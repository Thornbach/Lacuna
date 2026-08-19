//! Integrated pipeline tab:
//! Segmentation → Tiling → Anomaly detection → Restitch → (Reconstruction/Clustering).
//!
//! Slice 2: end-to-end Segment → Tile → Detect(DINO) → Restitch, on the worker
//! thread, with a leaf gallery + anomaly overlay. Clustering + the right-panel
//! UI (S3) and reconstruction stats (S4) build on this.

pub mod bank;
pub mod dino;
pub mod meta;
pub mod detect;
pub mod fewshot;
pub mod channels;
pub mod cluster;
pub mod tiling;
pub mod projection;
pub mod worker;
pub mod hardneg_mining;
pub mod shortcuts;

use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{mpsc, Arc},
    sync::atomic::{AtomicBool, Ordering},
};

use egui::{Color32, Context, RichText, Ui};
use egui_extras::{Column, TableBuilder};
use egui_phosphor::regular as icon;
use egui_plot::{Plot, Points};

use crate::settings::{AppDefaults, AppSettings, ClusterAlgo, CutMode};
use crate::tabs::leaf_seg::inference::{list_images, scan_image_count};
use crate::tabs::mask_tools::{
    dist_to_polygon_boundary, dist_to_polyline, fill_polygon_mask, mask_connected_components,
    point_in_polygon, reclaim_kerf, wand_flood_fill,
};
use crate::tabs::train::head::{
    spawn_retrain, RetrainCfg, RetrainMsg, spawn_calibrate, CalibrateCfg, rewrite_curated_family,
};
use crate::ui_kit;
use crate::widgets::ToastManager;
use hardneg_mining::{
    spawn_mine, spawn_mine_unmarked, LeafMineInput, MineConfig, MineMsg, MineUnmarkedConfig,
};
use worker::{spawn_pipeline, spawn_rank, AnomalyRegion, PipeConfig, PipeMsg, PipelineLeaf,
             RankLeaf, RankMsg, RankRegion};

/// Okabe–Ito, the de-facto standard qualitative palette for scientific figures:
/// eight hues chosen to stay distinguishable under protanopia, deuteranopia and
/// tritanopia.
///
/// The previous palette had red at index 0 and green at index 2 — the canonical
/// red/green confusion, and the two colours a run produces FIRST. It also had
/// two blues (1/8) and two greens (2/7) that were close even for normal vision.
/// Roughly 8% of men have a colour-vision deficiency, and class colour is the
/// primary information channel of this entire product.
///
/// Bonus, and not a small one for this user: these are the same values
/// conventionally used in published figures, so on-screen classes and the paper
/// end up the same colour.
const CLUSTER_PALETTE: [[u8; 3]; 8] = [
    [230, 159,   0], // orange
    [ 86, 180, 233], // sky blue
    [  0, 158, 115], // bluish green
    [240, 228,  66], // yellow
    [  0, 114, 178], // blue
    [213,  94,   0], // vermilion
    [204, 121, 167], // reddish purple
    [140, 140, 140], // neutral grey (Okabe–Ito's black, lightened for dark UIs)
];

/// Outline styles, cycled one step slower than the colours so that when the
/// palette wraps the STYLE differs.
///
/// This is the fix for two problems at once. Colour alone is not an accessible
/// encoding (WCAG 1.4.1), and `id % 8` means family 8 gets family 0's colour with
/// nothing else to tell them apart — routine over a long session, since every
/// hand-typed family name allocates a fresh id. A dash pattern survives both
/// colour blindness and greyscale printing.
///
/// `None` = solid. `Some(dash_len)` = dashes of that length in leaf pixels.
const CLUSTER_DASH: [Option<f32>; 4] = [None, Some(6.0), Some(2.5), Some(12.0)];

const GALLERY_PER_PAGE: usize = 60;

/// Per-family colour overrides, set by clicking a swatch.
///
/// A module-level table rather than tab state because `cluster_color` is a free
/// function called from ~20 places — the canvas painter, every swatch, the
/// export overlays, the leaf gallery. Threading a `&PipelineTab` through all of
/// them to support a cosmetic preference would be a worse trade than one
/// process-wide map, and there is only ever one pipeline tab.
static FAMILY_COLORS: std::sync::OnceLock<std::sync::Mutex<HashMap<i32, [u8; 3]>>> =
    std::sync::OnceLock::new();

fn family_colors() -> &'static std::sync::Mutex<HashMap<i32, [u8; 3]>> {
    FAMILY_COLORS.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

pub fn set_family_color(id: i32, rgb: Option<[u8; 3]>) {
    if let Ok(mut m) = family_colors().lock() {
        match rgb {
            Some(c) => { m.insert(id, c); }
            None => { m.remove(&id); }
        }
    }
}

pub fn family_color_overrides() -> HashMap<i32, [u8; 3]> {
    family_colors().lock().map(|m| m.clone()).unwrap_or_default()
}

fn cluster_color(id: i32) -> [u8; 3] {
    if let Ok(m) = family_colors().lock() {
        if let Some(c) = m.get(&id) {
            return *c;
        }
    }
    if id < 0 {
        [150, 150, 150] // noise
    } else {
        CLUSTER_PALETTE[id as usize % CLUSTER_PALETTE.len()]
    }
}

/// Where results go when the user has not said otherwise: a sibling of the
/// source folder, named after it.
///
/// A sibling rather than a child, because `list_images` walks the source folder
/// recursively — an output nested inside it would be re-scanned on the next run
/// and every leaf cut-out would come back as a new "photograph". Suffixed and
/// de-duplicated so a second run on the same folder does not silently write into
/// the first run's results.
fn derived_output_for(src: &std::path::Path) -> Option<PathBuf> {
    let parent = src.parent()?;
    let stem = src.file_name()?.to_string_lossy().to_string();
    let base = parent.join(format!("{stem}_lacuna"));
    if !base.exists() {
        return Some(base);
    }
    for n in 2..100 {
        let c = parent.join(format!("{stem}_lacuna{n}"));
        if !c.exists() {
            return Some(c);
        }
    }
    Some(base)
}

/// Centre of mass of a region's mask, in LEAF coordinates.
///
/// `AnomalyRegion` keeps a bbox and a bbox-local mask but no centroid — the
/// detection path had one on its own `Region` type and dropped it on conversion.
/// `embed_crop` centres its window here, so this has to be the mask's centroid
/// and not the bbox centre: for an L-shaped or crescent region those differ
/// enough to shift the crop off the tissue being embedded.
fn region_centroid(r: &AnomalyRegion) -> [f32; 2] {
    let [bx, by, bw, bh] = r.bbox_leaf;
    let (mut sx, mut sy, mut n) = (0f64, 0f64, 0usize);
    for y in 0..bh {
        for x in 0..bw {
            if r.mask.get((y * bw + x) as usize).copied().unwrap_or(false) {
                sx += (bx + x) as f64;
                sy += (by + y) as f64;
                n += 1;
            }
        }
    }
    if n == 0 {
        [(bx + bw / 2) as f32, (by + bh / 2) as f32]
    } else {
        [(sx / n as f64) as f32, (sy / n as f64) as f32]
    }
}

/// Draw a family's swatch: its colour AND its outline style, so the legend
/// teaches the second cue rather than leaving it to be discovered on the canvas.
/// Without this the dash pattern is a code with no key.
fn family_swatch(ui: &mut Ui, id: i32, size: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(size * 1.8, size), egui::Sense::hover());
    let c = cluster_color(id);
    let col = Color32::from_rgb(c[0], c[1], c[2]);
    let p = ui.painter();
    match cluster_dash(id) {
        None => {
            p.rect_filled(rect, 2.0, col);
        }
        Some(dash) => {
            // Fainter fill plus a dashed rule across it — the same visual
            // vocabulary the canvas outline uses for this family.
            p.rect_filled(rect, 2.0, col.linear_multiply(0.30));
            let y = rect.center().y;
            let d = (dash * 0.9).clamp(2.0, 6.0);
            p.add(egui::Shape::dashed_line(
                &[egui::pos2(rect.left() + 1.0, y), egui::pos2(rect.right() - 1.0, y)],
                egui::Stroke::new(2.0, col),
                d,
                d * 0.7,
            ));
        }
    }
}

/// The dash pattern that goes with `cluster_color(id)`. Advances every time the
/// colour wraps, so `id` and `id + 8` share a hue but never a style.
fn cluster_dash(id: i32) -> Option<f32> {
    if id < 0 {
        Some(3.0) // noise: always dashed, so it reads as provisional
    } else {
        let cycle = (id as usize) / CLUSTER_PALETTE.len();
        CLUSTER_DASH[cycle % CLUSTER_DASH.len()]
    }
}

/// Blend a cluster color toward gray — Focus mode's "dim" for every cluster
/// except the one currently selected. Never hides (that was the QA
/// complaint), just visually de-emphasizes so the rest stays legible as
/// context.
fn dim_color(c: [u8; 3]) -> [u8; 3] {
    const GRAY: u8 = 130;
    [lerp_u8(c[0], GRAY, 0.65), lerp_u8(c[1], GRAY, 0.65), lerp_u8(c[2], GRAY, 0.65)]
}

struct ClusterInfo {
    id:      i32,
    members: Vec<usize>, // indices into `regions`
}

/// One tile-picker-style hard-negative stamp: `(x,y)` = leaf-pixel top-left
/// (may overhang the leaf's own bounds; those pixels are saved transparent).
#[derive(Clone)]
struct HardnegStamp {
    x:    i32,
    y:    i32,
    file: PathBuf,
}

#[derive(Clone, Copy)]
enum Pick { Source, Output, Yolo, Dino, Bank, Meta, Recon, Head, MineHealthyDir, BaseSet }

/// Active canvas tool — Photoshop-style, mutually exclusive, and always
/// visibly indicated (see `show_toolbox`/the canvas options bar) so the same
/// click/right-click gesture never silently means two different things.
#[derive(Clone, Copy, PartialEq)]
enum CanvasTool { Select, MarkHealthy, Brush, Eraser, Knife, Scissor, Lasso, Wand, Polygon }

/// One entry per undoable structural edit — `Ctrl+Z`/the gallery's "Undo"
/// button always pops the most recent one, regardless of which action
/// produced it. `Cut` (Knife tool) never touches `removed`/`persisted`:
/// the pieces it creates were never independently rejected, so undoing a
/// cut is purely a visibility flip (`merged_away`), not a reject-reversal.
/// One region's mutable geometry, captured so an edit to it can be reversed.
///
/// Indices are stable: removal in this app is a visibility flag (`removed` /
/// `merged_away`), never an actual `Vec::remove`, so a stored index still means
/// the same region after any number of edits.
#[derive(Clone)]
struct RegionGeom {
    idx:  usize,
    mask: Vec<bool>,
    area: u32,
    crop: Vec<u8>,
}

enum UndoEntry {
    Remove(Vec<usize>),
    Cut { originals: Vec<usize>, created: Vec<usize> },
    /// Regions brought into existence by a brush / wand / polygon stroke.
    ///
    /// Mask edits used to push NOTHING onto this stack, so Ctrl+Z after
    /// painting silently reached past the stroke to whatever structural edit
    /// came before — reported repeatedly as undo "doing something in the
    /// regions from 10 steps ago" that could not then be taken back.
    ///
    /// A stroke touching existing regions MERGES with them, which hides the
    /// others and rewrites the survivor, so reversing it needs more than the
    /// created id. `merged` and the geometry pair are recorded by diffing state
    /// around the merge rather than by reaching into `merge_region_group` —
    /// correct regardless of what that function does internally.
    Paint {
        created: Vec<usize>,
        merged:  Vec<usize>,
        before:  Vec<RegionGeom>,
        after:   Vec<RegionGeom>,
    },
    /// An eraser stroke: the geometry of every region it changed, before and
    /// after, plus any region it erased down to nothing.
    ///
    /// `after` exists so redo is exact rather than a re-run of the gesture,
    /// which would depend on cursor position that no longer exists.
    Erase {
        before:  Vec<RegionGeom>,
        after:   Vec<RegionGeom>,
        emptied: Vec<usize>,
    },
    /// Regions confirmed into the curation set. Only ids that were NOT already
    /// persisted are recorded — undoing a confirm must not delete a curation the
    /// user had written earlier by some other route.
    ///
    /// Confirm was the most-used write in the app and the only one of the four
    /// editing actions with no undo at all: a mis-aimed "Confirm all remaining"
    /// wrote hundreds of rows to disk with nothing to take them back.
    Confirm(Vec<usize>),
}

#[derive(Clone, Copy, PartialEq)]
enum BrushShape { Square, Circle }

/// Which stage screen the Analyse tab is showing.
#[derive(Clone, Copy, PartialEq)]
enum StageView { Review, Done }

const BASE_ROWS_HELP: &str = "\
How many rows of the ORIGINAL training set get mixed into a retrain so it does \
not forget what the head already knew.\n\n\
Auto keeps roughly ten base rows per curated example, with a floor of 10,000 — \
the configuration measured best (LEARNS 0.942 / KEEPS 0.476). The floor is the \
measured part; the 10:1 ratio above it is a heuristic to hold that balance as \
curations grow, not a separately validated optimum.\n\n\
Turn it off to set the number by hand.";

/// Canonical filenames for the retrain base set, searched in `models/`.
/// `_headids` first: `base_set_gt.bin` uses the GROUND-TRUTH legend, whose class
/// ids do not match the head's (it drops Holes and swaps Sucker/Nekrosis), so it
/// must never be picked up automatically.
const BASE_SET_NAMES: [&str; 2] = ["base_set_headids.bin", "base_set.bin"];

/// Gallery ordering. The default is deliberately NOT detection order.
#[derive(Clone, Copy, PartialEq)]
enum GallerySort {
    /// Least like its own family first — the review order that scales.
    Unusual,
    /// Biggest first: what dominates the measured area, so errors cost most.
    Largest,
}

impl GallerySort {
    fn label(self) -> &'static str {
        match self {
            Self::Unusual => "Unusual first",
            Self::Largest => "Largest first",
        }
    }
    const ALL: [Self; 2] = [Self::Unusual, Self::Largest];
}

/// `2905` -> `2,905`. Counts in the thousands are the norm here and unseparated
/// digits are genuinely hard to compare at a glance.
fn fmt_thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// "45s", "17 min", "2 h 05 min" — coarse on purpose. False precision ("11:47
/// remaining") invites the user to trust a number that is an extrapolation from
/// an average, and then to notice it was wrong.
fn humanize_secs(s: f64) -> String {
    let s = s.max(0.0) as u64;
    if s < 90 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{} min", (s + 30) / 60)
    } else {
        format!("{} h {:02} min", s / 3600, (s % 3600) / 60)
    }
}

/// Image-writing half of an export, done a slice at a time.
///
/// A 10,000-leaf export is 10,000 full-resolution PNG encodes; doing it in one
/// call froze the window for minutes with no progress and no way out — which
/// reads as a crash. A worker thread is the obvious fix but the wrong one here:
/// the overlays are composited from each leaf's own pixel buffer, and shipping
/// 10,000 of those across a thread boundary means cloning gigabytes.
///
/// So it runs on the UI thread but yields: a bounded chunk per frame, progress
/// on screen, cancellable. The CSV — the part that actually matters — is written
/// up front and completely, so a cancelled export still leaves valid results.
struct ExportJob {
    crops_dir:  PathBuf,
    leaves_dir: PathBuf,
    /// Regions still to write, as (region index, filename).
    crops:      Vec<(usize, String)>,
    /// Leaves still to composite+write.
    leaves:     Vec<usize>,
    crop_cur:   usize,
    leaf_cur:   usize,
    written:    usize,
    failed:     usize,
    total:      usize,
}

/// Which action asked to wipe the run's in-memory review state. Both callers of
/// `reset_run_state` discard the whole session — the leaf you were on, every
/// rejection, the undo stack — and both were one unguarded click, one of them the
/// largest, greenest button in the tab.
#[derive(Clone, Copy, PartialEq)]
enum PendingReset {
    /// "Run Pipeline" pressed while a reviewed run is already loaded.
    Rerun,
    /// "Use this head now" after an in-place retrain.
    SwitchHead,
}

/// Right-panel sub-tabs — splits what used to be one long scrolling column
/// (leaf/morphology / stats / curation / retrain / export / log all stacked)
/// into focused views, mirroring Photoshop's tabbed panel dock
/// (Layers/Channels/Paths).

pub struct PipelineTab {
    source_folder: Option<PathBuf>,
    output_folder: Option<PathBuf>,
    yolo_model:    Option<PathBuf>,
    dino_model:    Option<PathBuf>,
    bank_path:     Option<PathBuf>,
    meta_path:     Option<PathBuf>,
    recon_ckpt:    Option<PathBuf>,
    head_path:     Option<PathBuf>,
    use_fewshot:     bool,
    use_patchcore:   bool,  // run the coreset bank ALONGSIDE the few-shot head (open-set safety net)
    unsupervised_families: bool, // DBSCAN over descriptors assigns family, not the head's argmax
    domain_projection: bool, // train+use a curation-adapted embedding projection (projection.rs)
    head_tau:        f32,
    head_grow:       f32,
    tile_size:       u32,
    margin_erode_px: u32,
    /// See `PipeConfig::detect_holes` — a hole eaten through the leaf is
    /// TRANSPARENT after segmentation, so it is excluded from tiling and no
    /// detector ever sees it. Only holes the segmenter failed to cut out get
    /// found, which means better segmentation reports FEWER holes. On by
    /// default; the geometry is unambiguous where the appearance model is blind.
    detect_holes:    bool,
    min_hole_area:   u32,
    /// See `PipeConfig::filter_margin_holes` — suppress head-detected "Holes"
    /// regions that hug the leaf outline instead of sitting inside it.
    filter_margin_holes: bool,
    hole_margin_px:      u32,
    /// Write one PNG per anomaly into `export/crops/`. Off by default — on a
    /// large run this dominates export time (one encode per anomaly), and the
    /// CSV row plus the per-leaf overlay already describe every region.
    export_crops:        bool,
    /// Write one full-size overlay PNG per leaf into `export/leaves/`.
    ///
    /// Gating the per-anomaly crops but not these missed the larger cost: crops
    /// are small thumbnails, whereas this is one FULL-RESOLUTION RGBA encode per
    /// leaf — 10,000 of them on a 10,000-leaf batch, each with every region
    /// painted in first. On a big run this is the dominant term in export time,
    /// not the crops.
    ///
    /// Defaults ON because it is the existing behaviour and the overlays are the
    /// main visual artefact; the cost is stated on the control so it can be an
    /// informed choice rather than a surprise.
    export_overlays:     bool,
    conf:            f32,
    seg_alpha_lo:    f32,   // YOLO cutout edge tightness (feather start)
    seg_chroma_min:  i32,   // YOLO cutout background-chroma rejection
    cluster_eps:     f32,   // DBSCAN radius; lower = more/smaller/looser clusters
    cluster_min_pts: usize, // DBSCAN min points; lower = more/smaller/looser clusters
    cluster_algo:    ClusterAlgo,
    target_k:        usize, // FixedK only; 0 = auto via suggest_k
    cut_mode:        CutMode,
    adaptive_threshold: f32, // Adaptive only: inconsistency sensitivity
    // Set only when the most recent run used Hierarchical clustering — enables
    // instant post-hoc re-cutting (drag K / adjust sensitivity, no pipeline
    // rerun) instead of another full run per guess. Cleared on a fresh run.
    hcluster:        Option<worker::HierarchicalClusterState>,
    recut_k:         usize,    // live UI value for the Fixed K re-cut slider
    recut_mode:      CutMode,  // live UI value for the re-cut mode toggle
    recut_threshold: f32,      // live UI value for the Adaptive re-cut slider

    // segmentation preview (tune the cutout edge before a full run)
    preview_busy: bool,
    preview_note: String,

    // optional pre-detection calibration: preview+mark a leaf, derive a
    // versioned few-shot head from the marks, skippable entirely
    calib_preview_rx: Option<(PathBuf, mpsc::Receiver<Result<(Vec<u8>, u32, u32), String>>)>,
    calib_preview_n:  usize, // cycles through source images across repeated previews
    calib_name:       String,
    calib_scale:      f32, // ABSOLUTE target coefficient-row norm for calibrated classes (not relative to the base head)
    calib_rx:         Option<mpsc::Receiver<crate::tabs::train::head::RetrainMsg>>,
    calib_running:    bool,
    calib_cancel:     Arc<AtomicBool>,
    calib_log:        Vec<String>,
    calib_selected:   Option<PathBuf>, // currently-applied calibration file, for the picker's highlight
    calib_out_path:   Option<PathBuf>, // set by start_calibrate, applied on RetrainMsg::Done
    // Real detection over the calibration-preview leaf (not a blank canvas)
    // — a separate channel/state from `self.rx`/`poll_worker` deliberately,
    // since that handler REPLACES self.regions/labels wholesale and would
    // wipe out any existing results this appends alongside instead.
    calib_detect_rx:       Option<mpsc::Receiver<PipeMsg>>,
    calib_detect_cancel:   Arc<AtomicBool>,
    calib_detect_leaf_idx: Option<usize>, // where this leaf landed in self.results, for remapping region.leaf
    // Leaves created by `start_calibration_preview` — scopes calibration to
    // examples marked on THESE leaves, not `labels.jsonl`'s entire history
    // for the output folder (which also holds real-run confirms/rejects,
    // renames, hard-negatives — a catastrophic mix when fed to calibration).
    calib_preview_leaves: HashSet<usize>,

    // worker
    rx:             Option<mpsc::Receiver<PipeMsg>>,
    cancel_flag:    Arc<AtomicBool>,
    running:        bool,
    progress_done:  usize,
    progress_total: usize,
    stage:          String,
    log:            Vec<String>,

    // results + view
    results:      Vec<PipelineLeaf>,
    thumbs:       Vec<Option<egui::TextureHandle>>, // parallel to results
    leaf_valid_px: Vec<u32>,                // parallel to results; cached once (was a per-frame megapixel scan)
    selected_idx: Option<usize>,
    overlay_tex:  Option<egui::TextureHandle>,
    // leaf, cluster, recon, opacity%, outline, regions.len().
    // The region count is part of the key so the live per-leaf preview is
    // replaced the moment clustering delivers real regions — without it the
    // cached preview texture would survive and the leaf would keep showing
    // provisional colours after the run finished.
    overlay_key:  Option<(usize, Option<i32>, bool, u32, bool, usize)>,
    show_recon:   bool,   // overlay the reconstructed (filled-in) leaf area on the canvas
    overlay_alpha: f32,   // cluster overlay opacity (fill mode) — see the leaf beneath
    overlay_outline: bool, // draw cluster OUTLINES instead of filled pixels

    // canvas zoom/pan — a universal gesture (scroll to zoom, hold middle-
    // mouse to pan), not a tool; applied on top of the existing fit-to-panel
    // rect, every other canvas interaction already funnels through the
    // resulting img_rect/s, so nothing else needs to change.
    canvas_zoom: f32,
    canvas_pan:  egui::Vec2,
    /// Region the canvas should scroll into view on the next draw, set when the
    /// selection changes from the region grid. Honoured (and cleared) inside the
    /// canvas draw, which is the only place the leaf->screen mapping exists.
    center_on_region: Option<usize>,
    // lasso tool: live screen-space points while dragging (cleared on release)
    lasso_points: Vec<egui::Pos2>,
    // brush tool: accumulated leaf-pixel coords while dragging (cleared on
    // release), converted to a bbox-local mask only once the stroke ends. A
    // HashSet (not Vec) specifically so it can be rendered live every frame
    // during the drag without ballooning from duplicate re-stamps while the
    // cursor lingers in one spot — dedup keeps the render cost bounded to
    // the stroke's true unique-pixel footprint.
    brush_shape:  BrushShape,
    brush_size:   u32,
    brush_stroke: HashSet<(i32, i32)>,
    wand_mask:       HashSet<(i32, i32)>,            // pending wand selection, uncommitted until Fill
    wand_tolerance:  f32,                             // Lab a/b distance threshold for the flood-fill
    wand_lab_cache:  Option<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>, // (leaf_idx, L, a, b) — recomputed only on leaf change
    wand_mask_tex:   Option<egui::TextureHandle>, // cached render of wand_mask, rebuilt only when the mask changes

    // clustering (filled by PipeMsg::Clusters)
    regions:          Vec<AnomalyRegion>,
    region_area:      Vec<u32>,             // parallel to regions; cached mask pixel count
    labels:           Vec<i32>,             // parallel to regions
    coords:           Vec<[f32; 2]>,        // PCA-2, parallel to regions
    clusters:         Vec<ClusterInfo>,
    // Cached parse of `eff_head()`'s file, refreshed whenever that path
    // changes — lets pickers (`cluster_picker_rows`) surface the head's REAL
    // family list (even classes with zero members in THIS run's clusters),
    // so typing an existing family name always matches instead of silently
    // forking a duplicate class.
    head_cache:       Option<(PathBuf, fewshot::FewShotHead)>,
    selected_cluster: Option<i32>,
    /// Whether the canvas DIMS everything outside `selected_cluster`.
    ///
    /// Deliberately separate from the selection itself. Dimming used to be an
    /// unavoidable consequence of picking a cluster — one field meant "what is
    /// selected" and "hide everything else" at once — and reviewers disliked it
    /// strongly, because selecting a family to work on is not the same as asking
    /// for the rest of the leaf to disappear. Selection now highlights; focus is
    /// opt-in via the button or `/`, and only bites when exactly one family is
    /// selected.
    focus_mode:       bool,
    // Curate gallery: restrict to `selected_idx`'s own regions. Off by
    // default (whole-dataset view, unchanged). Small regions are easy to
    // miss in the dataset-wide gallery, which made it hard to tell when a
    // single leaf was actually fully reviewed — this + the per-leaf status
    // readout above the gallery fix that directly.
    filter_leaf_only: bool,
    selected_region:  Option<usize>,   // anomaly highlighted with a bbox on the leaf
    gallery_page:     usize,           // anomaly gallery pagination
    /// Gallery ordering — see GallerySort. Defaults to Unusual: with thousands
    /// of regions per leaf, detection order is not a review order.
    gallery_sort:     GallerySort,
    scroll_to_selected: bool,          // one-shot: scroll the gallery to selected_region
    // one-shot: scroll the LEAF strip to selected_idx. Set by the arrow-key
    // hotkeys only, not by clicking — a click already puts the tile under the
    // cursor, and re-centring it there would yank the strip out from under you.
    scroll_to_leaf: bool,
    region_thumbs:    Vec<Option<egui::TextureHandle>>, // parallel to regions
    removed:          HashSet<usize>,       // region indices removed by the user
    struct_undo:      Vec<UndoEntry>,       // undo stack: one entry per edit gesture
    /// Entries popped off `struct_undo`, awaiting redo. Cleared by any NEW edit
    /// (`push_undo`), which is the standard branch-invalidation rule: once you
    /// edit after undoing, the path you undid is no longer reachable.
    struct_redo:      Vec<UndoEntry>,
    persisted:        HashSet<usize>,       // region indices already written to curations/labels.jsonl this run
    // region indices absorbed into another region by merge_touching_regions.
    // `regions` is append-only — a merge never removes/renumbers entries (that
    // would require touching every index-keyed field below); the survivor's
    // own mask/bbox/crop get updated in place instead, and every other member
    // of the merge group lands here. Semantically distinct from `removed`
    // (not "rejected," never subject to `UndoEntry::Remove`) — see `region_visible`.
    // Knife-cut originals also land here (see `UndoEntry::Cut`), restored the
    // same way a merge survivor's absorbed members would be.
    merged_away:      HashSet<usize>,
    // Leaf indices the user threw out wholesale (bad segmentation, a cut-off
    // leaf, debris the segmenter called a leaf). Distinct from `removed`, which
    // rejects ONE region and — deliberately — keeps it as training signal: a
    // rejected leaf is not a statement about any anomaly on it, it says the leaf
    // itself should never have entered the run, so nothing on it may be counted,
    // exported, or mined. Enforced centrally in `region_visible` so every
    // existing counter/renderer/export inherits it; the two places that ask about
    // a LEAF rather than a region (`count_fully_reviewed_leaves`,
    // `build_unmarked_mine_inputs`) check it directly, since "all regions
    // invisible" would otherwise read as "fully reviewed" and feed the whole
    // rejected leaf to the miner as healthy tissue.
    rejected_leaves:  HashSet<usize>,
    // Leaf indices the user has looked at and is happy with. Purely a review
    // bookmark: unlike `rejected_leaves` it changes NOTHING about export,
    // counting or mining — it exists so a 10,000-leaf batch can be worked across
    // several sittings without losing your place.
    reviewed:         HashSet<usize>,
    // Keyboard bindings, and the help overlay's state. `rebinding` holds the
    // action id awaiting a keypress; while it is set, normal key dispatch is
    // suppressed so pressing a key to BIND it cannot also FIRE it.
    keymap:           shortcuts::Keymap,
    help_open:        bool,
    rebinding:        Option<String>,
    // Pending irreversible actions, held until confirmed. `Option` rather than a
    // bool so the dialog can name exactly what it is about to destroy — a dialog
    // that says "are you sure?" without saying what teaches people to click yes.
    // Command palette state. palette_focused exists because egui only honours
    // request_focus once per widget lifetime, so it must fire on the frame the
    // window opens and not on every frame after.
    // On-demand appearance ranking (see worker::spawn_rank). Mutually exclusive
    // with the other GPU consumers — it loads its own DINO extractor.
    rank_rx:        Option<mpsc::Receiver<RankMsg>>,
    rank_cancel:    Arc<AtomicBool>,
    ranking:        bool,
    rank_done:      usize,
    rank_total:     usize,
    /// Settings column visible. Collapsed, the canvas gains ~250px — which is
    /// what review actually needs, since the folders and Run button are set once
    /// at the start and never touched again during a batch.
    /// Run-setup window (folders, calibration, Run) — opened from the top bar.
    setup_open:       bool,
    /// Family whose colour is being changed, if any.
    /// Which stage screen is showing. Review is the workspace; Done is the
    /// finish screen reached from the Export pill.
    stage_view:       StageView,
    /// One-shot request to force the Done screen's "Improve the model" section
    /// open (set by "Teach the model"); cleared the frame it is honoured.
    improve_open_req: bool,
    recolour_family:  Option<i32>,
    /// Action requested from a &self-ish draw context, run after the frame's
    /// panels close so it can take &mut self without fighting the borrow.
    perform_action_deferred: Option<String>,
    /// Regions set aside for a second pass. Purely a review aid — never
    /// written to disk and never affects export.
    flagged:          HashSet<usize>,
    filter_flagged:   bool,
    /// In-flight image export, stepped a chunk per frame. See ExportJob.
    export_job:       Option<ExportJob>,
    palette_open:     bool,
    palette_query:    String,
    palette_sel:      usize,
    palette_focused:  bool,
    pending_delete_cluster: Option<i32>,
    pending_reset:          Option<PendingReset>,
    /// When the current run started, for throughput and ETA. `None` when idle.
    run_started_at:         Option<std::time::Instant>,
    // Review state loaded from disk at run start, keyed by (source path relative
    // to the source folder, ordinal of the leaf WITHIN that photo).
    //
    // Deliberately NOT keyed by leaf index: indices are assigned in emit order at
    // run time and are not stable across runs, so index-keyed state silently
    // reattaches to the wrong leaf when anything upstream changes. That is the
    // same failure that let the base set's class ids swap Sucker and Nekrosis
    // unnoticed. Value carries (state, w, h) so a segmentation change can be
    // DETECTED rather than silently mis-restored.
    review_marks:     HashMap<(String, u32), (String, u32, u32)>,
    // Leaves whose stored mark referred to a differently-sized leaf — reported
    // once, never silently applied.
    review_mismatch:  usize,
    cluster_names:    HashMap<i32, String>,
    multi_selected:   HashSet<usize>,       // gallery tiles OR canvas rubber-band picks, for bulk reassign
    reassign_name:    String,               // target cluster name typed for bulk reassign
    canvas_drag_start: Option<egui::Pos2>,  // rubber-band select drag, screen-space (normal mode only)
    quick_reassign_open: bool,              // "R" hotkey: standalone Move-to-cluster popup, independent of right-click
    quick_reassign_pos: egui::Pos2,         // pointer position captured when the "R" hotkey opened the popup
    last_clicked_region: Option<usize>,     // gallery shift-click range-select anchor
    // Polygon tool: leaf-space nodes placed so far (not screen-space, so
    // already-placed nodes stay correctly positioned across a pan/zoom
    // mid-draw). `poly_pending` holds a closed polygon's rasterized stroke
    // while its family-choice popup (below) is open — only used when NO
    // region was selected at close time; a selected region commits
    // immediately via the same path Brush already uses, no popup needed.
    poly_points:  Vec<(f32, f32)>,
    poly_pending: Option<HashSet<(i32, i32)>>,
    poly_pick_pos: egui::Pos2,
    poly_pick_name: String,                 // "or new:" text field in the family-choice popup

    // hard-negative capture: tile-picker-style stamp tool — a magnifier-assisted
    // fixed-size square follows the cursor on the leaf canvas; click stamps that
    // exact patch straight into the flywheel curations, right-click/Ctrl+Z reverts.
    canvas_tool:       CanvasTool,
    hardneg_label:     String,
    hardneg_tile:      u32,
    hardneg_zoom:      f32,
    hardneg_stamps:    HashMap<usize, Vec<HardnegStamp>>, // leaf idx -> stamps (undo stack + overlay)
    hardneg_loupe_tex: Option<egui::TextureHandle>,

    // in-place retrain (flywheel, embedded): reuses train::head::spawn_retrain
    // unchanged, auto-filled from this tab's own output/head/dino fields — no
    // tab-switch needed to fine-tune from what you just curated.
    retrain_rx:     Option<mpsc::Receiver<RetrainMsg>>,
    retrain_cancel: Arc<AtomicBool>,
    retraining:     bool,
    retrain_stage:  String,
    retrain_log:    Vec<String>,
    retrain_done:   Option<PathBuf>, // Some(new head path) while the "use this head now" banner is showing
    /// Diagnostic escape hatch for the "retrain keeps getting worse"
    /// problem: zero-initializes any class that has curated rows THIS run
    /// instead of warm-starting from the current head's own coefficients
    /// (see `RetrainCfg::cold_start`'s doc comment for the full reasoning).
    retrain_cold_start: bool,
    /// Diagnostic: dump the exact training matrix + before/after head so an
    /// independent solver can be fitted on identical data and compared —
    /// see `RetrainCfg::dump_dir`. Writes hundreds of MB, so opt-in.
    retrain_dump: bool,
    /// See `RetrainCfg::base_set` — original training rows mixed into every
    /// retrain so curations fine-tune the head instead of replacing it.
    retrain_base_set:  Option<PathBuf>,
    retrain_base_rows: usize,
    /// Derive `base_rows` from the curation count instead of the manual value.
    retrain_auto_base_rows: bool,
    /// Gate for the destructive "delete every curation" confirm dialog.
    confirm_clear_curations: bool,
    /// results.csv shape: false = one row per anomaly, true = one row per leaf
    /// with per-family columns.
    export_wide: bool,
    /// Cached line count of `<output>/curations/labels.jsonl`, keyed by the
    /// file's (len, mtime) so the panel does not re-read a growing jsonl on
    /// every frame it draws.
    curation_count_cache: Option<(u64, Option<std::time::SystemTime>, usize)>,
    /// See `RetrainCfg::anchor` - pull the L2 penalty toward the current head
    /// instead of toward zero, so curations correct without competing for influence.
    retrain_anchor:    f32,

    // hard-negative MINING (flywheel, embedded, automated): scans an
    // independent folder of known-healthy tiles for patches the CURRENT
    // head wrongly calls defect, and stamps them into curations exactly
    // like a manual hardneg stamp — see hardneg_mining.rs.
    mine_healthy_dir:    Option<PathBuf>,
    mine_tau:            f32,
    mine_max:            usize,
    mine_rx:             Option<mpsc::Receiver<MineMsg>>,
    mine_cancel:         Arc<AtomicBool>,
    mining:              bool,
    mine_progress_done:  usize,
    mine_progress_total: usize,
    mine_found:          usize,
    mine_log:            Vec<String>,

    // single file-dialog channel (tagged with which field it fills)
    pick_rx:      Option<(Pick, mpsc::Receiver<Option<PathBuf>>)>,
    source_count: usize,
    defaults:     AppDefaults, // shared model-path defaults from Settings (inherited)
}

impl PipelineTab {
    pub fn new() -> Self {
        Self {
            source_folder: None,
            output_folder: None,
            yolo_model:    None,
            dino_model:    None,
            bank_path:     None,
            meta_path:     None,
            recon_ckpt:    None,
            head_path:     None,
            use_fewshot:     true,
            use_patchcore:   false, // opt-in: only worth it once the bank reflects real healthy data
            unsupervised_families: false, // opt-in: requires use_patchcore's bank+meta too
            domain_projection: false, // opt-in: needs unsupervised_families + real curations
            head_tau:        crate::settings::default_head_tau(),
            head_grow:       0.7,
            tile_size:       256,
            margin_erode_px: 6,
            detect_holes:    true,
            min_hole_area:   16,
            filter_margin_holes: false, // opt-in: changes what gets reported
            hole_margin_px:      16,
            export_crops:        false, // one encode per anomaly
            export_overlays:     true,  // existing behaviour; one encode per leaf
            conf:            0.25,
            seg_alpha_lo:    0.0,
            seg_chroma_min:  0,
            cluster_eps:     1.5,
            cluster_min_pts: 5,
            cluster_algo:    ClusterAlgo::Dbscan,
            target_k:        0,
            cut_mode:        CutMode::FixedK,
            adaptive_threshold: 8.0,
            hcluster:        None,
            recut_k:         0,
            recut_mode:      CutMode::FixedK,
            recut_threshold: 8.0,

            preview_busy: false,
            preview_note: String::new(),

            calib_preview_rx: None,
            calib_preview_n:  0,
            calib_name:       String::new(),
            calib_scale:      4.0,
            calib_rx:         None,
            calib_running:    false,
            calib_cancel:     Arc::new(AtomicBool::new(false)),
            calib_log:        Vec::new(),
            calib_selected:   None,
            calib_out_path:   None,
            calib_detect_rx:       None,
            calib_detect_cancel:   Arc::new(AtomicBool::new(false)),
            calib_detect_leaf_idx: None,
            calib_preview_leaves: HashSet::new(),

            rx:             None,
            cancel_flag:    Arc::new(AtomicBool::new(false)),
            running:        false,
            progress_done:  0,
            progress_total: 0,
            stage:          String::new(),
            log:            Vec::new(),

            results:      Vec::new(),
            leaf_valid_px: Vec::new(),
            thumbs:       Vec::new(),
            selected_idx: None,
            overlay_tex:  None,
            overlay_key:  None,
            show_recon:   false,
            overlay_alpha: 0.6,
            overlay_outline: false,

            canvas_zoom:  1.0,
            canvas_pan:   egui::Vec2::ZERO,
            center_on_region: None,
            lasso_points: Vec::new(),
            brush_shape:  BrushShape::Circle,
            brush_size:   32,
            brush_stroke: HashSet::new(),
            wand_mask:       HashSet::new(),
            wand_tolerance:  14.0,
            wand_lab_cache:  None,
            wand_mask_tex:   None,

            regions:          Vec::new(),
            region_area:      Vec::new(),
            labels:           Vec::new(),
            coords:           Vec::new(),
            clusters:         Vec::new(),
            head_cache:       None,
            selected_cluster: None,
            focus_mode:       false,
            filter_leaf_only: true,
            selected_region:  None,
            gallery_page:     0,
            gallery_sort:     GallerySort::Unusual,
            scroll_to_selected: false,
            scroll_to_leaf:     false,
            region_thumbs:    Vec::new(),
            removed:          HashSet::new(),
            struct_undo:      Vec::new(),
            struct_redo:      Vec::new(),
            persisted:        HashSet::new(),
            merged_away:      HashSet::new(),
            rejected_leaves:  HashSet::new(),
            reviewed:         HashSet::new(),
            keymap:           shortcuts::Keymap::default(),
            help_open:        false,
            rebinding:        None,
            rank_rx:        None,
            rank_cancel:    Arc::new(AtomicBool::new(false)),
            ranking:        false,
            rank_done:      0,
            rank_total:     0,
            setup_open:       false,
            stage_view:       StageView::Review,
            improve_open_req: false,
            recolour_family:  None,
            perform_action_deferred: None,
            flagged:          HashSet::new(),
            filter_flagged:   false,
            export_job:       None,
            palette_open:     false,
            palette_query:    String::new(),
            palette_sel:      0,
            palette_focused:  false,
            pending_delete_cluster: None,
            pending_reset:          None,
            run_started_at:         None,
            review_marks:     HashMap::new(),
            review_mismatch:  0,
            cluster_names:    HashMap::new(),
            multi_selected:   HashSet::new(),
            reassign_name:    String::new(),
            canvas_drag_start: None,
            quick_reassign_open: false,
            quick_reassign_pos: egui::Pos2::new(400.0, 300.0),
            last_clicked_region: None,
            poly_points: Vec::new(),
            poly_pending: None,
            poly_pick_pos: egui::Pos2::new(400.0, 300.0),
            poly_pick_name: String::new(),

            canvas_tool:       CanvasTool::Select,
            hardneg_label:     "healthy".to_string(),
            hardneg_tile:      64,
            hardneg_zoom:      4.0,
            hardneg_stamps:    HashMap::new(),
            hardneg_loupe_tex: None,

            retrain_rx:     None,
            retrain_cancel: Arc::new(AtomicBool::new(false)),
            retraining:     false,
            retrain_stage:  String::new(),
            retrain_log:    Vec::new(),
            retrain_done:   None,
            retrain_cold_start: false,
            retrain_dump:       false,
            retrain_base_set:   Self::default_base_set(),
            retrain_base_rows:  10_000,
            retrain_auto_base_rows: true,
            confirm_clear_curations: false,
            export_wide:        true,  // leaf is the sampling unit; long needs a pivot
            curation_count_cache:   None,
            retrain_anchor:     1.0,

            mine_healthy_dir:    None,
            mine_tau:            0.6,
            mine_max:            300,
            mine_rx:             None,
            mine_cancel:         Arc::new(AtomicBool::new(false)),
            mining:              false,
            mine_progress_done:  0,
            mine_progress_total: 0,
            mine_found:          0,
            mine_log:            Vec::new(),

            pick_rx:      None,
            source_count: 0,
            defaults:     AppDefaults::default(),
        }
    }

    /// Receive the latest shared defaults from Settings (called each frame by app.rs).
    pub fn set_defaults(&mut self, d: &AppDefaults) {
        self.defaults = d.clone();
    }

    fn eff_yolo(&self) -> Option<PathBuf> { self.yolo_model.clone().or_else(|| self.defaults.yolo.clone()) }
    fn eff_dino(&self) -> Option<PathBuf> { self.dino_model.clone().or_else(|| self.defaults.dino.clone()) }
    fn eff_bank(&self) -> Option<PathBuf> { self.bank_path.clone().or_else(|| self.defaults.bank.clone()) }
    fn eff_meta(&self) -> Option<PathBuf> { self.meta_path.clone().or_else(|| self.defaults.meta.clone()) }
    fn eff_recon(&self) -> Option<PathBuf> { self.recon_ckpt.clone().or_else(|| self.defaults.recon.clone()) }
    fn eff_head(&self) -> Option<PathBuf> { self.head_path.clone().or_else(|| self.defaults.head.clone()) }

    /// Whether the run will use the few-shot head: toggle on AND a head path resolved.
    fn fewshot_active(&self) -> bool { self.use_fewshot && self.eff_head().is_some() }

    /// Lazily (re)loads `eff_head()`'s file into `head_cache`, reloading only
    /// when the resolved path changes — the source of truth `cluster_picker_rows`
    /// unions in so a class already known to the head, but with zero members
    /// in this run's own clustering, still shows up as a pick target.
    fn cached_head(&mut self) -> Option<&fewshot::FewShotHead> {
        let path = self.eff_head()?;
        let stale = match &self.head_cache {
            Some((p, _)) => *p != path,
            None => true,
        };
        if stale {
            match fewshot::FewShotHead::load(&path) {
                Ok(h) => self.head_cache = Some((path.clone(), h)),
                Err(_) => self.head_cache = None,
            }
        }
        self.head_cache.as_ref().map(|(_, h)| h)
    }

    fn inherited(&self, which: Pick) -> Option<PathBuf> {
        match which {
            Pick::Yolo => self.defaults.yolo.clone(),
            Pick::Dino => self.defaults.dino.clone(),
            Pick::Bank => self.defaults.bank.clone(),
            Pick::Meta => self.defaults.meta.clone(),
            Pick::Recon => self.defaults.recon.clone(),
            Pick::Head => self.defaults.head.clone(),
            _ => None,
        }
    }

    // ── lifecycle ─────────────────────────────────────────────────────────

    pub fn needs_repaint(&self) -> bool {
        self.running || self.retraining || self.mining || self.ranking || self.export_job.is_some()
    }

    pub fn save_settings(&self, s: &mut AppSettings) {
        let r = &mut s.pipeline;
        r.last_source_folder = self.source_folder.clone();
        r.last_output_folder = self.output_folder.clone();
        r.yolo_model_path    = self.yolo_model.clone();
        r.dino_model_path    = self.dino_model.clone();
        r.bank_path          = self.bank_path.clone();
        r.meta_path          = self.meta_path.clone();
        r.recon_ckpt         = self.recon_ckpt.clone();
        r.head_path          = self.head_path.clone();
        r.use_fewshot        = self.use_fewshot;
        r.use_patchcore      = self.use_patchcore;
        r.unsupervised_families = self.unsupervised_families;
        r.domain_projection = self.domain_projection;
        r.head_tau           = self.head_tau;
        r.head_grow          = self.head_grow;
        r.shortcuts          = self.keymap.to_map();
        // These seven change what results.csv contains and were silently lost on
        // every restart — see PipelineSettings for the full note.
        r.conf               = self.conf;
        r.seg_alpha_lo       = self.seg_alpha_lo;
        r.seg_chroma_min     = self.seg_chroma_min;
        r.detect_holes       = self.detect_holes;
        r.min_hole_area      = self.min_hole_area;
        r.filter_margin_holes = self.filter_margin_holes;
        r.hole_margin_px     = self.hole_margin_px;
        r.export_crops       = self.export_crops;
        r.export_overlays    = self.export_overlays;
        r.export_wide        = self.export_wide;
        r.filter_leaf_only   = self.filter_leaf_only;
        r.family_colors      = family_color_overrides()
            .into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        r.tile_size          = self.tile_size;
        r.margin_erode_px    = self.margin_erode_px;
        r.cluster_eps        = self.cluster_eps;
        r.cluster_min_pts    = self.cluster_min_pts;
        r.cluster_algo       = self.cluster_algo;
        r.target_k           = self.target_k;
        r.cut_mode           = self.cut_mode;
        r.adaptive_threshold = self.adaptive_threshold;
    }

    pub fn load_settings(&mut self, s: &AppSettings) {
        let r = &s.pipeline;
        self.source_folder = r.last_source_folder.clone();
        self.output_folder = r.last_output_folder.clone();
        self.yolo_model    = r.yolo_model_path.clone();
        self.dino_model    = r.dino_model_path.clone();
        self.bank_path     = r.bank_path.clone();
        self.meta_path     = r.meta_path.clone();
        self.recon_ckpt    = r.recon_ckpt.clone();
        self.head_path     = r.head_path.clone();
        self.use_fewshot   = r.use_fewshot;
        self.use_patchcore = r.use_patchcore;
        self.unsupervised_families = r.unsupervised_families;
        self.domain_projection = r.domain_projection;
        self.head_tau      = r.head_tau;
        self.head_grow     = r.head_grow;
        self.keymap        = shortcuts::Keymap::from_map(&r.shortcuts);
        self.conf          = r.conf;
        self.seg_alpha_lo  = r.seg_alpha_lo;
        self.seg_chroma_min = r.seg_chroma_min;
        self.detect_holes  = r.detect_holes;
        self.min_hole_area = r.min_hole_area;
        self.filter_margin_holes = r.filter_margin_holes;
        self.hole_margin_px = r.hole_margin_px;
        self.export_crops  = r.export_crops;
        self.export_overlays = r.export_overlays;
        self.export_wide     = r.export_wide;
        self.filter_leaf_only = r.filter_leaf_only;
        for (k, v) in &r.family_colors {
            if let Ok(id) = k.parse::<i32>() { set_family_color(id, Some(*v)); }
        }
        self.tile_size     = r.tile_size;
        self.margin_erode_px = r.margin_erode_px;
        self.cluster_eps     = r.cluster_eps;
        self.cluster_min_pts = r.cluster_min_pts;
        self.cluster_algo    = r.cluster_algo;
        self.target_k        = r.target_k;
        self.cut_mode        = r.cut_mode;
        self.adaptive_threshold = r.adaptive_threshold;
        if let Some(f) = self.source_folder.clone() {
            self.source_count = scan_image_count(&f);
        }
    }

    // ── show ──────────────────────────────────────────────────────────────

    pub fn show(&mut self, ui: &mut Ui, ctx: &Context, toasts: &mut ToastManager) {
        self.poll_worker(toasts);
        self.poll_retrain(toasts);
        self.poll_mine(toasts);
        self.poll_rank(toasts);
        self.step_export(toasts);
        self.poll_calibration_preview(ctx);
        self.poll_calibration_detect(ctx, toasts);
        self.poll_calibrate(toasts);
        self.handle_leaf_hotkeys(ctx, toasts);

        egui::TopBottomPanel::top("pipeline_stepper")
            .exact_height(28.0)
            .show_inside(ui, |ui| self.show_stepper(ui));
        egui::TopBottomPanel::top("pipeline_qol_bar")
            .exact_height(32.0)
            .show_inside(ui, |ui| self.show_qol_bar(ui, toasts));
        // ── nothing loaded: the flow screen IS the app ──────────────────────
        // Not a panel among panels. Before a run there is exactly one thing to
        // do, and surrounding it with an empty canvas, an empty gallery and a
        // cluster panel that says "run the pipeline first" made the app look
        // complicated at the precise moment it is simplest.
        if self.results.is_empty() && !self.running {
            egui::CentralPanel::default().show_inside(ui, |ui| {
                egui::ScrollArea::vertical().id_salt("start_scroll").show(ui, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(28.0);
                        ui.allocate_ui_with_layout(
                            egui::vec2(460.0, 0.0),
                            egui::Layout::top_down(egui::Align::Min),
                            |ui| self.show_start_screen(ui),
                        );
                    });
                });
            });
            self.show_shortcuts_window(ctx);
            // MUST be here too. This branch returns before the shared panel code
            // below, so without this the start screen's "Change folders…" set
            // `setup_open = true` and nothing ever drew the window — the button
            // was dead on the one screen that most needs it.
            self.show_setup_window(ctx);
            self.show_command_palette(ctx, toasts);
            self.show_confirm_dialogs(ctx, toasts);
            return;
        }

        // Done screen: a full screen, not a panel — an ending should feel like
        // arriving somewhere, not like another tab.
        if self.stage_view == StageView::Done && !self.running {
            egui::CentralPanel::default().show_inside(ui, |ui| {
                egui::ScrollArea::vertical().id_salt("done_scroll").show(ui, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(24.0);
                        ui.allocate_ui_with_layout(
                            egui::vec2(520.0, 0.0),
                            egui::Layout::top_down(egui::Align::Min),
                            |ui| self.show_done_screen(ui, toasts),
                        );
                    });
                });
            });
            self.show_shortcuts_window(ctx);
            self.show_setup_window(ctx);
            self.show_command_palette(ctx, toasts);
            self.show_confirm_dialogs(ctx, toasts);
            if let Some(id) = self.perform_action_deferred.take() {
                self.perform_action(&id, toasts);
            }
            return;
        }

        // ── tools rail, always visible ──────────────────────────────────────
        // Split out of the 300px control column so the canvas can have the room.
        // The old panel mixed "what my mouse does right now" with "what a
        // six-hour batch job will do", separated by a hairline, and cost 300px
        // permanently — on a 1440-wide window the two side panels took 46% of the
        // width before the image got any.
        egui::SidePanel::left("pipeline_rail")
            .exact_width(52.0)
            .resizable(false)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().id_salt("rail_scroll").show(ui, |ui| {
                    self.show_toolbox(ui);
                });
            });
        // The active tool's settings, floating just beside the rail rather than
        // stretched across a global bar.
        //
        // Where tool settings belong is a spatial-mapping question: they configure
        // THAT tool, so they should sit next to it, not in a strip shared with
        // opacity and zoom where they read as another global control. Floating
        // also means they cost no layout — nothing reflows when the active tool
        // changes, which is what made the top bar wiggle.
        self.show_tool_options_popover(ctx);
        egui::SidePanel::right("pipeline_clusters")
            .default_width(380.0)
            // BOUNDED. `ui.available_width()` in the legend and verdict rows,
            // inside a content-sized scroll area, inside a resizable panel, is a
            // feedback loop: the row asks for the available width, the panel
            // grows to fit the row, which makes more width available… The panel
            // grew without limit as the pointer moved. The scroll area is also
            // pinned (`auto_shrink false`) so its width comes from the panel
            // rather than from its contents.
            .min_width(300.0)
            .max_width(560.0)
            .resizable(true)
            .show_inside(ui, |ui| self.show_cluster_panel(ui, ctx, toasts));
        egui::TopBottomPanel::bottom("pipeline_gallery")
            .resizable(false)
            .min_height(108.0)
            .show_inside(ui, |ui| self.show_gallery(ui, ctx));
        egui::CentralPanel::default().show_inside(ui, |ui| self.show_canvas(ui, ctx, toasts));
        // Last, so these float above every panel.
        self.show_shortcuts_window(ctx);
        self.show_setup_window(ctx);
        self.show_command_palette(ctx, toasts);
        self.show_confirm_dialogs(ctx, toasts);
        if let Some(id) = self.perform_action_deferred.take() {
            self.perform_action(&id, toasts);
        }
    }

    /// Top QoL strip: focus-mode status + the overlay-appearance toggles that
    /// used to be scattered across the left controls panel (`show_recon`) and
    /// the right cluster panel (`overlay_outline`/`overlay_alpha`) — pulled up
    /// here so they're reachable from every sub-tab, not just where they
    /// happened to live before.
    fn show_qol_bar(&mut self, ui: &mut Ui, toasts: &mut ToastManager) {
        ui.horizontal(|ui| {
            ui.add_space(6.0);
            self.show_reject_leaf_button(ui, toasts);
            ui.separator();
            // Say plainly that the amber overlay is provisional. Without this the
            // preview is indistinguishable from a finished result whose families
            // all happen to be one colour.
            if let Some(i) = self.selected_idx {
                let previewing = self.results.get(i).map_or(false, |l| !l.anomaly.is_empty())
                    && !self.regions.iter().any(|r| r.leaf == i);
                if previewing {
                    ui.label(RichText::new("live preview — families assigned after clustering")
                        .small().color(Color32::from_rgb(235, 165, 60)));
                    ui.separator();
                }
            }
            // Selection and focus are shown as two separate facts, because they
            // ARE two separate things now — you can have a family selected
            // without the rest of the leaf being dimmed. ASCII only: the bundled
            // fonts have no U+2715 and render it as tofu.
            if let Some(cid) = self.selected_cluster {
                let name = self.cluster_names.get(&cid).cloned().unwrap_or_else(|| format!("Cluster {cid}"));
                family_swatch(ui, cid, 10.0);
                ui.label(format!("Selected: {name}"));
                let label = if self.focus_mode { "Focus on  (/)" } else { "Focus off  (/)" };
                if ui.selectable_label(self.focus_mode, label)
                    .on_hover_text("Dim every family except the selected one. Shortcut: /")
                    .clicked()
                {
                    self.toggle_focus_mode(toasts);
                }
                if ui.small_button("x Clear selection").clicked() {
                    self.selected_cluster = None;
                    self.selected_region = None;
                    self.focus_mode = false;
                    self.overlay_tex = None;
                }
            } else {
                ui.label(RichText::new("No family selected").color(Color32::GRAY));
            }
            ui.separator();

            // Every control below is ALWAYS present, disabled rather than hidden
            // when it does not apply.
            //
            // Two bugs came out of doing it the other way. The recon checkbox and
            // the opacity slider were each wrapped in an `if`, so ticking Outline
            // or Show-reconstruction added or removed a widget and everything to
            // its right jumped — reported as the bar "wiggling". And the
            // right-aligned Shortcuts button was emitted BEFORE the slider:
            // `right_to_left` claims the rest of the row, so the slider had no
            // space left and simply never appeared.
            let has_recon = self.selected_idx.and_then(|i| self.results.get(i))
                .map_or(false, |l| !l.recon_mask.is_empty());
            ui.add_enabled_ui(has_recon, |ui| {
                if ui.checkbox(&mut self.show_recon, "Show reconstruction")
                    .on_hover_text("Tint (under the anomalies) the area the model reconstructed —\n\
                                    where the leaf was damaged/missing — so you see the whole intact\n\
                                    leaf with the damage as holes.")
                    .on_disabled_hover_text("This leaf has no reconstruction — enable it in \
                                             Settings and re-run.")
                    .changed()
                {
                    self.overlay_tex = None;
                }
            });
            ui.separator();
            if ui.checkbox(&mut self.overlay_outline, "Outline")
                .on_hover_text("Draw cluster OUTLINES (leaf fully visible inside) instead of\n\
                                filled pixels. Same family colours and dash styles.")
                .changed()
            {
                self.overlay_tex = None;
            }
            // Opacity applies to fill-mode regions and to the reconstruction tint
            // (which paints in both modes), so it is live unless neither is.
            let alpha_applies = !self.overlay_outline || (has_recon && self.show_recon);
            ui.scope(|ui| {
                ui.spacing_mut().slider_width = 150.0;
                ui.add_enabled_ui(alpha_applies, |ui| {
                    if ui.add(egui::Slider::new(&mut self.overlay_alpha, 0.1..=1.0)
                        .text("opacity")
                        .fixed_decimals(2))
                        .on_disabled_hover_text("Outline mode draws contours, which have no fill \
                                                 to make transparent.")
                        .changed()
                    {
                        self.overlay_tex = None;
                    }
                });
            });

            // Zoom lives here now, beside opacity, rather than in a banner over
            // the canvas — it is a view control like the other two.
            ui.separator();
            ui.label(RichText::new(format!("{:.0}%", self.canvas_zoom * 100.0))
                .small().color(ui_kit::MUTED()));
            if ui.small_button("Fit")
                .on_hover_text("Reset zoom and pan. Scroll to zoom, middle-drag to pan.")
                .clicked()
            {
                self.canvas_zoom = 1.0;
                self.canvas_pan = egui::Vec2::ZERO;
            }

            // Right-aligned LAST, so it cannot steal the row from the controls
            // above it. A visible entry point matters more than the F1 binding:
            // someone who does not know shortcuts exist will never press a key to
            // find out.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(6.0);
                let k = shortcuts::key_label(self.keymap.key("help"));
                if ui.small_button(format!("Shortcuts  ({k})"))
                    .on_hover_text("Every keyboard shortcut, what it does, and how to change it.")
                    .clicked()
                {
                    self.help_open = !self.help_open;
                }
                // No "Commands" button. Anyone who uses a command palette opens
                // it with the key; anyone who does not was never going to click a
                // button labelled with a keyboard shortcut. The Shortcuts window
                // beside this lists the binding for the few who go looking.
                // Reachable, but not a permanent column: folders, calibration and
                // Run are set once at the start of a batch and then never touched
                // for hours, so they do not earn standing screen space.
                if ui.small_button("Setup")
                    .on_hover_text("Folders, calibration, and starting another run.")
                    .clicked()
                {
                    self.setup_open = !self.setup_open;
                }
            });
        });
    }

    /// Leaf-level keyboard navigation, for reviewing a large batch quickly:
    /// `←`/`→` step through leaves, `X` rejects/restores the current one.
    ///
    /// Gated on nothing having keyboard focus, the same guard the tool hotkeys
    /// use (`show_canvas`'s `focused` check) — otherwise typing an "x" into the
    /// cluster-rename field would throw the leaf out of the run.
    ///
    /// Arrows CLAMP rather than wrap. Wrapping would silently send you from the
    /// last leaf back to the first, which during a long review reads as "the
    /// list reset" and is very easy to not notice.
    fn handle_leaf_hotkeys(&mut self, ctx: &Context, toasts: &mut ToastManager) {
        // Help is reachable even with no results — it is how a new user finds out
        // what any of this does.
        if self.rebinding.is_none()
            && !ctx.memory(|m| m.focused().is_some())
            && ctx.input(|i| self.keymap.pressed(i, "help"))
        {
            self.help_open = !self.help_open;
        }
        // Ctrl+K anywhere, including with a field focused — a palette you cannot
        // reach without first clicking away is a palette people stop using.
        // Space does the same thing, but ONLY when nothing has keyboard focus:
        // it is a printable character, so firing it into a text field would make
        // the palette impossible to type a space into.
        let plain_palette_key = self.keymap.key("palette");
        let focused = ctx.memory(|m| m.focused().is_some());
        let by_ctrl_k = ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::K));
        let by_plain = !focused
            && !shortcuts::is_unbound(plain_palette_key)
            && ctx.input(|i| !i.modifiers.any() && i.key_pressed(plain_palette_key));
        if !self.palette_open && self.rebinding.is_none() && (by_ctrl_k || by_plain) {
            self.palette_open = true;
            self.palette_query.clear();
            self.palette_sel = 0;
            return;
        }
        if self.results.is_empty()
            || self.rebinding.is_some()
            || self.palette_open
            || ctx.memory(|m| m.focused().is_some())
        {
            return;
        }
        // Unmodified keys only: Ctrl+K must not also fire whatever K is bound to,
        // and Ctrl+Z belongs to undo rather than to the tool on Z.
        let plain = ctx.input(|i| !i.modifiers.any());
        let (prev, next, reject, mark, jump) = ctx.input(|i| (
            plain && self.keymap.pressed(i, "leaf.prev"),
            plain && self.keymap.pressed(i, "leaf.next"),
            plain && self.keymap.pressed(i, "leaf.reject"),
            plain && self.keymap.pressed(i, "leaf.reviewed"),
            plain && self.keymap.pressed(i, "leaf.next_unreviewed"),
        ));
        // ── 1..9 assign a family to the current selection ───────────────────
        // The mockup's central review gesture. Order matches the cluster panel's
        // own listing, so the number a user sees beside a family is the key that
        // assigns it — position and mapping stay in step.
        if plain {
            let pressed = ctx.input(|i| {
                use egui::Key::*;
                [Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9]
                    .iter().position(|k| i.key_pressed(*k))
            });
            if let Some(slot) = pressed {
                let ids: Vec<i32> = self.clusters.iter().map(|c| c.id).filter(|&id| id >= 0).collect();
                if let Some(&cid) = ids.get(slot) {
                    let sel = self.effective_selection();
                    if !sel.is_empty() {
                        let name = self.class_display_name(cid);
                        self.reassign_ids(&sel, cid, &name, toasts);
                    }
                }
            }
        }

        // View toggles have no dedicated UI keys elsewhere, so they dispatch here.
        for id in ["region.next", "region.prev", "region.flag",
                   "view.outline", "view.recon", "view.focus", "view.clear_focus", "view.fit", "view.panel",
                   "run.start", "run.cancel", "review.export", "review.confirm_family",
                   "review.undo", "review.redo"] {
            let k = self.keymap.key(id);
            if !shortcuts::is_unbound(k) && plain && ctx.input(|i| i.key_pressed(k)) {
                self.perform_action(id, toasts);
            }
        }
        if prev || next {
            let n = self.results.len();
            let target = match self.selected_idx {
                None => 0, // nothing selected yet: either arrow starts at the first leaf
                Some(cur) if next => (cur + 1).min(n - 1),
                Some(cur) => cur.saturating_sub(1),
            };
            if self.selected_idx != Some(target) {
                self.selected_idx = Some(target);
                self.selected_region = None;
                self.overlay_tex = None;
                self.scroll_to_leaf = true;
            }
        }
        if reject {
            if let Some(li) = self.selected_idx {
                self.toggle_reject_leaf(li, toasts);
            }
        }
        // M marks the current leaf done. Deliberately does NOT auto-advance:
        // marking and moving are separate decisions, and binding them would make
        // an accidental M skip a leaf you never looked at. N is the advance.
        if mark {
            if let Some(li) = self.selected_idx {
                self.toggle_reviewed(li, toasts);
            }
        }
        if jump {
            let from = self.selected_idx.map_or(0, |i| i + 1);
            match self.next_unreviewed(from) {
                Some(t) => {
                    self.selected_idx = Some(t);
                    self.selected_region = None;
                    self.overlay_tex = None;
                    self.scroll_to_leaf = true;
                }
                None => toasts.success("Every leaf has been reviewed or rejected."),
            }
        }
    }

    /// Whole-leaf reject/restore, first control in the top bar.
    ///
    /// Deliberately NOT next to the per-region reject in the gallery: that one
    /// says "this detection is wrong" and keeps the region as training signal,
    /// this one says "this leaf should not be in the run at all". Conflating them
    /// would poison the curation set with rejects the user never meant as labels.
    fn show_reject_leaf_button(&mut self, ui: &mut Ui, toasts: &mut ToastManager) {
        let Some(li) = self.selected_idx else {
            ui.add_enabled(false, egui::Button::new("✕ Reject leaf"))
                .on_disabled_hover_text("Select a leaf in the gallery below first.");
            return;
        };
        let rejected = self.rejected_leaves.contains(&li);
        let n = self.regions.iter().filter(|r| r.leaf == li).count();

        // "Reviewed" sits beside reject because they are the two whole-leaf
        // verdicts and both persist. The hotkey is printed on the button — that
        // is the mechanism by which mouse users ever learn it exists.
        if !rejected {
            let done = self.reviewed.contains(&li);
            let label = if done { "Reviewed  (M)" } else { "Mark reviewed  (M)" };
            let mut btn = egui::Button::new(RichText::new(label).strong());
            if done {
                btn = btn.fill(ui_kit::ACCENT());
            }
            if ui.add(btn)
                .on_hover_text(
                    "Bookmark this leaf as looked-at. Saved to disk, so a long batch \
                     can be worked over several sittings.\n\nUnlike Reject, it changes \
                     nothing about the export — it only drives progress and the N key.\n\n\
                     M toggles · N jumps to the next unreviewed leaf",
                )
                .clicked()
            {
                self.toggle_reviewed(li, toasts);
            }
        }

        if rejected {
            ui.label(RichText::new(format!("Leaf {li} rejected"))
                .color(Color32::from_rgb(220, 110, 110)).strong());
            if ui.button("Restore leaf  (X)")
                .on_hover_text("Put this leaf back into the run — its anomalies count and export again.\n\n\
                                Hotkey: X   ·   ← / → step between leaves")
                .clicked()
            {
                self.toggle_reject_leaf(li, toasts);
            }
        } else {
            let btn = egui::Button::new(
                RichText::new("Reject leaf  (X)").color(Color32::WHITE).strong(),
            ).fill(Color32::from_rgb(170, 55, 55));
            if ui.add(btn)
                .on_hover_text(format!(
                    "Throw leaf {li} out of the run.\n\n\
                     Its {n} anomalies stop being counted, it is left out of the CSV \
                     and the exported images, and it is never mined for training data.\n\n\
                     Reversible — press again to restore.\n\n\
                     Hotkey: X   ·   ← / → step between leaves",
                ))
                .clicked()
            {
                self.toggle_reject_leaf(li, toasts);
            }
        }
    }

    /// Icon-only toolbox, rendered INSIDE the folders panel (`show_controls`)
    /// — not a separate docked panel. Switching tool re-defaults the stamp
    /// label so it always matches the new tool's intent. Tool-specific
    /// settings (label/tile/zoom/undo) live in the options bar above the
    /// canvas (`show_canvas_options_bar`), not here.
    fn show_toolbox(&mut self, ui: &mut Ui) {
        ui_kit::section_header(ui, "Tools");
        let mut switched_to: Option<CanvasTool> = None;
        // The key is PRINTED under the icon, not just mentioned in the tooltip.
        // Measured shortcut adoption is ~10% even among experienced users, and the
        // reliable transfer mechanism is showing the binding at the point of the
        // mouse action — a tooltip is still hidden until you go looking.
        // Each tool is ONE fixed-size cell. `horizontal_wrapped` decides whether to
        // wrap from the width a child reports, and a nested `ui.vertical` only
        // reports its width after laying out its contents — so the row overflowed
        // the 300px panel and the last tool was clipped instead of wrapping to a
        // second line. Allocating the cell up front gives the wrap logic a width
        // it can act on before the content exists.
        // Vertical rhythm, all of it explicit — see the `item_spacing.y = 0`
        // below, which makes the pitch exactly CELL.y with nothing added behind
        // my back.
        //
        // The numbers encode GROUPING, not just size. A key label 2px under its
        // button and 6px above the next one sits almost equidistant between the
        // two, so it reads as floating in the gap rather than as belonging to
        // the button above it, and the whole rail looks like one clamped stack.
        // Tight above (LABEL_GAP), roomy below (the remainder), so each
        // button+key is unambiguously one unit.
        const BTN_H: f32 = 28.0;
        const LABEL_GAP: f32 = 2.0;
        const CELL: egui::Vec2 = egui::vec2(34.0, 56.0);
        // Both earlier attempts at this (wrapped rows, then an explicit vertical
        // with a width clamp) still drifted, so nothing here derives an x from
        // layout any more. Every cell is placed at an x pinned to the panel's
        // own left edge, captured ONCE before the first tool is drawn. Whatever
        // was accumulating — measured widths, scrollbar reservation, fractional
        // rounding under a non-integer zoom factor — cannot accumulate through a
        // constant.
        const BTN_W: f32 = 30.0;

        // The tools as DATA, so the column can be laid out arithmetically.
        // (tool, icon, name, tooltip, keymap id)
        let tools: [(CanvasTool, &str, &str, &str, &str); 9] = [
            (CanvasTool::Select, icon::CURSOR, "Select",
             "Click to select · drag to box-select · ctrl+click to multi-select · right-click for actions.",
             "tool.select"),
            (CanvasTool::MarkHealthy, icon::CHECK_CIRCLE, "Mark Healthy",
             "Stamp a patch straight off the canvas as a HEALTHY training example — teaches the \
              model this texture is not an anomaly (e.g. a vein it sometimes confuses with necrosis).",
             "tool.mark_healthy"),
            (CanvasTool::Brush, icon::PAINT_BRUSH, "Brush",
             "Paint a freeform region using a cluster's color — extends that cluster's region if the \
              stroke touches one, or creates a new region otherwise.",
             "tool.brush"),
            (CanvasTool::Eraser, icon::ERASER, "Eraser",
             "Paint over region pixels to remove them — shrinks the region, or removes it entirely \
              if nothing's left.",
             "tool.eraser"),
            (CanvasTool::Knife, icon::KNIFE, "Knife",
             "Drag a straight line starting and ending OUTSIDE any region to split whatever it \
              crosses into two. Drag a freeform loop starting INSIDE a region to carve that piece \
              out on its own.",
             "tool.knife"),
            (CanvasTool::Scissor, icon::SCISSORS, "Scissor",
             "Click to place vertices instead of dragging — precise, deliberate cuts. Click \
              back on the FIRST vertex to close the loop and carve it out (needs the first click \
              inside a region); otherwise press Enter to cut along the open polyline like a bent knife.",
             "tool.scissor"),
            (CanvasTool::Lasso, icon::LASSO, "Lasso select",
             "Drag a freeform outline; every region whose center falls inside it is added to the \
              selection (feeds the same Confirm/Reject/Reassign actions as box-select).",
             "tool.lasso"),
            (CanvasTool::Wand, icon::MAGIC_WAND, "Wand",
             "Click a pixel to grow a mask outward by color similarity (shift-click adds another \
              blob) — review the pending selection, then \"Fill\" to label + commit it, or \"Clear\" \
              to discard. A fast alternative to hand-painting with the Brush.",
             "tool.wand"),
            (CanvasTool::Polygon, icon::POLYGON, "Polygon",
             "Click to place nodes, click the first node again to close the shape and fill it. \
              With a region selected, it extends that region's own cluster; with nothing \
              selected, you'll be asked which cluster to assign.",
             "tool.polygon"),
        ];

        // ONE allocation for the whole column, then every cell's rect is
        // `top + i * CELL.y`. No per-cell allocation, no `ui.put` (which
        // allocates its rect in the parent and so fought the cursor this code
        // had already advanced), no measured widths anywhere — the position of
        // tool i does not depend on tool i-1 at all, so nothing can accumulate.
        let (col, _) = ui.allocate_exact_size(
            egui::vec2(CELL.x, CELL.y * tools.len() as f32),
            egui::Sense::hover(),
        );
        let painter = ui.painter().clone();
        for (i, (tool, ic, name, tip, key_id)) in tools.iter().enumerate() {
            let active = self.canvas_tool == *tool;
            let key = shortcuts::key_label(self.keymap.key(key_id));
            let btn = egui::Rect::from_min_size(
                egui::pos2(
                    (col.left() + (CELL.x - BTN_W) / 2.0).round(),
                    (col.top() + CELL.y * i as f32).round(),
                ),
                egui::vec2(BTN_W, BTN_H),
            );
            let resp = ui
                .interact(btn, ui.id().with(("tool", i)), egui::Sense::click())
                .on_hover_text(format!("{name}  ({key})\n{tip}"));

            // Painted by hand rather than via `Button`, so the visuals follow
            // the rect instead of the rect following the widget.
            let vis = ui.style().interact(&resp);
            let (fill, fg) = if active {
                (ui_kit::ACCENT(), ui_kit::on_accent())
            } else {
                (vis.bg_fill, vis.fg_stroke.color)
            };
            painter.rect(btn, egui::Rounding::same(5.0), fill, vis.bg_stroke);
            painter.text(btn.center(), egui::Align2::CENTER_CENTER, ic,
                         egui::FontId::proportional(15.0), fg);
            painter.text(
                egui::pos2(btn.center().x, btn.bottom() + LABEL_GAP),
                egui::Align2::CENTER_TOP,
                key,
                // 8.5px was under the 9px floor the type scale sets for Small,
                // which exists because smaller is not reliably readable.
                egui::FontId::proportional(9.5),
                if active { ui_kit::ACCENT() } else { ui_kit::MUTED() },
            );
            if resp.clicked() && self.canvas_tool != *tool {
                self.canvas_tool = *tool;
                switched_to = Some(*tool);
            }
        }

        if let Some(tool) = switched_to {
            self.on_tool_switched(tool);
        }
    }

    /// The active tool's own settings (brush size, stamp label, wand tolerance…),
    /// plus zoom.
    ///
    /// Split out of `show_toolbox` when the tools moved into a 52px rail: these
    /// are wide, wordy controls and rendering them in a 52px column wrapped the
    /// help text to roughly one word per line. They now sit above the canvas,
    /// where there is room and where they are still visible when the settings
    /// column is collapsed — a Brush with no reachable size control would be
    /// useless.
    fn show_tool_options(&mut self, ui: &mut Ui) {
        ui.add_space(2.0);
        // Theme-derived, not a hardcoded near-black: from_gray(28) painted a black
        // rectangle in the middle of a LIGHT panel, with grey helper text inside
        // it — the most obviously broken thing a student picking a light theme
        // would see. `faint_bg_color` is the theme's own "slightly inset surface"
        // and reads correctly on both polarities.
        let opts_bg = ui.visuals().faint_bg_color;
        egui::Frame::none()
            .fill(opts_bg)
            .inner_margin(egui::Margin::same(6.0))
            .rounding(egui::Rounding::same(3.0))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                match self.canvas_tool {
                    CanvasTool::Select => {
                        ui.label(RichText::new(
                            "click = select\ndrag = box-select\nctrl+click = multi-select\nright-click = menu"
                        ).small().color(Color32::GRAY));
                    }
                    CanvasTool::MarkHealthy => {
                        ui.label(RichText::new("Label:").small());
                        if !self.clusters.is_empty() || self.cached_head().is_some() {
                            let current = self.hardneg_label.clone();
                            if let Some(id) = self.cluster_picker_rows(ui, &current) {
                                self.hardneg_label = self.class_display_name(id);
                            }
                            ui.add_space(4.0);
                        }
                        ui.label(RichText::new("or type (\"healthy\"/blank = healthy example):").small().color(Color32::GRAY));
                        ui.add(egui::TextEdit::singleline(&mut self.hardneg_label)
                            .desired_width(ui.available_width())
                            .hint_text("healthy"));
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Tile").small());
                            ui.add(egui::DragValue::new(&mut self.hardneg_tile).range(16..=256).speed(2));
                        });
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Zoom").small());
                            ui.add(egui::Slider::new(&mut self.hardneg_zoom, 1.5..=8.0).fixed_decimals(1));
                        });
                        let cur_stamps = self.selected_idx
                            .and_then(|i| self.hardneg_stamps.get(&i))
                            .map(|v| v.len()).unwrap_or(0);
                        ui.add_enabled_ui(cur_stamps > 0, |ui| {
                            if ui.small_button(format!("Undo ({cur_stamps})"))
                                .on_hover_text("Undo the last stamp on this leaf.")
                                .clicked()
                            {
                                if let Some(i) = self.selected_idx {
                                    self.undo_hardneg(i);
                                }
                            }
                        });
                        ui.label(RichText::new("click = stamp\nright-click/Ctrl+Z = undo")
                            .small().color(Color32::GRAY));
                    }
                    CanvasTool::Brush => {
                        ui.label(RichText::new("Cluster:").small());
                        // Primary picker: one clickable colored row per existing
                        // cluster (swatch + name), selects it directly — no typing
                        // needed for the common case of painting more of an
                        // already-detected family.
                        if !self.clusters.is_empty() {
                            let current = self.hardneg_label.clone();
                            if let Some(id) = self.cluster_picker_rows(ui, &current) {
                                self.hardneg_label = self.class_display_name(id);
                            }
                            ui.add_space(4.0);
                        }
                        // secondary: type a brand new cluster name instead
                        ui.label(RichText::new("or new:").small().color(Color32::GRAY));
                        ui.add(egui::TextEdit::singleline(&mut self.hardneg_label)
                            .desired_width(ui.available_width())
                            .hint_text("type a new cluster name"));
                        ui.horizontal(|ui| {
                            if ui.selectable_label(self.brush_shape == BrushShape::Circle, icon::CIRCLE).clicked() {
                                self.brush_shape = BrushShape::Circle;
                            }
                            if ui.selectable_label(self.brush_shape == BrushShape::Square, icon::SQUARE).clicked() {
                                self.brush_shape = BrushShape::Square;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Size").small());
                            ui.add(egui::Slider::new(&mut self.brush_size, 4..=200));
                        });
                        ui.label(RichText::new(
                            "drag to paint (ctrl+scroll = resize); touches an existing region of this \
                             cluster → extends it, otherwise creates a new one"
                        ).small().color(Color32::GRAY));
                    }
                    CanvasTool::Eraser => {
                        ui.horizontal(|ui| {
                            if ui.selectable_label(self.brush_shape == BrushShape::Circle, icon::CIRCLE).clicked() {
                                self.brush_shape = BrushShape::Circle;
                            }
                            if ui.selectable_label(self.brush_shape == BrushShape::Square, icon::SQUARE).clicked() {
                                self.brush_shape = BrushShape::Square;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Size").small());
                            ui.add(egui::Slider::new(&mut self.brush_size, 4..=200));
                        });
                        ui.label(RichText::new(
                            "drag over a region to erase those pixels (ctrl+scroll = resize); a region \
                             erased down to nothing is removed entirely"
                        ).small().color(Color32::GRAY));
                    }
                    CanvasTool::Knife => {
                        ui.label(RichText::new(
                            "straight drag starting/ending OUTSIDE any region: splits whatever it \
                             crosses into two\n\
                             freeform loop starting INSIDE a region: carves that piece out on its own"
                        ).small().color(Color32::GRAY));
                    }
                    CanvasTool::Scissor => {
                        ui.label(RichText::new(
                            "click to place vertices — first click INSIDE a region, click back on \
                             it again to close the loop and carve it out\n\
                             first click OUTSIDE any region: press Enter to cut along the open \
                             polyline instead, like a bent knife\n\
                             Esc/right-click cancels the pending path"
                        ).small().color(Color32::GRAY));
                    }
                    CanvasTool::Lasso => {
                        ui.label(RichText::new(
                            "drag a freeform outline; release to select every region whose center falls inside it"
                        ).small().color(Color32::GRAY));
                    }
                    CanvasTool::Wand => {
                        ui.label(RichText::new("Cluster:").small());
                        if !self.clusters.is_empty() {
                            let current = self.hardneg_label.clone();
                            if let Some(id) = self.cluster_picker_rows(ui, &current) {
                                self.hardneg_label = self.class_display_name(id);
                            }
                            ui.add_space(4.0);
                        }
                        ui.label(RichText::new("or new:").small().color(Color32::GRAY));
                        ui.add(egui::TextEdit::singleline(&mut self.hardneg_label)
                            .desired_width(ui.available_width())
                            .hint_text("type a new cluster name"));
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Tolerance").small());
                            ui.add(egui::Slider::new(&mut self.wand_tolerance, 1.0..=50.0));
                        });
                        ui.label(RichText::new(
                            "click to grow a mask by color similarity (shift-click adds another \
                             blob); review the pending selection on the canvas, then Fill or Clear"
                        ).small().color(Color32::GRAY));
                    }
                    CanvasTool::Polygon => {
                        ui.label(RichText::new(
                            "click = place node\nclick first node = close + fill\nEsc = cancel"
                        ).small().color(Color32::GRAY));
                    }
                }
            });

    }

    /// Which of the four real phases the worker is in, from the stage string it
    /// actually sends.
    ///
    /// The old five-step bar was `["Segment","Tile","Detect","Restitch","Done"]`
    /// matched with `stage.starts_with(step)` — but the worker never emits a
    /// stage beginning "Tile" or "Restitch", so two of five steps could NEVER
    /// light. Worse, "Loading models", "Reconstruct", "Morphology" and
    /// "Clustering" matched nothing at all, so during those every step rendered
    /// grey. Clustering runs after the last image, over the whole batch, which
    /// meant the end of a long run showed a frozen bar, an all-grey stepper and a
    /// spinner — indistinguishable from a hang, at exactly the point where
    /// killing the run is most expensive.
    fn current_phase(&self) -> usize {
        let s = self.stage.as_str();
        if s.is_empty() { return 0; }
        if s == "Done" { return 3; }
        if s.starts_with("Clustering") { return 2; }
        if s.starts_with("Detect") || s.starts_with("Reconstruct") || s.starts_with("Morphology") {
            return 1;
        }
        0 // Loading models, Leaf … (pre-cut), Segment …
    }

    fn show_stepper(&mut self, ui: &mut Ui) {
        // Numbered pills with a completion tick, as in the mockup. The stage bar
        // is the app's answer to "where am I and what is left", so it should read
        // as a route rather than as four grey words.
        const PHASES: [&str; 4] = ["Leaves", "Detect", "Review", "Export"];
        let phase = self.current_phase();
        let done_all = !self.results.is_empty() && !self.running;
        ui.horizontal_centered(|ui| {
            ui.add_space(4.0);
            for (i, s) in PHASES.iter().enumerate() {
                // "Review" and "Export" are reachable once there are results;
                // the first two describe work the worker does, not places to go.
                let complete = i < phase || (done_all && i < 2);
                let current = i == phase && (self.running || done_all);
                let (fg, bg) = if current {
                    (ui_kit::on_accent(), ui_kit::ACCENT())
                } else if complete {
                    (ui_kit::ACCENT(), ui_kit::ACCENT().linear_multiply(0.14))
                } else {
                    (ui_kit::MUTED(), Color32::TRANSPARENT)
                };
                let text = if complete && !current {
                    format!("{}  {s}", i + 1)
                } else {
                    format!("{}  {s}", i + 1)
                };
                let galley = ui.painter().layout_no_wrap(
                    text.clone(), egui::FontId::proportional(12.5), fg);
                let tick_w = if complete { 16.0 } else { 0.0 };
                let size = egui::vec2(galley.size().x + 22.0 + tick_w, 23.0);
                let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
                if bg != Color32::TRANSPARENT {
                    ui.painter().rect_filled(rect, 11.0, bg);
                }
                ui.painter().galley(
                    egui::pos2(rect.center().x - galley.size().x / 2.0,
                               rect.center().y - galley.size().y / 2.0),
                    galley, fg);
                if complete {
                    // drawn tick, not a glyph — U+2713 is not in the bundled fonts
                    let c = egui::pos2(rect.right() - 9.0, rect.center().y);
                    let st = egui::Stroke::new(1.6, fg);
                    ui.painter().line_segment([c + egui::vec2(-3.0, 0.0), c + egui::vec2(-1.0, 2.5)], st);
                    ui.painter().line_segment([c + egui::vec2(-1.0, 2.5), c + egui::vec2(3.0, -3.0)], st);
                }
                if resp.clicked() {
                    match i {
                        0 => self.setup_open = true,
                        2 => self.stage_view = StageView::Review,
                        3 => self.stage_view = StageView::Done,
                        _ => {}
                    }
                }
                let _ = resp.on_hover_text(match i {
                    0 => "Choose the photographs and where results go.",
                    1 => "Segment and detect — runs automatically.",
                    2 => "Judge the detections. You are here for most of the batch.",
                    _ => "Write results.csv and the images.",
                });
                if i < PHASES.len() - 1 {
                    ui.add_space(2.0);
                }
            }
            if !self.stage.is_empty() {
                ui.separator();
                ui.label(RichText::new(&self.stage).small().color(ui_kit::MUTED()));
            }
            // Throughput and remaining time, right-aligned. "How long is this
            // going to take" is the single most-wanted number during a
            // multi-hour batch and appeared nowhere in the app.
            if self.running {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(8.0);
                    if let Some(eta) = self.eta_text() {
                        ui.label(RichText::new(eta).text_style(ui_kit::numeric()).color(ui_kit::MUTED()));
                    }
                });
            }
        });
    }

    /// The region indices the Curate gallery is currently showing, in the order
    /// it shows them. Extracted so keyboard stepping and the grid cannot disagree
    /// about what "next" means.
    fn gallery_order(&self) -> Vec<usize> {
        let mut v: Vec<usize> = (0..self.regions.len())
            .filter(|&i| {
                self.region_visible(i)
                    && self.selected_cluster.map_or(true, |c| self.labels[i] == c)
                    && (!self.filter_leaf_only
                        || self.selected_idx.map_or(true, |li| self.regions[i].leaf == li))
                    && (!self.filter_flagged || self.flagged.contains(&i))
            })
            .collect();
        match self.gallery_sort {
            GallerySort::Largest => {
                v.sort_by_key(|&i| std::cmp::Reverse(self.region_area.get(i).copied().unwrap_or(0)));
            }
            GallerySort::Unusual => {
                let score = self.atypicality();
                v.sort_by(|&a, &b| {
                    score.get(&b).unwrap_or(&0.0)
                        .partial_cmp(score.get(&a).unwrap_or(&0.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
        v
    }

    /// Regions the "Rank by appearance" action would embed: the focused family,
    /// narrowed to the current leaf when the leaf filter is on.
    ///
    /// Already-embedded regions are excluded, so pressing it twice costs nothing
    /// and it can be used to top up after new regions appear.
    fn rank_targets(&self) -> Vec<usize> {
        let Some(cid) = self.selected_cluster else { return Vec::new() };
        (0..self.regions.len())
            .filter(|&i| {
                self.region_visible(i)
                    && self.labels[i] == cid
                    && self.regions[i].dino_embed.is_empty()
                    && (!self.filter_leaf_only
                        || self.selected_idx.map_or(true, |li| self.regions[i].leaf == li))
            })
            .collect()
    }

    fn start_rank_appearance(&mut self, toasts: &mut ToastManager) {
        let Some(dino) = self.eff_dino() else {
            toasts.error("Set the DINO model first.");
            return;
        };
        let targets = self.rank_targets();
        if targets.is_empty() {
            toasts.info("Nothing to rank — this family is already embedded.");
            return;
        }
        // Group by leaf so each leaf's pixel buffer crosses the thread boundary
        // once instead of once per region.
        let mut by_leaf: HashMap<usize, Vec<RankRegion>> = HashMap::new();
        for i in targets {
            let r = &self.regions[i];
            by_leaf.entry(r.leaf).or_default().push(RankRegion {
                idx: i,
                centroid: region_centroid(r),
                bbox_leaf: r.bbox_leaf,
                mask: r.mask.clone(),
            });
        }
        let leaves: Vec<RankLeaf> = by_leaf.into_iter()
            .filter_map(|(li, regions)| {
                let l = self.results.get(li)?;
                Some(RankLeaf { rgba: l.rgba.clone(), w: l.w, h: l.h, regions })
            })
            .collect();
        let total: usize = leaves.iter().map(|l| l.regions.len()).sum();

        let (tx, rx) = mpsc::channel();
        self.rank_rx = Some(rx);
        self.rank_cancel = Arc::new(AtomicBool::new(false));
        self.ranking = true;
        self.rank_done = 0;
        self.rank_total = total;
        spawn_rank(dino, 512, leaves, tx, self.rank_cancel.clone());
    }

    fn poll_rank(&mut self, toasts: &mut ToastManager) {
        let msgs: Vec<RankMsg> = match &self.rank_rx {
            Some(rx) => rx.try_iter().collect(),
            None => return,
        };
        for m in msgs {
            match m {
                RankMsg::Progress { done, total } => {
                    self.rank_done = done;
                    self.rank_total = total;
                }
                RankMsg::Done(pairs) => {
                    let n = pairs.len();
                    for (i, e) in pairs {
                        if let Some(r) = self.regions.get_mut(i) {
                            r.dino_embed = e;
                        }
                    }
                    self.ranking = false;
                    self.rank_rx = None;
                    // Switch the gallery to the ordering this just made possible,
                    // otherwise the work is invisible.
                    self.gallery_sort = GallerySort::Unusual;
                    self.gallery_page = 0;
                    toasts.success(format!(
                        "Ranked {n} regions by appearance — the least typical are now first."
                    ));
                }
                RankMsg::Error(e) => {
                    self.ranking = false;
                    self.rank_rx = None;
                    toasts.error(format!("Appearance ranking failed: {e}"));
                }
            }
        }
    }

    /// How unlike its own family each visible region is, 0 = typical.
    ///
    /// Uses the mask-aware DINO embedding when the run computed one
    /// (`unsupervised_families`): cosine distance from the family's centroid,
    /// which is a real measure of "this doesn't look like its siblings".
    ///
    /// Falls back to relative area otherwise — a much weaker signal, but the
    /// embeddings are only populated on one code path and an ordering that
    /// silently does nothing would be worse than an honest approximation. The UI
    /// says which one is in force.
    fn atypicality(&self) -> HashMap<usize, f32> {
        let mut out = HashMap::new();
        let have_embeds = self.regions.iter().any(|r| !r.dino_embed.is_empty());

        if have_embeds {
            // family centroid over the regions that have an embedding
            let mut sums: HashMap<i32, (Vec<f32>, usize)> = HashMap::new();
            for (i, r) in self.regions.iter().enumerate() {
                if r.dino_embed.is_empty() || !self.region_visible(i) { continue; }
                let e = sums.entry(self.labels[i]).or_insert_with(|| (vec![0.0; r.dino_embed.len()], 0));
                for (s, v) in e.0.iter_mut().zip(&r.dino_embed) { *s += *v; }
                e.1 += 1;
            }
            for (i, r) in self.regions.iter().enumerate() {
                if r.dino_embed.is_empty() || !self.region_visible(i) { continue; }
                let Some((sum, n)) = sums.get(&self.labels[i]) else { continue };
                if *n == 0 { continue; }
                // Embeddings are L2-normalized, so a dot with the (unnormalized)
                // mean is monotone in cosine similarity — enough for ordering.
                let dot: f32 = sum.iter().zip(&r.dino_embed).map(|(a, b)| a * b).sum();
                let norm = (sum.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-6);
                out.insert(i, 1.0 - dot / norm);
            }
            return out;
        }

        // fallback: distance from the family's median area, in relative terms
        let mut by_fam: HashMap<i32, Vec<u32>> = HashMap::new();
        for i in 0..self.regions.len() {
            if !self.region_visible(i) { continue; }
            by_fam.entry(self.labels[i]).or_default().push(self.region_area.get(i).copied().unwrap_or(0));
        }
        let med: HashMap<i32, f32> = by_fam.into_iter().map(|(k, mut v)| {
            v.sort_unstable();
            (k, v[v.len() / 2].max(1) as f32)
        }).collect();
        for i in 0..self.regions.len() {
            if !self.region_visible(i) { continue; }
            let a = self.region_area.get(i).copied().unwrap_or(0) as f32;
            let m = med.get(&self.labels[i]).copied().unwrap_or(1.0);
            out.insert(i, ((a / m).max(m / a.max(1.0)) - 1.0).max(0.0));
        }
        out
    }

    /// What the app's bottom strip should say while this tab is open.
    ///
    /// Answers the questions a long session actually raises — where am I, how much
    /// is left, is my work saved — instead of repeating the tab's own title.
    pub fn status_line(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.results.is_empty() {
            out.push(if self.running { "Running…".into() } else { "No run loaded".into() });
        } else {
            let (rev, rej, tot) = self.review_counts();
            let left = tot.saturating_sub(rev + rej);
            out.push(format!("{tot} leaves"));
            out.push(format!("{rev} reviewed · {rej} rejected · {left} to go"));
            if let Some(i) = self.selected_idx {
                out.push(format!("on leaf {}", i + 1));
            }
        }
        if self.running {
            if let Some(e) = self.eta_text() { out.push(e); }
        }
        if self.retraining { out.push("retraining".into()); }
        if self.mining     { out.push("mining".into()); }
        if self.ranking    { out.push(format!("ranking {}/{}", self.rank_done, self.rank_total)); }
        if let Some(j) = &self.export_job {
            let done = j.crop_cur + j.leaf_cur;
            out.push(format!("exporting images {done}/{}", j.total));
        }
        // Review marks and curations are written the moment they happen, so this
        // is a statement of fact rather than a save button's absence.
        if self.output_folder.is_some() && !self.results.is_empty() {
            out.push("saved".into());
        }
        out
    }

    /// "1,204 / 10,000 · 8.4/min · ~17 min left", or `None` before there is
    /// enough history to say anything honest.
    fn eta_text(&self) -> Option<String> {
        let started = self.run_started_at?;
        let done = self.progress_done;
        let total = self.progress_total;
        if done < 3 || total == 0 {
            // Deliberately silent rather than wrong: an estimate from one or two
            // images swings wildly and then "corrects" by minutes, which reads as
            // the app being confused.
            return Some(format!("{done} / {total}"));
        }
        let elapsed = started.elapsed().as_secs_f64();
        if elapsed <= 0.0 { return None; }
        let per = elapsed / done as f64;
        let remaining = per * (total.saturating_sub(done)) as f64;
        Some(format!(
            "{done} / {total} · {:.1}/min · ~{} left",
            60.0 / per.max(1e-6),
            humanize_secs(remaining),
        ))
    }

    /// Shown in place of everything else until the app can actually run.
    ///
    /// The first thing a new user met was a greyed-out "Run Pipeline", one line of
    /// grey text naming missing files, and no visible way to supply them — the
    /// pickers live in Settings → Pipeline, on a different screen, with nothing
    /// linking there. That is a dead end at step one, and it is probably the
    /// single largest contributor to "the workflow is not intuitive".
    ///
    /// Returns true when it took over the panel.
    fn show_setup_card(&mut self, ui: &mut Ui) -> bool {
        let missing_models = !self.all_paths_ok();
        let missing_folders = self.source_folder.is_none() || self.output_folder.is_none();
        if !missing_models && !missing_folders {
            return false;
        }
        ui.add_space(6.0);
        ui.label(RichText::new("Start an analysis").text_style(ui_kit::subhead()).strong());
        ui.label(RichText::new(
            "Choose a folder of photographs. Everything else has a working default.")
            .small().color(ui_kit::MUTED()));
        ui.add_space(10.0);

        // ── one decision ────────────────────────────────────────────────────
        // Picking a source folder derives the output folder beside it, so the
        // common case is a single choice rather than two. Asking for an output
        // location is a question with an obvious answer, and questions with
        // obvious answers are the ones worth not asking.
        if ui_kit::primary_button(ui, "Choose a folder of photographs…").clicked() {
            if self.pick_rx.is_none() {
                self.pick_rx = Some((Pick::Source, spawn_dialog(Pick::Source)));
            }
        }
        if let Some(src) = self.source_folder.clone() {
            ui.add_space(4.0);
            ui.label(RichText::new(src.display().to_string()).small().color(ui_kit::MUTED()));
            ui.label(
                RichText::new(format!("{} images found (including sub-folders)", self.source_count))
                    .small()
                    .color(if self.source_count > 0 { ui_kit::ACCENT() } else { Color32::from_rgb(220, 150, 130) }),
            );
            if self.output_folder.is_none() {
                if let Some(derived) = derived_output_for(&src) {
                    self.output_folder = Some(derived);
                }
            }
        }
        if let Some(out) = self.output_folder.clone() {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new("Results").small().color(ui_kit::MUTED()));
                if ui.small_button("change")
                    .on_hover_text("Put the results somewhere else. By default they go in a \
                                    folder beside the photographs.")
                    .clicked()
                {
                    if self.pick_rx.is_none() {
                        self.pick_rx = Some((Pick::Output, spawn_dialog(Pick::Output)));
                    }
                }
            });
            ui.label(RichText::new(out.display().to_string()).small().color(ui_kit::MUTED()));
        }
        ui.add_space(10.0);
        ui.separator();
        ui.add_space(6.0);

        // Models: normally already resolved from the bundled folder, so this only
        // becomes visible when something is genuinely missing. It used to be the
        // blocking step with its pickers on a different screen entirely.
        if missing_models {
            ui.label(RichText::new("Models").strong());
            ui.label(RichText::new(
                "Normally filled in automatically from the models/ folder next to \
                 Lacuna. Set whichever is missing:")
                .small().color(ui_kit::MUTED()));
            self.pick_row(ui, "Leaf segmentation (YOLO)", Pick::Yolo);
            self.pick_row(ui, "Features (DINO)", Pick::Dino);
            self.pick_row(ui, "Anomaly classifier (head)", Pick::Head);
            ui.add_space(8.0);
        }

        // The green button the mockup calls for: it lights up the moment the
        // folder has images, and states what it is about to do.
        let ready = self.source_count > 0 && !missing_models
            && self.output_folder.is_some() && self.output_inside_source().is_none();
        ui.add_enabled_ui(ready && !self.running, |ui| {
            if ui_kit::primary_button(ui, &format!("Analyse {} photographs", self.source_count)).clicked() {
                self.start();
            }
        });
        if !ready {
            ui.label(RichText::new(if self.source_folder.is_none() {
                "Choose a folder to begin."
            } else if self.source_count == 0 {
                "No images in that folder — pick another."
            } else if missing_models {
                "One or more models still need setting."
            } else {
                "Output folder overlaps the source folder — choose another."
            }).small().color(ui_kit::MUTED()));
        }
        true
    }

    /// Does the active tool have anything to configure? Select, Knife, Scissor
    /// and Polygon do not, and a popover that appears empty is worse than none.
    fn tool_has_options(&self) -> bool {
        matches!(
            self.canvas_tool,
            CanvasTool::MarkHealthy | CanvasTool::Brush | CanvasTool::Eraser | CanvasTool::Wand
        )
    }

    fn show_tool_options_popover(&mut self, ctx: &Context) {
        if !self.tool_has_options() {
            return;
        }
        egui::Area::new(egui::Id::new("tool_options_popover"))
            .order(egui::Order::Foreground)
            .anchor(egui::Align2::LEFT_TOP, egui::vec2(58.0, 104.0))
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style())
                    .inner_margin(egui::Margin::symmetric(10.0, 8.0))
                    .show(ui, |ui| {
                        ui.set_max_width(230.0);
                        let name = match self.canvas_tool {
                            CanvasTool::MarkHealthy => "Mark healthy",
                            CanvasTool::Brush       => "Brush",
                            CanvasTool::Eraser      => "Eraser",
                            CanvasTool::Wand        => "Magic wand",
                            _ => "Tool",
                        };
                        ui.label(RichText::new(name).small().strong().color(ui_kit::MUTED()));
                        ui.add_space(2.0);
                        self.show_tool_options(ui);
                    });
            });
    }

    /// The finish screen — the mockup's fourth stage.
    ///
    /// Every task needs an unambiguous ending; a review session that just trails
    /// off leaves you unsure whether you finished. This states what happened,
    /// offers the export, and makes the one worthwhile follow-on — teaching the
    /// model from your corrections — an offer rather than a control panel.
    fn show_done_screen(&mut self, ui: &mut Ui, toasts: &mut ToastManager) {
        let (rev, rej, tot) = self.review_counts();
        let confirmed = self.persisted.len();
        let aside = self.flagged.len();
        let corrections = self.removed.len();
        let agree = if confirmed + corrections > 0 {
            100.0 * confirmed as f32 / (confirmed + corrections) as f32
        } else { 0.0 };

        ui.add_space(6.0);
        ui.label(RichText::new("Review complete").text_style(ui_kit::display()).strong());
        ui.add_space(12.0);

        let stat = |ui: &mut Ui, v: String, k: &str, col: Color32| {
            ui.vertical(|ui| {
                ui.label(RichText::new(v).text_style(ui_kit::numeric()).size(22.0).color(col));
                ui.label(RichText::new(k).small().color(ui_kit::MUTED()));
            });
        };
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 26.0;
            stat(ui, fmt_thousands(tot), "leaves", ui.visuals().text_color());
            stat(ui, fmt_thousands(rev), "reviewed", ui.visuals().text_color());
            stat(ui, fmt_thousands(rej), "rejected", Color32::from_rgb(212, 121, 74));
            stat(ui, fmt_thousands(confirmed), "confirmed", ui.visuals().text_color());
            if confirmed + corrections > 0 {
                stat(ui, format!("{agree:.1}%"), "model agreed", ui_kit::ACCENT());
            }
            if aside > 0 {
                stat(ui, fmt_thousands(aside), "set aside", Color32::from_rgb(225, 180, 90));
            }
        });

        ui.add_space(14.0);
        ui.separator();
        ui.add_space(10.0);

        // Read the counters out before the closure — borrowing `export_job` while
        // the closure also wants to clear it is a unique-access conflict.
        let job = self.export_job.as_ref().map(|j| (j.crop_cur + j.leaf_cur, j.total));
        if let Some((done, total_imgs)) = job {
            let mut stop = false;
            ui.horizontal(|ui| {
                ui_kit::busy(ui, &format!("writing images {done}/{total_imgs}"));
                // Export CAN be cancelled now. The CSV is already down, so
                // stopping costs only images.
                if ui.button("Stop").clicked() {
                    stop = true;
                }
            });
            ui.add(egui::ProgressBar::new(done as f32 / total_imgs.max(1) as f32)
                .desired_height(5.0));
            if stop {
                self.export_job = None;
                toasts.info("Stopped — results.csv was already written.");
            }
        } else if ui_kit::primary_button(ui, "Export measurements").clicked() {
            self.export_results(toasts);
        }
        ui.label(RichText::new(
            "results.csv, plus the images you ticked under Export. A provenance \
             line records the model, thresholds and which rows you verified.")
            .small().color(ui_kit::MUTED()));

        // Table shape. Placed with the export controls rather than in settings
        // because it changes what the file IS, and that is a decision made at
        // the moment of writing it.
        ui.add_space(10.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Table shape").small().color(ui_kit::MUTED()));
            ui.selectable_value(&mut self.export_wide, false, "Long")
                .on_hover_text("ONE ROW PER ANOMALY.\n\n\
                                Every region with its own area, bounding box, recon % \
                                and family, with the leaf's morphology repeated on each \
                                row. Use when the anomaly is the unit of analysis, or \
                                when you want to filter/aggregate yourself.");
            ui.selectable_value(&mut self.export_wide, true, "Wide")
                .on_hover_text("ONE ROW PER LEAF.\n\n\
                                Leaf morphology once, plus four columns per family: \
                                count, total area, average area and % of leaf. Use when \
                                the leaf is the sampling unit — this is the shape that \
                                joins directly to per-leaf field data with no pivot.");
        });
        if self.export_wide {
            let fams = self.clusters.iter().filter(|c| c.id >= 0).count();
            ui.label(RichText::new(format!(
                "{} families x 4 columns. A family with no regions on a leaf gets 0 \
                 counts and a blank average.", fams))
                .small().color(ui_kit::MUTED()));
        }

        // The way BACK. The stage pill was the only route to review from here,
        // and a pill in a header does not read as a control — the finish screen
        // looked like a one-way door.
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            // No arrow glyph: U+2190 is not in the bundled fonts and rendered as
            // a tofu box. Same reason the stage pills draw their tick by hand.
            if ui.button("Back to review").clicked() {
                self.stage_view = StageView::Review;
            }
            if aside > 0 && ui.button(format!("Review the {aside} set aside")).clicked() {
                self.stage_view = StageView::Review;
                self.filter_flagged = true;
                self.gallery_page = 0;
            }
        });

        // Export options live here too — the Done screen is where someone decides
        // what to write, so making them go back to the review panel for the two
        // checkboxes that change what lands on disk is a pointless round trip.
        ui.add_space(10.0);
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.export_crops, "Anomaly crops")
                .on_hover_text("One small PNG per anomaly, into export/crops/.\n\n\
                                OFF by default: the CSV row and the leaf overlay already \
                                describe every region. Tick it only if you specifically need \
                                the individual crop images.");
            ui.checkbox(&mut self.export_overlays, "Leaf overlays")
                .on_hover_text("One FULL-SIZE PNG per leaf into export/leaves/, with every \
                                anomaly painted in its family colour.\n\n\
                                This is usually the most expensive part of an export — a \
                                10,000-leaf batch means 10,000 full-resolution encodes. Turn \
                                it off when you only need results.csv.");
        });
        // The export runs on the UI thread, so a big one freezes the window. Say
        // so rather than letting it look like a hang.
        let n_leaves = self.results.len().saturating_sub(self.rejected_leaves.len());
        if self.export_overlays && n_leaves > 500 {
            ui.label(RichText::new(format!(
                "{n_leaves} overlays to encode — this will take a while and the window \
                 will not respond until it finishes."
            )).small().color(Color32::from_rgb(220, 170, 90)));
        }

        // The flywheel, as one question rather than five controls.
        if corrections > 0 {
            ui.add_space(16.0);
            egui::Frame::none()
                .fill(ui.visuals().faint_bg_color)
                .inner_margin(egui::Margin::same(12.0))
                .rounding(egui::Rounding::same(5.0))
                .show(ui, |ui| {
                    ui.label(RichText::new(format!(
                        "You corrected the model {} times.", fmt_thousands(corrections)))
                        .strong());
                    ui.label(RichText::new(
                        "Teaching it from those corrections affects future runs only — this \
                         batch's results are already written.")
                        .small().color(ui_kit::MUTED()));
                    ui.add_space(6.0);
                    // Opens the panel BELOW, in place. It used to bounce you back
                    // to the review screen with a toast describing where to look,
                    // which is an instruction where a control belongs.
                    // Teaches, rather than revealing a panel that teaches.
                    //
                    // It used to only force the "Improve the model" section open
                    // below, which reads as doing nothing — the reporter was
                    // "confused why it doesn't teach immediately" and
                    // "overwhelmed by the choices". Every knob in that section
                    // has a working default, so the button now starts the
                    // retrain and opens the section so progress is visible.
                    // Nobody who just wants to train has to make a decision.
                    let can_teach = self.output_folder.is_some()
                        && self.eff_head().is_some()
                        && self.eff_dino().is_some()
                        && !self.retraining
                        && !self.running
                        && !self.mining;
                    let btn = ui.add_enabled(
                        can_teach,
                        egui::Button::new(if self.retraining {
                            "Teaching…"
                        } else {
                            "Teach the model"
                        }),
                    );
                    if btn
                        .on_hover_text(
                            "Retrain the head on everything you have curated, using the \
                             defaults below. Opens the section so you can watch it run.",
                        )
                        .clicked()
                    {
                        self.improve_open_req = true;
                        self.start_retrain();
                    }
                });
        }

        // Retraining lives here and only here — same moment as export, when the
        // run is done and the question is what to do with it.
        ui.add_space(14.0);
        self.show_improve_model(ui);
    }

    /// The flow screen: the only thing on screen before a run exists.
    fn show_start_screen(&mut self, ui: &mut Ui) {
        // `show_setup_card` returns false once everything is configured, in which
        // case it has drawn nothing — so draw the ready-to-run state instead.
        if !self.show_setup_card(ui) {
            ui.add_space(6.0);
            ui.label(RichText::new("Ready to analyse").text_style(ui_kit::subhead()).strong());
            ui.add_space(6.0);
            if let Some(src) = &self.source_folder {
                ui.label(RichText::new(src.display().to_string()).small().color(ui_kit::MUTED()));
            }
            ui.label(RichText::new(format!("{} photographs", self.source_count))
                .small().color(ui_kit::ACCENT()));
            ui.add_space(10.0);
            if ui_kit::primary_button(ui, &format!("Analyse {} photographs", self.source_count)).clicked() {
                self.start();
            }
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.small_button("Change folders…").clicked() { self.setup_open = true; }
            });
            ui.add_space(4.0);
            ui.label(RichText::new(
                "Results appear as each leaf finishes — you can start reviewing \
                 before the batch completes.")
                .small().color(ui_kit::MUTED()));
        }
    }

    /// Job configuration, in a window rather than a standing panel.
    fn show_setup_window(&mut self, ctx: &Context) {
        if !self.setup_open {
            return;
        }
        let mut open = self.setup_open;
        egui::Window::new("Run setup")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_width(360.0)
            .anchor(egui::Align2::LEFT_TOP, egui::vec2(70.0, 96.0))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().max_height(520.0)
                    .id_salt("setup_scroll")
                    .show(ui, |ui| self.show_controls_body(ui));
            });
        self.setup_open = open;
    }

    /// Folders, preview, calibration and Run — the content of the setup window.
    fn show_controls_body(&mut self, ui: &mut Ui) {
        ui_kit::section_header(ui, "Folders");
        self.pick_row(ui, "Source folder", Pick::Source);
        if self.source_folder.is_some() {
            ui.label(RichText::new(format!("{} images found", self.source_count)).small());
        }
        self.pick_row(ui, "Output folder", Pick::Output);

        // The standalone "Preview segmentation" button was REMOVED. It ran YOLO
        // on the first source image alone, which is not the pipeline: no tiling,
        // no margin erode, none of the per-leaf handling — so its cutout did not
        // resemble what a run actually produces, and its only advice was to tune
        // sliders that live somewhere else. Calibration below runs the real
        // detection path on a real leaf and is the thing to use instead.
        if !self.preview_note.is_empty() {
            ui.label(RichText::new(&self.preview_note).small().color(Color32::GRAY));
        }

        ui.add_space(10.0);
        ui_kit::section_header(ui, "Calibration (optional)");
        ui.label(RichText::new(
            "Mark a few example anomalies on a preview leaf to derive a model \
             tuned to this dataset — skip this and Run Pipeline just uses the \
             configured head/defaults as-is.")
            .small().color(Color32::GRAY));
        ui.horizontal(|ui| {
            let can_preview = self.eff_yolo().is_some() && self.source_count > 0 && !self.running;
            if ui.add_enabled(can_preview, egui::Button::new("Preview a leaf to mark")).clicked() {
                self.start_calibration_preview();
            }
        });
        // Matches exactly what `start_calibrate` will actually use — examples
        // persisted on THIS session's calibration-preview leaves, not the
        // output folder's entire curation history.
        let n_marked = self.persisted.iter()
            .filter(|&&i| self.regions.get(i).map_or(false, |r| self.calib_preview_leaves.contains(&r.leaf)))
            .count();
        if n_marked > 0 {
            ui.label(RichText::new(format!("{n_marked} example(s) marked so far")).small().color(Color32::GRAY));
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.calib_name)
                    .desired_width(110.0)
                    .hint_text("name"));
                let can_save = self.eff_head().is_some() && self.eff_dino().is_some()
                    && self.output_folder.is_some() && !self.calib_running;
                if ui.add_enabled(can_save, egui::Button::new("Save calibration")).clicked() {
                    self.start_calibrate();
                }
                if self.calib_running {
                    ui_kit::busy(ui, "calibrating…");
                    if ui.small_button("Cancel").clicked() {
                        self.calib_cancel.store(true, Ordering::Relaxed);
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label(RichText::new("Confidence").small())
                    .on_hover_text("Absolute assertiveness of calibrated classes (NOT relative \
                                     to the base head's own classes — matching the base head's \
                                     own scale is what caused the whole-leaf collapse bug, since \
                                     that scale can itself be unreasonably large). Default 4.0 is \
                                     a safe starting point; raise it if calibrated classes are \
                                     losing ties they should win, lower it if they're winning too \
                                     many.");
                ui.add(egui::Slider::new(&mut self.calib_scale, 0.5..=15.0).fixed_decimals(2));
            });
        }
        for line in self.calib_log.iter().rev().take(6) {
            ui.label(RichText::new(line).small().color(Color32::GRAY));
        }
        let calibrations = self.list_calibrations();
        if !calibrations.is_empty() {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Use:").small());
                let cur_label = self.calib_selected.as_ref()
                    .and_then(|p| calibrations.iter().find(|(cp, _, _)| cp == p))
                    .map(|(_, name, age)| format!("{name} ({})", format_age(*age)))
                    .unwrap_or_else(|| "None (configured head)".to_string());
                egui::ComboBox::from_id_salt("pipeline_use_calibration")
                    .selected_text(cur_label)
                    .show_ui(ui, |ui| {
                        if ui.selectable_label(self.calib_selected.is_none(), "None (configured head)").clicked() {
                            self.calib_selected = None;
                            self.head_path = None;
                            self.head_cache = None;
                        }
                        for (path, name, age) in &calibrations {
                            let sel = self.calib_selected.as_ref() == Some(path);
                            if ui.selectable_label(sel, format!("{name} ({})", format_age(*age))).clicked() {
                                self.calib_selected = Some(path.clone());
                                self.head_path = Some(path.clone());
                                self.head_cache = None;
                            }
                        }
                    });
            });
        }

        ui.add_space(10.0);
        // Blocking, not advisory: with the output inside the source folder the run
        // silently corrupts its own input, and the damage compounds every rerun.
        let folder_clash = self.output_inside_source();
        if let Some(why) = &folder_clash {
            egui::Frame::none()
                .fill(Color32::from_rgb(74, 30, 24))
                .inner_margin(egui::Margin::same(7.0))
                .rounding(egui::Rounding::same(4.0))
                .show(ui, |ui| {
                    ui.label(RichText::new("Output folder is inside the source folder")
                        .strong().color(Color32::from_rgb(240, 180, 160)));
                    ui.label(RichText::new(why).small().color(Color32::from_rgb(228, 196, 186)));
                });
            ui.add_space(6.0);
        }
        let can_start = self.all_paths_ok() && self.source_count > 0 && !self.running
            && !self.retraining && !self.mining && folder_clash.is_none();
        ui.add_enabled_ui(can_start, |ui| {
            if ui_kit::primary_button(ui, "Run Pipeline").clicked() {
                // Only ask when there is a session to lose. A first run, or one
                // where nothing has been reviewed, goes straight through — a
                // dialog that fires when nothing is at stake is how people learn
                // to click past dialogs that matter.
                let (rev, rej, _) = self.review_counts();
                if !self.results.is_empty() && (rev + rej > 0) {
                    self.pending_reset = Some(PendingReset::Rerun);
                } else {
                    self.start();
                }
            }
        });
        if self.running {
            if ui.add_sized([ui.available_width(), 26.0], egui::Button::new("Cancel")).clicked() {
                self.cancel_flag.store(true, Ordering::Relaxed);
            }
            let frac = if self.progress_total > 0 {
                self.progress_done as f32 / self.progress_total as f32
            } else { 0.0 };
            ui.add(egui::ProgressBar::new(frac).show_percentage());
            if !self.stage.is_empty() {
                ui.horizontal(|ui| ui_kit::busy(ui, &self.stage));
            }
        }
        if !can_start && !self.running {
            let need = if self.retraining {
                "Retraining in progress — Run is disabled until it finishes \
                 (both use the GPU/DINO and can't safely run together)."
            } else if self.fewshot_active() {
                "Set YOLO + DINO + few-shot head + source/output folders."
            } else {
                "Set YOLO + DINO + coreset bank + meta + source/output folders."
            };
            ui.label(RichText::new(need).small().color(Color32::GRAY));
        }

        // log moved to the right panel's "Log" tab (show_cluster_panel) —
        // see the "move the log somewhere else" feedback.
    }

    pub fn show_settings_panel(&mut self, ui: &mut Ui) {
        ui_kit::section_header(ui, "Models");
        self.pick_row(ui, "YOLO seg (.onnx)", Pick::Yolo);
        self.pick_row(ui, "DINOv3 (.onnx)", Pick::Dino);

        ui_kit::section_header(ui, "Detector");
        ui.checkbox(&mut self.use_fewshot, "Few-shot head (recommended)")
            .on_hover_text("Supervised head trained on your labeled families.\n\
                            Classifies each patch into healthy / family, assigns\n\
                            detected regions a family, and skips the 0.9 GB coreset\n\
                            bank. Falls back to PatchCore when no head is set.");
        if self.use_fewshot {
            self.pick_row(ui, "Few-shot head (.json)", Pick::Head);
            egui::Grid::new("pipeline_fewshot").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                ui.label("Seed threshold τ:")
                    .on_hover_text("Hysteresis SEED: a region must contain a patch with\n\
                                    P(defect) ≥ τ. Higher = fewer false positives, lower\n\
                                    recall. Default 0.85; raise toward 0.90 to cut FP.");
                ui.add(egui::Slider::new(&mut self.head_tau, 0.5..=0.98).fixed_decimals(2));
                ui.end_row();
                ui.label("Region tightness (grow τ):")
                    .on_hover_text("Hysteresis GROW: from each seed, the region expands\n\
                                    into connected patches with P(defect) ≥ this. HIGHER =\n\
                                    TIGHTER regions hugging the high-confidence core (smaller\n\
                                    boxes, same detections); lower = larger regions. Clamped ≤ seed τ.");
                ui.add(egui::Slider::new(&mut self.head_grow, 0.4..=0.9).fixed_decimals(2));
                ui.end_row();
            });
            if self.eff_head().is_none() {
                ui.label(RichText::new("No head set — will use PatchCore bank.")
                    .small().color(Color32::GRAY));
                self.pick_row(ui, "Coreset bank (.bin)", Pick::Bank);
                self.pick_row(ui, "Detector meta (.json)", Pick::Meta);
            } else {
                ui.checkbox(&mut self.use_patchcore, "Also run PatchCore (open-set safety net)")
                    .on_hover_text("Runs the coreset-bank detector alongside the few-shot head.\n\
                                    Catches anomaly types the head was never trained on, tagged\n\
                                    \"Novel (PatchCore)\". Off by default — only worth enabling once\n\
                                    the bank reflects your actual healthy data; otherwise it mostly\n\
                                    surfaces its own uncalibrated false positives.");
                if self.use_patchcore {
                    self.pick_row(ui, "Coreset bank (.bin)", Pick::Bank);
                    self.pick_row(ui, "Detector meta (.json)", Pick::Meta);
                    ui.checkbox(&mut self.unsupervised_families, "Cluster families unsupervised instead of the head's fixed classes")
                        .on_hover_text("The head's family label is always one of its trained\n\
                                        classes, forced by argmax — no reject/novelty option, so an\n\
                                        anomaly type it never saw gets shoehorned into the closest\n\
                                        known class anyway. Turning this on ignores the head's family\n\
                                        (its healthy/anomalous DETECTION is unaffected) and groups\n\
                                        regions by appearance instead — clusters emerge from this run's\n\
                                        actual data. Which algorithm does the grouping (DBSCAN or\n\
                                        Hierarchical) is picked separately below, under \"Clustering\n\
                                        looseness\". Trade-off: cluster IDs/names are NOT stable across\n\
                                        runs or datasets (rename them same as today); needs real tuning\n\
                                        per dataset either way.");
                    if self.unsupervised_families {
                        ui.checkbox(&mut self.domain_projection, "Train clustering projection from curations")
                            .on_hover_text("Trains a small domain-adapted embedding from THIS run's\n\
                                            own confirmed curations (curations/labels.jsonl), fresh in\n\
                                            memory every run — no separate training step, no saved file.\n\
                                            Closes some of the gap between a raw generic DINO embedding\n\
                                            and a head that was partially trained on this exact dataset,\n\
                                            without giving up open-set flexibility (clustering still runs\n\
                                            on the result, not a forced class list). Falls back to the\n\
                                            raw embedding automatically (logged, not an error) if there\n\
                                            isn't enough curated data yet — needs at least 2 confirmed\n\
                                            families with a handful of examples each.");
                    }
                }
            }
        } else {
            self.pick_row(ui, "Coreset bank (.bin)", Pick::Bank);
            self.pick_row(ui, "Detector meta (.json)", Pick::Meta);
        }

        ui_kit::section_header(ui, "Models (optional)");
        self.pick_row(ui, "Recon checkpoint (optional)", Pick::Recon);

        ui_kit::section_header(ui, "Settings");
        egui::Grid::new("pipeline_settings").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
            ui.label("Tile size:");
            egui::ComboBox::from_id_salt("pipeline_tile")
                .selected_text(format!("{}", self.tile_size))
                .show_ui(ui, |ui| {
                    for sz in [128u32, 256, 512] {
                        ui.selectable_value(&mut self.tile_size, sz, format!("{sz}"));
                    }
                });
            ui.end_row();
            ui.label("YOLO conf:");
            ui.add(egui::Slider::new(&mut self.conf, 0.0..=1.0).fixed_decimals(2));
            ui.end_row();
            ui.label("Margin erode (px):")
                .on_hover_text("Trim this many pixels inward from the leaf edge before\n\
                                detection, so the background ring left by the cutout\n\
                                isn't flagged as anomalous.");
            ui.add(egui::Slider::new(&mut self.margin_erode_px, 0..=20));
            ui.end_row();
            ui.label("Detect holes:")
                .on_hover_text("Flag interior holes — transparent regions fully enclosed by\n\
                                leaf — as defects on geometry alone.\n\n\
                                A hole eaten clean through the leaf is TRANSPARENT after\n\
                                segmentation, so it is excluded from tiling and no detector\n\
                                ever looks at it. Without this, only holes the segmenter\n\
                                FAILED to cut out (still showing background) are found — so\n\
                                improving segmentation reports fewer holes, not more.\n\n\
                                Regions land in the head's 'Holes' class when it has one,\n\
                                otherwise in the novel bucket.");
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.detect_holes, "");
                ui.add_enabled(self.detect_holes,
                    egui::DragValue::new(&mut self.min_hole_area).range(4..=2000).speed(4))
                    .on_hover_text("Minimum hole area in pixels — below this a transparent\n\
                                    blob is cutout anti-aliasing speckle, not damage.");
                ui.label(RichText::new("min px").small().color(Color32::GRAY));
            });
            ui.end_row();
            ui.label("Holes must be interior:")
                .on_hover_text("Drop 'Holes' detections that hug the leaf OUTLINE instead of\n\
                                sitting inside the leaf.\n\n\
                                A hole is enclosed tissue loss, but the head only sees\n\
                                appearance — and after transparent pixels are filled with the\n\
                                tile's mean colour, the leaf margin looks exactly like a hole\n\
                                rim. So the head fires along the edge. Raising 'Margin erode'\n\
                                does NOT help: it makes the eroded band transparent, which\n\
                                mean-fills it too, moving the edge instead of removing it.\n\n\
                                This measures how deep the region reaches (distance to the\n\
                                background OUTSIDE the leaf) and drops the shallow ones.\n\
                                Applies ONLY to the class named 'Holes'.");
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.filter_margin_holes, "");
                ui.add_enabled(self.filter_margin_holes,
                    egui::DragValue::new(&mut self.hole_margin_px).range(2..=200).speed(1))
                    .on_hover_text("Minimum depth in px. A region whose DEEPEST pixel is closer\n\
                                    than this to the outside is treated as a margin artifact.\n\
                                    Raise it if margin false positives survive; lower it if\n\
                                    genuine holes near the edge start disappearing.");
                ui.label(RichText::new("min depth px").small().color(Color32::GRAY));
            });
            ui.end_row();
            ui.label("Cutout edge:")
                .on_hover_text("YOLO cutout edge tightness (alpha feather start).\n\
                                Higher = tighter cut, less background rim\n\
                                (may nibble soft leaf edges).\n\
                                Lower (down to 0.0) = looser/more inclusive edge — \n\
                                try this if real leaf margin is getting cut off.");
            ui.add(egui::Slider::new(&mut self.seg_alpha_lo, 0.0..=0.75).fixed_decimals(2));
            ui.end_row();
            ui.label("Bg chroma reject:")
                .on_hover_text("Drop colourless rim pixels (grey/white/black, incl. the\n\
                                shadowed background next to the leaf). Higher = more\n\
                                aggressive; 0 = off.");
            ui.add(egui::Slider::new(&mut self.seg_chroma_min, 0..=60));
            ui.end_row();
        });

        ui_kit::section_header(ui, "Clustering looseness");
        ui.label(RichText::new("Affects PatchCore-only runs, and few-shot runs with \
                                 \"Cluster families unsupervised\" enabled above — both\n\
                                 cluster region features using the algorithm below.")
            .small().color(Color32::GRAY));
        egui::Grid::new("pipeline_clustering").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
            ui.label("Algorithm:")
                .on_hover_text("DBSCAN: density-radius based, can leave most regions as\n\
                                \"noise\" or collapse everything into 1-2 blobs if the data\n\
                                doesn't have a clean density gap (common with a mix of many\n\
                                different real anomaly types). Hierarchical: you pick (or\n\
                                auto-get a suggested) cluster COUNT directly instead of\n\
                                guessing a radius — more predictable, resists collapsing\n\
                                everything together the way DBSCAN can.");
            egui::ComboBox::from_id_salt("pipeline_cluster_algo")
                .selected_text(self.cluster_algo.label())
                .show_ui(ui, |ui| {
                    for &algo in ClusterAlgo::ALL {
                        ui.selectable_value(&mut self.cluster_algo, algo, algo.label());
                    }
                });
            ui.end_row();
            match self.cluster_algo {
                ClusterAlgo::Dbscan => {
                    ui.label("Cluster radius (eps):")
                        .on_hover_text("DBSCAN neighborhood radius over standardized region\n\
                                        features. LOWER = more, smaller clusters (looser —\n\
                                        similar-but-distinct anomalies split apart more readily).\n\
                                        Higher = fewer, broader clusters. Default 1.5 (8-D PatchCore\n\
                                        descriptor) — the DINO-embedding path (unsupervised_families)\n\
                                        needs a noticeably larger value; watch the log's k-distance\n\
                                        elbow suggestion after a run and dial toward that.");
                    ui.add(egui::Slider::new(&mut self.cluster_eps, 0.5..=8.0).fixed_decimals(2));
                    ui.end_row();
                    ui.label("Cluster min points:")
                        .on_hover_text("DBSCAN minimum neighbors to seed a cluster. Lower = more\n\
                                        clusters form (including from just a couple similar\n\
                                        regions); higher = only well-populated groups survive.\n\
                                        Default 5.");
                    ui.add(egui::Slider::new(&mut self.cluster_min_pts, 2..=10));
                    ui.end_row();
                }
                ClusterAlgo::Hierarchical => {
                    ui.label("Cut mode:")
                        .on_hover_text("Fixed K: one global cut depth for the whole merge tree —\n\
                                        simple, but if one branch (e.g. two defect types that keep\n\
                                        merging together) needs a much deeper split than the rest,\n\
                                        no single K gets it right. Adaptive: cuts each branch at its\n\
                                        own locally-appropriate depth instead — the fix for exactly\n\
                                        that \"some clusters fine, one stays stubbornly merged no\n\
                                        matter what K\" symptom.");
                    egui::ComboBox::from_id_salt("pipeline_cut_mode")
                        .selected_text(self.cut_mode.label())
                        .show_ui(ui, |ui| {
                            for &mode in CutMode::ALL {
                                ui.selectable_value(&mut self.cut_mode, mode, mode.label());
                            }
                        });
                    ui.end_row();
                    match self.cut_mode {
                        CutMode::FixedK => {
                            ui.label("Target clusters (0 = auto):")
                                .on_hover_text("How many groups to cut the merge tree into. 0 = auto,\n\
                                                via the biggest gap in merge distances (logged with\n\
                                                alternatives — the single largest gap is often the LAST\n\
                                                merge, i.e. a degenerate K=1-2, so treat auto as a\n\
                                                starting point, not gospel). You can also re-cut this\n\
                                                instantly after a run finishes, in the Clusters tab —\n\
                                                no need to get it right here and rerun the pipeline.");
                            ui.add(egui::Slider::new(&mut self.target_k, 0..=60));
                            ui.end_row();
                        }
                        CutMode::Adaptive => {
                            ui.label("Sensitivity:")
                                .on_hover_text("How locally-sharp a jump in merge height has to be\n\
                                                before that branch gets split. Higher = fewer, more\n\
                                                conservative splits; lower = more aggressive. There is\n\
                                                NO universally correct value here — like cluster_eps,\n\
                                                this needs tuning per dataset. Default 8.0 has real\n\
                                                margin above typical same-cluster \"finishing\" noise,\n\
                                                but start there and adjust from what the resulting\n\
                                                clusters actually look like. Re-cut instantly in the\n\
                                                Clusters tab afterward instead of rerunning the pipeline\n\
                                                per guess.");
                            ui.add(egui::Slider::new(&mut self.adaptive_threshold, 0.5..=30.0).fixed_decimals(1));
                            ui.end_row();
                        }
                    }
                    ui.label("Cluster min points:")
                        .on_hover_text("Minimum final cluster size. Groups smaller than this\n\
                                        get folded into noise (-1), mirroring DBSCAN's own noise\n\
                                        convention — different mechanism (Fixed K: undersized after\n\
                                        cutting to the target count; Adaptive: undersized after the\n\
                                        per-branch cut), same UI treatment. Default 5.");
                    ui.add(egui::Slider::new(&mut self.cluster_min_pts, 2..=10));
                    ui.end_row();
                }
            }
        });
    }

    fn pick_row(&mut self, ui: &mut Ui, label: &str, which: Pick) {
        let own = self.field_path(which).clone();
        let inherited = self.inherited(which);
        if ui.button(label).clicked() && self.pick_rx.is_none() {
            self.pick_rx = Some((which, spawn_dialog(which)));
        }
        let (txt, col) = match (&own, &inherited) {
            (Some(p), _) => (p.display().to_string(), Color32::GRAY),
            (None, Some(p)) => (format!("inherits: {}", p.display()), ui_kit::ACCENT()),
            (None, None) => ("- not set -".to_string(), Color32::GRAY),
        };
        ui.label(RichText::new(txt).small().color(col));
    }

    fn field_path(&self, which: Pick) -> &Option<PathBuf> {
        match which {
            Pick::Source => &self.source_folder,
            Pick::Output => &self.output_folder,
            Pick::Yolo => &self.yolo_model,
            Pick::Dino => &self.dino_model,
            Pick::Bank => &self.bank_path,
            Pick::Meta => &self.meta_path,
            Pick::Recon => &self.recon_ckpt,
            Pick::Head => &self.head_path,
            Pick::MineHealthyDir => &self.mine_healthy_dir,
            Pick::BaseSet => &self.retrain_base_set,
        }
    }

    fn show_gallery(&mut self, ui: &mut Ui, ctx: &Context) {
        ui.horizontal(|ui| {
            let (rev, rej, tot) = self.review_counts();
            ui.label(RichText::new(format!("Leaves — {tot} done")).small().color(Color32::GRAY));
            if tot > 0 {
                let left = tot.saturating_sub(rev + rej);
                ui.label(
                    RichText::new(format!("· {rev} reviewed · {rej} rejected · {left} to go"))
                        .small()
                        .color(if left == 0 { ui_kit::ACCENT() } else { Color32::GRAY }),
                )
                .on_hover_text(
                    "Saved to <output>/review_state.jsonl and restored next time you \
                     run this folder.\n\nM marks the current leaf reviewed · N jumps to \
                     the next unreviewed one.",
                );
            }
            if self.review_mismatch > 0 {
                ui.label(
                    RichText::new(format!("{} stale", self.review_mismatch))
                        .small().color(Color32::from_rgb(220, 170, 90)),
                )
                .on_hover_text(
                    "Saved review marks were recorded against differently-sized leaves \
                     — the segmentation model or its settings changed since.\n\nThose \
                     marks were NOT applied, because a tick on a leaf nobody has seen is \
                     worse than a missing tick.",
                );
            }
            if self.running {
                ui_kit::busy(ui, "processing…");
            }
        });
        for i in 0..self.results.len() {
            self.ensure_thumb(ctx, i); // lazily build missing thumbnails
        }
        egui::ScrollArea::horizontal().id_salt("pipeline_gallery_scroll").show(ui, |ui| {
            ui.horizontal(|ui| {
                for i in 0..self.results.len() {
                    let n = self.results[i].n_regions;
                    let Some(tex) = &self.thumbs[i] else { continue };
                    let rejected = self.rejected_leaves.contains(&i);
                    let reviewed = self.reviewed.contains(&i);
                    let hover = if rejected {
                        format!("leaf {i} — REJECTED ({n} regions excluded)")
                    } else if reviewed {
                        format!("leaf {i} — reviewed · {n} regions")
                    } else {
                        format!("leaf {i} — {n} regions · not yet reviewed")
                    };
                    let resp = ui
                        .add(egui::ImageButton::new((tex.id(), tex.size_vec2())))
                        .on_hover_text(hover);
                    // Rejection has to be legible from the gallery, not just from
                    // the top bar while the leaf happens to be selected — otherwise
                    // a leaf silently drops out of the export with nothing on screen
                    // saying so.
                    if rejected {
                        let red = Color32::from_rgb(190, 60, 60);
                        ui.painter().rect_filled(
                            resp.rect, 3.0, Color32::from_rgba_unmultiplied(150, 30, 30, 110),
                        );
                        ui.painter().rect_stroke(resp.rect, 3.0, egui::Stroke::new(2.0, red));
                        ui.painter().line_segment(
                            [resp.rect.left_top(), resp.rect.right_bottom()],
                            egui::Stroke::new(2.0, red),
                        );
                        ui.painter().line_segment(
                            [resp.rect.right_top(), resp.rect.left_bottom()],
                            egui::Stroke::new(2.0, red),
                        );
                    }
                    // Reviewed: dim the tile and tick it, so the strip reads as a
                    // map of what is left rather than an undifferentiated filmstrip.
                    // Drawn before the selection ring so selection still wins.
                    if reviewed && !rejected {
                        let green = Color32::from_rgb(120, 200, 130);
                        ui.painter().rect_filled(
                            resp.rect, 3.0, Color32::from_rgba_unmultiplied(10, 20, 12, 105),
                        );
                        // Drawn, not typed: U+2713 is absent from the bundled fonts
                        // and rendered as a tofu box. The reject cross beside it is
                        // already two line segments, so this also matches it.
                        let c = resp.rect.right_bottom() - egui::vec2(9.0, 8.0);
                        let stroke = egui::Stroke::new(2.0, green);
                        ui.painter().line_segment(
                            [c + egui::vec2(-4.0, 0.0), c + egui::vec2(-1.0, 3.5)], stroke,
                        );
                        ui.painter().line_segment(
                            [c + egui::vec2(-1.0, 3.5), c + egui::vec2(5.0, -4.0)], stroke,
                        );
                    }
                    if self.selected_idx == Some(i) {
                        ui.painter().rect_stroke(
                            resp.rect, 3.0,
                            egui::Stroke::new(2.0, Color32::from_rgb(120, 200, 130)),
                        );
                        // Keep the keyboard-selected leaf on screen. Without this,
                        // arrow-stepping past the visible end of the strip moves a
                        // selection you can no longer see.
                        if self.scroll_to_leaf {
                            resp.scroll_to_me(Some(egui::Align::Center));
                        }
                    }
                    if resp.clicked() {
                        self.selected_idx = Some(i);
                        self.overlay_tex = None;
                    }
                }
                self.scroll_to_leaf = false; // one-shot, consumed above
                // trailing tile: the leaf currently being processed
                if self.running {
                    ui.vertical(|ui| {
                        ui.add_space(20.0);
                        ui.add(egui::Spinner::new().size(22.0));
                        ui.label(RichText::new("processing").small().color(Color32::GRAY));
                    });
                }
            });
        });
    }

    /// Common tail of switching tools, however the switch happened (rail
    /// button click or hotkey): every tool shares scratch fields with at
    /// least one other (`canvas_drag_start`: Select+Knife's line-cut,
    /// `lasso_points`: Lasso+Knife's polycut, `brush_stroke`: Brush+Eraser)
    /// — leftover state from whichever tool was active before must never
    /// bleed into the next tool's first frame (e.g. Select's rubber-band
    /// keys off `canvas_drag_start.is_some()` rather than re-deriving it
    /// every frame, so a stale `Some()` left behind by another tool would
    /// make it think a drag is already in progress). Also re-defaults the
    /// stamp/brush label so it matches the new tool's intent.
    fn on_tool_switched(&mut self, tool: CanvasTool) {
        self.brush_stroke.clear();
        self.lasso_points.clear();
        self.canvas_drag_start = None;
        self.wand_mask.clear();
        self.wand_mask_tex = None;
        self.poly_points.clear();
        match tool {
            CanvasTool::MarkHealthy => self.hardneg_label = "healthy".to_string(),
            CanvasTool::Brush | CanvasTool::Wand => {
                self.hardneg_label = self.brush_default_family().unwrap_or_default();
            }
            _ => {}
        }
    }

    /// The cluster the canvas should dim everything else for — `None` unless
    /// focus mode is actually on. Every dimming site reads this, never
    /// `selected_cluster` directly.
    fn focus_cluster(&self) -> Option<i32> {
        if self.focus_mode { self.selected_cluster } else { None }
    }

    /// Focus needs exactly one family to focus ON, so the toggle is inert
    /// without a selection — reported as the control appearing to do nothing.
    fn focus_available(&self) -> bool {
        self.selected_cluster.is_some()
    }

    fn toggle_focus_mode(&mut self, toasts: &mut ToastManager) {
        if !self.focus_available() {
            toasts.info("Select a family first — focus dims everything except one family.");
            return;
        }
        self.focus_mode = !self.focus_mode;
        self.overlay_tex = None; // the dim state is baked into the overlay
        if self.focus_mode {
            let name = self
                .selected_cluster
                .and_then(|c| self.cluster_names.get(&c).cloned())
                .unwrap_or_else(|| "family".to_string());
            toasts.info(format!("Focus on {name}"));
        } else {
            toasts.info("Focus off");
        }
    }

    /// The colour the brush should preview and paint in: the target family's
    /// own colour, resolved through `cluster_color` so user overrides apply.
    ///
    /// The preview used to be a hardcoded orange for every family, so a custom
    /// category looked one colour under the brush and another in the legend —
    /// "eigene Kategorie Blattschaden hat bei Pinsel andere Farbe als bei
    /// Familien". Falls back to that orange only when the typed name is a family
    /// that does not exist yet, where there is genuinely no colour to show.
    ///
    /// Read-only on purpose: `resolve_cluster_id` would MINT an id for an unknown
    /// name, and a hover preview must not create a family as a side effect.
    fn brush_preview_color(&self) -> Color32 {
        let name = self.hardneg_label.trim();
        if name.is_empty() {
            return Color32::from_rgb(255, 150, 60);
        }
        match self
            .cluster_names
            .iter()
            .find(|(_, n)| n.eq_ignore_ascii_case(name))
            .map(|(&id, _)| id)
        {
            Some(id) => {
                let c = cluster_color(id);
                Color32::from_rgb(c[0], c[1], c[2])
            }
            None => Color32::from_rgb(255, 150, 60),
        }
    }

    /// Which family the brush should paint into unless told otherwise.
    ///
    /// Prefers the family of the REGION currently selected, and only then the
    /// focus-mode cluster. That order is the whole point: this used to read
    /// `selected_cluster` alone, so after reviewing a Nekrosis region the brush
    /// was still set to whatever cluster had last been focused — reported as
    /// people painting the wrong family without noticing, because reaching for
    /// the brush right after reviewing a region obviously means "more of THAT".
    fn brush_default_family(&self) -> Option<String> {
        let cid = self
            .selected_region
            .and_then(|i| self.labels.get(i).copied())
            .filter(|&c| c >= 0)
            .or(self.selected_cluster)?;
        self.cluster_names.get(&cid).cloned()
    }

    /// Tool-hotkey entry point (Photoshop-style single letters) — a no-op if
    /// `tool` is already active, otherwise identical to clicking its rail
    /// button.
    fn switch_tool_hotkey(&mut self, tool: CanvasTool) {
        if self.canvas_tool != tool {
            self.canvas_tool = tool;
            self.on_tool_switched(tool);
        }
    }

    /// Clickable cluster picker: one colored swatch+name row per known
    /// class, returning the clicked class's id if any row was clicked this
    /// frame. `current` (case-insensitive) highlights the matching row; pass
    /// "" for no highlight. Shared by the Brush tool's own picker, the
    /// right-click "Move to cluster…" submenu, the quick-reassign popup, the
    /// gallery bulk-reassign bar, and the MarkHealthy/calibration label
    /// field — one click-to-pick affordance instead of near-duplicate
    /// implementations, and the CANONICAL name source: rows come from the
    /// union of this run's own discovered clusters AND the loaded head's
    /// real family list (even a class with zero members this run), so
    /// typing an existing family's name always finds it instead of silently
    /// forking a duplicate class.
    /// Display name for a class id, checking this run's own `cluster_names`
    /// first, then the loaded head's real family list, then falling back to
    /// "Cluster N" — the same resolution order `cluster_picker_rows` uses to
    /// build its rows, so a head-only class (picked but never confirmed in
    /// this run) still resolves to its real name instead of a placeholder.
    fn class_display_name(&mut self, id: i32) -> String {
        if let Some(n) = self.cluster_names.get(&id) {
            return n.clone();
        }
        if let Some(h) = self.cached_head() {
            if h.classes.contains(&id) {
                return h.family_name(id);
            }
        }
        format!("Cluster {id}")
    }

    fn cluster_picker_rows(&mut self, ui: &mut Ui, current: &str) -> Option<i32> {
        let mut rows: HashMap<i32, String> = HashMap::new();
        for c in &self.clusters {
            if c.id < 0 {
                continue; // skip "noise"
            }
            let name = self.cluster_names.get(&c.id).cloned().unwrap_or_else(|| format!("Cluster {}", c.id));
            rows.insert(c.id, name);
        }
        if let Some(head) = self.cached_head() {
            for &cid in &head.classes {
                if cid <= 0 {
                    continue; // skip healthy (0) same as "noise" above
                }
                rows.entry(cid).or_insert_with(|| head.family_name(cid));
            }
        }
        let mut rows: Vec<(i32, String)> = rows.into_iter().collect();
        rows.sort_by(|a, b| a.1.cmp(&b.1));

        let mut picked = None;
        for (cid, name) in rows {
            let col = cluster_color(cid);
            let selected = current.eq_ignore_ascii_case(&name);
            let clicked = ui.horizontal(|ui| {
                let (rect, swatch_resp) = ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::click());
                ui.painter().rect_filled(rect, 2.0, Color32::from_rgb(col[0], col[1], col[2]));
                if selected {
                    ui.painter().rect_stroke(rect, 2.0, egui::Stroke::new(1.5, Color32::WHITE));
                }
                let label_resp = ui.selectable_label(selected, &name);
                swatch_resp.clicked() || label_resp.clicked()
            }).inner;
            if clicked {
                picked = Some(cid);
            }
        }
        picked
    }

    /// The canvas's current "selection" for context-menu/hotkey actions: the
    /// Toggle region `i` into/out of the multi-select set. On the FIRST
    /// toggle (`multi_selected` still empty), also seeds it with whatever
    /// single region was already highlighted by a plain click —
    /// `effective_selection` below prefers `multi_selected` the instant it's
    /// non-empty, so without this seed step a plain click followed by
    /// ctrl+click-ing MORE regions would silently drop that very first
    /// region from every bulk action (confirm/reject/reassign): a real
    /// reported bug ("ctrl+click a few regions, mass-reassign, the first one
    /// I clicked never changes"). Shared by the canvas Select tool and the
    /// gallery — both had this exact gap independently.
    fn toggle_multi_select(&mut self, i: usize) {
        if self.multi_selected.is_empty() {
            if let Some(prev) = self.selected_region {
                if prev != i {
                    self.multi_selected.insert(prev);
                }
            }
        }
        if !self.multi_selected.remove(&i) {
            self.multi_selected.insert(i);
        }
    }

    /// gallery/rubber-band multi-select set if non-empty, else whichever
    /// single region is highlighted by a plain click — so every action
    /// (Enter/Delete/Ctrl+Z/the reassign hotkey/the context menu) works the
    /// same way regardless of how the selection was made.
    fn effective_selection(&self) -> Vec<usize> {
        if !self.multi_selected.is_empty() {
            self.multi_selected.iter().copied().collect()
        } else {
            self.selected_region.into_iter().collect()
        }
    }

    /// Reassign `ids` to cluster `target_id` (persisted under `name`) — the
    /// shared tail of every reassign path: the gallery's typed Reassign
    /// button, the context-menu submenu (typed name AND its cluster-picker
    /// rows), and the quick-reassign popup.
    fn reassign_ids(&mut self, ids: &[usize], target_id: i32, name: &str, toasts: &mut ToastManager) {
        if ids.is_empty() {
            return;
        }
        for &i in ids {
            self.labels[i] = target_id;
            self.persist_region(i, name, false, toasts);
        }
        self.build_clusters(toasts);
        self.overlay_tex = None;
    }

    fn ensure_thumb(&mut self, ctx: &Context, i: usize) {
        if i >= self.results.len() || self.thumbs[i].is_some() {
            return;
        }
        let leaf = &self.results[i];
        let Some(src) = image::RgbaImage::from_raw(leaf.w, leaf.h, leaf.rgba.clone()) else { return };
        let scale = 84.0 / leaf.w.max(leaf.h).max(1) as f32;
        let tw = ((leaf.w as f32 * scale).round() as u32).max(1);
        let th = ((leaf.h as f32 * scale).round() as u32).max(1);
        let small = image::imageops::resize(&src, tw, th, image::imageops::FilterType::Triangle);
        let pixels: Vec<Color32> = small
            .into_raw()
            .chunks_exact(4)
            .map(|c| Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]))
            .collect();
        let tex = ctx.load_texture(
            format!("pipe_thumb_{i}"),
            egui::ColorImage { size: [tw as usize, th as usize], pixels },
            egui::TextureOptions::LINEAR,
        );
        self.thumbs[i] = Some(tex);
    }

    fn show_canvas(&mut self, ui: &mut Ui, ctx: &Context, toasts: &mut ToastManager) {
        self.ensure_overlay(ctx);
        let Some(tex) = self.overlay_tex.clone() else {
            ui.centered_and_justified(|ui| {
                ui.label(RichText::new(
                    "Set the models + folders, then Run Pipeline.\n\
                     Detected leaves appear in the gallery; click one to see its overlay.")
                    .color(Color32::GRAY));
            });
            return;
        };
        let leaf_idx = self.overlay_key.map(|k| k.0).unwrap_or(0);
        let avail = ui.available_size();
        let sz = tex.size_vec2();
        let scale = (avail.x / sz.x).min(avail.y / sz.y).min(1.0).max(0.01);
        let disp = sz * scale;
        // Allocate the panel area and draw the image centred in it, keeping the TRUE
        // image rect so the bbox + click hit-test map to leaf pixels correctly.
        let (area, resp) = ui.allocate_exact_size(avail, egui::Sense::click_and_drag());
        // Zoom/pan (Zoom/Pan tool) applies ON TOP of the fit-to-panel rect —
        // every downstream leaf-pixel<->screen conversion already funnels
        // through img_rect/s below, so generalizing THIS ONE computation is
        // all that's needed; no other call site changes.
        let fit_rect = egui::Rect::from_center_size(area.center(), disp);
        // Honour a pending "bring the selected region into view" request now that
        // fit_rect is known — it is the only place the mapping from leaf pixels
        // to screen exists.
        //
        // Only while zoomed in: at 1.0 the whole leaf is already visible and
        // moving the view would be a jump with no purpose. Reported as wanting
        // the view to follow the selection instead of having to zoom out, find
        // the next region, and zoom back in ("Leute sind bequemlich").
        if let Some(ri) = self.center_on_region.take() {
            if self.canvas_zoom > 1.01 {
                if let Some(r) = self.regions.get(ri) {
                    let [bx, by, bw, bh] = r.bbox_leaf;
                    let (cx, cy) = (bx as f32 + bw as f32 / 2.0, by as f32 + bh as f32 / 2.0);
                    let z = self.canvas_zoom;
                    self.canvas_pan = egui::vec2(
                        fit_rect.width() * z * (0.5 - cx / sz.x.max(1.0)),
                        fit_rect.height() * z * (0.5 - cy / sz.y.max(1.0)),
                    );
                }
            }
        }
        let img_rect = egui::Rect::from_center_size(
            fit_rect.center() + self.canvas_pan, fit_rect.size() * self.canvas_zoom,
        );
        egui::Image::new((tex.id(), img_rect.size())).paint_at(ui, img_rect);
        let s = img_rect.width() / sz.x.max(1.0);

        // outline mode: draw smooth vector contours of the visible regions —
        // Focus mode (a cluster selected via scatter plot/gallery) dims every
        // OTHER cluster's contour rather than skipping it, so the rest of the
        // leaf stays visible as context instead of disappearing.
        if self.overlay_outline {
            let sel = self.focus_cluster();
            for (ri, r) in self.regions.iter().enumerate() {
                if r.leaf != leaf_idx || !self.region_visible(ri) {
                    continue;
                }
                let focused = sel.map_or(true, |cid| self.labels[ri] == cid);
                let [bx, by, bw, bh] = r.bbox_leaf;
                let raw = trace_contour(&r.mask, bw, bh);
                if raw.len() < 4 {
                    continue;
                }
                let pts: Vec<egui::Pos2> = raw.iter().step_by(2).map(|&(cx, cy)| {
                    egui::pos2(
                        img_rect.min.x + (bx as f32 + cx + 0.5) * s,
                        img_rect.min.y + (by as f32 + cy + 0.5) * s,
                    )
                }).collect();
                let sm = chaikin(&pts, 2);
                let cid = self.labels[ri];
                let col = if focused { cluster_color(cid) } else { dim_color(cluster_color(cid)) };
                let width = if focused { 2.0 } else { 1.2 };
                let stroke = egui::Stroke::new(width, Color32::from_rgb(col[0], col[1], col[2]));
                // Dash pattern is the family's SECOND, non-colour cue: it survives
                // colour blindness, greyscale, and the palette wrapping past eight
                // families. Scaled by `s` so the dashes keep their on-screen length
                // as the canvas zooms.
                match cluster_dash(cid) {
                    None => ui.painter().add(egui::Shape::closed_line(sm, stroke)),
                    Some(dash) => {
                        let d = (dash * s).max(2.0);
                        let mut closed = sm.clone();
                        if let Some(&first) = sm.first() {
                            closed.push(first); // dashed_line does not close for us
                        }
                        ui.painter().add(egui::Shape::dashed_line(
                            &closed, stroke, d, d * 0.75,
                        ))
                    }
                };
            }
        }

        // which leaf is shown (so it's clear navigation worked)
        let name = self.results.get(leaf_idx)
            .and_then(|l| l.src.file_name())
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        ui.painter().text(
            area.min + egui::vec2(8.0, 6.0), egui::Align2::LEFT_TOP,
            format!("Leaf {} — {}  ·  click a region to find its tile", leaf_idx + 1, name),
            egui::FontId::proportional(13.0), Color32::from_rgb(120, 200, 130),
        );
        // (active-tool status now lives in the options bar above the canvas,
        // show_canvas_options_bar — no longer overlaid on the leaf image.)
        // highlight the selected anomaly with a bounding box on the leaf
        if let Some(ri) = self.selected_region {
            if let Some(r) = self.regions.get(ri) {
                if r.leaf == leaf_idx && self.region_visible(ri) {
                    let [bx, by, bw, bh] = r.bbox_leaf;
                    let pad = 2.0;
                    let mn = img_rect.min + egui::vec2(bx as f32 * s - pad, by as f32 * s - pad);
                    let mx = img_rect.min + egui::vec2((bx + bw) as f32 * s + pad, (by + bh) as f32 * s + pad);
                    ui.painter().rect_stroke(
                        egui::Rect::from_min_max(mn, mx), 1.0,
                        egui::Stroke::new(2.0, Color32::from_rgb(255, 230, 0)),
                    );
                }
            }
        }
        // outline every multi-selected region on this leaf (rubber-band / ctrl+click
        // picks from the gallery) — distinct blue from the single-selection yellow,
        // matching the gallery tile's own multi-select border colour.
        for &ri in &self.multi_selected {
            if let Some(r) = self.regions.get(ri) {
                if r.leaf == leaf_idx && self.region_visible(ri) {
                    let [bx, by, bw, bh] = r.bbox_leaf;
                    let pad = 2.0;
                    let mn = img_rect.min + egui::vec2(bx as f32 * s - pad, by as f32 * s - pad);
                    let mx = img_rect.min + egui::vec2((bx + bw) as f32 * s + pad, (by + bh) as f32 * s + pad);
                    ui.painter().rect_stroke(
                        egui::Rect::from_min_max(mn, mx), 1.0,
                        egui::Stroke::new(2.0, Color32::from_rgb(80, 170, 255)),
                    );
                }
            }
        }
        match self.canvas_tool {
        CanvasTool::MarkHealthy => {
            // ── tile-picker-style stamp tool: a square follows the cursor,
            // click stamps it (crop + save), right-click/Ctrl+Z reverts ──
            let t = self.hardneg_tile as f32;
            let mut sq_tl: Option<(f32, f32)> = None; // leaf-px top-left of the square
            let mut hover_leaf: Option<(f32, f32)> = None;
            if let Some(p) = resp.hover_pos() {
                let lx = (p.x - img_rect.min.x) / s.max(1e-3);
                let ly = (p.y - img_rect.min.y) / s.max(1e-3);
                hover_leaf = Some((lx, ly));
                sq_tl = Some((lx - t / 2.0, ly - t / 2.0));
            }
            // already-stamped tiles on this leaf
            if let Some(list) = self.hardneg_stamps.get(&leaf_idx) {
                for st in list {
                    let mn = img_rect.min + egui::vec2(st.x as f32 * s, st.y as f32 * s);
                    let mx = mn + egui::vec2(t * s, t * s);
                    let r = egui::Rect::from_min_max(mn, mx);
                    ui.painter().rect_filled(r, 0.0, Color32::from_rgba_unmultiplied(255, 180, 60, 28));
                    ui.painter().rect_stroke(r, 0.0, egui::Stroke::new(1.5, Color32::from_rgb(255, 180, 60)));
                }
            }
            // live placement square
            if let Some((tlx, tly)) = sq_tl {
                let mn = img_rect.min + egui::vec2(tlx * s, tly * s);
                let mx = mn + egui::vec2(t * s, t * s);
                ui.painter().rect_stroke(egui::Rect::from_min_max(mn, mx), 0.0,
                    egui::Stroke::new(2.0, Color32::from_rgb(80, 170, 255)));
            }
            // magnifier, pinned beside the cursor
            if let (Some((tlx, tly)), Some(p)) = (sq_tl, resp.hover_pos()) {
                self.show_hardneg_loupe(ui, ctx, leaf_idx, tlx + t / 2.0, tly + t / 2.0, p, s);
            }
            if resp.clicked() {
                if let Some((tlx, tly)) = sq_tl {
                    self.stamp_hardneg(leaf_idx, tlx.round() as i32, tly.round() as i32, toasts);
                }
            }
            if resp.secondary_clicked() {
                if let Some((lx, ly)) = hover_leaf {
                    self.remove_hardneg_at(leaf_idx, lx, ly);
                }
            }
            if ui.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Z)) {
                self.undo_hardneg(leaf_idx);
            }
        }
        CanvasTool::Select => {
            // ── click selects a region; drag rubber-band multi-selects
            // (feeds the gallery's bulk reassign/remove) ──
            if resp.drag_started_by(egui::PointerButton::Primary) {
                self.canvas_drag_start = resp.interact_pointer_pos();
            }
            let mut did_rubberband = false;
            if let Some(start) = self.canvas_drag_start {
                if let Some(cur) = resp.interact_pointer_pos() {
                    ui.painter().rect_stroke(
                        egui::Rect::from_two_pos(start, cur), 0.0,
                        egui::Stroke::new(1.5, Color32::from_rgb(80, 170, 255)),
                    );
                }
                if resp.drag_stopped_by(egui::PointerButton::Primary) {
                    if let Some(end) = resp.interact_pointer_pos() {
                        let rect = egui::Rect::from_two_pos(start, end);
                        if rect.width() > 4.0 && rect.height() > 4.0 {
                            let lx0 = (rect.min.x - img_rect.min.x) / s.max(1e-3);
                            let ly0 = (rect.min.y - img_rect.min.y) / s.max(1e-3);
                            let lx1 = (rect.max.x - img_rect.min.x) / s.max(1e-3);
                            let ly1 = (rect.max.y - img_rect.min.y) / s.max(1e-3);
                            self.select_regions_in_rect(leaf_idx, lx0, ly0, lx1, ly1);
                            did_rubberband = true;
                        }
                    }
                    self.canvas_drag_start = None;
                }
            }
            if !did_rubberband && resp.clicked() {
                let ctrl = ui.input(|i| i.modifiers.ctrl);
                if ctrl {
                    // ctrl+click toggles into the multi-select set — same
                    // gesture the gallery already supports, previously
                    // canvas-only-missing (worth fixing regardless of the
                    // tool rail: it's a real gap, not new behavior).
                    if let Some(p) = resp.interact_pointer_pos() {
                        let lx = (p.x - img_rect.min.x) / s.max(1e-3);
                        let ly = (p.y - img_rect.min.y) / s.max(1e-3);
                        if let Some(i) = self.region_at(leaf_idx, lx, ly) {
                            self.toggle_multi_select(i);
                        }
                    }
                } else if !self.multi_selected.is_empty() {
                    // a plain click anywhere dismisses the current multi-selection —
                    // click-away-to-deselect, cheaper than the context menu's
                    // "Clear selection" for the common case of just backing out.
                    self.multi_selected.clear();
                } else if let Some(p) = resp.interact_pointer_pos() {
                    // click a region on the leaf -> select it (highlights its gallery tile)
                    let lx = (p.x - img_rect.min.x) / s.max(1e-3);
                    let ly = (p.y - img_rect.min.y) / s.max(1e-3);
                    self.select_region_at(leaf_idx, lx, ly);
                }
            }
        }
        CanvasTool::Brush => {
            // ── paint a freeform mask: bounds-check in i32 space BEFORE any
            // cast to u32 — a saturating float->u32 cast would silently clamp
            // out-of-bounds pixels to 0 instead of excluding them (mirrors
            // stamp_hardneg's explicit bounds-then-continue pattern) ──
            let (lw, lh) = self.results.get(leaf_idx).map(|l| (l.w as i32, l.h as i32)).unwrap_or((0, 0));
            let half = (self.brush_size / 2) as i32;
            let hover_leaf = resp.hover_pos().map(|p| {
                ((p.x - img_rect.min.x) / s.max(1e-3), (p.y - img_rect.min.y) / s.max(1e-3))
            });
            // live cursor-following preview shape, in the TARGET FAMILY's colour
            // so the brush and the legend agree about what is being painted
            let brush_col = self.brush_preview_color();
            if let Some((cx, cy)) = hover_leaf {
                let mn = img_rect.min + egui::vec2((cx - half as f32) * s, (cy - half as f32) * s);
                let sz = (self.brush_size as f32) * s;
                match self.brush_shape {
                    BrushShape::Square => {
                        ui.painter().rect_stroke(egui::Rect::from_min_size(mn, egui::vec2(sz, sz)), 0.0,
                            egui::Stroke::new(2.0, brush_col));
                    }
                    BrushShape::Circle => {
                        ui.painter().circle_stroke(mn + egui::vec2(sz, sz) / 2.0, sz / 2.0,
                            egui::Stroke::new(2.0, brush_col));
                    }
                }
            }
            if resp.dragged_by(egui::PointerButton::Primary) {
                if let Some((cx, cy)) = hover_leaf {
                    let (cxi, cyi) = (cx.round() as i32, cy.round() as i32);
                    for oy in -half..=half {
                        for ox in -half..=half {
                            if self.brush_shape == BrushShape::Circle && ox * ox + oy * oy > half * half {
                                continue;
                            }
                            let (px, py) = (cxi + ox, cyi + oy);
                            if px < 0 || py < 0 || px >= lw || py >= lh {
                                continue;
                            }
                            // Bounds are not enough: the leaf is a CUTOUT inside
                            // its bounding box, so "inside the image" still
                            // includes transparent background. Painting there
                            // produced regions floating off the leaf, which then
                            // carry into area stats and curation crops. Alpha 10
                            // is the same validity bar tile_leaf and the hole
                            // detector already use.
                            if !self.leaf_pixel_valid(leaf_idx, px, py) {
                                continue;
                            }
                            self.brush_stroke.insert((px, py));
                        }
                    }
                }
            }
            // paint the stroke LIVE as it's built, not just once on release —
            // otherwise nothing visibly happens until the mouse button comes
            // up, which reads as a broken/delayed tool.
            if !self.brush_stroke.is_empty() {
                let px_sz = s.max(1.0);
                for &(px, py) in &self.brush_stroke {
                    let mn = img_rect.min + egui::vec2(px as f32 * s, py as f32 * s);
                    ui.painter().rect_filled(
                        egui::Rect::from_min_size(mn, egui::vec2(px_sz, px_sz)),
                        0.0,
                        Color32::from_rgba_unmultiplied(
                            brush_col.r(), brush_col.g(), brush_col.b(), 110,
                        ),
                    );
                }
            }
            if resp.drag_stopped_by(egui::PointerButton::Primary) && !self.brush_stroke.is_empty() {
                self.finish_brush_stroke(leaf_idx, toasts);
            }
        }
        CanvasTool::Eraser => {
            // ── same paint-a-stroke interaction as Brush, but the stroke
            // ERASES mask pixels from whatever regions it covers instead of
            // creating/extending one ──
            let (lw, lh) = self.results.get(leaf_idx).map(|l| (l.w as i32, l.h as i32)).unwrap_or((0, 0));
            let half = (self.brush_size / 2) as i32;
            let hover_leaf = resp.hover_pos().map(|p| {
                ((p.x - img_rect.min.x) / s.max(1e-3), (p.y - img_rect.min.y) / s.max(1e-3))
            });
            if let Some((cx, cy)) = hover_leaf {
                let mn = img_rect.min + egui::vec2((cx - half as f32) * s, (cy - half as f32) * s);
                let sz = (self.brush_size as f32) * s;
                match self.brush_shape {
                    BrushShape::Square => {
                        ui.painter().rect_stroke(egui::Rect::from_min_size(mn, egui::vec2(sz, sz)), 0.0,
                            egui::Stroke::new(2.0, Color32::from_rgb(230, 60, 60)));
                    }
                    BrushShape::Circle => {
                        ui.painter().circle_stroke(mn + egui::vec2(sz, sz) / 2.0, sz / 2.0,
                            egui::Stroke::new(2.0, Color32::from_rgb(230, 60, 60)));
                    }
                }
            }
            if resp.dragged_by(egui::PointerButton::Primary) {
                if let Some((cx, cy)) = hover_leaf {
                    let (cxi, cyi) = (cx.round() as i32, cy.round() as i32);
                    for oy in -half..=half {
                        for ox in -half..=half {
                            if self.brush_shape == BrushShape::Circle && ox * ox + oy * oy > half * half {
                                continue;
                            }
                            let (px, py) = (cxi + ox, cyi + oy);
                            if px < 0 || py < 0 || px >= lw || py >= lh {
                                continue;
                            }
                            self.brush_stroke.insert((px, py));
                        }
                    }
                }
            }
            if !self.brush_stroke.is_empty() {
                let px_sz = s.max(1.0);
                for &(px, py) in &self.brush_stroke {
                    let mn = img_rect.min + egui::vec2(px as f32 * s, py as f32 * s);
                    ui.painter().rect_filled(
                        egui::Rect::from_min_size(mn, egui::vec2(px_sz, px_sz)),
                        0.0, Color32::from_rgba_unmultiplied(230, 60, 60, 110),
                    );
                }
            }
            if resp.drag_stopped_by(egui::PointerButton::Primary) && !self.brush_stroke.is_empty() {
                self.erase_stroke(leaf_idx, toasts);
            }
        }
        CanvasTool::Knife => {
            self.handle_knife(ui, &resp, leaf_idx, img_rect, s, toasts);
        }
        CanvasTool::Scissor => {
            self.handle_scissor(ui, &resp, leaf_idx, img_rect, s, toasts);
        }
        CanvasTool::Lasso => {
            // ── freehand select: collect screen-space points while dragging,
            // draw them live, and on release select every region whose
            // bbox-center falls inside the closed polygon (same precision
            // level as the existing rubber-band, which also only tests bbox
            // intersection rather than exact mask overlap) ──
            if resp.drag_started_by(egui::PointerButton::Primary) {
                self.lasso_points.clear();
            }
            if resp.dragged_by(egui::PointerButton::Primary) {
                if let Some(p) = resp.interact_pointer_pos() {
                    if self.lasso_points.last().map_or(true, |last| last.distance(p) > 2.0) {
                        self.lasso_points.push(p);
                    }
                }
            }
            if self.lasso_points.len() >= 2 {
                let mut pts = self.lasso_points.clone();
                pts.push(pts[0]);
                ui.painter().add(egui::Shape::line(pts, egui::Stroke::new(1.5, Color32::from_rgb(80, 170, 255))));
            }
            if resp.drag_stopped_by(egui::PointerButton::Primary) && self.lasso_points.len() >= 3 {
                let poly: Vec<(f32, f32)> = self.lasso_points.iter()
                    .map(|p| ((p.x - img_rect.min.x) / s.max(1e-3), (p.y - img_rect.min.y) / s.max(1e-3)))
                    .collect();
                for (i, r) in self.regions.iter().enumerate() {
                    if r.leaf != leaf_idx || !self.region_visible(i) {
                        continue;
                    }
                    let [bx, by, bw, bh] = r.bbox_leaf;
                    let cx = bx as f32 + bw as f32 / 2.0;
                    let cy = by as f32 + bh as f32 / 2.0;
                    if point_in_polygon(cx, cy, &poly) {
                        self.multi_selected.insert(i);
                    }
                }
                self.lasso_points.clear();
            }
        }
        CanvasTool::Wand => {
            // ── click: grow a pending pixel selection by Lab color similarity
            // from the clicked seed (shift-click adds another blob instead of
            // replacing). Nothing is committed until "Fill" in the options
            // panel — reviewable/discardable before it becomes a region. ──
            let (lw, lh) = self.results.get(leaf_idx).map(|l| (l.w as usize, l.h as usize)).unwrap_or((0, 0));
            // `clicked_by` alone requires ZERO pointer movement between press
            // and release — any tiny jitter gets reclassified as a drag and
            // silently does nothing, which read as "the wand barely ever
            // works." Also accept a drag-stop, using the release position.
            let fired = resp.clicked_by(egui::PointerButton::Primary)
                || resp.drag_stopped_by(egui::PointerButton::Primary);
            if fired {
                if let Some(p) = resp.interact_pointer_pos() {
                    let lx = ((p.x - img_rect.min.x) / s.max(1e-3)).round() as i32;
                    let ly = ((p.y - img_rect.min.y) / s.max(1e-3)).round() as i32;
                    let tol = self.wand_tolerance;
                    self.ensure_wand_lab(leaf_idx);
                    let grown = self.wand_lab_cache.as_ref()
                        .filter(|(li, _, _, _)| *li == leaf_idx)
                        .map(|(_, l, a, b)| wand_flood_fill(l, a, b, lw, lh, lx, ly, tol));
                    if let Some(grown) = grown {
                        let shift = ui.input(|i| i.modifiers.shift);
                        if !shift {
                            self.wand_mask.clear();
                        }
                        self.wand_mask.extend(grown);
                        self.rebuild_wand_mask_tex(ctx, lw, lh);
                    }
                }
            }
            // live highlight of the pending (uncommitted) selection — a single
            // cached texture blit, not a per-pixel paint call every frame.
            if let Some(tex) = &self.wand_mask_tex {
                egui::Image::new((tex.id(), img_rect.size())).paint_at(ui, img_rect);
            }
        }
        CanvasTool::Polygon => {
            let to_leaf = |p: egui::Pos2| ((p.x - img_rect.min.x) / s.max(1e-3), (p.y - img_rect.min.y) / s.max(1e-3));
            // Clamp every placed node to the actual leaf bounds — a click
            // in the canvas's letterboxed margin, or far outside the image
            // while zoomed, otherwise produces leaf-space coordinates way
            // beyond the real image, which fed a bbox spanning tens of
            // thousands of pixels into the polygon fill and crashed on the
            // resulting allocation (confirmed real, not theoretical).
            let (lw, lh) = self.results.get(leaf_idx).map(|l| (l.w as f32, l.h as f32)).unwrap_or((0.0, 0.0));
            if let Some(p) = resp.hover_pos() {
                let (rx, ry) = to_leaf(p);
                let hp = (rx.clamp(0.0, lw.max(1.0)), ry.clamp(0.0, lh.max(1.0)));
                const SNAP_PX: f32 = 10.0;
                let near_start = self.poly_points.len() >= 3 && {
                    let (fx, fy) = self.poly_points[0];
                    let first_screen = img_rect.min + egui::vec2(fx * s, fy * s);
                    first_screen.distance(p) <= SNAP_PX
                };
                if resp.clicked() {
                    if near_start {
                        self.finish_polygon(leaf_idx, p, toasts);
                    } else {
                        self.poly_points.push(hp);
                    }
                }
                if let Some(&(lx, ly)) = self.poly_points.last() {
                    let last_screen = img_rect.min + egui::vec2(lx * s, ly * s);
                    ui.painter().line_segment([last_screen, p], egui::Stroke::new(1.5, Color32::from_rgb(80, 170, 255)));
                }
                // `near_start` was computed before the click above may have
                // just closed (and cleared) the polygon this same frame —
                // re-check emptiness, don't trust the stale bool, or this
                // indexes poly_points[0] on an empty vec and panics EXACTLY
                // at the moment of closing (confirmed as the real crash,
                // independent of area size).
                if near_start && !self.poly_points.is_empty() {
                    let (fx, fy) = self.poly_points[0];
                    let first_screen = img_rect.min + egui::vec2(fx * s, fy * s);
                    ui.painter().circle_stroke(first_screen, SNAP_PX, egui::Stroke::new(2.0, Color32::from_rgb(140, 230, 150)));
                }
            }
            if self.poly_points.len() > 1 {
                let screen: Vec<egui::Pos2> = self.poly_points.iter()
                    .map(|&(x, y)| img_rect.min + egui::vec2(x * s, y * s)).collect();
                ui.painter().add(egui::Shape::line(screen.clone(), egui::Stroke::new(1.5, Color32::from_rgb(80, 170, 255))));
                for pt in screen {
                    ui.painter().circle_filled(pt, 3.0, Color32::from_rgb(80, 170, 255));
                }
            } else if let Some(&(x, y)) = self.poly_points.first() {
                let pt = img_rect.min + egui::vec2(x * s, y * s);
                ui.painter().circle_filled(pt, 3.0, Color32::from_rgb(80, 170, 255));
            }
        }
        }

        // Polygon's family-choice popup — only shown when a polygon was
        // closed with NOTHING selected (see `finish_polygon`); with a
        // region selected, the fill commits immediately using that
        // region's own family, same as Brush.
        if let Some(stroke) = self.poly_pending.clone() {
            let mut applied: Option<(i32, String)> = None;
            let mut cancel = false;
            egui::Window::new("Assign new region to")
                .id(egui::Id::new("pipeline_poly_pick"))
                .collapsible(false)
                .resizable(false)
                .current_pos(self.poly_pick_pos)
                .show(ctx, |ui| {
                    if let Some(id) = self.cluster_picker_rows(ui, "") {
                        let name = self.class_display_name(id);
                        applied = Some((id, name));
                    }
                    if !self.clusters.is_empty() {
                        ui.separator();
                    }
                    ui.label(RichText::new("or new:").small().color(Color32::GRAY));
                    ui.horizontal(|ui| {
                        ui.add(egui::TextEdit::singleline(&mut self.poly_pick_name)
                            .desired_width(140.0)
                            .hint_text("cluster name"));
                        if ui.button("Apply").clicked() {
                            let name = self.poly_pick_name.trim().to_string();
                            if !name.is_empty() {
                                let id = self.resolve_cluster_id(&name);
                                applied = Some((id, name));
                            }
                        }
                    });
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
            if let Some((_, name)) = applied {
                self.poly_pending = None;
                self.poly_pick_name.clear();
                self.brush_stroke = stroke;
                self.hardneg_label = name;
                self.finish_brush_stroke(leaf_idx, toasts);
            } else if cancel {
                self.poly_pending = None;
                self.poly_pick_name.clear();
            }
        }

        // Wand's commit/discard actions — floated over the canvas rather than
        // in the left toolbox panel, since committing needs `leaf_idx`/`toasts`
        // which only this function (not the toolbox) has access to.
        if self.canvas_tool == CanvasTool::Wand && !self.wand_mask.is_empty() {
            egui::Area::new(egui::Id::new("wand_fill_bar"))
                .fixed_pos(area.min + egui::vec2(8.0, 8.0))
                .show(ctx, |ui| {
                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(format!("{} px selected", self.wand_mask.len()));
                            if ui.button("🪣 Fill").clicked() {
                                self.brush_stroke = std::mem::take(&mut self.wand_mask);
                                self.finish_brush_stroke(leaf_idx, toasts);
                                self.wand_mask_tex = None;
                            }
                            if ui.button("Clear").clicked() {
                                self.wand_mask.clear();
                                self.wand_mask_tex = None;
                            }
                        });
                    });
                });
        }

        // Zoom/pan are universal canvas gestures, not a tool you switch into —
        // scroll to zoom, hold middle-mouse to pan, regardless of active tool
        // (harmless — neither conflicts with any tool's own click/drag).
        // EXCEPTION: while the Brush/Eraser is active, ctrl+scroll resizes the
        // brush instead of zooming, so you can adjust size without switching tools.
        if resp.hovered() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            let ctrl = ui.input(|i| i.modifiers.ctrl);
            let brush_sized = matches!(self.canvas_tool, CanvasTool::Brush | CanvasTool::Eraser);
            if scroll != 0.0 && ctrl && brush_sized {
                let delta = (scroll * 0.15).round() as i32;
                self.brush_size = (self.brush_size as i32 + delta).clamp(4, 200) as u32;
            } else if scroll != 0.0 {
                // clamp allows zooming OUT below 100% (was 1.0..=6.0, which made
                // scrolling out a no-op once already at the fit-to-window default)
                self.canvas_zoom = (self.canvas_zoom * (1.0 + scroll * 0.0015)).clamp(0.25, 6.0);
            }
            if ui.input(|i| i.pointer.middle_down()) {
                self.canvas_pan += ui.input(|i| i.pointer.delta());
            }
        }

        // Tool hotkeys (Photoshop-style): switch tools by keypress, always
        // available regardless of which tool is currently active — guarded
        // by `!focused` so typing in any text field (cluster name, etc.)
        // never accidentally swaps the active tool out from under you.
        // `rebinding` also gates this: while the help overlay is capturing a key,
        // pressing (say) B to bind it must not simultaneously switch to Brush.
        let focused = ctx.memory(|m| m.focused().is_some()) || self.rebinding.is_some();
        if !focused {
            let km = &self.keymap;
            let key_tool = ui.input(|i| {
                if km.pressed(i, "tool.select") { Some(CanvasTool::Select) }
                else if km.pressed(i, "tool.mark_healthy") { Some(CanvasTool::MarkHealthy) }
                else if km.pressed(i, "tool.brush") { Some(CanvasTool::Brush) }
                else if km.pressed(i, "tool.eraser") { Some(CanvasTool::Eraser) }
                else if km.pressed(i, "tool.knife") { Some(CanvasTool::Knife) }
                else if km.pressed(i, "tool.scissor") { Some(CanvasTool::Scissor) }
                else if km.pressed(i, "tool.lasso") { Some(CanvasTool::Lasso) }
                else if km.pressed(i, "tool.wand") { Some(CanvasTool::Wand) }
                else if km.pressed(i, "tool.polygon") { Some(CanvasTool::Polygon) }
                else { None }
            });
            if let Some(tool) = key_tool {
                self.switch_tool_hotkey(tool);
            }
        }

        // ESC: universal cancel — clears selection (canvas OR the Curate
        // tab's gallery, same underlying fields), exits focus mode, and
        // closes the quick-reassign popup. Deliberately OUTSIDE the
        // MarkHealthy gate below (unlike Enter/Delete/R/Ctrl+Z) — it doesn't
        // conflict with anything MarkHealthy binds, and scoping it to "not
        // MarkHealthy" meant switching to that tool silently broke ESC
        // everywhere else in the tab, including the Curate gallery.
        if !focused && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.multi_selected.clear();
            self.selected_region = None;
            self.selected_cluster = None;
            self.quick_reassign_open = false;
            self.overlay_tex = None;
            self.lasso_points.clear(); // cancels a pending Scissor/Knife-polycut/Lasso path
            self.poly_points.clear();
            self.poly_pending = None;
        }

        // right-click context menu + Enter/Delete/Ctrl+Z/reassign-popup
        // shortcuts — act on whatever is currently selected, available for
        // every tool EXCEPT the stamp tool (which already uses right-click
        // for its own undo gesture)
        if self.canvas_tool != CanvasTool::MarkHealthy {
            let effective_sel = self.effective_selection();
            let n_sel = effective_sel.len();
            let mut do_confirm = false;
            let mut do_remove = false;
            let mut do_reassign = false;
            let mut do_reassign_id: Option<i32> = None;
            let mut do_clear = false;
            resp.context_menu(|ui| {
                if n_sel == 0 {
                    ui.label(RichText::new("No regions selected — click or drag a box to select").color(Color32::GRAY));
                    return;
                }
                ui.label(format!("{n_sel} region(s) selected"));
                ui.separator();
                if ui.button("✅ Confirm selected").clicked() {
                    do_confirm = true;
                    ui.close_menu();
                }
                if ui.button("🗑 Reject selected").clicked() {
                    do_remove = true;
                    ui.close_menu();
                }
                ui.menu_button("↪ Move to cluster…", |ui| {
                    // Clickable list of every cluster that already exists
                    // right now — no need to remember/retype a name for the
                    // common case of moving into something already curated.
                    if let Some(id) = self.cluster_picker_rows(ui, "") {
                        do_reassign_id = Some(id);
                        ui.close_menu();
                    }
                    if !self.clusters.is_empty() {
                        ui.separator();
                    }
                    ui.label(RichText::new("or new:").small().color(Color32::GRAY));
                    ui.add(egui::TextEdit::singleline(&mut self.reassign_name)
                        .desired_width(160.0)
                        .hint_text("cluster name"));
                    if ui.button("Apply").clicked() {
                        do_reassign = true;
                        ui.close_menu();
                    }
                });
                if ui.button("✖ Clear selection").clicked() {
                    do_clear = true;
                    ui.close_menu();
                }
            });
            // Keyboard shortcuts for the same effective selection — Enter to
            // confirm, Delete to reject, "R" to pop the quick-reassign
            // picker (below) without needing to right-click first — only
            // when no widget (e.g. the reassign text field) currently has
            // keyboard focus, so typing a cluster name never accidentally
            // triggers one of these.
            if !focused && n_sel > 0 {
                if ui.input(|i| self.keymap.pressed(i, "region.confirm")) {
                    do_confirm = true;
                }
                if ui.input(|i| self.keymap.pressed(i, "region.reject")) {
                    do_remove = true;
                }
                if ui.input(|i| self.keymap.pressed(i, "region.reassign")) {
                    if let Some(pos) = ctx.pointer_hover_pos() {
                        self.quick_reassign_pos = pos;
                    }
                    self.quick_reassign_open = true;
                }
            }
            // General undo (last remove or knife cut) — Ctrl+Z. Scoped the
            // same way as Enter/Delete above (not while a text field has
            // focus) and gated on `canvas_tool != MarkHealthy` by the outer
            // `if`, so it stays mutually exclusive with the stamp tool's own
            // Ctrl+Z (undo_hardneg) — never both firing off one keypress.
            if !focused && ui.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Z)) {
                // Mid-draw, Ctrl+Z undoes the last placed polygon NODE, not
                // the last committed structural edit — nothing from this
                // polygon has been committed yet.
                if self.canvas_tool == CanvasTool::Polygon && !self.poly_points.is_empty() {
                    self.poly_points.pop();
                } else {
                    self.undo_last_edit(toasts);
                }
            }
            // Redo, scoped identically to the Ctrl+Z above. Ctrl+Y rather than
            // Ctrl+Shift+Z: Shift is already the additive modifier for canvas
            // selection, and overloading it here would make redo depend on
            // whether a drag happened to be in progress.
            if !focused && ui.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Y)) {
                self.redo_last_edit(toasts);
            }
            if do_confirm {
                self.confirm_regions(&effective_sel, toasts);
            }
            if do_remove {
                self.remove_regions(&effective_sel, toasts, true);
                self.multi_selected.clear();
                self.selected_region = None;
            }
            if let Some(id) = do_reassign_id {
                let name = self.class_display_name(id);
                self.reassign_ids(&effective_sel, id, &name, toasts);
                self.multi_selected.clear();
                self.selected_region = None;
            }
            if do_reassign {
                let name = self.reassign_name.trim().to_string();
                if !name.is_empty() {
                    let id = self.resolve_cluster_id(&name);
                    self.reassign_ids(&effective_sel, id, &name, toasts);
                }
                self.reassign_name.clear();
                self.multi_selected.clear();
                self.selected_region = None;
            }
            if do_clear {
                self.multi_selected.clear();
                self.selected_region = None;
            }
        }

        // Standalone quick-reassign popup ("R" hotkey above) — the same
        // cluster-picker affordance as the context-menu submenu, but
        // reachable without right-clicking first.
        if self.quick_reassign_open {
            let effective_sel = self.effective_selection();
            let mut applied: Option<(i32, String)> = None;
            let mut close = false;
            egui::Window::new("Move to cluster")
                .id(egui::Id::new("pipeline_quick_reassign"))
                .collapsible(false)
                .resizable(false)
                .current_pos(self.quick_reassign_pos)
                .show(ctx, |ui| {
                    if effective_sel.is_empty() {
                        ui.label(RichText::new("Nothing selected.").color(Color32::GRAY));
                    } else {
                        ui.label(format!("{} region(s) selected", effective_sel.len()));
                        ui.separator();
                        if let Some(id) = self.cluster_picker_rows(ui, "") {
                            let name = self.class_display_name(id);
                            applied = Some((id, name));
                            close = true;
                        }
                        if !self.clusters.is_empty() {
                            ui.separator();
                        }
                        ui.label(RichText::new("or new:").small().color(Color32::GRAY));
                        ui.horizontal(|ui| {
                            ui.add(egui::TextEdit::singleline(&mut self.reassign_name)
                                .desired_width(140.0)
                                .hint_text("cluster name"));
                            if ui.button("Apply").clicked() {
                                let name = self.reassign_name.trim().to_string();
                                if !name.is_empty() {
                                    let id = self.resolve_cluster_id(&name);
                                    applied = Some((id, name));
                                }
                                close = true;
                            }
                        });
                    }
                    if ui.small_button("Cancel").clicked() {
                        close = true;
                    }
                });
            if let Some((id, name)) = applied {
                self.reassign_ids(&effective_sel, id, &name, toasts);
                self.reassign_name.clear();
                self.multi_selected.clear();
                self.selected_region = None;
            }
            if close {
                self.quick_reassign_open = false;
            }
        }
    }

    /// Rubber-band multi-select: add every anomaly region on `leaf_idx` whose bbox
    /// intersects the leaf-pixel rect `[lx0,ly0]..[lx1,ly1]` to `multi_selected`,
    /// for bulk reassign/remove in the gallery panel.
    fn select_regions_in_rect(&mut self, leaf_idx: usize, lx0: f32, ly0: f32, lx1: f32, ly1: f32) {
        let (x0, x1) = (lx0.min(lx1), lx0.max(lx1));
        let (y0, y1) = (ly0.min(ly1), ly0.max(ly1));
        for (i, r) in self.regions.iter().enumerate() {
            if r.leaf != leaf_idx || !self.region_visible(i) {
                continue;
            }
            if let Some(cid) = self.selected_cluster {
                if self.labels[i] != cid {
                    continue;
                }
            }
            let [bx, by, bw, bh] = r.bbox_leaf;
            let (rx0, ry0, rx1, ry1) = (bx as f32, by as f32, (bx + bw) as f32, (by + bh) as f32);
            if rx0 < x1 && rx1 > x0 && ry0 < y1 && ry1 > y0 {
                self.multi_selected.insert(i);
            }
        }
    }

    /// Tile-picker-style magnifier: a zoomed, crosshair-marked view of the leaf
    /// RGBA around leaf-px `(cx, cy)`, pinned beside `cursor_screen`.
    fn show_hardneg_loupe(
        &mut self, ui: &mut Ui, ctx: &Context, leaf_idx: usize,
        cx: f32, cy: f32, cursor_screen: egui::Pos2, scale: f32,
    ) {
        let Some(leaf) = self.results.get(leaf_idx) else { return };
        let t = self.hardneg_tile as f32;
        const MARGIN: f32 = 0.10; // show ~10% beyond the placement square on each side
        let r_src = t * (1.0 + 2.0 * MARGIN);
        let avail = ui.max_rect();
        let max_sz = (avail.width().min(avail.height()) - 24.0).clamp(120.0, 500.0);
        let loupe_sz = (r_src * scale * self.hardneg_zoom).clamp(120.0, max_sz);
        let factor = loupe_sz / r_src;
        let s = loupe_sz.round() as usize;
        let half = s as f32 / 2.0;
        let (lw, lh) = (leaf.w as f32, leaf.h as f32);
        let mut pixels = Vec::with_capacity(s * s);
        for j in 0..s {
            for i in 0..s {
                let sx = cx + (i as f32 - half) / factor;
                let sy = cy + (j as f32 - half) / factor;
                let col = if sx >= 0.0 && sy >= 0.0 && sx < lw && sy < lh {
                    let o = ((sy as u32 * leaf.w + sx as u32) * 4) as usize;
                    Color32::from_rgba_unmultiplied(leaf.rgba[o], leaf.rgba[o + 1], leaf.rgba[o + 2], leaf.rgba[o + 3])
                } else {
                    Color32::from_gray(30)
                };
                pixels.push(col);
            }
        }
        let ci = egui::ColorImage { size: [s, s], pixels };
        let tex = self.hardneg_loupe_tex.get_or_insert_with(|| {
            ctx.load_texture("pipe_hardneg_loupe", egui::ColorImage::new([1, 1], Color32::TRANSPARENT),
                egui::TextureOptions::NEAREST)
        });
        tex.set(ci, egui::TextureOptions::NEAREST);
        let lid = tex.id();

        let sq_half = t * scale / 2.0;
        let gap = 14.0;
        let mut lx = cursor_screen.x + sq_half + gap;
        if lx + loupe_sz > avail.max.x {
            lx = cursor_screen.x - sq_half - gap - loupe_sz;
        }
        let lx = lx.clamp(avail.min.x, (avail.max.x - loupe_sz).max(avail.min.x));
        let ly = (cursor_screen.y - loupe_sz / 2.0).clamp(avail.min.y, (avail.max.y - loupe_sz).max(avail.min.y));
        let lrect = egui::Rect::from_min_size(egui::pos2(lx, ly), egui::Vec2::splat(loupe_sz));

        let painter = ui.painter();
        painter.rect_filled(lrect.expand(4.0), egui::Rounding::same(6.0), Color32::from_black_alpha(160));
        painter.image(lid, lrect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
        let c = lrect.center();
        painter.line_segment([c - egui::vec2(7.0, 0.0), c + egui::vec2(7.0, 0.0)], egui::Stroke::new(1.0, Color32::WHITE));
        painter.line_segment([c - egui::vec2(0.0, 7.0), c + egui::vec2(0.0, 7.0)], egui::Stroke::new(1.0, Color32::WHITE));
        let half_t = t * factor;
        painter.rect_stroke(egui::Rect::from_center_size(c, egui::Vec2::splat(half_t)), 0.0,
            egui::Stroke::new(1.5, ui_kit::ACCENT()));
        painter.rect_stroke(lrect.expand(1.5), egui::Rounding::same(3.0), egui::Stroke::new(2.0, Color32::from_gray(235)));
    }

    /// Find the anomaly region at leaf-pixel (lx, ly) on `leaf_idx` (smallest match)
    /// and select it — highlights its gallery tile and jumps to its page.
    /// Hit-test: the smallest-area (topmost, in overlap terms) non-removed
    /// region on `leaf_idx` covering leaf-pixel `(lx, ly)`, if any. Shared by
    /// `select_region_at` (plain click) and the Select tool's ctrl+click
    /// multi-select toggle.
    fn region_at(&self, leaf_idx: usize, lx: f32, ly: f32) -> Option<usize> {
        if lx < 0.0 || ly < 0.0 {
            return None;
        }
        let (px, py) = (lx as u32, ly as u32);
        let mut best: Option<usize> = None;
        let mut best_area = u32::MAX;
        for (i, r) in self.regions.iter().enumerate() {
            if r.leaf != leaf_idx || !self.region_visible(i) {
                continue;
            }
            let [bx, by, bw, bh] = r.bbox_leaf;
            if px < bx || py < by || px >= bx + bw || py >= by + bh {
                continue;
            }
            if r.mask[((py - by) * bw + (px - bx)) as usize] {
                let a = bw * bh;
                if a < best_area {
                    best_area = a;
                    best = Some(i);
                }
            }
        }
        best
    }

    fn select_region_at(&mut self, leaf_idx: usize, lx: f32, ly: f32) {
        if let Some(i) = self.region_at(leaf_idx, lx, ly) {
            self.selected_region = Some(i);
            self.selected_cluster = Some(self.labels[i]);
            // jump the gallery to the page that shows this region
            let cl = self.labels[i];
            let pos = (0..self.regions.len())
                .filter(|&j| self.region_visible(j) && self.labels[j] == cl)
                .position(|j| j == i)
                .unwrap_or(0);
            self.gallery_page = pos / GALLERY_PER_PAGE;
            self.scroll_to_selected = true; // scroll the gallery to this tile
            self.overlay_tex = None; // cluster changed → rebuild
        }
    }

    fn ensure_overlay(&mut self, ctx: &Context) {
        let Some(idx) = self.selected_idx else { return };
        let sel = self.focus_cluster();
        let key = (idx, sel, self.show_recon, (self.overlay_alpha * 100.0) as u32,
                   self.overlay_outline, self.regions.len());
        if self.overlay_key == Some(key) && self.overlay_tex.is_some() {
            return;
        }
        let Some(leaf) = self.results.get(idx) else { return };
        let (w, h) = (leaf.w as usize, leaf.h as usize);
        let mut px = leaf.rgba.clone();

        // Which leaf pixels an (un-removed) anomaly region already covers —
        // computed BEFORE the recon-tint pass so the tint can skip them. A
        // pixel that's both "reconstructed" and part of a detected anomaly
        // should show the cluster colour alone (dimmed or not — Focus mode
        // dims non-selected regions, it never hides them), not a blend of the
        // two stacked on top of each other. Deliberately NOT filtered by
        // `sel`: a dimmed region still occupies this pixel, so the tint must
        // still skip it — filtering here would let cyan bleed under regions
        // that are merely dimmed, not actually absent.
        let mut covered = vec![false; w * h];
        if !self.overlay_outline {
            for ri in 0..self.regions.len() {
                if !self.region_visible(ri) {
                    continue;
                }
                let r = &self.regions[ri];
                if r.leaf != idx {
                    continue;
                }
                let [bx, by, bw, bh] = r.bbox_leaf;
                for ly in 0..bh {
                    for lx in 0..bw {
                        if !r.mask[(ly * bw + lx) as usize] {
                            continue;
                        }
                        let (gx, gy) = (bx + lx, by + ly);
                        if (gx as usize) < w && (gy as usize) < h {
                            covered[gy as usize * w + gx as usize] = true;
                        }
                    }
                }
            }
        }

        // reconstruction preview: tint the FILLED-IN area (reconstructed leaf where
        // the visible cutout is missing) in cyan, so the damage reads as holes in
        // the whole intact leaf. Skips anything an anomaly region already covers —
        // the cluster colour always wins, painted next; never stack the two.
        if self.show_recon && !leaf.recon_mask.is_empty() {
            let rs = worker::RECON_PREVIEW;
            for y in 0..h {
                let my = (y * rs / h.max(1)).min(rs - 1);
                for x in 0..w {
                    if covered[y * w + x] {
                        continue;
                    }
                    let mx = (x * rs / w.max(1)).min(rs - 1);
                    let o = (y * w + x) * 4;
                    if leaf.recon_mask[my * rs + mx] && px[o + 3] < 128 {
                        // Was a hardcoded 0.5 blend, completely ignoring the
                        // opacity slider — a real QA-reported bug ("opacity
                        // doesn't work on Hole recon").
                        px[o] = lerp_u8(px[o], 70, self.overlay_alpha);
                        px[o + 1] = lerp_u8(px[o + 1], 200, self.overlay_alpha);
                        px[o + 2] = lerp_u8(px[o + 2], 225, self.overlay_alpha);
                        px[o + 3] = 255;
                    }
                }
            }
        }

        // ── live preview, while the run is still going ──
        // Clustering only happens once, over every leaf, at the very end — so
        // until then a finished leaf has no regions and shows nothing, and the
        // detection you already paid for is invisible until the whole batch is
        // done. `PipelineLeaf.anomaly` is the restitched per-leaf mask and has
        // been streaming with every leaf all along (unread until now), so this
        // costs one extra pass over pixels the worker already computed.
        //
        // One flat colour, not family colours: families come from the clustering
        // that hasn't run yet, and colouring this as if it were final would imply
        // a classification nothing has made. Painted in BOTH fill and outline
        // mode, since outline mode draws vector contours from `regions` — which
        // by definition don't exist during the preview.
        let previewing = leaf.anomaly.len() == w * h
            && !self.regions.iter().any(|r| r.leaf == idx);
        if previewing {
            for i in 0..w * h {
                let o = i * 4;
                if leaf.anomaly[i] && px[o + 3] > 0 {
                    px[o]     = lerp_u8(px[o],     255, self.overlay_alpha);
                    px[o + 1] = lerp_u8(px[o + 1], 170, self.overlay_alpha);
                    px[o + 2] = lerp_u8(px[o + 2],  40, self.overlay_alpha);
                }
            }
        }

        // Fill mode: bake the regions into the texture (family colour, opacity =
        // slider). Outline mode keeps the texture clean and draws smooth vector
        // contours in show_canvas instead. Focus mode (item 1): a selected
        // cluster paints at full opacity, every other visible region paints
        // dimmed — never skipped, so the rest of the leaf stays visible as
        // context instead of vanishing when you click a cluster.
        if !self.overlay_outline {
            for ri in 0..self.regions.len() {
                if !self.region_visible(ri) {
                    continue;
                }
                let r = &self.regions[ri];
                if r.leaf != idx {
                    continue;
                }
                let focused = sel.map_or(true, |cid| self.labels[ri] == cid);
                let col = cluster_color(self.labels[ri]);
                if focused {
                    paint_region(&mut px, w, h, r, col, self.overlay_alpha);
                } else {
                    paint_region(&mut px, w, h, r, dim_color(col), self.overlay_alpha * 0.35);
                }
            }
        }

        let color: Vec<Color32> = px
            .chunks_exact(4)
            .map(|c| Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]))
            .collect();
        let tex = ctx.load_texture(
            format!("pipeline_overlay_{idx}_{sel:?}_{}_{}_{}", self.show_recon, key.3, self.overlay_outline),
            egui::ColorImage { size: [w, h], pixels: color },
            egui::TextureOptions::LINEAR,
        );
        self.overlay_tex = Some(tex);
        self.overlay_key = Some(key);
    }

    fn ensure_region_thumb(&mut self, ctx: &Context, i: usize) {
        if i >= self.regions.len() || self.region_thumbs[i].is_some() {
            return;
        }
        let r = &self.regions[i];
        let sz = r.crop_size as usize;
        let pixels: Vec<Color32> = r
            .crop
            .chunks_exact(4)
            .map(|c| Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]))
            .collect();
        let tex = ctx.load_texture(
            format!("region_thumb_{i}"),
            egui::ColorImage { size: [sz, sz], pixels },
            egui::TextureOptions::LINEAR,
        );
        self.region_thumbs[i] = Some(tex);
    }

    /// Compact EC/MC morphology readout for the currently-selected leaf.
    fn show_leaf_morphology(&self, ui: &mut Ui) {
        ui_kit::section_header(ui, "Selected leaf — morphology");
        let morph = self
            .selected_idx
            .and_then(|i| self.results.get(i))
            .and_then(|l| l.morph.as_ref());
        let Some(m) = morph else {
            ui_kit::caption(ui, "EC/MC metrics appear once the leaf is processed.");
            ui.separator();
            return;
        };
        egui::Grid::new("pipe_morph").num_columns(3).striped(true).show(ui, |ui| {
            ui.label("");
            ui.label(RichText::new("EC").small().strong());
            ui.label(RichText::new("MC").small().strong());
            ui.end_row();
            let row = |ui: &mut Ui, name: &str, a: String, b: String| {
                ui.label(RichText::new(name).small());
                ui.label(RichText::new(a).small());
                ui.label(RichText::new(b).small());
                ui.end_row();
            };
            row(ui, "Length", format!("{:.1}", m.ec_length), format!("{:.1}", m.mc_length));
            row(ui, "Width", format!("{:.1}", m.ec_width), format!("{:.1}", m.mc_width));
            row(ui, "Area (px)", format!("{}", m.ec_area), format!("{}", m.mc_area));
            row(ui, "Outline pts", format!("{}", m.ec_outline_count), format!("{}", m.mc_outline_count));
            row(ui, "Shape idx", format!("{:.3}", m.ec_shape_index), format!("{:.3}", m.mc_shape_index));
            row(ui, "Circularity", format!("{:.3}", m.ec_circularity), format!("{:.3}", m.mc_circularity));
            row(
                ui,
                "Entropy",
                format!("{:.4}", m.ec_approximate_entropy),
                format!("{:.4}", m.mc_spectral_entropy),
            );
        });
        ui.separator();
    }

    /// The review panel, arranged the way the work actually goes:
    /// **this detection → the families → this leaf's detections**, with
    /// everything else folded away underneath.
    ///
    /// It used to be four peer tabs — Metrics, Clusters, Curate, Log — which put
    /// the surface used ten thousand times a batch behind a tab switch, and gave
    /// a static seven-row table the same prominence as the review grid. Tabs
    /// also hid state: with Metrics open you could not see what was selected.
    fn show_cluster_panel(&mut self, ui: &mut Ui, ctx: &Context, toasts: &mut ToastManager) {
        if self.regions.is_empty() {
            ui.add_space(8.0);
            ui.label(RichText::new(if self.running {
                "Detecting… anomalies appear here as each leaf finishes."
            } else {
                "Run an analysis to review anomalies here."
            }).small().color(ui_kit::MUTED()));
            return;
        }
        // Measured BEFORE the scroll area: a width read inside a scrollable
        // parent takes part in that parent's own width negotiation, which is the
        // loop that made the panel creep wider on every mouse move. Measured out
        // here it is just a number.
        let row_w = (ui.available_width() - 14.0).max(180.0);
        egui::ScrollArea::vertical().id_salt("review_panel").auto_shrink([false, false])
            // Always reserve the scrollbar. Otherwise selecting a detection adds
            // the card, content passes the panel height, the bar appears and
            // everything shifts by its width — the 1-2px jump on first click.
            .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
            .show(ui, |ui| {
            // HARD width cap for everything below.
            //
            // `auto_shrink(false)` alone was not enough: a child that asks for
            // more than the panel has — the detection card's three buttons in a
            // non-wrapping row — still reports a wider minimum, the resizable
            // SidePanel grows to satisfy it, and next frame there is more width
            // to fill. It crept wider on every mouse move until it hit the max
            // I had set, which is why capping the panel hid the symptom without
            // removing the cause. Pinning max_width here means no descendant can
            // ever ask for more than the panel already has.
            ui.set_max_width(ui.available_width());
            self.show_selected_detection(ui, toasts);
            self.show_family_legend(ui, row_w);
            ui.add_space(6.0);
            self.show_curate_tab(ui, ctx, toasts);
            self.show_verdict_block(ui, row_w, toasts);

            ui.add_space(10.0);
            // The way OUT of review. "Improve the model" and "Export" used to sit
            // here as two collapsed headers; they are decisions about what to do
            // with a finished run, so they belong on the Done screen and nowhere
            // else. What review needs at this point is not machinery but an exit,
            // and the stage pill alone was not discoverable enough to be it.
            if ui_kit::primary_button(ui, "Finish and proceed to export").clicked() {
                self.perform_action_deferred = Some("flow.finish".into());
            }
            {
                let (rev, _rej, tot) = self.review_counts();
                if rev < tot {
                    ui.label(RichText::new(format!(
                        "{} of {} leaves reviewed — you can export at any point.",
                        fmt_thousands(rev), fmt_thousands(tot)))
                        .small().color(ui_kit::MUTED()));
                }
            }
            ui.add_space(10.0);
            egui::CollapsingHeader::new("Leaf measurements")
                .id_salt("panel_metrics").default_open(false)
                .show(ui, |ui| self.show_leaf_morphology(ui));
            egui::CollapsingHeader::new("All families across the run")
                .id_salt("panel_clusters").default_open(false)
                .show(ui, |ui| self.show_clusters_tab(ui, toasts));
            egui::CollapsingHeader::new("Run log")
                .id_salt("panel_log").default_open(false)
                .show(ui, |ui| self.show_log_tab(ui));
            // Messages, including the ones that self-dismissed. Toasts were being
            // recorded into a history that nothing could read — so a message that
            // flashed while you were looking elsewhere was simply gone.
            egui::CollapsingHeader::new("Recent messages")
                .id_salt("panel_msgs").default_open(false)
                .show(ui, |ui| {
                    let h = toasts.history();
                    if h.is_empty() {
                        ui.label(RichText::new("Nothing yet.").small().color(ui_kit::MUTED()));
                        return;
                    }
                    egui::ScrollArea::vertical().max_height(200.0).id_salt("msg_hist")
                        .show(ui, |ui| {
                            for r in h.iter().rev().take(60) {
                                let col = match r.kind {
                                    crate::widgets::ToastKind::Error   => Color32::from_rgb(226, 122, 106),
                                    crate::widgets::ToastKind::Warning => Color32::from_rgb(222, 178, 96),
                                    crate::widgets::ToastKind::Success => ui_kit::ACCENT(),
                                    crate::widgets::ToastKind::Info    => ui_kit::MUTED(),
                                };
                                ui.horizontal_top(|ui| {
                                    ui.label(RichText::new(&r.at)
                                        .text_style(ui_kit::numeric()).small().color(ui_kit::MUTED()));
                                    ui.label(RichText::new(&r.message).small().color(col));
                                });
                            }
                        });
                });
        });
    }

    /// The current detection, as a card: what the model called it, how sure it
    /// was, how big it is, and the three verdicts with their keys printed.
    fn show_selected_detection(&mut self, ui: &mut Ui, toasts: &mut ToastManager) {
        let Some(ri) = self.selected_region.filter(|&i| self.region_visible(i)) else {
            ui.add_space(6.0);
            ui.label(RichText::new("No detection selected").small().color(ui_kit::MUTED()));
            ui.label(RichText::new("Click one on the leaf, or press Down to step through them.")
                .small().color(ui_kit::MUTED()));
            ui.add_space(6.0);
            return;
        };
        let cid = self.labels[ri];
        let fam = self.class_display_name(cid);
        let area = self.region_area.get(ri).copied().unwrap_or(0);
        let leaf = self.regions[ri].leaf;
        let leaf_px = self.leaf_valid_px.get(leaf).copied().unwrap_or(1).max(1);
        let pct = 100.0 * area as f32 / leaf_px as f32;
        let confirmed = self.persisted.contains(&ri);
        let flagged = self.flagged.contains(&ri);

        egui::Frame::none()
            .fill(ui.visuals().faint_bg_color)
            .inner_margin(egui::Margin::same(9.0))
            .rounding(egui::Rounding::same(5.0))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    family_swatch(ui, cid, 12.0);
                    ui.label(RichText::new(&fam).strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // Human-confirmed vs model-proposed must be legible at a
                        // glance — they are different scientific claims, and the
                        // export now carries the distinction too.
                        if confirmed {
                            ui.label(RichText::new("confirmed").small().color(ui_kit::ACCENT()));
                        } else {
                            ui.label(RichText::new("model-proposed").small().color(ui_kit::MUTED()));
                        }
                        if flagged {
                            ui.label(RichText::new("set aside").small()
                                .color(Color32::from_rgb(225, 180, 90)));
                        }
                    });
                });
                ui.label(RichText::new(format!("{area} px · {pct:.2}% of leaf"))
                    .text_style(ui_kit::numeric()).color(ui_kit::MUTED()));
                ui.add_space(6.0);
                // Key labels resolved up front: capturing `self` in the closure
                // would hold an immutable borrow across the `perform_action`
                // calls below, which need `&mut self`.
                let k_ok = shortcuts::key_label(self.keymap.key("region.confirm"));
                let k_no = shortcuts::key_label(self.keymap.key("region.reject"));
                let k_fl = shortcuts::key_label(self.keymap.key("region.flag"));
                let mut act: Option<&str> = None;
                // WRAPPING. A plain `horizontal` reports the full un-wrapped width
                // as its minimum, which is what pushed the panel wider every
                // frame — see the note in show_cluster_panel.
                ui.horizontal_wrapped(|ui| {
                    if ui.button(format!("Accept  {k_ok}")).clicked() { act = Some("region.confirm"); }
                    if ui.button(format!("Reject  {k_no}")).clicked() { act = Some("region.reject"); }
                    if ui.button(format!("Aside  {k_fl}")).clicked() { act = Some("region.flag"); }
                });
                if let Some(a) = act {
                    self.perform_action(a, toasts);
                }
            });
        ui.add_space(8.0);
    }

    /// Mining and retraining. Lives on the DONE screen only.
    ///
    /// It was in the review panel, which put long-running machinery most users
    /// never touch in the column they work in all day. Teaching the model is
    /// something you decide once a run is finished — same moment as export — so
    /// it belongs at the end, next to that decision.
    ///
    /// The export controls that used to sit beside it here are gone entirely:
    /// the Done screen already has them, and two sets of the same checkboxes in
    /// different places is worse than none.
    fn show_improve_model(&mut self, ui: &mut Ui) {
        // `Some(true)` FORCES the header open for one frame, then reverts to
        // None so the user keeps control of it afterwards. This is what the
        // "Teach the model" button drives — it used to send you to another
        // screen with a toast telling you what to look for.
        let force_open = if self.improve_open_req {
            self.improve_open_req = false;
            Some(true)
        } else {
            None
        };
        egui::CollapsingHeader::new("Improve the model")
            .id_salt("curate_improve")
            .default_open(false)
            .open(force_open)
            .show(ui, |ui| {
        let can_retrain = self.output_folder.is_some() && self.eff_head().is_some()
            && self.eff_dino().is_some() && !self.retraining && !self.running && !self.mining;

        // ── the simple path ────────────────────────────────────────────────
        // What someone teaching the model actually needs to know: what it will
        // learn from, and that the base set is handled. Everything that used to
        // live here — the base-set picker, base rows, the anchor slider, cold
        // start, the diagnostics dump and hard-negative mining — is a tuning
        // knob, and having six of them above the one button that does the work
        // is what made this section read as an expert panel.
        // Re-probe whenever it is still unset. Resolving only once in `default()`
        // meant a base set that was not present at launch — or a launch from a
        // working directory the search did not cover — stayed unset for the whole
        // session with no way back except the manual picker.
        if self.retrain_base_set.is_none() {
            self.retrain_base_set = Self::default_base_set();
        }
        let curated = self.curation_row_count();
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!(
                "{} curated example{} to learn from.",
                fmt_thousands(curated), if curated == 1 { "" } else { "s" }))
                .strong());
            if curated > 0 && ui.small_button("Delete all…")
                .on_hover_text("Erase every curated example in this output folder and \
                                start the flywheel clean. Use when the folder has been \
                                contaminated by mislabelled or experimental curations.\n\n\
                                Does NOT touch any head file — a head already retrained \
                                from bad curations stays as it is; reselect the original.")
                .clicked()
            {
                self.confirm_clear_curations = true;
            }
        });
        match &self.retrain_base_set {
            Some(p) => {
                let name = p.file_name().map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                ui.label(RichText::new(format!(
                    "Base set: {name} — {} rows mixed in automatically.",
                    fmt_thousands(self.effective_base_rows())))
                    .small().color(ui_kit::MUTED()))
                    .on_hover_text(BASE_ROWS_HELP);
            }
            None => {
                let looked: Vec<String> = Self::base_set_candidates()
                    .iter().map(|p| p.display().to_string()).collect();
                ui.label(RichText::new(
                    "No base training set found in models/ — a retrain without one \
                     discards most of what the head already knew (IoU 0.475 -> 0.125). \
                     Set one under Advanced.")
                    .small().color(Color32::from_rgb(220, 170, 90)))
                    .on_hover_text(format!("Searched:\n{}", looked.join("\n")));
            }
        }
        // The retrain button itself, THEN the knobs. The action someone came
        // here for should not be below six settings they were told not to touch.
        ui.horizontal(|ui| {
            ui.add_enabled_ui(can_retrain, |ui| {
                if ui.button("🔄 Retrain from curations")
                    .on_hover_text("Fine-tune the current few-shot head on everything confirmed/\n\
                                    rejected so far this run (and any prior curations in the same\n\
                                    output folder) — writes a NEW sibling file, never overwrites\n\
                                    the original, so you can review before switching to it.\n\
                                    Requires: few-shot head + DINO model + output folder.")
                    .clicked()
                {
                    self.start_retrain();
                }
            });
            if self.retraining {
                ui_kit::busy(ui, &self.retrain_stage);
                if ui.small_button("Cancel")
                    .on_hover_text("Stop the retrain. The current head file is left untouched — \
                                    a new one is only written when training finishes.")
                    .clicked()
                {
                    self.retrain_cancel.store(true, Ordering::Relaxed);
                }
            }
        });
        if self.running && !self.retraining {
            ui.label(RichText::new("Retrain is disabled while the pipeline is running.")
                .small().color(Color32::GRAY));
        }
        if !self.retrain_log.is_empty() {
            egui::ScrollArea::vertical().max_height(80.0).id_salt("pipeline_retrain_log")
                .show(ui, |ui| {
                    for line in self.retrain_log.iter().rev().take(20) {
                        ui.label(RichText::new(line).small());
                    }
                });
        }

        // ── everything else ────────────────────────────────────────────────
        ui.add_space(8.0);
        egui::CollapsingHeader::new("Advanced")
            .id_salt("improve_advanced")
            .default_open(false)
            .show(ui, |ui| {
        self.pick_row(ui, "Base training set (.bin)", Pick::BaseSet);
        ui.checkbox(&mut self.retrain_auto_base_rows, "Scale base rows with the curation count")
            .on_hover_text(BASE_ROWS_HELP);
        ui.horizontal(|ui| {
            ui.label("base rows");
            ui.add_enabled(!self.retrain_auto_base_rows,
                egui::DragValue::new(&mut self.retrain_base_rows)
                    .range(0..=400_000).speed(1000));
            if self.retrain_auto_base_rows {
                ui.label(RichText::new(format!("auto: {}", fmt_thousands(self.effective_base_rows())))
                    .small().color(ui_kit::MUTED()));
            }
        });
        ui.horizontal(|ui| {
            ui.label("anchor to current head");
            ui.add(egui::Slider::new(&mut self.retrain_anchor, 0.0..=1.0).fixed_decimals(2))
                .on_hover_text("Pull the L2 penalty toward the CURRENT head instead of toward \
                                zero.\n\n\
                                0 = old behaviour: curations must out-vote the base rows for \
                                influence.\n\
                                1 = anchored: curations are the only data, and the penalty only \
                                bounds how far the solution may travel. Where curations say \
                                nothing, those weights simply stay put.\n\n\
                                Measured on held-out leaves (LEARNS = agreement with held-out \
                                curations, KEEPS = ground-truth IoU):\n\
                                   no retrain          0.195 / 0.475\n\
                                   base 50k, no anchor 0.969 / 0.460\n\
                                   base 10k, no anchor 0.985 / 0.431\n\
                                   base 10k + anchor   0.942 / 0.476  <- best\n\n\
                                A brand-new class has no prior, so it falls back to ordinary \
                                zero-centered L2 and can still learn freely.");
        });
        ui.label(RichText::new(
            "Base rows stop a retrain from discarding what the head already knew (without \
             them: IoU 0.475 -> 0.125). The anchor does the same job by bounding travel \
             rather than out-voting the curations, so 10k rows + anchor beats 50k rows \
             alone on BOTH learning and retention. Build a base set with \
             1Help/eval/export_base_set.py."
        ).small().color(Color32::GRAY));
        ui.add_space(4.0);
        ui.checkbox(&mut self.retrain_cold_start, "Train from scratch (no warm start)")
            .on_hover_text("Every retrain already reads ALL accumulated curations, not just new \
                            ones — this only changes where the solver STARTS. Normally it warm-\
                            starts from the current head's own coefficients; with this on, any \
                            class that has curated examples this run starts from zero instead, so \
                            its result depends only on the curated evidence itself, not on \
                            whatever the head happened to already believe (which may itself be a \
                            degraded result from an earlier retrain). Classes with NO curated \
                            examples in this output folder's history are unaffected either way — \
                            there's nothing to retrain them from, so they keep their existing \
                            (e.g. original bulk-trained) weights exactly as normal retrain does. \
                            Good for diagnosing a retrain that keeps getting worse: if this comes \
                            out meaningfully different (and better), the warm start was the \
                            problem; if it's the same or worse, look at the curated data instead.");
        ui.checkbox(&mut self.retrain_dump, "Dump training data (diagnostics)")
            .on_hover_text("Writes <output>/retrain_diag/: retrain_dump.bin (the EXACT feature \
                            rows, classes and weights this retrain trains on) and \
                            retrain_diag.json (per-class coefficient norms + intercepts before \
                            and after, plus solver settings and convergence). Lets an \
                            independent solver be fitted on byte-identical data, so the DATA and \
                            the SOLVER can finally be told apart — every data-side explanation \
                            for 'retrains come back too conservative' has tested null so far, and \
                            this holds the data fixed to isolate warm-start/freeze/L-BFGS. \
                            The .bin is rows x dim x 4 bytes — often several hundred MB — so \
                            leave this off for normal runs.");
        // Hard-negative mining: powerful, slow, and permanently inflates every
        // future retrain, so it belongs behind Advanced rather than being the
        // first thing in the section.
        ui.add_space(6.0);
        ui.separator();
        self.show_mine_hardneg(ui);
            }); // end "Advanced"
            }); // end "Improve the model"
    }
    /// The verdicts, as the panel's visual anchor: one big affirmative, two
    /// smaller corrections, and the leaf-level reject in a warning colour.
    ///
    /// They sit BELOW the grid on purpose — you look, then you decide, and the
    /// buttons should be where your eye ends up rather than something you scroll
    /// past on the way in. Sizes encode frequency: the accept path is pressed
    /// far more than the others, so it gets the width and the colour.
    fn show_verdict_block(&mut self, ui: &mut Ui, row_w: f32, toasts: &mut ToastManager) {
        // No family focused: still offer the leaf-level verdicts rather than
        // showing nothing. Landing on a leaf and finding no buttons at all read
        // as a dead end — the panel looked broken until you happened to click a
        // family.
        let Some(cid) = self.selected_cluster else {
            ui.add_space(10.0);
            ui.label(RichText::new("Pick a family above to judge it as a group.")
                .small().color(ui_kit::MUTED()));
            ui.add_space(4.0);
            if let Some(li) = self.selected_idx {
                let k_x = shortcuts::key_label(self.keymap.key("leaf.reject"));
                let k_m = shortcuts::key_label(self.keymap.key("leaf.reviewed"));
                let rejected = self.rejected_leaves.contains(&li);
                let reviewed = self.reviewed.contains(&li);
                if !rejected && ui.add_sized([row_w, 30.0],
                    egui::Button::new(RichText::new(if reviewed {
                        format!("Reviewed   {k_m}")
                    } else {
                        format!("Mark this leaf reviewed   {k_m}")
                    }).strong().color(if reviewed { ui_kit::on_accent() } else { ui.visuals().text_color() }))
                    .fill(if reviewed { ui_kit::ACCENT() } else { ui.visuals().widgets.inactive.bg_fill }))
                    .clicked()
                {
                    self.toggle_reviewed(li, toasts);
                }
                ui.add_space(4.0);
                let (txt, fill) = if rejected {
                    (format!("Restore this leaf   {k_x}"), Color32::from_rgb(90, 90, 95))
                } else {
                    (format!("Reject whole leaf   {k_x}"), Color32::from_rgb(168, 72, 27))
                };
                if ui.add_sized([row_w, 30.0],
                    egui::Button::new(RichText::new(txt).color(Color32::WHITE).strong()).fill(fill))
                    .clicked()
                {
                    self.toggle_reject_leaf(li, toasts);
                }
            }
            return;
        };
        let fam = self.class_display_name(cid);
        let pending: Vec<usize> = (0..self.regions.len())
            .filter(|&i| {
                self.region_visible(i) && self.labels[i] == cid && !self.persisted.contains(&i)
                    && (!self.filter_leaf_only
                        || self.selected_idx.map_or(true, |li| self.regions[i].leaf == li))
            })
            .collect();

        ui.add_space(10.0);
        let w = row_w;
        let k_ok = shortcuts::key_label(self.keymap.key("region.confirm"));
        let k_re = shortcuts::key_label(self.keymap.key("region.reassign"));
        let k_no = shortcuts::key_label(self.keymap.key("region.reject"));
        let k_x  = shortcuts::key_label(self.keymap.key("leaf.reject"));

        let mut act: Option<&str> = None;
        let mut confirm_family = false;

        ui.add_enabled_ui(!pending.is_empty(), |ui| {
            let label = RichText::new(format!("{fam} is correct   {k_ok}"))
                .strong().color(ui_kit::on_accent());
            if ui.add_sized([w, 34.0], egui::Button::new(label).fill(ui_kit::ACCENT()))
                .on_hover_text(format!(
                    "Accept all {} unreviewed {fam} regions shown above.\n\n\
                     Check the first rows first — with \u{201c}Unusual first\u{201d} the ones least \
                     like the rest of the family are at the top.\n\nUndoable with Ctrl+Z.",
                    pending.len()))
                .on_disabled_hover_text("Everything in this family has already been reviewed.")
                .clicked()
            {
                confirm_family = true;
            }
        });
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let half = (w - 6.0) / 2.0;
            let has_sel = !self.effective_selection().is_empty();
            ui.add_enabled_ui(has_sel, |ui| {
                if ui.add_sized([half, 28.0], egui::Button::new(format!("Reclassify   {k_re}")))
                    .on_disabled_hover_text("Select a detection first.")
                    .clicked()
                {
                    act = Some("region.reassign");
                }
                if ui.add_sized([half, 28.0], egui::Button::new(format!("Discard   {k_no}")))
                    .on_disabled_hover_text("Select a detection first.")
                    .clicked()
                {
                    act = Some("region.reject");
                }
            });
        });
        ui.add_space(4.0);
        if let Some(li) = self.selected_idx {
            let rejected = self.rejected_leaves.contains(&li);
            let (txt, fill) = if rejected {
                (format!("Restore this leaf   {k_x}"), Color32::from_rgb(90, 90, 95))
            } else {
                (format!("Reject whole leaf   {k_x}"), Color32::from_rgb(168, 72, 27))
            };
            if ui.add_sized([w, 30.0],
                egui::Button::new(RichText::new(txt).color(Color32::WHITE).strong()).fill(fill))
                .on_hover_text("Throw this leaf out of the run entirely — excluded from the CSV, \
                                the counts and from mining. Reversible, and saved to disk.")
                .clicked()
            {
                self.toggle_reject_leaf(li, toasts);
            }
        }
        if confirm_family {
            let n = pending.len();
            self.confirm_regions(&pending, toasts);
            toasts.success(format!("Confirmed {n} as {fam}"));
        }
        if let Some(a) = act {
            self.perform_action(a, toasts);
        }
    }

    /// Families as a persistent legend: swatch, dash style, name, count, and the
    /// number key that assigns it.
    ///
    /// One element doing three jobs — external memory (what the colours mean),
    /// signifier (click to focus), and the surface that teaches the number keys.
    /// Previously this lived in a table behind a tab, so the colour code was only
    /// visible if you went looking for it.
    fn show_family_legend(&mut self, ui: &mut Ui, row_w: f32) {
        ui_kit::section_header(ui, if self.filter_leaf_only {
            "Families on this leaf"
        } else {
            "Families in this run"
        });
        let ids: Vec<i32> = self.clusters.iter().map(|c| c.id).collect();
        let mut focus: Option<Option<i32>> = None;

        for (n, cid) in ids.iter().copied().enumerate() {
            // Count what the CURRENT filter shows, so the number beside a family
            // always matches what clicking it will give you.
            let count = (0..self.regions.len())
                .filter(|&i| {
                    self.region_visible(i) && self.labels[i] == cid
                        && (!self.filter_leaf_only
                            || self.selected_idx.map_or(true, |li| self.regions[i].leaf == li))
                })
                .count();
            if count == 0 { continue; }
            let selected = self.selected_cluster == Some(cid);
            let name = self.class_display_name(cid);

            // A selected row is a filled bar with an accent edge — the mockup's
            // treatment. Plain bold text did not read as "this is the one", and
            // the row did not read as clickable at all.
            let row_h = 24.0;
            let (rect, resp) = ui.allocate_exact_size(
                egui::vec2(row_w, row_h), egui::Sense::click(),
            );
            let hovered = resp.hovered();
            if selected || hovered {
                let fill = if selected {
                    ui_kit::ACCENT().linear_multiply(0.16)
                } else {
                    ui.visuals().faint_bg_color
                };
                ui.painter().rect_filled(rect, 3.0, fill);
            }
            if selected {
                ui.painter().rect_filled(
                    egui::Rect::from_min_size(rect.min, egui::vec2(3.0, row_h)),
                    1.0, ui_kit::ACCENT(),
                );
            }
            let mut x = rect.left() + 10.0;
            let mid = rect.center().y;
            let dim = ui_kit::MUTED();
            if n < 9 {
                ui.painter().text(
                    egui::pos2(x, mid), egui::Align2::LEFT_CENTER, format!("{}", n + 1),
                    egui::FontId::monospace(11.0), dim,
                );
            }
            x += 16.0;
            let c = cluster_color(cid);
            let col = Color32::from_rgb(c[0], c[1], c[2]);
            let sw = egui::Rect::from_min_size(egui::pos2(x, mid - 5.0), egui::vec2(16.0, 10.0));
            match cluster_dash(cid) {
                None => {
                    ui.painter().rect_filled(sw, 2.0, col);
                }
                Some(d) => {
                    ui.painter().rect_filled(sw, 2.0, col.linear_multiply(0.3));
                    let dd = (d * 0.9).clamp(2.0, 5.0);
                    ui.painter().add(egui::Shape::dashed_line(
                        &[egui::pos2(sw.left() + 1.0, mid), egui::pos2(sw.right() - 1.0, mid)],
                        egui::Stroke::new(2.0, col), dd, dd * 0.7,
                    ));
                }
            }
            x += 24.0;
            ui.painter().text(
                egui::pos2(x, mid), egui::Align2::LEFT_CENTER, &name,
                egui::FontId::proportional(13.5),
                if selected { ui.visuals().strong_text_color() } else { ui.visuals().text_color() },
            );
            ui.painter().text(
                egui::pos2(rect.right() - 10.0, mid), egui::Align2::RIGHT_CENTER,
                fmt_thousands(count),
                egui::FontId::monospace(12.0), dim,
            );

            // Clicking the SWATCH recolours; clicking anywhere else focuses.
            // The swatch is the thing that represents the colour, so it is where
            // a user reaches to change it.
            let on_swatch = resp.hover_pos().map_or(false, |p| sw.expand(3.0).contains(p));
            if resp.clicked() {
                if on_swatch {
                    self.recolour_family = Some(cid);
                } else {
                    focus = Some(if selected { None } else { Some(cid) });
                }
            }
            resp.on_hover_text(format!(
                "Show only {name}.\nClick the colour patch to change it.{}",
                if n < 9 { format!("\nKey {} assigns the selection to it.", n + 1) } else { String::new() }
            ));
        }

        if let Some(f) = focus {
            self.selected_cluster = f;
            self.selected_region = None;
            self.gallery_page = 0;
            self.overlay_tex = None;
        }

        // Recolour popup: the eight colour-blind-safe hues plus a reset.
        // A fixed palette rather than a free colour wheel on purpose — the whole
        // point of Okabe-Ito is that the set stays distinguishable, and letting
        // someone pick two near-identical greens would undo that.
        if let Some(cid) = self.recolour_family {
            let mut open = true;
            egui::Window::new(format!("Colour for {}", self.class_display_name(cid)))
                .collapsible(false).resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .open(&mut open)
                .show(ui.ctx(), |ui| {
                    ui.horizontal_wrapped(|ui| {
                        for c in CLUSTER_PALETTE {
                            let (rect, r) = ui.allocate_exact_size(
                                egui::vec2(30.0, 24.0), egui::Sense::click());
                            ui.painter().rect_filled(rect, 3.0, Color32::from_rgb(c[0], c[1], c[2]));
                            if cluster_color(cid) == c {
                                ui.painter().rect_stroke(rect.expand(1.0), 3.0,
                                    egui::Stroke::new(2.0, Color32::WHITE));
                            }
                            if r.clicked() {
                                set_family_color(cid, Some(c));
                                self.overlay_tex = None;
                            }
                        }
                    });
                    ui.add_space(6.0);
                    if ui.button("Back to the default").clicked() {
                        set_family_color(cid, None);
                        self.overlay_tex = None;
                    }
                });
            if !open {
                self.recolour_family = None;
            }
        }
    }

    /// Selected leaf + its morphology metrics — was always-visible above the
    /// other tabs, now its own tab per feedback.
    /// Selected leaf's morphology metrics. The reconstruction-preview toggle
    /// lives in the toolbox's "View" section now (grouped with zoom), not
    /// here — it's about how the canvas is displayed, not this leaf's stats.
    fn show_leaf_tab(&mut self, ui: &mut Ui) {
        self.show_leaf_morphology(ui);
    }

    fn show_clusters_tab(&mut self, ui: &mut Ui, toasts: &mut ToastManager) {
        ui.label(RichText::new("Defect clusters").strong());
        if self.regions.is_empty() {
            let msg = if self.running {
                "Detecting leaves… clusters are built once the whole dataset is processed."
            } else {
                "Run the pipeline to detect + cluster anomalies."
            };
            ui.label(RichText::new(msg).small().color(Color32::GRAY));
            return;
        }

        // Instant re-cut (Hierarchical only): replay the already-computed merge
        // tree at a different K or sensitivity threshold in O(n) — no pipeline
        // rerun, no O(n³) recompute. `build_clusters` needs `&mut self` as a
        // whole, which can't happen while `state` (borrowed from
        // `self.hcluster`) is still alive — so the slider interaction and the
        // resulting label/rebuild are kept in separate steps rather than one
        // closure, so `state`'s borrow ends before build_clusters is called.
        let mut recut_changed = false;
        if let Some(state) = &self.hcluster {
            let max_k = state.real_idx.len().max(1).min(100);
            let resp = ui.horizontal(|ui| {
                ui.label("Re-cut:")
                    .on_hover_text("Instantly re-partition — no pipeline rerun needed, this\n\
                                    replays the existing merge tree from this run in O(n).");
                egui::ComboBox::from_id_salt("recut_mode")
                    .selected_text(self.recut_mode.label())
                    .show_ui(ui, |ui| {
                        for &mode in CutMode::ALL {
                            ui.selectable_value(&mut self.recut_mode, mode, mode.label());
                        }
                    });
                match self.recut_mode {
                    CutMode::FixedK => ui.add(egui::Slider::new(&mut self.recut_k, 1..=max_k)),
                    CutMode::Adaptive => ui.add(egui::Slider::new(&mut self.recut_threshold, 0.5..=30.0).fixed_decimals(1)),
                }
            }).inner;
            if resp.changed() {
                let sub_labels = match self.recut_mode {
                    CutMode::FixedK => cluster::labels_for_k(
                        &state.merges, state.real_idx.len(), self.recut_k, state.min_cluster_size,
                    ),
                    CutMode::Adaptive => cluster::labels_adaptive(
                        &state.merges, state.real_idx.len(), self.recut_threshold, state.min_cluster_size,
                    ),
                };
                for (pos, &region_i) in state.real_idx.iter().enumerate() {
                    self.labels[region_i] = sub_labels[pos];
                }
                recut_changed = true;
            }
            ui.separator();
        }
        if recut_changed {
            self.build_clusters(toasts);
        }

        // ── PCA scatter (click → nearest point's cluster) ──
        let sel = self.focus_cluster();
        let plot = Plot::new("cluster_scatter").height(200.0).show(ui, |plot_ui| {
            for c in &self.clusters {
                let focused = sel.map_or(true, |cid| c.id == cid);
                let col = if focused { cluster_color(c.id) } else { dim_color(cluster_color(c.id)) };
                let pts: Vec<[f64; 2]> = c
                    .members
                    .iter()
                    .filter(|&&i| self.region_visible(i))
                    .map(|&i| [self.coords[i][0] as f64, self.coords[i][1] as f64])
                    .collect();
                let radius = if focused { 4.0 } else { 2.5 };
                plot_ui.points(
                    Points::new(pts).radius(radius).color(Color32::from_rgb(col[0], col[1], col[2])),
                );
            }
            plot_ui.pointer_coordinate()
        });
        if plot.response.clicked() {
            if let Some(coord) = plot.inner {
                let mut best = None;
                let mut bd = f64::INFINITY;
                for i in 0..self.coords.len() {
                    if !self.region_visible(i) {
                        continue;
                    }
                    let dx = self.coords[i][0] as f64 - coord.x;
                    let dy = self.coords[i][1] as f64 - coord.y;
                    let d = dx * dx + dy * dy;
                    if d < bd {
                        bd = d;
                        best = Some(i);
                    }
                }
                if let Some(i) = best {
                    self.selected_cluster = Some(self.labels[i]);
                    self.overlay_tex = None;
                }
            }
        }
        if self.selected_cluster.is_some() && ui.small_button("Clear selection").clicked() {
            self.selected_cluster = None;
            self.selected_region = None;
            self.overlay_tex = None;
        }
        ui.separator();

        // ── stats table ──
        let cur_leaf = self.selected_idx;
        let cur = cur_leaf.and_then(|i| self.results.get(i));
        let leaf_valid_px: f32 = cur_leaf
            .and_then(|i| self.leaf_valid_px.get(i))
            .copied()
            .unwrap_or(1)
            .max(1) as f32;
        let recon_area: f32 = cur.map(|l| l.recon_area as f32).unwrap_or(0.0);   // added = lost tissue
        let recon_whole: f32 = cur.map(|l| l.recon_whole as f32).unwrap_or(0.0); // whole intact leaf
        if recon_whole > 0.0 {
            ui.label(
                RichText::new(format!(
                    "Lost tissue: {:.1}%  ({} px reconstructed)",
                    100.0 * recon_area / recon_whole, recon_area as u64
                ))
                .small()
                .color(ui_kit::ACCENT()),
            )
            .on_hover_text("Leaf area the model reconstructed (damaged/missing tissue) as a \
                            fraction of the whole intact leaf.");
        }
        egui::ScrollArea::vertical().max_height(150.0).id_salt("cluster_table").show(ui, |ui| {
            TableBuilder::new(ui)
                .striped(true)
                .column(Column::exact(24.0))
                .column(Column::remainder().at_least(160.0))
                .column(Column::auto())
                .column(Column::auto())
                .column(Column::auto())
                .column(Column::exact(28.0))
                .header(20.0, |mut header| {
                    header.col(|_ui| {});
                    header.col(|ui| { ui.label("Cluster"); });
                    header.col(|ui| { ui.label("% leaf"); });
                    header.col(|ui| { ui.label("total px"); });
                    header.col(|ui| {
                        ui.label("Recon %")
                            .on_hover_text("This cluster's damaged area as a fraction of the RECONSTRUCTED \
                                            intact leaf (damage relative to the whole undamaged leaf). \
                                            Needs a recon checkpoint — bundled at models/recon/gen.mpk.");
                    });
                    header.col(|_ui| {});
                })
                .body(|mut body| {
                    for ci in 0..self.clusters.len() {
                        let cid = self.clusters[ci].id;
                        // colour + dash now come from family_swatch
                        let (mut total, mut leaf_px) = (0u64, 0u64);
                        for k in 0..self.clusters[ci].members.len() {
                            let ri = self.clusters[ci].members[k];
                            if !self.region_visible(ri) {
                                continue;
                            }
                            let area = self.region_area[ri] as u64;
                            total += area;
                            if Some(self.regions[ri].leaf) == cur_leaf {
                                leaf_px += area;
                            }
                        }
                        body.row(22.0, |mut row| {
                            row.col(|ui| {
                                if ui.small_button("☑")
                                    .on_hover_text("Select every region in this cluster (all leaves), \
                                                    ready for bulk reassign/remove.")
                                    .clicked()
                                {
                                    self.multi_selected = self.clusters[ci].members.iter()
                                        .copied()
                                        .filter(|&ri| self.region_visible(ri))
                                        .collect();
                                    self.selected_region = None;
                                }
                            });
                            row.col(|ui| {
                                if ci < 9 {
                                    ui.label(RichText::new(format!("{}", ci + 1))
                                        .text_style(ui_kit::numeric()).color(ui_kit::MUTED()))
                                        .on_hover_text("Press this number to assign the selected detections to this family.");
                                }
                                family_swatch(ui, cid, 10.0);
                                let mut name = self.cluster_names.get(&cid).cloned()
                                    .unwrap_or_else(|| format!("Cluster {cid}"));
                                // COMMIT on blur/Enter, not on every keystroke.
                                // `.changed()` fires per character, so typing an
                                // 8-letter name performed 8 full head-file
                                // rewrites — and each one overwrote the .json.bak,
                                // so the "backup before either edit" the tooltip
                                // promises ended up holding the state after the
                                // second-to-last keystroke rather than the
                                // original. It also re-persisted every member of
                                // the cluster 8 times over.
                                let te = ui.add(
                                    egui::TextEdit::singleline(&mut name).desired_width(190.0),
                                );
                                // Keep the in-memory label live so the field does
                                // not fight the typist; only the DISK writes wait.
                                if te.changed() {
                                    self.cluster_names.insert(cid, name.clone());
                                }
                                // lost_focus() covers Enter, Tab and clicking away
                                // — every way a person signals "I'm done typing".
                                if te.lost_focus() {
                                    self.cluster_names.insert(cid, name.clone());
                                    // propagate the rename to disk: any member already
                                    // persisted needs its family string re-written so it
                                    // doesn't silently diverge from what's now shown.
                                    let members: Vec<usize> = (0..self.regions.len())
                                        .filter(|&i| self.labels[i] == cid && self.persisted.contains(&i))
                                        .collect();
                                    for i in members {
                                        self.persist_region(i, &name, false, toasts);
                                    }
                                    // AND propagate into the head file itself, if `cid` is a
                                    // real head class — otherwise renames never stick and
                                    // "Cluster N" keeps coming back every time the head is
                                    // reloaded/retrained. Ephemeral unsupervised-cluster ids
                                    // (not a real head class) are skipped on purpose — those
                                    // aren't stable across runs, nothing to rename there.
                                    self.rename_head_class_if_real(cid, &name, toasts);
                                }
                            });
                            row.col(|ui| { ui.label(format!("{:.1}%", 100.0 * leaf_px as f32 / leaf_valid_px)); });
                            row.col(|ui| { ui.label(format!("{total}")); });
                            row.col(|ui| {
                                if recon_whole > 0.0 {
                                    ui.label(format!("{:.1}%", 100.0 * leaf_px as f32 / recon_whole));
                                } else {
                                    ui.label(RichText::new("-").color(Color32::DARK_GRAY));
                                }
                            });
                            row.col(|ui| {
                                let mut target = None;
                                let mut delete = false;
                                ui.menu_button("⇒", |ui| {
                                    ui.label(RichText::new("Merge into…").small().color(Color32::GRAY));
                                    if let Some(into_id) = self.cluster_picker_rows(ui, "") {
                                        if into_id != cid {
                                            target = Some(into_id);
                                            ui.close_menu();
                                        }
                                    }
                                    ui.separator();
                                    if ui.button(RichText::new("🗑 Delete entirely").color(Color32::from_rgb(230, 90, 90))).clicked() {
                                        delete = true;
                                        ui.close_menu();
                                    }
                                }).response.on_hover_text(
                                    "Merge this class into another — fixes an accidental \
                                     duplicate (e.g. \"hole\" typed when \"Hole\" already \
                                     existed). Or delete it entirely (kick a meaningless class \
                                     out of the head — e.g. a weird-discoloration class you'd \
                                     rather leave to the PatchCore safety net). A backup of the \
                                     head file (.json.bak) is written before either edit. \
                                     Deleting doesn't guarantee those patches fall through to \
                                     PatchCore — only patches that don't clearly match any \
                                     REMAINING class will."
                                );
                                if let Some(into_id) = target {
                                    self.merge_cluster_names(cid, into_id, toasts);
                                }
                                // Confirm, because this is the one action in the
                                // tab that is both irreversible and reachable by
                                // two clicks in a 28px-wide menu column: it
                                // rewrites EVERY curated row of the family to
                                // "rejected" on disk and removes the class from
                                // the head. The .json.bak is best-effort, so there
                                // may be nothing to go back to.
                                //
                                // Routine curation actions deliberately have NO
                                // dialog — a reviewer pressing a key 20,000 times
                                // learns to dismiss them, which removes the
                                // protection exactly where it matters. Rare and
                                // unrecoverable is the only case that earns one.
                                if delete {
                                    self.pending_delete_cluster = Some(cid);
                                }
                            });
                        });
                    }
                });
        });
        ui.label(RichText::new(
            "Not sure what a cluster actually is? See its example crops in the \
             Curate tab's gallery (grouped by cluster), then rename it above.")
            .small().color(Color32::GRAY));
    }

    fn show_curate_tab(&mut self, ui: &mut Ui, ctx: &Context, toasts: &mut ToastManager) {
        if self.regions.is_empty() {
            let msg = if self.running {
                "Detecting leaves… review becomes available once regions are found."
            } else {
                "Run the pipeline to detect anomalies, then review them here."
            };
            ui.label(RichText::new(msg).small().color(Color32::GRAY));
            return;
        }

        // The cluster combo, the "This leaf only" checkbox, the per-leaf pending
        // line, the dataset-wide counts and the bulk-confirm button all used to
        // sit HERE, above the grid — five controls and two status lines between
        // the family list and the pictures the reviewer came to look at.
        //
        // The family legend now does the filtering (click a family), the status
        // bar carries the counts, and bulk confirm moved next to the other
        // verdicts at the bottom where the decisions live. "This leaf only" moved
        // to the section header, since that is the one place its meaning is
        // obvious.

        // ── in-place retrain: fine-tune the head from this run's curations
        // without leaving Pipeline ──
        if let Some(new_head) = self.retrain_done.clone() {
            egui::Frame::none()
                .fill(Color32::from_rgb(40, 60, 45))
                .inner_margin(egui::Margin::same(8.0))
                .rounding(egui::Rounding::same(4.0))
                .show(ui, |ui| {
                    ui.label(RichText::new("✅ New head trained").strong().color(Color32::from_rgb(140, 230, 150)));
                    ui.label(RichText::new(new_head.display().to_string()).small().color(Color32::GRAY));
                    ui.horizontal(|ui| {
                        if ui.button("Use this head now").clicked() {
                            self.head_path = Some(new_head.clone());
                            self.retrain_done = None;
                            let (rev, rej, _) = self.review_counts();
                            if !self.results.is_empty() && (rev + rej > 0) {
                                self.pending_reset = Some(PendingReset::SwitchHead);
                            } else {
                                self.reset_run_state();
                                toasts.info("Switched to the retrained head — click Run Pipeline to see corrected results.");
                            }
                        }
                        if ui.button("Dismiss").clicked() {
                            self.retrain_done = None;
                        }
                    });
                });
            ui.add_space(4.0);
        }
        // ── everything below here is model-engineering, folded away by default ──
        //
        // These ~20 controls (mining, base set, base rows, anchor, cold start,
        // diagnostics dump, export) used to sit ABOVE the review gallery, so the
        // surface used ten thousand times a batch was reached by scrolling past
        // machinery most users never touch. Collapsed, they cost one line each and
        // the gallery is immediately visible; nothing is removed, and anyone who
        // wants them opens the section once and egui remembers it.

        // ── anomaly gallery (filtered to the selected cluster, paginated) ──
        const PER_PAGE: usize = GALLERY_PER_PAGE;
        let mut filtered: Vec<usize> = (0..self.regions.len())
            .filter(|&i| {
                self.region_visible(i)
                    && self.selected_cluster.map_or(true, |c| self.labels[i] == c)
                    && (!self.filter_leaf_only || self.selected_idx.map_or(true, |li| self.regions[i].leaf == li))
                    && (!self.filter_flagged || self.flagged.contains(&i))
            })
            .collect();
        // ── ordering ────────────────────────────────────────────────────────
        // With 6,000+ regions on a single leaf, judging them one by one is not a
        // workflow — 6,000 x ~1.5s is over two hours for ONE leaf. The unit of
        // review has to be the family, and the only way that is safe is if the
        // few members that DON'T belong surface first. So: sort by how unlike its
        // own family each region is, worst first, and the reviewer checks the
        // head of the list instead of all of it.
        match self.gallery_sort {
            GallerySort::Largest => {
                filtered.sort_by_key(|&i| std::cmp::Reverse(self.region_area.get(i).copied().unwrap_or(0)));
            }
            GallerySort::Unusual => {
                let score = self.atypicality();
                filtered.sort_by(|&a, &b| {
                    score.get(&b).unwrap_or(&0.0)
                        .partial_cmp(score.get(&a).unwrap_or(&0.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
        let total = filtered.len();
        let n_pages = total.div_ceil(PER_PAGE).max(1);
        if self.gallery_page >= n_pages {
            self.gallery_page = 0;
        }
        // ── family-level review ─────────────────────────────────────────────
        // The action that makes thousands of regions tractable: judge the whole
        // family at once, having checked the head of an unusual-first list, then
        // spend the remaining attention on the few that stood out.
        // Family header line: swatch + name + how much of it is still unreviewed.
        if let Some(cid) = self.selected_cluster {
            let fam = self.class_display_name(cid);
            let pending = filtered.iter().filter(|i| !self.persisted.contains(i)).count();
            ui.horizontal(|ui| {
                family_swatch(ui, cid, 11.0);
                ui.label(RichText::new(&fam).strong());
                ui.label(RichText::new(format!("{pending} unreviewed"))
                    .small().color(ui_kit::MUTED()));
            });
        }

        // ── section header, mockup style ────────────────────────────────────
        // "NEKROSIS — 341 REGIONS · 4 LOOK UNUSUAL" says what you are looking at
        // and where the risk is, in one line. The old header was a bare count
        // plus five controls.
        {
            let unusual = if self.regions.iter().any(|r| !r.dino_embed.is_empty()) {
                let score = self.atypicality();
                filtered.iter().filter(|i| score.get(i).copied().unwrap_or(0.0) > 0.35).count()
            } else { 0 };
            let title = match self.selected_cluster {
                Some(cid) => format!("{} — {} regions", self.class_display_name(cid).to_uppercase(),
                                     fmt_thousands(total)),
                None => format!("ALL FAMILIES — {} regions", fmt_thousands(total)),
            };
            ui.horizontal(|ui| {
                ui.label(RichText::new(title).small().strong().color(ui_kit::MUTED()));
                if unusual > 0 {
                    ui.label(RichText::new(format!("· {unusual} look unusual"))
                        .small().strong().color(Color32::from_rgb(225, 175, 85)));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_enabled_ui(self.selected_idx.is_some(), |ui| {
                        ui.checkbox(&mut self.filter_leaf_only, "This leaf only")
                            .on_hover_text("Limit everything above to the leaf currently open.");
                    });
                });
            });
        }
        // Replaces a line of mouse-gesture documentation that belongs in the
        // shortcuts window, not above the pictures.
        {
            let shown = (self.gallery_page + 1) * PER_PAGE;
            let more = total.saturating_sub(shown.min(total));
            let how = match self.gallery_sort {
                GallerySort::Unusual if self.regions.iter().any(|r| !r.dino_embed.is_empty()) =>
                    "sorted least-typical first",
                GallerySort::Unusual => "sorted by unusual size",
                GallerySort::Largest => "sorted largest first",
            };
            ui.label(RichText::new(if more > 0 {
                format!("{how} · {} more", fmt_thousands(more))
            } else {
                how.to_string()
            }).small().color(ui_kit::MUTED()));
        }
        let curate_sel = self.effective_selection();
        if !curate_sel.is_empty() {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{} selected", curate_sel.len())).small());
                if ui.small_button("✅ Confirm").clicked() {
                    self.confirm_regions(&curate_sel, toasts);
                    self.multi_selected.clear();
                    self.selected_region = None;
                }
                if ui.small_button("Reassign").clicked() {
                    self.reassign_selected(toasts);
                }
                if ui.small_button("Clear").clicked() {
                    self.multi_selected.clear();
                    self.selected_region = None;
                }
            });
            // On its own row, filling the panel's full width — this field is
            // typed into far more than the buttons above are clicked, so it
            // gets the room (was a cramped fixed 160px squeezed in with the
            // buttons).
            ui.add_sized([ui.available_width(), 0.0], egui::TextEdit::singleline(&mut self.reassign_name)
                .hint_text("cluster name"));
        }
        let show_idxs: Vec<usize> =
            filtered.iter().copied().skip(self.gallery_page * PER_PAGE).take(PER_PAGE).collect();
        for &i in &show_idxs {
            self.ensure_region_thumb(ctx, i);
        }
        // ── ONE uniform grid ────────────────────────────────────────────────
        // It used to start a new headed group each time the label changed while
        // walking the page. That was fine when the sort was BY cluster, but the
        // sort is now by atypicality, so families interleave and you got
        // "Nekrosis (1)", "Sucker (9)", "Nekrosis (2)", "Sucker (3)"... the same
        // family announced over and over down the column. Which family a tile
        // belongs to is already carried by the coloured stripe on the tile.
        egui::ScrollArea::vertical()
            .id_salt("anomaly_gallery")
            .max_height(360.0)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(6.0, 6.0);
                    for &i in &show_idxs {
                        let Some(tex) = &self.region_thumbs[i] else { continue };
                        let sel = self.selected_region == Some(i);
                        let resp = ui
                            .add(egui::ImageButton::new((tex.id(), egui::vec2(54.0, 54.0))))
                            .on_hover_text(format!(
                                "{} - leaf {} - {} px",
                                self.class_display_name(self.labels[i]),
                                self.regions[i].leaf + 1,
                                self.region_area.get(i).copied().unwrap_or(0),
                            ));
                        let r = resp.rect;
                        // Family stripe along the bottom edge: identity without a
                        // heading, and it survives a mixed-family grid.
                        let c = cluster_color(self.labels[i]);
                        ui.painter().rect_filled(
                            egui::Rect::from_min_max(
                                egui::pos2(r.left(), r.bottom() - 3.0), r.right_bottom()),
                            0.0, Color32::from_rgb(c[0], c[1], c[2]),
                        );
                        if self.persisted.contains(&i) {
                            ui.painter().rect_filled(
                                r, 3.0, Color32::from_rgba_unmultiplied(10, 25, 14, 90));
                            // An explicit tick, not just the dim above. A 90-alpha
                            // tint is easy to miss across a grid of small tiles,
                            // and reviewers lost track of what they had already
                            // checked — "vielleicht wäre eine Art Haken auf der
                            // Darstellung noch besser sichtbar".
                            //
                            // Painted by hand rather than drawn as U+2713: the
                            // bundled fonts do not cover it and it renders as
                            // tofu. Same two-line_segment trick the stage pills
                            // already use.
                            let cc = egui::pos2(r.left() + 10.0, r.top() + 10.0);
                            ui.painter().circle_filled(cc, 7.5, Color32::from_rgb(56, 152, 86));
                            let tick = egui::Stroke::new(2.0, Color32::WHITE);
                            ui.painter().line_segment(
                                [cc + egui::vec2(-3.5, 0.2), cc + egui::vec2(-1.0, 3.0)], tick);
                            ui.painter().line_segment(
                                [cc + egui::vec2(-1.0, 3.0), cc + egui::vec2(3.6, -3.2)], tick);
                        }
                        if self.flagged.contains(&i) {
                            ui.painter().rect_stroke(r, 3.0,
                                egui::Stroke::new(2.0, Color32::from_rgb(225, 180, 90)));
                        }
                        if sel {
                            ui.painter().rect_stroke(r.expand(1.0), 3.0,
                                egui::Stroke::new(2.0, ui_kit::ACCENT()));
                            if self.scroll_to_selected {
                                resp.scroll_to_me(Some(egui::Align::Center));
                                self.scroll_to_selected = false;
                            }
                        }
                        if self.multi_selected.contains(&i) {
                            ui.painter().rect_stroke(r, 3.0,
                                egui::Stroke::new(2.0, Color32::from_rgb(80, 170, 255)));
                        }
                        if resp.clicked() {
                            if ui.input(|inp| inp.modifiers.shift) {
                                let range = self.last_clicked_region.and_then(|a| {
                                    let x = show_idxs.iter().position(|&v| v == a)?;
                                    let y = show_idxs.iter().position(|&v| v == i)?;
                                    Some((x.min(y), x.max(y)))
                                });
                                if let Some((lo, hi)) = range {
                                    for &ri in &show_idxs[lo..=hi] { self.multi_selected.insert(ri); }
                                } else {
                                    self.multi_selected.insert(i);
                                }
                            } else if ui.input(|inp| inp.modifiers.ctrl) {
                                self.toggle_multi_select(i);
                            } else {
                                self.selected_idx = Some(self.regions[i].leaf);
                                self.selected_region = Some(i);
                                // Bring it into view if the canvas is zoomed in.
                                self.center_on_region = Some(i);
                                self.overlay_tex = None;
                            }
                            self.last_clicked_region = Some(i);
                        }
                        // Right-click opens a MENU; it does not delete.
                        //
                        // It used to call remove_regions directly, so anyone
                        // right-clicking a tile to see what options existed
                        // destroyed the region instead — reported as "da kann
                        // man sich schnell vertun". A destructive action must be
                        // something you choose, never something you discover.
                        resp.context_menu(|ui| {
                            ui.label(
                                RichText::new(format!("Region {}", i + 1))
                                    .small()
                                    .color(Color32::GRAY),
                            );
                            ui.separator();
                            if ui.button("Select").clicked() {
                                self.selected_idx = Some(self.regions[i].leaf);
                                self.selected_region = Some(i);
                                self.center_on_region = Some(i);
                                self.overlay_tex = None;
                                self.last_clicked_region = Some(i);
                                ui.close_menu();
                            }
                            if ui.button("Remove region").clicked() {
                                self.remove_regions(&[i], toasts, true);
                                self.multi_selected.remove(&i);
                                if self.selected_region == Some(i) {
                                    self.selected_region = None;
                                }
                                ui.close_menu();
                            }
                        });
                    }
                });
            });
        // Ordering, paging and appearance-ranking sit BELOW the grid — they
        // configure how it is presented, so they read as its footer rather than
        // as a toolbar you pass through on the way in.
        self.show_gallery_controls(ui, filtered.len(), n_pages, toasts);
    }

    /// The grid's footer: what order, which page, undo, and the ranking action.
    fn show_gallery_controls(
        &mut self, ui: &mut Ui, total: usize, n_pages: usize, toasts: &mut ToastManager,
    ) {
        ui.add_space(6.0);
        // FIRST row: what is being shown, and by what signal.
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 8.0;
            egui::ComboBox::from_id_salt("gallery_sort")
                .selected_text(self.gallery_sort.label())
                .width(150.0)
                .show_ui(ui, |ui| {
                    for s in GallerySort::ALL {
                        if ui.selectable_label(self.gallery_sort == s, s.label()).clicked() {
                            self.gallery_sort = s;
                            self.gallery_page = 0;
                        }
                    }
                });
            // Say which signal is actually in force — appearance ranking only
            // exists once embeddings have been computed for these regions.
            if self.gallery_sort == GallerySort::Unusual {
                let exact = self.regions.iter().any(|r| !r.dino_embed.is_empty());
                ui.label(RichText::new(if exact { "by appearance" } else { "by size (approx.)" })
                    .small().color(ui_kit::MUTED()))
                    .on_hover_text(if exact {
                        "Ranked by how far each region's DINO embedding sits from its family's \
                         centre — a real measure of \u{201c}this does not look like its siblings\u{201d}."
                    } else {
                        "No per-region embeddings for these yet — ranking falls back to how far \
                         each region's size is from its family's median. Press \u{201c}Rank by \
                         appearance\u{201d} for the real thing."
                    });
            }
        });

        // SECOND row. These used to share one wrapped row with the sort combo,
        // which in a 300px panel broke as "combo, qualifier, ‹" then
        // "1/3  ›  Undo" — the page arrows split across two lines with the page
        // number orphaned from them. Splitting the rows by MEANING (what is
        // shown / how to move through it) means the wrap can no longer cut
        // through the middle of a control group.
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 8.0;
            let n_flag = self.flagged.len();
            if n_flag > 0 || self.filter_flagged {
                if ui.selectable_label(self.filter_flagged, format!("Set aside ({n_flag})"))
                    .on_hover_text("Show only the detections you pressed F on — a second pass \
                                    for the hard calls, made when you are fresh.")
                    .clicked()
                {
                    self.filter_flagged = !self.filter_flagged;
                    self.gallery_page = 0;
                }
            }
            if n_pages > 1 {
                // Kept atomic in their own row: the three parts are one control
                // and are meaningless apart. Narrow enough (~70px) that this
                // non-wrapping row cannot push the panel wider.
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    if ui.small_button("‹").clicked() && self.gallery_page > 0 {
                        self.gallery_page -= 1;
                    }
                    ui.label(RichText::new(format!("{}/{}", self.gallery_page + 1, n_pages))
                        .small().color(ui_kit::MUTED()));
                    if ui.small_button("›").clicked() && self.gallery_page + 1 < n_pages {
                        self.gallery_page += 1;
                    }
                });
            }
            ui.add_enabled_ui(!self.struct_undo.is_empty(), |ui| {
                if ui.small_button(format!("Undo ({})", self.struct_undo.len()))
                    .on_hover_text("Reverse the last edit, including a paint or eraser stroke. Ctrl+Z.")
                    .clicked()
                {
                    self.undo_last_edit(toasts);
                }
            });
            // Sits next to Undo, always visible and merely disabled when empty,
            // so it is discoverable BEFORE you need it — the reviewer who wanted
            // this had already undone one step too far by the time they looked.
            ui.add_enabled_ui(!self.struct_redo.is_empty(), |ui| {
                if ui.small_button(format!("Redo ({})", self.struct_redo.len()))
                    .on_hover_text("Put back what Undo just reversed. Ctrl+Y.")
                    .clicked()
                {
                    self.redo_last_edit(toasts);
                }
            });
        });

        // Appearance ranking — the action that makes confirming a whole family
        // defensible. Cost stated up front from the measured ~45 ms/region.
        let to_rank = self.rank_targets().len();
        if self.ranking {
            ui.horizontal(|ui| {
                ui_kit::busy(ui, &format!("ranking {}/{}", self.rank_done, self.rank_total));
                if ui.small_button("Cancel").clicked() {
                    self.rank_cancel.store(true, Ordering::Relaxed);
                }
            });
            let frac = self.rank_done as f32 / self.rank_total.max(1) as f32;
            ui.add(egui::ProgressBar::new(frac).desired_height(4.0));
        } else if to_rank > 0 && self.selected_cluster.is_some() {
            let busy = self.running || self.retraining || self.mining;
            ui.add_enabled_ui(!busy && self.eff_dino().is_some(), |ui| {
                let secs = (to_rank as f64 * 0.045).round().max(1.0) as u64;
                if ui.button(format!("Rank {to_rank} by appearance  (~{secs}s)"))
                    .on_hover_text(
                        "Run DINO over each region in this family and re-order the grid so the \
                         ones LEAST like the rest come first.\n\n\
                         Cancellable; partial results are kept.")
                    .on_disabled_hover_text("Needs the DINO model, and no other GPU job running.")
                    .clicked()
                {
                    self.start_rank_appearance(toasts);
                }
            });
        }
        let _ = total;
    }

    /// Pipeline run log — moved here from the bottom of the folders panel
    /// per feedback ("we can move the log somewhere else").
    fn show_log_tab(&self, ui: &mut Ui) {
        if self.log.is_empty() {
            ui.label(RichText::new("No log entries yet — run the pipeline to see output here.")
                .small().color(Color32::GRAY));
            return;
        }
        // max_height, because this now renders inside the review panel's own
        // scroll area — an unbounded nested scroller swallows the outer one's
        // wheel events and the panel stops scrolling wherever the cursor is.
        egui::ScrollArea::vertical().max_height(160.0).id_salt("pipeline_run_log").show(ui, |ui| {
            for line in self.log.iter().rev().take(500) {
                ui.label(RichText::new(line).small());
            }
        });
    }

    // ── actions / polling ─────────────────────────────────────────────────

    /// Reset all per-run leaf/region/curation state. Shared by a fresh Run
    /// (`start`) and by "Use this head now" after an in-place retrain, so
    /// neither leaves stale, still-interactive gallery/canvas state behind —
    /// the anomaly gallery is driven directly off `regions.len()`, independent
    /// of `clusters`, so a partial reset would leave old regions fully
    /// clickable/confirmable against a leaf/head that's no longer loaded.
    fn reset_run_state(&mut self) {
        self.results.clear();
        self.thumbs.clear();
        self.leaf_valid_px.clear();
        self.selected_idx = None;
        self.overlay_tex = None;
        self.regions.clear();
        self.region_area.clear();
        self.labels.clear();
        self.coords.clear();
        self.clusters.clear();
        self.region_thumbs.clear();
        self.removed.clear();
        self.struct_undo.clear();
        self.persisted.clear();
        self.merged_away.clear();
        self.rejected_leaves.clear();
        self.flagged.clear();
        self.filter_flagged = false;
        // NOT review_marks: those are the on-disk history, reloaded per run and
        // re-applied as leaves arrive. Only the live per-index sets reset.
        self.reviewed.clear();
        self.review_mismatch = 0;
        self.cluster_names.clear();
        self.selected_cluster = None;
        self.selected_region = None;
        self.gallery_page = 0;
        self.hcluster = None;
        self.recut_k = 0;
        self.recut_mode = CutMode::FixedK;
        self.recut_threshold = self.adaptive_threshold;
    }

    /// A region counts as visible unless it's been rejected by the user, absorbed
    /// into another region by `merge_touching_regions`, or sits on a leaf the user
    /// threw out entirely — everything that iterates/renders/counts regions should
    /// gate on this, not on `removed` alone, now that merges and whole-leaf
    /// rejection can also hide an index.
    fn region_visible(&self, i: usize) -> bool {
        if self.removed.contains(&i) || self.merged_away.contains(&i) {
            return false;
        }
        // An out-of-range index stays visible rather than silently vanishing:
        // callers that hand this a bad index have a bug worth surfacing where it
        // happens, and this gate is not the place to swallow it.
        self.regions.get(i).map_or(true, |r| !self.rejected_leaves.contains(&r.leaf))
    }

    /// Can this action do anything right now? Drives both the palette's greying
    /// and whether a keypress should be swallowed.
    fn action_enabled(&self, id: &str) -> bool {
        match id {
            "run.start"  => self.all_paths_ok() && self.source_count > 0
                && !self.running && !self.retraining && !self.mining
                && self.output_inside_source().is_none(),
            "run.cancel" => self.running,
            "review.export" => !self.regions.is_empty() && self.output_folder.is_some(),
            // Only a navigation change, so it needs results to go and land on —
            // but deliberately NOT a completed review. Leaving early is allowed.
            "flow.finish" => !self.results.is_empty() && !self.running
                && self.stage_view != StageView::Done,
            "review.confirm_family" => self.selected_cluster.is_some(),
            "review.undo" => !self.struct_undo.is_empty(),
            "review.redo" => !self.struct_redo.is_empty(),
            "leaf.prev" | "leaf.next" | "leaf.next_unreviewed"
            | "leaf.reviewed" | "leaf.reject" => !self.results.is_empty(),
            "region.confirm" | "region.reject" | "region.reassign" =>
                !self.effective_selection().is_empty(),
            "view.recon" => self.selected_idx.and_then(|i| self.results.get(i))
                .map_or(false, |l| !l.recon_mask.is_empty()),
            "view.focus" => self.focus_available(),
            "view.clear_focus" => self.selected_cluster.is_some(),
            _ => true,
        }
    }

    /// THE dispatch point. Keys, the palette and (where sensible) buttons all
    /// route here, so an action cannot behave differently depending on how it was
    /// invoked — which is exactly how `Enter` ended up meaning two things.
    fn perform_action(&mut self, id: &str, toasts: &mut ToastManager) {
        if !self.action_enabled(id) {
            return;
        }
        match id {
            "leaf.prev" | "leaf.next" => {
                let n = self.results.len();
                let next = id == "leaf.next";
                let target = match self.selected_idx {
                    None => 0,
                    Some(cur) if next => (cur + 1).min(n - 1),
                    Some(cur) => cur.saturating_sub(1),
                };
                if self.selected_idx != Some(target) {
                    self.selected_idx = Some(target);
                    self.selected_region = None;
                    self.overlay_tex = None;
                    self.scroll_to_leaf = true;
                }
            }
            "leaf.next_unreviewed" => {
                let from = self.selected_idx.map_or(0, |i| i + 1);
                match self.next_unreviewed(from) {
                    Some(t) => {
                        self.selected_idx = Some(t);
                        self.selected_region = None;
                        self.overlay_tex = None;
                        self.scroll_to_leaf = true;
                    }
                    None => toasts.success("Every leaf has been reviewed or rejected."),
                }
            }
            "leaf.reviewed" => {
                if let Some(li) = self.selected_idx { self.toggle_reviewed(li, toasts); }
            }
            "leaf.reject" => {
                if let Some(li) = self.selected_idx { self.toggle_reject_leaf(li, toasts); }
            }
            "region.confirm" => {
                let sel = self.effective_selection();
                self.confirm_regions(&sel, toasts);
            }
            "region.reject" => {
                let sel = self.effective_selection();
                // write_reject = true: an explicit reject IS training signal
                // ("the model was wrong here"), unlike an eraser stroke that
                // merely empties a mask.
                self.remove_regions(&sel, toasts, true);
            }
            "review.confirm_family" => {
                if let Some(cid) = self.selected_cluster {
                    let pending: Vec<usize> = (0..self.regions.len())
                        .filter(|&i| self.region_visible(i) && self.labels[i] == cid
                            && !self.persisted.contains(&i)
                            && (!self.filter_leaf_only
                                || self.selected_idx.map_or(true, |li| self.regions[i].leaf == li)))
                        .collect();
                    let n = pending.len();
                    self.confirm_regions(&pending, toasts);
                    toasts.success(format!("Confirmed {n} region(s)"));
                }
            }
            "region.next" | "region.prev" => {
                // Walk the SAME order the gallery is showing, so keyboard and eye
                // agree — stepping in detection order while the grid is sorted by
                // atypicality would be its own kind of confusing.
                let order = self.gallery_order();
                if order.is_empty() {
                    return;
                }
                let cur = self.selected_region.and_then(|s| order.iter().position(|&i| i == s));
                let next = match (cur, id == "region.next") {
                    (None, _) => 0,
                    (Some(p), true)  => (p + 1).min(order.len() - 1),
                    (Some(p), false) => p.saturating_sub(1),
                };
                let target = order[next];
                self.selected_region = Some(target);
                // Follow the region onto its leaf, otherwise stepping past the
                // end of one leaf's detections selects something invisible.
                let leaf = self.regions[target].leaf;
                if self.selected_idx != Some(leaf) {
                    self.selected_idx = Some(leaf);
                    self.scroll_to_leaf = true;
                }
                self.gallery_page = next / GALLERY_PER_PAGE;
                self.scroll_to_selected = true;
                self.overlay_tex = None;
            }
            "region.flag" => {
                if let Some(i) = self.selected_region {
                    if self.flagged.remove(&i) {
                        toasts.info("Un-flagged");
                    } else {
                        self.flagged.insert(i);
                        toasts.info("Set aside — find it again with the Flagged filter");
                    }
                }
            }
            "review.undo"   => self.undo_last_edit(toasts),
            "review.redo"   => self.redo_last_edit(toasts),
            "review.export" => self.export_results(toasts),
            "flow.finish" => self.stage_view = StageView::Done,
            "run.start" => {
                let (rev, rej, _) = self.review_counts();
                if !self.results.is_empty() && (rev + rej > 0) {
                    self.pending_reset = Some(PendingReset::Rerun);
                } else {
                    self.start();
                }
            }
            "run.cancel" => self.cancel_flag.store(true, Ordering::Relaxed),
            "view.outline" => { self.overlay_outline = !self.overlay_outline; self.overlay_tex = None; }
            "view.recon"   => { self.show_recon = !self.show_recon; self.overlay_tex = None; }
            "view.focus" => self.toggle_focus_mode(toasts),
            "view.clear_focus" => {
                self.selected_cluster = None;
                self.selected_region = None;
                self.focus_mode = false;
                self.overlay_tex = None;
            }
            "view.fit" => { self.canvas_zoom = 1.0; self.canvas_pan = egui::Vec2::ZERO; }
            "view.panel" => self.setup_open = !self.setup_open,
            "help"    => self.help_open = !self.help_open,
            "palette" => { self.palette_open = true; self.palette_query.clear(); }
            _ => {
                // Tools: one arm rather than ten, since the id encodes the tool.
                let tool = match id {
                    "tool.select"       => Some(CanvasTool::Select),
                    "tool.mark_healthy" => Some(CanvasTool::MarkHealthy),
                    "tool.brush"        => Some(CanvasTool::Brush),
                    "tool.eraser"       => Some(CanvasTool::Eraser),
                    "tool.knife"        => Some(CanvasTool::Knife),
                    "tool.scissor"      => Some(CanvasTool::Scissor),
                    "tool.lasso"        => Some(CanvasTool::Lasso),
                    "tool.wand"         => Some(CanvasTool::Wand),
                    "tool.polygon"      => Some(CanvasTool::Polygon),
                    _ => None,
                };
                if let Some(t) = tool {
                    self.switch_tool_hotkey(t);
                }
            }
        }
    }

    /// Fuzzy-searchable list of every action, opened with Ctrl+K.
    ///
    /// This is the standard resolution of the expert-tool tension: a keyboard-
    /// first interface is fast but undiscoverable, a menu-driven one is
    /// discoverable but slow. A palette is both — and it teaches the binding by
    /// printing it beside each command, which is the transfer mechanism that
    /// actually moves people onto shortcuts.
    fn show_command_palette(&mut self, ctx: &Context, toasts: &mut ToastManager) {
        if !self.palette_open {
            return;
        }
        // Rank once per frame; twenty-odd actions, so cost is irrelevant.
        let mut hits: Vec<(i32, &'static shortcuts::ActionDef)> = shortcuts::ACTIONS.iter()
            .filter_map(|a| {
                let hay = format!("{} {}", a.label, a.group);
                shortcuts::fuzzy_score(&hay, &self.palette_query).map(|s| (s, a))
            })
            .collect();
        // Enabled actions first, then by score: an exact match you cannot run is
        // less useful than a near match you can.
        hits.sort_by_key(|(s, a)| (!self.action_enabled(a.id), -s));
        self.palette_sel = self.palette_sel.min(hits.len().saturating_sub(1));

        let mut run: Option<String> = None;
        let mut close = false;

        // Keys are read BEFORE the window so Enter/arrows drive the list rather
        // than the text field.
        ctx.input(|i| {
            if i.key_pressed(egui::Key::Escape) { close = true; }
            if i.key_pressed(egui::Key::ArrowDown) {
                self.palette_sel = (self.palette_sel + 1).min(hits.len().saturating_sub(1));
            }
            if i.key_pressed(egui::Key::ArrowUp) {
                self.palette_sel = self.palette_sel.saturating_sub(1);
            }
            if i.key_pressed(egui::Key::Enter) {
                if let Some((_, a)) = hits.get(self.palette_sel) {
                    run = Some(a.id.to_string());
                }
            }
        });

        egui::Window::new("Commands")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            // A real height. `fixed_size(_, 0.0)` collapsed the window to its
            // minimum, so the scroll area inside got almost no room and showed
            // two and a half rows — the palette's whole value is seeing the
            // candidates at a glance.
            .fixed_size(egui::vec2(520.0, 420.0))
            .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 80.0))
            .show(ctx, |ui| {
                let te = ui.add(
                    egui::TextEdit::singleline(&mut self.palette_query)
                        .hint_text("Search commands…")
                        .desired_width(f32::INFINITY),
                );
                if !self.palette_focused {
                    te.request_focus();
                    self.palette_focused = true;
                }
                if te.changed() {
                    self.palette_sel = 0;
                }
                ui.add_space(4.0);
                // One line per command, not two: two-line rows fit about four
                // suggestions in the same space that now holds ten, and the whole
                // value of a palette is seeing the candidates without scrolling.
                // The group moves onto the same line, dimmed, and the hint moves
                // to hover.
                egui::ScrollArea::vertical()
                    .max_height(340.0)
                    .auto_shrink([false, false]) // fill the window instead of hugging content
                    .id_salt("palette_list").show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 1.0;
                    for (row, (_, a)) in hits.iter().enumerate() {
                        let enabled = self.action_enabled(a.id);
                        let selected = row == self.palette_sel;
                        let key = self.keymap.key(a.id);
                        let resp = ui.add_enabled(
                            enabled,
                            egui::SelectableLabel::new(selected, {
                                let mut s = a.label.to_string();
                                if !shortcuts::is_unbound(key) {
                                    s.push_str(&format!("   [{}]", shortcuts::key_label(key)));
                                }
                                s.push_str(&format!("   ·  {}", a.group));
                                RichText::new(s)
                            }),
                        ).on_hover_text(a.hint);
                        if resp.clicked() {
                            run = Some(a.id.to_string());
                        }
                    }
                    if hits.is_empty() {
                        ui.label(RichText::new("No command matches that.")
                            .small().color(ui_kit::MUTED()));
                    }
                });
                ui.add_space(2.0);
                // Spelled out rather than "↑ ↓": those arrow glyphs are not in the
                // bundled fonts and rendered as tofu boxes — the same gap that
                // turned the reviewed-tick into a square.
                ui.label(RichText::new(format!(
                    "{} commands · Up/Down to move · Enter to run · Esc to close", hits.len()
                )).small().color(ui_kit::MUTED()));
            });

        if let Some(id) = run {
            self.perform_action(&id, toasts);
            close = true;
        }
        if close {
            self.palette_open = false;
            self.palette_focused = false;
            self.palette_query.clear();
            self.palette_sel = 0;
        }
    }

    /// Confirmation dialogs for the two irreversible actions — deleting a class
    /// everywhere, and discarding a review session.
    ///
    /// Each states the exact scope ("this will affect N regions across M leaves")
    /// rather than asking a generic "are you sure?", and neither makes the
    /// destructive option the default button.
    fn show_confirm_dialogs(&mut self, ctx: &Context, toasts: &mut ToastManager) {
        if self.confirm_clear_curations {
            let n = self.curation_row_count();
            let dir = self.output_folder.as_ref().map(|o| o.join("curations"));
            let mut decided = None;
            egui::Window::new("Delete all curations?")
                .collapsible(false).resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.set_min_width(400.0);
                    ui.label(RichText::new(format!(
                        "{} curated example{} will be erased.",
                        fmt_thousands(n), if n == 1 { "" } else { "s" })).strong());
                    if let Some(d) = &dir {
                        ui.add_space(4.0);
                        ui.label(RichText::new(d.display().to_string())
                            .small().color(ui_kit::MUTED()));
                    }
                    ui.add_space(4.0);
                    ui.label("Every label, stamp and mined hard negative in this output \
                              folder's curations/ is deleted. Detections in the current \
                              run stay on screen — only the training record is cleared.");
                    ui.add_space(2.0);
                    ui.label(RichText::new(
                        "This cannot be undone, and no head file is changed: a head \
                         already retrained from these curations keeps whatever it \
                         learned. Reselect the original head if you need to undo that too.")
                        .small().color(Color32::from_rgb(220, 150, 130)));
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() { decided = Some(false); }
                        if ui.add(egui::Button::new(
                            RichText::new("Delete all curations").color(Color32::WHITE))
                            .fill(Color32::from_rgb(170, 55, 55))).clicked()
                        {
                            decided = Some(true);
                        }
                    });
                });
            match decided {
                Some(true)  => { self.clear_curations(toasts); self.confirm_clear_curations = false; }
                Some(false) => { self.confirm_clear_curations = false; }
                None => {}
            }
        }
        if let Some(cid) = self.pending_delete_cluster {
            let name = self.class_display_name(cid);
            let n = self.regions.iter().enumerate()
                .filter(|(i, r)| self.labels.get(*i) == Some(&cid) && self.region_visible(*i)
                    && self.results.get(r.leaf).is_some())
                .count();
            let mut decided = None;
            egui::Window::new("Delete this class?")
                .collapsible(false).resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.set_min_width(380.0);
                    ui.label(RichText::new(format!("\u{201c}{name}\u{201d}")).strong());
                    ui.add_space(4.0);
                    ui.label(format!(
                        "Every curated example of this class on disk is rewritten to \
                         \u{201c}rejected\u{201d}, and the class is removed from the head file. \
                         {n} region(s) in this run carry it."
                    ));
                    ui.add_space(2.0);
                    ui.label(RichText::new(
                        "This cannot be undone. The .json.bak beside the head is best-effort \
                         and may not exist.")
                        .small().color(Color32::from_rgb(220, 150, 130)));
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Keep it").clicked() { decided = Some(false); }
                        if ui.add(egui::Button::new(
                            RichText::new("Delete everywhere").color(Color32::WHITE))
                            .fill(Color32::from_rgb(170, 55, 55))).clicked()
                        {
                            decided = Some(true);
                        }
                    });
                });
            match decided {
                Some(true)  => { self.delete_cluster_from_head(cid, toasts);
                                 self.pending_delete_cluster = None; }
                Some(false) => { self.pending_delete_cluster = None; }
                None => {}
            }
        }

        if let Some(kind) = self.pending_reset {
            let (rev, rej, tot) = self.review_counts();
            let mut decided = None;
            egui::Window::new("Discard this review session?")
                .collapsible(false).resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.set_min_width(400.0);
                    ui.label(match kind {
                        PendingReset::Rerun =>
                            "Running the pipeline again clears the results currently loaded.",
                        PendingReset::SwitchHead =>
                            "Switching to the retrained head clears the results currently loaded.",
                    });
                    ui.add_space(4.0);
                    ui.label(format!(
                        "{tot} leaves are loaded \u{2014} {rev} marked reviewed, {rej} rejected. \
                         The undo stack and which leaf you were on are lost."
                    ));
                    ui.add_space(2.0);
                    ui.label(RichText::new(
                        "Confirmed curations and the reviewed/rejected marks are already on \
                         disk and will be restored on the next run.")
                        .small().color(ui_kit::MUTED()));
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Stay here").clicked() { decided = Some(false); }
                        if ui.button("Discard and continue").clicked() { decided = Some(true); }
                    });
                });
            match decided {
                Some(true) => {
                    match kind {
                        PendingReset::Rerun => { self.pending_reset = None; self.start(); }
                        PendingReset::SwitchHead => {
                            self.pending_reset = None;
                            self.reset_run_state();
                            toasts.info("Switched to the retrained head — click Run Pipeline to see corrected results.");
                        }
                    }
                }
                Some(false) => { self.pending_reset = None; }
                None => {}
            }
        }
    }

    /// The keyboard-shortcuts window: every binding, what it does, whether it has
    /// been customised, click-to-rebind, and one button back to defaults.
    ///
    /// Rendered from `shortcuts::ACTIONS`, so it cannot fall out of step with what
    /// the keys actually do — the previous arrangement documented bindings in code
    /// comments and a couple of tooltips, which is how `Enter` ended up bound twice
    /// without anyone noticing.
    fn show_shortcuts_window(&mut self, ctx: &Context) {
        if !self.help_open {
            self.rebinding = None;
            return;
        }
        let mut open = self.help_open;

        // Capture a keypress for the action awaiting one. Done before the window so
        // Escape can cancel without the window's own widgets seeing the key first.
        if let Some(id) = self.rebinding.clone() {
            let pressed = ctx.input(|i| {
                i.events.iter().find_map(|e| match e {
                    egui::Event::Key { key, pressed: true, .. } => Some(*key),
                    _ => None,
                })
            });
            if let Some(k) = pressed {
                if k != egui::Key::Escape {
                    self.keymap.set(&id, k);
                }
                self.rebinding = None;
            }
        }

        egui::Window::new("Keyboard shortcuts")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_width(520.0)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ctx, |ui| {
                let conflicts = self.keymap.conflicts();
                if !conflicts.is_empty() {
                    for (k, ids) in &conflicts {
                        let names: Vec<&str> = ids
                            .iter()
                            .filter_map(|id| shortcuts::action(id).map(|a| a.label))
                            .collect();
                        ui.label(
                            RichText::new(format!(
                                "Conflict: {} is bound to {} actions: {}",
                                shortcuts::key_label(*k), ids.len(), names.join(", ")
                            ))
                            .small()
                            .color(Color32::from_rgb(220, 170, 90)),
                        )
                        .on_hover_text(
                            "Not necessarily wrong — two actions can share a key when they are \
                             never active at the same time. Shown so you can decide, rather than \
                             the app refusing a binding it cannot prove is a mistake.",
                        );
                    }
                    ui.add_space(4.0);
                }

                if self.rebinding.is_some() {
                    ui.label(
                        RichText::new("Press any key…  (Esc cancels)")
                            .strong().color(ui_kit::ACCENT()),
                    );
                    ui.add_space(4.0);
                }

                egui::ScrollArea::vertical()
                    .max_height(420.0)
                    .id_salt("shortcut_list")
                    .show(ui, |ui| {
                        let mut last_group = "";
                        // ACTIONS is ordered most-used first; render in that order
                        // rather than alphabetically so the list teaches priority.
                        for a in shortcuts::ACTIONS {
                            if a.group != last_group {
                                if !last_group.is_empty() {
                                    ui.add_space(6.0);
                                }
                                ui_kit::section_header(ui, a.group);
                                last_group = a.group;
                            }
                            ui.horizontal(|ui| {
                                let capturing = self.rebinding.as_deref() == Some(a.id);
                                let label = if capturing {
                                    "…".to_string()
                                } else {
                                    shortcuts::key_label(self.keymap.key(a.id)).to_string()
                                };
                                let mut btn = egui::Button::new(
                                    RichText::new(label).monospace().strong(),
                                );
                                if capturing {
                                    btn = btn.fill(ui_kit::ACCENT());
                                }
                                if ui
                                    .add_sized([64.0, 22.0], btn)
                                    .on_hover_text("Click, then press the key you want.")
                                    .clicked()
                                {
                                    self.rebinding = Some(a.id.to_string());
                                }
                                ui.label(a.label).on_hover_text(a.hint);
                                if !self.keymap.is_default(a.id) {
                                    ui.label(
                                        RichText::new("changed").small().color(ui_kit::MUTED()),
                                    )
                                    .on_hover_text(format!(
                                        "Default: {}",
                                        shortcuts::key_label(a.default_key)
                                    ));
                                    if ui.small_button("reset").on_hover_text("Back to default").clicked()
                                    {
                                        self.keymap.reset_one(a.id);
                                    }
                                }
                            });
                        }
                    });

                ui.separator();
                ui.horizontal(|ui| {
                    let n = self.keymap.n_customised();
                    if ui
                        .add_enabled(n > 0, egui::Button::new("Reset all to defaults"))
                        .on_hover_text("Discards every custom binding and restores the shipped set.")
                        .on_disabled_hover_text("Nothing has been customised.")
                        .clicked()
                    {
                        self.keymap.reset_all();
                        self.rebinding = None;
                    }
                    ui.label(
                        RichText::new(if n == 0 {
                            "all defaults".to_string()
                        } else {
                            format!("{n} customised")
                        })
                        .small()
                        .color(ui_kit::MUTED()),
                    );
                });
                ui.label(
                    RichText::new(
                        "Shortcuts are inactive while a text field has focus, so typing a \
                         cluster name never triggers one.",
                    )
                    .small()
                    .color(ui_kit::MUTED()),
                );
            });

        self.help_open = open;
        if !self.help_open {
            self.rebinding = None;
        }
    }

    // ── review state: survives closing the app ────────────────────────────
    //
    // Written to `<output>/review_state.jsonl`, append-only, last line per key
    // wins. Append-only rather than rewrite-in-place so a crash mid-write costs
    // one line instead of the whole file, matching the `labels.jsonl` convention
    // that already works for curations.

    /// Stable identity of a leaf: its source photo relative to the source folder,
    /// plus which leaf it was within that photo. `None` when the source folder is
    /// unknown or the path lies outside it — in which case the leaf simply is not
    /// persisted, rather than being written under a key that won't match later.
    fn leaf_key(&self, li: usize) -> Option<(String, u32)> {
        let leaf = self.results.get(li)?;
        let root = self.source_folder.as_ref()?;
        let rel = leaf.src.strip_prefix(root).unwrap_or(&leaf.src);
        // Forward slashes so a state file stays valid if the project is opened
        // from a different OS or a moved drive letter.
        let rel = rel.to_string_lossy().replace('\\', "/");
        let ordinal = self.results[..li].iter().filter(|l| l.src == leaf.src).count() as u32;
        Some((rel, ordinal))
    }

    fn review_state_path(&self) -> Option<PathBuf> {
        Some(self.output_folder.as_ref()?.join("review_state.jsonl"))
    }

    /// Read the whole file into `review_marks`, later lines overwriting earlier
    /// ones for the same key. Called once when a run starts, before any leaf
    /// arrives, so marks can be applied as leaves stream in.
    fn load_review_state(&mut self) {
        self.review_marks.clear();
        self.review_mismatch = 0;
        let Some(p) = self.review_state_path() else { return };
        let Ok(text) = std::fs::read_to_string(&p) else { return };
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
            let (Some(src), Some(n), Some(state)) = (
                v.get("src").and_then(|x| x.as_str()),
                v.get("n").and_then(|x| x.as_u64()),
                v.get("state").and_then(|x| x.as_str()),
            ) else { continue };
            let w = v.get("w").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
            let h = v.get("h").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
            self.review_marks
                .insert((src.to_string(), n as u32), (state.to_string(), w, h));
        }
    }

    /// Apply any stored mark to a leaf that has just arrived.
    ///
    /// The stored width/height are a fingerprint of the segmentation that produced
    /// the mark. If they disagree, the leaf under this key is not the leaf the
    /// user reviewed — so the mark is COUNTED and DROPPED rather than applied.
    /// Silently restoring it would put a "reviewed" tick on a leaf nobody has
    /// seen, which is worse than losing the tick.
    fn apply_review_mark(&mut self, li: usize) {
        let Some(key) = self.leaf_key(li) else { return };
        let Some((state, w, h)) = self.review_marks.get(&key).cloned() else { return };
        let Some(leaf) = self.results.get(li) else { return };
        if w != 0 && h != 0 && (w != leaf.w || h != leaf.h) {
            self.review_mismatch += 1;
            return;
        }
        match state.as_str() {
            "reviewed" => { self.reviewed.insert(li); }
            "rejected" => { self.rejected_leaves.insert(li); }
            _ => {}
        }
    }

    /// Append one line. Best-effort by nature (a review bookmark is not worth
    /// interrupting a session over), but unlike the curation writes this one
    /// reports failure instead of pretending it succeeded.
    fn write_review_mark(&mut self, li: usize, state: &str, toasts: &mut ToastManager) {
        let (Some(key), Some(path)) = (self.leaf_key(li), self.review_state_path()) else { return };
        let Some(leaf) = self.results.get(li) else { return };
        let line = format!(
            "{{\"src\":\"{}\",\"n\":{},\"state\":\"{}\",\"w\":{},\"h\":{}}}\n",
            json_escape(&key.0), key.1, state, leaf.w, leaf.h,
        );
        use std::io::Write;
        let res = std::fs::OpenOptions::new()
            .create(true).append(true).open(&path)
            .and_then(|mut f| f.write_all(line.as_bytes()));
        if let Err(e) = res {
            toasts.error(format!("could not save review state: {e}"));
        }
        self.review_marks.insert(key, (state.to_string(), leaf.w, leaf.h));
    }

    /// Toggle "I have looked at this leaf and I'm happy with it".
    ///
    /// Refuses on a rejected leaf rather than silently un-rejecting it: the
    /// button is hidden in that state, so having the hotkey quietly do something
    /// the UI does not offer would make M destroy a rejection by accident.
    fn toggle_reviewed(&mut self, li: usize, toasts: &mut ToastManager) {
        if self.rejected_leaves.contains(&li) {
            toasts.info(format!("Leaf {li} is rejected — press X to restore it first."));
            return;
        }
        if self.reviewed.remove(&li) {
            self.write_review_mark(li, "none", toasts);
        } else {
            self.reviewed.insert(li);
            self.write_review_mark(li, "reviewed", toasts);
        }
        self.overlay_tex = None;
    }

    /// First leaf at or after `from` that is neither reviewed nor rejected.
    /// Wraps once so the search covers the whole batch from wherever you are.
    fn next_unreviewed(&self, from: usize) -> Option<usize> {
        let n = self.results.len();
        (0..n)
            .map(|off| (from + off) % n)
            .find(|i| !self.reviewed.contains(i) && !self.rejected_leaves.contains(i))
    }

    fn review_counts(&self) -> (usize, usize, usize) {
        (self.reviewed.len(), self.rejected_leaves.len(), self.results.len())
    }

    /// Toggle whole-leaf rejection. Reversible on purpose — it is one click in a
    /// top bar, next to the destructive-looking controls, and an accidental press
    /// would otherwise silently drop a leaf's anomalies from the export with no
    /// way back short of re-running the pipeline.
    fn toggle_reject_leaf(&mut self, li: usize, toasts: &mut ToastManager) {
        if self.rejected_leaves.remove(&li) {
            self.write_review_mark(li, "none", toasts);
            toasts.success(format!("Leaf {li} restored"));
        } else {
            self.rejected_leaves.insert(li);
            // A rejected leaf is not also "reviewed" — see toggle_reviewed.
            self.reviewed.remove(&li);
            let n = self.regions.iter().filter(|r| r.leaf == li).count();
            self.write_review_mark(li, "rejected", toasts);
            toasts.success(format!("Leaf {li} rejected — {n} anomalies excluded"));
        }
        self.overlay_tex = None;  // canvas overlay is cached; force a repaint
    }

    /// Is the output folder the source folder, or nested inside it?
    ///
    /// `segment_one` writes every leaf as `{stem}_leafNN.png` into the output
    /// folder ([leaf_seg/inference.rs] `cutout_path`), and `list_images` walks the
    /// source folder RECURSIVELY (walkdir). So an output at or under the source
    /// means the next run scans its own cutouts as if they were new photographs,
    /// segments cutouts of cutouts, and duplicates every leaf — reported from the
    /// field as "2 leaves became 4 after rerunning".
    ///
    /// It also quietly breaks review state, because the duplicates are new leaves
    /// with their own keys, so progress appears to reset.
    ///
    /// Returns the explanation to show, or `None` when the folders are fine.
    fn output_inside_source(&self) -> Option<String> {
        let (src, out) = (self.source_folder.as_ref()?, self.output_folder.as_ref()?);
        // Canonicalize so `.\out` vs `out` and differing case on Windows still
        // compare equal; fall back to the raw path when the folder is not yet
        // created, which is the common case for a fresh output folder.
        let s = src.canonicalize().unwrap_or_else(|_| src.clone());
        let o = out.canonicalize().unwrap_or_else(|_| out.clone());
        if o == s {
            Some(format!(
                "Both are {}.\n\nEach run writes its leaf cut-outs into the output \
                 folder, and the source folder is scanned recursively — so the next \
                 run would treat those cut-outs as new photographs and duplicate \
                 every leaf. Choose an output folder outside the source folder.",
                s.display()
            ))
        } else if o.starts_with(&s) {
            Some(format!(
                "Output {} sits under source {}.\n\nThe source scan is recursive, so \
                 the leaf cut-outs written there would be picked up as new \
                 photographs on the next run and every leaf would be duplicated. \
                 Choose an output folder outside the source folder.",
                o.display(), s.display()
            ))
        } else {
            None
        }
    }

    fn all_paths_ok(&self) -> bool {
        let ex = |p: Option<PathBuf>| p.map(|p| p.exists()).unwrap_or(false);
        // detector: EITHER the few-shot head or the PatchCore bank+meta is enough
        // to run (they're complementary and run together when both are present —
        // see worker.rs — but only one is strictly required).
        let detector_ok = if self.fewshot_active() {
            ex(self.eff_head())
        } else {
            ex(self.eff_bank()) && ex(self.eff_meta())
        };
        ex(self.eff_yolo())
            && ex(self.eff_dino())
            && detector_ok
            && self.source_folder.is_some()
            && self.output_folder.is_some()
    }

    fn start(&mut self) {
        let (Some(yolo), Some(dino), Some(src), Some(out)) = (
            self.eff_yolo(),
            self.eff_dino(),
            self.source_folder.clone(),
            self.output_folder.clone(),
        ) else { return };

        // Detector selection: few-shot head (preferred) OR PatchCore bank+meta.
        // For the few-shot path bank/meta are unused; pass whatever is resolved
        // (the worker ignores them when a head is present).
        let head = if self.fewshot_active() { self.eff_head() } else { None };
        let (bank, meta) = if head.is_some() {
            (self.eff_bank().unwrap_or_default(), self.eff_meta().unwrap_or_default())
        } else {
            let (Some(bank), Some(meta)) = (self.eff_bank(), self.eff_meta()) else { return };
            (bank, meta)
        };

        let image_paths = list_images(&src);
        if image_paths.is_empty() {
            self.log.push("No images in source folder.".into());
            return;
        }
        self.reset_run_state();
        // Load BEFORE the worker starts: leaves stream in one at a time and each
        // applies its own mark on arrival, so the table has to be populated first.
        self.load_review_state();
        self.log.clear();
        self.cancel_flag = Arc::new(AtomicBool::new(false));
        self.progress_done = 0;
        self.progress_total = image_paths.len();
        self.running = true;
        self.run_started_at = Some(std::time::Instant::now());
        self.stage = "Loading models".into();

        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);
        spawn_pipeline(
            PipeConfig {
                image_paths,
                output_dir: out,
                yolo_model: yolo,
                dino_model: dino,
                bank_path: bank,
                meta_path: meta,
                tile_size: self.tile_size,
                margin_erode: self.margin_erode_px,
                detect_holes: self.detect_holes,
                min_hole_area: self.min_hole_area,
                filter_margin_holes: self.filter_margin_holes,
                hole_margin_px: self.hole_margin_px,
                dino_res: crate::tabs::pipeline::worker::default_dino_res(),
                conf: self.conf,
                recon_ckpt: self.eff_recon(),
                head_path: head,
                use_patchcore: self.use_patchcore,
                unsupervised_families: self.unsupervised_families,
                domain_projection: self.domain_projection,
                head_tau: self.head_tau,
                head_grow: self.head_grow.min(self.head_tau),
                seg_alpha_lo: self.seg_alpha_lo,
                seg_chroma_min: self.seg_chroma_min,
                cluster_eps: self.cluster_eps,
                cluster_min_pts: self.cluster_min_pts,
                cluster_algo: self.cluster_algo,
                target_k: self.target_k,
                cut_mode: self.cut_mode,
                adaptive_threshold: self.adaptive_threshold,
            },
            tx,
            self.cancel_flag.clone(),
        );
    }

    /// Find the retrain base set in `models/` without the user picking it.
    ///
    /// It is a ~1.2 GB export that is the same for every run, so making it a
    /// manual per-session choice only created a way to forget it — and a retrain
    /// with no base set silently drops the head's retained IoU from 0.475 to
    /// 0.125. Uses the same search order as the model weights.
    fn default_base_set() -> Option<PathBuf> {
        Self::base_set_candidates().into_iter().find(|p| p.exists())
    }

    /// Every path checked for the base set, in order. Also drives the warning's
    /// hover text — "not found" is only actionable if it says where it looked.
    fn base_set_candidates() -> Vec<PathBuf> {
        if let Ok(p) = std::env::var("LACUNA_BASE_SET") {
            return vec![PathBuf::from(p)];
        }
        let mut out = Vec::new();
        for name in BASE_SET_NAMES {
            // Next to the executable first: that is the packaged layout, and it
            // is the only one that does not depend on the working directory.
            if let Some(exe) = crate::paths::exe_dir() {
                out.push(exe.join("models").join(name));
                // `target/release/lacuna.exe` -> repo `models/`, so a cargo build
                // run from anywhere still finds it.
                if let Some(repo) = exe.parent().and_then(|p| p.parent()) {
                    out.push(repo.join("models").join(name));
                }
            }
            out.push(PathBuf::from("models").join(name));
        }
        out
    }

    /// Erase the output folder's curation record.
    ///
    /// Deletes the whole `curations/` tree — `labels.jsonl`, the label crops, and
    /// the mined hard negatives — because a partial delete is what produces a
    /// contaminated folder in the first place: a `labels.jsonl` line whose PNG is
    /// gone, or an orphan PNG no line refers to.
    ///
    /// The healthy-tile feature cache is deliberately kept: it is derived from
    /// image data, not from labels, so it is not part of the contamination and
    /// re-extracting it costs a full DINO pass over every tile.
    fn clear_curations(&mut self, toasts: &mut ToastManager) {
        let Some(out) = self.output_folder.clone() else { return };
        let dir = out.join("curations");
        if !dir.exists() {
            toasts.info("Nothing to delete — no curations folder in this output folder.");
            return;
        }
        let cache = dir.join("healthy_feature_cache");
        let keep = cache.exists().then(|| out.join(".lacuna_feature_cache_tmp"));
        if let Some(tmp) = &keep {
            let _ = std::fs::remove_dir_all(tmp);
            if std::fs::rename(&cache, tmp).is_err() {
                // Could not park it — better to keep everything than to delete a
                // cache the user did not ask to lose.
                toasts.error("Could not move the feature cache aside; nothing was deleted.");
                return;
            }
        }
        match std::fs::remove_dir_all(&dir) {
            Ok(()) => {
                let _ = std::fs::create_dir_all(&dir);
                if let Some(tmp) = &keep {
                    let _ = std::fs::rename(tmp, &cache);
                }
                self.curation_count_cache = None;
                toasts.success("Curations deleted — the flywheel starts clean.");
            }
            Err(e) => {
                if let Some(tmp) = &keep {
                    let _ = std::fs::rename(tmp, &cache);
                }
                toasts.error(format!("Could not delete curations: {e}"));
            }
        }
    }

    /// Number of curated examples accumulated in the current output folder.
    ///
    /// One `labels.jsonl` line per example. Cached on (len, mtime): this is read
    /// from draw code, and the file grows to tens of thousands of lines.
    fn curation_row_count(&mut self) -> usize {
        let Some(out) = self.output_folder.clone() else { return 0 };
        let path = out.join("curations").join("labels.jsonl");
        let Ok(meta) = std::fs::metadata(&path) else {
            self.curation_count_cache = None;
            return 0;
        };
        let key = (meta.len(), meta.modified().ok());
        if let Some((len, mtime, n)) = &self.curation_count_cache {
            if (*len, *mtime) == key {
                return *n;
            }
        }
        let n = std::fs::read_to_string(&path)
            .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count())
            .unwrap_or(0);
        self.curation_count_cache = Some((key.0, key.1, n));
        n
    }

    /// The base-row count a retrain will actually use.
    fn effective_base_rows(&mut self) -> usize {
        if !self.retrain_auto_base_rows {
            return self.retrain_base_rows;
        }
        // Floor at the measured-best 10k, then hold ~10 base rows per curated
        // example so the balance does not drift as curations accumulate. Capped
        // so a very large curation set cannot make every retrain unbearable —
        // retrain re-featurizes each row from scratch.
        (self.curation_row_count() * 10).clamp(10_000, 100_000)
    }

    /// Picks the next source image (cycling through the folder on repeated
    /// clicks) and, if a detector is actually configured (head, or
    /// bank+meta), runs the FULL existing pipeline on just that one image —
    /// so the user sees what the model currently gets right/wrong on this
    /// leaf and can correct it, instead of starting from a blank canvas.
    /// Falls back to `start_calibration_preview_blank` (segmentation only,
    /// no detection) when nothing is configured to detect with yet.
    fn start_calibration_preview(&mut self) {
        let (Some(yolo), Some(dino), Some(src), Some(out)) = (
            self.eff_yolo(), self.eff_dino(), self.source_folder.clone(), self.output_folder.clone(),
        ) else { return };
        let images = crate::tabs::leaf_seg::inference::list_images(&src);
        if images.is_empty() {
            self.preview_note = "No source images found.".into();
            return;
        }
        let idx = self.calib_preview_n % images.len();
        self.calib_preview_n += 1;
        let picked = images[idx].clone();

        // Same detector resolution as `start()` (mod.rs `fn start`) — head if
        // configured, else bank+meta.
        let head = if self.fewshot_active() { self.eff_head() } else { None };
        let (bank, meta) = if head.is_some() {
            (self.eff_bank().unwrap_or_default(), self.eff_meta().unwrap_or_default())
        } else if let (Some(b), Some(m)) = (self.eff_bank(), self.eff_meta()) {
            (b, m)
        } else {
            self.start_calibration_preview_blank(yolo, picked);
            return;
        };

        let (tx, rx) = mpsc::channel();
        self.calib_detect_rx = Some(rx);
        self.calib_detect_leaf_idx = None;
        self.calib_detect_cancel = Arc::new(AtomicBool::new(false));
        self.preview_busy = true;
        self.preview_note = "Running detection on this leaf…".into();
        spawn_pipeline(
            PipeConfig {
                image_paths: vec![picked],
                output_dir: out,
                yolo_model: yolo,
                dino_model: dino,
                bank_path: bank,
                meta_path: meta,
                tile_size: self.tile_size,
                margin_erode: self.margin_erode_px,
                detect_holes: self.detect_holes,
                min_hole_area: self.min_hole_area,
                filter_margin_holes: self.filter_margin_holes,
                hole_margin_px: self.hole_margin_px,
                dino_res: crate::tabs::pipeline::worker::default_dino_res(),
                conf: self.conf,
                recon_ckpt: self.eff_recon(),
                head_path: head,
                use_patchcore: self.use_patchcore,
                // Always the head's direct classification for a single-leaf
                // preview — DBSCAN/Hierarchical/domain_projection on a
                // handful of points from one leaf would be statistically
                // meaningless, regardless of what the Settings panel has
                // configured for full batch runs.
                unsupervised_families: false,
                domain_projection: false,
                head_tau: self.head_tau,
                head_grow: self.head_grow.min(self.head_tau),
                seg_alpha_lo: self.seg_alpha_lo,
                seg_chroma_min: self.seg_chroma_min,
                cluster_eps: self.cluster_eps,
                cluster_min_pts: self.cluster_min_pts,
                cluster_algo: self.cluster_algo,
                target_k: self.target_k,
                cut_mode: self.cut_mode,
                adaptive_threshold: self.adaptive_threshold,
            },
            tx,
            self.calib_detect_cancel.clone(),
        );
    }

    /// Fallback when no detector is configured yet: segmentation-only
    /// preview, pushes a real (but zero-region) `PipelineLeaf` so the
    /// existing canvas/Brush/Wand machinery works on it unchanged — the
    /// original calibration-preview behavior, kept for bootstrapping from
    /// nothing.
    fn start_calibration_preview_blank(&mut self, yolo: PathBuf, picked: PathBuf) {
        let (alpha_lo, chroma_min) = (self.seg_alpha_lo, self.seg_chroma_min);
        let (tx, rx) = mpsc::channel();
        self.calib_preview_rx = Some((picked.clone(), rx));
        self.preview_busy = true;
        self.preview_note = "Segmenting a leaf to mark (no detector configured — \
                              set a few-shot head or PatchCore bank+meta to detect first)…".into();
        std::thread::spawn(move || {
            let _ = tx.send(crate::tabs::leaf_seg::inference::preview_cutout(&yolo, &picked, alpha_lo, chroma_min));
        });
    }

    fn poll_calibration_preview(&mut self, ctx: &Context) {
        if self.calib_preview_rx.is_some() {
            ctx.request_repaint();
        }
        let Some((src_path, rx)) = &self.calib_preview_rx else { return };
        let Ok(res) = rx.try_recv() else { return };
        let src_path = src_path.clone();
        self.calib_preview_rx = None;
        self.preview_busy = false;
        match res {
            Ok((rgba, w, h)) => {
                let n = (w * h) as usize;
                let leaf = worker::PipelineLeaf {
                    src: src_path,
                    w, h,
                    rgba,
                    anomaly: vec![false; n],
                    n_regions: 0,
                    recon_area: 0,
                    recon_whole: 0,
                    recon_mask: Vec::new(),
                    morph: None,
                };
                let vp = leaf.rgba.chunks_exact(4).filter(|c| c[3] > 10).count() as u32;
                self.leaf_valid_px.push(vp.max(1));
                self.results.push(leaf);
                self.thumbs.push(None);
                let new_idx = self.results.len() - 1;
                self.selected_idx = Some(new_idx);
                self.calib_preview_leaves.insert(new_idx);
                self.overlay_tex = None;
                // Brush/Wand — NOT MarkHealthy's fixed-square stamp — so a marked
                // example is a precise pixel mask, not a square that drags in
                // whatever background surrounds it and dilutes the centroid.
                self.canvas_tool = CanvasTool::Brush;
                self.on_tool_switched(CanvasTool::Brush);
                self.preview_note = "Paint (Brush) or wand-select + Fill a few precise examples \
                                      per class, then \"Preview another leaf\" or Save calibration.".into();
            }
            Err(e) => self.preview_note = format!("Calibration preview failed: {e}"),
        }
    }

    /// Drains `calib_detect_rx`, APPENDING each message's data rather than
    /// replacing — unlike `poll_worker`'s handling of the same `PipeMsg`
    /// variants, which wholesale-replaces `self.regions`/`labels`/`coords`
    /// (correct for a real batch run, wrong here: this must coexist with
    /// whatever's already loaded, including earlier calibration-preview leaves).
    fn poll_calibration_detect(&mut self, ctx: &Context, toasts: &mut ToastManager) {
        if self.calib_detect_rx.is_some() {
            ctx.request_repaint();
        }
        let Some(rx) = &self.calib_detect_rx else { return };
        let mut done = false;
        let mut errored = false;
        for msg in rx.try_iter().take(64) {
            match msg {
                PipeMsg::Stage(s) => self.preview_note = s,
                PipeMsg::Progress { .. } => {}
                PipeMsg::Leaf(leaf) => {
                    let vp = leaf.rgba.chunks_exact(4).filter(|c| c[3] > 10).count() as u32;
                    self.leaf_valid_px.push(vp.max(1));
                    self.results.push(leaf);
                    self.thumbs.push(None);
                    let new_idx = self.results.len() - 1;
                    self.selected_idx = Some(new_idx);
                    self.calib_preview_leaves.insert(new_idx);
                    self.calib_detect_leaf_idx = Some(new_idx);
                }
                PipeMsg::Clusters { labels, coords, names, mut regions, .. } => {
                    // This mini-run's regions all carry `leaf == 0` (its own
                    // local, single-leaf indexing) — remap to where the leaf
                    // actually landed in self.results before extending.
                    if let Some(new_idx) = self.calib_detect_leaf_idx {
                        for r in &mut regions {
                            r.leaf = new_idx;
                        }
                    }
                    self.region_area.extend(
                        regions.iter().map(|r| r.mask.iter().filter(|&&b| b).count() as u32)
                    );
                    self.region_thumbs.extend(regions.iter().map(|_| None));
                    self.regions.extend(regions);
                    self.labels.extend(labels);
                    self.coords.extend(coords);
                    for (id, name) in names {
                        self.cluster_names.entry(id).or_insert(name);
                    }
                    // `hcluster` deliberately ignored — belongs to the main
                    // run's re-cut state, and is always None here anyway
                    // (unsupervised_families forced off for this mini-run).
                }
                PipeMsg::Log(_) => {}
                PipeMsg::Error(e) => {
                    toasts.error(format!("Calibration detection failed: {e}"));
                    errored = true;
                }
                PipeMsg::Finished => done = true,
            }
        }
        if done || errored {
            self.preview_busy = false;
            self.calib_detect_rx = None;
            self.overlay_tex = None;
            self.build_clusters(toasts);
            // Reviewing/rejecting what detection already found is the
            // natural first action, not painting — Brush/Wand are one
            // hotkey away once the user's ready to add missed spots.
            self.canvas_tool = CanvasTool::Select;
            self.on_tool_switched(CanvasTool::Select);
            if done {
                self.preview_note = "Detection done — confirm/reject what's right, paint/wand-fill \
                                      what's missing, then Save calibration.".into();
            }
        }
    }

    /// Kicks off centroid calibration (`train::head::spawn_calibrate`) from
    /// whatever's accumulated in `curations/labels.jsonl` so far (this
    /// run's MarkHealthy stamps, same file the flywheel retrain reads),
    /// against the currently configured base head, saved as a new versioned
    /// file under `<output>/calibrated_heads/`.
    fn start_calibrate(&mut self) {
        let (Some(out), Some(head), Some(dino)) =
            (self.output_folder.clone(), self.eff_head(), self.eff_dino())
        else { return };
        let name = self.calib_name.trim();
        let stem = if name.is_empty() { "calibration".to_string() } else { sanitize_filename(name) };
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs()).unwrap_or(0);
        let out_path = out.join("calibrated_heads").join(format!("{stem}_{ts}.json"));
        // Only crops from regions persisted on THIS session's calibration-preview
        // leaves — never the output folder's whole curation history (see
        // `CalibrateCfg::only_crops`'s doc comment for why that matters).
        let only_crops: std::collections::HashSet<String> = self.persisted.iter()
            .filter(|&&i| self.regions.get(i).map_or(false, |r| self.calib_preview_leaves.contains(&r.leaf)))
            .map(|&i| format!("region_{i}.png"))
            .collect();
        self.calib_out_path = Some(out_path.clone());
        self.calib_log.clear();
        self.calib_cancel = Arc::new(AtomicBool::new(false));
        self.calib_running = true;
        let (tx, rx) = mpsc::channel();
        self.calib_rx = Some(rx);
        spawn_calibrate(
            CalibrateCfg {
                base_head_path: head,
                dino_model: dino,
                curations_dir: out.join("curations"),
                out_path,
                scale: self.calib_scale,
                only_crops,
            },
            tx,
            self.calib_cancel.clone(),
        );
    }

    fn poll_calibrate(&mut self, toasts: &mut ToastManager) {
        let Some(rx) = &self.calib_rx else { return };
        for msg in rx.try_iter().take(64) {
            match msg {
                RetrainMsg::Stage(s) => self.calib_log.push(s),
                RetrainMsg::Log(s) => self.calib_log.push(s),
                RetrainMsg::Error(e) => {
                    self.calib_running = false;
                    toasts.error(format!("Calibration failed: {e}"));
                }
                RetrainMsg::Done(summary) => {
                    self.calib_running = false;
                    self.calib_log.push(summary);
                    if let Some(path) = self.calib_out_path.take() {
                        toasts.success("Calibration saved and applied.");
                        self.calib_selected = Some(path.clone());
                        self.head_path = Some(path);
                        self.head_cache = None;
                    } else {
                        toasts.success("Calibration saved.");
                    }
                }
            }
        }
        if !self.calib_running {
            self.calib_rx = None;
        }
    }

    /// Lists saved calibrations for THIS output folder, newest first: (path,
    /// display name, seconds since saved).
    fn list_calibrations(&self) -> Vec<(PathBuf, String, u64)> {
        let Some(out) = &self.output_folder else { return Vec::new() };
        let dir = out.join("calibrated_heads");
        let Ok(entries) = std::fs::read_dir(&dir) else { return Vec::new() };
        let now = std::time::SystemTime::now();
        let mut out: Vec<(PathBuf, String, u64)> = entries.flatten()
            .filter_map(|e| {
                let path = e.path();
                if path.extension()?.to_str()? != "json" {
                    return None;
                }
                let stem = path.file_stem()?.to_str()?;
                let (name, ts) = stem.rsplit_once('_')?;
                let ts: u64 = ts.parse().ok()?;
                let name = name.to_string();
                let age = now.duration_since(std::time::UNIX_EPOCH + std::time::Duration::from_secs(ts))
                    .map(|d| d.as_secs()).unwrap_or(0);
                Some((path, name, age))
            })
            .collect();
        out.sort_by_key(|(_, _, age)| *age);
        out
    }

    fn poll_worker(&mut self, toasts: &mut ToastManager) {
        let mut finished = false;
        let mut got_clusters = false;
        // Leaves that arrived this frame. Their stored review marks are applied
        // AFTER the loop: the loop holds `&self.rx` for its whole body, so no
        // `&mut self` method can be called inside it.
        let mut arrived: Vec<usize> = Vec::new();
        if let Some(rx) = &self.rx {
            for msg in rx.try_iter().take(64) {
                match msg {
                    PipeMsg::Stage(s) => self.stage = s,
                    PipeMsg::Progress { done, total } => {
                        self.progress_done = done;
                        self.progress_total = total;
                    }
                    PipeMsg::Leaf(leaf) => {
                        if self.selected_idx.is_none() {
                            self.selected_idx = Some(self.results.len());
                        }
                        // cache valid-pixel count once (was an O(pixels) scan every frame)
                        let vp = leaf.rgba.chunks_exact(4).filter(|c| c[3] > 10).count() as u32;
                        self.leaf_valid_px.push(vp.max(1));
                        self.results.push(leaf);
                        self.thumbs.push(None);
                        arrived.push(self.results.len() - 1);
                    }
                    PipeMsg::Clusters { labels, coords, names, regions, hcluster } => {
                        self.region_thumbs = vec![None; regions.len()];
                        // cache each region's mask area once (was re-summed every frame in the table)
                        self.region_area = regions
                            .iter()
                            .map(|r| r.mask.iter().filter(|&&b| b).count() as u32)
                            .collect();
                        self.regions = regions;
                        self.labels = labels;
                        self.coords = coords;
                        // seed cluster names from the head's families (few-shot path)
                        for (id, name) in names {
                            self.cluster_names.entry(id).or_insert(name);
                        }
                        // Some only when Hierarchical ran this run — enables the live
                        // re-cut slider (Clusters tab) instead of a full pipeline rerun
                        // per K/sensitivity guess. Seed the live re-cut controls from
                        // whatever mode/threshold this run actually used, not a fixed
                        // default — there's no equivalent "K" to back-derive a
                        // threshold from when FixedK ran, so the Adaptive seed is just
                        // whatever adaptive_threshold was configured this run.
                        self.recut_mode = self.cut_mode;
                        self.recut_threshold = self.adaptive_threshold;
                        self.recut_k = hcluster.as_ref()
                            .map(|h| self.labels.iter().copied().filter(|&l| l >= 0).collect::<std::collections::HashSet<_>>().len().max(1))
                            .unwrap_or(0);
                        self.hcluster = hcluster;
                        got_clusters = true;
                    }
                    PipeMsg::Log(l) => self.log.push(l),
                    PipeMsg::Error(e) => {
                        self.log.push(format!("ERROR: {e}"));
                        toasts.error(format!("Pipeline failed: {e}"));
                        finished = true;
                    }
                    PipeMsg::Finished => finished = true,
                }
            }
        }
        for li in arrived {
            self.apply_review_mark(li);
        }
        if got_clusters {
            self.build_clusters(toasts);
            self.overlay_tex = None; // rebuild to reflect cluster colours
        }
        if finished {
            self.running = false;
            self.run_started_at = None;
            self.rx = None;
            toasts.success(format!("Pipeline done — {} leaves, {} clusters", self.results.len(), self.clusters.len()));
        }
    }

    /// Retrain the few-shot head from this run's curations, in place — no tab
    /// switch, no re-picking folders. Auto-fills from what Pipeline already
    /// has loaded. Gated mutually exclusive with a pipeline Run (see
    /// `can_retrain`/`can_start` in `show_controls`/`show_cluster_panel`):
    /// both keep a DINO instance resident on the GPU with zero coordination
    /// between them, and this codebase's own convention (Train tab already
    /// gates its Fit-job against its own Retrain-job the same way) is
    /// disable-the-button, not allow-and-hope.
    fn start_retrain(&mut self) {
        let (Some(out), Some(head), Some(dino)) =
            (self.output_folder.clone(), self.eff_head(), self.eff_dino())
        else { return };
        // Resolved BEFORE the config literal: in auto mode this reads the
        // curation file, which needs &mut self for the count cache.
        let base_rows = self.effective_base_rows();
        let curations_dir = out.join("curations");
        let out_path = head.parent().map(|p| p.join("fewshot_head_retrained.json"))
            .unwrap_or_else(|| PathBuf::from("fewshot_head_retrained.json"));
        self.retrain_log.clear();
        self.retrain_done = None;
        self.retrain_cancel = Arc::new(AtomicBool::new(false));
        self.retraining = true;
        self.retrain_stage = "Retraining head".into();
        let (tx, rx) = mpsc::channel();
        self.retrain_rx = Some(rx);
        spawn_retrain(
            RetrainCfg {
                head_path: head,
                dino_model: dino,
                curations_dir,
                out_path,
                // Full incident history in head.rs's RetrainCfg doc comments
                // and `retrain`'s own comments. Short version: an
                // anchor-toward-the-base-head scheme never worked (three
                // rounds of tuning it), replaced with standard zero-centered
                // L2; fixed-epoch/fixed-lr gradient descent was ALSO a direct
                // root cause of repeated under-convergence bugs, replaced with
                // a real L-BFGS solve (`max_iters` is a safety ceiling now,
                // not a target).
                max_iters: 2000,
                // sklearn's C, matching the base head's own
                // LogisticRegression(C=1.0). Was 0.02 as a RAW coefficient,
                // which made the effective strength ~34x too strong and
                // collapsed every retrained class — see RetrainCfg::l2_reg.
                l2_reg: 1.0,
                max_patches_per_crop: 8,
                // Reuses whatever folder Mining (above) already points at,
                // if any — no separate setting needed, and it stays purely
                // informational (logged, never blocks).
                validate_healthy_dir: self.mine_healthy_dir.clone(),
                validate_tau: self.mine_tau,
                cold_start: self.retrain_cold_start,
                dump_dir: self.retrain_dump.then(|| out.join("retrain_diag")),
                base_set: self.retrain_base_set.clone(),
                base_rows,
                anchor: self.retrain_anchor,
            },
            tx,
            self.retrain_cancel.clone(),
        );
    }

    fn poll_retrain(&mut self, toasts: &mut ToastManager) {
        let mut done = false;
        if let Some(rx) = &self.retrain_rx {
            for msg in rx.try_iter().take(64) {
                match msg {
                    RetrainMsg::Stage(s) => self.retrain_stage = s,
                    RetrainMsg::Log(l) => self.retrain_log.push(l),
                    RetrainMsg::Error(e) => {
                        self.retrain_log.push(format!("ERROR: {e}"));
                        toasts.error(format!("Retrain failed: {e}"));
                        done = true;
                    }
                    RetrainMsg::Done(s) => {
                        self.retrain_log.push(s.clone());
                        // out_path was computed in start_retrain from eff_head()'s
                        // parent — recompute the same way rather than parsing it
                        // back out of the summary string.
                        if let Some(head) = self.eff_head() {
                            self.retrain_done = Some(
                                head.parent().map(|p| p.join("fewshot_head_retrained.json"))
                                    .unwrap_or_else(|| PathBuf::from("fewshot_head_retrained.json"))
                            );
                        }
                        toasts.success("Retrain finished — new head ready to review.");
                        done = true;
                    }
                }
            }
        }
        if done {
            self.retraining = false;
            self.retrain_rx = None;
        }
    }

    // ── hard-negative mining (automated, patch-level) ───────────────────────

    fn show_mine_hardneg(&mut self, ui: &mut Ui) {
        ui_kit::section_header(ui, "Mine hard negatives");
        ui.label(RichText::new(
            "Scan a folder of KNOWN-HEALTHY tiles for patches the current head \
             wrongly calls defect, and stamp them as new hard-negative curations \
             — the automated, patch-level counterpart to manually stamping with \
             the Hardneg tool above.")
            .small().color(Color32::GRAY));
        self.pick_row(ui, "Healthy tiles folder", Pick::MineHealthyDir);
        ui.horizontal(|ui| {
            ui.label(RichText::new("τ_mine").small())
                .on_hover_text("A patch is mined when defect_prob ≥ this value \
                                 — matches the original Python pipeline's \
                                 tau_mine default of 0.6.");
            ui.add(egui::Slider::new(&mut self.mine_tau, 0.3..=0.95).fixed_decimals(2));
        });
        ui.horizontal(|ui| {
            ui.label(RichText::new("Cap").small())
                .on_hover_text("Max hard negatives this run can add. Every one \
                                 mined permanently adds to curations/labels.jsonl \
                                 and gets re-featurized on EVERY future Retrain \
                                 — keep this modest.");
            ui.add(egui::DragValue::new(&mut self.mine_max).range(50..=2000).speed(10));
        });
        let can_mine = self.output_folder.is_some() && self.eff_head().is_some()
            && self.eff_dino().is_some() && self.mine_healthy_dir.is_some()
            && !self.running && !self.retraining && !self.mining;
        ui.horizontal(|ui| {
            ui.add_enabled_ui(can_mine, |ui| {
                if ui.button("⛏ Mine hard negatives").clicked() {
                    self.start_mine();
                }
            });
            if self.mining {
                ui_kit::busy(ui, &format!("mining… {} found", self.mine_found));
                if ui.small_button("Cancel")
                    .on_hover_text("Stop mining. Hard negatives already written to the curation \
                                    set are kept — they are appended as they are found.")
                    .clicked()
                {
                    self.mine_cancel.store(true, Ordering::Relaxed);
                }
            }
        });
        if self.mining && self.mine_progress_total > 0 {
            let frac = self.mine_progress_done as f32 / self.mine_progress_total as f32;
            ui.add(egui::ProgressBar::new(frac).show_percentage());
        }
        if !self.mine_log.is_empty() {
            egui::ScrollArea::vertical().max_height(80.0).id_salt("pipeline_mine_log").show(ui, |ui| {
                for line in self.mine_log.iter().rev().take(20) {
                    ui.label(RichText::new(line).small());
                }
            });
        }

        ui.add_space(6.0);
        ui.separator();
        ui.label(RichText::new(
            "Or: mine the area of every fully-reviewed leaf already loaded \
             here that ISN'T covered by any anomaly region — no separate \
             healthy-tile folder needed, since curating this batch already \
             tells you what's healthy. \"Fully reviewed\" = every region \
             detected on that leaf has been confirmed or rejected, nothing \
             left pending.")
            .small().color(Color32::GRAY));
        let n_reviewed = self.count_fully_reviewed_leaves();
        ui.label(RichText::new(format!("{n_reviewed} fully-reviewed leaf(ves) eligible")).small().color(Color32::GRAY));
        let can_mine_unmarked = self.output_folder.is_some() && self.eff_head().is_some()
            && self.eff_dino().is_some() && n_reviewed > 0
            && !self.running && !self.retraining && !self.mining;
        ui.add_enabled_ui(can_mine_unmarked, |ui| {
            if ui.button("⛏ Mine unmarked leaf area").clicked() {
                self.start_mine_unmarked();
            }
        });
    }

    /// A leaf counts as eligible once every region detected on it has been
    /// acted on (confirmed or rejected) — a leaf the detector never flagged
    /// at all is skipped too, not because it's unsafe, but because a
    /// tau-gated scan of it is expected to find nothing (the same head-ish
    /// scoring already said "no" there once) and would just cost a wasted
    /// DINO pass.
    fn count_fully_reviewed_leaves(&self) -> usize {
        (0..self.results.len()).filter(|&li| {
            if self.rejected_leaves.contains(&li) {
                return false; // thrown out, not reviewed — never a mining candidate
            }
            let mut any = false;
            let mut all_done = true;
            for (i, r) in self.regions.iter().enumerate() {
                if r.leaf != li { continue; }
                any = true;
                if self.region_visible(i) && !self.persisted.contains(&i) { all_done = false; }
            }
            any && all_done
        }).count()
    }

    /// Builds one `LeafMineInput` per fully-reviewed leaf: `marked` is every
    /// region ever detected there (visible OR removed/rejected — a rejected
    /// region already has its own `"source":"reject"` crop, re-mining it here
    /// would just duplicate it) OR'd onto a leaf-sized canvas, EXCEPT
    /// merged-away entries, whose area the surviving merged region's own
    /// mask already covers.
    fn build_unmarked_mine_inputs(&self) -> Vec<LeafMineInput> {
        let mut out = Vec::new();
        for (leaf_idx, leaf) in self.results.iter().enumerate() {
            // A rejected leaf must never be mined. `region_visible` already hides
            // all its regions, which makes the fully_reviewed test below pass
            // trivially — so without this guard the miner would treat every
            // unmarked pixel of a leaf the user threw out as healthy tissue and
            // write it straight into the training set as a hard negative.
            if self.rejected_leaves.contains(&leaf_idx) { continue; }
            let leaf_regions: Vec<usize> = self.regions.iter().enumerate()
                .filter(|(_, r)| r.leaf == leaf_idx).map(|(i, _)| i).collect();
            if leaf_regions.is_empty() { continue; }
            let fully_reviewed = leaf_regions.iter()
                .all(|&i| !self.region_visible(i) || self.persisted.contains(&i));
            if !fully_reviewed { continue; }

            let (w, h) = (leaf.w as usize, leaf.h as usize);
            let mut marked = vec![false; w * h];
            for &i in &leaf_regions {
                if self.merged_away.contains(&i) { continue; }
                let r = &self.regions[i];
                let [bx, by, bw, bh] = r.bbox_leaf;
                for yy in 0..bh {
                    for xx in 0..bw {
                        if !r.mask[(yy * bw + xx) as usize] { continue; }
                        let (gx, gy) = ((bx + xx) as usize, (by + yy) as usize);
                        if gx < w && gy < h { marked[gy * w + gx] = true; }
                    }
                }
            }
            let Some(rgba) = image::RgbaImage::from_raw(leaf.w, leaf.h, leaf.rgba.clone()) else { continue };
            out.push(LeafMineInput { leaf_idx, src: leaf.src.clone(), rgba, marked });
        }
        out
    }

    fn start_mine_unmarked(&mut self) {
        let (Some(out), Some(head), Some(dino)) = (
            self.output_folder.clone(), self.eff_head(), self.eff_dino(),
        ) else { return };
        let leaves = self.build_unmarked_mine_inputs();
        if leaves.is_empty() { return; }
        self.mine_log.clear();
        self.mine_cancel = Arc::new(AtomicBool::new(false));
        self.mining = true;
        self.mine_progress_done = 0;
        self.mine_progress_total = 0;
        self.mine_found = 0;
        let (tx, rx) = mpsc::channel();
        self.mine_rx = Some(rx);
        spawn_mine_unmarked(
            leaves,
            MineUnmarkedConfig {
                head_path: head,
                dino_model: dino,
                curations_dir: out.join("curations"),
                tau_mine: self.mine_tau,
                hardneg_tile: self.hardneg_tile,
                max_hardneg: self.mine_max,
            },
            tx,
            self.mine_cancel.clone(),
        );
    }

    fn start_mine(&mut self) {
        let (Some(out), Some(head), Some(dino), Some(healthy_dir)) = (
            self.output_folder.clone(), self.eff_head(), self.eff_dino(), self.mine_healthy_dir.clone(),
        ) else { return };
        self.mine_log.clear();
        self.mine_cancel = Arc::new(AtomicBool::new(false));
        self.mining = true;
        self.mine_progress_done = 0;
        self.mine_progress_total = 0;
        self.mine_found = 0;
        let (tx, rx) = mpsc::channel();
        self.mine_rx = Some(rx);
        spawn_mine(
            MineConfig {
                healthy_dir,
                head_path: head,
                dino_model: dino,
                curations_dir: out.join("curations"), // matches start_retrain's own convention
                tau_mine: self.mine_tau,
                hardneg_tile: self.hardneg_tile,
                max_hardneg: self.mine_max,
            },
            tx,
            self.mine_cancel.clone(),
        );
    }

    fn poll_mine(&mut self, toasts: &mut ToastManager) {
        let mut done = false;
        if let Some(rx) = &self.mine_rx {
            for msg in rx.try_iter().take(64) {
                match msg {
                    MineMsg::Progress { done: d, total } => {
                        self.mine_progress_done = d;
                        self.mine_progress_total = total;
                    }
                    MineMsg::Found { n_so_far } => self.mine_found = n_so_far,
                    MineMsg::Log(l) => self.mine_log.push(l),
                    MineMsg::Error(e) => {
                        self.mine_log.push(format!("ERROR: {e}"));
                        toasts.error(format!("Mining failed: {e}"));
                        done = true;
                    }
                    MineMsg::Done(s) => {
                        self.mine_log.push(s);
                        toasts.success(format!("Mining done — {} hard negative(s) found.", self.mine_found));
                        done = true;
                    }
                }
            }
        }
        if done {
            self.mining = false;
            self.mine_rx = None;
        }
    }

    /// Flywheel, single region: write its crop + a `labels.jsonl` line
    /// IMMEDIATELY (an upsert — any existing line for this region's crop
    /// filename is dropped first, then the new line appended) and mark it
    /// `persisted`. Filenames are STABLE per region index (`region_{idx}.png`,
    /// no timestamp) specifically so re-persisting after a change of mind
    /// (Reject → Undo → Confirm, or a cluster rename) overwrites in place
    /// instead of leaving an orphaned duplicate — `train::head::retrain` has
    /// no dedup-by-crop, so two stale lines for the same crop would silently
    /// become two contradictory training examples. Called by every curation
    /// action below (Confirm, Reject, Reassign, the bulk accelerator, and
    /// cluster-rename propagation) — this is the ONLY place that writes to
    /// `curations/`, so every action persists the same way, immediately.
    fn persist_region(&mut self, idx: usize, family: &str, is_reject: bool, toasts: &mut ToastManager) {
        let Some(out) = self.output_folder.clone() else {
            toasts.error("Set an output folder first.");
            return;
        };
        let Some(r) = self.regions.get(idx) else { return };
        let labels_dir = out.join("curations").join("labels");
        if let Err(e) = std::fs::create_dir_all(&labels_dir) {
            toasts.error(format!("curations dir: {e}"));
            return;
        }
        // A region bigger than one tile only ever had ONE fixed-size crop
        // saved, centered on its centroid — for a big region that silently
        // discards most of what was actually marked (confirmed on a real
        // example: ~80% missing). Cover the full extent with a grid of
        // tiles instead; small regions (the common case) get exactly
        // today's single crop, unchanged.
        const TILE: u32 = 128;
        let tiles = match self.results.get(r.leaf) {
            Some(l) => build_region_tiles(r, &l.rgba, l.w, l.h, TILE),
            None => vec![(r.crop.clone(), r.crop_size, build_crop_mask_png(r))],
        };
        let src = self.results.get(r.leaf).map(|l| l.src.display().to_string()).unwrap_or_default();
        let run = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let jsonl_path = out.join("curations").join("labels.jsonl");

        // upsert: drop ALL previous lines for this region idx — a
        // single-crop "region_{idx}.png" OR any number of previous
        // "region_{idx}_tileN.png" — a re-persist can change tile count,
        // so an exact-filename match isn't enough anymore. The two needles
        // cover the only two possible next characters after the bare index
        // (`.` for the single-crop name, `_` for a tile name), so
        // `region_46` can never collide with `region_463`.
        let needle_dot = format!("\"crop\":\"region_{idx}.");
        let needle_us = format!("\"crop\":\"region_{idx}_");
        if let Ok(text) = std::fs::read_to_string(&jsonl_path) {
            let kept: String = text.lines()
                .filter(|l| !l.contains(&needle_dot) && !l.contains(&needle_us))
                .map(|l| format!("{l}\n")).collect();
            let _ = std::fs::write(&jsonl_path, kept);
        }

        let n_tiles = tiles.len();
        let mut lines = String::new();
        for (t, (crop_bytes, crop_size, mask_png)) in tiles.into_iter().enumerate() {
            let stem = if n_tiles == 1 { format!("region_{idx}") } else { format!("region_{idx}_tile{t}") };
            let fname = format!("{stem}.png");
            // Track the crop write instead of discarding it: `persisted.insert`
            // below used to run unconditionally, so a full disk or a read-only
            // share marked the region confirmed in the UI with nothing on disk.
            // That is the worst possible failure for a curation set — it is
            // discovered months later, if ever.
            if let Some(img) = image::RgbaImage::from_raw(crop_size, crop_size, crop_bytes) {
                if let Err(e) = img.save(labels_dir.join(&fname)) {
                    toasts.error(format!("could not write crop {fname}: {e}"));
                    return;
                }
            }
            let mask_fname = format!("{stem}_mask.png");
            let mask_saved = mask_png.map(|m| m.save(labels_dir.join(&mask_fname)).is_ok()).unwrap_or(false);
            lines.push_str(&format!(
                "{{\"crop\":\"{}\",\"mask\":\"{}\",\"family\":\"{}\",\"source\":\"{}\",\"leaf_src\":\"{}\",\"ts\":{}}}\n",
                fname, if mask_saved { mask_fname.as_str() } else { "" }, json_escape(family),
                if is_reject { "reject" } else { "confirm" }, json_escape(&src), run,
            ));
        }

        use std::io::Write;
        // `persisted` is only marked once the bytes are actually down. It is the
        // set that drives "fully reviewed", which gates mining — so a row claimed
        // as written but absent would feed the miner a leaf it has no labels for.
        match std::fs::OpenOptions::new().create(true).append(true).open(&jsonl_path) {
            Ok(mut f) => match f.write_all(lines.as_bytes()) {
                Ok(()) => { self.persisted.insert(idx); }
                Err(e) => toasts.error(format!("labels.jsonl write failed: {e}")),
            },
            Err(e) => toasts.error(format!("labels.jsonl: {e}")),
        }
    }

    /// Explicit "I reviewed this and it's correct" gesture — persists each
    /// region's CURRENT cluster name immediately. Unlike Reject, this never
    /// touches `labels`/`removed` (a confirmed region's family is just
    /// whatever it already is); it only catches disk state up to what's shown.
    fn confirm_regions(&mut self, ids: &[usize], toasts: &mut ToastManager) {
        // Record only what this call actually creates, so undo takes back exactly
        // what it wrote — never a curation that already existed.
        let mut newly: Vec<usize> = Vec::new();
        for &i in ids {
            if !self.region_visible(i) { continue; }
            let cid = self.labels[i];
            if cid < 0 { continue; } // nothing meaningful to confirm yet
            let was = self.persisted.contains(&i);
            let family = self.cluster_names.get(&cid).cloned().unwrap_or_else(|| format!("Cluster {cid}"));
            self.persist_region(i, &family, false, toasts);
            // persist_region only inserts on a successful write, so this also
            // means a failed write leaves nothing on the undo stack to retract.
            if !was && self.persisted.contains(&i) {
                newly.push(i);
            }
        }
        if !newly.is_empty() {
            self.push_undo(UndoEntry::Confirm(newly));
        }
    }

    /// Bulk accelerator (was "Save curations for retraining"): confirm every
    /// region NOT yet individually touched this run — for the common case
    /// where the auto-clustering was already correct and the user just wants
    /// to fast-approve everything they didn't individually review. Everything
    /// ELSE (Confirm/Reject/Reassign) already persisted the instant it
    /// happened, so this can never lose anything — it's a convenience, not a
    /// save gate.
    fn confirm_all_remaining(&mut self, toasts: &mut ToastManager) {
        if self.regions.is_empty() {
            toasts.error("Nothing to confirm — run + curate first.");
            return;
        }
        let ids: Vec<usize> = (0..self.regions.len())
            .filter(|&i| !self.persisted.contains(&i) && self.labels.get(i).copied().unwrap_or(-1) >= 0
                && self.region_visible(i))
            .collect();
        let n = ids.len();
        self.confirm_regions(&ids, toasts);
        toasts.success(format!("Confirmed {n} remaining region(s) -> curations/"));
    }

    /// Flywheel, drag capture: crop the exact leaf-pixel rect `[lx0,ly0]..[lx1,ly1]`
    /// dragged on the canvas for `leaf_idx` and append it to `<output>/curations/` in
    /// the SAME format `persist_region` writes, so `train::head::retrain` consumes it
    /// identically. An empty/"healthy"/"reject" label maps to the reject convention
    /// (class 0); any other typed name is a normal positive family label (new or
    /// existing). This is the fast path for hard negatives the detector never even
    /// flagged as a region.
    /// Tile-picker-style stamp: crop the fixed `hardneg_tile`×`hardneg_tile` square
    /// at leaf-pixel top-left `(x, y)` (may overhang the leaf's own bounds — those
    /// pixels save transparent, same convention as Tile Picker) and append it to
    /// `<output>/curations/` in the SAME format `persist_region` writes, so
    /// `train::head::retrain` consumes it identically. An empty/"healthy"/"reject"
    /// label maps to the reject convention (class 0); any other typed name is a
    /// normal positive family label.
    fn stamp_hardneg(&mut self, leaf_idx: usize, x: i32, y: i32, toasts: &mut ToastManager) {
        let Some(out) = self.output_folder.clone() else {
            toasts.error("Set an output folder first.");
            return;
        };
        let Some(leaf) = self.results.get(leaf_idx) else { return };
        let (lw, lh) = (leaf.w as i32, leaf.h as i32);
        let tu = self.hardneg_tile;
        let t = tu as i32;
        let mut buf = vec![0u8; (tu * tu * 4) as usize];
        // Alpha-valid mask, same window — so `crop_feature`'s mask-aware
        // pooling (already used for confirmed/rejected regions) excludes
        // any transparent padding a stamp placed near the leaf's edge would
        // otherwise silently blend into "Healthy" training signal, matching
        // the same fix applied to both mining paths (hardneg_mining.rs).
        let mut mask_buf = vec![0u8; (tu * tu) as usize];
        for row in 0..t {
            let sy = y + row;
            if sy < 0 || sy >= lh {
                continue;
            }
            for col in 0..t {
                let sx = x + col;
                if sx < 0 || sx >= lw {
                    continue;
                }
                let si = ((sy * lw + sx) * 4) as usize;
                let di = ((row * t + col) * 4) as usize;
                buf[di..di + 4].copy_from_slice(&leaf.rgba[si..si + 4]);
                if leaf.rgba[si + 3] > 10 {
                    mask_buf[(row * t + col) as usize] = 255;
                }
            }
        }

        let labels_dir = out.join("curations").join("labels");
        if let Err(e) = std::fs::create_dir_all(&labels_dir) {
            toasts.error(format!("curations dir: {e}"));
            return;
        }
        let label = self.hardneg_label.trim();
        let is_reject = label.is_empty()
            || label.eq_ignore_ascii_case("healthy")
            || label.eq_ignore_ascii_case("reject")
            || label.eq_ignore_ascii_case("rejected");
        let family = if is_reject { "rejected".to_string() } else { label.to_string() };
        let run = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let fname = format!("{run}_hardneg_{leaf_idx}_{x}_{y}.png");
        let file = labels_dir.join(&fname);
        let Some(img) = image::RgbaImage::from_raw(tu, tu, buf) else { return };
        if let Err(e) = img.save(&file) {
            toasts.error(format!("save crop: {e}"));
            return;
        }
        let mask_fname = format!("{run}_hardneg_{leaf_idx}_{x}_{y}_mask.png");
        if let Some(mask_img) = image::GrayImage::from_raw(tu, tu, mask_buf) {
            let _ = mask_img.save(labels_dir.join(&mask_fname));
        }
        let src = leaf.src.display().to_string();
        let line = format!(
            "{{\"crop\":\"{}\",\"mask\":\"{}\",\"family\":\"{}\",\"source\":\"{}\",\"leaf_src\":\"{}\",\"ts\":{}}}\n",
            fname, mask_fname, json_escape(&family), if is_reject { "reject" } else { "manual" },
            json_escape(&src), run,
        );
        use std::io::Write;
        match std::fs::OpenOptions::new().create(true).append(true)
            .open(out.join("curations").join("labels.jsonl"))
        {
            Ok(mut f) => {
                let _ = f.write_all(line.as_bytes());
                self.hardneg_stamps.entry(leaf_idx).or_default().push(HardnegStamp { x, y, file });
                toasts.success(format!("Stamped \"{family}\""));
            }
            Err(e) => toasts.error(format!("labels.jsonl: {e}")),
        }
    }

    /// Remove the topmost stamp on `leaf_idx` containing leaf-pixel `(lx, ly)`.
    fn remove_hardneg_at(&mut self, leaf_idx: usize, lx: f32, ly: f32) {
        let t = self.hardneg_tile as f32;
        if let Some(list) = self.hardneg_stamps.get_mut(&leaf_idx) {
            if let Some(pos) = list.iter().rposition(|st| {
                lx >= st.x as f32 && lx < st.x as f32 + t && ly >= st.y as f32 && ly < st.y as f32 + t
            }) {
                let st = list.remove(pos);
                self.retract_hardneg(&st);
            }
        }
    }

    /// Undo the most recent stamp on `leaf_idx`.
    fn undo_hardneg(&mut self, leaf_idx: usize) {
        if let Some(list) = self.hardneg_stamps.get_mut(&leaf_idx) {
            if let Some(st) = list.pop() {
                self.retract_hardneg(&st);
            }
        }
    }

    /// Delete a stamp's crop file AND its `labels.jsonl` line (filter-rewrite).
    fn retract_hardneg(&self, st: &HardnegStamp) {
        let _ = std::fs::remove_file(&st.file);
        let Some(out) = self.output_folder.clone() else { return };
        let Some(name) = st.file.file_name().map(|f| f.to_string_lossy().to_string()) else { return };
        let jsonl_path = out.join("curations").join("labels.jsonl");
        let Ok(text) = std::fs::read_to_string(&jsonl_path) else { return };
        let needle = format!("\"crop\":\"{name}\"");
        let kept: String = text.lines().filter(|l| !l.contains(&needle)).map(|l| format!("{l}\n")).collect();
        let _ = std::fs::write(&jsonl_path, kept);
    }

    /// Export everything to `<output>/export/`: a SINGLE combined `results.csv`
    /// (one row per anomaly = cluster + region stats + Recon % + the leaf's
    /// morphology), `crops/` (each anomaly image) and `leaves/` (each leaf with
    /// anomalies colour-coded by family).
    fn export_results(&mut self, toasts: &mut ToastManager) {
        let Some(out) = self.output_folder.clone() else {
            toasts.error("Set an output folder first.");
            return;
        };
        if self.regions.is_empty() {
            toasts.error("Nothing to export — run the pipeline first.");
            return;
        }
        let dir = out.join("export");
        let crops_dir = dir.join("crops");
        let leaves_dir = dir.join("leaves");
        // Only create crops/ when it will actually be filled — an empty crops/
        // folder next to the results reads as "the crops failed", not "crops
        // were deliberately skipped".
        let dirs_ok = std::fs::create_dir_all(&dir).is_ok()
            && (!self.export_overlays || std::fs::create_dir_all(&leaves_dir).is_ok())
            && (!self.export_crops || std::fs::create_dir_all(&crops_dir).is_ok());
        if !dirs_ok {
            toasts.error("could not create export folder");
            return;
        }

        // `leaf` is a 0-based index into this run's leaves and says nothing about
        // which photograph a leaf came from — with several leaves per photo,
        // "leaf 7" is unresolvable without grouping by leaf_src by hand. Three
        // columns fix that:
        //
        //   image          the photograph's file stem
        //   leaf_in_image  1-based leaf number WITHIN that photograph
        //   leaf_file      the cut-out's own filename stem, so a row joins
        //                  directly to the PNG that `segment_one` wrote
        //                  (`{stem}_leaf{NN}.png`, 0-based to match the file)
        //
        // `leaf` itself keeps its meaning so existing scripts still work.
        let mut csv = String::from(
            "leaf,image,leaf_in_image,leaf_file,leaf_src,region,cluster_id,family,\
             area_px,pct_leaf,recon_pct,lost_tissue_pct,\
             bbox_x,bbox_y,bbox_w,bbox_h,crop_file,\
             ec_length,ec_width,ec_area,ec_shape_index,ec_circularity,ec_entropy,ec_outline,\
             mc_length,mc_width,mc_area,mc_shape_index,mc_circularity,mc_entropy,mc_outline\n",
        );
        // Ordinal of each leaf within its own photograph, computed once. Same
        // definition the review-state file uses, so the two agree.
        let mut ordinal: Vec<usize> = Vec::with_capacity(self.results.len());
        {
            let mut seen: HashMap<&std::path::Path, usize> = HashMap::new();
            for l in &self.results {
                let e = seen.entry(l.src.as_path()).or_insert(0);
                ordinal.push(*e);
                *e += 1;
            }
        }
        let mut n = 0usize;
        let mut pending_crops: Vec<(usize, String)> = Vec::new();
        // Per-(leaf, family) tallies for the WIDE format. Always accumulated —
        // it costs one hash insert per region and keeps the two formats reading
        // the exact same filtered set of regions, so the numbers cannot diverge.
        let mut agg: HashMap<(usize, i32), (usize, u64)> = HashMap::new();
        for (i, r) in self.regions.iter().enumerate() {
            if !self.region_visible(i) {
                continue;
            }
            {
                let e = agg.entry((r.leaf, self.labels[i])).or_insert((0, 0));
                e.0 += 1;
                e.1 += self.region_area.get(i).copied().unwrap_or(0) as u64;
            }
            let cid = self.labels[i];
            let fam = self.cluster_names.get(&cid).cloned().unwrap_or_else(|| format!("Cluster {cid}"));
            let leaf = r.leaf;
            let area = self.region_area.get(i).copied().unwrap_or(0);
            let leaf_px = self.leaf_valid_px.get(leaf).copied().unwrap_or(1).max(1);
            let l = self.results.get(leaf);
            let src = l.map(|l| l.src.display().to_string()).unwrap_or_default();
            let recon_whole = l.map(|l| l.recon_whole).unwrap_or(0);
            let recon_pct = if recon_whole > 0 {
                format!("{:.3}", 100.0 * area as f32 / recon_whole as f32)
            } else {
                String::new()
            };
            let lost_pct = if recon_whole > 0 {
                format!("{:.3}", 100.0 * l.map(|l| l.recon_area).unwrap_or(0) as f32 / recon_whole as f32)
            } else {
                String::new()
            };
            let [bx, by, bw, bh] = r.bbox_leaf;
            // The filename stays in the CSV even when the PNG isn't written: it
            // identifies the region, and blanking the column would break any
            // downstream script that reads it. A later export with crops on
            // produces exactly these names.
            let crop_file = format!("{leaf}_{i}.png");
            if self.export_crops {
                pending_crops.push((i, crop_file.clone()));
            }
            // Identity of the leaf within its photograph.
            let image_stem = l.and_then(|l| l.src.file_stem())
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let nth = ordinal.get(leaf).copied().unwrap_or(0);
            // `{stem}_leaf{NN}` matches the cut-out PNG segment_one writes, so a
            // CSV row can be joined straight to its image.
            let leaf_file = format!("{image_stem}_leaf{nth:02}");
            let mut cols = vec![
                leaf.to_string(),
                csv_escape(&image_stem),
                (nth + 1).to_string(), // 1-based: "the 2nd leaf in this photo"
                csv_escape(&leaf_file),
                csv_escape(&src), i.to_string(), cid.to_string(), csv_escape(&fam),
                area.to_string(), format!("{:.3}", 100.0 * area as f32 / leaf_px as f32), recon_pct, lost_pct,
                bx.to_string(), by.to_string(), bw.to_string(), bh.to_string(), crop_file,
            ];
            match l.and_then(|l| l.morph.as_ref()) {
                Some(m) => cols.extend([
                    format!("{:.2}", m.ec_length), format!("{:.2}", m.ec_width), m.ec_area.to_string(),
                    format!("{:.4}", m.ec_shape_index), format!("{:.4}", m.ec_circularity),
                    format!("{:.5}", m.ec_approximate_entropy), m.ec_outline_count.to_string(),
                    format!("{:.2}", m.mc_length), format!("{:.2}", m.mc_width), m.mc_area.to_string(),
                    format!("{:.4}", m.mc_shape_index), format!("{:.4}", m.mc_circularity),
                    format!("{:.5}", m.mc_spectral_entropy), m.mc_outline_count.to_string(),
                ]),
                None => cols.extend(std::iter::repeat(String::new()).take(14)),
            }
            if !self.export_wide {
                csv.push_str(&cols.join(","));
                csv.push('\n');
            }
            n += 1;
        }

        // ── wide format: ONE row per leaf ──────────────────────────────────
        // The long format answers "tell me about this region"; the wide one
        // answers "tell me about this leaf", which is the shape most statistics
        // want — a leaf is the sampling unit, and a long file has to be pivoted
        // before it can be joined to anything measured per leaf.
        if self.export_wide {
            // A fixed family column order, so every row lines up and two runs
            // over the same families produce comparable files.
            let mut fam_ids: Vec<i32> =
                self.clusters.iter().map(|c| c.id).filter(|&id| id >= 0).collect();
            fam_ids.sort_unstable();
            let fam_names: Vec<String> = fam_ids.iter()
                .map(|&id| self.cluster_names.get(&id).cloned()
                    .unwrap_or_else(|| format!("Cluster {id}")))
                .collect();

            let mut head = String::from(
                "leaf,image,leaf_in_image,leaf_file,leaf_src,leaf_area_px,n_anomalies,\
                 anomaly_area_px,anomaly_pct_leaf,lost_tissue_pct,\
                 ec_length,ec_width,ec_area,ec_shape_index,ec_circularity,ec_entropy,ec_outline,\
                 mc_length,mc_width,mc_area,mc_shape_index,mc_circularity,mc_entropy,mc_outline",
            );
            for name in &fam_names {
                let slug = csv_header_slug(name);
                // count / total / mean / share, per family. The mean alone is not
                // interpretable without the count behind it, and the total is what
                // sums back to anomaly_area_px.
                head.push_str(&format!(
                    ",{slug}_count,{slug}_area_px,{slug}_avg_area_px,{slug}_pct_leaf"));
            }
            head.push('\n');

            let mut wide = head;
            let mut rows = 0usize;
            for (leaf, l) in self.results.iter().enumerate() {
                // Rejected leaves contribute no visible regions, so a row for one
                // would be all zeros — indistinguishable from a genuinely clean
                // leaf. Same set the overlay export writes.
                if self.rejected_leaves.contains(&leaf) {
                    continue;
                }
                let leaf_px = self.leaf_valid_px.get(leaf).copied().unwrap_or(1).max(1);
                let image_stem = l.src.file_stem()
                    .map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                let nth = ordinal.get(leaf).copied().unwrap_or(0);
                let total_n: usize = fam_ids.iter()
                    .map(|&id| agg.get(&(leaf, id)).map_or(0, |a| a.0)).sum();
                let total_a: u64 = fam_ids.iter()
                    .map(|&id| agg.get(&(leaf, id)).map_or(0, |a| a.1)).sum();
                let lost_pct = if l.recon_whole > 0 {
                    format!("{:.3}", 100.0 * l.recon_area as f32 / l.recon_whole as f32)
                } else {
                    String::new()
                };
                let mut cols = vec![
                    leaf.to_string(),
                    csv_escape(&image_stem),
                    (nth + 1).to_string(),
                    csv_escape(&format!("{image_stem}_leaf{nth:02}")),
                    csv_escape(&l.src.display().to_string()),
                    leaf_px.to_string(),
                    total_n.to_string(),
                    total_a.to_string(),
                    format!("{:.3}", 100.0 * total_a as f32 / leaf_px as f32),
                    lost_pct,
                ];
                match l.morph.as_ref() {
                    Some(m) => cols.extend([
                        format!("{:.2}", m.ec_length), format!("{:.2}", m.ec_width), m.ec_area.to_string(),
                        format!("{:.4}", m.ec_shape_index), format!("{:.4}", m.ec_circularity),
                        format!("{:.5}", m.ec_approximate_entropy), m.ec_outline_count.to_string(),
                        format!("{:.2}", m.mc_length), format!("{:.2}", m.mc_width), m.mc_area.to_string(),
                        format!("{:.4}", m.mc_shape_index), format!("{:.4}", m.mc_circularity),
                        format!("{:.5}", m.mc_spectral_entropy), m.mc_outline_count.to_string(),
                    ]),
                    None => cols.extend(std::iter::repeat(String::new()).take(14)),
                }
                for &id in &fam_ids {
                    let (c, a) = agg.get(&(leaf, id)).copied().unwrap_or((0, 0));
                    cols.push(c.to_string());
                    cols.push(a.to_string());
                    // Mean of ZERO regions is undefined, not 0 — an empty cell
                    // keeps it out of a downstream mean instead of dragging it
                    // toward zero.
                    cols.push(if c > 0 { format!("{:.1}", a as f64 / c as f64) } else { String::new() });
                    cols.push(format!("{:.3}", 100.0 * a as f32 / leaf_px as f32));
                }
                wide.push_str(&cols.join(","));
                wide.push('\n');
                rows += 1;
            }
            csv = wide;
            n = rows;
        }

        // The CSV goes down NOW, in full. It is the actual result; the images are
        // an aid. So a cancelled or interrupted export still leaves a complete,
        // valid results.csv rather than a truncated one.
        if let Err(e) = std::fs::write(dir.join("results.csv"), csv) {
            toasts.error(format!("write results.csv: {e}"));
            return;
        }

        // Images are queued and written a chunk per frame — see `ExportJob`.
        let leaves: Vec<usize> = if self.export_overlays {
            (0..self.results.len()).filter(|li| !self.rejected_leaves.contains(li)).collect()
        } else {
            Vec::new()
        };
        let total = pending_crops.len() + leaves.len();
        if total == 0 {
            toasts.success(format!("Exported {n} anomalies -> export/results.csv"));
            return;
        }
        self.export_job = Some(ExportJob {
            crops_dir, leaves_dir,
            crops: pending_crops,
            leaves,
            crop_cur: 0,
            leaf_cur: 0,
            written: 0,
            failed: 0,
            total,
        });
        toasts.info(format!("results.csv written · {total} images queued"));
    }

    /// Write one bounded slice of the queued export. Called once per frame.
    ///
    /// Chunk sizes are deliberately lopsided: a crop is a small thumbnail, an
    /// overlay is a full-resolution composite plus PNG encode. Two overlays per
    /// frame keeps a 60 Hz UI comfortably interactive and still clears 10,000
    /// leaves in a minute or so.
    fn step_export(&mut self, toasts: &mut ToastManager) {
        const CROPS_PER_FRAME:  usize = 24;
        const LEAVES_PER_FRAME: usize = 2;
        let Some(job) = self.export_job.take() else { return };
        let mut job = job;

        let end = (job.crop_cur + CROPS_PER_FRAME).min(job.crops.len());
        for k in job.crop_cur..end {
            let (ri, ref name) = job.crops[k];
            if let Some(r) = self.regions.get(ri) {
                if let Some(img) = image::RgbaImage::from_raw(r.crop_size, r.crop_size, r.crop.clone()) {
                    match img.save(job.crops_dir.join(name)) {
                        Ok(()) => job.written += 1,
                        Err(_) => job.failed += 1,
                    }
                }
            }
        }
        job.crop_cur = end;

        if job.crop_cur >= job.crops.len() {
            let end = (job.leaf_cur + LEAVES_PER_FRAME).min(job.leaves.len());
            for k in job.leaf_cur..end {
                let li = job.leaves[k];
                let Some(leaf) = self.results.get(li) else { continue };
                let (w, h) = (leaf.w as usize, leaf.h as usize);
                let mut px = leaf.rgba.clone();
                for (ri, r) in self.regions.iter().enumerate() {
                    if r.leaf == li && self.region_visible(ri) {
                        paint_region(&mut px, w, h, r, cluster_color(self.labels[ri]), 0.6);
                    }
                }
                let stem = leaf.src.file_stem().map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| format!("leaf_{li}"));
                if let Some(img) = image::RgbaImage::from_raw(leaf.w, leaf.h, px) {
                    match img.save(job.leaves_dir.join(format!("{stem}_{li}.png"))) {
                        Ok(()) => job.written += 1,
                        Err(_) => job.failed += 1,
                    }
                }
            }
            job.leaf_cur = end;
        }

        if job.crop_cur >= job.crops.len() && job.leaf_cur >= job.leaves.len() {
            // Report failures rather than swallowing them: the old code discarded
            // every image write result and then said "Exported … + images".
            if job.failed > 0 {
                toasts.warning(format!(
                    "Export finished — {} images written, {} failed (disk full or read-only?)",
                    job.written, job.failed,
                ));
            } else {
                toasts.success(format!("Export finished — {} images written", job.written));
            }
        } else {
            self.export_job = Some(job);
        }
    }

    fn build_clusters(&mut self, toasts: &mut ToastManager) {
        // standing invariant: touching same-cluster regions become one
        // region BEFORE we group by label, so ClusterInfo.members never
        // contains an index that a merge is about to hide.
        self.merge_touching_regions(toasts);

        let mut by_label: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &l) in self.labels.iter().enumerate() {
            if !self.region_visible(i) {
                continue;
            }
            by_label.entry(l).or_default().push(i);
        }
        let mut clusters: Vec<ClusterInfo> =
            by_label.into_iter().map(|(id, members)| ClusterInfo { id, members }).collect();
        // real clusters first (by size desc), noise (-1) last
        clusters.sort_by(|a, b| match (a.id < 0, b.id < 0) {
            (false, true) => std::cmp::Ordering::Less,
            (true, false) => std::cmp::Ordering::Greater,
            _ => b.members.len().cmp(&a.members.len()),
        });
        // Name any cluster id seen for the first time, resolving through the
        // LOADED HEAD before falling back to a placeholder.
        //
        // This used to stamp "Cluster N" unconditionally, which silently renamed
        // real classes and then wrote the placeholder to disk. The path: a class
        // the head knows (say Skeletonizer) has no detections this run, so it
        // never reaches `cluster_names` — but `cluster_picker_rows` still offers
        // it, because that reads the head. The moment the user assigns the FIRST
        // region to it, `build_clusters` runs, sees a brand-new cluster id, and
        // inserts "Cluster N". Every display site reads `cluster_names` first,
        // so the real name vanishes mid-session; worse, `persist_region` writes
        // that same string into labels.jsonl, so every subsequent curation is
        // saved under "Cluster N" instead of its family. `retrain` joins by
        // NAME, so those rows then allocate a DUPLICATE class rather than
        // training the intended one. Reported from the field exactly as "the
        // suggestion was suddenly gone and replaced by cluster 4".
        //
        // `class_display_name` already implements the right order (runtime name,
        // then head family, then placeholder); ids are collected first so it can
        // take &mut self without borrowing `clusters`.
        let new_ids: Vec<i32> = clusters.iter()
            .map(|c| c.id)
            .filter(|id| !self.cluster_names.contains_key(id))
            .collect();
        for id in new_ids {
            let name = if id < 0 { "noise".to_string() } else { self.class_display_name(id) };
            self.cluster_names.insert(id, name);
        }
        self.clusters = clusters;
    }

    /// Standing invariant: two regions of the SAME cluster on the SAME leaf
    /// that touch (8-connected — matches `detect::connected_components`'s own
    /// definition of "connected", not an arbitrary choice) are treated as one
    /// region. Without this, a blob that's visually continuous can end up
    /// fragmented into several `AnomalyRegion`s (from separate detector
    /// passes, a reassign, or a brush stroke) with split area stats and an
    /// Eyedropper readout that flips identity depending on which pixel you're
    /// over. Called at the START of `build_clusters` (before its label-
    /// grouping loop) so `ClusterInfo.members` is always post-merge.
    fn merge_touching_regions(&mut self, toasts: &mut ToastManager) {
        // Group by (leaf, label), VISIBLE regions only — grouping over
        // removed/merged_away regions would (a) silently resurrect rejected
        // content by unioning it back into a survivor, and (b) let a zombie
        // merged_away entry with a stale label bridge two unrelated real
        // clusters together on a later pass.
        let mut by_group: HashMap<(usize, i32), Vec<usize>> = HashMap::new();
        for i in 0..self.regions.len() {
            if !self.region_visible(i) {
                continue;
            }
            let key = (self.regions[i].leaf, self.labels[i]);
            by_group.entry(key).or_default().push(i);
        }

        for (_, members) in by_group {
            let n = members.len();
            if n < 2 {
                continue;
            }
            // union-find over the pairwise touching graph, transitively
            let mut parent: Vec<usize> = (0..n).collect();
            fn find(parent: &mut [usize], x: usize) -> usize {
                if parent[x] != x {
                    parent[x] = find(parent, parent[x]);
                }
                parent[x]
            }
            for a in 0..n {
                for b in (a + 1)..n {
                    let ra = &self.regions[members[a]];
                    let rb = &self.regions[members[b]];
                    if !bbox_proximate(ra.bbox_leaf, rb.bbox_leaf) {
                        continue;
                    }
                    if regions_touch(ra, rb) {
                        let (pa, pb) = (find(&mut parent, a), find(&mut parent, b));
                        if pa != pb {
                            parent[pa] = pb;
                        }
                    }
                }
            }
            let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
            for a in 0..n {
                let root = find(&mut parent, a);
                groups.entry(root).or_default().push(members[a]);
            }
            for group in groups.into_values() {
                if group.len() >= 2 {
                    self.merge_region_group(&group, toasts);
                }
            }
        }
    }

    /// Merge 2+ touching same-cluster regions into one survivor (lowest
    /// index, for determinism). Every other member becomes `merged_away`
    /// (never removed/renumbered — see the field's own doc comment). The
    /// survivor's bbox/mask/area/crop/thumbnail are all updated in place so
    /// nothing downstream shows stale geometry, and disk state is
    /// reconciled so `curations/` never ends up with orphaned or stale
    /// entries for either the absorbed members or the survivor.
    fn merge_region_group(&mut self, indices: &[usize], toasts: &mut ToastManager) {
        let mut sorted = indices.to_vec();
        sorted.sort_unstable();
        let survivor = sorted[0];
        let leaf = self.regions[survivor].leaf;

        let (mut min_x, mut min_y) = (u32::MAX, u32::MAX);
        let (mut max_x, mut max_y) = (0u32, 0u32);
        for &i in &sorted {
            let [bx, by, bw, bh] = self.regions[i].bbox_leaf;
            min_x = min_x.min(bx);
            min_y = min_y.min(by);
            max_x = max_x.max(bx + bw);
            max_y = max_y.max(by + bh);
        }
        let (uw, uh) = (max_x - min_x, max_y - min_y);
        let mut union_mask = vec![false; (uw * uh) as usize];
        for &i in &sorted {
            let r = &self.regions[i];
            let [bx, by, bw, bh] = r.bbox_leaf;
            for ly in 0..bh {
                for lx in 0..bw {
                    if r.mask[(ly * bw + lx) as usize] {
                        let (gx, gy) = (bx + lx - min_x, by + ly - min_y);
                        union_mask[(gy * uw + gx) as usize] = true;
                    }
                }
            }
        }
        let area = union_mask.iter().filter(|&&b| b).count() as u32;

        // centroid of the union mask, for the regenerated crop
        let (mut sx, mut sy, mut cnt) = (0u64, 0u64, 0u64);
        for gy in 0..uh {
            for gx in 0..uw {
                if union_mask[(gy * uw + gx) as usize] {
                    sx += gx as u64;
                    sy += gy as u64;
                    cnt += 1;
                }
            }
        }
        let (ccx, ccy) = if cnt > 0 {
            (min_x as f32 + sx as f32 / cnt as f32, min_y as f32 + sy as f32 / cnt as f32)
        } else {
            (min_x as f32, min_y as f32)
        };
        let crop_size = self.regions[survivor].crop_size;
        let new_crop = self.results.get(leaf)
            .map(|l| worker::context_crop(&l.rgba, l.w, l.h, ccx, ccy, crop_size));

        self.regions[survivor].bbox_leaf = [min_x, min_y, uw, uh];
        self.regions[survivor].mask = union_mask;
        if let Some(c) = new_crop {
            self.regions[survivor].crop = c;
        }
        if let Some(a) = self.region_area.get_mut(survivor) {
            *a = area;
        }
        if let Some(t) = self.region_thumbs.get_mut(survivor) {
            *t = None; // force ensure_region_thumb to reload from the new crop
        }

        let survivor_family = self.labels[survivor];
        let survivor_was_persisted = self.persisted.contains(&survivor);
        for &i in &sorted[1..] {
            if self.persisted.contains(&i) {
                self.retract_persisted(i);
            }
            self.merged_away.insert(i);
            if self.selected_region == Some(i) {
                self.selected_region = Some(survivor);
            }
            if self.multi_selected.remove(&i) {
                self.multi_selected.insert(survivor);
            }
        }
        if survivor_was_persisted {
            let name = self.cluster_names.get(&survivor_family).cloned()
                .unwrap_or_else(|| format!("Cluster {survivor_family}"));
            self.persist_region(survivor, &name, false, toasts);
        }
    }

    /// Resolve a completed brush stroke (accumulated leaf-pixel coords) into
    /// region geometry: build its bbox-local mask, exclude any pixels
    /// already owned by a DIFFERENT cluster's visible region (mirrors the
    /// non-overlap assumption `ensure_overlay`'s `covered` tracking and the
    /// stats table's area summation already depend on), then either extend/
    /// merge into whichever existing visible region(s) of the resolved
    /// cluster the stroke touches, or create a brand new region. Persists
    /// the result immediately, consistent with every other curation action.
    /// Lazily (re)computes the Wand tool's per-pixel Lab (a,b) chroma arrays
    /// for `leaf_idx`, cached until the leaf changes — reuses `channels::lab_ab`
    /// (the same CIELAB conversion the color-deviation detection channel uses)
    /// rather than a fresh color-distance metric.
    fn ensure_wand_lab(&mut self, leaf_idx: usize) {
        if let Some((li, _, _, _)) = &self.wand_lab_cache {
            if *li == leaf_idx {
                return;
            }
        }
        let Some(leaf) = self.results.get(leaf_idx) else { return };
        let (w, h) = (leaf.w as usize, leaf.h as usize);
        // Full L+a+b, not just a/b — necrosis/discoloration is often as much
        // a LIGHTNESS change as a hue change; a/b-only distance (fine for
        // `color_deviation_map`'s different purpose) throws that signal away
        // and made the wand's boundary unreliable.
        let mut l = vec![0f32; w * h];
        let mut a = vec![0f32; w * h];
        let mut b = vec![0f32; w * h];
        for i in 0..w * h {
            let (r, g, bl) = (leaf.rgba[i * 4], leaf.rgba[i * 4 + 1], leaf.rgba[i * 4 + 2]);
            let (li_, ai, bi) = channels::rgb_to_lab(r, g, bl);
            l[i] = li_;
            a[i] = ai;
            b[i] = bi;
        }
        self.wand_lab_cache = Some((leaf_idx, l, a, b));
    }

    /// Rebuilds `wand_mask_tex` from `wand_mask` — called only when the mask
    /// actually changes (on click/shift-click, Fill, Clear), NEVER per
    /// frame. The live highlight used to redraw every pixel individually via
    /// `painter().rect_filled` EVERY FRAME, which at a loose tolerance
    /// (tens/hundreds of thousands of pixels) meant that many paint calls
    /// 60x/second — the real cause of the wand "nearly crashing the PC" at
    /// large tolerances, not the flood-fill itself. One texture upload per
    /// click, one blit per frame, regardless of mask size.
    fn rebuild_wand_mask_tex(&mut self, ctx: &Context, lw: usize, lh: usize) {
        if self.wand_mask.is_empty() || lw == 0 || lh == 0 {
            self.wand_mask_tex = None;
            return;
        }
        let mut px = vec![0u8; lw * lh * 4];
        for &(x, y) in &self.wand_mask {
            if x < 0 || y < 0 {
                continue;
            }
            let (x, y) = (x as usize, y as usize);
            if x >= lw || y >= lh {
                continue;
            }
            let o = (y * lw + x) * 4;
            px[o] = 120;
            px[o + 1] = 220;
            px[o + 2] = 120;
            px[o + 3] = 150;
        }
        let ci = egui::ColorImage::from_rgba_unmultiplied([lw, lh], &px);
        self.wand_mask_tex = Some(ctx.load_texture("wand_mask_pending", ci, egui::TextureOptions::NEAREST));
    }

    /// Is this leaf-space pixel actual leaf tissue rather than cutout background?
    ///
    /// Alpha > 10 matches `HOLE_ALPHA_THR` / `tile_leaf`, so a pixel can never be
    /// simultaneously "paintable" here and "not leaf" to the detector.
    fn leaf_pixel_valid(&self, leaf_idx: usize, px: i32, py: i32) -> bool {
        let Some(leaf) = self.results.get(leaf_idx) else { return false };
        if px < 0 || py < 0 || px >= leaf.w as i32 || py >= leaf.h as i32 {
            return false;
        }
        let o = ((py as usize * leaf.w as usize) + px as usize) * 4 + 3;
        leaf.rgba.get(o).is_some_and(|&a| a > 10)
    }

    fn finish_brush_stroke(&mut self, leaf_idx: usize, toasts: &mut ToastManager) {
        let pts = std::mem::take(&mut self.brush_stroke);
        if pts.is_empty() {
            return;
        }
        // Fall back to the selected region's family rather than refusing.
        //
        // This used to hard-error "Type a cluster name first" whenever the free-
        // text field was empty, even with a family plainly selected — so a user
        // who had just reviewed Sucker and painted more Sucker got a success
        // toast and a "type a name" error in the same breath. Typing is now only
        // required when there is genuinely nothing to infer from.
        let label_name = match self.hardneg_label.trim() {
            "" => match self.brush_default_family() {
                Some(f) => {
                    self.hardneg_label = f.clone();
                    f
                }
                None => {
                    toasts.error("Select a family first, or type a new name for it.");
                    return;
                }
            },
            s => s.to_string(),
        };

        let (mut min_x, mut min_y) = (i32::MAX, i32::MAX);
        let (mut max_x, mut max_y) = (i32::MIN, i32::MIN);
        for &(x, y) in &pts {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
        let (uw, uh) = ((max_x - min_x + 1) as u32, (max_y - min_y + 1) as u32);
        let mut mask = vec![false; (uw * uh) as usize];
        for &(x, y) in &pts {
            let (gx, gy) = ((x - min_x) as u32, (y - min_y) as u32);
            mask[(gy * uw + gx) as usize] = true;
        }
        let bbox_leaf = [min_x as u32, min_y as u32, uw, uh];
        let label = self.resolve_cluster_id(&label_name);

        // exclude pixels already owned by a DIFFERENT cluster's visible region
        for i in 0..self.regions.len() {
            if self.regions[i].leaf != leaf_idx || !self.region_visible(i) || self.labels[i] == label {
                continue;
            }
            let r = &self.regions[i];
            let [rx, ry, rw, rh] = r.bbox_leaf;
            for ly in 0..rh {
                for lx in 0..rw {
                    if !r.mask[(ly * rw + lx) as usize] {
                        continue;
                    }
                    let (gx, gy) = (rx + lx, ry + ly);
                    if gx < bbox_leaf[0] || gy < bbox_leaf[1]
                        || gx >= bbox_leaf[0] + uw || gy >= bbox_leaf[1] + uh
                    {
                        continue;
                    }
                    let (mx, my) = (gx - bbox_leaf[0], gy - bbox_leaf[1]);
                    mask[(my * uw + mx) as usize] = false;
                }
            }
        }
        let area = mask.iter().filter(|&&b| b).count() as u32;
        if area == 0 {
            toasts.info("Nothing painted — it all overlapped a different cluster's region.");
            return;
        }

        // centroid + crop for the new region
        let (mut sx, mut sy, mut cnt) = (0u64, 0u64, 0u64);
        for gy in 0..uh {
            for gx in 0..uw {
                if mask[(gy * uw + gx) as usize] {
                    sx += gx as u64;
                    sy += gy as u64;
                    cnt += 1;
                }
            }
        }
        let (ccx, ccy) = (min_x as f32 + sx as f32 / cnt.max(1) as f32, min_y as f32 + sy as f32 / cnt.max(1) as f32);
        let crop = self.results.get(leaf_idx)
            .map(|l| worker::context_crop(&l.rgba, l.w, l.h, ccx, ccy, worker::CROP_WIN))
            .unwrap_or_default();

        let stroke_region = AnomalyRegion {
            leaf: leaf_idx, bbox_leaf, mask,
            descriptor: [0.0; 8], family: label,
            crop, crop_size: worker::CROP_WIN,
            dino_embed: Vec::new(), // hand-painted regions never had a detector-derived one
            dino_embed_whole: Vec::new(),
        };

        // does the stroke touch any existing VISIBLE region of this cluster?
        let mut touched: Vec<usize> = Vec::new();
        for i in 0..self.regions.len() {
            if self.regions[i].leaf != leaf_idx || !self.region_visible(i) || self.labels[i] != label {
                continue;
            }
            if bbox_proximate(self.regions[i].bbox_leaf, stroke_region.bbox_leaf)
                && regions_touch(&self.regions[i], &stroke_region)
            {
                touched.push(i);
            }
        }

        self.regions.push(stroke_region);
        let new_idx = self.regions.len() - 1;
        self.region_area.push(area);
        self.labels.push(label);
        self.coords.push([0.0, 0.0]);
        self.region_thumbs.push(None);

        // Everything below is recorded so ONE stroke is ONE undo step. State is
        // diffed around the merge rather than predicted from it.
        let merged_before: HashSet<usize> = self.merged_away.clone();
        let geom_before: Vec<RegionGeom> =
            touched.iter().filter_map(|&i| self.snapshot_region(i)).collect();

        let idx = if touched.is_empty() {
            new_idx
        } else {
            let mut group = touched;
            group.push(new_idx);
            self.merge_region_group(&group, toasts);
            *group.iter().min().unwrap()
        };

        let merged: Vec<usize> = self
            .merged_away
            .difference(&merged_before)
            .copied()
            .collect();
        let geom_after: Vec<RegionGeom> =
            geom_before.iter().filter_map(|g| self.snapshot_region(g.idx)).collect();
        self.push_undo(UndoEntry::Paint {
            created: vec![new_idx],
            merged,
            before: geom_before,
            after: geom_after,
        });

        self.build_clusters(toasts);
        self.overlay_tex = None;
        let name = self.cluster_names.get(&label).cloned().unwrap_or_else(|| format!("Cluster {label}"));
        self.persist_region(idx, &name, false, toasts);
        toasts.success(format!("Painted \"{name}\""));
    }

    /// Rasterizes the closed polygon into a leaf-pixel stroke, then either
    /// commits it immediately (a region is selected — use ITS family,
    /// exactly like `finish_brush_stroke`'s existing touching-merge
    /// behavior) or opens the family-choice popup (nothing selected —
    /// there's no family to infer, ask).
    fn finish_polygon(&mut self, leaf_idx: usize, click_pos: egui::Pos2, toasts: &mut ToastManager) {
        let poly = std::mem::take(&mut self.poly_points);
        let Some(([bx, by, bw, bh], mask)) = fill_polygon_mask(&poly) else { return };
        let mut stroke: HashSet<(i32, i32)> = HashSet::new();
        for yy in 0..bh {
            for xx in 0..bw {
                if mask[(yy * bw + xx) as usize] {
                    stroke.insert((bx as i32 + xx as i32, by as i32 + yy as i32));
                }
            }
        }
        let sel = self.effective_selection();
        if let Some(&i) = sel.first() {
            let cid = self.labels[i];
            let name = self.cluster_names.get(&cid).cloned().unwrap_or_else(|| format!("Cluster {cid}"));
            self.brush_stroke = stroke;
            self.hardneg_label = name;
            self.finish_brush_stroke(leaf_idx, toasts);
        } else {
            self.poly_pick_pos = click_pos;
            self.poly_pending = Some(stroke);
        }
    }

    /// Resolve a completed eraser stroke (accumulated leaf-pixel coords,
    /// shares `brush_stroke` with the Brush tool — they're never active at
    /// once): for every visible region on this leaf the stroke overlaps,
    /// clear those mask pixels. A region erased down to nothing is removed
    /// outright via the existing `remove_regions` (lands on the same undo
    /// stack as any other reject); one that survives gets its mask/area
    /// updated AND its crop/centroid regenerated around the shrunk mask,
    /// mirroring `merge_region_group`'s own survivor-crop refresh — the
    /// displayed thumbnail and any future export both read `regions[i].crop`
    /// directly, so leaving it stale would silently show/export the
    /// pre-erase pixels.
    fn erase_stroke(&mut self, leaf_idx: usize, toasts: &mut ToastManager) {
        let pts = std::mem::take(&mut self.brush_stroke);
        if pts.is_empty() {
            return;
        }
        let mut emptied: Vec<usize> = Vec::new();
        // Captured BEFORE any mutation so the stroke is reversible as one
        // gesture. Only regions this stroke actually touches are recorded, so
        // the cost is proportional to the edit, not to the leaf.
        let mut before: Vec<RegionGeom> = Vec::new();
        for i in 0..self.regions.len() {
            if self.regions[i].leaf != leaf_idx || !self.region_visible(i) {
                continue;
            }
            let [bx, by, bw, bh] = self.regions[i].bbox_leaf;
            let mut touched = false;
            // Snapshot LAZILY, on the first pixel actually cleared. Snapshotting
            // up front would clone the mask and crop of every visible region on
            // the leaf, most of which the stroke never reaches.
            let mut snap: Option<RegionGeom> = None;
            for &(x, y) in &pts {
                if x < bx as i32 || y < by as i32 || x >= (bx + bw) as i32 || y >= (by + bh) as i32 {
                    continue;
                }
                let (mx, my) = ((x - bx as i32) as u32, (y - by as i32) as u32);
                let mi = (my * bw + mx) as usize;
                if self.regions[i].mask[mi] {
                    if snap.is_none() {
                        snap = self.snapshot_region(i);
                    }
                    self.regions[i].mask[mi] = false;
                    touched = true;
                }
            }
            if !touched {
                continue;
            }
            if let Some(s) = snap {
                before.push(s);
            }
            let new_area = self.regions[i].mask.iter().filter(|&&b| b).count() as u32;
            if new_area == 0 {
                emptied.push(i);
                continue;
            }
            self.region_area[i] = new_area;
            // regenerate crop + centroid around the SHRUNK mask
            let (mut sx, mut sy, mut cnt) = (0u64, 0u64, 0u64);
            for ly in 0..bh {
                for lx in 0..bw {
                    if self.regions[i].mask[(ly * bw + lx) as usize] {
                        sx += lx as u64;
                        sy += ly as u64;
                        cnt += 1;
                    }
                }
            }
            let (ccx, ccy) = (bx as f32 + sx as f32 / cnt.max(1) as f32, by as f32 + sy as f32 / cnt.max(1) as f32);
            let crop_size = self.regions[i].crop_size;
            if let Some(new_crop) = self.results.get(leaf_idx)
                .map(|l| worker::context_crop(&l.rgba, l.w, l.h, ccx, ccy, crop_size))
            {
                self.regions[i].crop = new_crop;
            }
            if let Some(t) = self.region_thumbs.get_mut(i) {
                *t = None;
            }
            // a previously-confirmed region's crop just changed underneath
            // it — re-persist so the on-disk copy isn't stale, same reason
            // merge_region_group re-persists its survivor.
            if self.persisted.contains(&i) {
                let cid = self.labels[i];
                let name = self.cluster_names.get(&cid).cloned().unwrap_or_else(|| format!("Cluster {cid}"));
                self.persist_region(i, &name, false, toasts);
            }
        }
        // One gesture, one undo entry — including any region erased to nothing.
        // `remove_regions_inner` is used deliberately so the removal does not
        // push its own competing entry and make one stroke take two Ctrl+Z.
        if !emptied.is_empty() {
            self.remove_regions_inner(&emptied, toasts, false);
        }
        if !before.is_empty() || !emptied.is_empty() {
            let after: Vec<RegionGeom> =
                before.iter().filter_map(|g| self.snapshot_region(g.idx)).collect();
            self.push_undo(UndoEntry::Erase { before, after, emptied });
        }
        self.overlay_tex = None;
    }

    /// Pixels within this perpendicular/boundary distance of a knife cut are
    /// initially excluded from BOTH resulting pieces — a real gap, not just
    /// a classification line. Without SOME gap the two pieces stay
    /// 8-connected-adjacent (`regions_touch`'s definition, matching
    /// `detect::connected_components`) and `merge_touching_regions` (run
    /// from `build_clusters`) would silently re-fuse them on the very next
    /// cluster rebuild. 2px comfortably exceeds the Chebyshev-distance-1
    /// touch threshold regardless of the cut's angle, while staying thin
    /// enough to read as a clean cut.
    const KNIFE_KERF: f32 = 2.0;
    /// The INNERMOST band that stays permanently excluded even after
    /// `reclaim_kerf` — everything between this and `KNIFE_KERF` gets
    /// returned to whichever piece is actually nearest (real marked pixels
    /// were being silently discarded outright before), while this thin
    /// residual band still guarantees the two pieces stay non-adjacent, so
    /// the cut can't quietly heal itself on the next `build_clusters`.
    const KNIFE_KERF_KEEP: f32 = 1.0;

    /// Knife tool: auto-detects line-cut vs polycut from where the drag
    /// starts (no separate mode toggle) — reuses `canvas_drag_start`
    /// (line-cut's start point, same field Select's rubber-band uses) and
    /// `lasso_points` (polycut's freeform loop, same field Lasso uses);
    /// safe since exactly one tool is ever active and tool-switching clears
    /// both (see the tool-switch handler in `show_toolbox`).
    fn handle_knife(
        &mut self, ui: &mut Ui, resp: &egui::Response, leaf_idx: usize,
        img_rect: egui::Rect, s: f32, toasts: &mut ToastManager,
    ) {
        let to_leaf = |p: egui::Pos2| ((p.x - img_rect.min.x) / s.max(1e-3), (p.y - img_rect.min.y) / s.max(1e-3));

        if resp.drag_started_by(egui::PointerButton::Primary) {
            if let Some(p) = resp.interact_pointer_pos() {
                let (lx, ly) = to_leaf(p);
                if self.region_at(leaf_idx, lx, ly).is_some() {
                    // starts INSIDE a region -> polycut: collect a freeform
                    // loop exactly like Lasso.
                    self.lasso_points.clear();
                    self.lasso_points.push(p);
                } else {
                    // starts OUTSIDE any region -> line-cut candidate; only
                    // the two endpoints matter, the dragged path's wiggle is
                    // ignored.
                    self.canvas_drag_start = Some(p);
                    self.lasso_points.clear();
                }
            }
        }

        if let Some(start) = self.canvas_drag_start {
            if let Some(cur) = resp.interact_pointer_pos() {
                ui.painter().line_segment([start, cur], egui::Stroke::new(2.0, Color32::from_rgb(255, 90, 90)));
            }
            if resp.drag_stopped_by(egui::PointerButton::Primary) {
                if let Some(end) = resp.interact_pointer_pos() {
                    let (ax, ay) = to_leaf(start);
                    let (bx, by) = to_leaf(end);
                    if self.region_at(leaf_idx, bx, by).is_some() {
                        toasts.error("Line cut needs both ends outside any region.");
                    } else if (bx - ax).hypot(by - ay) > 4.0 {
                        self.do_line_cut(leaf_idx, &[(ax, ay), (bx, by)], toasts);
                    }
                }
                self.canvas_drag_start = None;
            }
        } else if !self.lasso_points.is_empty() {
            if resp.dragged_by(egui::PointerButton::Primary) {
                if let Some(p) = resp.interact_pointer_pos() {
                    if self.lasso_points.last().map_or(true, |last| last.distance(p) > 2.0) {
                        self.lasso_points.push(p);
                    }
                }
            }
            if self.lasso_points.len() >= 2 {
                let mut pts = self.lasso_points.clone();
                pts.push(pts[0]);
                ui.painter().add(egui::Shape::line(pts, egui::Stroke::new(1.5, Color32::from_rgb(255, 90, 90))));
            }
            if resp.drag_stopped_by(egui::PointerButton::Primary) {
                if self.lasso_points.len() >= 3 {
                    let poly: Vec<(f32, f32)> = self.lasso_points.iter().map(|p| to_leaf(*p)).collect();
                    let (sx, sy) = poly[0];
                    // The lasso cuts the region it STARTS on. Starting anywhere
                    // else used to do nothing at all, silently — reported as
                    // "das Lasso Tool ging eben und jetzt nicht mehr", because a
                    // tool that fails without saying so is indistinguishable
                    // from a broken one. Especially easy to hit right after an
                    // undo, when the region you started on may be hidden again.
                    match self.region_at(leaf_idx, sx, sy) {
                        Some(i) => self.do_polycut(leaf_idx, i, &poly, toasts),
                        None => toasts.info(
                            "Lasso cuts the region it starts on — begin the loop inside one.",
                        ),
                    }
                } else if self.lasso_points.len() >= 1 {
                    toasts.info("Lasso needs a loop — drag around the part you want to cut off.");
                }
                self.lasso_points.clear();
            }
        }
    }

    /// Click-to-place-vertices alternative to Knife's drag gestures — more
    /// deliberate, and lets a line-cut bend through more than one straight
    /// segment. Reuses `lasso_points` (same field Knife's own polycut
    /// sub-mode already shares, cleared on tool switch and by Esc). Each
    /// click appends a vertex UNLESS it lands near the FIRST vertex with
    /// ≥3 already placed — that specific gesture closes the loop and
    /// immediately carves it out via the existing `do_polycut` (first
    /// vertex must have been inside a region, same precondition Knife's
    /// polycut already has). Enter with an open (≥2-point, not closed)
    /// path instead finalizes it as a line-cut via the generalized
    /// `do_line_cut` — both endpoints must be outside any region, same
    /// precondition Knife's straight line-cut already has.
    fn handle_scissor(
        &mut self, ui: &mut Ui, resp: &egui::Response, leaf_idx: usize,
        img_rect: egui::Rect, s: f32, toasts: &mut ToastManager,
    ) {
        let to_leaf = |p: egui::Pos2| ((p.x - img_rect.min.x) / s.max(1e-3), (p.y - img_rect.min.y) / s.max(1e-3));
        const CLOSE_RADIUS: f32 = 10.0;

        if resp.clicked_by(egui::PointerButton::Primary) {
            if let Some(p) = resp.interact_pointer_pos() {
                let closes_loop = self.lasso_points.len() >= 3
                    && self.lasso_points.first().map_or(false, |&f| f.distance(p) < CLOSE_RADIUS);
                if closes_loop {
                    let poly: Vec<(f32, f32)> = self.lasso_points.iter().map(|q| to_leaf(*q)).collect();
                    let (sx, sy) = poly[0];
                    if let Some(i) = self.region_at(leaf_idx, sx, sy) {
                        self.do_polycut(leaf_idx, i, &poly, toasts);
                    } else {
                        toasts.error("Scissor loop needs to start inside a region.");
                    }
                    self.lasso_points.clear();
                } else {
                    self.lasso_points.push(p);
                }
            }
        }
        if resp.secondary_clicked() {
            self.lasso_points.clear();
        }
        if !self.lasso_points.is_empty() {
            let mut pts = self.lasso_points.clone();
            if let Some(hover) = resp.hover_pos() {
                pts.push(hover); // live rubber-band to the cursor
            }
            ui.painter().add(egui::Shape::line(pts, egui::Stroke::new(1.5, Color32::from_rgb(255, 90, 90))));
        }
        // Focus guard: without it, pressing Enter to commit a typed cluster name
        // also committed a Scissor cut, because this site checked the raw key with
        // no regard for what had keyboard focus. `Enter` is additionally the
        // confirm-selection binding, so one press could fire both.
        let typing = ui.memory(|m| m.focused().is_some());
        if !typing
            && self.lasso_points.len() >= 2
            && ui.input(|i| i.key_pressed(egui::Key::Enter))
        {
            let poly: Vec<(f32, f32)> = self.lasso_points.iter().map(|q| to_leaf(*q)).collect();
            let (sx, sy) = poly[0];
            let (ex, ey) = poly[poly.len() - 1];
            if self.region_at(leaf_idx, sx, sy).is_some() || self.region_at(leaf_idx, ex, ey).is_some() {
                toasts.error("Line cut needs both ends outside any region.");
            } else {
                self.do_line_cut(leaf_idx, &poly, toasts);
            }
            self.lasso_points.clear();
        }
    }

    /// Straight-line (or bent polyline, see below) cut: every point in
    /// `pts` is outside any region by construction (see `handle_knife`).
    /// Deliberately NOT "classify every pixel by which side of the line
    /// it's on" — that split a big/sprawling or intertwined region ALL THE
    /// WAY THROUGH once any part of it was near the segment, even far from
    /// where the drag actually was ("cuts through the whole region instead
    /// of just where I cut"). This carves a kerf corridor out of the
    /// region's mask only along the actual finite path (clamped distance to
    /// the nearest segment, not an infinite line), then runs connected
    /// components on what's left — only if that corridor alone
    /// geometrically separates the region into 2+ disjoint pieces does a
    /// cut happen at all, and `reclaim_kerf` recovers everything beyond the
    /// minimal `KNIFE_KERF_KEEP` safety margin back into its nearest piece
    /// instead of discarding it. `pts.len() == 2` (a straight drag) is just
    /// this function's smallest case — Knife and Scissor share it
    /// unchanged. A single gesture can split more than one region; all
    /// splits land in ONE `UndoEntry::Cut` so one Ctrl+Z undoes the whole
    /// gesture, matching `remove_regions`'s batch-into-one-entry precedent.
    fn do_line_cut(&mut self, leaf_idx: usize, pts: &[(f32, f32)], toasts: &mut ToastManager) {
        if pts.len() < 2 {
            return;
        }
        let (mut seg_x0, mut seg_y0) = (f32::INFINITY, f32::INFINITY);
        let (mut seg_x1, mut seg_y1) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &(x, y) in pts {
            seg_x0 = seg_x0.min(x - Self::KNIFE_KERF);
            seg_y0 = seg_y0.min(y - Self::KNIFE_KERF);
            seg_x1 = seg_x1.max(x + Self::KNIFE_KERF);
            seg_y1 = seg_y1.max(y + Self::KNIFE_KERF);
        }
        let mut originals = Vec::new();
        let mut created = Vec::new();
        for i in 0..self.regions.len() {
            if self.regions[i].leaf != leaf_idx || !self.region_visible(i) {
                continue;
            }
            let [rx, ry, rw, rh] = self.regions[i].bbox_leaf;
            let region_in_segment_span = (rx as f32) < seg_x1 && ((rx + rw) as f32) > seg_x0
                && (ry as f32) < seg_y1 && ((ry + rh) as f32) > seg_y0;
            if !region_in_segment_span {
                continue;
            }
            let mut carved = self.regions[i].mask.clone();
            let mut any_carved = false;
            for ly in 0..rh {
                for lx in 0..rw {
                    let idx = (ly * rw + lx) as usize;
                    if !carved[idx] {
                        continue;
                    }
                    let (gx, gy) = ((rx + lx) as f32, (ry + ly) as f32);
                    if dist_to_polyline(gx, gy, pts) < Self::KNIFE_KERF {
                        carved[idx] = false;
                        any_carved = true;
                    }
                }
            }
            if !any_carved {
                continue; // the path never actually got close to this region's real pixels
            }
            let mut comps = mask_connected_components(&carved, rw, rh);
            if comps.len() < 2 {
                continue; // didn't fully separate it — no-op, region stays whole
            }
            let permanent_gap = |lx: u32, ly: u32| {
                dist_to_polyline((rx + lx) as f32, (ry + ly) as f32, pts) < Self::KNIFE_KERF_KEEP
            };
            reclaim_kerf(&mut comps, &self.regions[i].mask, rw, rh, permanent_gap);
            let pieces: Vec<Vec<(u32, u32)>> = comps.into_iter()
                .map(|idxs| idxs.into_iter()
                    .map(|p| (rx + (p as u32 % rw), ry + (p as u32 / rw)))
                    .collect())
                .collect();
            self.apply_cut(leaf_idx, i, pieces, &mut originals, &mut created, toasts);
        }
        if created.is_empty() {
            toasts.info("Line didn't fully separate any region.");
            return;
        }
        self.push_undo(UndoEntry::Cut { originals, created });
        self.build_clusters(toasts);
        self.overlay_tex = None;
        toasts.success("Knife: split region(s)");
    }

    /// Polygon cut ("polycut"): the drag started INSIDE region `i` (checked
    /// by `handle_knife`), so this only ever touches THAT region — carving a
    /// piece out of it, not a leaf-wide operation like the line-cut. Pixels
    /// inside the (now-closed) loop become one new piece, the rest become a
    /// second; same kerf principle as the line-cut, measured as distance to
    /// the nearest polygon edge instead of to a line.
    fn do_polycut(&mut self, leaf_idx: usize, i: usize, poly: &[(f32, f32)], toasts: &mut ToastManager) {
        if !self.region_visible(i) {
            return;
        }
        let [rx, ry, rw, rh] = self.regions[i].bbox_leaf;
        let mask = self.regions[i].mask.clone();
        let mut inside: Vec<usize> = Vec::new();
        let mut outside: Vec<usize> = Vec::new();
        for ly in 0..rh {
            for lx in 0..rw {
                let idx = (ly * rw + lx) as usize;
                if !mask[idx] {
                    continue;
                }
                let (gx, gy) = ((rx + lx) as f32, (ry + ly) as f32);
                if dist_to_polygon_boundary(gx, gy, poly) < Self::KNIFE_KERF {
                    continue; // carved out for now — reclaim_kerf may return it below
                }
                if point_in_polygon(gx, gy, poly) {
                    inside.push(idx);
                } else {
                    outside.push(idx);
                }
            }
        }
        if inside.is_empty() || outside.is_empty() {
            toasts.info("Loop didn't carve out part of the region.");
            return;
        }
        let mut pieces = vec![inside, outside];
        let permanent_gap = |lx: u32, ly: u32| {
            dist_to_polygon_boundary((rx + lx) as f32, (ry + ly) as f32, poly) < Self::KNIFE_KERF_KEEP
        };
        reclaim_kerf(&mut pieces, &mask, rw, rh, permanent_gap);
        let leaf_pieces: Vec<Vec<(u32, u32)>> = pieces.into_iter()
            .map(|idxs| idxs.into_iter()
                .map(|p| (rx + (p as u32 % rw), ry + (p as u32 / rw)))
                .collect())
            .collect();
        let mut originals = Vec::new();
        let mut created = Vec::new();
        self.apply_cut(leaf_idx, i, leaf_pieces, &mut originals, &mut created, toasts);
        self.push_undo(UndoEntry::Cut { originals, created });
        self.build_clusters(toasts);
        self.overlay_tex = None;
        toasts.success("Knife: carved out piece");
    }

    /// Build the `AnomalyRegion` for one knife-cut piece — a fresh crop
    /// around the piece's own centroid, same "no detector-derived signal"
    /// convention `finish_brush_stroke`'s hand-painted regions already use.
    fn build_cut_piece(&self, leaf_idx: usize, family: i32, pts: &[(u32, u32)]) -> AnomalyRegion {
        let (bbox_leaf, mask) = tight_bbox_and_mask(pts);
        let (mut sx, mut sy) = (0u64, 0u64);
        for &(x, y) in pts {
            sx += x as u64;
            sy += y as u64;
        }
        let n = pts.len().max(1) as f32;
        let (ccx, ccy) = (sx as f32 / n, sy as f32 / n);
        let crop = self.results.get(leaf_idx)
            .map(|l| worker::context_crop(&l.rgba, l.w, l.h, ccx, ccy, worker::CROP_WIN))
            .unwrap_or_default();
        AnomalyRegion {
            leaf: leaf_idx, bbox_leaf, mask,
            descriptor: [0.0; 8], family,
            crop, crop_size: worker::CROP_WIN,
            dino_embed: Vec::new(), // cut pieces never had a detector-derived one
            dino_embed_whole: Vec::new(),
        }
    }

    /// Shared bookkeeping for one region `i` splitting into two pieces
    /// (`pts_a`/`pts_b`, disjoint leaf-global pixel sets): builds both new
    /// `AnomalyRegion`s, retracts+re-persists across the split if `i` was
    /// already confirmed (mirrors `merge_region_group`'s survivor-persist
    /// pattern — a cut must not silently revert a confirmed region to
    /// "unreviewed"), marks `i` `merged_away`, and appends the new indices
    /// to `originals`/`created` for the caller's single combined undo entry.
    fn apply_cut(
        &mut self, leaf_idx: usize, i: usize, pieces: Vec<Vec<(u32, u32)>>,
        originals: &mut Vec<usize>, created: &mut Vec<usize>, toasts: &mut ToastManager,
    ) {
        let family = self.labels[i];
        let was_persisted = self.persisted.contains(&i);
        if was_persisted {
            self.retract_persisted(i);
        }
        self.merged_away.insert(i);

        let mut new_idxs = Vec::new();
        for pts in &pieces {
            let piece = self.build_cut_piece(leaf_idx, family, pts);
            self.regions.push(piece);
            let idx = self.regions.len() - 1;
            self.region_area.push(pts.len() as u32);
            self.labels.push(family);
            self.coords.push([0.0, 0.0]);
            self.region_thumbs.push(None);
            new_idxs.push(idx);
        }

        if was_persisted {
            let name = self.cluster_names.get(&family).cloned().unwrap_or_else(|| format!("Cluster {family}"));
            for &idx in &new_idxs {
                self.persist_region(idx, &name, false, toasts);
            }
        }
        if let Some(&first) = new_idxs.first() {
            if self.selected_region == Some(i) {
                self.selected_region = Some(first);
            }
            if self.multi_selected.remove(&i) {
                self.multi_selected.insert(first);
            }
        }
        originals.push(i);
        created.extend(new_idxs);
    }

    /// Bulk-reassign every gallery-multi-selected region to the cluster named
    /// `self.reassign_name` — reuses an existing cluster with that name (matched
    /// case-insensitively) or allocates a fresh id. Lets the user correct a batch of
    /// misclustered regions (e.g. nervature wrongly grouped with necrosis) in one action
    /// instead of one at a time.
    fn reassign_selected(&mut self, toasts: &mut ToastManager) {
        let name = self.reassign_name.trim().to_string();
        let ids = self.effective_selection();
        if ids.is_empty() || name.is_empty() {
            return;
        }
        let id = self.resolve_cluster_id(&name);
        self.reassign_ids(&ids, id, &name, toasts);
        self.multi_selected.clear();
        self.selected_region = None;
        self.reassign_name.clear();
    }

    /// Resolve a typed cluster name to its id — reuses an existing cluster
    /// with that name (case-insensitive) or allocates a fresh one. Shared by
    /// `reassign_selected` and the brush tool so both go through the same
    /// allocation logic. Skips the reserved `NOVEL_FAMILY` sentinel so a
    /// typed name can never collide with the "Novel (PatchCore)" id.
    fn resolve_cluster_id(&mut self, name: &str) -> i32 {
        self.cluster_names.iter()
            .find(|(_, n)| n.eq_ignore_ascii_case(name))
            .map(|(&id, _)| id)
            .unwrap_or_else(|| {
                let mut next = self.labels.iter().copied()
                    .chain(self.cluster_names.keys().copied())
                    .max().unwrap_or(-1) + 1;
                while next == worker::NOVEL_FAMILY {
                    next += 1;
                }
                self.cluster_names.insert(next, name.to_string());
                next
            })
    }

    /// Merges class `from_id` into `into_id`: rewrites any already-saved
    /// curated labels on disk, merges the two classes in the head file
    /// (matched BY NAME, since under unsupervised clustering this run's
    /// label ids and the head's own class ids are unrelated numbering
    /// spaces), and relabels this run's in-memory regions immediately.
    /// Writes a renamed display name into the ACTUAL head file's `families`
    /// map — a no-op if `id` isn't a real class in the currently configured
    /// head (e.g. an ephemeral id from this run's own unsupervised
    /// clustering, which isn't stable across runs and was never meant to be
    /// named in the head anyway).
    fn rename_head_class_if_real(&mut self, id: i32, new_name: &str, toasts: &mut ToastManager) {
        let Some(head_path) = self.eff_head() else { return };
        let is_real = self.cached_head().map_or(false, |h| h.classes.contains(&id));
        if !is_real {
            return;
        }
        match fewshot::FewShotHead::load(&head_path) {
            Ok(mut h) => {
                h.families.insert(id.to_string(), new_name.to_string());
                match save_head_backed_up(&h, &head_path) {
                    Ok(()) => self.head_cache = None,
                    Err(e) => toasts.error(format!("rename: save failed: {e}")),
                }
            }
            Err(e) => toasts.error(format!("rename: load head: {e}")),
        }
    }

    fn merge_cluster_names(&mut self, from_id: i32, into_id: i32, toasts: &mut ToastManager) {
        let from_name = self.class_display_name(from_id);
        let into_name = self.class_display_name(into_id);

        if let Some(out) = self.output_folder.clone() {
            let curations_dir = out.join("curations");
            if let Err(e) = rewrite_curated_family(&curations_dir, &from_name, &into_name) {
                toasts.error(format!("merge: {e}"));
            }
        }

        if let Some(head_path) = self.eff_head() {
            match fewshot::FewShotHead::load(&head_path) {
                Ok(mut h) => {
                    let from_hid = h.classes.iter().copied().find(|&c| h.family_name(c) == from_name);
                    let into_hid = h.classes.iter().copied().find(|&c| h.family_name(c) == into_name);
                    if let (Some(fhid), Some(ihid)) = (from_hid, into_hid) {
                        match h.merge_class(fhid, ihid) {
                            Ok(()) => match save_head_backed_up(&h, &head_path) {
                                Ok(()) => {
                                    toasts.success(format!("Merged \"{from_name}\" into \"{into_name}\" in the head file"));
                                    self.head_cache = None;
                                }
                                Err(e) => toasts.error(format!("merge: save failed: {e}")),
                            },
                            Err(e) => toasts.error(format!("merge: {e}")),
                        }
                    }
                    // else: neither/one side isn't a real head class (e.g. a
                    // purely this-run unsupervised cluster name) — nothing to
                    // merge at the head level, the curation rewrite above and
                    // the in-run relabel below still apply.
                }
                Err(e) => toasts.error(format!("merge: load head: {e}")),
            }
        }

        for i in 0..self.labels.len() {
            if self.labels[i] == from_id {
                self.labels[i] = into_id;
            }
        }
        self.cluster_names.remove(&from_id);
        self.build_clusters(toasts);
    }

    /// Removes a class from the head entirely — no merge target. See
    /// `FewShotHead::delete_class`'s doc comment for the important caveat:
    /// this does NOT guarantee those patches fall through to PatchCore, only
    /// narrows the argmax.
    fn delete_cluster_from_head(&mut self, id: i32, toasts: &mut ToastManager) {
        let name = self.class_display_name(id);

        if let Some(out) = self.output_folder.clone() {
            let curations_dir = out.join("curations");
            if let Err(e) = rewrite_curated_family(&curations_dir, &name, "rejected") {
                toasts.error(format!("delete: {e}"));
            }
        }

        if let Some(head_path) = self.eff_head() {
            match fewshot::FewShotHead::load(&head_path) {
                Ok(mut h) => {
                    let hid = h.classes.iter().copied().find(|&c| h.family_name(c) == name);
                    if let Some(hid) = hid {
                        match h.delete_class(hid) {
                            Ok(()) => match save_head_backed_up(&h, &head_path) {
                                Ok(()) => {
                                    toasts.success(format!("Deleted \"{name}\" from the head file"));
                                    self.head_cache = None;
                                }
                                Err(e) => toasts.error(format!("delete: save failed: {e}")),
                            },
                            Err(e) => toasts.error(format!("delete: {e}")),
                        }
                    }
                    // else: not a real head class — nothing to delete at the
                    // head level, the curation rewrite above still applies.
                }
                Err(e) => toasts.error(format!("delete: load head: {e}")),
            }
        }

        // Deliberately NOT force-relabeling this run's existing regions to
        // Novel (an earlier version did) — that presents a specific guess as
        // fact. What the head will ACTUALLY call these patches once the
        // deleted class is gone is unknown without rerunning inference: they
        // might land on Novel, or they might land on whichever remaining
        // class is now their next-best match — `delete_class`'s own doc
        // comment says exactly this. So just drop the display name (they'll
        // show as "Cluster N" — visibly stale, not a false certainty) and
        // tell the user how to see the real answer.
        self.cluster_names.remove(&id);
        toasts.info("Re-run detection (or Preview a leaf to mark) to see how the \
                      updated head actually classifies these now.");
        self.build_clusters(toasts);
    }

    /// Reject `ids` (both the gallery's single right-click and the canvas
    /// context menu's bulk "Remove selected" route through this), recording
    /// the batch on an undo stack so an accidental removal — one region or
    /// many — can be reversed in one action. Persists immediately (see
    /// `persist_region`) — this is where the old "forgot to click Save"
    /// failure mode used to live.
    ///
    /// `write_reject`: true for a DELIBERATE reject (right-click "Reject
    /// selected", Delete key, gallery right-click) — genuine hard-negative
    /// signal, "this is not an anomaly." false for Eraser's full-erase-to-
    /// empty case — that's "I don't want this piece of my own edit
    /// anymore," not "this is healthy," so it just retracts any existing
    /// curation of it (no training signal either direction) instead of
    /// actively teaching healthy. Without this distinction, erasing a
    /// region and repainting a different class at the same spot writes a
    /// direct contradiction into the training data (confirmed via a real
    /// reassign→erase→repaint workflow).
    fn remove_regions(&mut self, ids: &[usize], toasts: &mut ToastManager, write_reject: bool) {
        if ids.is_empty() {
            return;
        }
        self.remove_regions_inner(ids, toasts, write_reject);
        self.push_undo(UndoEntry::Remove(ids.to_vec()));
        self.overlay_tex = None;
    }

    /// The removal itself, WITHOUT touching the undo stack.
    ///
    /// Split out for the eraser: a stroke that erases a region down to nothing
    /// is one gesture, and pushing a separate `Remove` entry for it would make
    /// that single stroke take two Ctrl+Z to reverse. The caller folds the
    /// emptied ids into its own entry instead.
    fn remove_regions_inner(&mut self, ids: &[usize], toasts: &mut ToastManager, write_reject: bool) {
        for &i in ids {
            self.removed.insert(i);
            if write_reject {
                self.persist_region(i, "rejected", true, toasts);
            } else if self.persisted.contains(&i) {
                self.retract_persisted(i);
            }
        }
    }

    /// Capture a region's reversible geometry.
    fn snapshot_region(&self, idx: usize) -> Option<RegionGeom> {
        let r = self.regions.get(idx)?;
        Some(RegionGeom {
            idx,
            mask: r.mask.clone(),
            area: self.region_area.get(idx).copied().unwrap_or(0),
            crop: r.crop.clone(),
        })
    }

    /// Put a captured geometry back, and invalidate the cached thumbnail so the
    /// tile does not keep showing the state we just reversed.
    fn restore_region(&mut self, g: &RegionGeom) {
        if let Some(r) = self.regions.get_mut(g.idx) {
            r.mask = g.mask.clone();
            r.crop = g.crop.clone();
        }
        if let Some(a) = self.region_area.get_mut(g.idx) {
            *a = g.area;
        }
        if let Some(t) = self.region_thumbs.get_mut(g.idx) {
            *t = None;
        }
    }

    /// Push a new undo entry, capping the stack the same way the old
    /// remove-only stack did.
    fn push_undo(&mut self, entry: UndoEntry) {
        self.struct_undo.push(entry);
        // A fresh edit invalidates any redo branch.
        self.struct_redo.clear();
        const MAX_UNDO: usize = 50;
        if self.struct_undo.len() > MAX_UNDO {
            self.struct_undo.remove(0);
        }
    }

    /// Undo the most recent structural edit — a removal (gallery reject or
    /// bulk "Remove selected") or a knife-cut gesture, whichever happened
    /// last. Bound to the gallery's "Undo" button and Ctrl+Z.
    fn undo_last_edit(&mut self, toasts: &mut ToastManager) {
        let Some(entry) = self.struct_undo.pop() else {
            toasts.info("Nothing to undo.");
            return;
        };
        self.apply_undo(&entry, toasts);
        self.struct_redo.push(entry);
        self.overlay_tex = None;
    }

    /// Re-apply the most recently undone edit.
    ///
    /// Added because undo alone was a trap: someone who undid one step too far
    /// had no way back, and said so — "es wäre richtig gut, wenn es nicht nur
    /// Undo gäbe, sondern auch einen Wiederherstellen Knopf".
    fn redo_last_edit(&mut self, toasts: &mut ToastManager) {
        let Some(entry) = self.struct_redo.pop() else {
            toasts.info("Nothing to redo.");
            return;
        };
        self.apply_redo(&entry, toasts);
        self.struct_undo.push(entry);
        self.overlay_tex = None;
    }

    fn apply_undo(&mut self, entry: &UndoEntry, toasts: &mut ToastManager) {
        match entry {
            UndoEntry::Paint { created, merged, before, .. } => {
                // Soft-hide rather than delete: indices are stable and redo
                // needs the region to still be there.
                for &i in created {
                    self.removed.insert(i);
                    self.retract_persisted(i);
                }
                for &i in merged {
                    self.merged_away.remove(&i);
                }
                for g in before {
                    self.restore_region(g);
                }
                toasts.success("Undid paint stroke");
            }
            UndoEntry::Erase { before, emptied, .. } => {
                for g in before {
                    self.restore_region(g);
                }
                for &i in emptied {
                    self.removed.remove(&i);
                }
                toasts.success("Undid eraser stroke");
            }
            UndoEntry::Remove(ids) => {
                for &i in ids {
                    self.removed.remove(&i);
                    // undo the disk write too — a restored region goes back to
                    // "unreviewed," not "rejected," so its persisted reject line
                    // (if any) shouldn't linger; the stable region_{i}.png filename
                    // means this is a clean delete, no orphan left behind.
                    self.retract_persisted(i);
                }
                toasts.success(format!("Restored {} region(s)", ids.len()));
            }
            UndoEntry::Confirm(ids) => {
                // Same retraction the Remove arm uses: the stable
                // `region_{i}.png` filename makes this a clean delete of the row
                // and its crop, with no orphan left behind.
                for &i in ids {
                    self.retract_persisted(i);
                }
                toasts.success(format!("Un-confirmed {} region(s)", ids.len()));
            }
            UndoEntry::Cut { originals, created } => {
                // Cut pieces were never independently rejected/persisted as
                // their own reject event, so undoing is a pure visibility
                // flip: hide the pieces, restore the pre-cut region(s).
                for &i in created {
                    self.merged_away.insert(i);
                }
                for &i in originals {
                    self.merged_away.remove(&i);
                }
                toasts.success("Undid knife cut");
                self.build_clusters(toasts);
            }
        }
        self.overlay_tex = None;
    }

    /// The exact inverse of `apply_undo`, entry for entry.
    ///
    /// Deliberately state-restoring rather than gesture-replaying: an eraser
    /// redo puts back the recorded `after` geometry instead of re-running the
    /// stroke, which would depend on a cursor path that no longer exists.
    fn apply_redo(&mut self, entry: &UndoEntry, toasts: &mut ToastManager) {
        match entry {
            UndoEntry::Paint { created, merged, after, .. } => {
                for &i in created {
                    self.removed.remove(&i);
                }
                for &i in merged {
                    self.merged_away.insert(i);
                }
                for g in after {
                    self.restore_region(g);
                }
                toasts.success("Redid paint stroke");
            }
            UndoEntry::Erase { after, emptied, .. } => {
                for g in after {
                    self.restore_region(g);
                }
                for &i in emptied {
                    self.removed.insert(i);
                }
                toasts.success("Redid eraser stroke");
            }
            UndoEntry::Remove(ids) => {
                for &i in ids {
                    self.removed.insert(i);
                }
                toasts.success(format!("Removed {} region(s) again", ids.len()));
            }
            UndoEntry::Confirm(ids) => {
                // Re-persist under the region's CURRENT family name — the label
                // is read now rather than stored, so a redo after a reassign
                // writes the family the region actually has.
                for &i in ids {
                    let cid = self.labels.get(i).copied().unwrap_or(-1);
                    let name = self
                        .cluster_names
                        .get(&cid)
                        .cloned()
                        .unwrap_or_else(|| format!("Cluster {cid}"));
                    self.persist_region(i, &name, false, toasts);
                }
                toasts.success(format!("Re-confirmed {} region(s)", ids.len()));
            }
            UndoEntry::Cut { originals, created } => {
                for &i in created {
                    self.merged_away.remove(&i);
                }
                for &i in originals {
                    self.merged_away.insert(i);
                }
                toasts.success("Redid knife cut");
                self.build_clusters(toasts);
            }
        }
        self.overlay_tex = None;
    }

    /// Delete a persisted region's crop file AND its `labels.jsonl` line
    /// (filter-rewrite by the stable `region_{idx}.png` crop name — mirrors
    /// `retract_hardneg`'s same pattern for stamp-tool crops), and un-mark it
    /// `persisted`. Used by Reject-undo; a later re-Confirm/re-Reject simply
    /// calls `persist_region` again and writes a fresh line.
    fn retract_persisted(&mut self, idx: usize) {
        if !self.persisted.remove(&idx) {
            return;
        }
        let Some(out) = self.output_folder.clone() else { return };
        let fname = format!("region_{idx}.png");
        let _ = std::fs::remove_file(out.join("curations").join("labels").join(&fname));
        let jsonl_path = out.join("curations").join("labels.jsonl");
        let Ok(text) = std::fs::read_to_string(&jsonl_path) else { return };
        let needle = format!("\"crop\":\"{fname}\"");
        let kept: String = text.lines().filter(|l| !l.contains(&needle)).map(|l| format!("{l}\n")).collect();
        let _ = std::fs::write(&jsonl_path, kept);
    }

    /// Applies a background file/folder dialog's result to whichever field
    /// requested it. Must run every frame regardless of which app tab is
    /// active — `show_settings_panel` (the Settings screen's Pipeline
    /// category) spawns these same dialogs via `pick_row` but is a separate
    /// entry point from `show()`, so this can't be polled only from there:
    /// picking a path while on the Settings tab would otherwise sit in the
    /// channel unconsumed (label stuck showing the old path) until the user
    /// happened to switch to the Pipeline tab itself.
    pub fn poll_pick(&mut self) {
        if let Some((which, rx)) = &self.pick_rx {
            if let Ok(res) = rx.try_recv() {
                let which = *which;
                if let Some(p) = res {
                    match which {
                        Pick::Source => {
                            self.source_count = scan_image_count(&p);
                            self.source_folder = Some(p);
                        }
                        Pick::Output => self.output_folder = Some(p),
                        Pick::Yolo => self.yolo_model = Some(p),
                        Pick::Dino => self.dino_model = Some(p),
                        Pick::Bank => self.bank_path = Some(p),
                        Pick::Meta => self.meta_path = Some(p),
                        Pick::Recon => self.recon_ckpt = Some(p),
                        Pick::Head => self.head_path = Some(p),
                        Pick::MineHealthyDir => self.mine_healthy_dir = Some(p),
                        Pick::BaseSet => self.retrain_base_set = Some(p),
                    }
                }
                self.pick_rx = None;
            }
        }
    }
}

// ── free helpers ────────────────────────────────────────────────────────────

/// Minimal JSON string escaping for the curation label file (user-entered names).
pub(crate) fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Minimal CSV field escaping (quote if it contains a comma/quote/newline).
fn csv_escape(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Family name -> a safe CSV column-name fragment.
///
/// Family names are user-editable free text, and the wide format builds column
/// NAMES out of them ("Sucker_avg_area_px"). Quoting would survive a comma but
/// leaves a header that R and pandas both mangle on import, so the characters
/// are removed instead: anything outside `[A-Za-z0-9]` becomes `_`, runs
/// collapse, and a name that reduces to nothing falls back to `family`.
fn csv_header_slug(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() { "family".to_string() } else { trimmed.to_string() }
}

fn spawn_dialog(which: Pick) -> mpsc::Receiver<Option<PathBuf>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let res = match which {
            Pick::Source | Pick::Output | Pick::MineHealthyDir => rfd::FileDialog::new().pick_folder(),
            Pick::BaseSet => rfd::FileDialog::new().add_filter("base set", &["bin"]).pick_file(),
            // Stored as the CONTAINING folder (worker.rs looks for `gen.mpk`
            // inside it), but a folder-picker dialog can't show files at all —
            // let the user click the .mpk file directly, then take its parent.
            Pick::Recon => rfd::FileDialog::new()
                .add_filter("checkpoint", &["mpk"])
                .pick_file()
                .and_then(|p| p.parent().map(|d| d.to_path_buf())),
            Pick::Yolo | Pick::Dino => rfd::FileDialog::new()
                .add_filter("model weights", &["safetensors", "onnx"])
                .add_filter("all files", &["*"])
                .pick_file(),
            Pick::Bank => rfd::FileDialog::new().add_filter("bank", &["bin"]).pick_file(),
            Pick::Meta | Pick::Head => rfd::FileDialog::new().add_filter("json", &["json"]).pick_file(),
        };
        let _ = tx.send(res);
    });
    rx
}


/// Backs up `path` to `<path>.bak` (best-effort, overwrites any previous
/// backup) before writing `head` over it — in-place head edits (merge,
/// delete, rename) have no undo otherwise, unlike calibration's versioned
/// files.
fn save_head_backed_up(head: &fewshot::FewShotHead, path: &std::path::Path) -> Result<(), String> {
    if path.exists() {
        let bak = path.with_extension("json.bak");
        let _ = std::fs::copy(path, bak);
    }
    head.save(path)
}

/// Filesystem-safe stem for a user-typed calibration name: keeps
/// alphanumerics/`-`/`_`, collapses everything else to `_`.
fn sanitize_filename(s: &str) -> String {
    let out: String = s.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    if out.is_empty() { "calibration".to_string() } else { out }
}

/// Human-scale relative age string ("just now", "42m ago", "3h ago", "5d ago").
fn format_age(secs: u64) -> String {
    if secs < 60 { "just now".to_string() }
    else if secs < 3600 { format!("{}m ago", secs / 60) }
    else if secs < 86400 { format!("{}h ago", secs / 3600) }
    else { format!("{}d ago", secs / 86400) }
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 * (1.0 - t) + b as f32 * t).round().clamp(0.0, 255.0) as u8
}

/// Tint a region's masked pixels (bbox-local mask at `bbox_leaf`) into an RGBA buffer.
fn paint_region(px: &mut [u8], w: usize, h: usize, r: &AnomalyRegion, col: [u8; 3], t: f32) {
    let [bx, by, bw, bh] = r.bbox_leaf;
    for ly in 0..bh {
        for lx in 0..bw {
            if !r.mask[(ly * bw + lx) as usize] {
                continue;
            }
            let (gx, gy) = ((bx + lx) as usize, (by + ly) as usize);
            if gx >= w || gy >= h {
                continue;
            }
            let o = (gy * w + gx) * 4;
            px[o] = lerp_u8(px[o], col[0], t);
            px[o + 1] = lerp_u8(px[o + 1], col[1], t);
            px[o + 2] = lerp_u8(px[o + 2], col[2], t);
            px[o + 3] = 255;
        }
    }
}


/// Cheap prefilter for `merge_touching_regions`: do these two bboxes come
/// within 1px of each other? Inflating one side by 1px before the AABB test
/// is enough — two bboxes ending/starting on adjacent columns/rows can still
/// contain 8-connected touching pixels even though their raw AABBs don't
/// overlap at all.
fn bbox_proximate(a: [u32; 4], b: [u32; 4]) -> bool {
    let [ax, ay, aw, ah] = a;
    let [bx, by, bw, bh] = b;
    let ax0 = ax.saturating_sub(1);
    let ay0 = ay.saturating_sub(1);
    let ax1 = ax + aw + 1;
    let ay1 = ay + ah + 1;
    ax0 < bx + bw && ax1 > bx && ay0 < by + bh && ay1 > by
}

/// True 8-connected pixel adjacency (including direct overlap) between two
/// regions' masks, in absolute leaf coordinates — matches
/// `detect::connected_components`'s own definition of "connected" exactly,
/// so this is consistent with however the regions were extracted in the
/// first place, not an arbitrary different choice.
fn regions_touch(a: &AnomalyRegion, b: &AnomalyRegion) -> bool {
    let [ax, ay, aw, ah] = a.bbox_leaf;
    let [bx, by, bw, bh] = b.bbox_leaf;
    for ly in 0..ah {
        for lx in 0..aw {
            if !a.mask[(ly * aw + lx) as usize] {
                continue;
            }
            let (gx, gy) = (ax + lx, ay + ly);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = gx as i32 + dx;
                    let ny = gy as i32 + dy;
                    if nx < bx as i32 || ny < by as i32
                        || nx >= (bx + bw) as i32 || ny >= (by + bh) as i32
                    {
                        continue;
                    }
                    let (lbx, lby) = (nx as u32 - bx, ny as u32 - by);
                    if b.mask[(lby * bw + lbx) as usize] {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Tight bbox + bbox-local mask for a set of LEAF-GLOBAL pixel coordinates.
/// Shared by the Eraser (shrunk-region crop bounds) and Knife (each cut
/// piece's own geometry) tools — neither existing region-creation path
/// (`finish_brush_stroke`, `merge_region_group`) factored this out, they
/// each inline their own min/max scan. `pts` must be non-empty.
fn tight_bbox_and_mask(pts: &[(u32, u32)]) -> ([u32; 4], Vec<bool>) {
    let (mut min_x, mut min_y) = (u32::MAX, u32::MAX);
    let (mut max_x, mut max_y) = (0u32, 0u32);
    for &(x, y) in pts {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    let (uw, uh) = (max_x - min_x + 1, max_y - min_y + 1);
    let mut mask = vec![false; (uw * uh) as usize];
    for &(x, y) in pts {
        let (gx, gy) = (x - min_x, y - min_y);
        mask[(gy * uw + gx) as usize] = true;
    }
    ([min_x, min_y, uw, uh], mask)
}

/// Moore-neighbour boundary trace of a single connected region mask → ordered
/// boundary points (bbox-local pixel coords). Used to draw a smooth vector outline.
fn trace_contour(mask: &[bool], bw: u32, bh: u32) -> Vec<(f32, f32)> {
    let (w, h) = (bw as i32, bh as i32);
    if w < 1 || h < 1 {
        return Vec::new();
    }
    let at = |x: i32, y: i32| -> bool {
        x >= 0 && y >= 0 && x < w && y < h && mask[(y * w + x) as usize]
    };
    let mut start = (-1i32, -1i32);
    'f: for y in 0..h {
        for x in 0..w {
            if at(x, y) {
                start = (x, y);
                break 'f;
            }
        }
    }
    if start.0 < 0 {
        return Vec::new();
    }
    const NB: [(i32, i32); 8] = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];
    let mut contour = Vec::new();
    let mut p = start;
    let mut b = 4usize; // entered from the West
    let max_steps = (w * h * 4) as usize + 8;
    for _ in 0..max_steps {
        contour.push((p.0 as f32, p.1 as f32));
        let mut moved = false;
        for k in 1..=8 {
            let dir = (b + k) % 8;
            let q = (p.0 + NB[dir].0, p.1 + NB[dir].1);
            if at(q.0, q.1) {
                b = (dir + 4) % 8;
                p = q;
                moved = true;
                break;
            }
        }
        if !moved || p == start {
            break;
        }
    }
    contour
}

/// Chaikin corner-cutting: rounds a polyline into a smooth closed curve.
fn chaikin(pts: &[egui::Pos2], iters: usize) -> Vec<egui::Pos2> {
    let mut p = pts.to_vec();
    for _ in 0..iters {
        if p.len() < 3 {
            break;
        }
        let n = p.len();
        let mut q = Vec::with_capacity(n * 2);
        for i in 0..n {
            let (a, c) = (p[i], p[(i + 1) % n]);
            q.push(a + (c - a) * 0.25);
            q.push(a + (c - a) * 0.75);
        }
        p = q;
    }
    p
}

/// 8-connected components of a bbox-local boolean mask (`w`x`h`) — each
/// component returned as its flat (`w`-major) pixel indices. Used by the
/// knife's line-cut to check whether carving a kerf corridor out of a
/// region's mask actually separated it into disjoint pieces, rather than
/// assuming a line's sign always produces a valid split regardless of the
/// region's real shape.
/// Reprojects `r.mask` (bbox-local) into the exact same `r.crop_size`-square
/// window `context_crop` built `r.crop` around — recomputing the centroid
/// FRESH from the mask's own true pixels (mean leaf coordinate) rather than
/// storing one on `AnomalyRegion`, since that's the same convention every
/// region-creation path (`finish_brush_stroke`, `build_cut_piece`, worker.rs's
/// `rg.centroid`) already uses to build the crop in the first place — so this
/// lines up with whatever centroid the saved crop was actually centered on.
/// White (255) = masked, black (0) = context-only. `None` if the mask is
/// empty/degenerate (nothing to reproject).
fn build_crop_mask_png(r: &AnomalyRegion) -> Option<image::GrayImage> {
    let [bx, by, bw, bh] = r.bbox_leaf;
    if bw == 0 || bh == 0 || r.mask.is_empty() {
        return None;
    }
    let (mut sx, mut sy, mut n) = (0u64, 0u64, 0u64);
    for ly in 0..bh {
        for lx in 0..bw {
            if r.mask[(ly * bw + lx) as usize] {
                sx += (bx + lx) as u64;
                sy += (by + ly) as u64;
                n += 1;
            }
        }
    }
    if n == 0 {
        return None;
    }
    let (cx, cy) = (sx as f32 / n as f32, sy as f32 / n as f32);
    let win = r.crop_size;
    let half = (win / 2) as i32;
    let (cxi, cyi) = (cx.round() as i32, cy.round() as i32);
    let mut px = vec![0u8; (win * win) as usize];
    for oy in 0..win as i32 {
        let sy_ = cyi - half + oy;
        if sy_ < by as i32 || sy_ >= (by + bh) as i32 {
            continue;
        }
        for ox in 0..win as i32 {
            let sx_ = cxi - half + ox;
            if sx_ < bx as i32 || sx_ >= (bx + bw) as i32 {
                continue;
            }
            let (lx, ly) = ((sx_ - bx as i32) as u32, (sy_ - by as i32) as u32);
            if r.mask[(ly * bw + lx) as usize] {
                px[(oy as u32 * win + ox as u32) as usize] = 255;
            }
        }
    }
    image::GrayImage::from_raw(win, win, px)
}

/// If `r`'s bbox fits within one `tile`×`tile` window, returns the single
/// existing crop+mask unchanged (today's behavior — the common case, zero
/// change). Otherwise covers the FULL bbox with a grid of ≤`tile`×`tile`
/// windows, so a big region no longer loses everything outside one small
/// centered slice (confirmed on a real region: ~80% missing). Tiles with
/// zero true mask pixels are dropped — no point persisting pure background
/// as a positive example.
fn build_region_tiles(
    r: &AnomalyRegion, leaf_rgba: &[u8], lw: u32, lh: u32, tile: u32,
) -> Vec<(Vec<u8>, u32, Option<image::GrayImage>)> {
    let [bx, by, bw, bh] = r.bbox_leaf;
    if bw <= tile && bh <= tile {
        return vec![(r.crop.clone(), r.crop_size, build_crop_mask_png(r))];
    }
    let cols = bw.div_ceil(tile).max(1);
    let rows = bh.div_ceil(tile).max(1);
    let mut out = Vec::new();
    for ty in 0..rows {
        for tx in 0..cols {
            let x0 = bx + tx * tile;
            let y0 = by + ty * tile;
            let (mx1, my1) = ((x0 + tile).min(bx + bw), (y0 + tile).min(by + bh));
            let mut any = false;
            'chk: for gy in y0..my1 {
                for gx in x0..mx1 {
                    let (lx, ly) = (gx - bx, gy - by);
                    if r.mask[(ly * bw + lx) as usize] {
                        any = true;
                        break 'chk;
                    }
                }
            }
            if !any {
                continue;
            }
            let crop = tile_crop(leaf_rgba, lw, lh, x0, y0, tile, tile);
            let mask_png = build_tile_mask_png(r, x0, y0, tile);
            out.push((crop, tile, mask_png));
        }
    }
    if out.is_empty() {
        // Shouldn't happen if the region has any mask pixel at all — fall
        // back to the single centered crop rather than persist nothing.
        return vec![(r.crop.clone(), r.crop_size, build_crop_mask_png(r))];
    }
    out
}

/// Like `worker::context_crop`, but positioned by an explicit top-left
/// corner instead of centered on a point — lets `build_region_tiles` cover
/// a big region with a grid instead of one fixed window around its centroid.
fn tile_crop(rgba: &[u8], w: u32, h: u32, x0: u32, y0: u32, tw: u32, th: u32) -> Vec<u8> {
    let mut out = vec![0u8; (tw * th * 4) as usize];
    for oy in 0..th {
        let sy = (y0 + oy).min(h.saturating_sub(1));
        for ox in 0..tw {
            let sx = (x0 + ox).min(w.saturating_sub(1));
            let si = ((sy * w + sx) * 4) as usize;
            let oi = ((oy * tw + ox) * 4) as usize;
            out[oi..oi + 4].copy_from_slice(&rgba[si..si + 4]);
        }
    }
    out
}

/// `build_crop_mask_png`'s sibling for an explicit tile rect instead of a
/// centroid-centered window — reprojects `r.mask` into the tile's own local
/// frame. `None` if the tile happens to contain no true mask pixel.
fn build_tile_mask_png(r: &AnomalyRegion, x0: u32, y0: u32, tile: u32) -> Option<image::GrayImage> {
    let [bx, by, bw, bh] = r.bbox_leaf;
    let mut px = vec![0u8; (tile * tile) as usize];
    let mut any = false;
    for oy in 0..tile {
        let gy = y0 + oy;
        if gy < by || gy >= by + bh {
            continue;
        }
        for ox in 0..tile {
            let gx = x0 + ox;
            if gx < bx || gx >= bx + bw {
                continue;
            }
            let (lx, ly) = (gx - bx, gy - by);
            if r.mask[(ly * bw + lx) as usize] {
                px[(oy * tile + ox) as usize] = 255;
                any = true;
            }
        }
    }
    if any { image::GrayImage::from_raw(tile, tile, px) } else { None }
}

// mask_connected_components, reclaim_kerf, wand_flood_fill, point_in_polygon,
// dist_to_polygon_boundary, dist_to_polyline, dist_point_to_segment moved to
// `crate::tabs::mask_tools` (shared with the Field Review tab) — see the
// `use` near the top of this file.
