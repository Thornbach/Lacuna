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
use worker::{spawn_pipeline, AnomalyRegion, PipeConfig, PipeMsg, PipelineLeaf};

const CLUSTER_PALETTE: [[u8; 3]; 10] = [
    [230, 80, 80], [80, 160, 230], [120, 200, 110], [230, 170, 60], [170, 110, 210],
    [70, 200, 190], [230, 110, 170], [150, 190, 70], [110, 130, 220], [220, 140, 90],
];

const GALLERY_PER_PAGE: usize = 60;

fn cluster_color(id: i32) -> [u8; 3] {
    if id < 0 {
        [150, 150, 150] // noise
    } else {
        CLUSTER_PALETTE[id as usize % CLUSTER_PALETTE.len()]
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
enum CanvasTool { Select, MarkHealthy, Brush, Eraser, Knife, Scissor, Lasso, Wand, Eyedropper, Polygon }

/// One entry per undoable structural edit — `Ctrl+Z`/the gallery's "Undo"
/// button always pops the most recent one, regardless of which action
/// produced it. `Cut` (Knife tool) never touches `removed`/`persisted`:
/// the pieces it creates were never independently rejected, so undoing a
/// cut is purely a visibility flip (`merged_away`), not a reject-reversal.
enum UndoEntry {
    Remove(Vec<usize>),
    Cut { originals: Vec<usize>, created: Vec<usize> },
}

#[derive(Clone, Copy, PartialEq)]
enum BrushShape { Square, Circle }

/// Right-panel sub-tabs — splits what used to be one long scrolling column
/// (leaf/morphology / stats / curation / retrain / export / log all stacked)
/// into focused views, mirroring Photoshop's tabbed panel dock
/// (Layers/Channels/Paths).
#[derive(Clone, Copy, PartialEq)]
enum ClusterPanelTab { Leaf, Clusters, Curate, Log }

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
    preview_tex:  Option<egui::TextureHandle>,
    preview_rx:   Option<mpsc::Receiver<Result<(Vec<u8>, u32, u32), String>>>,
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
    overlay_key:  Option<(usize, Option<i32>, bool, u32, bool)>, // leaf, cluster, recon, opacity%, outline
    show_recon:   bool,   // overlay the reconstructed (filled-in) leaf area on the canvas
    overlay_alpha: f32,   // cluster overlay opacity (fill mode) — see the leaf beneath
    overlay_outline: bool, // draw cluster OUTLINES instead of filled pixels

    // canvas zoom/pan — a universal gesture (scroll to zoom, hold middle-
    // mouse to pan), not a tool; applied on top of the existing fit-to-panel
    // rect, every other canvas interaction already funnels through the
    // resulting img_rect/s, so nothing else needs to change.
    canvas_zoom: f32,
    canvas_pan:  egui::Vec2,
    // lasso tool: live screen-space points while dragging (cleared on release)
    lasso_points: Vec<egui::Pos2>,
    // eyedropper tool: last hovered region's readout, shown in the options bar
    inspect_info: Option<String>,
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
    cluster_panel_tab: ClusterPanelTab,

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
    // Curate gallery: restrict to `selected_idx`'s own regions. Off by
    // default (whole-dataset view, unchanged). Small regions are easy to
    // miss in the dataset-wide gallery, which made it hard to tell when a
    // single leaf was actually fully reviewed — this + the per-leaf status
    // readout above the gallery fix that directly.
    filter_leaf_only: bool,
    selected_region:  Option<usize>,   // anomaly highlighted with a bbox on the leaf
    gallery_page:     usize,           // anomaly gallery pagination
    scroll_to_selected: bool,          // one-shot: scroll the gallery to selected_region
    // one-shot: scroll the LEAF strip to selected_idx. Set by the arrow-key
    // hotkeys only, not by clicking — a click already puts the tile under the
    // cursor, and re-centring it there would yank the strip out from under you.
    scroll_to_leaf: bool,
    region_thumbs:    Vec<Option<egui::TextureHandle>>, // parallel to regions
    removed:          HashSet<usize>,       // region indices removed by the user
    struct_undo:      Vec<UndoEntry>,       // undo stack: one entry per removal or knife-cut gesture
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
            head_tau:        0.85,
            head_grow:       0.7,
            tile_size:       256,
            margin_erode_px: 6,
            detect_holes:    true,
            min_hole_area:   16,
            filter_margin_holes: false, // opt-in: changes what gets reported
            hole_margin_px:      16,
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

            preview_tex:  None,
            preview_rx:   None,
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
            lasso_points: Vec::new(),
            inspect_info: None,
            brush_shape:  BrushShape::Circle,
            brush_size:   32,
            brush_stroke: HashSet::new(),
            wand_mask:       HashSet::new(),
            wand_tolerance:  14.0,
            wand_lab_cache:  None,
            wand_mask_tex:   None,
            cluster_panel_tab: ClusterPanelTab::Curate,

            regions:          Vec::new(),
            region_area:      Vec::new(),
            labels:           Vec::new(),
            coords:           Vec::new(),
            clusters:         Vec::new(),
            head_cache:       None,
            selected_cluster: None,
            filter_leaf_only: false,
            selected_region:  None,
            gallery_page:     0,
            scroll_to_selected: false,
            scroll_to_leaf:     false,
            region_thumbs:    Vec::new(),
            removed:          HashSet::new(),
            struct_undo:      Vec::new(),
            persisted:        HashSet::new(),
            merged_away:      HashSet::new(),
            rejected_leaves:  HashSet::new(),
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
            retrain_base_set:   None,
            retrain_base_rows:  10_000,
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

    pub fn needs_repaint(&self) -> bool { self.running || self.retraining || self.mining }

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
        self.poll_preview(ctx);
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
        egui::SidePanel::left("pipeline_controls")
            .exact_width(ui_kit::CONTROL_W)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("pipeline_ctrl_scroll")
                    .show(ui, |ui| self.show_controls(ui));
            });
        egui::SidePanel::right("pipeline_clusters")
            .default_width(360.0)
            .resizable(true)
            .show_inside(ui, |ui| self.show_cluster_panel(ui, ctx, toasts));
        egui::TopBottomPanel::bottom("pipeline_gallery")
            .resizable(false)
            .min_height(108.0)
            .show_inside(ui, |ui| self.show_gallery(ui, ctx));
        egui::CentralPanel::default().show_inside(ui, |ui| self.show_canvas(ui, ctx, toasts));
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
            if let Some(cid) = self.selected_cluster {
                let name = self.cluster_names.get(&cid).cloned().unwrap_or_else(|| format!("Cluster {cid}"));
                let col = cluster_color(cid);
                let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter().rect_filled(rect, 2.0, Color32::from_rgb(col[0], col[1], col[2]));
                ui.label(format!("Focused: {name}"));
                if ui.small_button("✕ Exit focus").clicked() {
                    self.selected_cluster = None;
                    self.selected_region = None;
                    self.overlay_tex = None;
                }
            } else {
                ui.label(RichText::new("No cluster focused").color(Color32::GRAY));
            }
            ui.separator();
            let has_recon = self.selected_idx.and_then(|i| self.results.get(i))
                .map_or(false, |l| !l.recon_mask.is_empty());
            if has_recon {
                ui.checkbox(&mut self.show_recon, "Show reconstruction")
                    .on_hover_text("Tint (under the anomalies) the area the model reconstructed —\n\
                                    where the leaf was damaged/missing — so you see the whole intact\n\
                                    leaf with the damage as holes.");
                ui.separator();
            }
            if ui.checkbox(&mut self.overlay_outline, "Outline")
                .on_hover_text("Draw cluster OUTLINES (leaf fully visible inside) instead of\n\
                                filled pixels. Same family colours.")
                .changed()
            {
                self.overlay_tex = None;
            }
            // Shown whenever anything on the canvas actually respects this
            // slider: fill-mode regions (not outline mode), OR the
            // reconstruction-preview tint (`show_recon`, which paints in
            // BOTH outline and fill mode).
            if !self.overlay_outline || self.show_recon {
                ui.scope(|ui| {
                    ui.spacing_mut().slider_width = 160.0;
                    if ui.add(egui::Slider::new(&mut self.overlay_alpha, 0.1..=1.0)
                        .text("opacity")
                        .fixed_decimals(2)).changed()
                    {
                        self.overlay_tex = None;
                    }
                });
            }
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
        if self.results.is_empty() || ctx.memory(|m| m.focused().is_some()) {
            return;
        }
        let (prev, next, reject) = ctx.input(|i| (
            i.key_pressed(egui::Key::ArrowLeft),
            i.key_pressed(egui::Key::ArrowRight),
            i.key_pressed(egui::Key::X),
        ));
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

        if rejected {
            ui.label(RichText::new(format!("⚠ Leaf {li} rejected"))
                .color(Color32::from_rgb(220, 110, 110)).strong());
            if ui.button("✓ Restore leaf  (X)")
                .on_hover_text("Put this leaf back into the run — its anomalies count and export again.\n\n\
                                Hotkey: X   ·   ← / → step between leaves")
                .clicked()
            {
                self.toggle_reject_leaf(li, toasts);
            }
        } else {
            let btn = egui::Button::new(
                RichText::new("✕ Reject leaf  (X)").color(Color32::WHITE).strong(),
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
        let tool_btn = |ui: &mut Ui, tool: CanvasTool, icon: &str, name: &str, tip: &str,
                             cur: &mut CanvasTool, switched: &mut Option<CanvasTool>| {
            let active = *cur == tool;
            let text = RichText::new(icon).size(15.0);
            let btn = if active {
                egui::Button::new(text.color(Color32::BLACK)).fill(ui_kit::ACCENT)
            } else {
                egui::Button::new(text)
            };
            if ui.add_sized([30.0, 28.0], btn).on_hover_text(format!("{name}\n{tip}")).clicked()
                && *cur != tool
            {
                *cur = tool;
                *switched = Some(tool);
            }
        };
        ui.horizontal_wrapped(|ui| {
            tool_btn(ui, CanvasTool::Select, icon::CURSOR, "Select",
                "Click to select · drag to box-select · ctrl+click to multi-select · right-click for actions.",
                &mut self.canvas_tool, &mut switched_to);
            tool_btn(ui, CanvasTool::MarkHealthy, icon::CHECK_CIRCLE, "Mark Healthy",
                "Stamp a patch straight off the canvas as a HEALTHY training example — teaches the \
                 model this texture is not an anomaly (e.g. a vein it sometimes confuses with necrosis).",
                &mut self.canvas_tool, &mut switched_to);
            tool_btn(ui, CanvasTool::Brush, icon::PAINT_BRUSH, "Brush",
                "Paint a freeform region using a cluster's color — extends that cluster's region if the \
                 stroke touches one, or creates a new region otherwise.",
                &mut self.canvas_tool, &mut switched_to);
            tool_btn(ui, CanvasTool::Eraser, icon::ERASER, "Eraser",
                "Paint over region pixels to remove them — shrinks the region, or removes it entirely \
                 if nothing's left.",
                &mut self.canvas_tool, &mut switched_to);
            tool_btn(ui, CanvasTool::Knife, icon::KNIFE, "Knife",
                "Drag a straight line starting and ending OUTSIDE any region to split whatever it \
                 crosses into two. Drag a freeform loop starting INSIDE a region to carve that piece \
                 out on its own.",
                &mut self.canvas_tool, &mut switched_to);
            tool_btn(ui, CanvasTool::Scissor, icon::SCISSORS, "Scissor",
                "Click to place vertices instead of dragging — precise, deliberate cuts. Click \
                 back on the FIRST vertex to close the loop and carve it out (needs the first click \
                 inside a region); otherwise press Enter to cut along the open polyline like a bent knife.",
                &mut self.canvas_tool, &mut switched_to);
            tool_btn(ui, CanvasTool::Lasso, icon::LASSO, "Lasso select",
                "Drag a freeform outline; every region whose center falls inside it is added to the \
                 selection (feeds the same Confirm/Reject/Reassign actions as box-select).",
                &mut self.canvas_tool, &mut switched_to);
            tool_btn(ui, CanvasTool::Wand, icon::MAGIC_WAND, "Wand",
                "Click a pixel to grow a mask outward by color similarity (shift-click adds another \
                 blob) — review the pending selection, then \"Fill\" to label + commit it, or \"Clear\" \
                 to discard. A fast alternative to hand-painting with the Brush.",
                &mut self.canvas_tool, &mut switched_to);
            tool_btn(ui, CanvasTool::Eyedropper, icon::EYEDROPPER, "Eyedropper",
                "Hover a region to see its cluster, area, and review status — read-only, doesn't select anything.",
                &mut self.canvas_tool, &mut switched_to);
            tool_btn(ui, CanvasTool::Polygon, icon::POLYGON, "Polygon",
                "Click to place nodes, click the first node again to close the shape and fill it. \
                 With a region selected, it extends that region's own cluster; with nothing \
                 selected, you'll be asked which cluster to assign.",
                &mut self.canvas_tool, &mut switched_to);
        });

        if let Some(tool) = switched_to {
            self.on_tool_switched(tool);
        }

        // ── active tool's own options, directly below its icon — per
        // feedback, NOT a separate bar above the canvas ──
        ui.add_space(4.0);
        egui::Frame::none()
            .fill(Color32::from_gray(28))
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
                            if ui.small_button(format!("↩ Undo ({cur_stamps})"))
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
                    CanvasTool::Eyedropper => {
                        let txt = self.inspect_info.clone().unwrap_or_else(|| "hover a region…".to_string());
                        ui.label(RichText::new(txt).small().color(ui_kit::ACCENT));
                    }
                    CanvasTool::Polygon => {
                        ui.label(RichText::new(
                            "click = place node\nclick first node = close + fill\nEsc = cancel"
                        ).small().color(Color32::GRAY));
                    }
                }
            });

        // ── view controls: zoom is a universal capability now (scroll to
        // zoom, hold middle-mouse to pan — works regardless of active tool),
        // not a tool you switch into. Grouped here with the reconstruction
        // preview toggle since both are "how the canvas is displayed," not
        // an interaction mode. ──
        ui.add_space(4.0);
        ui_kit::section_header(ui, "View");
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("Zoom {:.0}%", self.canvas_zoom * 100.0)).small());
            if ui.small_button("Fit").on_hover_text("Reset zoom/pan (scroll to zoom, hold middle-mouse to pan).").clicked() {
                self.canvas_zoom = 1.0;
                self.canvas_pan = egui::Vec2::ZERO;
            }
        });
    }

    fn show_stepper(&self, ui: &mut Ui) {
        ui.horizontal_centered(|ui| {
            let steps = ["Segment", "Tile", "Detect", "Restitch", "Done"];
            for (i, s) in steps.iter().enumerate() {
                let active = self.stage.starts_with(s) || (self.stage == "Done" && i == steps.len() - 1);
                let col = if active { Color32::from_rgb(120, 200, 130) } else { Color32::GRAY };
                ui.label(RichText::new(*s).color(col).strong());
                if i < steps.len() - 1 {
                    ui.label(RichText::new(">").color(Color32::DARK_GRAY));
                }
            }
            if !self.stage.is_empty() {
                ui.separator();
                ui.label(RichText::new(&self.stage).small().color(Color32::GRAY));
            }
        });
    }

    fn show_controls(&mut self, ui: &mut Ui) {
        self.show_toolbox(ui);
        ui.separator();
        ui_kit::section_header(ui, "Folders");
        self.pick_row(ui, "Source folder", Pick::Source);
        if self.source_folder.is_some() {
            ui.label(RichText::new(format!("{} images found", self.source_count)).small());
        }
        self.pick_row(ui, "Output folder", Pick::Output);

        // Segmentation edge preview (runs YOLO on the first source image so you can
        // check the cutout edge before committing to a full pipeline run; tune the
        // underlying thresholds in Settings > Pipeline).
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            let can_preview = self.eff_yolo().is_some() && self.source_count > 0
                && !self.preview_busy && !self.running;
            if ui.add_enabled(can_preview, egui::Button::new("🔍 Preview segmentation")).clicked() {
                self.start_preview();
            }
            if self.preview_busy { ui_kit::busy(ui, "segmenting…"); }
        });
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
        let can_start = self.all_paths_ok() && self.source_count > 0 && !self.running && !self.retraining && !self.mining;
        ui.add_enabled_ui(can_start, |ui| {
            if ui_kit::primary_button(ui, "Run Pipeline").clicked() {
                self.start();
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
            (None, Some(p)) => (format!("inherits: {}", p.display()), ui_kit::ACCENT),
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
            ui.label(RichText::new(format!("Leaves — {} done", self.results.len()))
                .small().color(Color32::GRAY));
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
                    let hover = if rejected {
                        format!("leaf {i} — REJECTED ({n} regions excluded)")
                    } else {
                        format!("leaf {i} — {n} regions")
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
                self.hardneg_label = self.selected_cluster
                    .and_then(|c| self.cluster_names.get(&c).cloned())
                    .unwrap_or_default();
            }
            _ => {}
        }
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
            // Before a run: show the segmentation preview (if any) so the cutout edge
            // can be judged; otherwise the get-started hint.
            if let Some(pv) = self.preview_tex.clone() {
                let avail = ui.available_size();
                let sz = pv.size_vec2();
                let scale = (avail.x / sz.x).min(avail.y / sz.y).min(1.0).max(0.01);
                let disp = sz * scale;
                let (area, _) = ui.allocate_exact_size(avail, egui::Sense::hover());
                let rect = egui::Rect::from_center_size(area.center(), disp);
                egui::Image::new((pv.id(), disp)).paint_at(ui, rect);
                return;
            }
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
            let sel = self.selected_cluster;
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
                let col = if focused { cluster_color(self.labels[ri]) } else { dim_color(cluster_color(self.labels[ri])) };
                let width = if focused { 2.0 } else { 1.2 };
                ui.painter().add(egui::Shape::closed_line(
                    sm, egui::Stroke::new(width, Color32::from_rgb(col[0], col[1], col[2])),
                ));
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
            // live cursor-following preview shape
            if let Some((cx, cy)) = hover_leaf {
                let mn = img_rect.min + egui::vec2((cx - half as f32) * s, (cy - half as f32) * s);
                let sz = (self.brush_size as f32) * s;
                match self.brush_shape {
                    BrushShape::Square => {
                        ui.painter().rect_stroke(egui::Rect::from_min_size(mn, egui::vec2(sz, sz)), 0.0,
                            egui::Stroke::new(2.0, Color32::from_rgb(255, 150, 60)));
                    }
                    BrushShape::Circle => {
                        ui.painter().circle_stroke(mn + egui::vec2(sz, sz) / 2.0, sz / 2.0,
                            egui::Stroke::new(2.0, Color32::from_rgb(255, 150, 60)));
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
            // paint the stroke LIVE as it's built, not just once on release —
            // otherwise nothing visibly happens until the mouse button comes
            // up, which reads as a broken/delayed tool.
            if !self.brush_stroke.is_empty() {
                let px_sz = s.max(1.0);
                for &(px, py) in &self.brush_stroke {
                    let mn = img_rect.min + egui::vec2(px as f32 * s, py as f32 * s);
                    ui.painter().rect_filled(
                        egui::Rect::from_min_size(mn, egui::vec2(px_sz, px_sz)),
                        0.0, Color32::from_rgba_unmultiplied(255, 150, 60, 110),
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
        CanvasTool::Eyedropper => {
            // ── read-only inspect: hover updates a readout in the options
            // bar, never touches selection/confirm/reject state ──
            self.inspect_info = resp.hover_pos().and_then(|p| {
                let lx = (p.x - img_rect.min.x) / s.max(1e-3);
                let ly = (p.y - img_rect.min.y) / s.max(1e-3);
                self.region_at(leaf_idx, lx, ly).map(|i| {
                    let cid = self.labels[i];
                    let family = self.cluster_names.get(&cid).cloned().unwrap_or_else(|| format!("Cluster {cid}"));
                    let area = self.region_area.get(i).copied().unwrap_or(0);
                    let status = if self.removed.contains(&i) { "rejected" }
                        else if self.persisted.contains(&i) { "confirmed" }
                        else { "unreviewed" };
                    format!("Region {i} · {family} · {area}px · {status}")
                })
            });
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
        let focused = ctx.memory(|m| m.focused().is_some());
        if !focused {
            let key_tool = ui.input(|i| {
                if i.key_pressed(egui::Key::V) { Some(CanvasTool::Select) }
                else if i.key_pressed(egui::Key::H) { Some(CanvasTool::MarkHealthy) }
                else if i.key_pressed(egui::Key::B) { Some(CanvasTool::Brush) }
                else if i.key_pressed(egui::Key::E) { Some(CanvasTool::Eraser) }
                else if i.key_pressed(egui::Key::K) { Some(CanvasTool::Knife) }
                else if i.key_pressed(egui::Key::S) { Some(CanvasTool::Scissor) }
                else if i.key_pressed(egui::Key::L) { Some(CanvasTool::Lasso) }
                else if i.key_pressed(egui::Key::W) { Some(CanvasTool::Wand) }
                else if i.key_pressed(egui::Key::I) { Some(CanvasTool::Eyedropper) }
                else if i.key_pressed(egui::Key::P) { Some(CanvasTool::Polygon) }
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
                if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    do_confirm = true;
                }
                if ui.input(|i| i.key_pressed(egui::Key::Delete)) {
                    do_remove = true;
                }
                if ui.input(|i| i.key_pressed(egui::Key::R)) {
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
            egui::Stroke::new(1.5, ui_kit::ACCENT));
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
        let sel = self.selected_cluster;
        let key = (idx, sel, self.show_recon, (self.overlay_alpha * 100.0) as u32, self.overlay_outline);
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

    fn show_cluster_panel(&mut self, ui: &mut Ui, ctx: &Context, toasts: &mut ToastManager) {
        ui.add_space(4.0);
        // ── tabs: split what used to be one long scrolling column (leaf/
        // morphology / stats / curation / retrain / export / log all
        // stacked) into focused views ──
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.cluster_panel_tab, ClusterPanelTab::Leaf, "Metrics");
            ui.selectable_value(&mut self.cluster_panel_tab, ClusterPanelTab::Clusters, "Clusters");
            ui.selectable_value(&mut self.cluster_panel_tab, ClusterPanelTab::Curate, "Curate");
            ui.selectable_value(&mut self.cluster_panel_tab, ClusterPanelTab::Log, "Log");
        });
        ui.separator();

        match self.cluster_panel_tab {
            ClusterPanelTab::Leaf => self.show_leaf_tab(ui),
            ClusterPanelTab::Clusters => self.show_clusters_tab(ui, toasts),
            ClusterPanelTab::Curate => self.show_curate_tab(ui, ctx, toasts),
            ClusterPanelTab::Log => self.show_log_tab(ui),
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
        let sel = self.selected_cluster;
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
                .color(ui_kit::ACCENT),
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
                        let col = cluster_color(cid);
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
                                let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                                ui.painter().rect_filled(rect, 2.0, Color32::from_rgb(col[0], col[1], col[2]));
                                let mut name = self.cluster_names.get(&cid).cloned()
                                    .unwrap_or_else(|| format!("Cluster {cid}"));
                                if ui.add(egui::TextEdit::singleline(&mut name).desired_width(190.0)).changed() {
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
                                if delete {
                                    self.delete_cluster_from_head(cid, toasts);
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

        // ── filter by cluster directly, without needing to click a scatter
        // point or an existing thumbnail first ──
        ui.horizontal(|ui| {
            ui.label(RichText::new("Filter:").small());
            let cur_label = self.selected_cluster
                .map(|c| self.cluster_names.get(&c).cloned().unwrap_or_else(|| format!("Cluster {c}")))
                .unwrap_or_else(|| "All clusters".to_string());
            egui::ComboBox::from_id_salt("curate_cluster_filter")
                .selected_text(cur_label)
                .show_ui(ui, |ui| {
                    if ui.selectable_label(self.selected_cluster.is_none(), "All clusters").clicked() {
                        self.selected_cluster = None;
                    }
                    for c in &self.clusters {
                        let name = self.cluster_names.get(&c.id).cloned().unwrap_or_else(|| format!("Cluster {}", c.id));
                        if ui.selectable_label(self.selected_cluster == Some(c.id), name).clicked() {
                            self.selected_cluster = Some(c.id);
                        }
                    }
                });
            ui.add_enabled_ui(self.selected_idx.is_some(), |ui| {
                ui.checkbox(&mut self.filter_leaf_only, "This leaf only")
                    .on_hover_text("Show only the regions on the leaf currently open in the \
                                    Leaf view — small regions are easy to miss in the full \
                                    dataset gallery, which makes it hard to tell when a single \
                                    leaf is actually fully reviewed.");
            });
        });

        // Per-leaf review status — the concrete "is this leaf done" readout
        // the dataset-wide counts above can't give you, and what the new
        // unmarked-leaf-area miner (below) actually gates on per leaf.
        if let Some(li) = self.selected_idx {
            let leaf_regions: Vec<usize> = self.regions.iter().enumerate()
                .filter(|(_, r)| r.leaf == li).map(|(i, _)| i).collect();
            if !leaf_regions.is_empty() {
                let pending = leaf_regions.iter()
                    .filter(|&&i| self.region_visible(i) && !self.persisted.contains(&i))
                    .count();
                let (msg, color) = if pending == 0 {
                    (format!("Leaf {li}: {} region(s), fully reviewed ✅", leaf_regions.len()), Color32::from_rgb(140, 230, 150))
                } else {
                    (format!("Leaf {li}: {pending} of {} region(s) still pending", leaf_regions.len()), Color32::from_rgb(230, 190, 90))
                };
                ui.label(RichText::new(msg).small().color(color));
            }
        }
        ui.separator();

        // ── flywheel: every Confirm/Reject/Reassign below already writes to
        // curations/ the instant it happens — nothing here is "unsaved." This
        // button is a bulk accelerator for the common case of fast-approving
        // everything you didn't individually touch. ──
        ui.horizontal(|ui| {
            let (n_persisted, n_removed) = (self.persisted.len(), self.removed.len());
            let n_unreviewed = (0..self.regions.len())
                .filter(|&i| !self.persisted.contains(&i) && self.region_visible(i))
                .count();
            ui.label(RichText::new(format!(
                "{n_persisted} confirmed · {n_removed} rejected · {n_unreviewed} unreviewed"
            )).small().color(Color32::GRAY));
        });
        if ui.button("✅ Confirm all remaining")
            .on_hover_text("Confirm every region you haven't individually reviewed yet\n\
                            (writes each one's current cluster name to curations/ now).\n\
                            Confirm/Reject/Reassign already save immediately as you do them —\n\
                            this is just a shortcut for the rest.")
            .clicked()
        {
            self.confirm_all_remaining(toasts);
        }
        ui.separator();

        self.show_mine_hardneg(ui);
        ui.separator();

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
                            self.reset_run_state();
                            toasts.info("Switched to the retrained head — click Run Pipeline to see corrected results.");
                        }
                        if ui.button("Dismiss").clicked() {
                            self.retrain_done = None;
                        }
                    });
                });
            ui.add_space(4.0);
        }
        let can_retrain = self.output_folder.is_some() && self.eff_head().is_some()
            && self.eff_dino().is_some() && !self.retraining && !self.running && !self.mining;
        self.pick_row(ui, "Base training set (.bin)", Pick::BaseSet);
        ui.horizontal(|ui| {
            ui.label("base rows");
            ui.add(egui::DragValue::new(&mut self.retrain_base_rows)
                .range(0..=400_000).speed(1000));
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
            }
        });
        if self.running && !self.retraining {
            ui.label(RichText::new("Retrain is disabled while the pipeline is running.")
                .small().color(Color32::GRAY));
        }
        if !self.retrain_log.is_empty() {
            egui::ScrollArea::vertical().max_height(80.0).id_salt("pipeline_retrain_log").show(ui, |ui| {
                for line in self.retrain_log.iter().rev().take(20) {
                    ui.label(RichText::new(line).small());
                }
            });
        }
        ui.separator();
        if ui.button("📤 Export results (CSV + images)")
            .on_hover_text("Write <output>/export/: results.csv (ONE row per anomaly — cluster, \n\
                            region stats, Recon %, AND the leaf's morphology, all in one file), \n\
                            crops/ (each anomaly image) and leaves/ (each leaf with anomalies \n\
                            colour-coded by family).")
            .clicked()
        {
            self.export_results(toasts);
        }
        ui.separator();

        // ── anomaly gallery (filtered to the selected cluster, paginated) ──
        const PER_PAGE: usize = GALLERY_PER_PAGE;
        let mut filtered: Vec<usize> = (0..self.regions.len())
            .filter(|&i| {
                self.region_visible(i)
                    && self.selected_cluster.map_or(true, |c| self.labels[i] == c)
                    && (!self.filter_leaf_only || self.selected_idx.map_or(true, |li| self.regions[i].leaf == li))
            })
            .collect();
        // group by cluster (stable within each cluster, so pagination stays
        // predictable) — matters when no single cluster is selected, so the
        // gallery doesn't interleave unrelated families in raw detection order.
        filtered.sort_by_key(|&i| self.labels[i]);
        let total = filtered.len();
        let n_pages = total.div_ceil(PER_PAGE).max(1);
        if self.gallery_page >= n_pages {
            self.gallery_page = 0;
        }
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("Anomalies — {total} total")).small().color(Color32::GRAY));
            if n_pages > 1 {
                if ui.small_button("◀").clicked() && self.gallery_page > 0 {
                    self.gallery_page -= 1;
                }
                ui.label(RichText::new(format!("page {}/{}", self.gallery_page + 1, n_pages)).small());
                if ui.small_button("▶").clicked() && self.gallery_page + 1 < n_pages {
                    self.gallery_page += 1;
                }
            }
            ui.add_enabled_ui(!self.struct_undo.is_empty(), |ui| {
                if ui.small_button(format!("↩ Undo ({})", self.struct_undo.len()))
                    .on_hover_text("Undo the most recent remove or knife cut (Ctrl+Z).")
                    .clicked()
                {
                    self.undo_last_edit(toasts);
                }
            });
        });
        ui.label(RichText::new("click = highlight on leaf · right-click = reject · ctrl+click = multi-select \
                                 · Enter = confirm · Delete = reject")
            .small().color(Color32::DARK_GRAY));
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
        egui::ScrollArea::vertical().id_salt("anomaly_gallery").show(ui, |ui| {
            // grouped by cluster (show_idxs is pre-sorted by label above) —
            // a small heading per run so the sort is visually obvious, not
            // just an implicit ordering.
            let mut idx = 0;
            while idx < show_idxs.len() {
                let cid = self.labels[show_idxs[idx]];
                let mut end = idx + 1;
                while end < show_idxs.len() && self.labels[show_idxs[end]] == cid {
                    end += 1;
                }
                let col = cluster_color(cid);
                let name = self.cluster_names.get(&cid).cloned().unwrap_or_else(|| format!("Cluster {cid}"));
                ui.horizontal(|ui| {
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                    ui.painter().rect_filled(rect, 2.0, Color32::from_rgb(col[0], col[1], col[2]));
                    ui.label(RichText::new(format!("{name} ({})", end - idx)).small().strong());
                });
                ui.horizontal_wrapped(|ui| {
                    for &i in &show_idxs[idx..end] {
                        if let Some(tex) = &self.region_thumbs[i] {
                            let resp = ui
                                .add(egui::ImageButton::new((tex.id(), egui::vec2(48.0, 48.0))))
                                .on_hover_text(format!("region {i} · leaf {}", self.regions[i].leaf));
                            if self.selected_region == Some(i) {
                                ui.painter().rect_stroke(resp.rect, 2.0,
                                    egui::Stroke::new(2.0, Color32::from_rgb(255, 230, 0)));
                                if self.scroll_to_selected {
                                    resp.scroll_to_me(Some(egui::Align::Center));
                                    self.scroll_to_selected = false;
                                }
                            }
                            if self.multi_selected.contains(&i) {
                                ui.painter().rect_stroke(resp.rect, 2.0,
                                    egui::Stroke::new(2.0, Color32::from_rgb(80, 170, 255)));
                            }
                            if resp.clicked() {
                                if ui.input(|inp| inp.modifiers.shift) {
                                    // Range-select from the last-clicked tile to this
                                    // one, in the gallery's own cluster-sorted order
                                    // (`show_idxs`) — not insertion/region-id order.
                                    let range = self.last_clicked_region.and_then(|anchor| {
                                        let a = show_idxs.iter().position(|&x| x == anchor)?;
                                        let b = show_idxs.iter().position(|&x| x == i)?;
                                        Some((a.min(b), a.max(b)))
                                    });
                                    if let Some((lo, hi)) = range {
                                        for &ri in &show_idxs[lo..=hi] {
                                            self.multi_selected.insert(ri);
                                        }
                                    } else {
                                        self.multi_selected.insert(i);
                                    }
                                } else if ui.input(|inp| inp.modifiers.ctrl) {
                                    self.toggle_multi_select(i);
                                } else {
                                    self.selected_idx = Some(self.regions[i].leaf);
                                    self.selected_cluster = Some(self.labels[i]);
                                    self.selected_region = Some(i);
                                    self.overlay_tex = None;
                                }
                                self.last_clicked_region = Some(i);
                            }
                            if resp.secondary_clicked() {
                                self.remove_regions(&[i], toasts, true);
                                self.multi_selected.remove(&i);
                                if self.selected_region == Some(i) {
                                    self.selected_region = None;
                                }
                            }
                        }
                    }
                });
                ui.add_space(4.0);
                idx = end;
            }
        });
    }

    /// Pipeline run log — moved here from the bottom of the folders panel
    /// per feedback ("we can move the log somewhere else").
    fn show_log_tab(&self, ui: &mut Ui) {
        if self.log.is_empty() {
            ui.label(RichText::new("No log entries yet — run the pipeline to see output here.")
                .small().color(Color32::GRAY));
            return;
        }
        egui::ScrollArea::vertical().id_salt("pipeline_run_log").show(ui, |ui| {
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

    /// Toggle whole-leaf rejection. Reversible on purpose — it is one click in a
    /// top bar, next to the destructive-looking controls, and an accidental press
    /// would otherwise silently drop a leaf's anomalies from the export with no
    /// way back short of re-running the pipeline.
    fn toggle_reject_leaf(&mut self, li: usize, toasts: &mut ToastManager) {
        if self.rejected_leaves.remove(&li) {
            toasts.success(format!("Leaf {li} restored"));
        } else {
            self.rejected_leaves.insert(li);
            let n = self.regions.iter().filter(|r| r.leaf == li).count();
            toasts.success(format!("Leaf {li} rejected — {n} anomalies excluded"));
        }
        self.overlay_tex = None;  // canvas overlay is cached; force a repaint
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
        self.log.clear();
        self.cancel_flag = Arc::new(AtomicBool::new(false));
        self.progress_done = 0;
        self.progress_total = image_paths.len();
        self.running = true;
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
                dino_res: 512,
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

    fn start_preview(&mut self) {
        let (Some(yolo), Some(src)) = (self.eff_yolo(), self.source_folder.clone()) else { return };
        let images = crate::tabs::leaf_seg::inference::list_images(&src);
        let Some(first) = images.into_iter().next() else {
            self.preview_note = "No source images found.".into();
            return;
        };
        let (alpha_lo, chroma_min) = (self.seg_alpha_lo, self.seg_chroma_min);
        let (tx, rx) = mpsc::channel();
        self.preview_rx = Some(rx);
        self.preview_busy = true;
        self.preview_note = "Running YOLO on the first image…".into();
        std::thread::spawn(move || {
            let _ = tx.send(crate::tabs::leaf_seg::inference::preview_cutout(&yolo, &first, alpha_lo, chroma_min));
        });
    }

    fn poll_preview(&mut self, ctx: &Context) {
        if self.preview_busy { ctx.request_repaint(); }
        let msg = self.preview_rx.as_ref().and_then(|rx| rx.try_recv().ok());
        if let Some(res) = msg {
            self.preview_busy = false;
            self.preview_rx = None;
            match res {
                Ok((rgba, w, h)) => {
                    let ci = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
                    self.preview_tex = Some(ctx.load_texture("seg_preview", ci, egui::TextureOptions::LINEAR));
                    self.preview_note =
                        format!("Preview {w}×{h} — tune the sliders + re-preview; happy → Run Pipeline.");
                }
                Err(e) => self.preview_note = format!("Preview failed: {e}"),
            }
        }
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
                dino_res: 512,
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
        if got_clusters {
            self.build_clusters(toasts);
            self.overlay_tex = None; // rebuild to reflect cluster colours
        }
        if finished {
            self.running = false;
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
                base_rows: self.retrain_base_rows,
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
            if let Some(img) = image::RgbaImage::from_raw(crop_size, crop_size, crop_bytes) {
                let _ = img.save(labels_dir.join(&fname));
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
        match std::fs::OpenOptions::new().create(true).append(true).open(&jsonl_path) {
            Ok(mut f) => {
                let _ = f.write_all(lines.as_bytes());
                self.persisted.insert(idx);
            }
            Err(e) => toasts.error(format!("labels.jsonl: {e}")),
        }
    }

    /// Explicit "I reviewed this and it's correct" gesture — persists each
    /// region's CURRENT cluster name immediately. Unlike Reject, this never
    /// touches `labels`/`removed` (a confirmed region's family is just
    /// whatever it already is); it only catches disk state up to what's shown.
    fn confirm_regions(&mut self, ids: &[usize], toasts: &mut ToastManager) {
        for &i in ids {
            if !self.region_visible(i) { continue; }
            let cid = self.labels[i];
            if cid < 0 { continue; } // nothing meaningful to confirm yet
            let family = self.cluster_names.get(&cid).cloned().unwrap_or_else(|| format!("Cluster {cid}"));
            self.persist_region(i, &family, false, toasts);
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
        toasts.success(format!("Confirmed {n} remaining region(s) → curations/"));
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
    fn export_results(&self, toasts: &mut ToastManager) {
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
        if std::fs::create_dir_all(&crops_dir).and(std::fs::create_dir_all(&leaves_dir)).is_err() {
            toasts.error("could not create export folder");
            return;
        }

        let mut csv = String::from(
            "leaf,leaf_src,region,cluster_id,family,area_px,pct_leaf,recon_pct,lost_tissue_pct,\
             bbox_x,bbox_y,bbox_w,bbox_h,crop_file,\
             ec_length,ec_width,ec_area,ec_shape_index,ec_circularity,ec_entropy,ec_outline,\
             mc_length,mc_width,mc_area,mc_shape_index,mc_circularity,mc_entropy,mc_outline\n",
        );
        let mut n = 0usize;
        for (i, r) in self.regions.iter().enumerate() {
            if !self.region_visible(i) {
                continue;
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
            let crop_file = format!("{leaf}_{i}.png");
            if let Some(img) = image::RgbaImage::from_raw(r.crop_size, r.crop_size, r.crop.clone()) {
                let _ = img.save(crops_dir.join(&crop_file));
            }
            let mut cols = vec![
                leaf.to_string(), csv_escape(&src), i.to_string(), cid.to_string(), csv_escape(&fam),
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
            csv.push_str(&cols.join(","));
            csv.push('\n');
            n += 1;
        }

        // per-leaf overlays (anomalies colour-coded by family). The CSV loop above
        // inherits rejection through `region_visible`; this one iterates LEAVES,
        // so it needs the check itself or a rejected leaf would still ship an
        // image (an empty one, which reads as "analysed, nothing found").
        for (li, leaf) in self.results.iter().enumerate() {
            if self.rejected_leaves.contains(&li) { continue; }
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
                let _ = img.save(leaves_dir.join(format!("{stem}_{li}.png")));
            }
        }

        match std::fs::write(dir.join("results.csv"), csv) {
            // Name the excluded leaves explicitly. A silently smaller export is
            // exactly the kind of thing that gets noticed months later, in the
            // stats, with no way left to tell whether it was intentional.
            Ok(_) => {
                let skipped = self.rejected_leaves.len();
                let note = if skipped > 0 {
                    format!(" ({skipped} rejected leaves excluded)")
                } else {
                    String::new()
                };
                toasts.success(format!("Exported {n} anomalies + images → export/{note}"))
            }
            Err(e) => toasts.error(format!("write results.csv: {e}")),
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

    fn finish_brush_stroke(&mut self, leaf_idx: usize, toasts: &mut ToastManager) {
        let pts = std::mem::take(&mut self.brush_stroke);
        if pts.is_empty() {
            return;
        }
        let label_name = self.hardneg_label.trim().to_string();
        if label_name.is_empty() {
            toasts.error("Type a cluster name first.");
            return;
        }

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

        let idx = if touched.is_empty() {
            new_idx
        } else {
            let mut group = touched;
            group.push(new_idx);
            self.merge_region_group(&group, toasts);
            *group.iter().min().unwrap()
        };

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
        for i in 0..self.regions.len() {
            if self.regions[i].leaf != leaf_idx || !self.region_visible(i) {
                continue;
            }
            let [bx, by, bw, bh] = self.regions[i].bbox_leaf;
            let mut touched = false;
            for &(x, y) in &pts {
                if x < bx as i32 || y < by as i32 || x >= (bx + bw) as i32 || y >= (by + bh) as i32 {
                    continue;
                }
                let (mx, my) = ((x - bx as i32) as u32, (y - by as i32) as u32);
                let mi = (my * bw + mx) as usize;
                if self.regions[i].mask[mi] {
                    self.regions[i].mask[mi] = false;
                    touched = true;
                }
            }
            if !touched {
                continue;
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
        if !emptied.is_empty() {
            self.remove_regions(&emptied, toasts, false);
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
                    if let Some(i) = self.region_at(leaf_idx, sx, sy) {
                        self.do_polycut(leaf_idx, i, &poly, toasts);
                    }
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
        if self.lasso_points.len() >= 2 && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
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
        for &i in ids {
            self.removed.insert(i);
            if write_reject {
                self.persist_region(i, "rejected", true, toasts);
            } else if self.persisted.contains(&i) {
                self.retract_persisted(i);
            }
        }
        self.push_undo(UndoEntry::Remove(ids.to_vec()));
        self.overlay_tex = None;
    }

    /// Push a new undo entry, capping the stack the same way the old
    /// remove-only stack did.
    fn push_undo(&mut self, entry: UndoEntry) {
        self.struct_undo.push(entry);
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
        match entry {
            UndoEntry::Remove(ids) => {
                for &i in &ids {
                    self.removed.remove(&i);
                    // undo the disk write too — a restored region goes back to
                    // "unreviewed," not "rejected," so its persisted reject line
                    // (if any) shouldn't linger; the stable region_{i}.png filename
                    // means this is a clean delete, no orphan left behind.
                    self.retract_persisted(i);
                }
                toasts.success(format!("Restored {} region(s)", ids.len()));
            }
            UndoEntry::Cut { originals, created } => {
                // Cut pieces were never independently rejected/persisted as
                // their own reject event, so undoing is a pure visibility
                // flip: hide the pieces, restore the pre-cut region(s).
                for &i in &created {
                    self.merged_away.insert(i);
                }
                for &i in &originals {
                    self.merged_away.remove(&i);
                }
                toasts.success("Undid knife cut");
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
            Pick::Yolo | Pick::Dino => rfd::FileDialog::new().add_filter("ONNX", &["onnx"]).pick_file(),
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
