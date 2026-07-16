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
pub mod worker;

use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{mpsc, Arc},
    sync::atomic::{AtomicBool, Ordering},
};

use egui::{Color32, Context, RichText, Ui};
use egui_phosphor::regular as icon;
use egui_plot::{Plot, Points};

use crate::settings::{AppDefaults, AppSettings};
use crate::tabs::leaf_seg::inference::{list_images, scan_image_count};
use crate::tabs::train::head::{spawn_retrain, RetrainCfg, RetrainMsg};
use crate::ui_kit;
use crate::widgets::ToastManager;
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
enum Pick { Source, Output, Yolo, Dino, Bank, Meta, Recon, Head }

/// Active canvas tool — Photoshop-style, mutually exclusive, and always
/// visibly indicated (see `show_toolbox`/the canvas options bar) so the same
/// click/right-click gesture never silently means two different things.
#[derive(Clone, Copy, PartialEq)]
enum CanvasTool { Select, MarkHealthy, Brush, Lasso, Eyedropper }

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
    head_tau:        f32,
    head_grow:       f32,
    tile_size:       u32,
    margin_erode_px: u32,
    conf:            f32,
    seg_alpha_lo:    f32,   // YOLO cutout edge tightness (feather start)
    seg_chroma_min:  i32,   // YOLO cutout background-chroma rejection
    cluster_eps:     f32,   // DBSCAN radius; lower = more/smaller/looser clusters
    cluster_min_pts: usize, // DBSCAN min points; lower = more/smaller/looser clusters

    // segmentation preview (tune the cutout edge before a full run)
    preview_tex:  Option<egui::TextureHandle>,
    preview_rx:   Option<mpsc::Receiver<Result<(Vec<u8>, u32, u32), String>>>,
    preview_busy: bool,
    preview_note: String,

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
    cluster_panel_tab: ClusterPanelTab,

    // clustering (filled by PipeMsg::Clusters)
    regions:          Vec<AnomalyRegion>,
    region_area:      Vec<u32>,             // parallel to regions; cached mask pixel count
    labels:           Vec<i32>,             // parallel to regions
    coords:           Vec<[f32; 2]>,        // PCA-2, parallel to regions
    clusters:         Vec<ClusterInfo>,
    selected_cluster: Option<i32>,
    selected_region:  Option<usize>,   // anomaly highlighted with a bbox on the leaf
    gallery_page:     usize,           // anomaly gallery pagination
    scroll_to_selected: bool,          // one-shot: scroll the gallery to selected_region
    region_thumbs:    Vec<Option<egui::TextureHandle>>, // parallel to regions
    removed:          HashSet<usize>,       // region indices removed by the user
    remove_undo:      Vec<Vec<usize>>,      // undo stack: each entry = one removal action's ids
    persisted:        HashSet<usize>,       // region indices already written to curations/labels.jsonl this run
    // region indices absorbed into another region by merge_touching_regions.
    // `regions` is append-only — a merge never removes/renumbers entries (that
    // would require touching every index-keyed field below); the survivor's
    // own mask/bbox/crop get updated in place instead, and every other member
    // of the merge group lands here. Semantically distinct from `removed`
    // (not "rejected," never subject to remove_undo) — see `region_visible`.
    merged_away:      HashSet<usize>,
    cluster_names:    HashMap<i32, String>,
    multi_selected:   HashSet<usize>,       // gallery tiles OR canvas rubber-band picks, for bulk reassign
    reassign_name:    String,               // target cluster name typed for bulk reassign
    canvas_drag_start: Option<egui::Pos2>,  // rubber-band select drag, screen-space (normal mode only)

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
            head_tau:        0.85,
            head_grow:       0.7,
            tile_size:       256,
            margin_erode_px: 6,
            conf:            0.25,
            seg_alpha_lo:    0.0,
            seg_chroma_min:  0,
            cluster_eps:     1.5,
            cluster_min_pts: 5,

            preview_tex:  None,
            preview_rx:   None,
            preview_busy: false,
            preview_note: String::new(),

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
            cluster_panel_tab: ClusterPanelTab::Curate,

            regions:          Vec::new(),
            region_area:      Vec::new(),
            labels:           Vec::new(),
            coords:           Vec::new(),
            clusters:         Vec::new(),
            selected_cluster: None,
            selected_region:  None,
            gallery_page:     0,
            scroll_to_selected: false,
            region_thumbs:    Vec::new(),
            removed:          HashSet::new(),
            remove_undo:      Vec::new(),
            persisted:        HashSet::new(),
            merged_away:      HashSet::new(),
            cluster_names:    HashMap::new(),
            multi_selected:   HashSet::new(),
            reassign_name:    String::new(),
            canvas_drag_start: None,

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

    pub fn needs_repaint(&self) -> bool { self.running || self.retraining }

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
        r.head_tau           = self.head_tau;
        r.head_grow          = self.head_grow;
        r.tile_size          = self.tile_size;
        r.margin_erode_px    = self.margin_erode_px;
        r.cluster_eps        = self.cluster_eps;
        r.cluster_min_pts    = self.cluster_min_pts;
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
        self.head_tau      = r.head_tau;
        self.head_grow     = r.head_grow;
        self.tile_size     = r.tile_size;
        self.margin_erode_px = r.margin_erode_px;
        self.cluster_eps     = r.cluster_eps;
        self.cluster_min_pts = r.cluster_min_pts;
        if let Some(f) = self.source_folder.clone() {
            self.source_count = scan_image_count(&f);
        }
    }

    // ── show ──────────────────────────────────────────────────────────────

    pub fn show(&mut self, ui: &mut Ui, ctx: &Context, toasts: &mut ToastManager) {
        self.poll_pick();
        self.poll_worker(toasts);
        self.poll_retrain(toasts);
        self.poll_preview(ctx);

        egui::TopBottomPanel::top("pipeline_stepper")
            .exact_height(28.0)
            .show_inside(ui, |ui| self.show_stepper(ui));
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
            tool_btn(ui, CanvasTool::Lasso, icon::LASSO, "Lasso select",
                "Drag a freeform outline; every region whose center falls inside it is added to the \
                 selection (feeds the same Confirm/Reject/Reassign actions as box-select).",
                &mut self.canvas_tool, &mut switched_to);
            tool_btn(ui, CanvasTool::Eyedropper, icon::EYEDROPPER, "Eyedropper",
                "Hover a region to see its cluster, area, and review status — read-only, doesn't select anything.",
                &mut self.canvas_tool, &mut switched_to);
        });

        match switched_to {
            Some(CanvasTool::MarkHealthy) => self.hardneg_label = "healthy".to_string(),
            Some(CanvasTool::Brush) => {
                self.hardneg_label = self.selected_cluster
                    .and_then(|c| self.cluster_names.get(&c).cloned())
                    .unwrap_or_default();
            }
            _ => {}
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
                        ui.label(RichText::new("Label: \"healthy\"").small().color(Color32::GRAY));
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
                            for c in &self.clusters {
                                if c.id < 0 {
                                    continue; // skip "noise"
                                }
                                let col = cluster_color(c.id);
                                let name = self.cluster_names.get(&c.id).cloned()
                                    .unwrap_or_else(|| format!("Cluster {}", c.id));
                                let selected = self.hardneg_label.eq_ignore_ascii_case(&name);
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
                                    self.hardneg_label = name;
                                }
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
                    CanvasTool::Lasso => {
                        ui.label(RichText::new(
                            "drag a freeform outline; release to select every region whose center falls inside it"
                        ).small().color(Color32::GRAY));
                    }
                    CanvasTool::Eyedropper => {
                        let txt = self.inspect_info.clone().unwrap_or_else(|| "hover a region…".to_string());
                        ui.label(RichText::new(txt).small().color(ui_kit::ACCENT));
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
        let has_recon = self.selected_idx.and_then(|i| self.results.get(i))
            .map_or(false, |l| !l.recon_mask.is_empty());
        if has_recon {
            ui.checkbox(&mut self.show_recon, "Show reconstruction")
                .on_hover_text("Tint (under the anomalies) the area the model reconstructed —\n\
                                where the leaf was damaged/missing — so you see the whole intact\n\
                                leaf with the damage as holes.");
        }
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
        let can_start = self.all_paths_ok() && self.source_count > 0 && !self.running && !self.retraining;
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
        ui.label(RichText::new("Only affects PatchCore-only runs (no few-shot head) — \
                                 few-shot regions are already grouped by their trained family.")
            .small().color(Color32::GRAY));
        egui::Grid::new("pipeline_clustering").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
            ui.label("Cluster radius (eps):")
                .on_hover_text("DBSCAN neighborhood radius over standardized region\n\
                                descriptors. LOWER = more, smaller clusters (looser —\n\
                                similar-but-distinct anomalies split apart more readily).\n\
                                Higher = fewer, broader clusters. Default 1.5.");
            ui.add(egui::Slider::new(&mut self.cluster_eps, 0.5..=3.0).fixed_decimals(2));
            ui.end_row();
            ui.label("Cluster min points:")
                .on_hover_text("DBSCAN minimum neighbors to seed a cluster. Lower = more\n\
                                clusters form (including from just a couple similar\n\
                                regions); higher = only well-populated groups survive.\n\
                                Default 5.");
            ui.add(egui::Slider::new(&mut self.cluster_min_pts, 2..=10));
            ui.end_row();
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
                    let resp = ui
                        .add(egui::ImageButton::new((tex.id(), tex.size_vec2())))
                        .on_hover_text(format!("leaf {i} — {n} regions"));
                    if self.selected_idx == Some(i) {
                        ui.painter().rect_stroke(
                            resp.rect, 3.0,
                            egui::Stroke::new(2.0, Color32::from_rgb(120, 200, 130)),
                        );
                    }
                    if resp.clicked() {
                        self.selected_idx = Some(i);
                        self.overlay_tex = None;
                    }
                }
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

        // outline mode: draw smooth vector contours of the visible regions
        if self.overlay_outline {
            let sel = self.selected_cluster;
            for (ri, r) in self.regions.iter().enumerate() {
                if r.leaf != leaf_idx || !self.region_visible(ri) {
                    continue;
                }
                if let Some(cid) = sel {
                    if self.labels[ri] != cid {
                        continue;
                    }
                }
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
                let col = cluster_color(self.labels[ri]);
                ui.painter().add(egui::Shape::closed_line(
                    sm, egui::Stroke::new(2.0, Color32::from_rgb(col[0], col[1], col[2])),
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
            if ui.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Z)) {
                self.undo_hardneg(leaf_idx);
            }
        }
        CanvasTool::Select => {
            // ── click selects a region; drag rubber-band multi-selects
            // (feeds the gallery's bulk reassign/remove) ──
            if resp.drag_started() {
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
                if resp.drag_stopped() {
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
                            if !self.multi_selected.remove(&i) {
                                self.multi_selected.insert(i);
                            }
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
            if resp.dragged() {
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
            if resp.drag_stopped() && !self.brush_stroke.is_empty() {
                self.finish_brush_stroke(leaf_idx, toasts);
            }
        }
        CanvasTool::Lasso => {
            // ── freehand select: collect screen-space points while dragging,
            // draw them live, and on release select every region whose
            // bbox-center falls inside the closed polygon (same precision
            // level as the existing rubber-band, which also only tests bbox
            // intersection rather than exact mask overlap) ──
            if resp.drag_started() {
                self.lasso_points.clear();
            }
            if resp.dragged() {
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
            if resp.drag_stopped() && self.lasso_points.len() >= 3 {
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
        }
        // Zoom/pan are universal canvas gestures, not a tool you switch into —
        // scroll to zoom, hold middle-mouse to pan, regardless of active tool
        // (harmless — neither conflicts with any tool's own click/drag).
        // EXCEPTION: while the Brush is active, ctrl+scroll resizes the brush
        // instead of zooming, so you can adjust size without switching tools.
        if resp.hovered() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            let ctrl = ui.input(|i| i.modifiers.ctrl);
            if scroll != 0.0 && ctrl && self.canvas_tool == CanvasTool::Brush {
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
        // right-click context menu + Enter/Delete shortcuts — act on whatever
        // is currently selected, available for every tool EXCEPT the stamp
        // tool (which already uses right-click for its own undo gesture)
        if self.canvas_tool != CanvasTool::MarkHealthy {
            // right-click the canvas -> act on the current selection: the
            // multi-select set if non-empty, else whichever single region is
            // currently selected (a plain click on one anomaly) — so the menu
            // works the same way regardless of how you got there.
            let effective_sel: Vec<usize> = if !self.multi_selected.is_empty() {
                self.multi_selected.iter().copied().collect()
            } else {
                self.selected_region.into_iter().collect()
            };
            let n_sel = effective_sel.len();
            let mut do_confirm = false;
            let mut do_remove = false;
            let mut do_reassign = false;
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
            // confirm, Delete to reject — only when no widget (e.g. the
            // reassign text field) currently has keyboard focus, so typing a
            // cluster name never accidentally confirms/rejects a region.
            let focused = ctx.memory(|m| m.focused().is_some());
            if !focused && n_sel > 0 {
                if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    do_confirm = true;
                }
                if ui.input(|i| i.key_pressed(egui::Key::Delete)) {
                    do_remove = true;
                }
            }
            if do_confirm {
                self.confirm_regions(&effective_sel, toasts);
            }
            if do_remove {
                self.remove_regions(&effective_sel, toasts);
                self.multi_selected.clear();
                self.selected_region = None;
            }
            if do_reassign {
                for &i in &effective_sel {
                    self.multi_selected.insert(i);
                }
                self.reassign_selected(toasts);
                self.selected_region = None;
            }
            if do_clear {
                self.multi_selected.clear();
                self.selected_region = None;
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

        // Which leaf pixels an (un-removed, filter-matching) anomaly region already
        // covers — computed BEFORE the recon-tint pass so the tint can skip them.
        // A pixel that's both "reconstructed" and part of a detected anomaly (e.g.
        // a hole region) should show the cluster colour alone, not a blend of the
        // two stacked on top of each other.
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
                if let Some(cid) = sel {
                    if self.labels[ri] != cid {
                        continue;
                    }
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
                        px[o] = lerp_u8(px[o], 70, 0.5);
                        px[o + 1] = lerp_u8(px[o + 1], 200, 0.5);
                        px[o + 2] = lerp_u8(px[o + 2], 225, 0.5);
                        px[o + 3] = 255;
                    }
                }
            }
        }

        // Fill mode: bake the regions into the texture (family colour, opacity =
        // slider). Outline mode keeps the texture clean and draws smooth vector
        // contours in show_canvas instead.
        if !self.overlay_outline {
            for ri in 0..self.regions.len() {
                if !self.region_visible(ri) {
                    continue;
                }
                let r = &self.regions[ri];
                if r.leaf != idx {
                    continue;
                }
                if let Some(cid) = sel {
                    if self.labels[ri] != cid {
                        continue;
                    }
                }
                // Holes get an OUTLINE, never an opaque fill: the whole point is
                // that there's nothing there, so painting it solid erases exactly
                // the cue that makes it read as a gap rather than texture (this
                // was confusing — the raw leaf thumbnail still showed the true
                // gap while this composited view was covering it up).
                if self.labels[ri] == worker::HOLE_FAMILY {
                    paint_region_outline(&mut px, w, h, r, cluster_color(self.labels[ri]));
                } else {
                    paint_region(&mut px, w, h, r, cluster_color(self.labels[ri]), self.overlay_alpha);
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

        // overlay appearance: outline-vs-fill + opacity (see the leaf beneath)
        ui.horizontal(|ui| {
            if ui.checkbox(&mut self.overlay_outline, "Outline")
                .on_hover_text("Draw cluster OUTLINES (leaf fully visible inside) instead of\n\
                                filled pixels. Same family colours.")
                .changed()
            {
                self.overlay_tex = None;
            }
            if !self.overlay_outline {
                ui.label(RichText::new("opacity").small());
                if ui.add(egui::Slider::new(&mut self.overlay_alpha, 0.1..=1.0).fixed_decimals(2)
                    .show_value(false)).changed()
                {
                    self.overlay_tex = None;
                }
            }
        });
        ui.separator();

        // ── PCA scatter (click → nearest point's cluster) ──
        let sel = self.selected_cluster;
        let plot = Plot::new("cluster_scatter").height(200.0).show(ui, |plot_ui| {
            for c in &self.clusters {
                let col = cluster_color(c.id);
                let pts: Vec<[f64; 2]> = c
                    .members
                    .iter()
                    .filter(|&&i| self.region_visible(i))
                    .map(|&i| [self.coords[i][0] as f64, self.coords[i][1] as f64])
                    .collect();
                let radius = if Some(c.id) == sel { 4.0 } else { 2.5 };
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
            egui::Grid::new("cluster_stats").num_columns(4).striped(true).show(ui, |ui| {
                ui.label(RichText::new("Cluster").small());
                ui.label(RichText::new("% leaf").small());
                ui.label(RichText::new("total px").small());
                ui.label(RichText::new("Recon %").small())
                    .on_hover_text("This cluster's damaged area as a fraction of the RECONSTRUCTED \
                                    intact leaf (damage relative to the whole undamaged leaf). \
                                    Needs a recon checkpoint — bundled at models/recon/gen.mpk.");
                ui.end_row();
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
                    ui.horizontal(|ui| {
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
                        }
                    });
                    ui.label(format!("{:.1}%", 100.0 * leaf_px as f32 / leaf_valid_px));
                    ui.label(format!("{total}"));
                    if recon_whole > 0.0 {
                        ui.label(format!("{:.1}%", 100.0 * leaf_px as f32 / recon_whole));
                    } else {
                        ui.label(RichText::new("-").color(Color32::DARK_GRAY));
                    }
                    ui.end_row();
                }
            });
        });
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
        });
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
            && self.eff_dino().is_some() && !self.retraining && !self.running;
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
            ui.add_enabled_ui(!self.remove_undo.is_empty(), |ui| {
                if ui.small_button(format!("↩ Undo remove ({})", self.remove_undo.len()))
                    .on_hover_text("Restore the most recently removed region(s).")
                    .clicked()
                {
                    self.undo_remove(toasts);
                }
            });
        });
        ui.label(RichText::new("click = highlight on leaf · right-click = reject · ctrl+click = multi-select \
                                 · Enter = confirm · Delete = reject")
            .small().color(Color32::DARK_GRAY));
        if !self.multi_selected.is_empty() {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{} selected", self.multi_selected.len())).small());
                if ui.small_button("✅ Confirm").clicked() {
                    let ids: Vec<usize> = self.multi_selected.iter().copied().collect();
                    self.confirm_regions(&ids, toasts);
                }
                ui.add(egui::TextEdit::singleline(&mut self.reassign_name)
                    .desired_width(160.0)
                    .hint_text("cluster name"));
                if ui.small_button("Reassign").clicked() {
                    self.reassign_selected(toasts);
                }
                if ui.small_button("Clear").clicked() {
                    self.multi_selected.clear();
                }
            });
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
                                if ui.input(|inp| inp.modifiers.ctrl) {
                                    if !self.multi_selected.remove(&i) {
                                        self.multi_selected.insert(i);
                                    }
                                } else {
                                    self.selected_idx = Some(self.regions[i].leaf);
                                    self.selected_cluster = Some(self.labels[i]);
                                    self.selected_region = Some(i);
                                    self.overlay_tex = None;
                                }
                            }
                            if resp.secondary_clicked() {
                                self.remove_regions(&[i], toasts);
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
        self.remove_undo.clear();
        self.persisted.clear();
        self.merged_away.clear();
        self.cluster_names.clear();
        self.selected_cluster = None;
        self.selected_region = None;
        self.gallery_page = 0;
    }

    /// A region counts as visible unless it's been rejected by the user OR
    /// absorbed into another region by `merge_touching_regions` — everything
    /// that iterates/renders/counts regions should gate on this, not on
    /// `removed` alone, now that merges can also hide an index.
    fn region_visible(&self, i: usize) -> bool {
        !self.removed.contains(&i) && !self.merged_away.contains(&i)
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
                dino_res: 512,
                conf: self.conf,
                recon_ckpt: self.eff_recon(),
                head_path: head,
                use_patchcore: self.use_patchcore,
                head_tau: self.head_tau,
                head_grow: self.head_grow.min(self.head_tau),
                seg_alpha_lo: self.seg_alpha_lo,
                seg_chroma_min: self.seg_chroma_min,
                cluster_eps: self.cluster_eps,
                cluster_min_pts: self.cluster_min_pts,
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
                    PipeMsg::Clusters { labels, coords, names, regions } => {
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
                epochs: 150,
                lr: 0.5,
                l2_anchor: 0.05,
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
        let fname = format!("region_{idx}.png");
        if let Some(img) = image::RgbaImage::from_raw(r.crop_size, r.crop_size, r.crop.clone()) {
            let _ = img.save(labels_dir.join(&fname));
        }
        let src = self.results.get(r.leaf).map(|l| l.src.display().to_string()).unwrap_or_default();
        let run = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let jsonl_path = out.join("curations").join("labels.jsonl");
        // upsert: drop any existing line for this crop filename before appending
        let needle = format!("\"crop\":\"{fname}\"");
        if let Ok(text) = std::fs::read_to_string(&jsonl_path) {
            let kept: String = text.lines().filter(|l| !l.contains(&needle)).map(|l| format!("{l}\n")).collect();
            let _ = std::fs::write(&jsonl_path, kept);
        }
        let line = format!(
            "{{\"crop\":\"{}\",\"family\":\"{}\",\"source\":\"{}\",\"leaf_src\":\"{}\",\"ts\":{}}}\n",
            fname, json_escape(family), if is_reject { "reject" } else { "confirm" },
            json_escape(&src), run,
        );
        use std::io::Write;
        match std::fs::OpenOptions::new().create(true).append(true).open(&jsonl_path) {
            Ok(mut f) => {
                let _ = f.write_all(line.as_bytes());
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
        let src = leaf.src.display().to_string();
        let line = format!(
            "{{\"crop\":\"{}\",\"family\":\"{}\",\"source\":\"{}\",\"leaf_src\":\"{}\",\"ts\":{}}}\n",
            fname, json_escape(&family), if is_reject { "reject" } else { "manual" },
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

        // per-leaf overlays (anomalies colour-coded by family)
        for (li, leaf) in self.results.iter().enumerate() {
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
            Ok(_) => toasts.success(format!("Exported {n} anomalies + images → export/")),
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
        for c in &clusters {
            self.cluster_names.entry(c.id).or_insert_with(|| {
                if c.id < 0 { "noise".to_string() } else { format!("Cluster {}", c.id) }
            });
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

    /// Bulk-reassign every gallery-multi-selected region to the cluster named
    /// `self.reassign_name` — reuses an existing cluster with that name (matched
    /// case-insensitively) or allocates a fresh id. Lets the user correct a batch of
    /// misclustered regions (e.g. nervature wrongly grouped with necrosis) in one action
    /// instead of one at a time.
    fn reassign_selected(&mut self, toasts: &mut ToastManager) {
        let name = self.reassign_name.trim().to_string();
        if self.multi_selected.is_empty() || name.is_empty() {
            return;
        }
        let id = self.resolve_cluster_id(&name);
        let ids: Vec<usize> = self.multi_selected.iter().copied().collect();
        for &i in &ids {
            self.labels[i] = id;
            self.persist_region(i, &name, false, toasts);
        }
        self.multi_selected.clear();
        self.reassign_name.clear();
        self.build_clusters(toasts);
        self.overlay_tex = None;
    }

    /// Resolve a typed cluster name to its id — reuses an existing cluster
    /// with that name (case-insensitive) or allocates a fresh one. Shared by
    /// `reassign_selected` and the brush tool so both go through the same
    /// allocation logic. Skips BOTH reserved sentinels (previously only
    /// `HOLE_FAMILY` was guarded here, not `NOVEL_FAMILY` — a real bug: a
    /// typed name could have collided with the "Novel (PatchCore)" id).
    fn resolve_cluster_id(&mut self, name: &str) -> i32 {
        self.cluster_names.iter()
            .find(|(_, n)| n.eq_ignore_ascii_case(name))
            .map(|(&id, _)| id)
            .unwrap_or_else(|| {
                let mut next = self.labels.iter().copied()
                    .chain(self.cluster_names.keys().copied())
                    .max().unwrap_or(-1) + 1;
                while next == worker::HOLE_FAMILY || next == worker::NOVEL_FAMILY {
                    next += 1;
                }
                self.cluster_names.insert(next, name.to_string());
                next
            })
    }

    /// Reject `ids` (both the gallery's single right-click and the canvas
    /// context menu's bulk "Remove selected" route through this), recording
    /// the batch on an undo stack so an accidental removal — one region or
    /// many — can be reversed in one action. Persists immediately (see
    /// `persist_region`) — this is where the old "forgot to click Save"
    /// failure mode used to live.
    fn remove_regions(&mut self, ids: &[usize], toasts: &mut ToastManager) {
        if ids.is_empty() {
            return;
        }
        for &i in ids {
            self.removed.insert(i);
            self.persist_region(i, "rejected", true, toasts);
        }
        self.remove_undo.push(ids.to_vec());
        const MAX_UNDO: usize = 50;
        if self.remove_undo.len() > MAX_UNDO {
            self.remove_undo.remove(0);
        }
        self.overlay_tex = None;
    }

    /// Restore the most recently removed batch (one gallery reject, or one
    /// bulk "Remove selected" — whichever happened last).
    fn undo_remove(&mut self, toasts: &mut ToastManager) {
        let Some(ids) = self.remove_undo.pop() else {
            toasts.info("Nothing to undo.");
            return;
        };
        for &i in &ids {
            self.removed.remove(&i);
            // undo the disk write too — a restored region goes back to
            // "unreviewed," not "rejected," so its persisted reject line
            // (if any) shouldn't linger; the stable region_{i}.png filename
            // means this is a clean delete, no orphan left behind.
            self.retract_persisted(i);
        }
        self.overlay_tex = None;
        toasts.success(format!("Restored {} region(s)", ids.len()));
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

    fn poll_pick(&mut self) {
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
                    }
                }
                self.pick_rx = None;
            }
        }
    }
}

// ── free helpers ────────────────────────────────────────────────────────────

/// Minimal JSON string escaping for the curation label file (user-entered names).
fn json_escape(s: &str) -> String {
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
            Pick::Source | Pick::Output | Pick::Recon => rfd::FileDialog::new().pick_folder(),
            Pick::Yolo | Pick::Dino => rfd::FileDialog::new().add_filter("ONNX", &["onnx"]).pick_file(),
            Pick::Bank => rfd::FileDialog::new().add_filter("bank", &["bin"]).pick_file(),
            Pick::Meta | Pick::Head => rfd::FileDialog::new().add_filter("json", &["json"]).pick_file(),
        };
        let _ = tx.send(res);
    });
    rx
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

/// Outline-only paint: only mask pixels with at least one non-mask (or
/// out-of-bounds) 4-neighbor get colored; interior pixels are left
/// completely untouched. Used for HOLE regions specifically, so a genuinely
/// transparent gap stays visibly transparent — a solid fill would erase the
/// exact cue that makes it read as "gap" instead of "texture."
fn paint_region_outline(px: &mut [u8], w: usize, h: usize, r: &AnomalyRegion, col: [u8; 3]) {
    let [bx, by, bw, bh] = r.bbox_leaf;
    for ly in 0..bh {
        for lx in 0..bw {
            if !r.mask[(ly * bw + lx) as usize] {
                continue;
            }
            let is_boundary = lx == 0 || ly == 0 || lx == bw - 1 || ly == bh - 1
                || !r.mask[(ly * bw + (lx - 1)) as usize]
                || !r.mask[(ly * bw + (lx + 1)) as usize]
                || !r.mask[((ly - 1) * bw + lx) as usize]
                || !r.mask[((ly + 1) * bw + lx) as usize];
            if !is_boundary {
                continue;
            }
            let (gx, gy) = ((bx + lx) as usize, (by + ly) as usize);
            if gx >= w || gy >= h {
                continue;
            }
            let o = (gy * w + gx) * 4;
            px[o] = col[0];
            px[o + 1] = col[1];
            px[o + 2] = col[2];
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

/// Standard even-odd point-in-polygon test (ray casting). Used by the Lasso
/// tool to decide which regions' bbox-centers fall inside a freehand outline.
fn point_in_polygon(x: f32, y: f32, poly: &[(f32, f32)]) -> bool {
    let mut inside = false;
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}
