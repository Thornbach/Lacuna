//! Field Review tab — the real-data correction flywheel. Batch-runs the
//! current segmentation checkpoint over real field photos, then lets a human
//! fix its mistakes (Brush/Eraser/Wand/Knife/delete/paint-new-instance) with
//! the same mask-editing primitives the Pipeline tab uses, and exports the
//! corrected instances as YOLO-format polygon labels into the same dataset
//! `generate_leaf_dataset.py` produces — closing the synthetic-to-real
//! domain gap with real, human-verified examples instead of more synthetic
//! realism.
//!
//! Deliberately NOT built on `AnomalyRegion`/`PipelineTab` — those carry
//! PatchCore/few-shot-specific baggage (`descriptor`, `family`, `dino_embed`)
//! this feature doesn't need. `RealScene`/`RealInstance` are the analogous,
//! purpose-built types; the six portable geometry primitives are shared via
//! `crate::tabs::mask_tools`.

pub mod dataset_export;
pub mod polygon;
pub mod sam;

use std::{
    collections::HashSet,
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    sync::{mpsc, Arc, Mutex},
};

use egui::{Color32, Context, RichText, Ui};
use egui_phosphor::regular as icon;

use crate::settings::AppSettings;
use crate::tabs::leaf_seg::inference::{
    list_images, scan_image_count, spawn_segmentation, SegConfig, SegItem, SegMsg,
};
use crate::tabs::mask_tools::{
    dist_to_polyline, fill_polygon_mask, mask_connected_components, point_in_polygon, reclaim_kerf,
    wand_flood_fill,
};
use crate::tabs::pipeline::channels::rgb_to_lab;
use crate::ui_kit;
use crate::widgets::ToastManager;

const PALETTE: [[u8; 3]; 8] = [
    [230, 80, 80], [80, 180, 230], [120, 210, 120], [230, 180, 70],
    [180, 120, 220], [70, 210, 200], [230, 120, 180], [160, 200, 80],
];

// Kerf carve/keep margins for the Knife tool — same two-tier design as
// Pipeline's Knife (see `mask_tools::reclaim_kerf`'s doc comment): a wider
// carve boundary during the initial split, a narrower permanent-gap margin
// during kerf reclaim so cut pieces stay non-adjacent.
const KNIFE_KERF: f32 = 2.0;
const KNIFE_KERF_KEEP: f32 = 1.0;

#[derive(Clone)]
pub struct RealInstance {
    pub bbox:     [u32; 4], // x, y, w, h — SCENE (original photo) coords
    pub mask:     Vec<bool>, // bbox-local raster, true = leaf pixel
    pub class_id: u32,      // always 0 ("leaf") today; threaded through end to
                             // end so a future class picker is a small add,
                             // not a rework.
}

pub struct RealScene {
    pub filename:     String,
    pub path:         PathBuf,
    pub size:         [u32; 2],
    pub instances:    Vec<RealInstance>,
    /// Export gate. Without this, an "export everything loaded" button would
    /// silently write raw uncorrected model predictions into the dataset as
    /// verified ground truth — the exact failure this tool exists to prevent.
    pub reviewed:     bool,
    pub exported_idx: Option<u32>,
}

#[derive(Clone, Copy, PartialEq)]
enum FieldTool { Select, Lasso, Brush, Eraser, Wand, Knife, Polygon, Sam }

/// Background-thread results for the Sam tool's model load + per-scene
/// encode (see the `sam_*` fields' doc comment) — both are too slow for the
/// UI thread, unlike the per-click decoder call.
enum SamMsg {
    Loaded(Result<sam::SamModel, String>),
    Encoded(usize, Result<sam::SamEmbedding, String>),
}

#[derive(Clone, Copy, PartialEq)]
enum BrushShape { Square, Circle }

/// A full snapshot of one scene's instances, taken right before an edit
/// that would otherwise destroy the previous state — simpler than Pipeline's
/// own per-entry `UndoEntry` (Remove/Cut) because `RealInstance` carries no
/// heavy per-region data (descriptor/family/dino_embed) or disk-persistence
/// state to reconcile, just bbox+mask+class_id, so a plain "restore the
/// whole Vec" is correct and cheap enough not to need anything cleverer.
struct UndoEntry {
    scene_idx: usize,
    instances: Vec<RealInstance>,
}

pub struct FieldReviewTab {
    // ── settings ──────────────────────────────────────────────────────────
    source_folder: Option<PathBuf>,
    dataset_root:  Option<PathBuf>,
    model_path:    Option<PathBuf>,
    conf:  f32,
    imgsz: u32,

    // ── worker ────────────────────────────────────────────────────────────
    seg_rx:         Option<mpsc::Receiver<SegMsg>>,
    cancel_flag:    Arc<AtomicBool>,
    running:        bool,
    progress_done:  usize,
    progress_total: usize,

    // ── scenes + selection ───────────────────────────────────────────────
    scenes:            Vec<RealScene>,
    selected_idx:      Option<usize>,
    selected_instance: Option<usize>,
    multi_selected:    HashSet<usize>,

    // ── correction tools ─────────────────────────────────────────────────
    canvas_tool:      FieldTool,
    brush_size:       u32,
    brush_shape:      BrushShape,
    wand_tolerance:   f32,
    brush_stroke:     HashSet<(i32, i32)>,
    wand_mask:        HashSet<(i32, i32)>,
    lasso_points:     Vec<egui::Pos2>,
    knife_drag_start:  Option<egui::Pos2>,
    /// Polygon tool: leaf/image-space nodes placed so far (NOT screen-space
    /// like `lasso_points` — a polygon is built over several clicks, and
    /// storing leaf-space keeps already-placed nodes correctly positioned
    /// even if the user pans/zooms mid-draw; converted to screen space
    /// fresh every frame for rendering).
    poly_points:      Vec<(f32, f32)>,
    undo_stack:       Vec<UndoEntry>,

    // ── SAM click tool: a folder containing sam_encoder.onnx/sam_decoder.onnx
    // (see field_review::sam's doc comment for the export procedure). Both
    // model load and per-scene embedding (a real ViT forward pass,
    // expensive) run on a background thread (`sam_rx`/`sam_busy`) so the UI
    // never freezes; only the per-click decoder call runs on the UI thread,
    // and it's cheap by design. Clicks accumulate as points (positive from
    // left-click, negative/exclude from right-click) refining one live
    // preview — Fill/Cancel commit or discard it, same pattern as Wand. ──
    sam_dir:         Option<PathBuf>,
    sam_dir_rx:      Option<mpsc::Receiver<Option<PathBuf>>>,
    sam_model:       Option<Arc<Mutex<sam::SamModel>>>,
    sam_embed_cache: Option<(usize, sam::SamEmbedding)>,
    sam_status:      String,
    sam_rx:          Option<mpsc::Receiver<SamMsg>>,
    sam_busy:        bool,
    sam_points:      Vec<(f32, f32, bool)>,
    sam_preview:     HashSet<(i32, i32)>,

    // ── view (universal, not a tool) ─────────────────────────────────────
    canvas_zoom: f32,
    canvas_pan:  egui::Vec2,

    // ── per-scene caches ──────────────────────────────────────────────────
    image_cache: Option<(usize, image::RgbaImage)>,
    lab_cache:   Option<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>,
    overlay_tex: Option<(usize, egui::TextureHandle)>,

    // ── file dialogs ──────────────────────────────────────────────────────
    source_rx:  Option<mpsc::Receiver<Option<PathBuf>>>,
    dataset_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    model_rx:   Option<mpsc::Receiver<Option<PathBuf>>>,
    import_rx:  Option<mpsc::Receiver<Option<PathBuf>>>,

    source_count: usize,
    log: Vec<String>,
}

impl FieldReviewTab {
    pub fn new() -> Self {
        Self {
            source_folder: None,
            dataset_root:  None,
            model_path:    None,
            conf:  0.25,
            imgsz: 1280, // see settings.rs::FieldReviewSettings::default for the measurement

            seg_rx: None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            running: false,
            progress_done: 0,
            progress_total: 0,

            scenes: Vec::new(),
            selected_idx: None,
            selected_instance: None,
            multi_selected: HashSet::new(),

            canvas_tool: FieldTool::Select,
            brush_size: 24,
            brush_shape: BrushShape::Circle,
            wand_tolerance: 12.0,
            brush_stroke: HashSet::new(),
            wand_mask: HashSet::new(),
            lasso_points: Vec::new(),
            knife_drag_start: None,
            poly_points: Vec::new(),
            undo_stack: Vec::new(),

            sam_dir: None,
            sam_dir_rx: None,
            sam_model: None,
            sam_embed_cache: None,
            sam_status: String::new(),
            sam_rx: None,
            sam_busy: false,
            sam_points: Vec::new(),
            sam_preview: HashSet::new(),

            canvas_zoom: 1.0,
            canvas_pan: egui::Vec2::ZERO,

            image_cache: None,
            lab_cache: None,
            overlay_tex: None,

            source_rx: None,
            dataset_rx: None,
            model_rx: None,
            import_rx: None,

            source_count: 0,
            log: Vec::new(),
        }
    }

    // ── lifecycle ─────────────────────────────────────────────────────────

    pub fn needs_repaint(&self) -> bool { self.running || self.sam_busy }

    pub fn save_settings(&self, s: &mut AppSettings) {
        let r = &mut s.field_review;
        r.last_source_folder = self.source_folder.clone();
        r.last_dataset_root  = self.dataset_root.clone();
        r.last_model_path    = self.model_path.clone();
        r.conf  = self.conf;
        r.imgsz = self.imgsz;
    }

    pub fn load_settings(&mut self, s: &AppSettings) {
        let r = &s.field_review;
        self.source_folder = r.last_source_folder.clone();
        self.dataset_root  = r.last_dataset_root.clone();
        self.model_path    = r.last_model_path.clone();
        self.conf  = r.conf;
        self.imgsz = r.imgsz;
        if let Some(f) = self.source_folder.clone() {
            self.source_count = scan_image_count(&f);
        }
    }

    // ── show ──────────────────────────────────────────────────────────────

    pub fn show(&mut self, ui: &mut Ui, ctx: &Context, toasts: &mut ToastManager) {
        self.poll_dialogs(toasts);
        self.poll_worker(toasts);
        self.poll_sam(toasts);

        egui::SidePanel::left("field_review_controls")
            .exact_width(300.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("fr_ctrl_scroll")
                    .show(ui, |ui| self.show_controls(ui, toasts));
            });

        egui::TopBottomPanel::bottom("field_review_gallery")
            .resizable(false)
            .min_height(96.0)
            .show_inside(ui, |ui| self.show_gallery(ui));

        egui::CentralPanel::default().show_inside(ui, |ui| self.show_canvas(ui, ctx, toasts));
    }

    fn show_controls(&mut self, ui: &mut Ui, toasts: &mut ToastManager) {
        ui_kit::section_header(ui, "Model");
        if ui.button("Model (best.pt export / .onnx)…").clicked() && self.model_rx.is_none() {
            self.model_rx = Some(spawn_file_dialog(&["onnx"]));
        }
        ui.label(RichText::new(path_str(&self.model_path)).small().color(Color32::GRAY));

        ui_kit::section_header(ui, "Folders");
        if ui.button("Real photos folder…").clicked() && self.source_rx.is_none() {
            self.source_rx = Some(spawn_folder_dialog());
        }
        ui.label(RichText::new(path_str(&self.source_folder)).small().color(Color32::GRAY));
        if self.source_folder.is_some() {
            ui.label(RichText::new(format!("{} images found", self.source_count)).small());
        }
        if ui.button("Dataset root (export target)…").clicked() && self.dataset_rx.is_none() {
            self.dataset_rx = Some(spawn_folder_dialog());
        }
        ui.label(RichText::new(path_str(&self.dataset_root)).small().color(Color32::GRAY));
        if ui.button("SAM model folder…").clicked() && self.sam_dir_rx.is_none() {
            self.sam_dir_rx = Some(spawn_folder_dialog());
        }
        ui.label(RichText::new(path_str(&self.sam_dir)).small().color(Color32::GRAY));
        ui.label(RichText::new(
            "Folder containing sam_encoder.onnx + sam_decoder.onnx — enables \
             the Sam tool below (click a leaf to auto-segment it)."
        ).small().color(Color32::GRAY));

        ui.add_space(10.0);
        let can_start = self.model_ok() && self.source_folder.is_some() && self.source_count > 0 && !self.running;
        ui.add_enabled_ui(can_start, |ui| {
            if ui_kit::primary_button(ui, "Run Segmentation").clicked() {
                self.start();
            }
        });
        if self.running {
            if ui.add_sized([ui.available_width(), 26.0], egui::Button::new("Cancel")).clicked() {
                self.cancel_flag.store(true, Ordering::Relaxed);
            }
            let frac = if self.progress_total > 0 { self.progress_done as f32 / self.progress_total as f32 } else { 0.0 };
            ui.add(egui::ProgressBar::new(frac).show_percentage());
        }

        ui.add_space(4.0);
        ui.label(RichText::new("— or —").small().color(Color32::GRAY));
        if ui.button("Import pre-labeled scenes…").clicked() && self.import_rx.is_none() {
            self.import_rx = Some(spawn_folder_dialog());
        }
        ui.label(RichText::new(
            "Loads a folder already in images/+labels/ (YOLO-seg) format \
             — e.g. from an offline YOLO+SAM pre-labeling pass — as new, \
             unreviewed scenes, same as a live segmentation run. Appends \
             to whatever's already loaded."
        ).small().color(Color32::GRAY));

        if !self.scenes.is_empty() {
            ui_kit::section_header(ui, "Correction tools");
            let mut switched = false;
            let tool_btn = |ui: &mut Ui, tool: FieldTool, icon: &str, name: &str, tip: &str,
                            cur: &mut FieldTool, switched: &mut bool| {
                let active = *cur == tool;
                let text = RichText::new(icon).size(15.0);
                let btn = if active {
                    egui::Button::new(text.color(ui_kit::on_accent())).fill(ui_kit::ACCENT())
                } else {
                    egui::Button::new(text)
                };
                if ui.add_sized([30.0, 28.0], btn).on_hover_text(format!("{name}\n{tip}")).clicked()
                    && *cur != tool
                {
                    *cur = tool;
                    *switched = true;
                }
            };
            ui.horizontal_wrapped(|ui| {
                tool_btn(ui, FieldTool::Select, icon::CURSOR, "Select",
                    "Click an instance to select it · ctrl+click to multi-select.",
                    &mut self.canvas_tool, &mut switched);
                tool_btn(ui, FieldTool::Lasso, icon::LASSO, "Lasso select",
                    "Drag a freeform outline; every instance whose center falls inside it \
                     is added to the selection.",
                    &mut self.canvas_tool, &mut switched);
                tool_btn(ui, FieldTool::Brush, icon::PAINT_BRUSH, "Brush",
                    "With a leaf selected, paint extends ONLY that leaf. With \
                     nothing selected, paint always starts a NEW leaf — it \
                     never auto-merges into a neighbor it happens to touch.",
                    &mut self.canvas_tool, &mut switched);
                tool_btn(ui, FieldTool::Eraser, icon::ERASER, "Eraser",
                    "With a leaf selected, erases ONLY from that leaf. With \
                     nothing selected, erases from whatever it touches. \
                     Removes the leaf entirely if nothing's left.",
                    &mut self.canvas_tool, &mut switched);
                tool_btn(ui, FieldTool::Wand, icon::MAGIC_WAND, "Wand",
                    "Click a pixel to grow a selection by color similarity, then Fill \
                     to commit it like a brush stroke.",
                    &mut self.canvas_tool, &mut switched);
                tool_btn(ui, FieldTool::Knife, icon::KNIFE, "Knife",
                    "Select an instance, then drag a line across it to split it in two.",
                    &mut self.canvas_tool, &mut switched);
                tool_btn(ui, FieldTool::Polygon, icon::POLYGON, "Polygon",
                    "Click to place nodes, click the first node again to close the \
                     shape. Same rule as Brush: with a leaf selected, it extends ONLY \
                     that leaf; with nothing selected, it always starts a new one.",
                    &mut self.canvas_tool, &mut switched);
                tool_btn(ui, FieldTool::Sam, icon::SPARKLE, "Sam",
                    "Click a leaf to auto-segment it with SAM (needs a SAM model \
                     folder set above). Same rule as Brush: with a leaf selected, \
                     the result extends ONLY that leaf; with nothing selected, it \
                     always starts a new one.",
                    &mut self.canvas_tool, &mut switched);
            });
            if switched {
                self.wand_mask.clear();
                self.lasso_points.clear();
                self.knife_drag_start = None;
                self.poly_points.clear();
                self.sam_points.clear();
                self.sam_preview.clear();
            }
            match self.canvas_tool {
                FieldTool::Brush | FieldTool::Eraser => {
                    ui.add(egui::Slider::new(&mut self.brush_size, 4..=120).text("size"));
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("shape").small());
                        if ui.selectable_label(self.brush_shape == BrushShape::Circle, icon::CIRCLE).clicked() {
                            self.brush_shape = BrushShape::Circle;
                        }
                        if ui.selectable_label(self.brush_shape == BrushShape::Square, icon::SQUARE).clicked() {
                            self.brush_shape = BrushShape::Square;
                        }
                    });
                }
                FieldTool::Wand => {
                    ui.add(egui::Slider::new(&mut self.wand_tolerance, 1.0..=60.0).text("tolerance"));
                    if !self.wand_mask.is_empty() {
                        ui.horizontal(|ui| {
                            if ui.button("Fill").clicked() { self.commit_wand(); }
                            if ui.button("Cancel").clicked() { self.wand_mask.clear(); }
                        });
                    }
                }
                FieldTool::Knife => {
                    ui.label(RichText::new("Select an instance, then drag a line across it.").small().color(Color32::GRAY));
                }
                FieldTool::Sam => {
                    if self.sam_dir.is_none() {
                        ui.label(RichText::new("Set a SAM model folder above first.").small().color(Color32::GRAY));
                    } else {
                        let msg = if !self.sam_status.is_empty() { &self.sam_status } else { "Left-click: add · Right-click: exclude." };
                        ui.label(RichText::new(msg).small().color(Color32::GRAY));
                    }
                    if !self.sam_preview.is_empty() {
                        ui.horizontal(|ui| {
                            if ui.button("Fill").clicked() { self.commit_sam(); }
                            if ui.button("Cancel").clicked() {
                                self.sam_points.clear();
                                self.sam_preview.clear();
                            }
                        });
                    }
                }
                _ => {}
            }
            ui.horizontal(|ui| {
                ui.label(RichText::new("Del: remove selected · Ctrl+Z: undo").small().color(Color32::GRAY));
                ui.add_enabled_ui(!self.undo_stack.is_empty(), |ui| {
                    if ui.small_button(format!("Undo ({})", self.undo_stack.len()))
                        .on_hover_text("Undo the most recent paint/erase/cut/delete (Ctrl+Z).")
                        .clicked()
                    {
                        self.undo(toasts);
                    }
                });
            });

            ui.add_space(6.0);
            ui_kit::section_header(ui, "View");
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("Zoom {:.0}%", self.canvas_zoom * 100.0)).small());
                if ui.small_button("Fit").on_hover_text("Reset zoom/pan (scroll to zoom, hold middle-mouse to pan).").clicked() {
                    self.canvas_zoom = 1.0;
                    self.canvas_pan = egui::Vec2::ZERO;
                }
            });

            self.show_instance_list(ui);

            ui_kit::section_header(ui, "Review");
            if let Some(idx) = self.selected_idx {
                if let Some(scene) = self.scenes.get_mut(idx) {
                    ui.checkbox(&mut scene.reviewed, "Reviewed — ready to export");
                    ui.label(RichText::new(format!("{} instance(s)", scene.instances.len())).small());
                }
            }
            let reviewed_count = self.scenes.iter().filter(|s| s.reviewed).count();
            ui.add_space(6.0);
            let can_export = self.dataset_root.is_some() && reviewed_count > 0;
            ui.add_enabled_ui(can_export, |ui| {
                if ui_kit::primary_button(ui, &format!("Export {reviewed_count} reviewed scenes")).clicked() {
                    self.export(toasts);
                }
            });
        }

        if !self.log.is_empty() {
            ui_kit::section_header(ui, "Log");
            egui::ScrollArea::vertical().max_height(160.0).id_salt("fr_log").show(ui, |ui| {
                for line in self.log.iter().rev().take(200) {
                    ui.label(RichText::new(line).small());
                }
            });
        }
    }

    /// Scrollable list of the selected scene's instances — the "color
    /// overview" needed to tell touching leaves apart before painting, and
    /// a quick way to select/merge/delete without hunting for a leaf on the
    /// canvas. Click selects (ctrl-click multi-selects) exactly like the
    /// canvas Select tool, since both write the same `selected_instance`/
    /// `multi_selected` state.
    fn show_instance_list(&mut self, ui: &mut Ui) {
        let Some(scene_idx) = self.selected_idx else { return };
        let Some(scene) = self.scenes.get(scene_idx) else { return };
        if scene.instances.is_empty() { return; }

        ui_kit::section_header(ui, "Instances");
        let mut click_select: Option<(usize, bool)> = None;
        let mut click_delete: Option<usize> = None;
        egui::ScrollArea::vertical().max_height(200.0).id_salt("fr_instance_list").show(ui, |ui| {
            for (i, inst) in scene.instances.iter().enumerate() {
                let c = PALETTE[i % PALETTE.len()];
                let color = Color32::from_rgb(c[0], c[1], c[2]);
                let is_sel = self.selected_instance == Some(i) || self.multi_selected.contains(&i);
                ui.horizontal(|ui| {
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                    ui.painter().rect_filled(rect, 2.0, color);
                    let [_, _, bw, bh] = inst.bbox;
                    let resp = ui.selectable_label(is_sel, format!("Leaf {i}  {bw}×{bh}"));
                    if resp.clicked() {
                        click_select = Some((i, ui.input(|inp| inp.modifiers.ctrl)));
                    }
                    if ui.small_button("🗑").on_hover_text("Delete this leaf").clicked() {
                        click_delete = Some(i);
                    }
                });
            }
        });

        if let Some((i, ctrl)) = click_select {
            if ctrl {
                if !self.multi_selected.remove(&i) { self.multi_selected.insert(i); }
            } else {
                self.selected_instance = Some(i);
                self.multi_selected.clear();
            }
        }
        if let Some(i) = click_delete {
            self.delete_instances(scene_idx, &[i]);
        }

        let mut merge_targets: Vec<usize> = self.multi_selected.iter().copied().collect();
        if let Some(sel) = self.selected_instance { merge_targets.push(sel); }
        ui.horizontal(|ui| {
            ui.add_enabled_ui(merge_targets.len() >= 2, |ui| {
                if ui.small_button(format!("Merge selected ({})", merge_targets.len())).clicked() {
                    self.merge_instances(scene_idx, &merge_targets);
                }
            });
            let del_targets = merge_targets.len();
            ui.add_enabled_ui(del_targets >= 1, |ui| {
                if ui.small_button("Delete selected").clicked() {
                    self.delete_instances(scene_idx, &merge_targets);
                }
            });
        });
    }

    pub fn show_settings_panel(&mut self, ui: &mut Ui) {
        ui_kit::section_header(ui, "Settings");
        egui::Grid::new("field_review_settings").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
            ui.label("Confidence:");
            ui.add(egui::Slider::new(&mut self.conf, 0.0..=1.0).fixed_decimals(2));
            ui.end_row();
            ui.label("Image size:");
            egui::ComboBox::from_id_salt("field_review_imgsz")
                .selected_text(format!("{}", self.imgsz))
                .show_ui(ui, |ui| {
                    for sz in [320u32, 640, 960, 1280] {
                        ui.selectable_value(&mut self.imgsz, sz, format!("{sz}"));
                    }
                });
            ui.end_row();
        });
    }

    fn show_gallery(&mut self, ui: &mut Ui) {
        ui.add_space(2.0);
        ui.label(RichText::new("Loaded scenes").small().color(Color32::GRAY));
        egui::ScrollArea::horizontal().id_salt("fr_gallery_scroll").show(ui, |ui| {
            ui.horizontal(|ui| {
                for (i, scene) in self.scenes.iter().enumerate() {
                    let mark = if scene.reviewed { "\u{2713} " } else { "" };
                    let label = format!("{mark}{}\n{} leaves", scene.filename, scene.instances.len());
                    if ui.selectable_label(self.selected_idx == Some(i), label).clicked() {
                        self.selected_idx = Some(i);
                        self.selected_instance = None;
                        self.multi_selected.clear();
                        self.overlay_tex = None;
                    }
                }
            });
        });
    }

    // ── canvas ────────────────────────────────────────────────────────────

    fn show_canvas(&mut self, ui: &mut Ui, ctx: &Context, toasts: &mut ToastManager) {
        let Some(scene_idx) = self.selected_idx else {
            ui.centered_and_justified(|ui| {
                ui.label(RichText::new(
                    "Set a model + real-photo folder, then Run Segmentation.\n\
                     Click a scene in the gallery below to review/correct it.",
                ).color(Color32::GRAY));
            });
            return;
        };
        self.ensure_overlay(ctx, scene_idx);
        let Some((_, tex)) = self.overlay_tex.clone() else { return };

        let avail = ui.available_size();
        let sz = tex.size_vec2();
        let scale = (avail.x / sz.x).min(avail.y / sz.y).min(1.0).max(0.01);
        let disp = sz * scale;
        let (area, resp) = ui.allocate_exact_size(avail, egui::Sense::click_and_drag());
        // Zoom/pan apply ON TOP of the fit-to-panel rect — every downstream
        // instance-pixel<->screen conversion already funnels through
        // img_rect/s below, matching Pipeline's own canvas (mod.rs).
        let fit_rect = egui::Rect::from_center_size(area.center(), disp);
        let img_rect = egui::Rect::from_center_size(
            fit_rect.center() + self.canvas_pan, fit_rect.size() * self.canvas_zoom,
        );
        egui::Image::new((tex.id(), img_rect.size())).paint_at(ui, img_rect);
        let s = img_rect.width() / sz.x.max(1.0);

        // Zoom/pan are universal canvas gestures, not a tool — scroll to
        // zoom, hold middle-mouse to pan, regardless of active tool.
        // EXCEPTION (matches Pipeline's canvas exactly): while Brush/Eraser
        // is active, ctrl+scroll resizes the brush instead of zooming, so
        // size can be adjusted without switching tools or hunting for the
        // slider.
        if resp.hovered() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            let ctrl = ui.input(|i| i.modifiers.ctrl);
            let brush_sized = matches!(self.canvas_tool, FieldTool::Brush | FieldTool::Eraser);
            if scroll != 0.0 && ctrl && brush_sized {
                let delta = (scroll * 0.15).round() as i32;
                self.brush_size = (self.brush_size as i32 + delta).clamp(4, 200) as u32;
            } else if scroll != 0.0 {
                self.canvas_zoom = (self.canvas_zoom * (1.0 + scroll * 0.0015)).clamp(0.25, 6.0);
            }
            if ui.input(|i| i.pointer.middle_down()) {
                self.canvas_pan += ui.input(|i| i.pointer.delta());
            }
        }

        // Tool hotkeys (Photoshop-style, matches Pipeline's canvas exactly):
        // switch tools by keypress regardless of which tool is currently
        // active — guarded by `!focused` so typing in any text field never
        // accidentally swaps the active tool out from under you.
        let focused = ctx.memory(|m| m.focused().is_some());
        if !focused {
            let key_tool = ui.input(|i| {
                if i.key_pressed(egui::Key::V) { Some(FieldTool::Select) }
                else if i.key_pressed(egui::Key::B) { Some(FieldTool::Brush) }
                else if i.key_pressed(egui::Key::E) { Some(FieldTool::Eraser) }
                else if i.key_pressed(egui::Key::K) { Some(FieldTool::Knife) }
                else if i.key_pressed(egui::Key::L) { Some(FieldTool::Lasso) }
                else if i.key_pressed(egui::Key::W) { Some(FieldTool::Wand) }
                else if i.key_pressed(egui::Key::P) { Some(FieldTool::Polygon) }
                else if i.key_pressed(egui::Key::S) { Some(FieldTool::Sam) }
                else { None }
            });
            if let Some(tool) = key_tool {
                if self.canvas_tool != tool {
                    self.canvas_tool = tool;
                    self.wand_mask.clear();
                    self.lasso_points.clear();
                    self.knife_drag_start = None;
                    self.poly_points.clear();
                    self.sam_points.clear();
                    self.sam_preview.clear();
                }
            }
        }

        // outlines: selected instance (yellow), multi-selected (blue)
        if let Some(scene) = self.scenes.get(scene_idx) {
            if let Some(ii) = self.selected_instance {
                if let Some(inst) = scene.instances.get(ii) {
                    paint_bbox_outline(ui, img_rect, s, inst.bbox, Color32::from_rgb(255, 230, 0));
                }
            }
            for &ii in &self.multi_selected {
                if let Some(inst) = scene.instances.get(ii) {
                    paint_bbox_outline(ui, img_rect, s, inst.bbox, Color32::from_rgb(80, 170, 255));
                }
            }
        }

        let to_img = |p: egui::Pos2| ((p.x - img_rect.min.x) / s.max(1e-3), (p.y - img_rect.min.y) / s.max(1e-3));

        match self.canvas_tool {
            FieldTool::Select => {
                if resp.clicked() {
                    if let Some(p) = resp.interact_pointer_pos() {
                        let (lx, ly) = to_img(p);
                        let ctrl = ui.input(|i| i.modifiers.ctrl);
                        if let Some(scene) = self.scenes.get(scene_idx) {
                            if let Some(ii) = instance_at(scene, lx, ly) {
                                if ctrl {
                                    if !self.multi_selected.remove(&ii) { self.multi_selected.insert(ii); }
                                } else {
                                    self.selected_instance = Some(ii);
                                    self.multi_selected.clear();
                                }
                            } else if !ctrl {
                                self.selected_instance = None;
                                self.multi_selected.clear();
                            }
                        }
                    }
                }
            }
            FieldTool::Lasso => {
                if resp.drag_started_by(egui::PointerButton::Primary) {
                    self.lasso_points.clear();
                }
                if resp.dragged_by(egui::PointerButton::Primary) {
                    if let Some(p) = resp.interact_pointer_pos() { self.lasso_points.push(p); }
                }
                if self.lasso_points.len() > 1 {
                    ui.painter().add(egui::Shape::closed_line(
                        self.lasso_points.clone(),
                        egui::Stroke::new(1.5, Color32::from_rgb(80, 170, 255)),
                    ));
                }
                if resp.drag_stopped_by(egui::PointerButton::Primary) {
                    let poly: Vec<(f32, f32)> = self.lasso_points.iter().map(|&p| to_img(p)).collect();
                    if poly.len() > 2 {
                        if let Some(scene) = self.scenes.get(scene_idx) {
                            self.multi_selected.clear();
                            for (i, inst) in scene.instances.iter().enumerate() {
                                let [bx, by, bw, bh] = inst.bbox;
                                let (cx, cy) = (bx as f32 + bw as f32 / 2.0, by as f32 + bh as f32 / 2.0);
                                if point_in_polygon(cx, cy, &poly) { self.multi_selected.insert(i); }
                            }
                        }
                    }
                    self.lasso_points.clear();
                }
            }
            FieldTool::Brush | FieldTool::Eraser => {
                let (sw, sh) = self.scenes.get(scene_idx).map(|s| (s.size[0] as i32, s.size[1] as i32)).unwrap_or((0, 0));
                let half = (self.brush_size / 2) as i32;
                let col = if self.canvas_tool == FieldTool::Brush {
                    Color32::from_rgb(255, 150, 60)
                } else {
                    Color32::from_rgb(230, 80, 80)
                };
                if let Some(p) = resp.hover_pos() {
                    let (cx, cy) = to_img(p);
                    let mn = img_rect.min + egui::vec2((cx - half as f32) * s, (cy - half as f32) * s);
                    let side = self.brush_size as f32 * s;
                    match self.brush_shape {
                        BrushShape::Square => {
                            ui.painter().rect_stroke(egui::Rect::from_min_size(mn, egui::vec2(side, side)), 0.0, egui::Stroke::new(2.0, col));
                        }
                        BrushShape::Circle => {
                            ui.painter().circle_stroke(mn + egui::vec2(side, side) / 2.0, side / 2.0, egui::Stroke::new(2.0, col));
                        }
                    }
                }
                if resp.dragged_by(egui::PointerButton::Primary) {
                    if let Some(p) = resp.interact_pointer_pos() {
                        let (cx, cy) = to_img(p);
                        let (cxi, cyi) = (cx.round() as i32, cy.round() as i32);
                        for oy in -half..=half {
                            for ox in -half..=half {
                                if self.brush_shape == BrushShape::Circle && ox * ox + oy * oy > half * half { continue; }
                                let (px, py) = (cxi + ox, cyi + oy);
                                if px < 0 || py < 0 || px >= sw || py >= sh { continue; }
                                self.brush_stroke.insert((px, py));
                            }
                        }
                    }
                }
                // paint the stroke LIVE as it accumulates, not just once on
                // release — otherwise nothing visibly happens until the
                // mouse button comes up, which reads as a broken/delayed tool
                // (mirrors Pipeline's identical fix for the same complaint).
                if !self.brush_stroke.is_empty() {
                    let px_sz = s.max(1.0);
                    for &(px, py) in &self.brush_stroke {
                        let mn = img_rect.min + egui::vec2(px as f32 * s, py as f32 * s);
                        ui.painter().rect_filled(
                            egui::Rect::from_min_size(mn, egui::vec2(px_sz, px_sz)),
                            0.0, Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 110),
                        );
                    }
                }
                if resp.drag_stopped_by(egui::PointerButton::Primary) {
                    if self.canvas_tool == FieldTool::Brush {
                        self.finish_paint_stroke(scene_idx);
                    } else {
                        self.erase_stroke(scene_idx);
                    }
                }
            }
            FieldTool::Wand => {
                if resp.clicked() {
                    if let Some(p) = resp.interact_pointer_pos() {
                        let (lx, ly) = to_img(p);
                        self.run_wand(scene_idx, lx.round() as i32, ly.round() as i32);
                    }
                }
                if !self.wand_mask.is_empty() {
                    for &(px, py) in &self.wand_mask {
                        let mn = img_rect.min + egui::vec2(px as f32 * s, py as f32 * s);
                        ui.painter().rect_filled(
                            egui::Rect::from_min_size(mn, egui::vec2(s.max(1.0), s.max(1.0))),
                            0.0, Color32::from_rgba_unmultiplied(255, 200, 60, 90),
                        );
                    }
                }
            }
            FieldTool::Knife => {
                if resp.drag_started_by(egui::PointerButton::Primary) {
                    self.knife_drag_start = resp.interact_pointer_pos();
                }
                if let (Some(start), Some(cur)) = (self.knife_drag_start, resp.interact_pointer_pos()) {
                    ui.painter().line_segment([start, cur], egui::Stroke::new(2.0, Color32::from_rgb(255, 60, 60)));
                }
                if resp.drag_stopped_by(egui::PointerButton::Primary) {
                    if let (Some(start), Some(end), Some(ii)) = (self.knife_drag_start, resp.interact_pointer_pos(), self.selected_instance) {
                        let a = to_img(start);
                        let b = to_img(end);
                        self.do_knife_cut(scene_idx, ii, a, b);
                    }
                    self.knife_drag_start = None;
                }
            }
            FieldTool::Polygon => {
                // Clamp every placed node to the actual scene bounds — a
                // click can land in the canvas's letterboxed margin, or
                // land far outside the image while zoomed, which otherwise
                // produces leaf-space coordinates way beyond the real
                // image. That fed a bbox spanning tens of thousands of
                // pixels into the polygon fill and crashed on the
                // resulting allocation — confirmed as a real bug, not
                // theoretical.
                let (sw, sh) = self.scenes.get(scene_idx)
                    .map(|s| (s.size[0] as f32, s.size[1] as f32)).unwrap_or((0.0, 0.0));
                if let Some(p) = resp.hover_pos() {
                    let (rx, ry) = to_img(p);
                    let hp = (rx.clamp(0.0, sw.max(1.0)), ry.clamp(0.0, sh.max(1.0)));
                    // Snap-to-close: within this many SCREEN px of the first
                    // node (with at least 3 placed), a click closes the loop
                    // instead of adding another node.
                    const SNAP_PX: f32 = 10.0;
                    let near_start = self.poly_points.len() >= 3 && {
                        let (fx, fy) = self.poly_points[0];
                        let first_screen = img_rect.min + egui::vec2(fx * s, fy * s);
                        first_screen.distance(p) <= SNAP_PX
                    };
                    if resp.clicked() {
                        if near_start {
                            self.finish_polygon(scene_idx);
                        } else {
                            self.poly_points.push(hp);
                        }
                    }
                    // live rubber-band from the last node to the cursor
                    if let Some(&(lx, ly)) = self.poly_points.last() {
                        let last_screen = img_rect.min + egui::vec2(lx * s, ly * s);
                        ui.painter().line_segment([last_screen, p],
                            egui::Stroke::new(1.5, Color32::from_rgb(80, 170, 255)));
                    }
                    // `near_start` was computed before the click above may
                    // have just closed (and cleared) the polygon this same
                    // frame — re-check emptiness, don't trust the stale
                    // bool, or this indexes poly_points[0] on an empty
                    // vec and panics EXACTLY at the moment of closing
                    // (confirmed as the real crash, independent of area size).
                    if near_start && !self.poly_points.is_empty() {
                        let (fx, fy) = self.poly_points[0];
                        let first_screen = img_rect.min + egui::vec2(fx * s, fy * s);
                        ui.painter().circle_stroke(first_screen, SNAP_PX, egui::Stroke::new(2.0, Color32::from_rgb(140, 230, 150)));
                    }
                }
                // placed nodes + connecting segments
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
                if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    self.poly_points.clear();
                }
            }
            FieldTool::Sam => {
                self.ensure_sam_ready(scene_idx);
                let ready = self.sam_model.is_some()
                    && matches!(&self.sam_embed_cache, Some((i, _)) if *i == scene_idx);
                let click = if resp.clicked() {
                    resp.interact_pointer_pos().map(|p| (p, true))
                } else if resp.clicked_by(egui::PointerButton::Secondary) {
                    resp.interact_pointer_pos().map(|p| (p, false))
                } else {
                    None
                };
                if let Some((p, positive)) = click {
                    if ready {
                        let (lx, ly) = to_img(p);
                        self.sam_points.push((lx, ly, positive));
                        self.run_sam_preview(scene_idx, toasts);
                    } else {
                        toasts.info("Still preparing SAM for this scene — try again in a moment.");
                    }
                }
                for &(px, py, positive) in &self.sam_points {
                    let pt = img_rect.min + egui::vec2(px * s, py * s);
                    let col = if positive { Color32::from_rgb(140, 230, 150) } else { Color32::from_rgb(230, 90, 90) };
                    ui.painter().circle_filled(pt, 4.0, col);
                    ui.painter().circle_stroke(pt, 4.0, egui::Stroke::new(1.0, Color32::WHITE));
                }
                if !self.sam_preview.is_empty() {
                    for &(px, py) in &self.sam_preview {
                        let mn = img_rect.min + egui::vec2(px as f32 * s, py as f32 * s);
                        ui.painter().rect_filled(
                            egui::Rect::from_min_size(mn, egui::vec2(s.max(1.0), s.max(1.0))),
                            0.0, Color32::from_rgba_unmultiplied(80, 230, 140, 90),
                        );
                    }
                }
                if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    self.sam_points.clear();
                    self.sam_preview.clear();
                }
            }
        }

        if ui.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace)) {
            self.delete_selected(scene_idx);
        }
        if ui.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Z)) {
            // Mid-draw, Ctrl+Z undoes the last placed NODE, not the last
            // committed edit — nothing's been committed yet, so falling
            // through to the instance-level undo would either do nothing
            // useful or undo some earlier, unrelated edit instead of the
            // misclick that's actually annoying you right now.
            if self.canvas_tool == FieldTool::Polygon && !self.poly_points.is_empty() {
                self.poly_points.pop();
            } else {
                self.undo(toasts);
            }
        }
    }

    // ── undo ──────────────────────────────────────────────────────────────

    /// Snapshots the scene's instances BEFORE a destructive edit. Called by
    /// every structural mutation (paint-merge, erase, knife cut, delete,
    /// list-panel merge/delete) right after their own early-return guards,
    /// so a stroke/erase that ends up touching nothing never clutters the
    /// stack with a no-op entry.
    fn push_undo(&mut self, scene_idx: usize) {
        if let Some(scene) = self.scenes.get(scene_idx) {
            self.undo_stack.push(UndoEntry { scene_idx, instances: scene.instances.clone() });
            const MAX_UNDO: usize = 50;
            if self.undo_stack.len() > MAX_UNDO {
                self.undo_stack.remove(0);
            }
        }
    }

    fn undo(&mut self, toasts: &mut ToastManager) {
        let Some(entry) = self.undo_stack.pop() else {
            toasts.info("Nothing to undo.");
            return;
        };
        if let Some(scene) = self.scenes.get_mut(entry.scene_idx) {
            scene.instances = entry.instances;
        }
        self.selected_instance = None;
        self.multi_selected.clear();
        self.overlay_tex = None;
    }

    // ── overlay ───────────────────────────────────────────────────────────

    fn ensure_overlay(&mut self, ctx: &Context, scene_idx: usize) {
        if matches!(&self.overlay_tex, Some((i, _)) if *i == scene_idx) {
            return;
        }
        let Some(scene) = self.scenes.get(scene_idx) else { return };
        if !matches!(&self.image_cache, Some((i, _)) if *i == scene_idx) {
            let Ok(img) = image::open(&scene.path) else { return };
            self.image_cache = Some((scene_idx, img.to_rgba8()));
        }
        let Some((_, rgba_img)) = &self.image_cache else { return };
        let (w, h) = rgba_img.dimensions();
        let mut px: Vec<u8> = rgba_img.as_raw().clone();

        for (k, inst) in scene.instances.iter().enumerate() {
            let c = PALETTE[k % PALETTE.len()];
            let [bx, by, bw, bh] = inst.bbox;
            for yy in 0..bh {
                for xx in 0..bw {
                    if !inst.mask[(yy * bw + xx) as usize] { continue; }
                    let (gx, gy) = (bx + xx, by + yy);
                    if gx >= w || gy >= h { continue; }
                    let o = ((gy * w + gx) * 4) as usize;
                    for ch in 0..3 { px[o + ch] = lerp_u8(px[o + ch], c[ch], 0.45); }
                }
            }
        }

        let color: Vec<Color32> = px.chunks_exact(4).map(|c| Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3])).collect();
        let tex = ctx.load_texture(
            format!("field_review_overlay_{scene_idx}"),
            egui::ColorImage { size: [w as usize, h as usize], pixels: color },
            egui::TextureOptions::LINEAR,
        );
        self.overlay_tex = Some((scene_idx, tex));
    }

    fn ensure_lab(&mut self, scene_idx: usize) {
        if matches!(&self.lab_cache, Some((i, ..)) if *i == scene_idx) {
            return;
        }
        let Some((_, rgba_img)) = &self.image_cache else { return };
        let (w, h) = rgba_img.dimensions();
        let n = (w * h) as usize;
        let (mut l, mut a, mut b) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
        let raw = rgba_img.as_raw();
        for i in 0..n {
            let (r, g, bch) = (raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2]);
            let (ll, aa, bb) = rgb_to_lab(r, g, bch);
            l[i] = ll; a[i] = aa; b[i] = bb;
        }
        self.lab_cache = Some((scene_idx, l, a, b));
    }

    // ── correction actions ───────────────────────────────────────────────

    /// Shared by Brush and Wand-Fill. Merging is now an explicit action, not
    /// an implicit side effect of proximity: with an instance selected, a
    /// stroke ONLY ever extends that one instance, even if it happens to
    /// overlap a different, unselected instance nearby — two touching
    /// leaves painted with a selected target can no longer silently fuse.
    /// With nothing selected, a stroke ALWAYS starts a brand-new instance,
    /// even over existing pixels; merging two existing instances is only
    /// ever done explicitly (select both in the instance list, Merge).
    fn finish_paint_stroke(&mut self, scene_idx: usize) {
        let stroke = std::mem::take(&mut self.brush_stroke);
        if stroke.is_empty() { return; }
        self.push_undo(scene_idx);
        let Some(scene) = self.scenes.get_mut(scene_idx) else { return };

        if let Some(sel) = self.selected_instance.filter(|&ii| ii < scene.instances.len()) {
            let inst = &scene.instances[sel];
            let [bx, by, bw, bh] = inst.bbox;
            let (mut minx, mut miny, mut maxx, mut maxy) = (bx, by, bx + bw, by + bh);
            for &(x, y) in &stroke {
                minx = minx.min(x as u32); miny = miny.min(y as u32);
                maxx = maxx.max(x as u32 + 1); maxy = maxy.max(y as u32 + 1);
            }
            let (nw, nh) = (maxx - minx, maxy - miny);
            let mut merged = vec![false; (nw * nh) as usize];
            for yy in 0..bh {
                for xx in 0..bw {
                    if !inst.mask[(yy * bw + xx) as usize] { continue; }
                    let (gx, gy) = (bx + xx - minx, by + yy - miny);
                    merged[(gy * nw + gx) as usize] = true;
                }
            }
            for &(x, y) in &stroke {
                let (ox, oy) = (x as u32 - minx, y as u32 - miny);
                merged[(oy * nw + ox) as usize] = true;
            }
            let class_id = inst.class_id;
            scene.instances[sel] = RealInstance { bbox: [minx, miny, nw, nh], mask: merged, class_id };
        } else {
            let (mut minx, mut miny, mut maxx, mut maxy) = (u32::MAX, u32::MAX, 0u32, 0u32);
            for &(x, y) in &stroke {
                minx = minx.min(x as u32); miny = miny.min(y as u32);
                maxx = maxx.max(x as u32 + 1); maxy = maxy.max(y as u32 + 1);
            }
            let (nw, nh) = (maxx - minx, maxy - miny);
            let mut mask = vec![false; (nw * nh) as usize];
            for &(x, y) in &stroke {
                let (ox, oy) = (x as u32 - minx, y as u32 - miny);
                mask[(oy * nw + ox) as usize] = true;
            }
            scene.instances.push(RealInstance { bbox: [minx, miny, nw, nh], mask, class_id: 0 });
            // Auto-selecting the just-created instance lets a leaf be built
            // up from several short strokes without reselecting each time —
            // this only ever extends the SAME new instance, never a
            // different pre-existing one, so it can't reintroduce the
            // touching-leaves merge bug above.
            self.selected_instance = Some(scene.instances.len() - 1);
        }
        self.overlay_tex = None;
    }

    /// Mirrors `finish_paint_stroke`'s scope rule: with an instance
    /// selected, the eraser ONLY strips pixels from that instance — a
    /// stroke near a touching neighbor can no longer shave the neighbor
    /// too. With nothing selected, it erases from whatever it touches
    /// (there's no explicit target to protect).
    fn erase_stroke(&mut self, scene_idx: usize) {
        let stroke = std::mem::take(&mut self.brush_stroke);
        if stroke.is_empty() { return; }
        self.push_undo(scene_idx);
        let Some(scene) = self.scenes.get_mut(scene_idx) else { return };

        let targets: Vec<usize> = match self.selected_instance.filter(|&ii| ii < scene.instances.len()) {
            Some(sel) => vec![sel],
            None => (0..scene.instances.len()).collect(),
        };

        let mut to_remove = Vec::new();
        for &i in &targets {
            let inst = &mut scene.instances[i];
            let [bx, by, bw, bh] = inst.bbox;
            for &(x, y) in &stroke {
                let (xu, yu) = (x as u32, y as u32);
                if xu >= bx && xu < bx + bw && yu >= by && yu < by + bh {
                    inst.mask[((yu - by) * bw + (xu - bx)) as usize] = false;
                }
            }
            if !inst.mask.iter().any(|&m| m) { to_remove.push(i); }
        }
        if !to_remove.is_empty() {
            for &i in to_remove.iter().rev() { scene.instances.remove(i); }
            // Index shifts from removal make the old selection unreliable —
            // safe reset, matching the pre-existing conservative behavior.
            self.selected_instance = None;
            self.multi_selected.clear();
        }
        self.overlay_tex = None;
    }

    /// Unions two or more instances' masks into one, replacing them —
    /// backs the instance list's explicit "Merge selected" action.
    fn merge_instances(&mut self, scene_idx: usize, idxs: &[usize]) {
        let mut idxs: Vec<usize> = idxs.to_vec();
        idxs.sort_unstable();
        idxs.dedup();
        if idxs.len() < 2 { return; }
        self.push_undo(scene_idx);
        let Some(scene) = self.scenes.get_mut(scene_idx) else { return };
        if idxs.iter().any(|&i| i >= scene.instances.len()) { return; }

        let (mut minx, mut miny, mut maxx, mut maxy) = (u32::MAX, u32::MAX, 0u32, 0u32);
        for &i in &idxs {
            let [bx, by, bw, bh] = scene.instances[i].bbox;
            minx = minx.min(bx); miny = miny.min(by);
            maxx = maxx.max(bx + bw); maxy = maxy.max(by + bh);
        }
        let (nw, nh) = (maxx - minx, maxy - miny);
        let mut merged = vec![false; (nw * nh) as usize];
        for &i in &idxs {
            let inst = &scene.instances[i];
            let [bx, by, bw, bh] = inst.bbox;
            for yy in 0..bh {
                for xx in 0..bw {
                    if !inst.mask[(yy * bw + xx) as usize] { continue; }
                    let (gx, gy) = (bx + xx - minx, by + yy - miny);
                    merged[(gy * nw + gx) as usize] = true;
                }
            }
        }
        let class_id = scene.instances[idxs[0]].class_id;
        for &i in idxs.iter().rev() { scene.instances.remove(i); }
        scene.instances.push(RealInstance { bbox: [minx, miny, nw, nh], mask: merged, class_id });
        self.selected_instance = Some(scene.instances.len() - 1);
        self.multi_selected.clear();
        self.overlay_tex = None;
    }

    /// Deletes specific instances by index — backs both the keyboard
    /// Delete/Backspace gesture (via `delete_selected`) and the instance
    /// list's per-row and bulk delete buttons.
    fn delete_instances(&mut self, scene_idx: usize, idxs: &[usize]) {
        let mut idxs: Vec<usize> = idxs.to_vec();
        idxs.sort_unstable();
        idxs.dedup();
        if idxs.is_empty() { return; }
        self.push_undo(scene_idx);
        let Some(scene) = self.scenes.get_mut(scene_idx) else { return };
        for &i in idxs.iter().rev() {
            if i < scene.instances.len() { scene.instances.remove(i); }
        }
        self.selected_instance = None;
        self.multi_selected.clear();
        self.overlay_tex = None;
    }

    fn run_wand(&mut self, scene_idx: usize, sx: i32, sy: i32) {
        self.ensure_lab(scene_idx);
        let Some((_, l, a, b)) = &self.lab_cache else { return };
        let Some((_, rgba_img)) = &self.image_cache else { return };
        let (w, h) = rgba_img.dimensions();
        self.wand_mask = wand_flood_fill(l, a, b, w as usize, h as usize, sx, sy, self.wand_tolerance);
        let _ = scene_idx;
    }

    fn commit_wand(&mut self) {
        self.brush_stroke = std::mem::take(&mut self.wand_mask);
        if let Some(idx) = self.selected_idx { self.finish_paint_stroke(idx); }
    }

    /// Rasterizes the closed polygon into `brush_stroke` and hands it to
    /// `finish_paint_stroke` — same reuse pattern as `commit_wand`, so
    /// Polygon gets the EXACT same selection-gated merge-or-new rule Brush
    /// already has for free: a leaf selected -> extend ONLY that leaf;
    /// nothing selected -> always a new leaf, never an implicit merge.
    fn finish_polygon(&mut self, scene_idx: usize) {
        let poly = std::mem::take(&mut self.poly_points);
        let Some(([bx, by, bw, bh], mask)) = fill_polygon_mask(&poly) else { return };
        let mut stroke = HashSet::new();
        for yy in 0..bh {
            for xx in 0..bw {
                if mask[(yy * bw + xx) as usize] {
                    stroke.insert((bx as i32 + xx as i32, by as i32 + yy as i32));
                }
            }
        }
        self.brush_stroke = stroke;
        self.finish_paint_stroke(scene_idx);
    }

    /// Kicks off whatever background step the Sam tool currently needs —
    /// model load, then per-scene embedding — without blocking the UI
    /// thread. Both are slow (loading ~360MB of ONNX + building the GPU
    /// execution provider; a full ViT forward pass), unlike the per-click
    /// decoder call, which is designed to be fast and stays synchronous.
    /// Called every frame the Sam tool is active; a no-op once everything
    /// for the current scene is already ready or already in flight.
    fn ensure_sam_ready(&mut self, scene_idx: usize) {
        if self.sam_busy || self.sam_rx.is_some() { return; }
        if self.sam_model.is_none() {
            let Some(dir) = self.sam_dir.clone() else { return };
            self.sam_status = "Loading SAM model…".into();
            self.sam_busy = true;
            let (tx, rx) = mpsc::channel();
            self.sam_rx = Some(rx);
            std::thread::spawn(move || {
                let res = sam::SamModel::load(&dir.join("sam_encoder.onnx"), &dir.join("sam_decoder.onnx"));
                let _ = tx.send(SamMsg::Loaded(res));
            });
            return;
        }
        let need_encode = !matches!(&self.sam_embed_cache, Some((i, _)) if *i == scene_idx);
        if need_encode {
            let Some((_, rgba)) = &self.image_cache else { return };
            self.sam_status = "Encoding scene…".into();
            self.sam_busy = true;
            let model = self.sam_model.clone().unwrap();
            let rgba = rgba.clone();
            let (tx, rx) = mpsc::channel();
            self.sam_rx = Some(rx);
            std::thread::spawn(move || {
                let res = model.lock().unwrap().encode(&rgba);
                let _ = tx.send(SamMsg::Encoded(scene_idx, res));
            });
        }
    }

    fn poll_sam(&mut self, toasts: &mut ToastManager) {
        let Some(rx) = &self.sam_rx else { return };
        let Ok(msg) = rx.try_recv() else { return };
        self.sam_busy = false;
        self.sam_rx = None;
        match msg {
            SamMsg::Loaded(Ok(model)) => {
                self.sam_model = Some(Arc::new(Mutex::new(model)));
                self.sam_status = "Model loaded — encoding scene…".into();
            }
            SamMsg::Loaded(Err(e)) => {
                self.sam_status = format!("SAM load failed: {e}");
                toasts.error(format!("SAM load failed: {e}"));
            }
            SamMsg::Encoded(idx, Ok(emb)) => {
                self.sam_embed_cache = Some((idx, emb));
                self.sam_status = "Left-click: add · Right-click: exclude.".into();
            }
            SamMsg::Encoded(_, Err(e)) => {
                self.sam_status = format!("SAM encode failed: {e}");
                toasts.error(format!("SAM encode failed: {e}"));
            }
        }
    }

    /// Reruns the decoder over every accumulated point (see the `sam_*`
    /// fields' doc comment) and refreshes the live preview — mirrors
    /// `run_wand`'s preview-then-`commit_wand` split, just point-driven
    /// instead of tolerance-driven.
    fn run_sam_preview(&mut self, scene_idx: usize, toasts: &mut ToastManager) {
        let Some(model) = &self.sam_model else { return };
        let Some((_, emb)) = &self.sam_embed_cache else { return };
        match model.lock().unwrap().click(emb, &self.sam_points) {
            Ok(mask) => {
                let (sw, sh) = self.scenes.get(scene_idx)
                    .map(|s| (s.size[0] as usize, s.size[1] as usize)).unwrap_or((0, 0));
                let mut preview = HashSet::new();
                for y in 0..sh {
                    for x in 0..sw {
                        if mask.get(y * sw + x).copied().unwrap_or(false) {
                            preview.insert((x as i32, y as i32));
                        }
                    }
                }
                self.sam_status = format!("{} px — Fill to commit, Esc to discard.", preview.len());
                self.sam_preview = preview;
            }
            Err(e) => {
                self.sam_status = format!("SAM click failed: {e}");
                toasts.error(format!("SAM click failed: {e}"));
            }
        }
    }

    /// Commits the live preview — same reuse pattern as `commit_wand` and
    /// `finish_polygon`, so Sam gets the same selection-gated
    /// merge-or-new rule as every other tool for free.
    fn commit_sam(&mut self) {
        self.sam_points.clear();
        self.brush_stroke = std::mem::take(&mut self.sam_preview);
        if let Some(idx) = self.selected_idx { self.finish_paint_stroke(idx); }
    }

    fn do_knife_cut(&mut self, scene_idx: usize, inst_idx: usize, a: (f32, f32), b: (f32, f32)) {
        // Snapshotted up front (before the mutable borrow below) even though
        // a cut that fails to produce 2+ pieces is a no-op — pushing an
        // occasional harmless no-op undo entry is simpler than threading a
        // second mutable pass through this function just to avoid it.
        self.push_undo(scene_idx);
        let Some(scene) = self.scenes.get_mut(scene_idx) else { return };
        let Some(inst) = scene.instances.get(inst_idx) else { return };
        let [bx, by, bw, bh] = inst.bbox;
        let pts = [(a.0 - bx as f32, a.1 - by as f32), (b.0 - bx as f32, b.1 - by as f32)];

        let original = inst.mask.clone();
        let mut carved = vec![false; original.len()];
        for y in 0..bh {
            for x in 0..bw {
                let idx = (y * bw + x) as usize;
                if !original[idx] { continue; }
                carved[idx] = dist_to_polyline(x as f32 + 0.5, y as f32 + 0.5, &pts) > KNIFE_KERF;
            }
        }
        let mut pieces = mask_connected_components(&carved, bw, bh);
        if pieces.len() < 2 { return; }
        reclaim_kerf(&mut pieces, &original, bw, bh, |x, y| {
            dist_to_polyline(x as f32 + 0.5, y as f32 + 0.5, &pts) <= KNIFE_KERF_KEEP
        });

        let class_id = inst.class_id;
        let mut new_instances = Vec::new();
        for piece in pieces {
            if piece.is_empty() { continue; }
            let (mut minx, mut miny, mut maxx, mut maxy) = (bw, bh, 0u32, 0u32);
            for &idx in &piece {
                let (lx, ly) = (idx as u32 % bw, idx as u32 / bw);
                minx = minx.min(lx); miny = miny.min(ly);
                maxx = maxx.max(lx + 1); maxy = maxy.max(ly + 1);
            }
            let (nw, nh) = (maxx - minx, maxy - miny);
            let mut nmask = vec![false; (nw * nh) as usize];
            for &idx in &piece {
                let (lx, ly) = (idx as u32 % bw, idx as u32 / bw);
                nmask[((ly - miny) * nw + (lx - minx)) as usize] = true;
            }
            new_instances.push(RealInstance { bbox: [bx + minx, by + miny, nw, nh], mask: nmask, class_id });
        }
        if new_instances.len() < 2 { return; }
        scene.instances.splice(inst_idx..=inst_idx, new_instances);
        self.selected_instance = None;
        self.overlay_tex = None;
    }

    fn delete_selected(&mut self, scene_idx: usize) {
        let mut idxs: Vec<usize> = self.multi_selected.iter().copied().collect();
        if let Some(ii) = self.selected_instance { idxs.push(ii); }
        self.delete_instances(scene_idx, &idxs);
    }

    // ── export ────────────────────────────────────────────────────────────

    fn export(&mut self, toasts: &mut ToastManager) {
        let Some(root) = self.dataset_root.clone() else { return };
        let summary = dataset_export::export_scenes(&root, &mut self.scenes);
        if !summary.failed.is_empty() {
            for (name, err) in &summary.failed {
                self.log.push(format!("ERROR exporting {name}: {err}"));
            }
            toasts.error(format!("Export finished with {} error(s) — see log", summary.failed.len()));
        } else {
            toasts.success(format!("Exported {} scene(s)", summary.exported));
        }
        self.log.push(format!(
            "Export -> {}: {} written, {} skipped (unreviewed), {} failed",
            root.display(), summary.exported, summary.skipped_unreviewed, summary.failed.len()
        ));
    }

    // ── actions ───────────────────────────────────────────────────────────

    fn model_ok(&self) -> bool {
        self.model_path.as_ref().map(|p| p.exists()).unwrap_or(false)
    }

    fn start(&mut self) {
        let (Some(model), Some(src)) = (self.model_path.clone(), self.source_folder.clone()) else { return };
        let image_paths = list_images(&src);
        if image_paths.is_empty() {
            self.log.push("No images in source folder.".into());
            return;
        }
        self.scenes.clear();
        self.selected_idx = None;
        self.selected_instance = None;
        self.multi_selected.clear();
        self.overlay_tex = None;
        self.image_cache = None;
        self.lab_cache = None;
        self.log.clear();
        self.cancel_flag = Arc::new(AtomicBool::new(false));
        self.progress_done = 0;
        self.progress_total = image_paths.len();
        self.running = true;

        // segment_one's own cutout side-effect (writing per-instance PNGs)
        // needs an output_dir; scratch is fine, Field Review only cares
        // about SegItem's in-memory masks, not those cutout files.
        let output_dir = std::env::temp_dir().join("lacuna_field_review_scratch");
        let _ = std::fs::create_dir_all(&output_dir);

        let (tx, rx) = mpsc::channel();
        self.seg_rx = Some(rx);
        let cfg = SegConfig {
            model_path: model,
            image_paths,
            output_dir,
            imgsz: self.imgsz,
            conf: self.conf,
            alpha_lo: 0.5,
            chroma_min: 0,
        };
        spawn_segmentation(cfg, tx, self.cancel_flag.clone());
    }

    // ── polling ───────────────────────────────────────────────────────────

    fn poll_worker(&mut self, toasts: &mut ToastManager) {
        let mut finished = false;
        if let Some(rx) = &self.seg_rx {
            for msg in rx.try_iter().take(128) {
                match msg {
                    SegMsg::Progress { done, total } => { self.progress_done = done; self.progress_total = total; }
                    SegMsg::Item(item) => {
                        if self.selected_idx.is_none() { self.selected_idx = Some(self.scenes.len()); }
                        self.scenes.push(seg_item_to_scene(item));
                    }
                    SegMsg::Log(l) => self.log.push(l),
                    SegMsg::Error(e) => {
                        self.log.push(format!("ERROR: {e}"));
                        toasts.error(format!("Segmentation failed: {e}"));
                        finished = true;
                    }
                    SegMsg::Finished => finished = true,
                }
            }
        }
        if finished {
            self.running = false;
            self.seg_rx = None;
            toasts.success(format!("Segmentation done — {} scenes", self.scenes.len()));
        }
    }

    fn poll_dialogs(&mut self, toasts: &mut ToastManager) {
        if let Some(rx) = &self.model_rx {
            if let Ok(res) = rx.try_recv() {
                if let Some(p) = res { self.model_path = Some(p); }
                self.model_rx = None;
            }
        }
        if let Some(rx) = &self.source_rx {
            if let Ok(res) = rx.try_recv() {
                if let Some(p) = res {
                    self.source_count = scan_image_count(&p);
                    self.source_folder = Some(p);
                }
                self.source_rx = None;
            }
        }
        if let Some(rx) = &self.dataset_rx {
            if let Ok(res) = rx.try_recv() {
                if let Some(p) = res { self.dataset_root = Some(p); }
                self.dataset_rx = None;
            }
        }
        if let Some(rx) = &self.import_rx {
            if let Ok(res) = rx.try_recv() {
                if let Some(dir) = res {
                    match dataset_export::import_prelabeled(&dir) {
                        Ok(mut new_scenes) => {
                            let n = new_scenes.len();
                            if self.selected_idx.is_none() && !new_scenes.is_empty() {
                                self.selected_idx = Some(self.scenes.len());
                            }
                            self.scenes.append(&mut new_scenes);
                            toasts.success(format!("Imported {n} pre-labeled scene(s)."));
                        }
                        Err(e) => toasts.error(format!("Import failed: {e}")),
                    }
                }
                self.import_rx = None;
            }
        }
        if let Some(rx) = &self.sam_dir_rx {
            if let Ok(res) = rx.try_recv() {
                if let Some(p) = res {
                    self.sam_dir = Some(p);
                    // A new folder means any loaded model/cached embedding
                    // is stale — reload lazily next time the tool is used.
                    // Drops any in-flight load/encode too (its result would
                    // belong to the old folder); the abandoned background
                    // thread finishes harmlessly, its send just goes nowhere.
                    self.sam_model = None;
                    self.sam_embed_cache = None;
                    self.sam_status.clear();
                    self.sam_rx = None;
                    self.sam_busy = false;
                    self.sam_points.clear();
                    self.sam_preview.clear();
                }
                self.sam_dir_rx = None;
            }
        }
    }
}

// ── free helpers ────────────────────────────────────────────────────────────

fn seg_item_to_scene(item: SegItem) -> RealScene {
    let instances = item.instances.into_iter().map(|inst| RealInstance {
        bbox: inst.bbox,
        mask: inst.mask.iter().map(|&m| m != 0).collect(),
        class_id: 0,
    }).collect();
    RealScene {
        filename: item.filename,
        path: item.path,
        size: item.size,
        instances,
        reviewed: false,
        exported_idx: None,
    }
}

fn instance_at(scene: &RealScene, lx: f32, ly: f32) -> Option<usize> {
    let (lxi, lyi) = (lx.round() as i64, ly.round() as i64);
    scene.instances.iter().position(|inst| {
        let [bx, by, bw, bh] = inst.bbox;
        let (ox, oy) = (lxi - bx as i64, lyi - by as i64);
        ox >= 0 && oy >= 0 && (ox as u32) < bw && (oy as u32) < bh
            && inst.mask[(oy as u32 * bw + ox as u32) as usize]
    })
}

fn paint_bbox_outline(ui: &Ui, img_rect: egui::Rect, s: f32, bbox: [u32; 4], color: Color32) {
    let [bx, by, bw, bh] = bbox;
    let pad = 2.0;
    let mn = img_rect.min + egui::vec2(bx as f32 * s - pad, by as f32 * s - pad);
    let mx = img_rect.min + egui::vec2((bx + bw) as f32 * s + pad, (by + bh) as f32 * s + pad);
    ui.painter().rect_stroke(egui::Rect::from_min_max(mn, mx), 1.0, egui::Stroke::new(2.0, color));
}

fn spawn_folder_dialog() -> mpsc::Receiver<Option<PathBuf>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || { let _ = tx.send(rfd::FileDialog::new().pick_folder()); });
    rx
}

fn spawn_file_dialog(exts: &'static [&'static str]) -> mpsc::Receiver<Option<PathBuf>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(rfd::FileDialog::new().add_filter("model", exts).pick_file());
    });
    rx
}

fn path_str(p: &Option<PathBuf>) -> String {
    p.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "\u{2014} not set \u{2014}".to_string())
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 * (1.0 - t) + b as f32 * t).round().clamp(0.0, 255.0) as u8
}
