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
use egui_plot::{Plot, Points};

use crate::settings::{AppDefaults, AppSettings};
use crate::tabs::leaf_seg::inference::{list_images, scan_image_count};
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

#[derive(Clone, Copy)]
enum Pick { Source, Output, Yolo, Dino, Bank, Meta, Recon, Head }

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
    head_tau:        f32,
    head_grow:       f32,
    tile_size:       u32,
    margin_erode_px: u32,
    conf:            f32,
    seg_alpha_lo:    f32,   // YOLO cutout edge tightness (feather start)
    seg_chroma_min:  i32,   // YOLO cutout background-chroma rejection

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
    cluster_names:    HashMap<i32, String>,

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
            head_tau:        0.85,
            head_grow:       0.7,
            tile_size:       256,
            margin_erode_px: 6,
            conf:            0.25,
            seg_alpha_lo:    0.50,
            seg_chroma_min:  28,

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
            cluster_names:    HashMap::new(),

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

    pub fn needs_repaint(&self) -> bool { self.running }

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
        r.head_tau           = self.head_tau;
        r.head_grow          = self.head_grow;
        r.tile_size          = self.tile_size;
        r.margin_erode_px    = self.margin_erode_px;
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
        self.head_tau      = r.head_tau;
        self.head_grow     = r.head_grow;
        self.tile_size     = r.tile_size;
        self.margin_erode_px = r.margin_erode_px;
        if let Some(f) = self.source_folder.clone() {
            self.source_count = scan_image_count(&f);
        }
    }

    // ── show ──────────────────────────────────────────────────────────────

    pub fn show(&mut self, ui: &mut Ui, ctx: &Context, toasts: &mut ToastManager) {
        self.poll_pick();
        self.poll_worker(toasts);
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
        egui::CentralPanel::default().show_inside(ui, |ui| self.show_canvas(ui, ctx));
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
            }
        } else {
            self.pick_row(ui, "Coreset bank (.bin)", Pick::Bank);
            self.pick_row(ui, "Detector meta (.json)", Pick::Meta);
        }

        ui_kit::section_header(ui, "Models (optional)");
        self.pick_row(ui, "Recon checkpoint (optional)", Pick::Recon);

        ui_kit::section_header(ui, "Folders");
        self.pick_row(ui, "Source folder", Pick::Source);
        if self.source_folder.is_some() {
            ui.label(RichText::new(format!("{} images found", self.source_count)).small());
        }
        self.pick_row(ui, "Output folder", Pick::Output);

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
                                (may nibble soft leaf edges).");
            ui.add(egui::Slider::new(&mut self.seg_alpha_lo, 0.50..=0.75).fixed_decimals(2));
            ui.end_row();
            ui.label("Bg chroma reject:")
                .on_hover_text("Drop colourless rim pixels (grey/white/black, incl. the\n\
                                shadowed background next to the leaf). Higher = more\n\
                                aggressive; 0 = off.");
            ui.add(egui::Slider::new(&mut self.seg_chroma_min, 0..=60));
            ui.end_row();
        });

        // Segmentation edge preview (runs YOLO on the first source image so you can
        // dial the cutout edge before committing to a full pipeline run).
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
        let can_start = self.all_paths_ok() && self.source_count > 0 && !self.running;
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
            let need = if self.fewshot_active() {
                "Set YOLO + DINO + few-shot head + source/output folders."
            } else {
                "Set YOLO + DINO + coreset bank + meta + source/output folders."
            };
            ui.label(RichText::new(need).small().color(Color32::GRAY));
        }

        if !self.log.is_empty() {
            ui_kit::section_header(ui, "Log");
            egui::ScrollArea::vertical().max_height(160.0).id_salt("pipeline_log").show(ui, |ui| {
                for line in self.log.iter().rev().take(200) {
                    ui.label(RichText::new(line).small());
                }
            });
        }
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

    fn show_canvas(&mut self, ui: &mut Ui, ctx: &Context) {
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
        let (area, resp) = ui.allocate_exact_size(avail, egui::Sense::click());
        let img_rect = egui::Rect::from_center_size(area.center(), disp);
        egui::Image::new((tex.id(), disp)).paint_at(ui, img_rect);
        let s = img_rect.width() / sz.x.max(1.0);

        // outline mode: draw smooth vector contours of the visible regions
        if self.overlay_outline {
            let sel = self.selected_cluster;
            for (ri, r) in self.regions.iter().enumerate() {
                if r.leaf != leaf_idx || self.removed.contains(&ri) {
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
        // highlight the selected anomaly with a bounding box on the leaf
        if let Some(ri) = self.selected_region {
            if let Some(r) = self.regions.get(ri) {
                if r.leaf == leaf_idx && !self.removed.contains(&ri) {
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
        // click a region on the leaf -> select it (highlights its gallery tile)
        if resp.clicked() {
            if let Some(p) = resp.interact_pointer_pos() {
                let lx = (p.x - img_rect.min.x) / s.max(1e-3);
                let ly = (p.y - img_rect.min.y) / s.max(1e-3);
                self.select_region_at(leaf_idx, lx, ly);
            }
        }
    }

    /// Find the anomaly region at leaf-pixel (lx, ly) on `leaf_idx` (smallest match)
    /// and select it — highlights its gallery tile and jumps to its page.
    fn select_region_at(&mut self, leaf_idx: usize, lx: f32, ly: f32) {
        if lx < 0.0 || ly < 0.0 {
            return;
        }
        let (px, py) = (lx as u32, ly as u32);
        let mut best: Option<usize> = None;
        let mut best_area = u32::MAX;
        for (i, r) in self.regions.iter().enumerate() {
            if r.leaf != leaf_idx || self.removed.contains(&i) {
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
        if let Some(i) = best {
            self.selected_region = Some(i);
            self.selected_cluster = Some(self.labels[i]);
            // jump the gallery to the page that shows this region
            let cl = self.labels[i];
            let pos = (0..self.regions.len())
                .filter(|&j| !self.removed.contains(&j) && self.labels[j] == cl)
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

        // reconstruction preview: tint the FILLED-IN area (reconstructed leaf where
        // the visible cutout is missing) in cyan, so the damage reads as holes in
        // the whole intact leaf. Painted first; anomalies draw on top.
        if self.show_recon && !leaf.recon_mask.is_empty() {
            let rs = worker::RECON_PREVIEW;
            for y in 0..h {
                let my = (y * rs / h.max(1)).min(rs - 1);
                for x in 0..w {
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
                if self.removed.contains(&ri) {
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
                paint_region(&mut px, w, h, r, cluster_color(self.labels[ri]), self.overlay_alpha);
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
        self.show_leaf_morphology(ui);

        // reconstruction preview toggle (only when recon ran for this leaf)
        let has_recon = self.selected_idx.and_then(|i| self.results.get(i))
            .map_or(false, |l| !l.recon_mask.is_empty());
        if has_recon {
            ui.checkbox(&mut self.show_recon, "Show reconstruction")
                .on_hover_text("Tint (under the anomalies) the area the model reconstructed —\n\
                                where the leaf was damaged/missing — so you see the whole intact\n\
                                leaf with the damage as holes.");
        }

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
                    .filter(|i| !self.removed.contains(i))
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
                    if self.removed.contains(&i) {
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
                        if self.removed.contains(&ri) {
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
                        if ui.add(egui::TextEdit::singleline(&mut name).desired_width(110.0)).changed() {
                            self.cluster_names.insert(cid, name);
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
        ui.separator();

        // ── flywheel: persist this session's curation as training labels ──
        if ui.button("💾 Save curations for retraining")
            .on_hover_text("Write each kept region (labeled by its cluster name) and each\n\
                            removed region (as a reject) to <output>/curations/ as crops +\n\
                            labels.jsonl. They accumulate across runs and feed the Train tab\n\
                            so the model improves the more you use it.")
            .clicked()
        {
            self.save_curations(toasts);
        }
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
        let filtered: Vec<usize> = (0..self.regions.len())
            .filter(|&i| {
                !self.removed.contains(&i)
                    && self.selected_cluster.map_or(true, |c| self.labels[i] == c)
            })
            .collect();
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
        });
        ui.label(RichText::new("click = highlight on leaf · right-click = remove")
            .small().color(Color32::DARK_GRAY));
        let show_idxs: Vec<usize> =
            filtered.iter().copied().skip(self.gallery_page * PER_PAGE).take(PER_PAGE).collect();
        for &i in &show_idxs {
            self.ensure_region_thumb(ctx, i);
        }
        egui::ScrollArea::vertical().id_salt("anomaly_gallery").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                for &i in &show_idxs {
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
                        if resp.clicked() {
                            self.selected_idx = Some(self.regions[i].leaf);
                            self.selected_cluster = Some(self.labels[i]);
                            self.selected_region = Some(i);
                            self.overlay_tex = None;
                        }
                        if resp.secondary_clicked() {
                            self.removed.insert(i);
                            if self.selected_region == Some(i) {
                                self.selected_region = None;
                            }
                            self.overlay_tex = None;
                        }
                    }
                }
            });
        });
    }

    // ── actions / polling ─────────────────────────────────────────────────

    fn all_paths_ok(&self) -> bool {
        let ex = |p: Option<PathBuf>| p.map(|p| p.exists()).unwrap_or(false);
        // detector: the few-shot head replaces the PatchCore bank+meta.
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
        self.cluster_names.clear();
        self.selected_cluster = None;
        self.selected_region = None;
        self.gallery_page = 0;
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
                head_tau: self.head_tau,
                head_grow: self.head_grow.min(self.head_tau),
                seg_alpha_lo: self.seg_alpha_lo,
                seg_chroma_min: self.seg_chroma_min,
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
            self.build_clusters();
            self.overlay_tex = None; // rebuild to reflect cluster colours
        }
        if finished {
            self.running = false;
            self.rx = None;
            toasts.success(format!("Pipeline done — {} leaves, {} clusters", self.results.len(), self.clusters.len()));
        }
    }

    /// Flywheel: persist the user's curation — each kept region labeled by its
    /// cluster name (a positive), each removed region as a reject — to
    /// `<output>/curations/` as RGBA crops + an append-only `labels.jsonl`. Labels
    /// accumulate across runs and feed the Train tab. The crop IMAGE is stored (not
    /// DINO features) so a future backbone re-featurizes the whole label history.
    fn save_curations(&self, toasts: &mut ToastManager) {
        let Some(out) = self.output_folder.clone() else {
            toasts.error("Set an output folder first.");
            return;
        };
        if self.regions.is_empty() {
            toasts.error("Nothing to save — run + curate first.");
            return;
        }
        let labels_dir = out.join("curations").join("labels");
        if let Err(e) = std::fs::create_dir_all(&labels_dir) {
            toasts.error(format!("curations dir: {e}"));
            return;
        }
        let run = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut jsonl = String::new();
        let (mut n_conf, mut n_rej) = (0usize, 0usize);
        for (i, r) in self.regions.iter().enumerate() {
            let cid = self.labels[i];
            let removed = self.removed.contains(&i);
            if cid < 0 && !removed {
                continue; // unactioned noise: not a label
            }
            let family = if removed {
                "rejected".to_string()
            } else {
                self.cluster_names.get(&cid).cloned().unwrap_or_else(|| format!("Cluster {cid}"))
            };
            let fname = format!("{run}_{i}.png");
            if let Some(img) = image::RgbaImage::from_raw(r.crop_size, r.crop_size, r.crop.clone()) {
                let _ = img.save(labels_dir.join(&fname));
            }
            let src = self.results.get(r.leaf).map(|l| l.src.display().to_string()).unwrap_or_default();
            jsonl.push_str(&format!(
                "{{\"crop\":\"{}\",\"family\":\"{}\",\"source\":\"{}\",\"leaf_src\":\"{}\",\"ts\":{}}}\n",
                fname, json_escape(&family), if removed { "reject" } else { "confirm" },
                json_escape(&src), run,
            ));
            if removed { n_rej += 1 } else { n_conf += 1 }
        }
        use std::io::Write;
        match std::fs::OpenOptions::new().create(true).append(true)
            .open(out.join("curations").join("labels.jsonl"))
        {
            Ok(mut f) => {
                let _ = f.write_all(jsonl.as_bytes());
                toasts.success(format!("Saved {n_conf} labels + {n_rej} rejects → curations/"));
            }
            Err(e) => toasts.error(format!("labels.jsonl: {e}")),
        }
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
            if self.removed.contains(&i) {
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
                if r.leaf == li && !self.removed.contains(&ri) {
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

    fn build_clusters(&mut self) {
        let mut by_label: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &l) in self.labels.iter().enumerate() {
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
