//! Standalone leaf instance-segmentation tab (YOLO via ONNX Runtime).
//!
//! Slice 1: load a YOLO-seg ONNX model, run it over a source folder on a worker
//! thread, decode instance masks, write per-leaf cutouts, and visualize the
//! selected image with coloured mask overlays + a navigation gallery.

pub mod inference;

use std::{
    path::PathBuf,
    sync::{mpsc, Arc},
    sync::atomic::{AtomicBool, Ordering},
};

use egui::{Color32, Context, RichText, Ui};

use crate::settings::AppSettings;
use crate::ui_kit;
use crate::widgets::ToastManager;
use inference::{scan_image_count, spawn_segmentation, SegConfig, SegItem, SegMsg};

const PALETTE: [[u8; 3]; 8] = [
    [230, 80, 80], [80, 180, 230], [120, 210, 120], [230, 180, 70],
    [180, 120, 220], [70, 210, 200], [230, 120, 180], [160, 200, 80],
];

pub struct LeafSegTab {
    // ── settings ──────────────────────────────────────────────────────────
    source_folder: Option<PathBuf>,
    output_folder: Option<PathBuf>,
    model_path:    Option<PathBuf>,
    conf:  f32,
    iou:   f32,
    imgsz: u32,
    alpha_lo:   f32,   // cutout edge tightness (alpha feather start)
    chroma_min: i32,   // background-chroma rejection (grey/white/black incl. shadowed rim)

    // ── segmentation preview (tune the edge before a full run) ─────────────
    preview_tex:  Option<egui::TextureHandle>,
    preview_rx:   Option<mpsc::Receiver<Result<(Vec<u8>, u32, u32), String>>>,
    preview_busy: bool,
    preview_note: String,

    // ── worker ────────────────────────────────────────────────────────────
    seg_rx:        Option<mpsc::Receiver<SegMsg>>,
    cancel_flag:   Arc<AtomicBool>,
    running:       bool,
    progress_done: usize,
    progress_total: usize,

    // ── results + view ────────────────────────────────────────────────────
    results:      Vec<SegItem>,
    selected_idx: Option<usize>,
    overlay_tex:  Option<(usize, egui::TextureHandle)>, // (results idx, texture)

    // ── file dialogs ──────────────────────────────────────────────────────
    source_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    output_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    model_rx:  Option<mpsc::Receiver<Option<PathBuf>>>,

    source_count: usize,
    log: Vec<String>,
}

impl LeafSegTab {
    pub fn new() -> Self {
        Self {
            source_folder: None,
            output_folder: None,
            model_path:    None,
            conf:  0.25,
            iou:   0.45,
            imgsz: 640,
            alpha_lo:   0.50,
            chroma_min: 28,

            preview_tex:  None,
            preview_rx:   None,
            preview_busy: false,
            preview_note: String::new(),

            seg_rx:        None,
            cancel_flag:   Arc::new(AtomicBool::new(false)),
            running:       false,
            progress_done: 0,
            progress_total: 0,

            results:      Vec::new(),
            selected_idx: None,
            overlay_tex:  None,

            source_rx: None,
            output_rx: None,
            model_rx:  None,

            source_count: 0,
            log: Vec::new(),
        }
    }

    // ── lifecycle ─────────────────────────────────────────────────────────

    pub fn needs_repaint(&self) -> bool { self.running }

    pub fn save_settings(&self, s: &mut AppSettings) {
        let r = &mut s.leaf_seg;
        r.last_source_folder = self.source_folder.clone();
        r.last_output_folder = self.output_folder.clone();
        r.last_model_path    = self.model_path.clone();
        r.conf  = self.conf;
        r.iou   = self.iou;
        r.imgsz = self.imgsz;
    }

    pub fn load_settings(&mut self, s: &AppSettings) {
        let r = &s.leaf_seg;
        self.source_folder = r.last_source_folder.clone();
        self.output_folder = r.last_output_folder.clone();
        self.model_path    = r.last_model_path.clone();
        self.conf  = r.conf;
        self.iou   = r.iou;
        self.imgsz = r.imgsz;
        if let Some(f) = self.source_folder.clone() {
            self.source_count = scan_image_count(&f);
        }
    }

    // ── show ──────────────────────────────────────────────────────────────

    pub fn show(&mut self, ui: &mut Ui, ctx: &Context, toasts: &mut ToastManager) {
        self.poll_dialogs();
        self.poll_worker(toasts);
        self.poll_preview(ctx);

        egui::SidePanel::left("leafseg_controls")
            .exact_width(300.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("leafseg_ctrl_scroll")
                    .show(ui, |ui| self.show_controls(ui));
            });

        egui::TopBottomPanel::bottom("leafseg_gallery")
            .resizable(false)
            .min_height(96.0)
            .show_inside(ui, |ui| self.show_gallery(ui));

        egui::CentralPanel::default().show_inside(ui, |ui| self.show_canvas(ui, ctx));
    }

    fn show_controls(&mut self, ui: &mut Ui) {
        ui_kit::section_header(ui, "Model");
        if ui.button("Model (.onnx)…").clicked() && self.model_rx.is_none() {
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                let _ = tx.send(
                    rfd::FileDialog::new()
                        .add_filter("ONNX", &["onnx"])
                        .pick_file(),
                );
            });
            self.model_rx = Some(rx);
        }
        ui.label(RichText::new(path_str(&self.model_path)).small().color(Color32::GRAY));

        ui_kit::section_header(ui, "Folders");
        if ui.button("Source folder…").clicked() && self.source_rx.is_none() {
            self.source_rx = Some(spawn_folder_dialog());
        }
        ui.label(RichText::new(path_str(&self.source_folder)).small().color(Color32::GRAY));
        if self.source_folder.is_some() {
            ui.label(RichText::new(format!("{} images found", self.source_count)).small());
        }
        if ui.button("Output folder…").clicked() && self.output_rx.is_none() {
            self.output_rx = Some(spawn_folder_dialog());
        }
        ui.label(RichText::new(path_str(&self.output_folder)).small().color(Color32::GRAY));

        // Edge preview on the first source image (tune thresholds in Settings > Leaf Seg).
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            let can_preview = self.model_ok() && self.source_folder.is_some()
                && self.source_count > 0 && !self.preview_busy && !self.running;
            if ui.add_enabled(can_preview, egui::Button::new("🔍 Preview")).clicked() {
                self.start_preview();
            }
            if self.preview_busy { ui_kit::busy(ui, "segmenting…"); }
        });
        if !self.preview_note.is_empty() {
            ui.label(RichText::new(&self.preview_note).small().color(Color32::GRAY));
        }

        ui.add_space(10.0);
        let can_start = self.model_ok()
            && self.source_folder.is_some()
            && self.output_folder.is_some()
            && self.source_count > 0
            && !self.running;
        ui.add_enabled_ui(can_start, |ui| {
            if ui_kit::primary_button(ui, "Run Segmentation").clicked() {
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
            ui.horizontal(|ui| {
                ui_kit::busy(ui, &format!("Segmenting {}/{}", self.progress_done, self.progress_total));
            });
        }

        if !self.log.is_empty() {
            ui_kit::section_header(ui, "Log");
            egui::ScrollArea::vertical().max_height(160.0).id_salt("leafseg_log").show(ui, |ui| {
                for line in self.log.iter().rev().take(200) {
                    ui.label(RichText::new(line).small());
                }
            });
        }
    }

    pub fn show_settings_panel(&mut self, ui: &mut Ui) {
        ui_kit::section_header(ui, "Settings");
        egui::Grid::new("leafseg_settings").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
            ui.label("Confidence:");
            ui.add(egui::Slider::new(&mut self.conf, 0.0..=1.0).fixed_decimals(2));
            ui.end_row();
            ui.label("Image size:");
            egui::ComboBox::from_id_salt("leafseg_imgsz")
                .selected_text(format!("{}", self.imgsz))
                .show_ui(ui, |ui| {
                    for sz in [320u32, 640, 960, 1280] {
                        ui.selectable_value(&mut self.imgsz, sz, format!("{sz}"));
                    }
                });
            ui.end_row();
            ui.label("Cutout edge:")
                .on_hover_text("Cutout edge tightness (alpha feather start).\n\
                                Higher = tighter cut, less background rim.\n\
                                Lower (down to 0.0) = looser/more inclusive edge — \n\
                                try this if real leaf margin is getting cut off.");
            ui.add(egui::Slider::new(&mut self.alpha_lo, 0.0..=0.75).fixed_decimals(2));
            ui.end_row();
            ui.label("Bg chroma reject:")
                .on_hover_text("Drop colourless rim pixels (grey/white/black, incl. the\n\
                                shadowed background next to the leaf). Higher = more\n\
                                aggressive; 0 = off.");
            ui.add(egui::Slider::new(&mut self.chroma_min, 0..=60));
            ui.end_row();
        });
        ui.label(
            RichText::new("Image size must match the exported ONNX (640).")
                .small().color(Color32::GRAY),
        );
    }

    fn show_gallery(&mut self, ui: &mut Ui) {
        ui.add_space(2.0);
        ui.label(RichText::new("Segmented images").small().color(Color32::GRAY));
        egui::ScrollArea::horizontal().id_salt("leafseg_gallery_scroll").show(ui, |ui| {
            ui.horizontal(|ui| {
                for (i, item) in self.results.iter().enumerate() {
                    let label = format!("{}\n{} leaves", item.filename, item.instances.len());
                    if ui.selectable_label(self.selected_idx == Some(i), label).clicked() {
                        self.selected_idx = Some(i);
                        self.overlay_tex = None; // rebuild on next frame
                    }
                }
            });
        });
    }

    fn show_canvas(&mut self, ui: &mut Ui, ctx: &Context) {
        self.ensure_overlay(ctx);
        match &self.overlay_tex {
            Some((_, tex)) => {
                let avail = ui.available_size();
                let sz = tex.size_vec2();
                let scale = (avail.x / sz.x).min(avail.y / sz.y).min(1.0).max(0.01);
                ui.centered_and_justified(|ui| {
                    ui.add(egui::Image::new((tex.id(), sz * scale)));
                });
            }
            None => {
                if let Some(pv) = self.preview_tex.clone() {
                    let avail = ui.available_size();
                    let sz = pv.size_vec2();
                    let scale = (avail.x / sz.x).min(avail.y / sz.y).min(1.0).max(0.01);
                    ui.centered_and_justified(|ui| {
                        ui.add(egui::Image::new((pv.id(), sz * scale)));
                    });
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.label(
                            RichText::new(
                                "Select a model, source & output folder, then Run Segmentation.\n\
                                 Click an image in the gallery below to view its leaf masks.\n\
                                 Tip: use Preview to tune the cutout edge first.",
                            )
                            .color(Color32::GRAY),
                        );
                    });
                }
            }
        }
    }

    fn start_preview(&mut self) {
        let (Some(yolo), Some(src)) = (self.model_path.clone(), self.source_folder.clone()) else { return };
        let images = crate::tabs::leaf_seg::inference::list_images(&src);
        let Some(first) = images.into_iter().next() else {
            self.preview_note = "No source images found.".into();
            return;
        };
        let (alpha_lo, chroma_min) = (self.alpha_lo, self.chroma_min);
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
                    self.preview_tex = Some(ctx.load_texture("leafseg_preview", ci, egui::TextureOptions::LINEAR));
                    self.preview_note = format!("Preview {w}×{h} — tune the sliders + re-preview.");
                }
                Err(e) => self.preview_note = format!("Preview failed: {e}"),
            }
        }
    }

    // ── overlay ───────────────────────────────────────────────────────────

    fn ensure_overlay(&mut self, ctx: &Context) {
        let Some(idx) = self.selected_idx else { return };
        if matches!(&self.overlay_tex, Some((i, _)) if *i == idx) {
            return;
        }
        let Some(item) = self.results.get(idx) else { return };
        let Ok(img) = image::open(&item.path) else { return };
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        let mut px: Vec<u8> = rgba.into_raw(); // RGBA8

        // tint each instance's masked pixels with a palette colour
        for (k, inst) in item.instances.iter().enumerate() {
            let c = PALETTE[k % PALETTE.len()];
            let [bx, by, bw, bh] = inst.bbox;
            for yy in 0..bh {
                for xx in 0..bw {
                    if inst.mask[(yy * bw + xx) as usize] == 0 {
                        continue;
                    }
                    let gx = bx + xx;
                    let gy = by + yy;
                    if gx >= w || gy >= h {
                        continue;
                    }
                    let o = ((gy * w + gx) * 4) as usize;
                    for ch in 0..3 {
                        px[o + ch] = lerp_u8(px[o + ch], c[ch], 0.45);
                    }
                }
            }
        }

        let color: Vec<Color32> = px
            .chunks_exact(4)
            .map(|c| Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]))
            .collect();
        let tex = ctx.load_texture(
            format!("leafseg_overlay_{idx}"),
            egui::ColorImage { size: [w as usize, h as usize], pixels: color },
            egui::TextureOptions::LINEAR,
        );
        self.overlay_tex = Some((idx, tex));
    }

    // ── actions ───────────────────────────────────────────────────────────

    fn model_ok(&self) -> bool {
        self.model_path.as_ref().map(|p| p.exists()).unwrap_or(false)
    }

    fn start(&mut self) {
        let (Some(model), Some(src), Some(out)) =
            (self.model_path.clone(), self.source_folder.clone(), self.output_folder.clone())
        else { return };

        let image_paths = inference::list_images(&src);
        if image_paths.is_empty() {
            self.log.push("No images in source folder.".into());
            return;
        }
        self.results.clear();
        self.selected_idx = None;
        self.overlay_tex = None;
        self.log.clear();
        self.cancel_flag = Arc::new(AtomicBool::new(false));
        self.progress_done = 0;
        self.progress_total = image_paths.len();
        self.running = true;

        let (tx, rx) = mpsc::channel();
        self.seg_rx = Some(rx);
        let cfg = SegConfig {
            model_path: model,
            image_paths,
            output_dir: out,
            imgsz: self.imgsz,
            conf: self.conf,
            alpha_lo: self.alpha_lo,
            chroma_min: self.chroma_min,
        };
        spawn_segmentation(cfg, tx, self.cancel_flag.clone());
    }

    // ── polling ───────────────────────────────────────────────────────────

    fn poll_worker(&mut self, toasts: &mut ToastManager) {
        let mut finished = false;
        if let Some(rx) = &self.seg_rx {
            for msg in rx.try_iter().take(128) {
                match msg {
                    SegMsg::Progress { done, total } => {
                        self.progress_done = done;
                        self.progress_total = total;
                    }
                    SegMsg::Item(item) => {
                        if self.selected_idx.is_none() {
                            self.selected_idx = Some(self.results.len());
                        }
                        self.results.push(item);
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
            toasts.success(format!("Segmentation done — {} images", self.results.len()));
        }
    }

    fn poll_dialogs(&mut self) {
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
        if let Some(rx) = &self.output_rx {
            if let Ok(res) = rx.try_recv() {
                if let Some(p) = res { self.output_folder = Some(p); }
                self.output_rx = None;
            }
        }
    }
}

// ── free helpers ────────────────────────────────────────────────────────────

fn spawn_folder_dialog() -> mpsc::Receiver<Option<PathBuf>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(rfd::FileDialog::new().pick_folder());
    });
    rx
}

fn path_str(p: &Option<PathBuf>) -> String {
    p.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "— not set —".to_string())
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 * (1.0 - t) + b as f32 * t).round().clamp(0.0, 255.0) as u8
}
