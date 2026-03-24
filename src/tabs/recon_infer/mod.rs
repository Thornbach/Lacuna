pub mod inference;

use std::{
    path::PathBuf,
    sync::{mpsc, Arc},
    sync::atomic::{AtomicBool, Ordering},
};

use egui::{Color32, RichText, Ui, Context};

use crate::settings::{AppSettings, LeafShape, MarginType};
use crate::widgets::ToastManager;
use inference::{InferConfig, InferMsg, InferResult, spawn_inference, write_csv};

// ── State ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum RightTab { Preview, Results }

pub struct ReconInferTab {
    // ── Settings ───────────────────────────────────────────────────────────
    checkpoint_dir: Option<PathBuf>,
    source_folder:  Option<PathBuf>,
    output_folder:  Option<PathBuf>,
    image_size_px:   u32,
    threshold:       f32,
    pre_damage_pct:  f32,
    tta_enabled:     bool,
    debug_output:    bool,
    leaf_shape:      LeafShape,
    margin_type:     MarginType,

    // ── Background inference ───────────────────────────────────────────────
    infer_rx:       Option<mpsc::Receiver<InferMsg>>,
    cancel_flag:    Arc<AtomicBool>,
    inferring:      bool,
    progress_done:  usize,
    progress_total: usize,

    // ── Results ────────────────────────────────────────────────────────────
    results: Vec<InferResult>,

    // ── Preview (loaded from disk on demand, not kept for every image) ─────
    selected_idx:    Option<usize>,
    input_texture:   Option<egui::TextureHandle>,
    recon_texture:   Option<egui::TextureHandle>,
    /// Background thread loads (input_rgba, recon_rgba, size)
    preview_load_rx: Option<mpsc::Receiver<(Vec<u8>, Vec<u8>, u32)>>,

    // ── File dialog receivers ──────────────────────────────────────────────
    ckpt_rx:   Option<mpsc::Receiver<Option<PathBuf>>>,
    source_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    output_rx: Option<mpsc::Receiver<Option<PathBuf>>>,

    // ── UI state ───────────────────────────────────────────────────────────
    source_count: usize,
    right_tab:    RightTab,
    log:          Vec<String>,
}

impl ReconInferTab {
    pub fn new() -> Self {
        Self {
            checkpoint_dir: None,
            source_folder:  None,
            output_folder:  None,
            image_size_px:   256,
            threshold:       0.35,
            pre_damage_pct:  1.0,
            tta_enabled:     false,
            debug_output:    false,
            leaf_shape:      LeafShape::default(),
            margin_type:     MarginType::default(),

            infer_rx:       None,
            cancel_flag:    Arc::new(AtomicBool::new(false)),
            inferring:      false,
            progress_done:  0,
            progress_total: 0,

            results:        Vec::new(),

            selected_idx:    None,
            input_texture:   None,
            recon_texture:   None,
            preview_load_rx: None,

            ckpt_rx:   None,
            source_rx: None,
            output_rx: None,

            source_count: 0,
            right_tab:    RightTab::Preview,
            log:          Vec::new(),
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    pub fn needs_repaint(&self) -> bool {
        self.inferring || self.preview_load_rx.is_some()
    }

    pub fn is_inferring(&self) -> bool { self.inferring }

    pub fn progress(&self) -> (usize, usize) {
        (self.progress_done, self.progress_total)
    }

    pub fn save_settings(&self, s: &mut AppSettings) {
        let r = &mut s.recon_infer;
        r.last_checkpoint_dir = self.checkpoint_dir.clone();
        r.last_source_folder  = self.source_folder.clone();
        r.last_output_folder  = self.output_folder.clone();
        r.image_size_px       = self.image_size_px;
        r.threshold           = self.threshold;
        r.pre_damage_pct      = self.pre_damage_pct;
        r.tta_enabled         = self.tta_enabled;
        r.leaf_shape          = self.leaf_shape;
        r.margin_type         = self.margin_type;
    }

    pub fn load_settings(&mut self, s: &AppSettings) {
        let r = &s.recon_infer;
        self.checkpoint_dir = r.last_checkpoint_dir.clone();
        self.source_folder  = r.last_source_folder.clone();
        self.output_folder   = r.last_output_folder.clone();
        self.image_size_px   = r.image_size_px;
        self.threshold       = r.threshold;
        self.pre_damage_pct  = r.pre_damage_pct;
        self.tta_enabled     = r.tta_enabled;
        self.leaf_shape      = r.leaf_shape;
        self.margin_type     = r.margin_type;
        if let Some(folder) = &self.source_folder.clone() {
            self.source_count = scan_image_count(folder);
        }
    }

    // ── Main show ─────────────────────────────────────────────────────────────

    pub fn show(&mut self, ui: &mut Ui, ctx: &Context, toasts: &mut ToastManager) {
        self.poll_file_dialogs();
        self.poll_inference(ctx, toasts);
        self.poll_preview_load(ctx);

        egui::SidePanel::left("infer_controls")
            .exact_width(280.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("infer_ctrl_scroll")
                    .show(ui, |ui| self.show_controls(ui, toasts));
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.show_right_panel(ui, ctx);
        });
    }

    // ── Controls panel ────────────────────────────────────────────────────────

    fn show_controls(&mut self, ui: &mut Ui, toasts: &mut ToastManager) {
        ui.add_space(4.0);

        // ── Checkpoint ────────────────────────────────────────────────────────
        ui.label(RichText::new("Checkpoint").strong());
        ui.separator();

        if ui.button("Select checkpoint folder…").clicked() && self.ckpt_rx.is_none() {
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                let _ = tx.send(rfd::FileDialog::new().pick_folder());
            });
            self.ckpt_rx = Some(rx);
        }
        if let Some(p) = &self.checkpoint_dir {
            ui.label(RichText::new(p.display().to_string()).small().color(Color32::GRAY));
            let has_gen = p.join("gen.mpk").exists();
            if has_gen {
                ui.label(RichText::new("gen.mpk found").small().color(Color32::from_rgb(80, 200, 100)));
            } else {
                ui.label(RichText::new("gen.mpk NOT found").small().color(Color32::from_rgb(200, 80, 80)));
            }
        }
        ui.label(RichText::new("Select a checkpoint_best or checkpoint_epoch_XXXX folder.")
            .small().color(Color32::GRAY));

        ui.add_space(8.0);

        // ── Source images ─────────────────────────────────────────────────────
        ui.label(RichText::new("Source images (damaged)").strong());
        ui.separator();

        if ui.button("Source folder…").clicked() && self.source_rx.is_none() {
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                let _ = tx.send(rfd::FileDialog::new().pick_folder());
            });
            self.source_rx = Some(rx);
        }
        if let Some(p) = &self.source_folder {
            ui.label(RichText::new(p.display().to_string()).small().color(Color32::GRAY));
            ui.label(RichText::new(format!("{} images found", self.source_count)).small());
        }

        ui.add_space(8.0);

        // ── Output folder ─────────────────────────────────────────────────────
        ui.label(RichText::new("Output folder").strong());
        ui.separator();

        if ui.button("Output folder…").clicked() && self.output_rx.is_none() {
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                let _ = tx.send(rfd::FileDialog::new().pick_folder());
            });
            self.output_rx = Some(rx);
        }
        if let Some(p) = &self.output_folder {
            ui.label(RichText::new(p.display().to_string()).small().color(Color32::GRAY));
        }

        ui.add_space(8.0);

        // ── Leaf descriptors ──────────────────────────────────────────────────
        ui.label(RichText::new("Leaf Descriptors").strong());
        ui.separator();
        ui.label(RichText::new("Must match what was selected during training.")
            .small().color(Color32::GRAY));
        ui.add_space(4.0);

        egui::Grid::new("infer_descriptor_grid")
            .num_columns(2)
            .spacing([8.0, 6.0])
            .show(ui, |ui| {
                ui.label("Shape:");
                egui::ComboBox::from_id_salt("infer_leaf_shape")
                    .selected_text(self.leaf_shape.label())
                    .show_ui(ui, |ui| {
                        for &shape in LeafShape::ALL {
                            ui.selectable_value(&mut self.leaf_shape, shape, shape.label());
                        }
                    });
                ui.end_row();

                ui.label("Margin:");
                egui::ComboBox::from_id_salt("infer_margin_type")
                    .selected_text(self.margin_type.label())
                    .show_ui(ui, |ui| {
                        for &margin in MarginType::ALL {
                            ui.selectable_value(&mut self.margin_type, margin, margin.label());
                        }
                    });
                ui.end_row();
            });

        ui.add_space(8.0);

        // ── Model settings ────────────────────────────────────────────────────
        ui.label(RichText::new("Model settings").strong());
        ui.separator();

        egui::Grid::new("infer_settings_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label("Image size:");
                egui::ComboBox::from_id_salt("infer_img_sz")
                    .selected_text(format!("{}×{}", self.image_size_px, self.image_size_px))
                    .show_ui(ui, |ui| {
                        for sz in [256u32, 512, 1024] {
                            ui.selectable_value(&mut self.image_size_px, sz,
                                format!("{}×{}", sz, sz));
                        }
                    });
                ui.end_row();

                ui.label("Mask threshold:");
                ui.add(egui::Slider::new(&mut self.threshold, 0.1..=0.9)
                    .fixed_decimals(2));
                ui.end_row();

                ui.label("Pre-damage:")
                    .on_hover_text(
                        "Apply this % of synthetic margin damage before inference.\n\
                         Puts naturally-damaged leaves inside the training distribution\n\
                         so the model reconstructs real damage. 1% is usually enough.\n\
                         Set 0 to disable."
                    );
                ui.add(egui::Slider::new(&mut self.pre_damage_pct, 0.0..=5.0)
                    .suffix("%").fixed_decimals(1));
                ui.end_row();

                ui.label("TTA (4× rotations):")
                    .on_hover_text(
                        "Test-time augmentation: run inference at 0°/90°/180°/270°\n\
                         and average the four probability maps before thresholding.\n\
                         Slightly slower but more robust on non-canonical orientations."
                    );
                ui.checkbox(&mut self.tta_enabled, "Enable");
                ui.end_row();

                ui.label("Debug output:")
                    .on_hover_text(
                        "Save extra PNGs per image:\n\
                         _debug_input.png  — exact input the model saw\n\
                         _debug_prob.png   — raw probability map (jet, before threshold)\n\
                         Use to diagnose wrong predictions."
                    );
                ui.checkbox(&mut self.debug_output, "Save debug PNGs");
                ui.end_row();
            });
        ui.label(RichText::new("Image size must match the checkpoint's training size.")
            .small().color(Color32::GRAY));

        ui.add_space(10.0);

        // ── Start / Cancel ────────────────────────────────────────────────────
        let checkpoint_ok = self.checkpoint_dir.as_ref()
            .map(|p| p.join("gen.mpk").exists())
            .unwrap_or(false);
        let can_start = checkpoint_ok
            && self.source_folder.is_some()
            && self.output_folder.is_some()
            && self.source_count > 0
            && !self.inferring;

        ui.add_enabled_ui(can_start, |ui| {
            if ui.add_sized([ui.available_width(), 32.0],
                egui::Button::new(RichText::new("Run Batch Inference").strong())).clicked() {
                self.start_inference();
            }
        });

        if self.inferring {
            if ui.add_sized([ui.available_width(), 26.0],
                egui::Button::new("Cancel")).clicked() {
                self.cancel_flag.store(true, Ordering::Relaxed);
            }
        }

        // Reason why disabled
        if !can_start && !self.inferring {
            let reason = if self.checkpoint_dir.is_none() {
                Some("Select a checkpoint folder")
            } else if !checkpoint_ok {
                Some("gen.mpk not found in checkpoint folder")
            } else if self.source_folder.is_none() {
                Some("Select a source folder")
            } else if self.output_folder.is_none() {
                Some("Select an output folder")
            } else if self.source_count == 0 {
                Some("No images found in source folder")
            } else {
                None
            };
            if let Some(r) = reason {
                ui.label(RichText::new(r).small().color(Color32::from_rgb(180, 120, 60)));
            }
        }

        // Progress
        if self.inferring || (self.progress_total > 0 && self.progress_done > 0) {
            ui.add_space(6.0);
            ui.label(format!("Progress: {} / {}", self.progress_done, self.progress_total));
            if self.progress_total > 0 {
                let frac = self.progress_done as f32 / self.progress_total as f32;
                ui.add(egui::ProgressBar::new(frac).desired_width(ui.available_width()));
            }
        }

        ui.add_space(8.0);

        // ── CSV export ────────────────────────────────────────────────────────
        if !self.results.is_empty() {
            ui.label(RichText::new("Export").strong());
            ui.separator();

            if ui.add_sized([ui.available_width(), 26.0],
                egui::Button::new("Export results as CSV…")).clicked() {
                let results_ref = &self.results;
                // Open save dialog in a background thread
                let (tx, rx) = mpsc::channel::<Option<PathBuf>>();
                std::thread::spawn(move || {
                    let _ = tx.send(
                        rfd::FileDialog::new()
                            .add_filter("CSV", &["csv"])
                            .set_file_name("recon_results.csv")
                            .save_file()
                    );
                });
                // Collect path immediately (dialog is synchronous in rfd)
                // We use a small spin here — dialog returns quickly
                if let Ok(Some(path)) = rx.recv() {
                    match write_csv(results_ref, &path) {
                        Ok(())  => toasts.success(format!("CSV saved: {}", path.display())),
                        Err(e)  => toasts.error(format!("CSV export failed: {e}")),
                    }
                }
            }

            ui.label(RichText::new(format!(
                "{} images processed", self.results.len()
            )).small().color(Color32::GRAY));
        }

        ui.add_space(8.0);

        // ── Log ───────────────────────────────────────────────────────────────
        if !self.log.is_empty() {
            ui.label(RichText::new("Log").strong());
            ui.separator();
            egui::ScrollArea::vertical()
                .id_salt("infer_log_scroll")
                .max_height(150.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for entry in self.log.iter().rev().take(30) {
                        ui.label(RichText::new(entry).small().monospace());
                    }
                });
        }
    }

    // ── Right panel ───────────────────────────────────────────────────────────

    fn show_right_panel(&mut self, ui: &mut Ui, _ctx: &Context) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.right_tab, RightTab::Preview, "  Preview  ");
            ui.selectable_value(&mut self.right_tab, RightTab::Results, "  Results  ");
        });
        ui.separator();

        match self.right_tab {
            RightTab::Preview => self.show_preview(ui),
            RightTab::Results => self.show_results(ui),
        }
    }

    fn show_preview(&self, ui: &mut Ui) {
        match (&self.input_texture, &self.recon_texture) {
            (Some(inp), Some(rec)) => {
                // Display damaged vs reconstructed side by side
                let avail = ui.available_size();
                let half_w = (avail.x - 16.0) / 2.0;
                let tex_h  = inp.size_vec2().y * (half_w / inp.size_vec2().x);
                let show_h = tex_h.min(avail.y - 24.0);
                let show_w = inp.size_vec2().x * (show_h / inp.size_vec2().y);

                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label(RichText::new("Damaged input").small().color(Color32::GRAY));
                        ui.image((inp.id(), egui::vec2(show_w, show_h)));
                    });
                    ui.add_space(8.0);
                    ui.vertical(|ui| {
                        ui.label(RichText::new("Reconstructed").small().color(Color32::from_rgb(80, 200, 120)));
                        ui.image((rec.id(), egui::vec2(show_w, show_h)));
                    });
                });

                if let Some(idx) = self.selected_idx {
                    if let Some(r) = self.results.get(idx) {
                        ui.add_space(8.0);
                        egui::Grid::new("preview_stats_grid")
                            .num_columns(2)
                            .spacing([12.0, 2.0])
                            .show(ui, |ui| {
                                stat_row(ui, "File",             &r.filename);
                                stat_row(ui, "Input leaf px",    &r.input_leaf_px.to_string());
                                stat_row(ui, "Predicted total",  &r.predicted_total_px.to_string());
                                stat_row(ui, "Damage px",        &r.damage_px.to_string());
                                stat_row(ui, "% Damage",         &format!("{:.1}%", r.pct_damage));
                            });
                    }
                }
            }
            _ => {
                ui.centered_and_justified(|ui| {
                    ui.label(RichText::new(
                        if self.results.is_empty() {
                            "Run batch inference, then select a result row to preview."
                        } else {
                            "Select a row in the Results tab to preview the image pair."
                        }
                    ).color(Color32::GRAY));
                });
            }
        }
    }

    fn show_results(&mut self, ui: &mut Ui) {
        if self.results.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label(RichText::new("No results yet. Run batch inference to populate this table.")
                    .color(Color32::GRAY));
            });
            return;
        }

        // Column header
        ui.horizontal(|ui| {
            ui.add_sized([220.0, 18.0], egui::Label::new(
                RichText::new("Filename").strong().small()));
            ui.add_sized([90.0, 18.0], egui::Label::new(
                RichText::new("Input px").strong().small()));
            ui.add_sized([90.0, 18.0], egui::Label::new(
                RichText::new("Predicted px").strong().small()));
            ui.add_sized([90.0, 18.0], egui::Label::new(
                RichText::new("Damage px").strong().small()));
            ui.add_sized([80.0, 18.0], egui::Label::new(
                RichText::new("% Damage").strong().small()));
        });
        ui.separator();

        let selected_idx = self.selected_idx;
        let mut new_selected = selected_idx;

        egui::ScrollArea::vertical()
            .id_salt("infer_results_scroll")
            .show(ui, |ui| {
                for (i, r) in self.results.iter().enumerate() {
                    let is_selected = selected_idx == Some(i);
                    let dmg_color = damage_color(r.pct_damage);

                    let row_resp = ui.horizontal(|ui| {
                        ui.add_sized([220.0, 18.0],
                            egui::Label::new(RichText::new(&r.filename).small())
                                .truncate());
                        ui.add_sized([90.0, 18.0],
                            egui::Label::new(RichText::new(r.input_leaf_px.to_string()).small()));
                        ui.add_sized([90.0, 18.0],
                            egui::Label::new(RichText::new(r.predicted_total_px.to_string()).small()));
                        ui.add_sized([90.0, 18.0],
                            egui::Label::new(RichText::new(r.damage_px.to_string()).small()));
                        ui.add_sized([80.0, 18.0],
                            egui::Label::new(
                                RichText::new(format!("{:.1}%", r.pct_damage))
                                    .small().color(dmg_color)));
                    });

                    // Highlight selected row
                    if is_selected {
                        let rect = row_resp.response.rect;
                        ui.painter().rect_filled(rect, 0.0, Color32::from_rgba_unmultiplied(80, 160, 240, 30));
                    }

                    if row_resp.response.clicked() {
                        new_selected = Some(i);
                    }
                }
            });

        // Trigger preview load when selection changes
        if new_selected != selected_idx {
            self.selected_idx = new_selected;
            if let Some(idx) = new_selected {
                // Clone paths before the mutable borrow for request_preview
                if let Some(paths) = self.results.get(idx)
                    .map(|r| (r.path.clone(), r.out_path.clone()))
                {
                    self.request_preview_paths(paths.0, paths.1);
                    self.right_tab = RightTab::Preview;
                }
            }
        }
    }

    // ── Preview loading ───────────────────────────────────────────────────────

    fn request_preview_paths(&mut self, input_path: PathBuf, recon_path: PathBuf) {
        let (tx, rx) = mpsc::channel::<(Vec<u8>, Vec<u8>, u32)>();
        self.preview_load_rx = Some(rx);
        self.input_texture = None;
        self.recon_texture = None;

        std::thread::spawn(move || {
            const PREV: u32 = 512;
            let load = |p: &PathBuf| -> Option<Vec<u8>> {
                let img = image::open(p).ok()?;
                let img = img.resize_exact(PREV, PREV, image::imageops::FilterType::Triangle);
                Some(img.to_rgba8().into_raw())
            };
            if let (Some(inp), Some(rec)) = (load(&input_path), load(&recon_path)) {
                let _ = tx.send((inp, rec, PREV));
            }
        });
    }

    fn poll_preview_load(&mut self, ctx: &Context) {
        if let Some(rx) = self.preview_load_rx.take() {
            match rx.try_recv() {
                Ok((inp_rgba, rec_rgba, sz)) => {
                    let sz = sz as usize;
                    self.input_texture = Some(ctx.load_texture(
                        "infer_input",
                        egui::ColorImage::from_rgba_unmultiplied([sz, sz], &inp_rgba),
                        egui::TextureOptions::LINEAR,
                    ));
                    self.recon_texture = Some(ctx.load_texture(
                        "infer_recon",
                        egui::ColorImage::from_rgba_unmultiplied([sz, sz], &rec_rgba),
                        egui::TextureOptions::LINEAR,
                    ));
                }
                Err(mpsc::TryRecvError::Empty) => {
                    // Still loading — put the receiver back
                    self.preview_load_rx = Some(rx);
                    ctx.request_repaint();
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    // Load failed silently — textures stay None
                }
            }
        }
    }

    // ── Inference message polling ─────────────────────────────────────────────

    fn poll_inference(&mut self, ctx: &Context, toasts: &mut ToastManager) {
        let mut msgs = Vec::new();
        let mut done = false;

        if let Some(rx) = &self.infer_rx {
            for _ in 0..64 {
                match rx.try_recv() {
                    Ok(msg) => msgs.push(msg),
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => { done = true; break; }
                }
            }
        }

        for msg in msgs {
            self.handle_msg(msg, ctx, toasts);
        }
        if done { self.infer_rx = None; self.inferring = false; }
    }

    fn handle_msg(&mut self, msg: InferMsg, _ctx: &Context, toasts: &mut ToastManager) {
        match msg {
            InferMsg::Progress { done, total } => {
                self.progress_done  = done;
                self.progress_total = total;
            }
            InferMsg::ImageResult(result) => {
                self.results.push(result);
            }
            InferMsg::Log(msg) => {
                self.push_log(msg);
            }
            InferMsg::Finished => {
                self.inferring = false;
                toasts.success(format!(
                    "Inference complete — {} images processed.", self.results.len()
                ));
                self.push_log("Inference finished.".to_string());
            }
            InferMsg::Error(e) => {
                self.inferring = false;
                toasts.error(format!("Inference error: {e}"));
                self.push_log(format!("ERROR: {e}"));
            }
        }
    }

    fn push_log(&mut self, msg: String) {
        self.log.push(msg);
        if self.log.len() > 500 { self.log.drain(0..100); }
    }

    // ── File dialog polling ───────────────────────────────────────────────────

    fn poll_file_dialogs(&mut self) {
        if let Some(rx) = self.ckpt_rx.take() {
            match rx.try_recv() {
                Ok(Some(path)) => { self.checkpoint_dir = Some(path); }
                Ok(None) => {}
                Err(mpsc::TryRecvError::Empty) => self.ckpt_rx = Some(rx),
                Err(_) => {}
            }
        }
        if let Some(rx) = self.source_rx.take() {
            match rx.try_recv() {
                Ok(Some(path)) => {
                    self.source_count = scan_image_count(&path);
                    self.source_folder = Some(path);
                }
                Ok(None) => {}
                Err(mpsc::TryRecvError::Empty) => self.source_rx = Some(rx),
                Err(_) => {}
            }
        }
        if let Some(rx) = self.output_rx.take() {
            match rx.try_recv() {
                Ok(Some(path)) => { self.output_folder = Some(path); }
                Ok(None) => {}
                Err(mpsc::TryRecvError::Empty) => self.output_rx = Some(rx),
                Err(_) => {}
            }
        }
    }

    // ── Start inference ───────────────────────────────────────────────────────

    fn start_inference(&mut self) {
        let checkpoint_dir = match &self.checkpoint_dir {
            Some(p) => p.clone(),
            None    => return,
        };
        let source = match &self.source_folder {
            Some(p) => p.clone(),
            None    => return,
        };
        let output = match &self.output_folder {
            Some(p) => p.clone(),
            None    => return,
        };

        let image_paths: Vec<PathBuf> = walkdir::WalkDir::new(&source)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file() && is_image(e.path()))
            .map(|e| e.into_path())
            .collect();

        if image_paths.is_empty() { return; }

        self.results.clear();
        self.selected_idx    = None;
        self.input_texture   = None;
        self.recon_texture   = None;
        self.progress_done   = 0;
        self.progress_total  = image_paths.len();

        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel_flag = cancel.clone();

        let (tx, rx) = mpsc::channel::<InferMsg>();
        self.infer_rx = Some(rx);
        self.inferring = true;

        let cfg = InferConfig {
            checkpoint_dir,
            image_paths,
            output_dir:     output,
            image_size:     self.image_size_px as usize,
            threshold:      self.threshold,
            pre_damage_pct: self.pre_damage_pct,
            tta_enabled:    self.tta_enabled,
            debug_output:   self.debug_output,
            leaf_shape:     self.leaf_shape.index(),
            margin_type:    self.margin_type.index(),
        };

        spawn_inference(cfg, tx, cancel);
        self.push_log("Inference started.".to_string());
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn scan_image_count(folder: &PathBuf) -> usize {
    walkdir::WalkDir::new(folder)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file() && is_image(e.path()))
        .count()
}

fn is_image(p: &std::path::Path) -> bool {
    matches!(
        p.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref(),
        Some("png") | Some("tif") | Some("tiff") | Some("jpg") | Some("jpeg")
    )
}

fn damage_color(pct: f32) -> Color32 {
    if pct < 20.0 {
        Color32::from_rgb(80, 200, 100)
    } else if pct < 50.0 {
        Color32::from_rgb(220, 180, 60)
    } else {
        Color32::from_rgb(200, 80, 80)
    }
}

fn stat_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.label(format!("{label}:"));
    ui.label(value);
    ui.end_row();
}
