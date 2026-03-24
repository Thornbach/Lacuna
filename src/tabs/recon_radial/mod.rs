pub mod model;
pub mod training;
pub mod inference;

use std::{
    path::PathBuf,
    sync::{mpsc, Arc},
    sync::atomic::{AtomicBool, Ordering},
};

use egui::{Color32, RichText, Ui, Context};
use egui_plot::{Legend, Line, Plot, PlotPoints};

use crate::settings::AppSettings;
use crate::widgets::ToastManager;
use crate::tabs::recon_train::training::DamageParams;
use training::{RadialTrainConfig, RadialTrainMsg, spawn_radial_training};
use inference::{RadialInferConfig, RadialInferMsg, RadialInferResult, spawn_radial_inference, write_csv};

// ── Enums ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Mode { Train, Infer }

#[derive(Clone, Copy, PartialEq)]
enum TrainRightTab { LossCurve, SampleGrid }

#[derive(Clone, Copy, PartialEq)]
enum InferRightTab { Results, Preview }

// ── Tab state ─────────────────────────────────────────────────────────────────

pub struct ReconRadialTab {
    mode: Mode,

    // ── Training state ────────────────────────────────────────────────────────
    source_folder:  Option<PathBuf>,
    output_folder:  Option<PathBuf>,
    resume_folder:  Option<PathBuf>,

    epochs:            usize,
    batch_size:        usize,
    learning_rate:     f64,
    checkpoint_every:  usize,
    sample_grid_every: usize,
    image_size_px:     u32,
    curriculum_epochs: usize,
    accum_steps:       usize,
    damage_loss_weight: f32,

    damage_min_pct:     f32,
    damage_max_pct:     f32,
    curriculum_max_pct: f32,
    zero_damage_prob:   f32,
    coastal_enabled:    bool,  coastal_weight:  f32,
    snake_enabled:      bool,  snake_weight:    f32,
    lobe_enabled:       bool,  lobe_weight:     f32,

    training:       bool,
    cancel_flag:    Arc<AtomicBool>,
    rx:             Option<mpsc::Receiver<RadialTrainMsg>>,
    log_lines:      Vec<String>,
    current_epoch:  usize,
    total_epochs:   usize,

    b_step:   Vec<f64>,
    b_loss:   Vec<f64>,
    e_step:   Vec<f64>,
    e_mse:    Vec<f64>,
    e_b_mae:  Vec<f64>,

    latest_mse:        Option<f32>,
    latest_b_mae:      Option<f32>,
    latest_checkpoint: Option<String>,

    sample_texture: Option<egui::TextureHandle>,
    sample_epoch:   usize,

    source_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    output_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    resume_rx: Option<mpsc::Receiver<Option<PathBuf>>>,

    train_right_tab: TrainRightTab,
    source_count:    usize,

    // ── Inference state ───────────────────────────────────────────────────────
    infer_ckpt_dir:    Option<PathBuf>,
    infer_src_folder:  Option<PathBuf>,
    infer_out_folder:  Option<PathBuf>,
    infer_image_size:  u32,

    inferring:         bool,
    infer_cancel:      Arc<AtomicBool>,
    infer_rx:          Option<mpsc::Receiver<RadialInferMsg>>,
    infer_progress:    (usize, usize),  // (done, total)
    infer_results:     Vec<RadialInferResult>,
    infer_log:         Vec<String>,

    infer_selected:    Option<usize>,
    infer_inp_texture: Option<egui::TextureHandle>,
    infer_rec_texture: Option<egui::TextureHandle>,
    infer_preview_rx:  Option<mpsc::Receiver<(Vec<u8>, Vec<u8>, u32)>>,

    infer_ckpt_rx:   Option<mpsc::Receiver<Option<PathBuf>>>,
    infer_src_rx:    Option<mpsc::Receiver<Option<PathBuf>>>,
    infer_out_rx:    Option<mpsc::Receiver<Option<PathBuf>>>,
    infer_src_count: usize,

    infer_right_tab: InferRightTab,
}

impl ReconRadialTab {
    pub fn new() -> Self {
        Self {
            mode: Mode::Train,

            source_folder:  None,
            output_folder:  None,
            resume_folder:  None,

            epochs:            500,
            batch_size:        16,
            learning_rate:     1e-3,
            checkpoint_every:  20,
            sample_grid_every: 10,
            image_size_px:     512,
            curriculum_epochs: 50,
            accum_steps:       1,
            damage_loss_weight: 5.0,

            damage_min_pct:     5.0,
            damage_max_pct:     75.0,
            curriculum_max_pct: 30.0,
            zero_damage_prob:   0.20,
            coastal_enabled:    true,  coastal_weight:  0.70,
            snake_enabled:      true,  snake_weight:    0.05,
            lobe_enabled:       true,  lobe_weight:     0.50,

            training:       false,
            cancel_flag:    Arc::new(AtomicBool::new(false)),
            rx:             None,
            log_lines:      Vec::new(),
            current_epoch:  0,
            total_epochs:   0,

            b_step:  Vec::new(),
            b_loss:  Vec::new(),
            e_step:  Vec::new(),
            e_mse:   Vec::new(),
            e_b_mae: Vec::new(),

            latest_mse:        None,
            latest_b_mae:      None,
            latest_checkpoint: None,

            sample_texture: None,
            sample_epoch:   0,

            source_rx: None,
            output_rx: None,
            resume_rx: None,

            train_right_tab: TrainRightTab::LossCurve,
            source_count:    0,

            infer_ckpt_dir:    None,
            infer_src_folder:  None,
            infer_out_folder:  None,
            infer_image_size:  512,

            inferring:         false,
            infer_cancel:      Arc::new(AtomicBool::new(false)),
            infer_rx:          None,
            infer_progress:    (0, 0),
            infer_results:     Vec::new(),
            infer_log:         Vec::new(),

            infer_selected:    None,
            infer_inp_texture: None,
            infer_rec_texture: None,
            infer_preview_rx:  None,

            infer_ckpt_rx:   None,
            infer_src_rx:    None,
            infer_out_rx:    None,
            infer_src_count: 0,

            infer_right_tab: InferRightTab::Results,
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    pub fn needs_repaint(&self) -> bool {
        self.training || self.inferring || self.infer_preview_rx.is_some()
    }
    pub fn is_training(&self)    -> bool { self.training }
    pub fn epoch_progress(&self) -> (usize, usize) { (self.current_epoch, self.total_epochs) }

    pub fn save_settings(&self, s: &mut AppSettings) {
        let r = &mut s.radial_infer;
        r.last_checkpoint_dir = self.infer_ckpt_dir.clone();
        r.last_source_folder  = self.infer_src_folder.clone();
        r.last_output_folder  = self.infer_out_folder.clone();
        r.image_size_px       = self.infer_image_size;
    }

    pub fn load_settings(&mut self, s: &AppSettings) {
        let r = &s.radial_infer;
        self.infer_ckpt_dir   = r.last_checkpoint_dir.clone();
        self.infer_src_folder = r.last_source_folder.clone();
        self.infer_out_folder = r.last_output_folder.clone();
        self.infer_image_size = r.image_size_px;
        if let Some(f) = &self.infer_src_folder.clone() {
            self.infer_src_count = scan_image_count(f);
        }
    }

    // ── Main show ─────────────────────────────────────────────────────────────

    pub fn show(&mut self, ui: &mut Ui, ctx: &Context, toasts: &mut ToastManager) {
        self.poll_file_dialogs();
        self.handle_msgs(ctx, toasts);
        self.poll_infer_msgs(ctx, toasts);
        self.poll_infer_preview(ctx);

        egui::SidePanel::left("recon_radial_controls")
            .exact_width(290.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("recon_radial_ctrl_scroll")
                    .show(ui, |ui| { self.show_controls(ui, toasts); });
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.show_right_panel(ui, ctx);
        });
    }

    // ── Controls ──────────────────────────────────────────────────────────────

    fn show_controls(&mut self, ui: &mut Ui, toasts: &mut ToastManager) {
        ui.add_space(4.0);

        // Mode selector
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.mode, Mode::Train, "  Train  ");
            ui.selectable_value(&mut self.mode, Mode::Infer, "  Infer  ");
        });
        ui.separator();
        ui.add_space(4.0);

        match self.mode {
            Mode::Train => self.show_train_controls(ui),
            Mode::Infer => self.show_infer_controls(ui, toasts),
        }
    }

    // ── Training controls ─────────────────────────────────────────────────────

    fn show_train_controls(&mut self, ui: &mut Ui) {
        // Backend banner
        {
            let backend = model::backend_name();
            let is_cpu  = backend == "CPU";
            let (bg, fg, label) = if is_cpu {
                (Color32::from_rgb(160, 60, 10), Color32::WHITE, "CPU — training will be very slow")
            } else if backend == "CUDA" {
                (Color32::from_rgb(30, 110, 30), Color32::WHITE, "GPU ready (CUDA)")
            } else {
                (Color32::from_rgb(30, 80, 140), Color32::WHITE, "GPU ready (wgpu)")
            };
            egui::Frame::none()
                .fill(bg)
                .inner_margin(egui::Margin::symmetric(8.0, 5.0))
                .rounding(egui::Rounding::same(4.0))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(RichText::new(label).color(fg).strong().small());
                });
        }

        ui.add_space(4.0);
        egui::Frame::none()
            .fill(Color32::from_rgb(30, 50, 30))
            .inner_margin(egui::Margin::symmetric(8.0, 5.0))
            .rounding(egui::Rounding::same(4.0))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new(
                    "Radial profile completion — margin damage only.\n\
                     Interior spots/holes are correctly ignored."
                ).small().color(Color32::from_rgb(160, 220, 160)));
            });

        ui.add_space(6.0);

        // Data
        ui.label(RichText::new("Data").strong());
        ui.separator();

        if ui.button("Source folder…").clicked() && self.source_rx.is_none() {
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || { let _ = tx.send(rfd::FileDialog::new().pick_folder()); });
            self.source_rx = Some(rx);
        }
        if let Some(f) = &self.source_folder {
            ui.label(RichText::new(f.display().to_string()).small().color(Color32::GRAY));
            ui.label(RichText::new(format!("{} images", self.source_count)).small());
        }

        ui.add_space(4.0);
        if ui.button("Output folder…").clicked() && self.output_rx.is_none() {
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || { let _ = tx.send(rfd::FileDialog::new().pick_folder()); });
            self.output_rx = Some(rx);
        }
        if let Some(f) = &self.output_folder {
            ui.label(RichText::new(f.display().to_string()).small().color(Color32::GRAY));
        }

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.button("Resume checkpoint…").clicked() && self.resume_rx.is_none() {
                let (tx, rx) = mpsc::channel();
                std::thread::spawn(move || { let _ = tx.send(rfd::FileDialog::new().pick_folder()); });
                self.resume_rx = Some(rx);
            }
            if self.resume_folder.is_some() && ui.small_button("Clear").clicked() {
                self.resume_folder = None;
            }
        });
        if let Some(f) = &self.resume_folder {
            ui.label(RichText::new(format!("Resume: {}",
                f.file_name().and_then(|n| n.to_str()).unwrap_or("?")))
                .small().color(Color32::from_rgb(100, 180, 120)));
        }

        ui.add_space(8.0);

        // Damage
        ui.label(RichText::new("Damage — margin only").strong());
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Min %");
            ui.add(egui::Slider::new(&mut self.damage_min_pct, 1.0..=30.0).suffix("%"));
        });
        ui.horizontal(|ui| {
            ui.label("Max %");
            ui.add(egui::Slider::new(&mut self.damage_max_pct, 10.0..=95.0).suffix("%"));
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.coastal_enabled, "Coastal ");
            if self.coastal_enabled {
                ui.add(egui::Slider::new(&mut self.coastal_weight, 0.0..=1.0)
                    .show_value(true).fixed_decimals(2));
            }
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.snake_enabled, "Snake   ");
            if self.snake_enabled {
                ui.add(egui::Slider::new(&mut self.snake_weight, 0.0..=1.0)
                    .show_value(true).fixed_decimals(2));
            }
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.lobe_enabled, "Lobes   ");
            if self.lobe_enabled {
                ui.add(egui::Slider::new(&mut self.lobe_weight, 0.0..=1.0)
                    .show_value(true).fixed_decimals(2));
            }
        });
        ui.horizontal(|ui| {
            ui.label("0% prob:");
            ui.add(egui::Slider::new(&mut self.zero_damage_prob, 0.0..=0.5).fixed_decimals(2));
        });

        ui.add_space(8.0);

        // Loss
        ui.label(RichText::new("Loss").strong());
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Damage region weight:").on_hover_text(
                "Multiplier on MSE loss in the damaged (missing) region.\n\
                 Higher = penalise incomplete reconstruction more.");
            ui.add(egui::Slider::new(&mut self.damage_loss_weight, 1.0..=20.0).fixed_decimals(1));
        });

        ui.add_space(8.0);

        // Training settings
        ui.label(RichText::new("Training").strong());
        ui.separator();

        egui::Grid::new("radial_train_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label("Image size:");
                egui::ComboBox::from_id_salt("radial_img_sz")
                    .selected_text(format!("{}×{}", self.image_size_px, self.image_size_px))
                    .show_ui(ui, |ui| {
                        for sz in [256u32, 512] {
                            ui.selectable_value(&mut self.image_size_px, sz,
                                format!("{}×{}", sz, sz));
                        }
                    });
                ui.end_row();

                ui.label("Epochs:");
                ui.add(egui::DragValue::new(&mut self.epochs).range(1..=10000));
                ui.end_row();

                ui.label("Batch size:");
                ui.add(egui::DragValue::new(&mut self.batch_size).range(1..=128));
                ui.end_row();

                ui.label("Learning rate:");
                ui.add(egui::DragValue::new(&mut self.learning_rate).speed(1e-5).fixed_decimals(5));
                ui.end_row();

                ui.label("Grad accum:");
                ui.add(egui::DragValue::new(&mut self.accum_steps).range(1..=16));
                ui.end_row();

                ui.label("Checkpoint every:");
                ui.add(egui::DragValue::new(&mut self.checkpoint_every).range(1..=200).suffix(" ep"));
                ui.end_row();

                ui.label("Sample grid every:");
                ui.add(egui::DragValue::new(&mut self.sample_grid_every).range(1..=100).suffix(" ep"));
                ui.end_row();

                ui.label("Curriculum epochs:");
                ui.add(egui::DragValue::new(&mut self.curriculum_epochs).range(0..=500).suffix(" ep"));
                ui.end_row();

                ui.label("Curriculum max:");
                ui.add(egui::Slider::new(&mut self.curriculum_max_pct, 5.0..=80.0).suffix("%"));
                ui.end_row();
            });

        ui.add_space(10.0);

        // Start / Cancel
        let can_start = self.source_folder.is_some()
            && self.output_folder.is_some()
            && self.source_count > 1
            && !self.training;

        ui.add_enabled_ui(can_start, |ui| {
            if ui.add_sized(
                [ui.available_width(), 32.0],
                egui::Button::new(RichText::new("Start Radial Training").strong()),
            ).clicked() {
                self.start_training();
            }
        });
        if self.training {
            if ui.add_sized([ui.available_width(), 26.0], egui::Button::new("Cancel")).clicked() {
                self.cancel_flag.store(true, Ordering::Relaxed);
            }
        }

        if !can_start && !self.training {
            let reason = if self.source_folder.is_none()      { "Select a source folder" }
                         else if self.output_folder.is_none() { "Select an output folder" }
                         else if self.source_count <= 1       { "Need at least 2 images" }
                         else                                  { "" };
            if !reason.is_empty() {
                ui.label(RichText::new(reason).small().color(Color32::from_rgb(180, 120, 60)));
            }
        }

        ui.add_space(8.0);

        // Status
        if self.training || self.current_epoch > 0 {
            ui.label(RichText::new("Status").strong());
            ui.separator();
            ui.label(format!("Epoch: {} / {}", self.current_epoch, self.total_epochs));
            if let Some(mse) = self.latest_mse {
                ui.label(format!("Val MSE:      {:.6}", mse));
            }
            if let Some(b_mae) = self.latest_b_mae {
                ui.label(format!("Boundary MAE: {:.4}", b_mae));
            }
            if let Some(ckpt) = &self.latest_checkpoint {
                ui.label(RichText::new(format!("Last checkpoint:\n{ckpt}")).small().color(Color32::GRAY));
            }
            ui.add_space(6.0);
        }

        if !self.log_lines.is_empty() {
            ui.label(RichText::new("Log").strong());
            ui.separator();
            egui::ScrollArea::vertical()
                .id_salt("recon_radial_log")
                .max_height(150.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for line in self.log_lines.iter().rev().take(30) {
                        ui.label(RichText::new(line).small().monospace());
                    }
                });
        }
    }

    // ── Inference controls ────────────────────────────────────────────────────

    fn show_infer_controls(&mut self, ui: &mut Ui, toasts: &mut ToastManager) {
        // Checkpoint
        ui.label(RichText::new("Checkpoint").strong());
        ui.separator();

        if ui.button("Select checkpoint folder…").clicked() && self.infer_ckpt_rx.is_none() {
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || { let _ = tx.send(rfd::FileDialog::new().pick_folder()); });
            self.infer_ckpt_rx = Some(rx);
        }
        if let Some(p) = &self.infer_ckpt_dir {
            ui.label(RichText::new(p.display().to_string()).small().color(Color32::GRAY));
            let ok = p.join("radial.mpk").exists();
            ui.label(RichText::new(if ok { "radial.mpk found" } else { "radial.mpk NOT found" })
                .small()
                .color(if ok { Color32::from_rgb(80, 200, 100) } else { Color32::from_rgb(200, 80, 80) }));
        }
        ui.label(RichText::new("Select a checkpoint_best or checkpoint_epoch_XXXX folder.")
            .small().color(Color32::GRAY));

        ui.add_space(8.0);

        // Source images
        ui.label(RichText::new("Source images (damaged)").strong());
        ui.separator();

        if ui.button("Source folder…").clicked() && self.infer_src_rx.is_none() {
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || { let _ = tx.send(rfd::FileDialog::new().pick_folder()); });
            self.infer_src_rx = Some(rx);
        }
        if let Some(p) = &self.infer_src_folder {
            ui.label(RichText::new(p.display().to_string()).small().color(Color32::GRAY));
            ui.label(RichText::new(format!("{} images found", self.infer_src_count)).small());
        }

        ui.add_space(8.0);

        // Output folder
        ui.label(RichText::new("Output folder").strong());
        ui.separator();

        if ui.button("Output folder…").clicked() && self.infer_out_rx.is_none() {
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || { let _ = tx.send(rfd::FileDialog::new().pick_folder()); });
            self.infer_out_rx = Some(rx);
        }
        if let Some(p) = &self.infer_out_folder {
            ui.label(RichText::new(p.display().to_string()).small().color(Color32::GRAY));
        }

        ui.add_space(8.0);

        // Settings
        ui.label(RichText::new("Settings").strong());
        ui.separator();

        egui::Grid::new("radial_infer_settings_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label("Image size:");
                egui::ComboBox::from_id_salt("radial_infer_img_sz")
                    .selected_text(format!("{}×{}", self.infer_image_size, self.infer_image_size))
                    .show_ui(ui, |ui| {
                        for sz in [256u32, 512] {
                            ui.selectable_value(&mut self.infer_image_size, sz,
                                format!("{}×{}", sz, sz));
                        }
                    });
                ui.end_row();
            });
        ui.label(RichText::new("Image size must match the training checkpoint.")
            .small().color(Color32::GRAY));

        ui.add_space(10.0);

        // Start / Cancel
        let ckpt_ok = self.infer_ckpt_dir.as_ref()
            .map(|p| p.join("radial.mpk").exists())
            .unwrap_or(false);
        let can_start = ckpt_ok
            && self.infer_src_folder.is_some()
            && self.infer_out_folder.is_some()
            && self.infer_src_count > 0
            && !self.inferring;

        ui.add_enabled_ui(can_start, |ui| {
            if ui.add_sized([ui.available_width(), 32.0],
                egui::Button::new(RichText::new("Run Radial Inference").strong())).clicked() {
                self.start_inference();
            }
        });

        if self.inferring {
            if ui.add_sized([ui.available_width(), 26.0], egui::Button::new("Cancel")).clicked() {
                self.infer_cancel.store(true, Ordering::Relaxed);
            }
        }

        if !can_start && !self.inferring {
            let reason = if self.infer_ckpt_dir.is_none()       { "Select a checkpoint folder" }
                         else if !ckpt_ok                        { "radial.mpk not found" }
                         else if self.infer_src_folder.is_none() { "Select a source folder" }
                         else if self.infer_out_folder.is_none() { "Select an output folder" }
                         else if self.infer_src_count == 0       { "No images found" }
                         else                                     { "" };
            if !reason.is_empty() {
                ui.label(RichText::new(reason).small().color(Color32::from_rgb(180, 120, 60)));
            }
        }

        // Progress
        let (done, total) = self.infer_progress;
        if self.inferring || (total > 0) {
            ui.add_space(6.0);
            ui.label(format!("Progress: {done} / {total}"));
            if total > 0 {
                ui.add(egui::ProgressBar::new(done as f32 / total as f32)
                    .desired_width(ui.available_width()));
            }
        }

        ui.add_space(8.0);

        // CSV export
        if !self.infer_results.is_empty() {
            ui.label(RichText::new("Export").strong());
            ui.separator();

            if ui.add_sized([ui.available_width(), 26.0],
                egui::Button::new("Export results as CSV…")).clicked() {
                let (tx, rx) = mpsc::channel::<Option<PathBuf>>();
                std::thread::spawn(move || {
                    let _ = tx.send(
                        rfd::FileDialog::new()
                            .add_filter("CSV", &["csv"])
                            .set_file_name("radial_results.csv")
                            .save_file()
                    );
                });
                if let Ok(Some(path)) = rx.recv() {
                    match write_csv(&self.infer_results, &path) {
                        Ok(())  => toasts.success(format!("CSV saved: {}", path.display())),
                        Err(e)  => toasts.error(format!("CSV export failed: {e}")),
                    }
                }
            }

            ui.label(RichText::new(format!("{} images processed", self.infer_results.len()))
                .small().color(Color32::GRAY));
        }

        ui.add_space(8.0);

        // Log
        if !self.infer_log.is_empty() {
            ui.label(RichText::new("Log").strong());
            ui.separator();
            egui::ScrollArea::vertical()
                .id_salt("radial_infer_log")
                .max_height(150.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for line in self.infer_log.iter().rev().take(30) {
                        ui.label(RichText::new(line).small().monospace());
                    }
                });
        }
    }

    // ── Right panel ───────────────────────────────────────────────────────────

    fn show_right_panel(&mut self, ui: &mut Ui, ctx: &Context) {
        match self.mode {
            Mode::Train => {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.train_right_tab, TrainRightTab::LossCurve,  "  Loss Curve  ");
                    ui.selectable_value(&mut self.train_right_tab, TrainRightTab::SampleGrid, "  Sample Grid  ");
                });
                ui.separator();
                match self.train_right_tab {
                    TrainRightTab::LossCurve  => self.show_loss_curve(ui),
                    TrainRightTab::SampleGrid => self.show_sample_grid(ui),
                }
            }
            Mode::Infer => {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.infer_right_tab, InferRightTab::Results, "  Results  ");
                    ui.selectable_value(&mut self.infer_right_tab, InferRightTab::Preview, "  Preview  ");
                });
                ui.separator();
                match self.infer_right_tab {
                    InferRightTab::Results => self.show_infer_results(ui),
                    InferRightTab::Preview => self.show_infer_preview(ui, ctx),
                }
            }
        }
    }

    // ── Training right panels ─────────────────────────────────────────────────

    fn show_loss_curve(&self, ui: &mut Ui) {
        let half_h = (ui.available_height() / 2.0 - 12.0).max(100.0);

        ui.label(RichText::new("Weighted MSE loss (per batch)").small());
        Plot::new("radial_batch_loss")
            .height(half_h)
            .legend(Legend::default())
            .show(ui, |plt| {
                plt.line(make_line(&self.b_step, &self.b_loss, "MSE",
                    Color32::from_rgb(80, 200, 120)));
            });

        ui.add_space(4.0);
        ui.label(RichText::new("Validation metrics (per epoch)").small());
        Plot::new("radial_epoch_metric")
            .height(half_h)
            .legend(Legend::default())
            .show(ui, |plt| {
                plt.line(make_line(&self.e_step, &self.e_mse,   "Val MSE",
                    Color32::from_rgb(80, 200, 120)));
                plt.line(make_line(&self.e_step, &self.e_b_mae, "Boundary MAE",
                    Color32::from_rgb(220, 140, 50)));
            });
    }

    fn show_sample_grid(&self, ui: &mut Ui) {
        if let Some(tex) = &self.sample_texture {
            ui.label(RichText::new(
                format!("Epoch {} — Damaged | GT | Predicted | Profile overlay (red=damaged, green=GT, blue=pred)",
                    self.sample_epoch)
            ).small().color(Color32::GRAY));
            ui.add_space(2.0);
            let size  = tex.size_vec2();
            let avail = ui.available_size();
            let scale = (avail.x / size.x).min(avail.y / size.y).min(1.0);
            ui.image((tex.id(), size * scale));
        } else {
            ui.centered_and_justified(|ui| {
                ui.label(RichText::new(
                    if self.training { "Waiting for first sample grid…" }
                    else { "Start training to generate sample grids." }
                ).color(Color32::GRAY));
            });
        }
    }

    // ── Inference right panels ────────────────────────────────────────────────

    fn show_infer_results(&mut self, ui: &mut Ui) {
        if self.infer_results.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label(RichText::new(
                    "No results yet. Run Radial Inference to populate this table."
                ).color(Color32::GRAY));
            });
            return;
        }

        ui.horizontal(|ui| {
            ui.add_sized([220.0, 18.0], egui::Label::new(RichText::new("Filename").strong().small()));
            ui.add_sized([ 90.0, 18.0], egui::Label::new(RichText::new("Input px").strong().small()));
            ui.add_sized([ 90.0, 18.0], egui::Label::new(RichText::new("Predicted px").strong().small()));
            ui.add_sized([ 90.0, 18.0], egui::Label::new(RichText::new("Damage px").strong().small()));
            ui.add_sized([ 80.0, 18.0], egui::Label::new(RichText::new("% Damage").strong().small()));
        });
        ui.separator();

        let current_selected = self.infer_selected;
        let mut new_selected = current_selected;

        egui::ScrollArea::vertical()
            .id_salt("radial_results_scroll")
            .show(ui, |ui| {
                for (i, r) in self.infer_results.iter().enumerate() {
                    let is_sel   = current_selected == Some(i);
                    let dmg_col  = damage_color(r.pct_damage);

                    let row = ui.horizontal(|ui| {
                        ui.add_sized([220.0, 18.0],
                            egui::Label::new(RichText::new(&r.filename).small()).truncate());
                        ui.add_sized([ 90.0, 18.0],
                            egui::Label::new(RichText::new(r.input_leaf_px.to_string()).small()));
                        ui.add_sized([ 90.0, 18.0],
                            egui::Label::new(RichText::new(r.predicted_total_px.to_string()).small()));
                        ui.add_sized([ 90.0, 18.0],
                            egui::Label::new(RichText::new(r.damage_px.to_string()).small()));
                        ui.add_sized([ 80.0, 18.0],
                            egui::Label::new(
                                RichText::new(format!("{:.1}%", r.pct_damage)).small().color(dmg_col)));
                    });

                    if is_sel {
                        let rect = row.response.rect;
                        ui.painter().rect_filled(rect, 0.0,
                            Color32::from_rgba_unmultiplied(80, 160, 240, 30));
                    }
                    if row.response.clicked() { new_selected = Some(i); }
                }
            });

        if new_selected != current_selected {
            self.infer_selected = new_selected;
            if let Some(idx) = new_selected {
                if let Some(r) = self.infer_results.get(idx) {
                    let inp = r.path.clone();
                    let rec = r.out_path.clone();
                    self.load_preview(inp, rec);
                    self.infer_right_tab = InferRightTab::Preview;
                }
            }
        }
    }

    fn show_infer_preview(&self, ui: &mut Ui, _ctx: &Context) {
        match (&self.infer_inp_texture, &self.infer_rec_texture) {
            (Some(inp), Some(rec)) => {
                let avail  = ui.available_size();
                let half_w = (avail.x - 16.0) / 2.0;
                let tex_h  = inp.size_vec2().y * (half_w / inp.size_vec2().x);
                let show_h = tex_h.min(avail.y - 40.0);
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

                if let Some(idx) = self.infer_selected {
                    if let Some(r) = self.infer_results.get(idx) {
                        ui.add_space(8.0);
                        egui::Grid::new("radial_preview_stats").num_columns(2).spacing([12.0, 2.0])
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
                        if self.infer_results.is_empty() {
                            "Run inference, then select a result row to preview."
                        } else {
                            "Select a row in the Results tab to preview the image pair."
                        }
                    ).color(Color32::GRAY));
                });
            }
        }
    }

    // ── Preview loading ───────────────────────────────────────────────────────

    fn load_preview(&mut self, inp_path: PathBuf, rec_path: PathBuf) {
        let (tx, rx) = mpsc::channel::<(Vec<u8>, Vec<u8>, u32)>();
        self.infer_preview_rx  = Some(rx);
        self.infer_inp_texture = None;
        self.infer_rec_texture = None;

        std::thread::spawn(move || {
            const PREV: u32 = 512;
            let load = |p: &PathBuf| -> Option<Vec<u8>> {
                let img = image::open(p).ok()?;
                Some(img.resize_exact(PREV, PREV, image::imageops::FilterType::Triangle)
                    .to_rgba8().into_raw())
            };
            if let (Some(i), Some(r)) = (load(&inp_path), load(&rec_path)) {
                let _ = tx.send((i, r, PREV));
            }
        });
    }

    fn poll_infer_preview(&mut self, ctx: &Context) {
        if let Some(rx) = self.infer_preview_rx.take() {
            match rx.try_recv() {
                Ok((inp, rec, sz)) => {
                    let sz = sz as usize;
                    self.infer_inp_texture = Some(ctx.load_texture(
                        "radial_inp", egui::ColorImage::from_rgba_unmultiplied([sz, sz], &inp),
                        egui::TextureOptions::LINEAR,
                    ));
                    self.infer_rec_texture = Some(ctx.load_texture(
                        "radial_rec", egui::ColorImage::from_rgba_unmultiplied([sz, sz], &rec),
                        egui::TextureOptions::LINEAR,
                    ));
                }
                Err(mpsc::TryRecvError::Empty) => {
                    self.infer_preview_rx = Some(rx);
                    ctx.request_repaint();
                }
                Err(mpsc::TryRecvError::Disconnected) => {}
            }
        }
    }

    // ── Message polling ───────────────────────────────────────────────────────

    fn handle_msgs(&mut self, ctx: &Context, toasts: &mut ToastManager) {
        let mut msgs = Vec::new();
        let mut done = false;
        if let Some(rx) = &self.rx {
            for _ in 0..64 {
                match rx.try_recv() {
                    Ok(m)  => msgs.push(m),
                    Err(mpsc::TryRecvError::Empty)        => break,
                    Err(mpsc::TryRecvError::Disconnected) => { done = true; break; }
                }
            }
        }
        for msg in msgs { self.process_train_msg(msg, ctx, toasts); }
        if done { self.rx = None; self.training = false; }
    }

    fn process_train_msg(&mut self, msg: RadialTrainMsg, ctx: &Context, toasts: &mut ToastManager) {
        const MAX_PTS: usize = 10_000;
        match msg {
            RadialTrainMsg::BatchMetrics { step, loss } => {
                push_capped(&mut self.b_step, step as f64, MAX_PTS);
                push_capped(&mut self.b_loss, loss  as f64, MAX_PTS);
            }
            RadialTrainMsg::EpochMetrics { epoch, mse, boundary_mae } => {
                self.current_epoch = epoch;
                self.e_step.push(epoch as f64);
                self.e_mse.push(mse as f64);
                self.e_b_mae.push(boundary_mae as f64);
                self.latest_mse   = Some(mse);
                self.latest_b_mae = Some(boundary_mae);
            }
            RadialTrainMsg::SampleGrid { epoch, pixels, width, height } => {
                self.sample_epoch   = epoch;
                self.sample_texture = Some(ctx.load_texture(
                    "radial_sample_grid",
                    egui::ColorImage::from_rgba_unmultiplied([width, height], &pixels),
                    egui::TextureOptions::LINEAR,
                ));
            }
            RadialTrainMsg::Checkpoint { path } => {
                self.latest_checkpoint = Some(path);
            }
            RadialTrainMsg::Log(msg) => {
                self.log_lines.push(msg);
                if self.log_lines.len() > 500 { self.log_lines.drain(0..100); }
            }
            RadialTrainMsg::Finished => {
                self.training = false;
                let m = self.latest_mse.map(|v| format!("Final MSE: {:.6}", v)).unwrap_or_default();
                toasts.success(format!("Radial training complete. {m}"));
            }
            RadialTrainMsg::Error(e) => {
                self.training = false;
                toasts.error(format!("Radial training error: {e}"));
                self.log_lines.push(format!("ERROR: {e}"));
            }
        }
    }

    fn poll_infer_msgs(&mut self, _ctx: &Context, toasts: &mut ToastManager) {
        let mut msgs = Vec::new();
        let mut done = false;
        if let Some(rx) = &self.infer_rx {
            for _ in 0..64 {
                match rx.try_recv() {
                    Ok(m)  => msgs.push(m),
                    Err(mpsc::TryRecvError::Empty)        => break,
                    Err(mpsc::TryRecvError::Disconnected) => { done = true; break; }
                }
            }
        }
        for msg in msgs { self.process_infer_msg(msg, toasts); }
        if done { self.infer_rx = None; self.inferring = false; }
    }

    fn process_infer_msg(&mut self, msg: RadialInferMsg, toasts: &mut ToastManager) {
        match msg {
            RadialInferMsg::Progress { done, total } => {
                self.infer_progress = (done, total);
            }
            RadialInferMsg::ImageResult(r) => {
                self.infer_results.push(r);
            }
            RadialInferMsg::Log(msg) => {
                self.infer_log.push(msg);
                if self.infer_log.len() > 500 { self.infer_log.drain(0..100); }
            }
            RadialInferMsg::Finished => {
                self.inferring = false;
                toasts.success(format!(
                    "Radial inference complete — {} images processed.", self.infer_results.len()
                ));
                self.infer_log.push("Inference finished.".to_string());
            }
            RadialInferMsg::Error(e) => {
                self.inferring = false;
                toasts.error(format!("Radial inference error: {e}"));
                self.infer_log.push(format!("ERROR: {e}"));
            }
        }
    }

    // ── File dialog polling ───────────────────────────────────────────────────

    fn poll_file_dialogs(&mut self) {
        // Training dialogs
        if let Some(rx) = self.source_rx.take() {
            match rx.try_recv() {
                Ok(Some(p)) => { self.source_count = scan_image_count(&p); self.source_folder = Some(p); }
                Ok(None)    => {}
                Err(mpsc::TryRecvError::Empty) => self.source_rx = Some(rx),
                Err(_)      => {}
            }
        }
        if let Some(rx) = self.output_rx.take() {
            match rx.try_recv() {
                Ok(Some(p)) => { self.output_folder = Some(p); }
                Ok(None)    => {}
                Err(mpsc::TryRecvError::Empty) => self.output_rx = Some(rx),
                Err(_)      => {}
            }
        }
        if let Some(rx) = self.resume_rx.take() {
            match rx.try_recv() {
                Ok(Some(p)) => { self.resume_folder = Some(p); }
                Ok(None)    => {}
                Err(mpsc::TryRecvError::Empty) => self.resume_rx = Some(rx),
                Err(_)      => {}
            }
        }

        // Inference dialogs
        if let Some(rx) = self.infer_ckpt_rx.take() {
            match rx.try_recv() {
                Ok(Some(p)) => { self.infer_ckpt_dir = Some(p); }
                Ok(None)    => {}
                Err(mpsc::TryRecvError::Empty) => self.infer_ckpt_rx = Some(rx),
                Err(_)      => {}
            }
        }
        if let Some(rx) = self.infer_src_rx.take() {
            match rx.try_recv() {
                Ok(Some(p)) => { self.infer_src_count = scan_image_count(&p); self.infer_src_folder = Some(p); }
                Ok(None)    => {}
                Err(mpsc::TryRecvError::Empty) => self.infer_src_rx = Some(rx),
                Err(_)      => {}
            }
        }
        if let Some(rx) = self.infer_out_rx.take() {
            match rx.try_recv() {
                Ok(Some(p)) => { self.infer_out_folder = Some(p); }
                Ok(None)    => {}
                Err(mpsc::TryRecvError::Empty) => self.infer_out_rx = Some(rx),
                Err(_)      => {}
            }
        }
    }

    // ── Start training ────────────────────────────────────────────────────────

    fn start_training(&mut self) {
        let source = match &self.source_folder { Some(p) => p.clone(), None => return };
        let output = match &self.output_folder { Some(p) => p.clone(), None => return };
        if self.source_count < 2 { return; }

        let mut all_paths: Vec<PathBuf> = walkdir::WalkDir::new(&source)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file() && is_image(e.path()))
            .map(|e| e.into_path())
            .collect();

        if all_paths.len() < 2 { return; }

        use rand::{rngs::SmallRng, SeedableRng, seq::SliceRandom};
        let mut rng = SmallRng::seed_from_u64(42);
        all_paths.shuffle(&mut rng);
        let split = ((all_paths.len() as f32 * 0.9).ceil() as usize).min(all_paths.len() - 1);
        let train_paths = all_paths[..split].to_vec();
        let val_paths   = all_paths[split..].to_vec();

        let resume_from = if let Some(ref manual) = self.resume_folder {
            Some(manual.clone())
        } else {
            let best = output.join("checkpoint_best");
            if best.join("radial.mpk").exists() { Some(best) } else { None }
        };

        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel_flag   = cancel.clone();
        let (tx, rx)       = mpsc::channel();
        self.rx            = Some(rx);
        self.training      = true;
        self.total_epochs  = self.epochs;
        self.current_epoch = 0;

        self.b_step.clear(); self.b_loss.clear();
        self.e_step.clear(); self.e_mse.clear(); self.e_b_mae.clear();
        self.latest_mse        = None;
        self.latest_b_mae      = None;
        self.latest_checkpoint = None;

        let cfg = RadialTrainConfig {
            train_paths,
            val_paths,
            output_dir:        output,
            epochs:            self.epochs,
            batch_size:        self.batch_size,
            lr:                self.learning_rate,
            checkpoint_every:  self.checkpoint_every,
            sample_every:      self.sample_grid_every,
            image_size:        self.image_size_px as usize,
            curriculum_epochs: self.curriculum_epochs,
            resume_from,
            accum_steps:       self.accum_steps,
            damage_loss_weight: self.damage_loss_weight,
            damage_params: DamageParams {
                min_pct:          self.damage_min_pct,
                max_pct:          self.damage_max_pct,
                coastal:          self.coastal_enabled,
                coastal_w:        self.coastal_weight,
                spots:            false,  spots_w:    0.0,
                snake:            self.snake_enabled,
                snake_w:          self.snake_weight,
                ellipses:         false,  ellipses_w: 0.0,
                apex:             false,  apex_w:     0.0,
                clusters:         false,  clusters_w: 0.0,
                lobe:             self.lobe_enabled,
                lobe_w:           self.lobe_weight,
                focal_sector:     false,  focal_sector_w: 0.0,
                zero_damage_prob: self.zero_damage_prob,
                curriculum_max:   self.curriculum_max_pct,
            },
        };

        if model::backend_name() == "CPU" {
            self.log_lines.push(
                "WARNING: CPU backend — radial training is lightweight, but GPU preferred.".into()
            );
        }
        self.log_lines.push("Starting radial profile training…".into());
        spawn_radial_training(cfg, tx, cancel);
    }

    // ── Start inference ───────────────────────────────────────────────────────

    fn start_inference(&mut self) {
        let ckpt   = match &self.infer_ckpt_dir   { Some(p) => p.clone(), None => return };
        let source = match &self.infer_src_folder  { Some(p) => p.clone(), None => return };
        let output = match &self.infer_out_folder  { Some(p) => p.clone(), None => return };

        let image_paths: Vec<PathBuf> = walkdir::WalkDir::new(&source)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file() && is_image(e.path()))
            .map(|e| e.into_path())
            .collect();

        if image_paths.is_empty() { return; }

        self.infer_results.clear();
        self.infer_selected    = None;
        self.infer_inp_texture = None;
        self.infer_rec_texture = None;
        self.infer_progress    = (0, image_paths.len());

        let cancel = Arc::new(AtomicBool::new(false));
        self.infer_cancel = cancel.clone();

        let (tx, rx) = mpsc::channel();
        self.infer_rx  = Some(rx);
        self.inferring = true;

        let cfg = RadialInferConfig {
            checkpoint_dir: ckpt,
            image_paths,
            output_dir:  output,
            image_size:  self.infer_image_size as usize,
        };

        self.infer_log.push("Starting radial inference…".to_string());
        spawn_radial_inference(cfg, tx, cancel);
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_line(x: &[f64], y: &[f64], name: &str, color: Color32) -> Line {
    let pts: PlotPoints = x.iter().zip(y.iter()).map(|(&xi, &yi)| [xi, yi]).collect();
    Line::new(pts).name(name).color(color)
}

fn push_capped(v: &mut Vec<f64>, val: f64, cap: usize) {
    v.push(val);
    if v.len() > cap { v.drain(0..cap / 10); }
}

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
    if pct < 20.0      { Color32::from_rgb(80, 200, 100) }
    else if pct < 50.0 { Color32::from_rgb(220, 180, 60) }
    else               { Color32::from_rgb(200, 80, 80) }
}

fn stat_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.label(format!("{label}:"));
    ui.label(value);
    ui.end_row();
}
