pub mod model;
pub mod training;
pub mod inference;

use std::{
    path::PathBuf,
    sync::{mpsc, Arc},
    sync::atomic::{AtomicBool, Ordering},
};

use egui::{Color32, RichText, Ui, Context};

use crate::settings::AppSettings;
use crate::widgets::ToastManager;
use training::{AreaMsg, AreaTrainConfig, DamageParams, spawn_area_training};
use inference::{AreaInferConfig, AreaInferMsg, AreaResult, spawn_inference, write_csv};
use crate::tabs::recon_train::model::backend_name;

// ── State ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum RightTab { Log, Results }

pub struct ReconAreaTab {
    // ── Folder paths ──────────────────────────────────────────────────────
    source_folder:     Option<PathBuf>,
    output_folder:     Option<PathBuf>,
    checkpoint_folder: Option<PathBuf>,

    // ── Persisted settings ────────────────────────────────────────────────
    species_label:     String,
    image_size_px:     u32,
    epochs:            usize,
    batch_size:        usize,
    learning_rate:     f64,
    checkpoint_every:  usize,
    damage_min_pct:    f32,
    damage_max_pct:    f32,

    // ── Background training ───────────────────────────────────────────────
    training:          bool,
    train_rx:          Option<mpsc::Receiver<AreaMsg>>,
    cancel_train:      Arc<AtomicBool>,
    current_epoch:     usize,
    total_epochs:      usize,

    // ── Background inference ──────────────────────────────────────────────
    inferring:         bool,
    infer_rx:          Option<mpsc::Receiver<AreaInferMsg>>,
    cancel_infer:      Arc<AtomicBool>,
    infer_done:        usize,
    infer_total:       usize,

    // ── Results ───────────────────────────────────────────────────────────
    results:           Vec<AreaResult>,
    selected_idx:      Option<usize>,

    // ── Log ───────────────────────────────────────────────────────────────
    log:               Vec<String>,

    // ── File dialog receivers ─────────────────────────────────────────────
    source_rx:     Option<mpsc::Receiver<Option<PathBuf>>>,
    output_rx:     Option<mpsc::Receiver<Option<PathBuf>>>,
    checkpoint_rx: Option<mpsc::Receiver<Option<PathBuf>>>,

    // ── UI state ──────────────────────────────────────────────────────────
    source_count:  usize,
    right_tab:     RightTab,
}

impl ReconAreaTab {
    pub fn new() -> Self {
        Self {
            source_folder:     None,
            output_folder:     None,
            checkpoint_folder: None,

            species_label:     "Quercus".to_string(),
            image_size_px:     512,
            epochs:            100,
            batch_size:        4,
            learning_rate:     2e-4,
            checkpoint_every:  10,
            damage_min_pct:    2.0,
            damage_max_pct:    30.0,

            training:          false,
            train_rx:          None,
            cancel_train:      Arc::new(AtomicBool::new(false)),
            current_epoch:     0,
            total_epochs:      0,

            inferring:         false,
            infer_rx:          None,
            cancel_infer:      Arc::new(AtomicBool::new(false)),
            infer_done:        0,
            infer_total:       0,

            results:           Vec::new(),
            selected_idx:      None,

            log:               Vec::new(),

            source_rx:     None,
            output_rx:     None,
            checkpoint_rx: None,

            source_count:  0,
            right_tab:     RightTab::Log,
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    pub fn needs_repaint(&self) -> bool { self.training || self.inferring }
    pub fn is_training(&self)  -> bool { self.training }
    pub fn is_inferring(&self) -> bool { self.inferring }

    pub fn training_progress(&self) -> (usize, usize) {
        (self.current_epoch, self.total_epochs)
    }

    pub fn save_settings(&self, s: &mut AppSettings) {
        let r = &mut s.recon_area;
        r.last_source_folder     = self.source_folder.clone();
        r.last_output_folder     = self.output_folder.clone();
        r.last_checkpoint_folder = self.checkpoint_folder.clone();
        r.species_label          = self.species_label.clone();
        r.image_size_px          = self.image_size_px;
        r.epochs                 = self.epochs;
        r.batch_size             = self.batch_size;
        r.learning_rate          = self.learning_rate;
        r.checkpoint_every       = self.checkpoint_every;
        r.damage_min_pct         = self.damage_min_pct;
        r.damage_max_pct         = self.damage_max_pct;
    }

    pub fn load_settings(&mut self, s: &AppSettings) {
        let r = &s.recon_area;
        self.source_folder     = r.last_source_folder.clone();
        self.output_folder     = r.last_output_folder.clone();
        self.checkpoint_folder = r.last_checkpoint_folder.clone();
        self.species_label     = r.species_label.clone();
        self.image_size_px     = r.image_size_px;
        self.epochs            = r.epochs;
        self.batch_size        = r.batch_size;
        self.learning_rate     = r.learning_rate;
        self.checkpoint_every  = r.checkpoint_every;
        self.damage_min_pct    = r.damage_min_pct;
        self.damage_max_pct    = r.damage_max_pct;
        if let Some(folder) = &self.source_folder.clone() {
            self.source_count = scan_image_count(folder);
        }
    }

    // ── Main show ─────────────────────────────────────────────────────────────

    pub fn show(&mut self, ui: &mut Ui, ctx: &Context, toasts: &mut ToastManager) {
        self.poll_file_dialogs();
        self.poll_training(ctx, toasts);
        self.poll_inference(ctx, toasts);

        egui::SidePanel::left("area_controls")
            .exact_width(290.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("area_ctrl_scroll")
                    .show(ui, |ui| self.show_controls(ui, toasts));
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.show_right_panel(ui, ctx, toasts);
        });
    }

    // ── Controls panel ────────────────────────────────────────────────────────

    fn show_controls(&mut self, ui: &mut Ui, toasts: &mut ToastManager) {
        ui.add_space(4.0);

        // ── Backend banner ────────────────────────────────────────────────────
        let bname = backend_name();
        let (banner_color, banner_text) = match bname {
            "CUDA" => (
                Color32::from_rgb(40, 140, 70),
                format!("Backend: CUDA (GPU)"),
            ),
            "wgpu" => (
                Color32::from_rgb(50, 100, 180),
                format!("Backend: wgpu (GPU)"),
            ),
            _ => (
                Color32::from_rgb(160, 90, 30),
                format!("Backend: CPU (ndarray) — training will be slow"),
            ),
        };
        egui::Frame::none()
            .fill(banner_color)
            .inner_margin(egui::Margin::symmetric(8.0, 4.0))
            .rounding(4.0)
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new(&banner_text).small().color(Color32::WHITE).strong());
            });

        if bname == "CPU" {
            ui.collapsing("How to rebuild for GPU", |ui| {
                ui.label(RichText::new(
                    "CUDA:  cargo build --release\n\
                     wgpu:  cargo build --release --no-default-features --features wgpu-gpu"
                ).small().monospace().color(Color32::GRAY));
            });
        }

        ui.add_space(6.0);

        // ── Source folder ─────────────────────────────────────────────────────
        ui.label(RichText::new("Source folder").strong());
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

        // ── Checkpoint folder ─────────────────────────────────────────────────
        ui.label(RichText::new("Checkpoint folder").strong());
        ui.separator();

        if ui.button("Checkpoint folder…").clicked() && self.checkpoint_rx.is_none() {
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                let _ = tx.send(rfd::FileDialog::new().pick_folder());
            });
            self.checkpoint_rx = Some(rx);
        }
        if let Some(p) = &self.checkpoint_folder {
            ui.label(RichText::new(p.display().to_string()).small().color(Color32::GRAY));
            let has_model = p.join("area_regressor.mpk").exists();
            if has_model {
                ui.label(RichText::new("area_regressor.mpk found")
                    .small().color(Color32::from_rgb(80, 200, 100)));
            } else {
                ui.label(RichText::new("area_regressor.mpk not found")
                    .small().color(Color32::from_rgb(180, 120, 60)));
            }
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

        // ── Shared settings ───────────────────────────────────────────────────
        ui.label(RichText::new("Settings").strong());
        ui.separator();

        egui::Grid::new("area_shared_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label("Species label:");
                ui.text_edit_singleline(&mut self.species_label);
                ui.end_row();

                ui.label("Image size:");
                egui::ComboBox::from_id_salt("area_img_sz")
                    .selected_text(format!("{}×{}", self.image_size_px, self.image_size_px))
                    .show_ui(ui, |ui| {
                        for sz in [256u32, 512, 1024] {
                            ui.selectable_value(&mut self.image_size_px, sz,
                                format!("{}×{}", sz, sz));
                        }
                    });
                ui.end_row();
            });

        ui.add_space(10.0);

        // ── Training section ──────────────────────────────────────────────────
        ui.label(RichText::new("Training").strong());
        ui.separator();

        egui::Grid::new("area_train_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label("Epochs:");
                ui.add(egui::DragValue::new(&mut self.epochs)
                    .range(1..=1000).speed(1.0));
                ui.end_row();

                ui.label("Batch size:");
                ui.add(egui::DragValue::new(&mut self.batch_size)
                    .range(1..=8).speed(1.0));
                ui.end_row();

                ui.label("Learning rate:");
                ui.add(egui::DragValue::new(&mut self.learning_rate)
                    .speed(1e-6)
                    .fixed_decimals(6));
                ui.end_row();

                ui.label("Checkpoint every:");
                ui.add(egui::DragValue::new(&mut self.checkpoint_every)
                    .range(1..=500).speed(1.0));
                ui.end_row();
            });

        ui.add_space(4.0);
        ui.label(RichText::new("Damage range (%)").small().color(Color32::GRAY));
        egui::Grid::new("area_damage_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label("Min %:");
                ui.add(egui::Slider::new(&mut self.damage_min_pct, 0.0..=50.0)
                    .fixed_decimals(1));
                ui.end_row();

                ui.label("Max %:");
                ui.add(egui::Slider::new(&mut self.damage_max_pct, 1.0..=99.0)
                    .fixed_decimals(1));
                ui.end_row();
            });
        // Clamp min ≤ max
        if self.damage_min_pct > self.damage_max_pct {
            self.damage_min_pct = self.damage_max_pct;
        }

        ui.add_space(6.0);

        let can_train = self.source_folder.is_some()
            && self.output_folder.is_some()
            && self.source_count >= 2
            && !self.training
            && !self.inferring;

        ui.add_enabled_ui(can_train, |ui| {
            if ui.add_sized([ui.available_width(), 30.0],
                egui::Button::new(RichText::new("Start Training").strong())).clicked()
            {
                self.start_training(toasts);
            }
        });

        if self.training {
            ui.add_space(4.0);
            if self.total_epochs > 0 {
                let frac = self.current_epoch as f32 / self.total_epochs as f32;
                ui.add(egui::ProgressBar::new(frac)
                    .text(format!("Epoch {} / {}", self.current_epoch, self.total_epochs))
                    .desired_width(ui.available_width()));
            }
            if ui.add_sized([ui.available_width(), 24.0],
                egui::Button::new("Cancel Training")).clicked()
            {
                self.cancel_train.store(true, Ordering::Relaxed);
            }
        }

        if !can_train && !self.training {
            let reason = if self.source_folder.is_none() {
                Some("Select a source folder")
            } else if self.source_count < 2 {
                Some("Need at least 2 images for train/val split")
            } else if self.output_folder.is_none() {
                Some("Select an output folder")
            } else if self.inferring {
                Some("Inference is running")
            } else {
                None
            };
            if let Some(r) = reason {
                ui.label(RichText::new(r).small().color(Color32::from_rgb(180, 120, 60)));
            }
        }

        ui.add_space(10.0);

        // ── Inference section ─────────────────────────────────────────────────
        ui.label(RichText::new("Inference").strong());
        ui.separator();

        let ckpt_ok = self.checkpoint_folder.as_ref()
            .map(|p| p.join("area_regressor.mpk").exists())
            .unwrap_or(false);
        let can_infer = ckpt_ok
            && self.source_folder.is_some()
            && self.output_folder.is_some()
            && self.source_count > 0
            && !self.inferring
            && !self.training;

        ui.add_enabled_ui(can_infer, |ui| {
            if ui.add_sized([ui.available_width(), 30.0],
                egui::Button::new(RichText::new("Run Inference").strong())).clicked()
            {
                self.start_inference(toasts);
            }
        });

        if self.inferring {
            ui.add_space(4.0);
            if self.infer_total > 0 {
                let frac = self.infer_done as f32 / self.infer_total as f32;
                ui.add(egui::ProgressBar::new(frac)
                    .text(format!("{} / {}", self.infer_done, self.infer_total))
                    .desired_width(ui.available_width()));
            }
            if ui.add_sized([ui.available_width(), 24.0],
                egui::Button::new("Cancel Inference")).clicked()
            {
                self.cancel_infer.store(true, Ordering::Relaxed);
            }
        }

        if !can_infer && !self.inferring {
            let reason = if self.checkpoint_folder.is_none() {
                Some("Select a checkpoint folder")
            } else if !ckpt_ok {
                Some("area_regressor.mpk not found in checkpoint folder")
            } else if self.source_folder.is_none() {
                Some("Select a source folder")
            } else if self.output_folder.is_none() {
                Some("Select an output folder")
            } else if self.source_count == 0 {
                Some("No images found in source folder")
            } else if self.training {
                Some("Training is running")
            } else {
                None
            };
            if let Some(r) = reason {
                ui.label(RichText::new(r).small().color(Color32::from_rgb(180, 120, 60)));
            }
        }

        ui.add_space(8.0);
    }

    // ── Right panel ───────────────────────────────────────────────────────────

    fn show_right_panel(&mut self, ui: &mut Ui, _ctx: &Context, toasts: &mut ToastManager) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.right_tab, RightTab::Log, "  Training Log  ");
            ui.selectable_value(&mut self.right_tab, RightTab::Results, "  Results  ");
        });
        ui.separator();

        match self.right_tab {
            RightTab::Log     => self.show_log(ui),
            RightTab::Results => self.show_results(ui, toasts),
        }
    }

    fn show_log(&self, ui: &mut Ui) {
        if self.log.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label(RichText::new("No log entries yet. Start training to see output here.")
                    .color(Color32::GRAY));
            });
            return;
        }

        egui::ScrollArea::vertical()
            .id_salt("area_log_scroll")
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for entry in &self.log {
                    ui.label(RichText::new(entry).small().monospace());
                }
            });
    }

    fn show_results(&mut self, ui: &mut Ui, toasts: &mut ToastManager) {
        // Export button
        if !self.results.is_empty() {
            ui.horizontal(|ui| {
                if ui.button("Export CSV…").clicked() {
                    let (tx, rx) = mpsc::channel::<Option<PathBuf>>();
                    std::thread::spawn(move || {
                        let _ = tx.send(
                            rfd::FileDialog::new()
                                .add_filter("CSV", &["csv"])
                                .set_file_name("area_results.csv")
                                .save_file()
                        );
                    });
                    if let Ok(Some(path)) = rx.recv() {
                        match write_csv(&self.results, &path) {
                            Ok(()) => toasts.success(format!("CSV saved: {}", path.display())),
                            Err(e) => toasts.error(format!("CSV export failed: {e}")),
                        }
                    }
                }
                ui.label(RichText::new(format!("{} results", self.results.len()))
                    .small().color(Color32::GRAY));
            });
            ui.separator();
        }

        if self.results.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label(RichText::new("No results yet. Run inference to populate this table.")
                    .color(Color32::GRAY));
            });
            return;
        }

        // Column headers
        ui.horizontal(|ui| {
            ui.add_sized([220.0, 18.0], egui::Label::new(
                RichText::new("Filename").strong().small()));
            ui.add_sized([90.0, 18.0], egui::Label::new(
                RichText::new("Surviving px").strong().small()));
            ui.add_sized([100.0, 18.0], egui::Label::new(
                RichText::new("Predicted px").strong().small()));
            ui.add_sized([80.0, 18.0], egui::Label::new(
                RichText::new("Damage px").strong().small()));
            ui.add_sized([80.0, 18.0], egui::Label::new(
                RichText::new("% Damage").strong().small()));
        });
        ui.separator();

        let selected = self.selected_idx;
        let mut new_selected = selected;

        egui::ScrollArea::vertical()
            .id_salt("area_results_scroll")
            .show(ui, |ui| {
                for (i, r) in self.results.iter().enumerate() {
                    let is_selected = selected == Some(i);
                    let dmg_color  = damage_color(r.pct_damage);

                    let row_resp = ui.horizontal(|ui| {
                        ui.add_sized([220.0, 18.0],
                            egui::Label::new(RichText::new(&r.filename).small())
                                .truncate());
                        ui.add_sized([90.0, 18.0],
                            egui::Label::new(RichText::new(r.surviving_px.to_string()).small()));
                        ui.add_sized([100.0, 18.0],
                            egui::Label::new(RichText::new(r.predicted_total_px.to_string()).small()));
                        ui.add_sized([80.0, 18.0],
                            egui::Label::new(RichText::new(r.damage_px.to_string()).small()));
                        ui.add_sized([80.0, 18.0],
                            egui::Label::new(
                                RichText::new(format!("{:.1}%", r.pct_damage))
                                    .small().color(dmg_color)));
                    });

                    if is_selected {
                        let rect = row_resp.response.rect;
                        ui.painter().rect_filled(
                            rect, 0.0,
                            Color32::from_rgba_unmultiplied(80, 160, 240, 30),
                        );
                    }

                    if row_resp.response.clicked() {
                        new_selected = Some(i);
                    }
                }
            });

        if new_selected != selected {
            self.selected_idx = new_selected;
        }
    }

    // ── Training polling ──────────────────────────────────────────────────────

    fn poll_training(&mut self, _ctx: &Context, toasts: &mut ToastManager) {
        let mut msgs = Vec::new();
        let mut done = false;

        if let Some(rx) = &self.train_rx {
            for _ in 0..64 {
                match rx.try_recv() {
                    Ok(msg) => msgs.push(msg),
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => { done = true; break; }
                }
            }
        }

        for msg in msgs {
            self.handle_train_msg(msg, toasts);
        }
        if done { self.train_rx = None; self.training = false; }
    }

    fn handle_train_msg(&mut self, msg: AreaMsg, toasts: &mut ToastManager) {
        match msg {
            AreaMsg::BatchLoss { step, loss } => {
                if step % 50 == 0 {
                    self.push_log(format!("  step {step}: loss={loss:.6}"));
                }
            }
            AreaMsg::EpochMetrics { epoch, val_mae, val_rmse } => {
                self.current_epoch = epoch;
                self.push_log(format!(
                    "Epoch {}/{}: val_MAE={val_mae:.4}  val_RMSE={val_rmse:.4}",
                    epoch, self.total_epochs,
                ));
            }
            AreaMsg::Checkpoint { path } => {
                self.push_log(format!("Checkpoint saved: {path}"));
            }
            AreaMsg::Log(msg) => {
                self.push_log(msg);
            }
            AreaMsg::Finished => {
                self.training = false;
                toasts.success(format!(
                    "Training complete — {} epochs.", self.current_epoch
                ));
                self.push_log("Training finished.".to_string());
            }
            AreaMsg::Error(e) => {
                self.training = false;
                toasts.error(format!("Training error: {e}"));
                self.push_log(format!("ERROR: {e}"));
            }
        }
    }

    // ── Inference polling ─────────────────────────────────────────────────────

    fn poll_inference(&mut self, _ctx: &Context, toasts: &mut ToastManager) {
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
            self.handle_infer_msg(msg, toasts);
        }
        if done { self.infer_rx = None; self.inferring = false; }
    }

    fn handle_infer_msg(&mut self, msg: AreaInferMsg, toasts: &mut ToastManager) {
        match msg {
            AreaInferMsg::Progress { done, total } => {
                self.infer_done  = done;
                self.infer_total = total;
            }
            AreaInferMsg::Result(result) => {
                self.results.push(result);
                // Switch to results tab when first result arrives
                if self.results.len() == 1 {
                    self.right_tab = RightTab::Results;
                }
            }
            AreaInferMsg::Log(msg) => {
                self.push_log(msg);
            }
            AreaInferMsg::Finished => {
                self.inferring = false;
                toasts.success(format!(
                    "Inference complete — {} images.", self.results.len()
                ));
                self.push_log("Inference finished.".to_string());
            }
            AreaInferMsg::Error(e) => {
                self.inferring = false;
                toasts.error(format!("Inference error: {e}"));
                self.push_log(format!("ERROR: {e}"));
            }
        }
    }

    // ── File dialog polling ───────────────────────────────────────────────────

    fn poll_file_dialogs(&mut self) {
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
        if let Some(rx) = self.checkpoint_rx.take() {
            match rx.try_recv() {
                Ok(Some(path)) => { self.checkpoint_folder = Some(path); }
                Ok(None) => {}
                Err(mpsc::TryRecvError::Empty) => self.checkpoint_rx = Some(rx),
                Err(_) => {}
            }
        }
    }

    // ── Start training ────────────────────────────────────────────────────────

    fn start_training(&mut self, _toasts: &mut ToastManager) {
        let source = match &self.source_folder { Some(p) => p.clone(), None => return };
        let output = match &self.output_folder { Some(p) => p.clone(), None => return };

        // Collect all image paths then 90/10 split
        let mut all_paths: Vec<PathBuf> = walkdir::WalkDir::new(&source)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file() && is_image(e.path()))
            .map(|e| e.into_path())
            .collect();
        all_paths.sort();
        if all_paths.len() < 2 { return; }

        let split = ((all_paths.len() as f32 * 0.9) as usize).max(1);
        let val_paths   = all_paths.split_off(split);
        let train_paths = all_paths;

        let damage_params = DamageParams {
            min_pct:              self.damage_min_pct,
            max_pct:              self.damage_max_pct,
            coastal:              true,
            coastal_w:            0.7,
            spots:                true,
            spots_w:              0.3,
            snake:                false,
            snake_w:              0.0,
            ellipses:             true,
            ellipses_w:           0.5,
            apex:                 true,
            apex_w:               0.4,
            clusters:             true,
            clusters_w:           0.6,
            lobe:                 true,
            lobe_w:               0.5,
            zero_damage_prob:     0.1,
            curriculum_max: self.damage_min_pct,
        };

        let cfg = AreaTrainConfig {
            train_paths,
            val_paths,
            output_dir:       output,
            species_label:    self.species_label.clone(),
            epochs:           self.epochs,
            batch_size:       self.batch_size,
            lr:               self.learning_rate,
            checkpoint_every: self.checkpoint_every,
            image_size:       self.image_size_px as usize,
            damage_params,
            resume_from:      None,
        };

        self.results.clear();
        self.selected_idx = None;
        self.current_epoch = 0;
        self.total_epochs  = self.epochs;
        self.right_tab     = RightTab::Log;

        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel_train = cancel.clone();
        let (tx, rx) = mpsc::channel::<AreaMsg>();
        self.train_rx = Some(rx);
        self.training = true;

        spawn_area_training(cfg, tx, cancel);
        self.push_log("Training started.".to_string());
    }

    // ── Start inference ───────────────────────────────────────────────────────

    fn start_inference(&mut self, _toasts: &mut ToastManager) {
        let checkpoint = match &self.checkpoint_folder { Some(p) => p.clone(), None => return };
        let source     = match &self.source_folder     { Some(p) => p.clone(), None => return };
        let output     = match &self.output_folder     { Some(p) => p.clone(), None => return };

        let output_csv = output.join("area_results.csv");

        let cfg = AreaInferConfig {
            checkpoint_path: checkpoint,
            source_folder:   source,
            output_csv,
            image_size:      self.image_size_px as usize,
            species_label:   self.species_label.clone(),
        };

        self.results.clear();
        self.selected_idx = None;
        self.infer_done   = 0;
        self.infer_total  = self.source_count;

        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel_infer = cancel.clone();
        let (tx, rx) = mpsc::channel::<AreaInferMsg>();
        self.infer_rx = Some(rx);
        self.inferring = true;

        spawn_inference(cfg, tx, cancel);
        self.push_log("Inference started.".to_string());
    }

    // ── Utilities ─────────────────────────────────────────────────────────────

    fn push_log(&mut self, msg: String) {
        self.log.push(msg);
        if self.log.len() > 500 { self.log.drain(0..100); }
    }
}

// ── Module-level helpers ──────────────────────────────────────────────────────

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
