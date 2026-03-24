pub mod training;
pub mod model;

use std::{
    path::PathBuf,
    sync::{mpsc, Arc},
    sync::atomic::{AtomicBool, Ordering},
};

use egui::{Color32, RichText, Ui, Context};
use egui_plot::{Legend, Line, Plot, PlotPoints};

use crate::settings::{AppSettings, LeafShape, MarginType};
use crate::widgets::ToastManager;
use crate::tabs::recon_train::training::DamageParams;
use training::{SimpleTrainConfig, SimpleTrainMsg, spawn_simple_training};

use crate::tabs::recon_train::metrics::{MetricsSnapshot};

// ── UI state ──────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum RightTab { LossCurve, SampleGrid }

pub struct ReconSimpleTab {
    // Folders
    source_folder:  Option<PathBuf>,
    output_folder:  Option<PathBuf>,
    resume_folder:  Option<PathBuf>,

    // Training params
    epochs:             usize,
    batch_size:         usize,
    learning_rate:      f64,
    l1_lambda:          f32,
    tv_lambda:          f32,
    conf_lambda:        f32,
    recon_focus_weight: f32,
    tversky_alpha:      f32,
    tversky_beta:       f32,
    checkpoint_every:   usize,
    sample_grid_every:  usize,
    image_size_px:      u32,
    curriculum_epochs:  usize,

    // Damage params
    damage_min_pct:     f32,
    damage_max_pct:     f32,
    curriculum_max_pct: f32,
    zero_damage_prob:   f32,
    coastal_enabled:    bool,   coastal_weight:   f32,
    spots_enabled:      bool,   spots_weight:     f32,
    snake_enabled:      bool,   snake_weight:     f32,
    // ellipses and apex removed — interior/apex damage is not realistic herbivory
    clusters_enabled:      bool,   clusters_weight:      f32,
    lobe_enabled:          bool,   lobe_weight:          f32,
    focal_sector_enabled:  bool,   focal_sector_weight:  f32,
    accum_steps:           usize,

    // GAN / discriminator settings (kept for config compat, area_lambda replaces adv_lambda functionally)
    pretrain_epochs: usize,
    d_lr_factor:     f64,
    adv_lambda:      f32,
    area_lambda:     f32,

    // Runtime state
    training:       bool,
    cancel_flag:    Arc<AtomicBool>,
    rx:             Option<mpsc::Receiver<SimpleTrainMsg>>,
    log_lines:      Vec<String>,
    current_epoch:  usize,
    total_epochs:   usize,

    // Loss history
    b_step:         Vec<f64>,
    b_g_recon:      Vec<f64>,
    b_g_adv:        Vec<f64>,
    b_d_real:       Vec<f64>,
    b_d_fake:       Vec<f64>,
    e_step:         Vec<f64>,
    e_iou:          Vec<f64>,
    e_dice:         Vec<f64>,
    e_prec:         Vec<f64>,
    e_rec:          Vec<f64>,

    // Latest metrics for status panel
    latest_metrics:    Option<MetricsSnapshot>,
    latest_checkpoint: Option<String>,

    // Sample grid texture
    sample_texture:    Option<egui::TextureHandle>,
    sample_epoch:      usize,

    // File dialog receivers
    source_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    output_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    resume_rx: Option<mpsc::Receiver<Option<PathBuf>>>,

    // Descriptor conditioning
    leaf_shape:  LeafShape,
    margin_type: MarginType,

    // UI state
    right_tab:    RightTab,
    source_count: usize,
}

impl ReconSimpleTab {
    pub fn new() -> Self {
        Self {
            source_folder:  None,
            output_folder:  None,
            resume_folder:  None,

            epochs:             200,
            batch_size:         2,
            learning_rate:      2e-4,
            l1_lambda:          10.0,
            tv_lambda:          0.05,
            conf_lambda:        0.5,
            recon_focus_weight: 4.0,
            tversky_alpha:      0.92,  // ↑ penalise over-reconstruction (fills sinuses/rim)
            tversky_beta:       0.08,
            checkpoint_every:   10,
            sample_grid_every:  5,
            image_size_px:      512,
            curriculum_epochs:  20,

            damage_min_pct:     10.0,
            damage_max_pct:     40.0,  // realistic oak margin loss; 85% trained convex over-fill
            curriculum_max_pct: 40.0,
            zero_damage_prob:   0.40,  // ↑ more "intact in → add nothing out" (don't fill sinuses)
            coastal_enabled:    true,   coastal_weight:   0.70,
            spots_enabled:      false,  spots_weight:     0.20,
            snake_enabled:      true,   snake_weight:     0.05,
            clusters_enabled:      false,  clusters_weight:      0.60,
            lobe_enabled:          true,   lobe_weight:          0.50,
            focal_sector_enabled:  true,   focal_sector_weight:  1.0,
            accum_steps:           1,

            pretrain_epochs: 10,
            d_lr_factor:     0.5,
            adv_lambda:      20.0,
            area_lambda:     3.0,   // ↑ area head = robust scalar lost-area for asymmetric oak

            training:       false,
            cancel_flag:    Arc::new(AtomicBool::new(false)),
            rx:             None,
            log_lines:      Vec::new(),
            current_epoch:  0,
            total_epochs:   0,

            b_step:    Vec::new(),
            b_g_recon: Vec::new(),
            b_g_adv:   Vec::new(),
            b_d_real:  Vec::new(),
            b_d_fake:  Vec::new(),
            e_step:    Vec::new(),
            e_iou:     Vec::new(),
            e_dice:    Vec::new(),
            e_prec:    Vec::new(),
            e_rec:     Vec::new(),

            latest_metrics:    None,
            latest_checkpoint: None,

            sample_texture: None,
            sample_epoch:   0,

            source_rx: None,
            output_rx: None,
            resume_rx: None,

            leaf_shape:  LeafShape::default(),
            margin_type: MarginType::default(),

            right_tab:    RightTab::LossCurve,
            source_count: 0,
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    pub fn needs_repaint(&self) -> bool { self.training }
    pub fn is_training(&self)   -> bool { self.training }

    pub fn epoch_progress(&self) -> (usize, usize) {
        (self.current_epoch, self.total_epochs)
    }

    pub fn save_settings(&self, s: &mut AppSettings) {
        let r = &mut s.recon_simple;
        r.last_source_folder  = self.source_folder.clone();
        r.last_output_folder  = self.output_folder.clone();
        r.last_resume_folder  = self.resume_folder.clone();
        r.epochs              = self.epochs;
        r.batch_size          = self.batch_size;
        r.learning_rate       = self.learning_rate;
        r.l1_lambda           = self.l1_lambda;
        r.tv_lambda           = self.tv_lambda;
        r.conf_lambda         = self.conf_lambda;
        r.recon_focus_weight  = self.recon_focus_weight;
        r.tversky_alpha       = self.tversky_alpha;
        r.tversky_beta        = self.tversky_beta;
        r.checkpoint_every    = self.checkpoint_every;
        r.sample_grid_every   = self.sample_grid_every;
        r.image_size_px       = self.image_size_px;
        r.curriculum_epochs   = self.curriculum_epochs;
        r.damage_min_pct      = self.damage_min_pct;
        r.damage_max_pct      = self.damage_max_pct;
        r.curriculum_max_pct  = self.curriculum_max_pct;
        r.zero_damage_prob    = self.zero_damage_prob;
        r.coastal_enabled     = self.coastal_enabled;  r.coastal_weight   = self.coastal_weight;
        r.spots_enabled       = self.spots_enabled;    r.spots_weight     = self.spots_weight;
        r.snake_enabled       = self.snake_enabled;    r.snake_weight     = self.snake_weight;
        r.clusters_enabled    = self.clusters_enabled; r.clusters_weight  = self.clusters_weight;
        r.lobe_enabled        = self.lobe_enabled;     r.lobe_weight      = self.lobe_weight;
        r.focal_sector_enabled = self.focal_sector_enabled;
        r.focal_sector_weight  = self.focal_sector_weight;
        r.accum_steps         = self.accum_steps;
        r.leaf_shape          = self.leaf_shape;
        r.margin_type         = self.margin_type;
        r.pretrain_epochs     = self.pretrain_epochs;
        r.d_lr_factor         = self.d_lr_factor;
        r.adv_lambda          = self.adv_lambda;
    }

    pub fn load_settings(&mut self, s: &AppSettings) {
        let r = &s.recon_simple;
        self.source_folder   = r.last_source_folder.clone();
        self.output_folder   = r.last_output_folder.clone();
        self.resume_folder   = r.last_resume_folder.clone();
        self.epochs          = r.epochs;
        self.batch_size      = r.batch_size;
        self.learning_rate   = r.learning_rate;
        self.l1_lambda       = r.l1_lambda;
        self.tv_lambda       = r.tv_lambda;
        self.conf_lambda     = r.conf_lambda;
        self.recon_focus_weight = r.recon_focus_weight;
        self.tversky_alpha   = r.tversky_alpha;
        self.tversky_beta    = r.tversky_beta;
        self.checkpoint_every  = r.checkpoint_every;
        self.sample_grid_every = r.sample_grid_every;
        self.image_size_px   = r.image_size_px;
        self.curriculum_epochs = r.curriculum_epochs;
        self.damage_min_pct  = r.damage_min_pct;
        self.damage_max_pct  = r.damage_max_pct;
        self.curriculum_max_pct = r.curriculum_max_pct;
        self.zero_damage_prob = r.zero_damage_prob;
        self.coastal_enabled  = r.coastal_enabled;   self.coastal_weight   = r.coastal_weight;
        self.spots_enabled    = r.spots_enabled;     self.spots_weight     = r.spots_weight;
        self.snake_enabled    = r.snake_enabled;     self.snake_weight     = r.snake_weight;
        self.clusters_enabled = r.clusters_enabled;  self.clusters_weight  = r.clusters_weight;
        self.lobe_enabled     = r.lobe_enabled;      self.lobe_weight      = r.lobe_weight;
        self.focal_sector_enabled = r.focal_sector_enabled;
        self.focal_sector_weight  = r.focal_sector_weight;
        self.accum_steps      = r.accum_steps;
        self.leaf_shape       = r.leaf_shape;
        self.margin_type      = r.margin_type;
        self.pretrain_epochs  = r.pretrain_epochs;
        self.d_lr_factor      = r.d_lr_factor;
        self.adv_lambda       = r.adv_lambda;
        if let Some(folder) = &self.source_folder.clone() {
            self.source_count = scan_image_count(folder);
        }
    }

    // ── Main show ─────────────────────────────────────────────────────────────

    pub fn show(&mut self, ui: &mut Ui, ctx: &Context, toasts: &mut ToastManager) {
        self.poll_file_dialogs();
        self.handle_msgs(ctx, toasts);

        egui::SidePanel::left("recon_simple_controls")
            .exact_width(290.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("recon_simple_ctrl_scroll")
                    .show(ui, |ui| {
                        self.show_controls(ui);
                    });
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.show_right_panel(ui);
        });
    }

    // ── Controls ──────────────────────────────────────────────────────────────

    fn show_controls(&mut self, ui: &mut Ui) {
        ui.add_space(4.0);

        // ── Backend banner ────────────────────────────────────────────────────
        {
            let backend = crate::tabs::recon_train::model::backend_name();
            let is_cpu  = backend == "CPU";
            let (bg, fg, label) = if is_cpu {
                (Color32::from_rgb(160, 60, 10), Color32::WHITE, "CPU — training will be very slow")
            } else if backend == "CUDA" {
                (Color32::from_rgb(30, 110, 30),  Color32::WHITE, "GPU ready (CUDA)")
            } else {
                (Color32::from_rgb(30, 80, 140),  Color32::WHITE, "GPU ready (wgpu)")
            };
            egui::Frame::none()
                .fill(bg)
                .inner_margin(egui::Margin::symmetric(8.0, 5.0))
                .rounding(egui::Rounding::same(4.0))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(RichText::new(label).color(fg).strong().small());
                });
            if is_cpu {
                egui::CollapsingHeader::new(RichText::new("How to rebuild for GPU").small().color(Color32::GRAY))
                    .id_salt("simple_backend_help")
                    .show(ui, |ui| {
                        ui.label(RichText::new(
                            "CUDA: cargo build --release\n\
                             wgpu: cargo build --release --no-default-features --features wgpu-gpu"
                        ).small().monospace().color(Color32::GRAY));
                    });
            }
        }
        ui.add_space(4.0);

        // ── Data ──────────────────────────────────────────────────────────────
        ui.label(RichText::new("Data").strong());
        ui.separator();

        ui.horizontal(|ui| {
            if ui.button("Source folder…").clicked() && self.source_rx.is_none() {
                let (tx, rx) = mpsc::channel();
                std::thread::spawn(move || { let _ = tx.send(rfd::FileDialog::new().pick_folder()); });
                self.source_rx = Some(rx);
            }
        });
        if let Some(f) = &self.source_folder {
            ui.label(RichText::new(f.display().to_string()).small().color(Color32::GRAY));
            ui.label(RichText::new(format!("{} images found", self.source_count)).small());
        }

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.button("Output folder…").clicked() && self.output_rx.is_none() {
                let (tx, rx) = mpsc::channel();
                std::thread::spawn(move || { let _ = tx.send(rfd::FileDialog::new().pick_folder()); });
                self.output_rx = Some(rx);
            }
        });
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
            ui.label(RichText::new(format!("Resume: {}", f.file_name()
                .and_then(|n| n.to_str()).unwrap_or("?")))
                .small().color(Color32::from_rgb(100, 180, 120)));
        }

        ui.add_space(8.0);

        // ── Leaf descriptors ──────────────────────────────────────────────────
        ui.label(RichText::new("Leaf Descriptors").strong());
        ui.separator();
        ui.label(RichText::new(
            "These condition the model's FiLM layer. Train one model per leaf type."
        ).small().color(Color32::GRAY));
        ui.add_space(4.0);

        egui::Grid::new("simple_descriptor_grid")
            .num_columns(2)
            .spacing([8.0, 6.0])
            .show(ui, |ui| {
                ui.label("Shape:");
                egui::ComboBox::from_id_salt("simple_leaf_shape")
                    .selected_text(self.leaf_shape.label())
                    .show_ui(ui, |ui| {
                        for &shape in LeafShape::ALL {
                            ui.selectable_value(&mut self.leaf_shape, shape, shape.label());
                        }
                    });
                ui.end_row();

                ui.label("Margin:");
                egui::ComboBox::from_id_salt("simple_margin_type")
                    .selected_text(self.margin_type.label())
                    .show_ui(ui, |ui| {
                        for &margin in MarginType::ALL {
                            ui.selectable_value(&mut self.margin_type, margin, margin.label());
                        }
                    });
                ui.end_row();
            });

        ui.add_space(8.0);

        // ── Damage ────────────────────────────────────────────────────────────
        ui.label(RichText::new("Damage (on-the-fly)").strong());
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Min %");
            ui.add(egui::Slider::new(&mut self.damage_min_pct, 1.0..=30.0).suffix("%"));
        });
        ui.horizontal(|ui| {
            ui.label("Max %");
            ui.add(egui::Slider::new(&mut self.damage_max_pct, 10.0..=95.0).suffix("%"));
        });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.coastal_enabled, "Coastal ");
            if self.coastal_enabled {
                ui.add(egui::Slider::new(&mut self.coastal_weight, 0.0..=1.0).show_value(true).fixed_decimals(2));
            }
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.spots_enabled, "Spots   ");
            if self.spots_enabled {
                ui.add(egui::Slider::new(&mut self.spots_weight, 0.0..=1.0).show_value(true).fixed_decimals(2));
            }
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.snake_enabled, "Snake   ");
            if self.snake_enabled {
                ui.add(egui::Slider::new(&mut self.snake_weight, 0.0..=1.0).show_value(true).fixed_decimals(2));
            }
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.clusters_enabled, "Clusters");
            if self.clusters_enabled {
                ui.add(egui::Slider::new(&mut self.clusters_weight, 0.0..=1.0).show_value(true).fixed_decimals(2));
            }
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.lobe_enabled, "Lobes   ");
            if self.lobe_enabled {
                ui.add(egui::Slider::new(&mut self.lobe_weight, 0.0..=1.0).show_value(true).fixed_decimals(2));
            }
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.focal_sector_enabled, "Focal Sector");
            if self.focal_sector_enabled {
                ui.add(egui::Slider::new(&mut self.focal_sector_weight, 0.0..=2.0).show_value(true).fixed_decimals(2));
            }
        });
        ui.label(RichText::new("Half-plane cut — targeted one-side damage (30-40 %)")
            .small().color(Color32::GRAY));

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("0% damage prob:");
            ui.add(egui::Slider::new(&mut self.zero_damage_prob, 0.0..=0.5).show_value(true).fixed_decimals(2));
        });

        ui.add_space(8.0);

        // ── Loss weights ──────────────────────────────────────────────────────
        ui.label(RichText::new("Loss weights").strong());
        ui.separator();

        egui::Grid::new("simple_loss_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label("Recon (L1) lambda:").on_hover_text(
                    "Multiplier applied to the reconstruction loss (Tversky+BCE).");
                ui.add(egui::DragValue::new(&mut self.l1_lambda).range(0.1..=100.0).speed(0.1));
                ui.end_row();

                ui.label("TV lambda:").on_hover_text(
                    "Total Variation loss. Penalises rapid pixel changes — reduces checkerboard.");
                ui.add(egui::Slider::new(&mut self.tv_lambda, 0.0..=1.0).fixed_decimals(3));
                ui.end_row();

                ui.label("Conf lambda:").on_hover_text(
                    "Confidence loss. Penalises uncertain outputs near 0.5.");
                ui.add(egui::Slider::new(&mut self.conf_lambda, 0.0..=5.0).fixed_decimals(2));
                ui.end_row();

                ui.label("Recon focus wt:").on_hover_text(
                    "Loss multiplier on damaged pixels the model must reconstruct.");
                ui.add(egui::Slider::new(&mut self.recon_focus_weight, 1.0..=20.0).fixed_decimals(1));
                ui.end_row();

                ui.label("Tversky α (FP):").on_hover_text(
                    "FP weight in Tversky loss. α>0.5 penalises over-prediction.");
                ui.add(egui::Slider::new(&mut self.tversky_alpha, 0.1..=0.9).fixed_decimals(2));
                ui.end_row();

                ui.label("Tversky β (FN):").on_hover_text(
                    "FN weight in Tversky loss. β<0.5 tolerates under-prediction.");
                ui.add(egui::Slider::new(&mut self.tversky_beta, 0.1..=0.9).fixed_decimals(2));
                ui.end_row();
            });

        ui.add_space(8.0);

        // ── Training ─────────────────────────────────────────────────────────
        ui.label(RichText::new("Training").strong());
        ui.separator();

        egui::Grid::new("simple_train_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label("Image size:");
                egui::ComboBox::from_id_salt("simple_img_sz")
                    .selected_text(format!("{}×{}", self.image_size_px, self.image_size_px))
                    .show_ui(ui, |ui| {
                        for sz in [256u32, 512, 1024] {
                            ui.selectable_value(&mut self.image_size_px, sz,
                                format!("{}×{}", sz, sz));
                        }
                    });
                ui.end_row();

                ui.label("Epochs:");
                ui.add(egui::DragValue::new(&mut self.epochs).range(1..=10000));
                ui.end_row();

                ui.label("Batch size:");
                ui.add(egui::DragValue::new(&mut self.batch_size).range(1..=8));
                ui.end_row();

                ui.label("Learning rate:");
                ui.add(egui::DragValue::new(&mut self.learning_rate).speed(1e-6).fixed_decimals(6));
                ui.end_row();

                ui.label("Grad accum steps:").on_hover_text(
                    "Accumulate gradients over N batches before stepping. Effective batch = batch_size × N.");
                ui.add(egui::DragValue::new(&mut self.accum_steps).range(1..=8));
                ui.end_row();

                ui.label("Checkpoint every:");
                ui.add(egui::DragValue::new(&mut self.checkpoint_every).range(1..=100).suffix(" epochs"));
                ui.end_row();

                ui.label("Sample grid every:");
                ui.add(egui::DragValue::new(&mut self.sample_grid_every).range(1..=50).suffix(" epochs"));
                ui.end_row();

                ui.label("Curriculum epochs:");
                ui.add(egui::DragValue::new(&mut self.curriculum_epochs).range(0..=500).suffix(" epochs"));
                ui.end_row();

                ui.label("Curriculum max:");
                ui.add(egui::Slider::new(&mut self.curriculum_max_pct, 5.0..=85.0).suffix("% max"));
                ui.end_row();
            });

        ui.label(RichText::new(
            "Curriculum: damage ramps from min% → curriculum max% over N epochs,\n\
             then uses full max% for the rest of training."
        ).small().color(Color32::GRAY));

        ui.add_space(8.0);

        // ── Area head ─────────────────────────────────────────────────────────
        ui.label(RichText::new("Area head").strong());
        ui.separator();
        egui::Grid::new("simple_area_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label("Area lambda:").on_hover_text(
                    "Weight on the area-head MSE loss. Constrains global predicted leaf area to match GT, reducing overprediction bias.");
                ui.add(egui::DragValue::new(&mut self.area_lambda).range(0.0..=10.0).speed(0.05).fixed_decimals(2));
                ui.end_row();
            });

        ui.add_space(10.0);

        // ── Start / Cancel ────────────────────────────────────────────────────
        let can_start = self.source_folder.is_some()
            && self.output_folder.is_some()
            && self.source_count > 1
            && !self.training;

        ui.add_enabled_ui(can_start, |ui| {
            if ui.add_sized(
                [ui.available_width(), 32.0],
                egui::Button::new(RichText::new("Start Training").strong()),
            ).clicked() {
                self.start_training();
            }
        });

        if self.training {
            if ui.add_sized([ui.available_width(), 26.0], egui::Button::new("Cancel")).clicked() {
                self.cancel_flag.store(true, Ordering::Relaxed);
            }
        }

        if let Some(reason) = (!can_start && !self.training).then(|| {
            if self.source_folder.is_none()       { "Select a source folder" }
            else if self.output_folder.is_none()  { "Select an output folder" }
            else if self.source_count <= 1        { "Need at least 2 images" }
            else { "" }
        }) {
            if !reason.is_empty() {
                ui.label(RichText::new(reason).small().color(Color32::from_rgb(180, 120, 60)));
            }
        }

        ui.add_space(8.0);

        // ── Status ────────────────────────────────────────────────────────────
        if self.training || self.current_epoch > 0 {
            ui.label(RichText::new("Status").strong());
            ui.separator();
            ui.label(format!("Epoch: {} / {}", self.current_epoch, self.total_epochs));

            if let Some(m) = &self.latest_metrics {
                egui::Grid::new("simple_metrics_grid")
                    .num_columns(2)
                    .spacing([12.0, 2.0])
                    .show(ui, |ui| {
                        metric_row(ui, "Val IoU",     m.iou);
                        metric_row(ui, "Val F1/Dice", m.dice);
                        metric_row(ui, "Precision",   m.precision);
                        metric_row(ui, "Recall",      m.recall);
                        metric_row(ui, "Pixel Acc",   m.pixel_acc);
                    });
            }

            if let Some(ckpt) = &self.latest_checkpoint {
                ui.label(RichText::new(format!("Last checkpoint:\n{ckpt}"))
                    .small().color(Color32::GRAY));
            }

            ui.add_space(8.0);
        }

        // ── Log ───────────────────────────────────────────────────────────────
        if !self.log_lines.is_empty() {
            ui.label(RichText::new("Log").strong());
            ui.separator();
            egui::ScrollArea::vertical()
                .id_salt("recon_simple_log_scroll")
                .max_height(150.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for entry in self.log_lines.iter().rev().take(30) {
                        ui.label(RichText::new(entry).small().monospace());
                    }
                });
        }
    }

    // ── Right panel ───────────────────────────────────────────────────────────

    fn show_right_panel(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.right_tab, RightTab::LossCurve,  "  Loss Curve  ");
            ui.selectable_value(&mut self.right_tab, RightTab::SampleGrid, "  Sample Grid  ");
        });
        ui.separator();

        match self.right_tab {
            RightTab::LossCurve  => self.show_loss_curve(ui),
            RightTab::SampleGrid => self.show_sample_grid(ui),
        }
    }

    fn show_loss_curve(&self, ui: &mut Ui) {
        let third_h = (ui.available_height() / 3.0 - 10.0).max(100.0);

        ui.label(RichText::new("Reconstruction loss (per batch)").small());
        Plot::new("simple_batch_loss_plot")
            .height(third_h)
            .legend(Legend::default())
            .show(ui, |plt| {
                plt.line(make_line(&self.b_step, &self.b_g_recon, "G_recon",
                    Color32::from_rgb(80, 200, 120)));
            });

        ui.add_space(2.0);

        ui.label(RichText::new("Validation metrics (per epoch)").small());
        Plot::new("simple_epoch_metric_plot")
            .height(third_h)
            .legend(Legend::default())
            .show(ui, |plt| {
                plt.line(make_line(&self.e_step, &self.e_iou,  "IoU",       Color32::from_rgb( 80, 220, 120)));
                plt.line(make_line(&self.e_step, &self.e_dice, "Dice",      Color32::from_rgb( 60, 210, 210)));
                plt.line(make_line(&self.e_step, &self.e_prec, "Precision", Color32::from_rgb( 80, 140, 240)));
                plt.line(make_line(&self.e_step, &self.e_rec,  "Recall",    Color32::from_rgb(230, 140,  50)));
            });
    }

    fn show_sample_grid(&self, ui: &mut Ui) {
        if let Some(tex) = &self.sample_texture {
            ui.label(RichText::new(
                format!("Epoch {} — Damaged | GT Mask | Predicted | Diagnostic", self.sample_epoch)
            ).small().color(Color32::GRAY));
            ui.add_space(2.0);
            let size    = tex.size_vec2();
            let avail   = ui.available_size();
            let scale   = (avail.x / size.x).min(avail.y / size.y).min(1.0);
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

    // ── Message polling ───────────────────────────────────────────────────────

    fn handle_msgs(&mut self, ctx: &Context, toasts: &mut ToastManager) {
        let mut msgs = Vec::new();
        let mut done = false;

        if let Some(rx) = &self.rx {
            for _ in 0..64 {
                match rx.try_recv() {
                    Ok(msg) => msgs.push(msg),
                    Err(mpsc::TryRecvError::Empty)        => break,
                    Err(mpsc::TryRecvError::Disconnected) => { done = true; break; }
                }
            }
        }

        for msg in msgs {
            self.process_msg(msg, ctx, toasts);
        }
        if done { self.rx = None; self.training = false; }
    }

    fn process_msg(&mut self, msg: SimpleTrainMsg, ctx: &Context, toasts: &mut ToastManager) {
        const MAX_PTS: usize = 10_000;

        match msg {
            SimpleTrainMsg::BatchMetrics { step, g_recon, g_adv, d_real, d_fake } => {
                push_capped(&mut self.b_step,    step as f64,   MAX_PTS);
                push_capped(&mut self.b_g_recon, g_recon as f64, MAX_PTS);
                push_capped(&mut self.b_g_adv,   g_adv as f64,  MAX_PTS);
                push_capped(&mut self.b_d_real,  d_real as f64, MAX_PTS);
                push_capped(&mut self.b_d_fake,  d_fake as f64, MAX_PTS);
            }

            SimpleTrainMsg::EpochMetrics { epoch, metrics } => {
                self.current_epoch = epoch;
                self.e_step.push(epoch as f64);
                self.e_iou.push(metrics.iou   as f64);
                self.e_dice.push(metrics.dice  as f64);
                self.e_prec.push(metrics.precision as f64);
                self.e_rec.push(metrics.recall as f64);
                self.latest_metrics = Some(metrics);
            }

            SimpleTrainMsg::SampleGrid { epoch, pixels, width, height } => {
                self.sample_epoch   = epoch;
                self.sample_texture = Some(ctx.load_texture(
                    "simple_sample_grid",
                    egui::ColorImage::from_rgba_unmultiplied([width, height], &pixels),
                    egui::TextureOptions::LINEAR,
                ));
            }

            SimpleTrainMsg::Checkpoint { path } => {
                self.latest_checkpoint = Some(path);
            }

            SimpleTrainMsg::Log(msg) => {
                self.push_log(msg);
            }

            SimpleTrainMsg::Finished => {
                self.training = false;
                let m = self.latest_metrics.as_ref()
                    .map(|m| format!("Final IoU: {:.4}", m.iou))
                    .unwrap_or_default();
                toasts.success(format!("Recon training complete. {m}"));
                self.push_log("Training finished.".to_string());
            }

            SimpleTrainMsg::Error(e) => {
                self.training = false;
                toasts.error(format!("Recon training error: {e}"));
                self.push_log(format!("ERROR: {e}"));
            }
        }
    }

    fn push_log(&mut self, msg: String) {
        self.log_lines.push(msg);
        if self.log_lines.len() > 500 { self.log_lines.drain(0..100); }
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
        if let Some(rx) = self.resume_rx.take() {
            match rx.try_recv() {
                Ok(Some(path)) => { self.resume_folder = Some(path); }
                Ok(None) => {}
                Err(mpsc::TryRecvError::Empty) => self.resume_rx = Some(rx),
                Err(_) => {}
            }
        }
    }

    // ── Start training ────────────────────────────────────────────────────────

    fn start_training(&mut self) {
        let source = match &self.source_folder { Some(p) => p.clone(), None => return };
        let output = match &self.output_folder { Some(p) => p.clone(), None => return };
        if self.source_count < 2 { return; }

        // Collect all image paths
        let mut all_paths: Vec<PathBuf> = walkdir::WalkDir::new(&source)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.into_path())
            .filter(|p| is_image(p))
            .collect();

        if all_paths.len() < 2 { return; }

        // Deterministic shuffle + 90/10 split
        use rand::{rngs::SmallRng, SeedableRng, seq::SliceRandom};
        let mut rng = SmallRng::seed_from_u64(42);
        all_paths.shuffle(&mut rng);
        let split = ((all_paths.len() as f32 * 0.9).ceil() as usize)
            .min(all_paths.len() - 1);
        let train_paths = all_paths[..split].to_vec();
        let val_paths   = all_paths[split..].to_vec();

        // Resume only from an explicitly chosen folder (no auto-resume).
        // Auto-resume is disabled to prevent silently loading stale/incompatible checkpoints.
        let resume_from = self.resume_folder.clone();

        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel_flag = cancel.clone();

        let (tx, rx) = mpsc::channel::<SimpleTrainMsg>();
        self.rx           = Some(rx);
        self.training     = true;
        self.total_epochs = self.epochs;
        self.current_epoch = 0;

        // Clear plot history
        self.b_step.clear(); self.b_g_recon.clear();
        self.b_g_adv.clear(); self.b_d_real.clear(); self.b_d_fake.clear();
        self.e_step.clear(); self.e_iou.clear(); self.e_dice.clear();
        self.e_prec.clear(); self.e_rec.clear();
        self.latest_metrics = None;
        self.latest_checkpoint = None;

        let cfg = SimpleTrainConfig {
            train_paths,
            val_paths,
            output_dir: output,
            epochs:     self.epochs,
            batch_size: self.batch_size,
            lr:         self.learning_rate,
            l1_lambda:  self.l1_lambda,
            tv_lambda:  self.tv_lambda,
            conf_lambda: self.conf_lambda,
            recon_focus_weight: self.recon_focus_weight,
            tversky_alpha:      self.tversky_alpha,
            tversky_beta:       self.tversky_beta,
            checkpoint_every:   self.checkpoint_every,
            sample_every:       self.sample_grid_every,
            image_size:         self.image_size_px as usize,
            curriculum_epochs:  self.curriculum_epochs,
            resume_from,
            damage_params: DamageParams {
                min_pct:          self.damage_min_pct,
                max_pct:          self.damage_max_pct,
                coastal:          self.coastal_enabled,
                coastal_w:        self.coastal_weight,
                spots:            self.spots_enabled,
                spots_w:          self.spots_weight,
                snake:            self.snake_enabled,
                snake_w:          self.snake_weight,
                ellipses:         false,   // interior damage not realistic herbivory
                ellipses_w:       0.0,
                apex:             false,
                apex_w:           0.0,
                clusters:         self.clusters_enabled,
                clusters_w:       self.clusters_weight,
                lobe:             self.lobe_enabled,
                lobe_w:           self.lobe_weight,
                focal_sector:     self.focal_sector_enabled,
                focal_sector_w:   self.focal_sector_weight,
                zero_damage_prob: self.zero_damage_prob,
                curriculum_max:   self.curriculum_max_pct,
            },
            accum_steps:     self.accum_steps,
            leaf_shape:      self.leaf_shape.index(),
            margin_type:     self.margin_type.index(),
            pretrain_epochs: self.pretrain_epochs,
            d_lr_factor:     self.d_lr_factor,
            adv_lambda:      self.adv_lambda,
            area_lambda:     self.area_lambda,
            // Margin-contour emphasis (no symmetry: Quercus is asymmetric).
            boundary_lambda: 3.0,
            boundary_px:     3,
        };

        if crate::tabs::recon_train::model::backend_name() == "CPU" {
            self.push_log("WARNING: CPU backend — rebuild for GPU to train in reasonable time.".to_string());
        }
        self.push_log("Starting reconstruction training…".to_string());
        spawn_simple_training(cfg, tx, cancel);
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

fn metric_row(ui: &mut egui::Ui, label: &str, val: f32) {
    ui.label(format!("{}:", label));
    let color = if val > 0.85 { Color32::from_rgb(80, 200, 100) }
                else if val > 0.7 { Color32::from_rgb(200, 180, 60) }
                else { Color32::from_rgb(200, 80, 60) };
    ui.label(RichText::new(format!("{:.4}", val)).color(color));
    ui.end_row();
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
