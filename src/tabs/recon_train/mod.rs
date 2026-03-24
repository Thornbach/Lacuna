pub mod model;
pub mod metrics;
pub mod visualization;
pub mod training;

use std::{
    path::PathBuf,
    sync::{mpsc, Arc},
    sync::atomic::{AtomicBool, Ordering},
};

use egui::{Color32, RichText, Ui, Context};
use egui_plot::{Legend, Line, Plot, PlotPoints};

use crate::settings::AppSettings;
use crate::widgets::ToastManager;
use metrics::MetricsSnapshot;
use training::{TrainConfig, TrainMsg, DamageParams, spawn_training};

// ── State ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum RightTab { LossCurves, SampleGrid }

pub struct ReconTrainTab {
    // ── Persisted settings ─────────────────────────────────────────────────
    source_folder:     Option<PathBuf>,
    output_folder:     Option<PathBuf>,
    species_label:     String,
    damage_min_pct:    f32,
    damage_max_pct:    f32,
    coastal_enabled:   bool,
    coastal_weight:    f32,
    spots_enabled:     bool,
    spots_weight:      f32,
    snake_enabled:     bool,
    snake_weight:      f32,
    ellipses_enabled:  bool,
    ellipses_weight:   f32,
    apex_enabled:      bool,
    apex_weight:       f32,
    clusters_enabled:  bool,
    clusters_weight:   f32,
    lobe_enabled:      bool,
    lobe_weight:       f32,
    focal_sector_enabled: bool,
    focal_sector_weight:  f32,
    zero_damage_prob:          f32,
    batch_checkpoint_every:    usize,
    curriculum_epochs:         usize,
    curriculum_max_pct:  f32,
    pretrain_epochs:           usize,
    d_lr_factor:               f64,
    recon_focus_weight:        f32,
    adv_lambda:                f32,
    area_lambda:               f32,
    tv_lambda:                 f32,
    conf_lambda:               f32,
    adaptive_ctrl:             bool,
    ms_lambda:                f32,
    sym_lambda:               f32,
    tversky_alpha:            f32,
    tversky_beta:             f32,
    boundary_exclusion_px:    usize,
    accum_steps:              usize,
    epochs:                    usize,
    batch_size:        usize,
    learning_rate:     f64,
    l1_lambda:         f32,
    checkpoint_every:  usize,
    sample_grid_every: usize,
    image_size_px:     u32,

    // ── Background training ────────────────────────────────────────────────
    train_rx:    Option<mpsc::Receiver<TrainMsg>>,
    cancel_flag: Arc<AtomicBool>,
    training:    bool,

    // ── Epoch tracking ─────────────────────────────────────────────────────
    current_epoch: usize,
    total_epochs:  usize,

    // ── Plot history (batch-level) ─────────────────────────────────────────
    b_step:     Vec<f64>,
    b_g_adv:    Vec<f64>,
    b_g_recon:     Vec<f64>,
    b_d_loss:   Vec<f64>,
    b_d_real:   Vec<f64>,
    b_d_fake:   Vec<f64>,

    // ── Plot history (epoch-level) ─────────────────────────────────────────
    e_step:     Vec<f64>,
    e_iou:      Vec<f64>,
    e_dice:     Vec<f64>,
    e_prec:     Vec<f64>,
    e_rec:      Vec<f64>,

    // ── Current metrics display ────────────────────────────────────────────
    latest_metrics:    Option<MetricsSnapshot>,
    latest_d_real:     f32,
    latest_d_fake:     f32,
    latest_checkpoint: Option<String>,

    // ── Sample grid texture ────────────────────────────────────────────────
    sample_texture: Option<egui::TextureHandle>,
    sample_epoch:   usize,
    /// Per-row (iou, f1) stats from the last sample grid
    sample_stats:   Vec<(f32, f32)>,

    // ── File dialog receivers ──────────────────────────────────────────────
    source_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    output_rx: Option<mpsc::Receiver<Option<PathBuf>>>,

    // ── Log ───────────────────────────────────────────────────────────────
    log: Vec<String>,

    // ── UI state ──────────────────────────────────────────────────────────
    right_tab:       RightTab,
    source_count:    usize,
    /// Timestamp of the last BatchMetrics received — used to detect GPU stalls
    last_batch_tick: Option<std::time::Instant>,
    /// Wall-clock start of the current training run (for log timestamps)
    train_start:     Option<std::time::Instant>,
    /// Path of the training log file — set when training starts
    log_file_path:   Option<std::path::PathBuf>,
}

impl ReconTrainTab {
    pub fn new() -> Self {
        Self {
            source_folder:     None,
            output_folder:     None,
            species_label:     "Quercus".to_string(),
            damage_min_pct:    2.0,
            damage_max_pct:    40.0,
            coastal_enabled:   true,
            coastal_weight:    0.7,
            spots_enabled:     false,
            spots_weight:      0.3,
            snake_enabled:     true,
            snake_weight:      0.05,
            ellipses_enabled:  false,
            ellipses_weight:   0.5,
            apex_enabled:      false,
            apex_weight:       0.4,
            clusters_enabled:  false,
            clusters_weight:   0.6,
            lobe_enabled:      true,
            lobe_weight:       0.5,
            focal_sector_enabled: true,
            focal_sector_weight:  1.0,
            zero_damage_prob:          0.30,
            batch_checkpoint_every:    200,
            curriculum_epochs:         60,
            curriculum_max_pct:        40.0,
            pretrain_epochs:           0,
            d_lr_factor:               0.5,
            recon_focus_weight:        3.0,
            adv_lambda:                20.0,
            area_lambda:               1.0,
            tv_lambda:                 0.02,
            conf_lambda:               0.5,
            adaptive_ctrl:             true,
            ms_lambda:                0.3,
            sym_lambda:               0.0,
            tversky_alpha:            0.8,
            tversky_beta:             0.2,
            boundary_exclusion_px:    0,
            accum_steps:              1,
            epochs:                    100,
            batch_size:        2,
            learning_rate:     2e-4,
            l1_lambda:         0.5,
            checkpoint_every:  10,
            sample_grid_every: 5,
            image_size_px:     512,

            train_rx:    None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            training:    false,

            current_epoch: 0,
            total_epochs:  0,

            b_step:   Vec::new(),
            b_g_adv:  Vec::new(),
            b_g_recon:   Vec::new(),
            b_d_loss: Vec::new(),
            b_d_real: Vec::new(),
            b_d_fake: Vec::new(),

            e_step: Vec::new(),
            e_iou:  Vec::new(),
            e_dice: Vec::new(),
            e_prec: Vec::new(),
            e_rec:  Vec::new(),

            latest_metrics:    None,
            latest_d_real:     0.0,
            latest_d_fake:     0.0,
            latest_checkpoint: None,

            sample_texture: None,
            sample_epoch:   0,
            sample_stats:   Vec::new(),

            source_rx: None,
            output_rx: None,
            log:        Vec::new(),
            right_tab:  RightTab::LossCurves,
            source_count:    0,
            last_batch_tick: None,
            train_start:     None,
            log_file_path:   None,
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    pub fn needs_repaint(&self) -> bool { self.training }

    pub fn is_training(&self) -> bool   { self.training }

    pub fn epoch_progress(&self) -> (usize, usize) {
        (self.current_epoch, self.total_epochs)
    }

    pub fn save_settings(&self, s: &mut AppSettings) {
        let r = &mut s.recon_train;
        r.last_source_folder   = self.source_folder.clone();
        r.last_output_folder   = self.output_folder.clone();
        r.species_label        = self.species_label.clone();
        r.damage_min_pct       = self.damage_min_pct;
        r.damage_max_pct       = self.damage_max_pct;
        r.coastal_enabled      = self.coastal_enabled;
        r.coastal_weight       = self.coastal_weight;
        r.spots_enabled        = self.spots_enabled;
        r.spots_weight         = self.spots_weight;
        r.snake_enabled        = self.snake_enabled;
        r.snake_weight         = self.snake_weight;
        r.ellipses_enabled     = self.ellipses_enabled;
        r.ellipses_weight      = self.ellipses_weight;
        r.apex_enabled         = self.apex_enabled;
        r.apex_weight          = self.apex_weight;
        r.clusters_enabled     = self.clusters_enabled;
        r.clusters_weight      = self.clusters_weight;
        r.lobe_enabled         = self.lobe_enabled;
        r.lobe_weight          = self.lobe_weight;
        r.focal_sector_enabled = self.focal_sector_enabled;
        r.focal_sector_weight  = self.focal_sector_weight;
        r.zero_damage_prob             = self.zero_damage_prob;
        r.batch_checkpoint_every       = self.batch_checkpoint_every;
        r.curriculum_epochs            = self.curriculum_epochs;
        r.curriculum_max_pct     = self.curriculum_max_pct;
        r.pretrain_epochs              = self.pretrain_epochs;
        r.d_lr_factor                  = self.d_lr_factor;
        r.recon_focus_weight           = self.recon_focus_weight;
        r.adv_lambda                   = self.adv_lambda;
        r.tv_lambda                    = self.tv_lambda;
        r.conf_lambda                  = self.conf_lambda;
        r.adaptive_ctrl                = self.adaptive_ctrl;
        r.ms_lambda                    = self.ms_lambda;
        r.sym_lambda                   = self.sym_lambda;
        r.tversky_alpha                = self.tversky_alpha;
        r.tversky_beta                 = self.tversky_beta;
        r.boundary_exclusion_px        = self.boundary_exclusion_px;
        r.accum_steps                  = self.accum_steps;
        r.epochs                       = self.epochs;
        r.batch_size           = self.batch_size;
        r.learning_rate        = self.learning_rate;
        r.l1_lambda            = self.l1_lambda;
        r.checkpoint_every     = self.checkpoint_every;
        r.sample_grid_every    = self.sample_grid_every;
        r.image_size_px        = self.image_size_px;
    }

    pub fn load_settings(&mut self, s: &AppSettings) {
        let r = &s.recon_train;
        self.source_folder     = r.last_source_folder.clone();
        self.output_folder     = r.last_output_folder.clone();
        self.species_label     = r.species_label.clone();
        self.damage_min_pct    = r.damage_min_pct;
        self.damage_max_pct    = r.damage_max_pct;
        self.coastal_enabled   = r.coastal_enabled;
        self.coastal_weight    = r.coastal_weight;
        self.spots_enabled     = r.spots_enabled;
        self.spots_weight      = r.spots_weight;
        self.snake_enabled     = r.snake_enabled;
        self.snake_weight      = r.snake_weight;
        self.ellipses_enabled  = r.ellipses_enabled;
        self.ellipses_weight   = r.ellipses_weight;
        self.apex_enabled      = r.apex_enabled;
        self.apex_weight       = r.apex_weight;
        self.clusters_enabled  = r.clusters_enabled;
        self.clusters_weight   = r.clusters_weight;
        self.lobe_enabled      = r.lobe_enabled;
        self.lobe_weight       = r.lobe_weight;
        self.focal_sector_enabled = r.focal_sector_enabled;
        self.focal_sector_weight  = r.focal_sector_weight;
        self.zero_damage_prob              = r.zero_damage_prob;
        self.batch_checkpoint_every        = r.batch_checkpoint_every;
        self.curriculum_epochs             = r.curriculum_epochs;
        self.curriculum_max_pct      = r.curriculum_max_pct;
        self.pretrain_epochs               = r.pretrain_epochs;
        self.d_lr_factor                   = r.d_lr_factor;
        self.recon_focus_weight            = r.recon_focus_weight;
        self.adv_lambda                    = r.adv_lambda;
        self.tv_lambda                     = r.tv_lambda;
        self.conf_lambda                   = r.conf_lambda;
        self.adaptive_ctrl                 = r.adaptive_ctrl;
        self.ms_lambda                     = r.ms_lambda;
        self.sym_lambda                    = r.sym_lambda;
        self.tversky_alpha                 = r.tversky_alpha;
        self.tversky_beta                  = r.tversky_beta;
        self.boundary_exclusion_px         = r.boundary_exclusion_px;
        self.accum_steps                   = r.accum_steps;
        self.epochs                        = r.epochs;
        self.batch_size        = r.batch_size;
        self.learning_rate     = r.learning_rate;
        self.l1_lambda         = r.l1_lambda;
        self.checkpoint_every  = r.checkpoint_every;
        self.sample_grid_every = r.sample_grid_every;
        self.image_size_px     = r.image_size_px;
        // Refresh source count if folder is known
        if let Some(folder) = &self.source_folder.clone() {
            self.source_count = scan_image_count(folder);
        }
    }

    // ── Main show ─────────────────────────────────────────────────────────────

    pub fn show(&mut self, ui: &mut Ui, ctx: &Context, toasts: &mut ToastManager) {
        self.poll_file_dialogs();
        self.poll_training(ctx, toasts);

        egui::SidePanel::left("recon_controls")
            .exact_width(290.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("recon_ctrl_scroll")
                    .show(ui, |ui| {
                        self.show_controls(ui);
                    });
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.show_right_panel(ui, ctx);
        });
    }

    // ── Controls panel ────────────────────────────────────────────────────────

    fn show_controls(&mut self, ui: &mut Ui) {
        ui.add_space(4.0);

        // ── Backend banner ────────────────────────────────────────────────────
        {
            let backend = crate::tabs::recon_train::model::backend_name();
            let is_cpu = backend == "CPU";
            let (bg, fg, label) = if is_cpu {
                (
                    egui::Color32::from_rgb(160, 60, 10),
                    egui::Color32::WHITE,
                    "⚠ CPU — training will be extremely slow",
                )
            } else if backend == "CUDA" {
                (
                    egui::Color32::from_rgb(30, 110, 30),
                    egui::Color32::WHITE,
                    "GPU ready (CUDA)",
                )
            } else {
                (
                    egui::Color32::from_rgb(30, 80, 140),
                    egui::Color32::WHITE,
                    "GPU ready (wgpu)",
                )
            };

            egui::Frame::none()
                .fill(bg)
                .inner_margin(egui::Margin::symmetric(8.0, 5.0))
                .rounding(egui::Rounding::same(4.0))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(egui::RichText::new(label).color(fg).strong().small());
                });

            if is_cpu {
                egui::CollapsingHeader::new(
                    egui::RichText::new("How to rebuild for GPU").small().color(Color32::GRAY)
                )
                .id_salt("backend_help_hdr")
                .show(ui, |ui| {
                    ui.label(egui::RichText::new(
                        "CUDA (Windows / NVIDIA — default):\n\
                         cargo build --release\n\n\
                         wgpu (Vulkan/DX12/Metal, cross-platform):\n\
                         cargo build --release --no-default-features --features wgpu-gpu\n\n\
                         CPU (current — for testing only):\n\
                         cargo build --release --no-default-features"
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
                std::thread::spawn(move || {
                    let _ = tx.send(rfd::FileDialog::new().pick_folder());
                });
                self.source_rx = Some(rx);
            }
        });
        if let Some(f) = &self.source_folder {
            ui.label(RichText::new(f.display().to_string())
                .small().color(Color32::GRAY));
            ui.label(RichText::new(format!("{} images found", self.source_count)).small());
        }

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.button("Output folder…").clicked() && self.output_rx.is_none() {
                let (tx, rx) = mpsc::channel();
                std::thread::spawn(move || {
                    let _ = tx.send(rfd::FileDialog::new().pick_folder());
                });
                self.output_rx = Some(rx);
            }
        });
        if let Some(f) = &self.output_folder {
            ui.label(RichText::new(f.display().to_string())
                .small().color(Color32::GRAY));
        }

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("Species:");
            ui.text_edit_singleline(&mut self.species_label);
        });
        ui.label(RichText::new("(label passed as cGAN condition channel)")
            .small().color(Color32::GRAY));

        ui.add_space(8.0);

        // ── Damage ────────────────────────────────────────────────────────────
        ui.label(RichText::new("Damage (dynamic, on-the-fly)").strong());
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
                ui.add(egui::Slider::new(&mut self.coastal_weight, 0.0..=1.0)
                    .show_value(true).fixed_decimals(2));
            }
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.spots_enabled, "Spots   ");
            if self.spots_enabled {
                ui.add(egui::Slider::new(&mut self.spots_weight, 0.0..=1.0)
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
            ui.checkbox(&mut self.ellipses_enabled, "Ellipses");
            if self.ellipses_enabled {
                ui.add(egui::Slider::new(&mut self.ellipses_weight, 0.0..=1.0)
                    .show_value(true).fixed_decimals(2));
            }
        });
        ui.label(RichText::new("Large interior holes anywhere on leaf")
            .small().color(Color32::GRAY));
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.apex_enabled, "Apex    ");
            if self.apex_enabled {
                ui.add(egui::Slider::new(&mut self.apex_weight, 0.0..=1.0)
                    .show_value(true).fixed_decimals(2)
                    .suffix(" prob"));
            }
        });
        ui.label(RichText::new("Probability of tip/side strip removal per sample")
            .small().color(Color32::GRAY));
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.clusters_enabled, "Clusters");
            if self.clusters_enabled {
                ui.add(egui::Slider::new(&mut self.clusters_weight, 0.0..=1.0)
                    .show_value(true).fixed_decimals(2));
            }
        });
        ui.label(RichText::new("1–3 focused bite clusters along leaf margin")
            .small().color(Color32::GRAY));
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.lobe_enabled, "Lobes");
            if self.lobe_enabled {
                ui.add(egui::Slider::new(&mut self.lobe_weight, 0.0..=1.0)
                    .show_value(true).fixed_decimals(2));
            }
        });
        ui.label(RichText::new("1–3 whole-lobe disc bites from margin inward")
            .small().color(Color32::GRAY));
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.focal_sector_enabled, "Focal Sector");
            if self.focal_sector_enabled {
                ui.add(egui::Slider::new(&mut self.focal_sector_weight, 0.0..=2.0)
                    .show_value(true).fixed_decimals(2));
            }
        });
        ui.label(RichText::new("Half-plane cut: removes concentrated 30-40 % one-side damage")
            .small().color(Color32::GRAY));

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("0% damage prob:");
            ui.add(egui::Slider::new(&mut self.zero_damage_prob, 0.0..=0.5)
                .show_value(true).fixed_decimals(2));
        });
        ui.label(RichText::new("Chance each sample is undamaged (identity training)")
            .small().color(Color32::GRAY));

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("Boundary exclusion px:");
            ui.add(egui::DragValue::new(&mut self.boundary_exclusion_px).range(0..=50));
        });
        ui.label(RichText::new("Zeros out N pixels inward from the damage edge, forcing the model to use long-range texture cues rather than just interpolating the boundary. 0 = disabled.")
            .small().color(Color32::GRAY));

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("Recon focus weight:");
            ui.add(egui::Slider::new(&mut self.recon_focus_weight, 1.0..=20.0)
                .show_value(true).fixed_decimals(1));
        });
        ui.label(RichText::new("Loss multiplier on reconstructed pixels (GT=leaf, input=missing). Higher = model penalised more for missing damage.")
            .small().color(Color32::GRAY));

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("Adv lambda:");
            ui.add(egui::Slider::new(&mut self.adv_lambda, 1.0..=50.0)
                .show_value(true).fixed_decimals(1));
        });
        ui.label(RichText::new("Multiplier on adversarial loss in G step. Higher = generator cares more about fooling D relative to reconstruction.")
            .small().color(Color32::GRAY));

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("TV lambda:");
            ui.add(egui::Slider::new(&mut self.tv_lambda, 0.0..=1.0)
                .show_value(true).fixed_decimals(3));
        });
        ui.label(RichText::new("Total Variation loss weight. Penalises rapid pixel-to-pixel changes — suppresses checkerboard artifacts. Start at 0.02.")
            .small().color(Color32::GRAY));

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("Conf lambda:");
            ui.add(egui::Slider::new(&mut self.conf_lambda, 0.0..=5.0)
                .show_value(true).fixed_decimals(2));
        });
        ui.label(RichText::new("Confidence loss weight. Penalises predictions near 0.5 — pushes G to commit to crisp 0/1 outputs. Start at 0.5.")
            .small().color(Color32::GRAY));

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.adaptive_ctrl, "Adaptive GAN controller");
        });
        ui.label(RichText::new("Monitors D(x)/D(G(x)) rolling means each epoch and automatically nudges adv_lambda, conf_lambda, d_lr, and G steps within safe bounds to correct lockstep, D-dominance, or G-dominance. Actions appear in the log.")
            .small().color(Color32::GRAY));

        ui.add_space(8.0);

        // ── Multi-scale & symmetry ────────────────────────────────────────────
        ui.label(RichText::new("Shape losses").strong());
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Multi-scale λ:");
            ui.add(egui::Slider::new(&mut self.ms_lambda, 0.0..=5.0).fixed_decimals(2));
        });
        ui.label(RichText::new("Reconstruction loss also computed at 1/2 and 1/4 resolution. Helps G learn global shape for large missing areas.")
            .small().color(Color32::GRAY));

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("Tversky α (FP weight):");
            ui.add(egui::Slider::new(&mut self.tversky_alpha, 0.1..=0.9).fixed_decimals(2));
        });
        ui.horizontal(|ui| {
            ui.label("Tversky β (FN weight):");
            ui.add(egui::Slider::new(&mut self.tversky_beta, 0.1..=0.9).fixed_decimals(2));
        });
        ui.label(RichText::new("Tversky replaces Dice: FP weight α > β penalises over-prediction → better precision. α=β=0.5 equals Dice. Default α=0.7 β=0.3.")
            .small().color(Color32::GRAY));

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("Symmetry λ:");
            ui.add(egui::Slider::new(&mut self.sym_lambda, 0.0..=1.0).fixed_decimals(3));
        });
        ui.label(RichText::new("Penalises asymmetry in predicted mask. Keep at 0 for oak leaves — they are not bilaterally symmetric.")
            .small().color(Color32::GRAY));

        ui.add_space(8.0);

        // ── Training ─────────────────────────────────────────────────────────
        ui.label(RichText::new("Training").strong());
        ui.separator();

        egui::Grid::new("train_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label("Image size:");
                egui::ComboBox::from_id_salt("img_sz")
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

                ui.label("Grad accum steps:");
                ui.add(egui::DragValue::new(&mut self.accum_steps).range(1..=8));
                ui.end_row();

                ui.label("Learning rate:");
                ui.add(egui::DragValue::new(&mut self.learning_rate)
                    .speed(1e-6).fixed_decimals(6));
                ui.end_row();

                ui.label("D LR factor:")
                    .on_hover_text("D LR = G LR × factor. < 1.0 slows discriminator vs generator.\nPrevents D domination. Default 0.5.");
                ui.add(egui::Slider::new(&mut self.d_lr_factor, 0.1..=1.0)
                    .fixed_decimals(2).step_by(0.05));
                ui.end_row();

                ui.label("L1 lambda:");
                ui.add(egui::DragValue::new(&mut self.l1_lambda).range(1.0..=500.0));
                ui.end_row();

                ui.label("Checkpoint every:");
                ui.add(egui::DragValue::new(&mut self.checkpoint_every).range(1..=100)
                    .suffix(" epochs"));
                ui.end_row();

                ui.label("Batch checkpoint:");
                ui.add(egui::DragValue::new(&mut self.batch_checkpoint_every).range(0..=5000)
                    .suffix(" steps"));
                ui.end_row();

                ui.label("Sample grid every:");
                ui.add(egui::DragValue::new(&mut self.sample_grid_every).range(1..=50)
                    .suffix(" epochs"));
                ui.end_row();

                ui.label("Pretrain epochs:");
                ui.add(egui::DragValue::new(&mut self.pretrain_epochs).range(0..=100)
                    .suffix(" epochs"));
                ui.end_row();

                ui.label("Curriculum epochs:");
                ui.add(egui::DragValue::new(&mut self.curriculum_epochs).range(0..=500)
                    .suffix(" epochs"));
                ui.end_row();

                ui.label("Curriculum max:");
                ui.add(egui::Slider::new(&mut self.curriculum_max_pct, 5.0..=85.0)
                    .suffix("% max"));
                ui.end_row();
            });

        ui.label(RichText::new(
            "Pretrain: reconstruction-only before GAN kicks in.\n\
             Curriculum: damage ramps from min% → curriculum max% over N epochs,\n\
             then jumps to full max% for the rest of training."
        ).small().color(Color32::GRAY));

        ui.add_space(10.0);

        // ── Start / Cancel ────────────────────────────────────────────────────
        let can_start = self.source_folder.is_some()
            && self.output_folder.is_some()
            && self.source_count > 1
            && !self.training;

        ui.add_enabled_ui(can_start, |ui| {
            if ui.add_sized([ui.available_width(), 32.0],
                egui::Button::new(RichText::new("Start Training").strong())).clicked() {
                self.start_training();
            }
        });

        if self.training {
            if ui.add_sized([ui.available_width(), 26.0],
                egui::Button::new("Cancel")).clicked() {
                self.cancel_flag.store(true, Ordering::Relaxed);
            }

            // Stall detector: warn if no batch metrics for 5 min.
            // At 512×512 a single step with large damage can take 60–120 s,
            // so 90 s produced false positives; 300 s is a safer threshold.
            let stalled = self.last_batch_tick
                .map(|t| t.elapsed().as_secs() > 300)
                .unwrap_or(false);
            if stalled {
                ui.label(RichText::new(
                    "No activity for 5 min — GPU may be stalled.\n\
                     Cancel will NOT work while the GPU thread is frozen.\n\
                     1. Press Win+Ctrl+Shift+B to restart the NVIDIA display driver.\n\
                     2. Then restart this app — training will resume from the last checkpoint.\n\
                     (Do NOT skip step 1 — accumulated driver state causes faster re-crashes.)"
                ).small().color(Color32::from_rgb(220, 140, 40)));
            }
        }

        if let Some(reason) = (!can_start && !self.training).then(|| {
            if self.source_folder.is_none() { "Select a source folder" }
            else if self.output_folder.is_none() { "Select an output folder" }
            else if self.source_count <= 1 { "Need at least 2 images" }
            else { "" }
        }) {
            if !reason.is_empty() {
                ui.label(RichText::new(reason).small().color(Color32::from_rgb(180,120,60)));
            }
        }

        ui.add_space(8.0);

        // ── Status ────────────────────────────────────────────────────────────
        if self.training || self.current_epoch > 0 {
            ui.label(RichText::new("Status").strong());
            ui.separator();

            ui.label(format!("Epoch: {} / {}", self.current_epoch, self.total_epochs));

            if let Some(m) = &self.latest_metrics {
                egui::Grid::new("metrics_grid")
                    .num_columns(2)
                    .spacing([12.0, 2.0])
                    .show(ui, |ui| {
                        metric_row(ui, "Val IoU",     m.iou);
                        metric_row(ui, "Val F1/Dice", m.dice);
                        metric_row(ui, "Precision",   m.precision);
                        metric_row(ui, "Recall",      m.recall);
                        metric_row(ui, "Pixel Acc",   m.pixel_acc);
                        ui.label("D(x) / D(G(x)):");
                        ui.label(format!("{:.3} / {:.3}", self.latest_d_real, self.latest_d_fake));
                        ui.end_row();
                    });
            }

            if let Some(ckpt) = &self.latest_checkpoint {
                ui.label(RichText::new(format!("Last checkpoint:\n{ckpt}"))
                    .small().color(Color32::GRAY));
            }

            ui.add_space(8.0);
        }

        // ── Log ───────────────────────────────────────────────────────────────
        if !self.log.is_empty() {
            ui.label(RichText::new("Log").strong());
            ui.separator();
            egui::ScrollArea::vertical()
                .id_salt("recon_log_scroll")
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
        // Tab selector
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.right_tab, RightTab::LossCurves, "  Loss Curves  ");
            ui.selectable_value(&mut self.right_tab, RightTab::SampleGrid, "  Sample Grid  ");
        });
        ui.separator();

        match self.right_tab {
            RightTab::LossCurves => self.show_loss_curves(ui),
            RightTab::SampleGrid => self.show_sample_grid(ui),
        }
    }

    fn show_loss_curves(&self, ui: &mut Ui) {
        let half_h = (ui.available_height() / 2.0 - 12.0).max(120.0);

        // ── Batch-level: Generator / Discriminator losses ─────────────────────
        ui.label(RichText::new("Generator & Discriminator loss (per batch)").small());
        Plot::new("batch_loss_plot")
            .height(half_h)
            .legend(Legend::default())
            .show(ui, |plt| {
                plt.line(make_line(&self.b_step, &self.b_g_adv,  "G_adv",  Color32::from_rgb( 80,160,240)));
                plt.line(make_line(&self.b_step, &self.b_g_recon, "G_recon", Color32::from_rgb( 80,200,120)));
                plt.line(make_line(&self.b_step, &self.b_d_loss,  "D_loss", Color32::from_rgb(220, 80, 80)));
                plt.line(make_line(&self.b_step, &self.b_d_real,  "D(x)",   Color32::from_rgb(230,190, 60)));
                plt.line(make_line(&self.b_step, &self.b_d_fake,  "D(G(x))",Color32::from_rgb(160, 80,220)));
            });

        ui.add_space(4.0);

        // ── Epoch-level: Validation metrics ───────────────────────────────────
        ui.label(RichText::new("Validation metrics (per epoch)").small());
        Plot::new("epoch_metric_plot")
            .height(half_h)
            .legend(Legend::default())
            .show(ui, |plt| {
                plt.line(make_line(&self.e_step, &self.e_iou,  "IoU",       Color32::from_rgb( 80,220,120)));
                plt.line(make_line(&self.e_step, &self.e_dice, "Dice",      Color32::from_rgb( 60,210,210)));
                plt.line(make_line(&self.e_step, &self.e_prec, "Precision", Color32::from_rgb( 80,140,240)));
                plt.line(make_line(&self.e_step, &self.e_rec,  "Recall",    Color32::from_rgb(230,140, 50)));
            });
    }

    fn show_sample_grid(&self, ui: &mut Ui) {
        if let Some(tex) = &self.sample_texture {
            let label = format!("Epoch {} — Damaged | GT Mask | Predicted | Diagnostic (TP=green FP=red FN=orange)",
                                self.sample_epoch);
            ui.label(RichText::new(label).small().color(Color32::GRAY));
            ui.add_space(2.0);

            // Per-sample stats row: one entry per grid row
            if !self.sample_stats.is_empty() {
                ui.horizontal(|ui| {
                    for (i, &(iou, f1)) in self.sample_stats.iter().enumerate() {
                        let iou_col = if iou > 0.85 { Color32::from_rgb(80,200,100) }
                                      else if iou > 0.70 { Color32::from_rgb(200,180,60) }
                                      else { Color32::from_rgb(200,80,60) };
                        ui.label(RichText::new(format!("S{}: IoU {:.3}  F1 {:.3}", i + 1, iou, f1))
                            .small().color(iou_col));
                        if i + 1 < self.sample_stats.len() { ui.separator(); }
                    }
                });
                ui.add_space(2.0);
            }

            // Fit texture to available space while preserving aspect ratio
            let size    = tex.size_vec2();
            let avail   = ui.available_size();
            let scale   = (avail.x / size.x).min(avail.y / size.y).min(1.0);
            let show_sz = size * scale;
            ui.image((tex.id(), show_sz));
        } else {
            ui.centered_and_justified(|ui| {
                ui.label(RichText::new(
                    if self.training { "Waiting for first sample grid…" }
                    else { "Start training to generate sample grids." }
                ).color(Color32::GRAY));
            });
        }
    }

    // ── Training message polling ──────────────────────────────────────────────

    fn poll_training(&mut self, ctx: &Context, toasts: &mut ToastManager) {
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
            self.handle_msg(msg, ctx, toasts);
        }
        if done { self.train_rx = None; self.training = false; }
    }

    fn handle_msg(&mut self, msg: TrainMsg, ctx: &Context, toasts: &mut ToastManager) {
        // Any message from the training thread proves it is alive — reset the
        // stall timer here so validation, sample grid, and checkpoint operations
        // don't trigger false stall warnings.
        if self.training {
            self.last_batch_tick = Some(std::time::Instant::now());
        }

        const MAX_BATCH_PTS: usize = 10_000;

        match msg {
            TrainMsg::BatchMetrics { step, g_adv, g_recon, d_loss, d_real, d_fake } => {
                push_capped(&mut self.b_step,    step   as f64, MAX_BATCH_PTS);
                push_capped(&mut self.b_g_adv,   g_adv  as f64, MAX_BATCH_PTS);
                push_capped(&mut self.b_g_recon, g_recon as f64, MAX_BATCH_PTS);
                push_capped(&mut self.b_d_loss,  d_loss as f64, MAX_BATCH_PTS);
                push_capped(&mut self.b_d_real,  d_real as f64, MAX_BATCH_PTS);
                push_capped(&mut self.b_d_fake,  d_fake as f64, MAX_BATCH_PTS);
                self.latest_d_real = d_real;
                self.latest_d_fake = d_fake;
            }

            TrainMsg::EpochMetrics { epoch, metrics } => {
                self.current_epoch = epoch;
                self.e_step.push(epoch as f64);
                self.e_iou.push(metrics.iou as f64);
                self.e_dice.push(metrics.dice as f64);
                self.e_prec.push(metrics.precision as f64);
                self.e_rec.push(metrics.recall as f64);
                self.write_log_file(&format!(
                    "EPOCH {epoch}: IoU={:.4} Dice={:.4} Prec={:.4} Rec={:.4}",
                    metrics.iou, metrics.dice, metrics.precision, metrics.recall,
                ));
                self.latest_metrics = Some(metrics);
            }

            TrainMsg::SampleGrid { epoch, pixels, width, height, sample_stats } => {
                self.sample_epoch   = epoch;
                self.sample_stats   = sample_stats;
                self.sample_texture = Some(ctx.load_texture(
                    "sample_grid",
                    egui::ColorImage::from_rgba_unmultiplied([width, height], &pixels),
                    egui::TextureOptions::LINEAR,
                ));
            }

            TrainMsg::Checkpoint { path } => {
                self.write_log_file(&format!("CHECKPOINT: {path}"));
                self.latest_checkpoint = Some(path);
            }

            TrainMsg::Log(msg) => {
                self.write_log_file(&msg);
                self.push_log(msg);
            }

            TrainMsg::Finished => {
                self.training = false;
                let m = self.latest_metrics.as_ref()
                    .map(|m| format!("Final IoU: {:.4}", m.iou))
                    .unwrap_or_default();
                self.write_log_file(&format!("FINISHED. {m}"));
                toasts.success(format!("Training complete. {m}"));
                self.push_log("Training finished.".to_string());
            }

            TrainMsg::Error(e) => {
                self.training = false;
                self.write_log_file(&format!("ERROR: {e}"));
                toasts.error(format!("Training error: {e}"));
                self.push_log(format!("ERROR: {e}"));
            }
        }
    }

    /// Append a timestamped line to the on-disk training log file.
    fn write_log_file(&self, line: &str) {
        let Some(path) = &self.log_file_path else { return };
        let elapsed = self.train_start
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(path) {
            use std::io::Write;
            let _ = writeln!(f, "[{elapsed:8.1}s] {line}");
        }
    }

    fn push_log(&mut self, msg: String) {
        self.log.push(msg);
        if self.log.len() > 500 { self.log.drain(0..100); }
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
    }

    // ── Start training ────────────────────────────────────────────────────────

    fn start_training(&mut self) {
        let source = match &self.source_folder {
            Some(p) => p.clone(),
            None    => return,
        };
        let output = match &self.output_folder {
            Some(p) => p.clone(),
            None    => return,
        };
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

        // Deterministic shuffle for reproducibility (fixed seed for split)
        use rand::{rngs::SmallRng, SeedableRng, seq::SliceRandom};
        let mut rng = SmallRng::seed_from_u64(42);
        all_paths.shuffle(&mut rng);

        // 90/10 grouped split by file (each file is one unique leaf)
        let split = ((all_paths.len() as f32 * 0.9).ceil() as usize)
            .min(all_paths.len() - 1);
        let train_paths = all_paths[..split].to_vec();
        let val_paths   = all_paths[split..].to_vec();

        // Check for existing checkpoint to resume.
        // Prefer checkpoint_batch_latest (most recent mid-batch save) over
        // checkpoint_best (best-IoU epoch save) — the batch checkpoint is
        // always closer to where training stopped.
        let batch_ckpt = output.join("checkpoint_batch_latest");
        let best_ckpt  = output.join("checkpoint_best");
        let resume_from = if batch_ckpt.join("gen.mpk").exists() {
            Some(batch_ckpt)
        } else if best_ckpt.join("gen.mpk").exists() {
            Some(best_ckpt)
        } else {
            None
        };

        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel_flag = cancel.clone();

        let (tx, rx) = mpsc::channel::<TrainMsg>();
        self.train_rx        = Some(rx);
        self.training        = true;
        self.total_epochs    = self.epochs;
        self.current_epoch   = 0;
        self.last_batch_tick = None;
        self.train_start     = Some(std::time::Instant::now());

        // Open (or create) the log file in the output folder.
        // Append mode so multiple runs accumulate; a separator marks each start.
        let log_path = output.join("training_log.txt");
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&log_path) {
            use std::io::Write;
            let _ = writeln!(f, "\n=== Training started ===");
        }
        self.log_file_path = Some(log_path);

        // Clear all plot history so the new run starts with a fresh graph
        self.b_step.clear();  self.b_g_adv.clear(); self.b_g_recon.clear();
        self.b_d_loss.clear(); self.b_d_real.clear(); self.b_d_fake.clear();
        self.e_step.clear();  self.e_iou.clear();  self.e_dice.clear();
        self.e_prec.clear();  self.e_rec.clear();

        let cfg = TrainConfig {
            train_paths,
            val_paths,
            output_dir:       output,
            species_label:    self.species_label.clone(),
            epochs:           self.epochs,
            batch_size:       self.batch_size,
            lr:               self.learning_rate,
            l1_lambda:        self.l1_lambda,
            checkpoint_every:       self.checkpoint_every,
            batch_checkpoint_every: self.batch_checkpoint_every,
            sample_every:           self.sample_grid_every,
            image_size:       self.image_size_px as usize,
            damage_params: DamageParams {
                min_pct:          self.damage_min_pct,
                max_pct:          self.damage_max_pct,
                coastal:          self.coastal_enabled,
                coastal_w:        self.coastal_weight,
                spots:            self.spots_enabled,
                spots_w:          self.spots_weight,
                snake:            self.snake_enabled,
                snake_w:          self.snake_weight,
                ellipses:         self.ellipses_enabled,
                ellipses_w:       self.ellipses_weight,
                apex:             self.apex_enabled,
                apex_w:           self.apex_weight,
                clusters:         self.clusters_enabled,
                clusters_w:       self.clusters_weight,
                lobe:             self.lobe_enabled,
                lobe_w:           self.lobe_weight,
                focal_sector:     self.focal_sector_enabled,
                focal_sector_w:   self.focal_sector_weight,
                zero_damage_prob:     self.zero_damage_prob,
                curriculum_max: self.curriculum_max_pct,
            },
            resume_from,
            bg_color:            [40, 40, 40],
            curriculum_epochs:   self.curriculum_epochs,
            pretrain_epochs:     self.pretrain_epochs,
            d_lr_factor:         self.d_lr_factor,
            recon_focus_weight:  self.recon_focus_weight,
            adv_lambda:          self.adv_lambda,
            tv_lambda:           self.tv_lambda,
            conf_lambda:         self.conf_lambda,
            adaptive_ctrl:       self.adaptive_ctrl,
            ms_lambda:           self.ms_lambda,
            sym_lambda:          self.sym_lambda,
            tversky_alpha:       self.tversky_alpha,
            tversky_beta:        self.tversky_beta,
            boundary_exclusion_px: self.boundary_exclusion_px,
            accum_steps:           self.accum_steps,
            leaf_shape:            0,
            margin_type:           0,
            area_lambda:           self.area_lambda,
        };

        spawn_training(cfg, tx, cancel);
        if crate::tabs::recon_train::model::backend_name() == "CPU" {
            self.push_log("WARNING: Running on CPU backend — this binary has no GPU support.".to_string());
            self.push_log("  For CUDA:  cargo build --release".to_string());
            self.push_log("  For wgpu:  cargo build --release --no-default-features --features wgpu-gpu".to_string());
        }
        self.push_log("Training started.".to_string());
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_line(x: &[f64], y: &[f64], name: &str, color: Color32) -> Line {
    let pts: PlotPoints = x.iter().zip(y.iter())
        .map(|(&xi, &yi)| [xi, yi])
        .collect();
    Line::new(pts).name(name).color(color)
}

fn push_capped(v: &mut Vec<f64>, val: f64, cap: usize) {
    v.push(val);
    if v.len() > cap { v.drain(0..cap / 10); }
}

fn metric_row(ui: &mut egui::Ui, label: &str, val: f32) {
    ui.label(format!("{}:", label));
    let color = if val > 0.85 {
        Color32::from_rgb(80, 200, 100)
    } else if val > 0.7 {
        Color32::from_rgb(200, 180, 60)
    } else {
        Color32::from_rgb(200, 80, 60)
    };
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
