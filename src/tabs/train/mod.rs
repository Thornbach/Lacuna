//! In-app anomaly-detector trainer (Step 2.5).
//!
//! Healthy folder -> DINOv3 patch features -> coreset -> calibration ->
//! `coreset_bank.bin` + `detector_meta.json`. Removes Python from training and
//! enables the active-learning loop (mark consistent false-positives, add them to
//! the healthy set, re-fit) entirely inside FoliarToolbox.

pub mod coreset;
pub mod fit;
pub mod head;

use std::{
    path::PathBuf,
    sync::{mpsc, Arc},
    sync::atomic::{AtomicBool, Ordering},
};

use egui::{Color32, Context, RichText, Ui};

use crate::settings::{AppDefaults, AppSettings};
use crate::tabs::leaf_seg::inference::{list_images, scan_image_count};
use crate::tabs::pipeline::hardneg_mining::{spawn_mine, MineConfig, MineMsg};
use crate::ui_kit;
use crate::widgets::ToastManager;
use fit::{spawn_fit, FitConfig, FitMsg};
use head::{spawn_retrain, RetrainCfg, RetrainMsg};

/// Fixed hard-negative crop size for mining, matching the Pipeline tab's
/// own `hardneg_tile` default — not exposed as a separate control here to
/// keep this tab lean; Pipeline's own Hardneg stamp tool is where that
/// size actually matters for manual stamping precision.
const MINE_TILE_SIZE: u32 = 64;

#[derive(Clone, Copy)]
enum Pick { Healthy, Dino, Out, Curations, Head, MineHealthyDir, BaseSet }

pub struct TrainTab {
    healthy_folder: Option<PathBuf>,
    dino_model:     Option<PathBuf>,
    out_dir:        Option<PathBuf>,
    coreset_target: u32,
    n_fit:          u32,
    n_calib:        u32,
    max_fp_rate:    f32,
    margin_erode:   u32,

    rx:             Option<mpsc::Receiver<FitMsg>>,
    cancel_flag:    Arc<AtomicBool>,
    running:        bool,
    progress_done:  usize,
    progress_total: usize,
    stage:          String,
    log:            Vec<String>,

    pick_rx:      Option<(Pick, mpsc::Receiver<Option<PathBuf>>)>,
    healthy_count: usize,
    defaults:     AppDefaults, // shared defaults from Settings (dino, healthy)

    // flywheel: retrain the few-shot head from saved curations
    curations_dir: Option<PathBuf>,
    head_path:     Option<PathBuf>,
    retrain_rx:    Option<mpsc::Receiver<RetrainMsg>>,
    retrain_cancel: Arc<AtomicBool>,
    retraining:    bool,
    retrain_log:   Vec<String>,
    /// See `RetrainCfg::cold_start`'s doc comment — kept in sync with the
    /// Pipeline tab's own identical checkbox.
    retrain_cold_start: bool,
    /// See `RetrainCfg::dump_dir` — same diagnostic as the Pipeline tab's.
    retrain_dump: bool,
    /// See `RetrainCfg::base_set` — the original training rows mixed into every
    /// retrain so curations fine-tune the head instead of replacing it.
    retrain_base_set:  Option<PathBuf>,
    retrain_base_rows: usize,
    /// See `RetrainCfg::anchor` - pull the L2 penalty toward the current head
    /// instead of toward zero, so curations correct without competing for influence.
    retrain_anchor:    f32,

    // hard-negative mining (automated, patch-level) — see
    // pipeline::hardneg_mining. Reuses the same curations_dir/head_path
    // above (mining and retrain operate on the same curations folder).
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
}

impl TrainTab {
    pub fn new() -> Self {
        Self {
            healthy_folder: None,
            dino_model:     None,
            out_dir:        None,
            coreset_target: 150_000,
            n_fit:          3000,
            n_calib:        200,
            max_fp_rate:    0.0005,
            margin_erode:   6,

            rx:             None,
            cancel_flag:    Arc::new(AtomicBool::new(false)),
            running:        false,
            progress_done:  0,
            progress_total: 0,
            stage:          String::new(),
            log:            Vec::new(),

            pick_rx:        None,
            healthy_count:  0,
            defaults:       AppDefaults::default(),

            curations_dir:  None,
            head_path:      None,
            retrain_rx:     None,
            retrain_cancel: Arc::new(AtomicBool::new(false)),
            retraining:     false,
            retrain_log:    Vec::new(),
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
        }
    }

    pub fn needs_repaint(&self) -> bool { self.running || self.retraining || self.mining }

    pub fn set_defaults(&mut self, d: &AppDefaults) {
        self.defaults = d.clone();
    }

    fn eff_dino(&self) -> Option<PathBuf> {
        self.dino_model.clone().or_else(|| self.defaults.dino.clone())
    }
    fn eff_healthy(&self) -> Option<PathBuf> {
        self.healthy_folder.clone().or_else(|| self.defaults.healthy.clone())
    }

    pub fn save_settings(&self, s: &mut AppSettings) {
        let t = &mut s.train;
        t.healthy_folder = self.healthy_folder.clone();
        t.dino_model     = self.dino_model.clone();
        t.out_dir        = self.out_dir.clone();
        t.coreset_target = self.coreset_target;
        t.n_fit          = self.n_fit;
    }

    pub fn load_settings(&mut self, s: &AppSettings) {
        let t = &s.train;
        self.healthy_folder = t.healthy_folder.clone();
        self.dino_model     = t.dino_model.clone();
        self.out_dir        = t.out_dir.clone();
        self.coreset_target = t.coreset_target;
        self.n_fit          = t.n_fit;
        if let Some(f) = self.healthy_folder.clone() {
            self.healthy_count = scan_image_count(&f);
        }
    }

    pub fn show(&mut self, ui: &mut Ui, _ctx: &Context, toasts: &mut ToastManager) {
        self.poll_pick();
        self.poll_worker(toasts);
        self.poll_retrain(toasts);
        self.poll_mine(toasts);

        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ScrollArea::vertical().id_salt("train_scroll").show(ui, |ui| {
                ui.add_space(6.0);
                ui.heading("Train Detector");
                ui.label(
                    RichText::new(
                        "Fit a fresh anomaly bank from a HEALTHY tile folder (DINO features -> \
                         random coreset -> calibration). Writes coreset_bank.bin + \
                         detector_meta.json for the Pipeline tab. Only add tiles you are sure are \
                         healthy — PatchCore treats anything in the bank as normal.",
                    )
                    .color(Color32::GRAY),
                );
                ui.separator();

                ui_kit::section_header(ui, "Inputs");
                self.pick_row(ui, "Healthy tiles folder", Pick::Healthy);
                if self.healthy_folder.is_some() {
                    ui.label(RichText::new(format!("{} tiles found", self.healthy_count)).small());
                }
                self.pick_row(ui, "DINOv3 model (.onnx)", Pick::Dino);
                self.pick_row(ui, "Output folder", Pick::Out);

                ui.add_space(10.0);
                let can_start = self.ready() && !self.running;
                ui.add_enabled_ui(can_start, |ui| {
                    if ui.add_sized([220.0, 32.0],
                        egui::Button::new(
                            RichText::new("Fit Detector").strong().color(ui_kit::on_accent()),
                        ).fill(ui_kit::ACCENT())).clicked()
                    {
                        self.start();
                    }
                });
                if self.running {
                    if ui.add_sized([220.0, 26.0], egui::Button::new("Cancel")).clicked() {
                        self.cancel_flag.store(true, Ordering::Relaxed);
                    }
                    let frac = if self.progress_total > 0 {
                        self.progress_done as f32 / self.progress_total as f32
                    } else { 0.0 };
                    ui.add(egui::ProgressBar::new(frac).desired_width(220.0).show_percentage());
                    if !self.stage.is_empty() {
                        ui.horizontal(|ui| ui_kit::busy(ui, &self.stage));
                    }
                } else if !can_start {
                    ui.label(RichText::new("Set healthy folder, DINO model and output folder.")
                        .small().color(Color32::GRAY));
                }

                if !self.log.is_empty() {
                    ui_kit::section_header(ui, "Log");
                    for line in self.log.iter().rev().take(40) {
                        ui.label(RichText::new(line).small());
                    }
                }

                // ── flywheel: retrain the few-shot head from saved curations ──
                ui.add_space(14.0);
                ui.separator();
                ui_kit::section_header(ui, "Flywheel — retrain head from curations");
                ui.label(RichText::new(
                    "Fold your Pipeline curations (renamed clusters = labels, removed = rejects) \
                     back into the few-shot head: warm-started from the current head, zero-centered \
                     L2-regularized (a class's confidence reflects only its own evidence, not \
                     whatever it happened to inherit). New cluster names become new families. \
                     Writes fewshot_head_retrained.json next to the head.")
                    .small().color(Color32::GRAY));
                self.pick_row(ui, "Curations folder (<output>/curations)", Pick::Curations);
                self.pick_row(ui, "Head to refine (.json)", Pick::Head);
                ui.add_space(6.0);

                // ── automated, patch-level hard-negative mining — shares the
                // same curations folder + head above, so results land where
                // Retrain (right below) will pick them straight up ──
                ui_kit::section_header(ui, "Mine hard negatives");
                ui.label(RichText::new(
                    "Scan a folder of KNOWN-HEALTHY tiles for patches the current head wrongly \
                     calls defect, and stamp them into the SAME curations folder as new \
                     hard-negative examples — the automated, patch-level counterpart to Pipeline's \
                     manual Hardneg stamp tool.")
                    .small().color(Color32::GRAY));
                self.pick_row(ui, "Healthy tiles folder", Pick::MineHealthyDir);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("τ_mine").small())
                        .on_hover_text("A patch is mined when defect_prob ≥ this value.");
                    ui.add(egui::Slider::new(&mut self.mine_tau, 0.3..=0.95).fixed_decimals(2));
                });
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Cap").small())
                        .on_hover_text("Max hard negatives this run can add. Every one mined \
                                         permanently adds to labels.jsonl and gets re-featurized \
                                         on EVERY future Retrain — keep this modest.");
                    ui.add(egui::DragValue::new(&mut self.mine_max).range(50..=2000).speed(10));
                });
                let can_mine = self.curations_dir.is_some() && self.eff_head().is_some()
                    && self.eff_dino().is_some() && self.mine_healthy_dir.is_some()
                    && !self.mining && !self.retraining && !self.running;
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
                    egui::ScrollArea::vertical().max_height(80.0).id_salt("train_mine_log").show(ui, |ui| {
                        for line in self.mine_log.iter().rev().take(20) {
                            ui.label(RichText::new(line).small());
                        }
                    });
                }
                ui.add_space(10.0);

                self.pick_row(ui, "Base training set (.bin)", Pick::BaseSet);
                ui.horizontal(|ui| {
                    ui.label("base rows");
                    ui.add(egui::DragValue::new(&mut self.retrain_base_rows)
                        .range(0..=400_000).speed(1000));
                });
                ui.horizontal(|ui| {
                    ui.label("anchor to current head");
                    ui.add(egui::Slider::new(&mut self.retrain_anchor, 0.0..=1.0).fixed_decimals(2))
                        .on_hover_text("0 = penalty pulls toward zero (old behaviour); \
                                        1 = pulls toward the current head, so curations \
                                        correct without having to out-vote the base rows. \
                                        Measured best: 10k base rows + anchor 1.0.");
                });
                ui.label(RichText::new(
                    "Base rows stop a retrain discarding what the head knew (without them: \
                     IoU 0.475 -> 0.125). The anchor achieves the same by bounding travel \
                     instead of out-voting the curations — 10k + anchor beat 50k alone on \
                     both learning (0.942 vs 0.969) and retention (0.476 vs 0.460)."
                ).small().color(Color32::GRAY));
                ui.add_space(4.0);
                ui.checkbox(&mut self.retrain_cold_start, "Train from scratch (no warm start)")
                    .on_hover_text("Every retrain already reads ALL accumulated curations, not \
                                    just new ones — this only changes where the solver STARTS. \
                                    With this on, any class with curated examples this run starts \
                                    from zero instead of the current head's own coefficients, so \
                                    its result depends only on the curated evidence itself. \
                                    Classes with no curated history are unaffected either way.");
                ui.checkbox(&mut self.retrain_dump, "Dump training data (diagnostics)")
                    .on_hover_text("Writes <curations>/retrain_diag/: retrain_dump.bin (the exact \
                                    feature rows/classes/weights this retrain uses) and \
                                    retrain_diag.json (per-class norms + intercepts before/after, \
                                    solver settings, convergence). Lets an independent solver be \
                                    fitted on identical data to separate DATA problems from \
                                    SOLVER problems. Several hundred MB — opt-in.");
                let can_retrain = self.curations_dir.is_some() && self.eff_head().is_some()
                    && self.eff_dino().is_some() && !self.retraining && !self.running && !self.mining;
                ui.add_enabled_ui(can_retrain, |ui| {
                    if ui.add_sized([220.0, 28.0],
                        egui::Button::new(RichText::new("Retrain head").strong())).clicked()
                    {
                        self.start_retrain();
                    }
                });
                if self.retraining {
                    if ui.add_sized([220.0, 24.0], egui::Button::new("Cancel")).clicked() {
                        self.retrain_cancel.store(true, Ordering::Relaxed);
                    }
                    ui.horizontal(|ui| ui_kit::busy(ui, &self.stage));
                } else if !can_retrain {
                    ui.label(RichText::new("Need: curations folder + a head + DINO model.")
                        .small().color(Color32::GRAY));
                }
                for line in self.retrain_log.iter().rev().take(12) {
                    ui.label(RichText::new(line).small());
                }
            });
        });
    }

    pub fn show_settings_panel(&mut self, ui: &mut Ui) {
        ui_kit::section_header(ui, "Parameters");
        egui::Grid::new("train_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
            ui.label("Coreset size:");
            ui.add(egui::DragValue::new(&mut self.coreset_target).range(1000..=1_000_000).speed(1000));
            ui.end_row();
            ui.label("Fit tiles:");
            ui.add(egui::DragValue::new(&mut self.n_fit).range(50..=100_000).speed(50));
            ui.end_row();
            ui.label("Calibration tiles:");
            ui.add(egui::DragValue::new(&mut self.n_calib).range(20..=5000).speed(10));
            ui.end_row();
            ui.label("Max FP rate:")
                .on_hover_text("Fraction of clean pixels allowed to fire. Lower = stricter.");
            ui.add(egui::Slider::new(&mut self.max_fp_rate, 0.0001..=0.01).logarithmic(true));
            ui.end_row();
            ui.label("Margin erode (px):");
            ui.add(egui::Slider::new(&mut self.margin_erode, 0..=20));
            ui.end_row();
        });
    }

    fn start_retrain(&mut self) {
        let (Some(cur), Some(head), Some(dino)) =
            (self.curations_dir.clone(), self.eff_head(), self.eff_dino())
        else { return };
        let out_path = head.parent().map(|p| p.join("fewshot_head_retrained.json"))
            .unwrap_or_else(|| PathBuf::from("fewshot_head_retrained.json"));
        self.retrain_log.clear();
        self.retrain_cancel = Arc::new(AtomicBool::new(false));
        self.retraining = true;
        self.stage = "Retraining head".into();
        let (tx, rx) = mpsc::channel();
        self.retrain_rx = Some(rx);
        spawn_retrain(
            RetrainCfg {
                head_path: head,
                dino_model: dino,
                dump_dir: self.retrain_dump.then(|| cur.join("retrain_diag")),
                base_set: self.retrain_base_set.clone(),
                base_rows: self.retrain_base_rows,
                anchor: self.retrain_anchor,
                curations_dir: cur,
                out_path,
                // Keep in sync with the Pipeline tab's own retrain call site
                // (pipeline/mod.rs) — see its comment and head.rs's
                // RetrainCfg doc comments for the full incident history.
                // `max_iters` is a safety ceiling, not a target — a real
                // L-BFGS solve, not fixed-epoch gradient descent.
                max_iters: 2000,
                // sklearn's C (inverse strength), matching the base head's own
                // LogisticRegression(C=1.0) — see RetrainCfg::l2_reg for why
                // the previous raw 0.02 was ~34x too strong.
                l2_reg: 1.0,
                max_patches_per_crop: 8,
                validate_healthy_dir: self.mine_healthy_dir.clone(),
                validate_tau: self.mine_tau,
                cold_start: self.retrain_cold_start,
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
                    RetrainMsg::Stage(s) => self.stage = s,
                    RetrainMsg::Log(l) => self.retrain_log.push(l),
                    RetrainMsg::Error(e) => {
                        self.retrain_log.push(format!("ERROR: {e}"));
                        toasts.error(format!("Retrain failed: {e}"));
                        done = true;
                    }
                    RetrainMsg::Done(s) => {
                        self.retrain_log.push(s.clone());
                        toasts.success(s);
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

    fn start_mine(&mut self) {
        let (Some(cur), Some(head), Some(dino), Some(healthy_dir)) = (
            self.curations_dir.clone(), self.eff_head(), self.eff_dino(), self.mine_healthy_dir.clone(),
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
                curations_dir: cur,
                tau_mine: self.mine_tau,
                hardneg_tile: MINE_TILE_SIZE,
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

    fn pick_row(&mut self, ui: &mut Ui, label: &str, which: Pick) {
        let own = self.field(which).clone();
        let inherited = match which {
            Pick::Dino => self.defaults.dino.clone(),
            Pick::Healthy => self.defaults.healthy.clone(),
            Pick::Head => self.defaults.head.clone(),
            _ => None,
        };
        if ui.button(label).clicked() && self.pick_rx.is_none() {
            self.pick_rx = Some((which, spawn_dialog(which)));
        }
        let (txt, col) = match (&own, &inherited) {
            (Some(p), _) => (p.display().to_string(), Color32::GRAY),
            (None, Some(p)) => (format!("inherits: {}", p.display()), ui_kit::ACCENT()),
            (None, None) => ("- not set -".to_string(), Color32::GRAY),
        };
        ui.label(RichText::new(txt).small().color(col));
    }

    fn field(&self, which: Pick) -> &Option<PathBuf> {
        match which {
            Pick::Healthy => &self.healthy_folder,
            Pick::Dino => &self.dino_model,
            Pick::Out => &self.out_dir,
            Pick::Curations => &self.curations_dir,
            Pick::Head => &self.head_path,
            Pick::MineHealthyDir => &self.mine_healthy_dir,
            Pick::BaseSet => &self.retrain_base_set,
        }
    }

    fn eff_head(&self) -> Option<PathBuf> {
        self.head_path.clone().or_else(|| self.defaults.head.clone())
    }

    fn ready(&self) -> bool {
        let ex = |p: Option<PathBuf>| p.map(|p| p.exists()).unwrap_or(false);
        ex(self.eff_healthy()) && ex(self.eff_dino()) && self.out_dir.is_some()
    }

    fn start(&mut self) {
        let (Some(healthy), Some(dino), Some(out)) =
            (self.eff_healthy(), self.eff_dino(), self.out_dir.clone())
        else { return };
        let healthy_paths = list_images(&healthy);
        if healthy_paths.is_empty() {
            self.log.push("No tiles in healthy folder.".into());
            return;
        }
        self.log.clear();
        self.cancel_flag = Arc::new(AtomicBool::new(false));
        self.progress_done = 0;
        self.progress_total = (self.n_fit as usize).min(healthy_paths.len());
        self.running = true;
        self.stage = "Loading DINOv3".into();

        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);
        spawn_fit(
            FitConfig {
                healthy_paths,
                dino_model: dino,
                out_dir: out,
                dino_res: 512,
                coreset_target: self.coreset_target as usize,
                n_fit: self.n_fit as usize,
                n_calib: self.n_calib as usize,
                max_fp_rate: self.max_fp_rate,
                scale_floor_pct: 25.0,
                margin_erode: self.margin_erode,
                knn_k: 3,
                seed: 0,
            },
            tx,
            self.cancel_flag.clone(),
        );
    }

    fn poll_worker(&mut self, toasts: &mut ToastManager) {
        let mut finished = false;
        if let Some(rx) = &self.rx {
            for msg in rx.try_iter().take(64) {
                match msg {
                    FitMsg::Stage(s) => self.stage = s,
                    FitMsg::Progress { done, total } => {
                        self.progress_done = done;
                        self.progress_total = total;
                    }
                    FitMsg::Log(l) => self.log.push(l),
                    FitMsg::Error(e) => {
                        self.log.push(format!("ERROR: {e}"));
                        toasts.error(format!("Fit failed: {e}"));
                        finished = true;
                    }
                    FitMsg::Finished => finished = true,
                }
            }
        }
        if finished {
            self.running = false;
            self.rx = None;
            if let Some(out) = &self.out_dir {
                toasts.success(format!("Detector written to {}", out.display()));
            }
        }
    }

    fn poll_pick(&mut self) {
        if let Some((which, rx)) = &self.pick_rx {
            if let Ok(res) = rx.try_recv() {
                let which = *which;
                if let Some(p) = res {
                    match which {
                        Pick::Healthy => {
                            self.healthy_count = scan_image_count(&p);
                            self.healthy_folder = Some(p);
                        }
                        Pick::Dino => self.dino_model = Some(p),
                        Pick::Out => self.out_dir = Some(p),
                        Pick::Curations => self.curations_dir = Some(p),
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

fn spawn_dialog(which: Pick) -> mpsc::Receiver<Option<PathBuf>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let res = match which {
            Pick::Healthy | Pick::Out | Pick::Curations | Pick::MineHealthyDir => rfd::FileDialog::new().pick_folder(),
            Pick::Dino => rfd::FileDialog::new()
                .add_filter("model weights", &["safetensors", "onnx"])
                .add_filter("all files", &["*"])
                .pick_file(),
            Pick::Head => rfd::FileDialog::new().add_filter("json", &["json"]).pick_file(),
            Pick::BaseSet => rfd::FileDialog::new().add_filter("base set", &["bin"]).pick_file(),
        };
        let _ = tx.send(res);
    });
    rx
}

