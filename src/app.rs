use std::path::PathBuf;
use std::sync::mpsc;

use egui::{Color32, Context, RichText, Ui};
use egui_phosphor::regular as icon;

use crate::settings::{AppDefaults, AppSettings};
use crate::tabs::recon_train::model::backend_name;
use crate::tabs::{
    EroderTab, FieldReviewTab, LeafSegTab, MorphologyTab, PipelineTab, ReconInferTab,
    ReconSimpleTab, SorterTab, TilePickerTab, TrainTab,
};
use crate::ui_kit;
use crate::widgets::ToastManager;

// ── Tab enum ──────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    // production flow
    LeafSeg,
    FieldReview,
    Pipeline,
    Train,
    Morphology,
    // tools
    TilePicker,
    Eroder,
    Sorter,
    // reconstruction
    ReconSimple,
    ReconInfer,
    // app
    Settings,
}

#[derive(Clone, Copy, PartialEq)]
enum SettingsCategory {
    General, Pipeline, ReconTrain, ReconInfer, LeafSeg, FieldReview,
    Eroder, Train, TilePicker, Morphology, Sorter,
}

// ── Root app ──────────────────────────────────────────────────────────────────

pub struct LacunaApp {
    active_tab:    Tab,
    eroder:        EroderTab,
    sorter:        SorterTab,
    recon_simple:  ReconSimpleTab,
    recon_infer:   ReconInferTab,
    leaf_seg:      LeafSegTab,
    field_review:  FieldReviewTab,
    pipeline:      PipelineTab,
    train:         TrainTab,
    tile_picker:   TilePickerTab,
    morphology:    MorphologyTab,
    settings:      AppSettings,
    toasts:        ToastManager,
    default_pick:  Option<(DefaultField, mpsc::Receiver<Option<PathBuf>>)>,
    settings_category: SettingsCategory,
    // tab-switch fade-in transition
    prev_tab:       Tab,
    tab_changed_at: f64,
}

#[derive(Clone, Copy)]
enum DefaultField { Yolo, Dino, Bank, Meta, Head, Recon, Healthy }

impl LacunaApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut settings: AppSettings = cc.storage
            .and_then(|s| eframe::get_value(s, "app_settings"))
            .or_else(AppSettings::load_from_file)
            .unwrap_or_default();

        // Fill any unset shared-model defaults from the bundled `models/` folder
        // shipped next to the exe, so a fresh install works without picking paths.
        fill_bundled_defaults(&mut settings.defaults);

        // Icon font for the nav rail + buttons.
        let mut fonts = egui::FontDefinitions::default();
        egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Regular);
        cc.egui_ctx.set_fonts(fonts);

        crate::theme::apply(&cc.egui_ctx, &settings.theme_name);
        ui_kit::apply_style(&cc.egui_ctx);
        cc.egui_ctx.set_pixels_per_point(settings.ui_scale.clamp(0.7, 2.0));

        let mut eroder        = EroderTab::new();
        let mut sorter        = SorterTab::new();
        let mut recon_simple  = ReconSimpleTab::new();
        let mut recon_infer   = ReconInferTab::new();
        let mut leaf_seg      = LeafSegTab::new();
        let mut field_review  = FieldReviewTab::new();
        let mut pipeline      = PipelineTab::new();
        let mut train         = TrainTab::new();
        let mut tile_picker   = TilePickerTab::new();
        let mut morphology    = MorphologyTab::new();
        eroder.load_settings(&settings);
        sorter.load_settings(&settings);
        recon_simple.load_settings(&settings);
        recon_infer.load_settings(&settings);
        leaf_seg.load_settings(&settings);
        field_review.load_settings(&settings);
        pipeline.load_settings(&settings);
        train.load_settings(&settings);
        tile_picker.load_settings(&settings);
        morphology.load_settings(&settings);

        Self {
            active_tab: Tab::Pipeline, // production flow front and centre
            eroder, sorter, recon_simple, recon_infer,
            leaf_seg, field_review, pipeline, train, tile_picker, morphology,
            settings, toasts: ToastManager::new(),
            default_pick: None,
            settings_category: SettingsCategory::General,
            prev_tab: Tab::Pipeline,
            tab_changed_at: 0.0,
        }
    }

    fn poll_default_pick(&mut self) {
        if let Some((field, rx)) = &self.default_pick {
            if let Ok(res) = rx.try_recv() {
                let field = *field;
                if let Some(p) = res {
                    set_default(&mut self.settings.defaults, field, Some(p));
                }
                self.default_pick = None;
            }
        }
    }

    fn active_title(&self) -> &'static str {
        match self.active_tab {
            Tab::LeafSeg => "Leaf Segmentation",
            Tab::FieldReview => "Field Review",
            Tab::Pipeline => "Integrated Pipeline",
            Tab::Train => "Train Detector",
            Tab::Morphology => "Morphology",
            Tab::TilePicker => "Tile Picker",
            Tab::Eroder => "Eroder",
            Tab::Sorter => "Sorter",
            Tab::ReconSimple => "Recon Train",
            Tab::ReconInfer => "Recon Infer",
            Tab::Settings => "Settings",
        }
    }

    fn any_busy(&self) -> bool {
        self.eroder.is_processing()
            || self.sorter.needs_repaint()
            || self.recon_simple.needs_repaint()
            || self.recon_infer.needs_repaint()
            || self.leaf_seg.needs_repaint()
            || self.field_review.needs_repaint()
            || self.pipeline.needs_repaint()
            || self.train.needs_repaint()
            || self.tile_picker.needs_repaint()
            || self.morphology.needs_repaint()
    }

    // ── Top menu bar (replaces the old left nav rail) ───────────────────────────
    //
    // Grouped by WORKFLOW, not by model type — mirrors the old nav rail's own
    // 3-section split, just re-cut: File = process/analyze a dataset, Train =
    // the three training workflows (Field Review teaches the segmentation
    // model, Recon Train the reconstruction model, Train Detector the
    // anomaly-classification head), Tools = the three standalone utilities.

    fn tab_button(ui: &mut Ui, tab: &mut Tab, target: Tab, label: &str) {
        if ui.selectable_label(*tab == target, label).clicked() {
            *tab = target;
        }
    }

    // ── Slim context bar (active screen + global busy indicator) ────────────────

    fn show_top_bar(&mut self, ctx: &Context) {
        egui::TopBottomPanel::top("top_bar").exact_height(36.0).show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.add_space(4.0);
                ui.label(
                    RichText::new(format!("{}  Lacuna", icon::LEAF))
                        .size(16.0)
                        .strong()
                        .color(ui_kit::ACCENT),
                );
                ui.separator();

                Self::tab_button(ui, &mut self.active_tab, Tab::Pipeline, "Pipeline");
                Self::tab_button(ui, &mut self.active_tab, Tab::LeafSeg, "Leaf Seg");
                Self::tab_button(ui, &mut self.active_tab, Tab::Morphology, "Morphology");
                Self::tab_button(ui, &mut self.active_tab, Tab::ReconInfer, "Recon Infer");
                ui.separator();
                Self::tab_button(ui, &mut self.active_tab, Tab::FieldReview, "Field Review");
                Self::tab_button(ui, &mut self.active_tab, Tab::ReconSimple, "Recon Train");
                Self::tab_button(ui, &mut self.active_tab, Tab::Train, "Train Detector");
                ui.separator();
                Self::tab_button(ui, &mut self.active_tab, Tab::TilePicker, "Tile Picker");
                Self::tab_button(ui, &mut self.active_tab, Tab::Eroder, "Eroder");
                Self::tab_button(ui, &mut self.active_tab, Tab::Sorter, "Sorter");

                // right-aligned: current tab title, busy indicator, Settings —
                // Settings is a single destination, not a dropdown.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.selectable_label(self.active_tab == Tab::Settings, format!("{} Settings", icon::GEAR))
                        .clicked()
                    {
                        self.active_tab = Tab::Settings;
                    }
                    ui.separator();
                    if self.any_busy() {
                        ui_kit::busy(ui, "working…");
                        ui.add_space(10.0);
                    }
                    ui.label(RichText::new(self.active_title()).size(15.0).strong());
                });
            });
        });
    }

    // ── Settings screen ─────────────────────────────────────────────────────────

    fn show_settings(&mut self, ui: &mut Ui, ctx: &Context) {
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            use SettingsCategory::*;
            for (cat, label) in [
                (General, "General"), (Pipeline, "Pipeline"), (ReconTrain, "Recon Train"),
                (ReconInfer, "Recon Infer"), (LeafSeg, "Leaf Seg"), (FieldReview, "Field Review"), (Eroder, "Eroder"),
                (Train, "Train"), (TilePicker, "Tile Picker"), (Morphology, "Morphology"),
                (Sorter, "Sorter"),
            ] {
                ui.selectable_value(&mut self.settings_category, cat, label);
            }
        });
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.set_max_width(560.0);
            ui.add_space(6.0);

            match self.settings_category {
                SettingsCategory::General => {
                    ui_kit::section_header(ui, "Appearance");
                    egui::Grid::new("settings_appearance")
                        .num_columns(2)
                        .spacing([12.0, 8.0])
                        .show(ui, |ui| {
                            ui.label("Theme");
                            let current = self.settings.theme_name.clone();
                            egui::ComboBox::from_id_salt("theme_picker")
                                .selected_text(&current)
                                .width(200.0)
                                .show_ui(ui, |ui| {
                                    for name in crate::theme::theme_names() {
                                        if ui.selectable_label(current == name, name).clicked() {
                                            self.settings.theme_name = name.to_string();
                                            crate::theme::apply(ctx, name);
                                            ui_kit::apply_style(ctx);
                                        }
                                    }
                                });
                            ui.end_row();

                            ui.label("UI scale");
                            if ui
                                .add(egui::Slider::new(&mut self.settings.ui_scale, 0.8..=1.6).fixed_decimals(2))
                                .changed()
                            {
                                ctx.set_pixels_per_point(self.settings.ui_scale);
                            }
                            ui.end_row();
                        });

                    ui_kit::section_header(ui, "Compute");
                    ui.label(format!("Backend: {}", backend_name()));

                    ui_kit::section_header(ui, "Shared model defaults");
                    ui_kit::caption(
                        ui,
                        "Set these once; the Pipeline & Train tabs inherit them when their own field is empty.",
                    );
                    ui.add_space(4.0);
                    self.default_row(ui, "YOLO seg (.onnx)", DefaultField::Yolo);
                    self.default_row(ui, "DINOv3 (.onnx)", DefaultField::Dino);
                    self.default_row(ui, "Few-shot head (.json)", DefaultField::Head);
                    self.default_row(ui, "Coreset bank (.bin)", DefaultField::Bank);
                    self.default_row(ui, "Detector meta (.json)", DefaultField::Meta);
                    self.default_row(ui, "Recon checkpoint (.mpk)", DefaultField::Recon);
                    self.default_row(ui, "Healthy tiles (folder)", DefaultField::Healthy);

                    ui_kit::section_header(ui, "About");
                    ui.label("Lacuna");
                    ui_kit::caption(ui, &format!("v{} - egui 0.29 + Burn 0.16", env!("CARGO_PKG_VERSION")));
                    // Third-party model attributions (required by their licenses).
                    ui_kit::caption(ui, "Built with DINOv3 (Meta Platforms, Inc.)");
                    ui_kit::caption(ui, "Leaf segmentation powered by Ultralytics YOLO (AGPL-3.0)");
                    ui_kit::caption(ui, "See THIRD_PARTY_LICENSES.md for full terms.");
                }
                SettingsCategory::Pipeline    => self.pipeline.show_settings_panel(ui),
                SettingsCategory::ReconTrain  => self.recon_simple.show_settings_panel(ui),
                SettingsCategory::ReconInfer  => self.recon_infer.show_settings_panel(ui),
                SettingsCategory::LeafSeg     => self.leaf_seg.show_settings_panel(ui),
                SettingsCategory::FieldReview => self.field_review.show_settings_panel(ui),
                SettingsCategory::Eroder      => self.eroder.show_settings_panel(ui),
                SettingsCategory::Train       => self.train.show_settings_panel(ui),
                SettingsCategory::TilePicker  => self.tile_picker.show_settings_panel(ui),
                SettingsCategory::Morphology  => self.morphology.show_settings_panel(ui),
                SettingsCategory::Sorter      => self.sorter.show_settings_panel(ui),
            }
        });
    }

    fn default_row(&mut self, ui: &mut Ui, label: &str, field: DefaultField) {
        let cur = default_path(&self.settings.defaults, field).map(|p| p.display().to_string());
        ui.horizontal(|ui| {
            if ui.button(label).clicked() && self.default_pick.is_none() {
                self.default_pick = Some((field, spawn_default_dialog(field)));
            }
            match &cur {
                Some(s) => { ui.label(RichText::new(s).small().color(ui_kit::MUTED)); }
                None => { ui.label(RichText::new("not set").small().color(ui_kit::MUTED)); }
            }
            if cur.is_some() && ui.small_button("clear").clicked() {
                set_default(&mut self.settings.defaults, field, None);
            }
        });
    }

    // ── Status bar ────────────────────────────────────────────────────────────

    fn show_status_bar(&self, ctx: &Context) {
        egui::TopBottomPanel::bottom("status_bar").exact_height(24.0).show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                match self.active_tab {
                    Tab::Eroder => {
                        let n = self.eroder.loaded_count();
                        ui.label(RichText::new(format!(
                            "Eroder  |  {} image{} loaded", n, if n == 1 { "" } else { "s" }
                        )).small());
                        if self.eroder.is_processing() {
                            ui.separator();
                            ui.label(RichText::new("Processing…").small().color(Color32::YELLOW));
                        }
                    }
                    Tab::Sorter => {
                        let (done, total) = self.sorter.progress();
                        ui.label(RichText::new(format!("Sorter  |  {}/{} sorted", done, total)).small());
                    }
                    Tab::ReconSimple => {
                        let (epoch, total) = self.recon_simple.epoch_progress();
                        ui.label(RichText::new(format!("Recon Train  |  Epoch {}/{}", epoch, total)).small());
                        if self.recon_simple.is_training() {
                            ui.separator();
                            ui.label(RichText::new("Training…").small().color(Color32::YELLOW));
                        }
                    }
                    Tab::ReconInfer => {
                        let (done, total) = self.recon_infer.progress();
                        ui.label(RichText::new(format!("Recon Infer  |  {}/{} images", done, total)).small());
                        if self.recon_infer.is_inferring() {
                            ui.separator();
                            ui.label(RichText::new("Inferring…").small().color(Color32::YELLOW));
                        }
                    }
                    Tab::LeafSeg => { ui.label(RichText::new("Leaf Segmentation").small()); }
                    Tab::FieldReview => { ui.label(RichText::new("Field Review").small()); }
                    Tab::Pipeline => { ui.label(RichText::new("Integrated Pipeline").small()); }
                    Tab::Train => { ui.label(RichText::new("Train Detector").small()); }
                    Tab::Morphology => { ui.label(RichText::new("Morphology").small()); }
                    Tab::TilePicker => { ui.label(RichText::new("Tile Picker").small()); }
                    Tab::Settings => { ui.label(RichText::new("Settings").small()); }
                }
            });
        });
    }
}

fn default_path(d: &AppDefaults, f: DefaultField) -> Option<&PathBuf> {
    match f {
        DefaultField::Yolo => d.yolo.as_ref(),
        DefaultField::Dino => d.dino.as_ref(),
        DefaultField::Bank => d.bank.as_ref(),
        DefaultField::Meta => d.meta.as_ref(),
        DefaultField::Head => d.head.as_ref(),
        DefaultField::Recon => d.recon.as_ref(),
        DefaultField::Healthy => d.healthy.as_ref(),
    }
}

fn set_default(d: &mut AppDefaults, f: DefaultField, v: Option<PathBuf>) {
    match f {
        DefaultField::Yolo => d.yolo = v,
        DefaultField::Dino => d.dino = v,
        DefaultField::Bank => d.bank = v,
        DefaultField::Meta => d.meta = v,
        DefaultField::Head => d.head = v,
        DefaultField::Recon => d.recon = v,
        DefaultField::Healthy => d.healthy = v,
    }
}

/// Bundled `models/` folder: next to the exe (shipped build) or in the working
/// directory (dev). First existing wins.
fn models_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("models")));
    let cwd = std::env::current_dir().ok().map(|d| d.join("models"));
    [exe, cwd].into_iter().flatten().find(|d| d.is_dir())
}

/// Fill each UNSET shared-model default from the bundled `models/` folder so a
/// fresh install works out of the box (user-set defaults always win). Expected
/// names: yolo(.onnx|_weights.safetensors), dino(.onnx|_weights.safetensors),
/// fewshot_head.json, recon/. The BURN backends resolve their `.safetensors` from
/// whichever path is set here (an `.onnx` sibling or the file itself), so a lean
/// weights-only package (no big `.onnx` anchors) auto-loads too.
fn fill_bundled_defaults(d: &mut AppDefaults) {
    let Some(dir) = models_dir() else { return };
    let pick = |name: &str| -> Option<PathBuf> {
        let p = dir.join(name);
        p.exists().then_some(p)
    };
    // `.onnx` first (keeps the ort fallback working on dev machines); fall back to
    // the BURN `.safetensors` so a shipped package needs only the weights.
    if d.yolo.is_none() { d.yolo = pick("yolo.onnx").or_else(|| pick("yolo_weights.safetensors")); }
    if d.dino.is_none() { d.dino = pick("dino.onnx").or_else(|| pick("dino_weights.safetensors")); }
    if d.bank.is_none() { d.bank = pick("coreset_bank.bin"); }
    if d.meta.is_none() { d.meta = pick("detector_meta.json"); }
    if d.head.is_none() { d.head = pick("fewshot_head.json"); }
    if d.recon.is_none() {
        let p = dir.join("recon");
        if p.is_dir() {
            d.recon = Some(p);
        }
    }
}

fn spawn_default_dialog(field: DefaultField) -> mpsc::Receiver<Option<PathBuf>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let res = match field {
            // Stored as the CONTAINING folder (worker.rs looks for `gen.mpk`
            // inside it), but a folder-picker dialog can't show files at all —
            // the user has no way to see/confirm gen.mpk is actually in there.
            // Let them click the .mpk file directly, then take its parent.
            DefaultField::Recon => rfd::FileDialog::new()
                .add_filter("checkpoint", &["mpk"])
                .pick_file()
                .and_then(|p| p.parent().map(|d| d.to_path_buf())),
            DefaultField::Healthy => rfd::FileDialog::new().pick_folder(),
            DefaultField::Yolo | DefaultField::Dino => {
                rfd::FileDialog::new().add_filter("ONNX", &["onnx"]).pick_file()
            }
            DefaultField::Bank => rfd::FileDialog::new().add_filter("bank", &["bin"]).pick_file(),
            DefaultField::Meta | DefaultField::Head => {
                rfd::FileDialog::new().add_filter("json", &["json"]).pick_file()
            }
        };
        let _ = tx.send(res);
    });
    rx
}

/// A full-width navigation row: leading icon + label, highlighted when active.
// ── eframe::App impl ──────────────────────────────────────────────────────────

impl eframe::App for LacunaApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        self.poll_default_pick();
        // Must run regardless of active tab — `show_settings_panel` (Settings
        // → Pipeline) spawns file-pick dialogs too, but is a separate entry
        // point from `PipelineTab::show()`.
        self.pipeline.poll_pick();
        // share global defaults with the tabs that inherit them
        self.pipeline.set_defaults(&self.settings.defaults);
        self.train.set_defaults(&self.settings.defaults);

        let dropped: Vec<egui::DroppedFile> = ctx.input(|i| i.raw.dropped_files.clone());
        if !dropped.is_empty() {
            match self.active_tab {
                Tab::Eroder => self.eroder.handle_dropped_files(&dropped),
                Tab::Sorter => self.sorter.handle_dropped_files(&dropped),
                _ => {}
            }
        }

        self.show_top_bar(ctx);
        self.show_status_bar(ctx);

        // Subtle fade-in when switching screens (opacity cascades into nested panels).
        let now = ctx.input(|i| i.time);
        if self.active_tab != self.prev_tab {
            self.prev_tab = self.active_tab;
            self.tab_changed_at = now;
        }
        let fade = (((now - self.tab_changed_at) as f32) / 0.18).clamp(0.0, 1.0);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.set_opacity(fade);
            match self.active_tab {
                Tab::Eroder => {
                    self.eroder.show(ui, ctx, &mut self.toasts);
                    if self.eroder.check_start_flag() {
                        self.eroder.start_processing_pub();
                    }
                }
                Tab::Sorter      => { self.sorter.show(ui, ctx, &mut self.toasts); }
                Tab::ReconSimple => { self.recon_simple.show(ui, ctx, &mut self.toasts); }
                Tab::ReconInfer  => { self.recon_infer.show(ui, ctx, &mut self.toasts); }
                Tab::LeafSeg     => { self.leaf_seg.show(ui, ctx, &mut self.toasts); }
                Tab::FieldReview => { self.field_review.show(ui, ctx, &mut self.toasts); }
                Tab::Pipeline    => { self.pipeline.show(ui, ctx, &mut self.toasts); }
                Tab::Train       => { self.train.show(ui, ctx, &mut self.toasts); }
                Tab::Morphology  => { self.morphology.show(ui, ctx, &mut self.toasts); }
                Tab::TilePicker  => { self.tile_picker.show(ui, ctx, &mut self.toasts); }
                Tab::Settings    => { self.show_settings(ui, ctx); }
            }
        });

        self.toasts.show(ctx);

        if self.any_busy() || self.default_pick.is_some() || fade < 1.0 {
            ctx.request_repaint();
        }
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        self.eroder.save_settings(&mut self.settings);
        self.sorter.save_settings(&mut self.settings);
        self.recon_simple.save_settings(&mut self.settings);
        self.recon_infer.save_settings(&mut self.settings);
        self.leaf_seg.save_settings(&mut self.settings);
        self.field_review.save_settings(&mut self.settings);
        self.pipeline.save_settings(&mut self.settings);
        self.train.save_settings(&mut self.settings);
        self.tile_picker.save_settings(&mut self.settings);
        self.morphology.save_settings(&mut self.settings);
        eframe::set_value(storage, "app_settings", &self.settings);
        AppSettings::save_to_file(&self.settings);
    }
}
