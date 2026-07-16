//! Morphology tab — EC/MC leaf complexity analysis.
//!
//! Wraps the `leaf_complex_rust` (ShapeComplexity) library: pick a workspace
//! folder of transparent-background leaf PNGs, analyze them (single or batch on a
//! worker thread), and view the EC (edge) / MC (macro-shape) overlays, complexity
//! curves and metrics. The analysis lib is `image` 0.24 internally; we only ever
//! receive raw RGBA bytes + scalars from it, so it never clashes with our 0.25.

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
};

use egui::{
    Align2, Color32, ColorImage, Context, FontId, Rect, RichText, TextureHandle, TextureOptions,
    Ui, Vec2,
};
use egui_plot::{Line, Plot, PlotPoints};

use leaf_complex_rust_lib::config::{Config, ReferencePointChoice};
use leaf_complex_rust_lib::{analyze, MorphMetrics, MorphResult};

use crate::settings::AppSettings;
use crate::tabs::leaf_seg::inference::list_images;
use crate::ui_kit;
use crate::widgets::ToastManager;

struct Overlay {
    w: u32,
    h: u32,
    rgba: Vec<u8>,
}

/// Per-image analysis output kept in RAM (textures are built lazily for the
/// currently-viewed leaf only, so GPU memory stays bounded).
struct MorphData {
    metrics: MorphMetrics,
    ec_data: Vec<[f64; 2]>,
    mc_data: Vec<[f64; 2]>,
    ec_csv: String,
    mc_csv: String,
    original: Overlay,
    ec: Overlay,
    mc: Overlay,
}

impl From<MorphResult> for MorphData {
    fn from(r: MorphResult) -> Self {
        let ov = |o: leaf_complex_rust_lib::MorphOverlay| Overlay { w: o.width, h: o.height, rgba: o.rgba };
        MorphData {
            metrics: r.metrics,
            ec_data: r.ec_data.into_iter().map(|(x, y)| [x, y]).collect(),
            mc_data: r.mc_data.into_iter().map(|(x, y)| [x, y]).collect(),
            ec_csv: r.ec_csv,
            mc_csv: r.mc_csv,
            original: ov(r.original),
            ec: ov(r.ec_overlay),
            mc: ov(r.mc_overlay),
        }
    }
}

struct CurTex {
    idx: usize,
    dims: (u32, u32),
    original: TextureHandle,
    ec: TextureHandle,
    mc: TextureHandle,
}

enum MorphMsg {
    Done(usize, Box<MorphResult>),
    Error(usize, String),
    Progress(usize, usize),
    Finished,
}

pub struct MorphologyTab {
    workspace: Option<PathBuf>,
    output: Option<PathBuf>,
    paths: Vec<PathBuf>,
    selected: Vec<bool>,
    cur: usize,

    config: Config,
    results: HashMap<usize, MorphData>,
    cur_tex: Option<CurTex>,

    show_original: bool,
    show_ec: bool,
    show_mc: bool,

    rx: Option<mpsc::Receiver<MorphMsg>>,
    cancel: Arc<AtomicBool>,
    running: bool,
    done_n: usize,
    total_n: usize,
    status: String,

    ws_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    out_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
}

impl MorphologyTab {
    pub fn new() -> Self {
        Self {
            workspace: None,
            output: None,
            paths: Vec::new(),
            selected: Vec::new(),
            cur: 0,
            config: Config::default(),
            results: HashMap::new(),
            cur_tex: None,
            show_original: true,
            show_ec: true,
            show_mc: true,
            rx: None,
            cancel: Arc::new(AtomicBool::new(false)),
            running: false,
            done_n: 0,
            total_n: 0,
            status: String::new(),
            ws_rx: None,
            out_rx: None,
        }
    }

    pub fn needs_repaint(&self) -> bool {
        self.running || self.ws_rx.is_some() || self.out_rx.is_some()
    }

    pub fn save_settings(&self, s: &mut AppSettings) {
        s.morphology.workspace = self.workspace.clone();
        s.morphology.output = self.output.clone();
    }

    pub fn load_settings(&mut self, s: &AppSettings) {
        self.output = s.morphology.output.clone();
        if let Some(ws) = s.morphology.workspace.clone() {
            self.load_workspace(ws);
        }
    }

    pub fn show(&mut self, ui: &mut Ui, ctx: &Context, toasts: &mut ToastManager) {
        self.poll_dialogs();
        self.poll_worker(toasts);

        egui::SidePanel::left("morph_controls")
            .exact_width(ui_kit::CONTROL_W)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("morph_ctrl")
                    .show(ui, |ui| self.show_controls(ui));
            });
        egui::SidePanel::right("morph_results")
            .default_width(360.0)
            .resizable(true)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("morph_res")
                    .show(ui, |ui| self.show_results(ui));
            });
        egui::CentralPanel::default().show_inside(ui, |ui| self.show_viewer(ui, ctx));
    }

    // ── left controls ───────────────────────────────────────────────────────

    fn show_controls(&mut self, ui: &mut Ui) {
        ui_kit::section_header(ui, "Workspace");
        if ui.button("Workspace folder…").clicked() && self.ws_rx.is_none() {
            self.ws_rx = Some(spawn_folder_dialog());
        }
        ui.label(RichText::new(path_str(&self.workspace)).small().color(Color32::GRAY));
        if !self.paths.is_empty() {
            ui.label(RichText::new(format!("{} leaves", self.paths.len())).small());
        }
        if ui.button("Output folder…").clicked() && self.out_rx.is_none() {
            self.out_rx = Some(spawn_folder_dialog());
        }
        ui.label(RichText::new(path_str(&self.output)).small().color(Color32::GRAY));

        ui_kit::section_header(ui, "Analyze");
        let can = !self.paths.is_empty() && !self.running;
        ui.add_enabled_ui(can, |ui| {
            if ui_kit::primary_button(ui, "Analyze current").clicked() {
                self.start(vec![self.cur]);
            }
        });
        let sel: Vec<usize> = (0..self.paths.len()).filter(|&i| self.selected[i]).collect();
        ui.add_enabled_ui(can && !sel.is_empty(), |ui| {
            if ui.button(format!("Analyze selected ({})", sel.len())).clicked() {
                self.start(sel.clone());
            }
        });
        if self.running {
            if ui.button("Cancel").clicked() {
                self.cancel.store(true, Ordering::Relaxed);
            }
            let frac = if self.total_n > 0 { self.done_n as f32 / self.total_n as f32 } else { 0.0 };
            ui.add(egui::ProgressBar::new(frac).show_percentage());
            ui.horizontal(|ui| ui_kit::busy(ui, &format!("Analyzing {}/{}", self.done_n, self.total_n)));
        }

        ui_kit::section_header(ui, "Overlays");
        ui.checkbox(&mut self.show_original, "Original leaf");
        ui.checkbox(&mut self.show_ec, "EC — edge (pink)");
        ui.checkbox(&mut self.show_mc, "MC — macro-shape (gold)");

        ui_kit::section_header(ui, "Export");
        let have_out = self.output.is_some();
        ui.add_enabled_ui(have_out && !self.results.is_empty(), |ui| {
            if ui.button("Export summary CSV (all analyzed)").clicked() {
                self.export_summary();
            }
        });
        ui.add_enabled_ui(have_out && self.results.contains_key(&self.cur), |ui| {
            if ui.button("Export current curves").clicked() {
                self.export_current();
            }
        });
        if !self.status.is_empty() {
            ui.label(RichText::new(&self.status).small().color(ui_kit::ACCENT));
        }

        ui_kit::section_header(ui, "Leaves");
        ui.horizontal(|ui| {
            if ui.small_button("Select all").clicked() {
                self.selected.iter_mut().for_each(|s| *s = true);
            }
            if ui.small_button("Clear").clicked() {
                self.selected.iter_mut().for_each(|s| *s = false);
            }
        });
        egui::ScrollArea::vertical().max_height(220.0).id_salt("morph_list").show(ui, |ui| {
            for i in 0..self.paths.len() {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.selected[i], "");
                    let done = self.results.contains_key(&i);
                    let name = self
                        .paths
                        .get(i)
                        .and_then(|p| p.file_name())
                        .map(|f| f.to_string_lossy().to_string())
                        .unwrap_or_default();
                    let txt = if done { format!("● {name}") } else { name };
                    let col = if done { ui_kit::ACCENT } else { Color32::GRAY };
                    if ui
                        .selectable_label(self.cur == i, RichText::new(txt).small().color(col))
                        .clicked()
                    {
                        self.cur = i;
                    }
                });
            }
        });
    }

    pub fn show_settings_panel(&mut self, ui: &mut Ui) {
        ui_kit::section_header(ui, "Parameters");
        egui::Grid::new("morph_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
            ui.label("Reference:");
            egui::ComboBox::from_id_salt("morph_ref")
                .selected_text(match self.config.reference_point_choice {
                    ReferencePointChoice::Com => "Center of Mass",
                    ReferencePointChoice::Ep => "Emerge Point",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.config.reference_point_choice,
                        ReferencePointChoice::Com,
                        "Center of Mass",
                    );
                    ui.selectable_value(
                        &mut self.config.reference_point_choice,
                        ReferencePointChoice::Ep,
                        "Emerge Point",
                    );
                });
            ui.end_row();

            ui.label("Resize:");
            let mut resize_on = self.config.resize_dimensions.is_some();
            ui.horizontal(|ui| {
                if ui.checkbox(&mut resize_on, "").changed() {
                    self.config.resize_dimensions = if resize_on { Some([512, 512]) } else { None };
                }
                if let Some(dims) = self.config.resize_dimensions.as_mut() {
                    ui.add(egui::DragValue::new(&mut dims[0]).range(64..=2048).speed(16));
                    dims[1] = dims[0]; // keep square
                    ui.label("px");
                }
            });
            ui.end_row();

            ui.label("Opening density:")
                .on_hover_text("adaptive_opening_max_density — pink-region marking sensitivity.");
            ui.add(egui::Slider::new(&mut self.config.adaptive_opening_max_density, 10.0..=100.0));
            ui.end_row();

            ui.label("MC smoothing:")
                .on_hover_text("thornfiddle_smoothing_strength — macro-shape signal smoothing.");
            ui.add(egui::Slider::new(&mut self.config.thornfiddle_smoothing_strength, 0.0..=10.0));
            ui.end_row();
        });
    }

    // ── right results (metrics + plots) ──────────────────────────────────────

    fn show_results(&self, ui: &mut Ui) {
        ui_kit::section_header(ui, "Metrics");
        let Some(d) = self.results.get(&self.cur) else {
            ui_kit::caption(ui, "Analyze a leaf to see its EC/MC metrics and curves.");
            return;
        };
        let m = &d.metrics;
        egui::Grid::new("morph_metrics").num_columns(3).striped(true).show(ui, |ui| {
            ui.label("");
            ui.label(RichText::new("EC").strong());
            ui.label(RichText::new("MC").strong());
            ui.end_row();
            metric_row(ui, "Length", format!("{:.1}", m.ec_length), format!("{:.1}", m.mc_length));
            metric_row(ui, "Width", format!("{:.1}", m.ec_width), format!("{:.1}", m.mc_width));
            metric_row(ui, "Shape index", format!("{:.3}", m.ec_shape_index), format!("{:.3}", m.mc_shape_index));
            metric_row(ui, "Circularity", format!("{:.3}", m.ec_circularity), format!("{:.3}", m.mc_circularity));
            metric_row(
                ui,
                "Entropy",
                format!("{:.4}", m.ec_approximate_entropy),
                format!("{:.4}", m.mc_spectral_entropy),
            );
            metric_row(ui, "Area", m.ec_area.to_string(), m.mc_area.to_string());
            metric_row(ui, "Outline", m.ec_outline_count.to_string(), m.mc_outline_count.to_string());
        });
        ui.label(
            RichText::new("EC entropy = Approximate Entropy · MC entropy = Spectral Entropy")
                .small()
                .color(Color32::GRAY),
        );

        ui_kit::section_header(ui, "EC — edge pink path");
        Plot::new("morph_ec_plot").height(150.0).show(ui, |p| {
            p.line(Line::new(PlotPoints::from(d.ec_data.clone())).color(Color32::from_rgb(230, 90, 200)));
        });

        ui_kit::section_header(ui, "MC — harmonic path");
        Plot::new("morph_mc_plot").height(150.0).show(ui, |p| {
            p.line(Line::new(PlotPoints::from(d.mc_data.clone())).color(Color32::from_rgb(230, 190, 60)));
        });
    }

    // ── central overlay viewer ───────────────────────────────────────────────

    fn show_viewer(&mut self, ui: &mut Ui, ctx: &Context) {
        let avail = ui.available_rect_before_wrap();
        let painter = ui.painter_at(avail);
        painter.rect_filled(avail, 0.0, Color32::from_gray(24));

        self.ensure_cur_tex(ctx);
        let Some(ct) = &self.cur_tex else {
            let msg = if self.running {
                "Analyzing…"
            } else if self.paths.is_empty() {
                "Open a workspace folder of transparent-background leaf PNGs."
            } else {
                "Select a leaf and click Analyze."
            };
            painter.text(avail.center(), Align2::CENTER_CENTER, msg, FontId::proportional(15.0), Color32::GRAY);
            return;
        };

        let (w, h) = (ct.dims.0 as f32, ct.dims.1 as f32);
        let scale = (avail.width() / w).min(avail.height() / h);
        let sz = Vec2::new(w * scale, h * scale);
        let rect = Rect::from_center_size(avail.center(), sz);

        if self.show_original {
            egui::Image::new((ct.original.id(), sz)).paint_at(ui, rect);
        } else {
            painter.rect_filled(rect, 0.0, Color32::from_gray(36));
        }
        if self.show_ec {
            egui::Image::new((ct.ec.id(), sz)).paint_at(ui, rect);
        }
        if self.show_mc {
            egui::Image::new((ct.mc.id(), sz)).paint_at(ui, rect);
        }
    }

    fn ensure_cur_tex(&mut self, ctx: &Context) {
        if self.cur_tex.as_ref().map(|c| c.idx) == Some(self.cur) {
            return;
        }
        let Some(d) = self.results.get(&self.cur) else {
            self.cur_tex = None;
            return;
        };
        let mk = |o: &Overlay, name: &str| {
            let ci = ColorImage::from_rgba_unmultiplied([o.w as usize, o.h as usize], &o.rgba);
            ctx.load_texture(name, ci, TextureOptions::LINEAR)
        };
        self.cur_tex = Some(CurTex {
            idx: self.cur,
            dims: (d.original.w, d.original.h),
            original: mk(&d.original, "morph_orig"),
            ec: mk(&d.ec, "morph_ec"),
            mc: mk(&d.mc, "morph_mc"),
        });
    }

    // ── actions ──────────────────────────────────────────────────────────────

    fn start(&mut self, indices: Vec<usize>) {
        if self.running {
            return;
        }
        let jobs: Vec<(usize, PathBuf)> = indices
            .iter()
            .filter_map(|&i| self.paths.get(i).map(|p| (i, p.clone())))
            .collect();
        if jobs.is_empty() {
            return;
        }
        self.cancel = Arc::new(AtomicBool::new(false));
        self.running = true;
        self.done_n = 0;
        self.total_n = jobs.len();
        self.status = "Analyzing…".into();

        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);
        let config = self.config.clone();
        let cancel = self.cancel.clone();
        std::thread::spawn(move || {
            let total = jobs.len();
            for (n, (idx, path)) in jobs.into_iter().enumerate() {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                match analyze(&path, &config) {
                    Ok(res) => {
                        let _ = tx.send(MorphMsg::Done(idx, Box::new(res)));
                    }
                    Err(e) => {
                        let _ = tx.send(MorphMsg::Error(idx, e));
                    }
                }
                let _ = tx.send(MorphMsg::Progress(n + 1, total));
            }
            let _ = tx.send(MorphMsg::Finished);
        });
    }

    fn poll_worker(&mut self, toasts: &mut ToastManager) {
        let mut finished = false;
        if let Some(rx) = &self.rx {
            for msg in rx.try_iter().take(16) {
                match msg {
                    MorphMsg::Done(idx, res) => {
                        self.results.insert(idx, (*res).into());
                        if idx == self.cur {
                            self.cur_tex = None; // rebuild texture for the viewed leaf
                        }
                    }
                    MorphMsg::Error(idx, e) => {
                        let name = self.filename(idx);
                        self.status = format!("{name}: {e}");
                        toasts.error(format!("{name}: {e}"));
                    }
                    MorphMsg::Progress(d, t) => {
                        self.done_n = d;
                        self.total_n = t;
                    }
                    MorphMsg::Finished => finished = true,
                }
            }
        }
        if finished {
            self.running = false;
            self.rx = None;
            self.status = format!("Done — {} leaves analyzed", self.results.len());
        }
    }

    fn export_summary(&mut self) {
        let Some(out) = self.output.clone() else { return };
        let mut s = String::new();
        s.push_str(
            "Filename,EC_Length,EC_Width,EC_ShapeIndex,EC_Circularity,EC_ApEn,EC_Area,EC_Outline,\
             MC_Length,MC_Width,MC_ShapeIndex,MC_Circularity,MC_SpEn,MC_Area,MC_Outline\n",
        );
        let mut idxs: Vec<usize> = self.results.keys().copied().collect();
        idxs.sort_unstable();
        for i in idxs {
            let name = self.filename(i);
            let m = &self.results[&i].metrics;
            s.push_str(&format!(
                "{name},{:.4},{:.4},{:.4},{:.4},{:.6},{},{},{:.4},{:.4},{:.4},{:.4},{:.6},{},{}\n",
                m.ec_length, m.ec_width, m.ec_shape_index, m.ec_circularity, m.ec_approximate_entropy,
                m.ec_area, m.ec_outline_count,
                m.mc_length, m.mc_width, m.mc_shape_index, m.mc_circularity, m.mc_spectral_entropy,
                m.mc_area, m.mc_outline_count,
            ));
        }
        let file = out.join("morphology_summary.csv");
        match std::fs::write(&file, s) {
            Ok(()) => self.status = format!("Wrote {}", file.display()),
            Err(e) => self.status = format!("Export failed: {e}"),
        }
    }

    fn export_current(&mut self) {
        let Some(out) = self.output.clone() else { return };
        let Some(d) = self.results.get(&self.cur) else { return };
        let stem = self
            .paths
            .get(self.cur)
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "leaf".into());
        let ec = out.join(format!("{stem}_EC.csv"));
        let mc = out.join(format!("{stem}_MC.csv"));
        let r1 = std::fs::write(&ec, &d.ec_csv);
        let r2 = std::fs::write(&mc, &d.mc_csv);
        self.status = match (r1, r2) {
            (Ok(()), Ok(())) => format!("Wrote {stem}_EC.csv + _MC.csv"),
            _ => "Export failed".into(),
        };
    }

    fn filename(&self, idx: usize) -> String {
        self.paths
            .get(idx)
            .and_then(|p| p.file_name())
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default()
    }

    fn load_workspace(&mut self, dir: PathBuf) {
        let mut paths: Vec<PathBuf> = list_images(&dir)
            .into_iter()
            .filter(|p| {
                let is_png = p
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.eq_ignore_ascii_case("png"))
                    .unwrap_or(false);
                let is_output = p
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.ends_with("_EC") || s.ends_with("_MC"))
                    .unwrap_or(false);
                is_png && !is_output
            })
            .collect();
        paths.sort();
        self.selected = vec![false; paths.len()];
        self.paths = paths;
        self.workspace = Some(dir);
        self.cur = 0;
        self.results.clear();
        self.cur_tex = None;
        self.status.clear();
    }

    fn poll_dialogs(&mut self) {
        if let Some(rx) = &self.ws_rx {
            if let Ok(res) = rx.try_recv() {
                if let Some(p) = res {
                    self.load_workspace(p);
                }
                self.ws_rx = None;
            }
        }
        if let Some(rx) = &self.out_rx {
            if let Ok(res) = rx.try_recv() {
                if let Some(p) = res {
                    self.output = Some(p);
                }
                self.out_rx = None;
            }
        }
    }
}

fn metric_row(ui: &mut Ui, name: &str, ec: String, mc: String) {
    ui.label(RichText::new(name).small());
    ui.label(RichText::new(ec).small());
    ui.label(RichText::new(mc).small());
    ui.end_row();
}

fn spawn_folder_dialog() -> mpsc::Receiver<Option<PathBuf>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(rfd::FileDialog::new().pick_folder());
    });
    rx
}

fn path_str(p: &Option<PathBuf>) -> String {
    p.as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "- not set -".to_string())
}
