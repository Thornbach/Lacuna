pub mod algorithm;

use algorithm::{EroderParams, ResizeSpec, process_image};
use egui::{Color32, Context, DroppedFile, ProgressBar, RichText, ScrollArea, Ui, Vec2};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, AtomicUsize, Ordering}};

use crate::settings::AppSettings;
use crate::widgets::ToastManager;

// ── Processing state shared with background threads ───────────────────────────

struct ProcessingState {
    total:     usize,
    completed: Arc<AtomicUsize>,
    cancelled: Arc<AtomicBool>,
    finished:  Arc<AtomicBool>,   // set by thread after par_iter returns
    log:       Arc<Mutex<Vec<String>>>,
    done:      bool,
}

// ── Preview result ────────────────────────────────────────────────────────────

struct PreviewResult {
    before_pixels: Vec<egui::Color32>,
    before_size:   [usize; 2],
    after_pixels:  Vec<egui::Color32>,
    after_size:    [usize; 2],
}

// ── Thumbnail entry ───────────────────────────────────────────────────────────

struct ThumbEntry {
    path:    PathBuf,
    texture: Option<egui::TextureHandle>,
}

// ── Eroder tab state ──────────────────────────────────────────────────────────

pub struct EroderTab {
    // inputs
    input_folder:         Option<PathBuf>,
    image_paths:          Vec<PathBuf>,
    thumbs:               Vec<ThumbEntry>,
    thumb_rx:             Option<std::sync::mpsc::Receiver<(PathBuf, Vec<egui::Color32>, usize, usize)>>,
    selected_preview_idx: usize,

    // parameters
    damage_levels:        u32,
    max_damage_pct:       f32,
    erosion_prob:         f32,
    smoothing_iterations: u32,
    coastal_enabled:      bool,
    coastal_weight:       f32,
    spots_enabled:        bool,
    spots_weight:         f32,
    snake_enabled:        bool,
    snake_weight:         f32,
    ellipses_enabled:     bool,
    ellipses_weight:      f32,
    apex_enabled:         bool,
    apex_weight:          f32,
    clusters_enabled:     bool,
    clusters_weight:      f32,
    lobe_enabled:         bool,
    lobe_weight:          f32,
    boundary_noise:       bool,
    independent_outputs:  bool,
    seed_enabled:         bool,
    seed_value:           u64,

    // resize
    resize_enabled:       bool,
    resize_use_percent:   bool,
    resize_percent:       f32,
    resize_max_dim:       u32,

    // output
    output_folder: Option<PathBuf>,

    // live preview — background thread based
    preview_before:  Option<egui::TextureHandle>,
    preview_after:   Option<egui::TextureHandle>,
    preview_dirty:   bool,
    preview_rx:      Option<std::sync::mpsc::Receiver<PreviewResult>>,

    // processing
    processing:  Option<ProcessingState>,
    log_entries: Vec<String>,

    // file-dialog channel
    folder_rx: Option<std::sync::mpsc::Receiver<Option<PathBuf>>>,
    output_rx: Option<std::sync::mpsc::Receiver<Option<PathBuf>>>,

    // recent folders
    recent_folders: Vec<PathBuf>,
}

impl EroderTab {
    pub fn new() -> Self {
        Self {
            input_folder:         None,
            image_paths:          Vec::new(),
            thumbs:               Vec::new(),
            thumb_rx:             None,
            selected_preview_idx: 0,
            damage_levels:        10,
            max_damage_pct:       30.0,
            erosion_prob:         0.0005,
            smoothing_iterations: 10,
            coastal_enabled:      true,
            coastal_weight:       0.7,
            spots_enabled:        true,
            spots_weight:         0.2,
            snake_enabled:        false,
            snake_weight:         0.1,
            ellipses_enabled:     false,
            ellipses_weight:      0.5,
            apex_enabled:         false,
            apex_weight:          0.3,
            clusters_enabled:     false,
            clusters_weight:      0.5,
            lobe_enabled:         false,
            lobe_weight:          0.5,
            boundary_noise:       false,
            independent_outputs:  false,
            seed_enabled:         false,
            seed_value:           42,
            resize_enabled:       false,
            resize_use_percent:   true,
            resize_percent:       50.0,
            resize_max_dim:       1024,
            output_folder:        None,
            preview_before:       None,
            preview_after:        None,
            preview_dirty:        false,
            preview_rx:           None,
            processing:           None,
            log_entries:          Vec::new(),
            folder_rx:            None,
            output_rx:            None,
            recent_folders:       Vec::new(),
        }
    }

    // ── public queries ────────────────────────────────────────────────────────

    pub fn loaded_count(&self) -> usize { self.image_paths.len() }

    pub fn is_processing(&self) -> bool {
        self.processing.as_ref().map_or(false, |p| !p.done)
    }

    pub fn handle_dropped_files(&mut self, files: &[DroppedFile]) {
        let paths: Vec<PathBuf> = files.iter()
            .filter_map(|f| f.path.clone())
            .collect();
        if paths.is_empty() { return; }
        if paths.len() == 1 && paths[0].is_dir() {
            self.load_folder(&paths[0].clone());
        } else {
            let images: Vec<PathBuf> = paths.into_iter()
                .filter(|p| is_image(p))
                .collect();
            if !images.is_empty() {
                self.set_image_paths(images, None);
            }
        }
    }

    pub fn save_settings(&self, settings: &mut AppSettings) {
        settings.eroder.damage_levels        = self.damage_levels;
        settings.eroder.max_damage_pct       = self.max_damage_pct;
        settings.eroder.erosion_prob         = self.erosion_prob;
        settings.eroder.smoothing_iterations = self.smoothing_iterations;
        settings.eroder.coastal_enabled      = self.coastal_enabled;
        settings.eroder.coastal_weight       = self.coastal_weight;
        settings.eroder.spots_enabled        = self.spots_enabled;
        settings.eroder.spots_weight         = self.spots_weight;
        settings.eroder.snake_enabled        = self.snake_enabled;
        settings.eroder.snake_weight         = self.snake_weight;
        settings.eroder.ellipses_enabled     = self.ellipses_enabled;
        settings.eroder.ellipses_weight      = self.ellipses_weight;
        settings.eroder.apex_enabled         = self.apex_enabled;
        settings.eroder.apex_weight          = self.apex_weight;
        settings.eroder.clusters_enabled     = self.clusters_enabled;
        settings.eroder.clusters_weight      = self.clusters_weight;
        settings.eroder.lobe_enabled         = self.lobe_enabled;
        settings.eroder.lobe_weight          = self.lobe_weight;
        settings.eroder.boundary_noise       = self.boundary_noise;
        settings.eroder.independent_outputs  = self.independent_outputs;
        settings.eroder.resize_enabled       = self.resize_enabled;
        settings.eroder.resize_use_percent   = self.resize_use_percent;
        settings.eroder.resize_percent       = self.resize_percent;
        settings.eroder.resize_max_dim       = self.resize_max_dim;
        settings.eroder.last_input_folder    = self.input_folder.clone();
        settings.eroder.last_output_folder   = self.output_folder.clone();
        settings.eroder.recent_input_folders = self.recent_folders.clone();
        settings.eroder.seed = if self.seed_enabled { Some(self.seed_value) } else { None };
    }

    pub fn load_settings(&mut self, settings: &AppSettings) {
        self.damage_levels        = settings.eroder.damage_levels;
        self.max_damage_pct       = settings.eroder.max_damage_pct;
        self.erosion_prob         = settings.eroder.erosion_prob;
        self.smoothing_iterations = settings.eroder.smoothing_iterations;
        self.coastal_enabled      = settings.eroder.coastal_enabled;
        self.coastal_weight       = settings.eroder.coastal_weight;
        self.spots_enabled        = settings.eroder.spots_enabled;
        self.spots_weight         = settings.eroder.spots_weight;
        self.snake_enabled        = settings.eroder.snake_enabled;
        self.snake_weight         = settings.eroder.snake_weight;
        self.ellipses_enabled     = settings.eroder.ellipses_enabled;
        self.ellipses_weight      = settings.eroder.ellipses_weight;
        self.apex_enabled         = settings.eroder.apex_enabled;
        self.apex_weight          = settings.eroder.apex_weight;
        self.clusters_enabled     = settings.eroder.clusters_enabled;
        self.clusters_weight      = settings.eroder.clusters_weight;
        self.lobe_enabled         = settings.eroder.lobe_enabled;
        self.lobe_weight          = settings.eroder.lobe_weight;
        self.boundary_noise       = settings.eroder.boundary_noise;
        self.independent_outputs  = settings.eroder.independent_outputs;
        self.resize_enabled       = settings.eroder.resize_enabled;
        self.resize_use_percent   = settings.eroder.resize_use_percent;
        self.resize_percent       = settings.eroder.resize_percent;
        self.resize_max_dim       = settings.eroder.resize_max_dim;
        self.output_folder        = settings.eroder.last_output_folder.clone();
        self.recent_folders       = settings.eroder.recent_input_folders.clone();
        if let Some(s) = settings.eroder.seed {
            self.seed_enabled = true;
            self.seed_value   = s;
        }
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    fn load_folder(&mut self, folder: &PathBuf) {
        let paths = collect_images(folder);
        if !self.recent_folders.contains(folder) {
            self.recent_folders.insert(0, folder.clone());
            self.recent_folders.truncate(10);
        }
        if self.output_folder.is_none() {
            if let Some(parent) = folder.parent() {
                self.output_folder = Some(parent.join("eroded"));
            }
        }
        self.set_image_paths(paths, Some(folder.clone()));
    }

    fn set_image_paths(&mut self, paths: Vec<PathBuf>, folder: Option<PathBuf>) {
        self.input_folder = folder;
        self.image_paths  = paths;
        self.selected_preview_idx = 0;
        self.preview_dirty = true;
        self.preview_rx    = None; // discard any running preview

        let (tx, rx) = std::sync::mpsc::channel();
        self.thumb_rx = Some(rx);
        self.thumbs = self.image_paths.iter().map(|p| ThumbEntry {
            path: p.clone(),
            texture: None,
        }).collect();

        let paths_clone = self.image_paths.clone();
        std::thread::spawn(move || {
            for path in paths_clone {
                if let Ok(img) = image::open(&path) {
                    let thumb = img.thumbnail(96, 96).to_rgba8();
                    let (w, h) = thumb.dimensions();
                    let pixels: Vec<egui::Color32> = thumb.into_raw()
                        .chunks_exact(4)
                        .map(|c| egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]))
                        .collect();
                    let _ = tx.send((path, pixels, w as usize, h as usize));
                }
            }
        });
    }

    fn damage_fractions(&self) -> Vec<f32> {
        (1..=self.damage_levels)
            .map(|i| (self.max_damage_pct / self.damage_levels as f32) * i as f32 / 100.0)
            .collect()
    }

    /// Kick off a background thread to compute the preview.
    /// Replaces any in-flight preview — old results are silently discarded.
    fn start_preview_compute(&mut self) {
        if self.image_paths.is_empty() { return; }

        let path = self.image_paths[
            self.selected_preview_idx.min(self.image_paths.len() - 1)
        ].clone();

        let frac0 = *self.damage_fractions().first().unwrap_or(&0.05);
        let coastal_en      = self.coastal_enabled;
        let coastal_weight  = self.coastal_weight;
        let spots_en        = self.spots_enabled;
        let spots_weight    = self.spots_weight;
        let snake_en        = self.snake_enabled;
        let snake_weight    = self.snake_weight;
        let ellipses_en     = self.ellipses_enabled;
        let ellipses_weight = self.ellipses_weight;
        let apex_en         = self.apex_enabled;
        let apex_weight     = self.apex_weight;
        let clusters_en     = self.clusters_enabled;
        let clusters_weight = self.clusters_weight;
        let lobe_en         = self.lobe_enabled;
        let lobe_weight     = self.lobe_weight;
        let boundary_noise  = self.boundary_noise;
        let erosion_prob    = self.erosion_prob;
        let smoothing       = self.smoothing_iterations;
        let seed_value      = self.seed_value;
        let resize_en       = self.resize_enabled;
        let resize_pct      = self.resize_percent;
        let resize_dim      = self.resize_max_dim;
        let resize_pct_mode = self.resize_use_percent;

        let (tx, rx) = std::sync::mpsc::channel();
        self.preview_rx    = Some(rx);
        self.preview_dirty = false;

        std::thread::spawn(move || {
            let dyn_img = match image::open(&path) {
                Ok(i) => i,
                Err(_) => return,
            };

            // Apply optional resize
            let dyn_img = if resize_en {
                let spec = ResizeSpec {
                    use_percent: resize_pct_mode,
                    percent:     resize_pct,
                    max_dim:     resize_dim,
                };
                spec.apply(dyn_img)
            } else {
                dyn_img
            };

            let rgba = dyn_img.to_rgba8();
            let (w, h) = rgba.dimensions();
            let raw = rgba.into_raw();

            let pixels_before: Vec<egui::Color32> = raw.chunks_exact(4)
                .map(|c| egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]))
                .collect();
            let before_size = [w as usize, h as usize];

            // Compute eroded "after"
            let total_w = {
                let mut t = 0.0f32;
                if coastal_en  { t += coastal_weight;  }
                if spots_en    { t += spots_weight;    }
                if snake_en    { t += snake_weight;    }
                if ellipses_en { t += ellipses_weight; }
                if clusters_en { t += clusters_weight; }
                if lobe_en     { t += lobe_weight;     }
                if t == 0.0 { t = 1.0; }
                t
            };

            let mut mask: Vec<bool> = raw.chunks_exact(4).map(|c| c[3] > 0).collect();
            let mut rng = {
                use rand::SeedableRng;
                rand::rngs::SmallRng::seed_from_u64(seed_value)
            };

            if coastal_en {
                algorithm::erode_coastal(&mut mask, w as usize, h as usize,
                    frac0 * coastal_weight / total_w, erosion_prob, &mut rng);
            }
            if spots_en {
                algorithm::erode_spots(&mut mask, w as usize, h as usize,
                    frac0 * spots_weight / total_w, &mut rng);
            }
            if snake_en {
                algorithm::erode_margin_snake(&mut mask, w as usize, h as usize,
                    frac0 * snake_weight / total_w, &mut rng);
            }
            if ellipses_en {
                algorithm::erode_interior_ellipses(&mut mask, w as usize, h as usize,
                    frac0 * ellipses_weight / total_w, &mut rng);
            }
            if clusters_en {
                algorithm::erode_margin_clusters(&mut mask, w as usize, h as usize,
                    frac0 * clusters_weight / total_w, &mut rng);
            }
            if lobe_en {
                algorithm::erode_lobe(&mut mask, w as usize, h as usize,
                    frac0 * lobe_weight / total_w, &mut rng);
            }
            // Apex: probabilistic supplement — in preview, apply with probability = apex_weight
            if apex_en {
                use rand::Rng;
                if rng.gen::<f32>() < apex_weight {
                    let cut = (frac0 * 1.2_f32).clamp(0.08, 0.50);
                    algorithm::erode_apex(&mut mask, w as usize, h as usize, cut, &mut rng);
                }
            }
            algorithm::smooth_edges(&mut mask, w as usize, h as usize, smoothing);
            if boundary_noise {
                use rand::Rng;
                let ww = w as usize;
                let hh = h as usize;
                for i in 0..ww * hh {
                    if !mask[i] { continue; }
                    let x = i % ww;
                    let y = i / ww;
                    let is_border = (x > 0        && !mask[i - 1])
                        || (x + 1 < ww && !mask[i + 1])
                        || (y > 0        && !mask[i - ww])
                        || (y + 1 < hh && !mask[i + ww]);
                    if is_border && rng.gen::<f32>() < 0.12 {
                        mask[i] = false;
                    }
                }
            }

            let pixels_after: Vec<egui::Color32> = raw.chunks_exact(4)
                .enumerate()
                .map(|(i, c)| {
                    let a = if mask[i] { c[3] } else { 0 };
                    egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], a)
                })
                .collect();

            let _ = tx.send(PreviewResult {
                before_pixels: pixels_before,
                before_size,
                after_pixels: pixels_after,
                after_size: [w as usize, h as usize],
            });
        });
    }

    /// Called each frame: poll preview result channel and schedule compute
    /// when dirty.  Never blocks the UI thread.
    fn poll_preview(&mut self, ctx: &Context) {
        // Check for completed preview result
        if let Some(rx) = &self.preview_rx {
            if let Ok(result) = rx.try_recv() {
                self.preview_before = Some(ctx.load_texture(
                    "prev_before",
                    egui::ColorImage { size: result.before_size, pixels: result.before_pixels },
                    egui::TextureOptions::LINEAR,
                ));
                self.preview_after = Some(ctx.load_texture(
                    "prev_after",
                    egui::ColorImage { size: result.after_size, pixels: result.after_pixels },
                    egui::TextureOptions::LINEAR,
                ));
                self.preview_rx = None;
            }
        }

        // Start compute if dirty and no compute in flight
        if self.preview_dirty && self.preview_rx.is_none() {
            self.start_preview_compute();
        }
    }

    fn start_processing(&mut self) {
        if self.image_paths.is_empty() { return; }
        let output_root = match &self.output_folder {
            Some(p) => p.clone(),
            None    => return,
        };

        let resize = if self.resize_enabled {
            Some(ResizeSpec {
                use_percent: self.resize_use_percent,
                percent:     self.resize_percent,
                max_dim:     self.resize_max_dim,
            })
        } else {
            None
        };

        let params = Arc::new(EroderParams {
            damage_fractions:     self.damage_fractions(),
            erosion_prob:         self.erosion_prob,
            smoothing_iterations: self.smoothing_iterations,
            coastal_weight:       self.coastal_weight,
            spots_weight:         self.spots_weight,
            snake_weight:         self.snake_weight,
            ellipses_weight:      self.ellipses_weight,
            apex_weight:          self.apex_weight,
            clusters_weight:      self.clusters_weight,
            lobe_weight:          self.lobe_weight,
            coastal_enabled:      self.coastal_enabled,
            spots_enabled:        self.spots_enabled,
            snake_enabled:        self.snake_enabled,
            ellipses_enabled:     self.ellipses_enabled,
            apex_enabled:         self.apex_enabled,
            clusters_enabled:     self.clusters_enabled,
            lobe_enabled:         self.lobe_enabled,
            boundary_noise:       self.boundary_noise,
            independent_outputs:  self.independent_outputs,
            seed:                 if self.seed_enabled { Some(self.seed_value) } else { None },
            resize,
        });

        let total     = self.image_paths.len();
        let completed = Arc::new(AtomicUsize::new(0));
        let cancelled = Arc::new(AtomicBool::new(false));
        let finished  = Arc::new(AtomicBool::new(false));
        let log       = Arc::new(Mutex::new(Vec::<String>::new()));

        let paths      = self.image_paths.clone();
        let comp_cl    = Arc::clone(&completed);
        let canc_cl    = Arc::clone(&cancelled);
        let fin_cl     = Arc::clone(&finished);
        let log_cl     = Arc::clone(&log);
        let params_cl  = Arc::clone(&params);

        std::thread::spawn(move || {
            paths.par_iter().for_each(|path| {
                if canc_cl.load(Ordering::Relaxed) { return; }
                let entry = match process_image(path, &params_cl, &output_root, &canc_cl) {
                    Ok(msg)  => format!("✓  {}", msg),
                    Err(e) if e == "cancelled" => return,
                    Err(err) => format!("✗  {} — {}", path.display(), err),
                };
                log_cl.lock().unwrap().push(entry);
                comp_cl.fetch_add(1, Ordering::Relaxed);
            });
            fin_cl.store(true, Ordering::Relaxed);
        });

        self.log_entries.clear();
        self.processing = Some(ProcessingState {
            total,
            completed,
            cancelled,
            finished,
            log,
            done: false,
        });
    }

    // ── main render entry ─────────────────────────────────────────────────────

    pub fn show(&mut self, ui: &mut Ui, ctx: &Context, toasts: &mut ToastManager) {
        // ── poll async channels ────────────────────────────────────────────
        if let Some(rx) = self.folder_rx.take() {
            match rx.try_recv() {
                Ok(Some(folder)) => self.load_folder(&folder),
                Ok(None)         => {},
                Err(_)           => self.folder_rx = Some(rx),
            }
        }
        if let Some(rx) = self.output_rx.take() {
            match rx.try_recv() {
                Ok(Some(folder)) => self.output_folder = Some(folder),
                Ok(None)         => {},
                Err(_)           => self.output_rx = Some(rx),
            }
        }

        // ── poll thumbnail loader ──────────────────────────────────────────
        if let Some(rx) = &self.thumb_rx {
            let mut received = 0;
            while let Ok((path, pixels, w, h)) = rx.try_recv() {
                let texture = ctx.load_texture(
                    path.to_string_lossy(),
                    egui::ColorImage { size: [w, h], pixels },
                    egui::TextureOptions::LINEAR,
                );
                if let Some(entry) = self.thumbs.iter_mut().find(|e| e.path == path) {
                    entry.texture = Some(texture);
                }
                received += 1;
                if received > 8 { break; }
            }
        }

        // ── poll processing state ──────────────────────────────────────────
        if let Some(ps) = &mut self.processing {
            {
                let mut log = ps.log.lock().unwrap();
                self.log_entries.append(&mut *log);
            }
            if ps.finished.load(Ordering::Relaxed) && !ps.done {
                ps.done = true;
                let comp = ps.completed.load(Ordering::Relaxed);
                if ps.cancelled.load(Ordering::Relaxed) {
                    toasts.warning(format!("Cancelled — {}/{} done", comp, ps.total));
                } else {
                    toasts.success(format!("Done — {} image(s) eroded", comp));
                }
            }
        }

        // ── background preview ─────────────────────────────────────────────
        self.poll_preview(ctx);

        // ── layout ────────────────────────────────────────────────────────
        // Split into left/right without a columns() closure so we can call
        // &mut self methods on both sides without borrow conflicts.
        egui::SidePanel::left("eroder_controls")
            .resizable(true)
            .default_width(320.0)
            .show_inside(ui, |ui| {
                self.show_controls(ui, toasts);
            });

        egui::CentralPanel::default()
            .show_inside(ui, |ui| {
                self.show_preview(ui);
            });
    }

    // ── controls panel ────────────────────────────────────────────────────────

    fn show_controls(&mut self, ui: &mut Ui, toasts: &mut ToastManager) {
        ScrollArea::vertical().show(ui, |ui| {
            // ── Input ─────────────────────────────────────────────────────
            ui.group(|ui| {
                ui.label(RichText::new("Input").strong());
                ui.horizontal(|ui| {
                    if ui.button("📁 Open Folder").clicked() {
                        let (tx, rx) = std::sync::mpsc::channel();
                        self.folder_rx = Some(rx);
                        std::thread::spawn(move || {
                            let r = rfd::FileDialog::new()
                                .set_title("Select input folder").pick_folder();
                            let _ = tx.send(r);
                        });
                    }
                    if ui.button("📄 Open Files").clicked() {
                        let (tx, rx) = std::sync::mpsc::channel();
                        self.folder_rx = Some(rx);
                        std::thread::spawn(move || {
                            let picked = rfd::FileDialog::new()
                                .set_title("Select images")
                                .add_filter("Images", &["png", "tiff", "tif"])
                                .pick_files();
                            let first_parent = picked.as_ref()
                                .and_then(|v| v.first())
                                .and_then(|p| p.parent())
                                .map(|p| p.to_path_buf());
                            let _ = tx.send(first_parent);
                        });
                    }
                });

                if !self.recent_folders.is_empty() {
                    egui::ComboBox::from_label("Recent")
                        .selected_text("Open recent…")
                        .show_ui(ui, |ui| {
                            let mut chosen = None;
                            for folder in &self.recent_folders {
                                let label = folder.to_string_lossy();
                                if ui.selectable_label(false, label.as_ref()).clicked() {
                                    chosen = Some(folder.clone());
                                }
                            }
                            if let Some(f) = chosen {
                                self.load_folder(&f);
                            }
                        });
                }

                match &self.input_folder {
                    Some(f) => {
                        ui.label(RichText::new(f.to_string_lossy())
                            .color(Color32::GRAY).small());
                        ui.label(format!("{} images", self.image_paths.len()));
                    }
                    None => {
                        ui.label(RichText::new("No folder loaded  (or drag & drop here)")
                            .color(Color32::GRAY).italics());
                    }
                }

                // Thumbnail strip
                if !self.thumbs.is_empty() {
                    ui.add_space(4.0);
                    ScrollArea::horizontal()
                        .id_salt("thumb_strip")
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                for i in 0..self.thumbs.len() {
                                    let selected = self.selected_preview_idx == i;
                                    let size = Vec2::splat(72.0);
                                    let (rect, resp) = ui.allocate_exact_size(
                                        size + Vec2::splat(4.0),
                                        egui::Sense::click(),
                                    );
                                    if resp.clicked() {
                                        self.selected_preview_idx = i;
                                        self.preview_dirty = true;
                                    }
                                    if selected {
                                        ui.painter().rect_stroke(rect, 2.0,
                                            egui::Stroke::new(2.0, Color32::from_rgb(100, 180, 255)));
                                    }
                                    if let Some(tex) = &self.thumbs[i].texture {
                                        ui.put(rect, egui::Image::new((tex.id(), size))
                                            .fit_to_exact_size(size));
                                    } else {
                                        ui.painter().rect_filled(rect, 2.0, Color32::from_gray(50));
                                        ui.painter().text(rect.center(),
                                            egui::Align2::CENTER_CENTER, "…",
                                            egui::FontId::proportional(12.0), Color32::GRAY);
                                    }
                                }
                            });
                        });
                }
            });

            ui.add_space(6.0);

            // ── Parameters ────────────────────────────────────────────────
            ui.group(|ui| {
                ui.label(RichText::new("Parameters").strong());

                egui::Grid::new("eroder_params")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Damage levels:");
                        let prev_levels = self.damage_levels;
                        ui.add(egui::DragValue::new(&mut self.damage_levels).range(1..=99));
                        ui.end_row();

                        ui.label("Max damage (%):");
                        let prev_max = self.max_damage_pct;
                        ui.add(egui::DragValue::new(&mut self.max_damage_pct)
                            .range(1.0..=99.0).speed(0.5));
                        ui.end_row();

                        if prev_levels != self.damage_levels
                            || (prev_max - self.max_damage_pct).abs() > 0.001
                        {
                            self.preview_dirty = true;
                        }

                        ui.label("Erosion probability:");
                        ui.add(egui::DragValue::new(&mut self.erosion_prob)
                            .range(0.000001..=1.0)
                            .speed(0.00001)
                            .max_decimals(6));
                        ui.end_row();

                        ui.label("Smoothing iterations:");
                        let prev_sm = self.smoothing_iterations;
                        ui.add(egui::DragValue::new(&mut self.smoothing_iterations).range(0..=30));
                        if prev_sm != self.smoothing_iterations { self.preview_dirty = true; }
                        ui.end_row();
                    });

                // Damage level strip
                let fracs: Vec<f32> = (1..=self.damage_levels)
                    .map(|i| (self.max_damage_pct / self.damage_levels as f32) * i as f32)
                    .collect();
                ui.horizontal_wrapped(|ui| {
                    for f in &fracs {
                        ui.label(RichText::new(format!("{:.0}%", f))
                            .small().color(Color32::from_rgb(120, 170, 240)));
                    }
                });
            });

            ui.add_space(6.0);

            // ── Algorithms ────────────────────────────────────────────────
            ui.group(|ui| {
                ui.label(RichText::new("Algorithms").strong());

                let prev_coastal  = (self.coastal_enabled,  self.coastal_weight);
                let prev_spots    = (self.spots_enabled,    self.spots_weight);
                let prev_snake    = (self.snake_enabled,    self.snake_weight);
                let prev_ellipses = (self.ellipses_enabled, self.ellipses_weight);
                let prev_apex     = (self.apex_enabled,     self.apex_weight);
                let prev_clusters = (self.clusters_enabled, self.clusters_weight);
                let prev_lobe     = (self.lobe_enabled,     self.lobe_weight);
                let prev_noise    = self.boundary_noise;

                macro_rules! alg_row {
                    ($ui:expr, $en:expr, $w:expr, $label:expr) => {{
                        $ui.horizontal(|ui| {
                            ui.checkbox(&mut $en, $label);
                            if $en {
                                ui.add(egui::Slider::new(&mut $w, 0.0..=1.0)
                                    .text("weight").show_value(true));
                            }
                        });
                    }};
                }

                alg_row!(ui, self.coastal_enabled,  self.coastal_weight,  "Coastal erosion");
                alg_row!(ui, self.spots_enabled,    self.spots_weight,    "Interior spots (organic blobs)");
                alg_row!(ui, self.snake_enabled,    self.snake_weight,    "Margin snake (tapered notch)");
                if self.snake_enabled {
                    ui.label(RichText::new("  Wide at leaf edge, narrows toward leaf centre")
                        .small().color(Color32::GRAY));
                }
                alg_row!(ui, self.ellipses_enabled, self.ellipses_weight, "Interior ellipses (large holes)");
                if self.ellipses_enabled {
                    ui.label(RichText::new("  Large rotated ellipses anywhere on the leaf")
                        .small().color(Color32::GRAY));
                }
                alg_row!(ui, self.clusters_enabled, self.clusters_weight, "Margin clusters (herbivory bites)");
                if self.clusters_enabled {
                    ui.label(RichText::new("  1–3 Gaussian-weighted bite clusters along the border")
                        .small().color(Color32::GRAY));
                }
                alg_row!(ui, self.lobe_enabled,     self.lobe_weight,     "Lobe removal (whole-lobe bites)");
                if self.lobe_enabled {
                    ui.label(RichText::new("  1–3 disc bites centred on the leaf margin")
                        .small().color(Color32::GRAY));
                }

                ui.separator();

                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.apex_enabled, "Apex supplement");
                    if self.apex_enabled {
                        ui.add(egui::Slider::new(&mut self.apex_weight, 0.0..=1.0)
                            .text("prob").show_value(true))
                            .on_hover_text("Probability that apex/tip removal is applied on top of the main algorithm");
                    }
                });
                if self.apex_enabled {
                    ui.label(RichText::new("  Strips one bbox side (top/bottom/left/right)")
                        .small().color(Color32::GRAY));
                }

                let prev_noise_checked = ui.checkbox(&mut self.boundary_noise,
                    "Boundary noise (12 % border pixel removal)")
                    .on_hover_text("Matches training pipeline: removes ~12 % of border pixels for rough edge texture");

                if prev_coastal  != (self.coastal_enabled,  self.coastal_weight)
                    || prev_spots    != (self.spots_enabled,    self.spots_weight)
                    || prev_snake    != (self.snake_enabled,    self.snake_weight)
                    || prev_ellipses != (self.ellipses_enabled, self.ellipses_weight)
                    || prev_apex     != (self.apex_enabled,     self.apex_weight)
                    || prev_clusters != (self.clusters_enabled, self.clusters_weight)
                    || prev_lobe     != (self.lobe_enabled,     self.lobe_weight)
                    || prev_noise    != self.boundary_noise
                    || prev_noise_checked.changed()
                {
                    self.preview_dirty = true;
                }

                ui.separator();
                ui.checkbox(&mut self.independent_outputs,
                    "Independent outputs (one subfolder per algorithm)");

                ui.separator();
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.seed_enabled, "Fixed seed:");
                    if self.seed_enabled {
                        ui.add(egui::DragValue::new(&mut self.seed_value));
                    } else {
                        ui.label(RichText::new("random").italics().color(Color32::GRAY));
                    }
                });
            });

            ui.add_space(6.0);

            // ── Resize ────────────────────────────────────────────────────
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    let prev_en = self.resize_enabled;
                    ui.checkbox(&mut self.resize_enabled, "Resize images before processing");
                    if prev_en != self.resize_enabled { self.preview_dirty = true; }
                });
                if self.resize_enabled {
                    ui.indent("resize_opts", |ui| {
                        ui.horizontal(|ui| {
                            let prev_mode = self.resize_use_percent;
                            ui.radio_value(&mut self.resize_use_percent, true,  "Percentage");
                            ui.radio_value(&mut self.resize_use_percent, false, "Max dimension (px)");
                            if prev_mode != self.resize_use_percent { self.preview_dirty = true; }
                        });
                        if self.resize_use_percent {
                            let prev = self.resize_percent;
                            ui.add(egui::Slider::new(&mut self.resize_percent, 5.0..=200.0)
                                .suffix(" %").text("Scale"));
                            if (prev - self.resize_percent).abs() > 0.5 { self.preview_dirty = true; }
                        } else {
                            let prev = self.resize_max_dim;
                            ui.add(egui::DragValue::new(&mut self.resize_max_dim)
                                .range(64..=8192).suffix(" px"));
                            ui.label(RichText::new("longest side").small().color(Color32::GRAY));
                            if prev != self.resize_max_dim { self.preview_dirty = true; }
                        }
                    });
                }
            });

            ui.add_space(6.0);

            // ── Output ────────────────────────────────────────────────────
            ui.group(|ui| {
                ui.label(RichText::new("Output").strong());
                ui.horizontal(|ui| {
                    if ui.button("📁 Set output folder").clicked() {
                        let (tx, rx) = std::sync::mpsc::channel();
                        self.output_rx = Some(rx);
                        std::thread::spawn(move || {
                            let r = rfd::FileDialog::new()
                                .set_title("Output folder").pick_folder();
                            let _ = tx.send(r);
                        });
                    }
                    if let Some(p) = &self.output_folder {
                        ui.label(RichText::new(p.to_string_lossy()).small().color(Color32::GRAY));
                    } else {
                        ui.label(RichText::new("Not set").color(Color32::from_rgb(200, 100, 100)));
                    }
                });
            });

            ui.add_space(6.0);

            // ── Action buttons ────────────────────────────────────────────
            let is_running = self.processing.as_ref().map_or(false, |p| !p.done);
            let can_start  = !self.image_paths.is_empty()
                && self.output_folder.is_some()
                && !is_running;

            ui.horizontal(|ui| {
                if ui.add_enabled(can_start,
                    egui::Button::new(RichText::new("▶  Erode Images").strong())).clicked()
                {
                    toasts.info("Starting erosion…");
                    self.start_processing();
                }

                if is_running {
                    if ui.button("⏹ Cancel").clicked() {
                        if let Some(ps) = &self.processing {
                            ps.cancelled.store(true, Ordering::Relaxed);
                        }
                    }
                }
            });

            // ── Progress ──────────────────────────────────────────────────
            if let Some(ps) = &self.processing {
                let comp  = ps.completed.load(Ordering::Relaxed);
                let total = ps.total;
                let prog  = if total > 0 { comp as f32 / total as f32 } else { 0.0 };
                ui.add(ProgressBar::new(prog)
                    .text(format!("{}/{}", comp, total))
                    .animate(!ps.done));
            }

            // ── Log ───────────────────────────────────────────────────────
            if !self.log_entries.is_empty() {
                ui.add_space(4.0);
                ui.label(RichText::new("Log").strong());
                ScrollArea::vertical()
                    .id_salt("eroder_log")
                    .max_height(160.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for entry in &self.log_entries {
                            let color = if entry.starts_with('✓') {
                                Color32::from_rgb(100, 200, 120)
                            } else {
                                Color32::from_rgb(220, 80, 80)
                            };
                            ui.label(RichText::new(entry).monospace().color(color).small());
                        }
                    });
            }
        });
    }

    // ── preview panel ─────────────────────────────────────────────────────────

    fn show_preview(&self, ui: &mut Ui) {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Preview  (first damage level)").strong());
                // Show spinner while preview is being computed
                if self.preview_rx.is_some() {
                    ui.spinner();
                }
            });
            ui.separator();

            let avail  = ui.available_size();
            let half_w = (avail.x / 2.0 - 8.0).max(50.0);
            let half_h = (avail.y - 60.0).max(50.0);
            let preview_size = Vec2::new(half_w, half_h);

            ui.horizontal(|ui| {
                // Before
                ui.vertical(|ui| {
                    ui.label("Original");
                    match &self.preview_before {
                        Some(tex) => {
                            let aspect = tex.size()[0] as f32 / tex.size()[1] as f32;
                            let display_h = (half_w / aspect).min(half_h);
                            let display_size = Vec2::new(half_w, display_h);
                            ui.add(egui::Image::new((tex.id(), display_size))
                                .fit_to_exact_size(display_size));
                        }
                        None => draw_placeholder(ui, preview_size, "No image loaded"),
                    }
                });

                ui.separator();

                // After
                ui.vertical(|ui| {
                    ui.label("After erosion");
                    match &self.preview_after {
                        Some(tex) => {
                            let aspect = tex.size()[0] as f32 / tex.size()[1] as f32;
                            let display_h = (half_w / aspect).min(half_h);
                            let display_size = Vec2::new(half_w, display_h);
                            ui.add(egui::Image::new((tex.id(), display_size))
                                .fit_to_exact_size(display_size));
                        }
                        None => draw_placeholder(ui, preview_size, "No preview"),
                    }
                });
            });

            // Processing overlay
            if let Some(ps) = &self.processing {
                if !ps.done {
                    let comp = ps.completed.load(Ordering::Relaxed);
                    ui.separator();
                    ui.label(format!("Processing: {}/{}", comp, ps.total));
                }
            }
        });
    }
}

// These thin shims are kept so app.rs doesn't need changes for the processing
// flag mechanism — but we've removed the flag trick entirely.
impl EroderTab {
    pub fn check_start_flag(&mut self) -> bool { false }
    pub fn start_processing_pub(&mut self) {}
}

// ── file helpers ──────────────────────────────────────────────────────────────

fn is_image(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(str::to_lowercase).as_deref(),
        Some("png") | Some("tiff") | Some("tif")
    )
}

fn collect_images(folder: &Path) -> Vec<PathBuf> {
    walkdir::WalkDir::new(folder)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| is_image(p))
        .collect()
}

fn draw_placeholder(ui: &mut Ui, size: Vec2, text: &str) {
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    ui.painter().rect_filled(rect, 4.0, Color32::from_gray(40));
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        text,
        egui::FontId::proportional(13.0),
        Color32::GRAY,
    );
}
