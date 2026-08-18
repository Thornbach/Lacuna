// Main Application Structure
use eframe::egui;
use std::sync::{Arc, Mutex};
use std::thread;
use std::path::PathBuf;
use std::fs;

use crate::state::{AppState, AnalysisStatus, SummaryStats};
use crate::ui;
use crate::analysis::AnalysisEngine;
use crate::config_editor::ConfigEditor;
// analyze_image() takes &crate::config::Config (the GUI-local type).
// We store and pass that type everywhere.
use crate::config::Config;

pub struct LeafComplexApp {
    state: Arc<Mutex<AppState>>,
    config: Arc<Mutex<Config>>,
    analysis_engine: AnalysisEngine,
    config_editor: ConfigEditor,
    show_config_editor: bool,
}

impl LeafComplexApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // crate::config::Config is the single Config type used across the GUI.
        let config = Config::load(std::path::Path::new("config.toml"))
            .unwrap_or_else(|_| {
                eprintln!("Could not load config.toml, using defaults");
                Config::default()
            });

        Self {
            state: Arc::new(Mutex::new(AppState::default())),
            config: Arc::new(Mutex::new(config.clone())),
            analysis_engine: AnalysisEngine::new(),
            config_editor: ConfigEditor::new(config),
            show_config_editor: false,
        }
    }

    fn render_menu_bar(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("📁 Open Workspace...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .set_title("Select Workspace Folder")
                        .pick_folder()
                    {
                        let mut state = self.state.lock().unwrap();
                        state.load_workspace(path);
                        drop(state);

                        self.generate_all_thumbnails(ctx);
                        ui.close_menu();
                    }
                }

                ui.separator();

                if ui.button("💾 Export Selected Analysis...").clicked() {
                    self.export_selected_analysis();
                    ui.close_menu();
                }

                ui.separator();

                if ui.button("❌ Exit").clicked() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });

            ui.menu_button("View", |ui| {
                let mut state = self.state.lock().unwrap();

                ui.checkbox(&mut state.show_ec_overlay, "Show EC Overlay");
                ui.checkbox(&mut state.show_mc_overlay, "Show MC Overlay");
                ui.checkbox(&mut state.show_path_overlay, "Show Path");

                ui.separator();

                if ui.button("🔍 Reset Zoom").clicked() {
                    state.reset_view();
                    ui.close_menu();
                }
            });

            ui.menu_button("Analysis", |ui| {
                if ui.button("▶ Analyze Current Image").clicked() {
                    self.analyze_current_image(ctx);
                    ui.close_menu();
                }

                let selected_count = self.state.lock().unwrap().selected_count();
                let batch_label = if selected_count > 0 {
                    format!("⏩ Analyze Selected ({})", selected_count)
                } else {
                    "⏩ Analyze All Images".to_string()
                };

                if ui.button(batch_label).clicked() {
                    self.analyze_all_images(ctx);
                    ui.close_menu();
                }

                ui.separator();

                if ui.button("⚙️ Configuration...").clicked() {
                    self.show_config_editor = true;
                    ui.close_menu();
                }
            });

            ui.menu_button("Help", |ui| {
                if ui.button("ℹ️ About").clicked() {
                    ui.close_menu();
                }
            });
        });
    }

    fn generate_all_thumbnails(&self, ctx: &egui::Context) {
        let state = Arc::clone(&self.state);
        let engine = AnalysisEngine::new();
        let ctx_clone = ctx.clone();

        thread::spawn(move || {
            let images_clone: Vec<_> = {
                let state_guard = state.lock().unwrap();
                state_guard.images.iter().map(|img| img.path.clone()).collect()
            };

            for path in images_clone {
                if let Some(thumbnail) = engine.generate_thumbnail(&path, &ctx_clone) {
                    let mut state_guard = state.lock().unwrap();
                    if let Some(img_info) = state_guard.images.iter_mut().find(|img| img.path == path) {
                        img_info.thumbnail = Some(thumbnail);
                    }
                    drop(state_guard);
                    ctx_clone.request_repaint();
                }
            }
        });
    }

    fn analyze_current_image(&mut self, ctx: &egui::Context) {
        let state = Arc::clone(&self.state);
        let config = Arc::clone(&self.config);
        let ctx = ctx.clone();

        let image_path = {
            let state_guard = state.lock().unwrap();
            match state_guard.current_image() {
                Some(img) => img.path.clone(),
                None => return,
            }
        };

        {
            let mut state_guard = state.lock().unwrap();
            state_guard.analysis_in_progress = true;
            if let Some(idx) = state_guard.current_image_index {
                if let Some(img) = state_guard.images.get_mut(idx) {
                    img.status = AnalysisStatus::Running;
                }
            }
        }

        let engine = AnalysisEngine::new();
        thread::spawn(move || {
            let config_guard = config.lock().unwrap();
            // Deref MutexGuard<Config> with &* to get &crate::config::Config
            let result = engine.analyze_image(&image_path, &*config_guard, &ctx);
            drop(config_guard);

            let mut state_guard = state.lock().unwrap();
            state_guard.analysis_in_progress = false;

            match result {
                Ok(analysis_result) => {
                    state_guard.analysis_results.insert(image_path.clone(), analysis_result);
                    if let Some(idx) = state_guard.current_image_index {
                        if let Some(img) = state_guard.images.get_mut(idx) {
                            img.status = AnalysisStatus::Completed;
                        }
                    }
                }
                Err(e) => {
                    state_guard.last_error = Some(format!("Analysis failed: {}", e));
                    if let Some(idx) = state_guard.current_image_index {
                        if let Some(img) = state_guard.images.get_mut(idx) {
                            img.status = AnalysisStatus::Failed;
                        }
                    }
                }
            }

            ctx.request_repaint();
        });
    }

    /// Work-stealing batch processing — threads pick up new work when finished.
    fn analyze_all_images(&mut self, ctx: &egui::Context) {
        let state = Arc::clone(&self.state);
        let config = Arc::clone(&self.config);
        let ctx = ctx.clone();

        // Get selected images or all if none selected
        let image_paths: Vec<PathBuf> = {
            let state_guard = state.lock().unwrap();
            let selected = state_guard.get_selected_images();
            if selected.is_empty() {
                state_guard.images.iter().map(|img| img.path.clone()).collect()
            } else {
                selected
            }
        };

        if image_paths.is_empty() {
            return;
        }

        println!("Starting batch processing of {} images", image_paths.len());

        {
            let mut state_guard = state.lock().unwrap();
            state_guard.batch_processing = true;
            state_guard.current_batch_index = 0;
            state_guard.total_batch_count = image_paths.len();

            for img in state_guard.images.iter_mut() {
                if image_paths.contains(&img.path) {
                    img.status = AnalysisStatus::Running;
                }
            }
        }

        thread::spawn(move || {
            use std::sync::mpsc;
            use std::sync::atomic::{AtomicUsize, Ordering};

            let num_threads = std::cmp::min(num_cpus::get(), 8);
            println!("Using {} threads for batch processing", num_threads);

            let work_queue = Arc::new(Mutex::new(image_paths.clone()));
            let completed_count = Arc::new(AtomicUsize::new(0));
            let total = image_paths.len();

            let (tx, rx) = mpsc::channel();
            let mut handles = vec![];

            for thread_id in 0..num_threads {
                let queue = Arc::clone(&work_queue);
                let config = Arc::clone(&config);
                let ctx = ctx.clone();
                let tx = tx.clone();
                let completed = Arc::clone(&completed_count);

                let handle = thread::spawn(move || {
                    println!("Thread {} started", thread_id);
                    let engine = AnalysisEngine::new();

                    // Clone Config once and release the lock immediately so all
                    // threads can proceed concurrently.  Holding the lock across
                    // the work loop serialised all threads onto Thread 0.
                    let config_owned = config.lock().unwrap().clone();

                    let mut processed = 0;
                    loop {
                        let path = {
                            let mut queue_guard = queue.lock().unwrap();
                            queue_guard.pop()
                        };

                        match path {
                            Some(path) => {
                                processed += 1;
                                println!("Thread {}: Processing {} - {:?}",
                                    thread_id, processed, path.file_name().unwrap_or_default());

                                let result = engine.analyze_image(&path, &config_owned, &ctx);

                                if tx.send((path.clone(), result)).is_err() {
                                    eprintln!("Thread {}: Failed to send result", thread_id);
                                    break;
                                }

                                completed.fetch_add(1, Ordering::SeqCst);
                            }
                            None => {
                                println!("Thread {} finished ({} images)", thread_id, processed);
                                break;
                            }
                        }
                    }
                });

                handles.push(handle);
            }

            drop(tx);

            let mut completed = 0;
            for (path, result) in rx {
                let mut state_guard = state.lock().unwrap();
                completed += 1;
                state_guard.current_batch_index = completed;

                println!("Result {}/{} for {:?}",
                    completed, total, path.file_name().unwrap_or_default());

                match result {
                    Ok(analysis_result) => {
                        state_guard.analysis_results.insert(path.clone(), analysis_result);
                        if let Some(img) = state_guard.images.iter_mut().find(|i| i.path == path) {
                            img.status = AnalysisStatus::Completed;
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to analyze {:?}: {}", path, e);
                        if let Some(img) = state_guard.images.iter_mut().find(|i| i.path == path) {
                            img.status = AnalysisStatus::Failed;
                        }
                    }
                }

                drop(state_guard);
                ctx.request_repaint();
            }

            for (i, handle) in handles.into_iter().enumerate() {
                if let Err(e) = handle.join() {
                    eprintln!("Thread {} panicked: {:?}", i, e);
                }
            }

            let mut state_guard = state.lock().unwrap();
            state_guard.batch_processing = false;
            println!("Batch processing complete! Processed {} images", completed);
        });
    }

    /// Export selected images (or all analysed) with proper folder structure.
    fn export_selected_analysis(&mut self) {
        // ── 1. Snapshot pre-built CSV strings + summary (no GPU texture clones) ─
        // ec_csv / mc_csv are built once at analysis time and stored as Strings.
        // Cloning them here is just a memcpy of ~20–50 KB per image — very cheap.
        // img.path is also captured so we can compute Subfolder/Path columns.
        let (export_data, workspace_dir): (Vec<(String, std::path::PathBuf, String, String, SummaryStats)>, Option<std::path::PathBuf>) = {
            let state_guard = self.state.lock().unwrap();
            let selected = state_guard.get_selected_images();

            let selected_set: std::collections::HashSet<_> = selected.iter().cloned().collect();
            let filter_by_selection = !selected_set.is_empty();

            let mut seen = std::collections::HashSet::new();
            let data = state_guard.images.iter()
                .filter(|img| {
                    let in_selection = !filter_by_selection || selected_set.contains(&img.path);
                    let has_result = state_guard.analysis_results.contains_key(&img.path);
                    let not_seen = seen.insert(img.path.clone());
                    in_selection && has_result && not_seen
                })
                .filter_map(|img| {
                    state_guard.analysis_results.get(&img.path).map(|r| (
                        img.filename.clone(),
                        img.path.clone(),
                        r.ec_csv.clone(),   // pre-built — just memcpy
                        r.mc_csv.clone(),
                        r.summary.clone(),
                    ))
                })
                .collect();
            (data, state_guard.workspace_dir.clone())
        };

        if export_data.is_empty() {
            self.state.lock().unwrap().last_error =
                Some("No analysed images to export.".to_string());
            return;
        }

        // ── 2. Ask user for destination (still on main thread — requires UI) ──
        let export_base = match rfd::FileDialog::new()
            .set_title("Select Export Location")
            .pick_folder()
        {
            Some(p) => p,
            None => return,
        };

        // ── 3. Hand everything to a background thread — no more UI blocking ───
        let state = Arc::clone(&self.state);

        thread::spawn(move || {
            let results_dir = export_base.join("ShapeComplexityResults");
            let ec_dir = results_dir.join("EC");
            let mc_dir = results_dir.join("MC");
            let summary_dir = results_dir.join("summary");

            for dir in [&ec_dir, &mc_dir, &summary_dir] {
                if let Err(e) = fs::create_dir_all(dir) {
                    state.lock().unwrap().last_error =
                        Some(format!("Failed to create {:?}: {}", dir, e));
                    return;
                }
            }

            println!("Exporting {} images to {:?}", export_data.len(), results_dir);

            let mut all_summaries = Vec::with_capacity(export_data.len());
            let mut failed = 0usize;

            for (filename, img_path, ec_csv, mc_csv, summary) in &export_data {
                // Write pre-built CSV strings — plain byte dump, no formatting work.
                let ec_path = ec_dir.join(format!("{}_EC.csv", filename));
                if let Err(e) = fs::write(&ec_path, ec_csv.as_bytes()) {
                    eprintln!("EC export failed for {}: {}", filename, e);
                    failed += 1;
                    continue;
                }

                let mc_path = mc_dir.join(format!("{}_MC.csv", filename));
                if let Err(e) = fs::write(&mc_path, mc_csv.as_bytes()) {
                    eprintln!("MC export failed for {}: {}", filename, e);
                    failed += 1;
                    continue;
                }

                // Subfolder = immediate parent directory name.
                let subfolder = img_path.parent()
                    .and_then(|p| p.file_name())
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();

                // Relative path: root_name/subfolders/filename.png
                // Falls back to full absolute path when no workspace is loaded.
                let relative_path = match &workspace_dir {
                    Some(root) => {
                        let root_name = root.file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        match img_path.strip_prefix(root) {
                            Ok(within) => format!("{}/{}", root_name, within.display()),
                            Err(_) => img_path.to_string_lossy().into_owned(),
                        }
                    }
                    None => img_path.to_string_lossy().into_owned(),
                };

                all_summaries.push((filename.clone(), subfolder, relative_path, summary.clone()));
            }
            println!("Written {} CSV pairs", all_summaries.len());

            // Write combined summary CSV
            let summary_path = summary_dir.join("summary.csv");
            if let Err(e) = write_summary_data(&summary_path, &all_summaries) {
                state.lock().unwrap().last_error =
                    Some(format!("Failed to write summary.csv: {}", e));
                return;
            }

            let msg = if failed == 0 {
                format!("Export complete! {} images → {:?}", all_summaries.len(), results_dir)
            } else {
                format!("Export done with {} errors. {} images → {:?}",
                    failed, all_summaries.len(), results_dir)
            };
            println!("{}", msg);
        });
    }

    fn write_csv(&self, path: &PathBuf, data: &[(f64, f64)], header: &str) -> Result<(), String> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)
            .map_err(|e| format!("Failed to create file: {}", e))?;

        writeln!(file, "{}", header)
            .map_err(|e| format!("Failed to write header: {}", e))?;

        for (x, y) in data {
            writeln!(file, "{},{}", x, y)
                .map_err(|e| format!("Failed to write data: {}", e))?;
        }

        Ok(())
    }

    fn write_multi_summary_csv(
        &self,
        path: &PathBuf,
        summaries: &[(String, SummaryStats)],
    ) -> Result<(), String> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)
            .map_err(|e| format!("Failed to create file: {}", e))?;

        writeln!(
            file,
            "ID,MC,EC,EC_Length,MC_Length,EC_Width,MC_Width,\
             EC_ShapeIndex,MC_ShapeIndex,EC_Circularity,MC_Circularity,\
             EC_Area,MC_Area,EC_Outline_Count,MC_Outline_Count"
        )
        .map_err(|e| format!("Failed to write header: {}", e))?;

        for (filename, summary) in summaries {
            writeln!(
                file,
                "{},{:.4},{:.4},{:.1},{:.1},{:.1},{:.1},{:.3},{:.3},{:.3},{:.3},{},{},{},{}",
                filename,
                summary.mc_spectral_entropy,
                summary.ec_spectral_entropy,
                summary.ec_length,
                summary.mc_length,
                summary.ec_width,
                summary.mc_width,
                summary.ec_shape_index,
                summary.mc_shape_index,
                summary.ec_circularity,
                summary.mc_circularity,
                summary.ec_area,
                summary.mc_area,
                summary.ec_outline_count,
                summary.mc_outline_count
            )
            .map_err(|e| format!("Failed to write data: {}", e))?;
        }

        Ok(())
    }
}

// ── Free function used by the export background thread ───────────────────────

fn write_summary_data(
    path: &PathBuf,
    summaries: &[(String, String, String, SummaryStats)],
) -> Result<(), String> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)
        .map_err(|e| format!("Failed to create {:?}: {}", path, e))?;
    writeln!(
        file,
        "ID,Subfolder,Path,MC,EC,EC_Length,MC_Length,EC_Width,MC_Width,\
         EC_ShapeIndex,MC_ShapeIndex,EC_Circularity,MC_Circularity,\
         EC_Area,MC_Area,EC_Outline_Count,MC_Outline_Count"
    ).map_err(|e| format!("Failed to write header: {}", e))?;
    for (filename, subfolder, relative_path, s) in summaries {
        writeln!(
            file,
            "{},{},{},{:.4},{:.4},{:.1},{:.1},{:.1},{:.1},{:.3},{:.3},{:.3},{:.3},{},{},{},{}",
            filename,
            subfolder,
            relative_path,
            s.mc_spectral_entropy, s.ec_spectral_entropy,
            s.ec_length, s.mc_length,
            s.ec_width, s.mc_width,
            s.ec_shape_index, s.mc_shape_index,
            s.ec_circularity, s.mc_circularity,
            s.ec_area, s.mc_area,
            s.ec_outline_count, s.mc_outline_count
        ).map_err(|e| format!("Failed to write row: {}", e))?;
    }
    Ok(())
}

impl eframe::App for LeafComplexApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.show_config_editor {
            let mut show = self.show_config_editor;
            let config_updated = self.config_editor.show(ctx, &mut show);
            self.show_config_editor = show;

            if config_updated {
                // ConfigEditor saves to config.toml; reload from there.
                match Config::load(std::path::Path::new("config.toml")) {
                    Ok(new_config) => {
                        *self.config.lock().unwrap() = new_config;
                        println!("✅ Configuration updated and applied!");
                    }
                    Err(e) => {
                        self.state.lock().unwrap().last_error =
                            Some(format!("Failed to reload config after save: {}", e));
                    }
                }
            }
        }

        let mut analyze_clicked = false;
        let mut batch_clicked = false;

        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            self.render_menu_bar(ctx, ui);
        });

        egui::TopBottomPanel::bottom("thumbnails")
            .min_height(150.0)
            .max_height(150.0)
            .show(ctx, |ui| {
                ui::render_thumbnail_strip(ui, &self.state);
            });

        egui::SidePanel::left("image_view")
            .default_width(600.0)
            .min_width(400.0)
            .max_width(800.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui::render_image_view(ui, &self.state, ctx, &mut analyze_clicked, &mut batch_clicked);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui::render_analysis_panel(ui, &self.state, ctx);
        });

        if analyze_clicked {
            self.analyze_current_image(ctx);
        }
        if batch_clicked {
            self.analyze_all_images(ctx);
        }

        if let Some(error) = self.state.lock().unwrap().last_error.clone() {
            egui::Window::new("⚠️ Error")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(&error);
                    if ui.button("OK").clicked() {
                        self.state.lock().unwrap().last_error = None;
                    }
                });
        }

        {
            let state = self.state.lock().unwrap();

            if state.analysis_in_progress {
                egui::Window::new("⏳ Analyzing...")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        ui.spinner();
                        ui.label("Please wait...");
                    });
            }

            if state.batch_processing {
                egui::Window::new("⏳ Batch Processing...")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        ui.spinner();
                        ui.label(format!(
                            "Processing {}/{}...",
                            state.current_batch_index,
                            state.total_batch_count
                        ));
                    });
            }
        }

        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }
}