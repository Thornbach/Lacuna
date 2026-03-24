use egui::{Color32, Context, DroppedFile, Key, RichText, ScrollArea, Ui, Vec2};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::settings::{AppSettings, CategoryDef, SorterSession};
use crate::widgets::ToastManager;

// ── per-image status ──────────────────────────────────────────────────────────

#[derive(Clone, PartialEq)]
enum ImageStatus {
    Unsorted,
    Categorised(String), // category name
    Skipped,
    Flagged,
}

struct SortableImage {
    path:    PathBuf,
    status:  ImageStatus,
    texture: Option<egui::TextureHandle>,  // full-res for main view
    thumb:   Option<egui::TextureHandle>,  // small thumbnail for grid
}

// ── undo history ──────────────────────────────────────────────────────────────

struct UndoEntry {
    idx:      usize,
    previous: ImageStatus,
}

// ── filter mode ───────────────────────────────────────────────────────────────

#[derive(Clone, PartialEq)]
enum FilterMode {
    All,
    Unsorted,
    Categorised(String),
    Flagged,
    Skipped,
}

impl FilterMode {
    fn label(&self) -> &str {
        match self {
            FilterMode::All              => "All",
            FilterMode::Unsorted         => "Unsorted",
            FilterMode::Categorised(n)   => n.as_str(),
            FilterMode::Flagged          => "Flagged",
            FilterMode::Skipped          => "Skipped",
        }
    }
}

// ── category manager modal ────────────────────────────────────────────────────

struct CategoryEditor {
    open:     bool,
    editing:  Vec<CategoryDef>,
    new_name: String,
}

impl CategoryEditor {
    fn new() -> Self {
        Self { open: false, editing: Vec::new(), new_name: String::new() }
    }

    fn open_with(&mut self, cats: &[CategoryDef]) {
        self.open    = true;
        self.editing = cats.to_vec();
        self.new_name.clear();
    }

    /// Returns Some(new_categories) if the user clicked "Apply".
    fn show(&mut self, ctx: &Context) -> Option<Vec<CategoryDef>> {
        if !self.open { return None; }
        let mut result = None;

        egui::Window::new("Category Manager")
            .collapsible(false)
            .resizable(true)
            .min_width(380.0)
            .show(ctx, |ui| {
                ui.label("Drag to reorder  ·  Click colour swatch to change colour");
                ui.separator();

                let mut to_remove: Option<usize> = None;

                for (i, cat) in self.editing.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        // Colour swatch
                        let (r, g, b) = (cat.color[0], cat.color[1], cat.color[2]);
                        let swatch_color = Color32::from_rgb(r, g, b);
                        let (rect, _) = ui.allocate_exact_size(
                            Vec2::splat(18.0), egui::Sense::hover()
                        );
                        ui.painter().rect_filled(rect, 3.0, swatch_color);

                        // Name
                        ui.add(egui::TextEdit::singleline(&mut cat.name).desired_width(120.0));

                        // Hotkey
                        ui.label("key:");
                        let mut key_str = cat.hotkey.clone().unwrap_or_default();
                        let resp = ui.add(
                            egui::TextEdit::singleline(&mut key_str)
                                .desired_width(30.0)
                                .hint_text("?")
                        );
                        // Capture single-character hotkey
                        if resp.has_focus() {
                            ctx.input(|inp| {
                                for ev in &inp.events {
                                    if let egui::Event::Text(s) = ev {
                                        if !s.is_empty() {
                                            key_str = s.chars().next().unwrap().to_string();
                                        }
                                    }
                                }
                            });
                        }
                        cat.hotkey = if key_str.is_empty() { None } else { Some(key_str) };

                        // Output subfolder override
                        ui.label("→");
                        let mut sub = cat.output_subfolder.clone().unwrap_or_default();
                        ui.add(egui::TextEdit::singleline(&mut sub).desired_width(90.0).hint_text("(same as name)"));
                        cat.output_subfolder = if sub.is_empty() { None } else { Some(sub) };

                        if ui.button("🗑").clicked() {
                            to_remove = Some(i);
                        }
                    });
                }

                if let Some(i) = to_remove { self.editing.remove(i); }

                ui.separator();
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut self.new_name)
                        .desired_width(140.0).hint_text("New category name…"));
                    if ui.button("+ Add").clicked() && !self.new_name.trim().is_empty() {
                        let name = self.new_name.trim().to_string();
                        self.editing.push(CategoryDef {
                            name: name.clone(),
                            hotkey: None,
                            color: random_color(&name),
                            output_subfolder: None,
                        });
                        self.new_name.clear();
                    }
                });

                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Apply").clicked() {
                        result = Some(self.editing.clone());
                        self.open = false;
                    }
                    if ui.button("Cancel").clicked() {
                        self.open = false;
                    }
                });
            });

        result
    }
}

fn random_color(seed: &str) -> [u8; 3] {
    let h: u64 = seed.bytes().fold(5381u64, |acc, b| acc.wrapping_mul(33).wrapping_add(b as u64));
    [
        100 + (h & 0xFF) as u8 / 2,
        100 + ((h >> 8) & 0xFF) as u8 / 2,
        100 + ((h >> 16) & 0xFF) as u8 / 2,
    ]
}

// ── Sorter tab state ──────────────────────────────────────────────────────────

pub struct SorterTab {
    // data
    source_folder: Option<PathBuf>,
    output_root:   Option<PathBuf>,
    images:        Vec<SortableImage>,
    current_idx:   usize,

    // categories
    categories: Vec<CategoryDef>,

    // undo
    history: Vec<UndoEntry>,

    // filter
    filter: FilterMode,

    // zoom / pan for main view
    zoom: f32,
    pan:  Vec2,

    // thumbnail loader channel
    thumb_rx: Option<std::sync::mpsc::Receiver<(PathBuf, Vec<egui::Color32>, usize, usize)>>,
    // full-res loader channel
    full_rx:  Option<std::sync::mpsc::Receiver<(PathBuf, Vec<egui::Color32>, usize, usize)>>,

    // async folder dialog channel
    folder_rx: Option<std::sync::mpsc::Receiver<Option<PathBuf>>>,
    output_rx: Option<std::sync::mpsc::Receiver<Option<PathBuf>>>,

    // settings copy
    copy_not_move:  bool,
    thumbnail_size: f32,

    // session
    session: Option<SorterSession>,

    // UI state
    category_editor: CategoryEditor,
    search_query:    String,
    recent_folders:  Vec<PathBuf>,

    // flash feedback (category flash colour + timer)
    flash_color:  Option<Color32>,
    flash_frames: u8,

    // reset confirmation dialog
    confirm_reset: bool,

    // background colour picker
    bg_color:       [u8; 3],
    show_bg_picker: bool,
}

impl SorterTab {
    pub fn new() -> Self {
        Self {
            source_folder:  None,
            output_root:    None,
            images:         Vec::new(),
            current_idx:    0,
            categories:     vec![
                CategoryDef::new("Healthy", "1", [80, 180, 100]),
                CategoryDef::new("Damaged", "0", [200, 80, 80]),
            ],
            history:        Vec::new(),
            filter:         FilterMode::All,
            zoom:           1.0,
            pan:            Vec2::ZERO,
            thumb_rx:       None,
            full_rx:        None,
            folder_rx:      None,
            output_rx:      None,
            copy_not_move:  true,
            thumbnail_size: 96.0,
            session:        None,
            category_editor: CategoryEditor::new(),
            search_query:   String::new(),
            recent_folders: Vec::new(),
            flash_color:    None,
            flash_frames:   0,
            confirm_reset:  false,
            bg_color:       [40, 40, 40],
            show_bg_picker: false,
        }
    }

    // ── public queries ────────────────────────────────────────────────────────

    pub fn progress(&self) -> (usize, usize) {
        let done = self.images.iter()
            .filter(|i| i.status != ImageStatus::Unsorted)
            .count();
        (done, self.images.len())
    }

    pub fn needs_repaint(&self) -> bool {
        self.flash_frames > 0 ||
        self.thumb_rx.is_some() ||
        self.full_rx.is_some()
    }

    pub fn handle_dropped_files(&mut self, files: &[DroppedFile]) {
        let paths: Vec<PathBuf> = files.iter()
            .filter_map(|f| f.path.clone())
            .collect();
        if let Some(folder) = paths.into_iter().find(|p| p.is_dir()) {
            self.load_source_folder(&folder);
        }
    }

    pub fn save_settings(&self, settings: &mut AppSettings) {
        settings.sorter.categories        = self.categories.clone();
        settings.sorter.last_source_folder = self.source_folder.clone();
        settings.sorter.last_output_folder = self.output_root.clone();
        settings.sorter.recent_source_folders = self.recent_folders.clone();
        settings.sorter.copy_not_move     = self.copy_not_move;
        settings.sorter.thumbnail_size    = self.thumbnail_size;
        settings.sorter.bg_color          = self.bg_color;
        if let Some(sess) = &self.session { sess.save(); }
    }

    pub fn load_settings(&mut self, s: &AppSettings) {
        self.categories        = s.sorter.categories.clone();
        self.output_root       = s.sorter.last_output_folder.clone();
        self.recent_folders    = s.sorter.recent_source_folders.clone();
        self.copy_not_move     = s.sorter.copy_not_move;
        self.thumbnail_size    = s.sorter.thumbnail_size;
        self.bg_color          = s.sorter.bg_color;
    }

    // ── folder loading ────────────────────────────────────────────────────────

    fn load_source_folder(&mut self, folder: &PathBuf) {
        // Auto-suggest output folder FIRST so we can exclude it from the scan.
        if self.output_root.is_none() {
            if let Some(parent) = folder.parent() {
                self.output_root = Some(parent.join("sorted"));
            }
        }

        // Collect images, explicitly skipping the output folder to avoid
        // counting already-copied files as new source images.
        let exclude = self.output_root.clone();
        let paths: Vec<PathBuf> = walkdir::WalkDir::new(folder)
            .into_iter()
            .filter_entry(|e| {
                // Prune entire output subtree from the walk
                if let Some(ref ex) = exclude {
                    if e.path().starts_with(ex.as_path()) { return false; }
                }
                true
            })
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.into_path())
            .filter(|p| is_image(p))
            .collect();

        if paths.is_empty() { return; }

        // Remember recent
        if !self.recent_folders.contains(folder) {
            self.recent_folders.insert(0, folder.clone());
            self.recent_folders.truncate(10);
        }

        // ── Load state ──────────────────────────────────────────────────────
        // Primary: CSV log in the output folder (full-path keys → no collision).
        // Fallback: AppData session (filename keys, used only when no CSV exists).
        let csv_state = self.output_root.as_ref()
            .map(|out| load_csv_log(&out.join("_sorting_log.csv")))
            .unwrap_or_default();

        let session = SorterSession::load(folder).unwrap_or_else(|| SorterSession {
            source_folder: folder.clone(),
            categorised: HashMap::new(),
            skipped: Vec::new(),
        });

        // Build image list — use CSV (full path) first, then session (filename)
        self.images = paths.iter().map(|p| {
            let path_key = p.to_string_lossy().to_string();
            let fname    = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();

            let status = if let Some(s) = csv_state.get(&path_key) {
                // Full-path match from CSV — authoritative
                match s.as_str() {
                    "Unsorted" | "" => ImageStatus::Unsorted,
                    "Skipped"       => ImageStatus::Skipped,
                    "Flagged"       => ImageStatus::Flagged,
                    cat             => ImageStatus::Categorised(cat.to_string()),
                }
            } else if csv_state.is_empty() {
                // No CSV exists yet (pre-CSV sessions) — fall back to filename-keyed session
                // ONLY when the entire CSV is absent, so same-named files in other subfolders
                // are never incorrectly matched.
                if let Some(cat) = session.categorised.get(&fname) {
                    ImageStatus::Categorised(cat.clone())
                } else if session.skipped.contains(&fname) {
                    ImageStatus::Skipped
                } else {
                    ImageStatus::Unsorted
                }
            } else {
                // CSV exists but this path is not in it → genuinely unsorted
                ImageStatus::Unsorted
            };

            SortableImage { path: p.clone(), status, texture: None, thumb: None }
        }).collect();

        self.source_folder = Some(folder.clone());
        self.session       = Some(session);
        self.history.clear();

        // Jump to first unsorted
        self.current_idx = self.images.iter().position(|i| i.status == ImageStatus::Unsorted)
            .unwrap_or(0);

        self.start_thumb_loader();
        self.request_full_load(self.current_idx);
    }

    fn start_thumb_loader(&mut self) {
        let (tx, rx) = std::sync::mpsc::channel();
        self.thumb_rx = Some(rx);
        let paths: Vec<PathBuf> = self.images.iter().map(|i| i.path.clone()).collect();
        let size = self.thumbnail_size as u32;
        std::thread::spawn(move || {
            for path in paths {
                if let Ok(img) = image::open(&path) {
                    let thumb = img.thumbnail(size, size).to_rgba8();
                    let (w, h) = thumb.dimensions();
                    let pixels: Vec<egui::Color32> = thumb.into_raw().chunks_exact(4)
                        .map(|c| egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]))
                        .collect();
                    let _ = tx.send((path, pixels, w as usize, h as usize));
                }
            }
        });
    }

    fn request_full_load(&mut self, idx: usize) {
        if idx >= self.images.len() { return; }
        if self.images[idx].texture.is_some() { return; } // already loaded

        let (tx, rx) = std::sync::mpsc::channel();
        self.full_rx = Some(rx);
        let path = self.images[idx].path.clone();
        std::thread::spawn(move || {
            if let Ok(img) = image::open(&path) {
                let rgba = img.to_rgba8();
                let (w, h) = rgba.dimensions();
                let pixels: Vec<egui::Color32> = rgba.into_raw().chunks_exact(4)
                    .map(|c| egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]))
                    .collect();
                let _ = tx.send((path, pixels, w as usize, h as usize));
            }
        });
    }

    // ── categorisation ────────────────────────────────────────────────────────

    fn categorise(&mut self, cat_idx: usize, toasts: &mut ToastManager) {
        if self.images.is_empty() { return; }
        let idx = self.current_idx;
        if idx >= self.images.len() { return; }

        let cat = match self.categories.get(cat_idx) {
            Some(c) => c.clone(),
            None    => return,
        };

        let prev = self.images[idx].status.clone();
        self.images[idx].status = ImageStatus::Categorised(cat.name.clone());
        self.history.push(UndoEntry { idx, previous: prev });

        // Copy / move file
        if let (Some(output), Some(sess)) = (&self.output_root, &mut self.session) {
            let sub = cat.output_subfolder.clone().unwrap_or_else(|| cat.name.clone());

            // Preserve relative subfolder structure so files from different
            // source sub-directories with identical names don't overwrite each other.
            let rel_parent = self.source_folder.as_ref()
                .and_then(|src| self.images[idx].path.strip_prefix(src).ok())
                .and_then(|rel| rel.parent())
                .filter(|p| !p.as_os_str().is_empty());
            let dest_dir = match rel_parent {
                Some(parent) => output.join(&sub).join(parent),
                None         => output.join(&sub),
            };

            let _ = std::fs::create_dir_all(&dest_dir);
            let filename = self.images[idx].path
                .file_name().unwrap().to_string_lossy().to_string();
            let dest = dest_dir.join(&filename);

            let result = if self.copy_not_move {
                std::fs::copy(&self.images[idx].path, &dest).map(|_| ())
            } else {
                std::fs::rename(&self.images[idx].path, &dest)
            };
            if let Err(e) = result {
                toasts.error(format!("File error: {}", e));
            }

            // Update session
            sess.categorised.insert(filename, cat.name.clone());
        }

        // Write to CSV log (full path → no filename-collision issues)
        if let Some(csv_path) = self.output_root.as_ref().map(|o| o.join("_sorting_log.csv")) {
            append_csv_entry(&csv_path, &self.images[idx].path, &cat.name);
        }

        // Flash feedback
        self.flash_color  = Some(Color32::from_rgb(cat.color[0], cat.color[1], cat.color[2]));
        self.flash_frames = 12;

        self.advance();
    }

    fn skip(&mut self) {
        if self.images.is_empty() { return; }
        let idx = self.current_idx;
        let prev = self.images[idx].status.clone();
        self.images[idx].status = ImageStatus::Skipped;
        self.history.push(UndoEntry { idx, previous: prev });

        if let Some(sess) = &mut self.session {
            let fname = self.images[idx].path
                .file_name().unwrap().to_string_lossy().to_string();
            if !sess.skipped.contains(&fname) { sess.skipped.push(fname); }
        }

        if let Some(csv_path) = self.output_root.as_ref().map(|o| o.join("_sorting_log.csv")) {
            append_csv_entry(&csv_path, &self.images[idx].path, "Skipped");
        }

        self.advance();
    }

    fn flag(&mut self) {
        if self.images.is_empty() { return; }
        let idx = self.current_idx;
        let prev = self.images[idx].status.clone();
        self.images[idx].status = ImageStatus::Flagged;
        self.history.push(UndoEntry { idx, previous: prev });

        if let Some(csv_path) = self.output_root.as_ref().map(|o| o.join("_sorting_log.csv")) {
            append_csv_entry(&csv_path, &self.images[idx].path, "Flagged");
        }

        self.advance();
    }

    fn undo(&mut self) {
        if let Some(entry) = self.history.pop() {
            self.images[entry.idx].status = entry.previous.clone();
            // Record the reverted status in the CSV so "last entry wins" on reload
            let status_str = match &entry.previous {
                ImageStatus::Unsorted        => "Unsorted".to_string(),
                ImageStatus::Skipped         => "Skipped".to_string(),
                ImageStatus::Flagged         => "Flagged".to_string(),
                ImageStatus::Categorised(c)  => c.clone(),
            };
            if let Some(csv_path) = self.output_root.as_ref().map(|o| o.join("_sorting_log.csv")) {
                append_csv_entry(&csv_path, &self.images[entry.idx].path, &status_str);
            }
            self.navigate_to(entry.idx);
        }
    }

    fn advance(&mut self) {
        let n = self.images.len();
        if n == 0 { return; }
        // Find next unsorted image, wrapping around
        let start = (self.current_idx + 1) % n;
        for delta in 0..n {
            let i = (start + delta) % n;
            if matches!(&self.images[i].status, ImageStatus::Unsorted) {
                self.navigate_to(i);
                return;
            }
        }
        // All done — stay on current
        self.navigate_to(self.current_idx);
    }

    fn navigate_to(&mut self, idx: usize) {
        if idx == self.current_idx { return; }
        // Drop the previous full-res texture immediately so GPU/system RAM is freed.
        // Thumbnails stay loaded (they are small); full-res is reloaded on demand.
        if self.current_idx < self.images.len() {
            self.images[self.current_idx].texture = None;
        }
        self.current_idx = idx;
        self.zoom = 1.0;
        self.pan  = Vec2::ZERO;
        self.request_full_load(idx);
    }

    fn navigate_prev(&mut self) {
        if self.images.is_empty() { return; }
        let idx = if self.current_idx == 0 { self.images.len() - 1 } else { self.current_idx - 1 };
        self.navigate_to(idx);
    }

    fn navigate_next(&mut self) {
        if self.images.is_empty() { return; }
        let idx = (self.current_idx + 1) % self.images.len();
        self.navigate_to(idx);
    }

    fn jump_to_first_unsorted(&mut self) {
        if let Some(idx) = self.images.iter().position(|i| i.status == ImageStatus::Unsorted) {
            self.navigate_to(idx);
        }
    }

    fn reset_sorting(&mut self, toasts: &mut ToastManager) {
        for img in &mut self.images {
            img.status = ImageStatus::Unsorted;
        }
        self.history.clear();

        // Clear and re-save the session so the reset survives a restart
        if let Some(sess) = &mut self.session {
            sess.categorised.clear();
            sess.skipped.clear();
            sess.save();
        }

        // Jump back to first image
        self.current_idx = 0;
        self.zoom = 1.0;
        self.pan  = Vec2::ZERO;
        if !self.images.is_empty() {
            self.request_full_load(0);
        }

        // Archive the CSV log so the reset survives a restart
        if let Some(csv_path) = self.output_root.as_ref().map(|o| o.join("_sorting_log.csv")) {
            if csv_path.exists() {
                let backup = csv_path.with_file_name("_sorting_log_backup.csv");
                let _ = std::fs::rename(&csv_path, backup);
            }
        }

        toasts.info("Sorting reset — all images marked as unsorted.");
    }

    // ── visible indices given current filter / search ─────────────────────────

    fn filtered_indices(&self) -> Vec<usize> {
        let q = self.search_query.to_lowercase();
        self.images.iter().enumerate()
            .filter(|(_, img)| {
                // search filter
                if !q.is_empty() {
                    let name = img.path.file_name()
                        .and_then(|n| n.to_str()).unwrap_or("").to_lowercase();
                    if !name.contains(&q) { return false; }
                }
                // status filter
                match &self.filter {
                    FilterMode::All              => true,
                    FilterMode::Unsorted         => img.status == ImageStatus::Unsorted,
                    FilterMode::Categorised(n)   => img.status == ImageStatus::Categorised(n.clone()),
                    FilterMode::Flagged          => img.status == ImageStatus::Flagged,
                    FilterMode::Skipped          => img.status == ImageStatus::Skipped,
                }
            })
            .map(|(i, _)| i)
            .collect()
    }

    // ── main render entry ─────────────────────────────────────────────────────

    pub fn show(&mut self, ui: &mut Ui, ctx: &Context, toasts: &mut ToastManager) {
        // ── poll folder dialog ─────────────────────────────────────────────
        if let Some(rx) = self.folder_rx.take() {
            if let Ok(result) = rx.try_recv() {
                if let Some(folder) = result {
                    self.load_source_folder(&folder);
                }
            } else { self.folder_rx = Some(rx); }
        }
        if let Some(rx) = self.output_rx.take() {
            if let Ok(result) = rx.try_recv() {
                if let Some(folder) = result { self.output_root = Some(folder); }
            } else { self.output_rx = Some(rx); }
        }

        // ── poll thumbnail loader ──────────────────────────────────────────
        let mut thumb_done = false;
        if let Some(rx) = &self.thumb_rx {
            let mut n = 0;
            loop {
                match rx.try_recv() {
                    Ok((path, pixels, w, h)) => {
                        let tex = ctx.load_texture(
                            format!("thumb_{}", path.display()),
                            egui::ColorImage { size: [w, h], pixels },
                            egui::TextureOptions::LINEAR,
                        );
                        if let Some(img) = self.images.iter_mut().find(|i| i.path == path) {
                            img.thumb = Some(tex);
                        }
                        n += 1; if n > 8 { break; }
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        thumb_done = true;
                        break;
                    }
                    Err(_) => break,
                }
            }
        }
        if thumb_done { self.thumb_rx = None; }

        // ── poll full-res loader ───────────────────────────────────────────
        if let Some(rx) = self.full_rx.take() {
            match rx.try_recv() {
                Ok((path, pixels, w, h)) => {
                    let tex = ctx.load_texture(
                        format!("full_{}", path.display()),
                        egui::ColorImage { size: [w, h], pixels },
                        egui::TextureOptions::LINEAR,
                    );
                    if let Some(img) = self.images.iter_mut().find(|i| i.path == path) {
                        img.texture = Some(tex);
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    self.full_rx = Some(rx); // still loading
                }
                Err(_) => {} // sender dropped
            }
        }

        // ── flash tick ────────────────────────────────────────────────────
        if self.flash_frames > 0 {
            self.flash_frames -= 1;
            ctx.request_repaint();
        } else {
            self.flash_color = None;
        }

        // ── keyboard hotkeys ───────────────────────────────────────────────
        let mut do_undo         = false;
        let mut do_skip         = false;
        let mut do_flag         = false;
        let mut do_prev         = false;
        let mut do_next         = false;
        let mut do_jump_unsorted = false;
        let mut do_cat: Option<usize> = None;

        ctx.input(|inp| {
            // Undo
            if inp.key_pressed(Key::Z) && inp.modifiers.ctrl {
                do_undo = true;
            }
            // Skip (right arrow)
            if inp.key_pressed(Key::ArrowRight) { do_skip = true; }
            // Browse without sorting (left arrow)
            if inp.key_pressed(Key::ArrowLeft)  { do_prev = true; }
            // Next without sorting
            if inp.key_pressed(Key::ArrowDown)  { do_next = true; }
            // Flag
            if inp.key_pressed(Key::F)          { do_flag = true; }
            // Jump to first unsorted
            if inp.key_pressed(Key::Tab)        { do_jump_unsorted = true; }

            // Category hotkeys
            for (ci, cat) in self.categories.iter().enumerate() {
                if let Some(key_str) = &cat.hotkey {
                    let matched = inp.events.iter().any(|ev| {
                        matches!(ev, egui::Event::Text(s) if s == key_str.as_str())
                    });
                    if matched { do_cat = Some(ci); }
                }
            }

            // Zoom via scroll wheel is handled in show_main_image
        });

        if do_undo          { self.undo(); }
        if do_skip          { self.skip(); }
        if do_flag          { self.flag(); }
        if do_prev          { self.navigate_prev(); }
        if do_next          { self.navigate_next(); }
        if do_jump_unsorted { self.jump_to_first_unsorted(); }
        if let Some(ci) = do_cat { self.categorise(ci, toasts); }

        // ── category editor modal ──────────────────────────────────────────
        if let Some(new_cats) = self.category_editor.show(ctx) {
            self.categories = new_cats;
        }

        // ── reset confirmation modal ───────────────────────────────────────
        if self.confirm_reset {
            let mut do_reset  = false;
            let mut do_cancel = false;

            egui::Window::new("Confirm Reset")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    let (done, total) = self.progress();
                    ui.label(RichText::new("Reset all sorting progress?").strong());
                    ui.add_space(4.0);
                    ui.label(format!(
                        "{} of {} images have been sorted.\n\
                         All statuses will be set back to Unsorted.\n\
                         Files already copied / moved are NOT affected.",
                        done, total
                    ));
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.add(egui::Button::new(
                            RichText::new("Yes, reset everything").color(Color32::WHITE).strong()
                        ).fill(Color32::from_rgb(180, 50, 50))).clicked() {
                            do_reset = true;
                        }
                        if ui.button("Cancel").clicked() {
                            do_cancel = true;
                        }
                    });
                });

            if do_reset {
                self.confirm_reset = false;
                self.reset_sorting(toasts);
            } else if do_cancel {
                self.confirm_reset = false;
            }
        }

        // ── background colour picker window ────────────────────────────────
        if self.show_bg_picker {
            // (label, RGB) — chosen to contrast with green leaves and be easy on the eyes
            const PRESETS: &[(&str, [u8; 3])] = &[
                ("Charcoal (default)",  [40,  40,  40]),
                ("Dark gray",           [70,  70,  70]),
                ("Mid gray",            [128, 128, 128]),
                ("Off-white",           [225, 225, 220]),
                ("Pure white",          [255, 255, 255]),
                ("Black",               [5,   5,   5]),
                ("Dark teal",           [15,  30,  30]),
                ("Slate blue",          [25,  28,  38]),
            ];

            let current = self.bg_color;
            let mut new_color = self.bg_color;
            let mut close = false;
            let mut set_preset: Option<[u8; 3]> = None;

            egui::Window::new("Background Color")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("Presets:");
                    ui.horizontal_wrapped(|ui| {
                        for &(label, c) in PRESETS {
                            let color = Color32::from_rgb(c[0], c[1], c[2]);
                            let is_active = current == c;

                            let (rect, resp) = ui.allocate_exact_size(
                                Vec2::new(40.0, 26.0),
                                egui::Sense::click(),
                            );
                            let painter = ui.painter();
                            painter.rect_filled(rect, 3.0, color);
                            let border = if is_active || resp.hovered() {
                                Color32::WHITE
                            } else {
                                Color32::from_gray(90)
                            };
                            painter.rect_stroke(rect, 3.0, egui::Stroke::new(if is_active { 2.0 } else { 1.0 }, border));

                            if resp.clicked() { set_preset = Some(c); }
                            resp.on_hover_text(label);
                        }
                    });

                    ui.separator();
                    ui.label("Custom color:");
                    ui.horizontal(|ui| {
                        let mut rgb = [
                            new_color[0] as f32 / 255.0,
                            new_color[1] as f32 / 255.0,
                            new_color[2] as f32 / 255.0,
                        ];
                        egui::color_picker::color_edit_button_rgb(ui, &mut rgb);
                        new_color = [(rgb[0] * 255.0) as u8, (rgb[1] * 255.0) as u8, (rgb[2] * 255.0) as u8];
                        ui.label(RichText::new("← saved automatically").small().color(Color32::GRAY));
                    });

                    ui.separator();
                    if ui.button("Close").clicked() { close = true; }
                });

            if let Some(c) = set_preset {
                self.bg_color = c;
            } else {
                self.bg_color = new_color;
            }
            if close { self.show_bg_picker = false; }
        }

        // ── top bar (folder + progress + settings) ─────────────────────────
        ui.horizontal(|ui| {
            if ui.button("📁 Load Folder").clicked() {
                let (tx, rx) = std::sync::mpsc::channel();
                self.folder_rx = Some(rx);
                std::thread::spawn(move || {
                    let r = rfd::FileDialog::new().set_title("Select image folder").pick_folder();
                    let _ = tx.send(r);
                });
            }

            // Recent folders
            if !self.recent_folders.is_empty() {
                egui::ComboBox::from_label("Recent")
                    .selected_text("Open recent…")
                    .show_ui(ui, |ui| {
                        let mut chosen: Option<PathBuf> = None;
                        for f in &self.recent_folders {
                            if ui.selectable_label(false, f.to_string_lossy().as_ref()).clicked() {
                                chosen = Some(f.clone());
                            }
                        }
                        if let Some(folder) = chosen {
                            self.load_source_folder(&folder);
                        }
                    });
            }

            ui.separator();

            let (done, total) = self.progress();
            if total > 0 {
                let prog = done as f32 / total as f32;
                ui.add(egui::ProgressBar::new(prog)
                    .desired_width(180.0)
                    .text(format!("{}/{}", done, total)));
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("⚙ Categories").clicked() {
                    self.category_editor.open_with(&self.categories);
                }
                ui.separator();

                let bg_label = if self.show_bg_picker {
                    RichText::new("🎨").strong()
                } else {
                    RichText::new("🎨")
                };
                if ui.button(bg_label).on_hover_text("Background color").clicked() {
                    self.show_bg_picker = !self.show_bg_picker;
                }
                ui.separator();

                let has_sorted = self.images.iter()
                    .any(|i| i.status != ImageStatus::Unsorted);
                if ui.add_enabled(
                    has_sorted,
                    egui::Button::new(RichText::new("↺ Reset Sorting")
                        .color(Color32::from_rgb(220, 100, 80)))
                ).clicked() {
                    self.confirm_reset = true;
                }
                ui.separator();

                // Output folder
                if ui.button("📁 Output").clicked() {
                    let (tx, rx) = std::sync::mpsc::channel();
                    self.output_rx = Some(rx);
                    std::thread::spawn(move || {
                        let r = rfd::FileDialog::new().set_title("Output folder").pick_folder();
                        let _ = tx.send(r);
                    });
                }
                if let Some(out) = &self.output_root {
                    ui.label(RichText::new(out.to_string_lossy()).small().color(Color32::GRAY));
                }
                ui.separator();
                ui.checkbox(&mut self.copy_not_move, "Copy");
            });
        });

        ui.separator();

        // ── main area: image view | thumbnail grid ─────────────────────────
        let available = ui.available_size();
        let left_w  = (available.x * 0.62).max(200.0);
        let right_w = (available.x - left_w - 8.0).max(150.0);

        ui.horizontal(|ui| {
            // Left: main image + action buttons
            ui.vertical(|ui| {
                ui.set_width(left_w);
                self.show_main_image(ui, ctx, left_w, available.y - 120.0);
                self.show_action_bar(ui, ctx, toasts);
            });

            ui.separator();

            // Right: filter + thumbnail grid
            ui.vertical(|ui| {
                ui.set_width(right_w);
                self.show_thumbnail_panel(ui, ctx, right_w, available.y - 60.0);
            });
        });

        // ── statistics strip ───────────────────────────────────────────────
        ui.separator();
        self.show_stats_strip(ui);
    }

    fn show_main_image(&mut self, ui: &mut Ui, ctx: &Context, max_w: f32, max_h: f32) {
        // Flash overlay
        let flash_alpha = if self.flash_frames > 0 {
            self.flash_frames as f32 / 12.0
        } else { 0.0 };

        let image_area = ui.allocate_exact_size(
            Vec2::new(max_w, max_h.max(100.0)),
            egui::Sense::click_and_drag(),
        );
        let (rect, resp) = image_area;

        // Zoom with scroll wheel
        let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
        if resp.hovered() && scroll != 0.0 {
            let zoom_delta = 1.0 + scroll * 0.002;
            self.zoom = (self.zoom * zoom_delta).clamp(0.1, 20.0);
        }

        // Pan with drag
        if resp.dragged() {
            self.pan += resp.drag_delta();
        }

        // Reset zoom on double-click
        if resp.double_clicked() {
            self.zoom = 1.0;
            self.pan  = Vec2::ZERO;
        }

        let painter = ui.painter_at(rect);

        // Solid background
        let bg = Color32::from_rgb(self.bg_color[0], self.bg_color[1], self.bg_color[2]);
        painter.rect_filled(rect, 0.0, bg);

        // Draw the image (with zoom & pan)
        if self.current_idx < self.images.len() {
            let img = &self.images[self.current_idx];
            if let Some(tex) = &img.texture {
                let [tw, th] = tex.size();
                let base_scale = (max_w / tw as f32).min(max_h / th as f32).min(1.0);
                let scale = base_scale * self.zoom;
                let img_w  = tw as f32 * scale;
                let img_h  = th as f32 * scale;
                let center = rect.center() + self.pan;
                let img_rect = egui::Rect::from_center_size(center, Vec2::new(img_w, img_h));

                painter.image(
                    tex.id(),
                    img_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    Color32::WHITE,
                );

                // Flash overlay
                if flash_alpha > 0.0 {
                    if let Some(fc) = self.flash_color {
                        let flash = Color32::from_rgba_unmultiplied(fc.r(), fc.g(), fc.b(),
                            (flash_alpha * 80.0) as u8);
                        painter.rect_filled(img_rect, 0.0, flash);
                    }
                }

            } else {
                // Still loading
                painter.text(rect.center(), egui::Align2::CENTER_CENTER,
                    "Loading…", egui::FontId::proportional(16.0), Color32::GRAY);
            }

            // Image name + zoom overlay
            let fname = self.images[self.current_idx].path
                .file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
            painter.text(
                egui::pos2(rect.left() + 6.0, rect.bottom() - 20.0),
                egui::Align2::LEFT_BOTTOM,
                format!("{}  [{:.0}%]", fname, self.zoom * 100.0),
                egui::FontId::monospace(11.0),
                Color32::from_white_alpha(200),
            );
        } else {
            painter.text(rect.center(), egui::Align2::CENTER_CENTER,
                "No images loaded\nDrop a folder here",
                egui::FontId::proportional(18.0), Color32::GRAY);
        }
    }

    fn show_action_bar(&mut self, ui: &mut Ui, _ctx: &Context, toasts: &mut ToastManager) {
        ui.horizontal(|ui| {
            // Category buttons
            let mut do_cat: Option<usize> = None;
            let mut do_skip = false;
            let mut do_flag = false;
            let mut do_undo = false;

            for (i, cat) in self.categories.iter().enumerate() {
                let color = Color32::from_rgb(cat.color[0], cat.color[1], cat.color[2]);
                let key_hint = cat.hotkey.as_deref().unwrap_or("");
                let label = format!("{} [{}]", cat.name, key_hint);

                let btn = egui::Button::new(
                    RichText::new(&label).color(Color32::WHITE).strong()
                ).fill(color);

                if ui.add(btn).clicked() {
                    do_cat = Some(i);
                }
            }

            ui.separator();

            if ui.button("→ Skip").clicked() { do_skip = true; }
            if ui.button("⚑ Flag").clicked() { do_flag = true; }

            ui.separator();

            if ui.button("↩ Undo").clicked() { do_undo = true; }

            // Nav arrows
            ui.separator();
            if ui.button("◀").clicked() { self.navigate_prev(); }
            if ui.button("▶").clicked() { self.navigate_next(); }
            if ui.button("↠").on_hover_text("Jump to first unsorted [Tab]").clicked() {
                self.jump_to_first_unsorted();
            }

            // Zoom control
            ui.separator();
            ui.label("Zoom:");
            ui.add(egui::DragValue::new(&mut self.zoom).range(0.1..=10.0).speed(0.01).max_decimals(2));
            if ui.small_button("1:1").clicked() { self.zoom = 1.0; self.pan = Vec2::ZERO; }

            if let Some(ci) = do_cat { self.categorise(ci, toasts); }
            if do_skip { self.skip(); }
            if do_flag { self.flag(); }
            if do_undo { self.undo(); }
        });

        // Hotkey reminder (small, fading hint)
        if !self.categories.is_empty() {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Hotkeys:").small().color(Color32::GRAY));
                for cat in &self.categories {
                    if let Some(key) = &cat.hotkey {
                        let color = Color32::from_rgb(cat.color[0], cat.color[1], cat.color[2]);
                        ui.label(RichText::new(format!("[{}]={}", key, cat.name))
                            .small().color(color));
                    }
                }
                ui.label(RichText::new("  [→]=Skip  [←]=Prev  [↓]=Next  [F]=Flag  [Ctrl+Z]=Undo  [Tab]=First Unsorted")
                    .small().color(Color32::GRAY));
            });
        }
    }

    fn show_thumbnail_panel(&mut self, ui: &mut Ui, _ctx: &Context, panel_w: f32, _panel_h: f32) {
        // Filter bar
        ui.horizontal(|ui| {
            let filters = {
                let mut f = vec![FilterMode::All, FilterMode::Unsorted, FilterMode::Flagged, FilterMode::Skipped];
                for cat in &self.categories {
                    f.push(FilterMode::Categorised(cat.name.clone()));
                }
                f
            };
            for mode in &filters {
                let selected = &self.filter == mode;
                if ui.selectable_label(selected, mode.label()).clicked() {
                    self.filter = mode.clone();
                }
            }
        });

        // Search box
        ui.horizontal(|ui| {
            ui.label("🔍");
            ui.add(egui::TextEdit::singleline(&mut self.search_query)
                .desired_width(panel_w - 40.0)
                .hint_text("Search filename…"));
            if ui.small_button("✕").clicked() { self.search_query.clear(); }
        });

        ui.separator();

        // Compute thumbnail grid layout
        let thumb_sz = self.thumbnail_size;
        let thumb_with_pad = thumb_sz + 8.0;
        let cols = ((panel_w / thumb_with_pad).floor() as usize).max(1);

        let visible_idxs = self.filtered_indices();

        ScrollArea::vertical()
            .id_salt("sorter_thumb_grid")
            .show_rows(ui, thumb_with_pad + 4.0, (visible_idxs.len() + cols - 1) / cols,
                |ui, row_range| {
                    for row in row_range {
                        ui.horizontal(|ui| {
                            for col in 0..cols {
                                let slot = row * cols + col;
                                if slot >= visible_idxs.len() { break; }
                                let img_idx = visible_idxs[slot];
                                self.draw_thumbnail(ui, img_idx, thumb_sz);
                            }
                        });
                    }
                }
            );
    }

    fn draw_thumbnail(&mut self, ui: &mut Ui, img_idx: usize, size: f32) {
        let is_current = img_idx == self.current_idx;
        // Clone status eagerly to avoid borrow conflicts with navigate_to / categories
        let status = self.images[img_idx].status.clone();

        let (rect, resp) = ui.allocate_exact_size(
            Vec2::splat(size + 6.0),
            egui::Sense::click(),
        );

        if resp.clicked() {
            self.navigate_to(img_idx);
        }

        let inner = egui::Rect::from_center_size(rect.center(), Vec2::splat(size));

        // Border
        let border_color = if is_current {
            Color32::from_rgb(100, 180, 255)
        } else {
            Color32::from_gray(80)
        };
        let border_w = if is_current { 2.5 } else { 1.0 };
        ui.painter().rect_stroke(inner, 2.0, egui::Stroke::new(border_w, border_color));

        // Thumbnail image or placeholder
        if let Some(tex) = &self.images[img_idx].thumb {
            ui.painter().image(
                tex.id(), inner,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                Color32::WHITE,
            );
        } else {
            ui.painter().rect_filled(inner, 2.0, Color32::from_gray(40));
        }

        // Status badge (bottom-right corner)
        let badge_text: Option<(&'static str, Color32)> = match &status {
            ImageStatus::Categorised(name) => {
                let color = self.categories.iter()
                    .find(|c| c.name == *name)
                    .map(|c| Color32::from_rgb(c.color[0], c.color[1], c.color[2]))
                    .unwrap_or(Color32::GRAY);
                Some(("✓", color))
            }
            ImageStatus::Skipped => Some(("→", Color32::YELLOW)),
            ImageStatus::Flagged  => Some(("⚑", Color32::from_rgb(255, 140, 0))),
            ImageStatus::Unsorted => None,
        };

        if let Some((icon, color)) = badge_text {
            let bg = Color32::from_rgba_unmultiplied(0, 0, 0, 160);
            // badge_pos is the bottom-right anchor; min is 16px up-left from it
            let badge_min  = inner.right_bottom() + Vec2::new(-18.0, -18.0);
            let badge_rect = egui::Rect::from_min_size(badge_min, Vec2::splat(16.0));
            ui.painter().rect_filled(badge_rect, 3.0, bg);
            ui.painter().text(
                badge_rect.center(),
                egui::Align2::CENTER_CENTER,
                icon,
                egui::FontId::proportional(11.0),
                color,
            );
        }

        // Tooltip — on_hover_text is the egui-idiomatic approach
        let fname = self.images[img_idx].path
            .file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string();
        let status_str = match &status {
            ImageStatus::Unsorted        => "Unsorted".to_string(),
            ImageStatus::Categorised(n)  => format!("→ {}", n),
            ImageStatus::Skipped         => "Skipped".to_string(),
            ImageStatus::Flagged         => "Flagged".to_string(),
        };
        resp.on_hover_text(format!("{}\n{}", fname, status_str));
    }

    fn show_stats_strip(&self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            let unsorted = self.images.iter().filter(|i| i.status == ImageStatus::Unsorted).count();
            let flagged  = self.images.iter().filter(|i| i.status == ImageStatus::Flagged).count();
            let skipped  = self.images.iter().filter(|i| i.status == ImageStatus::Skipped).count();

            ui.label(format!("Total: {}  ", self.images.len()));
            ui.separator();
            ui.label(RichText::new(format!("Unsorted: {}", unsorted)).color(Color32::GRAY));
            ui.separator();

            for cat in &self.categories {
                let count = self.images.iter()
                    .filter(|i| i.status == ImageStatus::Categorised(cat.name.clone()))
                    .count();
                let color = Color32::from_rgb(cat.color[0], cat.color[1], cat.color[2]);
                ui.label(RichText::new(format!("{}: {}", cat.name, count)).color(color));
                ui.separator();
            }

            if flagged > 0 {
                ui.label(RichText::new(format!("Flagged: {}", flagged))
                    .color(Color32::from_rgb(255, 140, 0)));
                ui.separator();
            }
            if skipped > 0 {
                ui.label(RichText::new(format!("Skipped: {}", skipped)).color(Color32::YELLOW));
            }
        });
    }
}

// ── CSV log helpers ───────────────────────────────────────────────────────────
//
// Format: two-column, comma-separated, with a header row.
//   source_path,status
//   C:\...\leaf_001.png,Healthy
//   C:\...\leaf_002.png,Skipped
//
// Append-only; "last entry wins" per path on reload → handles re-sort & undo.

fn append_csv_entry(csv_path: &std::path::Path, source: &std::path::Path, status: &str) {
    use std::io::Write;
    if let Some(parent) = csv_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let write_header = !csv_path.exists();
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true).append(true).open(csv_path)
    {
        if write_header {
            let _ = writeln!(f, "source_path,status");
        }
        // Paths on disk essentially never contain commas; status strings never do.
        let _ = writeln!(f, "{},{}", source.display(), status);
    }
}

/// Read the CSV log and return a map from canonical path string → latest status string.
/// "Last entry wins" — iterating in file order with HashMap::insert gives this naturally.
fn load_csv_log(csv_path: &std::path::Path) -> HashMap<String, String> {
    let mut map = HashMap::new();
    if let Ok(content) = std::fs::read_to_string(csv_path) {
        for line in content.lines().skip(1) { // skip header
            // Use rfind so that even a path with commas would be handled correctly
            // (the status field is always the very last segment).
            if let Some(comma) = line.rfind(',') {
                let path   = line[..comma].trim().to_string();
                let status = line[comma + 1..].trim().to_string();
                if !path.is_empty() {
                    map.insert(path, status);
                }
            }
        }
    }
    map
}

fn is_image(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(str::to_lowercase).as_deref(),
        Some("png") | Some("tiff") | Some("tif") | Some("jpg") | Some("jpeg")
    )
}
