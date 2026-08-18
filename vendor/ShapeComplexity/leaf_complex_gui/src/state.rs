// Application State Management
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use eframe::egui;
use leaf_complex_rust_lib::feature_extraction::MarginalPointFeatures;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnalysisStatus {
    NotStarted,
    Running,
    Completed,
    Failed,
}

#[derive(Clone)]
pub struct AnalysisResult {
    pub ec_data: Vec<(f64, f64)>,
    pub mc_data: Vec<(f64, f64)>,
    pub summary: SummaryStats,
    pub ec_image_texture: Option<egui::TextureHandle>,
    pub mc_image_texture: Option<egui::TextureHandle>,
    pub original_texture: Option<egui::TextureHandle>,

    pub ec_contour: Vec<(u32, u32)>,
    pub mc_contour: Vec<(u32, u32)>,

    pub ec_features: Vec<MarginalPointFeatures>,
    pub mc_features: Vec<MarginalPointFeatures>,
    pub ec_reference_point: (u32, u32),
    pub mc_reference_point: (u32, u32),

    /// Pre-built CSV strings — written directly to disk at export time.
    /// Avoids re-formatting 1000+ rows per image during export.
    pub ec_csv: String,
    pub mc_csv: String,
}

#[derive(Clone, Default)]
pub struct SummaryStats {
    pub ec_length: f64,
    pub ec_width: f64,
    pub ec_shape_index: f64,
    pub ec_circularity: f64,
    pub ec_spectral_entropy: f64,
    pub ec_area: u32,
    pub ec_outline_count: u32,

    pub mc_length: f64,
    pub mc_width: f64,
    pub mc_shape_index: f64,
    pub mc_circularity: f64,
    pub mc_spectral_entropy: f64,
    pub mc_area: u32,
    pub mc_outline_count: u32,
}

pub struct ImageInfo {
    pub path: PathBuf,
    pub filename: String,
    pub thumbnail: Option<egui::TextureHandle>,
    pub status: AnalysisStatus,
    pub selected: bool,
}

pub struct AppState {
    // Workspace
    pub workspace_dir: Option<PathBuf>,
    pub images: Vec<ImageInfo>,
    pub current_image_index: Option<usize>,

    // Analysis
    pub analysis_results: HashMap<PathBuf, AnalysisResult>,
    pub analysis_in_progress: bool,
    pub batch_processing: bool,
    pub current_batch_index: usize,
    pub total_batch_count: usize,

    // UI State
    pub selected_point: Option<usize>,
    pub selected_point_type: PointType,
    pub show_ec_overlay: bool,
    pub show_mc_overlay: bool,
    pub show_path_overlay: bool,
    pub zoom_level: f32,
    pub pan_offset: egui::Vec2,

    pub thumbnail_scroll_offset: f32,

    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PointType {
    EC,
    MC,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            workspace_dir: None,
            images: Vec::new(),
            current_image_index: None,
            analysis_results: HashMap::new(),
            analysis_in_progress: false,
            batch_processing: false,
            current_batch_index: 0,
            total_batch_count: 0,
            selected_point: None,
            selected_point_type: PointType::EC,
            show_ec_overlay: true,
            show_mc_overlay: true,
            show_path_overlay: true,
            zoom_level: 1.0,
            pan_offset: egui::Vec2::ZERO,
            thumbnail_scroll_offset: 0.0,
            last_error: None,
        }
    }
}

impl AppState {
    pub fn current_image(&self) -> Option<&ImageInfo> {
        self.current_image_index
            .and_then(|idx| self.images.get(idx))
    }

    pub fn current_result(&self) -> Option<&AnalysisResult> {
        self.current_image()
            .and_then(|img| self.analysis_results.get(&img.path))
    }

    pub fn load_workspace(&mut self, dir: PathBuf) {
        self.workspace_dir = Some(dir.clone());
        self.images.clear();
        self.current_image_index = None;

        // FIX 1: Use recursive collection with output-folder/output-file filtering
        let mut png_paths = Vec::new();
        collect_png_files_recursive(&dir, &mut png_paths);

        for path in png_paths {
            if let Some(filename) = path.file_stem().and_then(|s| s.to_str()) {
                self.images.push(ImageInfo {
                    path: path.clone(),
                    filename: filename.to_string(),
                    thumbnail: None,
                    status: AnalysisStatus::NotStarted,
                    selected: false,
                });
            }
        }

        self.images.sort_by(|a, b| a.filename.cmp(&b.filename));

        if !self.images.is_empty() {
            self.current_image_index = Some(0);
        }
    }

    pub fn select_image(&mut self, index: usize) {
        if index < self.images.len() {
            self.current_image_index = Some(index);
            self.selected_point = None;
            self.reset_view();
        }
    }

    pub fn reset_view(&mut self) {
        self.zoom_level = 1.0;
        self.pan_offset = egui::Vec2::ZERO;
    }

    pub fn select_all(&mut self) {
        for img in &mut self.images {
            img.selected = true;
        }
    }

    pub fn deselect_all(&mut self) {
        for img in &mut self.images {
            img.selected = false;
        }
    }

    pub fn get_selected_images(&self) -> Vec<PathBuf> {
        self.images.iter()
            .filter(|img| img.selected)
            .map(|img| img.path.clone())
            .collect()
    }

    pub fn selected_count(&self) -> usize {
        self.images.iter().filter(|img| img.selected).count()
    }
}

/// Recursively collect PNG files from a directory tree.
///
/// Skips:
///   - Known output folders produced by the exporter (EC, MC, summary,
///     ShapeComplexityResults) so their images never pollute the workspace.
///   - PNG files whose stem ends with `_EC` or `_MC`, which are our own
///     visualisation outputs, even if they somehow end up in a non-output folder.
///
/// FIX 1 (transparency bug): output images have opaque backgrounds; loading
///   them caused the alpha-based calculations to fail intermittently depending
///   on where the user had previously exported results.
///
/// FIX 2 (subfolder support): the old implementation called `read_dir` only on
///   the top-level directory. This function recurses into every subfolder that
///   is not an output folder, enabling workspace trees like:
///
///   workspace/
///     species_a/
///       leaf1.png
///       leaf2.png
///     species_b/
///       leaf3.png
fn collect_png_files_recursive(dir: &Path, result: &mut Vec<PathBuf>) {
    // Folder names produced by the exporter — never recurse into these.
    const SKIP_DIRS: &[&str] = &["EC", "MC", "summary", "ShapeComplexityResults"];

    let Ok(entries) = std::fs::read_dir(dir) else { return };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            let dir_name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if !SKIP_DIRS.contains(&dir_name) {
                collect_png_files_recursive(&path, result);
            }
        } else if path.is_file() {
            let is_png = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("png"))
                .unwrap_or(false);

            if is_png {
                let stem = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("");
                // Skip our own output visualisations
                if !stem.ends_with("_EC") && !stem.ends_with("_MC") {
                    result.push(path);
                }
            }
        }
    }
}