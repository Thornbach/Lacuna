use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ── Leaf descriptor enums ─────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize, Default, Debug)]
pub enum LeafShape {
    #[default] Lobed,
    Lanceolate,
    Ovate,
    Cordate,
    Palmate,
    Orbicular,
}

impl LeafShape {
    pub fn index(self) -> u32 { self as u32 }
    pub fn label(self) -> &'static str {
        match self {
            Self::Lobed      => "Lobed",
            Self::Lanceolate => "Lanceolate",
            Self::Ovate      => "Ovate",
            Self::Cordate    => "Cordate",
            Self::Palmate    => "Palmate",
            Self::Orbicular  => "Orbicular",
        }
    }
    pub const ALL: &'static [LeafShape] = &[
        Self::Lobed, Self::Lanceolate, Self::Ovate,
        Self::Cordate, Self::Palmate, Self::Orbicular,
    ];
}

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize, Default, Debug)]
pub enum MarginType {
    #[default] Smooth,
    Crenate,
    Serrate,
    Spiny,
}

impl MarginType {
    pub fn index(self) -> u32 { self as u32 }
    pub fn label(self) -> &'static str {
        match self {
            Self::Smooth  => "Smooth",
            Self::Crenate => "Crenate",
            Self::Serrate => "Serrate",
            Self::Spiny   => "Spiny",
        }
    }
    pub const ALL: &'static [MarginType] = &[
        Self::Smooth, Self::Crenate, Self::Serrate, Self::Spiny,
    ];
}

// ── per-tab persistent settings ──────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct EroderSettings {
    pub damage_levels: u32,
    pub max_damage_pct: f32,
    pub erosion_prob: f32,
    pub smoothing_iterations: u32,
    pub coastal_enabled: bool,
    pub coastal_weight: f32,
    pub spots_enabled: bool,
    pub spots_weight: f32,
    pub snake_enabled: bool,
    pub snake_weight: f32,
    pub ellipses_enabled: bool,
    pub ellipses_weight: f32,
    pub apex_enabled: bool,
    pub apex_weight: f32,
    pub clusters_enabled: bool,
    pub clusters_weight: f32,
    pub lobe_enabled: bool,
    pub lobe_weight: f32,
    pub boundary_noise: bool,
    pub independent_outputs: bool,
    pub resize_enabled: bool,
    pub resize_use_percent: bool,
    pub resize_percent: f32,
    pub resize_max_dim: u32,
    pub last_input_folder: Option<PathBuf>,
    pub last_output_folder: Option<PathBuf>,
    pub recent_input_folders: Vec<PathBuf>,
    pub seed: Option<u64>,
}

impl Default for EroderSettings {
    fn default() -> Self {
        Self {
            damage_levels: 10,
            max_damage_pct: 30.0,
            erosion_prob: 0.0005,
            smoothing_iterations: 10,
            coastal_enabled: true,
            coastal_weight: 0.7,
            spots_enabled: true,
            spots_weight: 0.2,
            snake_enabled: false,
            snake_weight: 0.1,
            ellipses_enabled: false,
            ellipses_weight: 0.5,
            apex_enabled: false,
            apex_weight: 0.3,
            clusters_enabled: false,
            clusters_weight: 0.5,
            lobe_enabled: false,
            lobe_weight: 0.5,
            boundary_noise: false,
            independent_outputs: false,
            resize_enabled: false,
            resize_use_percent: true,
            resize_percent: 50.0,
            resize_max_dim: 1024,
            last_input_folder: None,
            last_output_folder: None,
            recent_input_folders: Vec::new(),
            seed: None,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CategoryDef {
    pub name: String,
    pub hotkey: Option<String>, // single char, e.g. "1", "a"
    pub color: [u8; 3],
    pub output_subfolder: Option<String>,
}

impl CategoryDef {
    pub fn new(name: impl Into<String>, hotkey: impl Into<String>, color: [u8; 3]) -> Self {
        Self {
            name: name.into(),
            hotkey: Some(hotkey.into()),
            color,
            output_subfolder: None,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SorterSettings {
    pub categories: Vec<CategoryDef>,
    pub last_source_folder: Option<PathBuf>,
    pub last_output_folder: Option<PathBuf>,
    pub recent_source_folders: Vec<PathBuf>,
    pub copy_not_move: bool,
    pub show_already_sorted: bool,
    pub thumbnail_size: f32,
    /// Solid image-viewer background colour
    #[serde(default = "SorterSettings::default_bg", alias = "bg_color_a")]
    pub bg_color: [u8; 3],
}

impl SorterSettings {
    fn default_bg() -> [u8; 3] { [40, 40, 40] }
}

impl Default for SorterSettings {
    fn default() -> Self {
        Self {
            categories: vec![
                CategoryDef::new("Healthy", "1", [80, 180, 100]),
                CategoryDef::new("Damaged", "0", [200, 80, 80]),
            ],
            last_source_folder: None,
            last_output_folder: None,
            recent_source_folders: Vec::new(),
            copy_not_move: true,
            show_already_sorted: true,
            thumbnail_size: 96.0,
            bg_color: [40, 40, 40],
        }
    }
}


// ── recon-infer settings ──────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct ReconInferSettings {
    pub last_checkpoint_dir: Option<PathBuf>,
    pub last_source_folder:  Option<PathBuf>,
    pub last_output_folder:  Option<PathBuf>,
    pub image_size_px:       u32,
    pub threshold:           f32,
    #[serde(default = "default_pre_damage_pct")]
    pub pre_damage_pct:      f32,
    /// Run 4-rotation test-time augmentation (TTA) and average the predictions.
    /// Slower but more accurate on leaves in non-canonical orientations.
    #[serde(default)]
    pub tta_enabled:         bool,
    /// Leaf shape descriptor used for FiLM conditioning at inference.
    #[serde(default)]
    pub leaf_shape:          LeafShape,
    /// Margin type descriptor used for FiLM conditioning at inference.
    #[serde(default)]
    pub margin_type:         MarginType,
}

fn default_pre_damage_pct() -> f32 { 1.0 }

impl Default for ReconInferSettings {
    fn default() -> Self {
        Self {
            last_checkpoint_dir: None,
            last_source_folder:  None,
            last_output_folder:  None,
            image_size_px:       256,
            threshold:           0.35,
            pre_damage_pct:      1.0,
            tta_enabled:         false,
            leaf_shape:          LeafShape::default(),
            margin_type:         MarginType::default(),
        }
    }
}

// ── recon-area settings ───────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct ReconAreaSettings {
    pub last_source_folder:     Option<PathBuf>,
    pub last_output_folder:     Option<PathBuf>,
    pub last_checkpoint_folder: Option<PathBuf>,
    pub species_label:          String,
    pub image_size_px:          u32,
    pub epochs:                 usize,
    pub batch_size:             usize,
    pub learning_rate:          f64,
    pub checkpoint_every:       usize,
    pub damage_min_pct:         f32,
    pub damage_max_pct:         f32,
}

impl Default for ReconAreaSettings {
    fn default() -> Self {
        Self {
            last_source_folder:     None,
            last_output_folder:     None,
            last_checkpoint_folder: None,
            species_label:          "Quercus".to_string(),
            image_size_px:          512,
            epochs:                 100,
            batch_size:             4,
            learning_rate:          2e-4,
            checkpoint_every:       10,
            damage_min_pct:         2.0,
            damage_max_pct:         30.0,
        }
    }
}

// ── recon-radial-infer settings ───────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct RadialInferSettings {
    pub last_checkpoint_dir: Option<PathBuf>,
    pub last_source_folder:  Option<PathBuf>,
    pub last_output_folder:  Option<PathBuf>,
    pub image_size_px:       u32,
}

impl Default for RadialInferSettings {
    fn default() -> Self {
        Self {
            last_checkpoint_dir: None,
            last_source_folder:  None,
            last_output_folder:  None,
            image_size_px:       512,
        }
    }
}

// ── recon-simple settings ─────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct ReconSimpleSettings {
    pub last_source_folder:  Option<PathBuf>,
    pub last_output_folder:  Option<PathBuf>,
    pub last_resume_folder:  Option<PathBuf>,
    pub epochs:              usize,
    pub batch_size:          usize,
    pub learning_rate:       f64,
    pub l1_lambda:           f32,
    pub tv_lambda:           f32,
    pub conf_lambda:         f32,
    pub recon_focus_weight:  f32,
    pub tversky_alpha:       f32,
    pub tversky_beta:        f32,
    pub checkpoint_every:    usize,
    pub sample_grid_every:   usize,
    pub image_size_px:       u32,
    pub curriculum_epochs:   usize,
    pub damage_min_pct:      f32,
    pub damage_max_pct:      f32,
    pub curriculum_max_pct:  f32,
    pub zero_damage_prob:    f32,
    pub coastal_enabled:     bool,  pub coastal_weight:   f32,
    pub spots_enabled:       bool,  pub spots_weight:     f32,
    pub snake_enabled:       bool,  pub snake_weight:     f32,
    pub ellipses_enabled:    bool,  pub ellipses_weight:  f32,
    pub apex_enabled:        bool,  pub apex_weight:      f32,
    pub clusters_enabled:    bool,  pub clusters_weight:  f32,
    pub lobe_enabled:        bool,  pub lobe_weight:      f32,
    #[serde(default = "ReconSimpleSettings::default_focal_sector_enabled")]
    pub focal_sector_enabled: bool,
    #[serde(default = "ReconSimpleSettings::default_focal_sector_weight")]
    pub focal_sector_weight:  f32,
    /// Pixels to zero out inward from the damage boundary on the intact side.
    /// Forces long-range reasoning rather than boundary interpolation. Default 0.
    pub boundary_exclusion_px: usize,
    pub accum_steps:           usize,
    /// Macro leaf shape used to condition the FiLM descriptor.
    #[serde(default)]
    pub leaf_shape:  LeafShape,
    /// Leaf margin type used to condition the FiLM descriptor.
    #[serde(default)]
    pub margin_type: MarginType,
    /// Reconstruction-only pre-training phase before enabling the discriminator.
    #[serde(default = "ReconSimpleSettings::default_pretrain_epochs")]
    pub pretrain_epochs: usize,
    /// D learning rate = G LR × d_lr_factor.
    #[serde(default = "ReconSimpleSettings::default_d_lr_factor")]
    pub d_lr_factor: f64,
    /// Adversarial loss weight in the generator step.
    #[serde(default = "ReconSimpleSettings::default_adv_lambda")]
    pub adv_lambda: f32,
    /// Cosine LR schedule floor, as a fraction of `learning_rate`, reached at
    /// the final epoch. 1.0 = flat LR (no decay).
    #[serde(default = "ReconSimpleSettings::default_lr_min_frac")]
    pub lr_min_frac: f64,
    /// 0-indexed epoch to resume at (only applied when a resume checkpoint is
    /// picked). Auto-filled from the checkpoint's `epoch_info.txt` if present.
    #[serde(default)]
    pub start_epoch: usize,
    /// Best val IoU already achieved before a resume, so a worse early epoch
    /// post-resume can't clobber an existing checkpoint_best.
    #[serde(default)]
    pub resume_best_iou: f32,
    /// Weight on the hole-consistency loss (penalises invented interior holes
    /// in the predicted silhouette). 0 = disabled.
    #[serde(default = "ReconSimpleSettings::default_hole_lambda")]
    pub hole_lambda: f32,
}

impl ReconSimpleSettings {
    fn default_focal_sector_enabled() -> bool  { true }
    fn default_focal_sector_weight()  -> f32   { 1.0  }
    fn default_pretrain_epochs()      -> usize { 10   }
    fn default_d_lr_factor()          -> f64   { 0.5  }
    fn default_adv_lambda()           -> f32   { 20.0 }
    fn default_lr_min_frac()          -> f64   { 0.05 }
    fn default_hole_lambda()          -> f32   { 3.0  }
}

impl Default for ReconSimpleSettings {
    fn default() -> Self {
        Self {
            last_source_folder:  None,
            last_output_folder:  None,
            last_resume_folder:  None,
            epochs:              200,
            batch_size:          2,
            learning_rate:       2e-4,
            l1_lambda:           10.0,
            tv_lambda:           0.05,
            conf_lambda:         0.5,
            recon_focus_weight:  4.0,
            tversky_alpha:       0.85,
            tversky_beta:        0.15,
            checkpoint_every:    10,
            sample_grid_every:   5,
            image_size_px:       512,
            curriculum_epochs:   20,
            damage_min_pct:      10.0,
            damage_max_pct:      85.0,
            curriculum_max_pct:  40.0,
            zero_damage_prob:    0.30,
            coastal_enabled:     true,   coastal_weight:   0.35,
            spots_enabled:       true,   spots_weight:     0.20,
            snake_enabled:       false,  snake_weight:     0.10,
            ellipses_enabled:    true,   ellipses_weight:  0.15,
            apex_enabled:        true,   apex_weight:      0.50,
            clusters_enabled:    true,   clusters_weight:  0.60,
            lobe_enabled:        true,   lobe_weight:      0.50,
            focal_sector_enabled: true,
            focal_sector_weight:  1.0,
            boundary_exclusion_px: 0,
            accum_steps:           1,
            leaf_shape:            LeafShape::default(),
            margin_type:           MarginType::default(),
            pretrain_epochs:       10,
            d_lr_factor:           0.5,
            adv_lambda:            20.0,
            lr_min_frac:           0.05,
            start_epoch:           0,
            resume_best_iou:       0.0,
            hole_lambda:           3.0,
        }
    }
}

// ── leaf segmentation tab ─────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct LeafSegSettings {
    pub last_source_folder: Option<PathBuf>,
    pub last_output_folder: Option<PathBuf>,
    pub last_model_path:    Option<PathBuf>,
    pub conf:  f32,
    pub iou:   f32,
    pub imgsz: u32,
}

impl Default for LeafSegSettings {
    fn default() -> Self {
        Self {
            last_source_folder: None,
            last_output_folder: None,
            last_model_path:    None,
            conf:  0.25,
            iou:   0.45,
            imgsz: 640,
        }
    }
}

// ── integrated pipeline tab ───────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct PipelineSettings {
    pub last_source_folder: Option<PathBuf>,
    pub last_output_folder: Option<PathBuf>,
    pub yolo_model_path:    Option<PathBuf>,
    pub dino_model_path:    Option<PathBuf>,
    pub bank_path:          Option<PathBuf>,
    pub meta_path:          Option<PathBuf>,
    pub tile_size:          u32,
    pub residual_max_area:  u32,
    pub region_close_px:    u32,
    pub min_region_area:    u32,
    #[serde(default = "default_margin_erode")]
    pub margin_erode_px:    u32,
    #[serde(default)]
    pub recon_ckpt:         Option<PathBuf>,
    /// Few-shot supervised head (`fewshot_head.json`). Per-tab override of the
    /// shared `AppDefaults.head`.
    #[serde(default)]
    pub head_path:          Option<PathBuf>,
    /// Use the few-shot head instead of the PatchCore bank (when a head is set).
    #[serde(default = "default_true")]
    pub use_fewshot:        bool,
    /// ALSO run the PatchCore bank alongside the few-shot head (open-set safety
    /// net for anomaly types the head was never trained on). Off by default —
    /// only worth enabling once the bank reflects real healthy data, otherwise
    /// it mostly surfaces its own uncalibrated false positives.
    #[serde(default)]
    pub use_patchcore:      bool,
    /// Few-shot defect-probability threshold τ (hysteresis SEED).
    #[serde(default = "default_head_tau")]
    pub head_tau:           f32,
    /// Few-shot hysteresis GROW threshold (higher = tighter regions).
    #[serde(default = "default_head_grow")]
    pub head_grow:          f32,
    /// DBSCAN radius over standardized region descriptors (PatchCore clustering
    /// only). Lower = more, smaller, possibly-overlapping-in-meaning clusters.
    #[serde(default = "default_cluster_eps")]
    pub cluster_eps:        f32,
    /// DBSCAN min points. Lower = more, smaller, looser clusters.
    #[serde(default = "default_cluster_min_pts")]
    pub cluster_min_pts:    usize,
}

fn default_margin_erode() -> u32 { 6 }
fn default_true() -> bool { true }
fn default_head_tau() -> f32 { 0.85 }
fn default_head_grow() -> f32 { 0.7 }
fn default_cluster_eps() -> f32 { 1.5 }
fn default_cluster_min_pts() -> usize { 5 }

impl Default for PipelineSettings {
    fn default() -> Self {
        Self {
            last_source_folder: None,
            last_output_folder: None,
            yolo_model_path:    None,
            dino_model_path:    None,
            bank_path:          None,
            meta_path:          None,
            tile_size:          256,
            residual_max_area:  100,
            region_close_px:    2,
            min_region_area:    3,
            margin_erode_px:    6,
            recon_ckpt:         None,
            head_path:          None,
            use_fewshot:        true,
            use_patchcore:      false,
            head_tau:           0.85,
            head_grow:          0.7,
            cluster_eps:        1.5,
            cluster_min_pts:    5,
        }
    }
}

// ── train detector tab ────────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct TrainSettings {
    pub healthy_folder: Option<PathBuf>,
    pub dino_model:     Option<PathBuf>,
    pub out_dir:        Option<PathBuf>,
    pub coreset_target: u32,
    pub n_fit:          u32,
}

impl Default for TrainSettings {
    fn default() -> Self {
        Self {
            healthy_folder: None,
            dino_model:     None,
            out_dir:        None,
            coreset_target: 150_000,
            n_fit:          3000,
        }
    }
}

// ── tile picker tab ───────────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct TilePickerSettings {
    pub source_folder: Option<PathBuf>,
    pub output_folder: Option<PathBuf>,
    pub tile:          u32,
    pub zoom:          f32,
    /// Loupe glued next to the cursor/stamp square instead of pinned to a corner.
    #[serde(default)]
    pub loupe_follow:  bool,
    /// Normalised [0,1] top-left anchor of the fixed loupe (default top-right).
    #[serde(default = "default_loupe_anchor")]
    pub loupe_anchor:  [f32; 2],
}

fn default_loupe_anchor() -> [f32; 2] { [1.0, 0.0] }

impl Default for TilePickerSettings {
    fn default() -> Self {
        Self {
            source_folder: None,
            output_folder: None,
            tile:          256,
            zoom:          3.0,
            loupe_follow:  false,
            loupe_anchor:  [1.0, 0.0],
        }
    }
}

// ── morphology tab ────────────────────────────────────────────────────────────

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct MorphologySettings {
    pub workspace: Option<PathBuf>,
    pub output:    Option<PathBuf>,
}

// ── root app settings ─────────────────────────────────────────────────────────

fn default_ui_scale() -> f32 { 1.0 }

/// Shared model-asset defaults, set once in Settings and inherited by the
/// Pipeline & Train tabs when their own field is empty.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct AppDefaults {
    pub yolo:    Option<PathBuf>,
    pub dino:    Option<PathBuf>,
    pub bank:    Option<PathBuf>,
    pub meta:    Option<PathBuf>,
    pub recon:   Option<PathBuf>,
    pub healthy: Option<PathBuf>,
    /// Few-shot detection head (`fewshot_head.json`). When set, the pipeline uses
    /// the supervised head instead of the PatchCore bank (skips the 0.9 GB bank).
    #[serde(default)]
    pub head:    Option<PathBuf>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AppSettings {
    pub dark_mode:    bool,
    pub theme_name:   String,
    #[serde(default = "default_ui_scale")]
    pub ui_scale:     f32,
    #[serde(default)]
    pub defaults:     AppDefaults,
    pub eroder:       EroderSettings,
    pub sorter:       SorterSettings,
    #[serde(default)]
    pub recon_infer:  ReconInferSettings,
    #[serde(default)]
    pub recon_area:   ReconAreaSettings,
    #[serde(default)]
    pub recon_simple: ReconSimpleSettings,
    #[serde(default)]
    pub radial_infer: RadialInferSettings,
    #[serde(default)]
    pub leaf_seg:     LeafSegSettings,
    #[serde(default)]
    pub pipeline:     PipelineSettings,
    #[serde(default)]
    pub train:        TrainSettings,
    #[serde(default)]
    pub tile_picker:  TilePickerSettings,
    #[serde(default)]
    pub morphology:   MorphologySettings,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            dark_mode:    true,
            theme_name:   "Catppuccin Mocha".to_string(),
            ui_scale:     1.0,
            defaults:     AppDefaults::default(),
            eroder:       EroderSettings::default(),
            sorter:       SorterSettings::default(),
            recon_infer:  ReconInferSettings::default(),
            recon_area:   ReconAreaSettings::default(),
            recon_simple: ReconSimpleSettings::default(),
            radial_infer: RadialInferSettings::default(),
            leaf_seg:     LeafSegSettings::default(),
            pipeline:     PipelineSettings::default(),
            train:        TrainSettings::default(),
            tile_picker:  TilePickerSettings::default(),
            morphology:   MorphologySettings::default(),
        }
    }
}

impl AppSettings {
    fn settings_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("Lacuna").join("settings.json"))
    }

    pub fn load_from_file() -> Option<Self> {
        let path = Self::settings_path()?;
        let text = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&text).ok()
    }

    pub fn save_to_file(settings: &Self) {
        if let Some(path) = Self::settings_path() {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Ok(json) = serde_json::to_string_pretty(settings) {
                let _ = std::fs::write(path, json);
            }
        }
    }
}

// ── sorter session (resume where you left off) ────────────────────────────────

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct SorterSession {
    /// source folder this session belongs to
    pub source_folder: PathBuf,
    /// maps filename → category name (already categorised)
    pub categorised: std::collections::HashMap<String, String>,
    /// filenames explicitly skipped
    pub skipped: Vec<String>,
}

impl SorterSession {
    fn session_path(source: &PathBuf) -> Option<PathBuf> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        source.hash(&mut h);
        let hash = h.finish();
        dirs::config_dir().map(|d| {
            d.join("Lacuna")
                .join("sessions")
                .join(format!("{:x}.json", hash))
        })
    }

    pub fn load(source: &PathBuf) -> Option<Self> {
        let path = Self::session_path(source)?;
        let text = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&text).ok()
    }

    pub fn save(&self) {
        if let Some(path) = Self::session_path(&self.source_folder) {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Ok(json) = serde_json::to_string_pretty(self) {
                let _ = std::fs::write(path, json);
            }
        }
    }
}

// ── tile-picker session (per-folder progress: done markers + resume index) ─────

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct TilePickerSession {
    /// source folder this session belongs to
    pub source_folder: PathBuf,
    /// filenames the user has marked "done"
    pub done: std::collections::HashSet<String>,
    /// last image index the user was on (so we resume there)
    pub last_index: usize,
}

impl TilePickerSession {
    fn session_path(source: &PathBuf) -> Option<PathBuf> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        source.hash(&mut h);
        let hash = h.finish();
        dirs::config_dir().map(|d| {
            d.join("Lacuna")
                .join("tile_sessions")
                .join(format!("{:x}.json", hash))
        })
    }

    pub fn load(source: &PathBuf) -> Option<Self> {
        let path = Self::session_path(source)?;
        let text = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&text).ok()
    }

    pub fn save(&self) {
        if let Some(path) = Self::session_path(&self.source_folder) {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Ok(json) = serde_json::to_string_pretty(self) {
                let _ = std::fs::write(path, json);
            }
        }
    }
}
