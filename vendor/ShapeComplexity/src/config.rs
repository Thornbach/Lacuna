// src/config.rs - Configuration management for EC/MC analysis

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};

use crate::errors::{LeafComplexError, Result};

/// Main configuration structure for LeafComplexR
///
/// All analysis parameters are configurable via TOML file.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Config {
    /// Input path (file or directory)
    pub input_path: String,
    
    /// Output base directory (EC, MC, and summary.csv will be created here)
    pub output_base_dir: String,
    
    /// Optional resize dimensions [width, height] for batch processing
    pub resize_dimensions: Option<[u32; 2]>,
    
    /// Kernel size for morphological opening (EC region marking)
    pub opening_kernel_size: u32,
    
    /// RGB color for marking opened regions (default: bright pink)
    pub marked_region_color_rgb: [u8; 3],
    
    /// Reference point choice: "EP" (Emerge Point) or "COM" (Center of Mass)
    pub reference_point_choice: ReferencePointChoice,
    
    /// Enable parallel processing for batch operations
    #[serde(default = "default_parallel")]
    pub use_parallel: bool,
    
    // Adaptive Opening Parameters (for pink region marking)
    /// Density threshold: >=this % non-transparent pixels triggers max opening
    #[serde(default = "default_adaptive_opening_max_density")]
    pub adaptive_opening_max_density: f64,
    
    /// Maximum opening percentage of image dimension at high density
    #[serde(default = "default_adaptive_opening_max_percentage")]
    pub adaptive_opening_max_percentage: f64,
    
    /// Minimum opening percentage of image dimension at low density
    #[serde(default = "default_adaptive_opening_min_percentage")]
    pub adaptive_opening_min_percentage: f64,
    
    // Petiole Filtering Parameters
    /// Enable petiole filtering in EC analysis pipeline
    #[serde(default = "default_enable_petiole_filter_ec")]
    pub enable_petiole_filter_ec: bool,
    
    /// Enable petiole filtering in EC calculation
    #[serde(default = "default_enable_petiole_filter_ec_complexity")]
    pub enable_petiole_filter_ec_complexity: bool,

    /// Enable petiole filtering in MC analysis (default off; GUI parity)
    #[serde(default = "default_enable_petiole_filter_mc")]
    pub enable_petiole_filter_mc: bool,
    
    /// true = remove petiole completely and merge ends, false = set to zero
    #[serde(default = "default_petiole_remove_completely")]
    pub petiole_remove_completely: bool,
    
    // Pink Threshold Filtering Parameters
    /// Enable threshold filtering for Geodesic_EC values
    #[serde(default = "default_enable_pink_threshold_filter")]
    pub enable_pink_threshold_filter: bool,
    
    /// Values <= this threshold will be set to zero
    #[serde(default = "default_pink_threshold_value")]
    pub pink_threshold_value: f64,
    
    // Thornfiddle (MC) Analysis Parameters
    /// Gaussian sigma for periodic smoothing of Thornfiddle_Path
    #[serde(default = "default_thornfiddle_smoothing_strength")]
    pub thornfiddle_smoothing_strength: f64,
    
    // Approximate Entropy Parameters (for EC)
    /// Pattern length for ApEn calculation (typical: 1-3)
    #[serde(default = "default_approximate_entropy_m")]
    pub approximate_entropy_m: usize,
    
    /// Tolerance for ApEn calculation (typical: 0.1-0.3 * std_dev)
    #[serde(default = "default_approximate_entropy_r")]
    pub approximate_entropy_r: f64,
    
    /// Scaling factor for edge complexity calculation
    #[serde(default = "default_ec_scaling_factor")]
    pub ec_scaling_factor: f64,
    
    // Dynamic Golden Lobe Detection Parameters (for MC)
    /// Maximum opening percentage for circular leaves (MC_ShapeIndex = 1.0)
    #[serde(default = "default_thornfiddle_max_opening_percentage")]
    pub thornfiddle_max_opening_percentage: f64,
    
    /// Minimum opening percentage for elongated leaves (MC_ShapeIndex >= 5.0)
    #[serde(default = "default_thornfiddle_min_opening_percentage")]
    pub thornfiddle_min_opening_percentage: f64,
    
    /// Minimum golden pixels crossed to trigger harmonic chain
    #[serde(default = "default_thornfiddle_pixel_threshold")]
    pub thornfiddle_pixel_threshold: u32,
    
    /// RGB color for marking golden lobe regions
    #[serde(default = "default_thornfiddle_marked_color_rgb")]
    pub thornfiddle_marked_color_rgb: [u8; 3],
    
    // Harmonic Enhancement Parameters
    /// Maximum number of harmonics for largest segments relative to leaf circumference
    #[serde(default = "default_harmonic_max_harmonics")]
    pub harmonic_max_harmonics: usize,
    
    /// Global harmonic strength multiplier (1.0 = normal, 2.0 = double)
    #[serde(default = "default_harmonic_strength_multiplier")]
    pub harmonic_strength_multiplier: f64,
    
    /// Minimum chain length (in contour points) to count as valid harmonic chain
    #[serde(default = "default_harmonic_min_chain_length")]
    pub harmonic_min_chain_length: usize,
    
    // Spectral Entropy Sigmoid Scaling Parameters (for MC)
    /// Steepness of sigmoid transition (higher = sharper around threshold)
    #[serde(default = "default_spectral_entropy_sigmoid_k")]
    pub spectral_entropy_sigmoid_k: f64,
    
    /// Center point of sigmoid transition (coefficient of variation threshold)
    #[serde(default = "default_spectral_entropy_sigmoid_c")]
    pub spectral_entropy_sigmoid_c: f64,
}

/// Reference point calculation method
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum ReferencePointChoice {
    /// Emerge Point - bottommost central point
    Ep,
    /// Center of Mass - weighted centroid
    Com,
}

// Default value functions
fn default_parallel() -> bool { true }
fn default_adaptive_opening_max_density() -> f64 { 75.0 }
fn default_adaptive_opening_max_percentage() -> f64 { 15.0 }
fn default_adaptive_opening_min_percentage() -> f64 { 1.0 }
fn default_enable_petiole_filter_ec() -> bool { true }
fn default_enable_petiole_filter_ec_complexity() -> bool { true }
fn default_enable_petiole_filter_mc() -> bool { false }
fn default_petiole_remove_completely() -> bool { true }
fn default_enable_pink_threshold_filter() -> bool { true }
fn default_pink_threshold_value() -> f64 { 3.0 }
fn default_thornfiddle_smoothing_strength() -> f64 { 2.0 }
fn default_approximate_entropy_m() -> usize { 2 }
fn default_approximate_entropy_r() -> f64 { 0.2 }
fn default_ec_scaling_factor() -> f64 { 3.0 }
fn default_thornfiddle_max_opening_percentage() -> f64 { 30.0 }
fn default_thornfiddle_min_opening_percentage() -> f64 { 5.0 }
fn default_thornfiddle_pixel_threshold() -> u32 { 5 }
fn default_thornfiddle_marked_color_rgb() -> [u8; 3] { [255, 215, 0] }
fn default_harmonic_max_harmonics() -> usize { 12 }
fn default_harmonic_strength_multiplier() -> f64 { 2.0 }
fn default_harmonic_min_chain_length() -> usize { 15 }
fn default_spectral_entropy_sigmoid_k() -> f64 { 20.0 }
fn default_spectral_entropy_sigmoid_c() -> f64 { 0.04 }

impl Config {
    /// Load configuration from a TOML file
    ///
    /// # Arguments
    /// * `path` - Path to the TOML configuration file
    ///
    /// # Returns
    /// Parsed configuration or error
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let content = fs::read_to_string(path).map_err(|e| {
            LeafComplexError::Config(format!("Failed to read config file '{}': {}", path.display(), e))
        })?;

        let config: Config = toml::from_str(&content).map_err(|e| {
            LeafComplexError::Config(format!("Failed to parse config file '{}': {}", path.display(), e))
        })?;

        Ok(config)
    }

    /// Create default configuration
    ///
    /// # Returns
    /// Configuration with sensible defaults
    pub fn default() -> Self {
        Self {
            input_path: "./input".to_string(),
            output_base_dir: "./output".to_string(),
            resize_dimensions: Some([512, 512]),
            opening_kernel_size: 9,
            marked_region_color_rgb: [255, 0, 255],
            reference_point_choice: ReferencePointChoice::Com,
            use_parallel: true,
            adaptive_opening_max_density: 75.0,
            adaptive_opening_max_percentage: 15.0,
            adaptive_opening_min_percentage: 1.0,
            enable_petiole_filter_ec: true,
            enable_petiole_filter_ec_complexity: true,
            enable_petiole_filter_mc: false,
            petiole_remove_completely: true,
            enable_pink_threshold_filter: true,
            pink_threshold_value: 3.0,
            thornfiddle_smoothing_strength: 2.0,
            approximate_entropy_m: 2,
            approximate_entropy_r: 0.2,
            ec_scaling_factor: 3.0,
            thornfiddle_max_opening_percentage: 30.0,
            thornfiddle_min_opening_percentage: 5.0,
            thornfiddle_pixel_threshold: 5,
            thornfiddle_marked_color_rgb: [255, 215, 0],
            harmonic_max_harmonics: 12,
            harmonic_strength_multiplier: 2.0,
            harmonic_min_chain_length: 15,
            spectral_entropy_sigmoid_k: 20.0,
            spectral_entropy_sigmoid_c: 0.04,
        }
    }

    /// Validate configuration parameters
    ///
    /// Checks that all values are within reasonable ranges and paths exist.
    ///
    /// # Returns
    /// Ok if valid, Err with description if invalid
    pub fn validate(&self) -> Result<()> {
        // Check input path exists
        let input_path = PathBuf::from(&self.input_path);
        if !input_path.exists() {
            return Err(LeafComplexError::InvalidPath(input_path));
        }

        // Check kernel size is reasonable
        if self.opening_kernel_size == 0 {
            return Err(LeafComplexError::Config(
                "opening_kernel_size must be > 0".to_string(),
            ));
        }

        // Validate adaptive opening parameters
        if !(0.0..=100.0).contains(&self.adaptive_opening_max_density) {
            return Err(LeafComplexError::Config(
                "adaptive_opening_max_density must be between 0.0 and 100.0".to_string(),
            ));
        }
        
        if !(0.0..=50.0).contains(&self.adaptive_opening_max_percentage) {
            return Err(LeafComplexError::Config(
                "adaptive_opening_max_percentage must be between 0.0 and 50.0".to_string(),
            ));
        }
        
        if self.adaptive_opening_min_percentage <= 0.0 || 
           self.adaptive_opening_min_percentage >= self.adaptive_opening_max_percentage {
            return Err(LeafComplexError::Config(
                "adaptive_opening_min_percentage must be > 0.0 and < max_percentage".to_string(),
            ));
        }

        // Validate approximate entropy parameters
        if self.approximate_entropy_m < 1 {
            return Err(LeafComplexError::Config(
                "approximate_entropy_m must be >= 1".to_string(),
            ));
        }

        if self.approximate_entropy_r <= 0.0 {
            return Err(LeafComplexError::Config(
                "approximate_entropy_r must be > 0.0".to_string(),
            ));
        }
        
        // Validate thornfiddle parameters
        if !(0.0..=50.0).contains(&self.thornfiddle_max_opening_percentage) {
            return Err(LeafComplexError::Config(
                "thornfiddle_max_opening_percentage must be between 0.0 and 50.0".to_string(),
            ));
        }
        
        if self.thornfiddle_min_opening_percentage <= 0.0 || 
           self.thornfiddle_min_opening_percentage >= self.thornfiddle_max_opening_percentage {
            return Err(LeafComplexError::Config(
                "thornfiddle_min_opening_percentage must be > 0.0 and < max_percentage".to_string(),
            ));
        }
        
        if self.thornfiddle_pixel_threshold == 0 {
            return Err(LeafComplexError::Config(
                "thornfiddle_pixel_threshold must be > 0".to_string(),
            ));
        }
        
        // Validate harmonic parameters
        if self.harmonic_max_harmonics == 0 {
            return Err(LeafComplexError::Config(
                "harmonic_max_harmonics must be > 0".to_string(),
            ));
        }
        
        if self.harmonic_strength_multiplier <= 0.0 {
            return Err(LeafComplexError::Config(
                "harmonic_strength_multiplier must be > 0.0".to_string(),
            ));
        }
        
        if self.harmonic_min_chain_length == 0 {
            return Err(LeafComplexError::Config(
                "harmonic_min_chain_length must be > 0".to_string(),
            ));
        }
        
        // Validate spectral entropy sigmoid parameters
        if self.spectral_entropy_sigmoid_k <= 0.0 {
            return Err(LeafComplexError::Config(
                "spectral_entropy_sigmoid_k must be > 0.0".to_string(),
            ));
        }
        
        if self.spectral_entropy_sigmoid_c <= 0.0 {
            return Err(LeafComplexError::Config(
                "spectral_entropy_sigmoid_c must be > 0.0".to_string(),
            ));
        }

        // Create output directories
        let base_dir = PathBuf::from(&self.output_base_dir);
        let ec_dir = base_dir.join("EC");
        let mc_dir = base_dir.join("MC");

        fs::create_dir_all(&ec_dir).map_err(|e| {
            LeafComplexError::Io(io::Error::new(
                ErrorKind::Other,
                format!("Failed to create EC output directory: {}", e),
            ))
        })?;

        fs::create_dir_all(&mc_dir).map_err(|e| {
            LeafComplexError::Io(io::Error::new(
                ErrorKind::Other,
                format!("Failed to create MC output directory: {}", e),
            ))
        })?;

        Ok(())
    }

    /// Save configuration to a TOML file
    ///
    /// # Arguments
    /// * `path` - Path where the TOML file should be saved
    ///
    /// # Returns
    /// Ok if successful, Err otherwise
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self).map_err(|e| {
            LeafComplexError::Config(format!("Failed to serialize config: {}", e))
        })?;

        fs::write(path, content).map_err(|e| LeafComplexError::Io(e))?;

        Ok(())
    }
}
