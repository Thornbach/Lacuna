// config.rs - Configuration for LeafComplexR GUI
// This wraps the library config with GUI-specific defaults

use leaf_complex_rust_lib::config::ReferencePointChoice;

/// GUI Configuration - matches library Config structure
#[derive(Debug, Clone)]
pub struct Config {
    // Image Processing
    pub opening_kernel_size: u32,
    /// Optional resize dimensions [width, height] for GUI processing
    pub resize_dimensions: Option<[u32; 2]>,
    pub marked_region_color_rgb: [u8; 3],
    pub reference_point_choice: ReferencePointChoice,
    
    // Adaptive Opening
    pub adaptive_opening_max_density: f64,
    pub adaptive_opening_max_percentage: f64,
    pub adaptive_opening_min_percentage: f64,
    
    // Petiole Filtering
    pub enable_petiole_filter_ec: bool,
    pub enable_petiole_filter_mc: bool,
    pub petiole_remove_completely: bool,
    
    // Pink Threshold Filtering
    pub enable_pink_threshold_filter: bool,
    pub pink_threshold_value: f64,
    
    // Thornfiddle Parameters
    pub thornfiddle_smoothing_strength: f64,
    pub thornfiddle_max_opening_percentage: f64,
    pub thornfiddle_min_opening_percentage: f64,
    pub thornfiddle_pixel_threshold: u32,
    pub thornfiddle_marked_color_rgb: [u8; 3],
    
    // Harmonic Enhancement
    pub harmonic_max_harmonics: usize,
    pub harmonic_strength_multiplier: f64,
    pub harmonic_min_chain_length: usize,
    
    // Entropy Parameters
    pub approximate_entropy_m: usize,
    pub approximate_entropy_r: f64,
    pub spectral_entropy_sigmoid_k: f64,
    pub spectral_entropy_sigmoid_c: f64,
    
    // EC Scaling
    pub ec_scaling_factor: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            opening_kernel_size: 9,
            resize_dimensions: Some([512, 512]),
            marked_region_color_rgb: [255, 0, 255],
            reference_point_choice: ReferencePointChoice::Com,
            
            adaptive_opening_max_density: 75.0,
            adaptive_opening_max_percentage: 15.0,
            adaptive_opening_min_percentage: 1.0,
            
            enable_petiole_filter_ec: true,
            enable_petiole_filter_mc: false,
            petiole_remove_completely: true,
            
            enable_pink_threshold_filter: true,
            pink_threshold_value: 3.0,
            
            thornfiddle_smoothing_strength: 2.0,
            thornfiddle_max_opening_percentage: 30.0,
            thornfiddle_min_opening_percentage: 5.0,
            thornfiddle_pixel_threshold: 5,
            thornfiddle_marked_color_rgb: [255, 215, 0],
            
            harmonic_max_harmonics: 12,
            harmonic_strength_multiplier: 2.0,
            harmonic_min_chain_length: 15,
            
            approximate_entropy_m: 2,
            approximate_entropy_r: 0.2,
            spectral_entropy_sigmoid_k: 20.0,
            spectral_entropy_sigmoid_c: 0.04,
            
            ec_scaling_factor: 3.0,
        }
    }
}

impl Config {
    /// Load config from a file (placeholder - returns default for now)
    pub fn load(_path: &std::path::Path) -> Result<Self, String> {
        // For simplicity, just return default config
        // Full implementation would need serde + toml dependencies
        Ok(Self::default())
    }
    
    /// Save config to a file (placeholder)
    pub fn save(&self, _path: &std::path::Path) -> Result<(), String> {
        // For simplicity, this is a no-op
        // Full implementation would need serde + toml dependencies
        Ok(())
    }
}