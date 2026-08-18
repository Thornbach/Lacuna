// src/lib.rs - Library interface for LeafComplexR

pub mod config;
pub mod errors;
pub mod feature_extraction;
pub mod image_io;
pub mod image_utils;
pub mod morphology;
pub mod path_algorithms;
pub mod pipeline;
pub mod point_analysis;
pub mod output;
pub mod thornfiddle;
pub mod shape_analysis;
pub mod gui_api;

// Re-export commonly used types and functions
pub use errors::{LeafComplexError, Result};
pub use config::Config;
pub use pipeline::process_image;
pub use image_io::{InputImage, load_image, save_image};

// Egui-free analysis entry point for embedding in other front-ends.
pub use gui_api::{analyze, analyze_rgba, MorphMetrics, MorphOverlay, MorphResult};

// Re-export shape analysis functions
pub use shape_analysis::{
    analyze_shape,
    analyze_shape_comprehensive,
    calculate_biological_dimensions,
    calculate_biological_dimensions_fast,
    calculate_bounding_box_dimensions,
    calculate_outline_count,
    calculate_outline_count_from_contour,
    calculate_circularity_from_contour,
    calculate_area,
    calculate_circularity,
    calculate_length_width_shape_index,
    calculate_length_width_shape_index_with_shorter,
    calculate_dynamic_opening_percentage,
    calculate_shape_index,
};

// Re-export thornfiddle analysis functions
pub use thornfiddle::{
    // Spectral entropy functions
    calculate_spectral_entropy_from_contour,
    calculate_spectral_entropy_from_pink_path,
    calculate_spectral_entropy_from_harmonic_thornfiddle_path,
    
    // Entropy and complexity functions
    calculate_approximate_entropy_from_pink_path,
    calculate_edge_feature_density,
    
    // Harmonic thornfiddle functions
    calculate_thornfiddle_path_harmonic,
    calculate_leaf_circumference,
    extract_harmonic_thornfiddle_path_signal,
    
    // Summary creation
    HarmonicResult,
    
    // Signal extraction utilities
    extract_pink_path_signal,
    extract_thornfiddle_path_signal,
    
    // Filtering functions
    filter_petiole_from_ec_features,
    detect_petiole_sequence,
    apply_petiole_filter,
    apply_pink_threshold_filter,
    
    // Basic thornfiddle calculations
    calculate_thornfiddle_multiplier,
    calculate_thornfiddle_path,
};

// Re-export morphology functions
pub use morphology::{
    trace_contour,
    apply_opening,
    calculate_center_of_mass,
    create_thornfiddle_image,
    create_mc_with_com_component,
};

// Re-export point analysis functions
pub use point_analysis::{
    calculate_emerge_point,
    get_reference_point,
    get_mc_reference_point,
};
