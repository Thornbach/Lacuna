// src/pipeline.rs - Main processing pipeline for EC/MC analysis

use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::errors::{LeafComplexError, Result};
use crate::feature_extraction::generate_features;
use crate::image_io::{InputImage, save_image};
use crate::image_utils::resize_image;
use crate::morphology::{
    apply_opening, mark_opened_regions, trace_contour, 
    create_mc_with_com_component, create_thornfiddle_image
};
use crate::output::{write_ec_csv, write_mc_csv, create_summary};
use crate::point_analysis::{get_reference_point, get_mc_reference_point};
use crate::shape_analysis::{
    analyze_shape_comprehensive, calculate_length_width_shape_index, 
    calculate_length_width_shape_index_with_shorter, calculate_dynamic_opening_percentage
};
use crate::thornfiddle;

/// Calculate adaptive opening kernel size based on pixel density
///
/// # Arguments
/// * `image` - Image to analyze
/// * `max_density` - Density threshold for max opening
/// * `max_percentage` - Maximum opening percentage at high density
/// * `min_percentage` - Minimum opening percentage at low density
///
/// # Returns
/// Calculated kernel size in pixels
fn calculate_adaptive_opening_kernel_size(
    image: &image::RgbaImage,
    max_density: f64,
    max_percentage: f64,
    min_percentage: f64,
) -> u32 {
    let (width, height) = image.dimensions();
    let total_pixels = (width * height) as f64;
    
    // Count non-transparent pixels
    let mut non_transparent_count = 0;
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);
            if pixel[3] > 0 {
                non_transparent_count += 1;
            }
        }
    }
    
    // Calculate percentage of non-transparent pixels
    let non_transparent_percentage = (non_transparent_count as f64 / total_pixels) * 100.0;
    
    // Calculate opening percentage using linear scaling
    let opening_percentage = if non_transparent_percentage >= max_density {
        max_percentage
    } else {
        let scaling_factor = non_transparent_percentage / max_density;
        min_percentage + scaling_factor * (max_percentage - min_percentage)
    };
    
    // Use the smaller dimension to avoid overly large kernels
    let image_dimension = std::cmp::min(width, height) as f64;
    let kernel_size = ((opening_percentage / 100.0) * image_dimension).round() as u32;
    let adaptive_kernel_size = kernel_size.max(1);
    
    println!("Adaptive opening: {:.1}% density -> {:.1}% opening -> {} px kernel", 
             non_transparent_percentage, opening_percentage, adaptive_kernel_size);
    
    adaptive_kernel_size
}

/// Process a single image through the complete EC/MC analysis pipeline
///
/// # Pipeline Steps
/// 1. Resize image (if configured)
/// 2. Apply adaptive morphological opening for EC region marking
/// 3. Create MC image by removing small components
/// 4. Calculate shape metrics (EC and MC)
/// 5. Create Thornfiddle image with golden lobes
/// 6. Calculate reference points (separate for EC and MC)
/// 7. Extract contours and generate features
/// 8. Apply filtering (petiole, threshold)
/// 9. Calculate harmonic enhancements
/// 10. Compute entropy metrics
/// 11. Write output CSVs
///
/// # Arguments
/// * `input_image` - Loaded input image with metadata
/// * `config` - Configuration parameters
/// * `debug` - Enable debug output and intermediate image saving
///
/// # Returns
/// Ok if successful, Err with description if failed
pub fn process_image(
    input_image: InputImage,
    config: &Config,
    debug: bool,
    root_dir: &Path,
) -> Result<()> {
    let InputImage { image, path, filename } = input_image;

    let subfolder = path.parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("root");

    // Relative path from (and including) the root directory name.
    // e.g. root_dir = /data/leaves, image = /data/leaves/sp1/leaf.png
    //   -> relative_path = "leaves/sp1/leaf.png"
    let relative_path = {
        let root_name = root_dir.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let within_root = path.strip_prefix(root_dir).unwrap_or(&path);
        if within_root == path.as_path() {
            // Could not strip root — fall back to just the filename
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        } else {
            format!("{}/{}", root_name, within_root.display())
        }
    };
    
    // Step 1: Resize if configured
    let processed_image = if let Some(dimensions) = config.resize_dimensions {
        resize_image(&image, dimensions)
    } else {
        image
    };
    
    // Step 2: Calculate adaptive opening kernel size
    let adaptive_opening_kernel_size = calculate_adaptive_opening_kernel_size(
        &processed_image,
        config.adaptive_opening_max_density,
        config.adaptive_opening_max_percentage,
        config.adaptive_opening_min_percentage,
    );
    
    // Apply morphological opening
    let opened_image = apply_opening(&processed_image, adaptive_opening_kernel_size)?;
    
    // Mark opened regions (pink)
    let mut marked_image = mark_opened_regions(
        &processed_image,
        &opened_image,
        config.marked_region_color_rgb,
    );

    // Step 3: Create MC image (formerly LMC)
    let mc_image = create_mc_with_com_component(
        &processed_image,
        &mut marked_image, 
        config.marked_region_color_rgb
    );
    
    // Step 4: Calculate shape metrics
    println!("Calculating EC shape metrics...");
    let (ec_length, ec_width, ec_shape_index) = calculate_length_width_shape_index(
        &processed_image, 
        config.marked_region_color_rgb
    );
    
    if debug {
        println!("EC Shape: Length={:.1}, Width={:.1}, Index={:.3}", 
                 ec_length, ec_width, ec_shape_index);
    }
    
    println!("Calculating MC shape metrics...");
    let (mc_length, mc_width, mc_shape_index, mc_shorter_dimension) = 
        calculate_length_width_shape_index_with_shorter(
            &mc_image, 
            config.marked_region_color_rgb
        );
    
    if debug {
        println!("MC Shape: Length={:.1}, Width={:.1}, Index={:.3}, Shorter={:.1}", 
                 mc_length, mc_width, mc_shape_index, mc_shorter_dimension);
    }
    
    // Step 5: Calculate dynamic opening percentage and create Thornfiddle image
    let dynamic_opening_percentage = calculate_dynamic_opening_percentage(
        mc_shape_index,
        config.thornfiddle_max_opening_percentage,
        config.thornfiddle_min_opening_percentage,
    );
    
    let dynamic_kernel_size = ((dynamic_opening_percentage / 100.0) * mc_shorter_dimension)
        .round() as u32;
    let dynamic_kernel_size = dynamic_kernel_size.max(1);
    
    println!("Dynamic thornfiddle: MC Shape Index {:.3} -> {:.1}% -> {} px kernel", 
             mc_shape_index, dynamic_opening_percentage, dynamic_kernel_size);
    
    let thornfiddle_image = create_thornfiddle_image(
        &mc_image,
        dynamic_kernel_size,
        config.thornfiddle_marked_color_rgb,
    )?;
    
    // Calculate comprehensive shape metrics
    let (area, ec_circularity, _, _, outline_count, _) = 
        analyze_shape_comprehensive(&processed_image, config.marked_region_color_rgb);
    
    if debug {
        println!("Shape metrics: Area={}, Outline={}, EC_Circularity={:.6}", 
                 area, outline_count, ec_circularity);
    }
    
    // Save debug images if requested
    if debug {
        let debug_dir = PathBuf::from(&config.output_base_dir).join("debug");
        std::fs::create_dir_all(&debug_dir).map_err(|e| LeafComplexError::Io(e))?;
        
        save_image(&marked_image, debug_dir.join(format!("{}_marked.png", filename)))?;
        save_image(&thornfiddle_image, debug_dir.join(format!("{}_thornfiddle.png", filename)))?;
    }
    
    // Step 6: Calculate reference points (separate for EC and MC)
    let ec_reference_point = get_reference_point(
        &processed_image,
        &marked_image,
        &config.reference_point_choice,
        config.marked_region_color_rgb,
    )?;
    
    let mc_reference_point = get_mc_reference_point(
        &mc_image,
        &marked_image,
        &config.reference_point_choice,
        config.marked_region_color_rgb,
    )?;
    
    if debug {
        println!("EC reference point: {:?}", ec_reference_point);
        println!("MC reference point: {:?}", mc_reference_point);
    }
    
    // Step 7: EC Analysis (pink regions are OPAQUE)
    let ec_contour = trace_contour(
        &marked_image,
        true, // is_pink_opaque = true for EC
        config.marked_region_color_rgb,
    );
    
    // Generate initial EC features
    let initial_ec_features = generate_features(
        ec_reference_point,
        &ec_contour,
        &processed_image,
        Some(&marked_image),
        config.marked_region_color_rgb,
        true, // is_ec = true
    )?;
    
    // Apply petiole filtering to EC features
    let (ec_features, petiole_info) = thornfiddle::filter_petiole_from_ec_features(
        &initial_ec_features,
        config.enable_petiole_filter_ec,
        config.petiole_remove_completely,
        1.0, // threshold for petiole detection
        config.enable_pink_threshold_filter,
        config.pink_threshold_value,
    );
    
    // Calculate harmonic values for EC
    let ec_circumference = thornfiddle::calculate_leaf_circumference(&ec_contour);
    let ec_harmonic_result = thornfiddle::calculate_thornfiddle_path_harmonic(
        &ec_features,
        ec_circumference,
        &thornfiddle_image,
        ec_reference_point,
        &ec_contour,
        config.thornfiddle_marked_color_rgb,
        config.thornfiddle_pixel_threshold,
        config.harmonic_min_chain_length,
        config.harmonic_strength_multiplier,
        config.harmonic_max_harmonics,
    );
    
    // Update EC features with harmonic and thornfiddle values
    let mut ec_features_final = ec_features;
    for (i, feature) in ec_features_final.iter_mut().enumerate() {
        if let Some(&harmonic_value) = ec_harmonic_result.harmonic_values.get(i) {
            feature.thornfiddle_path_harmonic = harmonic_value;
        }
        // Calculate thornfiddle_path
        feature.thornfiddle_path = thornfiddle::calculate_thornfiddle_path(feature);
    }
    
    if debug {
        println!("EC contour points: {}", ec_contour.len());
        if let Some(ref indices) = petiole_info {
            println!("Petiole detected: {} points", indices.len());
        }
        println!("EC harmonic chains: {}", ec_harmonic_result.valid_chain_count);
    }
    
    // Step 8: MC Analysis (pink regions are TRANSPARENT)
    let mc_contour = trace_contour(
        &mc_image,
        false, // is_pink_opaque = false for MC
        config.marked_region_color_rgb,
    );
    
    let mc_features = generate_features(
        mc_reference_point,
        &mc_contour,
        &mc_image,
        None, // No marked image needed for MC
        config.marked_region_color_rgb,
        false, // is_ec = false
    )?;
    
    // Calculate harmonic values for MC
    let mc_circumference = thornfiddle::calculate_leaf_circumference(&mc_contour);
    let mc_harmonic_result = thornfiddle::calculate_thornfiddle_path_harmonic(
        &mc_features,
        mc_circumference,
        &thornfiddle_image,
        mc_reference_point,
        &mc_contour,
        config.thornfiddle_marked_color_rgb,
        config.thornfiddle_pixel_threshold,
        config.harmonic_min_chain_length,
        config.harmonic_strength_multiplier,
        config.harmonic_max_harmonics,
    );
    
    // Update MC features with harmonic and thornfiddle values
    let mut mc_features_final = mc_features;
    for (i, feature) in mc_features_final.iter_mut().enumerate() {
        if let Some(&harmonic_value) = mc_harmonic_result.harmonic_values.get(i) {
            feature.thornfiddle_path_harmonic = harmonic_value;
        }
        // Calculate thornfiddle_path
        feature.thornfiddle_path = thornfiddle::calculate_thornfiddle_path(feature);
    }
    
    if debug {
        println!("MC contour points: {}", mc_contour.len());
        println!("MC harmonic chains: {}", mc_harmonic_result.valid_chain_count);
    }
    
    // Step 9: Calculate entropy metrics
    let mc_spectral_entropy = thornfiddle::calculate_spectral_entropy_from_harmonic_thornfiddle_path(
        &mc_features_final,
        mc_harmonic_result.valid_chain_count,
        config.thornfiddle_smoothing_strength,
        config.spectral_entropy_sigmoid_k,
        config.spectral_entropy_sigmoid_c,
    ).0; // We only need the entropy value, not the smoothed path
    
    let ec_approximate_entropy = thornfiddle::calculate_approximate_entropy_from_pink_path(
        &ec_features_final,
        config.approximate_entropy_m,
        config.approximate_entropy_r,
    );
    
    if debug {
        println!("MC Spectral Entropy: {:.6}", mc_spectral_entropy);
        println!("EC Approximate Entropy: {:.6}", ec_approximate_entropy);
    }
    
    // Step 10: Write output CSVs
    write_ec_csv(&ec_features_final, &config.output_base_dir, &filename)?;
    write_mc_csv(&mc_features_final, &config.output_base_dir, &filename)?;
    
    // Step 11: Create summary
    create_summary(
        &config.output_base_dir,
        &filename,
        subfolder,
        &relative_path,
        mc_spectral_entropy,
        ec_approximate_entropy,
        ec_length,
        mc_length,
        ec_width,
        mc_width,
        ec_shape_index,
        mc_shape_index,
        outline_count,
        mc_harmonic_result.valid_chain_count,
    )?;
    
    if debug {
        println!("Analysis complete for: {}", filename);
    }
    
    Ok(())
}