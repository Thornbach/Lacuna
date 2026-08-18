use std::path::{Path, PathBuf};
use image::{RgbaImage, Rgba, imageops};
use eframe::egui;

// FIX: Config comes from the GUI-local crate::config (same type the rest of the app uses).
//      Analysis functions are still pulled from the library crate.
use crate::config::Config;
use leaf_complex_rust_lib::{
    morphology, shape_analysis, point_analysis,
    feature_extraction, thornfiddle, load_image,
};

// Import from state.rs, not defining our own
use crate::state::{AnalysisResult, SummaryStats};

pub struct AnalysisEngine;

impl AnalysisEngine {
    pub fn new() -> Self {
        Self
    }

    // Generate a small thumbnail texture for the strip at the bottom of the GUI.
    pub fn generate_thumbnail(
        &self,
        image_path: &Path,
        ctx: &egui::Context,
    ) -> Option<egui::TextureHandle> {
        let input_image = load_image(image_path).ok()?;

        let thumbnail_size = 120;
        let (width, height) = input_image.image.dimensions();
        let aspect_ratio = width as f32 / height as f32;

        let (thumb_width, thumb_height) = if aspect_ratio > 1.0 {
            (thumbnail_size, (thumbnail_size as f32 / aspect_ratio) as u32)
        } else {
            ((thumbnail_size as f32 * aspect_ratio) as u32, thumbnail_size)
        };

        let thumbnail = imageops::resize(
            &input_image.image,
            thumb_width,
            thumb_height,
            imageops::FilterType::Lanczos3,
        );

        Some(load_texture_from_image(
            ctx,
            &thumbnail,
            format!("{}_thumb", input_image.filename),
        ))
    }

    pub fn analyze_image(
        &self,
        image_path: &PathBuf,
        config: &Config,
        ctx: &egui::Context,
    ) -> Result<AnalysisResult, String> {
        println!("\n=== Starting Analysis ===");
        println!("Image: {:?}", image_path);

        let filename = image_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        // ------------------------------------------------------------------ //
        //  Load & optionally resize
        // ------------------------------------------------------------------ //
        let image = image::open(image_path)
            .map_err(|e| format!("Failed to load image: {}", e))?
            .to_rgba8();

        // Use Triangle (bilinear) filter, NOT Lanczos3.
        // Lanczos3 introduces ringing artifacts: it creates semi-transparent pixels
        // (alpha 1-127) scattered in the background region near the leaf edge.
        // mark_opened_regions then marks these as pink (orig[3]>0, opened[3]==0),
        // and trace_contour finds them first (they sit at lower x values than the
        // real leaf), tracing a tiny 10-15 point cluster instead of the full ~1500
        // point leaf outline. Triangle keeps background at exactly alpha=0, matching
        // the CLI's behaviour (which also uses Triangle via image_utils::resize_image).
        let processed_image = if let Some(dimensions) = config.resize_dimensions {
            image::imageops::resize(
                &image,
                dimensions[0],
                dimensions[1],
                image::imageops::FilterType::Triangle,  // must match CLI — see above
            )
        } else {
            image
        };

        // ------------------------------------------------------------------ //
        //  FIX: Guard against images with no transparent background.
        //
        //  Output images (_EC / _MC) and screenshots are fully opaque.
        //  If such an image slips through the workspace loader it would cause
        //  the alpha-based calculations to produce wrong results silently.
        //  We catch it here and return a clear error instead.
        // ------------------------------------------------------------------ //
        let transparent_pixel_count = processed_image
            .pixels()
            .filter(|p| p[3] < 128)
            .count();

        if transparent_pixel_count == 0 {
            return Err(format!(
                "'{}' has no transparent background. Only RGBA PNG images \
                 with a transparent background are supported. Make sure you \
                 are not accidentally loading an output image (_EC / _MC) or \
                 a screenshot.",
                filename
            ));
        }

        // ------------------------------------------------------------------ //
        //  Adaptive morphological opening  →  mark pink regions  →  clean
        // ------------------------------------------------------------------ //
        let adaptive_opening_kernel_size = calculate_adaptive_opening_kernel_size(
            &processed_image,
            config.adaptive_opening_max_density,
            config.adaptive_opening_max_percentage,
            config.adaptive_opening_min_percentage,
        );

        println!("Adaptive opening kernel size: {}", adaptive_opening_kernel_size);

        let opened_image =
            morphology::apply_opening(&processed_image, adaptive_opening_kernel_size)
                .map_err(|e| format!("Opening failed: {}", e))?;

        // Use the library's mark_opened_regions — identical to what the CLI uses.
        // (A local reimplementation that checked opened_pixel[3]==0 was wrong because
        //  apply_opening does not zero the alpha channel of eroded pixels.)
        let mut marked_image = morphology::mark_opened_regions(
            &processed_image,
            &opened_image,
            config.marked_region_color_rgb,
        );

        // NOTE: clean_thin_artifacts is intentionally NOT applied to marked_image here.
        // The EC marked region is a thin boundary ring around the leaf edge.
        // Applying neighbour-count filtering to it removes the entire contour,
        // leaving 0 EC features. It is applied only to the visual overlay below.

        // ------------------------------------------------------------------ //
        //  Build MC image
        // ------------------------------------------------------------------ //
        let mc_image = morphology::create_mc_with_com_component(
            &processed_image,
            &mut marked_image,
            config.marked_region_color_rgb,
        );

        println!("Created MC image");

        // ------------------------------------------------------------------ //
        //  Reference points
        // ------------------------------------------------------------------ //
        let ec_reference_point = point_analysis::get_reference_point(
            &processed_image,
            &marked_image,
            &config.reference_point_choice,
            config.marked_region_color_rgb,
        )
        .map_err(|e| format!("Failed to get EC reference point: {}", e))?;

        let mc_reference_point = point_analysis::get_mc_reference_point(
            &mc_image,
            &marked_image,
            &config.reference_point_choice,
            config.marked_region_color_rgb,
        )
        .map_err(|e| format!("Failed to get MC reference point: {}", e))?;

        println!("EC reference point: {:?}", ec_reference_point);
        println!("MC reference point: {:?}", mc_reference_point);

        // ------------------------------------------------------------------ //
        //  Contours from original (pre-filtered) images
        // ------------------------------------------------------------------ //
        let ec_contour_original =
            morphology::trace_contour(&marked_image, true, config.marked_region_color_rgb);
        let mc_contour_original =
            morphology::trace_contour(&mc_image, false, config.marked_region_color_rgb);

        println!("Original EC contour points: {}", ec_contour_original.len());
        println!("Original MC contour points: {}", mc_contour_original.len());

        // ------------------------------------------------------------------ //
        //  Shape metrics  (area, circularity, outline count)
        // ------------------------------------------------------------------ //
        let ec_area = shape_analysis::calculate_area(&marked_image);
        let ec_outline_count = ec_contour_original.len() as u32;
        let ec_circularity = shape_analysis::calculate_circularity_from_contour(
            ec_area,
            &ec_contour_original,
        );

        let mc_area = shape_analysis::calculate_area(&mc_image);
        let mc_outline_count = mc_contour_original.len() as u32;
        let mc_circularity = shape_analysis::calculate_circularity_from_contour(
            mc_area,
            &mc_contour_original,
        );

        println!(
            "EC metrics: Area={}, Outline={}, Circ={:.3}",
            ec_area, ec_outline_count, ec_circularity
        );
        println!(
            "MC metrics: Area={}, Outline={}, Circ={:.3}",
            mc_area, mc_outline_count, mc_circularity
        );

        // Length / width / shape index
        let (ec_length, ec_width, ec_shape_index) =
            shape_analysis::calculate_length_width_shape_index(
                &processed_image,
                config.marked_region_color_rgb,
            );

        let (mc_length, mc_width, mc_shape_index, mc_shorter_dimension) =
            shape_analysis::calculate_length_width_shape_index_with_shorter(
                &mc_image,
                config.marked_region_color_rgb,
            );

        println!(
            "EC shape: Length={:.1}, Width={:.1}, Index={:.3}",
            ec_length, ec_width, ec_shape_index
        );
        println!(
            "MC shape: Length={:.1}, Width={:.1}, Index={:.3}, Shorter={:.1}",
            mc_length, mc_width, mc_shape_index, mc_shorter_dimension
        );

        // ------------------------------------------------------------------ //
        //  Feature extraction from original contours
        // ------------------------------------------------------------------ //
        let initial_ec_features = feature_extraction::generate_features(
            ec_reference_point,
            &ec_contour_original,
            &processed_image,
            Some(&marked_image),
            config.marked_region_color_rgb,
            true,
        )
        .map_err(|e| format!("EC feature extraction failed: {}", e))?;

        let initial_mc_features = feature_extraction::generate_features(
            mc_reference_point,
            &mc_contour_original,
            &mc_image,
            None,
            config.marked_region_color_rgb,
            false,
        )
        .map_err(|e| format!("MC feature extraction failed: {}", e))?;

        println!("Initial EC features: {}", initial_ec_features.len());
        println!("Initial MC features: {}", initial_mc_features.len());

        // ------------------------------------------------------------------ //
        //  Petiole filtering
        // ------------------------------------------------------------------ //
        let (ec_features, _ec_petiole_info) = thornfiddle::filter_petiole_from_ec_features(
            &initial_ec_features,
            config.enable_petiole_filter_ec,
            config.petiole_remove_completely,
            1.0,
            config.enable_pink_threshold_filter,
            config.pink_threshold_value,
        );

        let (mc_features, _mc_petiole_info) = thornfiddle::filter_petiole_from_ec_features(
            &initial_mc_features,
            config.enable_petiole_filter_mc,
            config.petiole_remove_completely,
            1.0,
            false,
            0.0,
        );

        println!(
            "After filtering - EC features: {}, MC features: {}",
            ec_features.len(),
            mc_features.len()
        );

        // Filtered contours that match the filtered feature indices
        let ec_contour_filtered: Vec<(u32, u32)> = ec_features
            .iter()
            .filter_map(|f| ec_contour_original.get(f.point_index))
            .copied()
            .collect();

        let mc_contour_filtered: Vec<(u32, u32)> = mc_features
            .iter()
            .filter_map(|f| mc_contour_original.get(f.point_index))
            .copied()
            .collect();

        println!(
            "Filtered contours - EC: {}, MC: {}",
            ec_contour_filtered.len(),
            mc_contour_filtered.len()
        );

        if ec_contour_filtered.len() != ec_features.len() {
            eprintln!(
                "WARNING: EC contour/feature mismatch! Contour={}, Features={}",
                ec_contour_filtered.len(),
                ec_features.len()
            );
        }
        if mc_contour_filtered.len() != mc_features.len() {
            eprintln!(
                "WARNING: MC contour/feature mismatch! Contour={}, Features={}",
                mc_contour_filtered.len(),
                mc_features.len()
            );
        }

        // ------------------------------------------------------------------ //
        //  Thornfiddle image (dynamic opening)
        // ------------------------------------------------------------------ //
        let dynamic_opening_percentage = shape_analysis::calculate_dynamic_opening_percentage(
            mc_shape_index,
            config.thornfiddle_max_opening_percentage,
            config.thornfiddle_min_opening_percentage,
        );

        let dynamic_kernel_size =
            ((dynamic_opening_percentage / 100.0) * mc_shorter_dimension).round() as u32;
        let dynamic_kernel_size = dynamic_kernel_size.max(1);

        println!(
            "Dynamic thornfiddle: MC Shape Index {:.3} -> {:.1}% -> {} px kernel",
            mc_shape_index, dynamic_opening_percentage, dynamic_kernel_size
        );

        let thornfiddle_image = morphology::create_thornfiddle_image(
            &mc_image,
            dynamic_kernel_size,
            config.thornfiddle_marked_color_rgb,
        )
        .map_err(|e| format!("Failed to create thornfiddle image: {}", e))?;

        // ------------------------------------------------------------------ //
        //  Harmonic analysis  (MC only — EC harmonic is not calculated,
        //  EC uses Approximate Entropy on the pink-pixel path instead)
        // ------------------------------------------------------------------ //
        let mc_circumference = thornfiddle::calculate_leaf_circumference(&mc_contour_original);

        let mc_harmonic_result = thornfiddle::calculate_thornfiddle_path_harmonic(
            &mc_features,
            mc_circumference,
            &thornfiddle_image,
            mc_reference_point,
            &mc_contour_original,
            config.thornfiddle_marked_color_rgb,
            config.thornfiddle_pixel_threshold,
            config.harmonic_min_chain_length,
            config.harmonic_strength_multiplier,
            config.harmonic_max_harmonics,
        );

        println!(
            "MC harmonic chains: {}",
            mc_harmonic_result.valid_chain_count
        );

        // ------------------------------------------------------------------ //
        //  Finalise features with thornfiddle_path values
        //  EC: harmonic stays at default (0); only ApEn is used for EC.
        //  MC: harmonic values from mc_harmonic_result.
        // ------------------------------------------------------------------ //
        let mut ec_features_final = ec_features.clone();
        for feature in ec_features_final.iter_mut() {
            feature.thornfiddle_path = thornfiddle::calculate_thornfiddle_path(feature);
            // feature.thornfiddle_path_harmonic left at default — not used for EC
        }

        let mut mc_features_final = mc_features.clone();
        for (i, feature) in mc_features_final.iter_mut().enumerate() {
            if let Some(&harmonic_value) = mc_harmonic_result.harmonic_values.get(i) {
                feature.thornfiddle_path_harmonic = harmonic_value;
            }
            feature.thornfiddle_path = thornfiddle::calculate_thornfiddle_path(feature);
        }

        // ------------------------------------------------------------------ //
        //  Entropy metrics
        // ------------------------------------------------------------------ //
        let mc_spectral_entropy =
            thornfiddle::calculate_spectral_entropy_from_harmonic_thornfiddle_path(
                &mc_features_final,
                mc_harmonic_result.valid_chain_count,
                config.thornfiddle_smoothing_strength,
                config.spectral_entropy_sigmoid_k,
                config.spectral_entropy_sigmoid_c,
            )
            .0; // only the entropy value; discard the smoothed-path vector

        let ec_approximate_entropy = thornfiddle::calculate_approximate_entropy_from_pink_path(
            &ec_features_final,
            config.approximate_entropy_m,
            config.approximate_entropy_r,
        );

        println!("EC Approximate Entropy: {:.6}", ec_approximate_entropy);
        println!("MC Spectral Entropy:    {:.6}", mc_spectral_entropy);

        // ------------------------------------------------------------------ //
        //  Graph data
        // ------------------------------------------------------------------ //
        let ec_data: Vec<(f64, f64)> = ec_features_final
            .iter()
            .enumerate()
            .map(|(i, f)| (i as f64, f.diego_path_pink.unwrap_or(0) as f64))
            .collect();

        let mc_data: Vec<(f64, f64)> = mc_features_final
            .iter()
            .enumerate()
            .map(|(i, f)| (i as f64, f.thornfiddle_path_harmonic))
            .collect();

        println!(
            "Graph data - EC: {} points (diego_path_pink), MC: {} points",
            ec_data.len(),
            mc_data.len()
        );

        // ------------------------------------------------------------------ //
        //  Pre-build CSV strings (cached for instant export)
        //  Built here once so export is a plain fs::write() with no per-row work.
        // ------------------------------------------------------------------ //
        let ec_csv = {
            let mut s = String::with_capacity(ec_data.len() * 20 + 32);
            s.push_str("Point_Index,Pink_Pixels\n");
            for (x, y) in &ec_data {
                s.push_str(&format!("{},{:.0}\n", *x as usize, y));
            }
            s
        };

        let mc_csv = {
            let mut s = String::with_capacity(mc_data.len() * 30 + 32);
            s.push_str("Point_Index,Geodesic_MC_H\n");
            for (x, y) in &mc_data {
                s.push_str(&format!("{},{:.6}\n", *x as usize, y));
            }
            s
        };

        // ------------------------------------------------------------------ //
        //  Textures for the GUI viewer
        // ------------------------------------------------------------------ //
        // Clean thin artefacts from the overlay only — purely cosmetic, does not
        // affect the marked_image used for contour tracing and feature extraction.
        let marked_image_display = clean_thin_artifacts(&marked_image, config.marked_region_color_rgb);
        let ec_overlay = create_transparent_overlay(&marked_image_display, &[255, 0, 255]);
        let mc_overlay = create_transparent_overlay(&thornfiddle_image, &[255, 215, 0]);

        let original_texture = load_texture_from_image(
            ctx,
            &processed_image,
            format!("{}_original", filename),
        );
        let ec_texture =
            load_texture_from_image(ctx, &ec_overlay, format!("{}_ec", filename));
        let mc_texture =
            load_texture_from_image(ctx, &mc_overlay, format!("{}_mc", filename));

        // ------------------------------------------------------------------ //
        //  Summary
        // ------------------------------------------------------------------ //
        let summary = SummaryStats {
            ec_length,
            ec_width,
            ec_shape_index,
            ec_circularity,
            ec_spectral_entropy: ec_approximate_entropy,
            ec_area,
            ec_outline_count,
            mc_length,
            mc_width,
            mc_shape_index,
            mc_circularity,
            mc_spectral_entropy,
            mc_area,
            mc_outline_count,
        };

        println!("=== Analysis Complete ===");
        println!("Final verification:");
        println!(
            "  EC: {} data points, {} contour points, {} features",
            ec_data.len(),
            ec_contour_filtered.len(),
            ec_features_final.len()
        );
        println!(
            "  MC: {} data points, {} contour points, {} features",
            mc_data.len(),
            mc_contour_filtered.len(),
            mc_features_final.len()
        );
        println!();

        Ok(AnalysisResult {
            ec_data,
            mc_data,
            summary,
            ec_image_texture: Some(ec_texture),
            mc_image_texture: Some(mc_texture),
            original_texture: Some(original_texture),
            ec_contour: ec_contour_filtered,
            mc_contour: mc_contour_filtered,
            ec_features: ec_features_final,
            mc_features: mc_features_final,
            ec_reference_point,
            mc_reference_point,
            ec_csv,
            mc_csv,
        })
    }
}

// ============================================================================
//  Private helpers
// ============================================================================

/// Adaptive opening kernel size from pixel density.
fn calculate_adaptive_opening_kernel_size(
    image: &RgbaImage,
    max_density: f64,
    max_percentage: f64,
    min_percentage: f64,
) -> u32 {
    let (width, height) = image.dimensions();
    let total_pixels = (width * height) as f64;

    let non_transparent_count = image.pixels().filter(|p| p[3] > 0).count() as f64;
    let non_transparent_percentage = (non_transparent_count / total_pixels) * 100.0;

    let opening_percentage = if non_transparent_percentage >= max_density {
        max_percentage
    } else {
        let scaling_factor = non_transparent_percentage / max_density;
        min_percentage + scaling_factor * (max_percentage - min_percentage)
    };

    let image_dimension = std::cmp::min(width, height) as f64;
    let kernel_size =
        ((opening_percentage / 100.0) * image_dimension).round() as u32;
    let adaptive_kernel_size = kernel_size.max(1);

    println!(
        "Adaptive opening: {:.1}% density -> {:.1}% opening -> {} px kernel",
        non_transparent_percentage, opening_percentage, adaptive_kernel_size
    );

    adaptive_kernel_size
}

/// Mark pixels that were opaque before opening but are now transparent.
// mark_opened_regions removed — use morphology::mark_opened_regions (library) instead

/// Keep only pixels of a specific colour; everything else becomes transparent.
fn create_transparent_overlay(image: &RgbaImage, color_to_keep: &[u8; 3]) -> RgbaImage {
    let (width, height) = image.dimensions();
    let mut overlay = RgbaImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);

            if pixel[3] > 0
                && pixel[0] == color_to_keep[0]
                && pixel[1] == color_to_keep[1]
                && pixel[2] == color_to_keep[2]
            {
                overlay.put_pixel(x, y, *pixel);
            } else {
                overlay.put_pixel(x, y, Rgba([0, 0, 0, 0]));
            }
        }
    }

    overlay
}

/// Upload a raw RGBA image as an egui texture.
fn load_texture_from_image(
    ctx: &egui::Context,
    image: &RgbaImage,
    name: String,
) -> egui::TextureHandle {
    let (width, height) = image.dimensions();
    let image_data: Vec<u8> = image.as_raw().clone();

    let color_image = egui::ColorImage::from_rgba_unmultiplied(
        [width as usize, height as usize],
        &image_data,
    );

    ctx.load_texture(name, color_image, egui::TextureOptions::default())
}

/// Remove single-pixel-wide artefact lines from the marked (pink) region.
///
/// Any pink pixel with ≤ 2 pink 8-neighbours is considered part of a thin
/// line artefact and is cleared.  Two passes ensure isolated pixels and
/// one-pixel-wide lines are both removed without touching real edge regions.
fn clean_thin_artifacts(marked_image: &RgbaImage, marked_color: [u8; 3]) -> RgbaImage {
    let (width, height) = marked_image.dimensions();
    let mut cleaned = marked_image.clone();

    let is_pink = |x: u32, y: u32| -> bool {
        if x >= width || y >= height {
            return false;
        }
        let pixel = marked_image.get_pixel(x, y);
        pixel[0] == marked_color[0]
            && pixel[1] == marked_color[1]
            && pixel[2] == marked_color[2]
            && pixel[3] > 0
    };

    let count_pink_neighbors = |x: u32, y: u32| -> usize {
        let mut count = 0;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && ny >= 0 && is_pink(nx as u32, ny as u32) {
                    count += 1;
                }
            }
        }
        count
    };

    let mut to_remove = Vec::new();
    for y in 0..height {
        for x in 0..width {
            if is_pink(x, y) && count_pink_neighbors(x, y) <= 2 {
                to_remove.push((x, y));
            }
        }
    }

    let removed = to_remove.len();
    for (x, y) in to_remove {
        cleaned.put_pixel(x, y, Rgba([0, 0, 0, 0]));
    }

    println!("Cleaned {} thin artefact pixels from marked image", removed);
    cleaned
}