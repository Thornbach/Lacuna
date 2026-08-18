//! Egui-free, front-end-agnostic analysis entry point.
//!
//! Mirrors the desktop GUI's `analyze_image` but returns plain data — raw RGBA
//! overlays (`Vec<u8>` + dimensions), scalar metrics and plot points — instead
//! of building egui textures. This lets any front-end consume the analysis
//! without sharing `image`-crate types across version boundaries (e.g. a host
//! app on `image` 0.25 calling this `image` 0.24 library).

use std::path::Path;

use image::{Rgba, RgbaImage};

use crate::config::Config;
use crate::{feature_extraction, morphology, point_analysis, shape_analysis, thornfiddle};

/// A straight (non-premultiplied) RGBA overlay buffer.
pub struct MorphOverlay {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>, // width * height * 4
}

/// Scalar EC/MC complexity + shape metrics for one leaf.
#[derive(Clone, Default)]
pub struct MorphMetrics {
    pub ec_length: f64,
    pub ec_width: f64,
    pub ec_shape_index: f64,
    pub ec_circularity: f64,
    pub ec_approximate_entropy: f64,
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

/// Full analysis output for one image.
pub struct MorphResult {
    pub metrics: MorphMetrics,
    /// Edge-complexity curve: (point index, pink-pixel count).
    pub ec_data: Vec<(f64, f64)>,
    /// Macro-shape curve: (point index, harmonic thornfiddle value).
    pub mc_data: Vec<(f64, f64)>,
    pub original: MorphOverlay,
    pub ec_overlay: MorphOverlay,
    pub mc_overlay: MorphOverlay,
    pub ec_csv: String,
    pub mc_csv: String,
}

fn to_overlay(img: &RgbaImage) -> MorphOverlay {
    let (width, height) = img.dimensions();
    MorphOverlay { width, height, rgba: img.as_raw().clone() }
}

/// Run the full EC/MC analysis on a single RGBA-PNG leaf image.
///
/// Returns an error string on load failure or if the image has no transparent
/// background (which would silently break the alpha-based calculations).
pub fn analyze(image_path: &Path, config: &Config) -> Result<MorphResult, String> {
    let image = image::open(image_path)
        .map_err(|e| format!("Failed to load image: {e}"))?
        .to_rgba8();
    analyze_core(image, config)
}

/// Same as [`analyze`] but from a raw RGBA buffer already in memory — no disk
/// round-trip. `rgba` must be exactly `width * height * 4` bytes. Lets a host
/// pipeline feed in-memory leaf cutouts straight in.
pub fn analyze_rgba(
    rgba: &[u8],
    width: u32,
    height: u32,
    config: &Config,
) -> Result<MorphResult, String> {
    let image = RgbaImage::from_raw(width, height, rgba.to_vec())
        .ok_or_else(|| "analyze_rgba: buffer is not width*height*4 bytes".to_string())?;
    analyze_core(image, config)
}

fn analyze_core(image: RgbaImage, config: &Config) -> Result<MorphResult, String> {
    // ── optionally resize ────────────────────────────────────────────────────
    // Triangle (bilinear), NOT Lanczos3: Lanczos rings semi-transparent pixels
    // into the background, which corrupts the marked-region / contour trace.
    let mut processed_image = if let Some(dimensions) = config.resize_dimensions {
        image::imageops::resize(
            &image,
            dimensions[0],
            dimensions[1],
            image::imageops::FilterType::Triangle,
        )
    } else {
        image
    };
    // Drop stray disconnected specks (resize anti-aliasing / segmentation crumbs) so
    // the EC contour trace can't start on one and collapse. No-op for a clean leaf.
    morphology::keep_largest_alpha_component(&mut processed_image);

    // Per-step timing (set MORPH_TIMING=1 to print). Diagnoses where morphology
    // spends its time so we optimise the real bottleneck.
    let morph_dbg = std::env::var("MORPH_TIMING").is_ok();
    let mut _tm = std::time::Instant::now();
    macro_rules! step { ($l:expr) => {{
        if morph_dbg { eprintln!("[morph] {:<16} {:>7.0}ms", $l, _tm.elapsed().as_secs_f64() * 1000.0); }
        #[allow(unused_assignments)] { _tm = std::time::Instant::now(); }
    }} }

    // Guard against fully-opaque images (e.g. an exported _EC/_MC or screenshot).
    let transparent_pixel_count = processed_image.pixels().filter(|p| p[3] < 128).count();
    if transparent_pixel_count == 0 {
        return Err(
            "image has no transparent background — only RGBA PNGs with a transparent \
             background are supported (not _EC/_MC outputs or screenshots)"
                .to_string(),
        );
    }

    // ── adaptive opening → mark pink regions ─────────────────────────────────
    let adaptive_opening_kernel_size = calculate_adaptive_opening_kernel_size(
        &processed_image,
        config.adaptive_opening_max_density,
        config.adaptive_opening_max_percentage,
        config.adaptive_opening_min_percentage,
    );

    let opened_image = morphology::apply_opening_fast(&processed_image, adaptive_opening_kernel_size)
        .map_err(|e| format!("Opening failed: {e}"))?;
    step!("opening#1");

    let mut marked_image = morphology::mark_opened_regions(
        &processed_image,
        &opened_image,
        config.marked_region_color_rgb,
    );

    // ── MC image ─────────────────────────────────────────────────────────────
    let mc_image = morphology::create_mc_with_com_component(
        &processed_image,
        &mut marked_image,
        config.marked_region_color_rgb,
    );

    // ── reference points ─────────────────────────────────────────────────────
    let ec_reference_point = point_analysis::get_reference_point(
        &processed_image,
        &marked_image,
        &config.reference_point_choice,
        config.marked_region_color_rgb,
    )
    .map_err(|e| format!("Failed to get EC reference point: {e}"))?;

    let mc_reference_point = point_analysis::get_mc_reference_point(
        &mc_image,
        &marked_image,
        &config.reference_point_choice,
        config.marked_region_color_rgb,
    )
    .map_err(|e| format!("Failed to get MC reference point: {e}"))?;

    // ── contours from original (pre-filtered) images ─────────────────────────
    let ec_contour_original =
        morphology::trace_contour(&marked_image, true, config.marked_region_color_rgb);
    let mc_contour_original =
        morphology::trace_contour(&mc_image, false, config.marked_region_color_rgb);
    step!("mark+mc+contours");

    // ── shape metrics ────────────────────────────────────────────────────────
    let ec_area = shape_analysis::calculate_area(&marked_image);
    let ec_outline_count = ec_contour_original.len() as u32;
    let ec_circularity =
        shape_analysis::calculate_circularity_from_contour(ec_area, &ec_contour_original);

    let mc_area = shape_analysis::calculate_area(&mc_image);
    let mc_outline_count = mc_contour_original.len() as u32;
    let mc_circularity =
        shape_analysis::calculate_circularity_from_contour(mc_area, &mc_contour_original);

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

    // ── feature extraction ───────────────────────────────────────────────────
    let initial_ec_features = feature_extraction::generate_features(
        ec_reference_point,
        &ec_contour_original,
        &processed_image,
        Some(&marked_image),
        config.marked_region_color_rgb,
        true,
    )
    .map_err(|e| format!("EC feature extraction failed: {e}"))?;
    step!("features_EC");

    let initial_mc_features = feature_extraction::generate_features(
        mc_reference_point,
        &mc_contour_original,
        &mc_image,
        None,
        config.marked_region_color_rgb,
        false,
    )
    .map_err(|e| format!("MC feature extraction failed: {e}"))?;
    step!("features_MC");

    // ── petiole filtering ────────────────────────────────────────────────────
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

    // ── thornfiddle image (dynamic opening) ──────────────────────────────────
    let dynamic_opening_percentage = shape_analysis::calculate_dynamic_opening_percentage(
        mc_shape_index,
        config.thornfiddle_max_opening_percentage,
        config.thornfiddle_min_opening_percentage,
    );
    let dynamic_kernel_size =
        (((dynamic_opening_percentage / 100.0) * mc_shorter_dimension).round() as u32).max(1);

    let thornfiddle_image = morphology::create_thornfiddle_image(
        &mc_image,
        dynamic_kernel_size,
        config.thornfiddle_marked_color_rgb,
    )
    .map_err(|e| format!("Failed to create thornfiddle image: {e}"))?;
    step!("thornfiddle_img");

    // ── harmonic analysis (MC) ───────────────────────────────────────────────
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
    step!("harmonic_MC");

    // ── finalise features ────────────────────────────────────────────────────
    let mut ec_features_final = ec_features.clone();
    for feature in ec_features_final.iter_mut() {
        feature.thornfiddle_path = thornfiddle::calculate_thornfiddle_path(feature);
    }

    let mut mc_features_final = mc_features.clone();
    for (i, feature) in mc_features_final.iter_mut().enumerate() {
        if let Some(&harmonic_value) = mc_harmonic_result.harmonic_values.get(i) {
            feature.thornfiddle_path_harmonic = harmonic_value;
        }
        feature.thornfiddle_path = thornfiddle::calculate_thornfiddle_path(feature);
    }

    // ── entropy metrics ──────────────────────────────────────────────────────
    let mc_spectral_entropy =
        thornfiddle::calculate_spectral_entropy_from_harmonic_thornfiddle_path(
            &mc_features_final,
            mc_harmonic_result.valid_chain_count,
            config.thornfiddle_smoothing_strength,
            config.spectral_entropy_sigmoid_k,
            config.spectral_entropy_sigmoid_c,
        )
        .0;

    let ec_approximate_entropy = thornfiddle::calculate_approximate_entropy_from_pink_path(
        &ec_features_final,
        config.approximate_entropy_m,
        config.approximate_entropy_r,
    );
    step!("finalize+entropy");

    // ── graph data ───────────────────────────────────────────────────────────
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

    // ── CSV strings (cached for instant export) ──────────────────────────────
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

    // ── overlays (raw RGBA) ──────────────────────────────────────────────────
    let marked_image_display = clean_thin_artifacts(&marked_image, config.marked_region_color_rgb);
    let ec_overlay = create_transparent_overlay(&marked_image_display, &[255, 0, 255]);
    let mc_overlay = create_transparent_overlay(&thornfiddle_image, &[255, 215, 0]);

    Ok(MorphResult {
        metrics: MorphMetrics {
            ec_length,
            ec_width,
            ec_shape_index,
            ec_circularity,
            ec_approximate_entropy,
            ec_area,
            ec_outline_count,
            mc_length,
            mc_width,
            mc_shape_index,
            mc_circularity,
            mc_spectral_entropy,
            mc_area,
            mc_outline_count,
        },
        ec_data,
        mc_data,
        original: to_overlay(&processed_image),
        ec_overlay: to_overlay(&ec_overlay),
        mc_overlay: to_overlay(&mc_overlay),
        ec_csv,
        mc_csv,
    })
}

// ── private helpers (ported from the GUI's analysis.rs) ──────────────────────

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
    (((opening_percentage / 100.0) * image_dimension).round() as u32).max(1)
}

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

/// Remove single-pixel-wide artefact lines from the marked (pink) region.
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
    for (x, y) in to_remove {
        cleaned.put_pixel(x, y, Rgba([0, 0, 0, 0]));
    }
    cleaned
}
