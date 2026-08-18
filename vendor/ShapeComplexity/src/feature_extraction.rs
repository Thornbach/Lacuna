// src/feature_extraction.rs - Feature extraction for EC/MC analysis

use image::RgbaImage;

use crate::errors::{LeafComplexError, Result};
use crate::path_algorithms::{
    calculate_straight_path_length, build_diego_prevmap, diego_path_from_prevmap,
    calculate_diego_path_length, calculate_diego_path_pink, trace_straight_line,
    check_straight_line_transparency, check_straight_line_boundary_zone,
};

#[derive(Debug, Clone)]
pub struct MarginalPointFeatures {
    pub point_index: usize,
    pub straight_path_length: f64,
    pub diego_path_length: f64,
    pub diego_path_pink: Option<u32>,
    pub thornfiddle_path: f64,
    pub thornfiddle_path_harmonic: f64,
}

/// Generate features for all marginal points on the contour.
///
/// For both EC and MC, the geodesic path is used:
///   - If the straight line is clear → straight line (fast path)
///   - If the straight line clips a transparent pixel → graph search from
///     reference_point:
///       EC → Dijkstra with boundary penalty (steers path through the leaf
///            interior, preventing false pink accumulation when the path would
///            otherwise hug the leaf margin around a lobe)
///       MC → plain BFS (true geodesic hop count for path-length metric)
pub fn generate_features(
    reference_point: (u32, u32),
    marginal_points: &[(u32, u32)],
    image: &RgbaImage,
    marked_image: Option<&RgbaImage>,
    marked_color: [u8; 3],
    is_ec: bool,
) -> Result<Vec<MarginalPointFeatures>> {
    if marginal_points.is_empty() {
        return Err(LeafComplexError::NoValidPoints);
    }

    let mut features = Vec::with_capacity(marginal_points.len());

    // EC navigates on marked_image (pink regions are opaque/walkable).
    // MC navigates on the original image (pink regions are transparent/blocked).
    let analysis_image = if is_ec {
        marked_image.unwrap_or(image)
    } else {
        image
    };

    // For EC: Dijkstra with a boundary penalty keeps the path through the leaf
    // interior, avoiding the continuous pink strip along the leaf margin that
    // would otherwise inflate pink counts when routing around lobes.
    // For MC: plain BFS (penalty 0.0) preserves the true geodesic distance.
    // 30.0 makes a boundary cardinal step cost 31× a normal interior step.
    // Combined with the 2-pixel penalty zone in calculate_diego_path this is
    // essentially prohibitive: the interior detour would have to be 31× longer
    // than the boundary shortcut for the margin route to still win — impossible
    // for any realistic leaf geometry.
    let boundary_penalty = if is_ec { 30.0_f64 } else { 0.0_f64 };

    // Single-source path tree from the reference point — computed ONCE and shared by
    // every margin point (all paths originate here). Replaces the per-point BFS/Dijkstra
    // that was ~95% of the pipeline runtime; reconstructed paths are byte-identical.
    let prev_map = build_diego_prevmap(reference_point, analysis_image, boundary_penalty);

    for (idx, &marginal_point) in marginal_points.iter().enumerate() {
        let straight_path_length =
            calculate_straight_path_length(reference_point, marginal_point);

        let straight_line = trace_straight_line(reference_point, marginal_point);
        let crosses_transparency =
            check_straight_line_transparency(&straight_line, analysis_image);

        // For EC: also check whether the straight line itself grazes the 2-pixel
        // boundary zone mid-path (before the endpoint vicinity).  This catches
        // the rare case where the straight line passes near a different part of
        // the leaf edge, picking up pink pixels from an unrelated tooth without
        // ever clipping a transparent pixel.  If detected, Dijkstra runs instead
        // so the interior-biased path is used.
        let needs_search = crosses_transparency
            || (is_ec && check_straight_line_boundary_zone(&straight_line, analysis_image));

        // When the straight line is blocked or grazes the boundary zone, use the
        // appropriate search: Dijkstra (EC) or BFS (MC) — both from reference_point.
        let diego_path = if needs_search {
            diego_path_from_prevmap(reference_point, marginal_point, analysis_image, &prev_map)
        } else {
            straight_line.clone()
        };

        let diego_path_length = if needs_search {
            calculate_diego_path_length(&diego_path)
        } else {
            straight_path_length
        };

        let diego_path_pink = if is_ec {
            if let Some(marked) = marked_image {
                Some(calculate_diego_path_pink(&diego_path, marked, marked_color))
            } else {
                None
            }
        } else {
            None
        };

        features.push(MarginalPointFeatures {
            point_index: idx,
            straight_path_length,
            diego_path_length,
            diego_path_pink,
            thornfiddle_path: 0.0,
            thornfiddle_path_harmonic: 0.0,
        });
    }

    Ok(features)
}