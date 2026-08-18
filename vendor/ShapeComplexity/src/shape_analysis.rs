// src/shape_analysis.rs - Added Shape Index calculation functions

use image::RgbaImage;
use crate::image_utils::is_non_transparent;
use crate::morphology::{trace_contour, smooth_contour};
use std::f64::consts::PI;

/// Calculate the area of non-transparent pixels in the image
pub fn calculate_area(image: &RgbaImage) -> u32 {
    let (width, height) = image.dimensions();
    let mut non_transparent_count = 0;
    
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);
            if is_non_transparent(pixel) {
                non_transparent_count += 1;
            }
        }
    }
    
    non_transparent_count
}

/// Calculate the bounding box dimensions of non-transparent pixels
/// Returns (width, height) of the bounding box
pub fn calculate_bounding_box_dimensions(image: &RgbaImage) -> (u32, u32) {
    let (img_width, img_height) = image.dimensions();
    
    let mut min_x = img_width;
    let mut max_x = 0;
    let mut min_y = img_height;
    let mut max_y = 0;
    let mut found_pixels = false;
    
    // Find the bounding box of all non-transparent pixels
    for y in 0..img_height {
        for x in 0..img_width {
            let pixel = image.get_pixel(x, y);
            if is_non_transparent(pixel) {
                found_pixels = true;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
    }
    
    if !found_pixels {
        return (0, 0);
    }
    
    // Calculate dimensions
    let width = if max_x >= min_x { max_x - min_x + 1 } else { 0 };
    let height = if max_y >= min_y { max_y - min_y + 1 } else { 0 };
    
    (width, height)
}

/// Calculate the outline count from a pre-computed contour
pub fn calculate_outline_count_from_contour(contour: &[(u32, u32)]) -> u32 {
    contour.len() as u32
}

/// Calculate the outline count (number of contour points) from the original image
pub fn calculate_outline_count(image: &RgbaImage, marked_color: [u8; 3]) -> u32 {
    // Trace the contour of the original image (treating pink as opaque)
    let contour = trace_contour(image, true, marked_color);
    contour.len() as u32
}

/// Calculate biological length and width from contour points
/// Length = longest straight-line distance between any two contour points
/// Width = maximum perpendicular distance to the length axis
pub fn calculate_biological_dimensions(contour: &[(u32, u32)]) -> (f64, f64) {
    if contour.len() < 2 {
        return (0.0, 0.0);
    }
    
    // Find the two points with maximum distance (length)
    let mut max_length = 0.0;
    let mut length_p1 = (0.0, 0.0);
    let mut length_p2 = (0.0, 0.0);
    
    for i in 0..contour.len() {
        for j in (i + 1)..contour.len() {
            let p1 = (contour[i].0 as f64, contour[i].1 as f64);
            let p2 = (contour[j].0 as f64, contour[j].1 as f64);
            
            let distance = ((p2.0 - p1.0).powi(2) + (p2.1 - p1.1).powi(2)).sqrt();
            
            if distance > max_length {
                max_length = distance;
                length_p1 = p1;
                length_p2 = p2;
            }
        }
    }
    
    // Calculate the direction vector of the length axis
    let length_vec = (length_p2.0 - length_p1.0, length_p2.1 - length_p1.1);
    let length_vec_normalized = {
        let len = (length_vec.0.powi(2) + length_vec.1.powi(2)).sqrt();
        if len > 0.0 {
            (length_vec.0 / len, length_vec.1 / len)
        } else {
            (1.0, 0.0)
        }
    };
    
    // Find maximum width (perpendicular distance to length axis)
    let mut min_width: f64 = 0.0;
    let mut max_width: f64 = 0.0;
    
    for point in contour {
        let p = (point.0 as f64, point.1 as f64);
        
        // Calculate perpendicular distance from point to length axis
        let to_point = (p.0 - length_p1.0, p.1 - length_p1.1);
        
        // Project onto the perpendicular direction
        let perp_vec = (-length_vec_normalized.1, length_vec_normalized.0);
        let perp_distance = to_point.0 * perp_vec.0 + to_point.1 * perp_vec.1;
        
        min_width = min_width.min(perp_distance);
        max_width = max_width.max(perp_distance);
    }
    
    // Width is the total span between min and max
    let width = max_width - min_width;
    
    (max_length, width)
}

/// Fast biological dimensions (optimized version for better performance)
/// Uses sampling for very large contours to reduce O(n²) complexity
pub fn calculate_biological_dimensions_fast(contour: &[(u32, u32)]) -> (f64, f64) {
    if contour.len() < 2 {
        return (0.0, 0.0);
    }
    
    // For performance, sample fewer points if contour is very large
    // This reduces complexity from O(n²) to O(s²) where s is sample size
    let sample_step = if contour.len() > 500 { 
        std::cmp::max(1, contour.len() / 250) 
    } else { 
        1 
    };
    
    let mut max_length = 0.0;
    let mut length_p1 = (0.0, 0.0);
    let mut length_p2 = (0.0, 0.0);
    
    // Sample points for length calculation
    for i in (0..contour.len()).step_by(sample_step) {
        for j in ((i + contour.len()/4)..contour.len()).step_by(sample_step) {
            let p1 = (contour[i].0 as f64, contour[i].1 as f64);
            let p2 = (contour[j].0 as f64, contour[j].1 as f64);
            
            let distance = ((p2.0 - p1.0).powi(2) + (p2.1 - p1.1).powi(2)).sqrt();
            
            if distance > max_length {
                max_length = distance;
                length_p1 = p1;
                length_p2 = p2;
            }
        }
    }
    
    // Calculate perpendicular width using all points for accuracy
    let length_vec = (length_p2.0 - length_p1.0, length_p2.1 - length_p1.1);
    let length_vec_normalized = {
        let len = (length_vec.0.powi(2) + length_vec.1.powi(2)).sqrt();
        if len > 0.0 {
            (length_vec.0 / len, length_vec.1 / len)
        } else {
            (1.0, 0.0)
        }
    };
    
    let mut min_width: f64 = 0.0;
    let mut max_width: f64 = 0.0;
    
    // Use a smaller sample step for width calculation to maintain accuracy
    let width_sample_step = std::cmp::max(1, sample_step / 2);
    
    for point in contour.iter().step_by(width_sample_step) {
        let p = (point.0 as f64, point.1 as f64);
        let to_point = (p.0 - length_p1.0, p.1 - length_p1.1);
        let perp_vec = (-length_vec_normalized.1, length_vec_normalized.0);
        let perp_distance = to_point.0 * perp_vec.0 + to_point.1 * perp_vec.1;
        
        min_width = min_width.min(perp_distance);
        max_width = max_width.max(perp_distance);
    }
    
    let width = max_width - min_width;
    
    (max_length, width)
}

/// NEW: Calculate Shape Index from length and width
/// Shape Index = Length / Width, with handling for wide leaves
pub fn calculate_shape_index(length: f64, width: f64) -> f64 {
    if width <= 0.0 {
        return 1.0; // Default to circular if width is invalid
    }
    
    // Handle wide leaves by ensuring we always use the longer dimension as "length"
    let (adjusted_length, adjusted_width) = if length >= width {
        (length, width)
    } else {
        // Flip for wide leaves: use width as length
        (width, length)
    };
    
    adjusted_length / adjusted_width
}

/// NEW: Calculate dynamic thornfiddle opening percentage based on shape index
/// Uses linear relationship: more elongated shapes get smaller opening percentages
pub fn calculate_dynamic_opening_percentage(
    shape_index: f64,
    max_percentage: f64,
    min_percentage: f64,
) -> f64 {
    if shape_index <= 1.0 {
        return max_percentage; // Perfect circle or wide leaf gets max percentage
    }
    
    // Linear interpolation between max and min based on deviation from 1.0
    // The more elongated (higher shape_index), the smaller the percentage
    
    // Define a reasonable upper bound for shape index (e.g., 5.0 for very elongated leaves)
    let max_shape_index = 5.0;
    let clamped_shape_index = shape_index.min(max_shape_index);
    
    // Linear interpolation: as shape_index goes from 1.0 to max_shape_index,
    // percentage goes from max_percentage to min_percentage
    let interpolation_factor = (clamped_shape_index - 1.0) / (max_shape_index - 1.0);
    let dynamic_percentage = max_percentage - (interpolation_factor * (max_percentage - min_percentage));
    
    // Ensure result is within bounds
    dynamic_percentage.max(min_percentage).min(max_percentage)
}

/// NEW: Calculate length, width, and shape index from image
/// Returns (length, width, shape_index)
pub fn calculate_length_width_shape_index(image: &RgbaImage, marked_color: [u8; 3]) -> (f64, f64, f64) {
    // Trace contour
    let contour = trace_contour(image, true, marked_color); // true = pink as opaque for LEC
    
    // Calculate biological dimensions
    let (length, width) = calculate_biological_dimensions_fast(&contour);
    
    // Calculate shape index
    let shape_index = calculate_shape_index(length, width);
    
    (length, width, shape_index)
}

/// NEW: Get the longer dimension from length and width
/// Returns the longer of the two dimensions (for display purposes)
pub fn get_longer_dimension(length: f64, width: f64) -> f64 {
    length.max(width)
}

/// NEW: Get the shorter dimension from length and width
/// Returns the shorter of the two dimensions (for kernel size calculation)
pub fn get_shorter_dimension(length: f64, width: f64) -> f64 {
    length.min(width)
}

/// NEW: Calculate length, width, shape index, and shorter dimension from image
/// Returns (length, width, shape_index, shorter_dimension)
pub fn calculate_length_width_shape_index_with_shorter(image: &RgbaImage, marked_color: [u8; 3]) -> (f64, f64, f64, f64) {
    let (length, width, shape_index) = calculate_length_width_shape_index(image, marked_color);
    let shorter_dimension = get_shorter_dimension(length, width);
    (length, width, shape_index, shorter_dimension)
}

/// LEGACY: Calculate length, width, shape index, and longer dimension from image
/// Returns (length, width, shape_index, longer_dimension)
/// DEPRECATED: Use calculate_length_width_shape_index_with_shorter instead
pub fn calculate_length_width_shape_index_with_longer(image: &RgbaImage, marked_color: [u8; 3]) -> (f64, f64, f64, f64) {
    let (length, width, shape_index) = calculate_length_width_shape_index(image, marked_color);
    let longer_dimension = get_longer_dimension(length, width);
    (length, width, shape_index, longer_dimension)
}

/// Calculate the perimeter of the leaf using contour points
pub fn calculate_perimeter(contour_points: &[(u32, u32)]) -> f64 {
    if contour_points.len() < 2 {
        return 0.0;
    }
    
    let mut perimeter = 0.0;
    let n = contour_points.len();
    
    for i in 0..n {
        let (x1, y1) = contour_points[i];
        let (x2, y2) = contour_points[(i + 1) % n]; // Wrap around to first point
        
        let dx = x2 as f64 - x1 as f64;
        let dy = y2 as f64 - y1 as f64;
        perimeter += (dx * dx + dy * dy).sqrt();
    }
    
    perimeter
}

/// Apply correction factor to adjust for digitization artifacts in perimeter calculation
pub fn correct_perimeter(perimeter: f64, circularity_estimate: f64) -> f64 {
    // Apply correction based on how circle-like the shape appears to be
    if circularity_estimate > 0.75 {
        // For circle-like shapes, apply stronger correction
        // The correction factor is empirically determined
        perimeter * 0.945
    } else if circularity_estimate > 0.5 {
        // For somewhat circular shapes, apply moderate correction
        perimeter * 0.975
    } else {
        // For non-circular shapes, apply minimal correction
        perimeter * 0.99
    }
}

/// Calculate circularity of the shape (4π * Area / Perimeter²)
/// 1.0 for a perfect circle, < 1.0 for other shapes
pub fn calculate_circularity(area: u32, perimeter: f64) -> f64 {
    if perimeter <= 0.0 {
        return 0.0;
    }
    
    // Initial circularity calculation
    let initial_circularity = (4.0 * PI * area as f64) / (perimeter * perimeter);
    
    // Apply correction for digitization artifacts
    let corrected_perimeter = correct_perimeter(perimeter, initial_circularity);
    
    // Recalculate with corrected perimeter
    (4.0 * PI * area as f64) / (corrected_perimeter * corrected_perimeter)
}

/// Calculate circularity from pre-computed area and contour
pub fn calculate_circularity_from_contour(area: u32, contour: &[(u32, u32)]) -> f64 {
    if contour.len() < 2 {
        return 0.0;
    }
    
    // Apply smoothing to reduce digitization artifacts
    let smoothed_contour = smooth_contour(contour, 3);
    
    // Calculate perimeter from smoothed contour
    let perimeter = calculate_perimeter(&smoothed_contour);
    
    // Calculate circularity with corrections
    calculate_circularity(area, perimeter)
}

/// Analyze shape of the image and return area and circularity
pub fn analyze_shape(image: &RgbaImage, marked_color: [u8; 3]) -> (u32, f64) {
    // Calculate area
    let area = calculate_area(image);
    
    // Trace contour for perimeter calculation
    let raw_contour = trace_contour(image, false, marked_color);
    
    // Calculate circularity from the contour
    let circularity = calculate_circularity_from_contour(area, &raw_contour);
    
    (area, circularity)
}

/// UPDATED: Comprehensive shape analysis with length, width, shape index, and outline count
/// Returns (area, circularity, length, width, outline_count, shape_index)
/// Uses biological length/width instead of axis-aligned bounding box
/// This function avoids redundant contour tracing by doing it only once
pub fn analyze_shape_comprehensive(image: &RgbaImage, marked_color: [u8; 3]) -> (u32, f64, f64, f64, u32, f64) {
    // Calculate area (fast - single pass through pixels)
    let area = calculate_area(image);
    
    // Trace contour ONLY ONCE (expensive operation)
    let raw_contour = trace_contour(image, true, marked_color); // Use true for pink as opaque
    
    // Calculate biological dimensions from the pre-computed contour
    let (length, width) = calculate_biological_dimensions_fast(&raw_contour);
    
    // Calculate shape index
    let shape_index = calculate_shape_index(length, width);
    
    // Calculate circularity from the pre-computed contour
    let circularity = calculate_circularity_from_contour(area, &raw_contour);
    
    // Calculate outline count from the pre-computed contour
    let outline_count = calculate_outline_count_from_contour(&raw_contour);
    
    (area, circularity, length, width, outline_count, shape_index)
}

/// LEGACY: Comprehensive shape analysis WITHOUT shape index (for backward compatibility)
/// Returns (area, circularity, length, width, outline_count)
pub fn analyze_shape_comprehensive_legacy(image: &RgbaImage, marked_color: [u8; 3]) -> (u32, f64, f64, f64, u32) {
    let (area, circularity, length, width, outline_count, _shape_index) = analyze_shape_comprehensive(image, marked_color);
    (area, circularity, length, width, outline_count)
}