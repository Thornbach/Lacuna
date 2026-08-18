use image::RgbaImage;

use crate::config::ReferencePointChoice;
use crate::errors::{LeafComplexError, Result};
use crate::image_utils::{is_non_transparent, has_rgb_color};

/// Calculate the Emerge Point (EP)
pub fn calculate_emerge_point(
    image: &RgbaImage,
    marked_color: [u8; 3],
) -> Result<(u32, u32)> {
    let (width, height) = image.dimensions();
    
    // First, find maximum Y among non-transparent, non-marked pixels
    let mut max_y = 0;
    let mut max_y_points = Vec::new();
    
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);
            
            // Check if pixel is non-transparent and not marked with the special color
            if is_non_transparent(pixel) && !has_rgb_color(pixel, marked_color) {
                if y > max_y {
                    max_y = y;
                    max_y_points.clear();
                    max_y_points.push(x);
                } else if y == max_y {
                    max_y_points.push(x);
                }
            }
        }
    }
    
    if max_y_points.is_empty() {
        return Err(LeafComplexError::NoValidPoints);
    }
    
    // Filter points to those within the central band
    let center_x = width as f32 * 0.5;
    let band_left = width as f32 * 0.49;
    let band_right = width as f32 * 0.51;
    
    // Fixed: clone max_y_points to avoid moving it before we might need it again
    let central_points: Vec<u32> = max_y_points.iter()
        .filter(|&&x| x as f32 >= band_left && x as f32 <= band_right)
        .cloned()
        .collect();
    
    // If there are central points, find the one closest to center
    if !central_points.is_empty() {
        let closest = central_points
            .into_iter()
            .min_by_key(|&x| {
                let dx = x as f32 - center_x;
                (dx * dx) as u32 // Square distance
            })
            .unwrap(); // Safe because we checked non-empty
        
        Ok((closest, max_y))
    } else if !max_y_points.is_empty() {
        // If no central points, find closest to center among all max_y points
        let closest = max_y_points
            .into_iter()
            .min_by_key(|&x| {
                let dx = x as f32 - center_x;
                (dx * dx) as u32 // Square distance
            })
            .unwrap(); // Safe because we checked non-empty
        
        Ok((closest, max_y))
    } else {
        Err(LeafComplexError::NoValidPoints)
    }
}

/// Calculate the Center of Mass (COM) for any image
pub fn calculate_center_of_mass(image: &RgbaImage) -> Result<(u32, u32)> {
    let (width, height) = image.dimensions();
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_alpha = 0.0;
    
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);
            let alpha = pixel[3] as f32;
            
            if alpha > 0.0 {
                sum_x += x as f32 * alpha;
                sum_y += y as f32 * alpha;
                sum_alpha += alpha;
            }
        }
    }
    
    if sum_alpha <= 0.0 {
        return Err(LeafComplexError::NoValidPoints);
    }
    
    // Calculate COM
    let com_x = sum_x / sum_alpha;
    let com_y = sum_y / sum_alpha;
    
    // Round to nearest integer
    Ok((com_x.round() as u32, com_y.round() as u32))
}

/// Get the reference point based on the configuration choice for any image
pub fn get_reference_point(
    image: &RgbaImage,
    marked_image: &RgbaImage,
    reference_point_choice: &ReferencePointChoice,
    marked_color: [u8; 3],
) -> Result<(u32, u32)> {
    match reference_point_choice {
        ReferencePointChoice::Ep => calculate_emerge_point(marked_image, marked_color),
        ReferencePointChoice::Com => calculate_center_of_mass(image),
    }
}

/// Get the reference point specifically for LMC analysis
/// This uses the LMC image for COM calculation if needed
pub fn get_mc_reference_point(
    lmc_image: &RgbaImage,
    marked_image: &RgbaImage,
    reference_point_choice: &ReferencePointChoice,
    marked_color: [u8; 3],
) -> Result<(u32, u32)> {
    match reference_point_choice {
        ReferencePointChoice::Ep => calculate_emerge_point(marked_image, marked_color),
        ReferencePointChoice::Com => calculate_center_of_mass(lmc_image), // Use LMC image for COM
    }
}