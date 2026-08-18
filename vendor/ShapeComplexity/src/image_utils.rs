use image::{ImageBuffer, Rgba, RgbaImage};

/// Constants
pub const ALPHA_THRESHOLD: u8 = 128; // Alpha value above which a pixel is considered non-transparent

/// Resize an image to the specified dimensions
pub fn resize_image(
    image: &RgbaImage,
    dimensions: [u32; 2],
) -> RgbaImage {
    let (width, height) = (dimensions[0], dimensions[1]);
    image::imageops::resize(
        image,
        width,
        height,
        image::imageops::FilterType::Triangle,
    )
}

/// Check if a pixel is transparent (alpha below threshold)
#[inline]
pub fn is_transparent(pixel: &Rgba<u8>) -> bool {
    pixel[3] < ALPHA_THRESHOLD
}

/// Check if a pixel is non-transparent (alpha >= threshold)
#[inline]
pub fn is_non_transparent(pixel: &Rgba<u8>) -> bool {
    pixel[3] >= ALPHA_THRESHOLD
}

/// Check if a pixel has the specified RGB color (ignoring alpha)
#[inline]
pub fn has_rgb_color(pixel: &Rgba<u8>, color: [u8; 3]) -> bool {
    pixel[0] == color[0] && pixel[1] == color[1] && pixel[2] == color[2]
}

/// Create a mask image from alpha channel (255 for non-transparent, 0 for transparent)
pub fn create_alpha_mask(image: &RgbaImage) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let (width, height) = image.dimensions();
    let mut mask = ImageBuffer::new(width, height);
    
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);
            let mask_value = if is_non_transparent(pixel) { 255 } else { 0 };
            mask.put_pixel(x, y, Rgba([mask_value, mask_value, mask_value, 255]));
        }
    }
    
    mask
}

/// Convert coordinates between image systems if needed
/// In most image processing systems, (0,0) is top-left and y increases downward
#[inline]
pub fn convert_coordinates(x: u32, y: u32, _height: u32) -> (u32, u32) {
    // This function could be modified if a different coordinate system is needed
    (x, y)
}

// In image_utils.rs
pub fn create_circular_kernel(diameter: u32) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    if diameter == 0 {
        return ImageBuffer::new(0, 0);
    }

    let mut kernel = ImageBuffer::new(diameter, diameter);
    let center = (diameter - 1) as f32 / 2.0; // Center coordinate for indices 0..diameter-1

    // For odd diameters, R_sq = ((D-1)/2)^2 ensures a typical cross for D=3, etc.
    // For even diameters, using (D/2)^2 is more common to fill out e.g. a 2x2.
    let radius_sq: f32;
    if diameter % 2 == 1 { // Odd
        radius_sq = ((diameter - 1) as f32 / 2.0).powi(2);
    } else { // Even
        radius_sq = (diameter as f32 / 2.0).powi(2);
        // For even diameters, an exact circle might not touch pixel centers well.
        // The R^2 for even D often implies pixels whose corners are within the circle,
        // or whose centers are within a slightly larger conceptual circle.
        // The (D/2.0)^2 radius will make a D=2 kernel a 2x2 square if dist_sq includes pixel centers.
    }

    for y_idx in 0..diameter {
        for x_idx in 0..diameter {
            let dx = x_idx as f32 - center;
            let dy = y_idx as f32 - center;
            let dist_sq = dx * dx + dy * dy;

            if diameter % 2 == 0 && diameter > 0 {
                // For even diameters, to ensure a (e.g.) 2x2 kernel for D=2,
                // we often consider a pixel (i,j) part of the disk if the square cell it represents
                // intersects the continuous disk. A common approximation is to check if its center
                // is within a slightly expanded radius or use specific rules.
                // The (D/2.0)^2 radius with dx/dy from integer coords to 'center' ( (D-1)/2.0 )
                // will result in a 2x2 for D=2:
                // D=2, center=0.5. radius_sq = (2/2)^2 = 1.
                // (0,0): dx=-0.5, dy=-0.5. dist_sq=0.5. 0.5 <= 1. IN.
                // This ensures D=2 creates a 2x2, D=4 creates a 4x4 etc.
                // For more "circular" even kernels, more complex rules or larger lookup tables are needed.
            }


            // Add a small epsilon for floating point comparisons to handle points exactly on the circumference.
            if dist_sq <= radius_sq + 1e-6 {
                kernel.put_pixel(x_idx, y_idx, Rgba([255, 255, 255, 255]));
            } else {
                kernel.put_pixel(x_idx, y_idx, Rgba([0, 0, 0, 0]));
            }
        }
    }
    kernel
}

/// Check if a point is inside the image bounds
#[inline]
pub fn in_bounds(x: i32, y: i32, width: u32, height: u32) -> bool {
    x >= 0 && y >= 0 && (x as u32) < width && (y as u32) < height
}

/// Create a debug image with specified points marked in color
pub fn create_debug_image(
    image: &RgbaImage,
    points: &[(u32, u32)],
    color: [u8; 3],
    point_size: u32,
) -> RgbaImage {
    let mut debug_image = image.clone();
    let (width, height) = debug_image.dimensions();
    
    for &(x, y) in points {
        let radius = point_size / 2;
        for dy in 0..point_size {
            for dx in 0..point_size {
                let px = x.saturating_sub(radius).saturating_add(dx);
                let py = y.saturating_sub(radius).saturating_add(dy);
                
                if px < width && py < height {
                    debug_image.put_pixel(px, py, Rgba([color[0], color[1], color[2], 255]));
                }
            }
        }
    }
    
    debug_image
}