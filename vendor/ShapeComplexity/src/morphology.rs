use image::{ImageBuffer, Rgba, RgbaImage};
use std::cmp::{max, min};
use std::collections::{HashMap, VecDeque};

use crate::errors::{LeafComplexError, Result};
use crate::image_utils::{create_circular_kernel, in_bounds, has_rgb_color, ALPHA_THRESHOLD};

pub fn erode_alpha(
    image: &RgbaImage,
    kernel: &ImageBuffer<Rgba<u8>, Vec<u8>>,
) -> RgbaImage {
    let (width, height) = image.dimensions();
    let (k_width, k_height) = kernel.dimensions();
    let k_radius_x = (k_width / 2) as i32;
    let k_radius_y = (k_height / 2) as i32;
    
    let mut result = RgbaImage::new(width, height);
    
    for y in 0..height {
        for x in 0..width {
            let mut min_alpha = 255u8;
            let mut erode = false;
            
            // Check if any kernel pixel is outside the object
            for ky in 0..k_height {
                for kx in 0..k_width {
                    if kernel.get_pixel(kx, ky)[0] > 0 {
                        let img_x = x as i32 + (kx as i32) - k_radius_x;
                        let img_y = y as i32 + (ky as i32) - k_radius_y;
                        
                        if !in_bounds(img_x, img_y, width, height) {
                            // Consider out-of-bounds as transparent
                            min_alpha = 0;
                            erode = true;
                            break;
                        }
                        
                        let img_alpha = image.get_pixel(img_x as u32, img_y as u32)[3];
                        min_alpha = min(min_alpha, img_alpha);
                        
                        if img_alpha < ALPHA_THRESHOLD {
                            erode = true;
                            break;
                        }
                    }
                }
                if erode {
                    break;
                }
            }
            
            // Copy RGB from original, but use eroded alpha
            let original = image.get_pixel(x, y);
            result.put_pixel(
                x, 
                y, 
                Rgba([original[0], original[1], original[2], if erode { 0 } else { original[3] }])
            );
        }
    }
    
    result
}

/// Applies morphological dilation to the alpha channel
pub fn dilate_alpha(
    image: &RgbaImage,
    kernel: &ImageBuffer<Rgba<u8>, Vec<u8>>,
) -> RgbaImage {
    let (width, height) = image.dimensions();
    let (k_width, k_height) = kernel.dimensions();
    let k_radius_x = (k_width / 2) as i32;
    let k_radius_y = (k_height / 2) as i32;
    
    let mut result = RgbaImage::new(width, height);
    
    for y in 0..height {
        for x in 0..width {
            let mut max_alpha = 0u8;
            let mut dilate = false;
            
            // Check if any kernel pixel overlaps with a non-transparent pixel
            for ky in 0..k_height {
                for kx in 0..k_width {
                    if kernel.get_pixel(kx, ky)[0] > 0 {
                        let img_x = x as i32 + (kx as i32) - k_radius_x;
                        let img_y = y as i32 + (ky as i32) - k_radius_y;
                        
                        if in_bounds(img_x, img_y, width, height) {
                            let img_alpha = image.get_pixel(img_x as u32, img_y as u32)[3];
                            max_alpha = max(max_alpha, img_alpha);
                            
                            if img_alpha >= ALPHA_THRESHOLD {
                                dilate = true;
                                break;
                            }
                        }
                    }
                }
                if dilate {
                    break;
                }
            }
            
            // Copy RGB from original, but use dilated alpha
            let original = image.get_pixel(x, y);
            result.put_pixel(
                x, 
                y, 
                Rgba([original[0], original[1], original[2], if dilate { original[3].max(1) } else { original[3] }])
            );
        }
    }
    
    result
}

/// Apply morphological opening (erosion followed by dilation)
pub fn apply_opening(
    image: &RgbaImage, 
    kernel_size: u32
) -> Result<RgbaImage> {
    if kernel_size == 0 {
        return Err(LeafComplexError::Morphology(
            "Kernel size must be greater than 0".to_string()
        ));
    }
    
    // Create circular kernel once
    let kernel = create_circular_kernel(kernel_size);
    
    // Pre-compute kernel properties
    let (k_width, k_height) = kernel.dimensions();
    let k_radius_x = (k_width / 2) as i32;
    let k_radius_y = (k_height / 2) as i32;
    
    // Create kernel lookup for faster access
    let mut kernel_pixels = vec![false; (k_width * k_height) as usize];
    for ky in 0..k_height {
        for kx in 0..k_width {
            if kernel.get_pixel(kx, ky)[0] > 0 {
                kernel_pixels[(ky * k_width + kx) as usize] = true;
            }
        }
    }
    
    // Image properties
    let (width, height) = image.dimensions();
    
    // Apply erosion - using a non-parallel implementation first to fix the issues
    let mut eroded = RgbaImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let original = image.get_pixel(x, y);
            
            // Skip transparent pixels - they stay transparent
            if original[3] < ALPHA_THRESHOLD {
                eroded.put_pixel(x, y, *original);
                continue;
            }
            
            let mut erode = false;
            // Check kernel
            'kernel_check: for ky in 0..k_height {
                for kx in 0..k_width {
                    if kernel_pixels[(ky * k_width + kx) as usize] {
                        let img_x = x as i32 + (kx as i32) - k_radius_x;
                        let img_y = y as i32 + (ky as i32) - k_radius_y;
                        
                        if img_x < 0 || img_y < 0 || img_x >= width as i32 || img_y >= height as i32 {
                            erode = true;
                            break 'kernel_check;
                        }
                        
                        let img_alpha = image.get_pixel(img_x as u32, img_y as u32)[3];
                        if img_alpha < ALPHA_THRESHOLD {
                            erode = true;
                            break 'kernel_check;
                        }
                    }
                }
            }
            
            let new_pixel = if erode {
                Rgba([original[0], original[1], original[2], 0])
            } else {
                *original
            };
            eroded.put_pixel(x, y, new_pixel);
        }
    }
    
    // Apply dilation
    let mut dilated = RgbaImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let original = eroded.get_pixel(x, y);
            
            let mut dilate = false;
            // Check kernel
            'kernel_check: for ky in 0..k_height {
                for kx in 0..k_width {
                    if kernel_pixels[(ky * k_width + kx) as usize] {
                        let img_x = x as i32 + (kx as i32) - k_radius_x;
                        let img_y = y as i32 + (ky as i32) - k_radius_y;
                        
                        if img_x >= 0 && img_y >= 0 && img_x < width as i32 && img_y < height as i32 {
                            let img_alpha = eroded.get_pixel(img_x as u32, img_y as u32)[3];
                            if img_alpha >= ALPHA_THRESHOLD {
                                dilate = true;
                                break 'kernel_check;
                            }
                        }
                    }
                }
            }
            
            let new_pixel = if dilate {
                Rgba([original[0], original[1], original[2], original[3].max(1)])
            } else {
                *original
            };
            dilated.put_pixel(x, y, new_pixel);
        }
    }
    
    Ok(dilated)
}

/// Fast morphological opening — byte-for-byte identical result to `apply_opening`
/// (disk erosion → dilation, original RGB preserved, opened alpha) but:
///   • iterates only the disk's TRUE pixels (an offset list), not the k×k box,
///   • reads the raw RGBA buffer instead of per-pixel `get_pixel`,
///   • runs erosion + dilation in parallel (rayon).
/// The naive version is O(W·H·k²) single-threaded with a content-adaptive kernel
/// that can reach ~77 px → seconds/leaf; this is the same work parallel + raw.
pub fn apply_opening_fast(image: &RgbaImage, kernel_size: u32) -> Result<RgbaImage> {
    use rayon::prelude::*;
    if kernel_size == 0 {
        return Err(LeafComplexError::Morphology(
            "Kernel size must be greater than 0".to_string(),
        ));
    }
    let kernel = create_circular_kernel(kernel_size);
    let (k_width, k_height) = kernel.dimensions();
    let k_radius_x = (k_width / 2) as i32;
    let k_radius_y = (k_height / 2) as i32;
    // Disk offsets (relative to centre) — only the pixels actually in the kernel.
    let mut offsets: Vec<(i32, i32)> = Vec::new();
    for ky in 0..k_height {
        for kx in 0..k_width {
            if kernel.get_pixel(kx, ky)[0] > 0 {
                offsets.push((kx as i32 - k_radius_x, ky as i32 - k_radius_y));
            }
        }
    }

    let (width, height) = image.dimensions();
    let (w, h, wu) = (width as i32, height as i32, width as usize);
    let raw = image.as_raw();
    let n = (width * height) as usize;
    let is_leaf: Vec<bool> = (0..n).map(|i| raw[i * 4 + 3] >= ALPHA_THRESHOLD).collect();

    // ── Erosion: a leaf pixel erodes if any disk offset is OOB or background ──
    let mut eroded_alpha = vec![0u8; n];
    eroded_alpha.par_iter_mut().enumerate().for_each(|(i, out)| {
        let a = raw[i * 4 + 3];
        if a < ALPHA_THRESHOLD {
            *out = a; // background stays as-is
            return;
        }
        let (x, y) = ((i % wu) as i32, (i / wu) as i32);
        let mut erode = false;
        for &(dx, dy) in &offsets {
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || ny < 0 || nx >= w || ny >= h {
                erode = true;
                break;
            }
            if !is_leaf[ny as usize * wu + nx as usize] {
                erode = true;
                break;
            }
        }
        *out = if erode { 0 } else { a };
    });
    let eroded_leaf: Vec<bool> = eroded_alpha.iter().map(|&a| a >= ALPHA_THRESHOLD).collect();

    // ── Dilation: a pixel is set if any disk offset lands on the eroded leaf ──
    let mut final_alpha = vec![0u8; n];
    final_alpha.par_iter_mut().enumerate().for_each(|(i, out)| {
        let (x, y) = ((i % wu) as i32, (i / wu) as i32);
        let mut dilate = false;
        for &(dx, dy) in &offsets {
            let (nx, ny) = (x + dx, y + dy);
            if nx >= 0 && ny >= 0 && nx < w && ny < h && eroded_leaf[ny as usize * wu + nx as usize] {
                dilate = true;
                break;
            }
        }
        *out = if dilate { eroded_alpha[i].max(1) } else { eroded_alpha[i] };
    });

    // Rebuild: original RGB, opened alpha (matches apply_opening exactly).
    let mut buf = raw.clone();
    for i in 0..n {
        buf[i * 4 + 3] = final_alpha[i];
    }
    RgbaImage::from_raw(width, height, buf)
        .ok_or_else(|| LeafComplexError::Morphology("opening rebuild failed".to_string()))
}

pub fn mark_opened_regions(
    original: &RgbaImage, 
    opened: &RgbaImage, 
    color: [u8; 3]
) -> RgbaImage {
    let (width, height) = original.dimensions();
    let mut marked = original.clone();
    
    // Mark pixels that were non-transparent in original but transparent in opened
    for y in 0..height {
        for x in 0..width {
            let orig_pixel = original.get_pixel(x, y);
            let opened_pixel = opened.get_pixel(x, y);
            
            // If pixel was originally non-transparent but is transparent after opening
            if orig_pixel[3] > 0 && opened_pixel[3] == 0 {
                // Mark it with the specified color
                marked.put_pixel(x, y, Rgba([color[0], color[1], color[2], orig_pixel[3]]));
            }
        }
    }
        
    marked
}

/// NEW: Create Thornfiddle Image with golden lobe regions
/// Takes LMC image as base, applies aggressive opening, marks removed regions as golden
pub fn create_thornfiddle_image(
    lmc_image: &RgbaImage,
    dynamic_kernel_size: u32,
    golden_color: [u8; 3],
) -> Result<RgbaImage> {
    let (width, height) = lmc_image.dimensions();
    
    // Ensure minimum kernel size of 1
    let aggressive_size = dynamic_kernel_size.max(1);
    
    println!("Creating Thornfiddle image with DYNAMIC kernel size: {} pixels (based on LMC SHORTER dimension)", 
             aggressive_size);
    
    // Apply aggressive opening to LMC image (fast parallel path — identical result)
    let aggressively_opened = apply_opening_fast(lmc_image, aggressive_size)?;
    
    // Create Thornfiddle image: LMC base + golden overlays for removed regions
    let mut thornfiddle_image = lmc_image.clone();
    
    // Mark pixels that were non-transparent in LMC but transparent after aggressive opening
    let mut golden_pixel_count = 0;
    for y in 0..height {
        for x in 0..width {
            let lmc_pixel = lmc_image.get_pixel(x, y);
            let opened_pixel = aggressively_opened.get_pixel(x, y);
            
            // If pixel was originally non-transparent in LMC but is transparent after aggressive opening
            if lmc_pixel[3] > 0 && opened_pixel[3] == 0 {
                // Mark it with golden color (lobe region)
                thornfiddle_image.put_pixel(x, y, Rgba([golden_color[0], golden_color[1], golden_color[2], lmc_pixel[3]]));
                golden_pixel_count += 1;
            }
        }
    }
    
    println!("Thornfiddle image created with {} golden lobe pixels using dynamic kernel size", golden_pixel_count);
    
    Ok(thornfiddle_image)
}

/// Find all connected components in an image
/// Returns a vector of component sizes and a map of pixel coordinates to component IDs
fn find_connected_components(image: &RgbaImage, pink_color: [u8; 3]) -> (Vec<usize>, HashMap<(u32, u32), u32>) {
    let (width, height) = image.dimensions();
    let mut visited = vec![false; (width * height) as usize];
    let mut component_map = HashMap::new();
    let mut component_sizes = Vec::new();
    let mut component_id = 0u32;
    
    // 8-connected neighbors
    let directions = [
        (0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)
    ];
    
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            
            if visited[idx] {
                continue;
            }
            
            let pixel = image.get_pixel(x, y);
            
            // Check if this pixel should be considered (non-transparent and not pink)
            if pixel[3] > 0 && !has_rgb_color(pixel, pink_color) {
                // Start a new connected component
                let mut queue = VecDeque::new();
                let mut component_pixels = Vec::new();
                
                queue.push_back((x, y));
                visited[idx] = true;
                
                // BFS to find all connected pixels
                while let Some((cx, cy)) = queue.pop_front() {
                    component_pixels.push((cx, cy));
                    component_map.insert((cx, cy), component_id);
                    
                    // Check all 8 neighbors
                    for &(dx, dy) in &directions {
                        let nx = cx as i32 + dx;
                        let ny = cy as i32 + dy;
                        
                        if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                            continue;
                        }
                        
                        let nx = nx as u32;
                        let ny = ny as u32;
                        let nidx = (ny * width + nx) as usize;
                        
                        if visited[nidx] {
                            continue;
                        }
                        
                        let npixel = image.get_pixel(nx, ny);
                        if npixel[3] > 0 && !has_rgb_color(npixel, pink_color) {
                            queue.push_back((nx, ny));
                            visited[nidx] = true;
                        }
                    }
                }
                
                component_sizes.push(component_pixels.len());
                component_id += 1;
            }
        }
    }
    
    (component_sizes, component_map)
}

/// Remove small connected components based on size threshold
fn filter_small_components(
    image: &RgbaImage, 
    component_sizes: &[usize], 
    component_map: &HashMap<(u32, u32), u32>,
    min_size_threshold: usize
) -> RgbaImage {
    let (width, height) = image.dimensions();
    let mut filtered_image = RgbaImage::new(width, height);
    
    // Initialize with transparent pixels
    for y in 0..height {
        for x in 0..width {
            filtered_image.put_pixel(x, y, Rgba([0, 0, 0, 0]));
        }
    }
    
    // Copy pixels from components that are large enough
    for y in 0..height {
        for x in 0..width {
            if let Some(&component_id) = component_map.get(&(x, y)) {
                let component_size = component_sizes[component_id as usize];
                
                if component_size >= min_size_threshold {
                    // Keep this pixel
                    filtered_image.put_pixel(x, y, *image.get_pixel(x, y));
                }
            }
        }
    }
    
    filtered_image
}

/// Keep only the largest connected component
fn keep_largest_component(
    image: &RgbaImage, 
    component_sizes: &[usize], 
    component_map: &HashMap<(u32, u32), u32>
) -> RgbaImage {
    let (width, height) = image.dimensions();
    let mut filtered_image = RgbaImage::new(width, height);
    
    // Initialize with transparent pixels
    for y in 0..height {
        for x in 0..width {
            filtered_image.put_pixel(x, y, Rgba([0, 0, 0, 0]));
        }
    }
    
    // Find the largest component
    if let Some((largest_component_id, _)) = component_sizes.iter()
        .enumerate()
        .max_by_key(|(_, &size)| size) {
        
        // Copy pixels from the largest component only
        for y in 0..height {
            for x in 0..width {
                if let Some(&component_id) = component_map.get(&(x, y)) {
                    if component_id == largest_component_id as u32 {
                        filtered_image.put_pixel(x, y, *image.get_pixel(x, y));
                    }
                }
            }
        }
    }
    
    filtered_image
}

/// Apply additional morphological cleaning to remove thin connections and shells
fn clean_thin_artifacts(image: &RgbaImage, pink_color: [u8; 3]) -> RgbaImage {
    let (width, height) = image.dimensions();
    
    // Step 1: Apply a small erosion to break thin connections (2-pixel radius)
    let small_kernel = create_circular_kernel(3); // 3x3 kernel to break 1-2 pixel connections
    let mut eroded = RgbaImage::new(width, height);
    
    let (k_width, k_height) = small_kernel.dimensions();
    let k_radius_x = (k_width / 2) as i32;
    let k_radius_y = (k_height / 2) as i32;
    
    // Create kernel lookup
    let mut kernel_pixels = vec![false; (k_width * k_height) as usize];
    for ky in 0..k_height {
        for kx in 0..k_width {
            if small_kernel.get_pixel(kx, ky)[0] > 0 {
                kernel_pixels[(ky * k_width + kx) as usize] = true;
            }
        }
    }
    
    // Apply erosion
    for y in 0..height {
        for x in 0..width {
            let original = image.get_pixel(x, y);
            
            if original[3] < ALPHA_THRESHOLD || has_rgb_color(original, pink_color) {
                eroded.put_pixel(x, y, Rgba([0, 0, 0, 0]));
                continue;
            }
            
            let mut erode = false;
            'kernel_check: for ky in 0..k_height {
                for kx in 0..k_width {
                    if kernel_pixels[(ky * k_width + kx) as usize] {
                        let img_x = x as i32 + (kx as i32) - k_radius_x;
                        let img_y = y as i32 + (ky as i32) - k_radius_y;
                        
                        if img_x < 0 || img_y < 0 || img_x >= width as i32 || img_y >= height as i32 {
                            erode = true;
                            break 'kernel_check;
                        }
                        
                        let check_pixel = image.get_pixel(img_x as u32, img_y as u32);
                        if check_pixel[3] < ALPHA_THRESHOLD || has_rgb_color(check_pixel, pink_color) {
                            erode = true;
                            break 'kernel_check;
                        }
                    }
                }
            }
            
            if erode {
                eroded.put_pixel(x, y, Rgba([0, 0, 0, 0]));
            } else {
                eroded.put_pixel(x, y, *original);
            }
        }
    }
    
    // Step 2: Find connected components after erosion
    let (component_sizes, component_map) = find_connected_components(&eroded, pink_color);
    
    // Step 3: Keep only the largest component
    let largest_only = keep_largest_component(&eroded, &component_sizes, &component_map);
    
    // Step 4: Apply a small dilation to restore size (1-pixel radius)
    let _restore_kernel = create_circular_kernel(3);
    let mut dilated = RgbaImage::new(width, height);
    
    for y in 0..height {
        for x in 0..width {
            let mut dilate = false;
            
            'kernel_check: for ky in 0..k_height {
                for kx in 0..k_width {
                    if kernel_pixels[(ky * k_width + kx) as usize] {
                        let img_x = x as i32 + (kx as i32) - k_radius_x;
                        let img_y = y as i32 + (ky as i32) - k_radius_y;
                        
                        if img_x >= 0 && img_y >= 0 && img_x < width as i32 && img_y < height as i32 {
                            let check_pixel = largest_only.get_pixel(img_x as u32, img_y as u32);
                            if check_pixel[3] >= ALPHA_THRESHOLD && !has_rgb_color(check_pixel, pink_color) {
                                dilate = true;
                                break 'kernel_check;
                            }
                        }
                    }
                }
            }
            
            if dilate {
                // Use the original pixel color from the input image
                let orig_pixel = image.get_pixel(x, y);
                if orig_pixel[3] > 0 && !has_rgb_color(orig_pixel, pink_color) {
                    dilated.put_pixel(x, y, *orig_pixel);
                } else {
                    dilated.put_pixel(x, y, Rgba([128, 128, 128, 255])); // Fallback gray
                }
            } else {
                dilated.put_pixel(x, y, Rgba([0, 0, 0, 0]));
            }
        }
    }
    
    dilated
}

/// Improved LMC creation with thin artifact removal
pub fn create_mc_with_com_component(
    processed_image: &RgbaImage, 
    marked_image: &mut RgbaImage, 
    pink_color: [u8; 3]
) -> RgbaImage {
    let (width, height) = processed_image.dimensions();
    
    // First, calculate the center of mass
    let (com_x, com_y) = calculate_center_of_mass(processed_image)
        .unwrap_or((width / 2, height / 2)); // Fallback to center if calculation fails
    
    println!("Center of Mass: ({}, {})", com_x, com_y);
    
    // Create a version of the image with pink pixels made transparent
    let mut temp_image = marked_image.clone();
    for y in 0..height {
        for x in 0..width {
            let pixel = temp_image.get_pixel_mut(x, y);
            if has_rgb_color(pixel, pink_color) {
                *pixel = image::Rgba([0, 0, 0, 0]);
            }
        }
    }
    
    // NEW: Apply morphological cleaning to remove thin artifacts
    println!("Cleaning thin artifacts...");
    let cleaned_image = clean_thin_artifacts(&temp_image, pink_color);
    
    // Find connected components in the cleaned image
    let (component_sizes, component_map) = find_connected_components(&cleaned_image, pink_color);
    
    println!("Found {} connected components after cleaning", component_sizes.len());
    
    // Calculate size threshold (e.g., components must be at least 0.5% of image area)
    let total_pixels = (width * height) as usize;
    let min_size_threshold = (total_pixels as f64 * 0.005).max(50.0) as usize; // Reduced to 0.5% since we've already cleaned
    
    // Apply size-based filtering
    let size_filtered = filter_small_components(&cleaned_image, &component_sizes, &component_map, min_size_threshold);
    
    // Keep only the largest remaining component
    let (final_component_sizes, final_component_map) = find_connected_components(&size_filtered, pink_color);
    let lmc_image = keep_largest_component(&size_filtered, &final_component_sizes, &final_component_map);
    
    println!("After cleaning and filtering: {} components, keeping largest with {} pixels", 
             final_component_sizes.len(),
             final_component_sizes.iter().max().unwrap_or(&0));
    
    // Update marking: If pixel is in original but not in final LMC, mark it pink
    for y in 0..height {
        for x in 0..width {
            let orig_pixel = processed_image.get_pixel(x, y);
            let lmc_pixel = lmc_image.get_pixel(x, y);
            
            // If pixel is non-transparent in original but transparent in LMC
            if orig_pixel[3] > 0 && lmc_pixel[3] == 0 {
                // Mark it pink in the marked image
                marked_image.put_pixel(x, y, image::Rgba([pink_color[0], pink_color[1], pink_color[2], orig_pixel[3]]));
            }
        }
    }
    
    lmc_image
}

/// Direction vectors for Moore-Neighbor contour tracing
static MOORE_NEIGHBORHOOD: [(i32, i32); 8] = [
    (1, 0),   // right
    (1, 1),   // down-right
    (0, 1),   // down
    (-1, 1),  // down-left
    (-1, 0),  // left
    (-1, -1), // up-left
    (0, -1),  // up
    (1, -1),  // up-right
];

/// Trace the contour of a region using Moore-Neighbor tracing algorithm
/// Keep only the largest connected component of non-transparent pixels; zero the
/// alpha of everything else. Resizing a non-square cutout to 512² leaves faint
/// disconnected specks (anti-aliasing / segmentation crumbs); the EC contour trace
/// starts at the leftmost boundary pixel, which can be one of those specks — it then
/// traces the tiny blob (~6 points) and stops, never reaching the leaf. This removes
/// the specks so EC traces the real leaf. (The MC path already does the equivalent
/// via its centre-of-mass component.) No-op for a clean single-component leaf.
pub fn keep_largest_alpha_component(image: &mut RgbaImage) {
    let (width, height) = image.dimensions();
    let (w, h, wu) = (width as i32, height as i32, width as usize);
    let n = (width * height) as usize;
    let solid: Vec<bool> = image.as_raw().chunks_exact(4).map(|p| p[3] > 0).collect();

    let mut label = vec![0u32; n];
    let dirs: [(i32, i32); 8] =
        [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)];
    let mut cur = 0u32;
    let (mut best_label, mut best_size) = (0u32, 0usize);
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..n {
        if !solid[start] || label[start] != 0 { continue; }
        cur += 1;
        label[start] = cur;
        stack.push(start);
        let mut size = 0usize;
        while let Some(p) = stack.pop() {
            size += 1;
            let (px, py) = ((p % wu) as i32, (p / wu) as i32);
            for &(dx, dy) in &dirs {
                let (nx, ny) = (px + dx, py + dy);
                if nx < 0 || ny < 0 || nx >= w || ny >= h { continue; }
                let nidx = ny as usize * wu + nx as usize;
                if solid[nidx] && label[nidx] == 0 {
                    label[nidx] = cur;
                    stack.push(nidx);
                }
            }
        }
        if size > best_size { best_size = size; best_label = cur; }
    }
    if best_label == 0 { return; }
    for i in 0..n {
        if label[i] != best_label && solid[i] {
            let (x, y) = ((i % wu) as u32, (i / wu) as u32);
            let mut px = *image.get_pixel(x, y);
            px[3] = 0;
            image.put_pixel(x, y, px);
        }
    }
}

pub fn trace_contour(image: &RgbaImage, is_pink_opaque: bool, pink_color: [u8; 3]) -> Vec<(u32, u32)> {
    use std::collections::HashSet;
    
    let (width, height) = image.dimensions();
    let mut contour = Vec::new();
    
    // Find the leftmost non-transparent pixel (first pixel encountered in scanning order)
    let mut start_point = None;
    
    'outer: for x in 0..width {
        for y in 0..height {
            let pixel = image.get_pixel(x, y);
            
            // Check if this is a valid contour pixel based on mode
            let is_valid = if is_pink_opaque {
                pixel[3] > 0  // For LEC, any non-transparent pixel is valid
            } else {
                pixel[3] > 0 && !has_rgb_color(pixel, pink_color)  // For LMC, non-transparent and non-pink
            };
            
            if is_valid {
                // Check if it's a boundary pixel by checking if it has any transparent/invalid neighbor
                let mut is_boundary = false;
                
                // Check 8-connected neighbors
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx == 0 && dy == 0 { continue; }
                        
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        
                        if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                            // Out of bounds counts as boundary
                            is_boundary = true;
                            break;
                        } else {
                            let neighbor = image.get_pixel(nx as u32, ny as u32);
                            let is_neighbor_valid = if is_pink_opaque {
                                neighbor[3] > 0
                            } else {
                                neighbor[3] > 0 && !has_rgb_color(neighbor, pink_color)
                            };
                            
                            if !is_neighbor_valid {
                                is_boundary = true;
                                break;
                            }
                        }
                    }
                    if is_boundary { break; }
                }
                
                if is_boundary {
                    start_point = Some((x, y));
                    break 'outer;
                }
            }
        }
    }
    
    // If no start point found, return empty contour
    let start = match start_point {
        Some(p) => p,
        None => return Vec::new(),
    };
    
    // Add start point to contour
    contour.push(start);
    
    // Initialize variables for contour tracing
    let mut current = start;
    let mut visited = HashSet::new();
    visited.insert(current);
    
    // Direction offsets for 8-connected neighbors (clockwise order from right)
    let directions = [
        (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)
    ];
    
    // Start with the right direction
    let mut direction_idx = 0;
    
    // Safety counter to prevent infinite loops
    let max_iterations = (width * height) as usize * 2;
    let mut iteration_count = 0;
    
    // Main tracing loop
    loop {
        iteration_count += 1;
        if iteration_count > max_iterations {
            println!("Warning: Contour tracing terminated after {} iterations to prevent infinite loop.", max_iterations);
            break;
        }
        
        // Find the next boundary pixel by checking neighbors in clockwise order
        let mut found_next = false;
        
        // Look in all 8 directions, starting from the backtracking direction + 2
        // This ensures we follow the boundary by making a right turn whenever possible
        for i in 0..8 {
            // Start from backtracking direction + 2 (90 degrees clockwise from backtrack)
            let check_idx = (direction_idx + 6 + i) % 8;
            let (dx, dy) = directions[check_idx];
            let nx = current.0 as i32 + dx;
            let ny = current.1 as i32 + dy;
            
            // Check bounds
            if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                continue;
            }
            
            let next = (nx as u32, ny as u32);
            let pixel = image.get_pixel(next.0, next.1);
            
            // Check if this pixel is valid according to our criteria
            let is_valid = if is_pink_opaque {
                pixel[3] > 0
            } else {
                pixel[3] > 0 && !has_rgb_color(pixel, pink_color)
            };
            
            if is_valid {
                // We found the next boundary pixel
                current = next;
                direction_idx = check_idx;
                
                // Only add to contour if we haven't visited this pixel yet
                if !visited.contains(&current) {
                    contour.push(current);
                    visited.insert(current);
                }
                
                found_next = true;
                break;
            }
        }
        
        // If we couldn't find a next pixel or we've returned to start and completed at least one circuit
        if !found_next || (current == start && contour.len() > 1) {
            break;
        }
    }
    
    // Return the traced contour
    contour
}

/// Calculate the Center of Mass (COM) - adapted from your existing code
pub fn calculate_center_of_mass(image: &RgbaImage) -> Option<(u32, u32)> {
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
        return None;
    }
    
    // Calculate COM
    let com_x = sum_x / sum_alpha;
    let com_y = sum_y / sum_alpha;
    
    // Round to nearest integer
    Some((com_x.round() as u32, com_y.round() as u32))
}

pub fn resample_contour(contour: &[(u32, u32)], target_points: usize) -> Vec<(u32, u32)> {
    if contour.len() <= 1 || target_points <= 1 {
        return contour.to_vec();
    }
    
    if target_points >= contour.len() {
        // If target is larger than current, just return the original
        return contour.to_vec();
    }
    
    // Calculate cumulative distances along the contour
    let mut cumulative_distances = vec![0.0; contour.len()];
    let mut total_perimeter = 0.0;
    
    for i in 1..contour.len() {
        let dx = contour[i].0 as f64 - contour[i-1].0 as f64;
        let dy = contour[i].1 as f64 - contour[i-1].1 as f64;
        let segment_length = (dx * dx + dy * dy).sqrt();
        total_perimeter += segment_length;
        cumulative_distances[i] = total_perimeter;
    }
    
    // Handle closed contour - add distance from last point back to first
    if contour.len() > 2 {
        let last_idx = contour.len() - 1;
        let dx = contour[0].0 as f64 - contour[last_idx].0 as f64;
        let dy = contour[0].1 as f64 - contour[last_idx].1 as f64;
        let closing_segment = (dx * dx + dy * dy).sqrt();
        total_perimeter += closing_segment;
    }
    
    // Generate target distances for resampled points
    let mut resampled_contour = Vec::with_capacity(target_points);
    
    for i in 0..target_points {
        let target_distance = (i as f64 * total_perimeter) / target_points as f64;
        
        // Find the segment containing this target distance
        let mut segment_start_idx = 0;
        for j in 1..cumulative_distances.len() {
            if cumulative_distances[j] > target_distance {
                segment_start_idx = j - 1;
                break;
            }
            if j == cumulative_distances.len() - 1 {
                segment_start_idx = j;
            }
        }
        
        // Handle the case where target_distance is in the closing segment
        if target_distance > cumulative_distances[cumulative_distances.len() - 1] {
            // Interpolate between last point and first point
            let excess_distance = target_distance - cumulative_distances[cumulative_distances.len() - 1];
            let last_idx = contour.len() - 1;
            let dx = contour[0].0 as f64 - contour[last_idx].0 as f64;
            let dy = contour[0].1 as f64 - contour[last_idx].1 as f64;
            let closing_segment_length = (dx * dx + dy * dy).sqrt();
            
            if closing_segment_length > 0.0 {
                let t = excess_distance / closing_segment_length;
                let x = contour[last_idx].0 as f64 + t * dx;
                let y = contour[last_idx].1 as f64 + t * dy;
                resampled_contour.push((x.round() as u32, y.round() as u32));
            } else {
                resampled_contour.push(contour[last_idx]);
            }
        } else {
            // Interpolate within the current segment
            let segment_end_idx = if segment_start_idx == cumulative_distances.len() - 1 {
                0 // Wrap to first point for closed contour
            } else {
                segment_start_idx + 1
            };
            
            if segment_start_idx == segment_end_idx {
                // Single point case
                resampled_contour.push(contour[segment_start_idx]);
            } else {
                let segment_start_distance = cumulative_distances[segment_start_idx];
                let segment_end_distance = if segment_end_idx == 0 {
                    total_perimeter
                } else {
                    cumulative_distances[segment_end_idx]
                };
                
                let segment_length = segment_end_distance - segment_start_distance;
                
                if segment_length > 0.0 {
                    let t = (target_distance - segment_start_distance) / segment_length;
                    
                    let start_point = contour[segment_start_idx];
                    let end_point = contour[segment_end_idx];
                    
                    let x = start_point.0 as f64 + t * (end_point.0 as f64 - start_point.0 as f64);
                    let y = start_point.1 as f64 + t * (end_point.1 as f64 - start_point.1 as f64);
                    
                    resampled_contour.push((x.round() as u32, y.round() as u32));
                } else {
                    resampled_contour.push(contour[segment_start_idx]);
                }
            }
        }
    }
    
    resampled_contour
}

/// Smooth contour points to reduce digitization artifacts
pub fn smooth_contour(contour: &[(u32, u32)], smoothing_strength: usize) -> Vec<(u32, u32)> {
    if contour.len() <= 3 || smoothing_strength == 0 {
        return contour.to_vec();
    }
    
    let mut smoothed = Vec::with_capacity(contour.len());
    let window_size = std::cmp::min(smoothing_strength * 2 + 1, contour.len());
    let half_window = window_size / 2;
    
    for i in 0..contour.len() {
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut count = 0;
        
        for j in 0..window_size {
            let idx = (i + j + contour.len() - half_window) % contour.len();
            sum_x += contour[idx].0 as f64;
            sum_y += contour[idx].1 as f64;
            count += 1;
        }
        
        smoothed.push(((sum_x / count as f64).round() as u32, 
                       (sum_y / count as f64).round() as u32));
    }
    
    smoothed
}