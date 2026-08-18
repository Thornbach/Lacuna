use std::path::{Path, PathBuf};
use std::fs;
use image::{ImageFormat, RgbaImage};

use crate::errors::{LeafComplexError, Result};

/// Represents an input image with its metadata
pub struct InputImage {
    pub image: RgbaImage,
    pub path: PathBuf,
    pub filename: String,
}

/// Get all PNG files from a directory (recursively)
pub fn get_png_files_in_dir<P: AsRef<Path>>(dir_path: P) -> Result<Vec<PathBuf>> {
    let dir_path = dir_path.as_ref();
    
    if !dir_path.exists() {
        return Err(LeafComplexError::InvalidPath(dir_path.to_path_buf()));
    }
    
    if !dir_path.is_dir() {
        return Err(LeafComplexError::Config(format!(
            "{} is not a directory", dir_path.display()
        )));
    }
    
    let mut png_files = Vec::new();
    find_png_files_recursive(dir_path, &mut png_files)?;
    
    Ok(png_files)
}

/// Helper function to recursively search for PNG files
fn find_png_files_recursive(dir_path: &Path, result: &mut Vec<PathBuf>) -> Result<()> {
    let entries = fs::read_dir(dir_path)
        .map_err(|e| LeafComplexError::Io(e))?;
    
    for entry in entries {
        let entry = entry.map_err(|e| LeafComplexError::Io(e))?;
        let path = entry.path();
        
        if path.is_dir() {
            // Recursively search subdirectories
            find_png_files_recursive(&path, result)?;
        } else if path.is_file() {
            // Check if it's a PNG file
            if let Some(ext) = path.extension() {
                if ext.to_ascii_lowercase() == "png" {
                    result.push(path);
                }
            }
        }
    }
    
    Ok(())
}

/// Load a PNG image ensuring RGBA format
pub fn load_image<P: AsRef<Path>>(path: P) -> Result<InputImage> {
    let path = path.as_ref();
    
    // Get filename without extension
    let filename = path.file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| LeafComplexError::InvalidPath(path.to_path_buf()))?
        .to_string();
    
    // Load the image
    let img = image::open(path)
        .map_err(|e| LeafComplexError::Image(e))?;
    
    // Convert to RGBA
    let rgba_img = img.to_rgba8();
    
    Ok(InputImage {
        image: rgba_img,
        path: path.to_path_buf(),
        filename,
    })
}

/// Save an RGBA image to the specified path
pub fn save_image<P: AsRef<Path>>(image: &RgbaImage, path: P) -> Result<()> {
    image.save_with_format(path, ImageFormat::Png)
        .map_err(|e| LeafComplexError::Image(e))?;
    
    Ok(())
}