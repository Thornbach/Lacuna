// src/output.rs - CSV output generation for EC/MC analysis

use std::fs;
use std::path::Path;
use csv::Writer;

use crate::errors::{LeafComplexError, Result};
use crate::feature_extraction::MarginalPointFeatures;

/// Write EC (Edge Complexity) features to CSV
///
/// # Arguments
/// * `features` - Vector of features for each contour point
/// * `output_dir` - Base output directory
/// * `filename` - Name of the input file (without extension)
///
/// # Output Columns
/// - Point_Index
/// - Geodesic (Diego path length)
/// - Geodesic_EC (Pink pixels crossed)
/// - GeodesicPath_MC (Thornfiddle path)
/// - Geodesic_MC_H (Harmonic thornfiddle path)
pub fn write_ec_csv<P: AsRef<Path>>(
    features: &[MarginalPointFeatures],
    output_dir: P,
    filename: &str,
) -> Result<()> {
    let output_path = output_dir.as_ref().join("EC").join(format!("{}.csv", filename));
    
    // Create directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|e| LeafComplexError::Io(e))?;
    }
    
    // Create CSV writer
    let mut writer = Writer::from_path(&output_path)
        .map_err(|e| LeafComplexError::CsvOutput(e))?;
    
    // Write header
    writer.write_record(&[
        "Point_Index",
        "Geodesic",
        "Geodesic_EC",
        "GeodesicPath_MC",
        "Geodesic_MC_H",
    ]).map_err(|e| LeafComplexError::CsvOutput(e))?;
    
    // Write data
    for feature in features {
        writer.write_record(&[
            feature.point_index.to_string(),
            format!("{:.6}", feature.diego_path_length),
            feature.diego_path_pink.unwrap_or(0).to_string(),
            format!("{:.6}", feature.thornfiddle_path),
            format!("{:.6}", feature.thornfiddle_path_harmonic),
        ]).map_err(|e| LeafComplexError::CsvOutput(e))?;
    }
    
    // Flush writer
    writer.flush().map_err(|e| LeafComplexError::CsvOutput(csv::Error::from(e)))?;
    
    Ok(())
}

/// Write MC (Margin Complexity) features to CSV
///
/// # Arguments
/// * `features` - Vector of features for each contour point
/// * `output_dir` - Base output directory
/// * `filename` - Name of the input file (without extension)
///
/// # Output Columns
/// - Point_Index
/// - Geodesic (Diego path length)
/// - Geodesic_EC (always 0 for MC)
/// - GeodesicPath_MC (Thornfiddle path)
/// - Geodesic_MC_H (Harmonic thornfiddle path)
pub fn write_mc_csv<P: AsRef<Path>>(
    features: &[MarginalPointFeatures],
    output_dir: P,
    filename: &str,
) -> Result<()> {
    let output_path = output_dir.as_ref().join("MC").join(format!("{}.csv", filename));
    
    // Create directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|e| LeafComplexError::Io(e))?;
    }
    
    // Create CSV writer
    let mut writer = Writer::from_path(&output_path)
        .map_err(|e| LeafComplexError::CsvOutput(e))?;
    
    // Write header
    writer.write_record(&[
        "Point_Index",
        "Geodesic",
        "Geodesic_EC",
        "GeodesicPath_MC",
        "Geodesic_MC_H",
    ]).map_err(|e| LeafComplexError::CsvOutput(e))?;
    
    // Write data
    for feature in features {
        writer.write_record(&[
            feature.point_index.to_string(),
            format!("{:.6}", feature.diego_path_length),
            "0".to_string(), // MC analysis doesn't have pink pixels
            format!("{:.6}", feature.thornfiddle_path),
            format!("{:.6}", feature.thornfiddle_path_harmonic),
        ]).map_err(|e| LeafComplexError::CsvOutput(e))?;
    }
    
    // Flush writer
    writer.flush().map_err(|e| LeafComplexError::CsvOutput(csv::Error::from(e)))?;
    
    Ok(())
}

/// Create summary CSV with aggregate metrics
///
/// # Arguments
/// * `output_dir` - Base output directory
/// * `filename` - Name of the input file (without extension)
/// * `subfolder` - Subfolder name for organization
/// * `mc_spectral_entropy` - Spectral entropy from MC analysis
/// * `ec_approximate_entropy` - Approximate entropy from EC analysis
/// * `ec_length` - Biological length from EC contour
/// * `mc_length` - Biological length from MC contour
/// * `ec_width` - Biological width from EC contour
/// * `mc_width` - Biological width from MC contour
/// * `ec_shape_index` - Shape index from EC analysis
/// * `mc_shape_index` - Shape index from MC analysis
/// * `outline_count` - Number of contour points
/// * `harmonic_chain_count` - Number of harmonic chains detected
///
/// # Output Columns
/// - ID
/// - Subfolder
/// - MC (Spectral entropy from margin complexity)
/// - EC (Approximate entropy from edge complexity)
/// - EC_Length, MC_Length
/// - EC_Width, MC_Width
/// - EC_ShapeIndex, MC_ShapeIndex
/// - Outline_Count
/// - Harmonic_Chain_Count
pub fn create_summary<P: AsRef<Path>>(
    output_dir: P,
    filename: &str,
    subfolder: &str,
    relative_path: &str,
    mc_spectral_entropy: f64,
    ec_approximate_entropy: f64,
    ec_length: f64,
    mc_length: f64,
    ec_width: f64,
    mc_width: f64,
    ec_shape_index: f64,
    mc_shape_index: f64,
    outline_count: u32,
    harmonic_chain_count: usize,
) -> Result<()> {
    // Summary goes directly in output directory
    let summary_path = output_dir.as_ref().join("summary.csv");
    
    // Check if summary file already exists
    let file_exists = summary_path.exists();
    
    // Open file in append mode if it exists, otherwise create new
    let mut writer = if file_exists {
        Writer::from_writer(fs::OpenOptions::new()
            .write(true)
            .append(true)
            .open(&summary_path)
            .map_err(|e| LeafComplexError::Io(e))?)
    } else {
        let mut writer = Writer::from_path(&summary_path)
            .map_err(|e| LeafComplexError::CsvOutput(e))?;
        
        // Write header for new file
        writer.write_record(&[
            "ID",
            "Subfolder",
            "Path",
            "MC",
            "EC",
            "EC_Length",
            "MC_Length",
            "EC_Width",
            "MC_Width",
            "EC_ShapeIndex",
            "MC_ShapeIndex",
            "Outline_Count",
            "Harmonic_Chain_Count",
        ]).map_err(|e| LeafComplexError::CsvOutput(e))?;
        
        writer
    };
    
    // Write data row
    writer.write_record(&[
        filename,
        subfolder,
        relative_path,
        &format!("{:.6}", mc_spectral_entropy),
        &format!("{:.6}", ec_approximate_entropy),
        &format!("{:.1}", ec_length),
        &format!("{:.1}", mc_length),
        &format!("{:.1}", ec_width),
        &format!("{:.1}", mc_width),
        &format!("{:.3}", ec_shape_index),
        &format!("{:.3}", mc_shape_index),
        &outline_count.to_string(),
        &harmonic_chain_count.to_string(),
    ]).map_err(|e| LeafComplexError::CsvOutput(e))?;
    
    // Flush writer
    writer.flush().map_err(|e| LeafComplexError::CsvOutput(csv::Error::from(e)))?;
    
    Ok(())
}
