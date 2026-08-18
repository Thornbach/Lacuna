// Revised src/thornfiddle.rs - Principled harmonic enhancement and continuous spectral entropy scaling

use std::path::Path;
use std::fs;
use std::collections::HashSet;
use rustfft::{FftPlanner, num_complex::Complex};
use csv::Writer;
use std::f64::consts::PI;
use image::RgbaImage;

use crate::errors::{LeafComplexError, Result};
use crate::feature_extraction::MarginalPointFeatures;
use crate::image_utils::has_rgb_color;
use crate::path_algorithms::trace_straight_line;

/// Represents a chain of consecutive golden pixel crossings (lobes)
#[derive(Debug, Clone)]
struct GoldenChain {
    start_index: usize,
    end_index: usize,
    length: usize,
    total_golden_pixels: u32,
    max_crossing_count: u32,
}

/// Result structure containing harmonic values, chain statistics, and weighted metrics
#[derive(Debug)]
pub struct HarmonicResult {
    pub harmonic_values: Vec<f64>,
    pub valid_chain_count: usize,
    pub total_chain_count: usize,
    pub weighted_chain_score: f64,
}

/// NEW: Calculate spectral entropy sigmoid scaling factor
/// S(CV) = 1 / (1 + exp(-k * (CV - c)))
/// This provides continuous scaling from near-zero for simple shapes to 1 for complex shapes
fn calculate_spectral_entropy_sigmoid_scaling(coefficient_of_variation: f64, k: f64, c: f64) -> f64 {
    let sigmoid = 1.0 / (1.0 + (-k * (coefficient_of_variation - c)).exp());
    sigmoid.powf(1.0) // Stays low, then climbs steeply
}

/// Extract contour signature using absolute distance deviations from mean radius
fn extract_contour_signature(contour: &[(u32, u32)], interpolation_points: usize) -> Vec<f64> {
    use crate::morphology::resample_contour;
    
    if contour.len() < 3 {
        return Vec::new();
    }
    
    // Resample contour to fixed number of points
    let resampled = resample_contour(contour, interpolation_points);
    if resampled.is_empty() {
        return Vec::new();
    }
    
    // Calculate centroid
    let n = resampled.len() as f64;
    let sum_x: f64 = resampled.iter().map(|&(x, _)| x as f64).sum();
    let sum_y: f64 = resampled.iter().map(|&(_, y)| y as f64).sum();
    
    let centroid_x = sum_x / n;
    let centroid_y = sum_y / n;
    
    // Calculate distances from centroid
    let distances: Vec<f64> = resampled.iter()
        .map(|&(x, y)| {
            let dx = x as f64 - centroid_x;
            let dy = y as f64 - centroid_y;
            (dx * dx + dy * dy).sqrt()
        })
        .collect();
    
    // Apply light smoothing to reduce digitization noise
    let smoothed_distances = smooth_signal(&distances, 2);
    
    // Calculate mean radius
    let mean_radius = smoothed_distances.iter().sum::<f64>() / n;
    
    // Return ABSOLUTE deviations from mean radius
    let absolute_deviations: Vec<f64> = smoothed_distances.iter()
        .map(|&d| (d - mean_radius).abs())
        .collect();
    
    absolute_deviations
}

/// Simple smoothing filter to reduce noise
fn smooth_signal(signal: &[f64], window_size: usize) -> Vec<f64> {
    if signal.len() < 3 || window_size == 0 {
        return signal.to_vec();
    }
    
    let mut smoothed = Vec::with_capacity(signal.len());
    let half_window = window_size / 2;
    
    for i in 0..signal.len() {
        let mut sum = 0.0;
        let mut count = 0;
        
        // Calculate window bounds
        let start = if i >= half_window { i - half_window } else { 0 };
        let end = std::cmp::min(i + half_window + 1, signal.len());
        
        // Average over window
        for j in start..end {
            sum += signal[j];
            count += 1;
        }
        
        smoothed.push(sum / count as f64);
    }
    
    smoothed
}

/// REVISED: Calculate spectral entropy from contour with continuous sigmoid scaling
pub fn calculate_spectral_entropy_from_contour(
    contour: &[(u32, u32)], 
    interpolation_points: usize,
    sigmoid_k: f64,
    sigmoid_c: f64,
) -> f64 {
    // Extract contour signature
    let signature = extract_contour_signature(contour, interpolation_points);
    if signature.is_empty() {
        return 0.0;
    }
    
    // Calculate coefficient of variation
    let mean = signature.iter().sum::<f64>() / signature.len() as f64;
    let variance = signature.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / signature.len() as f64;
    let std_dev = variance.sqrt();
    
    let coefficient_of_variation = if mean > 1e-6 { std_dev / mean } else { 0.0 };
    
    // Calculate raw spectral entropy
    let powers = calculate_power_spectrum_periodic(&signature);
    if powers.is_empty() {
        return 0.0;
    }
    
    let raw_entropy = calculate_shannon_entropy(&powers);
    
    // Apply continuous sigmoid scaling
    let sigmoid_scaling = calculate_spectral_entropy_sigmoid_scaling(coefficient_of_variation, sigmoid_k, sigmoid_c);
    
    raw_entropy * sigmoid_scaling
}

/// Legacy version for backward compatibility
pub fn calculate_spectral_entropy_from_contour_legacy(
    contour: &[(u32, u32)], 
    interpolation_points: usize
) -> f64 {
    // Use default sigmoid parameters
    calculate_spectral_entropy_from_contour(contour, interpolation_points, 20.0, 0.03)
}

/// Detect petiole sequence in a signal using outlier analysis.
///
/// Three-step approach:
///   1. Collect all contiguous runs of values > threshold (no extreme-value
///      filter yet, so we don't discard the tail of a wrap-around sequence).
///   2. Merge wrap-around: if the contour trace started mid-petiole the
///      petiole appears as two runs — one at index 0 and one at index n-1.
///      Merge them into one connected sequence.
///   3. Keep only runs that contain at least one value >= the 95th percentile
///      (outlier guard), then return the longest qualifying run.
pub fn detect_petiole_sequence(signal: &[f64], threshold: f64) -> Option<Vec<usize>> {
    if signal.is_empty() {
        return None;
    }
    let n = signal.len();

    // 95th-percentile outlier threshold
    let mut sorted_values = signal.to_vec();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let outlier_idx = ((n as f64 * 0.95) as usize).min(n - 1);
    let outlier_threshold = sorted_values[outlier_idx];

    println!("Petiole detection - Outlier threshold (95th percentile): {}", outlier_threshold);

    // Step 1: Collect all contiguous runs of values above threshold.
    let mut raw_sequences: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();

    for i in 0..n {
        if signal[i] > threshold {
            current.push(i);
        } else if !current.is_empty() {
            raw_sequences.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        raw_sequences.push(current);
    }

    // Step 2: Merge wrap-around sequences.
    // If both the first run starts at index 0 and the last run ends at index n-1,
    // they are the two halves of one petiole sequence that straddles the
    // contour's start/end boundary.  Merge tail + head (circular order).
    if raw_sequences.len() >= 2 {
        let first_at_zero = raw_sequences.first().and_then(|s| s.first()).copied() == Some(0);
        let last_at_end  = raw_sequences.last().and_then(|s| s.last()).copied()  == Some(n - 1);
        if first_at_zero && last_at_end {
            let tail = raw_sequences.pop().unwrap();   // ends at n-1
            let head = raw_sequences.remove(0);        // starts at 0
            let mut merged = tail;  // tail comes first in circular order
            merged.extend(head);
            raw_sequences.push(merged);
        }
    }

    // Step 3: Filter to runs that contain at least one extreme value,
    // then return the longest.
    let longest = raw_sequences
        .into_iter()
        .filter(|seq| seq.iter().any(|&i| signal[i] >= outlier_threshold))
        .max_by_key(|s| s.len());

    match longest {
        Some(seq) => {
            println!("Detected petiole sequence with {} points", seq.len());
            Some(seq)
        }
        None => {
            println!("No significant petiole sequence detected");
            None
        }
    }
}

/// Apply petiole filter to a signal
pub fn apply_petiole_filter(signal: &[f64], petiole_indices: &[usize], remove_completely: bool) -> Vec<f64> {
    if petiole_indices.is_empty() {
        return signal.to_vec();
    }
    
    if remove_completely {
        // Mode 2: Remove petiole completely and merge loose ends
        let mut filtered_signal = Vec::new();
        let petiole_set: std::collections::HashSet<usize> = petiole_indices.iter().cloned().collect();
        
        for (i, &value) in signal.iter().enumerate() {
            if !petiole_set.contains(&i) {
                filtered_signal.push(value);
            }
        }
        
        println!("Petiole removal: {} -> {} points", signal.len(), filtered_signal.len());
        filtered_signal
    } else {
        // Mode 1: Set petiole values to zero
        let mut filtered_signal = signal.to_vec();
        for &idx in petiole_indices {
            if idx < filtered_signal.len() {
                filtered_signal[idx] = 0.0;
            }
        }
        
        println!("Petiole zeroing: {} points set to zero", petiole_indices.len());
        filtered_signal
    }
}

/// Apply threshold filter to pink path values
pub fn apply_pink_threshold_filter(
    features: &mut [MarginalPointFeatures],
    enable_threshold_filter: bool,
    threshold: f64,
) {
    if !enable_threshold_filter {
        return;
    }
    
    let mut filtered_count = 0;
    
    for feature in features.iter_mut() {
        if let Some(pink_value) = feature.diego_path_pink {
            if (pink_value as f64) <= threshold {
                feature.diego_path_pink = Some(0);
                filtered_count += 1;
            }
        }
    }
    
    if filtered_count > 0 {
        println!("Pink threshold filter: {} values <= {:.1} set to zero", filtered_count, threshold);
    }
}

/// Filter petiole from LEC features with optional threshold filtering
pub fn filter_petiole_from_ec_features(
    features: &[MarginalPointFeatures],
    enable_petiole_filter: bool,
    remove_completely: bool,
    threshold: f64,
    enable_pink_threshold_filter: bool,
    pink_threshold: f64,
) -> (Vec<MarginalPointFeatures>, Option<Vec<usize>>) {
    if !enable_petiole_filter && !enable_pink_threshold_filter {
        return (features.to_vec(), None);
    }
    
    let mut working_features = features.to_vec();
    let mut petiole_indices = None;
    
    // Step 1: Apply petiole filtering if enabled
    if enable_petiole_filter && !working_features.is_empty() {
        let pink_signal = extract_pink_path_signal(&working_features);

        petiole_indices = detect_petiole_sequence(&pink_signal, threshold);
        
        if let Some(ref indices) = petiole_indices {
            if remove_completely {
                let petiole_set: HashSet<usize> = indices.iter().cloned().collect();
                let filtered_features: Vec<MarginalPointFeatures> = working_features.iter()
                    .enumerate()
                    .filter(|(i, _)| !petiole_set.contains(i))
                    .map(|(_, feature)| feature.clone())
                    .collect();
                
                // Keep original point_index values — they are contour-array positions
                // used by the GUI to look up pixels in ec_contour_original.
                // Overwriting them with sequential 0,1,2... would cause the GUI
                // to fetch the wrong contour pixel for every filtered feature.
                working_features = filtered_features;
                
                println!("LEC petiole removal: {} -> {} features", features.len(), working_features.len());
            } else {
                for &idx in indices {
                    if idx < working_features.len() {
                        working_features[idx].diego_path_pink = Some(0);
                    }
                }
                
                println!("LEC petiole zeroing: {} features modified", indices.len());
            }
        }
    }
    
    // Step 2: Apply pink threshold filtering if enabled
    apply_pink_threshold_filter(&mut working_features, enable_pink_threshold_filter, pink_threshold);
    
    (working_features, petiole_indices)
}

/// Calculate Edge Complexity using the provided algorithm with configurable petiole filtering
pub fn calculate_edge_feature_density(
    colored_path_values: &[f64],
    enable_petiole_filter: bool,
    petiole_remove_completely: bool,
    scaling_factor: f64,
) -> Result<f64> {
    if colored_path_values.is_empty() {
        return Err(LeafComplexError::Other("Empty colored path values for edge feature density calculation".to_string()));
    }
    
    let threshold = 1.0;
    
    let filtered_values = if enable_petiole_filter {
        if let Some(petiole_indices) = detect_petiole_sequence(colored_path_values, threshold) {
            apply_petiole_filter(colored_path_values, &petiole_indices, petiole_remove_completely)
        } else {
            colored_path_values.to_vec()
        }
    } else {
        colored_path_values.to_vec()
    };
    
    let feature_points = filtered_values.iter()
        .filter(|&&v| v > threshold)
        .count();
    
    let total_points = filtered_values.len();
    
    let sum_feature_magnitudes = filtered_values.iter()
        .filter(|&&v| v > threshold)
        .sum::<f64>();
    
    if total_points > 0 {
        let density = feature_points as f64 / total_points as f64;
        
        let avg_magnitude = if feature_points > 0 {
            sum_feature_magnitudes / feature_points as f64
        } else {
            0.0
        };
        
        let edge_complexity = density * (1.0 + avg_magnitude.sqrt()) * scaling_factor;          

        println!("Feature density: {:.4}, Avg magnitude: {:.4}, Edge complexity: {:.4}", 
                 density, avg_magnitude, edge_complexity);
        
        Ok(edge_complexity)
    } else {
        Ok(0.0)
    }
}

/// Apply periodic-aware Gaussian smoothing to a signal
pub fn periodic_gaussian_smooth(signal: &[f64], window_size: usize, sigma: f64) -> Vec<f64> {
    if signal.len() < 3 || window_size == 0 {
        return signal.to_vec();
    }
    
    let n = signal.len();
    let mut smoothed = Vec::with_capacity(n);
    
    // Generate Gaussian weights
    let half_window = window_size / 2;
    let mut weights = Vec::with_capacity(window_size);
    let mut weight_sum = 0.0;
    
    for i in 0..window_size {
        let offset = i as f64 - half_window as f64;
        let weight = (-0.5 * (offset / sigma).powi(2)).exp();
        weights.push(weight);
        weight_sum += weight;
    }
    
    // Normalize weights
    for weight in &mut weights {
        *weight /= weight_sum;
    }
    
    // Apply smoothing with periodic boundary conditions
    for i in 0..n {
        let mut weighted_sum = 0.0;
        
        for j in 0..window_size {
            let offset = j as i32 - half_window as i32;
            let idx = ((i as i32 + offset) + n as i32) % n as i32;
            let idx = if idx < 0 { idx + n as i32 } else { idx } as usize;
            
            weighted_sum += signal[idx] * weights[j];
        }
        
        smoothed.push(weighted_sum);
    }
    
    smoothed
}

/// Calculate power spectrum using FFT for periodic signal
fn calculate_power_spectrum_periodic(signal: &[f64]) -> Vec<f64> {
    if signal.len() < 4 {
        return Vec::new();
    }
    
    // Check if signal has significant variation
    let mean = signal.iter().sum::<f64>() / signal.len() as f64;
    let variance = signal.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / signal.len() as f64;
    
    if variance < 1e-6 {
        return Vec::new();
    }
    
    let mut fft_size = signal.len();
    
    // Convert to complex numbers (remove DC component)
    let mut complex_input: Vec<Complex<f64>> = signal
        .iter()
        .map(|&x| Complex::new(x - mean, 0.0))
        .collect();
    
    // Extend to next power of 2 for efficiency if needed
    if !fft_size.is_power_of_two() {
        fft_size = fft_size.next_power_of_two();
        complex_input.resize(fft_size, Complex::new(0.0, 0.0));
    }
    
    // Perform FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    fft.process(&mut complex_input);
    
    // Calculate power spectrum (magnitude squared)
    let mut powers = Vec::with_capacity(fft_size / 2);
    
    // Skip DC component (index 0) and use only positive frequencies
    for i in 1..fft_size / 2 {
        powers.push(complex_input[i].norm_sqr());
    }
    
    // Normalize so powers sum to 1
    let total_power: f64 = powers.iter().sum();
    if total_power > 0.0 {
        for p in &mut powers {
            *p /= total_power;
        }
    }
    
    powers
}

/// Calculate Shannon entropy from normalized power spectrum
fn calculate_shannon_entropy(powers: &[f64]) -> f64 {
    if powers.is_empty() {
        return 0.0;
    }
    
    // Calculate Shannon entropy: -Σ(p * log2(p))
    let entropy = -powers.iter()
        .filter(|&&p| p > 1e-12)
        .map(|&p| p * p.log2())
        .sum::<f64>();
    
    // Normalize by maximum possible entropy
    let max_entropy = (powers.len() as f64).log2();
    if max_entropy > 1e-6 {
        entropy / max_entropy
    } else {
        0.0
    }
}

pub fn calculate_spectral_entropy_from_harmonic_thornfiddle_path(
    features: &[MarginalPointFeatures],
    chain_count: usize,  // NEW: Pass chain count for linear scaling
    smoothing_strength: f64,
    sigmoid_k: f64,
    sigmoid_c: f64,
) -> (f64, Vec<f64>) {
    if features.is_empty() {
        return (0.0, Vec::new());
    }
    
    // Extract Harmonic Thornfiddle Path signal
    let harmonic_signal = extract_harmonic_thornfiddle_path_signal(features);
    
    if harmonic_signal.len() < 4 {
        return (0.0, harmonic_signal);
    }
    
    // Apply periodic-aware Gaussian smoothing
    let window_size = (harmonic_signal.len() / 8).max(3).min(21);
    let sigma = smoothing_strength.max(0.5);
    let smoothed_signal = periodic_gaussian_smooth(&harmonic_signal, window_size, sigma);
    
    // Calculate coefficient of variation
    let mean = smoothed_signal.iter().sum::<f64>() / smoothed_signal.len() as f64;
    let variance = smoothed_signal.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / smoothed_signal.len() as f64;
    let std_dev = variance.sqrt();
    
    let coefficient_of_variation = if mean > 1e-6 { std_dev / mean } else { 0.0 };
    
    // Calculate raw spectral entropy
    let powers = calculate_power_spectrum_periodic(&smoothed_signal);
    if powers.is_empty() {
        return (0.0, smoothed_signal);
    }
    
    let raw_entropy = calculate_shannon_entropy(&powers);

    // Apply sigmoid scaling
    let sigmoid_scaling = calculate_spectral_entropy_sigmoid_scaling(coefficient_of_variation, sigmoid_k, sigmoid_c);

    // NEW: Apply Weber-Fechner chain factor scaling with minimum baseline
    let chain_factor = if chain_count == 0 {
        0.1  // 10% minimum for leaves with no chains (digitization noise baseline)
    } else {
        (1.0 + chain_count as f64).ln() / (1.0 + 10.0_f64).ln()
    };
    let final_entropy = raw_entropy * sigmoid_scaling * chain_factor;
    let final_entropy = final_entropy.max(raw_entropy * 0.01); // Ensure minimum 1% of raw entropy

    (final_entropy, smoothed_signal)
}

/// Legacy version for backward compatibility (assumes 0 chains)
pub fn calculate_spectral_entropy_from_thornfiddle_path(
    features: &[MarginalPointFeatures],
    smoothing_strength: f64
) -> (f64, Vec<f64>) {
    // Use default sigmoid parameters and 0 chains for legacy compatibility
    calculate_spectral_entropy_from_harmonic_thornfiddle_path(features, 0, smoothing_strength, 20.0, 0.03)
}



// (Removed duplicate legacy function definition)

/// Calculate approximate entropy from Pink Path
pub fn calculate_approximate_entropy_from_pink_path(
    features: &[MarginalPointFeatures],
    m: usize,
    r: f64,
) -> f64 {
    if features.is_empty() {
        return 0.0;
    }
    
    let pink_signal = extract_pink_path_signal(features);
    
    if pink_signal.len() < 4 {
        return 0.0;
    }
    
    let mean = pink_signal.iter().sum::<f64>() / pink_signal.len() as f64;
    let variance = pink_signal.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / pink_signal.len() as f64;
    let std_dev = variance.sqrt();
    
    let adaptive_r = if std_dev > 1e-6 {
        r * std_dev
    } else {
        r
    };
    
    calculate_approximate_entropy(&pink_signal, m, adaptive_r)
}

/// Calculate approximate entropy for a given signal
pub fn calculate_approximate_entropy(signal: &[f64], m: usize, r: f64) -> f64 {
    let n = signal.len();
    if n <= m {
        return 0.0;
    }
    
    let phi_m = calculate_phi(signal, m, r);
    let phi_m1 = calculate_phi(signal, m + 1, r);
    
    phi_m - phi_m1
}

/// Helper function to calculate phi for approximate entropy
fn calculate_phi(signal: &[f64], m: usize, r: f64) -> f64 {
    let n = signal.len();
    let mut sum = 0.0;
    
    for i in 0..=(n - m) {
        let mut matches = 0;
        
        for j in 0..=(n - m) {
            let max_diff = calculate_max_distance(&signal[i..i+m], &signal[j..j+m]);
            
            if max_diff <= r {
                matches += 1;
            }
        }
        
        let ratio = matches as f64 / (n - m + 1) as f64;
        if ratio > 1e-12 {
            sum += ratio.ln();
        }
    }
    
    sum / (n - m + 1) as f64
}

/// Calculate maximum distance between two pattern vectors
fn calculate_max_distance(pattern1: &[f64], pattern2: &[f64]) -> f64 {
    pattern1.iter()
        .zip(pattern2.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0, |acc, diff| acc.max(diff))
}

/// Create Thornfiddle summary CSV with weighted chain metrics
pub fn create_thornfiddle_summary<P: AsRef<Path>>(
    output_dir: P,
    filename: &str,
    subfolder: &str,
    spectral_entropy: f64,
    spectral_entropy_pink: f64,
    spectral_entropy_contour: f64,
    approximate_entropy: f64,
    edge_complexity: f64,
    lec_circularity: f64,
    lmc_circularity: f64,
    area: u32,
    lec_length: f64,
    lec_width: f64,
    lec_shape_index: f64,
    lmc_length: f64,
    lmc_width: f64,
    lmc_shape_index: f64,
    dynamic_opening_percentage: f64,
    dynamic_kernel_size: u32,
    outline_count: u32,
    harmonic_chain_count: usize,
    weighted_chain_score: f64,
) -> Result<()> {
    let thornfiddle_dir = output_dir.as_ref().join("Thornfiddle");
    fs::create_dir_all(&thornfiddle_dir).map_err(|e| LeafComplexError::Io(e))?;
    
    let summary_path = thornfiddle_dir.join("summary.csv");
    
    let file_exists = summary_path.exists();
    
    let mut writer = if file_exists {
        Writer::from_writer(fs::OpenOptions::new()
            .write(true)
            .append(true)
            .open(&summary_path)
            .map_err(|e| LeafComplexError::Io(e))?)
    } else {
        let mut writer = Writer::from_path(&summary_path)
            .map_err(|e| LeafComplexError::CsvOutput(e))?;
        
        // Write header with weighted chain metrics
        writer.write_record(&[
            "ID",
            "Subfolder",
            "Spectral_Entropy",
            "Spectral_Entropy_Pink",
            "Spectral_Entropy_Contour",
            "Approximate_Entropy",
            "Edge_Complexity",
            "LEC_Circularity",
            "LMC_Circularity",
            "Area",
            "LEC_Length",
            "LEC_Width",
            "LEC_ShapeIndex",
            "LMC_Length",
            "LMC_Width",
            "LMC_ShapeIndex",
            "Dynamic_Opening_Percentage",
            "Dynamic_Kernel_Size",
            "Outline_Count",
            "Harmonic_Chain_Count",
            "Weighted_Chain_Score",
        ]).map_err(|e| LeafComplexError::CsvOutput(e))?;
        
        writer
    };
    
    // Write data with weighted chain metrics
    writer.write_record(&[
        filename,
        subfolder,
        &format!("{:.6}", spectral_entropy),
        &format!("{:.6}", spectral_entropy_pink),
        &format!("{:.6}", spectral_entropy_contour),
        &format!("{:.6}", approximate_entropy),
        &format!("{:.6}", edge_complexity),
        &format!("{:.6}", lec_circularity),
        &format!("{:.6}", lmc_circularity),
        &area.to_string(),
        &format!("{:.1}", lec_length),
        &format!("{:.1}", lec_width),
        &format!("{:.3}", lec_shape_index),
        &format!("{:.1}", lmc_length),
        &format!("{:.1}", lmc_width),
        &format!("{:.3}", lmc_shape_index),
        &format!("{:.1}", dynamic_opening_percentage),
        &dynamic_kernel_size.to_string(),
        &outline_count.to_string(),
        &harmonic_chain_count.to_string(),
        &format!("{:.2}", weighted_chain_score),
    ]).map_err(|e| LeafComplexError::CsvOutput(e))?;
    
    writer.flush().map_err(|e| LeafComplexError::CsvOutput(csv::Error::from(e)))?;
    
    Ok(())
}

/// Calculate a simple Thornfiddle Multiplier based on path complexity
pub fn calculate_thornfiddle_multiplier(feature: &MarginalPointFeatures) -> f64 {
    if feature.straight_path_length <= 0.0 || feature.diego_path_length <= 0.0 {
        return 1.0;
    }
    
    let path_ratio = feature.diego_path_length / feature.straight_path_length;
    let base_multiplier = path_ratio.max(1.0);
    
    base_multiplier
}

/// Calculate Thornfiddle Path with simple multiplier
pub fn calculate_thornfiddle_path(feature: &MarginalPointFeatures) -> f64 {
    let multiplier = calculate_thornfiddle_multiplier(feature);
    feature.diego_path_length * multiplier
}

/// REVISED: Calculate Thornfiddle Path with principled harmonic enhancement
pub fn calculate_thornfiddle_path_harmonic(
    features: &[MarginalPointFeatures],
    leaf_circumference: f64,
    thornfiddle_image: &RgbaImage,
    reference_point: (u32, u32),
    contour_points: &[(u32, u32)],
    golden_color: [u8; 3],
    pixel_threshold: u32,
    min_chain_length: usize,
    harmonic_strength_multiplier: f64,
    max_harmonics: usize,
) -> HarmonicResult {
    if features.is_empty() {
        return HarmonicResult {
            harmonic_values: Vec::new(),
            valid_chain_count: 0,
            total_chain_count: 0,
            weighted_chain_score: 0.0,
        };
    }
    
    println!("Calculating principled harmonic Thornfiddle with geometric enhancement");
    println!("Parameters: pixel_threshold={}, min_chain_length={}, max_harmonics={}, harmonic_strength={}",
             pixel_threshold, min_chain_length, max_harmonics, harmonic_strength_multiplier);
    
    // Step 1: Detect golden chains based on pixel crossings
    let golden_chains = detect_golden_chains(
        features,
        thornfiddle_image,
        reference_point,
        contour_points,
        golden_color,
        pixel_threshold,
    );
    
    let total_chain_count = golden_chains.len();
    
    // Step 2: Filter chains by minimum length requirement
    let valid_chains: Vec<&GoldenChain> = golden_chains.iter()
        .filter(|chain| chain.length >= min_chain_length)
        .collect();
    
    let valid_chain_count = valid_chains.len();
    
    // Step 2.5: Calculate weighted chain score (chain intensity weighting)
    let weighted_chain_score: f64 = valid_chains.iter()
        .map(|chain| (chain.total_golden_pixels as f64) * (chain.length as f64))
        .sum();
    
    println!("Detected {} total chains, {} valid chains (>= {} points), weighted score: {:.1}", 
             total_chain_count, valid_chain_count, min_chain_length, weighted_chain_score);
    
    // Step 3: Calculate base Thornfiddle values
    let base_thornfiddle: Vec<f64> = features.iter()
        .map(|feature| calculate_thornfiddle_path(feature))
        .collect();
    
    // Step 4: Apply principled harmonics to each valid chain
    let mut harmonic_thornfiddle = base_thornfiddle.clone();
    
    for chain in valid_chains.iter() {
        apply_principled_harmonic_enhancement(
            &mut harmonic_thornfiddle,
            chain,
            leaf_circumference,
            harmonic_strength_multiplier,
            max_harmonics,
            features, // Pass the features to access path lengths
        );
    }
    
    println!("Principled harmonic enhancement complete - {} valid chains processed, weighted score: {:.1}", 
             valid_chain_count, weighted_chain_score);
    
    HarmonicResult {
        harmonic_values: harmonic_thornfiddle,
        valid_chain_count,
        total_chain_count,
        weighted_chain_score,
    }
}

/// NEW: Apply principled harmonic enhancement based on deepest point weighting
fn apply_principled_harmonic_enhancement(
    harmonic_values: &mut [f64],
    chain: &GoldenChain,
    leaf_circumference: f64,
    harmonic_strength_multiplier: f64,
    max_harmonics: usize,
    features: &[MarginalPointFeatures], // Added to access path lengths
) {
    // Calculate segment length relative to leaf circumference
    let segment_length = chain.length as f64;
    let circumference_ratio = segment_length / leaf_circumference;
    
    // Enhancement Richness: N_h = floor(N_max * (L_seg / C_leaf))
    let num_harmonics = ((max_harmonics as f64) * circumference_ratio).floor() as usize;
    let num_harmonics = num_harmonics.max(1).min(max_harmonics);
    
    // Base frequency normalized by leaf circumference
    let base_frequency = 2.0 * PI / leaf_circumference;
    
    // CORRECTED: Find the deepest point (point with longest geodesic path) in the segment
    let mut deepest_point_idx = chain.start_index;
    let mut max_path_length = 0.0;
    
    for i in chain.start_index..=chain.end_index {
        if i < features.len() {
            let path_length = features[i].diego_path_length;
            if path_length > max_path_length {
                max_path_length = path_length;
                deepest_point_idx = i;
            }
        }
    }
    
    println!("Chain enhancement: length={}, circumference_ratio={:.4}, harmonics={}, deepest_point_idx={}", 
             segment_length, circumference_ratio, num_harmonics, deepest_point_idx);
    
    // Apply harmonic enhancement to each point in the chain
    for i in chain.start_index..=chain.end_index {
        if i >= harmonic_values.len() {
            break;
        }
        
        // CORRECTED: Enhancement Intensity based on distance from deepest point
        let distance_from_deepest = if deepest_point_idx >= chain.start_index && deepest_point_idx <= chain.end_index {
            // Calculate relative distance from the deepest point within the segment
            let dist_to_deepest = (i as i32 - deepest_point_idx as i32).abs() as f64;
            let max_distance_in_segment = ((chain.end_index - chain.start_index) as f64 / 2.0).max(1.0);
            
            // W_pos: 1.0 at deepest point, decreasing linearly to 0.0 at segment ends
            1.0 - (dist_to_deepest / max_distance_in_segment).min(1.0)
        } else {
            // Fallback to linear if deepest point detection fails
            let position_in_chain = i - chain.start_index;
            let chain_length = chain.end_index - chain.start_index + 1;
            position_in_chain as f64 / chain_length as f64
        };
        
        // Calculate harmonic sum: Σ (1/k) * sin(2π * k * f_base * L_pos)
        let mut harmonic_sum = 0.0;
        for k in 1..=num_harmonics {
            let amplitude = 1.0 / k as f64; // Standard harmonic series decay
            let phase = base_frequency * k as f64 * (i - chain.start_index) as f64;
            harmonic_sum += amplitude * phase.sin();
        }
        
        // Enhancement = W_pos * harmonic_sum * strength_multiplier
        let enhancement = distance_from_deepest * harmonic_sum * harmonic_strength_multiplier;
        
        // Apply enhancement to base value
        let base_value = harmonic_values[i];
        harmonic_values[i] = base_value + (base_value * enhancement);
        
        if i == deepest_point_idx {
            println!("  Deepest point enhancement: W_pos={:.3}, enhancement={:.3}", 
                     distance_from_deepest, enhancement);
        }
    }
}

/// Count golden pixels crossed by a path
fn count_golden_pixels_crossed(
    path: &[(u32, u32)],
    thornfiddle_image: &RgbaImage,
    golden_color: [u8; 3],
) -> u32 {
    let mut golden_count = 0;
    let (width, height) = thornfiddle_image.dimensions();
    
    for &(x, y) in path {
        if x < width && y < height {
            let pixel = thornfiddle_image.get_pixel(x, y);
            if has_rgb_color(pixel, golden_color) {
                golden_count += 1;
            }
        }
    }
    
    golden_count
}

/// Detect chains of consecutive golden pixel crossings
fn detect_golden_chains(
    features: &[MarginalPointFeatures],
    thornfiddle_image: &RgbaImage,
    reference_point: (u32, u32),
    contour_points: &[(u32, u32)],
    golden_color: [u8; 3],
    pixel_threshold: u32,
) -> Vec<GoldenChain> {
    let mut chains = Vec::new();
    let mut current_chain_start: Option<usize> = None;
    let mut chain_golden_counts = Vec::new();
    
    for (i, _) in features.iter().enumerate() {
        if i >= contour_points.len() {
            break;
        }
        
        let marginal_point = contour_points[i];
        
        // For MC we always trace the straight line to evaluate golden crossings
        let path_to_check = trace_straight_line(reference_point, marginal_point);
        
        let golden_count = count_golden_pixels_crossed(&path_to_check, thornfiddle_image, golden_color);
        let crosses_threshold = golden_count >= pixel_threshold;
        
        if crosses_threshold {
            if current_chain_start.is_none() {
                current_chain_start = Some(i);
                chain_golden_counts.clear();
            }
            chain_golden_counts.push(golden_count);
        } else if let Some(start) = current_chain_start {
            if !chain_golden_counts.is_empty() {
                let total_golden_pixels: u32 = chain_golden_counts.iter().sum();
                let max_crossing_count = *chain_golden_counts.iter().max().unwrap_or(&0);
                
                chains.push(GoldenChain {
                    start_index: start,
                    end_index: i - 1,
                    length: i - start,
                    total_golden_pixels,
                    max_crossing_count,
                });
                
                println!("Golden chain detected: indices {}-{}, length {}, total golden pixels {}", 
                         start, i - 1, i - start, total_golden_pixels);
            }
            current_chain_start = None;
            chain_golden_counts.clear();
        }
    }
    
    // Handle chain that extends to end of contour
    if let Some(start) = current_chain_start {
        if !chain_golden_counts.is_empty() {
            let total_golden_pixels: u32 = chain_golden_counts.iter().sum();
            let max_crossing_count = *chain_golden_counts.iter().max().unwrap_or(&0);
            
            chains.push(GoldenChain {
                start_index: start,
                end_index: features.len() - 1,
                length: features.len() - start,
                total_golden_pixels,
                max_crossing_count,
            });
            
            println!("Golden chain detected (end): indices {}-{}, length {}, total golden pixels {}", 
                     start, features.len() - 1, features.len() - start, total_golden_pixels);
        }
    }
    
    chains
}

/// Calculate leaf circumference from contour points
pub fn calculate_leaf_circumference(contour: &[(u32, u32)]) -> f64 {
    if contour.len() < 2 {
        return 0.0;
    }
    
    let mut circumference = 0.0;
    for i in 0..contour.len() {
        let current = contour[i];
        let next = contour[(i + 1) % contour.len()];
        
        let dx = next.0 as f64 - current.0 as f64;
        let dy = next.1 as f64 - current.1 as f64;
        circumference += (dx * dx + dy * dy).sqrt();
    }
    
    circumference
}

/// Extract Thornfiddle Path values from features
pub fn extract_thornfiddle_path_signal(features: &[MarginalPointFeatures]) -> Vec<f64> {
    features.iter()
        .map(|feature| calculate_thornfiddle_path(feature))
        .collect()
}

/// Extract Harmonic Thornfiddle Path values from features  
pub fn extract_harmonic_thornfiddle_path_signal(features: &[MarginalPointFeatures]) -> Vec<f64> {
    features.iter()
        .map(|feature| feature.thornfiddle_path_harmonic)
        .collect()
}

/// Extract Pink Path values from LEC features
pub fn extract_pink_path_signal(features: &[MarginalPointFeatures]) -> Vec<f64> {
    features.iter()
        .map(|feature| feature.diego_path_pink.unwrap_or(0) as f64)
        .collect()
}

/// Extract path-length ratio signal: diego_path_length / straight_path_length.
///
/// A ratio of 1.0 means the geodesic was a straight line (no detour needed).
/// A ratio >> 1.0 means the path had to navigate around a concavity — which
/// is exactly what the petiole attachment creates.  This makes the ratio a
/// routing-based but pink-independent signal for petiole detection.
pub fn extract_path_ratio_signal(features: &[MarginalPointFeatures]) -> Vec<f64> {
    features.iter()
        .map(|feature| {
            let s = feature.straight_path_length;
            if s < 1e-9 {
                1.0
            } else {
                feature.diego_path_length / s
            }
        })
        .collect()
}

/// REVISED: Calculate spectral entropy from Pink Path signal with continuous sigmoid scaling
pub fn calculate_spectral_entropy_from_pink_path(
    features: &[MarginalPointFeatures],
    sigmoid_k: f64,
    sigmoid_c: f64,
) -> f64 {
    if features.is_empty() {
        return 0.0;
    }
    
    let pink_signal = extract_pink_path_signal(features);
    
    if pink_signal.len() < 4 {
        return 0.0;
    }
    
    // Calculate coefficient of variation
    let mean = pink_signal.iter().sum::<f64>() / pink_signal.len() as f64;
    let variance = pink_signal.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / pink_signal.len() as f64;
    let std_dev = variance.sqrt();
    
    let coefficient_of_variation = if mean > 1e-6 { std_dev / mean } else { 0.0 };
    
    // Calculate raw spectral entropy
    let powers = calculate_power_spectrum_periodic(&pink_signal);
    if powers.is_empty() {
        return 0.0;
    }
    
    let raw_entropy = calculate_shannon_entropy(&powers);
    
    // Apply continuous sigmoid scaling
    let sigmoid_scaling = calculate_spectral_entropy_sigmoid_scaling(coefficient_of_variation, sigmoid_k, sigmoid_c);
    
    raw_entropy * sigmoid_scaling
}