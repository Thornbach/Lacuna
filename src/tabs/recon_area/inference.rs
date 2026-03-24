use std::{
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
    sync::atomic::{AtomicBool, Ordering},
};

use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn_core::{module::Module, record::CompactRecorder};

use crate::tabs::recon_train::model::{InferBackend, create_infer_device, backend_name};
use super::model::LeafAreaRegressor;

type InferDevice = <InferBackend as Backend>::Device;

// ── Public types ──────────────────────────────────────────────────────────────

/// Configuration for a batch area-inference run
pub struct AreaInferConfig {
    /// Folder containing `area_regressor.mpk`
    pub checkpoint_path: PathBuf,
    pub source_folder:   PathBuf,
    pub output_csv:      PathBuf,
    pub image_size:      usize,
    pub species_label:   String,
}

/// Per-image inference result
pub struct AreaResult {
    pub filename:           String,
    pub path:               String,
    pub surviving_px:       usize,
    pub predicted_total_px: usize,
    pub damage_px:          usize,
    pub pct_damage:         f32,
}

/// Messages sent from inference thread → UI thread
pub enum AreaInferMsg {
    Progress { done: usize, total: usize },
    Result(AreaResult),
    Log(String),
    Finished,
    Error(String),
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn spawn_inference(
    cfg:    AreaInferConfig,
    tx:     mpsc::Sender<AreaInferMsg>,
    cancel: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        match run_inference(&cfg, &tx, &cancel) {
            Err(e) => { let _ = tx.send(AreaInferMsg::Error(e)); }
            Ok(()) => {}
        }
        let _ = tx.send(AreaInferMsg::Finished);
    });
}

// ── Main inference function ───────────────────────────────────────────────────

fn run_inference(
    cfg:    &AreaInferConfig,
    tx:     &mpsc::Sender<AreaInferMsg>,
    cancel: &AtomicBool,
) -> Result<(), String> {
    log(tx, format!("Requesting {} device…", backend_name()));
    let device: InferDevice = create_infer_device();
    {
        let probe = Tensor::<InferBackend, 1>::zeros([1], &device);
        let _ = probe.into_data();
    }
    log(tx, format!("{} device ready.", backend_name()));

    // Load model weights
    log(tx, format!("Loading checkpoint: {}", cfg.checkpoint_path.display()));
    let rec = CompactRecorder::new();
    let model: LeafAreaRegressor<InferBackend> = LeafAreaRegressor::<InferBackend>::init(&device)
        .load_file(cfg.checkpoint_path.join("area_regressor"), &rec, &device)
        .map_err(|e| format!("Failed to load area_regressor: {e}"))?;
    log(tx, "Model loaded. Running inference…");

    // Collect image paths
    let image_paths: Vec<PathBuf> = walkdir::WalkDir::new(&cfg.source_folder)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file() && is_image(e.path()))
        .map(|e| e.into_path())
        .collect();

    if image_paths.is_empty() {
        return Err("No images found in source folder.".to_string());
    }

    let total = image_paths.len();
    let mut all_results: Vec<AreaResult> = Vec::with_capacity(total);

    for (idx, path) in image_paths.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            log(tx, "Inference cancelled.");
            break;
        }

        let stem = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image");

        match infer_one(path, &model, &device, cfg) {
            Ok(result) => {
                let _ = tx.send(AreaInferMsg::Progress { done: idx + 1, total });
                // Send a copy to UI (borrow AreaResult fields before moving)
                let filename           = result.filename.clone();
                let file_path          = result.path.clone();
                let surviving_px       = result.surviving_px;
                let predicted_total_px = result.predicted_total_px;
                let damage_px          = result.damage_px;
                let pct_damage         = result.pct_damage;

                let _ = tx.send(AreaInferMsg::Result(AreaResult {
                    filename:           filename.clone(),
                    path:               file_path.clone(),
                    surviving_px,
                    predicted_total_px,
                    damage_px,
                    pct_damage,
                }));

                all_results.push(AreaResult {
                    filename,
                    path: file_path,
                    surviving_px,
                    predicted_total_px,
                    damage_px,
                    pct_damage,
                });
            }
            Err(e) => {
                log(tx, format!("Skipped {stem}: {e}"));
                let _ = tx.send(AreaInferMsg::Progress { done: idx + 1, total });
            }
        }
    }

    // Write CSV
    if !all_results.is_empty() {
        log(tx, format!("Writing CSV: {}", cfg.output_csv.display()));
        match write_csv(&all_results, &cfg.output_csv) {
            Ok(()) => log(tx, format!("CSV saved ({} rows).", all_results.len())),
            Err(e) => log(tx, format!("Warning: CSV write failed — {e}")),
        }
    }

    Ok(())
}

// ── Per-image inference ───────────────────────────────────────────────────────

fn infer_one(
    path:   &Path,
    model:  &LeafAreaRegressor<InferBackend>,
    device: &InferDevice,
    cfg:    &AreaInferConfig,
) -> Result<AreaResult, String> {
    let sz   = cfg.image_size;
    let sz32 = sz as u32;
    let n    = sz * sz;

    // Load and resize
    let img  = image::open(path).map_err(|e| format!("{e}"))?;
    let img  = img.resize_exact(sz32, sz32, image::imageops::FilterType::Triangle);
    let mut rgba = img.to_rgba8().into_raw();

    // Zero RGB under transparent pixels — matches training convention
    for p in rgba.chunks_mut(4) {
        if p[3] <= 128 { p[0] = 0; p[1] = 0; p[2] = 0; p[3] = 0; }
    }

    // Count surviving pixels (at image_size scale, before building tensor)
    let surviving_px: usize = rgba.chunks(4).filter(|p| p[3] > 128).count();

    // Build [1, 5, sz, sz] input tensor
    let mut input_data = Vec::with_capacity(5 * n);
    for ch in 0..4usize {
        for i in 0..n {
            input_data.push((rgba[i * 4 + ch] as f32 / 127.5) - 1.0);
        }
    }
    for _ in 0..n { input_data.push(1.0f32); }  // species channel

    let input_t: Tensor<InferBackend, 4> = Tensor::from_data(
        TensorData::new(input_data, [1usize, 5, sz, sz]), device,
    );

    // Forward → predicted fraction ∈ (0, 1)
    let pred_vec: Vec<f32> = model.forward(input_t)
        .into_data().to_vec().unwrap_or_default();
    let fraction = pred_vec.first().copied().unwrap_or(0.0).clamp(0.0, 1.0);

    // Convert fraction to pixel counts
    let predicted_total_px = (fraction * n as f32).round() as usize;
    let damage_px = predicted_total_px.saturating_sub(surviving_px);
    let pct_damage = if predicted_total_px > 0 {
        (damage_px as f32 / predicted_total_px as f32 * 100.0).clamp(0.0, 100.0)
    } else {
        0.0
    };

    let filename = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();

    Ok(AreaResult {
        filename,
        path: path.display().to_string(),
        surviving_px,
        predicted_total_px,
        damage_px,
        pct_damage,
    })
}

// ── CSV export ────────────────────────────────────────────────────────────────

pub fn write_csv(results: &[AreaResult], path: &Path) -> Result<(), String> {
    use std::io::Write;
    let mut f = std::fs::File::create(path).map_err(|e| e.to_string())?;
    writeln!(f, "Filename,Path,Surviving_px,Predicted_total_px,Damage_px,Pct_Damage")
        .map_err(|e| e.to_string())?;
    for r in results {
        writeln!(f, "{},{},{},{},{},{:.2}",
            r.filename,
            r.path,
            r.surviving_px,
            r.predicted_total_px,
            r.damage_px,
            r.pct_damage,
        ).map_err(|e| e.to_string())?;
    }
    Ok(())
}

// ── Utilities ─────────────────────────────────────────────────────────────────

fn is_image(p: &Path) -> bool {
    matches!(
        p.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref(),
        Some("png") | Some("tif") | Some("tiff") | Some("jpg") | Some("jpeg")
    )
}

fn log(tx: &mpsc::Sender<AreaInferMsg>, msg: impl Into<String>) {
    let _ = tx.send(AreaInferMsg::Log(msg.into()));
}
