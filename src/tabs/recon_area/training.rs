use std::{
    path::PathBuf,
    sync::{mpsc, Arc},
    sync::atomic::{AtomicBool, Ordering},
};

use burn::{
    tensor::{Tensor, TensorData, backend::Backend},
    module::AutodiffModule,
};
use burn_core::{
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::CompactRecorder,
    module::Module,
};
use rand::{rngs::SmallRng, Rng, SeedableRng, seq::SliceRandom};

// ── Re-export backend aliases ─────────────────────────────────────────────────
use crate::tabs::recon_train::model::{TrainBackend, InferBackend, create_train_device, backend_name};
pub use crate::tabs::recon_train::training::DamageParams;
use crate::tabs::recon_train::training::apply_random_damage;

use super::model::LeafAreaRegressor;

type TrainDevice = <TrainBackend as Backend>::Device;

// ── Public types ──────────────────────────────────────────────────────────────

/// Messages sent from the area-training thread → UI thread
pub enum AreaMsg {
    /// Loss for a single batch step
    BatchLoss { step: usize, loss: f32 },
    /// Validation metrics produced once per epoch
    EpochMetrics { epoch: usize, val_mae: f32, val_rmse: f32 },
    /// A checkpoint was saved
    Checkpoint { path: String },
    /// Informational log line
    Log(String),
    /// Training completed normally
    Finished,
    /// Fatal error — includes message
    Error(String),
}

/// Configuration for one area-regression training run
pub struct AreaTrainConfig {
    pub train_paths:      Vec<PathBuf>,
    pub val_paths:        Vec<PathBuf>,
    pub output_dir:       PathBuf,
    pub species_label:    String,
    pub epochs:           usize,
    pub batch_size:       usize,
    pub lr:               f64,
    pub checkpoint_every: usize,
    pub image_size:       usize,
    pub damage_params:    DamageParams,
    pub resume_from:      Option<PathBuf>,
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn spawn_area_training(
    cfg:    AreaTrainConfig,
    tx:     mpsc::Sender<AreaMsg>,
    cancel: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_area_training(&cfg, &tx, &cancel)
        }));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => { let _ = tx.send(AreaMsg::Error(e)); }
            Err(payload) => {
                let msg = payload.downcast_ref::<String>().cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "GPU panic (no message)".to_string());
                let _ = tx.send(AreaMsg::Error(format!("GPU panic: {msg}")));
            }
        }
        let _ = tx.send(AreaMsg::Finished);
    });
}

// ── Main training function ────────────────────────────────────────────────────

fn run_area_training(
    cfg:    &AreaTrainConfig,
    tx:     &mpsc::Sender<AreaMsg>,
    cancel: &AtomicBool,
) -> Result<(), String> {
    log(tx, format!("Requesting {} compute device…", backend_name()));
    let device: TrainDevice = create_train_device();
    {
        let probe = Tensor::<TrainBackend, 1>::zeros([1], &device);
        let _ = probe.into_data();
    }
    log(tx, format!("{} device ready.", backend_name()));

    // ── Load & cache images ───────────────────────────────────────────────────
    log(tx, format!("Loading {} training images…", cfg.train_paths.len()));
    let train_cache = load_images(&cfg.train_paths, cfg.image_size)?;
    log(tx, format!("Loading {} validation images…", cfg.val_paths.len()));
    let val_cache = load_images(&cfg.val_paths, cfg.image_size)?;

    // ── Initialise model ──────────────────────────────────────────────────────
    log(tx, "Initialising LeafAreaRegressor…");
    let mut model = LeafAreaRegressor::<TrainBackend>::init(&device);
    log(tx, "Model ready.");

    // ── Optimiser (Adam β1=0.9, β2=0.999 — standard regression) ─────────────
    let mut optim = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .init::<TrainBackend, LeafAreaRegressor<TrainBackend>>();

    // ── Optional checkpoint resume ────────────────────────────────────────────
    if let Some(resume) = &cfg.resume_from {
        log(tx, format!("Resuming from checkpoint: {}", resume.display()));
        let rec = CompactRecorder::new();
        model = model
            .load_file(resume.join("area_regressor"), &rec, &device)
            .map_err(|e| format!("Failed to load checkpoint: {e}"))?;
    }

    std::fs::create_dir_all(&cfg.output_dir).map_err(|e| e.to_string())?;

    let size = cfg.image_size;
    let n = size * size;
    let total_px = n as f32;
    let mut rng = SmallRng::from_entropy();
    let mut step = 0usize;

    // ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in 0..cfg.epochs {
        if cancel.load(Ordering::Relaxed) {
            log(tx, "Training cancelled.");
            break;
        }

        let mut indices: Vec<usize> = (0..train_cache.len()).collect();
        indices.shuffle(&mut rng);

        // ── Batch loop ────────────────────────────────────────────────────────
        for batch_indices in indices.chunks(cfg.batch_size) {
            if cancel.load(Ordering::Relaxed) { break; }

            let b = batch_indices.len();

            // Build input [b, 5, size, size] and GT [b, 1] tensors on CPU
            let mut input_data = Vec::with_capacity(b * 5 * n);
            let mut gt_data    = Vec::with_capacity(b);

            for &idx in batch_indices {
                let orig = &train_cache[idx];

                // Ground-truth: fraction of original leaf pixels (before damage)
                let original_leaf_px: usize = orig.chunks(4)
                    .filter(|p| p[3] > 128)
                    .count();
                let gt_frac = original_leaf_px as f32 / total_px;
                gt_data.push(gt_frac);

                // Damaged input (training signal: model sees partial leaf)
                let damaged = apply_random_damage(orig, size, size, &cfg.damage_params, &mut rng);

                // RGBA channels normalised to [-1, 1]
                for ch in 0..4usize {
                    for i in 0..n {
                        input_data.push((damaged[i * 4 + ch] as f32 / 127.5) - 1.0);
                    }
                }
                // Species channel = 1.0
                for _ in 0..n { input_data.push(1.0f32); }
            }

            // Upload to GPU
            let input_t: Tensor<TrainBackend, 4> = Tensor::from_data(
                TensorData::new(input_data, [b, 5, size, size]), &device,
            );
            let gt_t: Tensor<TrainBackend, 2> = Tensor::from_data(
                TensorData::new(gt_data, [b, 1]), &device,
            );

            // Forward + MSE loss
            let pred_t = model.forward(input_t);
            let loss_t = (pred_t - gt_t).powf_scalar(2.0f32).mean();

            let loss: f32 = loss_t.clone().into_scalar();

            if loss.is_nan() || loss.is_infinite() {
                log(tx, format!("WARNING: NaN/Inf loss at step {step}"));
            }

            // Backward + optimiser step
            let grads  = loss_t.backward();
            let params = GradientsParams::from_grads(grads, &model);
            model = optim.step(cfg.lr, model, params);

            let _ = tx.send(AreaMsg::BatchLoss { step, loss });
            step += 1;

            if step % 20 == 0 {
                log(tx, format!("  step {step} loss={loss:.6}"));
            }
        }

        if cancel.load(Ordering::Relaxed) { break; }

        // ── Validation ────────────────────────────────────────────────────────
        log(tx, format!("Validating (epoch {})…", epoch + 1));
        let (val_mae, val_rmse) = run_val(
            &model, &val_cache, size, &cfg.damage_params, &mut rng, &device,
        );
        let _ = tx.send(AreaMsg::EpochMetrics { epoch: epoch + 1, val_mae, val_rmse });
        log(tx, format!(
            "Epoch {}/{}: val_MAE={:.4}  val_RMSE={:.4}",
            epoch + 1, cfg.epochs, val_mae, val_rmse,
        ));

        // ── Flush GPU before saving ───────────────────────────────────────────
        {
            let _: f32 = model.gpu_fence().into_scalar();
        }

        // ── Checkpoint ────────────────────────────────────────────────────────
        if (epoch + 1) % cfg.checkpoint_every == 0 || epoch + 1 == cfg.epochs {
            let dir = cfg.output_dir.join(format!("checkpoint_epoch_{:04}", epoch + 1));
            log(tx, format!("Saving checkpoint: {}", dir.display()));
            if let Err(e) = save_checkpoint(&model, &dir) {
                log(tx, format!("Warning: checkpoint save failed — {e}"));
            } else {
                let _ = tx.send(AreaMsg::Checkpoint { path: dir.display().to_string() });
                log(tx, "  Checkpoint saved.");
            }
        }
    }

    // Final checkpoint
    let final_dir = cfg.output_dir.join("checkpoint_final");
    if let Err(e) = save_checkpoint(&model, &final_dir) {
        log(tx, format!("Warning: final checkpoint failed — {e}"));
    } else {
        let _ = tx.send(AreaMsg::Checkpoint { path: final_dir.display().to_string() });
        log(tx, "Final checkpoint saved.");
    }

    Ok(())
}

// ── Validation ────────────────────────────────────────────────────────────────

fn run_val(
    model:  &LeafAreaRegressor<TrainBackend>,
    cache:  &[Vec<u8>],
    size:   usize,
    params: &DamageParams,
    rng:    &mut SmallRng,
    device: &TrainDevice,
) -> (f32, f32) {
    let model_infer = model.clone().valid();
    let n = size * size;
    let total_px = n as f32;
    let picks = evenly_spaced(cache.len(), 50);

    let mut sum_ae  = 0.0f32;
    let mut sum_se  = 0.0f32;
    let mut count   = 0usize;

    for &idx in &picks {
        let orig = &cache[idx];
        let original_leaf_px: usize = orig.chunks(4).filter(|p| p[3] > 128).count();
        let gt_frac = original_leaf_px as f32 / total_px;

        let damaged = apply_random_damage(orig, size, size, params, rng);

        let mut input_data = Vec::with_capacity(5 * n);
        for ch in 0..4usize {
            for i in 0..n {
                input_data.push((damaged[i * 4 + ch] as f32 / 127.5) - 1.0);
            }
        }
        for _ in 0..n { input_data.push(1.0f32); }

        let input_t: Tensor<InferBackend, 4> = Tensor::from_data(
            TensorData::new(input_data, [1usize, 5, size, size]), device,
        );
        let pred_vec: Vec<f32> = model_infer.forward(input_t)
            .into_data().to_vec().unwrap_or_default();

        if let Some(&pred) = pred_vec.first() {
            let err = (pred - gt_frac).abs();
            sum_ae += err;
            sum_se += err * err;
            count  += 1;
        }
    }

    if count == 0 { return (0.0, 0.0); }
    let mae  = sum_ae / count as f32;
    let rmse = (sum_se / count as f32).sqrt();
    (mae, rmse)
}

// ── Image loading ─────────────────────────────────────────────────────────────

fn load_images(paths: &[PathBuf], target_size: usize) -> Result<Vec<Vec<u8>>, String> {
    use rayon::prelude::*;
    let sz = target_size as u32;
    paths.par_iter().map(|p| {
        let img = image::open(p)
            .map_err(|e| format!("{}: {e}", p.display()))?;
        let img = img.resize_exact(sz, sz, image::imageops::FilterType::Triangle);
        let mut raw = img.to_rgba8().into_raw();
        // Zero background pixels at load time — match training/inference convention
        for px in raw.chunks_mut(4) {
            if px[3] <= 128 {
                px[0] = 0; px[1] = 0; px[2] = 0; px[3] = 0;
            }
        }
        Ok(raw)
    }).collect()
}

// ── Checkpoint save ───────────────────────────────────────────────────────────

fn save_checkpoint(
    model: &LeafAreaRegressor<TrainBackend>,
    dir:   &std::path::Path,
) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;

    // Flush pending GPU commands before parameter readback.
    // Use existing model weights — do NOT create a new InferDevice/stream.
    let _: f32 = model.gpu_fence().into_scalar();

    let rec = CompactRecorder::new();
    model.clone().valid()
        .save_file(dir.join("area_regressor"), &rec)
        .map_err(|e| format!("{e}"))
}

// ── Utilities ─────────────────────────────────────────────────────────────────

fn log(tx: &mpsc::Sender<AreaMsg>, msg: impl Into<String>) {
    let _ = tx.send(AreaMsg::Log(msg.into()));
}

fn evenly_spaced(total: usize, count: usize) -> Vec<usize> {
    if total == 0 { return vec![]; }
    (0..count.min(total))
        .map(|i| i * total / count.min(total))
        .collect()
}
