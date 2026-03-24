use std::{
    path::{Path, PathBuf},
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
use rand::{Rng, rngs::SmallRng, SeedableRng, seq::SliceRandom};

use crate::tabs::recon_train::training::{
    DamageParams, load_images, apply_random_damage,
};
use crate::tabs::recon_train::model::{TrainBackend, InferBackend};
use super::model::{
    RadialNet, create_train_device, backend_name,
    extract_radial, fill_from_radial,
    save_radial_checkpoint, N_RAYS,
};

type TrainDevice = <TrainBackend as burn::tensor::backend::Backend>::Device;

// ── Public message type ───────────────────────────────────────────────────────

pub enum RadialTrainMsg {
    BatchMetrics { step: u64, loss: f32 },
    EpochMetrics { epoch: usize, mse: f32, boundary_mae: f32 },
    SampleGrid   { epoch: usize, pixels: Vec<u8>, width: usize, height: usize },
    Checkpoint   { path: String },
    Log(String),
    Finished,
    Error(String),
}

// ── Config ────────────────────────────────────────────────────────────────────

pub struct RadialTrainConfig {
    pub train_paths:       Vec<PathBuf>,
    pub val_paths:         Vec<PathBuf>,
    pub output_dir:        PathBuf,
    pub epochs:            usize,
    pub batch_size:        usize,
    pub lr:                f64,
    pub checkpoint_every:  usize,
    pub sample_every:      usize,
    pub image_size:        usize,
    pub damage_params:     DamageParams,
    pub curriculum_epochs: usize,
    pub resume_from:       Option<PathBuf>,
    pub accum_steps:       usize,
    /// Weight given to the damaged (missing) region in the loss.
    /// Higher = penalise under-completion more.  Default 5.0.
    pub damage_loss_weight: f32,
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn spawn_radial_training(
    config: RadialTrainConfig,
    tx:     mpsc::Sender<RadialTrainMsg>,
    cancel: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            train_loop(&config, &tx, &cancel)
        }));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => { let _ = tx.send(RadialTrainMsg::Error(e)); }
            Err(p) => {
                let msg = p.downcast_ref::<String>().cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "GPU panic".to_string());
                let _ = tx.send(RadialTrainMsg::Error(msg));
            }
        }
        let _ = tx.send(RadialTrainMsg::Finished);
    });
}

fn log(tx: &mpsc::Sender<RadialTrainMsg>, msg: String) {
    let _ = tx.send(RadialTrainMsg::Log(msg));
}

// ── Training loop ─────────────────────────────────────────────────────────────

fn train_loop(
    cfg:    &RadialTrainConfig,
    tx:     &mpsc::Sender<RadialTrainMsg>,
    cancel: &AtomicBool,
) -> Result<(), String> {
    log(tx, format!("Backend: {}", backend_name()));
    log(tx, format!("Radial profile: {} rays, 1-D U-Net", N_RAYS));

    let device: TrainDevice = create_train_device();
    {
        let _ = Tensor::<TrainBackend, 1>::zeros([1], &device).into_data();
    }
    log(tx, format!("{} device ready.", backend_name()));

    // ── Load images ──────────────────────────────────────────────────────────
    log(tx, format!("Loading {} training images…", cfg.train_paths.len()));
    let train_cache = load_images(&cfg.train_paths, cfg.image_size)?;
    log(tx, format!("Loading {} validation images…", cfg.val_paths.len()));
    let val_cache = load_images(&cfg.val_paths, cfg.image_size)?;

    log(tx, "Precomputing GT radial profiles for training set…".to_string());
    let train_gt: Vec<(Vec<f32>, f32, f32)> = train_cache.iter()
        .map(|rgba| gt_profile_from_rgba(rgba, cfg.image_size))
        .collect();

    log(tx, "Precomputing GT radial profiles for validation set…".to_string());
    let val_gt: Vec<(Vec<f32>, f32, f32)> = val_cache.iter()
        .map(|rgba| gt_profile_from_rgba(rgba, cfg.image_size))
        .collect();

    // Fixed validation damages (seeded, reproducible)
    let val_picks: Vec<usize> = evenly_spaced(val_cache.len(), 20);
    let mut val_rng = SmallRng::seed_from_u64(0xdeadbeef);
    let val_dmg_profiles: Vec<Vec<f32>> = val_picks.iter()
        .map(|&idx| {
            let dmg = apply_random_damage(
                &val_cache[idx], cfg.image_size, cfg.image_size,
                &cfg.damage_params, &mut val_rng,
            );
            let alpha: Vec<bool> = dmg.chunks(4).map(|p| p[3] > 128).collect();
            let (prof, _, _, _) = extract_radial_with_valid(&alpha, cfg.image_size, cfg.image_size);
            prof
        })
        .collect();

    // ── Model ─────────────────────────────────────────────────────────────────
    log(tx, "Initialising RadialNet…".to_string());
    let mut model = RadialNet::<TrainBackend>::init(&device);
    log(tx, "RadialNet ready.".to_string());

    if let Some(ref resume) = cfg.resume_from {
        let rec = CompactRecorder::new();
        model = model.load_file(resume.join("radial"), &rec, &device)
            .map_err(|e| format!("load: {e}"))?;
        log(tx, "Checkpoint loaded.".to_string());
    }

    let mut optim = AdamConfig::new()
        .with_beta_1(0.5)
        .with_beta_2(0.999)
        .init::<TrainBackend, RadialNet<TrainBackend>>();

    std::fs::create_dir_all(&cfg.output_dir).map_err(|e| e.to_string())?;

    let mut best_mse = f32::MAX;
    let mut step     = 0u64;
    let mut rng      = SmallRng::from_entropy();
    let accum        = cfg.accum_steps.max(1);

    // ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in 0..cfg.epochs {
        if cancel.load(Ordering::Relaxed) { log(tx, "Cancelled.".to_string()); break; }

        log(tx, format!("Epoch {}/{}", epoch + 1, cfg.epochs));

        let eff_max = if cfg.curriculum_epochs > 0 && epoch < cfg.curriculum_epochs {
            let t = epoch as f32 / cfg.curriculum_epochs as f32;
            cfg.damage_params.min_pct
                + t * (cfg.damage_params.curriculum_max - cfg.damage_params.min_pct)
        } else {
            cfg.damage_params.max_pct
        };
        let epoch_params = DamageParams { max_pct: eff_max, ..cfg.damage_params.clone() };

        let mut indices: Vec<usize> = (0..train_cache.len()).collect();
        indices.shuffle(&mut rng);

        let batches: Vec<Vec<usize>> = indices.chunks(cfg.batch_size)
            .map(|c| c.to_vec()).collect();

        let mut accum_loss: Option<Tensor<TrainBackend, 1>> = None;
        let mut accum_count = 0u64;
        let mut batch_num   = 0usize;

        for batch_idx in &batches {
            if cancel.load(Ordering::Relaxed) { break; }

            // Build batch on CPU
            let b = batch_idx.len();
            let mut input_flat = Vec::with_capacity(b * 3 * N_RAYS);
            let mut gt_flat    = Vec::with_capacity(b * N_RAYS);
            let mut damage_mask_flat = Vec::with_capacity(b * N_RAYS); // 1 where damaged

            for &idx in batch_idx {
                let dmg_rgba = apply_random_damage(
                    &train_cache[idx], cfg.image_size, cfg.image_size,
                    &epoch_params, &mut rng,
                );
                let dmg_alpha: Vec<bool> = dmg_rgba.chunks(4).map(|p| p[3] > 128).collect();
                let (dmg_prof, valid, _, _) = extract_radial_with_valid(
                    &dmg_alpha, cfg.image_size, cfg.image_size,
                );
                let (gt_prof, _, _) = &train_gt[idx];

                // Circular shift augmentation — forces the model to use local context
                // rather than memorising angular position of lobes.
                let shift = rng.gen_range(0..N_RAYS);
                let mut dmg_s   = dmg_prof.clone();
                let mut valid_s = valid.clone();
                let mut gt_s    = gt_prof.clone();
                if shift > 0 {
                    dmg_s  .rotate_left(shift);
                    valid_s.rotate_left(shift);
                    gt_s   .rotate_left(shift);
                }

                // 3-channel input: damaged | valid | interp (after shift)
                let inp = build_3ch_input(&dmg_s, &valid_s);
                input_flat.extend_from_slice(&inp);

                // GT (shifted)
                for &v in &gt_s { gt_flat.push(v); }

                // Damage mask: 1 where gt > dmg (missing area)
                for i in 0..N_RAYS {
                    let is_damaged = gt_s[i] - dmg_s[i] > 0.01;
                    damage_mask_flat.push(if is_damaged { 1.0f32 } else { 0.0 });
                }
            }

            // Upload
            let input_t: Tensor<TrainBackend, 3> = Tensor::from_data(
                TensorData::new(input_flat, [b, 3, N_RAYS]), &device,
            );
            let gt_t: Tensor<TrainBackend, 3> = Tensor::from_data(
                TensorData::new(gt_flat, [b, 1, N_RAYS]), &device,
            );
            let dmg_mask_t: Tensor<TrainBackend, 3> = Tensor::from_data(
                TensorData::new(damage_mask_flat, [b, 1, N_RAYS]), &device,
            );

            // Forward
            let pred = model.forward(input_t);

            // Weighted MSE: damaged region gets higher weight
            let weight = dmg_mask_t.mul_scalar(cfg.damage_loss_weight - 1.0).add_scalar(1.0);
            let diff_sq = (pred - gt_t).powf_scalar(2.0);
            let loss = (diff_sq * weight).mean();
            let loss_v: f32 = loss.clone().into_scalar();

            let scaled = loss.mul_scalar(1.0 / accum as f32);
            accum_loss = Some(match accum_loss.take() {
                None    => scaled,
                Some(p) => p + scaled,
            });
            accum_count += 1;

            if accum_count % accum as u64 == 0 {
                let grads = accum_loss.take().unwrap().backward();
                let params = GradientsParams::from_grads(grads, &model);
                model = optim.step(cfg.lr, model, params);
            }

            let _ = tx.send(RadialTrainMsg::BatchMetrics { step, loss: loss_v });
            step += 1;
            batch_num += 1;
            if batch_num % 20 == 0 {
                log(tx, format!("  batch {batch_num}/{} (step {step})", batches.len()));
            }
        }

        // Flush leftover accumulation
        if let Some(leftover) = accum_loss.take() {
            let grads = leftover.backward();
            let params = GradientsParams::from_grads(grads, &model);
            model = optim.step(cfg.lr, model, params);
        }

        if cancel.load(Ordering::Relaxed) { break; }

        // ── Validation ────────────────────────────────────────────────────────
        log(tx, format!("Validating (epoch {})…", epoch + 1));
        let (mse, boundary_mae) = run_validation(
            &model, &val_cache, &val_picks, &val_dmg_profiles,
            &val_gt, cfg.image_size, &device,
        );
        let _ = tx.send(RadialTrainMsg::EpochMetrics { epoch: epoch + 1, mse, boundary_mae });
        log(tx, format!("  Val MSE={:.6}  Boundary MAE={:.4}px", mse, boundary_mae));

        // ── Checkpoint ────────────────────────────────────────────────────────
        if (epoch + 1) % cfg.checkpoint_every == 0 || epoch == cfg.epochs - 1 {
            let dir = cfg.output_dir.join(format!("checkpoint_epoch_{:04}", epoch + 1));
            match save_radial_checkpoint(&model, &dir) {
                Ok(()) => {
                    let _ = tx.send(RadialTrainMsg::Checkpoint {
                        path: dir.to_string_lossy().into_owned(),
                    });
                }
                Err(e) => log(tx, format!("Checkpoint save failed: {e}")),
            }
            if mse < best_mse {
                best_mse = mse;
                let best = cfg.output_dir.join("checkpoint_best");
                let _ = save_radial_checkpoint(&model, &best);
                log(tx, format!("  New best MSE: {:.6}", mse));
            }
        }

        // ── Sample grid ───────────────────────────────────────────────────────
        if (epoch + 1) % cfg.sample_every == 0 {
            log(tx, "Building sample grid…".to_string());
            let (pixels, w, h) = build_sample_grid(
                &model, &val_cache, &val_picks, &val_dmg_profiles,
                &val_gt, cfg.image_size, &device,
            );
            let _ = tx.send(RadialTrainMsg::SampleGrid { epoch: epoch + 1, pixels, width: w, height: h });
        }
    }

    log(tx, "Radial training complete.".to_string());
    Ok(())
}

// ── Validation ────────────────────────────────────────────────────────────────

fn run_validation(
    model:        &RadialNet<TrainBackend>,
    val_cache:    &[Vec<u8>],
    val_picks:    &[usize],
    dmg_profiles: &[Vec<f32>],
    gt_profiles:  &[(Vec<f32>, f32, f32)],
    _size:        usize,
    device:       &TrainDevice,
) -> (f32, f32) {
    let m = model.clone().valid();
    let mut total_mse = 0.0f32;
    let mut total_boundary_mae = 0.0f32;
    let n = val_picks.len();

    for (i, &idx) in val_picks.iter().enumerate() {
        let dmg = &dmg_profiles[i];
        let (gt, _, _) = &gt_profiles[idx];

        // valid flag: 0 where dmg profile is 0
        let valid: Vec<f32> = dmg.iter().map(|&r| if r > 0.0 { 1.0 } else { 0.0 }).collect();
        let inp = build_3ch_input(dmg, &valid);

        let input_t: Tensor<InferBackend, 3> = Tensor::from_data(
            TensorData::new(inp, [1, 3, N_RAYS]), device,
        );
        let pred_vec: Vec<f32> = m.forward(input_t)
            .reshape([N_RAYS])
            .into_data().to_vec().unwrap_or_default();

        let mse: f32 = pred_vec.iter().zip(gt.iter())
            .map(|(p, g)| (p - g).powi(2))
            .sum::<f32>() / N_RAYS as f32;
        total_mse += mse;

        // Boundary MAE only in damaged region (where gt > dmg + small threshold)
        let mut dmg_count = 0usize;
        let mut dmg_mae = 0.0f32;
        for j in 0..N_RAYS {
            if gt[j] - dmg[j] > 0.01 {
                // Convert normalised delta to pixels (rough: max_r ≈ size/2)
                dmg_mae += (pred_vec[j] - gt[j]).abs();
                dmg_count += 1;
            }
        }
        if dmg_count > 0 {
            total_boundary_mae += dmg_mae / dmg_count as f32;
        }
    }

    (total_mse / n as f32, total_boundary_mae / n as f32)
}

// ── Sample grid ───────────────────────────────────────────────────────────────
//
// For each of 4 validation samples, draw:
//   Col 0: Damaged alpha mask (grey)
//   Col 1: GT intact mask
//   Col 2: Predicted completion mask
//   Col 3: Radial profile overlay (damaged=red, gt=green, pred=blue on white bg)

fn build_sample_grid(
    model:        &RadialNet<TrainBackend>,
    val_cache:    &[Vec<u8>],
    val_picks:    &[usize],
    dmg_profiles: &[Vec<f32>],
    gt_profiles:  &[(Vec<f32>, f32, f32)],
    size:         usize,
    device:       &TrainDevice,
) -> (Vec<u8>, usize, usize) {
    let m       = model.clone().valid();
    let thumb   = 128usize;
    let picks   = evenly_spaced(val_picks.len(), 4);
    let n_cols  = 4usize;
    let n_rows  = picks.len();
    let w       = thumb * n_cols;
    let h       = thumb * n_rows;
    let mut out = vec![30u8; w * h * 4];  // dark background

    let max_r = (size.min(size) as f32) / 2.0;

    for (row, &pi) in picks.iter().enumerate() {
        let idx = val_picks[pi];
        let dmg = &dmg_profiles[pi];
        let (gt, cx, cy) = &gt_profiles[idx];

        // valid flag
        let valid: Vec<f32> = dmg.iter().map(|&r| if r > 0.0 { 1.0 } else { 0.0 }).collect();
        let inp = build_3ch_input(dmg, &valid);

        let input_t: Tensor<InferBackend, 3> = Tensor::from_data(
            TensorData::new(inp, [1, 3, N_RAYS]), device,
        );
        let pred: Vec<f32> = m.forward(input_t)
            .reshape([N_RAYS])
            .into_data().to_vec().unwrap_or_default();

        // Build full-res masks then thumbnail
        let orig_rgba = &val_cache[idx];
        let dmg_alpha: Vec<bool> = {
            // Re-derive from dmg profile (not perfectly reversible; use original alpha for display)
            let dmg_mask = fill_from_radial(dmg, size, size, *cx, *cy);
            dmg_mask
        };
        let gt_mask   = fill_from_radial(gt, size, size, *cx, *cy);
        let pred_mask = fill_from_radial(&pred, size, size, *cx, *cy);

        // Thumbnails: damaged (grey), GT (green tint), pred (blue tint)
        let col0 = mask_to_thumb(&dmg_alpha, size, [80, 100, 80], thumb);
        let col1 = mask_to_thumb(&gt_mask,   size, [60, 160, 60], thumb);
        let col2 = mask_to_thumb(&pred_mask, size, [60, 100, 200], thumb);
        let col3 = draw_radial_overlay(dmg, gt, &pred, thumb);

        for col_idx in 0..4 {
            let src = [&col0, &col1, &col2, &col3][col_idx];
            for ty in 0..thumb {
                for tx_i in 0..thumb {
                    let ox = col_idx * thumb + tx_i;
                    let oy = row * thumb + ty;
                    let di = (oy * w + ox) * 4;
                    let si = (ty * thumb + tx_i) * 4;
                    out[di..di+4].copy_from_slice(&src[si..si+4]);
                }
            }
        }
    }

    (out, w, h)
}

/// Draw radial profiles as a polar chart on a white thumbnail.
fn draw_radial_overlay(
    dmg:  &[f32],
    gt:   &[f32],
    pred: &[f32],
    sz:   usize,
) -> Vec<u8> {
    let mut img = vec![255u8; sz * sz * 4];
    let cx = sz as f32 / 2.0;
    let cy = sz as f32 / 2.0;
    let max_r = sz as f32 / 2.0 * 0.95;
    let tau = std::f32::consts::TAU;
    let n = N_RAYS;

    let draw = |img: &mut Vec<u8>, x: f32, y: f32, r: u8, g: u8, b: u8| {
        let px = x.round() as isize;
        let py = y.round() as isize;
        if px >= 0 && px < sz as isize && py >= 0 && py < sz as isize {
            let i = (py as usize * sz + px as usize) * 4;
            img[i]   = r; img[i+1] = g; img[i+2] = b; img[i+3] = 255;
        }
    };

    for i in 0..n {
        let angle = tau * (i as f32) / n as f32;
        let (cos, sin) = (angle.cos(), angle.sin());
        // Draw each profile as a dot at its radius
        for (profile, (r, g, b)) in [
            (dmg,  (220u8, 80u8,  80u8)),
            (gt,   (80u8,  200u8, 80u8)),
            (pred, (80u8,  120u8, 220u8)),
        ] {
            let radius = profile[i] * max_r;
            let x = cx + cos * radius;
            let y = cy + sin * radius;
            draw(&mut img, x, y, r, g, b);
            // Make 2px wide
            draw(&mut img, x + 0.5, y, r, g, b);
            draw(&mut img, x, y + 0.5, r, g, b);
        }
    }

    img
}

fn mask_to_thumb(mask: &[bool], src_size: usize, color: [u8; 3], thumb: usize) -> Vec<u8> {
    let mut out = vec![30u8; thumb * thumb * 4];
    let scale = src_size as f32 / thumb as f32;
    for ty in 0..thumb {
        for tx in 0..thumb {
            let sx = (tx as f32 * scale) as usize;
            let sy = (ty as f32 * scale) as usize;
            let idx = sy * src_size + sx;
            let o   = (ty * thumb + tx) * 4;
            if idx < mask.len() && mask[idx] {
                out[o]   = color[0];
                out[o+1] = color[1];
                out[o+2] = color[2];
                out[o+3] = 255;
            } else {
                out[o]   = 30; out[o+1] = 30; out[o+2] = 30; out[o+3] = 255;
            }
        }
    }
    out
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Extract radial profile + valid flag in one call.
pub fn extract_radial_with_valid(
    alpha: &[bool],
    w:     usize,
    h:     usize,
) -> (Vec<f32>, Vec<f32>, f32, f32) {
    let (profile, valid, cx, cy) = super::model::extract_radial(alpha, w, h);
    let valid_f: Vec<f32> = valid.iter().map(|&v| if v { 1.0f32 } else { 0.0 }).collect();
    (profile, valid_f, cx, cy)
}

/// Extract GT radial profile from raw RGBA bytes.
fn gt_profile_from_rgba(rgba: &[u8], size: usize) -> (Vec<f32>, f32, f32) {
    let alpha: Vec<bool> = rgba.chunks(4).map(|p| p[3] > 128).collect();
    let (profile, _, cx, cy) = super::model::extract_radial(&alpha, size, size);
    (profile, cx, cy)
}

/// Fill invalid profile positions by circular linear interpolation from valid neighbours.
///
/// Where `valid[i] > 0.5` the input value is copied unchanged.
/// Gaps are filled by finding the nearest valid sample on each side of the gap
/// (circularly) and linearly interpolating between them.
fn interp_fill(profile: &[f32], valid: &[f32]) -> Vec<f32> {
    let n = profile.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        if valid[i] > 0.5 {
            out[i] = profile[i];
            continue;
        }
        // Nearest valid to the left
        let (left_d, left_v) = (1..=n)
            .find_map(|d| {
                let j = (i + n - d) % n;
                if valid[j] > 0.5 { Some((d, profile[j])) } else { None }
            })
            .unwrap_or((n, 0.0));
        // Nearest valid to the right
        let (right_d, right_v) = (1..=n)
            .find_map(|d| {
                let j = (i + d) % n;
                if valid[j] > 0.5 { Some((d, profile[j])) } else { None }
            })
            .unwrap_or((n, 0.0));

        out[i] = if left_d < n && right_d < n {
            let t = left_d as f32 / (left_d + right_d) as f32;
            left_v * (1.0 - t) + right_v * t
        } else if left_d < n {
            left_v
        } else if right_d < n {
            right_v
        } else {
            0.0
        };
    }
    out
}

/// Build the 3-channel input flat array for one sample: [dmg | valid | interp].
fn build_3ch_input(dmg: &[f32], valid: &[f32]) -> Vec<f32> {
    let interp = interp_fill(dmg, valid);
    let mut v = Vec::with_capacity(3 * N_RAYS);
    v.extend_from_slice(dmg);
    v.extend_from_slice(valid);
    v.extend_from_slice(&interp);
    v
}

fn evenly_spaced(n: usize, count: usize) -> Vec<usize> {
    if n == 0 { return vec![]; }
    let count = count.min(n);
    (0..count).map(|i| i * n / count).collect()
}

// ── Inference helper ─────────────────────────────────────────────────────────

/// Run radial inference on a single damaged RGBA image.
/// Returns the reconstructed binary mask [w*h].
pub fn infer_one(
    model:  &RadialNet<InferBackend>,
    rgba:   &[u8],
    w:      usize,
    h:      usize,
    device: &<InferBackend as burn::tensor::backend::Backend>::Device,
) -> Vec<bool> {
    let alpha: Vec<bool> = rgba.chunks(4).map(|p| p[3] > 128).collect();
    let (dmg_prof, valid_f, cx, cy) = extract_radial_with_valid(&alpha, w, h);

    let inp = build_3ch_input(&dmg_prof, &valid_f);

    let input_t: Tensor<InferBackend, 3> = Tensor::from_data(
        TensorData::new(inp, [1, 3, N_RAYS]), device,
    );
    let pred: Vec<f32> = model.forward(input_t)
        .reshape([N_RAYS])
        .into_data().to_vec().unwrap_or_default();

    fill_from_radial(&pred, w, h, cx, cy)
}
