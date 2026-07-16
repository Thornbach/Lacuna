use std::{
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
    sync::atomic::{AtomicBool, Ordering},
};

use burn::{
    tensor::{Tensor, TensorData, activation, backend::Backend},
    module::AutodiffModule,
};
use burn::tensor::module::max_pool2d;
use burn_core::{
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::CompactRecorder,
    module::Module,
};
use rand::{rngs::SmallRng, SeedableRng, seq::SliceRandom};

use crate::tabs::recon_train::{
    metrics::{compute_damage_metrics, average_metrics, MetricsSnapshot},
    visualization::build_sample_grid,
    training::{
        DamageParams,
        load_images, apply_random_damage, fill_holes, augment_pair,
        recon_loss_from_logits, tv_loss, confidence_loss,
    },
};
use crate::tabs::recon_train::model::{TrainBackend, InferBackend};
use super::model::{
    UNetSimple, create_train_device, backend_name, save_simple_checkpoint,
};

type TrainDevice = <TrainBackend as burn::tensor::backend::Backend>::Device;

// ── Messages ──────────────────────────────────────────────────────────────────

pub enum SimpleTrainMsg {
    BatchMetrics { step: u64, g_recon: f32, g_adv: f32, d_real: f32, d_fake: f32 },
    EpochMetrics  { epoch: usize, metrics: MetricsSnapshot },
    SampleGrid    { epoch: usize, pixels: Vec<u8>, width: usize, height: usize },
    Checkpoint    { path: String },
    Log(String),
    Finished,
    Error(String),
}

// ── Config ────────────────────────────────────────────────────────────────────

pub struct SimpleTrainConfig {
    pub train_paths:        Vec<PathBuf>,
    pub val_paths:          Vec<PathBuf>,
    pub output_dir:         PathBuf,
    pub epochs:             usize,
    pub batch_size:         usize,
    pub lr:                 f64,
    pub l1_lambda:          f32,
    pub tv_lambda:          f32,
    pub conf_lambda:        f32,
    pub recon_focus_weight: f32,
    pub tversky_alpha:      f32,
    pub tversky_beta:       f32,
    pub checkpoint_every:   usize,
    pub sample_every:       usize,
    pub image_size:         usize,
    pub damage_params:      DamageParams,
    pub curriculum_epochs:  usize,
    pub resume_from:        Option<PathBuf>,
    /// 0-indexed epoch to start the loop at (for resuming mid-run). 0 = fresh start.
    pub start_epoch:        usize,
    /// Best val IoU already achieved before this run (for resuming), so a worse
    /// early epoch post-resume can't clobber an existing checkpoint_best.
    pub resume_best_iou:    f32,
    /// Cosine LR schedule floor, as a fraction of `lr`, reached at the final
    /// epoch. 1.0 = flat LR (no decay).
    pub lr_min_frac:        f64,
    pub accum_steps:        usize,
    /// Leaf shape class index — passed to UNetSimple FiLM conditioning.
    pub leaf_shape:  u32,
    /// Margin type class index — passed to UNetSimple FiLM conditioning.
    pub margin_type: u32,
    /// Reconstruction-only pre-training epochs before enabling the discriminator.
    pub pretrain_epochs: usize,
    /// D learning rate = G LR × d_lr_factor.
    pub d_lr_factor: f64,
    /// Adversarial loss weight in the generator step.
    pub adv_lambda: f32,
    /// Area head MSE loss weight.
    pub area_lambda: f32,
    /// Boundary-weighted BCE loss weight (focuses gradient on the margin contour).
    pub boundary_lambda: f32,
    /// Half-width (px) of the margin band for the boundary loss.
    pub boundary_px: usize,
    /// Weight on the hole-consistency loss: penalises pixels the model predicts
    /// as not-leaf but which its OWN prediction topologically encloses (i.e. an
    /// invented interior hole). 0 = disabled.
    pub hole_lambda: f32,
}

// ── Internal batch type for 4-channel data ────────────────────────────────────

struct SimpleBatch {
    /// Flat [B, 4, H, W]: channels R, G, B, A — all normalised to [−1, 1]
    input_data: Vec<f32>,
    /// Flat [B, 1, H, W]: GT leaf mask {0.0, 1.0}
    gt_data:    Vec<f32>,
    batch_len:  usize,
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn spawn_simple_training(
    config: SimpleTrainConfig,
    tx:     mpsc::Sender<SimpleTrainMsg>,
    cancel: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            train_loop(&config, &tx, &cancel)
        }));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => { let _ = tx.send(SimpleTrainMsg::Error(e)); }
            Err(payload) => {
                let msg = payload.downcast_ref::<String>().cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "GPU panic (no message)".to_string());
                let _ = tx.send(SimpleTrainMsg::Error(format!("GPU panic: {msg}")));
            }
        }
        let _ = tx.send(SimpleTrainMsg::Finished);
    });
}

fn log(tx: &mpsc::Sender<SimpleTrainMsg>, msg: String) {
    let _ = tx.send(SimpleTrainMsg::Log(msg));
}

fn train_loop(
    cfg:    &SimpleTrainConfig,
    tx:     &mpsc::Sender<SimpleTrainMsg>,
    cancel: &AtomicBool,
) -> Result<(), String> {
    log(tx, format!("Backend: {}", backend_name()));
    log(tx, format!("Input: 4-channel RGBA | shape={} margin={}", cfg.leaf_shape, cfg.margin_type));

    let device: TrainDevice = create_train_device();
    {
        let probe = Tensor::<TrainBackend, 1>::zeros([1], &device);
        let _ = probe.into_data();
    }
    log(tx, format!("{} device ready.", backend_name()));

    // ── Load images ──────────────────────────────────────────────────────────
    log(tx, format!("Loading {} training images…", cfg.train_paths.len()));
    let train_cache = Arc::new(load_images(&cfg.train_paths, cfg.image_size)?);
    log(tx, format!("Loading {} validation images…", cfg.val_paths.len()));
    let val_cache = load_images(&cfg.val_paths, cfg.image_size)?;

    let val_picks: Vec<usize> = evenly_spaced(val_cache.len(), 20);
    log(tx, format!("Pre-generating {} fixed validation damages…", val_picks.len()));
    let mut val_rng = SmallRng::seed_from_u64(0xdeadbeef_u64);
    let val_damaged: Vec<Vec<u8>> = val_picks.iter()
        .map(|&idx| apply_random_damage(
            &val_cache[idx], cfg.image_size, cfg.image_size,
            &cfg.damage_params, &mut val_rng,
        ))
        .collect();

    // ── Model ─────────────────────────────────────────────────────────────────
    log(tx, "Initialising UNetSimple (4-level, 4-channel RGBA + FiLM)…".into());
    let mut generator = UNetSimple::<TrainBackend>::init(&device);
    log(tx, "Generator ready.".into());

    if let Some(resume) = &cfg.resume_from {
        log(tx, format!("Resuming from {:?}", resume));
        let rec = CompactRecorder::new();
        generator = match generator.load_file(resume.join("gen"), &rec, &device) {
            Ok(g)  => { log(tx, "Checkpoint loaded.".into()); g }
            Err(e) => {
                log(tx, format!(
                    "WARNING: checkpoint incompatible ({e}) — starting from scratch."
                ));
                UNetSimple::<TrainBackend>::init(&device)
            }
        };
    }

    // beta_1=0.9 (standard high-momentum Adam), NOT the GAN-stabilization 0.5 —
    // this trainer has no discriminator/adversarial loss at all (pure supervised
    // Tversky+BCE/TV/confidence/no-delete/area/boundary), so there's no
    // oscillating minimax dynamic to damp. 0.5 here was inherited from the GAN
    // trainer's config and just throws away momentum that would otherwise help
    // push through plateaus — a likely cause of loss stalling mid-run.
    let mut g_optim = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .init::<TrainBackend, UNetSimple<TrainBackend>>();

    std::fs::create_dir_all(&cfg.output_dir).map_err(|e| e.to_string())?;

    // Resuming: seed best_iou from the prior run so an early post-resume epoch
    // (Adam's momentum/variance restart from zero on resume — only the weights
    // are checkpointed, not optimizer state) can't clobber a better existing
    // checkpoint_best just because the fresh default of 0.0 looks like an
    // improvement over anything.
    let mut best_iou  = cfg.resume_best_iou;
    let mut step      = 0u64;
    let mut rng       = SmallRng::from_entropy();
    let accum         = cfg.accum_steps.max(1);

    if cfg.start_epoch >= cfg.epochs {
        log(tx, format!(
            "start_epoch ({}) >= epochs ({}) — nothing to do.",
            cfg.start_epoch, cfg.epochs,
        ));
        return Ok(());
    }
    if cfg.start_epoch > 0 {
        log(tx, format!(
            "Resuming at epoch {} (best IoU so far: {:.4})",
            cfg.start_epoch + 1, best_iou,
        ));
    }

    if cfg.curriculum_epochs > 0 {
        log(tx, format!(
            "Curriculum: damage max ramps {:.0}%→{:.0}% over {} epochs",
            cfg.damage_params.min_pct, cfg.damage_params.curriculum_max,
            cfg.curriculum_epochs,
        ));
    }

    // ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in cfg.start_epoch..cfg.epochs {
        if cancel.load(Ordering::Relaxed) {
            log(tx, "Training cancelled.".into());
            break;
        }

        let cur_lr = cosine_lr(cfg.lr, epoch, cfg.epochs, cfg.lr_min_frac);

        log(tx, format!("Epoch {}/{}  LR={:.2e}", epoch + 1, cfg.epochs, cur_lr));

        let eff_max = if cfg.curriculum_epochs > 0 && epoch < cfg.curriculum_epochs {
            let t = epoch as f32 / cfg.curriculum_epochs as f32;
            (cfg.damage_params.min_pct
                + t * (cfg.damage_params.curriculum_max - cfg.damage_params.min_pct))
                .clamp(cfg.damage_params.min_pct, cfg.damage_params.curriculum_max)
        } else {
            cfg.damage_params.max_pct
        };
        let epoch_params = DamageParams { max_pct: eff_max, ..cfg.damage_params.clone() };

        let mut indices: Vec<usize> = (0..train_cache.len()).collect();
        indices.shuffle(&mut rng);

        let batches: Vec<Vec<usize>> = indices
            .chunks(cfg.batch_size)
            .map(|c| c.to_vec())
            .collect();

        // CPU pipeline for 2-channel data
        let (work_tx, work_rx) = mpsc::channel::<Vec<usize>>();
        let (prebuilt_tx, prebuilt_rx) = mpsc::sync_channel::<SimpleBatch>(1);

        spawn_simple_cpu_pipeline(
            Arc::clone(&train_cache),
            epoch_params.clone(),
            cfg.image_size,
            work_rx,
            prebuilt_tx,
        );

        let mut next_batch_idx = 0usize;
        for _ in 0..2 {
            if next_batch_idx < batches.len() {
                let _ = work_tx.send(batches[next_batch_idx].clone());
                next_batch_idx += 1;
            }
        }

        let mut batch_num   = 0usize;
        let mut accum_loss: Option<Tensor<TrainBackend, 1>> = None;
        let mut accum_count = 0u64;

        for _ in 0..batches.len() {
            if cancel.load(Ordering::Relaxed) { break; }

            let prebuilt = match prebuilt_rx.recv() {
                Ok(b) => b,
                Err(_) => break,
            };

            if next_batch_idx < batches.len() {
                let _ = work_tx.send(batches[next_batch_idx].clone());
                next_batch_idx += 1;
            }

            let (input_t, gt_t) = upload_simple_batch(prebuilt, cfg.image_size, &device);

            // ── Generator step (pure reconstruction + area head) ─────────────
            let (fake_logits, area_pred) = generator.forward_with_area(
                input_t.clone(), cfg.leaf_shape, cfg.margin_type,
            );
            let input_alpha = input_t.clone().narrow(1, 3, 1);

            let g_recon_t = recon_loss_from_logits(
                fake_logits.clone(), gt_t.clone(),
                input_alpha.clone(),
                cfg.tversky_alpha, cfg.tversky_beta,
            );
            let g_recon: f32 = g_recon_t.clone().into_scalar();

            let fake_probs = activation::sigmoid(fake_logits.clone());
            let g_tv_t   = tv_loss(fake_probs.clone());
            let g_conf_t = confidence_loss(fake_probs.clone());

            // No-delete: the visible leaf IS ground truth, so penalise predicting < 1
            // there. The net must PRESERVE real tissue and only ADD the missing margin —
            // never remove correct pixels (that wastes capacity and hurts reconstruction).
            // Loss = mean over visible pixels of (1 − pred).
            let visible_01  = input_alpha.add_scalar(1.0f32).div_scalar(2.0f32); // [B,1,H,W] in [0,1]
            let vis_sum     = visible_01.clone().sum().clamp_min(1.0f32);
            let no_delete_t = visible_01
                .mul(fake_probs.clone().mul_scalar(-1.0f32).add_scalar(1.0f32))
                .sum() / vis_sum;

            // Hole-consistency: penalise pixels the model itself topologically
            // encloses (per its own predicted silhouette) but predicts as NOT
            // leaf — an invented interior hole. Plain per-pixel losses barely
            // react to these (a tiny isolated dot moves aggregate BCE/Tversky/
            // IoU almost nothing), so nothing otherwise discourages them.
            let g_hole_t = if cfg.hole_lambda > 0.0 {
                hole_loss(fake_probs.clone(), cfg.image_size, &device)
            } else {
                Tensor::<TrainBackend, 1>::zeros([1], &device)
            };

            // Area loss: MSE between predicted area fraction and GT area fraction
            // gt_t is [B,1,H,W]; flatten spatial dims → [B,1], matching area_pred shape
            let b = gt_t.dims()[0];
            let gt_area_frac = gt_t.clone().reshape([b as i32, 1, -1]).mean_dim(2); // [B,1,1] keep; flatten to [B,1]
            let gt_area_frac = gt_area_frac.reshape([b as i32, 1]);                 // [B,1]
            let area_loss_t  = (area_pred - gt_area_frac).powf_scalar(2.0f32).mean();

            // Boundary-weighted BCE: focus gradient on the margin contour (within
            // boundary_px of the GT edge), where the reconstruction error lives.
            let g_bound_t = if cfg.boundary_lambda > 0.0 {
                let k   = cfg.boundary_px * 2 + 1;
                let pad = cfg.boundary_px;
                let dil = max_pool2d(gt_t.clone(), [k, k], [1, 1], [pad, pad], [1, 1]);
                let inv = gt_t.clone().mul_scalar(-1.0).add_scalar(1.0);
                let ero = max_pool2d(inv, [k, k], [1, 1], [pad, pad], [1, 1])
                    .mul_scalar(-1.0).add_scalar(1.0);
                let band = (dil - ero).clamp(0.0, 1.0); // [B,1,H,W], 1 in the margin ring
                let relu_x = fake_logits.clone().clamp_min(0.0);
                let abs_x  = fake_logits.clone().abs();
                let bce_px = relu_x - fake_logits.clone().mul(gt_t.clone())
                    + abs_x.mul_scalar(-1.0).exp().add_scalar(1.0).log();
                let denom = band.clone().sum().clamp_min(1.0);
                (bce_px.mul(band)).sum() / denom
            } else {
                Tensor::<TrainBackend, 1>::zeros([1], &device)
            };

            let g_loss_t = g_recon_t.mul_scalar(cfg.l1_lambda)
                + g_tv_t.mul_scalar(cfg.tv_lambda)
                + g_conf_t.mul_scalar(cfg.conf_lambda)
                + area_loss_t.mul_scalar(cfg.area_lambda)
                + g_bound_t.mul_scalar(cfg.boundary_lambda)
                + no_delete_t.mul_scalar(2.0f32)   // strong: never delete visible tissue
                + g_hole_t.mul_scalar(cfg.hole_lambda);

            if g_recon.is_nan() || g_recon.is_infinite() {
                log(tx, format!("WARNING: NaN/Inf at step {step}: g_recon={g_recon}"));
            }

            // Gradient accumulation
            let g_scaled = g_loss_t.mul_scalar(1.0 / accum as f32);
            accum_loss = Some(match accum_loss.take() {
                None    => g_scaled,
                Some(p) => p + g_scaled,
            });
            accum_count += 1;

            if accum_count % accum as u64 == 0 {
                let grads = accum_loss.take().unwrap().backward();
                let params = GradientsParams::from_grads(grads, &generator);
                generator = g_optim.step(cur_lr, generator, params);
            }

            let _ = tx.send(SimpleTrainMsg::BatchMetrics {
                step,
                g_recon,
                g_adv:  0.0,
                d_real: 0.0,
                d_fake: 0.0,
            });
            step += 1;
            batch_num += 1;
            if batch_num % 10 == 0 {
                log(tx, format!("  batch {batch_num}/{} (step {step})", batches.len()));
            }
        }

        // Flush remaining accumulated gradient at epoch end
        if let Some(leftover) = accum_loss.take() {
            let grads = leftover.backward();
            let params = GradientsParams::from_grads(grads, &generator);
            generator = g_optim.step(cur_lr, generator, params);
        }

        if cancel.load(Ordering::Relaxed) { break; }

        // ── Validation ────────────────────────────────────────────────────────
        log(tx, format!("Validating (epoch {})…", epoch + 1));
        let metrics = run_simple_validation(
            &generator, &val_cache, &val_picks, &val_damaged,
            cfg.image_size, cfg.leaf_shape, cfg.margin_type, &device,
        );
        let _ = tx.send(SimpleTrainMsg::EpochMetrics { epoch: epoch + 1, metrics: metrics.clone() });
        log(tx, format!(
            "  Val IoU={:.4} Dice={:.4} Prec={:.4} Rec={:.4}",
            metrics.iou, metrics.dice, metrics.precision, metrics.recall,
        ));

        // ── Checkpoint ────────────────────────────────────────────────────────
        if (epoch + 1) % cfg.checkpoint_every == 0 || epoch == cfg.epochs - 1 {
            let ckpt_dir = cfg.output_dir.join(format!("checkpoint_epoch_{:04}", epoch + 1));
            match save_simple_checkpoint(&generator, &ckpt_dir) {
                Ok(()) => {
                    write_epoch_info(&ckpt_dir, epoch + 1, metrics.iou);
                    let _ = tx.send(SimpleTrainMsg::Checkpoint {
                        path: ckpt_dir.to_string_lossy().into_owned(),
                    });
                }
                Err(e) => log(tx, format!("Checkpoint save failed: {e}")),
            }
            if metrics.iou > best_iou {
                best_iou = metrics.iou;
                let best_dir = cfg.output_dir.join("checkpoint_best");
                let _ = save_simple_checkpoint(&generator, &best_dir);
                write_epoch_info(&best_dir, epoch + 1, best_iou);
                log(tx, format!("  New best IoU: {:.4}", best_iou));
            }
        }

        // ── Sample grid ───────────────────────────────────────────────────────
        if (epoch + 1) % cfg.sample_every == 0 {
            log(tx, "Building sample grid…".into());
            let (grid_pixels, w, h) = build_simple_sample_grid(
                &generator, &val_cache, &val_picks, &val_damaged,
                cfg.image_size, cfg.leaf_shape, cfg.margin_type, &device,
            );
            let _ = tx.send(SimpleTrainMsg::SampleGrid {
                epoch: epoch + 1,
                pixels: grid_pixels,
                width: w,
                height: h,
            });
        }
    }

    log(tx, "Training complete.".into());
    Ok(())
}

// ── 2-channel CPU pipeline ────────────────────────────────────────────────────

/// Builds [B, 4, H, W] batches: channels R, G, B, A — all normalised to [−1, 1].
/// GT is [B, 1, H, W] filled binary mask.
/// Augmentation (flip/rot) is applied consistently to both RGBA and GT mask.
fn spawn_simple_cpu_pipeline(
    cache:   Arc<Vec<Vec<u8>>>,
    params:  DamageParams,
    size:    usize,
    work_rx: mpsc::Receiver<Vec<usize>>,
    out_tx:  mpsc::SyncSender<SimpleBatch>,
) {
    std::thread::Builder::new()
        .name("simple-data-pipeline".into())
        .spawn(move || {
            let mut rng = SmallRng::from_entropy();
            while let Ok(indices) = work_rx.recv() {
                let b = indices.len();
                let n = size * size;
                let mut input_data = Vec::with_capacity(b * 4 * n);
                let mut gt_data    = Vec::with_capacity(b * n);

                for idx in &indices {
                    let rgba = cache[*idx].clone();

                    // GT from intact leaf alpha, hole-filled
                    let gt_raw    = rgba.chunks(4).map(|p| p[3] > 128).collect::<Vec<_>>();
                    let gt_filled = fill_holes(&gt_raw, size, size);

                    // Apply random damage (zeroes damaged pixels completely)
                    let damaged = apply_random_damage(&rgba, size, size, &params, &mut rng);

                    // Augment RGBA and GT jointly
                    let (damaged_aug, gt_aug) = augment_pair(damaged, gt_filled, size, &mut rng);

                    // GT → f32
                    for &b in &gt_aug {
                        gt_data.push(if b { 1.0f32 } else { 0.0 });
                    }

                    // Channels 0-3: RGBA normalised to [−1, 1]
                    for ch in 0..4usize {
                        for i in 0..n {
                            input_data.push(damaged_aug[i * 4 + ch] as f32 / 127.5 - 1.0);
                        }
                    }
                }

                if out_tx.send(SimpleBatch { input_data, gt_data, batch_len: b }).is_err() {
                    break;
                }
            }
        })
        .expect("failed to spawn simple-data-pipeline thread");
}

fn upload_simple_batch(
    batch:  SimpleBatch,
    size:   usize,
    device: &TrainDevice,
) -> (Tensor<TrainBackend, 4>, Tensor<TrainBackend, 4>) {
    let b = batch.batch_len;
    let input_t = Tensor::from_data(
        TensorData::new(batch.input_data, [b, 4, size, size]), device,
    );
    let gt_t = Tensor::from_data(
        TensorData::new(batch.gt_data, [b, 1, size, size]), device,
    );
    (input_t, gt_t)
}

// ── Validation ────────────────────────────────────────────────────────────────

fn run_simple_validation(
    generator:   &UNetSimple<TrainBackend>,
    val_cache:   &[Vec<u8>],
    val_picks:   &[usize],
    val_damaged: &[Vec<u8>],
    size:        usize,
    shape_cls:   u32,
    margin_cls:  u32,
    device:      &TrainDevice,
) -> MetricsSnapshot {
    let gen_infer = generator.clone().valid();
    let mut snaps = Vec::new();

    for (i, &idx) in val_picks.iter().enumerate() {
        let img     = &val_cache[idx];
        let damaged = &val_damaged[i];

        let input_data = build_4ch_input(damaged, size);

        let input_t: Tensor<InferBackend, 4> = Tensor::from_data(
            TensorData::new(input_data, [1usize, 4, size, size]), device,
        );
        let pred_vec: Vec<f32> = gen_infer.forward_probs(input_t, shape_cls, margin_cls)
            .into_data().to_vec().unwrap_or_default();

        let gt_raw    = img.chunks(4).map(|p| p[3] > 128).collect::<Vec<_>>();
        let gt_filled = fill_holes(&gt_raw, size, size);
        let gt_f32: Vec<f32> = gt_filled.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();

        // Damage zone: pixels where the damaged input has alpha = 0
        let input_alpha: Vec<bool> = damaged.chunks(4).map(|p| p[3] > 128).collect();

        snaps.push(compute_damage_metrics(&pred_vec, &gt_f32, &input_alpha));
    }

    average_metrics(&snaps)
}

// ── Sample grid ───────────────────────────────────────────────────────────────

fn build_simple_sample_grid(
    generator:   &UNetSimple<TrainBackend>,
    val_cache:   &[Vec<u8>],
    val_picks:   &[usize],
    val_damaged: &[Vec<u8>],
    size:        usize,
    shape_cls:   u32,
    margin_cls:  u32,
    device:      &TrainDevice,
) -> (Vec<u8>, usize, usize) {
    let gen_infer = generator.clone().valid();
    let n = size * size;
    let mut samples: Vec<(Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();

    for &pi in &evenly_spaced(val_picks.len(), 4) {
        let orig_idx = val_picks[pi];
        let img      = &val_cache[orig_idx];
        let damaged  = &val_damaged[pi];

        let input_data = build_4ch_input(damaged, size);

        // Saliency (gradient of sum(sigmoid) w.r.t. alpha channel — ch 3)
        let sal_vec: Vec<f32> = {
            let inp_tr: Tensor<TrainBackend, 4> = Tensor::from_data(
                TensorData::new(input_data.clone(), [1usize, 4, size, size]), device,
            ).require_grad();
            let out = generator.forward(inp_tr.clone(), shape_cls, margin_cls);
            let grads = activation::sigmoid(out).sum().backward();
            if let Some(g) = inp_tr.grad(&grads) {
                // Sum absolute gradients across all 4 channels for saliency map
                // sum_dim(1): [1,4,H,W] → [1,1,H,W]; reshape to [n]
                let summed: Vec<f32> = g.abs()
                    .sum_dim(1)
                    .reshape([n])
                    .into_data().to_vec().unwrap_or_default();
                let max_v = summed.iter().cloned().fold(0.0f32, f32::max);
                if max_v > 1e-8 { summed.iter().map(|&v| v / max_v).collect() }
                else { vec![0.0f32; n] }
            } else { vec![0.0f32; n] }
        };

        let input_t: Tensor<InferBackend, 4> = Tensor::from_data(
            TensorData::new(input_data, [1usize, 4, size, size]), device,
        );
        let pred_vec: Vec<f32> = gen_infer.forward_probs(input_t, shape_cls, margin_cls)
            .into_data().to_vec().unwrap_or_default();

        let gt_raw    = img.chunks(4).map(|p| p[3] > 128).collect::<Vec<_>>();
        let gt_filled = fill_holes(&gt_raw, size, size);
        let gt_f32    = gt_filled.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();

        // Display: show actual RGB of damaged leaf (dark background where transparent)
        let display_rgba: Vec<u8> = damaged.chunks(4).flat_map(|p| {
            if p[3] > 128 { [p[0], p[1], p[2], 255u8] }
            else           { [30u8, 30, 30, 255] }
        }).collect();

        samples.push((display_rgba, gt_f32, pred_vec, sal_vec));
    }

    build_sample_grid(&samples, 128, [30, 30, 30], 0.4)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build flat [1, 4, H*W] input in NCHW order: R, G, B, A normalised to [−1,1].
fn build_4ch_input(damaged_rgba: &[u8], size: usize) -> Vec<f32> {
    let n = size * size;
    let mut data = Vec::with_capacity(4 * n);
    for ch in 0..4usize {
        for i in 0..n {
            data.push(damaged_rgba[i * 4 + ch] as f32 / 127.5 - 1.0);
        }
    }
    data
}

fn evenly_spaced(n: usize, count: usize) -> Vec<usize> {
    if n == 0 { return vec![]; }
    let count = count.min(n);
    (0..count).map(|i| i * n / count).collect()
}

/// Location weight for the hole-consistency loss: 1.0 at pixels the model
/// predicts as NOT-leaf (`prob <= 0.5`) but which its OWN thresholded
/// prediction topologically encloses (the exact "is this a real hole?" test
/// `pipeline/worker.rs`'s hole detector runs at inference, reused here at
/// train time). The mask itself is computed CPU-side from a detached copy of
/// `fake_probs` (non-differentiable — it only selects WHERE to penalise);
/// gradient flows back to the network through the differentiable `fake_probs`
/// factor in the returned loss, exactly like the existing `no_delete_t` term.
fn hole_loss(
    fake_probs: Tensor<TrainBackend, 4>,
    size:       usize,
    device:     &TrainDevice,
) -> Tensor<TrainBackend, 1> {
    const HOLE_BIN_THRESHOLD: f32 = 0.5;

    let dims = fake_probs.dims();
    let b = dims[0];
    let n = size * size;

    let pred_flat: Vec<f32> = fake_probs.clone().into_data().to_vec().unwrap_or_default();
    if pred_flat.len() != b * n {
        return Tensor::<TrainBackend, 1>::zeros([1], device);
    }

    let mut mask_flat = Vec::with_capacity(b * n);
    for s in 0..b {
        let slice = &pred_flat[s * n..(s + 1) * n];
        let pred_bin: Vec<bool> = slice.iter().map(|&p| p > HOLE_BIN_THRESHOLD).collect();
        let pred_filled = fill_holes(&pred_bin, size, size);
        for i in 0..n {
            mask_flat.push(if pred_filled[i] && !pred_bin[i] { 1.0f32 } else { 0.0f32 });
        }
    }

    let mask_t: Tensor<TrainBackend, 4> =
        Tensor::from_data(TensorData::new(mask_flat, [b, 1, size, size]), device);
    let denom = mask_t.clone().sum().clamp_min(1.0f32);
    mask_t.mul(fake_probs.mul_scalar(-1.0f32).add_scalar(1.0f32)).sum() / denom
}

/// Cosine-annealed LR: `base_lr` at epoch 0, decaying to `base_lr * lr_min_frac`
/// at the final epoch (`epoch == total_epochs - 1`). `lr_min_frac = 1.0` disables
/// decay entirely (flat LR, the old behaviour).
fn cosine_lr(base_lr: f64, epoch: usize, total_epochs: usize, lr_min_frac: f64) -> f64 {
    if total_epochs <= 1 { return base_lr; }
    let t     = (epoch as f64 / (total_epochs - 1) as f64).min(1.0);
    let cos   = 0.5 * (1.0 + (std::f64::consts::PI * t).cos());
    let floor = lr_min_frac.clamp(0.0, 1.0);
    base_lr * (floor + (1.0 - floor) * cos)
}

/// Sidecar written next to every checkpoint (`checkpoint_epoch_NNNN` and
/// `checkpoint_best`) so a later "Resume checkpoint…" pick can auto-fill the
/// resume epoch / best-IoU fields instead of the user having to remember them.
fn write_epoch_info(dir: &Path, epoch: usize, iou: f32) {
    let _ = std::fs::write(dir.join("epoch_info.txt"), format!("epoch={epoch}\niou={iou:.4}\n"));
}

/// Reads the sidecar `write_epoch_info` writes. Returns `None` for checkpoints
/// saved before this feature existed — caller falls back to manual entry.
pub fn read_epoch_info(dir: &Path) -> Option<(usize, f32)> {
    let text = std::fs::read_to_string(dir.join("epoch_info.txt")).ok()?;
    let mut epoch = None;
    let mut iou   = None;
    for line in text.lines() {
        if let Some(v) = line.strip_prefix("epoch=") { epoch = v.trim().parse().ok(); }
        if let Some(v) = line.strip_prefix("iou=")   { iou   = v.trim().parse().ok(); }
    }
    Some((epoch?, iou?))
}
