use std::{
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
    sync::atomic::{AtomicBool, Ordering},
    collections::VecDeque,
};

use burn::{
    tensor::{Tensor, TensorData, activation, backend::{Backend, AutodiffBackend}},
    module::AutodiffModule,
};
use burn_core::{
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::CompactRecorder,
    module::Module,
};
use rand::{rngs::SmallRng, Rng, SeedableRng, seq::SliceRandom};

use crate::tabs::eroder::algorithm::{
    erode_coastal, erode_spots, erode_margin_snake, smooth_edges,
    erode_interior_ellipses, erode_apex, erode_margin_clusters, erode_lobe,
    erode_focal_sector,
};
use super::metrics::{compute_damage_metrics, average_metrics, MetricsSnapshot};
use super::model::{
    UNetGenerator, PatchDiscriminator, TrainBackend, InferBackend,
    create_train_device, backend_name,
};
use super::visualization::build_sample_grid;

type TrainDevice = <TrainBackend as Backend>::Device;

// ── Public types ──────────────────────────────────────────────────────────────

/// Messages sent from training thread → UI thread
pub enum TrainMsg {
    /// Metrics produced once per batch (for live loss curves)
    BatchMetrics {
        step:    u64,
        g_adv:   f32,
        g_recon: f32,   // Dice+BCE reconstruction loss
        d_loss:  f32,
        d_real:  f32,
        d_fake:  f32,
    },
    /// Validation metrics produced once per epoch
    EpochMetrics {
        epoch:   usize,
        metrics: MetricsSnapshot,
    },
    /// Pre-composited 4×4 sample grid, sent every N epochs
    SampleGrid {
        epoch:  usize,
        pixels: Vec<u8>,
        width:  usize,
        height: usize,
        /// Per-sample (iou, f1/dice) — one entry per row in the grid
        sample_stats: Vec<(f32, f32)>,
    },
    /// A checkpoint was saved
    Checkpoint { path: String },
    /// Informational log line
    Log(String),
    /// Training completed normally
    Finished,
    /// Fatal error — includes message
    Error(String),
}

/// Parameters controlling on-the-fly damage generation
#[derive(Clone)]
pub struct DamageParams {
    pub min_pct:          f32,
    pub max_pct:          f32,
    pub coastal:          bool,
    pub coastal_w:        f32,
    pub spots:            bool,
    pub spots_w:          f32,
    pub snake:            bool,
    pub snake_w:          f32,
    /// Large interior ellipses (radius 15–50 px) at any leaf location.
    pub ellipses:         bool,
    pub ellipses_w:       f32,
    /// Apex / tip removal: strip from one random bbox side.
    /// `apex_w` is the per-sample probability of applying apex damage.
    pub apex:             bool,
    pub apex_w:           f32,
    /// Clustered margin damage: 1–3 focused bite clusters on the leaf border.
    /// Simulates natural herbivory patterns (insect feeds intensively at 1–3 spots).
    pub clusters:         bool,
    pub clusters_w:       f32,
    /// Whole-lobe removal: 1–3 circular disc bites centred on the leaf margin.
    pub lobe:             bool,
    pub lobe_w:           f32,
    /// Focused half-plane sector removal: projects all leaf pixels onto a random
    /// direction and removes the top `fraction` by projection score.  Directly
    /// targets the failure mode of concentrated 30-40 % single-side damage.
    pub focal_sector:     bool,
    pub focal_sector_w:   f32,
    /// Probability [0, 1] that a sample is presented with zero damage
    /// (input == GT). Teaches the model the identity mapping.
    pub zero_damage_prob:       f32,
    /// Maximum damage % reached at the END of the curriculum phase.
    /// Ramps from min_pct → curriculum_max over curriculum_epochs, then
    /// jumps to max_pct for the remainder of training.
    pub curriculum_max:   f32,
}

/// Full configuration for one training run
pub struct TrainConfig {
    pub train_paths:      Vec<PathBuf>,
    pub val_paths:        Vec<PathBuf>,
    pub output_dir:       PathBuf,
    #[allow(dead_code)]
    pub species_label:    String,
    pub epochs:           usize,
    pub batch_size:       usize,
    pub lr:               f64,
    pub l1_lambda:        f32,
    pub checkpoint_every:       usize,
    pub batch_checkpoint_every: usize,   // 0 = disabled; saves checkpoint_batch_latest/
    pub sample_every:           usize,
    pub image_size:             usize,
    pub damage_params:          DamageParams,
    pub resume_from:            Option<PathBuf>,
    pub bg_color:               [u8; 3],
    /// Ramp damage difficulty over first N epochs (0 = full damage from epoch 0).
    pub curriculum_epochs:      usize,
    /// Reconstruction-only pre-training before enabling adversarial loss (0 = disabled).
    pub pretrain_epochs:        usize,
    /// Discriminator LR = G LR × d_lr_factor.
    /// Values < 1.0 make D learn slower than G, preventing discriminator dominance.
    /// Default 0.5 (D trains at half G's learning rate).
    pub d_lr_factor:            f64,
    /// Loss weight multiplier for pixels the model must reconstruct
    /// (GT=1 but input alpha=0). Prevents identity-mapping collapse. Default 7.0.
    pub recon_focus_weight:     f32,
    /// Multiplier applied to the adversarial loss in the generator step.
    /// Boosts gradient signal from D relative to the reconstruction loss.
    /// Default 20.0.
    pub adv_lambda:             f32,
    /// Weight on the Total Variation loss applied to G's sigmoid output.
    /// Suppresses checkerboard artifacts from ConvTranspose2d and high adv_lambda.
    /// Default 0.02.
    pub tv_lambda:              f32,
    /// Weight on the confidence (entropy minimisation) loss.
    /// Penalises predictions near 0.5 and rewards predictions near 0 or 1.
    /// Directly combats "insecure" generator output where G hedges near 0.5.
    /// Default 0.5.
    pub conf_lambda:            f32,
    /// Enable adaptive GAN controller.
    /// Monitors rolling d_real / d_fake means, classifies training state
    /// (healthy / lockstep / D-dominant / G-dominant), and nudges adv_lambda,
    /// conf_lambda, and d_lr within safe bounds every few epochs.
    /// Default true.
    pub adaptive_ctrl:          bool,

    /// Multi-scale reconstruction loss weight. Loss is also computed at 1/2 and
    /// 1/4 resolution, helping G learn global shape structure for large missing areas.
    /// Default 0.0 (disabled).
    pub ms_lambda: f32,
    /// Bilateral symmetry loss weight. Penalises asymmetry in G's output.
    /// Default 0.0.
    pub sym_lambda: f32,
    /// Tversky FP weight (α). Higher than β → penalises over-prediction (FP) more than
    /// under-prediction (FN). Replaces the Dice term in reconstruction loss.
    /// Tversky with α=β=0.5 is equivalent to Dice. Default 0.7.
    pub tversky_alpha: f32,
    /// Tversky FN weight (β). Lower than α → FN penalised less than FP.
    /// Set higher than α to tolerate FP and penalise FN (boost recall). Default 0.3.
    pub tversky_beta:  f32,
    /// Pixels to zero out inward from the damage boundary on the intact side.
    /// Forces the model to use long-range texture cues rather than just
    /// interpolating the boundary. 0 = disabled. Default 0.
    pub boundary_exclusion_px: usize,
    /// Gradient accumulation steps. 1 = update every batch (disabled).
    /// N = accumulate N batches before calling backward + optimizer.step().
    /// Effective batch = batch_size × accum_steps with no extra VRAM cost.
    pub accum_steps: usize,
    /// Leaf shape class index for FiLM conditioning.
    pub leaf_shape:  u32,
    /// Margin type class index for FiLM conditioning.
    pub margin_type: u32,
    /// Weight on area-head MSE loss. 0.0 = disabled.
    pub area_lambda: f32,
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn spawn_training(
    config: TrainConfig,
    tx:     mpsc::Sender<TrainMsg>,
    cancel: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        // catch_unwind catches panics that propagate to this thread.
        // GPU worker-thread panics (cubecl OOM, cube-count overflow, etc.) may
        // propagate via PoisonError when the training thread next locks a shared
        // mutex — this ensures they show as an error toast rather than a silent hang.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_training(&config, &tx, &cancel)
        }));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => { let _ = tx.send(TrainMsg::Error(e)); }
            Err(payload) => {
                let msg = payload.downcast_ref::<String>().cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "GPU panic (no message — likely OOM or cubecl deadlock)".to_string());
                let _ = tx.send(TrainMsg::Error(format!("GPU panic: {msg}")));
            }
        }
        let _ = tx.send(TrainMsg::Finished);
    });
}

// ── Main training function ────────────────────────────────────────────────────

fn run_training(
    cfg:    &TrainConfig,
    tx:     &mpsc::Sender<TrainMsg>,
    cancel: &AtomicBool,
) -> Result<(), String> {
    // ── Compute device probe ──────────────────────────────────────────────────
    log(tx, format!("Requesting {} compute device…", backend_name()));
    let device: TrainDevice = create_train_device();
    {
        // Allocate + read back a single scalar — confirms device is alive and
        // forces synchronous initialisation (important for CUDA JIT warm-up).
        let probe = Tensor::<TrainBackend, 1>::zeros([1], &device);
        let _ = probe.into_data();
    }
    log(tx, format!("{} device ready.", backend_name()));

    // ── Load & cache training images ─────────────────────────────────────────
    log(tx, format!("Loading {} training images…", cfg.train_paths.len()));
    let train_cache = Arc::new(load_images(&cfg.train_paths, cfg.image_size)?);
    log(tx, format!("Loading {} validation images…", cfg.val_paths.len()));
    let val_cache = load_images(&cfg.val_paths, cfg.image_size)?;

    // ── Pre-generate fixed validation damages ─────────────────────────────────
    // Compute val_picks once (evenly spaced, capped at 50) and pre-damage only
    // those images.  This avoids regenerating damage every epoch (metrics vary)
    // and bounds CPU work to ≤50 images regardless of val set size.
    let val_picks = evenly_spaced(val_cache.len(), 20);
    log(tx, format!("Pre-generating fixed validation damages ({} images, seeded)…", val_picks.len()));
    let mut val_rng = SmallRng::seed_from_u64(0xdeadbeef_u64);
    let val_damaged: Vec<Vec<u8>> = val_picks.iter()
        .map(|&idx| apply_random_damage(&val_cache[idx], cfg.image_size, cfg.image_size, &cfg.damage_params, &mut val_rng))
        .collect();
    log(tx, format!("Fixed validation set ready ({} images).", val_damaged.len()));

    // ── Initialise models ─────────────────────────────────────────────────────
    // Shader JIT compilation happens here on first run (several minutes).
    // Subsequent runs use the cubecl disk cache and finish in <30 s.
    log(tx, "Initialising generator weights…");
    let mut generator     = UNetGenerator::<TrainBackend>::init(&device);
    log(tx, "Initialising discriminator weights…");
    let mut discriminator = PatchDiscriminator::<TrainBackend>::init(5, &device);
    log(tx, "Models ready.");

    // ── Optimisers (Adam β1=0.5, β2=0.999 — standard for GANs) ───────────────
    let mut g_optim = AdamConfig::new()
        .with_beta_1(0.5)
        .with_beta_2(0.999)
        .init::<TrainBackend, UNetGenerator<TrainBackend>>();
    let mut d_optim = AdamConfig::new()
        .with_beta_1(0.5)
        .with_beta_2(0.999)
        .init::<TrainBackend, PatchDiscriminator<TrainBackend>>();

    // ── Optional checkpoint resume ────────────────────────────────────────────
    if let Some(resume) = &cfg.resume_from {
        log(tx, format!("Resuming from checkpoint: {}", resume.display()));
        let rec = CompactRecorder::new();
        generator = generator
            .load_file(resume.join("gen"), &rec, &device)
            .map_err(|e| format!("Failed to load generator: {e}"))?;
        discriminator = discriminator
            .load_file(resume.join("disc"), &rec, &device)
            .map_err(|e| format!("Failed to load discriminator: {e}"))?;
    }

    let lr               = cfg.lr;
    let mut d_lr         = cfg.lr * cfg.d_lr_factor;
    let mut adv_lambda   = cfg.adv_lambda;
    let mut conf_lambda  = cfg.conf_lambda;
    let mut best_iou     = 0.0f32;
    let mut step         = 0u64;
    let mut rng          = SmallRng::from_entropy();

    // ── One-time curriculum / phase announcement ──────────────────────────────
    if cfg.curriculum_epochs > 0 {
        log(tx, format!(
            "Curriculum: damage max ramps {:.0}%→{:.0}% over {} epochs, then {:.0}% full range",
            cfg.damage_params.min_pct, cfg.damage_params.curriculum_max,
            cfg.curriculum_epochs, cfg.damage_params.max_pct,
        ));
    }
    if cfg.pretrain_epochs > 0 {
        log(tx, format!(
            "Phase 1: reconstruction-only for {} epochs (adversarial disabled)",
            cfg.pretrain_epochs,
        ));
    }

    // ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in 0..cfg.epochs {
        if cancel.load(Ordering::Relaxed) {
            log(tx, "Training cancelled.");
            break;
        }

        // ── Phase transition announcement ──────────────────────────────────────
        let pretrain = epoch < cfg.pretrain_epochs;
        if epoch == cfg.pretrain_epochs && cfg.pretrain_epochs > 0 {
            log(tx, "Phase 2: adversarial (GAN) training enabled");
        }

        // ── Curriculum: linearly ramp damage difficulty ───────────────────────
        // Phase 1 (epoch 0 → curriculum_epochs): ramp min_pct → curriculum_max.
        // Phase 2 (after curriculum_epochs): use full max_pct range.
        let eff_max = if cfg.curriculum_epochs > 0 && epoch < cfg.curriculum_epochs {
            let t = epoch as f32 / cfg.curriculum_epochs as f32;  // 0→1
            (cfg.damage_params.min_pct
                + t * (cfg.damage_params.curriculum_max - cfg.damage_params.min_pct))
                .clamp(cfg.damage_params.min_pct, cfg.damage_params.curriculum_max)
        } else {
            cfg.damage_params.max_pct
        };
        let epoch_params = DamageParams { max_pct: eff_max, ..cfg.damage_params.clone() };

        // Shuffle training indices each epoch
        let mut indices: Vec<usize> = (0..train_cache.len()).collect();
        indices.shuffle(&mut rng);

        // ── CPU data pipeline for this epoch ──────────────────────────────────
        // Collect all batch index slices upfront so they can be sent to the
        // background thread without borrowing `indices` across the loop body.
        let batches: Vec<Vec<usize>> = indices
            .chunks(cfg.batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        // work_tx   — training thread sends index vecs to pipeline thread
        // work_rx   — pipeline thread receives index vecs
        // prebuilt_tx — pipeline thread sends completed PrebuiltBatch (cap=1)
        // prebuilt_rx — training thread receives them
        let (work_tx, work_rx) = mpsc::channel::<Vec<usize>>();
        let (prebuilt_tx, prebuilt_rx) = mpsc::sync_channel::<PrebuiltBatch>(1);

        spawn_cpu_pipeline(
            Arc::clone(&train_cache),
            epoch_params.clone(),
            cfg.image_size,
            work_rx,
            prebuilt_tx,
            cfg.boundary_exclusion_px,
        );

        // Prime the pipeline: send the first two work items before entering the
        // loop so that the background thread can start working while we boot up.
        let mut next_batch_idx = 0usize;
        for _ in 0..2 {
            if next_batch_idx < batches.len() {
                let _ = work_tx.send(batches[next_batch_idx].clone());
                next_batch_idx += 1;
            }
        }

        // ── Batch loop ────────────────────────────────────────────────────────
        let accum = cfg.accum_steps.max(1);
        let mut accum_g_loss: Option<Tensor<TrainBackend, 1>> = None;
        let mut accum_count  = 0u64;
        let mut batch_num = 0usize;
        let mut batch_timer = std::time::Instant::now();
        for _ in 0..batches.len() {
            if cancel.load(Ordering::Relaxed) { break; }

            // Receive the pre-built batch from the CPU pipeline thread.
            // This blocks until the CPU work is done, but in steady state the
            // background thread finished it while the GPU was running last batch.
            let prebuilt = match prebuilt_rx.recv() {
                Ok(b) => b,
                Err(_) => break,   // pipeline thread exited — shouldn't happen
            };

            // Queue the next work item immediately so the background thread can
            // start on it while we run the GPU step.
            if next_batch_idx < batches.len() {
                let _ = work_tx.send(batches[next_batch_idx].clone());
                next_batch_idx += 1;
            }

            let (input_t, gt_t) = upload_batch(prebuilt, cfg.image_size, &device);

            // Diagnostic fence logging: first 40 steps + every 100 thereafter.
            // Format: "sN/fM <label>"
            let diag = step < 40 || step % 100 == 0;

            // ── Discriminator step (skipped during pretrain phase) ────────────
            let (d_real, d_fake, d_loss) = if pretrain {
                (0.0f32, 0.0f32, 0.0f32)
            } else {
                // G-forward (detached, for D input) — apply sigmoid so D sees [0,1]
                // same as the real GT mask.  Fence 0: drain before D-forward bursts.
                let fake_detached = {
                    let logits = generator.forward(input_t.clone(), cfg.leaf_shape, cfg.margin_type);
                    let probs  = activation::sigmoid(logits);
                    Tensor::from_inner(probs.inner())
                };
                let _: f32 = fake_detached.clone().mean().into_scalar(); // fence 0
                if diag { log(tx, format!("s{step}/f0 G_fwd_det_done")); }

                let d_real_out = discriminator.forward(input_t.clone(), gt_t.clone());
                let d_fake_out = discriminator.forward(input_t.clone(), fake_detached.clone());

                let d_real: f32 = d_real_out.clone().mean().into_scalar(); // fence 1
                if diag { log(tx, format!("s{step}/f1 d_real={d_real:.4}")); }

                let d_fake: f32 = d_fake_out.clone().mean().into_scalar(); // fence 2
                if diag { log(tx, format!("s{step}/f2 d_fake={d_fake:.4}")); }

                let d_loss_t = disc_loss(d_real_out, d_fake_out);
                let d_loss: f32 = d_loss_t.clone().into_scalar();           // fence 3
                if diag { log(tx, format!("s{step}/f3 d_loss={d_loss:.4}")); }

                // Conditional D-skip: when D(x) > 0.8 AND D(G(x)) < 0.1,
                // D is dominating — skip D's backward pass to let G catch up.
                let skip_d = d_real > 0.8 && d_fake < 0.1;
                if !skip_d {
                    let grads_d = d_loss_t.backward();
                    let _: f32 = discriminator.gpu_fence().into_scalar();   // fence 3b: drain D backward
                    if diag { log(tx, format!("s{step}/f3b D_backward_done")); }
                    let d_params = GradientsParams::from_grads(grads_d, &discriminator);
                    discriminator = d_optim.step(d_lr, discriminator, d_params);
                }

                let _: f32 = discriminator.gpu_fence().into_scalar();       // fence 4
                if diag { log(tx, format!("s{step}/f4 D_opt_done")); }

                (d_real, d_fake, d_loss)
            };

            // ── Generator step ────────────────────────────────────────────────
            let mut g_adv   = 0.0f32;
            let mut g_recon = 0.0f32;
            {
                // Fence 4b: drain G-forward before submitting D-forward.
                let fake_logits_g = generator.forward(input_t.clone(), cfg.leaf_shape, cfg.margin_type);
                let _: f32 = fake_logits_g.clone().mean().into_scalar();     // fence 4b
                if diag { log(tx, format!("s{step}/f4b G_fwd_done (pretrain={pretrain})")); }

                // Reconstruction loss (BCE-with-logits on raw logits).
                let input_alpha_t = input_t.clone().narrow(1, 3, 1);
                let g_recon_t = recon_loss_from_logits(
                    fake_logits_g.clone(), gt_t.clone(),
                    input_alpha_t,
                    cfg.tversky_alpha, cfg.tversky_beta,
                );
                g_recon = g_recon_t.clone().into_scalar();                   // fence 5
                if diag { log(tx, format!("s{step}/f5 g_recon={g_recon:.4}")); }

                // TV + confidence losses applied to sigmoid probs.
                let fake_probs_tv = activation::sigmoid(fake_logits_g.clone());
                let g_tv_t   = tv_loss(fake_probs_tv.clone());
                let g_conf_t = confidence_loss(fake_probs_tv.clone());
                let g_sym_t  = symmetry_loss(fake_probs_tv.clone());
                let g_tv:   f32 = g_tv_t.clone().into_scalar();
                let g_conf: f32 = g_conf_t.clone().into_scalar();
                let g_sym: f32  = g_sym_t.clone().into_scalar();
                if diag { log(tx, format!("s{step} g_tv={g_tv:.4} g_conf={g_conf:.4} g_sym={g_sym:.4}")); }

                // Multi-scale reconstruction loss (1/2 and 1/4 resolution).
                let input_alpha_t_ms = input_t.clone().narrow(1, 3, 1);
                let g_ms_t = if cfg.ms_lambda > 0.0 {
                    let l2 = recon_loss_from_logits(
                        avg_pool2x(fake_logits_g.clone()),
                        avg_pool2x(gt_t.clone()),
                        avg_pool2x(input_alpha_t_ms.clone()),
                        cfg.tversky_alpha, cfg.tversky_beta,
                    );
                    let l4 = recon_loss_from_logits(
                        avg_pool2x(avg_pool2x(fake_logits_g.clone())),
                        avg_pool2x(avg_pool2x(gt_t.clone())),
                        avg_pool2x(avg_pool2x(input_alpha_t_ms.clone())),
                        cfg.tversky_alpha, cfg.tversky_beta,
                    );
                    l2.mul_scalar(0.5_f32) + l4.mul_scalar(0.25_f32)
                } else {
                    Tensor::<TrainBackend, 1>::zeros([1], &device)
                };

                let (g_adv_cur, g_loss_t) = if pretrain {
                    (0.0f32, g_recon_t.mul_scalar(cfg.l1_lambda)
                           + g_tv_t.mul_scalar(cfg.tv_lambda)
                           + g_conf_t.mul_scalar(conf_lambda)
                           + g_sym_t.mul_scalar(cfg.sym_lambda)
                           + g_ms_t.mul_scalar(cfg.ms_lambda))
                } else {
                    let d_fake_score = discriminator.forward(input_t.clone(), fake_probs_tv.clone());
                    let g_adv_t  = gen_adv_loss(d_fake_score);
                    let g_adv_v: f32 = g_adv_t.clone().into_scalar();       // fence 6
                    if diag { log(tx, format!("s{step}/f6 g_adv={g_adv_v:.4}")); }
                    (g_adv_v, g_adv_t.mul_scalar(adv_lambda)
                          + g_recon_t.mul_scalar(cfg.l1_lambda)
                          + g_tv_t.mul_scalar(cfg.tv_lambda)
                          + g_conf_t.mul_scalar(conf_lambda)
                          + g_sym_t.mul_scalar(cfg.sym_lambda)
                          + g_ms_t.mul_scalar(cfg.ms_lambda))
                };
                g_adv = g_adv_cur;

                // NaN/Inf guard
                if g_recon.is_nan() || g_recon.is_infinite()
                    || d_loss.is_nan() || g_adv.is_nan()
                {
                    log(tx, format!(
                        "WARNING: NaN/Inf at step {step}: d_loss={d_loss} g_adv={g_adv} g_recon={g_recon}"
                    ));
                }

                // Gradient accumulation: scale and accumulate before stepping.
                let g_loss_scaled = g_loss_t.mul_scalar(1.0 / accum as f32);
                accum_g_loss = Some(match accum_g_loss.take() {
                    None    => g_loss_scaled,
                    Some(p) => p + g_loss_scaled,
                });
                accum_count += 1;

                if accum_count % accum as u64 == 0 {
                    let grads_g = accum_g_loss.take().unwrap().backward();
                    let _: f32  = generator.gpu_fence().into_scalar();       // fence 6b
                    if diag { log(tx, format!("s{step}/f6b G_backward_done")); }
                    let g_params = GradientsParams::from_grads(grads_g, &generator);
                    generator    = g_optim.step(lr, generator, g_params);
                    // Fence 7: AFTER G optimizer.
                    let _: f32 = generator.gpu_fence().into_scalar();
                    if diag { log(tx, format!("s{step}/f7 G_opt_done")); }
                }
            }

            let _ = tx.send(TrainMsg::BatchMetrics {
                step, d_real, d_fake, d_loss, g_adv, g_recon,
            });
            step += 1;

            // ── Mid-batch checkpoint ───────────────────────────────────────────
            if cfg.batch_checkpoint_every > 0
                && step % cfg.batch_checkpoint_every as u64 == 0
            {
                let dir = cfg.output_dir.join("checkpoint_batch_latest");
                log(tx, format!("Saving batch checkpoint (step {step})…"));
                if let Err(e) = save_checkpoint(&generator, &discriminator, &dir) {
                    log(tx, format!("Warning: batch checkpoint failed — {e}"));
                } else {
                    let _ = tx.send(TrainMsg::Checkpoint { path: dir.display().to_string() });
                    log(tx, format!("  Batch checkpoint saved (step {step})"));
                }
            }

            let batch_ms = batch_timer.elapsed().as_millis();
            batch_timer = std::time::Instant::now();
            batch_num += 1;
            if batch_num % 5 == 0 {
                log(tx, format!("  batch {batch_num} (step {step}) [{batch_ms}ms]"));
            }
        }

        // Flush any remaining accumulated gradient (when batches % accum_steps != 0)
        if let Some(remaining) = accum_g_loss.take() {
            let grads_g  = remaining.backward();
            let g_params = GradientsParams::from_grads(grads_g, &generator);
            generator    = g_optim.step(lr, generator, g_params);
        }

        if cancel.load(Ordering::Relaxed) { break; }

        // ── Validation ────────────────────────────────────────────────────────
        log(tx, format!("Validating (epoch {})…", epoch + 1));
        let metrics = run_validation(&generator, &val_cache, &val_picks, &val_damaged, cfg, &device);
        let _ = tx.send(TrainMsg::EpochMetrics { epoch: epoch + 1, metrics: metrics.clone() });

        // ── Sample grid ───────────────────────────────────────────────────────
        if (epoch + 1) % cfg.sample_every == 0 {
            log(tx, "Building sample grid…");
            let grid = generate_sample_grid(&generator, &val_cache, &val_picks, &val_damaged, cfg, &device);
            let _ = tx.send(TrainMsg::SampleGrid {
                epoch:        epoch + 1,
                pixels:       grid.0,
                width:        grid.1,
                height:       grid.2,
                sample_stats: grid.3,
            });
        }

        log(tx, format!(
            "Epoch {}/{}: IoU={:.4}  Dice={:.4}  Prec={:.4}  Rec={:.4}",
            epoch + 1, cfg.epochs,
            metrics.iou, metrics.dice, metrics.precision, metrics.recall,
        ));

        // ── Device stream flush (BEFORE checkpoint save) ───────────────────────
        // Reads a generator weight scalar to drain all pending GPU commands.
        // Uses the existing TrainBackend — no new CudaDevice/stream created.
        {
            let _flush: f32 = generator.gpu_fence().into_scalar();
        }

        // ── Checkpointing ─────────────────────────────────────────────────────
        let is_best = metrics.iou > best_iou;
        if is_best { best_iou = metrics.iou; }

        let save_this = is_best || (epoch + 1) % cfg.checkpoint_every == 0;
        if save_this {
            let dir = if is_best {
                cfg.output_dir.join("checkpoint_best")
            } else {
                cfg.output_dir.join(format!("checkpoint_epoch_{:04}", epoch + 1))
            };
            log(tx, format!("Saving checkpoint: {}", dir.display()));
            if let Err(e) = save_checkpoint(&generator, &discriminator, &dir) {
                log(tx, format!("Warning: checkpoint save failed — {e}"));
            } else {
                let _ = tx.send(TrainMsg::Checkpoint { path: dir.display().to_string() });
                log(tx, "  Checkpoint saved.");
            }
        }
    }

    Ok(())
}

// ── Validation ────────────────────────────────────────────────────────────────

pub(crate) fn run_validation(
    generator:   &UNetGenerator<TrainBackend>,
    val_cache:   &[Vec<u8>],
    val_picks:   &[usize],    // pre-selected indices into val_cache
    val_damaged: &[Vec<u8>],  // pre-generated damages aligned to val_picks
    cfg:         &TrainConfig,
    device:      &TrainDevice,
) -> MetricsSnapshot {
    let gen_infer = generator.clone().valid();
    let size = cfg.image_size;
    let n = size * size;
    let mut snaps = Vec::new();

    for (i, &idx) in val_picks.iter().enumerate() {
        let img     = &val_cache[idx];
        let damaged = &val_damaged[i];

        // Build single-image tensors: [1, 4, size, size] — RGBA
        let mut input_data = Vec::with_capacity(4 * n);
        for ch in 0..4usize {
            for i in 0..n {
                input_data.push((damaged[i * 4 + ch] as f32 / 127.5) - 1.0);
            }
        }

        let input_t: Tensor<InferBackend, 4> = Tensor::from_data(
            TensorData::new(input_data, [1usize, 4, size, size]), device,
        );
        let pred_t = gen_infer.forward_probs(input_t, cfg.leaf_shape, cfg.margin_type);

        // Extract predictions
        let pred_vec: Vec<f32> = pred_t.into_data().to_vec().unwrap_or_default();

        // GT: fill holes in original mask
        let gt_raw: Vec<bool> = img.chunks(4).map(|p| p[3] > 128).collect();
        let gt_filled = fill_holes(&gt_raw, size, size);
        let gt_f32: Vec<f32> = gt_filled.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();

        // Damage zone: pixels where the damaged input has alpha = 0
        let input_alpha: Vec<bool> = damaged.chunks(4).map(|p| p[3] > 128).collect();

        // Use damage-zone-only metrics — intact pixels are excluded entirely.
        // A model copying the input scores recall=0 here.
        snaps.push(compute_damage_metrics(&pred_vec, &gt_f32, &input_alpha));
    }

    average_metrics(&snaps)
}

// ── Sample grid generation ────────────────────────────────────────────────────

fn generate_sample_grid(
    generator:   &UNetGenerator<TrainBackend>,
    val_cache:   &[Vec<u8>],
    val_picks:   &[usize],    // pre-selected indices into val_cache
    val_damaged: &[Vec<u8>],  // pre-generated damages aligned to val_picks
    cfg:         &TrainConfig,
    device:      &TrainDevice,
) -> (Vec<u8>, usize, usize, Vec<(f32, f32)>) {
    let gen_infer = generator.clone().valid();
    let size = cfg.image_size;
    let n    = size * size;
    let mut samples: Vec<(Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();

    // Pick 4 evenly-spaced entries from the pre-generated set
    let grid_picks = evenly_spaced(val_picks.len(), 4);

    for &pi in &grid_picks {
        let orig_idx = val_picks[pi];
        let img      = &val_cache[orig_idx];
        let damaged  = val_damaged[pi].clone();

        let mut input_data = Vec::with_capacity(4 * n);
        for ch in 0..4usize {
            for i in 0..n {
                input_data.push((damaged[i * 4 + ch] as f32 / 127.5) - 1.0);
            }
        }

        // ── Gradient saliency (TrainBackend, forward+backward on input) ────────
        let sal_vec: Vec<f32> = {
            let input_train: Tensor<TrainBackend, 4> = Tensor::from_data(
                TensorData::new(input_data.clone(), [1usize, 4, size, size]), device,
            ).require_grad();
            let output   = generator.forward(input_train.clone(), cfg.leaf_shape, cfg.margin_type);
            let sal_loss = activation::sigmoid(output).sum();
            let grads    = sal_loss.backward();
            if let Some(grad) = input_train.grad(&grads) {
                let summed: Vec<f32> = grad.abs()
                    .narrow(1, 0, 4)
                    .sum_dim(1)
                    .reshape([n])
                    .into_data().to_vec().unwrap_or_default();
                let max_v = summed.iter().cloned().fold(0.0f32, f32::max);
                if max_v > 1e-8 { summed.iter().map(|&v| v / max_v).collect() }
                else             { vec![0.0f32; n] }
            } else {
                vec![0.0f32; n]
            }
        };

        // ── Inference (InferBackend) ───────────────────────────────────────────
        let input_t: Tensor<InferBackend, 4> = Tensor::from_data(
            TensorData::new(input_data, [1usize, 4, size, size]), device,
        );
        let pred_vec: Vec<f32> = gen_infer.forward_probs(input_t, cfg.leaf_shape, cfg.margin_type)
            .into_data().to_vec().unwrap_or_default();

        let gt_raw: Vec<bool> = img.chunks(4).map(|p| p[3] > 128).collect();
        let gt_filled = fill_holes(&gt_raw, size, size);
        let gt_f32: Vec<f32> = gt_filled.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();

        samples.push((damaged, gt_f32, pred_vec, sal_vec));
    }

    // Per-sample IoU + F1 (Dice) for the UI stats strip — damage zone only
    let sample_stats: Vec<(f32, f32)> = samples.iter().map(|(damaged, gt, pred, _)| {
        let input_alpha: Vec<bool> = damaged.chunks(4).map(|p| p[3] > 128).collect();
        let m = compute_damage_metrics(pred, gt, &input_alpha);
        (m.iou, m.dice)
    }).collect();

    let (pixels, w, h) = build_sample_grid(&samples, 256, cfg.bg_color, 0.4);
    (pixels, w, h, sample_stats)
}

// ── CPU data pipeline ─────────────────────────────────────────────────────────
//
// While the GPU runs the current batch a background thread pre-computes the
// next batch's float arrays on CPU.  The training loop receives pre-built
// (input_data, gt_data, batch_len) tuples and only does the GPU upload itself.
//
// Channel capacity = 1: back-pressure keeps the pipeline from running too far
// ahead and consuming excess RAM, while still hiding the full CPU cost of one
// batch behind the GPU step.

pub(crate) struct PrebuiltBatch {
    pub(crate) input_data: Vec<f32>,   // [b * 4 * n]
    pub(crate) gt_data:    Vec<f32>,   // [b * 1 * n]
    pub(crate) batch_len:  usize,      // actual b (last batch may be smaller)
}

/// Dilate the damage mask inward by `exclusion_px` pixels.
///
/// After `apply_random_damage`, the damaged leaf has some alpha=0 pixels (eroded).
/// This function expands that zero-alpha region further into the intact leaf by
/// `exclusion_px` pixels, forcing the model to use only pixels far from the
/// damage boundary as reconstruction cues (long-range reasoning).
///
/// Only expands into pixels that were originally leaf (original alpha > 0).
/// Background pixels are left untouched.
pub(crate) fn apply_boundary_exclusion(
    original:     &[u8],        // original RGBA (undamaged)
    damaged:      &mut Vec<u8>, // damaged RGBA (modified in place)
    exclusion_px: usize,
    w:            usize,
    h:            usize,
) {
    if exclusion_px == 0 { return; }
    let n = w * h;

    // Seed: pixels that are eroded (damaged alpha=0) AND were originally leaf (orig alpha>0)
    let mut frontier: Vec<bool> = (0..n).map(|i| {
        damaged[i * 4 + 3] == 0 && original[i * 4 + 3] > 0
    }).collect();

    // BFS-style dilation: expand frontier by 1 pixel per iteration
    for _ in 0..exclusion_px {
        let prev = frontier.clone();
        for y in 0..h {
            for x in 0..w {
                if prev[y * w + x] {
                    // Expand to 4-connected neighbours that are still intact leaf
                    for (nx, ny) in [
                        (x.wrapping_sub(1), y), (x + 1, y),
                        (x, y.wrapping_sub(1)), (x, y + 1),
                    ] {
                        if nx < w && ny < h {
                            let ni = ny * w + nx;
                            if original[ni * 4 + 3] > 0 && damaged[ni * 4 + 3] > 0 {
                                frontier[ni] = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // Zero out all pixels in the expanded zone
    for i in 0..n {
        if frontier[i] && damaged[i * 4 + 3] > 0 {
            damaged[i * 4]     = 0;
            damaged[i * 4 + 1] = 0;
            damaged[i * 4 + 2] = 0;
            damaged[i * 4 + 3] = 0;
        }
    }
}

/// Spawn a thread that reads image indices from `work_rx`, builds the float
/// arrays for each batch (damage, gt, normalisation), and sends the
/// result to `out_tx`.
///
/// The thread exits when `work_rx` is closed (sender dropped).
pub(crate) fn spawn_cpu_pipeline(
    cache:        Arc<Vec<Vec<u8>>>,
    params:       DamageParams,
    size:         usize,
    work_rx:      mpsc::Receiver<Vec<usize>>,
    out_tx:       mpsc::SyncSender<PrebuiltBatch>,
    exclusion_px: usize,
) {
    std::thread::Builder::new()
        .name("data-pipeline".into())
        .spawn(move || {
            let mut rng = SmallRng::from_entropy();
            while let Ok(indices) = work_rx.recv() {
                let b = indices.len();
                let n = size * size;
                let mut input_data = Vec::with_capacity(b * 4 * n);
                let mut gt_data    = Vec::with_capacity(b * n);

                for idx in &indices {
                    let rgba = cache[*idx].clone();

                    // GT: fill holes in original alpha mask
                    let gt_raw    = rgba.chunks(4).map(|p| p[3] > 128).collect::<Vec<_>>();
                    let gt_filled = fill_holes(&gt_raw, size, size);

                    // Damaged input: apply random erosion, then boundary exclusion zone
                    let mut damaged = apply_random_damage(&rgba, size, size, &params, &mut rng);
                    apply_boundary_exclusion(&rgba, &mut damaged, exclusion_px, size, size);

                    // Geometric augmentation: same random transform applied to
                    // both damaged input and GT mask.
                    let (damaged, gt_filled) = augment_pair(damaged, gt_filled, size, &mut rng);

                    // GT → f32 (after augmentation so GT and input share the transform)
                    for filled in &gt_filled {
                        gt_data.push(if *filled { 1.0f32 } else { 0.0 });
                    }

                    // RGBA channels normalised to [-1, 1]
                    for ch in 0..4usize {
                        for i in 0..n {
                            input_data.push((damaged[i * 4 + ch] as f32 / 127.5) - 1.0);
                        }
                    }
                }

                if out_tx.send(PrebuiltBatch { input_data, gt_data, batch_len: b }).is_err() {
                    break; // training thread dropped the receiver — exit cleanly
                }
            }
        })
        .expect("failed to spawn data-pipeline thread");
}

/// Upload a pre-built CPU batch to the GPU device.
pub(crate) fn upload_batch(
    batch:  PrebuiltBatch,
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

// ── Dynamic damage generation ─────────────────────────────────────────────────

pub(crate) fn apply_random_damage(
    rgba:   &[u8],
    w:      usize,
    h:      usize,
    params: &DamageParams,
    rng:    &mut SmallRng,
) -> Vec<u8> {
    // With probability zero_damage_prob, return the original unchanged.
    // This exposes the model to undamaged examples so it learns the identity
    // mapping and does not always try to reconstruct something that isn't there.
    if params.zero_damage_prob > 0.0 && rng.gen::<f32>() < params.zero_damage_prob {
        return rgba.to_vec();
    }

    let mut mask: Vec<bool> = rgba.chunks(4).map(|p| p[3] > 128).collect();
    let fraction = rng.gen_range(params.min_pct..=params.max_pct) / 100.0;

    // Pick ONE algorithm per sample by weighted random selection.
    // Stacking all algorithms simultaneously caused total damage to far exceed
    // the max_pct target (each removes its share from an already-reduced mask).
    // One algorithm per sample also matches biological reality (one feeding mode
    // per image) and guarantees total removal ≈ fraction.
    //
    // Weights serve as selection probabilities (unnormalised).
    // Apex is treated separately: it is always available as a supplement AFTER
    // the main algorithm with probability apex_w, since it targets the tip/side
    // strip specifically rather than general area removal.
    let mut candidates: Vec<(f32, u8)> = Vec::new(); // (weight, id)
    if params.coastal       { candidates.push((params.coastal_w,       0)); }
    if params.spots         { candidates.push((params.spots_w,         1)); }
    if params.snake         { candidates.push((params.snake_w,         2)); }
    if params.ellipses      { candidates.push((params.ellipses_w,      3)); }
    if params.clusters      { candidates.push((params.clusters_w,      4)); }
    if params.lobe          { candidates.push((params.lobe_w,          5)); }
    if params.focal_sector  { candidates.push((params.focal_sector_w,  6)); }

    if !candidates.is_empty() {
        let total_w: f32 = candidates.iter().map(|&(wt, _)| wt).sum();
        let mut pick = rng.gen::<f32>() * total_w;
        let mut chosen_id = candidates.last().unwrap().1;
        for &(wt, id) in &candidates {
            pick -= wt;
            if pick <= 0.0 { chosen_id = id; break; }
        }
        match chosen_id {
            0 => {
                // Divide total damage into 2–5 independent bites, each starting
                // from a fresh random coastal seed.  A single large-target call
                // wraps around the whole margin; small-target calls terminate
                // early and produce spatially isolated bites — matching the
                // localised feeding patterns of real herbivores.
                let n_bites  = rng.gen_range(2usize..=5);
                let per_bite = fraction / n_bites as f32;
                for _ in 0..n_bites {
                    erode_coastal(&mut mask, w, h, per_bite, 0.000005, rng);
                }
            }
            1 => erode_spots(&mut mask, w, h, fraction, rng),
            2 => erode_margin_snake(&mut mask, w, h, fraction, rng),
            3 => erode_interior_ellipses(&mut mask, w, h, fraction, rng),
            4 => erode_margin_clusters(&mut mask, w, h, fraction, rng),
            5 => erode_lobe(&mut mask, w, h, fraction, rng),
            6 => erode_focal_sector(&mut mask, w, h, fraction, rng),
            _ => {}
        }
    }

    // Apex strip: probabilistic supplement — removes a tip or side strip with
    // probability apex_w, independent of the main algorithm choice.
    if params.apex && rng.gen::<f32>() < params.apex_w {
        let cut = (fraction * 1.2_f32).clamp(0.08, 0.50);
        erode_apex(&mut mask, w, h, cut, rng);
    }

    smooth_edges(&mut mask, w, h, 1);

    let mut out = rgba.to_vec();
    for (i, &leaf) in mask.iter().enumerate() {
        if !leaf {
            // Zero all four channels: the network must not see leaf texture
            // beneath the damaged region, or the training signal is polluted.
            let b = i * 4;
            out[b]     = 0;
            out[b + 1] = 0;
            out[b + 2] = 0;
            out[b + 3] = 0;
        }
    }
    out
}

// ── Hole filling ──────────────────────────────────────────────────────────────

/// Fill interior holes in a binary mask by flood-filling background from border.
pub(crate) fn fill_holes(mask: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut background = vec![false; w * h];
    let mut queue: VecDeque<usize> = VecDeque::new();

    // Seed from border pixels that are background
    let seed = |x: usize, y: usize, bg: &mut Vec<bool>, q: &mut VecDeque<usize>| {
        let i = y * w + x;
        if !mask[i] && !bg[i] { bg[i] = true; q.push_back(i); }
    };
    for x in 0..w {
        seed(x, 0,     &mut background, &mut queue);
        seed(x, h - 1, &mut background, &mut queue);
    }
    for y in 0..h {
        seed(0,     y, &mut background, &mut queue);
        seed(w - 1, y, &mut background, &mut queue);
    }

    // BFS expansion
    while let Some(i) = queue.pop_front() {
        let x = i % w;
        let y = i / w;
        for (nx, ny) in [
            (x.wrapping_sub(1), y),
            (x + 1, y),
            (x, y.wrapping_sub(1)),
            (x, y + 1),
        ] {
            if nx < w && ny < h {
                let ni = ny * w + nx;
                if !mask[ni] && !background[ni] {
                    background[ni] = true;
                    queue.push_back(ni);
                }
            }
        }
    }

    // Any 0-pixel not reached from border = interior hole → fill to 1
    (0..w * h).map(|i| mask[i] || !background[i]).collect()
}

// ── Image loading ─────────────────────────────────────────────────────────────

pub(crate) fn load_images(paths: &[PathBuf], target_size: usize) -> Result<Vec<Vec<u8>>, String> {
    use rayon::prelude::*;
    let sz = target_size as u32;
    paths.par_iter().map(|p| {
        let img = image::open(p)
            .map_err(|e| format!("{}: {e}", p.display()))?;
        let img = img.resize_exact(sz, sz, image::imageops::FilterType::Triangle);
        let mut raw = img.to_rgba8().into_raw();
        // Zero background pixels at load time so that:
        //   1. zero_damage_prob early-return samples are consistent with damaged ones
        //   2. any source image with non-zero RGB under alpha=0 doesn't pollute training
        for p in raw.chunks_mut(4) {
            if p[3] <= 128 {
                p[0] = 0; p[1] = 0; p[2] = 0; p[3] = 0;
            } else {
                p[3] = 255; // binarise soft/feathered masks → always 0 or 255
            }
        }
        Ok(raw)
    }).collect()
}

// ── Scale jitter ──────────────────────────────────────────────────────────────

/// Randomly rescale the image+mask to 80–120% of `size`, then pad (if shrunk)
/// or random-crop (if grown) back to `size × size`.  Both image and mask use
/// nearest-neighbour sampling so no alpha blending artefacts are introduced.
pub(crate) fn scale_jitter(
    rgba: Vec<u8>,
    mask: Vec<bool>,
    size: usize,
    rng:  &mut SmallRng,
) -> (Vec<u8>, Vec<bool>) {
    let scale: f32 = rng.gen_range(0.80_f32..=1.20_f32);
    if (scale - 1.0).abs() < 0.02 { return (rgba, mask); }

    let scaled = ((size as f32 * scale).round() as usize).max(1);

    // Step 1: nearest-neighbour resize to `scaled × scaled`
    let mut s_rgba = vec![0u8; scaled * scaled * 4];
    let mut s_mask = vec![false;  scaled * scaled];
    for sy in 0..scaled {
        for sx in 0..scaled {
            let src_x = ((sx as f32 / scale).floor() as usize).min(size - 1);
            let src_y = ((sy as f32 / scale).floor() as usize).min(size - 1);
            let si = src_y * size + src_x;
            let di = sy * scaled + sx;
            s_mask[di] = mask[si];
            let (s4, d4) = (si * 4, di * 4);
            s_rgba[d4]     = rgba[s4];
            s_rgba[d4 + 1] = rgba[s4 + 1];
            s_rgba[d4 + 2] = rgba[s4 + 2];
            s_rgba[d4 + 3] = rgba[s4 + 3];
        }
    }

    // Step 2: fit back to `size × size`
    let mut out_rgba = vec![0u8; size * size * 4];
    let mut out_mask = vec![false;  size * size];

    if scaled < size {
        // Pad: centre the scaled content (background stays zero / transparent)
        let pad_x = (size - scaled) / 2;
        let pad_y = (size - scaled) / 2;
        for sy in 0..scaled {
            for sx in 0..scaled {
                let dy = sy + pad_y;
                let dx = sx + pad_x;
                if dy < size && dx < size {
                    let si = sy * scaled + sx;
                    let di = dy * size + dx;
                    out_mask[di] = s_mask[si];
                    let (s4, d4) = (si * 4, di * 4);
                    out_rgba[d4]     = s_rgba[s4];
                    out_rgba[d4 + 1] = s_rgba[s4 + 1];
                    out_rgba[d4 + 2] = s_rgba[s4 + 2];
                    out_rgba[d4 + 3] = s_rgba[s4 + 3];
                }
            }
        }
    } else {
        // Crop: random top-left offset within the oversized image
        let max_off_x = scaled - size;
        let max_off_y = scaled - size;
        let off_x = rng.gen_range(0..=max_off_x);
        let off_y = rng.gen_range(0..=max_off_y);
        for dy in 0..size {
            for dx in 0..size {
                let si = (dy + off_y) * scaled + (dx + off_x);
                let di = dy * size + dx;
                out_mask[di] = s_mask[si];
                let (s4, d4) = (si * 4, di * 4);
                out_rgba[d4]     = s_rgba[s4];
                out_rgba[d4 + 1] = s_rgba[s4 + 1];
                out_rgba[d4 + 2] = s_rgba[s4 + 2];
                out_rgba[d4 + 3] = s_rgba[s4 + 3];
            }
        }
    }

    (out_rgba, out_mask)
}

// ── Geometric augmentation ────────────────────────────────────────────────────

/// Applies an identical random geometric transform to both the damaged RGBA
/// image and the GT bool mask.
///
/// Transforms applied independently each call:
///   - horizontal flip  (50 % chance)
///   - vertical flip    (50 % chance)
///   - continuous rotation (uniform 0–360°, nearest-neighbour resample)
///
/// Forcing the model to see the same leaf in all orientations prevents it from
/// memorising damage locations and instead builds a rotation-invariant
/// leaf-shape prior.
pub(crate) fn augment_pair(
    mut rgba: Vec<u8>,
    mut mask: Vec<bool>,
    size:     usize,
    rng:      &mut SmallRng,
) -> (Vec<u8>, Vec<bool>) {
    // Horizontal flip
    if rng.gen::<bool>() {
        for y in 0..size {
            for x in 0..size / 2 {
                let x2 = size - 1 - x;
                let a  = (y * size + x)  * 4;
                let b  = (y * size + x2) * 4;
                for c in 0..4 { rgba.swap(a + c, b + c); }
                mask.swap(y * size + x, y * size + x2);
            }
        }
    }
    // Vertical flip
    if rng.gen::<bool>() {
        for y in 0..size / 2 {
            let y2 = size - 1 - y;
            for x in 0..size {
                let a = (y  * size + x) * 4;
                let b = (y2 * size + x) * 4;
                for c in 0..4 { rgba.swap(a + c, b + c); }
                mask.swap(y * size + x, y2 * size + x);
            }
        }
    }
    // Continuous random rotation: uniform 0–360°, nearest-neighbour sampling.
    // Produces every possible leaf orientation — replaces the discrete rot90
    // approach and gives the model a rotation-invariant leaf-shape prior.
    // Transparent (alpha=0) background fills corners exposed by rotation.
    {
        let angle: f32 = rng.gen::<f32>() * std::f32::consts::TAU;
        let (sin_a, cos_a) = angle.sin_cos();
        let c = (size as f32 - 1.0) * 0.5;
        let mut rr = vec![0u8; rgba.len()];
        let mut mm = vec![false; mask.len()];
        for dy in 0..size {
            for dx in 0..size {
                let fx = dx as f32 - c;
                let fy = dy as f32 - c;
                // Inverse rotation to find source pixel
                let sx = (cos_a * fx + sin_a * fy + c).round() as i32;
                let sy = (-sin_a * fx + cos_a * fy + c).round() as i32;
                if sx >= 0 && (sx as usize) < size && sy >= 0 && (sy as usize) < size {
                    let si = sy as usize * size + sx as usize;
                    let di = dy * size + dx;
                    mm[di] = mask[si];
                    let (s4, d4) = (si * 4, di * 4);
                    rr[d4]     = rgba[s4];
                    rr[d4 + 1] = rgba[s4 + 1];
                    rr[d4 + 2] = rgba[s4 + 2];
                    rr[d4 + 3] = rgba[s4 + 3];
                }
                // Out-of-bounds: leave as 0 (transparent = background)
            }
        }
        rgba = rr;
        mask = mm;
    }
    // ── Scale jitter ──────────────────────────────────────────────────────────
    // Resize ±20 % then pad/crop back — forces scale-invariant leaf-shape prior.
    let (mut rgba, mut mask) = scale_jitter(rgba, mask, size, rng);

    // ── Colour augmentation ───────────────────────────────────────────────────
    //
    // Random brightness scale applied uniformly to all three RGB channels.
    // Simulates different scanner calibrations / lighting conditions between
    // the training set and external (same-species) datasets.
    // Range: ×0.85 – ×1.15. Only applied to opaque (leaf) pixels.
    let brightness = rng.gen_range(0.85_f32..=1.15_f32);
    if (brightness - 1.0).abs() > 0.01 {
        for p in rgba.chunks_mut(4) {
            if p[3] > 128 {
                p[0] = ((p[0] as f32 * brightness).round() as u32).min(255) as u8;
                p[1] = ((p[1] as f32 * brightness).round() as u32).min(255) as u8;
                p[2] = ((p[2] as f32 * brightness).round() as u32).min(255) as u8;
            }
        }
    }

    (rgba, mask)
}

// ── Loss functions (LSGAN) ────────────────────────────────────────────────────

/// LSGAN discriminator loss with two-sided label smoothing.
///
/// Real target 0.85, fake target 0.10 — prevents the discriminator weights from
/// growing to infinity when it reaches perfect separation. The fake floor of 0.10
/// also makes it harder for D to anchor "fake" at zero, leaving more room for G.
fn disc_loss(d_real: Tensor<TrainBackend, 4>, d_fake: Tensor<TrainBackend, 4>) -> Tensor<TrainBackend, 1> {
    let real = d_real.sub_scalar(0.85f32).powf_scalar(2.0f32).mean();
    let fake = d_fake.sub_scalar(0.10f32).powf_scalar(2.0f32).mean();
    (real + fake).mul_scalar(0.5f32)
}

/// Average-pool a 4-D tensor by 2× (simple reshape-average, no parameters needed).
pub(crate) fn avg_pool2x(x: Tensor<TrainBackend, 4>) -> Tensor<TrainBackend, 4> {
    let [b, c, h, w] = x.dims();
    // [B,C,H,W] → [B,C,H/2,2,W/2,2] → mean over the 2-element dims
    let t = x.reshape([b, c, h / 2, 2, w / 2, 2]);
    // mean over dim 5 (w-pair), then dim 3 (h-pair)
    let t = t.mean_dim(5).reshape([b, c, h / 2, 2, w / 2]);
    t.mean_dim(3).reshape([b, c, h / 2, w / 2])
}

/// Bilateral symmetry regulariser: penalises the difference between the
/// predicted mask and its horizontal mirror image.  The lambda should be
/// small (0.05–0.2) so it acts as a soft prior, not a hard constraint.
pub(crate) fn symmetry_loss(pred: Tensor<TrainBackend, 4>) -> Tensor<TrainBackend, 1> {
    let flipped = pred.clone().flip([3]);
    (pred - flipped).abs().mean()
}

fn gen_adv_loss(d_fake: Tensor<TrainBackend, 4>) -> Tensor<TrainBackend, 1> {
    // Target 0.85 matches the real target in disc_loss — G wants its outputs to
    // score as high as real samples, not to reach the impossible score of 1.0.
    d_fake.sub_scalar(0.85f32).powf_scalar(2.0f32).mean().mul_scalar(0.5f32)
}

/// Total Variation loss on sigmoid probabilities [B,1,H,W].
///
/// Penalises rapid pixel-to-pixel changes by summing absolute horizontal and
/// vertical differences.  Directly suppresses 2×2 checkerboard patterns that
/// arise from ConvTranspose2d stride-2 overlap and from high-adv_lambda
/// pressure pushing G toward fake binary texture.  A small weight (0.01–0.10)
/// is sufficient — TV is zero for smooth outputs, large only for checkerboards.
pub(crate) fn tv_loss(probs: Tensor<TrainBackend, 4>) -> Tensor<TrainBackend, 1> {
    let [_b, _c, h, w] = probs.dims();
    let dy = probs.clone().narrow(2, 1, h - 1) - probs.clone().narrow(2, 0, h - 1);
    let dx = probs.clone().narrow(3, 1, w - 1) - probs.narrow(3, 0, w - 1);
    dy.abs().mean() + dx.abs().mean()
}

/// Confidence (entropy minimisation) loss on sigmoid probabilities [B,1,H,W].
///
/// Binary entropy H(p) = -p·log(p) - (1-p)·log(1-p) is maximised at p=0.5
/// (H = log(2) ≈ 0.693) and is 0 at p=0 or p=1.  Adding this loss to G's
/// objective directly penalises uncertain predictions and rewards committed
/// 0/1 outputs — producing the "confident generator" behaviour.
///
/// This is the key fix for the "insecure output" problem: the adversarial and
/// reconstruction losses don't punish p≈0.5 explicitly.  The entropy loss does.
pub(crate) fn confidence_loss(probs: Tensor<TrainBackend, 4>) -> Tensor<TrainBackend, 1> {
    let eps = 1e-6_f32;
    // Clamp to avoid log(0)
    let p = probs.clamp(eps, 1.0 - eps);
    // 1 - p
    let q = p.clone().sub_scalar(1.0_f32).abs();
    // H(p) = -p*log(p) - (1-p)*log(1-p)
    let h = p.clone().log().mul(p).mul_scalar(-1.0_f32)
           + q.clone().log().mul(q).mul_scalar(-1.0_f32);
    h.mean()
}

/// Reconstruction loss: Dice (on sigmoid probs) + numerically-stable BCE-with-logits.
///
/// **Why logits instead of sigmoid output for BCE?**
/// Standard BCE goes through a saturating sigmoid. Once the network outputs large
/// positive logits (→ pred≈1 everywhere), sigmoid gradient σ(x)*(1-σ(x))→0 and
/// ALL gradients flowing back through the output layer vanish — both from BCE and
/// from Dice. The network gets frozen in the all-white attractor.
///
/// BCE-with-logits uses the identity:
///   BCE(x, t) = max(x,0) - x*t + log(1 + exp(-|x|))
/// which has non-zero gradient for any finite logit, escaping the saturation trap.
pub(crate) fn recon_loss_from_logits(
    logits:        Tensor<TrainBackend, 4>,
    gt:            Tensor<TrainBackend, 4>,
    input_alpha:   Tensor<TrainBackend, 4>,  // [B,1,H,W] alpha channel, range [-1,1]
    tversky_alpha: f32,  // FP weight  — higher → penalise over-reconstruction more
    tversky_beta:  f32,  // FN weight  — higher → penalise under-reconstruction more
) -> Tensor<TrainBackend, 1> {
    // ── Zone masks ────────────────────────────────────────────────────────────
    // input_alpha ∈ [-1,1]: +1 = intact leaf, -1 = eroded/background
    let alpha_01 = input_alpha.add_scalar(1.0_f32).div_scalar(2.0_f32);

    // eroded_mask : GT=1 AND input was transparent — pixels to reconstruct
    let eroded_mask = (gt.clone() - alpha_01).clamp(0.0_f32, 1.0_f32);
    // bg_mask     : GT=0 — everything outside the original leaf
    let bg_mask     = gt.clone().mul_scalar(-1.0_f32).add_scalar(1.0_f32);
    // intact_mask : GT=1 AND input was opaque — trivial pass-through (excluded from main loss)
    let intact_mask = gt.clone().mul(eroded_mask.clone().mul_scalar(-1.0_f32).add_scalar(1.0_f32));

    let probs = activation::sigmoid(logits.clone());

    // ── Zone-isolated Tversky ─────────────────────────────────────────────────
    // Intact pixels are excluded entirely — they inflate TP and dilute the loss.
    // TP : predicted leaf  in the eroded zone   (correct reconstruction)
    // FP : predicted leaf  in the background    (over-reconstruction)
    // FN : predicted ~leaf in the eroded zone   (under-reconstruction)
    // α/β directly and honestly control the precision/recall tradeoff without
    // any "focus weight" workaround.
    let p_eroded = probs.clone().mul(eroded_mask.clone());
    let p_bg     = probs.clone().mul(bg_mask.clone());

    let tp  = p_eroded.clone().sum();
    let fp  = p_bg.sum();
    let fn_ = eroded_mask.clone().sum() - tp.clone();

    let eps = 1e-6_f32;
    let tversky_l =
        (tp.clone().add_scalar(eps)
        / (tp
            + fp.mul_scalar(tversky_alpha)
            + fn_.mul_scalar(tversky_beta))
            .add_scalar(eps))
        .mul_scalar(-1.0_f32).add_scalar(1.0_f32);

    // ── BCE-with-logits: max(x,0) - x*t + log(1 + exp(-|x|)) ────────────────
    let relu_x = logits.clone().clamp_min(0.0_f32);
    let abs_x  = logits.clone().abs();
    let bce_px = relu_x
        - logits.mul(gt)
        + abs_x.mul_scalar(-1.0_f32).exp().add_scalar(1.0_f32).log();

    // BCE on eroded zone — pixel-level gradient for reconstruction
    let n_eroded   = eroded_mask.clone().sum().clamp_min(1.0_f32);
    let bce_eroded = (bce_px.clone().mul(eroded_mask)).sum() / n_eroded;

    // BCE on background — pixel-level gradient against over-reconstruction
    let n_bg   = bg_mask.clone().sum().clamp_min(1.0_f32);
    let bce_bg = (bce_px.clone().mul(bg_mask)).sum() / n_bg;

    // Intact zone: very small weight (0.05) — not for grading reconstruction,
    // but to keep predictions smooth across the full image and suppress stripe
    // artifacts in regions outside the damage zone. Metrics use damage-zone only.
    let n_intact   = intact_mask.clone().sum().clamp_min(1.0_f32);
    let bce_intact = (bce_px.mul(intact_mask)).sum() / n_intact;

    tversky_l + bce_eroded + bce_bg + bce_intact.mul_scalar(0.05_f32)
}

// ── Shape cleanup ───────────────────────────────────────────────────────────────

/// "Shape only" cleanup of a predicted intact-leaf probability map: threshold at
/// `tau`, OR with the visible leaf, keep only the component connected to the visible
/// leaf (drops hallucinated islands), then fill interior holes → one solid
/// silhouette. Returns a bool mask of length `w*h`.
pub(crate) fn refine_silhouette(pred: &[f32], visible: &[bool], w: usize, h: usize, tau: f32) -> Vec<bool> {
    let n = w * h;
    let mask: Vec<bool> = (0..n)
        .map(|i| pred.get(i).copied().unwrap_or(0.0) > tau || visible[i])
        .collect();

    // Flood-fill (4-conn) from every visible pixel through `mask`.
    let mut keep = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..n {
        if visible[i] && mask[i] && !keep[i] { keep[i] = true; stack.push(i); }
    }
    while let Some(i) = stack.pop() {
        let (x, y) = (i % w, i / w);
        if x + 1 < w { let ni = i + 1; if mask[ni] && !keep[ni] { keep[ni] = true; stack.push(ni); } }
        if x > 0     { let ni = i - 1; if mask[ni] && !keep[ni] { keep[ni] = true; stack.push(ni); } }
        if y + 1 < h { let ni = i + w; if mask[ni] && !keep[ni] { keep[ni] = true; stack.push(ni); } }
        if y > 0     { let ni = i - w; if mask[ni] && !keep[ni] { keep[ni] = true; stack.push(ni); } }
    }
    fill_holes(&keep, w, h)
}

// ── Checkpoint save ───────────────────────────────────────────────────────────

fn save_checkpoint(
    generator:     &UNetGenerator<TrainBackend>,
    discriminator: &PatchDiscriminator<TrainBackend>,
    dir:           &Path,
) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;

    // Defensive flush: drain any pending GPU commands before the GPU→CPU
    // parameter readback that save_file() performs.
    // IMPORTANT: use the existing generator's parameter tensor — do NOT create
    // a new InferDevice here.  Every create_infer_device() allocates a new CUDA
    // stream in the WDDM kernel driver, and those streams are NOT freed between
    // process restarts on Windows.  Accumulated leaked streams destabilise the
    // driver, causing progressively earlier GPU hangs across training sessions.
    let _: f32 = generator.gpu_fence().into_scalar();

    let rec = CompactRecorder::new();

    // .valid() converts TrainBackend → InferBackend (strips autodiff wrapper).
    // This avoids any autodiff-graph state interacting with the parameter
    // readback, which could cause hangs if the autodiff engine has pending work.
    generator.clone().valid()
        .save_file(dir.join("gen"), &rec)
        .map_err(|e| format!("{e}"))?;
    discriminator.clone().valid()
        .save_file(dir.join("disc"), &rec)
        .map_err(|e| format!("{e}"))?;
    Ok(())
}

// ── Utilities ─────────────────────────────────────────────────────────────────

fn log(tx: &mpsc::Sender<TrainMsg>, msg: impl Into<String>) {
    let _ = tx.send(TrainMsg::Log(msg.into()));
}

fn evenly_spaced(total: usize, count: usize) -> Vec<usize> {
    if total == 0 { return vec![]; }
    (0..count.min(total))
        .map(|i| i * total / count.min(total))
        .collect()
}
