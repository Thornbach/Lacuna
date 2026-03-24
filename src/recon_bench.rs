//! Headless A/B reconstruction benchmark — Recon (UNet-only) vs GAN — for ONE
//! species (oak), with **margin-loss-only** synthetic damage, graded on
//! **AREA accuracy** (how well each recovers the % lost tissue).
//!
//! Run:
//!   lacuna --recon-bench <oak-intact-folder> [epochs] [max_images]
//!
//! The idea under test (your eroder-on-the-fly scheme): take fully-intact oak
//! leaves, destroy their margins with the herbivory-realistic eroder modes
//! (coastal bite / snake notch / apex loss / clustered bites / whole-lobe /
//! focal sector — interior spots+ellipses OFF), then reconstruct the intact
//! silhouette. Both architectures share the SAME U-Net generator
//! (`UNetSimple` == `UNetGenerator`); they differ only in the training loss:
//!   * Recon  (recon_simple): Tversky + TV + confidence + area-head MSE.
//!   * GAN    (recon_train) : same + a PatchGAN adversarial term.
//!
//! What this harness does (faithful, compute-matched comparison):
//!   1. list intact oak leaves; split by file (= by leaf) into train / val.
//!   2. build ONE margin-loss-only DamageParams used by BOTH trainers.
//!   3. train each architecture with identical data, damage, image size,
//!      epochs and batch — each otherwise in its OWN shipped hyperparameters.
//!   4. freeze a SINGLE fixed val-damage set and reconstruct it with each
//!      model's `checkpoint_best`, grading on AREA:
//!        - lost-tissue %  MAE   (|pred − true|, the headline number)
//!        - lost-tissue %  bias  (signed: + = over-estimates damage)
//!        - whole-leaf area MAPE (|pred_whole − true_whole| / true_whole)
//!        - damage-zone IoU / Dice (shape reference)
//!   5. print a side-by-side table and write `<folder>/../recon_bench_out/results.csv`.
//!
//! NOTE: training is heavy. Without cuDNN the CUDA EP falls back to CPU and this
//! is impractically slow — run it once a GPU build is available. For a quick
//! end-to-end sanity run use small args, e.g. `--recon-bench <folder> 2 60`.

use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};
use std::sync::atomic::AtomicBool;

use burn::tensor::{Tensor, TensorData, activation, backend::Backend};
use rand::{rngs::SmallRng, SeedableRng, seq::SliceRandom};

use crate::tabs::leaf_seg::inference::list_images;
use crate::tabs::recon_train::training::{
    DamageParams, TrainConfig, TrainMsg, spawn_training,
    load_images, apply_random_damage, fill_holes, refine_silhouette,
};
use crate::tabs::recon_simple::training::{
    SimpleTrainConfig, SimpleTrainMsg, spawn_simple_training,
};
use crate::tabs::recon_simple::model::{
    load_simple_infer, create_infer_device, UNetSimple, InferBackend,
};

// ── Bench knobs (edit here) ────────────────────────────────────────────────────
const IMG:           usize = 256;   // train + eval resolution (matches deploy: RECON_SIZE)
const DEFAULT_EPOCHS: usize = 60;
const BATCH:         usize = 4;
const VAL_FRACTION:  f32   = 0.12;  // fraction of leaves held out
const MAX_VAL_EVAL:  usize = 80;    // # held-out leaves actually graded
const SEED:          u64   = 0xA11CE5;

// Oak FiLM conditioning — Lobed shape (index 0). MUST match train + eval.
const OAK_SHAPE:  u32 = 0;          // LeafShape::Lobed
const OAK_MARGIN: u32 = 0;          // MarginType::default()

/// Margin-loss-only damage: marginal modes ON (coastal/snake/apex/clusters/lobe/
/// focal_sector), interior modes OFF (spots/ellipses). coastal/lobe/clusters/apex
/// up-weighted per the oak herbivory profile.
fn margin_loss_damage() -> DamageParams {
    DamageParams {
        min_pct:        6.0,
        max_pct:        40.0,   // realistic oak margin loss (matches the 40% cap)
        coastal:        true,  coastal_w:       1.00,  // edge nibble  (up-weighted)
        spots:          false, spots_w:         0.00,  // INTERIOR — off
        snake:          true,  snake_w:         0.50,  // margin notch
        ellipses:       false, ellipses_w:      0.00,  // INTERIOR — off
        apex:           true,  apex_w:          0.40,  // tip loss     (up-weighted)
        clusters:       true,  clusters_w:      1.00,  // margin bites (up-weighted)
        lobe:           true,  lobe_w:          1.00,  // whole-lobe   (up-weighted)
        focal_sector:   true,  focal_sector_w:  0.60,  // margin wedge
        zero_damage_prob: 0.40,  // ↑ suppress over-fill of intact margins / natural sinuses
        curriculum_max:   35.0,
    }
}

// ── Area report ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Default)]
struct AreaReport {
    n:             usize,
    tau:           f32,   // operating threshold chosen by the area-bias sweep
    lost_pct_mae:  f32,   // mean |pred_lost% − true_lost%|
    lost_pct_bias: f32,   // mean (pred_lost% − true_lost%)  (+ = over-estimates damage)
    whole_mape:    f32,   // mean |pred_whole − true_whole| / true_whole
    whole_iou:     f32,   // whole-silhouette IoU (refined pred vs GT, whole leaf)
    iou:           f32,   // damage-zone IoU
    dice:          f32,   // damage-zone Dice
    head_lost_mae: f32,   // lost-tissue % MAE from the AREA HEAD scalar (τ-independent)
}

/// One held-out leaf's raw prediction + ground truth, kept so the threshold can
/// be swept without re-running the model.
struct EvalSample {
    pred:      Vec<f32>,
    gt:        Vec<bool>,   // hole-filled intact silhouette
    visible:   Vec<bool>,   // damaged-input alpha>128
    head_area: f32,         // area-head prediction: intact-leaf area / image fraction
}

// ── Entry point ────────────────────────────────────────────────────────────────

pub fn run(folder: &str, epochs: usize, max_images: usize) {
    let epochs = if epochs == 0 { DEFAULT_EPOCHS } else { epochs };
    println!("[recon-bench] oak margin-loss reconstruction A/B");
    println!("[recon-bench] folder={folder}  epochs={epochs}  img={IMG}  batch={BATCH}");

    // ── 1. list + split by leaf (= by file) ───────────────────────────────────
    let mut all = list_images(&PathBuf::from(folder));
    if all.is_empty() {
        eprintln!("[recon-bench] no images in {folder} — abort");
        return;
    }
    let mut rng = SmallRng::seed_from_u64(SEED);
    all.shuffle(&mut rng);
    if max_images > 0 && all.len() > max_images {
        all.truncate(max_images);
    }
    let n_val = ((all.len() as f32 * VAL_FRACTION).round() as usize).clamp(1, all.len().saturating_sub(1));
    let val_paths:   Vec<PathBuf> = all[..n_val].to_vec();
    let train_paths: Vec<PathBuf> = all[n_val..].to_vec();
    println!("[recon-bench] {} leaves → {} train / {} val",
             all.len(), train_paths.len(), val_paths.len());

    let damage = margin_loss_damage();
    let out_root = PathBuf::from(folder).parent()
        .map(|p| p.join("recon_bench_out"))
        .unwrap_or_else(|| PathBuf::from("recon_bench_out"));
    let _ = std::fs::create_dir_all(&out_root);

    // ── 2. train Recon (UNet-only) ────────────────────────────────────────────
    let recon_dir = out_root.join("recon");
    println!("\n[recon-bench] ===== training RECON (UNet-only) =====");
    train_recon(&train_paths, &val_paths, &recon_dir, epochs, &damage);

    // ── 3. train GAN ──────────────────────────────────────────────────────────
    let gan_dir = out_root.join("gan");
    println!("\n[recon-bench] ===== training GAN =====");
    train_gan(&train_paths, &val_paths, &gan_dir, epochs, &damage);

    // ── 4. build ONE fixed val-damage set ─────────────────────────────────────
    println!("\n[recon-bench] ===== evaluating (area accuracy) =====");
    let val_cache = match load_images(&val_paths, IMG) {
        Ok(c) => c,
        Err(e) => { eprintln!("[recon-bench] val load failed: {e}"); return; }
    };
    let picks = evenly_spaced(val_cache.len(), MAX_VAL_EVAL);
    let mut dmg_rng = SmallRng::seed_from_u64(0xEFA1_u64); // fixed eval-damage seed
    let val_damaged: Vec<Vec<u8>> = picks.iter()
        .map(|&i| apply_random_damage(&val_cache[i], IMG, IMG, &damage, &mut dmg_rng))
        .collect();
    println!("[recon-bench] graded on {} held-out leaves (fixed damage)", picks.len());

    let device = create_infer_device();

    let recon_rep = match load_simple_infer(&recon_dir.join("checkpoint_best"), &device) {
        Ok(m) => eval_area(&m, &val_cache, &picks, &val_damaged, &device),
        Err(e) => { eprintln!("[recon-bench] RECON checkpoint load failed: {e}"); AreaReport::default() }
    };
    let gan_rep = match load_simple_infer(&gan_dir.join("checkpoint_best"), &device) {
        Ok(m) => eval_area(&m, &val_cache, &picks, &val_damaged, &device),
        Err(e) => { eprintln!("[recon-bench] GAN checkpoint load failed: {e}"); AreaReport::default() }
    };

    // ── 5. report ─────────────────────────────────────────────────────────────
    print_table(&recon_rep, &gan_rep);
    write_csv(&out_root.join("results.csv"), &recon_rep, &gan_rep, epochs, train_paths.len());
    println!("[recon-bench] wrote {}", out_root.join("results.csv").display());
}

// ── Standalone clean Recon training run (GPU) ───────────────────────────────────
//
// `lacuna --recon-train <folder> [epochs] [max_images]`
//
// One proper run with the CURRENT clean config: 4-channel RGBA input, no-delete loss,
// boundary loss, boosted area head, margin-loss-only damage capped at 40%, trained at
// 512px (matches the pipeline's RECON_SIZE). Saves checkpoint_best continuously + a
// sample-grid PNG every 5 epochs so progress is visible. Build with the CUDA feature
// (`cargo build --release`) to train on the GPU.

/// Train resolution for the standalone run (matches pipeline RECON_SIZE=512).
const TRAIN_IMG: usize = 512;

pub fn run_training(folder: &str, epochs: usize, max_images: usize) {
    let epochs     = if epochs == 0 { 120 } else { epochs };
    let max_images = if max_images == 0 { 8000 } else { max_images };

    let mut all = list_images(&PathBuf::from(folder));
    if all.is_empty() { eprintln!("[recon-train] no images in {folder} — abort"); return; }

    let out = PathBuf::from(folder).parent()
        .map(|p| p.join("recon_train_out"))
        .unwrap_or_else(|| PathBuf::from("recon_train_out"));
    let _ = std::fs::create_dir_all(&out);

    // Durable log: the release build is a `windows`-subsystem app (no console), so
    // mirror every line to <out>/train.log. Watch with `Get-Content train.log -Wait`.
    let mut logf = std::fs::File::create(out.join("train.log")).ok();
    let mut logln = |s: String| {
        println!("{s}");
        if let Some(f) = logf.as_mut() { use std::io::Write; let _ = writeln!(f, "{s}"); let _ = f.flush(); }
    };

    logln(format!("[recon-train] clean Recon run  folder={folder}  epochs={epochs}  img={TRAIN_IMG}  batch={BATCH_TRAIN}(x{ACCUM} accum)"));
    logln(format!("[recon-train] output → {}", out.display()));

    // Pre-flight: the recon needs RGBA CUTOUTS (alpha carries the leaf silhouette).
    // Solid images (alpha all opaque) would make GT = the whole frame → meaningless.
    if let Ok(img) = image::open(&all[0]) {
        let rgba = img.to_rgba8();
        let (mut opaque, mut transparent) = (0u64, 0u64);
        for p in rgba.pixels() { if p[3] > 128 { opaque += 1 } else { transparent += 1 } }
        let total = (opaque + transparent).max(1);
        if transparent == 0 {
            logln("[recon-train] WARNING: first image has NO transparent pixels — these look like".into());
            logln("              SOLID images, not leaf cutouts. Recon needs RGBA cutouts (alpha =".into());
            logln("              leaf silhouette). Training on solid images will be meaningless.".into());
        } else {
            logln(format!("[recon-train] cutout check OK  ({:.0}% leaf / {:.0}% background)",
                     100.0 * opaque as f64 / total as f64, 100.0 * transparent as f64 / total as f64));
        }
    }

    let mut rng = SmallRng::seed_from_u64(SEED);
    all.shuffle(&mut rng);
    if all.len() > max_images { all.truncate(max_images); }
    let n_val = ((all.len() as f32 * 0.10).round() as usize).clamp(1, all.len().saturating_sub(1));
    let val_paths:   Vec<PathBuf> = all[..n_val].to_vec();
    let train_paths: Vec<PathBuf> = all[n_val..].to_vec();
    logln(format!("[recon-train] {} leaves → {} train / {} val  (image cache ~{} MB RAM)",
             all.len(), train_paths.len(), val_paths.len(),
             all.len() * TRAIN_IMG * TRAIN_IMG * 4 / 1_000_000));

    let cfg = SimpleTrainConfig {
        train_paths,
        val_paths,
        output_dir:         out.clone(),
        epochs,
        batch_size:         BATCH_TRAIN,
        lr:                 2e-4,
        l1_lambda:          10.0,
        tv_lambda:          0.05,
        conf_lambda:        0.5,
        recon_focus_weight: 4.0,
        tversky_alpha:      0.92,
        tversky_beta:       0.08,
        checkpoint_every:   5,          // best updates every 5 epochs → safe to stop early
        sample_every:       5,          // save a preview grid every 5 epochs
        image_size:         TRAIN_IMG,
        damage_params:      margin_loss_damage(),
        curriculum_epochs:  epochs / 4,
        resume_from:        None,
        accum_steps:        ACCUM,
        leaf_shape:         OAK_SHAPE,
        margin_type:        OAK_MARGIN,
        pretrain_epochs:    0,
        d_lr_factor:        0.5,
        adv_lambda:         0.0,
        area_lambda:        3.0,
        boundary_lambda:    3.0,
        boundary_px:        3,
    };

    let (tx, rx) = mpsc::channel();
    spawn_simple_training(cfg, tx, Arc::new(AtomicBool::new(false)));
    let sdir = out.join("samples");
    let _ = std::fs::create_dir_all(&sdir);
    for msg in rx {
        match msg {
            SimpleTrainMsg::Log(s) => logln(format!("   {s}")),
            SimpleTrainMsg::EpochMetrics { epoch, metrics } =>
                logln(format!("[epoch {epoch}/{epochs}] IoU={:.4} Dice={:.4} Prec={:.4} Rec={:.4}",
                         metrics.iou, metrics.dice, metrics.precision, metrics.recall)),
            SimpleTrainMsg::SampleGrid { epoch, pixels, width, height } => {
                if let Some(img) = image::RgbaImage::from_raw(width as u32, height as u32, pixels) {
                    let _ = img.save(sdir.join(format!("epoch_{epoch:04}.png")));
                    logln(format!("   preview → samples/epoch_{epoch:04}.png"));
                }
            }
            SimpleTrainMsg::Checkpoint { path } => logln(format!("   ckpt: {path}")),
            SimpleTrainMsg::Error(e) => logln(format!("[recon-train ERROR] {e}")),
            SimpleTrainMsg::Finished => break,
            _ => {}
        }
    }
    logln(format!("\n[recon-train] DONE. Best model: {}\\checkpoint_best\\gen.mpk", out.display()));
    logln("[recon-train] Deploy it: copy that gen.mpk over".into());
    logln("              E:\\PhD_TobiMu\\02_code\\FoliarToolbox\\models\\recon\\gen.mpk".into());
}

const BATCH_TRAIN: usize = 1;   // batch 1 → half the activation VRAM (GroupNorm needs no batch stats)
const ACCUM:       usize = 4;   // effective batch 4, no extra VRAM

// ── Training drivers ────────────────────────────────────────────────────────────

fn train_recon(train: &[PathBuf], val: &[PathBuf], out: &Path, epochs: usize, damage: &DamageParams) {
    let cfg = SimpleTrainConfig {
        train_paths:        train.to_vec(),
        val_paths:          val.to_vec(),
        output_dir:         out.to_path_buf(),
        epochs,
        batch_size:         BATCH,
        lr:                 2e-4,
        l1_lambda:          10.0,
        tv_lambda:          0.05,
        conf_lambda:        0.5,
        recon_focus_weight: 4.0,
        tversky_alpha:      0.92,   // ↑ penalise over-reconstruction
        tversky_beta:       0.08,
        checkpoint_every:   (epochs / 8).max(1),
        sample_every:       epochs + 1,        // skip sample-grid compute
        image_size:         IMG,
        damage_params:      damage.clone(),
        curriculum_epochs:  epochs / 3,
        resume_from:        None,
        accum_steps:        1,
        leaf_shape:         OAK_SHAPE,
        margin_type:        OAK_MARGIN,
        pretrain_epochs:    0,
        d_lr_factor:        0.5,
        adv_lambda:         0.0,               // recon path ignores adv anyway
        area_lambda:        3.0,               // ↑ area head = robust scalar lost-area
        boundary_lambda:    3.0,               // margin-contour emphasis
        boundary_px:        3,
    };
    let (tx, rx) = mpsc::channel();
    spawn_simple_training(cfg, tx, Arc::new(AtomicBool::new(false)));
    for msg in rx {
        match msg {
            SimpleTrainMsg::Log(s)                      => println!("   [recon] {s}"),
            SimpleTrainMsg::EpochMetrics { epoch, metrics } =>
                println!("   [recon] epoch {epoch}: IoU={:.4} Dice={:.4} Prec={:.4} Rec={:.4}",
                         metrics.iou, metrics.dice, metrics.precision, metrics.recall),
            SimpleTrainMsg::Error(e)                    => eprintln!("   [recon ERROR] {e}"),
            SimpleTrainMsg::Finished                    => break,
            _ => {}
        }
    }
}

fn train_gan(train: &[PathBuf], val: &[PathBuf], out: &Path, epochs: usize, damage: &DamageParams) {
    let cfg = TrainConfig {
        train_paths:           train.to_vec(),
        val_paths:             val.to_vec(),
        output_dir:            out.to_path_buf(),
        species_label:         "oak".into(),
        epochs,
        batch_size:            BATCH,
        lr:                    2e-4,
        l1_lambda:             0.5,
        checkpoint_every:      (epochs / 8).max(1),
        batch_checkpoint_every: 0,
        sample_every:          epochs + 1,     // skip sample-grid compute
        image_size:            IMG,
        damage_params:         damage.clone(),
        resume_from:           None,
        bg_color:              [0, 0, 0],
        curriculum_epochs:     epochs / 3,
        pretrain_epochs:       (epochs / 6).max(1),   // reconstruction warm-up before adv
        d_lr_factor:           0.5,
        recon_focus_weight:    3.0,
        adv_lambda:            20.0,
        tv_lambda:             0.02,
        conf_lambda:           0.5,
        adaptive_ctrl:         true,
        ms_lambda:             0.3,
        sym_lambda:            0.0,
        tversky_alpha:         0.8,
        tversky_beta:          0.2,
        boundary_exclusion_px: 0,
        accum_steps:           1,
        leaf_shape:            OAK_SHAPE,
        margin_type:           OAK_MARGIN,
        area_lambda:           1.0,
    };
    let (tx, rx) = mpsc::channel();
    spawn_training(cfg, tx, Arc::new(AtomicBool::new(false)));
    for msg in rx {
        match msg {
            TrainMsg::Log(s)                      => println!("   [gan] {s}"),
            TrainMsg::EpochMetrics { epoch, metrics } =>
                println!("   [gan] epoch {epoch}: IoU={:.4} Dice={:.4} Prec={:.4} Rec={:.4}",
                         metrics.iou, metrics.dice, metrics.precision, metrics.recall),
            TrainMsg::Error(e)                    => eprintln!("   [gan ERROR] {e}"),
            TrainMsg::Finished                    => break,
            _ => {}
        }
    }
}

// ── Area evaluation ──────────────────────────────────────────────────────────────

/// Run the model once per held-out leaf and keep the raw predictions.
fn collect_samples(
    model:       &UNetSimple<InferBackend>,
    val_cache:   &[Vec<u8>],
    picks:       &[usize],
    val_damaged: &[Vec<u8>],
    device:      &<InferBackend as Backend>::Device,
) -> Vec<EvalSample> {
    let n = IMG * IMG;
    let mut out = Vec::with_capacity(picks.len());
    for (i, &idx) in picks.iter().enumerate() {
        let intact  = &val_cache[idx];
        let damaged = &val_damaged[i];
        let input = build_4ch_input(damaged, IMG);
        let input_t: Tensor<InferBackend, 4> =
            Tensor::from_data(TensorData::new(input, [1usize, 4, IMG, IMG]), device);
        let (logits, area_t) = model.forward_with_area(input_t, OAK_SHAPE, OAK_MARGIN);
        let pred: Vec<f32> = activation::sigmoid(logits).into_data().to_vec().unwrap_or_default();
        let head_area: f32 = area_t.into_data().to_vec::<f32>().unwrap_or_default()
            .first().copied().unwrap_or(0.0);
        if pred.len() != n { continue; }
        let gt_raw: Vec<bool> = intact.chunks(4).map(|p| p[3] > 128).collect();
        let gt = fill_holes(&gt_raw, IMG, IMG);
        let visible: Vec<bool> = damaged.chunks(4).map(|p| p[3] > 128).collect();
        out.push(EvalSample { pred, gt, visible, head_area });
    }
    out
}

/// Score all samples at one threshold, using the "shape only" refined silhouette.
fn score_at_tau(samples: &[EvalSample], tau: f32) -> AreaReport {
    let n = IMG * IMG;
    let (mut mae, mut bias, mut whole_mape, mut whole_iou, mut dz_iou, mut dz_dice, mut head_mae) =
        (0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    let mut graded = 0usize;

    for s in samples {
        let sil = refine_silhouette(&s.pred, &s.visible, IMG, IMG, tau); // solid whole silhouette

        let (mut true_whole, mut true_lost, mut pred_whole, mut pred_added) = (0u64, 0u64, 0u64, 0u64);
        let (mut inter, mut union) = (0u64, 0u64);                 // whole-silhouette IoU
        let (mut tp, mut fp, mut fnn) = (0u64, 0u64, 0u64);        // damage-zone (v==false)
        for k in 0..n {
            let g = s.gt[k];
            let v = s.visible[k];
            let p = sil[k];
            if g { true_whole += 1; if !v { true_lost += 1; } }
            if p { pred_whole += 1; if !v { pred_added += 1; } }
            if g || p { union += 1; if g && p { inter += 1; } }
            if !v {
                match (p, g) {
                    (true,  true)  => tp += 1,
                    (true,  false) => fp += 1,
                    (false, true)  => fnn += 1,
                    _ => {}
                }
            }
        }
        if true_whole == 0 { continue; }

        let true_lost_pct = true_lost as f64 / true_whole as f64;
        let pred_lost_pct = if pred_whole > 0 { pred_added as f64 / pred_whole as f64 } else { 0.0 };
        let d = pred_lost_pct - true_lost_pct;
        mae        += d.abs();
        bias       += d;
        whole_mape += (pred_whole as f64 - true_whole as f64).abs() / true_whole as f64;
        whole_iou  += inter as f64 / (union.max(1)) as f64;
        dz_iou     += tp as f64 / ((tp + fp + fnn).max(1)) as f64;
        dz_dice    += (2 * tp) as f64 / ((2 * tp + fp + fnn).max(1)) as f64;

        // Area-head estimate: predicted intact area (head_area·n) minus visible → lost %.
        let visible_count = s.visible.iter().filter(|&&v| v).count() as f64;
        let head_whole = (s.head_area as f64 * n as f64).max(1.0);
        let head_lost_pct = ((head_whole - visible_count) / head_whole).clamp(0.0, 1.0);
        head_mae += (head_lost_pct - true_lost_pct).abs();
        graded += 1;
    }

    if graded == 0 { return AreaReport::default(); }
    let g = graded as f64;
    AreaReport {
        n:             graded,
        tau,
        lost_pct_mae:  (mae / g) as f32,
        lost_pct_bias: (bias / g) as f32,
        whole_mape:    (whole_mape / g) as f32,
        whole_iou:     (whole_iou / g) as f32,
        iou:           (dz_iou / g) as f32,
        dice:          (dz_dice / g) as f32,
        head_lost_mae: (head_mae / g) as f32,
    }
}

/// Collect predictions once, then sweep τ and pick the operating point that drives
/// the lost-tissue area **bias** closest to zero (the correct point for an area
/// estimate). Returns the report at that τ.
fn eval_area(
    model:       &UNetSimple<InferBackend>,
    val_cache:   &[Vec<u8>],
    picks:       &[usize],
    val_damaged: &[Vec<u8>],
    device:      &<InferBackend as Backend>::Device,
) -> AreaReport {
    let samples = collect_samples(model, val_cache, picks, val_damaged, device);
    if samples.is_empty() { return AreaReport::default(); }
    let mut best = AreaReport::default();
    let mut best_bias = f32::INFINITY;
    for step in 6..=14 {                       // τ ∈ {0.30, 0.35, … 0.70}
        let tau = step as f32 * 0.05;
        let r = score_at_tau(&samples, tau);
        if r.lost_pct_bias.abs() < best_bias {
            best_bias = r.lost_pct_bias.abs();
            best = r;
        }
    }
    best
}

// ── Reporting ────────────────────────────────────────────────────────────────────

fn print_table(recon: &AreaReport, gan: &AreaReport) {
    let pct = |x: f32| format!("{:+.2}%", x * 100.0);
    let abs = |x: f32| format!("{:.2}%", x * 100.0);
    println!("\n┌─────────────────────────────┬────────────┬────────────┐");
    println!("│ metric (held-out, area)     │   RECON    │    GAN     │");
    println!("├─────────────────────────────┼────────────┼────────────┤");
    println!("│ lost-tissue % MAE  (↓)      │ {:>10} │ {:>10} │", abs(recon.lost_pct_mae),  abs(gan.lost_pct_mae));
    println!("│  └ area-head % MAE  (↓)      │ {:>10} │ {:>10} │", abs(recon.head_lost_mae), abs(gan.head_lost_mae));
    println!("│ lost-tissue % bias (→0)     │ {:>10} │ {:>10} │", pct(recon.lost_pct_bias), pct(gan.lost_pct_bias));
    println!("│ whole-area MAPE    (↓)      │ {:>10} │ {:>10} │", abs(recon.whole_mape),    abs(gan.whole_mape));
    println!("│ whole-silhouette IoU (↑)    │ {:>10.4} │ {:>10.4} │", recon.whole_iou, gan.whole_iou);
    println!("│ damage-zone IoU    (↑)      │ {:>10.4} │ {:>10.4} │", recon.iou,  gan.iou);
    println!("│ damage-zone Dice   (↑)      │ {:>10.4} │ {:>10.4} │", recon.dice, gan.dice);
    println!("│ operating τ (bias→0)        │ {:>10.2} │ {:>10.2} │", recon.tau, gan.tau);
    println!("└─────────────────────────────┴────────────┴────────────┘");
    println!("(graded leaves: recon={}, gan={} · headline = whole-silhouette IoU + lost-% MAE)", recon.n, gan.n);

    let winner = if recon.lost_pct_mae <= gan.lost_pct_mae { "RECON" } else { "GAN" };
    println!("\n[recon-bench] AREA winner (lower lost-tissue % MAE): {winner}");
}

fn write_csv(path: &Path, recon: &AreaReport, gan: &AreaReport, epochs: usize, n_train: usize) {
    let mut s = String::from(
        "arch,n_graded,tau,lost_pct_mae,head_lost_mae,lost_pct_bias,whole_area_mape,whole_silhouette_iou,damage_iou,damage_dice,epochs,n_train\n");
    for (name, r) in [("recon", recon), ("gan", gan)] {
        s.push_str(&format!(
            "{name},{},{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{epochs},{n_train}\n",
            r.n, r.tau, r.lost_pct_mae, r.head_lost_mae, r.lost_pct_bias, r.whole_mape, r.whole_iou, r.iou, r.dice));
    }
    let _ = std::fs::write(path, s);
}

// ── Helpers ──────────────────────────────────────────────────────────────────────

/// Flat [1, 4, H*W] NCHW input: R,G,B,A in [−1,1] (no symmetry channel).
fn build_4ch_input(rgba: &[u8], size: usize) -> Vec<f32> {
    let n = size * size;
    let mut data = Vec::with_capacity(4 * n);
    for ch in 0..4usize {
        for i in 0..n {
            data.push(rgba[i * 4 + ch] as f32 / 127.5 - 1.0);
        }
    }
    data
}

fn evenly_spaced(n: usize, count: usize) -> Vec<usize> {
    if n == 0 { return vec![]; }
    let count = count.min(n);
    (0..count).map(|i| i * n / count).collect()
}
