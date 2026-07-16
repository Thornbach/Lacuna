//! Standalone reconstruction-model validation report generator (headless CLI).
//!
//! Run:
//!   lacuna --recon-validate [checkpoint_dir] [source_folder] [max_images]
//!
//! Loads a trained checkpoint (default: the current `checkpoint_best`) and
//! evaluates it on the exact held-out validation split used by the "Recon
//! Train" tab (`recon_simple`) — same seed, same 90/10 split logic — so this
//! is a genuine held-out evaluation, not a re-test on training data.
//!
//! Reports metrics for BOTH the damage zone (the pixels that actually had to
//! be reconstructed) and the whole leaf silhouette (the number that matters
//! for downstream leaf-area measurements), broken down overall, per synthetic
//! damage type, and per damage severity bucket. Writes a CSV of raw
//! per-sample results plus a formatted Markdown report meant to be dropped
//! straight into a thesis/paper appendix.

use std::path::PathBuf;

use burn::tensor::{Tensor, TensorData, activation, backend::Backend};
use rand::{rngs::SmallRng, SeedableRng, seq::SliceRandom};

use crate::tabs::recon_train::training::{
    DamageParams, load_images, apply_random_damage, fill_holes, refine_silhouette,
};
use crate::tabs::recon_train::metrics::{
    compute_metrics, compute_damage_metrics, average_metrics, MetricsSnapshot,
};
use crate::tabs::recon_simple::model::{load_simple_infer, create_infer_device, UNetSimple, InferBackend};
use crate::tabs::recon_simple::is_image;

/// MUST match `recon_simple/mod.rs::start_training`'s split seed exactly —
/// this is what makes "held-out" a true statement rather than an assumption.
const SEED_SPLIT: u64 = 42;
const SEED_EVAL:  u64 = 0xE1A1_u64;

// FiLM conditioning class indices. MUST match the checkpoint's training config
// (settings.json → recon_simple.leaf_shape / margin_type at time of training).
const LEAF_SHAPE:  u32 = 0; // LeafShape::Lobed
const MARGIN_TYPE: u32 = 0; // MarginType::Smooth

// The deployed Pipeline's hole-detector operating point (worker.rs::RECON_THRESHOLD).
const DEPLOYED_TAU: f32 = 0.65;

const DEFAULT_CKPT: &str = r"E:\PhD_TobiMu\01_data\processed\RECONTRAIN\checkpoint_best";
const DEFAULT_SRC:  &str = r"E:\PhD_TobiMu\01_data\processed\sorted\Healthy";
const DEFAULT_SIZE: usize = 256; // must match the checkpoint's trained image_size_px

/// The damage profile actually used to train the currently-deployed checkpoint
/// (mirrors settings.json → recon_simple at time of writing). Keep in sync if
/// the Recon Train tab's damage settings change before the next validation run.
fn trained_damage_params() -> DamageParams {
    DamageParams {
        min_pct: 1.0, max_pct: 55.0,
        coastal: true,       coastal_w: 0.4,
        spots: false,        spots_w: 0.2,
        snake: false,        snake_w: 0.1,
        ellipses: true,      ellipses_w: 0.15,
        apex: true,          apex_w: 0.5,
        clusters: true,      clusters_w: 0.4,
        lobe: true,          lobe_w: 0.4,
        focal_sector: false, focal_sector_w: 1.0,
        // Force real damage on every evaluated sample — unlike training, a
        // validation report has no use for "no damage" identity-mapping
        // samples (compute_damage_metrics would just skip them anyway).
        zero_damage_prob: 0.0,
        curriculum_max: 55.0,
    }
}

/// Same base range, but with every mode disabled except one, weight forced to
/// 1.0, so `apply_random_damage` always applies exactly that mode in isolation.
fn isolated_damage_params(mode: &str, base: &DamageParams) -> DamageParams {
    let mut p = DamageParams {
        min_pct: base.min_pct, max_pct: base.max_pct,
        coastal: false,       coastal_w: 0.0,
        spots: false,         spots_w: 0.0,
        snake: false,         snake_w: 0.0,
        ellipses: false,      ellipses_w: 0.0,
        apex: false,          apex_w: 0.0,
        clusters: false,      clusters_w: 0.0,
        lobe: false,          lobe_w: 0.0,
        focal_sector: false,  focal_sector_w: 0.0,
        zero_damage_prob: 0.0,
        curriculum_max: base.max_pct,
    };
    match mode {
        "coastal"      => { p.coastal = true;      p.coastal_w = 1.0; }
        "spots"        => { p.spots = true;        p.spots_w = 1.0; }
        "snake"        => { p.snake = true;         p.snake_w = 1.0; }
        "ellipses"     => { p.ellipses = true;      p.ellipses_w = 1.0; }
        "clusters"     => { p.clusters = true;      p.clusters_w = 1.0; }
        "lobe"         => { p.lobe = true;          p.lobe_w = 1.0; }
        "focal_sector" => { p.focal_sector = true;  p.focal_sector_w = 1.0; }
        // apex is applied as a supplement independent of the main candidate
        // pool (see apply_random_damage) — with every main mode off and
        // apex_w=1.0 it always fires alone.
        "apex"         => { p.apex = true;          p.apex_w = 1.0; }
        _ => {}
    }
    p
}

// ── Per-sample evaluation record ────────────────────────────────────────────

struct EvalSample {
    pred:               Vec<f32>,   // raw sigmoid probabilities
    gt_bool:             Vec<bool>,  // filled solid intact-leaf silhouette (ground truth)
    gt_f32:             Vec<f32>,
    visible:            Vec<bool>,  // damaged input alpha>128
    head_area:          f32,       // model's own scalar area-fraction prediction
    damage_pct_actual:  f32,       // measured (not requested) % of true leaf area zeroed
    damage_type:        String,
}

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

fn eval_batch(
    model:   &UNetSimple<InferBackend>,
    cache:   &[Vec<u8>],
    size:    usize,
    damage:  &DamageParams,
    label:   &str,
    seed:    u64,
    device:  &<InferBackend as Backend>::Device,
) -> Vec<EvalSample> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let n = size * size;
    let mut out = Vec::with_capacity(cache.len());

    for img in cache {
        let damaged = apply_random_damage(img, size, size, damage, &mut rng);
        let input_data = build_4ch_input(&damaged, size);
        let input_t: Tensor<InferBackend, 4> =
            Tensor::from_data(TensorData::new(input_data, [1usize, 4, size, size]), device);
        let (logits, area_t) = model.forward_with_area(input_t, LEAF_SHAPE, MARGIN_TYPE);
        let pred: Vec<f32> = activation::sigmoid(logits).into_data().to_vec().unwrap_or_default();
        let head_area: f32 = area_t.into_data().to_vec::<f32>().unwrap_or_default()
            .first().copied().unwrap_or(0.0);
        if pred.len() != n { continue; }

        let gt_raw: Vec<bool> = img.chunks(4).map(|p| p[3] > 128).collect();
        let gt_bool = fill_holes(&gt_raw, size, size);
        let gt_f32: Vec<f32> = gt_bool.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let visible: Vec<bool> = damaged.chunks(4).map(|p| p[3] > 128).collect();

        let gt_whole = gt_bool.iter().filter(|&&b| b).count().max(1);
        let vis_count = visible.iter().filter(|&&b| b).count();
        let damage_pct_actual = 100.0 * (1.0 - vis_count as f32 / gt_whole as f32);

        out.push(EvalSample {
            pred, gt_bool, gt_f32, visible, head_area, damage_pct_actual,
            damage_type: label.to_string(),
        });
    }
    out
}

// ── Whole-silhouette / area-accuracy scoring at a given threshold ──────────

#[derive(Clone, Default)]
struct AreaReport {
    n:             usize,
    tau:           f32,
    lost_pct_mae:  f32,
    lost_pct_bias: f32,
    whole_mape:    f32,
    whole_iou:     f32,
    whole_dice:    f32,
    head_lost_mae: f32,
}

fn score_at_tau(samples: &[&EvalSample], tau: f32, size: usize) -> AreaReport {
    let n = size * size;
    let (mut mae, mut bias, mut whole_mape, mut whole_iou, mut whole_dice, mut head_mae) =
        (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    let mut graded = 0usize;

    for s in samples {
        let sil = refine_silhouette(&s.pred, &s.visible, size, size, tau);

        let (mut true_whole, mut true_lost, mut pred_whole, mut pred_added) = (0u64, 0u64, 0u64, 0u64);
        let (mut inter, mut union, mut card_a, mut card_b) = (0u64, 0u64, 0u64, 0u64);
        for k in 0..n {
            let g = s.gt_bool[k];
            let v = s.visible[k];
            let p = sil[k];
            if g { true_whole += 1; if !v { true_lost += 1; } }
            if p { pred_whole += 1; if !v { pred_added += 1; } }
            if g { card_a += 1; }
            if p { card_b += 1; }
            if g || p { union += 1; if g && p { inter += 1; } }
        }
        if true_whole == 0 { continue; }

        let true_lost_pct = true_lost as f64 / true_whole as f64;
        let pred_lost_pct = if pred_whole > 0 { pred_added as f64 / pred_whole as f64 } else { 0.0 };
        let d = pred_lost_pct - true_lost_pct;
        mae        += d.abs();
        bias       += d;
        whole_mape += (pred_whole as f64 - true_whole as f64).abs() / true_whole as f64;
        whole_iou  += inter as f64 / (union.max(1)) as f64;
        whole_dice += (2 * inter) as f64 / ((card_a + card_b).max(1)) as f64;

        let visible_count = s.visible.iter().filter(|&&v| v).count() as f64;
        let head_whole     = (s.head_area as f64 * n as f64).max(1.0);
        let head_lost_pct  = ((head_whole - visible_count) / head_whole).clamp(0.0, 1.0);
        head_mae += (head_lost_pct - true_lost_pct).abs();
        graded += 1;
    }

    if graded == 0 { return AreaReport::default(); }
    let g = graded as f64;
    AreaReport {
        n: graded, tau,
        lost_pct_mae:  (mae / g) as f32,
        lost_pct_bias: (bias / g) as f32,
        whole_mape:    (whole_mape / g) as f32,
        whole_iou:     (whole_iou / g) as f32,
        whole_dice:    (whole_dice / g) as f32,
        head_lost_mae: (head_mae / g) as f32,
    }
}

/// Sweeps τ and returns the report at the operating point that drives the
/// population-level lost-tissue-area **bias** closest to zero — the correct
/// threshold for an area estimate (not necessarily 0.5 or the deployed 0.65).
fn sweep_best(samples: &[&EvalSample], size: usize) -> AreaReport {
    let mut best = AreaReport::default();
    let mut best_bias = f32::INFINITY;
    let mut tau = 0.20f32;
    while tau <= 0.80 {
        let r = score_at_tau(samples, tau, size);
        if r.lost_pct_bias.abs() < best_bias {
            best_bias = r.lost_pct_bias.abs();
            best = r;
        }
        tau += 0.02;
    }
    best
}

fn zone_metrics(samples: &[&EvalSample]) -> MetricsSnapshot {
    let snaps: Vec<MetricsSnapshot> = samples.iter()
        .map(|s| compute_damage_metrics(&s.pred, &s.gt_f32, &s.visible))
        .collect();
    average_metrics(&snaps)
}

fn whole_metrics_fixed(samples: &[&EvalSample]) -> MetricsSnapshot {
    let snaps: Vec<MetricsSnapshot> = samples.iter()
        .map(|s| compute_metrics(&s.pred, &s.gt_f32))
        .collect();
    average_metrics(&snaps)
}

// ── Entry point ──────────────────────────────────────────────────────────────

pub fn run(checkpoint_dir: &str, source_folder: &str, max_images: usize) {
    let ckpt = if checkpoint_dir.is_empty() { DEFAULT_CKPT.to_string() } else { checkpoint_dir.to_string() };
    let src  = if source_folder.is_empty()  { DEFAULT_SRC.to_string()  } else { source_folder.to_string() };
    let max_images = if max_images == 0 { 200 } else { max_images };
    let size = DEFAULT_SIZE;

    println!("[recon-validate] checkpoint = {ckpt}");
    println!("[recon-validate] source     = {src}");
    println!("[recon-validate] image size = {size}px  max_images = {max_images}");

    // ── 1. reproduce the EXACT held-out split "Recon Train" trains against ──
    let mut all_paths: Vec<PathBuf> = walkdir::WalkDir::new(&src)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| is_image(p))
        .collect();
    if all_paths.is_empty() {
        eprintln!("[recon-validate] no images found in {src} — abort");
        return;
    }
    let mut split_rng = SmallRng::seed_from_u64(SEED_SPLIT);
    all_paths.shuffle(&mut split_rng);
    let split = ((all_paths.len() as f32 * 0.9).ceil() as usize).min(all_paths.len() - 1);
    let n_train = split;
    let mut val_paths: Vec<PathBuf> = all_paths[split..].to_vec();
    println!(
        "[recon-validate] {} total images -> {} train / {} held-out val (split seed={SEED_SPLIT}, \
         identical to Recon Train's own split — this set was never trained on)",
        all_paths.len(), n_train, val_paths.len(),
    );
    if val_paths.len() > max_images {
        let mut sub_rng = SmallRng::seed_from_u64(SEED_EVAL);
        val_paths.shuffle(&mut sub_rng); // deterministic sub-sample, not directory-order biased
        val_paths.truncate(max_images);
    }
    let n_val_used = val_paths.len();
    println!("[recon-validate] evaluating on {n_val_used} held-out leaves");

    // ── 2. load model + images ────────────────────────────────────────────────
    let device = create_infer_device();
    let model = match load_simple_infer(&PathBuf::from(&ckpt), &device) {
        Ok(m)  => m,
        Err(e) => { eprintln!("[recon-validate] checkpoint load failed: {e}"); return; }
    };
    let cache = match load_images(&val_paths, size) {
        Ok(c)  => c,
        Err(e) => { eprintln!("[recon-validate] image load failed: {e}"); return; }
    };
    if cache.is_empty() {
        eprintln!("[recon-validate] no images loaded — abort");
        return;
    }

    // ── 3. aggregate pass: mixed damage, matches the trained distribution ────
    let base = trained_damage_params();
    println!("[recon-validate] running aggregate pass ({} samples)…", cache.len());
    let agg = eval_batch(&model, &cache, size, &base, "mixed", SEED_EVAL, &device);
    let agg_refs: Vec<&EvalSample> = agg.iter().collect();

    // ── 4. per-damage-type breakdown (isolated modes, capped subset) ─────────
    let type_n = cache.len().min(60);
    let type_cache = &cache[..type_n];
    let modes = ["coastal", "ellipses", "apex", "clusters", "lobe"];
    let mut per_type: Vec<(String, Vec<EvalSample>)> = Vec::new();
    for &mode in &modes {
        let p = isolated_damage_params(mode, &base);
        println!("[recon-validate] running per-type pass: {mode} ({type_n} samples)…");
        let samples = eval_batch(&model, type_cache, size, &p, mode, SEED_EVAL ^ 0x9E37_79B9, &device);
        per_type.push((mode.to_string(), samples));
    }

    // ── 5. per-severity breakdown (from the aggregate pass) ──────────────────
    let light:  Vec<&EvalSample> = agg.iter().filter(|s| s.damage_pct_actual < 20.0).collect();
    let medium: Vec<&EvalSample> = agg.iter().filter(|s| s.damage_pct_actual >= 20.0 && s.damage_pct_actual < 40.0).collect();
    let heavy:  Vec<&EvalSample> = agg.iter().filter(|s| s.damage_pct_actual >= 40.0).collect();

    // ── 6. write outputs ──────────────────────────────────────────────────────
    let out_dir = PathBuf::from(&ckpt).parent()
        .map(|p| p.join("validation_report"))
        .unwrap_or_else(|| PathBuf::from("validation_report"));
    let _ = std::fs::create_dir_all(&out_dir);

    write_csv(&out_dir.join("validation_samples.csv"), &agg, &per_type);
    write_report(
        &out_dir.join("VALIDATION_APPENDIX.md"),
        &ckpt, &src, size, all_paths.len(), n_train, n_val_used, type_n,
        &base, &agg_refs, &per_type, &light, &medium, &heavy,
    );

    println!("[recon-validate] wrote {}", out_dir.join("VALIDATION_APPENDIX.md").display());
    println!("[recon-validate] wrote {}", out_dir.join("validation_samples.csv").display());
    println!("[recon-validate] done.");
}

// ── Output writers ───────────────────────────────────────────────────────────

fn write_csv(path: &PathBuf, agg: &[EvalSample], per_type: &[(String, Vec<EvalSample>)]) {
    let mut s = String::from(
        "damage_type,damage_pct_actual,zone_iou,zone_dice,zone_precision,zone_recall,\
         zone_pixel_acc,whole_iou_t05,whole_dice_t05,head_area_pred\n"
    );
    let mut push_rows = |samples: &[EvalSample]| {
        for smp in samples {
            let zone  = compute_damage_metrics(&smp.pred, &smp.gt_f32, &smp.visible);
            let whole = compute_metrics(&smp.pred, &smp.gt_f32);
            s.push_str(&format!(
                "{},{:.3},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
                smp.damage_type, smp.damage_pct_actual,
                zone.iou, zone.dice, zone.precision, zone.recall, zone.pixel_acc,
                whole.iou, whole.dice, smp.head_area,
            ));
        }
    };
    push_rows(agg);
    for (_, samples) in per_type { push_rows(samples); }
    let _ = std::fs::write(path, s);
}

fn row(name: &str, val: String) -> String { format!("| {name} | {val} |\n") }

#[allow(clippy::too_many_arguments)]
fn write_report(
    path:        &PathBuf,
    ckpt:        &str,
    src:         &str,
    size:        usize,
    n_total:     usize,
    n_train:     usize,
    n_val_used:  usize,
    type_n:      usize,
    damage:      &DamageParams,
    agg:         &[&EvalSample],
    per_type:    &[(String, Vec<EvalSample>)],
    light:       &[&EvalSample],
    medium:      &[&EvalSample],
    heavy:       &[&EvalSample],
) {
    let mut s = String::new();
    s.push_str("# Reconstruction Model — Validation Report\n\n");
    s.push_str("Generated by `lacuna --recon-validate`.\n\n");
    s.push_str(&format!("- **Checkpoint**: `{ckpt}`\n"));
    s.push_str(&format!("- **Source dataset**: `{src}`\n"));
    s.push_str(&format!("- **Image resolution**: {size}×{size} px\n\n"));

    s.push_str("## Methodology\n\n");
    s.push_str(&format!(
        "- **Data split**: deterministic 90/10 train/validation split (seed {SEED_SPLIT}), \
         identical to the split used by the \"Recon Train\" tab during training — \
         {n_total} total images → {n_train} train / {} held-out validation. \
         This validation set was never seen by the optimizer.\n",
        n_total - n_train,
    ));
    s.push_str(&format!(
        "- **Evaluation set size**: {n_val_used} held-out leaves used for the aggregate pass \
         (capped for runtime); {type_n} leaves per damage-type breakdown pass.\n"
    ));
    s.push_str(&format!(
        "- **Damage protocol**: synthetic damage applied on-the-fly with the same erosion \
         algorithms and {:.0}–{:.0}% severity range used in training. Unlike training, every \
         evaluated sample receives real damage (the training-time 0%-damage identity-mapping \
         case is excluded here, since it isn't informative about reconstruction quality). \
         A fixed evaluation seed makes every number below exactly reproducible.\n",
        damage.min_pct, damage.max_pct,
    ));
    s.push_str(
        "- **Damage-zone metrics** (IoU/Dice/Precision/Recall/Pixel-Acc) are restricted to \
         pixels the model never saw in its input (the actual reconstruction task) — a model \
         that just copies its input scores 0 here.\n"
    );
    s.push_str(
        "- **Whole-leaf metrics** score the full predicted silhouette (visible + reconstructed \
         pixels together) against the true intact leaf shape — this is what matters for \
         downstream leaf-area measurements. Reported at three operating points: τ=0.50 \
         (conventional default), τ=0.65 (the threshold actually used by the Pipeline's hole \
         detector in production), and the bias-optimal τ (swept to drive the population-level \
         lost-tissue-area bias to ≈0 — the theoretically correct threshold for an area estimate, \
         found independently for each subset below).\n"
    );
    s.push_str(
        "- **Area-head MAE** is a τ-independent sanity check: the model's own dedicated scalar \
         area-fraction output compared to the true area fraction, bypassing pixel thresholding \
         entirely.\n\n"
    );

    // ── Overall ──────────────────────────────────────────────────────────────
    s.push_str("## Overall Results (mixed damage, matches trained distribution)\n\n");
    let zone  = zone_metrics(agg);
    let whole_fixed05 = whole_metrics_fixed(agg);
    let whole_t05  = score_at_tau(agg, 0.50, size);
    let whole_dep  = score_at_tau(agg, DEPLOYED_TAU, size);
    let whole_best = sweep_best(agg, size);

    s.push_str("| Metric | Value |\n|---|---|\n");
    s.push_str(&row("N samples", format!("{}", agg.len())));
    s.push_str(&row("**Damage-zone IoU**", format!("{:.4}", zone.iou)));
    s.push_str(&row("Damage-zone Dice", format!("{:.4}", zone.dice)));
    s.push_str(&row("Damage-zone Precision", format!("{:.4}", zone.precision)));
    s.push_str(&row("Damage-zone Recall", format!("{:.4}", zone.recall)));
    s.push_str(&row("Damage-zone Pixel Accuracy", format!("{:.4}", zone.pixel_acc)));
    s.push_str(&row("Whole-leaf IoU (τ=0.50)", format!("{:.4}", whole_fixed05.iou)));
    s.push_str(&row("Whole-leaf Dice (τ=0.50)", format!("{:.4}", whole_fixed05.dice)));
    s.push_str(&row("Whole-leaf IoU, area-report (τ=0.50)", format!("{:.4}", whole_t05.whole_iou)));
    s.push_str(&row("Whole-leaf Dice, area-report (τ=0.50)", format!("{:.4}", whole_t05.whole_dice)));
    s.push_str(&row("**Whole-leaf IoU (τ=0.65, deployed)**", format!("{:.4}", whole_dep.whole_iou)));
    s.push_str(&row("Whole-leaf Dice (τ=0.65, deployed)", format!("{:.4}", whole_dep.whole_dice)));
    s.push_str(&row(
        &format!("**Whole-leaf IoU (bias-optimal τ={:.2})**", whole_best.tau),
        format!("{:.4}", whole_best.whole_iou),
    ));
    s.push_str(&row("Whole-leaf Dice (bias-optimal τ)", format!("{:.4}", whole_best.whole_dice)));
    s.push_str(&row("Lost-tissue % MAE (τ=0.65, deployed)", format!("{:.2}%", whole_dep.lost_pct_mae * 100.0)));
    s.push_str(&row("Lost-tissue % bias (τ=0.65, deployed)", format!("{:+.2}%", whole_dep.lost_pct_bias * 100.0)));
    s.push_str(&row("Lost-tissue % MAE (bias-optimal τ)", format!("{:.2}%", whole_best.lost_pct_mae * 100.0)));
    s.push_str(&row("Lost-tissue % bias (bias-optimal τ)", format!("{:+.2}%", whole_best.lost_pct_bias * 100.0)));
    s.push_str(&row("Whole-area MAPE (τ=0.65, deployed)", format!("{:.2}%", whole_dep.whole_mape * 100.0)));
    s.push_str(&row("**Area-head % MAE (τ-independent)**", format!("{:.2}%", whole_dep.head_lost_mae * 100.0)));
    s.push('\n');

    // ── Per damage type ──────────────────────────────────────────────────────
    s.push_str("## Per-Damage-Type Breakdown\n\n");
    s.push_str(
        "Only synthetic damage modes enabled in the trained damage profile are broken out — \
         disabled modes (spots, snake, focal_sector at time of writing) are excluded since the \
         model has no training exposure to them and testing them would not be meaningful.\n\n"
    );
    s.push_str("| Damage type | N | Zone IoU | Zone Dice | Zone Recall | Whole IoU (τ=0.65) | Whole IoU (best τ) | Lost% MAE (best τ) | Lost% bias (best τ) |\n");
    s.push_str("|---|---|---|---|---|---|---|---|---|\n");
    for (name, samples) in per_type {
        let refs: Vec<&EvalSample> = samples.iter().collect();
        let z    = zone_metrics(&refs);
        let dep  = score_at_tau(&refs, DEPLOYED_TAU, size);
        let best = sweep_best(&refs, size);
        s.push_str(&format!(
            "| {name} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.2}% | {:+.2}% |\n",
            refs.len(), z.iou, z.dice, z.recall, dep.whole_iou, best.whole_iou,
            best.lost_pct_mae * 100.0, best.lost_pct_bias * 100.0,
        ));
    }
    s.push('\n');

    // ── Per severity ─────────────────────────────────────────────────────────
    s.push_str("## Per-Severity Breakdown (aggregate pass, bucketed by measured damage %)\n\n");
    s.push_str("| Severity | N | Zone IoU | Zone Recall | Whole IoU (τ=0.65) | Whole IoU (best τ) | Lost% MAE (best τ) |\n");
    s.push_str("|---|---|---|---|---|---|---|\n");
    for (label, bucket) in [("Light (<20%)", light), ("Medium (20–40%)", medium), ("Heavy (≥40%)", heavy)] {
        if bucket.is_empty() {
            s.push_str(&format!("| {label} | 0 | — | — | — | — | — |\n"));
            continue;
        }
        let z    = zone_metrics(bucket);
        let dep  = score_at_tau(bucket, DEPLOYED_TAU, size);
        let best = sweep_best(bucket, size);
        s.push_str(&format!(
            "| {label} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.2}% |\n",
            bucket.len(), z.iou, z.recall, dep.whole_iou, best.whole_iou, best.lost_pct_mae * 100.0,
        ));
    }
    s.push('\n');

    s.push_str("## Caveats\n\n");
    s.push_str(
        "- All damage in this evaluation is **synthetic** (algorithmic erosion), not real \
         herbivory — this measures the model's ability to reconstruct plausible intact \
         silhouettes under the same distribution it was trained on, not agreement with \
         real-world ground-truth damage.\n"
    );
    s.push_str(
        "- The bias-optimal τ is chosen independently per subset (overall / per-type / \
         per-severity), reflecting the best achievable area-accuracy operating point for that \
         subset specifically. A single deployed threshold (0.65) necessarily trades off against \
         this ideal per-subset optimum — that's why both are reported.\n"
    );
    s.push_str(&format!(
        "- Per-damage-type N ({type_n}) is smaller than the aggregate pass ({}) for runtime \
         reasons; treat per-type numbers as indicative, not as statistically precise as the \
         overall aggregate.\n",
        agg.len(),
    ));

    let _ = std::fs::write(path, s);
}
