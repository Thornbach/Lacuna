//! Shared damage/augmentation/loss utilities for reconstruction training and
//! inference. Originally split across this file alongside a GAN training loop
//! (`spawn_training`/`run_training`/`PatchDiscriminator` etc.) — that loop and
//! its "GAN Train" tab were removed as unused legacy. Everything remaining
//! here is genuinely shared: `recon_simple`'s trainer (the "Recon Train" tab
//! actually in use) calls `apply_random_damage`, `augment_pair`, `tv_loss`,
//! `confidence_loss`, and `recon_loss_from_logits` directly, and the
//! Pipeline (`worker.rs`) calls `refine_silhouette` directly for its
//! reconstruction preview overlay.

use std::{
    collections::VecDeque,
    path::PathBuf,
};

use burn::tensor::{Tensor, activation, backend::Backend};
use rand::{rngs::SmallRng, Rng};

use crate::tabs::eroder::algorithm::{
    erode_coastal, erode_spots, erode_margin_snake, smooth_edges,
    erode_interior_ellipses, erode_apex, erode_margin_clusters, erode_lobe,
    erode_focal_sector,
};
use super::model::TrainBackend;

#[allow(dead_code)] // kept for signature parity with callers that don't use it
type TrainDevice = <TrainBackend as Backend>::Device;

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
    let (mut rgba, mask) = scale_jitter(rgba, mask, size, rng);

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

// ── Loss functions ────────────────────────────────────────────────────────────

/// Total Variation loss on sigmoid probabilities [B,1,H,W] — suppresses
/// checkerboard artifacts. A small weight (0.01–0.10) is sufficient — TV is
/// zero for smooth outputs, large only for checkerboards.
pub(crate) fn tv_loss(probs: Tensor<TrainBackend, 4>) -> Tensor<TrainBackend, 1> {
    let [_b, _c, h, w] = probs.dims();
    let dy = probs.clone().narrow(2, 1, h - 1) - probs.clone().narrow(2, 0, h - 1);
    let dx = probs.clone().narrow(3, 1, w - 1) - probs.narrow(3, 0, w - 1);
    dy.abs().mean() + dx.abs().mean()
}

/// Confidence (entropy minimisation) loss on sigmoid probabilities [B,1,H,W].
///
/// Binary entropy H(p) = -p·log(p) - (1-p)·log(1-p) is maximised at p=0.5
/// (H = log(2) ≈ 0.693) and is 0 at p=0 or p=1.  Adding this loss to the
/// objective directly penalises uncertain predictions and rewards committed
/// 0/1 outputs — producing "confident" generator behaviour instead of hedging
/// near p≈0.5, which the reconstruction loss alone doesn't explicitly punish.
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

/// Reconstruction loss: zone-isolated Tversky (on sigmoid probs) + numerically
/// stable BCE-with-logits.
///
/// **Why logits instead of sigmoid output for BCE?**
/// Standard BCE goes through a saturating sigmoid. Once the network outputs large
/// positive logits (→ pred≈1 everywhere), sigmoid gradient σ(x)*(1-σ(x))→0 and
/// ALL gradients flowing back through the output layer vanish — both from BCE and
/// from Tversky. The network gets frozen in the all-white attractor.
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
