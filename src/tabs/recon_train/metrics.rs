/// Per-epoch validation metrics snapshot
#[derive(Clone, Debug, Default)]
pub struct MetricsSnapshot {
    pub iou:       f32,
    pub dice:      f32,
    pub precision: f32,
    pub recall:    f32,
    pub pixel_acc: f32,
}

/// Compute binary segmentation metrics from flat float slices.
/// `pred`: predicted probabilities [0, 1], one value per pixel
/// `gt`:   ground truth binary values (0.0 or 1.0), one value per pixel
pub fn compute_metrics(pred: &[f32], gt: &[f32]) -> MetricsSnapshot {
    assert_eq!(pred.len(), gt.len());

    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut tn = 0u64;
    let mut r#fn = 0u64;

    for (&p, &g) in pred.iter().zip(gt.iter()) {
        let p_bin = p > 0.5;
        let g_bin = g > 0.5;
        match (p_bin, g_bin) {
            (true,  true)  => tp += 1,
            (true,  false) => fp += 1,
            (false, false) => tn += 1,
            (false, true)  => r#fn += 1,
        }
    }

    let total = (tp + fp + tn + r#fn) as f32;
    let eps = 1e-7_f32;

    MetricsSnapshot {
        iou:       tp as f32 / ((tp + fp + r#fn) as f32).max(eps),
        dice:      (2 * tp) as f32 / ((2 * tp + fp + r#fn) as f32).max(eps),
        precision: tp as f32 / ((tp + fp) as f32).max(eps),
        recall:    tp as f32 / ((tp + r#fn) as f32).max(eps),
        pixel_acc: (tp + tn) as f32 / total.max(eps),
    }
}

/// Compute metrics restricted to the **damage zone only** — pixels where
/// `input_alpha[i] == false` (alpha was 0 in the damaged input).
///
/// Only these pixels are graded:
/// - GT=1, pred=1 → TP  (correctly reconstructed)
/// - GT=1, pred=0 → FN  (missed reconstruction)
/// - GT=0, pred=1 → FP  (hallucinated outside leaf)
/// - GT=0, pred=0 → TN  (correctly empty background)
///
/// Intact pixels (input_alpha=true) contribute nothing.
/// A model that simply copies the input scores recall=0 and IoU=0 here.
pub fn compute_damage_metrics(
    pred:        &[f32],
    gt:          &[f32],
    input_alpha: &[bool],   // true = pixel present in damaged input
) -> MetricsSnapshot {
    assert_eq!(pred.len(), gt.len());
    assert_eq!(pred.len(), input_alpha.len());

    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut tn = 0u64;
    let mut r#fn = 0u64;
    let mut n_damage = 0u64;

    for ((&p, &g), &intact) in pred.iter().zip(gt.iter()).zip(input_alpha.iter()) {
        if intact { continue; }  // skip pixels already present in the input
        n_damage += 1;
        let p_bin = p > 0.5;
        let g_bin = g > 0.5;
        match (p_bin, g_bin) {
            (true,  true)  => tp += 1,
            (true,  false) => fp += 1,
            (false, false) => tn += 1,
            (false, true)  => r#fn += 1,
        }
    }

    if n_damage == 0 {
        // No damage zone (zero-damage sample) — return NaN-safe defaults
        return MetricsSnapshot::default();
    }

    let total = (tp + fp + tn + r#fn) as f32;
    let eps = 1e-7_f32;

    MetricsSnapshot {
        iou:       tp as f32 / ((tp + fp + r#fn) as f32).max(eps),
        dice:      (2 * tp) as f32 / ((2 * tp + fp + r#fn) as f32).max(eps),
        precision: tp as f32 / ((tp + fp) as f32).max(eps),
        recall:    tp as f32 / ((tp + r#fn) as f32).max(eps),
        pixel_acc: (tp + tn) as f32 / total.max(eps),
    }
}

/// Accumulate metrics across many predictions and average
pub fn average_metrics(snaps: &[MetricsSnapshot]) -> MetricsSnapshot {
    if snaps.is_empty() { return MetricsSnapshot::default(); }
    let n = snaps.len() as f32;
    MetricsSnapshot {
        iou:       snaps.iter().map(|s| s.iou).sum::<f32>()       / n,
        dice:      snaps.iter().map(|s| s.dice).sum::<f32>()      / n,
        precision: snaps.iter().map(|s| s.precision).sum::<f32>() / n,
        recall:    snaps.iter().map(|s| s.recall).sum::<f32>()    / n,
        pixel_acc: snaps.iter().map(|s| s.pixel_acc).sum::<f32>() / n,
    }
}
