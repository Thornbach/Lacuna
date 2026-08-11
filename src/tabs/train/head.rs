//! Flywheel consume-side: fine-tune the few-shot head from saved curations.
//!
//! Reads `<curations>/labels.jsonl` (+ crops written by the Pipeline tab's "Save
//! curations"), runs the frozen DINO on each crop, and nudges the existing head's
//! weights toward the user's confirmed family / rejected labels — warm-started from
//! the current head and L2-anchored to it, so it improves on the curated cases
//! WITHOUT forgetting the base. A cluster NAME the user typed that isn't a known
//! family becomes a NEW class (the loop discovers types). Writes an updated
//! `fewshot_head.json` the app reloads. No Python, no DINO features stored — the
//! crop IMAGE is re-featurized, so this survives a backbone swap.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Deserialize;

use argmin::core::{CostFunction, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use rayon::prelude::*;

use crate::tabs::pipeline::dino::DinoExtractor;
use crate::tabs::pipeline::fewshot::FewShotHead;
use crate::tabs::pipeline::hardneg_mining::{compute_fpr_for, pick_validation_tiles};

pub struct RetrainCfg {
    pub head_path:     PathBuf,
    pub dino_model:    PathBuf,
    pub curations_dir: PathBuf,
    pub out_path:      PathBuf,
    /// L-BFGS outer-iteration ceiling — a safety stop, not a target. Replaced
    /// the old fixed-epoch/fixed-`lr` gradient descent entirely (see `retrain`'s
    /// doc comment: that approach directly caused three separate convergence
    /// bugs). No `lr` field anymore — step size comes from the line search.
    ///
    /// Watch the `L-BFGS: N iteration(s)` log line: if N equals this ceiling the
    /// solve STOPPED rather than converged, and the head is whatever the solver
    /// happened to be holding. Measured instance: after the regularization fix
    /// the problem got genuinely harder (a weaker penalty means a less strongly
    /// convex objective), and a real 11,831-row run ran straight into the old
    /// 300 ceiling — costing ~10% of firing rate against a converged reference
    /// fit on identical data (17.2% vs 19.3% of defect rows at tau=0.9).
    /// Raised so that a normal run converges on its own; iterations are cheap
    /// next to the DINO feature extraction that precedes them.
    pub max_iters: u64,
    /// Standard zero-centered L2 regularization strength — matches how the
    /// ORIGINAL head was trained (`sklearn.LogisticRegression(C=1.0,
    /// class_weight="balanced")` in the Python pipeline: a from-scratch fit
    /// over all data, regularized toward zero). An earlier version of this
    /// function instead anchored each class toward the CURRENT head's own
    /// existing weights (`l2_anchor * (w - w0)`) — meant as anti-forgetting
    /// for incremental fine-tuning, but its correctness depended entirely
    /// on those existing weights being trustworthy, which repeatedly wasn't
    /// true in practice (already-saturated established classes kept
    /// getting pulled back to their own bad prior value regardless of
    /// anchor strength; brand-new all-zero classes needed a totally
    /// different, much weaker anchor just to be able to learn anything at
    /// all — two classes of bug from one root design choice). Regularizing
    /// toward zero instead means every class's resting coefficient
    /// magnitude reflects only the strength/consistency of its OWN curated
    /// evidence — no dependency on whatever it happened to inherit. The
    /// optimization still WARM-STARTS from the current head's values (fast
    /// convergence) — only the regularization TARGET changed, not the
    /// starting point. A class with NO samples this run is frozen entirely
    /// (see `retrain`'s freeze logic) rather than regularized toward zero —
    /// zero-centered L2 only makes sense for a class the current run
    /// actually has evidence about.
    ///
    /// UNITS: this is sklearn's `C` (INVERSE strength — larger = weaker
    /// regularization), not a raw penalty coefficient, so it means the same
    /// thing here as in the `LogisticRegression(C=1.0)` that fits the base
    /// head. The raw coefficient handed to the optimizer is derived as
    /// `1 / (C * n_norm)` — see `retrain`'s `l2_coef`.
    ///
    /// It used to be the raw coefficient (0.02), which was a real, measured
    /// bug rather than a tuning preference. `retrain` averages the data term
    /// over crops (`1/n_norm`) but applied L2 unscaled, so the effective
    /// strength was `C = 1/(n_norm * l2_reg)` — with 1721 crops that is
    /// C ≈ 0.029, roughly 34x stronger than the base head's C = 1.0, and it
    /// GOT WORSE the more the user curated. The solver then did the rational
    /// thing: crush W (penalized) and recover the fit through the intercepts
    /// (not penalized). Measured on a real dump (11,831 rows, 1721 crops):
    /// every trained class collapsed from ‖w‖ ≈ 90 to ≈ 1.2 while intercepts
    /// jumped to ≈ 28, leaving a near-constant predictor — defect_prob 0.507
    /// on defect rows vs 0.363 on healthy ones, and NOTHING firing at
    /// tau = 0.9. Refitting the same rows at C = 1.0 gave ‖w‖ ≈ 10, defect
    /// 0.723 vs healthy 0.246, and 19% firing at tau = 0.9. That collapse is
    /// the long-reported "retrains come back far more conservative".
    /// (An independent solver fitted on the identical dump matched Lacuna's
    /// own L-BFGS to cosine 1.000 per class, so the solver was never at
    /// fault — only this scaling.)
    pub l2_reg: f32,
    /// Cap on per-patch training rows contributed by a single curated crop
    /// (evenly strided, not just the first N) — every crop is resized to a
    /// FIXED DINO grid regardless of native pixel size (`dino.rs`'s
    /// `features_at`), so "every valid patch" from even a small 64px
    /// hardneg stamp can be 900+ near-duplicate, highly-correlated rows.
    /// Each crop's total weight (see `class_weight`) is divided across
    /// however many rows it actually contributes, so raising this doesn't
    /// change how much any ONE curation event can influence training —
    /// only how much internal diversity that event's own signal is spread
    /// across.
    pub max_patches_per_crop: usize,
    /// Optional validation gate: a folder of independent known-healthy
    /// tiles (reuses whatever folder Mining already points at, if any) to
    /// score BEFORE and AFTER this retrain with `hardneg_mining::compute_fpr`
    /// — purely informational (logged, never blocks), so a regression in
    /// false-positive rate on genuinely healthy material is visible before
    /// deciding whether to switch to the new head.
    pub validate_healthy_dir: Option<PathBuf>,
    pub validate_tau: f32,
    /// Diagnostic escape hatch for a real, reported problem: retraining
    /// keeps coming back WORSE, repeatedly, across several different
    /// algorithm fixes this session. The optimization is convex (softmax +
    /// zero-centered L2, see `l2_reg`'s doc comment) and already reads
    /// EVERY row of `labels.jsonl` every single call, not just new ones —
    /// so the data side is already "the same old data + additional data,"
    /// exactly what re-running from scratch would use. The one thing that
    /// ISN'T from scratch is the L-BFGS starting point: it normally warm-
    /// starts from the CURRENT head's own coefficients (fast convergence,
    /// but for a truly convex problem solved to convergence, the starting
    /// point shouldn't matter — if it demonstrably does, that's evidence
    /// the warm-started solve isn't actually reaching the optimum, not
    /// that warm-starting is inherently wrong). `cold_start` zero-
    /// initializes any class with samples THIS run (`trained_this_run`)
    /// instead, letting its result depend only on its own curated
    /// evidence. Deliberately does NOT zero classes with no samples this
    /// run — those get frozen at their existing value either way (nothing
    /// to retrain them from), so cold-starting them to zero would just
    /// delete them instead of leaving them untouched.
    pub cold_start: bool,
    /// Anti-forgetting: a pool of ORIGINAL training rows (dense ground truth,
    /// exported by `1Help/eval/export_base_set.py` in `write_training_dump`'s
    /// format) mixed into every retrain alongside the curations.
    ///
    /// Without this, `retrain` fits ONLY the curation rows against
    /// zero-centered L2 — so a head warm-started from hundreds of thousands of
    /// rows and refitted on ten thousand does not get fine-tuned, it gets
    /// REPLACED. The penalty pulls toward zero and the small set fully
    /// determines the solution; the warm start contributes nothing. Measured on
    /// a real run: IoU 0.475 -> 0.125 and family purity 0.985 -> 0.490 on
    /// held-out ground truth, which is the "everything is suddenly one class"
    /// failure users actually see.
    ///
    /// `base_rows` caps how many get mixed in, because the fix has to stay fast
    /// enough to run on the spot. It saturates early
    /// (`1Help/eval/base_size_curve.py`, scored on held-out leaves, with a
    /// deliberately BAD curation set):
    /// ```text
    ///   base rows      fit    IoU   family purity
    ///           0       2s  0.106           0.420
    ///      25,000       9s  0.414           0.944
    ///      50,000      16s  0.423           0.949
    ///     100,000      38s  0.430           0.954
    ///     400,000     150s  0.430           0.957
    /// ```
    /// 25k already recovers ~96% of the benefit and 400k adds 0.016 IoU for 16x
    /// the compute, so the default sits at 50k for headroom. Note those numbers
    /// come from a curation set with only 4 leaves and inverted healthy labels:
    /// the base pool does not merely prevent forgetting, it makes a bad
    /// curation set close to harmless.
    pub base_set:  Option<PathBuf>,
    pub base_rows: usize,
    /// Pull the L2 penalty toward the CURRENT head's coefficients instead of
    /// toward zero: `‖W - anchor·W₀‖²`. 0 = ordinary zero-centered L2 (the old
    /// behaviour), 1 = fully anchored.
    ///
    /// This is a different mechanism from `base_set`, not a variation of it, and
    /// the difference is why it wins. Base rows make the curations compete with
    /// tens of thousands of pseudo-observations for influence — measured on a
    /// real session, curations carried 1.9% of the total training weight. The
    /// anchor does not compete with them at all: the curations are the only
    /// data, and the penalty merely bounds how far the solution may travel.
    /// Where the curations say nothing, the gradient is zero and those weights
    /// simply stay at W₀.
    ///
    /// Measured (`1Help/eval/flywheel_learns.py`; LEARNS = balanced agreement
    /// with HELD-OUT leaves' curations, KEEPS = IoU on held-out ground truth):
    /// ```text
    ///   no retrain            LEARNS 0.195   KEEPS 0.475
    ///   base 50k              LEARNS 0.969   KEEPS 0.460
    ///   base 10k              LEARNS 0.985   KEEPS 0.431
    ///   base  5k              LEARNS 0.988   KEEPS 0.417
    ///   base  0k              LEARNS 1.000   KEEPS 0.000   (collapse)
    ///   anchor + base 10k     LEARNS 0.942   KEEPS 0.476   <- best
    /// ```
    /// Base-only trades one against the other monotonically; the anchor breaks
    /// that trade, reaching near-maximum learning at no cost to prior knowledge
    /// on a fifth of the base data.
    ///
    /// A brand-new class has W₀ = 0, so anchoring reduces to zero-centered L2
    /// for it automatically — it can still learn freely, with no special case.
    /// That was one of the two things that sank the earlier attempt at this.
    pub anchor: f32,
    /// Diagnostic: when set, write the EXACT training matrix this run built
    /// (every row's feature vector, class and weight) plus a before/after
    /// snapshot of the head, so an independent solver can be fitted on
    /// byte-identical data and the two results compared directly.
    ///
    /// Exists because a real, repeatedly-reported symptom — retrained heads
    /// coming back far more conservative — has survived every data-side
    /// explanation tested so far (mask-coverage gate, per-crop patch cap,
    /// curation crop size, hard-negative pooling shape, hard-negative boost,
    /// and feature-norm scale were each measured and each came back null or
    /// pointing the wrong way). What has NOT been isolated is this function's
    /// own solve: warm-start, the freeze of untouched classes, and the L-BFGS
    /// path. Dumping the inputs is the only way to hold the data fixed and
    /// vary ONLY the solver, instead of guessing at the symptom again.
    ///
    /// Writes `retrain_dump.bin` (large: rows × dim × 4 bytes, hundreds of MB
    /// is normal) and `retrain_diag.json` into this directory. Off unless a
    /// path is set.
    pub dump_dir: Option<PathBuf>,
}

pub struct CalibrateCfg {
    pub base_head_path: PathBuf,
    pub dino_model:     PathBuf,
    pub curations_dir:  PathBuf,
    pub out_path:       PathBuf,
    /// Confidence SCALE for calibrated classes, relative to the base head's
    /// ABSOLUTE target coefficient-row norm for calibrated classes (NOT a
    /// multiplier — an earlier version scaled relative to the base head's
    /// own median coefficient norm, which broke badly the moment that base
    /// norm itself was unreasonably large; see `calibrate()`'s comment).
    /// A sane default is a small single-digit value — high enough for a
    /// calibrated class to compete, nowhere near large enough for one
    /// tight-centroid class to dominate every softmax regardless of match.
    pub scale:           f32,
    /// Crop filenames (as they appear in `labels.jsonl`'s "crop" field) to
    /// actually use. REQUIRED, not optional — `labels.jsonl` accumulates
    /// every curation action for the whole output folder over the whole
    /// session (real-run confirms/rejects, renames, hard-negatives), not
    /// just deliberate calibration examples; reading the whole file mixes
    /// a small clean teaching set with unrelated historical noise.
    pub only_crops:     HashSet<String>,
}

pub enum RetrainMsg {
    Stage(String),
    Log(String),
    Error(String),
    Done(String), // summary
}

#[derive(Deserialize)]
struct LabelRow {
    crop: String,
    family: String,
    #[serde(default)]
    source: String,
    /// Companion mask filename (same dir as `crop`), reprojected into the
    /// crop's own window — empty for old rows (pre-mask) or hard-negative
    /// stamps (a stamp already IS the precise example, no shape to encode).
    #[serde(default)]
    mask: String,
}

pub fn spawn_retrain(cfg: RetrainCfg, tx: mpsc::Sender<RetrainMsg>, cancel: Arc<AtomicBool>) {
    std::thread::spawn(move || match retrain(&cfg, &tx, &cancel) {
        Ok(summary) => { let _ = tx.send(RetrainMsg::Done(summary)); }
        Err(e) => { let _ = tx.send(RetrainMsg::Error(e)); }
    });
}

fn read_labels(path: &Path) -> Result<Vec<LabelRow>, String> {
    let txt = std::fs::read_to_string(path)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for line in txt.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(r) = serde_json::from_str::<LabelRow>(line) {
            out.push(r);
        }
    }
    Ok(out)
}

/// The norm every real DINO patch feature has at INFERENCE time: `dino.rs`
/// concatenates two individually unit-normalised layer halves, so ‖x‖² ≈ 2
/// (asserted directly in `dino.rs`'s own test). Training rows must be scaled to
/// THIS, not to 1.
///
/// Why it matters: `fewshot.rs::predict` consumes features raw, at ‖x‖ = √2,
/// while these helpers used to rescale every training row to ‖x‖ = 1. A head
/// fitted against unit-norm rows and then fed √2-norm ones has its whole `w·x`
/// term inflated by √2 while the intercept — which carries the "healthy is the
/// common class" prior — stays put, so the data term overpowers the prior and
/// the head over-fires. Measured on held-out ground truth
/// (`1Help/eval/norm_mismatch.py`), train-unit → infer-raw costs ~35% of IoU
/// (0.181 vs 0.280) with recall inflated 0.885 → 0.957 and precision collapsing.
/// Either scale works as long as BOTH sides agree (unit→unit scored 0.274,
/// raw→raw 0.280); √2 is the correct choice here because the base head shipped
/// by `1Help/eval/export_fewshot_head.py` is itself fitted on raw √2 features,
/// so warm-starting from it only stays coherent at that scale.
const FEATURE_NORM: f32 = std::f32::consts::SQRT_2;

/// Rescales a pooled vector to `FEATURE_NORM` — averaging shrinks the norm
/// (the mean of N vectors is shorter than the vectors themselves), so a pooled
/// row would otherwise sit at a different scale than the per-patch rows and the
/// inference-time features alike.
fn rescale_to_feature_norm(m: &mut [f32]) {
    let nrm = m.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    let k = FEATURE_NORM / nrm;
    for x in m.iter_mut() {
        *x *= k;
    }
}

/// Mean DINO feature over a crop, rescaled to the inference-time patch norm
/// (see `FEATURE_NORM`).
fn mean_feature(feat: &[f32], grid: usize, dim: usize) -> Vec<f32> {
    let n = grid * grid;
    let mut m = vec![0f32; dim];
    for p in 0..n {
        let fp = &feat[p * dim..p * dim + dim];
        for d in 0..dim {
            m[d] += fp[d];
        }
    }
    for d in 0..dim {
        m[d] /= n as f32;
    }
    rescale_to_feature_norm(&mut m);
    m
}

/// Same as `mean_feature`, but only averages patches whose corresponding
/// region in `mask_img` (same win×win size as the crop the features were
/// extracted from) is MAJORITY-masked — the whole point of persisting a
/// mask alongside the crop: a precisely painted/wand-filled example should
/// train on the anomaly itself, not the square of context around it. Falls
/// back to the unmasked `mean_feature` if the mask selects zero patches
/// (degenerate/misaligned mask) rather than ever averaging an empty set.
fn mean_feature_masked(feat: &[f32], grid: usize, dim: usize, mask_img: &image::GrayImage) -> Vec<f32> {
    let (mw, mh) = (mask_img.width() as usize, mask_img.height() as usize);
    if mw == 0 || mh == 0 {
        return mean_feature(feat, grid, dim);
    }
    let mut m = vec![0f32; dim];
    let mut n_used = 0usize;
    for py in 0..grid {
        let y0 = py * mh / grid;
        let y1 = ((py + 1) * mh / grid).max(y0 + 1).min(mh);
        for px in 0..grid {
            let x0 = px * mw / grid;
            let x1 = ((px + 1) * mw / grid).max(x0 + 1).min(mw);
            let (mut on, mut tot) = (0usize, 0usize);
            for y in y0..y1 {
                for x in x0..x1 {
                    tot += 1;
                    if mask_img.get_pixel(x as u32, y as u32).0[0] > 128 {
                        on += 1;
                    }
                }
            }
            if tot > 0 && on * 2 >= tot {
                let p = py * grid + px;
                let fp = &feat[p * dim..p * dim + dim];
                for d in 0..dim {
                    m[d] += fp[d];
                }
                n_used += 1;
            }
        }
    }
    if n_used == 0 {
        return mean_feature(feat, grid, dim);
    }
    for d in 0..dim {
        m[d] /= n_used as f32;
    }
    rescale_to_feature_norm(&mut m);
    m
}

/// Loads a crop's DINO mean-feature, using mask-aware pooling when
/// `row.mask` names a real, loadable mask image alongside `row.crop` —
/// falls back to whole-crop `mean_feature` for old rows (no mask), hard-neg
/// stamps (mask deliberately empty), or a missing/corrupt mask file.
fn crop_feature(row: &LabelRow, crops_dir: &Path, feat: &[f32], grid: usize, dim: usize) -> Vec<f32> {
    if !row.mask.is_empty() {
        if let Ok(mi) = image::open(crops_dir.join(&row.mask)) {
            return mean_feature_masked(feat, grid, dim, &mi.to_luma8());
        }
    }
    mean_feature(feat, grid, dim)
}

/// Downsamples pixel-level mask validity to `grid×grid` by majority vote
/// per cell (same bucketing as `mean_feature_masked`), returning the FLAT
/// patch indices that pass — the patch-level counterpart to that function's
/// mean-pooling, used by `crop_patch_rows` below.
fn valid_patch_indices(grid: usize, mask_img: &image::GrayImage) -> Vec<usize> {
    let (mw, mh) = (mask_img.width() as usize, mask_img.height() as usize);
    let mut out = Vec::new();
    if mw == 0 || mh == 0 {
        return out;
    }
    for py in 0..grid {
        let y0 = py * mh / grid;
        let y1 = ((py + 1) * mh / grid).max(y0 + 1).min(mh);
        for px in 0..grid {
            let x0 = px * mw / grid;
            let x1 = ((px + 1) * mw / grid).max(x0 + 1).min(mw);
            let (mut on, mut tot) = (0usize, 0usize);
            for y in y0..y1 {
                for x in x0..x1 {
                    tot += 1;
                    if mask_img.get_pixel(x as u32, y as u32).0[0] > 128 {
                        on += 1;
                    }
                }
            }
            if tot > 0 && on * 2 >= tot {
                out.push(py * grid + px);
            }
        }
    }
    out
}

/// Per-patch training rows for one curated crop — passed through RAW, exactly
/// as `fewshot.rs::predict` will see them at inference.
///
/// History worth keeping, because this line has now been wrong in both
/// directions. Originally raw; then changed to per-patch L2-normalisation to
/// fix a real bug (raw patch rows were being mixed with unit-norm
/// `mean_feature` fallback rows in the SAME optimisation, so per-example
/// gradient magnitude tracked incidental feature scale rather than the intended
/// per-crop weight). That diagnosis was right but the remedy standardised on
/// the WRONG target: it made training rows unit-norm while inference stayed at
/// ‖x‖ = √2 (see `FEATURE_NORM`), trading an internal inconsistency for a
/// train/deploy one that measurably costs ~35% of IoU. The fix for the original
/// bug is to put every row on ONE scale — and that scale has to be the
/// inference scale. So patches stay raw (they already have ‖x‖ ≈ √2) and the
/// pooled fallbacks are rescaled up to meet them. Evenly
/// strided and capped at `cap` rows so one curation event can't contribute
/// hundreds of near-duplicate, spatially adjacent rows (see
/// `RetrainCfg::max_patches_per_crop`'s doc comment). Falls back to a
/// single mean-pooled row for legacy rows with no mask (nothing to select
/// a subset from).
fn crop_patch_rows(row: &LabelRow, crops_dir: &Path, feat: &[f32], grid: usize, dim: usize, cap: usize) -> Vec<Vec<f32>> {
    if !row.mask.is_empty() {
        if let Ok(mi) = image::open(crops_dir.join(&row.mask)) {
            let idxs = valid_patch_indices(grid, &mi.to_luma8());
            if !idxs.is_empty() {
                let cap = cap.max(1);
                let stride = idxs.len().div_ceil(cap).max(1);
                return idxs.iter().step_by(stride).take(cap)
                    .map(|&p| feat[p * dim..p * dim + dim].to_vec())
                    .collect();
            }
        }
    }
    vec![mean_feature(feat, grid, dim)]
}

/// Reads a base training set written in the `retrain_dump.bin` format
/// (see `write_training_dump`) and returns a CLASS-BALANCED subsample of at
/// most `target` rows, restricted to classes the head actually has.
///
/// Balanced rather than uniform on purpose: the ground-truth pool is ~80%
/// healthy, so a uniform draw would faithfully preserve the healthy manifold
/// and lose exactly the defect classes the base set exists to protect. Each
/// class gets an equal quota; whatever a small class can't fill is handed back
/// to the classes that have room, so the target is still met.
///
/// Rows come back at weight 1.0. That is what the measurement used
/// (`1Help/eval/base_size_curve.py`), and the balanced quota already equalises
/// the classes, so no extra per-class factor is applied on top.
fn read_base_set(
    path: &Path,
    target: usize,
    head_classes: &[i32],
) -> Result<Vec<(Vec<f32>, i32, f32)>, String> {
    use std::io::Read;
    let f = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut r = std::io::BufReader::new(f);
    let mut hdr = [0u8; 16];
    r.read_exact(&mut hdr).map_err(|e| format!("read base header: {e}"))?;
    let magic = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
    if magic != 0x4C52_4431 {
        return Err(format!("{} is not a base set (magic 0x{magic:08X})", path.display()));
    }
    let n_rows = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
    let n_cls = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let mut cls_buf = vec![0u8; n_cls * 4];
    r.read_exact(&mut cls_buf).map_err(|e| format!("read base classes: {e}"))?;

    // Pass 1: class of every row, without keeping the features — the file is
    // hundreds of MB and only `target` rows survive the quota below.
    let row_bytes = 8 + dim * 4;
    let mut classes = Vec::with_capacity(n_rows);
    let mut buf = vec![0u8; row_bytes];
    for _ in 0..n_rows {
        r.read_exact(&mut buf).map_err(|e| format!("scan base rows: {e}"))?;
        classes.push(i32::from_le_bytes(buf[0..4].try_into().unwrap()));
    }

    let mut by_class: HashMap<i32, Vec<usize>> = HashMap::new();
    for (i, &c) in classes.iter().enumerate() {
        if head_classes.contains(&c) {
            by_class.entry(c).or_default().push(i);
        }
    }
    if by_class.is_empty() {
        return Err(format!(
            "base set has no rows for any class in this head (base classes {:?}, head {:?})",
            classes.iter().copied().collect::<HashSet<_>>(), head_classes
        ));
    }

    // Equal quota, redistributing what small classes cannot fill.
    let k = by_class.len();
    let mut quota: HashMap<i32, usize> = by_class.iter()
        .map(|(&c, v)| (c, (target / k).min(v.len()))).collect();
    let mut leftover = target.saturating_sub(quota.values().sum::<usize>());
    while leftover > 0 {
        let growable: Vec<i32> = by_class.iter()
            .filter(|(c, v)| v.len() > quota[c]).map(|(&c, _)| c).collect();
        if growable.is_empty() {
            break;
        }
        let share = (leftover / growable.len()).max(1);
        for c in growable {
            let room = by_class[&c].len() - quota[&c];
            let take = share.min(room).min(leftover);
            *quota.get_mut(&c).unwrap() += take;
            leftover -= take;
            if leftover == 0 {
                break;
            }
        }
    }

    // Evenly strided pick per class — deterministic, and spreads the sample
    // across the file rather than taking one contiguous block (which would be
    // one region of one leaf, since rows are written in tile order).
    let mut wanted: Vec<(usize, i32)> = Vec::new();
    for (&c, idxs) in &by_class {
        let want = quota[&c];
        if want == 0 {
            continue;
        }
        let stride = (idxs.len() / want).max(1);
        for &i in idxs.iter().step_by(stride).take(want) {
            wanted.push((i, c));
        }
    }
    wanted.sort_unstable_by_key(|&(i, _)| i);

    // Pass 2: re-read, keeping only the chosen rows.
    let f = std::fs::File::open(path).map_err(|e| format!("reopen {}: {e}", path.display()))?;
    let mut r = std::io::BufReader::new(f);
    let mut skip = vec![0u8; 16 + n_cls * 4];
    r.read_exact(&mut skip).map_err(|e| format!("reskip base header: {e}"))?;
    let mut out = Vec::with_capacity(wanted.len());
    let mut cursor = 0usize;
    for (idx, cls) in wanted {
        while cursor < idx {
            r.read_exact(&mut buf).map_err(|e| format!("seek base rows: {e}"))?;
            cursor += 1;
        }
        r.read_exact(&mut buf).map_err(|e| format!("read base row: {e}"))?;
        cursor += 1;
        let feat: Vec<f32> = (0..dim)
            .map(|d| f32::from_le_bytes(buf[8 + d * 4..12 + d * 4].try_into().unwrap()))
            .collect();
        out.push((feat, cls, 1.0f32));
    }
    Ok(out)
}

/// Snapshot of one head's per-class scale, for the diagnostic dump.
fn head_snapshot(head: &FewShotHead) -> serde_json::Value {
    let rows: Vec<serde_json::Value> = head.classes.iter().enumerate().map(|(k, &c)| {
        let norm = head.coef[k].iter().map(|v| v * v).sum::<f32>().sqrt();
        serde_json::json!({
            "class": c,
            "name": head.families.get(&c.to_string()).cloned()
                .unwrap_or_else(|| if c == 0 { "Healthy".into() } else { c.to_string() }),
            "coef_norm": norm,
            "intercept": head.intercept[k],
        })
    }).collect();
    serde_json::Value::Array(rows)
}

/// Writes the exact training matrix to `retrain_dump.bin` so an independent
/// solver can be fitted on byte-identical data (see `RetrainCfg::dump_dir`).
///
/// Format — manual little-endian, mirroring `bank.rs`'s `CoresetBank` style
/// rather than adding a serialization dependency:
/// ```text
///   u32  magic 0x4C52_4431 ("LRD1")
///   u32  n_rows
///   u32  dim
///   u32  n_classes
///   [i32; n_classes]        class ids, in the head's own row order
///   per row:  i32 class, f32 weight, [f32; dim] feature
/// ```
fn write_training_dump(
    dir: &Path,
    samples: &[(Vec<f32>, i32, f32)],
    classes: &[i32],
    dim: usize,
) -> Result<u64, String> {
    use std::io::Write;
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let path = dir.join("retrain_dump.bin");
    let f = std::fs::File::create(&path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let mut w = std::io::BufWriter::new(f);
    let mut hdr = Vec::with_capacity(16 + classes.len() * 4);
    hdr.extend_from_slice(&0x4C52_4431u32.to_le_bytes());
    hdr.extend_from_slice(&(samples.len() as u32).to_le_bytes());
    hdr.extend_from_slice(&(dim as u32).to_le_bytes());
    hdr.extend_from_slice(&(classes.len() as u32).to_le_bytes());
    for &c in classes {
        hdr.extend_from_slice(&c.to_le_bytes());
    }
    w.write_all(&hdr).map_err(|e| format!("write dump header: {e}"))?;
    let mut row = Vec::with_capacity(8 + dim * 4);
    for (feat, cls, wt) in samples {
        row.clear();
        row.extend_from_slice(&cls.to_le_bytes());
        row.extend_from_slice(&wt.to_le_bytes());
        for v in feat {
            row.extend_from_slice(&v.to_le_bytes());
        }
        w.write_all(&row).map_err(|e| format!("write dump row: {e}"))?;
    }
    w.flush().map_err(|e| format!("flush dump: {e}"))?;
    std::fs::metadata(&path).map(|m| m.len()).map_err(|e| format!("stat dump: {e}"))
}

/// Multinomial softmax cross-entropy + zero-centered L2, as an `argmin`
/// `CostFunction`/`Gradient` pair — the same math the old hand-rolled
/// gradient-descent loop computed, just handed to a real quasi-Newton
/// solver instead of a fixed-epoch/fixed-`lr` loop (see `retrain`'s doc
/// comment). `Param` is a flat `Vec<f64>`: `coef` rows concatenated
/// (`[k*dim..(k+1)*dim]` per class `k`), followed by `kk` intercepts —
/// f64 throughout, not f32, since a line search evaluates small cost/
/// gradient differences near the optimum where f32 rounding could produce
/// spurious non-descent directions. `trained` marks which classes have any
/// samples this run — untouched classes' gradient (both coef AND
/// intercept) is forced to exactly zero every call, which keeps their
/// entire L-BFGS history (and therefore every future step) exactly zero in
/// those coordinates: a true freeze, not just "no L2 pull," since the
/// softmax cross term would otherwise still nudge an untouched class via
/// every OTHER class's samples.
struct HeadObjective<'a> {
    samples: &'a [(Vec<f32>, i32, f32)],
    class_row: &'a HashMap<i32, usize>,
    trained: &'a [bool],
    kk: usize,
    dim: usize,
    n_norm: f64,
    l2_reg: f64,
    /// The coefficient block the L2 penalty pulls TOWARD, already scaled by
    /// `RetrainCfg::anchor` — all zeros when anchoring is off, which is exactly
    /// ordinary zero-centered L2, so one code path serves both.
    anchor_w: &'a [f64],
}

impl HeadObjective<'_> {
    fn logits(&self, p: &[f64], x: &[f32]) -> Vec<f64> {
        (0..self.kk).map(|k| {
            let wk = &p[k * self.dim..(k + 1) * self.dim];
            let mut s = p[self.kk * self.dim + k];
            for d in 0..self.dim {
                s += wk[d] * x[d] as f64;
            }
            s
        }).collect()
    }
}

impl CostFunction for HeadObjective<'_> {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, p: &Vec<f64>) -> Result<f64, argmin::core::Error> {
        // Each sample's contribution is independent, summed at the end —
        // embarrassingly parallel. Line search evaluates this (and
        // `gradient` below) repeatedly per outer L-BFGS iteration, and
        // patch-level training can put tens of thousands of rows through
        // it, so single-threaded was a real, unnecessary slowdown once
        // `max_patches_per_crop` started expanding crops into many rows.
        let loss: f64 = self.samples.par_iter().map(|(x, cls, w)| {
            let row = self.class_row[cls];
            let logit = self.logits(p, x);
            let m = logit.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = logit.iter().map(|&l| (l - m).exp()).sum();
            -(*w as f64) * (logit[row] - m - sum.ln())
        }).sum();
        let mut loss = loss / self.n_norm;
        // ‖W - W₀‖² where W₀ is the anchor (all-zero when anchoring is off, so
        // this is plain zero-centered L2 in that case).
        let mut reg = 0f64;
        for (i, v) in p[..self.kk * self.dim].iter().enumerate() {
            let d = v - self.anchor_w[i];
            reg += d * d;
        }
        loss += 0.5 * self.l2_reg * reg;
        Ok(loss)
    }
}

impl Gradient for HeadObjective<'_> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;
    fn gradient(&self, p: &Vec<f64>) -> Result<Vec<f64>, argmin::core::Error> {
        let len = self.kk * self.dim + self.kk;
        let mut g: Vec<f64> = self.samples.par_iter()
            .fold(
                || vec![0f64; len],
                |mut acc, (x, cls, w)| {
                    let row = self.class_row[cls];
                    let logit = self.logits(p, x);
                    let m = logit.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let mut probs: Vec<f64> = logit.iter().map(|&l| (l - m).exp()).collect();
                    let sum: f64 = probs.iter().sum();
                    for pr in probs.iter_mut() {
                        *pr /= sum;
                    }
                    for k in 0..self.kk {
                        let gk = (*w as f64) * (probs[k] - if k == row { 1.0 } else { 0.0 });
                        acc[self.kk * self.dim + k] += gk;
                        for d in 0..self.dim {
                            acc[k * self.dim + d] += gk * x[d] as f64;
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![0f64; len],
                |mut a, b| {
                    for i in 0..len {
                        a[i] += b[i];
                    }
                    a
                },
            );
        for v in g.iter_mut() {
            *v /= self.n_norm;
        }
        for k in 0..self.kk {
            for d in 0..self.dim {
                let i = k * self.dim + d;
                g[i] += self.l2_reg * (p[i] - self.anchor_w[i]);
            }
        }
        for k in 0..self.kk {
            if !self.trained[k] {
                for d in 0..self.dim {
                    g[k * self.dim + d] = 0.0;
                }
                g[self.kk * self.dim + k] = 0.0;
            }
        }

        // ── keep frozen classes actually frozen, in the softmax's own terms ──
        // A softmax only reads DIFFERENCES between logits, so adding the same
        // constant to every intercept changes nothing — and correspondingly the
        // intercept gradients sum to exactly zero over all classes
        // (sum_k (p_k - y_k) = 1 - 1 = 0 per sample). Zeroing a frozen class's
        // entry above BREAKS that sum, leaving a spurious net push on every
        // remaining class. They then drift together while the frozen one cannot
        // follow, which moves it relative to everything else — the one thing
        // freezing exists to prevent.
        //
        // Observed doing exactly that: with Skeletonizer frozen, the four
        // trained intercepts all rose ~+18 while it stayed at -3.54, taking its
        // deficit from ~2.7-8.5 to ~21.4-22.8. With ‖w‖ 6.7 and ‖x‖ √2 it can
        // swing at most ±9.4, so the class became unable to win any patch. That
        // run also burned all 2000 iterations, because this flat direction is
        // slow to converge along as well.
        //
        // Re-centering the TRAINED intercept gradients to sum to zero removes
        // that collective direction while leaving every relative adjustment
        // among trained classes untouched. A no-op when nothing is frozen,
        // since the sum is already zero there.
        if self.trained.iter().any(|&t| !t) {
            let idx: Vec<usize> = (0..self.kk).filter(|&k| self.trained[k]).collect();
            if !idx.is_empty() {
                let base = self.kk * self.dim;
                let mean = idx.iter().map(|&k| g[base + k]).sum::<f64>() / idx.len() as f64;
                for &k in &idx {
                    g[base + k] -= mean;
                }
            }
        }
        Ok(g)
    }
}

fn retrain(
    cfg: &RetrainCfg,
    tx: &mpsc::Sender<RetrainMsg>,
    cancel: &AtomicBool,
) -> Result<String, String> {
    let _ = tx.send(RetrainMsg::Stage("Loading head + DINO".into()));
    let mut head = FewShotHead::load(&cfg.head_path)?;
    let dim = head.dim;
    let mut dino = DinoExtractor::load(&cfg.dino_model, head.infer_resolution)?;

    // Validation gate, BEFORE score: the CURRENT (about-to-be-replaced)
    // head's own false-positive rate on independent known-healthy tiles —
    // computed here, before anything below mutates `head`, so this is
    // exactly what's deployed right now, not an artifact of new classes
    // already having been pushed onto it. Purely informational (never
    // blocks) — see RetrainCfg::validate_healthy_dir's doc comment. The
    // tile SAMPLE is picked once and reused for the after-score below —
    // scoring two independently-shuffled subsets of a large pool would
    // compare noise, not the model (a real early version of this did
    // exactly that).
    let (old_fpr, validate_tiles) = match &cfg.validate_healthy_dir {
        Some(dir) => match pick_validation_tiles(dir) {
            Ok(tiles) => {
                match compute_fpr_for(&head, &mut dino, &tiles, cfg.validate_tau, |done, total| {
                    let _ = tx.send(RetrainMsg::Stage(format!("Validating (before): {done}/{total} tiles")));
                }) {
                    Ok(fpr) => {
                        let _ = tx.send(RetrainMsg::Log(format!(
                            "validation (before): {:.2}% of healthy patches wrongly fire (tau={}, {} tiles)",
                            fpr * 100.0, cfg.validate_tau, tiles.len()
                        )));
                        (Some(fpr), Some(tiles))
                    }
                    Err(e) => {
                        let _ = tx.send(RetrainMsg::Log(format!("validation skipped: {e}")));
                        (None, None)
                    }
                }
            }
            Err(e) => {
                let _ = tx.send(RetrainMsg::Log(format!("validation skipped: {e}")));
                (None, None)
            }
        },
        None => (None, None),
    };

    // Log exactly which file is being warm-started from and its starting
    // coefficient norms, BEFORE any training happens. Regularization is now
    // zero-centered (not anchored to these values, see RetrainCfg::l2_reg),
    // but this is still worth knowing: warm-starting from an already-large
    // point still shapes convergence speed/direction, even without an
    // anchor pulling toward it. Make the starting point undeniable instead
    // of something to remember correctly (this exact ambiguity cost real
    // debugging time earlier).
    {
        let mut start_norms: Vec<(i32, f32)> = head.classes.iter().zip(head.coef.iter())
            .map(|(&c, row)| (c, row.iter().map(|v| v * v).sum::<f32>().sqrt()))
            .collect();
        start_norms.sort_by_key(|&(c, _)| c);
        let start_str: String = start_norms.iter()
            .map(|&(c, n)| {
                let name = head.families.get(&c.to_string()).cloned()
                    .unwrap_or_else(|| if c == 0 { "Healthy".to_string() } else { c.to_string() });
                if n == 0.0 { format!("{name}=0 (new)") } else { format!("{name}={n:.2}") }
            })
            .collect::<Vec<_>>().join(", ");
        let _ = tx.send(RetrainMsg::Log(format!(
            "starting from {} — coef norms: {start_str}", cfg.head_path.display()
        )));
    }

    // Retraining a head onto itself compounds every round on top of the last
    // instead of branching from a known-good baseline, and destroys the only
    // copy you could roll back to. `save_head_backed_up` keeps one backup, but
    // that is a single step of history, not a baseline.
    if cfg.head_path == cfg.out_path {
        let _ = tx.send(RetrainMsg::Log(
            "WARNING: retraining a head ONTO ITSELF (input and output are the same \
             file). Each round then builds on the previous one's drift and there is \
             no baseline to return to. Prefer retraining from the original head each \
             time.".into(),
        ));
    }

    let rows = read_labels(&cfg.curations_dir.join("labels.jsonl"))?;
    if rows.is_empty() {
        return Err("no curation labels found".into());
    }

    // name -> class id, seeded from the head's families; new names get new ids.
    // Keyed by a trimmed-lowercased form of the name — defense-in-depth so a
    // stray case/whitespace difference ("hole" vs "Hole") can't fork a
    // duplicate class here even if it slipped past the Pipeline tab's own
    // canonical-name picker.
    let norm = |s: &str| s.trim().to_lowercase();
    let mut name2class: HashMap<String, i32> = HashMap::new();
    for (idx, name) in &head.families {
        if let Ok(i) = idx.parse::<i32>() {
            name2class.insert(norm(name), i);
        }
    }
    let healthy_class = 0i32;
    let mut next_id = head.classes.iter().copied().max().unwrap_or(0) + 1;
    let mut n_new = 0;

    // ── pass 1: assign each row a class id (creates new classes as a side
    // effect — pushes a fresh all-zero coef/intercept row), WITHOUT
    // touching images. Needed so crop-level class_counts/weights, `kk`, and
    // the pre-training norm snapshot (for the new-class sanity check below)
    // are all known BEFORE generating the — possibly much larger — set of
    // per-patch training rows in pass 2. ──
    let mut row_class: Vec<i32> = Vec::with_capacity(rows.len());
    for r in &rows {
        let cls = if r.source == "reject" || r.family == "rejected" {
            healthy_class
        } else if let Some(&c) = name2class.get(&norm(&r.family)) {
            c
        } else {
            let c = next_id;
            next_id += 1;
            n_new += 1;
            head.classes.push(c);
            head.intercept.push(0.0);
            head.coef.push(vec![0.0; dim]);
            head.families.insert(c.to_string(), r.family.clone());
            name2class.insert(norm(&r.family), c);
            c
        };
        row_class.push(cls);
    }
    let kk = head.classes.len();
    let class_row: HashMap<i32, usize> =
        head.classes.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    // Snapshot BEFORE training touches anything — a class at exactly 0.0
    // here is either genuinely brand-new (just pushed above) or was
    // already at 0 in the loaded head; both cases are "no established
    // signal to compare against" for the new-class sanity check below.
    let start_norm_by_row: Vec<f32> = head.coef.iter()
        .map(|row| row.iter().map(|v| v * v).sum::<f32>().sqrt())
        .collect();

    // ── class-balanced sample weights, computed on CROPS (curation
    // events), never on the expanded per-patch samples pass 2 produces —
    // so `class_counts`/MIN_CLASS_COUNT keep meaning "how many times did a
    // human curate this," not "how many patches happened to come out of
    // however many crops." ──
    // Plain per-sample averaging lets whichever class has the most curated
    // examples dominate the gradient outright (a real failure hit in
    // practice: 1000 "sucker" crops vs. 30 "hole" crops trained a head that
    // called EVERY anomaly "sucker"). Weight each crop by n_crops /
    // (n_classes * count[class]) — sklearn's "balanced" scheme — so every
    // class contributes the same TOTAL weight to the gradient regardless of
    // how lopsided the curation counts are.
    let mut class_counts: HashMap<i32, usize> = HashMap::new();
    for &cls in &row_class {
        *class_counts.entry(cls).or_insert(0) += 1;
    }
    let n_classes = class_counts.len() as f32;
    let n_crops = rows.len() as f32;
    let mut counts_log: Vec<(i32, usize)> = class_counts.iter().map(|(&c, &n)| (c, n)).collect();
    counts_log.sort_by_key(|&(c, _)| c);
    let counts_str: String = counts_log.iter()
        .map(|&(c, n)| {
            // Class 0 is always the healthy/rejected bucket (see
            // `healthy_class` above) — name it explicitly rather than
            // falling back to the bare numeric id, which otherwise reads as
            // an unexplained mystery class in this log line.
            let name = if c == healthy_class {
                "Healthy".to_string()
            } else {
                head.families.get(&c.to_string()).cloned().unwrap_or_else(|| c.to_string())
            };
            format!("{name}={n}")
        })
        .collect::<Vec<_>>().join(", ");
    let n_hardneg_crops = rows.iter().filter(|r| r.crop.contains("_hardneg_")).count();
    let _ = tx.send(RetrainMsg::Log(format!("class balance: {counts_str}  ({n_hardneg_crops} hard negatives)")));
    // A class with a handful of examples (real case hit: "Holes"=2) still
    // gets the SAME total weight budget (n_crops/n_classes) as a 900-example
    // class under plain "balanced" weighting — concentrated into just 1-2
    // examples, that's a 100x+ per-example weight multiplier over a
    // well-populated class's own examples. Confirmed in a real head:
    // Healthy(623)/Sucker(916)/Nekrosis(398)/Holes(~2) all converged to
    // nearly IDENTICAL coef norms (6.4-6.9) regardless of how much real
    // evidence backed each one. Flooring the count this weight is computed
    // from caps how extreme that per-example leverage can get without
    // changing anything for classes that already have reasonable support.
    const MIN_CLASS_COUNT: f32 = 15.0;
    for (&cls, &cnt) in &class_counts {
        if (cnt as f32) < MIN_CLASS_COUNT {
            let name = if cls == healthy_class { "Healthy".to_string() }
                else { head.families.get(&cls.to_string()).cloned().unwrap_or_else(|| cls.to_string()) };
            let _ = tx.send(RetrainMsg::Log(format!(
                "WARNING: '{name}' has only {cnt} example(s) — its weight is capped, but predictions \
                 for it will still be unreliable until you curate more."
            )));
        }
    }
    let class_weight = |cls: i32| -> f32 {
        let eff_count = (class_counts[&cls] as f32).max(MIN_CLASS_COUNT);
        n_crops / (n_classes * eff_count)
    };
    // Hard negatives are deliberately hand-picked, targeted corrective
    // examples — exactly the kind that should get OUTSIZED influence, not
    // be absorbed into a fixed budget alongside generic curation volume.
    const HARDNEG_BOOST: f32 = 4.0;
    // Only classes with ANY samples this run get trained — see
    // `HeadObjective`'s doc comment for why the rest are fully FROZEN
    // (not just left unregularized) instead of decaying toward zero.
    let mut trained_this_run = vec![false; kk];
    for &cls in class_counts.keys() {
        if let Some(&row) = class_row.get(&cls) {
            trained_this_run[row] = true;
        }
    }

    // ── pass 2: DINO + per-patch feature extraction, one bounded weight
    // per curated crop divided across however many patch rows it
    // contributes (see RetrainCfg::max_patches_per_crop's doc comment) ──
    let _ = tx.send(RetrainMsg::Stage("Extracting features from curations".into()));
    let crops_dir = cfg.curations_dir.join("labels");
    let mut samples: Vec<(Vec<f32>, i32, f32)> = Vec::new();
    let mut n_crops_used = 0usize;
    for (k, (r, &cls)) in rows.iter().zip(&row_class).enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let path = crops_dir.join(&r.crop);
        let img = match image::open(&path) {
            Ok(i) => i.to_rgb8(),
            Err(e) => {
                let _ = tx.send(RetrainMsg::Log(format!("skip {}: {e}", r.crop)));
                continue;
            }
        };
        let f = dino.features(&img)?;
        if f.dim != dim {
            return Err(format!("crop feature dim {} != head dim {dim}", f.dim));
        }
        let is_hardneg = r.crop.contains("_hardneg_");
        let crop_w = class_weight(cls) * if is_hardneg { HARDNEG_BOOST } else { 1.0 };
        let patch_rows = crop_patch_rows(r, &crops_dir, &f.feat, f.grid, dim, cfg.max_patches_per_crop);
        let per_row_w = crop_w / patch_rows.len() as f32;
        for pf in patch_rows {
            samples.push((pf, cls, per_row_w));
        }
        n_crops_used += 1;
        if (k + 1) % 25 == 0 {
            let _ = tx.send(RetrainMsg::Log(format!("features {}/{}", k + 1, rows.len())));
        }
    }
    if samples.is_empty() {
        return Err("no usable curation crops".into());
    }
    let _ = tx.send(RetrainMsg::Log(format!(
        "{n_crops_used} crop(s) -> {} training row(s) (cap {}/crop)",
        samples.len(), cfg.max_patches_per_crop
    )));

    // ── anti-forgetting: mix in the ORIGINAL training rows ──
    // Curations alone cannot fine-tune this head — zero-centered L2 plus a
    // small row count re-derives the solution from scratch and discards
    // whatever the base head knew (see RetrainCfg::base_set).
    let n_curation_rows = samples.len();
    if let Some(bp) = &cfg.base_set {
        let _ = tx.send(RetrainMsg::Stage("Loading base training set".into()));
        match read_base_set(bp, cfg.base_rows, &head.classes) {
            Ok(base) => {
                let mut per_class: HashMap<i32, usize> = HashMap::new();
                for (_, c, _) in &base {
                    *per_class.entry(*c).or_insert(0) += 1;
                }
                let mut counts: Vec<(i32, usize)> = per_class.into_iter().collect();
                counts.sort_by_key(|&(c, _)| c);
                let summary: String = counts.iter()
                    .map(|&(c, n)| {
                        let name = head.families.get(&c.to_string()).cloned()
                            .unwrap_or_else(|| if c == 0 { "Healthy".into() } else { c.to_string() });
                        format!("{name}={n}")
                    })
                    .collect::<Vec<_>>().join(", ");
                let _ = tx.send(RetrainMsg::Log(format!(
                    "base set: +{} row(s) from {} ({summary})", base.len(), bp.display()
                )));
                // A class carried ONLY by the base pool still has evidence this
                // run, so it must not be frozen — freezing it would pin it at a
                // value fitted under a different objective while every other
                // class moves, which is how one class ends up dominating.
                for (_, c, _) in &base {
                    if let Some(&row) = class_row.get(c) {
                        trained_this_run[row] = true;
                    }
                }
                samples.extend(base);
            }
            Err(e) => {
                // Loud, and NOT fatal: a retrain without the base pool still
                // produces a head, it just produces the forgetting failure —
                // so say exactly that rather than dying or staying quiet.
                let _ = tx.send(RetrainMsg::Log(format!(
                    "WARNING: base set unusable ({e}) — continuing WITHOUT it. \
                     Expect the retrained head to forget everything outside these \
                     curations."
                )));
            }
        }
    } else {
        let _ = tx.send(RetrainMsg::Log(
            "NOTE: no base training set configured — this fits the curations ALONE, \
             which discards what the head learned from its original training data. \
             Set a base set to make retraining incremental.".into(),
        ));
    }
    let _ = tx.send(RetrainMsg::Log(format!(
        "training on {} row(s) total ({} curation + {} base)",
        samples.len(), n_curation_rows, samples.len() - n_curation_rows
    )));

    // Snapshot the pre-solve head while it's still untouched — the dump below
    // needs the exact warm-start point to reproduce this run's conditions.
    let before_snapshot = head_snapshot(&head);
    if let Some(dir) = &cfg.dump_dir {
        let _ = tx.send(RetrainMsg::Stage("Writing training dump".into()));
        match write_training_dump(dir, &samples, &head.classes, dim) {
            Ok(bytes) => {
                let _ = tx.send(RetrainMsg::Log(format!(
                    "training dump: {} row(s) x {dim} dims -> {}/retrain_dump.bin ({:.1} MB)",
                    samples.len(), dir.display(), bytes as f64 / (1024.0 * 1024.0)
                )));
            }
            Err(e) => { let _ = tx.send(RetrainMsg::Log(format!("training dump FAILED: {e}"))); }
        }
    }

    // ── L-BFGS fine-tune (see RetrainCfg's doc comments for why this
    // replaced fixed-epoch/fixed-lr gradient descent) ──
    let _ = tx.send(RetrainMsg::Stage("Fine-tuning head (L-BFGS)".into()));
    let mut init_param = vec![0f64; kk * dim + kk];
    for k in 0..kk {
        // Cold-start only classes actually being retrained this run — a
        // class with no samples this run is frozen at its existing value
        // regardless (see HeadObjective's doc comment), so zeroing it here
        // would delete it instead of leaving it untouched.
        if cfg.cold_start && trained_this_run[k] {
            continue;
        }
        for d in 0..dim {
            init_param[k * dim + d] = head.coef[k][d] as f64;
        }
        init_param[kk * dim + k] = head.intercept[k] as f64;
    }
    if cfg.cold_start {
        let _ = tx.send(RetrainMsg::Log(
            "cold start: zero-initializing every class with curated examples this run \
             (classes with no curated examples keep their existing weights, as usual)".into(),
        ));
    }

    // The point the L2 penalty pulls toward. Built from the head as it stands
    // AFTER pass 1, so a class created this run contributes all zeros and is
    // therefore regularized toward zero exactly as before — new classes stay
    // free to learn without a special case.
    //
    // Independent of `cold_start`: that only moves the STARTING point, and for
    // a strictly convex objective solved to convergence the starting point does
    // not affect the answer. The anchor changes the objective itself, which is
    // the only thing that can change where the solver lands.
    let anchor_w: Vec<f64> = if cfg.anchor > 0.0 {
        let a = cfg.anchor as f64;
        let mut v = vec![0f64; kk * dim];
        for k in 0..kk {
            for d in 0..dim {
                v[k * dim + d] = head.coef[k][d] as f64 * a;
            }
        }
        let n_anchored = (0..kk).filter(|&k| head.coef[k].iter().any(|&x| x != 0.0)).count();
        let _ = tx.send(RetrainMsg::Log(format!(
            "anchoring to the current head at strength {a} ({n_anchored}/{kk} class(es) have a \
             prior; the rest fall back to zero-centered L2)"
        )));
        v
    } else {
        vec![0f64; kk * dim]
    };
    // `cfg.l2_reg` is sklearn's C (inverse strength). The cost averages the data
    // term over crops, so the matching raw penalty is 1/(C * n_norm) — keeping
    // this derived rather than hard-coded is what stops the effective strength
    // from drifting with the curation count (see RetrainCfg::l2_reg).
    let n_norm = n_crops_used.max(1) as f64;
    let l2_coef = 1.0 / (cfg.l2_reg.max(1e-6) as f64 * n_norm);
    let _ = tx.send(RetrainMsg::Log(format!(
        "regularization: C={} over {n_norm:.0} crop(s) -> L2 coefficient {l2_coef:.3e}",
        cfg.l2_reg
    )));
    let objective = HeadObjective {
        samples: &samples,
        class_row: &class_row,
        trained: &trained_this_run,
        kk, dim,
        n_norm,
        l2_reg: l2_coef,
        anchor_w: &anchor_w,
    };
    let linesearch = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(linesearch, 10);
    let result = Executor::new(objective, solver)
        .configure(|state| state.param(init_param).max_iters(cfg.max_iters))
        .run()
        .map_err(|e| format!("L-BFGS solve failed: {e}"))?;
    let best = result.state().best_param.clone()
        .ok_or_else(|| "L-BFGS produced no solution".to_string())?;
    let n_iters = result.state().get_iter();
    let final_cost = result.state().get_best_cost();
    let _ = tx.send(RetrainMsg::Log(format!("L-BFGS: {n_iters} iteration(s), final cost {final_cost:.4}")));
    if n_iters >= cfg.max_iters {
        let _ = tx.send(RetrainMsg::Log(format!(
            "WARNING: L-BFGS stopped at the {} iteration ceiling instead of converging — \
             the head below is wherever the solver happened to be. Raise max_iters.",
            cfg.max_iters
        )));
    }
    for k in 0..kk {
        for d in 0..dim {
            head.coef[k][d] = best[k * dim + d] as f32;
        }
        head.intercept[k] = best[kk * dim + k] as f32;
    }

    // Everything an independent solver needs to reproduce this run's conditions
    // exactly and be compared against it (see RetrainCfg::dump_dir).
    if let Some(dir) = &cfg.dump_dir {
        let diag = serde_json::json!({
            "dim": dim,
            "n_classes": kk,
            "n_rows": samples.len(),
            "n_crops_used": n_crops_used,
            "class_ids": head.classes,
            "trained_this_run": trained_this_run,
            "cold_start": cfg.cold_start,
            "l2_reg": cfg.l2_reg,
            "max_iters": cfg.max_iters,
            "max_patches_per_crop": cfg.max_patches_per_crop,
            "n_norm": n_crops_used.max(1),
            "lbfgs_iters": n_iters,
            "lbfgs_final_cost": final_cost,
            "head_path": cfg.head_path.display().to_string(),
            "out_path": cfg.out_path.display().to_string(),
            "before": before_snapshot,
            "after": head_snapshot(&head),
        });
        let p = dir.join("retrain_diag.json");
        match serde_json::to_string_pretty(&diag)
            .map_err(|e| e.to_string())
            .and_then(|s| std::fs::write(&p, s).map_err(|e| e.to_string()))
        {
            Ok(()) => { let _ = tx.send(RetrainMsg::Log(format!("diagnostics -> {}", p.display()))); }
            Err(e) => { let _ = tx.send(RetrainMsg::Log(format!("diagnostics FAILED: {e}"))); }
        }
    }

    // ── write updated head ──
    let max_norm: f32 = head.coef.iter()
        .map(|row| row.iter().map(|v| v * v).sum::<f32>().sqrt())
        .fold(0f32, f32::max);
    let _ = tx.send(RetrainMsg::Log(format!("post-training max coef norm = {max_norm:.3}")));
    // New-class representation sanity check: a class that started at
    // norm==0.0 needed to climb from nothing this run — warn if it ends up
    // drastically weaker than its established siblings, the exact symptom
    // observed once (a new class's norm left at 1.6 against ~6.5-6.9 for
    // everyone else) before the freeze + per-class-trained gradient fixes
    // above. Purely a log line; L-BFGS convergence + the freeze above
    // should already prevent this in practice.
    {
        let established: Vec<f32> = (0..kk)
            .filter(|&k| start_norm_by_row[k] > 0.0)
            .map(|k| head.coef[k].iter().map(|v| v * v).sum::<f32>().sqrt())
            .collect();
        if !established.is_empty() {
            let mut sorted = established.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = sorted[sorted.len() / 2];
            for k in 0..kk {
                if start_norm_by_row[k] == 0.0 {
                    let final_norm = head.coef[k].iter().map(|v| v * v).sum::<f32>().sqrt();
                    if final_norm < 0.5 * median {
                        let c = head.classes[k];
                        let name = head.families.get(&c.to_string()).cloned().unwrap_or_else(|| c.to_string());
                        let _ = tx.send(RetrainMsg::Log(format!(
                            "WARNING: new class '{name}' ended at coef norm {final_norm:.2}, well below \
                             established classes' median {median:.2} — likely under-represented; \
                             curate more examples for it before trusting its predictions."
                        )));
                    }
                }
            }
        }
    }
    // Validation gate, AFTER: SAME tile sample as before (see
    // `validate_tiles`'s doc comment above), now against the trained head,
    // still purely informational.
    if let (Some(tiles), Some(old)) = (&validate_tiles, old_fpr) {
        match compute_fpr_for(&head, &mut dino, tiles, cfg.validate_tau, |done, total| {
            let _ = tx.send(RetrainMsg::Stage(format!("Validating (after): {done}/{total} tiles")));
        }) {
            Ok(new_fpr) => {
                let delta = (new_fpr - old) * 100.0;
                let arrow = if delta > 0.5 { "WORSE" } else if delta < -0.5 { "better" } else { "~same" };
                let _ = tx.send(RetrainMsg::Log(format!(
                    "validation (after): {:.2}% wrongly fire on healthy ({arrow}, {delta:+.2}pp vs. before)",
                    new_fpr * 100.0
                )));
            }
            Err(e) => { let _ = tx.send(RetrainMsg::Log(format!("validation (after) skipped: {e}"))); }
        }
    }
    head.onnx_parity = None; // weights changed; parity no longer the exported one
    head.save(&cfg.out_path)?;
    Ok(format!(
        "Updated head: {n_crops_used} crops ({} rows), {kk} classes (+{n_new} new) -> {}",
        samples.len(), cfg.out_path.display()
    ))
}

/// Rewrites every `labels.jsonl` row naming family `from` to `to` — used
/// when merging a duplicate class so already-curated labels follow the
/// merge instead of silently orphaning back onto the deleted class name on
/// the next retrain. Mirrors the existing rewrite-in-place pattern already
/// used for retracting a single persisted label.
pub fn rewrite_curated_family(curations_dir: &Path, from: &str, to: &str) -> Result<(), String> {
    let path = curations_dir.join("labels.jsonl");
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(format!("read {}: {e}", path.display())),
    };
    let needle = format!("\"family\":\"{from}\"");
    let replacement = format!("\"family\":\"{to}\"");
    let rewritten: String = text.lines()
        .map(|l| if l.contains(&needle) { l.replace(&needle, &replacement) } else { l.to_string() })
        .map(|l| format!("{l}\n"))
        .collect();
    std::fs::write(&path, rewritten).map_err(|e| format!("write {}: {e}", path.display()))
}

pub fn spawn_calibrate(cfg: CalibrateCfg, tx: mpsc::Sender<RetrainMsg>, cancel: Arc<AtomicBool>) {
    std::thread::spawn(move || match calibrate(&cfg, &tx, &cancel) {
        Ok(summary) => { let _ = tx.send(RetrainMsg::Done(summary)); }
        Err(e) => { let _ = tx.send(RetrainMsg::Error(e)); }
    });
}

/// Nearest-centroid calibration: for each curated family, average the
/// L2-normalized DINO mean-features into a centroid, then express that
/// centroid as an equivalent linear row (nearest-centroid on unit-norm
/// features IS a linear classifier: argmin‖x-c‖² == argmax(2x·c - ‖c‖²)).
/// Produces an ordinary `FewShotHead`-shaped file — no changes needed
/// anywhere `predict`/`decide_global` are used. A COPY of the base head is
/// spliced: classes present in the curated data get their row replaced,
/// everything else keeps the base head's original (trained) row untouched —
/// same anti-forgetting spirit as `retrain`'s warm-start/anchor, but
/// per-class rather than gradient-based, and immune to sample-count
/// imbalance since a centroid is a mean regardless of how many examples fed it.
fn calibrate(cfg: &CalibrateCfg, tx: &mpsc::Sender<RetrainMsg>, cancel: &AtomicBool) -> Result<String, String> {
    let _ = tx.send(RetrainMsg::Stage("Loading head + DINO".into()));
    let base = FewShotHead::load(&cfg.base_head_path)?;
    let dim = base.dim;
    let mut dino = DinoExtractor::load(&cfg.dino_model, base.infer_resolution)?;

    let all_rows = read_labels(&cfg.curations_dir.join("labels.jsonl"))?;
    let rows: Vec<LabelRow> = all_rows.into_iter().filter(|r| cfg.only_crops.contains(&r.crop)).collect();
    if rows.is_empty() {
        return Err("no marked examples for this calibration round — preview a leaf \
                     and paint/wand-fill a few examples first".into());
    }

    let norm = |s: &str| s.trim().to_lowercase();
    let mut name2class: HashMap<String, i32> = HashMap::new();
    let mut class_name: HashMap<i32, String> = HashMap::new();
    for (idx, name) in &base.families {
        if let Ok(i) = idx.parse::<i32>() {
            name2class.insert(norm(name), i);
            class_name.insert(i, name.clone());
        }
    }
    let mut next_id = base.classes.iter().copied().max().unwrap_or(0) + 1;

    // ── accumulate a running L2-normalized-feature sum per class ──
    let _ = tx.send(RetrainMsg::Stage("Extracting features from curations".into()));
    let crops_dir = cfg.curations_dir.join("labels");
    let mut sums: HashMap<i32, (Vec<f32>, usize)> = HashMap::new();
    for (k, r) in rows.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let cls = if r.source == "reject" || r.family == "rejected" {
            0i32
        } else if let Some(&c) = name2class.get(&norm(&r.family)) {
            c
        } else {
            let c = next_id;
            next_id += 1;
            name2class.insert(norm(&r.family), c);
            class_name.insert(c, r.family.clone());
            c
        };
        let path = crops_dir.join(&r.crop);
        let img = match image::open(&path) {
            Ok(i) => i.to_rgb8(),
            Err(e) => {
                let _ = tx.send(RetrainMsg::Log(format!("skip {}: {e}", r.crop)));
                continue;
            }
        };
        let f = dino.features(&img)?;
        if f.dim != dim {
            return Err(format!("crop feature dim {} != head dim {dim}", f.dim));
        }
        let feat = crop_feature(r, &crops_dir, &f.feat, f.grid, dim);
        let entry = sums.entry(cls).or_insert_with(|| (vec![0f32; dim], 0));
        for d in 0..dim {
            entry.0[d] += feat[d];
        }
        entry.1 += 1;
        if (k + 1) % 25 == 0 {
            let _ = tx.send(RetrainMsg::Log(format!("features {}/{}", k + 1, rows.len())));
        }
    }
    if sums.is_empty() {
        return Err("no usable curation crops".into());
    }

    // ── splice centroid-derived linear rows into a COPY of the base head ──
    let _ = tx.send(RetrainMsg::Stage("Computing centroids".into()));
    let mut head = base.clone();

    // `cfg.scale` is now an ABSOLUTE target coefficient norm, not a
    // multiplier on the base head's own scale — an earlier version matched
    // the base head's median coefficient norm, which sounded principled but
    // broke badly in practice: a real base head measured at median norm
    // ~97, wildly outside the range a sane logistic-regression row over
    // unit-norm features should have (single digits). Matching that
    // reproduced the exact same "whole leaf collapses into one class" bug
    // through a different path — the base head's own scale isn't a
    // trustworthy reference, so calibration no longer depends on it at all.
    // Logged as a diagnostic only: an unusually large value here is itself
    // a sign the base head may be poorly regularized / overconfident.
    let ref_norm: f32 = {
        let mut norms: Vec<f32> = base.coef.iter()
            .map(|row| row.iter().map(|v| v * v).sum::<f32>().sqrt())
            .filter(|n| n.is_finite() && *n > 1e-6)
            .collect();
        if norms.is_empty() {
            0.0
        } else {
            norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            norms[norms.len() / 2]
        }
    };
    let target_norm = cfg.scale.max(1e-3);
    let mut ref_log = format!(
        "base head's own median coef norm = {ref_norm:.3} (diagnostic only, not used) \
         -> calibrated classes target {target_norm:.3}"
    );
    if ref_norm > 20.0 {
        ref_log.push_str(" — that base-head norm is unusually large; it may itself be \
                           overconfident/poorly regularized, worth a fresh retrain at some point");
    }
    let _ = tx.send(RetrainMsg::Log(ref_log));

    let mut n_marked = 0usize;
    for (&cls, (sum, n)) in &sums {
        if *n == 0 {
            continue;
        }
        n_marked += 1;
        let c: Vec<f32> = sum.iter().map(|v| v / *n as f32).collect();
        let c_norm = c.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
        let k = target_norm / c_norm; // rescale direction c to the target norm
        let coef_row: Vec<f32> = c.iter().map(|v| v * k).collect();
        let intercept_row = -c_norm * target_norm / 2.0;
        if let Some(pos) = head.classes.iter().position(|&x| x == cls) {
            head.coef[pos] = coef_row;
            head.intercept[pos] = intercept_row;
        } else {
            head.classes.push(cls);
            head.coef.push(coef_row);
            head.intercept.push(intercept_row);
        }
        if let Some(name) = class_name.get(&cls) {
            head.families.insert(cls.to_string(), name.clone());
        }
    }

    head.onnx_parity = None;
    head.save(&cfg.out_path)?;
    Ok(format!(
        "Calibrated {n_marked} classes from {} crops -> {}",
        rows.len(), cfg.out_path.display()
    ))
}
