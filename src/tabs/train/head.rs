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

use crate::tabs::pipeline::dino::DinoExtractor;
use crate::tabs::pipeline::fewshot::FewShotHead;

pub struct RetrainCfg {
    pub head_path:     PathBuf,
    pub dino_model:    PathBuf,
    pub curations_dir: PathBuf,
    pub out_path:      PathBuf,
    pub epochs:        usize,
    pub lr:            f32,
    pub l2_anchor:     f32, // pull weights toward the base head (anti-forgetting)
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

/// L2-normalised mean DINO feature over a crop (matches the unit-norm patch space
/// the head was trained on).
fn mean_feature(feat: &[f32], grid: usize, dim: usize) -> Vec<f32> {
    let n = grid * grid;
    let mut m = vec![0f32; dim];
    for p in 0..n {
        let fp = &feat[p * dim..p * dim + dim];
        for d in 0..dim {
            m[d] += fp[d];
        }
    }
    let mut nrm = 0f32;
    for d in 0..dim {
        m[d] /= n as f32;
        nrm += m[d] * m[d];
    }
    let nrm = nrm.sqrt().max(1e-8);
    for d in 0..dim {
        m[d] /= nrm;
    }
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
    let mut nrm = 0f32;
    for d in 0..dim {
        m[d] /= n_used as f32;
        nrm += m[d] * m[d];
    }
    let nrm = nrm.sqrt().max(1e-8);
    for d in 0..dim {
        m[d] /= nrm;
    }
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

fn retrain(
    cfg: &RetrainCfg,
    tx: &mpsc::Sender<RetrainMsg>,
    cancel: &AtomicBool,
) -> Result<String, String> {
    let _ = tx.send(RetrainMsg::Stage("Loading head + DINO".into()));
    let mut head = FewShotHead::load(&cfg.head_path)?;
    let dim = head.dim;
    let mut dino = DinoExtractor::load(&cfg.dino_model, head.infer_resolution)?;

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

    // ── gather samples (mean feature per crop, one label each) ──
    let _ = tx.send(RetrainMsg::Stage("Extracting features from curations".into()));
    let crops_dir = cfg.curations_dir.join("labels");
    let mut samples: Vec<(Vec<f32>, i32)> = Vec::new();
    for (k, r) in rows.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let cls = if r.source == "reject" || r.family == "rejected" {
            healthy_class
        } else if let Some(&c) = name2class.get(&norm(&r.family)) {
            c
        } else {
            // a new, user-named family -> grow the head with a fresh class
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
        samples.push((crop_feature(r, &crops_dir, &f.feat, f.grid, dim), cls));
        if (k + 1) % 25 == 0 {
            let _ = tx.send(RetrainMsg::Log(format!("features {}/{}", k + 1, rows.len())));
        }
    }
    if samples.is_empty() {
        return Err("no usable curation crops".into());
    }

    // ── class-balanced sample weights ──
    // Plain per-sample averaging lets whichever class has the most curated
    // examples dominate the gradient outright (a real failure hit in
    // practice: 1000 "sucker" crops vs. 30 "hole" crops trained a head that
    // called EVERY anomaly "sucker"). Weight each sample by n_total /
    // (n_classes * count[class]) — sklearn's "balanced" scheme — so every
    // class contributes the same TOTAL weight to the gradient regardless of
    // how lopsided the curation counts are. Weights are constructed to sum
    // to n_total overall, so dividing by `n` below still normalizes correctly
    // and no LR retuning is needed.
    let mut class_counts: HashMap<i32, usize> = HashMap::new();
    for (_, cls) in &samples {
        *class_counts.entry(*cls).or_insert(0) += 1;
    }
    let n_classes = class_counts.len() as f32;
    let n_total = samples.len() as f32;
    let mut counts_log: Vec<(i32, usize)> = class_counts.iter().map(|(&c, &n)| (c, n)).collect();
    counts_log.sort_by_key(|&(c, _)| c);
    let counts_str: String = counts_log.iter()
        .map(|&(c, n)| {
            let name = head.families.get(&c.to_string()).cloned().unwrap_or_else(|| c.to_string());
            format!("{name}={n}")
        })
        .collect::<Vec<_>>().join(", ");
    let _ = tx.send(RetrainMsg::Log(format!("class balance: {counts_str}")));
    let class_weight = |cls: i32| -> f32 { n_total / (n_classes * class_counts[&cls] as f32) };

    // ── warm-started, anchored softmax fine-tune ──
    let _ = tx.send(RetrainMsg::Stage("Fine-tuning head".into()));
    let kk = head.classes.len();
    let class_row: HashMap<i32, usize> =
        head.classes.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let w0 = head.coef.clone(); // anchor toward the base head
    let b0 = head.intercept.clone();
    let n = samples.len() as f32;
    for ep in 0..cfg.epochs {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let mut gw = vec![vec![0f32; dim]; kk];
        let mut gb = vec![0f32; kk];
        let mut loss = 0f32;
        for (x, cls) in &samples {
            let row = class_row[cls];
            let sw = class_weight(*cls);
            let mut logit = vec![0f32; kk];
            for k in 0..kk {
                let mut s = head.intercept[k];
                let wk = &head.coef[k];
                for d in 0..dim {
                    s += wk[d] * x[d];
                }
                logit[k] = s;
            }
            let m = logit.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for l in logit.iter_mut() {
                *l = (*l - m).exp();
                sum += *l;
            }
            for l in logit.iter_mut() {
                *l /= sum;
            }
            loss -= sw * (logit[row] + 1e-9).ln();
            for k in 0..kk {
                let g = sw * (logit[k] - if k == row { 1.0 } else { 0.0 });
                gb[k] += g;
                let gwk = &mut gw[k];
                for d in 0..dim {
                    gwk[d] += g * x[d];
                }
            }
        }
        for k in 0..kk {
            head.intercept[k] -= cfg.lr * (gb[k] / n + cfg.l2_anchor * (head.intercept[k] - b0[k]));
            let wk = &mut head.coef[k];
            let w0k = &w0[k];
            let gwk = &gw[k];
            for d in 0..dim {
                wk[d] -= cfg.lr * (gwk[d] / n + cfg.l2_anchor * (wk[d] - w0k[d]));
            }
            // Hard safety cap: nothing here (lr=0.5, a weak l2_anchor=0.05,
            // 150 epochs, full-batch) stopped a class's weight row from
            // walking arbitrarily far when a batch of many similar new
            // examples pushes it consistently in one direction — almost
            // certainly how a coefficient norm as extreme as 97 (observed
            // in a real deployed head) happened in the first place, and
            // exactly the mechanism behind "whole leaf gets one color"
            // regardless of what calibrate()'s own (separate) scale fix
            // does. Rescale back to a generous but bounded ceiling every
            // epoch rather than only checking at the end, so it can't
            // spend 150 epochs compounding past it first.
            const MAX_COEF_NORM: f32 = 25.0;
            let wnorm: f32 = wk.iter().map(|v| v * v).sum::<f32>().sqrt();
            if wnorm > MAX_COEF_NORM {
                let s = MAX_COEF_NORM / wnorm;
                for v in wk.iter_mut() {
                    *v *= s;
                }
            }
        }
        if (ep + 1) % 25 == 0 || ep == cfg.epochs - 1 {
            let _ = tx.send(RetrainMsg::Log(format!("epoch {}/{}  loss {:.4}", ep + 1, cfg.epochs, loss / n)));
        }
    }

    // ── write updated head ──
    let max_norm: f32 = head.coef.iter()
        .map(|row| row.iter().map(|v| v * v).sum::<f32>().sqrt())
        .fold(0f32, f32::max);
    let _ = tx.send(RetrainMsg::Log(format!("post-training max coef norm = {max_norm:.3}")));
    head.onnx_parity = None; // weights changed; parity no longer the exported one
    head.save(&cfg.out_path)?;
    Ok(format!(
        "Updated head: {} crops, {} classes (+{} new) -> {}",
        samples.len(), kk, n_new, cfg.out_path.display()
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
