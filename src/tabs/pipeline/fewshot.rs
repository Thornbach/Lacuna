//! Few-shot supervised detection head — a linear classifier on frozen DINO patch
//! features, trained in Python (`1Help/eval/export_fewshot_head.py`) and applied
//! here as `softmax(W·feat + b)` per patch. This is the validated replacement for
//! the label-free PatchCore kNN bank: each patch is classified into
//! {healthy, family1..K}; `defect_prob = 1 - P(healthy)`, family = argmax defect.
//!
//! The head consumes the SAME `DinoExtractor` features the PatchCore path uses
//! (512px, 32×32 grid, 1536-D, per-layer L2-normed), so no extra model is loaded.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::detect;

#[derive(Deserialize, Serialize, Clone)]
pub struct FewShotHead {
    pub infer_resolution: u32,
    #[allow(dead_code)] // metadata from the export; kept for provenance
    pub patch_size: u32,
    pub dim: usize,
    pub classes: Vec<i32>,   // class indices; 0 = healthy
    pub intercept: Vec<f32>, // [K]
    pub coef: Vec<Vec<f32>>, // [K][D]  (sklearn LogisticRegression.coef_)
    #[allow(dead_code)] // export-time default τ; the UI carries the live value
    pub tau_default: f32,
    #[serde(default)]
    pub families: HashMap<String, String>, // "idx" -> family name
    #[serde(default)]
    pub onnx_parity: Option<f32>,
    /// Optional per-family seed-threshold override (family idx → τ_hi). Empty = use
    /// the global seed τ for every family. Lets weak families (e.g. pink) seed at a
    /// lower threshold to recover recall without loosening the strong ones.
    #[serde(default, rename = "per_family_tau")]
    pub hi_fam: HashMap<i32, f32>,
}

/// Per-patch head output over the feature grid.
pub struct HeadOut {
    #[allow(dead_code)] // grid size echoed back for callers that want it
    pub grid: usize,
    pub defect_prob: Vec<f32>, // grid*grid, = 1 - P(healthy)
    pub family: Vec<i32>,      // grid*grid, argmax defect class (>=1); 0 if none win
}

impl FewShotHead {
    pub fn load(path: &Path) -> Result<Self, String> {
        let s = std::fs::read_to_string(path)
            .map_err(|e| format!("read head {}: {e}", path.display()))?;
        let h: FewShotHead = serde_json::from_str(&s).map_err(|e| format!("parse head: {e}"))?;
        let k = h.classes.len();
        if h.coef.len() != k || h.intercept.len() != k {
            return Err("few-shot head: classes/coef/intercept length mismatch".into());
        }
        if h.coef.iter().any(|r| r.len() != h.dim) {
            return Err("few-shot head: coef row dim != dim".into());
        }
        Ok(h)
    }

    /// Display name for a family index (falls back to "Cluster N").
    pub fn family_name(&self, idx: i32) -> String {
        self.families
            .get(&idx.to_string())
            .cloned()
            .unwrap_or_else(|| format!("Cluster {idx}"))
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string(self).map_err(|e| format!("serialize head: {e}"))?;
        if let Some(p) = path.parent() {
            let _ = std::fs::create_dir_all(p);
        }
        std::fs::write(path, json).map_err(|e| format!("write head {}: {e}", path.display()))
    }

    /// Fold class `from_id` into `into_id`: drops `from_id`'s row from
    /// `classes`/`coef`/`intercept` (parallel-by-position) and its entries
    /// from `families`/`hi_fam` (keyed by id) — `into_id`'s own trained row
    /// is untouched. Curated labels still naming the old family are NOT
    /// rewritten here; callers pair this with a `labels.jsonl` rewrite (see
    /// `rewrite_curated_family`) so the two stay consistent.
    pub fn merge_class(&mut self, from_id: i32, into_id: i32) -> Result<(), String> {
        if from_id == into_id {
            return Ok(());
        }
        let pos = self.classes.iter().position(|&c| c == from_id)
            .ok_or_else(|| format!("merge_class: no such class {from_id}"))?;
        if !self.classes.iter().any(|&c| c == into_id) {
            return Err(format!("merge_class: target class {into_id} does not exist"));
        }
        self.classes.remove(pos);
        self.coef.remove(pos);
        self.intercept.remove(pos);
        self.families.remove(&from_id.to_string());
        self.hi_fam.remove(&from_id);
        Ok(())
    }

    /// Removes class `id` entirely — no merge target, it just stops being a
    /// predictable class. NOTE: this narrows the argmax, it does NOT
    /// guarantee those patches fall through to PatchCore/novel — a patch
    /// that weakly matched the deleted class only ends up there if none of
    /// the REMAINING classes clear the seed threshold either; if another
    /// class is a decent match, the patch is reassigned to that class
    /// instead. Callers should pair this with a `labels.jsonl` rewrite (to
    /// "rejected", say) if past curations named this class, same caveat as
    /// `merge_class`.
    pub fn delete_class(&mut self, id: i32) -> Result<(), String> {
        let pos = self.classes.iter().position(|&c| c == id)
            .ok_or_else(|| format!("delete_class: no such class {id}"))?;
        self.classes.remove(pos);
        self.coef.remove(pos);
        self.intercept.remove(pos);
        self.families.remove(&id.to_string());
        self.hi_fam.remove(&id);
        Ok(())
    }

    /// Classify each patch. `feat` is `grid*grid*dim` row-major (DinoExtractor order).
    pub fn predict(&self, feat: &[f32], grid: usize, dim: usize) -> HeadOut {
        assert_eq!(dim, self.dim, "feature dim mismatch: head {} vs {dim}", self.dim);
        let k = self.classes.len();
        let healthy_col = self.classes.iter().position(|&c| c == 0);
        let n = grid * grid;
        let mut defect_prob = vec![0f32; n];
        let mut family = vec![0i32; n];
        let mut logits = vec![0f32; k];
        for p in 0..n {
            let fp = &feat[p * dim..p * dim + dim];
            for (c, slot) in logits.iter_mut().enumerate() {
                let w = &self.coef[c];
                let mut acc = self.intercept[c];
                for j in 0..dim {
                    acc += w[j] * fp[j];
                }
                *slot = acc;
            }
            let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for l in logits.iter_mut() {
                *l = (*l - m).exp();
                sum += *l;
            }
            let p_healthy = healthy_col.map(|hc| logits[hc] / sum).unwrap_or(0.0);
            defect_prob[p] = 1.0 - p_healthy;
            // argmax over the defect classes (skip healthy)
            let mut best = 0i32;
            let mut bestv = f32::NEG_INFINITY;
            for (c, &cls) in self.classes.iter().enumerate() {
                if cls != 0 && logits[c] > bestv {
                    bestv = logits[c];
                    best = cls;
                }
            }
            family[p] = best;
        }
        HeadOut { grid, defect_prob, family }
    }
}

/// A defect region from the few-shot detector (mirrors detect::Region + family).
pub struct FewShotRegion {
    pub bbox: [u32; 4],
    #[allow(dead_code)] // mirrors detect::Region; not consumed in the worker yet
    pub area: u32,
    pub centroid: [f32; 2],
    pub mask: Vec<bool>,
    pub family: i32, // head-assigned family (>=1)
}

/// Few-shot decision stage (validated in `1Help/eval/decide_eval.py`).
/// Hysteresis on the defect-prob patch grid: a patch SEEDS a region when
/// `prob ≥ tau_hi` (per predicted family via `hi_fam`, else `tau_hi`); the region
/// GROWS into 8-connected patches with `prob ≥ tau_lo`; a grow-blob with no seed or
/// fewer than `min_region` patches is dropped. This halves the false-positive
/// region rate vs a flat threshold (FP regions are low-confidence, real defects
/// have a high-confidence core).
pub const HEAD_MIN_REGION_PATCHES: usize = 2;  // drop grow-blobs smaller than this (patches)

/// Hysteresis over a `w`×`h` grid (used both at per-tile patch-grid resolution
/// and, since the tile-seam fix, at full-leaf PIXEL resolution — the function
/// doesn't care which, it just needs 8-connectivity over a rectangular field).
fn grid_hysteresis(
    prob: &[f32],
    fam: &[i32],
    w: usize,
    h: usize,
    tau_hi: f32,
    tau_lo: f32,
    hi_fam: &HashMap<i32, f32>,
    min_region: usize,
) -> Vec<bool> {
    let n = w * h;
    let grow: Vec<bool> = prob.iter().map(|&p| p >= tau_lo).collect();
    let mut out = vec![false; n];
    let mut visited = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut comp: Vec<usize> = Vec::new();
    for start in 0..n {
        if !grow[start] || visited[start] {
            continue;
        }
        stack.clear();
        comp.clear();
        stack.push(start);
        visited[start] = true;
        let mut has_seed = false;
        while let Some(i) = stack.pop() {
            comp.push(i);
            let hi_eff = hi_fam.get(&fam[i]).copied().unwrap_or(tau_hi);
            if prob[i] >= hi_eff {
                has_seed = true;
            }
            let (y, x) = ((i / w) as i32, (i % w) as i32);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dy == 0 && dx == 0 {
                        continue;
                    }
                    let (ny, nx) = (y + dy, x + dx);
                    if ny < 0 || nx < 0 || ny >= h as i32 || nx >= w as i32 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if grow[j] && !visited[j] {
                        visited[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        if has_seed && comp.len() >= min_region {
            for &i in &comp {
                out[i] = true;
            }
        }
    }
    out
}

/// Nearest-neighbour upscale of a `g`×`g` i32 grid (family labels) to pixel
/// resolution — the discrete-label counterpart of `detect::upscale`'s bilinear
/// float upscale (used for `defect_prob`).
pub fn upscale_family(grid: &[i32], g: usize, out_w: usize, out_h: usize) -> Vec<i32> {
    let mut out = vec![0i32; out_w * out_h];
    for oy in 0..out_h {
        let gy = (oy * g / out_h.max(1)).min(g - 1);
        for ox in 0..out_w {
            let gx = (ox * g / out_w.max(1)).min(g - 1);
            out[oy * out_w + ox] = grid[gy * g + gx];
        }
    }
    out
}

/// Few-shot decision stage, run ONCE over a full-leaf stitched `prob`/`fam` map
/// (both already at leaf-pixel resolution — each tile's grid output upscaled and
/// placed at its origin before calling this) rather than per-tile. Running
/// hysteresis and region-extraction per-tile independently let a real anomaly
/// spanning two tiles lose whatever half didn't reach ITS OWN tile's seed
/// threshold, producing a visible cut at the tile seam; deciding globally removes
/// that split entirely. Hysteresis: a pixel SEEDS at `prob ≥ tau_hi` (per-family
/// override via `hi_fam`), GROWS into 8-connected pixels down to `tau_lo`; a
/// grow-blob with no seed or fewer than `min_region` pixels is dropped.
///
/// Regions are then extracted in TWO passes, not once over the whole anomaly
/// mask:
///
/// 1. Decide "is this a real anomaly" exactly as a single family-agnostic
///    pass always did — connected components over the plain boolean mask,
///    gated by the full `min_area`. This is the realness gate; splitting by
///    type below must never be able to veto it.
/// 2. Within EACH validated blob, split by the head's own per-pixel family
///    assignment (already computed at this exact resolution — no extra
///    detector cost), working only in that blob's own small bbox. A uniform
///    blob stays one region; a blob spanning more than one family divides —
///    fixing the case where two visually distinct anomaly types physically
///    touch on the same leaf (e.g. necrosis right next to a cluster of
///    suckers) and would otherwise connected-component-merge into ONE
///    `AnomalyRegion`, before any embedding or clustering ever runs. That
///    permanently bakes "two different things" into one indivisible region:
///    clustering assigns a single label per region (pooled over its whole
///    mask), so no later re-cut/K change, nor even the Knife tool without
///    manual intervention, could ever separate them again.
///
/// Splitting is gated by a much smaller floor than `min_area` (a piece just
/// needs to not be lone-pixel noise) since realness was already decided in
/// step 1 — an EARLIER version of this function gated each family's
/// sub-mask by the FULL `min_area` independently, which meant a genuinely
/// large blob with noisy/mixed per-pixel family assignment near its
/// boundary could fragment into pieces each too small to individually pass,
/// silently dropping the whole anomaly. That regression is why step 1 is
/// family-agnostic and unconditional.
#[allow(clippy::too_many_arguments)]
pub fn decide_global(
    prob:               &[f32],
    fam:                &[i32],
    valid:              &[bool],
    w:                  usize,
    h:                  usize,
    tau_hi:             f32,
    tau_lo:             f32,
    hi_fam:             &HashMap<i32, f32>,
    min_region:         usize,
    region_close_px:    usize,
    min_area:           u32,
) -> (Vec<bool>, Vec<FewShotRegion>) {
    let grid_mask = grid_hysteresis(prob, fam, w, h, tau_hi, tau_lo, hi_fam, min_region);
    let n = w * h;
    let mut mask = vec![false; n];
    for i in 0..n {
        mask[i] = valid[i] && grid_mask[i];
    }
    let mask = detect::morph_close(&mask, w, h, region_close_px);

    // Step 1: family-agnostic realness gate, unchanged from before the
    // per-family split existed.
    let overall = detect::extract_regions(&mask, w, h, min_area);

    // Step 2: split each validated blob by family, within its own bbox
    // (cheap — that blob's area, not the whole leaf's) — but ONLY when a
    // secondary family is a SUBSTANTIAL fraction of the blob, not just a
    // handful of stray misclassified pixels. Splitting on any family
    // disagreement at all (an earlier version of this function did exactly
    // that) ballooned region counts ~4x on real data (a QA dataset went
    // from ~4000 to ~15600 regions) — since DINO-embedding one region costs
    // a fixed ~45-50ms regardless of its size, that turned directly into
    // ~4x longer embedding time (minutes -> tens of minutes) for what was,
    // for most of those extra regions, an insignificant sliver of a
    // different-but-noisy family call, not a genuine second anomaly type
    // like the necrosis-next-to-suckers case this splitting exists for.
    const SPLIT_MIN_AREA: u32 = 3;
    // Reverted 0.25 -> 0.15: 0.25 was bumped up purely to cut region count for
    // a performance complaint, but a necrosis patch that's a SMALL fraction of
    // a bigger sucker blob it's touching (the exact case this splitting exists
    // for) can fall under 0.25 and get silently re-swallowed into the majority
    // family — i.e. the original reported bug coming back disguised as "the
    // clustering got worse." Region-count growth from this should hurt less
    // now than it did the first time, since DINO batching + the O(n^2) NN-chain
    // clustering rewrite (both landed since) absorb a chunk of that cost.
    const MIN_SPLIT_FRACTION: f32 = 0.15; // a family needs >=15% of the blob to earn its own region
    let mut out: Vec<FewShotRegion> = Vec::new();
    for blob in &overall {
        let [bx, by, bw, bh] = blob.bbox;
        // Per-family pixel counts within this blob (family 0 = "no defect
        // class won" — excluded, same as the old majority-vote path used to
        // ignore it too; effectively never occurs for a pixel that already
        // passed hysteresis, since `predict()` only leaves `family` at 0
        // when literally no defect class beat healthy).
        let mut family_area: std::collections::HashMap<i32, u32> = std::collections::HashMap::new();
        for idx in 0..blob.mask.len() {
            if !blob.mask[idx] {
                continue;
            }
            let (lx, ly) = (idx as u32 % bw, idx as u32 / bw);
            let f = fam[(by + ly) as usize * w + (bx + lx) as usize];
            if f != 0 {
                *family_area.entry(f).or_insert(0) += 1;
            }
        }
        let total: u32 = family_area.values().sum();
        let significant: Vec<i32> = family_area.iter()
            .filter(|&(_, &area)| total > 0 && area as f32 / total as f32 >= MIN_SPLIT_FRACTION)
            .map(|(&f, _)| f)
            .collect();

        if significant.len() <= 1 {
            // Not a genuine multi-type blob (uniform, or one dominant
            // family plus noisy stray misclassifications) — keep it as ONE
            // region under the majority family, exactly the pre-split
            // behavior.
            let family = family_area.iter().max_by_key(|&(_, &c)| c).map(|(&f, _)| f).unwrap_or(0);
            out.push(FewShotRegion {
                bbox: blob.bbox, area: blob.area, centroid: blob.centroid,
                mask: blob.mask.clone(), family,
            });
            continue;
        }

        for &fam_id in &significant {
            let fam_local: Vec<bool> = (0..blob.mask.len()).map(|idx| {
                if !blob.mask[idx] {
                    return false;
                }
                let (lx, ly) = (idx as u32 % bw, idx as u32 / bw);
                fam[(by + ly) as usize * w + (bx + lx) as usize] == fam_id
            }).collect();
            let sub = detect::extract_regions(&fam_local, bw as usize, bh as usize, SPLIT_MIN_AREA);
            out.extend(sub.into_iter().map(|sr| {
                let [sx, sy, sw, sh] = sr.bbox;
                FewShotRegion {
                    bbox: [bx + sx, by + sy, sw, sh],
                    area: sr.area,
                    centroid: [bx as f32 + sr.centroid[0], by as f32 + sr.centroid[1]],
                    mask: sr.mask,
                    family: fam_id,
                }
            }));
        }
    }
    (mask, out)
}
