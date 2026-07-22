//! DINO-channel anomaly detection (Slice-2 single-channel skeleton).
//!
//! Ports the dino path of the 1Help detector:
//!   features --kNN--> per-patch score grid  (detector.anomaly_map)
//!   grid --bilinear--> tile map
//!   robust per-tile z with scale floor       (postprocess.robust_z)
//!   per-channel hysteresis + absolute floor  (detector.decide, dino only)
//!   connected components -> regions          (regions.extract_regions)
//!
//! Residual + color channels, scale-gate and morphological close come in Slice 5.

use image::RgbImage;

use super::bank::KnnScorer;
use super::channels;
use super::dino::DinoExtractor;
use super::meta::DetectorMeta;

pub struct Region {
    pub bbox:       [u32; 4], // x, y, w, h
    pub area:       u32,
    pub centroid:   [f32; 2],
    pub mask:       Vec<bool>, // bbox-local, w*h
    /// 8-D clustering descriptor (port of regions.region_descriptor):
    /// [z_dino, z_res, z_color, a_dev, b_dev, log_area, elongation, extent].
    pub descriptor: [f32; 8],
}

pub struct DetectResult {
    pub w:        usize,
    pub h:        usize,
    pub dino_map: Vec<f32>,  // raw kNN mean-dist, upscaled to tile (w*h)
    pub z_map:    Vec<f32>,  // per-tile robust z (w*h)
    // z_res/z_color/med_a/med_b: exposed so the worker can score OTHER detectors'
    // regions (e.g. few-shot head regions) against this same descriptor space for
    // unsupervised clustering — see AnomalyRegion::descriptor's doc comment.
    pub z_res:    Vec<f32>,
    pub z_color:  Vec<f32>,
    pub med_a:    f32,
    pub med_b:    f32,
    pub mask:     Vec<bool>, // final binary (w*h)
    pub regions:  Vec<Region>,
    pub dino_ms:  u128,      // DINOv3 ort feature time
    pub knn_ms:   u128,      // bank kNN time
}

pub struct DetectParams {
    pub knn_k:              usize,
    pub dino_low_ratio:     f32, // hysteresis grow ratio per channel
    pub residual_low_ratio: f32,
    pub color_low_ratio:    f32,
    pub residual_ksize:     usize,
    pub residual_max_area:  u32, // residual-only blobs larger than this need dino/color support
    pub region_close_px:    u32,
    pub min_area:           u32,
    pub use_color:          bool,
    pub use_dino_floor:     bool,
}

impl Default for DetectParams {
    fn default() -> Self {
        Self {
            knn_k: 3,
            dino_low_ratio: 0.4,
            residual_low_ratio: 0.9,
            color_low_ratio: 0.6,
            residual_ksize: 9,
            residual_max_area: 100,
            region_close_px: 1, // tighter regions (was 2 = more dilation/merging)
            min_area: 3,
            use_color: true,
            use_dino_floor: true,
        }
    }
}

/// Run the full detector on one tile (`img`, resized to the model res inside the
/// extractor). DINO + color are trusted at all scales; the residual channel is
/// scale-gated (a large residual-only blob = extended texture, dropped unless
/// dino/color agree). Channels missing from the calibrated meta are skipped, so
/// a dino-only bank still works.
pub fn detect(
    dino:     &mut DinoExtractor,
    bank:     &dyn KnnScorer,
    meta:     &DetectorMeta,
    params:   &DetectParams,
    img:      &RgbImage,
    out_w:    usize,
    out_h:    usize,
    valid_in: Option<&[bool]>, // tile alpha mask (out_w*out_h); None = all valid
) -> Result<DetectResult, String> {
    let n = out_w * out_h;
    let t0 = std::time::Instant::now();
    let f = dino.features(img)?;
    let dino_ms = t0.elapsed().as_millis();
    let t1 = std::time::Instant::now();
    let scores = bank.knn_mean_dist(&f.feat, f.grid * f.grid, params.knn_k);
    let knn_ms = t1.elapsed().as_millis();
    let dino_map = upscale(&scores, f.grid, f.grid, out_w, out_h);

    let all_true;
    let valid: &[bool] = match valid_in {
        Some(v) => v,
        None => { all_true = vec![true; n]; &all_true }
    };

    // ── dino channel (always) ──
    let z_dino = robust_z(&dino_map, valid, meta.scale_floor("dino"));
    let dino_mask = channel_mask(
        &z_dino, meta.ch_threshold("dino"), params.dino_low_ratio, valid, out_w, out_h);

    // ── color channel (necrosis blobs), trusted at all scales. LAB a/b are also
    //    used for the descriptor's colour-direction dims, so compute them once. ──
    let (lab_a, lab_b, med_a, med_b) = channels::lab_ab(img, valid);
    let do_color = params.use_color && meta.has("color");
    let z_color = if do_color {
        let mut color = vec![0f32; n];
        for i in 0..n {
            if valid[i] {
                color[i] = ((lab_a[i] - med_a).powi(2) + (lab_b[i] - med_b).powi(2)).sqrt();
            }
        }
        robust_z(&color, valid, meta.scale_floor("color"))
    } else {
        vec![0f32; n]
    };
    let color_mask = if do_color {
        channel_mask(&z_color, meta.ch_threshold("color"), params.color_low_ratio, valid, out_w, out_h)
    } else {
        vec![false; n]
    };

    // ── residual channel (small specks), SCALE-GATED ──
    let z_res = if meta.has("residual") {
        let residual = channels::residual_map(img, valid, params.residual_ksize);
        robust_z(&residual, valid, meta.scale_floor("residual"))
    } else {
        vec![0f32; n]
    };
    let res_mask = if meta.has("residual") {
        channel_mask(&z_res, meta.ch_threshold("residual"), params.residual_low_ratio, valid, out_w, out_h)
    } else {
        vec![false; n]
    };

    // combine: dino + color are support; residual kept only if small or supported
    let mut support = vec![false; n];
    for i in 0..n {
        support[i] = dino_mask[i] || color_mask[i];
    }
    let res_kept = scale_gate_residual(&res_mask, &support, out_w, out_h, params.residual_max_area);
    let mut mask = vec![false; n];
    for i in 0..n {
        mask[i] = support[i] || res_kept[i];
    }

    // absolute DINO floor (cross-tile-stable safety net)
    if params.use_dino_floor {
        for i in 0..n {
            if valid[i] && dino_map[i] >= meta.dino_threshold {
                mask[i] = true;
            }
        }
    }
    for i in 0..n {
        if !valid[i] {
            mask[i] = false;
        }
    }

    // morphological close merges fragments, then connected components -> regions
    let mask = morph_close(&mask, out_w, out_h, params.region_close_px as usize);
    let mut regions = extract_regions(&mask, out_w, out_h, params.min_area);
    for rg in &mut regions {
        rg.descriptor = region_descriptor(
            rg.bbox, &rg.mask, rg.area, &z_dino, &z_res, &z_color, &lab_a, &lab_b, med_a, med_b, out_w);
    }
    Ok(DetectResult {
        w: out_w, h: out_h, dino_map, z_map: z_dino, z_res, z_color, med_a, med_b,
        mask, regions, dino_ms, knn_ms,
    })
}

/// Per-tile raw signal ONLY (no z-score, no decision): bank kNN scores against
/// an ALREADY-EXTRACTED DINO feature grid (shared with the few-shot path — one
/// DINO forward per tile serves both detectors, not two) + raw LAB a/b +
/// residual, each upscaled/computed at the tile's own pixel size. Split out of
/// `detect()` so the worker can stitch every tile's raw signal into full-leaf
/// canvases BEFORE `decide_global` computes robust-z or reaches a decision —
/// see `decide_global`'s doc comment for why per-tile statistics and per-tile
/// decisions both break down at scale.
pub struct TileSignal {
    pub dino_map: Vec<f32>, // out_w*out_h, raw kNN mean-dist
    pub lab_a:    Vec<f32>,
    pub lab_b:    Vec<f32>,
    pub residual: Vec<f32>, // zeros if meta doesn't calibrate residual
    pub valid:    Vec<bool>,
    pub knn_ms:   u128,
}

pub fn extract_tile_signal(
    feat:     &super::dino::DinoFeatures,
    bank:     &dyn KnnScorer,
    meta:     &DetectorMeta,
    params:   &DetectParams,
    img:      &RgbImage, // for LAB/residual only — DINO features already extracted
    out_w:    usize,
    out_h:    usize,
    valid_in: Option<&[bool]>,
) -> TileSignal {
    let n = out_w * out_h;
    let t1 = std::time::Instant::now();
    let scores = bank.knn_mean_dist(&feat.feat, feat.grid * feat.grid, params.knn_k);
    let knn_ms = t1.elapsed().as_millis();
    let dino_map = upscale(&scores, feat.grid, feat.grid, out_w, out_h);

    let all_true;
    let valid: &[bool] = match valid_in {
        Some(v) => v,
        None => { all_true = vec![true; n]; &all_true }
    };

    // raw LAB a/b (tile-local median from lab_ab discarded — recomputed globally
    // in decide_global, which is the whole point of stitching first).
    let (lab_a, lab_b, _, _) = channels::lab_ab(img, valid);
    let residual = if meta.has("residual") {
        channels::residual_map(img, valid, params.residual_ksize)
    } else {
        vec![0f32; n]
    };

    TileSignal { dino_map, lab_a, lab_b, residual, valid: valid.to_vec(), knn_ms }
}

/// PatchCore decision stage, run ONCE over full-leaf stitched raw signal (every
/// tile's `TileSignal` upscaled/placed at its origin by the caller) rather than
/// per-tile. Two problems this fixes at once vs. the old per-tile `detect()`:
/// (1) seam artifacts — a real anomaly spanning two tiles was judged
/// independently by each tile's own hysteresis, so whichever half didn't reach
/// ITS tile's threshold got silently dropped, cutting the anomaly at the tile
/// boundary; (2) robust-z corruption — `robust_z`'s median/MAD assumes most of
/// what it's computed over is healthy, which fails when a large anomaly
/// dominates a SMALL tile (the "healthy" statistics end up describing the
/// anomaly itself). Computing median/MAD over the whole leaf's valid pixels
/// instead is far more robust to a big, localized anomaly.
#[allow(clippy::too_many_arguments)]
pub fn decide_global(
    dino_map: &[f32],
    lab_a:    &[f32],
    lab_b:    &[f32],
    residual: &[f32],
    valid:    &[bool],
    w:        usize,
    h:        usize,
    meta:     &DetectorMeta,
    params:   &DetectParams,
) -> DetectResult {
    let n = w * h;

    let z_dino = robust_z(dino_map, valid, meta.scale_floor("dino"));
    let dino_mask = channel_mask(&z_dino, meta.ch_threshold("dino"), params.dino_low_ratio, valid, w, h);

    let do_color = params.use_color && meta.has("color");
    let mut av: Vec<f32> = (0..n).filter(|&i| valid[i]).map(|i| lab_a[i]).collect();
    let mut bv: Vec<f32> = (0..n).filter(|&i| valid[i]).map(|i| lab_b[i]).collect();
    let (med_a, med_b) = if av.is_empty() { (0.0, 0.0) } else {
        (channels::median(&mut av), channels::median(&mut bv))
    };
    let z_color = if do_color {
        let mut color = vec![0f32; n];
        for i in 0..n {
            if valid[i] {
                color[i] = ((lab_a[i] - med_a).powi(2) + (lab_b[i] - med_b).powi(2)).sqrt();
            }
        }
        robust_z(&color, valid, meta.scale_floor("color"))
    } else {
        vec![0f32; n]
    };
    let color_mask = if do_color {
        channel_mask(&z_color, meta.ch_threshold("color"), params.color_low_ratio, valid, w, h)
    } else {
        vec![false; n]
    };

    let z_res = if meta.has("residual") {
        robust_z(residual, valid, meta.scale_floor("residual"))
    } else {
        vec![0f32; n]
    };
    let res_mask = if meta.has("residual") {
        channel_mask(&z_res, meta.ch_threshold("residual"), params.residual_low_ratio, valid, w, h)
    } else {
        vec![false; n]
    };

    let mut support = vec![false; n];
    for i in 0..n {
        support[i] = dino_mask[i] || color_mask[i];
    }
    let res_kept = scale_gate_residual(&res_mask, &support, w, h, params.residual_max_area);
    let mut mask = vec![false; n];
    for i in 0..n {
        mask[i] = support[i] || res_kept[i];
    }
    if params.use_dino_floor {
        for i in 0..n {
            if valid[i] && dino_map[i] >= meta.dino_threshold {
                mask[i] = true;
            }
        }
    }
    for i in 0..n {
        if !valid[i] {
            mask[i] = false;
        }
    }

    let mask = morph_close(&mask, w, h, params.region_close_px as usize);
    let mut regions = extract_regions(&mask, w, h, params.min_area);
    for rg in &mut regions {
        rg.descriptor = region_descriptor(
            rg.bbox, &rg.mask, rg.area, &z_dino, &z_res, &z_color, lab_a, lab_b, med_a, med_b, w);
    }
    DetectResult {
        w, h, dino_map: dino_map.to_vec(), z_map: z_dino, z_res, z_color, med_a, med_b,
        mask, regions, dino_ms: 0, knn_ms: 0,
    }
}

/// 8-D descriptor over a region's pixels: per-channel mean z, mean colour
/// deviation direction (a,b), and shape (log-area, elongation, extent).
/// Takes raw `(bbox, mask, area)` rather than `&Region` so callers with a
/// differently-typed region (e.g. `fewshot::FewShotRegion`) can score their
/// own regions against these same z-maps for unsupervised clustering.
pub(crate) fn region_descriptor(
    bbox: [u32; 4], mask: &[bool], area: u32,
    z_dino: &[f32], z_res: &[f32], z_color: &[f32],
    lab_a: &[f32], lab_b: &[f32], med_a: f32, med_b: f32, w: usize,
) -> [f32; 8] {
    let [bx, by, bw, bh] = bbox;
    let (mut sd, mut sr, mut sc, mut sa, mut sb) = (0f32, 0f32, 0f32, 0f32, 0f32);
    for ly in 0..bh {
        for lx in 0..bw {
            if !mask[(ly * bw + lx) as usize] {
                continue;
            }
            let gi = (by + ly) as usize * w + (bx + lx) as usize;
            sd += z_dino[gi];
            sr += z_res[gi];
            sc += z_color[gi];
            sa += lab_a[gi] - med_a;
            sb += lab_b[gi] - med_b;
        }
    }
    let areaf = area.max(1) as f32;
    let (elong, extent) = region_shape(bbox, mask, area);
    [sd / areaf, sr / areaf, sc / areaf, sa / areaf, sb / areaf, (1.0 + areaf).ln(), elong, extent]
}

/// (elongation, extent) of a region: elongation = sqrt(eigenvalue ratio of the
/// coordinate covariance) capped at 20 (line ≫1, blob ~1); extent = filled
/// fraction of the bounding box.
fn region_shape(bbox: [u32; 4], mask: &[bool], area: u32) -> (f32, f32) {
    let [_, _, bw, bh] = bbox;
    let extent = area as f32 / (bw * bh).max(1) as f32;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for ly in 0..bh {
        for lx in 0..bw {
            if mask[(ly * bw + lx) as usize] {
                xs.push(lx as f32);
                ys.push(ly as f32);
            }
        }
    }
    if xs.len() < 2 {
        return (1.0, extent);
    }
    let nf = xs.len() as f32;
    let mx = xs.iter().sum::<f32>() / nf;
    let my = ys.iter().sum::<f32>() / nf;
    let (mut cxx, mut cyy, mut cxy) = (0f32, 0f32, 0f32);
    for i in 0..xs.len() {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        cxx += dx * dx;
        cyy += dy * dy;
        cxy += dx * dy;
    }
    cxx /= nf;
    cyy /= nf;
    cxy /= nf;
    let tr = cxx + cyy;
    let det = cxx * cyy - cxy * cxy;
    let disc = (tr * tr / 4.0 - det).max(0.0).sqrt();
    let l1 = tr / 2.0 + disc;
    let l2 = (tr / 2.0 - disc).max(1e-6);
    ((l1 / l2).sqrt().min(20.0), extent)
}

/// One channel's hysteresis mask: seed at `high`, grow connected pixels down to
/// `high * low_ratio` that contain a seed.
fn channel_mask(z: &[f32], high: f32, low_ratio: f32, valid: &[bool], w: usize, h: usize) -> Vec<bool> {
    if !high.is_finite() {
        return vec![false; w * h];
    }
    let low = high * low_ratio;
    let seeds: Vec<bool> = z.iter().zip(valid).map(|(&z, &ok)| ok && z >= high).collect();
    let cand: Vec<bool> = z.iter().zip(valid).map(|(&z, &ok)| ok && z >= low).collect();
    grow_from_seeds(&cand, &seeds, w, h)
}

/// Drop residual-only components larger than `cap` (extended texture) unless they
/// overlap the dino/color `support`.
fn scale_gate_residual(res_mask: &[bool], support: &[bool], w: usize, h: usize, cap: u32) -> Vec<bool> {
    if cap == 0 {
        return res_mask.to_vec();
    }
    let (lbl, n) = connected_components(res_mask, w, h);
    if n == 0 {
        return vec![false; w * h];
    }
    let mut area = vec![0u32; n + 1];
    let mut sup = vec![false; n + 1];
    for i in 0..w * h {
        let l = lbl[i] as usize;
        if l != 0 {
            area[l] += 1;
            if support[i] {
                sup[l] = true;
            }
        }
    }
    (0..w * h)
        .map(|i| {
            let l = lbl[i] as usize;
            l != 0 && (area[l] <= cap || sup[l])
        })
        .collect()
}

pub fn morph_close(mask: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    if r == 0 {
        return mask.to_vec();
    }
    erode_bool(&dilate_bool(mask, w, h, r), w, h, r)
}

fn dilate_bool(mask: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    morph_sep(mask, w, h, r, true)
}
fn erode_bool(mask: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    morph_sep(mask, w, h, r, false)
}

/// Separable binary morphology: dilate (OR over window) if `grow`, else erode (AND).
fn morph_sep(mask: &[bool], w: usize, h: usize, r: usize, grow: bool) -> Vec<bool> {
    let want = grow; // dilate looks for any `true`; erode looks for any `false`
    let mut tmp = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let x0 = x.saturating_sub(r);
            let x1 = (x + r).min(w - 1);
            let hit = (x0..=x1).any(|xx| mask[y * w + xx] == want);
            tmp[y * w + x] = if grow { hit } else { !hit };
        }
    }
    let mut out = vec![false; w * h];
    for x in 0..w {
        for y in 0..h {
            let y0 = y.saturating_sub(r);
            let y1 = (y + r).min(h - 1);
            let hit = (y0..=y1).any(|yy| tmp[yy * w + x] == want);
            out[y * w + x] = if grow { hit } else { !hit };
        }
    }
    out
}

// ── per-tile robust z (one-sided lower MAD + scale floor) ────────────────────

pub fn robust_z(map: &[f32], valid: &[bool], min_scale: f32) -> Vec<f32> {
    let mut vals: Vec<f32> = map.iter().zip(valid).filter(|(_, &ok)| ok).map(|(&m, _)| m).collect();
    if vals.is_empty() {
        return vec![0.0; map.len()];
    }
    let med = median(&mut vals);
    let mut lower: Vec<f32> = vals.iter().filter(|&&v| v <= med).map(|&v| med - v).collect();
    let mad = if lower.is_empty() { 0.0 } else { median(&mut lower) };
    let scale = (1.4826 * mad).max(min_scale) + 1e-6;
    map.iter()
        .zip(valid)
        .map(|(&m, &ok)| if ok { (m - med) / scale } else { 0.0 })
        .collect()
}

/// Unfloored per-tile robust scale 1.4826 * MAD(lower half) — calibration pools
/// these over healthy tiles to set the per-channel scale floor.
pub fn robust_scale(map: &[f32], valid: &[bool]) -> f32 {
    let mut vals: Vec<f32> = map.iter().zip(valid).filter(|(_, &ok)| ok).map(|(&m, _)| m).collect();
    if vals.is_empty() {
        return 0.0;
    }
    let med = median(&mut vals);
    let mut lower: Vec<f32> = vals.iter().filter(|&&v| v <= med).map(|&v| med - v).collect();
    let mad = if lower.is_empty() { 0.0 } else { median(&mut lower) };
    1.4826 * mad
}

fn median(v: &mut [f32]) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

// ── bilinear upscale (align_corners = false, matching cv2/torch default) ──────

pub fn upscale(src: &[f32], gw: usize, gh: usize, ow: usize, oh: usize) -> Vec<f32> {
    if gw == ow && gh == oh {
        return src.to_vec();
    }
    let mut out = vec![0f32; ow * oh];
    let sx = gw as f32 / ow as f32;
    let sy = gh as f32 / oh as f32;
    for oy in 0..oh {
        let fy = ((oy as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(gh - 1);
        let wy = fy - y0 as f32;
        for ox in 0..ow {
            let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(gw - 1);
            let wx = fx - x0 as f32;
            let v00 = src[y0 * gw + x0];
            let v01 = src[y0 * gw + x1];
            let v10 = src[y1 * gw + x0];
            let v11 = src[y1 * gw + x1];
            let top = v00 * (1.0 - wx) + v01 * wx;
            let bot = v10 * (1.0 - wx) + v11 * wx;
            out[oy * ow + ox] = top * (1.0 - wy) + bot * wy;
        }
    }
    out
}

// ── connected components (8-connectivity) ────────────────────────────────────

fn connected_components(mask: &[bool], w: usize, h: usize) -> (Vec<u32>, usize) {
    let mut labels = vec![0u32; w * h];
    let mut next = 0u32;
    let mut stack = Vec::new();
    for start in 0..w * h {
        if !mask[start] || labels[start] != 0 {
            continue;
        }
        next += 1;
        labels[start] = next;
        stack.push(start);
        while let Some(p) = stack.pop() {
            let (px, py) = ((p % w) as i32, (p / w) as i32);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = px + dx;
                    let ny = py + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let np = ny as usize * w + nx as usize;
                    if mask[np] && labels[np] == 0 {
                        labels[np] = next;
                        stack.push(np);
                    }
                }
            }
        }
    }
    (labels, next as usize)
}

/// Hysteresis: keep low-threshold components that contain a high seed.
fn grow_from_seeds(cand: &[bool], seeds: &[bool], w: usize, h: usize) -> Vec<bool> {
    let (lbl, n) = connected_components(cand, w, h);
    if n == 0 {
        return vec![false; w * h];
    }
    let mut keep = vec![false; n + 1];
    for i in 0..w * h {
        if seeds[i] && lbl[i] != 0 {
            keep[lbl[i] as usize] = true;
        }
    }
    (0..w * h).map(|i| lbl[i] != 0 && keep[lbl[i] as usize]).collect()
}

pub fn extract_regions(mask: &[bool], w: usize, h: usize, min_area: u32) -> Vec<Region> {
    let (lbl, n) = connected_components(mask, w, h);
    let mut acc: Vec<(u32, u32, u32, u32, u32, f64, f64)> = vec![(u32::MAX, u32::MAX, 0, 0, 0, 0.0, 0.0); n + 1];
    // (minx, miny, maxx, maxy, area, sumx, sumy)
    for y in 0..h {
        for x in 0..w {
            let l = lbl[y * w + x] as usize;
            if l == 0 {
                continue;
            }
            let a = &mut acc[l];
            a.0 = a.0.min(x as u32);
            a.1 = a.1.min(y as u32);
            a.2 = a.2.max(x as u32);
            a.3 = a.3.max(y as u32);
            a.4 += 1;
            a.5 += x as f64;
            a.6 += y as f64;
        }
    }
    let mut out = Vec::new();
    for l in 1..=n {
        let (minx, miny, maxx, maxy, area, sx, sy) = acc[l];
        if area < min_area {
            continue;
        }
        let bw = maxx - minx + 1;
        let bh = maxy - miny + 1;
        let mut rmask = vec![false; (bw * bh) as usize];
        for y in miny..=maxy {
            for x in minx..=maxx {
                if lbl[y as usize * w + x as usize] as usize == l {
                    rmask[((y - miny) * bw + (x - minx)) as usize] = true;
                }
            }
        }
        out.push(Region {
            bbox: [minx, miny, bw, bh],
            area,
            centroid: [(sx / area as f64) as f32, (sy / area as f64) as f32],
            mask: rmask,
            descriptor: [0.0; 8],
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tabs::pipeline::bank::{CoresetBank, GpuBank};
    use burn::backend::NdArray;
    use burn::tensor::backend::Backend;
    use std::path::PathBuf;
    use std::time::Instant;

    const ASSETS: &str = r"E:\PhD_TobiMu\02_code\02paper\anomaly\rust_assets";
    const ONNX:   &str = r"E:\PhD_TobiMu\02_code\02paper\anomaly\dinov3_vitb16_512.onnx";

    /// End-to-end DINO detect on one tile against the real assets.
    /// cargo test --release --no-default-features detect_smoke -- --ignored --nocapture
    #[test]
    #[ignore]
    fn detect_smoke() {
        let onnx = PathBuf::from(ONNX);
        let bank_p = PathBuf::from(ASSETS).join("coreset_bank.bin");
        let meta_p = PathBuf::from(ASSETS).join("detector_meta.json");
        // override with DETECT_SMOKE_IMG; default = a healthy in-domain tile
        let image = PathBuf::from(std::env::var("DETECT_SMOKE_IMG").unwrap_or_else(|_| {
            r"E:\PhD_TobiMu\01_data\raw\SamplingSaturday\output256\Acer_l2nr002_2_3.png".to_string()
        }));
        for p in [&onnx, &bank_p, &meta_p, &image] {
            assert!(p.exists(), "missing: {}", p.display());
        }

        let t = Instant::now();
        let mut dino = DinoExtractor::load(&onnx, 512).expect("dino");
        let bank_cpu = CoresetBank::load(&bank_p).expect("bank");
        let meta = DetectorMeta::load(&meta_p).expect("meta");
        // GPU GEMM path on the NdArray backend here; Cuda in the default build.
        let device = <NdArray as Backend>::Device::default();
        let bank = GpuBank::<NdArray>::new(&bank_cpu, device);
        println!("loaded (bank N={} D={}) in {:?}", bank_cpu.n, bank_cpu.d, t.elapsed());

        let img = image::open(&image).expect("open").to_rgb8();
        let t2 = Instant::now();
        let r = detect(&mut dino, &bank, &meta, &DetectParams::default(), &img, 256, 256, None)
            .expect("detect");
        let dt = t2.elapsed();

        let (mut lo, mut hi, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
        for &v in &r.dino_map {
            lo = lo.min(v);
            hi = hi.max(v);
            sum += v as f64;
        }
        let flagged = r.mask.iter().filter(|&&b| b).count();
        println!(
            "detect in {dt:?} | dino_map[{lo:.3}..{hi:.3}] mean={:.3} | z_thr(dino)={:.2} | flagged {:.2}% | {} regions",
            sum / r.dino_map.len() as f64,
            meta.ch_threshold("dino"),
            100.0 * flagged as f32 / r.mask.len() as f32,
            r.regions.len(),
        );
        assert_eq!(r.dino_map.len(), 256 * 256);
        assert!(r.dino_map.iter().all(|v| v.is_finite()));
    }
}
