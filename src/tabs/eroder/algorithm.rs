use rand::Rng;
use rand::seq::SliceRandom;
use std::f32::consts::TAU;
use std::sync::atomic::{AtomicBool, Ordering};

// ── helpers ───────────────────────────────────────────────────────────────────

#[inline]
fn idx(x: usize, y: usize, w: usize) -> usize { y * w + x }

/// Returns true if the pixel at (x,y) is land but has at least one non-land
/// neighbour (8-connectivity).
fn is_coastal(mask: &[bool], x: usize, y: usize, w: usize, h: usize) -> bool {
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dx == 0 && dy == 0 { continue; }
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx < 0 || nx >= w as i32 || ny < 0 || ny >= h as i32 {
                return true; // image edge counts as "sea"
            }
            if !mask[idx(nx as usize, ny as usize, w)] {
                return true;
            }
        }
    }
    false
}

/// Accumulate immediate land-neighbours into the probability layer.
fn update_neighbours(
    mask: &[bool],
    prob: &mut [f32],
    x: usize, y: usize, w: usize, h: usize,
    increment: f32,
) {
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dx == 0 && dy == 0 { continue; }
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx < 0 || nx >= w as i32 || ny < 0 || ny >= h as i32 { continue; }
            let ni = idx(nx as usize, ny as usize, w);
            if mask[ni] { prob[ni] = (prob[ni] + increment).min(1.0); }
        }
    }
}

/// Centroid of all land pixels.
fn center_of_mass(mask: &[bool], w: usize, h: usize) -> (f32, f32) {
    let (mut sx, mut sy, mut n) = (0f64, 0f64, 0usize);
    for i in 0..w * h {
        if mask[i] { sx += (i % w) as f64; sy += (i / w) as f64; n += 1; }
    }
    if n == 0 { return (w as f32 / 2.0, h as f32 / 2.0); }
    ((sx / n as f64) as f32, (sy / n as f64) as f32)
}

// ── Algorithm A: coastal / edge erosion ──────────────────────────────────────

pub fn erode_coastal(
    mask: &mut Vec<bool>,
    w: usize, h: usize,
    fraction: f32,
    erosion_prob_start: f32,
    rng: &mut impl Rng,
) {
    let total_land = mask.iter().filter(|&&b| b).count();
    let target = ((total_land as f32) * fraction) as usize;
    if target == 0 { return; }

    let mut prob = vec![0.0f32; w * h];
    let mut land: Vec<usize> = (0..w * h).filter(|&i| mask[i]).collect();

    for &i in &land {
        if is_coastal(mask, i % w, i / w, w, h) { prob[i] = erosion_prob_start; }
    }

    let mut eroded = 0usize;

    while eroded < target {
        land.shuffle(rng);
        let prev = eroded;

        for &i in &land {
            if !mask[i] { continue; }
            if rng.gen::<f32>() < prob[i] {
                mask[i] = false;
                eroded += 1;
                update_neighbours(mask, &mut prob, i % w, i / w, w, h, 0.5);
                if eroded >= target { return; }
            }
        }

        if eroded == prev {
            // No progress — seed 1–3 random coastal pixels to spark cascades.
            // Bumping ALL coastal probs homogenises erosion around the whole
            // margin; seeding a small random subset preserves the localised
            // bite behaviour that matches real herbivory.
            let coastal: Vec<usize> = land.iter().copied()
                .filter(|&i| mask[i] && is_coastal(mask, i % w, i / w, w, h))
                .collect();
            if coastal.is_empty() { break; }
            let n_seeds = rng.gen_range(1usize..=3).min(coastal.len());
            for &i in coastal.choose_multiple(rng, n_seeds) {
                prob[i] = 0.7;
            }
        }
    }
}

// ── Algorithm B: organic interior spots ──────────────────────────────────────
//
// Angular-harmonic blobs with satellite blobs and per-pixel boundary noise.
// Produces highly irregular, organic-looking feeding holes.

/// Core blob stamper — reused for main blobs and satellites.
fn erode_blob(
    mask: &mut Vec<bool>,
    w: usize, h: usize,
    cx: i32, cy: i32,
    base_r: f32,
    aspect_x: f32, aspect_y: f32,
    cos_r: f32, sin_r: f32,
    harmonics: &[(f32, f32, f32)],
    rng: &mut impl Rng,
    eroded: &mut usize,
    target: usize,
) {
    let max_r = (base_r * 2.0) as i32 + 3;

    for dy in -max_r..=max_r {
        for dx in -max_r..=max_r {
            let fx = dx as f32;
            let fy = dy as f32;

            // Rotate into ellipse frame
            let rx =  fx * cos_r + fy * sin_r;
            let ry = -fx * sin_r + fy * cos_r;

            // Scale by aspect ratio
            let sx = rx / aspect_x;
            let sy = ry / aspect_y;
            let dist  = (sx * sx + sy * sy).sqrt();
            let angle = ry.atan2(rx);

            // Organic lobe modulation
            let modulation: f32 = harmonics.iter()
                .map(|(freq, amp, phase)| amp * (freq * angle + phase).cos())
                .sum();

            // Per-pixel boundary noise — roughens the silhouette organically
            let noise = rng.gen_range(-0.18..=0.18_f32);
            let effective_r = base_r * (1.0 + modulation + noise).max(0.08);
            if dist > effective_r { continue; }

            let nx = cx + dx;
            let ny = cy + dy;
            if nx < 0 || nx >= w as i32 || ny < 0 || ny >= h as i32 { continue; }
            let ni = ny as usize * w + nx as usize;
            if mask[ni] {
                mask[ni] = false;
                *eroded += 1;
                if *eroded >= target { return; }
            }
        }
    }
}

pub fn erode_spots(
    mask: &mut Vec<bool>,
    w: usize, h: usize,
    fraction: f32,
    rng: &mut impl Rng,
) {
    let land: Vec<usize> = (0..w * h).filter(|&i| mask[i]).collect();
    let total_land = land.len();
    let target = ((total_land as f32) * fraction) as usize;
    if target == 0 || land.is_empty() { return; }

    let interior: Vec<usize> = land.iter().copied()
        .filter(|&i| !is_coastal(mask, i % w, i / w, w, h))
        .collect();
    let candidates = if interior.is_empty() { &land[..] } else { &interior[..] };

    let mut eroded = 0usize;
    let mut stall  = 0usize;  // consecutive blob placements with zero new pixels

    while eroded < target {
        if stall >= 8 { break; }  // all reachable pixels already eroded

        let prev_eroded = eroded;
        let ci  = candidates[rng.gen_range(0..candidates.len())];
        let cx  = (ci % w) as i32;
        let cy  = (ci / w) as i32;

        let base_r = rng.gen_range(4..=18) as f32;

        // Random ellipse axes and rotation
        let aspect_x: f32 = rng.gen_range(0.45..=1.9);
        let aspect_y: f32 = rng.gen_range(0.45..=1.9);
        let rot: f32 = rng.gen::<f32>() * TAU;
        let cos_r = rot.cos();
        let sin_r = rot.sin();

        // Angular harmonics (3–7 lobes, larger amplitudes for more distortion)
        let n_harm = rng.gen_range(3usize..=7);
        let harmonics: Vec<(f32, f32, f32)> = (0..n_harm)
            .map(|_| (
                rng.gen_range(2..=8) as f32,
                rng.gen_range(0.12..=0.55_f32),
                rng.gen::<f32>() * TAU,
            ))
            .collect();

        erode_blob(mask, w, h, cx, cy, base_r, aspect_x, aspect_y, cos_r, sin_r,
                   &harmonics, rng, &mut eroded, target);
        if eroded >= target { return; }

        // 0–2 satellite blobs (~65 % chance each) for cluster-like appearance
        let n_sat = if rng.gen::<f32>() < 0.65 { rng.gen_range(1usize..=2) } else { 0 };
        for _ in 0..n_sat {
            if eroded >= target { return; }
            let offset_r   = base_r * rng.gen_range(0.55..=1.6);
            let offset_ang = rng.gen::<f32>() * TAU;
            let sat_cx = cx + (offset_r * offset_ang.cos()) as i32;
            let sat_cy = cy + (offset_r * offset_ang.sin()) as i32;
            if sat_cx < 0 || sat_cx >= w as i32 || sat_cy < 0 || sat_cy >= h as i32 { continue; }

            let sat_r   = base_r * rng.gen_range(0.25..=0.6);
            let sat_ax  = rng.gen_range(0.5..=1.5_f32);
            let sat_ay  = rng.gen_range(0.5..=1.5_f32);
            let sat_rot = rng.gen::<f32>() * TAU;
            let n_sh    = rng.gen_range(2usize..=4);
            let sat_harmonics: Vec<(f32, f32, f32)> = (0..n_sh)
                .map(|_| (
                    rng.gen_range(2..=5) as f32,
                    rng.gen_range(0.10..=0.40_f32),
                    rng.gen::<f32>() * TAU,
                ))
                .collect();

            erode_blob(mask, w, h, sat_cx, sat_cy, sat_r,
                       sat_ax, sat_ay, sat_rot.cos(), sat_rot.sin(),
                       &sat_harmonics, rng, &mut eroded, target);
        }
        if eroded == prev_eroded { stall += 1; } else { stall = 0; }
    }
}

// ── Algorithm C: margin-snake (tapered notch) ─────────────────────────────────

pub fn erode_margin_snake(
    mask: &mut Vec<bool>,
    w: usize, h: usize,
    fraction: f32,
    rng: &mut impl Rng,
) {
    let land: Vec<usize> = (0..w * h).filter(|&i| mask[i]).collect();
    let total_land = land.len();
    let target = ((total_land as f32) * fraction) as usize;
    if target == 0 || land.is_empty() { return; }

    let coastal: Vec<usize> = land.iter().copied()
        .filter(|&i| is_coastal(mask, i % w, i / w, w, h))
        .collect();
    if coastal.is_empty() { return; }

    let (com_x, com_y) = center_of_mass(mask, w, h);

    let mut eroded   = 0usize;
    let mut no_start = 0usize;  // consecutive tries that hit an already-eroded start

    while eroded < target {
        let start = coastal[rng.gen_range(0..coastal.len())];
        if !mask[start] {
            no_start += 1;
            if no_start > coastal.len() * 10 { break; }
            continue;
        }
        no_start = 0;

        let sx = (start % w) as f32;
        let sy = (start / w) as f32;

        let mut dir_x = com_x - sx;
        let mut dir_y = com_y - sy;
        let dir_len = (dir_x * dir_x + dir_y * dir_y).sqrt();
        if dir_len < 1.0 { continue; }
        dir_x /= dir_len;
        dir_y /= dir_len;

        let start_radius: f32 = rng.gen_range(8.0..=20.0);
        let steps:        usize = rng.gen_range(10..=30);
        let step_size:    f32 = rng.gen_range(2.5..=6.0);
        let wobble:       f32 = rng.gen_range(0.25..=0.65);

        let mut px = sx;
        let mut py = sy;
        let mut cur_dx = dir_x;
        let mut cur_dy = dir_y;

        for step in 0..steps {
            let t = step as f32 / steps as f32;
            let radius = (start_radius * (1.0 - t * 0.88)).max(2.0);
            let r = radius as i32;

            for dy in -r..=r {
                for dx in -r..=r {
                    if dx * dx + dy * dy > r * r { continue; }
                    let nx = px as i32 + dx;
                    let ny = py as i32 + dy;
                    if nx < 0 || nx >= w as i32 || ny < 0 || ny >= h as i32 { continue; }
                    let ni = ny as usize * w + nx as usize;
                    if mask[ni] {
                        mask[ni] = false;
                        eroded += 1;
                        if eroded >= target { return; }
                    }
                }
            }

            let angle = rng.gen_range(-wobble..=wobble);
            let (sin_a, cos_a) = angle.sin_cos();
            let new_dx =  cur_dx * cos_a + cur_dy * sin_a;
            let new_dy = -cur_dx * sin_a + cur_dy * cos_a;
            let len = (new_dx * new_dx + new_dy * new_dy).sqrt();
            if len > 0.001 {
                cur_dx = new_dx / len;
                cur_dy = new_dy / len;
            }

            px += cur_dx * step_size;
            py += cur_dy * step_size;

            if px < 0.0 || py < 0.0 || px >= w as f32 || py >= h as f32 { break; }
            let pi = py as usize * w + px as usize;
            if !mask[pi] { break; }
        }
    }
}

// ── Algorithm D: interior ellipses ────────────────────────────────────────────
//
// Stamps large, randomly-rotated ellipses at ANY position on the leaf
// (not just the margin). Produces the large contiguous interior holes that
// coastal / spots / snake never generate, teaching the model to reconstruct
// from minimal surviving evidence.

pub fn erode_interior_ellipses(
    mask: &mut Vec<bool>,
    w: usize, h: usize,
    fraction: f32,
    rng: &mut impl Rng,
) {
    let land: Vec<usize> = (0..w * h).filter(|&i| mask[i]).collect();
    let total = land.len();
    let target = ((total as f32) * fraction) as usize;
    if target == 0 || land.is_empty() { return; }

    let mut eroded = 0usize;
    let mut stall  = 0usize;  // consecutive ellipses with zero new pixels

    while eroded < target {
        if stall >= 8 { break; }

        let prev_eroded = eroded;
        let ci = land[rng.gen_range(0..land.len())];
        let cx = (ci % w) as i32;
        let cy = (ci / w) as i32;

        let base_r: f32    = rng.gen_range(15..=50) as f32;
        let aspect_x: f32  = rng.gen_range(0.5..=2.0);
        let aspect_y: f32  = rng.gen_range(0.5..=2.0);
        let rot: f32       = rng.gen::<f32>() * TAU;
        let (sin_r, cos_r) = rot.sin_cos();

        let max_r = (base_r * aspect_x.max(aspect_y) * 1.1) as i32 + 2;
        for dy in -max_r..=max_r {
            for dx in -max_r..=max_r {
                let rx  =  dx as f32 * cos_r + dy as f32 * sin_r;
                let ry  = -dx as f32 * sin_r + dy as f32 * cos_r;
                let dist = ((rx / aspect_x).powi(2) + (ry / aspect_y).powi(2)).sqrt();
                if dist > base_r { continue; }

                let nx = cx + dx;
                let ny = cy + dy;
                if nx < 0 || nx >= w as i32 || ny < 0 || ny >= h as i32 { continue; }
                let ni = ny as usize * w + nx as usize;
                if mask[ni] {
                    mask[ni] = false;
                    eroded += 1;
                    if eroded >= target { return; }
                }
            }
        }
        if eroded == prev_eroded { stall += 1; } else { stall = 0; }
    }
}

// ── Algorithm E: apex / tip removal ───────────────────────────────────────────
//
// Removes a rectangular strip from one randomly-chosen side of the leaf
// bounding box. Simulates tip burn, apical necrosis, and edge trimming.
// `cut_fraction` is the proportion of the bbox height/width to remove (0–1).

pub fn erode_apex(
    mask: &mut Vec<bool>,
    w: usize, h: usize,
    cut_fraction: f32,
    rng: &mut impl Rng,
) {
    let land: Vec<usize> = (0..w * h).filter(|&i| mask[i]).collect();
    if land.is_empty() { return; }

    let min_x = land.iter().map(|&i| i % w).min().unwrap() as f32;
    let max_x = land.iter().map(|&i| i % w).max().unwrap() as f32;
    let min_y = land.iter().map(|&i| i / w).min().unwrap() as f32;
    let max_y = land.iter().map(|&i| i / w).max().unwrap() as f32;

    let bbox_w = (max_x - min_x).max(1.0);
    let bbox_h = (max_y - min_y).max(1.0);

    // 0=top, 1=bottom, 2=left, 3=right
    let dir = rng.gen_range(0u8..4);

    for i in 0..w * h {
        if !mask[i] { continue; }
        let x = (i % w) as f32;
        let y = (i / w) as f32;
        let remove = match dir {
            0 => (y - min_y) / bbox_h < cut_fraction,
            1 => (max_y - y) / bbox_h < cut_fraction,
            2 => (x - min_x) / bbox_w < cut_fraction,
            _ => (max_x - x) / bbox_w < cut_fraction,
        };
        if remove { mask[i] = false; }
    }
}

// ── Algorithm F: clustered margin damage ──────────────────────────────────────
//
// Simulates natural herbivory patterns where an insect feeds intensively at
// 1–3 spots along the leaf margin, creating irregularly shaped bite clusters
// rather than the uniform ring erosion of Algorithm A.
//
// Strategy: pick 1–3 random border pixels as cluster centres, then initialise
// coastal erosion probabilities with a Gaussian boost near each centre (high
// probability ≈ 0.85 at the centre, tapering to ≈ 0.02 far away). The same
// propagation mechanism as erode_coastal is used so erosion always advances
// inward from the margin, preserving biological realism.

pub fn erode_margin_clusters(
    mask: &mut Vec<bool>,
    w: usize, h: usize,
    fraction: f32,
    rng: &mut impl Rng,
) {
    let total_land = mask.iter().filter(|&&b| b).count();
    let target = ((total_land as f32) * fraction) as usize;
    if target == 0 { return; }

    // Collect border pixels (leaf pixels with at least one non-leaf neighbour).
    let border: Vec<usize> = (0..w * h)
        .filter(|&i| mask[i] && is_coastal(mask, i % w, i / w, w, h))
        .collect();
    if border.is_empty() { return; }

    // Pick 1–3 cluster centres from the border.
    let n_clusters: usize = rng.gen_range(1usize..=3);
    let centers: Vec<(f32, f32)> = (0..n_clusters)
        .map(|_| {
            let ci = border[rng.gen_range(0..border.len())];
            ((ci % w) as f32, (ci / w) as f32)
        })
        .collect();

    // Cluster radius: 10–22 % of the smaller image dimension.
    // Larger radius → wider, shallower cluster; smaller → tight deep bite.
    let base_r  = (w.min(h) as f32) * rng.gen_range(0.10_f32..=0.22);
    let two_r2  = 2.0 * base_r * base_r;

    // Initialise erosion probabilities for border pixels only.
    let mut prob = vec![0.0f32; w * h];
    for &i in &border {
        let px = (i % w) as f32;
        let py = (i / w) as f32;
        let min_dsq = centers.iter()
            .map(|&(cx, cy)| (px - cx).powi(2) + (py - cy).powi(2))
            .fold(f32::MAX, f32::min);
        // 0.85 at cluster centre, ≈ 0.02 at 2× radius distance.
        prob[i] = 0.02 + 0.83 * (-min_dsq / two_r2).exp();
    }

    // Propagate erosion inward from the margin (identical to erode_coastal).
    let mut land: Vec<usize> = (0..w * h).filter(|&i| mask[i]).collect();
    let mut eroded      = 0;
    let mut no_progress = 0usize;
    while eroded < target {
        land.retain(|&i| mask[i]);
        if land.is_empty() { break; }
        let prev = eroded;
        land.shuffle(rng);
        for &i in &land {
            if !mask[i] { continue; }
            if rng.gen::<f32>() < prob[i] {
                mask[i] = false;
                eroded += 1;
                update_neighbours(mask, &mut prob, i % w, i / w, w, h, 0.45);
                if eroded >= target { return; }
            }
        }
        if eroded == prev {
            no_progress += 1;
            if no_progress >= 30 { break; }
            // Nudge coastal pixels so low-prob regions can still be reached.
            for &i in &land {
                if mask[i] && is_coastal(mask, i % w, i / w, w, h) {
                    prob[i] = (prob[i] + 0.06).min(1.0);
                }
            }
        } else {
            no_progress = 0;
        }
    }
}

// ── Algorithm G: whole-lobe removal ───────────────────────────────────────────
//
// Removes 1–3 circular lobes from the leaf margin — each lobe is a disc
// centred on a random border pixel.  Radius is chosen so the disc removes
// roughly (fraction / n_lobes) of the leaf area.  Simulates caterpillars or
// beetles consuming entire leaf lobes rather than just nibbling the margin.

pub fn erode_lobe(
    mask: &mut Vec<bool>,
    w: usize, h: usize,
    fraction: f32,
    rng: &mut impl Rng,
) {
    let total_land = mask.iter().filter(|&&b| b).count();
    let target = ((total_land as f32) * fraction) as usize;
    if target == 0 { return; }

    let initial_border: Vec<usize> = (0..w * h)
        .filter(|&i| mask[i] && is_coastal(mask, i % w, i / w, w, h))
        .collect();
    if initial_border.is_empty() { return; }

    let n_lobes: usize = rng.gen_range(1usize..=3);
    let mut eroded = 0usize;

    // Leaf centroid — used for angular distance calculations in adjacent mode.
    let centroid_x = (0..w * h).filter(|&i| mask[i]).map(|i| (i % w) as f32).sum::<f32>()
        / total_land.max(1) as f32;
    let centroid_y = (0..w * h).filter(|&i| mask[i]).map(|i| (i / w) as f32).sum::<f32>()
        / total_land.max(1) as f32;

    // Adjacent mode: 60 % of multi-lobe calls place subsequent lobes within
    // ±90 ° of the first lobe's angle from the centroid.  This creates the
    // "two neighbouring lobes removed" pattern the model must learn to restore.
    let adjacent_mode = n_lobes >= 2 && rng.gen::<f32>() < 0.60;
    let mut first_angle: Option<f32> = None;

    for lobe_idx in 0..n_lobes {
        if eroded >= target { break; }

        // Re-collect border after each lobe so next centre lands on live margin
        let border: Vec<usize> = if lobe_idx == 0 {
            initial_border.clone()
        } else {
            (0..w * h).filter(|&i| mask[i] && is_coastal(mask, i % w, i / w, w, h)).collect()
        };
        if border.is_empty() { break; }

        // In adjacent mode, constrain lobe 2+ to within ±90° of the first lobe.
        let ci = if adjacent_mode && lobe_idx > 0 {
            if let Some(ref_angle) = first_angle {
                let window = std::f32::consts::FRAC_PI_2; // 90 °
                let candidates: Vec<usize> = border.iter().filter(|&&bi| {
                    let bx = (bi % w) as f32 - centroid_x;
                    let by = (bi / w) as f32 - centroid_y;
                    let a = f32::atan2(by, bx);
                    let mut diff = (a - ref_angle).abs();
                    if diff > std::f32::consts::PI { diff = std::f32::consts::TAU - diff; }
                    diff <= window
                }).copied().collect();
                if candidates.is_empty() {
                    border[rng.gen_range(0..border.len())]
                } else {
                    candidates[rng.gen_range(0..candidates.len())]
                }
            } else {
                border[rng.gen_range(0..border.len())]
            }
        } else {
            border[rng.gen_range(0..border.len())]
        };

        let cx = (ci % w) as i32;
        let cy = (ci / w) as i32;

        // Record angle of first lobe for subsequent adjacent placements.
        if lobe_idx == 0 {
            let bx = cx as f32 - centroid_x;
            let by = cy as f32 - centroid_y;
            first_angle = Some(f32::atan2(by, bx));
        }

        // Radius so disc area ≈ target_px / remaining_lobes
        let remaining = (n_lobes - lobe_idx).max(1);
        let per_lobe  = (target.saturating_sub(eroded) / remaining).max(1);
        let base_r    = ((per_lobe as f32 / std::f32::consts::PI).sqrt() * 1.3)
            .clamp(8.0, 80.0) as i32;

        for dy in -base_r..=base_r {
            for dx in -base_r..=base_r {
                if dx * dx + dy * dy > base_r * base_r { continue; }
                let nx = cx + dx;
                let ny = cy + dy;
                if nx < 0 || nx >= w as i32 || ny < 0 || ny >= h as i32 { continue; }
                let ni = ny as usize * w + nx as usize;
                if mask[ni] {
                    mask[ni] = false;
                    eroded += 1;
                    if eroded >= target { return; }
                }
            }
        }
    }
}

// ── Focal sector damage ───────────────────────────────────────────────────────
// Removes a focused half-plane sector of the leaf.  A random direction is
// chosen; all leaf pixels are ranked by projection onto that direction and the
// top `fraction` are erased.  Simulates concentrated single-side damage (e.g.,
// 30-40 % of one lobe eaten away) that distributed primitives under-represent.

pub fn erode_focal_sector(
    mask:     &mut Vec<bool>,
    w:        usize,
    h:        usize,
    fraction: f32,
    rng:      &mut impl Rng,
) {
    let mut leaf_px: Vec<usize> = (0..w * h).filter(|&i| mask[i]).collect();
    let n_leaf = leaf_px.len();
    if n_leaf < 20 { return; }

    let cx = leaf_px.iter().map(|&i| (i % w) as f32).sum::<f32>() / n_leaf as f32;
    let cy = leaf_px.iter().map(|&i| (i / w) as f32).sum::<f32>() / n_leaf as f32;

    let angle = rng.gen::<f32>() * TAU;
    let (nx, ny) = (angle.cos(), angle.sin());

    leaf_px.sort_by(|&a, &b| {
        let pa = ((a % w) as f32 - cx) * nx + ((a / w) as f32 - cy) * ny;
        let pb = ((b % w) as f32 - cx) * nx + ((b / w) as f32 - cy) * ny;
        pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
    });

    let n_remove = ((n_leaf as f32 * fraction) as usize).min(n_leaf);
    for &idx in leaf_px.iter().take(n_remove) {
        mask[idx] = false;
    }
}

// ── Smoothing ─────────────────────────────────────────────────────────────────

pub fn smooth_edges(mask: &mut Vec<bool>, w: usize, h: usize, iterations: u32) {
    let mut floats: Vec<f32> = mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
    let mut tmp = floats.clone();

    for _ in 0..iterations {
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0f32;
                let mut cnt = 0u32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                            sum += floats[idx(nx as usize, ny as usize, w)];
                            cnt += 1;
                        }
                    }
                }
                tmp[idx(x, y, w)] = sum / cnt as f32;
            }
        }
        std::mem::swap(&mut floats, &mut tmp);
    }

    for (b, f) in mask.iter_mut().zip(floats.iter()) { *b = *f > 0.5; }
}

// ── Resize spec ───────────────────────────────────────────────────────────────

pub struct ResizeSpec {
    pub use_percent: bool,
    pub percent:     f32,   // e.g. 50.0 = 50 %
    pub max_dim:     u32,   // longest side target when !use_percent
}

impl ResizeSpec {
    pub fn apply(&self, img: image::DynamicImage) -> image::DynamicImage {
        let (w, h) = (img.width(), img.height());
        let (new_w, new_h) = if self.use_percent {
            let s = (self.percent / 100.0).max(0.01);
            ((w as f32 * s).round().max(1.0) as u32,
             (h as f32 * s).round().max(1.0) as u32)
        } else {
            let m = self.max_dim.max(1);
            if w >= h {
                let new_h = (h as f64 * m as f64 / w as f64).round() as u32;
                (m, new_h.max(1))
            } else {
                let new_w = (w as f64 * m as f64 / h as f64).round() as u32;
                (new_w.max(1), m)
            }
        };
        img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3)
    }
}

// ── Algorithm params & single-image processing ───────────────────────────────

pub struct EroderParams {
    pub damage_fractions:     Vec<f32>,
    pub erosion_prob:         f32,
    pub smoothing_iterations: u32,
    pub coastal_weight:       f32,
    pub spots_weight:         f32,
    pub snake_weight:         f32,
    pub ellipses_weight:      f32,
    pub clusters_weight:      f32,
    pub lobe_weight:          f32,
    /// Probability per damage level that apex damage is applied as a supplement
    pub apex_weight:          f32,
    pub coastal_enabled:      bool,
    pub spots_enabled:        bool,
    pub snake_enabled:        bool,
    pub ellipses_enabled:     bool,
    pub apex_enabled:         bool,
    pub clusters_enabled:     bool,
    pub lobe_enabled:         bool,
    /// Apply 12 % random border-pixel removal (matches training pipeline noise)
    pub boundary_noise:       bool,
    pub independent_outputs:  bool,
    pub seed:                 Option<u64>,
    pub resize:               Option<ResizeSpec>,
}

impl EroderParams {
    #[allow(dead_code)]
    pub fn from_levels(n: u32, max_pct: f32, _erosion_prob: f32, _smoothing: u32) -> Vec<f32> {
        (1..=n)
            .map(|i| (max_pct / n as f32) * i as f32 / 100.0)
            .collect()
    }
}

pub fn apply_mask_and_save(
    original_rgba: &[u8],
    mask: &[bool],
    w: u32, h: u32,
    out_path: &std::path::Path,
) -> Result<(), String> {
    use image::{ImageBuffer, Rgba};
    let mut out: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(w, h);
    for (i, pixel) in out.pixels_mut().enumerate() {
        let base = i * 4;
        // Eroded pixels must be fully zeroed (RGB + alpha) so the network cannot
        // see leaf texture beneath the transparent region.
        *pixel = if mask[i] {
            Rgba([original_rgba[base], original_rgba[base+1], original_rgba[base+2], original_rgba[base+3]])
        } else {
            Rgba([0, 0, 0, 0])
        };
    }
    out.save(out_path).map_err(|e| e.to_string())
}

pub fn process_image(
    path: &std::path::Path,
    params: &EroderParams,
    output_root: &std::path::Path,
    cancelled: &AtomicBool,
) -> Result<String, String> {
    if cancelled.load(Ordering::Relaxed) {
        return Err("cancelled".to_string());
    }

    let dyn_img = image::open(path).map_err(|e| format!("open: {}", e))?;

    // Apply optional resize before any algorithm
    let dyn_img = if let Some(spec) = &params.resize {
        spec.apply(dyn_img)
    } else {
        dyn_img
    };

    let rgba = dyn_img.to_rgba8();
    let (w, h) = rgba.dimensions();
    let raw = rgba.into_raw();

    let base_mask: Vec<bool> = raw.chunks_exact(4).map(|c| c[3] > 0).collect();

    let filename = path.file_name()
        .and_then(|n| n.to_str()).unwrap_or("image.png");

    // Normalise weights (apex is probabilistic supplement — not in the blend)
    let total_w = {
        let mut t = 0.0f32;
        if params.coastal_enabled  { t += params.coastal_weight;  }
        if params.spots_enabled    { t += params.spots_weight;    }
        if params.snake_enabled    { t += params.snake_weight;    }
        if params.ellipses_enabled { t += params.ellipses_weight; }
        if params.clusters_enabled { t += params.clusters_weight; }
        if params.lobe_enabled     { t += params.lobe_weight;     }
        if t == 0.0 { t = 1.0; }
        t
    };
    let coastal_frac  = if params.coastal_enabled  { params.coastal_weight  / total_w } else { 0.0 };
    let spots_frac    = if params.spots_enabled    { params.spots_weight    / total_w } else { 0.0 };
    let snake_frac    = if params.snake_enabled    { params.snake_weight    / total_w } else { 0.0 };
    let ellipses_frac = if params.ellipses_enabled { params.ellipses_weight / total_w } else { 0.0 };
    let clusters_frac = if params.clusters_enabled { params.clusters_weight / total_w } else { 0.0 };
    let lobe_frac     = if params.lobe_enabled     { params.lobe_weight     / total_w } else { 0.0 };

    for &frac in &params.damage_fractions {
        if cancelled.load(Ordering::Relaxed) {
            return Err("cancelled".to_string());
        }

        let folder_name = format!("{:02}", (frac * 100.0).round() as u32);

        if params.independent_outputs {
            macro_rules! run_independent {
                ($alg:expr, $sub:expr) => {{
                    let mut mask = base_mask.clone();
                    let mut rng  = make_rng(params.seed);
                    $alg(&mut mask, &mut rng);
                    smooth_edges(&mut mask, w as usize, h as usize, params.smoothing_iterations);
                    let dir = output_root.join(&folder_name).join($sub);
                    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
                    apply_mask_and_save(&raw, &mask, w, h, &dir.join(filename))?;
                }};
            }
            if params.coastal_enabled {
                run_independent!(|m: &mut Vec<bool>, r: &mut _| erode_coastal(m, w as usize, h as usize, frac, params.erosion_prob, r), "coastal");
            }
            if params.spots_enabled {
                run_independent!(|m: &mut Vec<bool>, r: &mut _| erode_spots(m, w as usize, h as usize, frac, r), "spots");
            }
            if params.snake_enabled {
                run_independent!(|m: &mut Vec<bool>, r: &mut _| erode_margin_snake(m, w as usize, h as usize, frac, r), "snake");
            }
            if params.ellipses_enabled {
                run_independent!(|m: &mut Vec<bool>, r: &mut _| erode_interior_ellipses(m, w as usize, h as usize, frac, r), "ellipses");
            }
            if params.apex_enabled {
                let cut = (frac * 1.2_f32).clamp(0.08, 0.50);
                run_independent!(|m: &mut Vec<bool>, r: &mut _| erode_apex(m, w as usize, h as usize, cut, r), "apex");
            }
            if params.clusters_enabled {
                run_independent!(|m: &mut Vec<bool>, r: &mut _| erode_margin_clusters(m, w as usize, h as usize, frac, r), "clusters");
            }
            if params.lobe_enabled {
                run_independent!(|m: &mut Vec<bool>, r: &mut _| erode_lobe(m, w as usize, h as usize, frac, r), "lobe");
            }
        } else {
            let mut mask = base_mask.clone();
            let mut rng  = make_rng(params.seed);

            if params.coastal_enabled && coastal_frac > 0.0 {
                erode_coastal(&mut mask, w as usize, h as usize,
                    frac * coastal_frac, params.erosion_prob, &mut rng);
            }
            if params.spots_enabled && spots_frac > 0.0 {
                erode_spots(&mut mask, w as usize, h as usize, frac * spots_frac, &mut rng);
            }
            if params.snake_enabled && snake_frac > 0.0 {
                erode_margin_snake(&mut mask, w as usize, h as usize, frac * snake_frac, &mut rng);
            }
            if params.ellipses_enabled && ellipses_frac > 0.0 {
                erode_interior_ellipses(&mut mask, w as usize, h as usize, frac * ellipses_frac, &mut rng);
            }
            if params.clusters_enabled && clusters_frac > 0.0 {
                erode_margin_clusters(&mut mask, w as usize, h as usize, frac * clusters_frac, &mut rng);
            }
            if params.lobe_enabled && lobe_frac > 0.0 {
                erode_lobe(&mut mask, w as usize, h as usize, frac * lobe_frac, &mut rng);
            }
            // Apex: probabilistic supplement (same logic as training pipeline)
            if params.apex_enabled && rng.gen::<f32>() < params.apex_weight {
                let cut = (frac * 1.2_f32).clamp(0.08, 0.50);
                erode_apex(&mut mask, w as usize, h as usize, cut, &mut rng);
            }

            smooth_edges(&mut mask, w as usize, h as usize, params.smoothing_iterations);

            // Boundary noise: ~12% of border pixels removed (matches training pipeline)
            if params.boundary_noise {
                for i in 0..w as usize * h as usize {
                    if !mask[i] { continue; }
                    let x = i % w as usize;
                    let y = i / w as usize;
                    let is_border = (x > 0            && !mask[i - 1])
                        || (x + 1 < w as usize && !mask[i + 1])
                        || (y > 0            && !mask[i - w as usize])
                        || (y + 1 < h as usize && !mask[i + w as usize]);
                    if is_border && rng.gen::<f32>() < 0.12 {
                        mask[i] = false;
                    }
                }
            }

            let dir = output_root.join(&folder_name);
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            apply_mask_and_save(&raw, &mask, w, h, &dir.join(filename))?;
        }
    }

    Ok(format!("{} → {} level(s)", filename, params.damage_fractions.len()))
}

fn make_rng(seed: Option<u64>) -> impl Rng {
    use rand::SeedableRng;
    match seed {
        Some(s) => rand::rngs::SmallRng::seed_from_u64(s),
        None    => rand::rngs::SmallRng::from_entropy(),
    }
}
