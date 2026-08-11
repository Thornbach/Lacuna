//! Shared pixel-mask geometry primitives used by the interactive mask-editing
//! tools (Brush/Eraser/Wand/Knife/Scissor/Lasso) across multiple tabs — the
//! Pipeline tab's anomaly-region curation and the Field Review tab's
//! instance-segmentation correction. Extracted from `pipeline/mod.rs` because
//! these are dense, already-debugged pixel algorithms (Lab-space flood fill,
//! kerf-reclaim topology) where a bugfix landing in one copy and not a
//! duplicate would be a real, expensive-to-diagnose failure mode. Every
//! function here is a pure function over plain `Vec<bool>`/slices/tuples —
//! none of them know about `AnomalyRegion`, `RealInstance`, or any other
//! tab-specific state; callers own that.

use std::collections::{HashSet, VecDeque};

/// Flood-fills 8-connected components of `mask` (row-major, `w*h`), returning
/// each component as a `Vec` of flat pixel indices.
pub fn mask_connected_components(mask: &[bool], w: u32, h: u32) -> Vec<Vec<usize>> {
    let (w, h) = (w as usize, h as usize);
    let mut seen = vec![false; w * h];
    let mut out = Vec::new();
    for start in 0..w * h {
        if !mask[start] || seen[start] {
            continue;
        }
        let mut comp = Vec::new();
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(p) = stack.pop() {
            comp.push(p);
            let (px, py) = ((p % w) as i32, (p / w) as i32);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (nx, ny) = (px + dx, py + dy);
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let np = ny as usize * w + nx as usize;
                    if mask[np] && !seen[np] {
                        seen[np] = true;
                        stack.push(np);
                    }
                }
            }
        }
        out.push(comp);
    }
    out
}

/// Multi-source 8-connected BFS reclaiming pixels that were carved out of a
/// cut's mask but discarded outright before — seeded from every pixel
/// already assigned to a piece (flat `w`-major indices into `pieces`),
/// grown into any `original`-true, not-yet-owned pixel until every
/// reclaimable one is claimed by whichever piece's frontier reaches it
/// first. `permanent_gap(x,y)` (LOCAL bbox coordinates) marks pixels that
/// must NEVER be reclaimed regardless — the thin residual band that keeps
/// the resulting pieces non-adjacent so a cut can't quietly heal itself on
/// the next region/cluster rebuild.
pub fn reclaim_kerf(pieces: &mut [Vec<usize>], original: &[bool], w: u32, h: u32, permanent_gap: impl Fn(u32, u32) -> bool) {
    let (wu, hu) = (w as usize, h as usize);
    let mut owner: Vec<i32> = vec![-1; wu * hu];
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for (pi, piece) in pieces.iter().enumerate() {
        for &idx in piece {
            owner[idx] = pi as i32;
            queue.push_back(idx);
        }
    }
    while let Some(p) = queue.pop_front() {
        let owner_id = owner[p];
        let (px, py) = ((p % wu) as i32, (p / wu) as i32);
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let (nx, ny) = (px + dx, py + dy);
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    continue;
                }
                let np = ny as usize * wu + nx as usize;
                if original[np] && owner[np] == -1 && !permanent_gap(nx as u32, ny as u32) {
                    owner[np] = owner_id;
                    pieces[owner_id as usize].push(np);
                    queue.push_back(np);
                }
            }
        }
    }
}

/// 8-connected flood-fill from `(sx,sy)`, growing while each candidate
/// pixel's full CIELAB (L,a,b) distance to the SEED pixel stays within
/// `tolerance`. Capped at half the image's area (and a hard 40,000px
/// ceiling) so a too-loose tolerance on a fairly uniform region can't turn
/// into an unbounded fill.
pub fn wand_flood_fill(
    l: &[f32], a: &[f32], b: &[f32], w: usize, h: usize, sx: i32, sy: i32, tolerance: f32,
) -> HashSet<(i32, i32)> {
    let mut out = HashSet::new();
    if sx < 0 || sy < 0 || sx >= w as i32 || sy >= h as i32 {
        return out;
    }
    let seed_idx = sy as usize * w + sx as usize;
    let (sl, sa, sb) = (l[seed_idx], a[seed_idx], b[seed_idx]);
    let tol2 = tolerance * tolerance;
    // Capped WAY below "half the image" — a loose tolerance on a fairly
    // uniform region could otherwise grow into hundreds of thousands of
    // pixels, and rendering (even as a texture) and hashing that many
    // pixels every click is real, felt latency, not just a theoretical
    // concern.
    let cap = (w * h / 2).min(40_000);
    // BFS (FIFO), not DFS (LIFO stack) — nearest-to-seed pixels are
    // included FIRST, so if the true connected region is bigger than
    // `cap`, the result is a blob AROUND the click point missing only the
    // farthest edges. A DFS/stack version (the original implementation)
    // greedily chases one direction as far as it can go before ever
    // backtracking — hitting the cap mid-chase leaves a long thin tendril
    // reaching across the image with the area right around the click still
    // unfilled. Confirmed as a real, reported bug: "wand paths through the
    // whole leaf to fill out a region at the end of the picture."
    let mut queue: VecDeque<(i32, i32)> = VecDeque::new();
    queue.push_back((sx, sy));
    out.insert((sx, sy));
    while let Some((px, py)) = queue.pop_front() {
        if out.len() >= cap {
            break;
        }
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let (nx, ny) = (px + dx, py + dy);
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    continue;
                }
                if out.contains(&(nx, ny)) {
                    continue;
                }
                let idx = ny as usize * w + nx as usize;
                let (dl, da, db) = (l[idx] - sl, a[idx] - sa, b[idx] - sb);
                if dl * dl + da * da + db * db <= tol2 {
                    out.insert((nx, ny));
                    queue.push_back((nx, ny));
                }
            }
        }
    }
    out
}

/// Rasterizes a closed polygon (image/leaf-space points, in order) into a
/// bbox-local boolean mask, via a per-pixel `point_in_polygon` test over
/// the polygon's own bounding box — shared by the Polygon tool in both
/// Pipeline and Field Review. `None` for a degenerate polygon (fewer than
/// 3 points, or a zero-area bbox).
pub fn fill_polygon_mask(poly: &[(f32, f32)]) -> Option<([u32; 4], Vec<bool>)> {
    if poly.len() < 3 {
        return None;
    }
    let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
    let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
    for &(x, y) in poly {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    let bx = min_x.floor().max(0.0) as u32;
    let by = min_y.floor().max(0.0) as u32;
    let bx1 = max_x.ceil().max(0.0) as u32;
    let by1 = max_y.ceil().max(0.0) as u32;
    if bx1 <= bx || by1 <= by {
        return None;
    }
    let (bw, bh) = (bx1 - bx, by1 - by);
    // Sanity cap — callers are expected to clamp nodes to the actual
    // image/scene bounds before calling this (a stray click in a canvas's
    // letterboxed margin, or while zoomed, can otherwise map to leaf-space
    // coordinates far outside the real image). Without this, such a click
    // produces a bbox spanning tens of thousands of pixels per side, and
    // the `Vec<bool>` allocation below balloons to gigabytes — confirmed
    // as a real crash, not just a theoretical concern. 40M is comfortably
    // above any realistic leaf/scene image (a few thousand px per side)
    // while still catching a genuinely out-of-bounds click.
    const MAX_MASK_PIXELS: u64 = 40_000_000;
    if (bw as u64) * (bh as u64) > MAX_MASK_PIXELS {
        return None;
    }
    let mut mask = vec![false; (bw * bh) as usize];
    for gy in 0..bh {
        let py = by as f32 + gy as f32 + 0.5;
        for gx in 0..bw {
            let px = bx as f32 + gx as f32 + 0.5;
            if point_in_polygon(px, py, poly) {
                mask[(gy * bw + gx) as usize] = true;
            }
        }
    }
    Some(([bx, by, bw, bh], mask))
}

/// Standard even-odd point-in-polygon test (ray casting). Used by the Lasso
/// tool to decide which items' bbox-centers fall inside a freehand outline.
pub fn point_in_polygon(x: f32, y: f32, poly: &[(f32, f32)]) -> bool {
    let mut inside = false;
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Shortest distance from `(px,py)` to any edge segment of `poly`, treated
/// as closed (edge from the last point back to the first included). Used by
/// the Knife/Scissor tools' polycut kerf — a per-pixel "how close is this to
/// the drawn loop's boundary" test, distinct from `point_in_polygon`'s
/// inside/outside test.
pub fn dist_to_polygon_boundary(px: f32, py: f32, poly: &[(f32, f32)]) -> f32 {
    let n = poly.len();
    if n < 2 {
        return f32::INFINITY;
    }
    let mut best = f32::INFINITY;
    for k in 0..n {
        let (ax, ay) = poly[k];
        let (bx, by) = poly[(k + 1) % n];
        best = best.min(dist_point_to_segment(px, py, ax, ay, bx, by));
    }
    best
}

/// Shortest distance from `(px,py)` to any edge of an OPEN polyline `pts`
/// (2+ points, consecutive segments — NO wraparound edge from the last
/// point back to the first, unlike `dist_to_polygon_boundary`). Used by the
/// knife/scissor line-cut's kerf so a bent multi-segment cut works exactly
/// like a single straight one — a straight 2-point drag is just this
/// function's smallest case.
pub fn dist_to_polyline(px: f32, py: f32, pts: &[(f32, f32)]) -> f32 {
    let mut best = f32::INFINITY;
    for w in pts.windows(2) {
        let (ax, ay) = w[0];
        let (bx, by) = w[1];
        best = best.min(dist_point_to_segment(px, py, ax, ay, bx, by));
    }
    best
}

pub fn dist_point_to_segment(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let dx = bx - ax;
    let dy = by - ay;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-6 {
        return ((px - ax).powi(2) + (py - ay).powi(2)).sqrt();
    }
    let t = (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0);
    let (cx, cy) = (ax + t * dx, ay + t * dy);
    ((px - cx).powi(2) + (py - cy).powi(2)).sqrt()
}
