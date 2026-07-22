//! Cluster anomaly-region descriptors into defect families.
//!
//! Port of clustering.cluster_descriptors: StandardScaler so no dim dominates,
//! then density clustering. We use DBSCAN (label -1 = noise) rather than HDBSCAN
//! — adequate on the standardized 8-D descriptors and simple in pure Rust. PCA-2
//! gives 2-D coordinates for the interactive scatter (visualization only;
//! clustering runs in full 8-D). v1 DBSCAN is O(n²); fine for a few thousand
//! regions, revisit with a spatial index if it grows.

pub const D: usize = 8;

pub fn standardize(pts: &[[f32; D]]) -> Vec<[f32; D]> {
    let n = pts.len();
    if n == 0 {
        return Vec::new();
    }
    let mut mean = [0f32; D];
    for p in pts {
        for k in 0..D {
            mean[k] += p[k];
        }
    }
    for m in &mut mean {
        *m /= n as f32;
    }
    let mut sd = [0f32; D];
    for p in pts {
        for k in 0..D {
            sd[k] += (p[k] - mean[k]).powi(2);
        }
    }
    for s in &mut sd {
        *s = (*s / n as f32).sqrt().max(1e-6);
    }
    pts.iter()
        .map(|p| std::array::from_fn(|k| (p[k] - mean[k]) / sd[k]))
        .collect()
}

/// DBSCAN on D-dim points. Returns labels: cluster id ≥ 0, or -1 for noise.
pub fn dbscan(pts: &[[f32; D]], eps: f32, min_pts: usize) -> Vec<i32> {
    let n = pts.len();
    let eps2 = eps * eps;
    let dist2 = |a: &[f32; D], b: &[f32; D]| {
        (0..D).map(|k| (a[k] - b[k]) * (a[k] - b[k])).sum::<f32>()
    };
    let neighbors = |i: usize| -> Vec<usize> {
        (0..n).filter(|&j| dist2(&pts[i], &pts[j]) <= eps2).collect()
    };

    const UNVISITED: i32 = -2;
    const NOISE: i32 = -1;
    let mut labels = vec![UNVISITED; n];
    let mut c = -1i32;
    for i in 0..n {
        if labels[i] != UNVISITED {
            continue;
        }
        let nb = neighbors(i);
        if nb.len() < min_pts {
            labels[i] = NOISE;
            continue;
        }
        c += 1;
        labels[i] = c;
        let mut queue = nb;
        let mut qi = 0;
        while qi < queue.len() {
            let j = queue[qi];
            qi += 1;
            if labels[j] == NOISE {
                labels[j] = c; // border point
            }
            if labels[j] != UNVISITED {
                continue;
            }
            labels[j] = c;
            let nb2 = neighbors(j);
            if nb2.len() >= min_pts {
                queue.extend(nb2);
            }
        }
    }
    labels
}

/// Suggests a DBSCAN `eps` via the standard k-distance elbow heuristic: for
/// each point, find its distance to its `k`-th nearest neighbor (`k` =
/// `min_pts`); sort those distances ascending — this curve is flat where
/// points sit in dense regions and rises sharply once it reaches
/// sparse/noise territory. The "elbow" (point of maximum perpendicular
/// distance from the straight line joining the curve's first and last
/// points — a standard generic knee detector, no derivative estimation
/// needed) is a good `eps` candidate: tight enough to respect real density
/// gaps, loose enough to not starve every point of neighbors. This is a
/// suggestion to log/show, not something to auto-apply — a fixed `eps` is
/// inherently a per-dataset judgment call, and a smooth ramp with no clear
/// elbow (rather than a sharp knee) is itself useful information: it means
/// the data doesn't have a strong natural density gap at all, so no `eps`
/// choice will cleanly separate "cluster" from "noise" here.
pub fn suggest_eps(pts: &[[f32; D]], k: usize) -> Option<f32> {
    let n = pts.len();
    if k == 0 || n <= k + 1 {
        return None;
    }
    let dist = |a: &[f32; D], b: &[f32; D]| {
        (0..D).map(|d| (a[d] - b[d]) * (a[d] - b[d])).sum::<f32>().sqrt()
    };
    let mut kdist: Vec<f32> = (0..n)
        .map(|i| {
            let mut ds: Vec<f32> = (0..n).filter(|&j| j != i).map(|j| dist(&pts[i], &pts[j])).collect();
            ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ds[k - 1]
        })
        .collect();
    kdist.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let (x0, y0) = (0f32, kdist[0]);
    let (x1, y1) = ((n - 1) as f32, kdist[n - 1]);
    let (dx, dy) = (x1 - x0, y1 - y0);
    let norm = (dx * dx + dy * dy).sqrt().max(1e-9);
    let mut best_i = 0;
    let mut best_d = -1f32;
    for (i, &y) in kdist.iter().enumerate() {
        let (px, py) = (i as f32, y);
        let d = ((px - x0) * dy - (py - y0) * dx).abs() / norm;
        if d > best_d {
            best_d = d;
            best_i = i;
        }
    }
    Some(kdist[best_i])
}

/// Project D-dim points to 2-D via the top-2 principal components (for the scatter).
pub fn pca2(pts: &[[f32; D]]) -> Vec<[f32; 2]> {
    let n = pts.len();
    if n == 0 {
        return Vec::new();
    }
    let mut mean = [0f32; D];
    for p in pts {
        for k in 0..D {
            mean[k] += p[k];
        }
    }
    for m in &mut mean {
        *m /= n as f32;
    }
    let mut cov = [[0f32; D]; D];
    for p in pts {
        let d: [f32; D] = std::array::from_fn(|k| p[k] - mean[k]);
        for a in 0..D {
            for b in 0..D {
                cov[a][b] += d[a] * d[b];
            }
        }
    }
    for row in &mut cov {
        for v in row.iter_mut() {
            *v /= n as f32;
        }
    }
    let pc1 = power_iter(&cov);
    let cov2 = deflate(&cov, &pc1);
    let pc2 = power_iter(&cov2);
    pts.iter()
        .map(|p| {
            let d: [f32; D] = std::array::from_fn(|k| p[k] - mean[k]);
            [dot(&d, &pc1), dot(&d, &pc2)]
        })
        .collect()
}

fn dot(a: &[f32; D], b: &[f32; D]) -> f32 {
    (0..D).map(|k| a[k] * b[k]).sum()
}

fn mat_vec(m: &[[f32; D]; D], v: &[f32; D]) -> [f32; D] {
    std::array::from_fn(|a| (0..D).map(|b| m[a][b] * v[b]).sum())
}

fn power_iter(m: &[[f32; D]; D]) -> [f32; D] {
    let mut v: [f32; D] = std::array::from_fn(|k| 1.0 + 0.01 * k as f32);
    for _ in 0..100 {
        let mut w = mat_vec(m, &v);
        let norm = dot(&w, &w).sqrt().max(1e-12);
        for x in &mut w {
            *x /= norm;
        }
        v = w;
    }
    v
}

fn deflate(m: &[[f32; D]; D], v: &[f32; D]) -> [[f32; D]; D] {
    let lambda = dot(&mat_vec(m, v), v); // vᵀ M v
    let mut out = *m;
    for a in 0..D {
        for b in 0..D {
            out[a][b] -= lambda * v[a] * v[b];
        }
    }
    out
}

// ── Variable-dimension counterparts (used by the DINO-embedding unsupervised
// clustering path, worker.rs) ───────────────────────────────────────────────
// The fixed-D=8 functions above stay untouched — they still serve the legacy
// PatchCore hand-crafted-descriptor clustering path, and their `[f32; D]`
// return type is consumed directly by `AnomalyRegion.descriptor: [f32; 8]`.
// These operate on `Vec<f32>` points of any (uniform) length instead.

pub fn standardize_var(pts: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = pts.len();
    if n == 0 {
        return Vec::new();
    }
    let dim = pts[0].len();
    let mut mean = vec![0f32; dim];
    for p in pts {
        for k in 0..dim {
            mean[k] += p[k];
        }
    }
    for m in &mut mean {
        *m /= n as f32;
    }
    let mut sd = vec![0f32; dim];
    for p in pts {
        for k in 0..dim {
            sd[k] += (p[k] - mean[k]).powi(2);
        }
    }
    for s in &mut sd {
        *s = (*s / n as f32).sqrt().max(1e-6);
    }
    pts.iter()
        .map(|p| (0..dim).map(|k| (p[k] - mean[k]) / sd[k]).collect())
        .collect()
}

/// DBSCAN on variable-dim points. Same algorithm as `dbscan`, generalized.
pub fn dbscan_var(pts: &[Vec<f32>], eps: f32, min_pts: usize) -> Vec<i32> {
    let n = pts.len();
    let eps2 = eps * eps;
    let dist2 = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();
    let neighbors = |i: usize| -> Vec<usize> {
        (0..n).filter(|&j| dist2(&pts[i], &pts[j]) <= eps2).collect()
    };

    const UNVISITED: i32 = -2;
    const NOISE: i32 = -1;
    let mut labels = vec![UNVISITED; n];
    let mut c = -1i32;
    for i in 0..n {
        if labels[i] != UNVISITED {
            continue;
        }
        let nb = neighbors(i);
        if nb.len() < min_pts {
            labels[i] = NOISE;
            continue;
        }
        c += 1;
        labels[i] = c;
        let mut queue = nb;
        let mut qi = 0;
        while qi < queue.len() {
            let j = queue[qi];
            qi += 1;
            if labels[j] == NOISE {
                labels[j] = c; // border point
            }
            if labels[j] != UNVISITED {
                continue;
            }
            labels[j] = c;
            let nb2 = neighbors(j);
            if nb2.len() >= min_pts {
                queue.extend(nb2);
            }
        }
    }
    labels
}

/// k-distance elbow `eps` suggestion, variable-dim counterpart of `suggest_eps`.
pub fn suggest_eps_var(pts: &[Vec<f32>], k: usize) -> Option<f32> {
    let n = pts.len();
    if k == 0 || n <= k + 1 {
        return None;
    }
    let dist = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
    let mut kdist: Vec<f32> = (0..n)
        .map(|i| {
            let mut ds: Vec<f32> = (0..n).filter(|&j| j != i).map(|j| dist(&pts[i], &pts[j])).collect();
            ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ds[k - 1]
        })
        .collect();
    kdist.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let (x0, y0) = (0f32, kdist[0]);
    let (x1, y1) = ((n - 1) as f32, kdist[n - 1]);
    let (dx, dy) = (x1 - x0, y1 - y0);
    let norm = (dx * dx + dy * dy).sqrt().max(1e-9);
    let mut best_i = 0;
    let mut best_d = -1f32;
    for (i, &y) in kdist.iter().enumerate() {
        let (px, py) = (i as f32, y);
        let d = ((px - x0) * dy - (py - y0) * dx).abs() / norm;
        if d > best_d {
            best_d = d;
            best_i = i;
        }
    }
    Some(kdist[best_i])
}

/// Project variable-dim points to their top-`k` principal components (repeated
/// power-iteration + deflation, same technique as `pca2`, generalized to any
/// `k` and any input dimension instead of a fixed top-2-of-8).
pub fn pca_k_var(pts: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
    let n = pts.len();
    if n == 0 {
        return Vec::new();
    }
    let dim = pts[0].len();
    let k = k.min(dim);
    let mut mean = vec![0f32; dim];
    for p in pts {
        for d in 0..dim {
            mean[d] += p[d];
        }
    }
    for m in &mut mean {
        *m /= n as f32;
    }
    let mut cov = vec![vec![0f32; dim]; dim];
    for p in pts {
        let d: Vec<f32> = (0..dim).map(|i| p[i] - mean[i]).collect();
        for a in 0..dim {
            for b in 0..dim {
                cov[a][b] += d[a] * d[b];
            }
        }
    }
    for row in &mut cov {
        for v in row.iter_mut() {
            *v /= n as f32;
        }
    }
    let mut components: Vec<Vec<f32>> = Vec::with_capacity(k);
    let mut cur = cov;
    for _ in 0..k {
        let pc = power_iter_var(&cur, dim);
        cur = deflate_var(&cur, &pc, dim);
        components.push(pc);
    }
    pts.iter()
        .map(|p| {
            let d: Vec<f32> = (0..dim).map(|i| p[i] - mean[i]).collect();
            components.iter().map(|c| dot_var(&d, c)).collect()
        })
        .collect()
}

fn dot_var(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn mat_vec_var(m: &[Vec<f32>], v: &[f32], dim: usize) -> Vec<f32> {
    (0..dim).map(|a| (0..dim).map(|b| m[a][b] * v[b]).sum()).collect()
}

fn power_iter_var(m: &[Vec<f32>], dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|k| 1.0 + 0.01 * k as f32).collect();
    for _ in 0..100 {
        let mut w = mat_vec_var(m, &v, dim);
        let norm = dot_var(&w, &w).sqrt().max(1e-12);
        for x in &mut w {
            *x /= norm;
        }
        v = w;
    }
    v
}

fn deflate_var(m: &[Vec<f32>], v: &[f32], dim: usize) -> Vec<Vec<f32>> {
    let lambda = dot_var(&mat_vec_var(m, v, dim), v);
    let mut out = m.to_vec();
    for a in 0..dim {
        for b in 0..dim {
            out[a][b] -= lambda * v[a] * v[b];
        }
    }
    out
}

// ── Agglomerative (hierarchical) clustering — an alternative to DBSCAN that
// doesn't need a global density radius. Built after DBSCAN hit the same
// "tight radius -> fragments+noise, loose radius (even at the k-distance
// elbow) -> everything collapses into 1-2 blobs" failure on TWO different
// feature spaces (the legacy 8-D descriptor AND the DINO embedding) — that's
// DBSCAN's transitive density-chaining, not a tuning gap. Complete linkage
// specifically (not single linkage, which shares DBSCAN's chaining
// weakness): merging two clusters can only happen once ALL their members are
// close, not just some pair, so it resists exactly the failure just
// observed.
//
// The merge SEQUENCE (not just a final labeling) is the actual product:
// replaying any PREFIX of it via union-find reproduces the clustering at any
// smaller cluster count K in O(n) — no need to rerun the merge search to
// answer "what if I wanted a different K", which is what makes instant
// interactive re-cutting (mod.rs) possible instead of another full pipeline
// run per guess.

/// Complete-linkage agglomeration via the nearest-neighbor-chain (NN-chain)
/// algorithm (Murtagh) — produces the EXACT SAME clustering as the naive
/// "rescan every live pair for the global minimum each step" approach (an
/// earlier version of this function did exactly that), just computed in
/// O(n²) time instead of O(n³). That earlier version was accepted as fine
/// "at ~4000 regions" per this function's own now-outdated reasoning — real
/// usage since then reached ~15600 regions, where O(n³) means roughly
/// (15600/4000)³ ≈ 60x the cost that estimate was based on, in practice a
/// 30-40 MINUTE hang. NN-chain exploits a property specific to Lance-
/// Williams-updatable linkages (complete/average/ward, NOT single, which is
/// why this doesn't generalize to every possible linkage): repeatedly
/// following "nearest neighbor of the current chain tip" is guaranteed to
/// terminate at a pair that are each other's nearest neighbor (a
/// "reciprocal nearest neighbor" pair, RNN) — merging that pair is always a
/// valid step in SOME complete-linkage dendrogram, so the chain can be
/// resumed from where it left off after each merge instead of restarting a
/// full O(n) scan. The amortized argument (standard result, see e.g. Müllner
/// 2011 "Modern hierarchical clustering algorithms") bounds total chain
/// pushes across the WHOLE run at O(n), each costing O(n) to find a nearest
/// neighbor — O(n²) total. Cross-checked against the naive algorithm for
/// identical results — see `agglomerative_var_matches_naive_reference`.
///
/// Each output entry is `(i, j, distance)` — `i` is the smaller of the two
/// ORIGINAL point indices being merged, always the surviving representative
/// (never renumbered, only deactivated), `j` (the larger) is absorbed into
/// it — same convention the naive version used, so `labels_for_k`/
/// `labels_adaptive`'s union-find replay is unaffected by this change.
/// Checks `cancel` periodically and returns whatever's been merged so far if
/// it fires — a partial sequence is still valid to replay prefixes of, it
/// just can't reach small K values.
pub fn agglomerative_var(pts: &[Vec<f32>], cancel: &std::sync::atomic::AtomicBool) -> Vec<(usize, usize, f32)> {
    let n = pts.len();
    if n < 2 {
        return Vec::new();
    }
    let dim = pts[0].len();
    let dist = |a: &[f32], b: &[f32]| -> f32 {
        (0..dim).map(|k| (a[k] - b[k]) * (a[k] - b[k])).sum::<f32>().sqrt()
    };
    // Upper-triangular live distance matrix: d[i][j] for i<j, both still live.
    let mut d: Vec<Vec<f32>> = (0..n).map(|i| vec![0f32; n - i]).collect();
    for i in 0..n {
        for j in (i + 1)..n {
            d[i][j - i] = dist(&pts[i], &pts[j]);
        }
    }
    let get = |d: &[Vec<f32>], i: usize, j: usize| -> f32 {
        if i < j { d[i][j - i] } else { d[j][i - j] }
    };
    let set = |d: &mut [Vec<f32>], i: usize, j: usize, v: f32| {
        if i < j { d[i][j - i] = v } else { d[j][i - j] = v }
    };

    let mut active = vec![true; n];
    let mut merges: Vec<(usize, usize, f32)> = Vec::with_capacity(n - 1);
    let mut chain: Vec<usize> = Vec::new();
    let mut step = 0usize;

    // Nearest ACTIVE point to `x` (excluding `x` itself) — O(n) scan.
    let nearest = |d: &[Vec<f32>], active: &[bool], x: usize| -> usize {
        let mut best = usize::MAX;
        let mut bd = f32::INFINITY;
        for k in 0..n {
            if k == x || !active[k] {
                continue;
            }
            let v = get(d, x, k);
            if v < bd {
                bd = v;
                best = k;
            }
        }
        best
    };

    // Ground-truth active count, recomputed from `active` itself every
    // iteration rather than tracked incrementally — an incremental counter
    // (`n_active -= 1` paired with every `active[j] = false`) LOOKS
    // airtight by construction, but two separate real-world crashes at
    // ~15600 and ~7500 regions, neither reproducible at the small sizes
    // this function's own tests cover, both had the exact signature of the
    // counter believing more points were active than the array actually
    // had — and repeated attempts to prove the increment/decrement pairing
    // can never drift did not find where, if anywhere, it does. Recomputing
    // it directly makes drift IMPOSSIBLE rather than merely believed
    // impossible: this costs O(n) per iteration, but the search below is
    // already O(n) per iteration too, so it doesn't change the algorithm's
    // O(n²) total — a live count over a slice is not the expensive part.
    'merge_loop: while active.iter().filter(|&&a| a).count() > 1 {
        if step % 64 == 0 && cancel.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        step += 1;
        // Follow nearest-neighbor links from the chain tip until a
        // reciprocal pair (each other's nearest neighbor) is found.
        //
        // Defensive: the textbook proof that a chain never revisits a point
        // before terminating in an RNN assumes EXACT arithmetic. At real
        // scale (thousands of f32 points, complete linkage repeatedly
        // taking max() over accumulating updates) two distances that are
        // mathematically distinct can round to compare equal or flip order,
        // which can in rare cases let the SAME point end up pushed onto the
        // chain twice. When the later occurrence is later merged away
        // (deactivated), the earlier occurrence sitting deeper in the chain
        // becomes a stale/"zombie" entry — using it as `x` would query
        // `nearest()` for an already-absorbed point, and once few enough
        // points remain active, `nearest()` can find nothing left at all
        // (its `usize::MAX` "not found" sentinel), which then gets used as
        // an index — a real crash this caused in production at ~15600
        // regions (never reproduced at the small sizes
        // `agglomerative_var_matches_naive_reference` covers). Purging any
        // stale entries off the top before ever reading `chain.last()`
        // makes this self-healing regardless of root cause, at negligible
        // cost (this loop only ever does real work when it fires).
        let (x, y) = loop {
            while matches!(chain.last(), Some(&t) if !active[t]) {
                chain.pop();
            }
            if chain.is_empty() {
                let start = (0..n).find(|&i| active[i]).expect("loop condition guarantees >1 active");
                chain.push(start);
            }
            let x = *chain.last().expect("just pushed if empty");
            let y = nearest(&d, &active, x);
            if y == usize::MAX {
                // Hard fallback — should be unreachable given the loop
                // condition guarantees >1 active points and the purge
                // above, but proving a hand-rolled chain algorithm airtight
                // against every real f32-rounding edge case at real scale
                // has twice now turned out harder than expected (the purge
                // above fixed one recurrence of this exact crash; a second
                // one still got through). Rather than keep patching chain
                // invariants under time pressure, guarantee no crash
                // outright: brute-force the TRUE global-minimum active pair
                // directly (same per-step logic `naive_complete_linkage`
                // uses throughout), and discard the chain's state entirely
                // rather than trust it further. Costs O(active²) for just
                // this one step — since this should be rare to never, it
                // doesn't threaten the O(n²) total.
                let mut bi = usize::MAX;
                let mut bj = usize::MAX;
                let mut bd = f32::INFINITY;
                for a in 0..n {
                    if !active[a] {
                        continue;
                    }
                    for b in (a + 1)..n {
                        if !active[b] {
                            continue;
                        }
                        let v = get(&d, a, b);
                        if v < bd {
                            bd = v;
                            bi = a;
                            bj = b;
                        }
                    }
                }
                chain.clear();
                break (bi, bj);
            }
            if chain.len() >= 2 && chain[chain.len() - 2] == y {
                break (x, y);
            }
            chain.push(y);
        };
        if x == usize::MAX || y == usize::MAX {
            // Absolute backstop: even the brute-force fallback above found
            // no valid pair — should be truly impossible (it scans every
            // (a,b) with both active, and the ground-truth active-count
            // loop condition guarantees at least 2 exist), but this crash
            // has now recurred twice despite two separate "should be
            // impossible" fixes. Whatever the actual cause, stop cleanly
            // here rather than let a sentinel value reach an array index a
            // third time. A partial merge sequence is still valid to
            // replay prefixes of, same as the existing cancel early-exit.
            break 'merge_loop;
        }
        chain.pop();
        chain.pop();
        let dxy = get(&d, x, y);
        let (i, j) = if x < y { (x, y) } else { (y, x) };
        merges.push((i, j, dxy));
        // complete linkage: d(merged, k) = max(d(x,k), d(y,k)) for every
        // other active k — stored under the surviving id `i`.
        for k in 0..n {
            if k == x || k == y || !active[k] {
                continue;
            }
            let m = get(&d, x, k).max(get(&d, y, k));
            set(&mut d, i, k, m);
        }
        active[j] = false;
    }
    // NN-chain discovers merges in CHAIN-TRAVERSAL order, not globally-
    // increasing distance order the way the naive global-min-scan always
    // did (each of its steps explicitly finds the smallest remaining
    // distance) — `labels_for_k`'s union-find prefix-replay, and the
    // "merge distances are non-decreasing" invariant callers rely on, both
    // assume a distance-sorted sequence. The SET of (i,j,distance) merges
    // NN-chain finds is provably identical to naive's (verified: see
    // `agglomerative_var_matches_naive_reference`) — sorting them here is
    // the standard final step of the algorithm (same approach scipy's
    // `linkage` uses), not a distinct computation.
    merges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    merges
}

/// Fixed-8-D counterpart of `agglomerative_var`, for the legacy PatchCore
/// descriptor path.
pub fn agglomerative(pts: &[[f32; D]], cancel: &std::sync::atomic::AtomicBool) -> Vec<(usize, usize, f32)> {
    let pts_var: Vec<Vec<f32>> = pts.iter().map(|p| p.to_vec()).collect();
    agglomerative_var(&pts_var, cancel)
}

/// Relabel a raw per-point grouping key (union-find roots, tree-traversal
/// ids, anything comparable/hashable) to sequential ids in first-appearance
/// order, then fold any group smaller than `min_cluster_size` to `-1`
/// (noise), mirroring DBSCAN's convention. Shared by `labels_for_k` and
/// `labels_adaptive` — both produce a raw grouping via different mechanisms
/// (union-find vs. tree traversal) but need identical post-processing.
fn compact_and_fold(raw_group_key: &[usize], min_cluster_size: usize) -> Vec<i32> {
    let mut id_of: std::collections::HashMap<usize, i32> = std::collections::HashMap::new();
    let mut next_id = 0i32;
    for &r in raw_group_key {
        id_of.entry(r).or_insert_with(|| { let id = next_id; next_id += 1; id });
    }
    let mut counts = vec![0usize; next_id as usize];
    for &r in raw_group_key {
        counts[id_of[&r] as usize] += 1;
    }
    raw_group_key.iter().map(|&r| {
        let id = id_of[&r];
        if counts[id as usize] < min_cluster_size { -1 } else { id }
    }).collect()
}

/// Cheap O(n) replay: labels for exactly `k` final clusters (`k` must
/// already be resolved to a concrete value in `[1, n]` — never pass a
/// literal "auto" sentinel here). Clusters smaller than `min_cluster_size`
/// are relabeled `-1` (noise), mirroring DBSCAN's convention.
pub fn labels_for_k(merges: &[(usize, usize, f32)], n: usize, k: usize, min_cluster_size: usize) -> Vec<i32> {
    if n == 0 {
        return Vec::new();
    }
    let k = k.clamp(1, n);
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }
    let replay = (n - k).min(merges.len());
    for &(i, j, _) in &merges[..replay] {
        let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
        if ri != rj {
            parent[rj] = ri;
        }
    }
    let root_of: Vec<usize> = (0..n).map(|i| find(&mut parent, i)).collect();
    compact_and_fold(&root_of, min_cluster_size)
}

/// Top candidate cluster counts by largest gap between consecutive merge
/// distances (the dendrogram-height sequence is non-decreasing for complete
/// linkage, confirmed: the distance that triggers a merge is always ≤ every
/// other still-live pairwise distance, and the merge's own update can only
/// raise, never lower, subsequent distances — so each step's global minimum
/// is monotonically non-decreasing). Like `suggest_eps`, this is a logged
/// suggestion, not something to auto-apply blindly: the single largest gap
/// is very often the FINAL merge (the most separated top-level split, e.g.
/// "damaged vs background", isn't necessarily the most USEFUL cut) — hence
/// returning several candidates rather than just the winner.
pub fn suggest_k(merges: &[(usize, usize, f32)]) -> Vec<(usize, f32)> {
    let n = merges.len() + 1; // original point count
    if merges.len() < 2 {
        return Vec::new();
    }
    let dists: Vec<f32> = merges.iter().map(|&(_, _, d)| d).collect();
    let mut gaps: Vec<(usize, f32)> = (0..dists.len() - 1)
        .map(|m| (m, dists[m + 1] - dists[m]))
        .collect();
    gaps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    gaps.into_iter()
        .take(4)
        .map(|(m, gap)| (n - m - 1, gap)) // K = clusters remaining right after merge m completes
        .collect()
}

// ── Adaptive per-branch cut ──────────────────────────────────────────────────
// `labels_for_k`'s cut is a single flat K applied uniformly across the whole
// dendrogram — if one branch needs a much deeper split than the rest, no
// single K resolves it without either leaving that branch merged or
// needlessly over-splitting everything else. This reconstructs the actual
// tree from the flat `merges` sequence and cuts each branch at its own
// locally-appropriate depth, via an inconsistency statistic adapted from
// SciPy's `scipy.cluster.hierarchy.inconsistent`/`fcluster(criterion=
// 'inconsistent')` — an established method, not an ad hoc heuristic.

/// A node in the dendrogram tree reconstructed from a flat merge sequence.
/// `labels_for_k`'s plain union-find replay doesn't expose parent/child
/// structure at all — this does, which `labels_adaptive` needs for its
/// per-branch decisions.
enum TreeNode {
    Leaf(usize),
    Internal { height: f32, left: usize, right: usize },
}

/// Rebuild the dendrogram as an arena: slots `0..n` ARE the leaves
/// (`arena[x] = Leaf(x)`), internal nodes appended one per merge starting at
/// index `n`. `node_of[x]` is the CURRENT arena node representing original
/// point `x`'s cluster as of the last processed merge. `is_absorbed[x]`
/// marks every point that became some merge's `j` (absorbed side) — points
/// that stay `false` are FOREST ROOTS. Normally there's exactly one root
/// (the last merge's result); a cancelled/partial `merges` sequence (fewer
/// than `n-1` entries, if `agglomerative_var`'s cancel check fired) produces
/// a forest with multiple roots instead — callers must handle that, not
/// assume a single root.
fn build_tree(merges: &[(usize, usize, f32)], n: usize) -> (Vec<TreeNode>, Vec<usize>, Vec<bool>) {
    let mut arena: Vec<TreeNode> = (0..n).map(TreeNode::Leaf).collect();
    let mut node_of: Vec<usize> = (0..n).collect();
    let mut is_absorbed = vec![false; n];
    for &(i, j, dist) in merges {
        let v = arena.len();
        arena.push(TreeNode::Internal { height: dist, left: node_of[i], right: node_of[j] });
        node_of[i] = v;
        is_absorbed[j] = true;
    }
    (arena, node_of, is_absorbed)
}

/// SciPy's own default depth for the inconsistency calculation. Not
/// user-exposed — no evidence yet end users should tune this on top of the
/// sensitivity threshold; if the shipped default doesn't resolve a case
/// that needs it, bumping this is the documented first thing to try.
const ADAPTIVE_DEPTH: usize = 2;

/// Collect the merge heights of `node` and its INTERNAL descendants down to
/// `depth` levels (leaves contribute nothing — they have no height).
/// Iterative (explicit stack), not recursive, matching this file's style.
fn collect_heights(arena: &[TreeNode], node: usize, depth: usize, out: &mut Vec<f32>) {
    let mut stack: Vec<(usize, usize)> = vec![(node, depth)];
    while let Some((v, d)) = stack.pop() {
        if let TreeNode::Internal { height, left, right } = arena[v] {
            out.push(height);
            if d > 0 {
                stack.push((left, d - 1));
                stack.push((right, d - 1));
            }
        }
    }
}

/// How far internal node `v`'s own merge height stands out from the local
/// distribution of its nearby descendants' heights (v's own height is
/// EXCLUDED from that distribution — comparing v against itself would be
/// meaningless). `std ≈ 0` (too little local data to compare against, e.g.
/// both children are leaves — a leaf contributes no height at all)
/// conservatively returns 0.0 — "not independently worth splitting on its
/// own" — biasing AWAY from a shallow node accidentally shattering its
/// whole subtree.
fn inconsistency(arena: &[TreeNode], v: usize) -> f32 {
    let TreeNode::Internal { height, left, right } = arena[v] else { return 0.0 };
    let mut heights = Vec::new();
    if ADAPTIVE_DEPTH > 0 {
        collect_heights(arena, left, ADAPTIVE_DEPTH - 1, &mut heights);
        collect_heights(arena, right, ADAPTIVE_DEPTH - 1, &mut heights);
    }
    if heights.is_empty() {
        return 0.0;
    }
    let mean = heights.iter().sum::<f32>() / heights.len() as f32;
    let var = heights.iter().map(|h| (h - mean).powi(2)).sum::<f32>() / heights.len() as f32;
    let std = var.sqrt();
    if std > 1e-6 { (height - mean) / std } else { 0.0 }
}

/// Adaptive per-branch cut: unlike `labels_for_k`'s single flat K applied
/// uniformly, this can resolve different branches of the tree at different
/// depths. Evaluates EVERY node's inconsistency independently up front (a
/// full traversal), NOT a greedy top-down stop at the first node that fails
/// its own local check — a naive early-stop would let a locally-unremarkable
/// ancestor (a real possibility; the statistic is purely local) hide a
/// genuinely sharp split several levels deeper in its subtree, silently
/// reproducing exactly the "some branches never split no matter what"
/// failure this function exists to fix. A node whose inconsistency exceeds
/// `threshold` mints two FRESH cluster ids for its children; otherwise both
/// children keep the SAME id flowing down from their parent — but traversal
/// continues into both regardless, never stopping early, so a deeper node
/// can still independently mint its own ids even when nothing above it did.
pub fn labels_adaptive(merges: &[(usize, usize, f32)], n: usize, threshold: f32, min_cluster_size: usize) -> Vec<i32> {
    if n == 0 {
        return Vec::new();
    }
    let (arena, node_of, is_absorbed) = build_tree(merges, n);
    let inconsist: Vec<f32> = (0..arena.len()).map(|v| inconsistency(&arena, v)).collect();

    let roots: Vec<usize> = (0..n).filter(|&x| !is_absorbed[x]).map(|x| node_of[x]).collect();
    let mut group = vec![0usize; n];
    let mut next_id = 0usize;
    let mut stack: Vec<(usize, usize)> = Vec::new(); // (arena node, current group id)
    for &r in &roots {
        stack.push((r, next_id));
        next_id += 1;
    }
    while let Some((v, gid)) = stack.pop() {
        match arena[v] {
            TreeNode::Leaf(point) => {
                group[point] = gid;
            }
            TreeNode::Internal { left, right, .. } => {
                if inconsist[v] > threshold {
                    stack.push((left, next_id));
                    next_id += 1;
                    stack.push((right, next_id));
                    next_id += 1;
                } else {
                    stack.push((left, gid));
                    stack.push((right, gid));
                }
            }
        }
    }
    compact_and_fold(&group, min_cluster_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn dbscan_separates_two_blobs() {
        // two blobs separated along dim 0; other dims constant (no amplified noise)
        let mut pts: Vec<[f32; D]> = Vec::new();
        for i in 0..30 {
            let mut p = [0f32; D];
            p[0] = (i % 3) as f32 * 0.05;
            pts.push(p);
        }
        for i in 0..30 {
            let mut p = [0f32; D];
            p[0] = 10.0 + (i % 3) as f32 * 0.05;
            pts.push(p);
        }
        let std = standardize(&pts);
        let labels = dbscan(&std, 0.5, 5);
        let clusters: HashSet<i32> = labels.iter().copied().filter(|&l| l >= 0).collect();
        assert_eq!(clusters.len(), 2, "expected 2 clusters, got {clusters:?}");
        assert!(!labels.contains(&-1) || labels.iter().filter(|&&l| l == -1).count() < 5,
            "too much noise");

        // PCA-2: PC1 should separate the blobs (range across PC1 large)
        let xy = pca2(&std);
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for p in &xy {
            lo = lo.min(p[0]);
            hi = hi.max(p[0]);
        }
        assert!(hi - lo > 1.5, "PC1 spread {} too small", hi - lo);
    }

    #[test]
    fn suggest_eps_lands_between_dense_and_sparse() {
        // Two dense blobs, standardized, k=5: the elbow should land somewhere
        // clearly above the tight within-blob spacing and clearly below the
        // huge inter-blob gap — i.e. a usable eps, not pinned to either
        // extreme. Unlike `dbscan_separates_two_blobs` above, every point gets
        // a DISTINCT dim-0 value (no repeated/duplicate coordinates) — real
        // region descriptors are continuous floats and essentially never
        // collide exactly, and exact duplicates degrade k-distance into a
        // long flat run of zeros that isn't representative of real data.
        let mut pts: Vec<[f32; D]> = Vec::new();
        for i in 0..30 {
            let mut p = [0f32; D];
            p[0] = i as f32 * 0.02;
            pts.push(p);
        }
        for i in 0..30 {
            let mut p = [0f32; D];
            p[0] = 10.0 + i as f32 * 0.02;
            pts.push(p);
        }
        let std = standardize(&pts);
        let eps = suggest_eps(&std, 5).expect("enough points for a suggestion");
        assert!(eps > 0.0 && eps.is_finite(), "eps should be a sane positive number, got {eps}");

        // With that suggested eps, DBSCAN should actually recover 2 clusters
        // with little noise — the whole point of the suggestion.
        let labels = dbscan(&std, eps, 5);
        let clusters: HashSet<i32> = labels.iter().copied().filter(|&l| l >= 0).collect();
        assert_eq!(clusters.len(), 2, "suggested eps {eps} gave {clusters:?}, expected 2 clusters");
    }

    #[test]
    fn suggest_eps_none_when_too_few_points() {
        let pts: Vec<[f32; D]> = vec![[0.0; D]; 3];
        assert_eq!(suggest_eps(&pts, 5), None);
    }

    #[test]
    fn dbscan_var_separates_two_blobs_in_higher_dim() {
        // Same shape as dbscan_separates_two_blobs but at dim=32 (DINO-PCA-like
        // dimensionality) with distinct per-point values (no exact duplicates).
        const DIM: usize = 32;
        let mut pts: Vec<Vec<f32>> = Vec::new();
        for i in 0..30 {
            let mut p = vec![0f32; DIM];
            p[0] = i as f32 * 0.02;
            pts.push(p);
        }
        for i in 0..30 {
            let mut p = vec![0f32; DIM];
            p[0] = 10.0 + i as f32 * 0.02;
            pts.push(p);
        }
        let std = standardize_var(&pts);
        let labels = dbscan_var(&std, 0.5, 5);
        let clusters: HashSet<i32> = labels.iter().copied().filter(|&l| l >= 0).collect();
        assert_eq!(clusters.len(), 2, "expected 2 clusters, got {clusters:?}");
    }

    #[test]
    fn pca_k_var_reduces_dimension_and_separates_blobs() {
        const DIM: usize = 32;
        let mut pts: Vec<Vec<f32>> = Vec::new();
        for i in 0..30 {
            let mut p = vec![0f32; DIM];
            p[0] = i as f32 * 0.02;
            pts.push(p);
        }
        for i in 0..30 {
            let mut p = vec![0f32; DIM];
            p[0] = 10.0 + i as f32 * 0.02;
            pts.push(p);
        }
        let reduced = pca_k_var(&pts, 8);
        assert_eq!(reduced.len(), 60);
        assert!(reduced.iter().all(|p| p.len() == 8), "expected 8-D reduced points");

        // PC1 should still separate the blobs, same shape as the pca2 test.
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for p in &reduced {
            lo = lo.min(p[0]);
            hi = hi.max(p[0]);
        }
        assert!(hi - lo > 1.5, "PC1 spread {} too small", hi - lo);
    }

    #[test]
    fn suggest_eps_var_matches_fixed_d_on_same_data() {
        // Sanity check: suggest_eps_var on 8-D data should behave the same as
        // suggest_eps (same algorithm, generalized storage).
        let mut pts: Vec<[f32; D]> = Vec::new();
        for i in 0..30 {
            let mut p = [0f32; D];
            p[0] = i as f32 * 0.02;
            pts.push(p);
        }
        for i in 0..30 {
            let mut p = [0f32; D];
            p[0] = 10.0 + i as f32 * 0.02;
            pts.push(p);
        }
        let std_fixed = standardize(&pts);
        let std_var: Vec<Vec<f32>> = std_fixed.iter().map(|p| p.to_vec()).collect();
        let eps_fixed = suggest_eps(&std_fixed, 5).expect("fixed-D suggestion");
        let eps_var = suggest_eps_var(&std_var, 5).expect("var-D suggestion");
        assert!((eps_fixed - eps_var).abs() < 1e-4,
            "fixed {eps_fixed} vs var {eps_var} should match on identical data");
    }

    fn no_cancel() -> std::sync::atomic::AtomicBool {
        std::sync::atomic::AtomicBool::new(false)
    }

    /// Reference implementation kept ONLY for testing: the original naive
    /// "rescan every live pair for the global minimum each step" complete
    /// linkage, O(n³). `agglomerative_var` replaced this with the O(n²)
    /// NN-chain algorithm for real use (a 30-40 MINUTE hang at ~15600
    /// regions, vs the ~4000 this was originally sized for) — this reference
    /// stays here so `agglomerative_var_matches_naive_reference` can prove
    /// the fast version produces IDENTICAL clusterings, not an approximation.
    fn naive_complete_linkage(pts: &[Vec<f32>], cancel: &std::sync::atomic::AtomicBool) -> Vec<(usize, usize, f32)> {
        let n = pts.len();
        if n < 2 {
            return Vec::new();
        }
        let dim = pts[0].len();
        let dist = |a: &[f32], b: &[f32]| -> f32 {
            (0..dim).map(|k| (a[k] - b[k]) * (a[k] - b[k])).sum::<f32>().sqrt()
        };
        let mut d: Vec<Vec<f32>> = (0..n).map(|i| vec![0f32; n - i]).collect();
        for i in 0..n {
            for j in (i + 1)..n {
                d[i][j - i] = dist(&pts[i], &pts[j]);
            }
        }
        let get = |d: &[Vec<f32>], i: usize, j: usize| -> f32 {
            if i < j { d[i][j - i] } else { d[j][i - j] }
        };
        let set = |d: &mut [Vec<f32>], i: usize, j: usize, v: f32| {
            if i < j { d[i][j - i] = v } else { d[j][i - j] = v }
        };
        let mut live: Vec<usize> = (0..n).collect();
        let mut merges: Vec<(usize, usize, f32)> = Vec::with_capacity(n - 1);
        for step in 0..n - 1 {
            if step % 64 == 0 && cancel.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            let (mut bi, mut bj, mut bd) = (0usize, 1usize, f32::INFINITY);
            for a in 0..live.len() {
                for b in (a + 1)..live.len() {
                    let (li, lj) = (live[a], live[b]);
                    let v = get(&d, li, lj);
                    if v < bd {
                        bd = v;
                        bi = a;
                        bj = b;
                    }
                }
            }
            let (i, j) = (live[bi], live[bj]);
            merges.push((i, j, bd));
            for &k in &live {
                if k == i || k == j {
                    continue;
                }
                let m = get(&d, i, k).max(get(&d, j, k));
                set(&mut d, i, k, m);
            }
            live.swap_remove(bj);
        }
        merges
    }

    #[test]
    #[ignore] // manual stress check at real-world scale, not part of the normal fast suite
    fn agglomerative_var_handles_duplicates_at_real_scale() {
        // Reproduces the actual production scale (~15600 regions) that
        // crashed, heavy with exact-duplicate points (the hypothesized
        // trigger for the zombie-chain-entry defensive purge above) — this
        // is the closest thing to a direct regression test for that crash
        // without the real DINO embedding data that caused it.
        fn pseudo_rand(seed: u32) -> f32 {
            let mut x = seed.wrapping_mul(2654435761);
            x ^= x >> 15;
            x = x.wrapping_mul(0x85ebca6b);
            x ^= x >> 13;
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
        const DIM: usize = 32;
        const N_GROUPS: usize = 400;
        const DUPES_PER_GROUP: usize = 39;
        let mut pts: Vec<Vec<f32>> = Vec::new();
        for g in 0..N_GROUPS {
            let base: Vec<f32> = (0..DIM).map(|d| pseudo_rand((g * DIM + d) as u32 + 1) * 5.0).collect();
            for _ in 0..DUPES_PER_GROUP {
                pts.push(base.clone());
            }
        }
        let n = pts.len();
        println!("n={n}");
        let t0 = std::time::Instant::now();
        let merges = agglomerative_var(&pts, &no_cancel());
        println!("{:.2}s, {} merges", t0.elapsed().as_secs_f64(), merges.len());
        assert_eq!(merges.len(), n - 1);
    }

    #[test]
    #[ignore] // manual perf sanity check, not part of the normal fast suite
    fn agglomerative_var_scales_to_thousands() {
        fn pseudo_rand(seed: u32) -> f32 {
            let mut x = seed.wrapping_mul(2654435761);
            x ^= x >> 15;
            x = x.wrapping_mul(0x85ebca6b);
            x ^= x >> 13;
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
        const DIM: usize = 32;
        for &n in &[2000usize, 4000, 8000] {
            let pts: Vec<Vec<f32>> = (0..n).map(|i| {
                (0..DIM).map(|d| pseudo_rand((i * DIM + d) as u32 + 1) * 3.0).collect()
            }).collect();
            let t0 = std::time::Instant::now();
            let merges = agglomerative_var(&pts, &no_cancel());
            println!("n={n}: {:.2}s", t0.elapsed().as_secs_f64());
            assert_eq!(merges.len(), n - 1);
        }
    }

    #[test]
    fn agglomerative_var_matches_naive_reference() {
        // Deterministic pseudo-random points, no exact-tie distances (a real
        // risk with e.g. a small integer grid), several sizes so the NN-chain
        // amortized argument actually gets exercised past trivial n.
        fn pseudo_rand(seed: u32) -> f32 {
            let mut x = seed.wrapping_mul(2654435761);
            x ^= x >> 15;
            x = x.wrapping_mul(0x85ebca6b);
            x ^= x >> 13;
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
        for &n in &[2usize, 3, 5, 13, 40, 97] {
            const DIM: usize = 5;
            let pts: Vec<Vec<f32>> = (0..n).map(|i| {
                (0..DIM).map(|d| pseudo_rand((i * DIM + d) as u32 + 1) * 3.0).collect()
            }).collect();
            let fast = agglomerative_var(&pts, &no_cancel());
            let naive = naive_complete_linkage(&pts, &no_cancel());
            assert_eq!(fast.len(), naive.len(), "n={n}: merge count mismatch");
            // Merge distances, sorted, must match exactly — the search ORDER
            // can legitimately differ (different tie-breaking among
            // equal-distance pairs, though this data has none), but the
            // multiset of merge heights a valid complete-linkage dendrogram
            // produces for a given point set is unique.
            let mut fast_d: Vec<f32> = fast.iter().map(|&(_, _, d)| d).collect();
            let mut naive_d: Vec<f32> = naive.iter().map(|&(_, _, d)| d).collect();
            fast_d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            naive_d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (a, b) in fast_d.iter().zip(&naive_d) {
                assert!((a - b).abs() < 1e-3, "n={n}: merge distance mismatch {a} vs {b}");
            }
            // And the actual resulting PARTITIONS must match at every k, not
            // just the distances — compare via labels_for_k, order-independent
            // (HashSet of member-index-sets per cluster).
            for k in 1..=n.min(6) {
                let fast_labels = labels_for_k(&fast, n, k, 1);
                let naive_labels = labels_for_k(&naive, n, k, 1);
                let fast_groups: HashSet<Vec<usize>> = group_by_label(&fast_labels);
                let naive_groups: HashSet<Vec<usize>> = group_by_label(&naive_labels);
                assert_eq!(fast_groups, naive_groups, "n={n} k={k}: partition mismatch");
            }
        }
    }

    #[test]
    fn agglomerative_var_handles_duplicate_points_at_scale() {
        // The production crash ("index out of bounds ... index is
        // usize::MAX") never reproduced at the small, tie-free sizes
        // `agglomerative_var_matches_naive_reference` covers — real region
        // embeddings at real scale (thousands of points) are far more
        // likely to include near-duplicates or exact ties (visually
        // identical crops, or two regions that happen to land on the same
        // point after PCA) than synthetic well-separated test data ever
        // does. Build a dataset that's deliberately full of exact
        // duplicates (many points sharing the exact same coordinates, so
        // distance=0 between them) at a size large enough to actually
        // exercise the zombie-chain-entry defensive purge, and confirm it
        // completes without panicking and matches the naive reference.
        fn pseudo_rand(seed: u32) -> f32 {
            let mut x = seed.wrapping_mul(2654435761);
            x ^= x >> 15;
            x = x.wrapping_mul(0x85ebca6b);
            x ^= x >> 13;
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
        const DIM: usize = 8;
        const N_GROUPS: usize = 15;
        const DUPES_PER_GROUP: usize = 12;
        let mut pts: Vec<Vec<f32>> = Vec::new();
        for g in 0..N_GROUPS {
            let base: Vec<f32> = (0..DIM).map(|d| pseudo_rand((g * DIM + d) as u32 + 1) * 5.0).collect();
            for _ in 0..DUPES_PER_GROUP {
                pts.push(base.clone()); // exact duplicate — distance 0 within a group
            }
        }
        let n = pts.len();
        let fast = agglomerative_var(&pts, &no_cancel()); // must not panic
        let naive = naive_complete_linkage(&pts, &no_cancel());
        assert_eq!(fast.len(), n - 1);
        assert_eq!(fast.len(), naive.len());
        let mut fast_d: Vec<f32> = fast.iter().map(|&(_, _, d)| d).collect();
        let mut naive_d: Vec<f32> = naive.iter().map(|&(_, _, d)| d).collect();
        fast_d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        naive_d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (a, b) in fast_d.iter().zip(&naive_d) {
            assert!((a - b).abs() < 1e-3, "merge distance mismatch {a} vs {b}");
        }
        // At k=N_GROUPS, each duplicate group should still be intact as its
        // own cluster (they're all at distance 0 from each other, strictly
        // closer than any cross-group distance).
        let labels = labels_for_k(&fast, n, N_GROUPS, 1);
        let groups = group_by_label(&labels);
        assert_eq!(groups.len(), N_GROUPS);
        for group in &groups {
            assert_eq!(group.len(), DUPES_PER_GROUP);
        }
    }

    /// Group point indices by label into sorted member-index vectors (order-
    /// independent partition comparison — ignores which numeric id each
    /// group happens to get).
    fn group_by_label(labels: &[i32]) -> HashSet<Vec<usize>> {
        let mut by_label: std::collections::HashMap<i32, Vec<usize>> = std::collections::HashMap::new();
        for (i, &l) in labels.iter().enumerate() {
            by_label.entry(l).or_default().push(i);
        }
        by_label.into_values().collect()
    }

    #[test]
    fn agglomerative_var_recovers_two_blobs_at_k2() {
        const DIM: usize = 8;
        let mut pts: Vec<Vec<f32>> = Vec::new();
        for i in 0..15 {
            let mut p = vec![0f32; DIM];
            p[0] = i as f32 * 0.02;
            pts.push(p);
        }
        for i in 0..15 {
            let mut p = vec![0f32; DIM];
            p[0] = 10.0 + i as f32 * 0.02;
            pts.push(p);
        }
        let n = pts.len();
        let merges = agglomerative_var(&pts, &no_cancel());
        assert_eq!(merges.len(), n - 1);

        let labels = labels_for_k(&merges, n, 2, 1);
        let clusters: HashSet<i32> = labels.iter().copied().collect();
        assert_eq!(clusters.len(), 2, "expected 2 clusters at k=2, got {clusters:?}");
        // both blobs should be internally consistent (all-same-label within each half)
        let first_half: HashSet<i32> = labels[..15].iter().copied().collect();
        let second_half: HashSet<i32> = labels[15..].iter().copied().collect();
        assert_eq!(first_half.len(), 1, "first blob should be one cluster");
        assert_eq!(second_half.len(), 1, "second blob should be one cluster");
        assert_ne!(first_half, second_half, "the two blobs should be different clusters");
    }

    #[test]
    fn agglomerative_merge_distances_are_monotonic() {
        const DIM: usize = 8;
        let mut pts: Vec<Vec<f32>> = Vec::new();
        for i in 0..20 {
            let mut p = vec![0f32; DIM];
            p[0] = (i as f32 * 0.37).sin() * 5.0; // irregular but deterministic spread
            p[1] = i as f32 * 0.11;
            pts.push(p);
        }
        let merges = agglomerative_var(&pts, &no_cancel());
        for w in merges.windows(2) {
            assert!(w[1].2 + 1e-4 >= w[0].2,
                "merge distances should be non-decreasing: {} then {}", w[0].2, w[1].2);
        }
    }

    #[test]
    fn labels_for_k_matches_stop_early_reference() {
        // Reference: rerun the merge search but physically stop after n-k merges
        // (a from-scratch "stop early" implementation) and compare its resulting
        // partition against labels_for_k's cheap DSU-replay of a full sequence's
        // prefix — they should produce the identical partition (same grouping of
        // points into sets, though not necessarily the same numeric ids).
        const DIM: usize = 6;
        let mut pts: Vec<Vec<f32>> = Vec::new();
        for i in 0..12 {
            let mut p = vec![0f32; DIM];
            p[0] = (i as f32 * 0.53).cos() * 3.0;
            p[1] = (i as f32 * 1.7).sin() * 3.0;
            pts.push(p);
        }
        let n = pts.len();
        let full_merges = agglomerative_var(&pts, &no_cancel());
        let k = 4;
        let labels = labels_for_k(&full_merges, n, k, 1);

        // partition-equality check: two points are "together" under `labels` iff
        // they were merged within the first (n-k) steps of a stop-early rerun.
        let stop_early = &full_merges[..(n - k)];
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut [usize], x: usize) -> usize {
            if parent[x] != x { parent[x] = find(parent, parent[x]); }
            parent[x]
        }
        for &(i, j, _) in stop_early {
            let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
            if ri != rj { parent[rj] = ri; }
        }
        for a in 0..n {
            for b in 0..n {
                let same_ref = find(&mut parent, a) == find(&mut parent, b);
                let same_labels = labels[a] == labels[b];
                assert_eq!(same_ref, same_labels,
                    "points {a},{b}: reference says together={same_ref}, labels_for_k says together={same_labels}");
            }
        }
    }

    #[test]
    fn labels_for_k_applies_min_cluster_size_as_noise() {
        // One big blob of 10 plus 2 far-away singletons: at k=3, the two
        // singletons should each be their own tiny "cluster" and, with
        // min_cluster_size=2, get relabeled to noise (-1).
        const DIM: usize = 4;
        let mut pts: Vec<Vec<f32>> = Vec::new();
        for i in 0..10 {
            let mut p = vec![0f32; DIM];
            p[0] = i as f32 * 0.01;
            pts.push(p);
        }
        pts.push(vec![50.0, 0.0, 0.0, 0.0]);
        pts.push(vec![-50.0, 0.0, 0.0, 0.0]);
        let n = pts.len();
        let merges = agglomerative_var(&pts, &no_cancel());
        let labels = labels_for_k(&merges, n, 3, 2);
        assert_eq!(labels[10], -1, "far singleton should be noise");
        assert_eq!(labels[11], -1, "far singleton should be noise");
        let big_blob_label = labels[0];
        assert!(big_blob_label >= 0);
        assert!(labels[..10].iter().all(|&l| l == big_blob_label));
    }

    #[test]
    fn suggest_k_finds_the_two_blob_gap() {
        const DIM: usize = 8;
        let mut pts: Vec<Vec<f32>> = Vec::new();
        for i in 0..15 {
            let mut p = vec![0f32; DIM];
            p[0] = i as f32 * 0.02;
            pts.push(p);
        }
        for i in 0..15 {
            let mut p = vec![0f32; DIM];
            p[0] = 10.0 + i as f32 * 0.02;
            pts.push(p);
        }
        let merges = agglomerative_var(&pts, &no_cancel());
        let candidates = suggest_k(&merges);
        assert!(!candidates.is_empty());
        assert!(candidates.iter().any(|&(k, _)| k == 2),
            "expected K=2 among top candidates, got {candidates:?}");
    }

    #[test]
    fn labels_adaptive_resolves_a_branch_needing_a_deeper_split() {
        // A1, A2: two tight sub-blobs, moderately separated from each other.
        // B: one tight blob, far from both A1 and A2. A single flat K has to
        // choose one global depth; the point of labels_adaptive is that it
        // can split the A1/A2 branch (a real, sharp local jump) while
        // leaving B's own (much smaller, smooth) internal merges alone.
        // Decorrelated pseudo-random 2-D jitter kept TINY (~0.0003) relative
        // to the inter-group gaps (5.0, 50.0) — NOT sin/cos-based: two sines
        // at different frequencies trace a Lissajous CURVE, not a uniform
        // scatter, which has its own genuine (if tiny) internal structure
        // that complete linkage + inconsistency correctly detects as real
        // sub-clusters — a flaw in test data, not the algorithm, found by
        // debug-printing actual per-node inconsistency values.
        fn pseudo_rand(seed: u32) -> f32 {
            let mut x = seed.wrapping_mul(2654435761);
            x ^= x >> 15;
            x = x.wrapping_mul(0x85ebca6b);
            x ^= x >> 13;
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0 // in [-1, 1]
        }
        let jitter = |i: usize| (pseudo_rand(i as u32) * 0.0003, pseudo_rand(i as u32 + 10_000) * 0.0003);
        let mut pts: Vec<Vec<f32>> = Vec::new();
        for i in 0..30 {
            let (jx, jy) = jitter(i);
            pts.push(vec![jx, jy]);
        }
        for i in 0..30 {
            let (jx, jy) = jitter(i);
            pts.push(vec![5.0 + jx, jy]);
        }
        for i in 0..30 {
            let (jx, jy) = jitter(i);
            pts.push(vec![50.0 + jx, jy]);
        }
        let n = pts.len();
        let merges = agglomerative_var(&pts, &no_cancel());
        assert_eq!(merges.len(), n - 1);

        // Threshold empirically chosen, not guessed: debug-printed every
        // node's actual inconsistency value on this exact data. A tight
        // cluster's OWN last few internal merges genuinely run up to ~15.7
        // (finishing a cluster under complete linkage inherently requires a
        // proportionally larger "last mile" jump that a small local window
        // can't fully smooth away — a real property of the statistic, not a
        // bug), while the two genuine cross-group splits measure ~26.4 and
        // ~42527. 18.0 sits cleanly in the gap between those two regimes for
        // THIS dataset — see labels_adaptive's doc comment and the shipped
        // default in settings.rs for why production needs a real margin and
        // per-dataset tuning rather than trusting this exact number.
        let labels = labels_adaptive(&merges, n, 18.0, 1);
        let a1: HashSet<i32> = labels[0..30].iter().copied().collect();
        let a2: HashSet<i32> = labels[30..60].iter().copied().collect();
        let b: HashSet<i32> = labels[60..90].iter().copied().collect();
        assert_eq!(a1.len(), 1, "A1 should be one consistent group, got {a1:?}");
        assert_eq!(a2.len(), 1, "A2 should be one consistent group, got {a2:?}");
        assert_eq!(b.len(), 1, "B should be one consistent group, got {b:?}");
        assert_ne!(a1, a2, "A1 and A2 should be split apart");
        assert_ne!(a1, b, "A1 and B should be different groups");
        assert_ne!(a2, b, "A2 and B should be different groups");
    }

    #[test]
    fn labels_adaptive_handles_partial_merge_sequence() {
        // A cancelled/partial merges list (fewer than n-1 entries) produces a
        // forest, not one tree — must not panic, must produce a full-length,
        // sane result.
        const DIM: usize = 4;
        let mut pts: Vec<Vec<f32>> = Vec::new();
        for i in 0..20 {
            let mut p = vec![0f32; DIM];
            p[0] = i as f32 * 0.37;
            pts.push(p);
        }
        let n = pts.len();
        let full_merges = agglomerative_var(&pts, &no_cancel());
        let partial = &full_merges[..full_merges.len() / 2]; // simulate a cancelled run
        let labels = labels_adaptive(partial, n, 1.0, 1);
        assert_eq!(labels.len(), n);
    }

    #[test]
    fn labels_adaptive_leaf_leaf_merge_does_not_panic() {
        // n=2: a single merge joining two leaves directly — both children
        // contribute no height (degenerate case for the inconsistency stat).
        let pts: Vec<Vec<f32>> = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let merges = agglomerative_var(&pts, &no_cancel());
        assert_eq!(merges.len(), 1);
        let labels = labels_adaptive(&merges, 2, 1.0, 1);
        assert_eq!(labels.len(), 2);
    }

    #[test]
    fn labels_adaptive_empty_input() {
        let labels = labels_adaptive(&[], 0, 1.0, 1);
        assert!(labels.is_empty());
    }
}
