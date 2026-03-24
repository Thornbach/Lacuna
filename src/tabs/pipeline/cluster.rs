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
}
