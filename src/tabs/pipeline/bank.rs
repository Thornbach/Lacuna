//! PatchCore coreset memory bank, loaded from `coreset_bank.bin`.
//!
//! Format (little-endian, written by 1Help/export_assets.py):
//!   u32 N, u32 D, then N*D float32 row-major.
//!
//! The DINO anomaly score for a query patch is the mean L2 distance to its k
//! nearest bank vectors (reproduces detector.anomaly_map). The features are the
//! concatenated per-layer L2-normed DINOv3 tokens, so they are NOT unit-norm
//! overall — use true squared-L2 (‖q‖²+‖b‖²−2q·b), not a cosine shortcut.
//!
//! v1 uses a straightforward CPU implementation. It is O(n_q · N · D); for the
//! full 150k bank this is the slow path — a GPU GEMM (burn) is the planned
//! optimization, but correctness comes first here.

use std::path::Path;

use burn::tensor::{backend::Backend, Tensor, TensorData};
use rayon::prelude::*;

pub struct CoresetBank {
    pub n: usize,
    pub d: usize,
    flat:   Vec<f32>, // n*d row-major
    sqnorm: Vec<f32>, // ‖row‖² precomputed
}

impl CoresetBank {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read bank: {e}"))?;
        if bytes.len() < 8 {
            return Err("coreset_bank.bin too small for header".into());
        }
        let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let d = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let need = 8 + n * d * 4;
        if bytes.len() < need {
            return Err(format!("coreset_bank.bin truncated: have {}, need {need}", bytes.len()));
        }
        let mut flat = vec![0f32; n * d];
        for (i, slot) in flat.iter_mut().enumerate() {
            let o = 8 + i * 4;
            *slot = f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        }
        let mut sqnorm = vec![0f32; n];
        for (r, s) in sqnorm.iter_mut().enumerate() {
            *s = flat[r * d..r * d + d].iter().map(|v| v * v).sum();
        }
        Ok(Self { n, d, flat, sqnorm })
    }

    /// Wrap an in-memory bank (e.g. freshly fit by the trainer) without a disk
    /// round-trip. `flat` is `n*d` row-major.
    pub fn from_parts(flat: Vec<f32>, n: usize, d: usize) -> Self {
        let mut sqnorm = vec![0f32; n];
        for (r, s) in sqnorm.iter_mut().enumerate() {
            *s = flat[r * d..r * d + d].iter().map(|v| v * v).sum();
        }
        Self { n, d, flat, sqnorm }
    }

    /// Mean L2 distance from each query row to its `k` nearest bank rows.
    /// `queries` is `n_q * d` row-major; returns `n_q` scores. Parallelized over
    /// queries with rayon (the inner full-bank scan is the cost).
    pub fn knn_mean_dist(&self, queries: &[f32], n_q: usize, k: usize) -> Vec<f32> {
        let d = self.d;
        let k = k.clamp(1, self.n.max(1));
        (0..n_q)
            .into_par_iter()
            .map(|qi| self.knn_one(&queries[qi * d..qi * d + d], k))
            .collect()
    }

    fn knn_one(&self, q: &[f32], k: usize) -> f32 {
        let d = self.d;
        let qsq: f32 = q.iter().map(|v| v * v).sum();
        let mut topk = vec![f32::INFINITY; k]; // ascending
        for r in 0..self.n {
            let b = &self.flat[r * d..r * d + d];
            let mut dot = 0f32;
            for j in 0..d {
                dot += q[j] * b[j];
            }
            let dist2 = (qsq + self.sqnorm[r] - 2.0 * dot).max(0.0);
            if dist2 < topk[k - 1] {
                let mut p = k - 1;
                while p > 0 && topk[p - 1] > dist2 {
                    topk[p] = topk[p - 1];
                    p -= 1;
                }
                topk[p] = dist2;
            }
        }
        topk.iter().map(|d2| d2.sqrt()).sum::<f32>() / k as f32
    }
}

/// Backend-agnostic kNN scoring, so `detect` doesn't care whether the bank is
/// the CPU (`CoresetBank`) or GPU (`GpuBank`) implementation.
pub trait KnnScorer {
    fn knn_mean_dist(&self, queries: &[f32], n_q: usize, k: usize) -> Vec<f32>;
}

impl KnnScorer for CoresetBank {
    fn knn_mean_dist(&self, q: &[f32], n_q: usize, k: usize) -> Vec<f32> {
        CoresetBank::knn_mean_dist(self, q, n_q, k)
    }
}

impl<B: Backend> KnnScorer for GpuBank<B> {
    fn knn_mean_dist(&self, q: &[f32], n_q: usize, k: usize) -> Vec<f32> {
        GpuBank::knn_mean_dist(self, q, n_q, k)
    }
}

/// GPU-resident bank for fast kNN via a single GEMM per query chunk.
/// Generic over the burn backend: `Cuda` in the default build (fast), `NdArray`
/// in the CPU build. Squared-L2: ‖q‖² + ‖b‖² − 2·(q·bᵀ).
pub struct GpuBank<B: Backend> {
    bank_dn: Tensor<B, 2>, // [d, n] — pre-transposed ONCE (not per call)
    bsq:     Tensor<B, 2>, // [1, n]
    pub n:   usize,
    pub d:   usize,
    device:  B::Device,
}

impl<B: Backend> GpuBank<B> {
    pub fn new(bank: &CoresetBank, device: B::Device) -> Self {
        let (n, d) = (bank.n, bank.d);
        let bank_nd = Tensor::<B, 2>::from_data(TensorData::new(bank.flat.clone(), [n, d]), &device);
        let bsq = bank_nd.clone().mul(bank_nd.clone()).sum_dim(1).reshape([1, n]); // [1, n]
        let bank_dn = bank_nd.transpose(); // [d, n] — materialize once, reuse every tile
        Self { bank_dn, bsq, n, d, device }
    }

    /// Mean L2 distance to the k nearest bank rows for each query row.
    /// `queries` is `n_q * d` row-major. Chunked to bound the [chunk, n] matrix.
    pub fn knn_mean_dist(&self, queries: &[f32], n_q: usize, k: usize) -> Vec<f32> {
        let d = self.d;
        let k = k.clamp(1, self.n.max(1));
        let chunk = 256usize;
        let mut out = Vec::with_capacity(n_q);
        let mut i = 0;
        while i < n_q {
            let c = chunk.min(n_q - i);
            let q = Tensor::<B, 2>::from_data(
                TensorData::new(queries[i * d..(i + c) * d].to_vec(), [c, d]),
                &self.device,
            );
            let qsq = q.clone().mul(q.clone()).sum_dim(1);          // [c, 1]
            let dot = q.matmul(self.bank_dn.clone());              // [c, n]
            let mut d2 = qsq + self.bsq.clone() - dot.mul_scalar(2.0); // [c, n]
            // k smallest via iterative min_dim — burn's topk over n=150k is a slow
            // sort; k min-reductions are O(k·n) and stay on the GPU as fast ops.
            let mut acc = Tensor::<B, 2>::zeros([c, 1], &self.device);
            for _ in 0..k {
                let mins = d2.clone().min_dim(1);                  // [c, 1]
                acc = acc + mins.clone().clamp_min(0.0).sqrt();
                // (d2 - rowmin) <= 0 is true only at the row minimum; broadcasting
                // sub works where `equal` does not.
                let is_min = (d2.clone() - mins).lower_equal_elem(0.0); // [c, n] bool
                d2 = d2.mask_fill(is_min, f32::INFINITY);         // exclude this neighbour
            }
            let mean = acc.div_scalar(k as f32);                  // [c, 1]
            let v: Vec<f32> = mean.into_data().to_vec().expect("knn to_vec");
            out.extend(v);
            i += c;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_bank(path: &Path, n: u32, d: u32, rows: &[f32]) {
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&n.to_le_bytes()).unwrap();
        f.write_all(&d.to_le_bytes()).unwrap();
        for v in rows {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }

    #[test]
    fn bank_roundtrip_and_knn() {
        // 4 bank rows in 2-D: (0,0) (1,0) (0,1) (5,5)
        let rows = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 5.0, 5.0];
        let path = std::env::temp_dir().join("foliar_test_bank.bin");
        write_bank(&path, 4, 2, &rows);

        let bank = CoresetBank::load(&path).unwrap();
        assert_eq!((bank.n, bank.d), (4, 2));

        // query (0,0): k=1 nearest is itself -> dist 0
        let s = bank.knn_mean_dist(&[0.0, 0.0], 1, 1);
        assert!(s[0].abs() < 1e-5, "k=1 self dist {}", s[0]);

        // query (0,0): k=2 -> nearest two are (0,0) d=0 and (1,0)/(0,1) d=1 -> mean 0.5
        let s2 = bank.knn_mean_dist(&[0.0, 0.0], 1, 2);
        assert!((s2[0] - 0.5).abs() < 1e-5, "k=2 mean {}", s2[0]);

        // far query (5,5): k=1 nearest is (5,5) -> 0
        let s3 = bank.knn_mean_dist(&[5.0, 5.0], 1, 1);
        assert!(s3[0].abs() < 1e-5, "far self dist {}", s3[0]);
    }

    #[test]
    fn gpu_matches_cpu() {
        use burn::backend::NdArray;
        // 6 bank rows in 4-D, 3 queries — deterministic pseudo-values
        let rows: Vec<f32> = (0..6 * 4).map(|i| ((i * 7 + 3) % 11) as f32 / 11.0).collect();
        let path = std::env::temp_dir().join("foliar_test_bank_gpu.bin");
        write_bank(&path, 6, 4, &rows);
        let bank = CoresetBank::load(&path).unwrap();

        let queries: Vec<f32> = (0..3 * 4).map(|i| ((i * 5 + 1) % 9) as f32 / 9.0).collect();
        let cpu = bank.knn_mean_dist(&queries, 3, 2);

        let device = <NdArray as Backend>::Device::default();
        let gpu = GpuBank::<NdArray>::new(&bank, device);
        let g = gpu.knn_mean_dist(&queries, 3, 2);

        for (a, b) in cpu.iter().zip(g.iter()) {
            assert!((a - b).abs() < 1e-4, "cpu {a} vs gpu {b}");
        }
    }
}
