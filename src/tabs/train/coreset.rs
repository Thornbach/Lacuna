//! Streaming coreset construction for the in-app detector trainer.
//!
//! v1 uses a bottom-k reservoir: every patch feature gets a uniform random key
//! and we keep the `cap` rows with the smallest keys — an exact uniform random
//! sample (random coreset) of everything seen, with RAM bounded at `cap` rows.
//! (Greedy k-center can be added later as a quality toggle.)

use rand::{rngs::SmallRng, Rng, SeedableRng};

pub struct ReservoirSampler {
    cap:  usize,
    d:    usize,
    buf:  Vec<f32>, // rows*d, row-major
    keys: Vec<f32>, // one per row
    rng:  SmallRng,
}

impl ReservoirSampler {
    pub fn new(cap: usize, d: usize, seed: u64) -> Self {
        Self { cap, d, buf: Vec::new(), keys: Vec::new(), rng: SmallRng::seed_from_u64(seed) }
    }

    /// Add `m` rows (`x` is `m*d` row-major).
    pub fn add_rows(&mut self, x: &[f32], m: usize) {
        if m == 0 {
            return;
        }
        debug_assert_eq!(x.len(), m * self.d);
        for _ in 0..m {
            self.keys.push(self.rng.gen::<f32>());
        }
        self.buf.extend_from_slice(x);
        if self.keys.len() > self.cap {
            self.trim();
        }
    }

    fn trim(&mut self) {
        let n = self.keys.len();
        let mut idx: Vec<usize> = (0..n).collect();
        // partition so the `cap` smallest keys come first
        idx.select_nth_unstable_by(self.cap, |&a, &b| {
            self.keys[a].partial_cmp(&self.keys[b]).unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(self.cap);

        let mut nb = Vec::with_capacity(self.cap * self.d);
        let mut nk = Vec::with_capacity(self.cap);
        for &j in &idx {
            nb.extend_from_slice(&self.buf[j * self.d..j * self.d + self.d]);
            nk.push(self.keys[j]);
        }
        self.buf = nb;
        self.keys = nk;
    }

    pub fn len(&self) -> usize { self.keys.len() }
    pub fn is_empty(&self) -> bool { self.keys.is_empty() }

    /// Consume the sampler into (flat rows, n_rows). `flat.len() == n_rows * d`.
    pub fn into_pool(self) -> (Vec<f32>, usize) {
        let n = self.keys.len();
        (self.buf, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservoir_bounded_and_uniform() {
        let cap = 1000;
        let d = 4;
        let mut s = ReservoirSampler::new(cap, d, 0);
        // stream 50 batches of 5000 rows; values = row index so we can check spread
        let mut seen = 0u32;
        for _ in 0..50 {
            let mut batch = Vec::with_capacity(5000 * d);
            for _ in 0..5000 {
                for c in 0..d {
                    batch.push((seen as f32) + c as f32 * 0.01);
                }
                seen += 1;
            }
            s.add_rows(&batch, 5000);
        }
        assert_eq!(s.len(), cap, "reservoir must be bounded at cap");
        let (flat, n) = s.into_pool();
        assert_eq!(n, cap);
        assert_eq!(flat.len(), cap * d);
        // uniform sample of 0..250000 -> mean of col0 should be ~125000
        let mean: f64 = (0..n).map(|r| flat[r * d] as f64).sum::<f64>() / n as f64;
        assert!((mean - 125_000.0).abs() < 20_000.0, "sample mean {mean} not ~uniform");
    }
}
