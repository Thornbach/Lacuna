//! DINOv3 multi-layer patch-feature extractor via ONNX Runtime (`ort`).
//!
//! Loads `dinov3_vitb16_<res>.onnx` (exported by 1Help/export_dinov3.py, which
//! bakes in ImageNet normalization and per-layer L2-norm). The Rust side only
//! has to resize a tile to `res`×`res`, feed RGB in 0..1 (CHW), and read back the
//! `[1, g*g, dim]` patch features (dim = 1536 for the 2-layer ViT-B export).
//!
//! NOTE (parity): the exported graph matched torch f32 to ~1e-6 (cosine 1.0). The
//! one residual difference vs the Python bank is the tile→res resize filter
//! (`image` Triangle here vs torch bilinear align_corners=False); good enough for
//! the S2 skeleton, to be matched exactly in S5 if it moves the thresholds.

//! ── BURN alternative (no ort/onnxruntime/cuDNN) ────────────────────────────────
//! Set `LACUNA_DINO_BURN=1` to run the hand-written pure-BURN DINOv3
//! (`crate::dino_burn`) on the same backend recon uses (burn-cuda on a cuda build,
//! NdArray on CPU). Weights come from a safetensors next to the .onnx (or
//! `LACUNA_DINO_WEIGHTS`, or the dev fallback in port/). Output is byte-for-byte the
//! same `DinoFeatures` layout, so the rest of the pipeline is unchanged. Validated
//! vs the ONNX oracle to max|Δ|≈7.7e-6 (see `--dino-burn-validate`).

use std::path::{Path, PathBuf};

use image::{imageops::FilterType, RgbImage};
#[cfg(feature = "ort-backend")]
use ort::value::Tensor as OrtTensor;
use rayon::prelude::*;

use crate::tabs::recon_train::model::{create_infer_device, InferBackend};

type BurnDevice = <InferBackend as burn::tensor::backend::Backend>::Device;

enum Model {
    #[cfg(feature = "ort-backend")]
    Ort(ort::session::Session),
    Burn(Box<crate::dino_burn::DinoV3Burn<InferBackend>>, BurnDevice),
}

/// True if the optional ort path is compiled AND requested at runtime.
#[cfg(feature = "ort-backend")]
fn use_ort() -> bool {
    // An explicit override always wins.
    if let Ok(v) = std::env::var("LACUNA_USE_ORT") {
        return v != "0" && !v.is_empty();
    }
    // Otherwise ORT is the right default ONLY when BURN would fall back to the
    // pure-Rust ndarray CPU backend: measured at 512, ORT CPU is 921 ms vs
    // ndarray's 4226 ms, a 4.6x win.
    //
    // It is the WRONG default when a GPU backend is compiled in. DINO is the
    // pipeline's bottleneck, and ORT here is the CPU execution provider (the
    // GPU EPs are the separate `ort-cuda` / `directml` features). Defaulting it
    // ON unconditionally would quietly move the single heaviest model OFF the
    // GPU that `wgpu-gpu`/`cuda` was enabled for — CUDA measured 212 ms at 512,
    // so that is a ~4x regression in the exact build meant to be fast.
    //
    // Note BURN also has no fixed input shape, so LACUNA_USE_ORT=0 remains the
    // way to run DINO at a resolution other than the ONNX's baked-in 512.
    cfg!(not(any(feature = "cuda", feature = "wgpu-gpu")))
}

pub struct DinoExtractor {
    model:   Model,
    res:     u32,
    /// Wall-time (ms) of the most recent forward — for the pipeline timer.
    pub last_ms: f32,
    /// Wall-time (ms) of the most recent call's image resize + CHW-repack
    /// step (CPU, before the forward pass) — split out from `last_ms` so
    /// callers can attribute cost accurately instead of it silently falling
    /// into whatever bucket happens to wrap the call (a real reported
    /// confusion: this used to be invisible, hidden inside a caller-side
    /// "pool" timing bucket that was actually measuring something else).
    pub last_prep_ms: f32,
}

pub struct DinoFeatures {
    pub feat: Vec<f32>, // tokens*dim row-major
    pub grid: usize,    // tokens per side (res/patch)
    pub dim:  usize,    // feature dim (1536)
}

impl DinoFeatures {
    /// Mean-pool all `grid*grid` patch tokens into one `dim`-length embedding —
    /// used to give a whole region (via its context crop) a single semantic
    /// feature vector for unsupervised clustering, instead of per-patch scores.
    pub fn mean_pool(&self) -> Vec<f32> {
        let n = self.grid * self.grid;
        let mut out = vec![0f32; self.dim];
        if n == 0 {
            return out;
        }
        for t in 0..n {
            let row = &self.feat[t * self.dim..(t + 1) * self.dim];
            for (o, &v) in out.iter_mut().zip(row) {
                *o += v;
            }
        }
        for o in &mut out {
            *o /= n as f32;
        }
        out
    }
}

/// Resolve the safetensors weights for the BURN path.
fn resolve_burn_weights(model_path: &Path) -> PathBuf {
    crate::paths::resolve_weights(model_path, "dino_weights.safetensors", "LACUNA_DINO_WEIGHTS")
}

impl DinoExtractor {
    pub fn load(model_path: &Path, res: u32) -> Result<Self, String> {
        // Optional ort fallback (only if compiled in AND LACUNA_USE_ORT=1).
        #[cfg(feature = "ort-backend")]
        if use_ort() {
            eprintln!("[dino] backend=ort (LACUNA_USE_ORT) {}", model_path.display());
            return Ok(Self { model: Model::Ort(crate::tabs::build_session(model_path)?), res, last_ms: 0.0, last_prep_ms: 0.0 });
        }
        // Default: pure-Rust BURN.
        let device = create_infer_device();
        let wpath = resolve_burn_weights(model_path);
        eprintln!("[dino] backend=BURN ({}) weights={}",
                  crate::tabs::recon_train::model::backend_name(), wpath.display());
        let net = crate::dino_burn::DinoV3Burn::<InferBackend>::load(&wpath.to_string_lossy(), &device)?;
        Ok(Self { model: Model::Burn(Box::new(net), device), res, last_ms: 0.0, last_prep_ms: 0.0 })
    }

    pub fn res(&self) -> u32 { self.res }

    /// Resize `img` to the extractor's own configured res×res, run the
    /// model, return per-patch features.
    pub fn features(&mut self, img: &RgbImage) -> Result<DinoFeatures, String> {
        self.features_at(img, self.res)
    }

    /// Like `features`, but resizes to an explicit `res` instead of the
    /// extractor's own configured resolution — lets a caller trade input
    /// resolution for speed on a per-call basis without a second loaded
    /// model instance (same weights; the Burn forward pass is already
    /// resolution-agnostic, reading `H,W` straight off the input tensor —
    /// confirmed when making it batch-generic — only the resize target and
    /// resulting patch grid differ). Used for region-embedding crops, which
    /// don't need the per-patch precision full-res per-tile detection does.
    /// `res` must be a multiple of the model's patch size (16).
    pub fn features_at(&mut self, img: &RgbImage, res: u32) -> Result<DinoFeatures, String> {
        let resized = image::imageops::resize(img, res, res, FilterType::Triangle);
        let n = (res * res) as usize;
        // CHW, [0,1] — identical layout for both backends (ImageNet-normalize is
        // baked into the ONNX graph AND into DinoV3Burn::forward).
        let mut data = vec![0f32; 3 * n];
        for y in 0..res {
            for x in 0..res {
                let px = resized.get_pixel(x, y);
                let idx = (y * res + x) as usize;
                data[idx] = px[0] as f32 / 255.0;
                data[n + idx] = px[1] as f32 / 255.0;
                data[2 * n + idx] = px[2] as f32 / 255.0;
            }
        }

        match &mut self.model {
            #[cfg(feature = "ort-backend")]
            Model::Ort(session) => {
                let input = OrtTensor::from_array(([1usize, 3, res as usize, res as usize], data))
                    .map_err(|e| format!("dino input tensor: {e}"))?;
                let t_run = std::time::Instant::now();
                let outputs = session
                    .run(ort::inputs!["images" => input])
                    .map_err(|e| format!("dino run: {e}"))?;
                self.last_ms = t_run.elapsed().as_secs_f32() * 1000.0;
                let (shape, feat) = outputs["features"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("dino extract: {e}"))?;
                let tokens = shape[1] as usize;
                let dim = shape[2] as usize;
                let grid = (tokens as f64).sqrt().round() as usize;
                Ok(DinoFeatures { feat: feat.to_vec(), grid, dim })
            }
            Model::Burn(net, device) => {
                let x = burn::tensor::Tensor::<InferBackend, 4>::from_data(
                    burn::tensor::TensorData::new(data, [1, 3, res as usize, res as usize]),
                    device,
                );
                let t_run = std::time::Instant::now();
                let out = net.forward(x); // [1, tokens, 1536]
                let dims = out.dims();
                let feat = out.into_data().to_vec::<f32>()
                    .map_err(|e| format!("dino burn extract: {e:?}"))?;
                self.last_ms = t_run.elapsed().as_secs_f32() * 1000.0;
                let dim = dims[2];
                let grid = (dims[1] as f64).sqrt().round() as usize;
                Ok(DinoFeatures { feat, grid, dim })
            }
        }
    }

    /// Batched variant of `features_at`: resizes every image to res×res and
    /// runs them through ONE forward pass instead of one call per image,
    /// returning one `DinoFeatures` per input in the same order. Exists
    /// because the region-embedding step's cost is dominated by the forward
    /// pass's fixed per-call overhead (not image size — every crop already
    /// gets resized to the same fixed `res` regardless of its own size), so
    /// calling it once per region (as `region_dino_embed` used to) scales
    /// directly with region count; batching amortizes that overhead across
    /// however many regions are passed in one call. `imgs` should be a
    /// bounded chunk (the caller decides the chunk size), not every region
    /// on a leaf at once — an unbounded batch risks a very large GPU
    /// allocation for a leaf with hundreds of regions.
    ///
    /// The `ort` backend (non-default, opt-in via `LACUNA_USE_ORT`) was
    /// never restructured for batching — falls back to a plain per-image
    /// loop there, still correct, just without the speedup, since `ort`
    /// isn't the path this app actually ships with.
    pub fn features_batch_at(&mut self, imgs: &[RgbImage], res: u32) -> Result<Vec<DinoFeatures>, String> {
        if imgs.is_empty() {
            return Ok(Vec::new());
        }
        #[cfg(feature = "ort-backend")]
        if matches!(self.model, Model::Ort(_)) {
            return imgs.iter().map(|img| self.features_at(img, res)).collect();
        }

        let n = imgs.len();
        let npx = (res * res) as usize;
        let t_prep = std::time::Instant::now();
        let mut data = vec![0f32; n * 3 * npx];
        // Per-image resize + CHW-repack, in parallel — each image writes
        // only to its own disjoint `3*npx`-sized chunk of `data`, so there's
        // no aliasing between rayon tasks. Previously sequential and
        // un-timed (its cost silently fell into a caller-side "pool"
        // bucket that was really measuring something else entirely).
        data.par_chunks_mut(3 * npx).zip(imgs.par_iter()).for_each(|(chunk, img)| {
            let resized = image::imageops::resize(img, res, res, FilterType::Triangle);
            for y in 0..res {
                for x in 0..res {
                    let px = resized.get_pixel(x, y);
                    let idx = (y * res + x) as usize;
                    chunk[idx] = px[0] as f32 / 255.0;
                    chunk[npx + idx] = px[1] as f32 / 255.0;
                    chunk[2 * npx + idx] = px[2] as f32 / 255.0;
                }
            }
        });
        self.last_prep_ms = t_prep.elapsed().as_secs_f32() * 1000.0;

        let (net, device) = match &mut self.model {
            #[cfg(feature = "ort-backend")]
            Model::Ort(_) => unreachable!("ort case already returned above"),
            Model::Burn(net, device) => (net, device),
        };
        let x = burn::tensor::Tensor::<InferBackend, 4>::from_data(
            burn::tensor::TensorData::new(data, [n, 3, res as usize, res as usize]),
            device,
        );
        let t_run = std::time::Instant::now();
        let out = net.forward(x); // [n, tokens, 1536]
        let dims = out.dims();
        let feat_all = out.into_data().to_vec::<f32>()
            .map_err(|e| format!("dino burn extract: {e:?}"))?;
        self.last_ms = t_run.elapsed().as_secs_f32() * 1000.0;
        let (tokens, dim) = (dims[1], dims[2]);
        let grid = (tokens as f64).sqrt().round() as usize;
        let per_img = tokens * dim;
        Ok((0..n).map(|b| DinoFeatures {
            feat: feat_all[b * per_img..(b + 1) * per_img].to_vec(),
            grid, dim,
        }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Smoke test the real exported ONNX end-to-end through ort.
    /// cargo test --no-default-features dino_smoke -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dino_smoke() {
        let model = PathBuf::from(
            r"E:\PhD_TobiMu\02_code\02paper\anomaly\dinov3_vitb16_512.onnx",
        );
        let image = PathBuf::from(
            r"E:\PhD_TobiMu\02_code\02paper\leaf_segmentation\test\8f6a70ebb6ae0c10090b5ac3ec98e837.jpg",
        );
        assert!(model.exists(), "onnx missing: {}", model.display());

        let mut dino = DinoExtractor::load(&model, 512).expect("load");
        let img = image::open(&image).expect("open").to_rgb8();
        let f = dino.features(&img).expect("features");

        println!("grid={} dim={} tokens={}", f.grid, f.dim, f.feat.len() / f.dim);
        assert_eq!(f.dim, 1536, "expected 1536-D");
        assert_eq!(f.grid, 32, "expected 32x32 grid at res 512");
        assert_eq!(f.feat.len(), 32 * 32 * 1536);
        assert!(f.feat.iter().all(|v| v.is_finite()), "non-finite features");

        // each layer half is L2-normed -> full row norm^2 ≈ 2.0
        let row0 = &f.feat[0..f.dim];
        let nrm2: f32 = row0.iter().map(|v| v * v).sum();
        println!("row0 ‖·‖² = {nrm2:.4} (expect ~2.0: two unit-norm layer halves)");
        assert!((nrm2 - 2.0).abs() < 0.1, "row norm² {nrm2}");
    }
}
