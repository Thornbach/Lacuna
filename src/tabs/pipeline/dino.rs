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
    std::env::var("LACUNA_USE_ORT").map(|v| v != "0" && !v.is_empty()).unwrap_or(false)
}

pub struct DinoExtractor {
    model:   Model,
    res:     u32,
    /// Wall-time (ms) of the most recent forward — for the pipeline timer.
    pub last_ms: f32,
}

pub struct DinoFeatures {
    pub feat: Vec<f32>, // tokens*dim row-major
    pub grid: usize,    // tokens per side (res/patch)
    pub dim:  usize,    // feature dim (1536)
}

/// Resolve the safetensors weights for the BURN path.
fn resolve_burn_weights(model_path: &Path) -> PathBuf {
    if let Ok(p) = std::env::var("LACUNA_DINO_WEIGHTS") {
        return PathBuf::from(p);
    }
    if let Some(dir) = model_path.parent() {
        let sib = dir.join("dino_weights.safetensors");
        if sib.exists() {
            return sib;
        }
    }
    PathBuf::from(r"E:\PhD_TobiMu\02_code\FoliarToolbox\port\dino_weights.safetensors")
}

impl DinoExtractor {
    pub fn load(model_path: &Path, res: u32) -> Result<Self, String> {
        // Optional ort fallback (only if compiled in AND LACUNA_USE_ORT=1).
        #[cfg(feature = "ort-backend")]
        if use_ort() {
            eprintln!("[dino] backend=ort (LACUNA_USE_ORT) {}", model_path.display());
            return Ok(Self { model: Model::Ort(crate::tabs::build_session(model_path)?), res, last_ms: 0.0 });
        }
        // Default: pure-Rust BURN.
        let device = create_infer_device();
        let wpath = resolve_burn_weights(model_path);
        eprintln!("[dino] backend=BURN ({}) weights={}",
                  crate::tabs::recon_train::model::backend_name(), wpath.display());
        let net = crate::dino_burn::DinoV3Burn::<InferBackend>::load(&wpath.to_string_lossy(), &device)?;
        Ok(Self { model: Model::Burn(Box::new(net), device), res, last_ms: 0.0 })
    }

    pub fn res(&self) -> u32 { self.res }

    /// Resize `img` to res×res, run the model, return per-patch features.
    pub fn features(&mut self, img: &RgbImage) -> Result<DinoFeatures, String> {
        let res = self.res;
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
