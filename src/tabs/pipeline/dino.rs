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
    /// Set when the loaded ort graph has a FIXED input resolution, which every
    /// export from `1Help/export_dinov3.py` does (`dynamic_axes=None`).
    ///
    /// When present it overrides the per-call `res` in `features_at`. That looks
    /// heavy-handed, but the alternative is worse: `features_at` takes a res
    /// argument and several callers pass their own (region-embed crops, the
    /// bench, projection), so any one of them can hand a fixed graph a shape it
    /// cannot accept and kill the run with a raw ort error. Honouring the graph
    /// makes those calls merely slower or coarser instead of fatal. `None` for
    /// BURN and for a genuinely dynamic graph, where the caller's res is used.
    ort_fixed_res: Option<u32>,
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

/// The input resolution an ONNX graph was exported at, if it is fixed.
///
/// `1Help/export_dinov3.py` exports with `dynamic_axes=None`, so the graph's
/// `images` input is a hard `[1, 3, RES, RES]`. Feeding it anything else fails
/// with "Got invalid dimensions for input: images", which is a *load-time*
/// mismatch reported at run time — the least useful moment.
///
/// Returns `None` for a dynamic axis (`-1`) or a non-square input, where the
/// caller's requested resolution is authoritative and nothing needs overriding.
#[cfg(feature = "ort-backend")]
fn ort_input_res(session: &ort::session::Session) -> Option<u32> {
    use ort::value::ValueType;
    let input = session.inputs.first()?;
    let ValueType::Tensor { shape, .. } = &input.input_type else {
        return None;
    };
    // NCHW.
    let (h, w) = (*shape.get(2)?, *shape.get(3)?);
    if h > 0 && h == w {
        Some(h as u32)
    } else {
        None
    }
}

/// Whether `model_path` is something ONNX Runtime can actually open.
///
/// Guards an upgrade path that would otherwise be a hard failure: the v0.4 cpu
/// package shipped `dino_weights.safetensors` and no `.onnx` at all, so an
/// existing `settings.json` can still point DINO at a safetensors file. On a CPU
/// build `use_ort()` is now true, and handing that to ort fails to parse.
///
/// Returning false here falls through to BURN, which reads exactly that file.
/// BURN on CPU is roughly 7x slower — but a slow run beats a dead one, and the
/// log says plainly what happened and how to fix it.
#[cfg(feature = "ort-backend")]
fn ort_loadable(model_path: &Path) -> bool {
    let is_onnx = model_path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("onnx"));
    if !is_onnx {
        eprintln!(
            "[dino] {} is not an .onnx - using BURN instead (slower). \
             Point the DINO model at models/dino.onnx for the fast path.",
            model_path.display()
        );
    }
    is_onnx
}

impl DinoExtractor {
    pub fn load(model_path: &Path, res: u32) -> Result<Self, String> {
        // Optional ort fallback (only if compiled in AND LACUNA_USE_ORT=1).
        #[cfg(feature = "ort-backend")]
        if use_ort() && ort_loadable(model_path) {
            let session = crate::tabs::build_session(model_path)?;
            // TRUST THE GRAPH, NOT THE CONFIG. The requested `res` comes from
            // `worker::default_dino_res()` (256 on CPU), but the .onnx actually
            // on disk decides what can run: a settings.json written by an older
            // build still points at the 512 export, and the dev tree has both
            // models/dino.onnx (512) and models/cpu256/dino.onnx (256).
            // Mismatching them used to abort the whole pipeline with a raw ort
            // shape error. Adapting instead means every combination runs — the
            // 512 graph is simply slower, which is a far better failure mode
            // than none at all a week before a conference.
            let fixed = ort_input_res(&session);
            let res = match fixed {
                Some(graph_res) if graph_res != res => {
                    eprintln!(
                        "[dino] backend=ort {} — graph is fixed at {graph_res}px; \
                         using that instead of the requested {res}px",
                        model_path.display()
                    );
                    graph_res
                }
                _ => {
                    eprintln!("[dino] backend=ort {} @{res}px", model_path.display());
                    res
                }
            };
            return Ok(Self {
                model: Model::Ort(session),
                res,
                ort_fixed_res: fixed,
                last_ms: 0.0,
                last_prep_ms: 0.0,
            });
        }
        // Default: pure-Rust BURN.
        let device = create_infer_device();
        let wpath = resolve_burn_weights(model_path);
        eprintln!("[dino] backend=BURN ({}) weights={}",
                  crate::tabs::recon_train::model::backend_name(), wpath.display());
        let net = crate::dino_burn::DinoV3Burn::<InferBackend>::load(&wpath.to_string_lossy(), &device)?;
        Ok(Self {
            model: Model::Burn(Box::new(net), device),
            res,
            // BURN reads H,W off the input tensor, so any res is valid.
            ort_fixed_res: None,
            last_ms: 0.0,
            last_prep_ms: 0.0,
        })
    }

    pub fn res(&self) -> u32 { self.res }

    /// The resolution a call will ACTUALLY run at, given what the model allows.
    fn effective_res(&self, requested: u32) -> u32 {
        match self.ort_fixed_res {
            Some(fixed) if fixed != requested => fixed,
            _ => requested,
        }
    }

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
        // A fixed-shape ort graph cannot honour an arbitrary res; see
        // `ort_fixed_res`. Silently coarser beats a dead pipeline.
        let res = self.effective_res(res);
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
        // Same graph constraint as the single-image path. Applied here too so
        // the BURN branch below allocates for the resolution actually used.
        let res = self.effective_res(res);
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
