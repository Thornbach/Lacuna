//! Headless validation of the BURN DINOv3 reimplementation against the golden
//! reference (port/dino_ref.safetensors: onnxruntime output + HF per-block hidden
//! states). Runs on the CPU NdArray backend for deterministic f32 parity.
//!
//!   lacuna --dino-burn-validate [ref.safetensors] [weights.safetensors]
//!
//! Reports max/mean |Δ| at each stage so any divergence is localized to a block.

use crate::dino_burn::DinoV3Burn;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use safetensors::SafeTensors;

type B = NdArray<f32>;

fn load_ref<const D: usize>(
    st: &SafeTensors,
    name: &str,
    shape: [usize; D],
    device: &NdArrayDevice,
) -> Tensor<B, D> {
    let v = st.tensor(name).unwrap_or_else(|_| panic!("missing ref tensor `{name}`"));
    let data: Vec<f32> = v
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Tensor::from_data(TensorData::new(data, shape), device)
}

fn compare<const D: usize>(label: &str, got: Tensor<B, D>, want: Tensor<B, D>) -> f32 {
    let g = got.into_data().to_vec::<f32>().unwrap();
    let w = want.into_data().to_vec::<f32>().unwrap();
    assert_eq!(g.len(), w.len(), "{label}: length mismatch {} vs {}", g.len(), w.len());
    let mut maxd = 0f32;
    let mut sumd = 0f64;
    for (a, b) in g.iter().zip(w.iter()) {
        let d = (a - b).abs();
        if d > maxd {
            maxd = d;
        }
        sumd += d as f64;
    }
    let mean = (sumd / g.len() as f64) as f32;
    let flag = if maxd < 1e-2 { "OK " } else { "!! " };
    println!("  {flag}{label:8}  max|Δ|={maxd:.3e}  mean|Δ|={mean:.3e}");
    maxd
}

pub fn run(ref_path: &str, weights_path: &str) {
    let device = NdArrayDevice::Cpu;
    println!("[dino-burn-validate]");
    println!("  weights: {weights_path}");
    println!("  ref    : {ref_path}");

    let model = match DinoV3Burn::<B>::load(weights_path, &device) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("load weights failed: {e}");
            return;
        }
    };

    let bytes = std::fs::read(ref_path).expect("read ref");
    let st = SafeTensors::deserialize(&bytes).expect("parse ref");

    let x = load_ref(&st, "x_chw", [1, 3, 512, 512], &device);
    let hs_embed = load_ref(&st, "hs_embed", [1, 1029, 768], &device);
    let hs_block0 = load_ref(&st, "hs_block0", [1, 1029, 768], &device);
    let hs_block6 = load_ref(&st, "hs_block6", [1, 1029, 768], &device);
    let hs_block11 = load_ref(&st, "hs_block11", [1, 1029, 768], &device);
    let onnx_out = load_ref(&st, "onnx_out", [1, 1024, 1536], &device);

    let t0 = std::time::Instant::now();
    let out = model.forward_debug(x);
    let ms = t0.elapsed().as_millis();

    compare("embed", out.embed, hs_embed);
    compare("block0", out.block0, hs_block0);
    compare("block6", out.block6, hs_block6);
    compare("block11", out.block11, hs_block11);
    let final_max = compare("final", out.feat, onnx_out);
    println!("  forward: {ms} ms");
    if final_max < 1e-2 {
        println!("  => PASS (BURN DINOv3 matches the ONNX oracle)");
    } else {
        println!("  => DIVERGENCE — inspect the first stage that flags !!");
    }

    extractor_selftest(weights_path);
}

/// Exercise the WIRED pipeline path: `DinoExtractor` with the BURN backend forced,
/// on a synthetic RGB image — verifies resize → tensor → forward → into_data glue
/// produces a structurally-correct `DinoFeatures` (grid 32, dim 1536, unit halves).
fn extractor_selftest(weights_path: &str) {
    use crate::tabs::pipeline::dino::DinoExtractor;
    use std::path::Path;
    println!("[extractor-selftest] (BURN path — now the default)");
    std::env::set_var("LACUNA_DINO_WEIGHTS", weights_path);
    let mut dino = match DinoExtractor::load(Path::new("unused-in-burn-mode.onnx"), 512) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("  load failed: {e}");
            return;
        }
    };
    // Synthetic gradient image (non-square, forces the resize path).
    let img = image::RgbImage::from_fn(320, 240, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
    });
    match dino.features(&img) {
        Ok(f) => {
            let tokens = f.feat.len() / f.dim;
            let finite = f.feat.iter().all(|v| v.is_finite());
            let n2: f32 = f.feat[0..f.dim].iter().map(|v| v * v).sum();
            println!(
                "  grid={} dim={} tokens={} finite={} row0‖·‖²={:.3} (expect grid=32 dim=1536 ‖²≈2.0) {}ms",
                f.grid, f.dim, tokens, finite, n2, dino.last_ms as u32
            );
            let ok = f.grid == 32 && f.dim == 1536 && finite && (n2 - 2.0).abs() < 0.1;
            println!("  => {}", if ok { "PASS (wired BURN extractor works)" } else { "FAIL" });
        }
        Err(e) => eprintln!("  features failed: {e}"),
    }
}
