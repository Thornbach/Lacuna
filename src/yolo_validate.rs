//! Headless validation of the BURN YOLO26-seg network against the golden reference
//! (port/yolo_ref.safetensors: per-layer + proto + raw head branches from the fused
//! ultralytics model). CPU NdArray backend for deterministic f32 parity.
//!
//!   lacuna --yolo-burn-validate [ref.safetensors] [weights.safetensors]
//!
//! The topk'd out0 is NOT compared here (ambiguous on random input); we validate the
//! deterministic raw network, which fully determines detection quality on real images.

use crate::yolo_burn::YoloV26Burn;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use safetensors::SafeTensors;

type B = NdArray<f32>;

fn load_ref<const D: usize>(st: &SafeTensors, name: &str, shape: [usize; D], dev: &NdArrayDevice) -> Tensor<B, D> {
    let v = st.tensor(name).unwrap_or_else(|_| panic!("missing ref tensor `{name}`"));
    let data: Vec<f32> = v
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Tensor::from_data(TensorData::new(data, shape), dev)
}

fn compare<const D: usize>(label: &str, got: Tensor<B, D>, want: Tensor<B, D>) {
    let g = got.into_data().to_vec::<f32>().unwrap();
    let w = want.into_data().to_vec::<f32>().unwrap();
    assert_eq!(g.len(), w.len(), "{label}: len {} vs {}", g.len(), w.len());
    let (mut maxd, mut sumd) = (0f32, 0f64);
    for (a, b) in g.iter().zip(w.iter()) {
        let d = (a - b).abs();
        maxd = maxd.max(d);
        sumd += d as f64;
    }
    let mean = (sumd / g.len() as f64) as f32;
    let flag = if maxd < 2e-2 { "OK " } else { "!! " };
    println!("  {flag}{label:9}  max|Δ|={maxd:.3e}  mean|Δ|={mean:.3e}");
}

pub fn run(ref_path: &str, weights_path: &str) {
    let dev = NdArrayDevice::Cpu;
    println!("[yolo-burn-validate]");
    println!("  weights: {weights_path}");
    println!("  ref    : {ref_path}");

    let net = match YoloV26Burn::<B>::load(weights_path, &dev) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("load weights failed: {e}");
            return;
        }
    };

    let bytes = std::fs::read(ref_path).expect("read ref");
    let st = SafeTensors::deserialize(&bytes).expect("parse ref");
    let x = load_ref(&st, "input", [1, 3, 640, 640], &dev);

    let t0 = std::time::Instant::now();
    let out = net.forward(x);
    let ms = t0.elapsed().as_millis();

    compare("layer0", out.layer0, load_ref(&st, "layer_0", [1, 64, 320, 320], &dev));
    compare("layer4", out.layer4, load_ref(&st, "layer_4", [1, 512, 80, 80], &dev));
    compare("layer9", out.layer9, load_ref(&st, "layer_9", [1, 512, 20, 20], &dev));
    compare("layer10", out.layer10, load_ref(&st, "layer_10", [1, 512, 20, 20], &dev));
    compare("layer16", out.layer16, load_ref(&st, "layer_16", [1, 256, 80, 80], &dev));
    compare("layer19", out.layer19, load_ref(&st, "layer_19", [1, 512, 40, 40], &dev));
    compare("layer22", out.layer22, load_ref(&st, "layer_22", [1, 512, 20, 20], &dev));
    compare("proto", out.proto, load_ref(&st, "proto", [1, 32, 160, 160], &dev));
    compare("head_box", out.head_box, load_ref(&st, "head_box", [1, 4, 8400], &dev));
    compare("head_cls", out.head_cls, load_ref(&st, "head_cls", [1, 1, 8400], &dev));
    compare("head_msk", out.head_msk, load_ref(&st, "head_msk", [1, 32, 8400], &dev));
    println!("  forward: {ms} ms");
}

/// End-to-end: segment a REAL leaf image with ort AND BURN, compare the results
/// (instance count, scores, full-image mask IoU). Needs the optional `ort-backend`
/// feature to have something to compare against.
#[cfg(not(feature = "ort-backend"))]
pub fn compare_seg(_image: &str, _onnx: &str, _weights: &str) {
    println!("[yolo-seg-compare] needs the `ort-backend` feature — build with --features ort-backend");
}

#[cfg(feature = "ort-backend")]
pub fn compare_seg(image: &str, onnx: &str, weights: &str) {
    use crate::tabs::leaf_seg::inference::{build_yolo, segment_one, SegConfig, SegItem};
    use std::path::{Path, PathBuf};

    let out_dir = std::env::temp_dir().join("yolo_cmp");
    std::fs::create_dir_all(&out_dir).ok();
    let cfg = || SegConfig {
        model_path: PathBuf::from(onnx),
        image_paths: vec![],
        output_dir: out_dir.clone(),
        imgsz: 640,
        conf: 0.25,
        alpha_lo: 0.50,
        chroma_min: 28,
    };

    println!("[yolo-seg-compare] image={image}");
    std::env::set_var("LACUNA_USE_ORT", "1");
    let mut m_ort = build_yolo(Path::new(onnx)).expect("ort load");
    let ort = segment_one(&mut m_ort, Path::new(image), &cfg()).expect("ort seg");

    std::env::remove_var("LACUNA_USE_ORT");
    std::env::set_var("LACUNA_YOLO_WEIGHTS", weights);
    let mut m_burn = build_yolo(Path::new(onnx)).expect("burn load");
    let burn = segment_one(&mut m_burn, Path::new(image), &cfg()).expect("burn seg");

    let scores = |it: &SegItem| {
        let mut s: Vec<f32> = it.instances.iter().map(|i| i.score).collect();
        s.sort_by(|a, b| b.partial_cmp(a).unwrap());
        s.into_iter().take(5).map(|v| format!("{v:.3}")).collect::<Vec<_>>().join(",")
    };
    println!("  ort : {} instances  scores[{}]", ort.instances.len(), scores(&ort));
    println!("  burn: {} instances  scores[{}]", burn.instances.len(), scores(&burn));

    let [w, h] = ort.size;
    let full = |it: &SegItem| {
        let mut m = vec![false; (w * h) as usize];
        for inst in &it.instances {
            let [bx, by, bw, bh] = inst.bbox;
            for yy in 0..bh {
                for xx in 0..bw {
                    if inst.mask[(yy * bw + xx) as usize] == 1 {
                        m[((by + yy) * w + (bx + xx)) as usize] = true;
                    }
                }
            }
        }
        m
    };
    let (a, b) = (full(&ort), full(&burn));
    let (mut inter, mut uni) = (0u64, 0u64);
    for (x, y) in a.iter().zip(b.iter()) {
        if *x || *y {
            uni += 1;
            if *x && *y {
                inter += 1;
            }
        }
    }
    let iou = if uni > 0 { inter as f64 / uni as f64 } else { 1.0 };
    println!("  full-image mask IoU (ort vs burn) = {iou:.4}");
    println!("  => {}", if iou > 0.98 { "PASS (BURN YOLO matches ort end-to-end)" } else { "CHECK" });
}
