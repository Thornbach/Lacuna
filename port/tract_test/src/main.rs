//! Probe: can `tract` (pure-Rust ONNX, zero DLLs) load + run our real models?
//! If yes → hassle-free shipping (single self-contained .exe, no ort/cuDNN/CUDA),
//! CPU speed. Tests op coverage on the actual YOLO26 + DINOv3 graphs.

use tract_onnx::prelude::*;
use std::time::Instant;

fn try_model(name: &str, path: &str, shape: [usize; 4]) { try_model_opt(name, path, shape, true) }

fn try_model_opt(name: &str, path: &str, shape: [usize; 4], optimize: bool) {
    println!("\n=== {name}: {path}  input {shape:?}  optimize={optimize} ===");
    let t = Instant::now();
    let res = (|| -> TractResult<()> {
        let m = tract_onnx::onnx()
            .model_for_path(path)?
            .with_input_fact(0, f32::fact(shape).into())?;
        let model = if optimize {
            m.into_optimized()?.into_runnable()?
        } else {
            m.into_typed()?.into_runnable()?
        };
        println!("  loaded in {:.1}s", t.elapsed().as_secs_f32());
        let input = Tensor::zero::<f32>(&shape)?;
        let t2 = Instant::now();
        let out = model.run(tvec!(input.into()))?;
        println!(
            "  RAN OK in {:.2}s — {} output(s): {:?}",
            t2.elapsed().as_secs_f32(),
            out.len(),
            out.iter().map(|o| o.shape().to_vec()).collect::<Vec<_>>()
        );
        Ok(())
    })();
    if let Err(e) = res {
        println!("  FAILED: {e}");
    }
}

fn main() {
    let base = r"E:\PhD_TobiMu\02_code\FoliarToolbox\models";
    // catch_unwind so a panic in one model still lets the other run.
    let _ = std::panic::catch_unwind(|| {
        try_model("DINOv3", &format!(r"{base}\dino.onnx"), [1, 3, 512, 512]);
    });
    let _ = std::panic::catch_unwind(|| {
        try_model("YOLO26", &format!(r"{base}\yolo.onnx"), [1, 3, 640, 640]);
    });
    // retry YOLO26 without tract's optimizer (avoids the panicking linalg kernel)
    let _ = std::panic::catch_unwind(|| {
        try_model_opt("YOLO26 (no-opt)", &format!(r"{base}\yolo.onnx"), [1, 3, 640, 640], false);
    });
}
