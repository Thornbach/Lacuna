//! Probe: does the ort CUDA EP actually load + run DINO on the GPU now that the
//! cuDNN DLLs are findable? Compares load/run with CUDA-requested vs CPU-only.

use ort::execution_providers::{CUDAExecutionProvider, ExecutionProvider};
use ort::session::Session;
use ort::value::Tensor;
use std::time::Instant;

const DINO: &str = r"E:\PhD_TobiMu\02_code\FoliarToolbox\models\dino.onnx";

fn run(label: &str, cuda: bool) {
    println!("\n=== {label} ===");
    let t = Instant::now();
    let mut b = Session::builder().unwrap();
    if cuda {
        b = b
            .with_execution_providers([CUDAExecutionProvider::default().build()])
            .unwrap();
    }
    let mut session = match b.commit_from_file(DINO) {
        Ok(s) => s,
        Err(e) => {
            println!("  session build FAILED: {e}");
            return;
        }
    };
    println!("  session built in {:.1}s", t.elapsed().as_secs_f32());

    let n = 3 * 512 * 512;
    for i in 0..2 {
        let input = Tensor::from_array(([1usize, 3, 512, 512], vec![0f32; n])).unwrap();
        let t2 = Instant::now();
        match session.run(ort::inputs!["images" => input]) {
            Ok(out) => {
                let (shape, _) = out["features"].try_extract_tensor::<f32>().unwrap();
                println!(
                    "  forward #{i}: {:.3}s  out {:?}",
                    t2.elapsed().as_secs_f32(),
                    shape
                );
            }
            Err(e) => println!("  forward FAILED: {e}"),
        }
    }
}

fn main() {
    println!("CUDA EP is_available: {:?}", CUDAExecutionProvider::default().is_available());
    run("CUDA requested (GPU if cuDNN found)", true);
    run("CPU only (baseline)", false);
    println!("\nInterpretation: if the CUDA run's forwards are ~10-30x faster than CPU,");
    println!("the GPU is engaged. If they're the same speed, ort still fell back to CPU.");
}
