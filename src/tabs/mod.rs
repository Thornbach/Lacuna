pub mod eroder;
pub mod sorter;
pub mod recon_simple;
pub mod recon_train;
pub mod recon_infer;
pub mod leaf_seg;
pub mod pipeline;
pub mod train;
pub mod tile_picker;
pub mod morphology;

#[cfg(feature = "ort-backend")]
use std::path::Path;

/// Build an ort `Session` (only with the optional `ort-backend` feature). Registers
/// GPU execution providers by priority (DirectML → CUDA), falling back to CPU if none
/// load. DINO/YOLO now default to BURN; this is used only when `LACUNA_USE_ORT=1`.
#[cfg(feature = "ort-backend")]
pub fn build_session(path: &Path) -> Result<ort::session::Session, String> {
    #[allow(unused_mut)]
    let mut eps: Vec<ort::execution_providers::ExecutionProviderDispatch> = Vec::new();
    #[cfg(feature = "directml")]
    {
        use ort::execution_providers::DirectMLExecutionProvider;
        eps.push(DirectMLExecutionProvider::default().build());
    }
    #[cfg(feature = "ort-cuda")]
    {
        use ort::execution_providers::CUDAExecutionProvider;
        eps.push(CUDAExecutionProvider::default().build());
    }
    if !eps.is_empty() {
        match commit_session(path, eps) {
            Ok(s) => return Ok(s),
            Err(e) => eprintln!(
                "[ort] GPU execution provider failed for {} — falling back to CPU.\n      {e}",
                path.display()
            ),
        }
    }
    commit_session(path, Vec::new())
}

/// One-line backend diagnostic logged at pipeline start. Reports the active BURN
/// backend (and, if the ort fallback is compiled, whether its GPU EPs can load).
pub fn gpu_diagnostics() -> String {
    let burn = crate::tabs::recon_train::model::backend_name();
    #[cfg(not(feature = "ort-backend"))]
    {
        format!("Backend: BURN {burn} (pure-Rust, no onnxruntime/cuDNN)")
    }
    #[cfg(feature = "ort-backend")]
    {
        #[allow(unused_mut)]
        let mut eps: Vec<String> = Vec::new();
        #[cfg(feature = "directml")]
        {
            use ort::execution_providers::{DirectMLExecutionProvider, ExecutionProvider};
            eps.push(format!("DirectML={}", DirectMLExecutionProvider::default().is_available().unwrap_or(false)));
        }
        #[cfg(feature = "ort-cuda")]
        {
            use ort::execution_providers::{CUDAExecutionProvider, ExecutionProvider};
            eps.push(format!("CUDA={}", CUDAExecutionProvider::default().is_available().unwrap_or(false)));
        }
        let ep = if eps.is_empty() { "CPU only".into() } else { eps.join(", ") };
        format!("Backend: BURN {burn}; ort fallback available (EPs: {ep})")
    }
}

#[cfg(feature = "ort-backend")]
fn commit_session(
    path: &Path,
    eps:  Vec<ort::execution_providers::ExecutionProviderDispatch>,
) -> Result<ort::session::Session, String> {
    use ort::session::{builder::GraphOptimizationLevel, Session};
    let mut builder = Session::builder().map_err(|e| format!("ort builder: {e}"))?;
    if !eps.is_empty() {
        builder = builder
            .with_execution_providers(eps)
            .map_err(|e| format!("ort EP: {e}"))?;
    }
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("ort opt level: {e}"))?
        .with_intra_threads(threads)
        .map_err(|e| format!("ort threads: {e}"))?;
    builder
        .commit_from_file(path)
        .map_err(|e| format!("load onnx {}: {e}", path.display()))
}

pub use eroder::EroderTab;
pub use sorter::SorterTab;
pub use recon_simple::ReconSimpleTab;
pub use recon_train::ReconTrainTab;
pub use recon_infer::ReconInferTab;
pub use leaf_seg::LeafSegTab;
pub use pipeline::PipelineTab;
pub use train::TrainTab;
pub use tile_picker::TilePickerTab;
pub use morphology::MorphologyTab;
