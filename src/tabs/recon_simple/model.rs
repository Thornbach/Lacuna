/// Thin re-export: UNetSimple is the same architecture as UNetGenerator so that
/// recon_simple training and recon_infer inference share checkpoint-compatible weights.
pub use crate::tabs::recon_train::model::{
    UNetGenerator as UNetSimple,
    create_train_device, create_infer_device, backend_name,
};
pub use crate::tabs::recon_train::model::{TrainBackend, InferBackend};

use burn_core::record::CompactRecorder;
use burn_core::module::Module as CoreModule;
use burn::module::AutodiffModule;
use std::path::Path;

pub fn save_simple_checkpoint(
    model: &UNetSimple<TrainBackend>,
    dir:   &Path,
) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let _: f32 = model.gpu_fence().into_scalar();
    let rec = CompactRecorder::new();
    model.clone().valid()
        .save_file(dir.join("gen"), &rec)
        .map_err(|e| format!("{e}"))
}

pub fn load_simple_checkpoint(
    model:  UNetSimple<TrainBackend>,
    dir:    &Path,
    device: &<TrainBackend as burn::tensor::backend::Backend>::Device,
) -> Result<UNetSimple<TrainBackend>, String> {
    let rec = CompactRecorder::new();
    model.load_file(dir.join("gen"), &rec, device)
        .map_err(|e| format!("{e}"))
}

pub fn load_simple_infer(
    dir:    &Path,
    device: &<InferBackend as burn::tensor::backend::Backend>::Device,
) -> Result<UNetSimple<InferBackend>, String> {
    let rec = CompactRecorder::new();
    UNetSimple::<InferBackend>::init(device)
        .load_file(dir.join("gen"), &rec, device)
        .map_err(|e| format!("{e}"))
}
