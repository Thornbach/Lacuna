use std::{
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
    sync::atomic::{AtomicBool, Ordering},
};

use burn::tensor::Tensor;

use crate::tabs::recon_train::model::{InferBackend, create_infer_device, backend_name};
use super::model::load_radial_infer;
use super::training::infer_one;

type InferDevice = <InferBackend as burn::tensor::backend::Backend>::Device;

// ── Public types ──────────────────────────────────────────────────────────────

pub struct RadialInferConfig {
    pub checkpoint_dir: PathBuf,
    pub image_paths:    Vec<PathBuf>,
    pub output_dir:     PathBuf,
    pub image_size:     usize,
}

pub struct RadialInferResult {
    pub filename:           String,
    pub path:               PathBuf,
    pub out_path:           PathBuf,
    pub input_leaf_px:      usize,
    pub predicted_total_px: usize,
    pub damage_px:          usize,
    pub pct_damage:         f32,
}

pub enum RadialInferMsg {
    Progress { done: usize, total: usize },
    ImageResult(RadialInferResult),
    Log(String),
    Finished,
    Error(String),
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn spawn_radial_inference(
    config: RadialInferConfig,
    tx:     mpsc::Sender<RadialInferMsg>,
    cancel: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        match run_inference(&config, &tx, &cancel) {
            Ok(())  => {}
            Err(e)  => { let _ = tx.send(RadialInferMsg::Error(e)); }
        }
        let _ = tx.send(RadialInferMsg::Finished);
    });
}

// ── Main loop ─────────────────────────────────────────────────────────────────

fn run_inference(
    cfg:    &RadialInferConfig,
    tx:     &mpsc::Sender<RadialInferMsg>,
    cancel: &AtomicBool,
) -> Result<(), String> {
    log(tx, format!("Backend: {}", backend_name()));
    let device: InferDevice = create_infer_device();
    { let _ = Tensor::<InferBackend, 1>::zeros([1], &device).into_data(); }
    log(tx, format!("{} device ready.", backend_name()));

    log(tx, format!("Loading checkpoint: {}", cfg.checkpoint_dir.display()));
    let model = load_radial_infer(&cfg.checkpoint_dir, &device)
        .map_err(|e| format!("Failed to load checkpoint: {e}"))?;
    log(tx, "Model loaded. Running inference…".to_string());

    std::fs::create_dir_all(&cfg.output_dir).map_err(|e| e.to_string())?;

    let total = cfg.image_paths.len();
    for (idx, path) in cfg.image_paths.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            log(tx, "Inference cancelled.".to_string());
            break;
        }

        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
        let out_path = cfg.output_dir.join(format!("{stem}_radial_recon.png"));

        match process_one(path, &model, &device, cfg, &out_path) {
            Ok(result) => {
                let _ = tx.send(RadialInferMsg::Progress { done: idx + 1, total });
                let _ = tx.send(RadialInferMsg::ImageResult(result));
            }
            Err(e) => {
                log(tx, format!("Skipped {stem}: {e}"));
                let _ = tx.send(RadialInferMsg::Progress { done: idx + 1, total });
            }
        }
    }

    Ok(())
}

// ── Per-image processing ──────────────────────────────────────────────────────

fn process_one(
    path:     &Path,
    model:    &super::model::RadialNet<InferBackend>,
    device:   &InferDevice,
    cfg:      &RadialInferConfig,
    out_path: &Path,
) -> Result<RadialInferResult, String> {
    let sz   = cfg.image_size;
    let sz32 = sz as u32;

    let img  = image::open(path).map_err(|e| e.to_string())?;
    let img  = img.resize_exact(sz32, sz32, image::imageops::FilterType::Triangle);
    let mut rgba = img.to_rgba8().into_raw();

    // Binarise alpha — matches training load_images() behaviour
    for p in rgba.chunks_mut(4) {
        if p[3] <= 128 { p[0] = 0; p[1] = 0; p[2] = 0; p[3] = 0; }
        else           { p[3] = 255; }
    }

    let input_leaf_px = rgba.chunks(4).filter(|p| p[3] > 128).count();

    // Run radial inference → reconstructed binary mask [sz*sz]
    let pred_mask = infer_one(model, &rgba, sz, sz, device);
    let predicted_total_px = pred_mask.iter().filter(|&&b| b).count();

    // Mean colour of existing (non-damaged) leaf pixels for fill
    let (fill_r, fill_g, fill_b) = {
        let mut sr = 0u64; let mut sg = 0u64; let mut sb = 0u64; let mut cnt = 0u64;
        for (i, p) in rgba.chunks(4).enumerate() {
            if p[3] > 128 && pred_mask.get(i).copied().unwrap_or(false) {
                sr += p[0] as u64; sg += p[1] as u64; sb += p[2] as u64; cnt += 1;
            }
        }
        if cnt > 0 { ((sr/cnt) as u8, (sg/cnt) as u8, (sb/cnt) as u8) }
        else        { (128u8, 128u8, 128u8) }
    };

    // Composite: keep original pixels, fill reconstructed gap with mean colour
    let mut recon = vec![0u8; sz * sz * 4];
    for i in 0..sz * sz {
        if !pred_mask[i] { continue; }
        let b = i * 4;
        if rgba[b + 3] > 128 {
            recon[b]     = rgba[b];
            recon[b + 1] = rgba[b + 1];
            recon[b + 2] = rgba[b + 2];
            recon[b + 3] = 255;
        } else {
            recon[b]     = fill_r;
            recon[b + 1] = fill_g;
            recon[b + 2] = fill_b;
            recon[b + 3] = 255;
        }
    }

    save_rgba_png(&recon, sz32, sz32, out_path)?;

    let damage_px = predicted_total_px.saturating_sub(input_leaf_px);
    let pct_damage = if predicted_total_px > 0 {
        damage_px as f32 / predicted_total_px as f32 * 100.0
    } else { 0.0 };

    Ok(RadialInferResult {
        filename: path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string(),
        path:     path.to_path_buf(),
        out_path: out_path.to_path_buf(),
        input_leaf_px,
        predicted_total_px,
        damage_px,
        pct_damage,
    })
}

// ── CSV export ────────────────────────────────────────────────────────────────

pub fn write_csv(results: &[RadialInferResult], path: &Path) -> Result<(), String> {
    use std::io::Write;
    let mut f = std::fs::File::create(path).map_err(|e| e.to_string())?;
    writeln!(f, "Filename,Path,Input_Leaf_px,Predicted_Total_px,Damage_px,Pct_Damage")
        .map_err(|e| e.to_string())?;
    for r in results {
        writeln!(f, "{},{},{},{},{},{:.2}",
            r.filename, r.path.display(),
            r.input_leaf_px, r.predicted_total_px,
            r.damage_px, r.pct_damage,
        ).map_err(|e| e.to_string())?;
    }
    Ok(())
}

// ── Utilities ─────────────────────────────────────────────────────────────────

fn save_rgba_png(pixels: &[u8], w: u32, h: u32, path: &Path) -> Result<(), String> {
    use image::{ImageBuffer, Rgba};
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_raw(w, h, pixels.to_vec())
            .ok_or("ImageBuffer::from_raw failed")?;
    img.save(path).map_err(|e| e.to_string())
}

fn log(tx: &mpsc::Sender<RadialInferMsg>, msg: String) {
    let _ = tx.send(RadialInferMsg::Log(msg));
}
