//! In-app detector training worker: healthy tiles -> DINO features -> random
//! coreset (reservoir) -> dino-channel calibration -> coreset_bank.bin +
//! detector_meta.json. Pure Rust; runs on a background thread (mpsc + cancel).
//!
//! It reuses the pipeline's DINO extractor, GPU bank kNN, robust-z and tiling so
//! the bank + thresholds it writes match exactly what the Pipeline tab consumes.

use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
    sync::atomic::{AtomicBool, Ordering},
};

use image::RgbImage;

use crate::tabs::pipeline::bank::{CoresetBank, GpuBank};
use crate::tabs::pipeline::channels;
use crate::tabs::pipeline::detect::{robust_scale, robust_z, upscale};
use crate::tabs::pipeline::dino::DinoExtractor;
use crate::tabs::pipeline::tiling::{erode_alpha, tile_leaf};
use crate::tabs::recon_train::model::{create_infer_device, InferBackend};

use super::coreset::ReservoirSampler;

pub struct FitConfig {
    pub healthy_paths:  Vec<PathBuf>,
    pub dino_model:     PathBuf,
    pub out_dir:        PathBuf,
    pub dino_res:       u32,
    pub coreset_target: usize,
    pub n_fit:          usize,
    pub n_calib:        usize,
    pub max_fp_rate:    f32,
    pub scale_floor_pct: f32,
    pub margin_erode:   u32,
    pub knn_k:          usize,
    pub seed:           u64,
}

pub enum FitMsg {
    Stage(String),
    Progress { done: usize, total: usize },
    Log(String),
    Error(String),
    Finished,
}

pub fn spawn_fit(cfg: FitConfig, tx: mpsc::Sender<FitMsg>, cancel: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        if let Err(e) = run_fit(&cfg, &tx, &cancel) {
            let _ = tx.send(FitMsg::Error(e));
        }
        let _ = tx.send(FitMsg::Finished);
    });
}

fn log(tx: &mpsc::Sender<FitMsg>, m: impl Into<String>) { let _ = tx.send(FitMsg::Log(m.into())); }
fn stage(tx: &mpsc::Sender<FitMsg>, m: impl Into<String>) { let _ = tx.send(FitMsg::Stage(m.into())); }

fn run_fit(cfg: &FitConfig, tx: &mpsc::Sender<FitMsg>, cancel: &AtomicBool) -> Result<(), String> {
    std::fs::create_dir_all(&cfg.out_dir).map_err(|e| e.to_string())?;
    stage(tx, "Loading DINOv3");
    let mut dino = DinoExtractor::load(&cfg.dino_model, cfg.dino_res)?;

    // ── 1. collect healthy patch features into a random coreset (reservoir) ──
    stage(tx, "Collecting healthy features");
    let n_fit = cfg.n_fit.min(cfg.healthy_paths.len());
    let mut sampler: Option<ReservoirSampler> = None;
    let mut seen = 0u64;
    for (i, path) in cfg.healthy_paths.iter().take(n_fit).enumerate() {
        if cancel.load(Ordering::Relaxed) {
            log(tx, "cancelled");
            break;
        }
        if i % 16 == 0 {
            let _ = tx.send(FitMsg::Progress { done: i, total: n_fit });
        }
        let Some(tile) = load_tile(path, cfg.margin_erode) else { continue };
        let f = dino.features(&tile.rgb)?;
        let tvalid = token_valid(&tile.valid, tile.size, f.grid);
        let s = sampler.get_or_insert_with(|| ReservoirSampler::new(cfg.coreset_target, f.dim, cfg.seed));
        // gather valid-token rows into one batch
        let mut batch = Vec::new();
        let mut m = 0usize;
        for t in 0..f.grid * f.grid {
            if tvalid[t] {
                batch.extend_from_slice(&f.feat[t * f.dim..t * f.dim + f.dim]);
                m += 1;
            }
        }
        seen += m as u64;
        s.add_rows(&batch, m);
    }
    let _ = tx.send(FitMsg::Progress { done: n_fit, total: n_fit });
    let sampler = sampler.ok_or("no valid features collected")?;
    let (flat, bank_n) = sampler.into_pool();
    let dim = flat.len() / bank_n;
    log(tx, format!("saw {seen} valid patches; coreset {bank_n} x {dim}"));

    // ── 2. write coreset_bank.bin ──
    stage(tx, "Writing coreset_bank.bin");
    write_bank(&cfg.out_dir.join("coreset_bank.bin"), &flat, bank_n, dim)?;

    // ── 3. calibrate all three channels (dino + residual + color) ──
    stage(tx, "Calibrating");
    let cpu_bank = CoresetBank::from_parts(flat, bank_n, dim);
    let device = create_infer_device();
    let bank = GpuBank::<InferBackend>::new(&cpu_bank, device);

    const RESIDUAL_KSIZE: usize = 9; // must match DetectParams::default().residual_ksize
    let n_calib = cfg.n_calib.min(n_fit);
    let (mut sc_d, mut sc_r, mut sc_c) = (Vec::new(), Vec::new(), Vec::new());
    // cache (dino_map, residual, color, valid) per tile for the second pass
    let mut cache: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<bool>)> = Vec::new();
    for (i, path) in cfg.healthy_paths.iter().take(n_calib).enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        if i % 8 == 0 {
            let _ = tx.send(FitMsg::Progress { done: i, total: n_calib });
        }
        let Some(tile) = load_tile(path, cfg.margin_erode) else { continue };
        let f = dino.features(&tile.rgb)?;
        let scores = bank.knn_mean_dist(&f.feat, f.grid * f.grid, cfg.knn_k);
        let t = tile.size as usize;
        let dino_map = upscale(&scores, f.grid, f.grid, t, t);
        let residual = channels::residual_map(&tile.rgb, &tile.valid, RESIDUAL_KSIZE);
        let color = channels::color_deviation_map(&tile.rgb, &tile.valid);
        let s = robust_scale(&dino_map, &tile.valid);
        if s > 0.0 { sc_d.push(s); }
        let s = robust_scale(&residual, &tile.valid);
        if s > 0.0 { sc_r.push(s); }
        let s = robust_scale(&color, &tile.valid);
        if s > 0.0 { sc_c.push(s); }
        cache.push((dino_map, residual, color, tile.valid));
    }
    if cache.is_empty() {
        return Err("no calibration tiles".into());
    }
    let floor_d = percentile(&mut sc_d, cfg.scale_floor_pct);
    let floor_r = percentile(&mut sc_r, cfg.scale_floor_pct);
    let floor_c = percentile(&mut sc_c, cfg.scale_floor_pct);

    // pass 2: pooled clean per-channel z (with floors) + raw dino distances
    let (mut z_d, mut z_r, mut z_c, mut raw_d) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (dmap, res, col, valid) in &cache {
        let zd = robust_z(dmap, valid, floor_d);
        let zr = robust_z(res, valid, floor_r);
        let zc = robust_z(col, valid, floor_c);
        for i in 0..dmap.len() {
            if valid[i] {
                z_d.push(zd[i]);
                z_r.push(zr[i]);
                z_c.push(zc[i]);
                raw_d.push(dmap[i]);
            }
        }
    }
    let q = (1.0 - cfg.max_fp_rate) * 100.0;
    let thr_d = percentile(&mut z_d, q);
    let thr_r = percentile(&mut z_r, q);
    let thr_c = percentile(&mut z_c, q);
    let dino_floor = percentile(&mut raw_d, q);

    // ── 4. write detector_meta.json (3-channel) ──
    stage(tx, "Writing detector_meta.json");
    write_meta(
        &cfg.out_dir.join("detector_meta.json"),
        [thr_d, thr_r, thr_c],
        dino_floor,
        [floor_d, floor_r, floor_c],
        dim,
    )?;
    log(tx, format!(
        "calibrated z-thr d/r/c = {thr_d:.2}/{thr_r:.2}/{thr_c:.2}  floors = {floor_d:.3}/{floor_r:.3}/{floor_c:.3}  dino_floor={dino_floor:.3}"
    ));
    stage(tx, "Done");
    Ok(())
}

// ── helpers ─────────────────────────────────────────────────────────────────

struct Tile {
    rgb:   RgbImage,
    valid: Vec<bool>,
    size:  u32,
}

/// Load a healthy tile RGBA, erode its margin, and produce a single square
/// mean-filled RGB tile + valid mask (same preprocessing the pipeline uses).
fn load_tile(path: &Path, margin_erode: u32) -> Option<Tile> {
    let img = image::open(path).ok()?.to_rgba8();
    let (w, h) = img.dimensions();
    let mut rgba = img.into_raw();
    erode_alpha(&mut rgba, w, h, margin_erode, 10);
    let tsz = w.max(h);
    let mut tiles = tile_leaf(&rgba, w, h, tsz, 10);
    let t = tiles.drain(..).next()?;
    Some(Tile { rgb: t.rgb, valid: t.valid, size: t.size })
}

/// A token (g×g grid) is valid iff its whole (t/g)² cell is valid.
fn token_valid(valid: &[bool], t: u32, g: usize) -> Vec<bool> {
    let t = t as usize;
    let block = (t / g).max(1);
    let mut out = vec![false; g * g];
    for gy in 0..g {
        for gx in 0..g {
            let mut ok = true;
            'cell: for by in 0..block {
                for bx in 0..block {
                    let y = gy * block + by;
                    let x = gx * block + bx;
                    if y >= t || x >= t || !valid[y * t + x] {
                        ok = false;
                        break 'cell;
                    }
                }
            }
            out[gy * g + gx] = ok;
        }
    }
    out
}

fn percentile(v: &mut [f32], p: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (((p / 100.0) * (v.len() - 1) as f32).round()).clamp(0.0, (v.len() - 1) as f32) as usize;
    v[idx]
}

fn write_bank(path: &Path, flat: &[f32], n: usize, d: usize) -> Result<(), String> {
    let mut f = std::fs::File::create(path).map_err(|e| e.to_string())?;
    f.write_all(&(n as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&(d as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    let mut bytes = Vec::with_capacity(flat.len() * 4);
    for v in flat {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&bytes).map_err(|e| e.to_string())?;
    Ok(())
}

fn write_meta(
    path: &Path,
    ch_thr: [f32; 3],   // dino, residual, color
    dino_thr: f32,
    floors: [f32; 3],   // dino, residual, color
    dim: usize,
) -> Result<(), String> {
    let [td, tr, tc] = ch_thr;
    let [fd, fr, fc] = floors;
    let meta = serde_json::json!({
        "threshold": td.max(tr).max(tc),
        "ch_thresholds": { "dino": td, "residual": tr, "color": tc },
        "dino_threshold": dino_thr,
        "scale_floors": { "dino": fd, "residual": fr, "color": fc },
        "dim": dim,
        "project_dim": serde_json::Value::Null,
    });
    std::fs::write(path, serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tabs::leaf_seg::inference::list_images;

    /// End-to-end fit on a few real healthy tiles (tiny bank to keep the
    /// NdArray-debug kNN fast). Verifies the written assets load.
    /// cargo test --no-default-features fit_smoke -- --ignored --nocapture
    #[test]
    #[ignore]
    fn fit_smoke() {
        let healthy = PathBuf::from(r"E:\PhD_TobiMu\01_data\raw\SamplingSaturday\output256");
        let dino = PathBuf::from(r"E:\PhD_TobiMu\02_code\02paper\anomaly\dinov3_vitb16_512.onnx");
        let out = std::env::temp_dir().join("foliar_fit_smoke");
        let paths = list_images(&healthy);
        assert!(!paths.is_empty(), "no healthy tiles at {}", healthy.display());
        let first = paths[0].clone();

        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        spawn_fit(
            FitConfig {
                healthy_paths: paths,
                dino_model: dino,
                out_dir: out.clone(),
                dino_res: 512,
                coreset_target: 2000,
                n_fit: 20,
                n_calib: 8,
                max_fp_rate: 0.0005,
                scale_floor_pct: 25.0,
                margin_erode: 6,
                knn_k: 3,
                seed: 0,
            },
            tx,
            cancel,
        );
        while let Ok(m) = rx.recv() {
            match m {
                FitMsg::Log(l) => println!("log: {l}"),
                FitMsg::Stage(s) => println!("stage: {s}"),
                FitMsg::Error(e) => panic!("fit error: {e}"),
                FitMsg::Finished => break,
                _ => {}
            }
        }
        let bank_p = out.join("coreset_bank.bin");
        let meta_p = out.join("detector_meta.json");
        assert!(bank_p.exists(), "coreset_bank.bin not written");
        assert!(meta_p.exists(), "detector_meta.json not written");
        let b = CoresetBank::load(&bank_p).expect("bank load");
        let m = crate::tabs::pipeline::meta::DetectorMeta::load(&meta_p).expect("meta load");
        println!("FIT OK: bank {}x{}  z_thr(dino)={:.3}", b.n, b.d, m.ch_threshold("dino"));
        assert_eq!(b.d, 1536);
        assert!(b.n > 0 && b.n <= 2000);
        assert!(m.has("residual") && m.has("color"), "meta missing channels");
        println!("  z-thr r/c = {:.2}/{:.2}", m.ch_threshold("residual"), m.ch_threshold("color"));

        // ── end-to-end S5: detect with the fresh 3-channel detector (self-consistent) ──
        use crate::tabs::pipeline::bank::GpuBank;
        use crate::tabs::pipeline::detect::{detect, DetectParams};
        use crate::tabs::pipeline::dino::DinoExtractor;
        use burn::backend::NdArray;
        use burn::tensor::backend::Backend;
        let onnx = PathBuf::from(r"E:\PhD_TobiMu\02_code\02paper\anomaly\dinov3_vitb16_512.onnx");
        let mut dino = DinoExtractor::load(&onnx, 512).expect("dino");
        let dev = <NdArray as Backend>::Device::default();
        let gpu = GpuBank::<NdArray>::new(&b, dev);
        let tile = super::load_tile(&first, 6).expect("tile");
        let r = detect(
            &mut dino, &gpu, &m, &DetectParams::default(),
            &tile.rgb, tile.size as usize, tile.size as usize, Some(&tile.valid),
        ).expect("detect");
        let flagged = r.mask.iter().filter(|&&x| x).count();
        println!(
            "  S5 detect on healthy tile: {:.1}% flagged, {} regions",
            100.0 * flagged as f32 / r.mask.len() as f32, r.regions.len()
        );
        assert!(r.dino_map.iter().all(|v| v.is_finite()));
    }
}
