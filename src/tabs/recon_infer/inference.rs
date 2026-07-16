use std::{
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
    sync::atomic::{AtomicBool, Ordering},
};

use burn::tensor::{Tensor, TensorData, backend::Backend};

use rand::{rngs::SmallRng, SeedableRng};
use crate::tabs::recon_simple::model::{UNetSimple, load_simple_infer};
use crate::tabs::recon_train::model::{InferBackend, create_infer_device, backend_name};
use crate::tabs::eroder::algorithm::erode_margin_clusters;

type InferDevice = <InferBackend as Backend>::Device;

// ── Public types ──────────────────────────────────────────────────────────────

/// Per-image inference statistics (no image data — images are saved to disk)
pub struct InferResult {
    pub filename:           String,
    pub path:               PathBuf,
    pub out_path:           PathBuf,
    pub input_leaf_px:      usize,
    pub predicted_total_px: usize,
    pub damage_px:          usize,
    pub pct_damage:         f32,
}

pub enum InferMsg {
    Progress { done: usize, total: usize },
    ImageResult(InferResult),
    Log(String),
    Finished,
    Error(String),
}

pub struct InferConfig {
    pub checkpoint_dir:  PathBuf,
    pub image_paths:     Vec<PathBuf>,
    pub output_dir:      PathBuf,
    pub image_size:      usize,
    pub threshold:       f32,
    /// Fraction of leaf area to erase synthetically before inference (0 = disabled).
    pub pre_damage_pct:  f32,
    /// Run TTA: 4 rotations (0°/90°/180°/270°), average probability maps.
    pub tta_enabled:     bool,
    /// Save debug PNGs alongside each output.
    pub debug_output:    bool,
    /// Leaf shape class index (LeafShape::index()).
    pub leaf_shape:      u32,
    /// Margin type class index (MarginType::index()).
    pub margin_type:     u32,
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn spawn_inference(
    config: InferConfig,
    tx:     mpsc::Sender<InferMsg>,
    cancel: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        match run_inference(&config, &tx, &cancel) {
            Err(e) => { let _ = tx.send(InferMsg::Error(e)); }
            Ok(()) => {}
        }
        let _ = tx.send(InferMsg::Finished);
    });
}

// ── Main inference function ───────────────────────────────────────────────────

fn run_inference(
    cfg:    &InferConfig,
    tx:     &mpsc::Sender<InferMsg>,
    cancel: &AtomicBool,
) -> Result<(), String> {
    log(tx, format!("Requesting {} device…", backend_name()));
    let device: InferDevice = create_infer_device();
    {
        let probe = Tensor::<InferBackend, 1>::zeros([1], &device);
        let _ = probe.into_data();
    }
    log(tx, format!("{} device ready.", backend_name()));

    // Load UNetSimple checkpoint (gen.mpk saved by Recon Train tab)
    log(tx, format!("Loading checkpoint: {}", cfg.checkpoint_dir.display()));
    let generator: UNetSimple<InferBackend> = load_simple_infer(&cfg.checkpoint_dir, &device)
        .map_err(|e| format!("Failed to load checkpoint: {e}"))?;
    log(tx, "Model loaded. Running inference…");

    std::fs::create_dir_all(&cfg.output_dir).map_err(|e| e.to_string())?;

    let total = cfg.image_paths.len();
    for (idx, path) in cfg.image_paths.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            log(tx, "Inference cancelled.");
            break;
        }

        let stem = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image");
        let out_path = cfg.output_dir.join(format!("{stem}_recon.png"));

        match infer_one(path, &generator, &device, cfg, cfg.leaf_shape, cfg.margin_type) {
            Ok((stats, recon_rgba)) => {
                let sz = cfg.image_size as u32;
                if let Err(e) = save_rgba_png(&recon_rgba, sz, sz, &out_path) {
                    log(tx, format!("Warning: could not save {stem}: {e}"));
                }

                let damage_px = stats.predicted_total_px
                    .saturating_sub(stats.input_leaf_px);
                let pct_damage = if stats.predicted_total_px > 0 {
                    damage_px as f32 / stats.predicted_total_px as f32 * 100.0
                } else {
                    0.0
                };

                let _ = tx.send(InferMsg::Progress { done: idx + 1, total });
                let _ = tx.send(InferMsg::ImageResult(InferResult {
                    filename: path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("")
                        .to_string(),
                    path:               path.clone(),
                    out_path,
                    input_leaf_px:      stats.input_leaf_px,
                    predicted_total_px: stats.predicted_total_px,
                    damage_px,
                    pct_damage,
                }));
            }
            Err(e) => {
                log(tx, format!("Skipped {stem}: {e}"));
                let _ = tx.send(InferMsg::Progress { done: idx + 1, total });
            }
        }
    }

    Ok(())
}

// ── BFS dilation ─────────────────────────────────────────────────────────────

/// Returns a mask that is `true` for every pixel within `radius` pixels
/// (Chebyshev / 8-connected BFS) of any opaque pixel in `rgba`.
///
/// Used to define the "allowed reconstruction zone": the model can only add
/// new leaf pixels that are reachable from the existing leaf boundary.
/// Scanner calibration marks far from the leaf are therefore excluded.
fn bfs_dilate(rgba: &[u8], sz: usize, radius: usize) -> Vec<bool> {
    let n = sz * sz;
    let mut dist  = vec![u32::MAX; n];
    let mut queue = std::collections::VecDeque::new();

    // Seed: all currently-opaque leaf pixels at distance 0
    for i in 0..n {
        if rgba[i * 4 + 3] > 128 {
            dist[i] = 0;
            queue.push_back(i);
        }
    }

    // BFS expanding up to `radius` steps
    let r = radius as u32;
    while let Some(idx) = queue.pop_front() {
        let d = dist[idx];
        if d >= r { continue; }
        let x = (idx % sz) as i32;
        let y = (idx / sz) as i32;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 { continue; }
                let nx = x + dx; let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= sz as i32 || ny >= sz as i32 { continue; }
                let ni = ny as usize * sz + nx as usize;
                if dist[ni] == u32::MAX {
                    dist[ni] = d + 1;
                    queue.push_back(ni);
                }
            }
        }
    }

    dist.iter().map(|&d| d <= r).collect()
}

// ── Largest-blob filter ───────────────────────────────────────────────────────

/// Remove connected components of opaque pixels that are smaller than
/// `min_fraction` of the total opaque area.
///
/// This removes scanner bar artefacts, corner markers, and stray debris
/// (which are tiny compared to the leaf) while keeping all significant leaf
/// fragments — including lobes that became disconnected due to heavy erosion.
///
/// Typical value: min_fraction = 0.02  (drop anything < 2 % of leaf area).
fn remove_small_blobs(rgba: &mut [u8], sz: usize, min_fraction: f32) {
    let n = sz * sz;
    let mut label  = vec![0u32; n];
    // (size, pixel-list) per component; component id = index + 1
    let mut comp_sizes:  Vec<u32>       = Vec::new();
    let mut comp_pixels: Vec<Vec<usize>> = Vec::new();
    let mut next_id: u32 = 1;

    for start in 0..n {
        if label[start] != 0 || rgba[start * 4 + 3] == 0 { continue; }

        let id = next_id;
        next_id += 1;

        let mut pixels = Vec::new();
        let mut queue  = std::collections::VecDeque::new();
        queue.push_back(start);
        label[start] = id;

        while let Some(idx) = queue.pop_front() {
            pixels.push(idx);
            let x = (idx % sz) as i32;
            let y = (idx / sz) as i32;
            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nx = x + dx; let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= sz as i32 || ny >= sz as i32 { continue; }
                let ni = ny as usize * sz + nx as usize;
                if label[ni] == 0 && rgba[ni * 4 + 3] > 0 {
                    label[ni] = id;
                    queue.push_back(ni);
                }
            }
        }
        comp_sizes.push(pixels.len() as u32);
        comp_pixels.push(pixels);
    }

    if comp_sizes.is_empty() { return; }

    let total: u32 = comp_sizes.iter().sum();
    let min_size = ((total as f32 * min_fraction) as u32).max(50);

    for (size, pixels) in comp_sizes.iter().zip(comp_pixels.iter()) {
        if *size < min_size {
            for &px in pixels {
                let b = px * 4;
                rgba[b] = 0; rgba[b+1] = 0; rgba[b+2] = 0; rgba[b+3] = 0;
            }
        }
    }
}

// ── Largest-blob prediction filter ───────────────────────────────────────────

/// Keep only the largest connected component of a boolean prediction mask.
///
/// The U-Net bottleneck sometimes produces blocky probability hotspots in
/// pure-background regions.  These are always spatially isolated from the main
/// leaf prediction, so keeping only the largest connected component removes them
/// cleanly without any tunable threshold.
fn largest_pred_blob(mask: Vec<bool>, sz: usize) -> Vec<bool> {
    let n = sz * sz;
    let mut visited = vec![false; n];
    let mut best_pixels: Vec<usize> = Vec::new();

    for start in 0..n {
        if !mask[start] || visited[start] { continue; }

        // BFS from this seed
        let mut pixels = Vec::new();
        let mut queue  = std::collections::VecDeque::new();
        queue.push_back(start);
        visited[start] = true;

        while let Some(idx) = queue.pop_front() {
            pixels.push(idx);
            let x = (idx % sz) as i32;
            let y = (idx / sz) as i32;
            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nx = x + dx; let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= sz as i32 || ny >= sz as i32 { continue; }
                let ni = ny as usize * sz + nx as usize;
                if mask[ni] && !visited[ni] {
                    visited[ni] = true;
                    queue.push_back(ni);
                }
            }
        }

        if pixels.len() > best_pixels.len() {
            best_pixels = pixels;
        }
    }

    let mut out = vec![false; n];
    for &i in &best_pixels { out[i] = true; }
    out
}

// ── TTA rotation helpers ──────────────────────────────────────────────────────

/// Rotate an RGBA flat buffer 90° clockwise: src(y,x) → dst(x, size-1-y).
fn rotate_rgba_cw(rgba: &[u8], size: usize) -> Vec<u8> {
    let mut out = vec![0u8; rgba.len()];
    for y in 0..size {
        for x in 0..size {
            let src = y * size + x;
            let dst = x * size + (size - 1 - y);
            let (s4, d4) = (src * 4, dst * 4);
            out[d4]     = rgba[s4];
            out[d4 + 1] = rgba[s4 + 1];
            out[d4 + 2] = rgba[s4 + 2];
            out[d4 + 3] = rgba[s4 + 3];
        }
    }
    out
}

/// Rotate a probability flat buffer 90° clockwise.
fn rotate_prob_cw(probs: &[f32], size: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; probs.len()];
    for y in 0..size {
        for x in 0..size {
            out[x * size + (size - 1 - y)] = probs[y * size + x];
        }
    }
    out
}

/// Apply k × 90° CW rotations (k = 0..3).
pub(crate) fn rotate_rgba_cw_k(rgba: &[u8], size: usize, k: u8) -> Vec<u8> {
    let mut buf = rgba.to_vec();
    for _ in 0..k { buf = rotate_rgba_cw(&buf, size); }
    buf
}

pub(crate) fn rotate_prob_cw_k(probs: &[f32], size: usize, k: u8) -> Vec<f32> {
    let mut buf = probs.to_vec();
    for _ in 0..k { buf = rotate_prob_cw(&buf, size); }
    buf
}

// ── Per-image inference ───────────────────────────────────────────────────────

struct Stats {
    input_leaf_px:      usize,
    predicted_total_px: usize,
}

fn infer_one(
    path:       &Path,
    generator:  &UNetSimple<InferBackend>,
    device:     &InferDevice,
    cfg:        &InferConfig,
    shape_cls:  u32,
    margin_cls: u32,
) -> Result<(Stats, Vec<u8>), String> {
    let sz   = cfg.image_size;
    let sz32 = sz as u32;

    // Load and resize the damaged input image
    let img  = image::open(path).map_err(|e| format!("{e}"))?;
    let img  = img.resize_exact(sz32, sz32, image::imageops::FilterType::Triangle);
    let mut rgba = img.to_rgba8().into_raw();   // length = sz*sz*4
    // Zero transparent pixels and binarise soft/feathered alpha edges.
    // Matches load_images() in training.rs.
    for p in rgba.chunks_mut(4) {
        if p[3] <= 128 { p[0] = 0; p[1] = 0; p[2] = 0; p[3] = 0; }
        else           { p[3] = 255; }
    }

    // Remove tiny disconnected blobs (scanner bars, corner markers, debris).
    // Keeps all fragments ≥ 2 % of total leaf area, so eroded leaf lobes survive.
    remove_small_blobs(&mut rgba, sz, 0.02);

    let n = sz * sz;
    // Count surviving pixels from the ORIGINAL (naturally damaged) image.
    // This is the reference for the damage % calculation regardless of pre-damage.
    let input_leaf_px: usize = rgba.chunks(4).filter(|p| p[3] > 128).count();

    // Optional synthetic pre-damage: apply a small erode_margin_clusters pass on
    // a copy of the image before feeding to the model.  This puts naturally-damaged
    // leaves inside the training distribution (alpha=0 pixels in a recognisable
    // herbivory pattern) so the model triggers its reconstruction pathway.
    // The original `rgba` is kept intact for the no-delete constraint and compositing.
    let model_input_rgba: Vec<u8> = if cfg.pre_damage_pct > 0.0 {
        let mut mask: Vec<bool> = rgba.chunks(4).map(|p| p[3] > 128).collect();
        let fraction = cfg.pre_damage_pct / 100.0;
        let mut rng = SmallRng::from_entropy();
        erode_margin_clusters(&mut mask, sz, sz, fraction, &mut rng);
        // Re-apply mask: eroded pixels become fully transparent
        let mut out = rgba.clone();
        for (i, &alive) in mask.iter().enumerate() {
            if !alive {
                out[i * 4]     = 0;
                out[i * 4 + 1] = 0;
                out[i * 4 + 2] = 0;
                out[i * 4 + 3] = 0;
            }
        }
        out
    } else {
        rgba.clone()
    };

    // Run inference — optionally with 4-rotation TTA.
    // Build 4-channel input: R,G,B,A (no symmetry channel; Quercus is asymmetric).
    let build_4ch = |rgba: &[u8]| -> Vec<f32> {
        let mut data = Vec::with_capacity(4 * n);
        for ch in 0..4usize {
            for i in 0..n {
                data.push((rgba[i * 4 + ch] as f32 / 127.5) - 1.0);
            }
        }
        data
    };

    let pred_vec: Vec<f32> = if cfg.tta_enabled {
        // Average predictions over 0°/90°/180°/270° CW rotations.
        let mut sum = vec![0.0f32; n];
        for k in 0u8..4 {
            let rotated = rotate_rgba_cw_k(&model_input_rgba, sz, k);
            let input_data = build_4ch(&rotated);
            let input_t: Tensor<InferBackend, 4> = Tensor::from_data(
                TensorData::new(input_data, [1usize, 4, sz, sz]), device,
            );
            let raw: Vec<f32> = generator.forward_probs(input_t, shape_cls, margin_cls)
                .into_data().to_vec().unwrap_or_default();
            let unrotated = rotate_prob_cw_k(&raw, sz, (4 - k) % 4);
            for (s, &v) in sum.iter_mut().zip(unrotated.iter()) { *s += v; }
        }
        sum.iter().map(|&v| v / 4.0).collect()
    } else {
        // Single-pass inference
        let input_data = build_4ch(&model_input_rgba);
        let input_t: Tensor<InferBackend, 4> = Tensor::from_data(
            TensorData::new(input_data, [1usize, 4, sz, sz]), device,
        );
        generator.forward_probs(input_t, shape_cls, margin_cls)
            .into_data().to_vec().unwrap_or_default()
    };

    // ── Debug output ──────────────────────────────────────────────────────────
    if cfg.debug_output {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("img");
        // Input PNG: the exact RGBA the model saw (white-composited so it's visible)
        let mut inp_rgba = vec![0u8; n * 4];
        for i in 0..n {
            let b = i * 4;
            let a = model_input_rgba[b + 3];
            let t = a as f32 / 255.0;
            inp_rgba[b]     = (model_input_rgba[b]     as f32 * t + 255.0 * (1.0 - t)) as u8;
            inp_rgba[b + 1] = (model_input_rgba[b + 1] as f32 * t + 255.0 * (1.0 - t)) as u8;
            inp_rgba[b + 2] = (model_input_rgba[b + 2] as f32 * t + 255.0 * (1.0 - t)) as u8;
            inp_rgba[b + 3] = 255;
        }
        let _ = save_rgba_png(&inp_rgba, sz32, sz32,
            &cfg.output_dir.join(format!("{stem}_debug_input.png")));
        // Probability map: jet colormap over the raw sigmoid output
        let mut prob_rgba = vec![0u8; n * 4];
        for i in 0..n {
            let p = pred_vec[i];
            let (r, g, b) = {
                let p = p.clamp(0.0, 1.0);
                if      p < 0.25 { let t = p / 0.25;       (0.0,  t,          1.0) }
                else if p < 0.5  { let t = (p-0.25)/0.25;  (0.0,  1.0,        1.0-t) }
                else if p < 0.75 { let t = (p-0.5)/0.25;   (t,    1.0,        0.0) }
                else             { let t = (p-0.75)/0.25;  (1.0,  1.0-t,      0.0) }
            };
            let b4 = i * 4;
            prob_rgba[b4]     = (r * 255.0) as u8;
            prob_rgba[b4 + 1] = (g * 255.0) as u8;
            prob_rgba[b4 + 2] = (b * 255.0) as u8;
            prob_rgba[b4 + 3] = 255;
        }
        let _ = save_rgba_png(&prob_rgba, sz32, sz32,
            &cfg.output_dir.join(format!("{stem}_debug_prob.png")));
    }

    // Threshold + no-delete constraint against the ORIGINAL rgba (not pre-damaged):
    // any pixel surviving real damage must appear in the reconstruction.
    let pred_mask_raw: Vec<bool> = pred_vec.iter().enumerate()
        .map(|(i, &v)| rgba[i * 4 + 3] > 128 || v >= cfg.threshold)
        .collect();

    // Keep only the largest connected component of the prediction.
    // Background hallucinations (blocky bottleneck artefacts far from the leaf)
    // are always spatially isolated from the main leaf prediction — this single
    // step removes them without any tunable threshold.
    let pred_mask = largest_pred_blob(pred_mask_raw, sz);
    let predicted_total_px = pred_mask.iter().filter(|&&b| b).count();

    // Mean colour of the intact (existing) leaf pixels — used to fill the
    // reconstructed area so the composite looks plausible.
    let (fill_r, fill_g, fill_b) = {
        let mut sr = 0u64; let mut sg = 0u64; let mut sb = 0u64; let mut cnt = 0u64;
        for (i, chunk) in rgba.chunks(4).enumerate() {
            if chunk[3] > 128 && pred_mask.get(i).copied().unwrap_or(false) {
                sr += chunk[0] as u64;
                sg += chunk[1] as u64;
                sb += chunk[2] as u64;
                cnt += 1;
            }
        }
        if cnt > 0 { ((sr/cnt) as u8, (sg/cnt) as u8, (sb/cnt) as u8) }
        else        { (128u8, 128u8, 128u8) }
    };

    // Composite output image:
    //   - predicted=1 AND input had alpha>0 → keep original pixel
    //   - predicted=1 AND input was transparent → fill with mean leaf colour
    //   - predicted=0 → fully transparent (0,0,0,0)
    let mut recon = vec![0u8; n * 4];
    for i in 0..n {
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

    Ok((Stats { input_leaf_px, predicted_total_px }, recon))
}

// ── Utilities ─────────────────────────────────────────────────────────────────

fn save_rgba_png(pixels: &[u8], w: u32, h: u32, path: &Path) -> Result<(), String> {
    use image::{ImageBuffer, Rgba};
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_raw(w, h, pixels.to_vec())
            .ok_or("ImageBuffer::from_raw failed")?;
    img.save(path).map_err(|e| e.to_string())
}

/// Write all results to a CSV file.
pub fn write_csv(results: &[InferResult], path: &Path) -> Result<(), String> {
    use std::io::Write;
    let mut f = std::fs::File::create(path).map_err(|e| e.to_string())?;
    writeln!(f, "Filename,Path,Input_Leaf_px,Predicted_Total_px,Damage_px,Pct_Damage")
        .map_err(|e| e.to_string())?;
    for r in results {
        writeln!(f, "{},{},{},{},{},{:.2}",
            r.filename,
            r.path.display(),
            r.input_leaf_px,
            r.predicted_total_px,
            r.damage_px,
            r.pct_damage,
        ).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn log(tx: &mpsc::Sender<InferMsg>, msg: impl Into<String>) {
    let _ = tx.send(InferMsg::Log(msg.into()));
}
