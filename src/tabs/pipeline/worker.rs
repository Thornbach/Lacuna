//! Integrated-pipeline worker (Slice 2): runs the staged
//!   Segment (YOLO/ort) -> Tile -> Detect (DINO/ort + bank kNN) -> Restitch
//! flow on a background thread, streaming results over an mpsc channel (mirrors
//! the recon_infer threading idiom). Reconstruction + clustering are added in
//! later slices. Models are created INSIDE the worker thread (ort sessions and
//! burn GPU tensors are not freely shareable across threads).

use std::{
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
    sync::atomic::{AtomicBool, Ordering},
};

use burn::tensor::{backend::Backend, Tensor, TensorData};
use leaf_complex_rust_lib::config::Config as MorphConfig;
use leaf_complex_rust_lib::{analyze_rgba, MorphMetrics};

use crate::tabs::leaf_seg::inference::{self as seg, SegConfig};
use crate::tabs::recon_simple::model::{load_simple_infer, UNetSimple};
use crate::tabs::recon_train::model::{create_infer_device, InferBackend};

use super::bank::{CoresetBank, GpuBank};
use super::cluster;
use super::detect::{detect, DetectParams};
use super::dino::DinoExtractor;
use super::fewshot::{self, FewShotHead};
use super::meta::DetectorMeta;
use super::tiling::{restitch_mask, tile_leaf};

const CROP_WIN: u32 = 64;        // context-crop size for the anomaly gallery
const CLUSTER_EPS: f32 = 1.5;    // DBSCAN radius in standardized descriptor space
const CLUSTER_MIN_PTS: usize = 5;
const RECON_SIZE: usize = 512;   // reconstruction model input size (match training image_size_px)
// Reconstruction decision threshold. The net tends to OVER-predict (fills sinuses →
// false positives), so bias above 0.5 to trim low-confidence over-fill. Tune live in
// the Recon Infer tab's threshold slider, then set the sweet-spot value here.
const RECON_THRESHOLD: f32 = 0.65;

pub struct PipeConfig {
    pub image_paths: Vec<PathBuf>,
    pub output_dir:  PathBuf,
    pub yolo_model:  PathBuf,
    pub dino_model:  PathBuf,
    pub bank_path:   PathBuf,
    pub meta_path:   PathBuf,
    pub tile_size:    u32,
    pub margin_erode: u32,
    pub dino_res:     u32,
    pub conf:         f32,
    pub recon_ckpt:   Option<PathBuf>, // folder with gen.mpk; None = skip reconstruction
    pub head_path:    Option<PathBuf>, // few-shot head json; Some = use few-shot detector (skips bank)
    pub head_tau:     f32,             // few-shot decision threshold (hysteresis SEED)
    pub head_grow:    f32,             // hysteresis GROW threshold (higher = tighter regions)
    pub seg_alpha_lo:   f32,           // YOLO cutout edge tightness (feather start)
    pub seg_chroma_min: i32,           // YOLO cutout background-chroma rejection
}

pub struct PipelineLeaf {
    pub src:        PathBuf,
    pub w:          u32,
    pub h:          u32,
    pub rgba:       Vec<u8>,   // leaf cutout RGBA
    pub anomaly:    Vec<bool>, // w*h, restitched from tiles
    pub n_regions:  usize,
    pub recon_area: usize,     // ADDED reconstructed area = lost tissue (px); 0 if no recon
    pub recon_whole: usize,    // whole reconstructed intact-leaf area (px); 0 if no recon
    pub recon_mask: Vec<bool>, // predicted intact-leaf mask at RECON_PREVIEW²; empty if no recon
    pub morph:      Option<MorphMetrics>, // EC/MC complexity metrics (None if analysis failed)
}

/// Side length of the stored reconstruction preview mask (= RECON_SIZE).
pub const RECON_PREVIEW: usize = RECON_SIZE;

/// A detected anomaly region, dataset-wide (the clustering atom). References its
/// leaf by index and carries everything the UI needs to highlight + display it.
pub struct AnomalyRegion {
    pub leaf:       usize,      // index into the tab's leaves Vec (emit order)
    pub bbox_leaf:  [u32; 4],   // x, y, w, h in LEAF coords (tile origin + region bbox)
    pub mask:       Vec<bool>,  // bbox-local
    pub descriptor: [f32; 8],   // 8-D clustering feature (PatchCore path; zeros for few-shot)
    pub family:     i32,        // head-assigned family ≥1 (few-shot path; 0 for PatchCore)
    pub crop:       Vec<u8>,    // RGBA context-crop thumbnail, crop_size²·4
    pub crop_size:  u32,
}

pub enum PipeMsg {
    Stage(String),
    Leaf(PipelineLeaf),
    Progress { done: usize, total: usize },
    Log(String),
    Error(String),
    /// Final clustering of all regions: `labels`/`coords` are parallel to `regions`.
    /// `names` maps a cluster/family id to a display name (few-shot supplies the
    /// head's family names; empty for the PatchCore path → UI names them "Cluster N").
    Clusters {
        labels:  Vec<i32>,
        coords:  Vec<[f32; 2]>,
        names:   std::collections::HashMap<i32, String>,
        regions: Vec<AnomalyRegion>,
    },
    Finished,
}

pub fn spawn_pipeline(cfg: PipeConfig, tx: mpsc::Sender<PipeMsg>, cancel: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        if let Err(e) = run_pipeline(&cfg, &tx, &cancel) {
            let _ = tx.send(PipeMsg::Error(e));
        }
        let _ = tx.send(PipeMsg::Finished);
    });
}

fn log(tx: &mpsc::Sender<PipeMsg>, m: impl Into<String>) {
    let _ = tx.send(PipeMsg::Log(m.into()));
}
fn stage(tx: &mpsc::Sender<PipeMsg>, m: impl Into<String>) {
    let _ = tx.send(PipeMsg::Stage(m.into()));
}

fn run_pipeline(
    cfg:    &PipeConfig,
    tx:     &mpsc::Sender<PipeMsg>,
    cancel: &AtomicBool,
) -> Result<(), String> {
    std::fs::create_dir_all(&cfg.output_dir).map_err(|e| e.to_string())?;

    stage(tx, "Loading models");
    log(tx, crate::tabs::gpu_diagnostics());
    let mut yolo = seg::build_yolo(&cfg.yolo_model)?;
    let mut dino = DinoExtractor::load(&cfg.dino_model, cfg.dino_res)?;
    // DINO GPU check: two warmup forwards on a dummy tile. The WARM time makes it
    // unambiguous whether DINO is actually on the GPU (~20ms) or silently on CPU
    // (~600ms) — regardless of what "CUDA loadable" says.
    {
        let dummy = image::RgbImage::new(256, 256);
        let _ = dino.features(&dummy);            // cold (CUDA/cuDNN autotune)
        let _ = dino.features(&dummy);            // warm
        log(tx, format!(
            "DINO warmup {:.0}ms/forward — {}", dino.last_ms,
            if dino.last_ms < 150.0 { "GPU ✓" } else { "CPU (slow!) — cuDNN not engaged for DINO" },
        ));
    }
    let device = create_infer_device();

    // Detector: the few-shot head (preferred, validated) OR the PatchCore bank.
    // When a head is present it fully replaces the kNN path, so we skip loading
    // the ~0.9 GB coreset bank + meta entirely (the deployment VRAM win).
    let head: Option<FewShotHead> = match &cfg.head_path {
        Some(p) => {
            let h = FewShotHead::load(p)?;
            log(tx, format!(
                "few-shot head: {} classes, dim {}, τ={:.2}{}",
                h.classes.len(), h.dim, cfg.head_tau,
                h.onnx_parity.map(|p| format!(" (parity {p:.3})")).unwrap_or_default(),
            ));
            if h.infer_resolution != cfg.dino_res {
                log(tx, format!(
                    "⚠ head trained at {}px but DINO runs at {}px — features may not align",
                    h.infer_resolution, cfg.dino_res,
                ));
            }
            Some(h)
        }
        None => None,
    };
    let (meta, bank): (Option<DetectorMeta>, Option<GpuBank<InferBackend>>) = if head.is_some() {
        (None, None)
    } else {
        let meta = DetectorMeta::load(&cfg.meta_path)?;
        log(tx, "loading coreset bank…");
        let cpu_bank = CoresetBank::load(&cfg.bank_path)?;
        let bank = GpuBank::<InferBackend>::new(&cpu_bank, device.clone());
        log(tx, format!("bank {}x{} ready", cpu_bank.n, cpu_bank.d));
        (Some(meta), Some(bank))
    };

    // optional reconstruction model (UNetSimple) for the intact-area stat
    let recon: Option<UNetSimple<InferBackend>> = match &cfg.recon_ckpt {
        Some(dir) if dir.join("gen.mpk").exists() => match load_simple_infer(dir, &device) {
            Ok(g) => {
                log(tx, "reconstruction model loaded");
                Some(g)
            }
            Err(e) => {
                log(tx, format!("recon load failed: {e}"));
                None
            }
        },
        _ => None,
    };

    let params = DetectParams::default();
    let morph_cfg = MorphConfig::default();
    let total = cfg.image_paths.len();
    let mut leaf_idx = 0usize;
    let mut all_regions: Vec<AnomalyRegion> = Vec::new();
    for (idx, path) in cfg.image_paths.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            log(tx, "cancelled");
            break;
        }
        let _ = tx.send(PipeMsg::Progress { done: idx, total });
        let fname = path.file_name().unwrap_or_default().to_string_lossy().to_string();

        // Probe the input: an image that already has a transparent background is a
        // pre-cut leaf, so skip YOLO and treat the whole image as one leaf. Only
        // opaque images (raw scans) go through segmentation.
        let probe = match image::open(path) {
            Ok(i) => i.to_rgba8(),
            Err(e) => {
                log(tx, format!("[skip] {}: {e}", path.display()));
                continue;
            }
        };
        let (iw, ih) = probe.dimensions();
        let raw = probe.into_raw();
        let total_px = (iw as usize) * (ih as usize);
        let transparent = raw.chunks_exact(4).filter(|p| p[3] < 128).count();
        let is_cutout = total_px > 0 && transparent * 100 > total_px; // >1% transparent

        if is_cutout {
            stage(tx, format!("Leaf {fname} (pre-cut)"));
            process_leaf(
                raw, iw, ih, path, &fname, &mut dino, bank.as_ref(), meta.as_ref(), head.as_ref(),
                &params, &recon, &device, cfg, &morph_cfg, tx, cancel, &mut leaf_idx,
                &mut all_regions,
            )?;
        } else {
            stage(tx, format!("Segment: {fname}"));
            let seg_cfg = SegConfig {
                model_path:  cfg.yolo_model.clone(),
                image_paths: Vec::new(),
                output_dir:  cfg.output_dir.clone(),
                imgsz:       640,
                conf:        cfg.conf,
                alpha_lo:    cfg.seg_alpha_lo,
                chroma_min:  cfg.seg_chroma_min,
            };
            let item = match seg::segment_one(&mut yolo, path, &seg_cfg) {
                Ok(it) => it,
                Err(e) => {
                    log(tx, format!("[skip] {}: {e}", path.display()));
                    continue;
                }
            };
            log(tx, format!("{}: {} leaves", item.filename, item.instances.len()));
            for inst in &item.instances {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                let leaf_img = match image::open(&inst.cutout_path) {
                    Ok(i) => i.to_rgba8(),
                    Err(e) => {
                        log(tx, format!("cutout open: {e}"));
                        continue;
                    }
                };
                let (lw, lh) = leaf_img.dimensions();
                process_leaf(
                    leaf_img.into_raw(), lw, lh, path, &item.filename, &mut dino, bank.as_ref(),
                    meta.as_ref(), head.as_ref(), &params, &recon, &device, cfg, &morph_cfg, tx,
                    cancel, &mut leaf_idx, &mut all_regions,
                )?;
            }
        }
    }

    // ── group all detected regions into families ──
    if !all_regions.is_empty() && !cancel.load(Ordering::Relaxed) {
        stage(tx, "Clustering");
        let mut names: std::collections::HashMap<i32, String> = std::collections::HashMap::new();
        let (labels, coords) = if let Some(head) = &head {
            // Few-shot: the head already assigns each region a family. v1 clusters
            // ARE those families (labels = head family), with a per-family jittered
            // scatter so the plot reads as separated clouds. (Open-set discovery on
            // the dense embedding is a follow-up; this is the validated path.)
            let labels: Vec<i32> = all_regions.iter().map(|r| r.family).collect();
            let coords = family_scatter(&labels);
            for &l in &labels {
                if l >= 0 {
                    names.entry(l).or_insert_with(|| head.family_name(l));
                }
            }
            log(tx, format!("{} regions -> {} families (few-shot)", all_regions.len(), names.len()));
            (labels, coords)
        } else {
            // PatchCore: cluster the 8-D descriptors (StandardScaler + DBSCAN + PCA-2).
            let descs: Vec<[f32; 8]> = all_regions.iter().map(|r| r.descriptor).collect();
            let std = cluster::standardize(&descs);
            let labels = cluster::dbscan(&std, CLUSTER_EPS, CLUSTER_MIN_PTS);
            let coords = cluster::pca2(&std);
            let n = labels.iter().copied().filter(|&l| l >= 0).max().map(|m| m + 1).unwrap_or(0);
            log(tx, format!("{} regions -> {} clusters", all_regions.len(), n));
            (labels, coords)
        };
        let _ = tx.send(PipeMsg::Clusters { labels, coords, names, regions: all_regions });
    }

    let _ = tx.send(PipeMsg::Progress { done: total, total });
    stage(tx, "Done");
    Ok(())
}

/// Run one leaf fully through tile -> detect -> restitch -> reconstruct ->
/// morphology, then emit it. Streams a completed leaf so the UI can show it
/// immediately; clustering happens once over all regions at the end.
#[allow(clippy::too_many_arguments)]
fn process_leaf(
    leaf_rgba:   Vec<u8>,
    lw:          u32,
    lh:          u32,
    src:         &Path,
    fname:       &str,
    dino:        &mut DinoExtractor,
    bank:        Option<&GpuBank<InferBackend>>,
    meta:        Option<&DetectorMeta>,
    head:        Option<&FewShotHead>,
    params:      &DetectParams,
    recon:       &Option<UNetSimple<InferBackend>>,
    device:      &<InferBackend as Backend>::Device,
    cfg:         &PipeConfig,
    morph_cfg:   &MorphConfig,
    tx:          &mpsc::Sender<PipeMsg>,
    cancel:      &AtomicBool,
    leaf_idx:    &mut usize,
    all_regions: &mut Vec<AnomalyRegion>,
) -> Result<(), String> {
    // erode a COPY's alpha so the cutout's background ring is excluded from
    // detection, while the displayed leaf keeps its full extent.
    let mut det_rgba = leaf_rgba.clone();
    super::tiling::erode_alpha(&mut det_rgba, lw, lh, cfg.margin_erode, 10);
    let tiles = tile_leaf(&det_rgba, lw, lh, cfg.tile_size, 10);
    let n_tiles = tiles.len();
    let t_leaf = std::time::Instant::now();
    let mut det_ms = 0f64;      // total per-tile detect wall time
    let mut dino_ms = 0f64;     // of which, DINO forward
    let mut dino_tiles = 0usize;// non-skipped tiles that ran DINO
    let mut tile_masks = Vec::with_capacity(n_tiles);
    let mut n_regions = 0usize;
    for (ti, t) in tiles.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        stage(tx, format!("Detect {fname} — tile {}/{}", ti + 1, n_tiles));
        let tw = cfg.tile_size as usize;
        // Background-tile skip: a tile with fewer leaf pixels than the minimum
        // region area can produce no detection (and too few to enclose a hole), so
        // the DINO ViT forward — the per-tile bottleneck — is wasted on it. Skip it
        // with an empty mask so restitch indices stay aligned.
        let valid_count = t.valid.iter().filter(|&&v| v).count() as u32;
        if valid_count < params.min_area.max(1) {
            tile_masks.push(vec![false; tw * tw]);
            continue;
        }
        let t_tile = std::time::Instant::now();
        if let Some(head) = head {
            // Few-shot path: head classifies each patch → defect prob + family,
            // then grid hysteresis (seed τ / grow τ / min-region patches). Higher
            // grow τ = regions hug the high-confidence core (tighter boxes).
            let tau_lo = cfg.head_grow.clamp(0.05, cfg.head_tau);
            let (mask, regions) = fewshot::fewshot_detect(
                dino, head, &t.rgb, tw, tw, Some(&t.valid),
                cfg.head_tau, tau_lo, fewshot::HEAD_MIN_REGION_PATCHES,
                params.region_close_px as usize, params.min_area,
            )?;
            n_regions += regions.len();
            for rg in &regions {
                let [rx, ry, rw, rh] = rg.bbox;
                let cx = t.origin[0] as f32 + rg.centroid[0];
                let cy = t.origin[1] as f32 + rg.centroid[1];
                let crop = context_crop(&leaf_rgba, lw, lh, cx, cy, CROP_WIN);
                all_regions.push(AnomalyRegion {
                    leaf: *leaf_idx,
                    bbox_leaf: [t.origin[0] + rx, t.origin[1] + ry, rw, rh],
                    mask: rg.mask.clone(),
                    descriptor: [0.0; 8],
                    family: rg.family,
                    crop,
                    crop_size: CROP_WIN,
                });
            }
            tile_masks.push(mask);
        } else {
            // PatchCore path: kNN vs bank + 3-channel decide.
            let bank = bank.ok_or("PatchCore path requires a coreset bank")?;
            let meta = meta.ok_or("PatchCore path requires detector meta")?;
            let r = detect(dino, bank, meta, params, &t.rgb, tw, tw, Some(&t.valid))?;
            log(tx, format!("  tile {}/{}: dino={}ms knn={}ms", ti + 1, n_tiles, r.dino_ms, r.knn_ms));
            n_regions += r.regions.len();
            for rg in &r.regions {
                let [rx, ry, rw, rh] = rg.bbox;
                let cx = t.origin[0] as f32 + rg.centroid[0];
                let cy = t.origin[1] as f32 + rg.centroid[1];
                let crop = context_crop(&leaf_rgba, lw, lh, cx, cy, CROP_WIN);
                all_regions.push(AnomalyRegion {
                    leaf: *leaf_idx,
                    bbox_leaf: [t.origin[0] + rx, t.origin[1] + ry, rw, rh],
                    mask: rg.mask.clone(),
                    descriptor: rg.descriptor,
                    family: 0,
                    crop,
                    crop_size: CROP_WIN,
                });
            }
            tile_masks.push(r.mask);
        }
        det_ms += t_tile.elapsed().as_secs_f64() * 1000.0;
        dino_ms += dino.last_ms as f64;
        dino_tiles += 1;
    }
    let anomaly = restitch_mask(&tiles, &tile_masks, lw, lh);
    let t_rec = std::time::Instant::now();
    let (recon_area, recon_whole, recon_mask) = match recon {
        Some(g) => {
            stage(tx, format!("Reconstruct {fname}"));
            reconstruct_area(g, device, &leaf_rgba, lw, lh, RECON_SIZE)
        }
        None => (0, 0, Vec::new()),
    };
    let rec_ms = t_rec.elapsed().as_secs_f64() * 1000.0;
    // morphology (EC/MC complexity) on the leaf cutout — metrics only (overlays
    // dropped, so retention stays ~constant per leaf).
    stage(tx, format!("Morphology {fname}"));
    let t_morph = std::time::Instant::now();
    let morph = analyze_rgba(&leaf_rgba, lw, lh, morph_cfg).ok().map(|r| r.metrics);
    let morph_ms = t_morph.elapsed().as_secs_f64() * 1000.0;
    log(tx, format!(
        "[timing] leaf {}: {dino_tiles} tiles · detect {det_ms:.0}ms (dino {dino_ms:.0}ms = \
         {:.1}ms/tile, post {:.0}ms) · recon {rec_ms:.0}ms · morph {morph_ms:.0}ms · total {:.0}ms",
        *leaf_idx + 1,
        dino_ms / dino_tiles.max(1) as f64,
        det_ms - dino_ms,
        t_leaf.elapsed().as_secs_f64() * 1000.0,
    ));

    let _ = tx.send(PipeMsg::Leaf(PipelineLeaf {
        src: src.to_path_buf(),
        w: lw,
        h: lh,
        rgba: leaf_rgba,
        anomaly,
        n_regions,
        recon_area,
        recon_whole,
        recon_mask,
        morph,
    }));
    *leaf_idx += 1;
    Ok(())
}

/// Returns `(added, whole, mask)` for a leaf cutout: `added` = reconstructed lost
/// tissue (predicted leaf where the cutout is missing), `whole` = the whole intact
/// leaf (predicted ∪ visible) — both in leaf-resolution px — and `mask` = the
/// predicted intact-leaf mask (`sz²`) for the canvas preview. Runs UNetSimple at
/// `sz`×`sz`.
fn reconstruct_area(
    gen:    &UNetSimple<InferBackend>,
    device: &<InferBackend as Backend>::Device,
    rgba:   &[u8],
    w:      u32,
    h:      u32,
    sz:     usize,
) -> (usize, usize, Vec<bool>) {
    let Some(img) = image::RgbaImage::from_raw(w, h, rgba.to_vec()) else { return (0, 0, Vec::new()) };
    let small = image::imageops::resize(&img, sz as u32, sz as u32, image::imageops::FilterType::Triangle);
    let mut r = small.into_raw();
    for p in r.chunks_mut(4) {
        if p[3] <= 128 {
            p[0] = 0; p[1] = 0; p[2] = 0; p[3] = 0;
        } else {
            p[3] = 255;
        }
    }
    let n = sz * sz;
    let mut data = Vec::with_capacity(4 * n);
    for ch in 0..4 {
        for i in 0..n {
            data.push((r[i * 4 + ch] as f32 / 127.5) - 1.0);
        }
    }
    let input: Tensor<InferBackend, 4> =
        Tensor::from_data(TensorData::new(data, [1usize, 4, sz, sz]), device);
    let pred: Vec<f32> = gen.forward_probs(input, 0, 0).into_data().to_vec().unwrap_or_default();
    if pred.len() != n {
        return (0, 0, Vec::new());
    }
    // "Shape only" cleanup: keep the silhouette connected to the visible leaf and
    // fill interior holes → one solid intact shape for both the preview and the area.
    let visible: Vec<bool> = (0..n).map(|i| r[i * 4 + 3] > 128).collect();
    let mask = crate::tabs::recon_train::training::refine_silhouette(&pred, &visible, sz, sz, RECON_THRESHOLD);
    let mut added = 0usize; // refined leaf where the cutout is missing = lost tissue
    let mut whole = 0usize; // refined silhouette = whole intact leaf
    for i in 0..n {
        if mask[i] {
            whole += 1;
            if !visible[i] { added += 1; }
        }
    }
    let scale = (w as f64 * h as f64) / n as f64;
    (
        (added as f64 * scale).round() as usize,
        (whole as f64 * scale).round() as usize,
        mask,
    )
}

/// Scatter coordinates for the few-shot path: there is no PCA embedding (regions
/// carry a family label, not an 8-D descriptor), so lay each family out as its own
/// jittered cloud on a grid. Deterministic per index (a tiny hash) so re-renders
/// are stable. The plot then reads as one separated blob per family.
fn family_scatter(labels: &[i32]) -> Vec<[f32; 2]> {
    let mut fams: Vec<i32> = labels.iter().copied().filter(|&l| l >= 0).collect();
    fams.sort_unstable();
    fams.dedup();
    let col_of = |f: i32| fams.iter().position(|&x| x == f);
    let cols = (fams.len() as f32).sqrt().ceil().max(1.0) as usize;
    labels
        .iter()
        .enumerate()
        .map(|(i, &l)| {
            let Some(c) = col_of(l) else { return [0.0, 0.0] };
            let (gx, gy) = ((c % cols) as f32, (c / cols) as f32);
            // hash the index into two [-0.35, 0.35] jitters
            let h = (i as u32).wrapping_mul(2654435761);
            let jx = ((h & 0xffff) as f32 / 65535.0 - 0.5) * 0.7;
            let jy = (((h >> 16) & 0xffff) as f32 / 65535.0 - 0.5) * 0.7;
            [gx * 1.6 + jx, gy * 1.6 + jy]
        })
        .collect()
}

/// Fixed-size context crop (RGBA) centered on (cx, cy), clamp-padded at borders.
fn context_crop(rgba: &[u8], w: u32, h: u32, cx: f32, cy: f32, win: u32) -> Vec<u8> {
    let half = (win / 2) as i32;
    let (cx, cy) = (cx.round() as i32, cy.round() as i32);
    let mut out = vec![0u8; (win * win * 4) as usize];
    for oy in 0..win as i32 {
        let sy = (cy - half + oy).clamp(0, h as i32 - 1) as u32;
        for ox in 0..win as i32 {
            let sx = (cx - half + ox).clamp(0, w as i32 - 1) as u32;
            let si = ((sy * w + sx) * 4) as usize;
            let oi = ((oy as u32 * win + ox as u32) * 4) as usize;
            out[oi..oi + 4].copy_from_slice(&rgba[si..si + 4]);
        }
    }
    out
}
