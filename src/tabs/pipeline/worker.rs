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
use rand::{rngs::SmallRng, SeedableRng};

use crate::tabs::eroder::algorithm::erode_margin_clusters;
use crate::tabs::leaf_seg::inference::{self as seg, SegConfig};
use crate::tabs::recon_infer::inference::{rotate_rgba_cw_k, rotate_prob_cw_k};
use crate::tabs::recon_simple::model::{load_simple_infer, UNetSimple};
use crate::tabs::recon_train::model::{create_infer_device, InferBackend};

use super::bank::{CoresetBank, GpuBank};
use super::cluster;
use super::detect::{self, DetectParams};
use super::dino::DinoExtractor;
use super::fewshot::{self, FewShotHead};
use super::meta::DetectorMeta;
use super::tiling::tile_leaf;

pub(crate) const CROP_WIN: u32 = 64; // context-crop size for the anomaly gallery
// Reconstruction model input size. MUST match the checkpoint's trained
// image_size_px (settings.json → recon_simple.image_size_px at training time —
// 256 as of 2026-07). A mismatch doesn't crash (the U-Net is resolution-
// agnostic in shape) but silently degrades quality: every feature ends up at
// the wrong physical scale. Was hardcoded to 512 while the deployed checkpoint
// was actually trained at 256 — root cause of "Pipeline reconstruction looks
// worse than Recon Infer on the same model" (2026-07-10). Keep in sync with
// whatever the currently-deployed checkpoint was trained at.
const RECON_SIZE: usize = 256;
// Reconstruction decision threshold. The net tends to OVER-predict (fills sinuses →
// false positives), so bias above 0.5 to trim low-confidence over-fill. Tune live in
// the Recon Infer tab's threshold slider, then set the sweet-spot value here.
// Used for hole detection and the visual reconstruction preview — deliberately
// conservative so natural margins/sinuses never get flagged as holes or painted
// into the preview overlay.
const RECON_THRESHOLD: f32 = 0.65;
// Separate, lower threshold used ONLY for the scalar lost-tissue-%/area stat
// (recon_area/recon_whole below), NOT the preview or hole detection. RECON_THRESHOLD's
// conservatism is correct for those two, but it introduces a systematic ~4%
// UNDER-estimate in the area stat specifically (measured via `--recon-validate`
// on checkpoint `RECONTRAIN/checkpoint_best`, 2026-07: lost-tissue-% bias -4.34%
// at τ=0.65 vs +0.12% at τ=0.28 — the bias-optimal operating point found by
// sweeping). If the checkpoint is retrained, re-run `--recon-validate` and update
// this constant to the new bias-optimal τ from its report.
const AREA_THRESHOLD: f32 = 0.28;
// Pre-damage nudge applied to the model's input ONLY (never to the true
// observed alpha used for hole detection/the area stat). Real herbivory
// doesn't always visually resemble the synthetic erosion patterns the model
// trained on, so a small additional synthetic erosion nudges the input back
// inside the training distribution, making the model reliably trigger its
// reconstruction pathway. Mirrors Recon Infer's `pre_damage_pct` (same
// technique, same default 1%, ported here 2026-07-10 alongside the
// RECON_SIZE fix and TTA below).
const PRE_DAMAGE_PCT: f32 = 1.0;
// Reserved family/cluster id for reconstruction-flagged holes: background-colored
// gaps the cutout treats as opaque leaf but the recon model believes are NOT leaf
// tissue. These bypass whatever detector clustering is active (few-shot family /
// PatchCore DBSCAN) — they carry no DINO descriptor, so they always form their own
// fixed cluster rather than being subject to either path's whims.
pub(crate) const HOLE_FAMILY: i32 = 9999;
const HOLE_MIN_AREA: u32 = 6; // min connected-component size (leaf-resolution px)
// Reserved family/cluster id for PatchCore-only findings: pixels the open-set
// kNN-vs-healthy-bank detector flags that the few-shot head's closed-set
// classifier did NOT — i.e. something that doesn't match any known trained
// defect family. Bypasses DBSCAN like HOLE_FAMILY, for the same reason (no
// meaningful descriptor to cluster on relative to the few-shot family ids).
pub(crate) const NOVEL_FAMILY: i32 = 9998;
// PatchCore's own false-positive rate was never put through the hard-negative-
// mining campaign that got the few-shot head's FP-region rate down (RESULTS.md:
// baseline 66% -> shipped few-shot 41%). A flat pass-through of "PatchCore
// disagrees with the head" mostly surfaces PatchCore's own uncalibrated
// misfires (veins, margins, glare), not genuine unseen anomalies — so the
// safety-net role needs a stricter bar than PatchCore's normal primary-
// detector operating point before something is worth surfacing as "novel".
const NOVEL_CONFIDENCE_MULT: f32 = 1.5;

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
    pub head_path:    Option<PathBuf>, // few-shot head json; runs alongside the bank if both resolve
    pub use_patchcore: bool,          // ALSO run the bank when a head is present (opt-in safety net)
    pub head_tau:     f32,             // few-shot decision threshold (hysteresis SEED)
    pub head_grow:    f32,             // hysteresis GROW threshold (higher = tighter regions)
    pub seg_alpha_lo:   f32,           // YOLO cutout edge tightness (feather start)
    pub seg_chroma_min: i32,           // YOLO cutout background-chroma rejection
    pub cluster_eps:     f32,          // DBSCAN radius; lower = more/smaller/looser clusters
    pub cluster_min_pts: usize,        // DBSCAN min points; lower = more/smaller/looser clusters
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
    pub descriptor: [f32; 8],   // 8-D clustering feature (PatchCore path; zeros for few-shot/sentinels)
    pub family:     i32,        // head-assigned family ≥1 (few-shot); 0 (PatchCore-only); or a
                                 // reserved sentinel (HOLE_FAMILY / NOVEL_FAMILY)
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
        // Catch panics, not just returned Err — without this, a panic anywhere
        // in run_pipeline (a model/checkpoint mismatch, a GPU/driver issue on
        // an unfamiliar machine, etc.) silently kills this thread. Neither
        // PipeMsg::Error NOR PipeMsg::Finished ever gets sent, so the UI just
        // sits showing whatever the last progress message was — indistinguishable
        // from a genuine hang, and on a release build (windows_subsystem =
        // "windows", no console) the panic message itself goes nowhere visible.
        // Confirmed root cause of an earlier "pipeline never gets past leaf 1"
        // report this same session (stale checkpoint -> GroupNorm channel-count
        // panic) — this turns any recurrence into a real, readable error instead.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_pipeline(&cfg, &tx, &cancel)
        }));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                let _ = tx.send(PipeMsg::Error(e));
            }
            Err(panic_payload) => {
                let msg = panic_payload.downcast_ref::<&str>().map(|s| s.to_string())
                    .or_else(|| panic_payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "no panic message captured".to_string());
                let _ = tx.send(PipeMsg::Error(format!(
                    "Pipeline crashed (not a hang): {msg}\n\
                     Likely a model/checkpoint mismatch or a GPU/driver issue on this \
                     machine — check that the models/ folder matches a known-working install."
                )));
            }
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

    // Detector: the few-shot head (closed-set, known-type recall) AND the
    // PatchCore bank (open-set — models "distance from healthy", not "matches a
    // known defect prototype"). These are complementary, not either/or: a
    // classifier trained on known families structurally can't recognize a novel
    // or diffuse/uniform anomaly type it never saw labeled examples of, which is
    // exactly the case PatchCore is validated to catch instead (leave-one-family-
    // out testing showed it beats the closed-set head on unseen types). Both now
    // run together when both are configured — bank/meta loading is non-fatal so
    // a head-only setup still works exactly as before.
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
    // With a head present, running PatchCore too is opt-in (`cfg.use_patchcore`)
    // — without a head, PatchCore is the only detector so it always runs
    // regardless of that flag.
    let want_patchcore = head.is_none() || cfg.use_patchcore;
    let (meta, bank): (Option<DetectorMeta>, Option<GpuBank<InferBackend>>) = if !want_patchcore {
        log(tx, "PatchCore disabled — running the few-shot head only".to_string());
        (None, None)
    } else {
        match (DetectorMeta::load(&cfg.meta_path), CoresetBank::load(&cfg.bank_path)) {
            (Ok(meta), Ok(cpu_bank)) => {
                let bank = GpuBank::<InferBackend>::new(&cpu_bank, device.clone());
                log(tx, format!(
                    "PatchCore bank {}x{} ready{}", cpu_bank.n, cpu_bank.d,
                    if head.is_some() { " — running alongside the few-shot head (open-set safety net)" } else { "" },
                ));
                (Some(meta), Some(bank))
            }
            (me, mb) => {
                if head.is_none() {
                    return Err(format!(
                        "no detector available — head: {:?} not configured; PatchCore meta/bank failed: {:?} / {:?}",
                        cfg.head_path, me.err(), mb.err(),
                    ));
                }
                log(tx, "PatchCore bank/meta not configured — running the few-shot head only".to_string());
                (None, None)
            }
        }
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
        names.insert(HOLE_FAMILY, "Hole (reconstruction)".to_string());
        names.insert(NOVEL_FAMILY, "Novel (PatchCore)".to_string());
        let is_sentinel = |f: i32| f == HOLE_FAMILY || f == NOVEL_FAMILY;
        let n_holes = all_regions.iter().filter(|r| r.family == HOLE_FAMILY).count();
        let n_novel = all_regions.iter().filter(|r| r.family == NOVEL_FAMILY).count();
        let (labels, coords) = if let Some(head) = &head {
            // Few-shot: the head already assigns each region a family. v1 clusters
            // ARE those families (labels = head family), with a per-family jittered
            // scatter so the plot reads as separated clouds. (Open-set discovery on
            // the dense embedding is a follow-up; this is the validated path.)
            // Hole/Novel regions already carry their sentinel family, so they flow
            // through unchanged and land in their own scatter cloud.
            let labels: Vec<i32> = all_regions.iter().map(|r| r.family).collect();
            let coords = family_scatter(&labels);
            for &l in &labels {
                if l >= 0 && !is_sentinel(l) {
                    names.entry(l).or_insert_with(|| head.family_name(l));
                }
            }
            log(tx, format!(
                "{} regions -> {} families (few-shot, {n_holes} holes, {n_novel} novel/PatchCore)",
                all_regions.len(), names.len(),
            ));
            (labels, coords)
        } else {
            // PatchCore-only: cluster the 8-D descriptors (StandardScaler + DBSCAN +
            // PCA-2) over the non-sentinel regions only — holes carry no meaningful
            // descriptor (zeros), so mixing them into DBSCAN would dump them wherever
            // the nearest zero-ish real cluster happens to be rather than isolating them.
            let real_idx: Vec<usize> = (0..all_regions.len()).filter(|&i| !is_sentinel(all_regions[i].family)).collect();
            let descs: Vec<[f32; 8]> = real_idx.iter().map(|&i| all_regions[i].descriptor).collect();
            let std = cluster::standardize(&descs);
            let db_labels = cluster::dbscan(&std, cfg.cluster_eps, cfg.cluster_min_pts);
            let db_coords = cluster::pca2(&std);
            // seed from each region's OWN family (preserves HOLE_FAMILY/NOVEL_FAMILY
            // sentinels; 0 — overwritten below — for every real PatchCore region).
            let mut labels: Vec<i32> = all_regions.iter().map(|r| r.family).collect();
            let mut coords = vec![[0.0f32; 2]; all_regions.len()];
            for (k, &i) in real_idx.iter().enumerate() {
                labels[i] = db_labels[k];
                coords[i] = db_coords[k];
            }
            let n = labels.iter().copied().filter(|&l| l >= 0 && !is_sentinel(l)).max().map(|m| m + 1).unwrap_or(0);
            log(tx, format!("{} regions -> {} clusters ({n_holes} holes, {n_novel} novel)", all_regions.len(), n));
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
    let mut n_regions = 0usize;

    // Full-leaf stitched signal canvases, filled in tile by tile and decided
    // ONCE at the end — this (not per-tile decisions) is what actually fixes
    // tile-seam artifacts: a real anomaly spanning two tiles is judged as one
    // connected region instead of two independently-thresholded halves, either
    // of which can fall below ITS OWN tile's seed bar and get silently dropped,
    // cutting the anomaly at the tile boundary. It also fixes a second, subtler
    // problem for PatchCore: `robust_z`'s median/MAD assumes most of what it's
    // computed over is healthy, which breaks down when a large anomaly
    // dominates a small tile — computing it over the whole leaf instead is far
    // more robust to a big, localized anomaly.
    let (lwu, lhu) = (lw as usize, lh as usize);
    let n_leaf = lwu * lhu;
    let mut fs_prob = vec![0f32; n_leaf];
    let mut fs_fam  = vec![0i32; n_leaf];
    let mut pc_dino = vec![0f32; n_leaf];
    let mut pc_a    = vec![0f32; n_leaf];
    let mut pc_b    = vec![0f32; n_leaf];
    let mut pc_res  = vec![0f32; n_leaf];
    let mut leaf_valid = vec![false; n_leaf];

    for (ti, t) in tiles.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        stage(tx, format!("Detect {fname} — tile {}/{}", ti + 1, n_tiles));
        let tw = cfg.tile_size as usize;
        // Background-tile skip: a tile with fewer leaf pixels than the minimum
        // region area can produce no detection (and too few to enclose a hole), so
        // the DINO ViT forward — the per-tile bottleneck — is wasted on it.
        let valid_count = t.valid.iter().filter(|&&v| v).count() as u32;
        if valid_count < params.min_area.max(1) {
            continue;
        }
        let t_tile = std::time::Instant::now();
        // ONE DINO forward per tile, shared by both detectors below — the old
        // code called this separately per path, doubling the per-tile bottleneck
        // whenever both a head and a bank were configured.
        let f = dino.features(&t.rgb)?;
        dino_ms += dino.last_ms as f64;

        if let Some(head) = head {
            let pred = head.predict(&f.feat, f.grid, f.dim);
            let prob_up = detect::upscale(&pred.defect_prob, pred.grid, pred.grid, tw, tw);
            let fam_up = fewshot::upscale_family(&pred.family, pred.grid, tw, tw);
            place_tile(&mut fs_prob, &prob_up, t.origin, tw, lwu, lhu);
            place_tile(&mut fs_fam, &fam_up, t.origin, tw, lwu, lhu);
        }
        if let (Some(bank), Some(meta)) = (bank, meta) {
            let sig = detect::extract_tile_signal(&f, bank, meta, params, &t.rgb, tw, tw, Some(&t.valid));
            place_tile(&mut pc_dino, &sig.dino_map, t.origin, tw, lwu, lhu);
            place_tile(&mut pc_a, &sig.lab_a, t.origin, tw, lwu, lhu);
            place_tile(&mut pc_b, &sig.lab_b, t.origin, tw, lwu, lhu);
            place_tile(&mut pc_res, &sig.residual, t.origin, tw, lwu, lhu);
        }
        place_tile(&mut leaf_valid, &t.valid, t.origin, tw, lwu, lhu);

        det_ms += t_tile.elapsed().as_secs_f64() * 1000.0;
        dino_tiles += 1;
    }

    // ── decide ONCE, globally ──
    let mut anomaly = vec![false; n_leaf];
    if let Some(head) = head {
        let tau_lo = cfg.head_grow.clamp(0.05, cfg.head_tau);
        let (fs_mask, fs_regions) = fewshot::decide_global(
            &fs_prob, &fs_fam, &leaf_valid, lwu, lhu,
            cfg.head_tau, tau_lo, &head.hi_fam, fewshot::HEAD_MIN_REGION_PATCHES,
            params.region_close_px as usize, params.min_area,
        );
        for i in 0..n_leaf {
            if fs_mask[i] { anomaly[i] = true; }
        }
        n_regions += fs_regions.len();
        for rg in &fs_regions {
            let crop = context_crop(&leaf_rgba, lw, lh, rg.centroid[0], rg.centroid[1], CROP_WIN);
            all_regions.push(AnomalyRegion {
                leaf: *leaf_idx,
                bbox_leaf: rg.bbox,
                mask: rg.mask.clone(),
                descriptor: [0.0; 8],
                family: rg.family,
                crop,
                crop_size: CROP_WIN,
            });
        }

        if let (Some(_), Some(meta)) = (bank, meta) {
            // PatchCore runs alongside as an open-set safety net: only keep what
            // it flags that the few-shot head did NOT, so a known defect never
            // gets double-counted as two overlapping regions — its job here is
            // specifically to catch what the closed-set classifier misses.
            let pc = detect::decide_global(&pc_dino, &pc_a, &pc_b, &pc_res, &leaf_valid, lwu, lhu, meta, params);
            let mut novel_mask = vec![false; n_leaf];
            for i in 0..n_leaf {
                novel_mask[i] = pc.mask[i] && !fs_mask[i];
            }
            let novel_mask = detect::morph_close(&novel_mask, lwu, lhu, params.region_close_px as usize);
            let novel_regions = detect::extract_regions(&novel_mask, lwu, lhu, params.min_area);
            let dino_thr = meta.ch_threshold("dino") * NOVEL_CONFIDENCE_MULT;
            for rg in &novel_regions {
                // Confidence gate: mean z_dino over the region must clear a much
                // higher bar than PatchCore's normal operating threshold — see
                // NOVEL_CONFIDENCE_MULT's doc comment for why a flat pass-through
                // would just be PatchCore's own noise floor.
                let [bx, by, bw, bh] = rg.bbox;
                let (mut sum, mut cnt) = (0f32, 0u32);
                for ly in 0..bh {
                    for lx in 0..bw {
                        if rg.mask[(ly * bw + lx) as usize] {
                            sum += pc.z_map[(by + ly) as usize * lwu + (bx + lx) as usize];
                            cnt += 1;
                        }
                    }
                }
                if cnt == 0 || (sum / cnt as f32) < dino_thr {
                    continue;
                }
                let crop = context_crop(&leaf_rgba, lw, lh, rg.centroid[0], rg.centroid[1], CROP_WIN);
                all_regions.push(AnomalyRegion {
                    leaf: *leaf_idx,
                    bbox_leaf: rg.bbox,
                    mask: rg.mask.clone(),
                    descriptor: [0.0; 8],
                    family: NOVEL_FAMILY,
                    crop,
                    crop_size: CROP_WIN,
                });
                n_regions += 1;
                for ly in 0..bh {
                    for lx in 0..bw {
                        if rg.mask[(ly * bw + lx) as usize] {
                            anomaly[(by + ly) as usize * lwu + (bx + lx) as usize] = true;
                        }
                    }
                }
            }
        }
    } else if let (Some(_), Some(meta)) = (bank, meta) {
        // PatchCore-only (no head configured at all) — same as the pre-refactor
        // behaviour, just decided globally instead of per-tile.
        let r = detect::decide_global(&pc_dino, &pc_a, &pc_b, &pc_res, &leaf_valid, lwu, lhu, meta, params);
        for i in 0..n_leaf {
            if r.mask[i] { anomaly[i] = true; }
        }
        n_regions += r.regions.len();
        for rg in &r.regions {
            let crop = context_crop(&leaf_rgba, lw, lh, rg.centroid[0], rg.centroid[1], CROP_WIN);
            all_regions.push(AnomalyRegion {
                leaf: *leaf_idx,
                bbox_leaf: rg.bbox,
                mask: rg.mask.clone(),
                descriptor: rg.descriptor,
                family: 0,
                crop,
                crop_size: CROP_WIN,
            });
        }
    }

    let t_rec = std::time::Instant::now();
    let (recon_area, recon_whole, recon_mask, recon_hole) = match recon {
        Some(g) => {
            stage(tx, format!("Reconstruct {fname}"));
            reconstruct_area(g, device, &leaf_rgba, lw, lh, RECON_SIZE)
        }
        None => (0, 0, Vec::new(), Vec::new()),
    };
    let rec_ms = t_rec.elapsed().as_secs_f64() * 1000.0;
    // Reconstruction-flagged holes: pixels the cutout treats as opaque leaf but the
    // recon model believes are background — catches large, background-colored gaps
    // (e.g. holes) the DINO/color tile detector can miss entirely, independent of it.
    if !recon_hole.is_empty() {
        let hole_up = upscale_bool_sq(&recon_hole, RECON_SIZE, lwu, lhu);
        let hole_up = detect::morph_close(&hole_up, lwu, lhu, 1);
        let hole_regions = detect::extract_regions(&hole_up, lwu, lhu, HOLE_MIN_AREA);
        for rg in &hole_regions {
            let [rx, ry, rw, rh] = rg.bbox;
            let crop = context_crop(&leaf_rgba, lw, lh, rg.centroid[0], rg.centroid[1], CROP_WIN);
            all_regions.push(AnomalyRegion {
                leaf: *leaf_idx,
                bbox_leaf: [rx, ry, rw, rh],
                mask: rg.mask.clone(),
                descriptor: [0.0; 8],
                family: HOLE_FAMILY,
                crop,
                crop_size: CROP_WIN,
            });
        }
        n_regions += hole_regions.len();
        for i in 0..(lwu * lhu) {
            if hole_up[i] { anomaly[i] = true; }
        }
    }
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

/// Returns `(added, whole, mask, hole)` for a leaf cutout: `added` = reconstructed
/// lost tissue (predicted leaf where the cutout is missing), `whole` = the whole
/// intact leaf (predicted ∪ visible) — both in leaf-resolution px — `mask` = the
/// predicted intact-leaf mask (`sz²`) for the canvas preview, and `hole` = the RAW
/// (pre-hole-filling) disagreement mask (`sz²`): pixels the cutout treats as opaque
/// leaf but the model's raw prediction says are NOT leaf (background-colored gaps).
/// Runs UNetSimple at `sz`×`sz`.
fn reconstruct_area(
    gen:    &UNetSimple<InferBackend>,
    device: &<InferBackend as Backend>::Device,
    rgba:   &[u8],
    w:      u32,
    h:      u32,
    sz:     usize,
) -> (usize, usize, Vec<bool>, Vec<bool>) {
    let Some(img) = image::RgbaImage::from_raw(w, h, rgba.to_vec()) else { return (0, 0, Vec::new(), Vec::new()) };
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

    // Pre-damage nudge: applied to a COPY fed to the model only — `r` (and the
    // `visible` derived from it below) stays the TRUE observed signal, used
    // for hole detection and the area stat. See PRE_DAMAGE_PCT's doc comment.
    let model_input_rgba: Vec<u8> = if PRE_DAMAGE_PCT > 0.0 {
        let mut mask: Vec<bool> = r.chunks(4).map(|p| p[3] > 128).collect();
        let fraction = PRE_DAMAGE_PCT / 100.0;
        let mut rng = SmallRng::from_entropy();
        erode_margin_clusters(&mut mask, sz, sz, fraction, &mut rng);
        let mut out = r.clone();
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
        r.clone()
    };

    let build_4ch = |rgba: &[u8]| -> Vec<f32> {
        let mut data = Vec::with_capacity(4 * n);
        for ch in 0..4usize {
            for i in 0..n {
                data.push(rgba[i * 4 + ch] as f32 / 127.5 - 1.0);
            }
        }
        data
    };

    // TTA: average predictions over 0°/90°/180°/270° CW rotations — mirrors
    // Recon Infer's tta_enabled, ported here alongside the pre-damage nudge
    // and the RECON_SIZE fix (2026-07-10).
    let mut pred = vec![0.0f32; n];
    for k in 0u8..4 {
        let rotated = rotate_rgba_cw_k(&model_input_rgba, sz, k);
        let input_data = build_4ch(&rotated);
        let input_t: Tensor<InferBackend, 4> =
            Tensor::from_data(TensorData::new(input_data, [1usize, 4, sz, sz]), device);
        let raw: Vec<f32> = gen.forward_probs(input_t, 0, 0).into_data().to_vec().unwrap_or_default();
        if raw.len() != n {
            return (0, 0, Vec::new(), Vec::new());
        }
        let unrotated = rotate_prob_cw_k(&raw, sz, (4 - k) % 4);
        for (s, v) in pred.iter_mut().zip(unrotated.iter()) { *s += v; }
    }
    for p in pred.iter_mut() { *p /= 4.0; }

    let visible: Vec<bool> = (0..n).map(|i| r[i * 4 + 3] > 128).collect();
    // Hole signal — TWO independent cases, both defined by the same test
    // (topologically enclosed by leaf, i.e. NOT reachable from the image border
    // through the same kind of pixel), applied to two different source signals:
    //
    // (A) OPAQUE but the model's raw prediction doesn't believe it's leaf —
    //     a background-colored gap that the cutout's alpha channel still marks
    //     visible. The enclosure check is essential here regardless: the model
    //     is legitimately less confident along ANY boundary, including ordinary
    //     serrated/lobed leaf margins, so a flat low-confidence threshold alone
    //     would flag the whole natural edge as "holes" — a genuine margin,
    //     however jagged, is always open to the true background at the image
    //     border, but a real interior hole is not.
    // (B) TRANSPARENT (alpha=0) but enclosed by the visible silhouette itself —
    //     a genuinely punched-through gap, independent of what the model
    //     predicts there. Pure alpha topology, no model dependency: a pixel here
    //     reads as pitch black against the app's canvas rather than a color, and
    //     the tile detector already skips transparent pixels via its valid mask,
    //     so nothing else in the pipeline can ever catch this case.
    // Both reuse the same border-seeded flood fill `refine_silhouette` uses for
    // its own hole-filling.
    let pred_bin: Vec<bool> = (0..n).map(|i| pred[i] >= RECON_THRESHOLD).collect();
    // Close a small radius on each mask before flood-filling for enclosure: a single-
    // pixel crack connecting a hole to the image border (plausible near a leaf's own
    // jagged margin, and increasingly likely the BIGGER the hole is, since it has more
    // boundary length) would otherwise defeat enclosure for the ENTIRE hole, not just
    // the crack pixel. Membership below still tests the ORIGINAL (unclosed) masks, so
    // this only bridges thin cracks — it doesn't grow what counts as "hole."
    let pred_bin_closed = detect::morph_close(&pred_bin, sz, sz, 2);
    let visible_closed = detect::morph_close(&visible, sz, sz, 2);
    let pred_filled = crate::tabs::recon_train::training::fill_holes(&pred_bin_closed, sz, sz);
    let visible_filled = crate::tabs::recon_train::training::fill_holes(&visible_closed, sz, sz);
    let hole: Vec<bool> = (0..n).map(|i| {
        let opaque_bg_colored = visible[i] && !pred_bin[i] && pred_filled[i];
        let punched_through = !visible[i] && visible_filled[i];
        opaque_bg_colored || punched_through
    }).collect();
    // "Shape only" cleanup: keep the silhouette connected to the visible leaf and
    // fill interior holes → one solid intact shape for the preview and hole test.
    let mask = crate::tabs::recon_train::training::refine_silhouette(&pred, &visible, sz, sz, RECON_THRESHOLD);
    // Separate, lower-confidence silhouette used ONLY for the area stat below —
    // see AREA_THRESHOLD's doc comment for why this must differ from `mask`.
    let area_mask = crate::tabs::recon_train::training::refine_silhouette(&pred, &visible, sz, sz, AREA_THRESHOLD);
    let mut added = 0usize; // refined leaf where the cutout is missing = lost tissue
    let mut whole = 0usize; // refined silhouette = whole intact leaf
    for i in 0..n {
        if area_mask[i] {
            whole += 1;
            if !visible[i] { added += 1; }
        }
    }
    let scale = (w as f64 * h as f64) / n as f64;
    (
        (added as f64 * scale).round() as usize,
        (whole as f64 * scale).round() as usize,
        mask,
        hole,
    )
}

/// Copy a `tw`×`tw` tile's data into its position within an `lw`×`lh` leaf-sized
/// canvas (the last row/col of tiles can overhang the leaf — clipped here).
/// Generic over whatever per-pixel signal is being stitched (probability,
/// family label, raw channel value, validity) — see `process_leaf`.
fn place_tile<T: Copy>(canvas: &mut [T], tile: &[T], origin: [u32; 2], tw: usize, lw: usize, lh: usize) {
    let (ox, oy) = (origin[0] as usize, origin[1] as usize);
    for ty in 0..tw {
        let gy = oy + ty;
        if gy >= lh {
            continue;
        }
        for tx in 0..tw {
            let gx = ox + tx;
            if gx >= lw {
                continue;
            }
            canvas[gy * lw + gx] = tile[ty * tw + tx];
        }
    }
}

/// Nearest-neighbour upscale of a square `g`×`g` bool grid to `out_w`×`out_h`.
fn upscale_bool_sq(grid: &[bool], g: usize, out_w: usize, out_h: usize) -> Vec<bool> {
    let mut out = vec![false; out_w * out_h];
    for oy in 0..out_h {
        let gy = (oy * g / out_h.max(1)).min(g - 1);
        for ox in 0..out_w {
            let gx = (ox * g / out_w.max(1)).min(g - 1);
            out[oy * out_w + ox] = grid[gy * g + gx];
        }
    }
    out
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
pub(crate) fn context_crop(rgba: &[u8], w: u32, h: u32, cx: f32, cy: f32, win: u32) -> Vec<u8> {
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
