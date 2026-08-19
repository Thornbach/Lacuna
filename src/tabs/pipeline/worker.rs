//! Integrated-pipeline worker (Slice 2): runs the staged
//!   Segment (YOLO/ort) -> Tile -> Detect (DINO/ort + bank kNN) -> Restitch
//! flow on a background thread, streaming results over an mpsc channel (mirrors
//! the recon_infer threading idiom). Reconstruction + clustering are added in
//! later slices. Models are created INSIDE the worker thread (ort sessions and
//! burn GPU tensors are not freely shareable across threads).

use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
    sync::atomic::{AtomicBool, Ordering},
};

use burn::tensor::{backend::Backend, Tensor, TensorData};
use leaf_complex_rust_lib::config::Config as MorphConfig;
use leaf_complex_rust_lib::{analyze_rgba, MorphMetrics};
use rand::{rngs::SmallRng, SeedableRng};
use rayon::prelude::*;

use crate::tabs::eroder::algorithm::erode_margin_clusters;
use crate::tabs::leaf_seg::inference::{self as seg, SegConfig};
use crate::tabs::recon_infer::inference::{rotate_rgba_cw_k, rotate_prob_cw_k};
use crate::tabs::recon_simple::model::{load_simple_infer, UNetSimple};
use crate::tabs::recon_train::model::{create_infer_device, InferBackend};
use crate::settings::{ClusterAlgo, CutMode};

use super::bank::{CoresetBank, GpuBank};
use super::cluster;
use super::detect::{self, DetectParams};
use super::dino::{DinoExtractor, DinoFeatures};
use super::fewshot::{self, FewShotHead};
use super::meta::DetectorMeta;
use super::projection;
use super::tiling::{crop_to_rgb_meanfill, tile_leaf};

// Context-crop size for the anomaly gallery + curation PNGs.
//
// Raised 64 -> 128 on measured evidence, not intuition. A curated crop is
// re-featurized by resizing it to DINO's 512px input, so the crop size sets how
// many REAL pixels each DINO patch describes: 64px -> 2x2 real px/patch, vs the
// 8x8 an inference tile gets. Training rows and inference patches therefore
// described different physical scales — a train/deploy mismatch that only the
// curation path suffers (dense training infers on the same tiles it trains on).
//
// Measured on held-out ground-truth leaves (1Help/eval/flywheel_fair.py, curated
// crops for BOTH classes so extraction scale carries no label information):
//     win  px/patch   IoU@0.85   recall@0.85
//      64    2x2        0.204       0.539
//     128    4x4        0.275       0.465     <- best IoU, +35%
//     256    8x8        0.237       0.339
// 256 matches inference scale exactly yet does WORSE: at a fixed patch budget a
// larger window spends most of its patches on context rather than the region, so
// 128 is the empirical optimum rather than the theoretically "matched" 256.
//
// NOTE: `crop_size` is persisted per region, so curations saved before this
// change keep their own 64px geometry and stay loadable — but a curations folder
// spanning the change trains on MIXED scales, which is the very inconsistency
// this raise exists to remove. Prefer re-mining/re-curating over mixing.
pub(crate) const CROP_WIN: u32 = 128;
// Region-adaptive embedding crop (unsupervised_families clustering only, NOT
// the gallery/curation crop above, which stays fixed at CROP_WIN): a big
// region only showing its centroid neighborhood in a fixed small window loses
// most of its own extent. EMBED_PAD is the margin added around the region's
// own bbox; MAX_EMBED_CROP caps the worst case so a huge region doesn't
// build a huge crop.
//
// MIN_EMBED_CROP is deliberately its OWN constant rather than reusing CROP_WIN
// (which it used to clamp to): the two knobs answer different questions —
// CROP_WIN is a supervised-training concern, this one only shapes unsupervised
// clustering — and tying them meant tuning the curation crop silently moved
// every clustering embedding too. Keeps its historical value so the CROP_WIN
// raise above is a no-op for clustering.
pub(crate) const EMBED_PAD: u32 = 16;
pub(crate) const MIN_EMBED_CROP: u32 = 64;
pub(crate) const MAX_EMBED_CROP: u32 = 256;
/// Alpha at or below which a leaf pixel counts as transparent when hunting for
/// interior holes. Matches the threshold `tile_leaf` already uses to decide
/// what is valid leaf tissue, so a pixel can never be simultaneously "valid
/// enough to detect on" and "transparent enough to be a hole".
const HOLE_ALPHA_THR: u8 = 10;
// PCA target dimensionality for the DINO-embedding unsupervised clustering path
// (worker.rs's final clustering block). Fixed rather than user-tunable: a 3rd
// interacting DBSCAN knob (on top of cluster_eps/cluster_min_pts) would compound
// tuning difficulty more than it'd help. Bumped 16->32 to retain more
// separating structure — NOTE: this broke the original rationale that 16 kept
// the standardized feature space's eps scale close to the legacy 8-D
// descriptor's; there's no such match at 32-D. `suggest_eps_var`'s logged
// elbow suggestion is the actual mitigant for picking eps now, not a matched
// constant.
pub(crate) const DINO_CLUSTER_PCA_DIM: usize = 32;
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
// Used for the visual reconstruction preview — deliberately conservative so
// natural margins/sinuses never get painted into the preview overlay.
const RECON_THRESHOLD: f32 = 0.65;
// Separate, lower threshold used ONLY for the scalar lost-tissue-%/area stat
// (recon_area/recon_whole below), NOT the preview. RECON_THRESHOLD's
// conservatism is correct for those two, but it introduces a systematic ~4%
// UNDER-estimate in the area stat specifically (measured via `--recon-validate`
// on checkpoint `RECONTRAIN/checkpoint_best`, 2026-07: lost-tissue-% bias -4.34%
// at τ=0.65 vs +0.12% at τ=0.28 — the bias-optimal operating point found by
// sweeping). If the checkpoint is retrained, re-run `--recon-validate` and update
// this constant to the new bias-optimal τ from its report.
const AREA_THRESHOLD: f32 = 0.28;
// Pre-damage nudge applied to the model's input ONLY (never to the true
// observed alpha used for the area stat). Real herbivory
// doesn't always visually resemble the synthetic erosion patterns the model
// trained on, so a small additional synthetic erosion nudges the input back
// inside the training distribution, making the model reliably trigger its
// reconstruction pathway. Mirrors Recon Infer's `pre_damage_pct` (same
// technique, same default 1%, ported here 2026-07-10 alongside the
// RECON_SIZE fix and TTA below).
const PRE_DAMAGE_PCT: f32 = 1.0;
// Reserved family/cluster id for PatchCore-only findings: pixels the open-set
// kNN-vs-healthy-bank detector flags that the few-shot head's closed-set
// classifier did NOT — i.e. something that doesn't match any known trained
// defect family. Bypasses DBSCAN, since there's no meaningful descriptor to
// cluster on relative to the few-shot family ids.
pub(crate) const NOVEL_FAMILY: i32 = 9998;
// PatchCore's own false-positive rate was never put through the hard-negative-
// mining campaign that got the few-shot head's FP-region rate down (RESULTS.md:
// baseline 66% -> shipped few-shot 41%). A flat pass-through of "PatchCore
// disagrees with the head" mostly surfaces PatchCore's own uncalibrated
// misfires (veins, margins, glare), not genuine unseen anomalies — so the
// safety-net role needs a stricter bar than PatchCore's normal primary-
// detector operating point before something is worth surfacing as "novel".
const NOVEL_CONFIDENCE_MULT: f32 = 1.5;

/// DINO input resolution for detection tiles. **512 on every backend.**
///
/// A CPU-only 256 default was tried and REVERTED (2026-08-18). It is ~5x cheaper
/// per tile, but 256 with the default 256 px tile means each patch describes 16
/// native pixels instead of 8, and the head was fit at 8. Measured over three
/// leaves, same binary and segmentation, tau held constant: 834 regions at 512
/// against 307 at 256. Field verdict on the same scale change via tile size was
/// blunter — "accuracy is super shitty" at 16 px/patch, "nearly perfect" at 8.
///
/// It turned out not to be a trade worth making, because DINO was never the real
/// cost on CPU once the backend was fixed: with the ort CPU EP a 512 tile is
/// ~500 ms and a whole leaf is a few seconds. The 30 s/leaf that prompted all of
/// this was PatchCore's bank kNN (~236 G MAC/tile on ndarray), which is opt-in,
/// off by default, and measured to change nothing on this data.
///
/// Resolution is still selectable — by choosing the MODEL, not this constant.
/// `1Help/export_dinov3.py` exports with `dynamic_axes=None`, so each .onnx has
/// its resolution baked in and `DinoExtractor::load` adopts whatever the loaded
/// graph declares. `models/dino.onnx` is 512; `models/cpu256/dino.onnx` is 256
/// for anyone who wants the speed and accepts the cost.
///
/// `LACUNA_DINO_RES` still overrides, which only affects the BURN path — ort
/// takes its shape from the graph regardless.
pub fn default_dino_res() -> u32 {
    if let Ok(v) = std::env::var("LACUNA_DINO_RES") {
        if let Ok(r) = v.trim().parse::<u32>() {
            // Must be a multiple of the ViT patch size or the grid maths breaks.
            if r >= 64 && r % 16 == 0 {
                return r;
            }
            eprintln!("[dino] ignoring LACUNA_DINO_RES={v} (need a multiple of 16, >= 64)");
        }
    }
    512
}

pub struct PipeConfig {
    pub image_paths: Vec<PathBuf>,
    pub output_dir:  PathBuf,
    pub yolo_model:  PathBuf,
    pub dino_model:  PathBuf,
    pub bank_path:   PathBuf,
    pub meta_path:   PathBuf,
    pub tile_size:    u32,
    pub margin_erode: u32,
    /// Flag interior holes (transparent regions enclosed by leaf) as defects on
    /// geometry alone — see the hole block in `process_leaf`. Off means a hole
    /// the segmenter cut out cleanly is invisible to every detector, because
    /// transparent pixels are excluded from tiling before detection runs.
    pub detect_holes:  bool,
    /// Drop head-detected regions of the "Holes" family that never get far
    /// enough inside the leaf to be a hole. A hole is *enclosed* tissue loss;
    /// the head only sees appearance, and the leaf margin looks identical to a
    /// hole rim once `tile_leaf` mean-fills the transparent side — so the head
    /// fires along the outline and produces false positives there. Geometry is
    /// the only thing that separates the two (see `exterior_distance`).
    /// Off by default: it changes what gets reported, and that should be a
    /// deliberate choice, not a silent one.
    pub filter_margin_holes: bool,
    /// How deep inside the leaf a "Holes" region must reach (px, distance to the
    /// exterior) to count. Measured as the region's MAXIMUM depth, not its mean:
    /// a genuine hole near the edge still has a core well inside, while a margin
    /// artifact hugs the outline and never gets deep. Mean would punish real
    /// holes for having a rim.
    pub hole_margin_px:      u32,
    /// Minimum hole area in leaf pixels; below this a transparent blob is
    /// cutout anti-aliasing speckle rather than real damage. Mirrors
    /// `1Help/config.py::min_hole_area`.
    pub min_hole_area: u32,
    pub dino_res:     u32,
    pub conf:         f32,
    pub recon_ckpt:   Option<PathBuf>, // folder with gen.mpk; None = skip reconstruction
    pub head_path:    Option<PathBuf>, // few-shot head json; runs alongside the bank if both resolve
    pub use_patchcore: bool,          // ALSO run the bank when a head is present (opt-in safety net)
    // Let DBSCAN over the 8-D descriptor assign family/cluster labels instead of the
    // head's own closed-set argmax (which has zero reject/novelty option — every
    // detected anomaly gets forced into one of its trained classes regardless of
    // fit). Opt-in; requires bank+meta to also be loaded (the descriptor leans on
    // PatchCore's z-score channels). No effect when no head is configured at all —
    // that case already clusters unsupervised today.
    pub unsupervised_families: bool,
    // Train a small domain-adapted projection (projection.rs) from this run's
    // OWN curations, fresh in-memory each run, and use it in place of the raw
    // DINO embedding for clustering. Opt-in; falls back to raw embeddings
    // (logged) if there isn't enough curated data yet.
    pub domain_projection: bool,
    pub head_tau:     f32,             // few-shot decision threshold (hysteresis SEED)
    pub head_grow:    f32,             // hysteresis GROW threshold (higher = tighter regions)
    pub seg_alpha_lo:   f32,           // YOLO cutout edge tightness (feather start)
    pub seg_chroma_min: i32,           // YOLO cutout background-chroma rejection
    pub cluster_eps:     f32,          // DBSCAN radius; lower = more/smaller/looser clusters
    pub cluster_min_pts: usize,        // DBSCAN min points; lower = more/smaller/looser clusters
    pub cluster_algo:    ClusterAlgo,  // Dbscan (default) or Hierarchical
    pub target_k:        usize,        // FixedK only; 0 = auto via suggest_k
    pub cut_mode:        CutMode,      // Hierarchical only: FixedK (default) or Adaptive
    pub adaptive_threshold: f32,       // Adaptive only: inconsistency sensitivity
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
    pub descriptor: [f32; 8],   // 8-D clustering feature; zeros for the Novel sentinel, and for
                                 // few-shot regions UNLESS unsupervised_families scored them (see
                                 // process_leaf)
    pub family:     i32,        // head-assigned family ≥1 (few-shot); 0 (PatchCore-only); or the
                                 // reserved NOVEL_FAMILY sentinel
    pub crop:       Vec<u8>,    // RGBA context-crop thumbnail, crop_size²·4
    pub crop_size:  u32,
    pub dino_embed: Vec<f32>,   // mask-aware, adaptive-crop, L2-normalized DINO embedding;
                                 // empty unless unsupervised_families scored it (see process_leaf)
    pub dino_embed_whole: Vec<f32>, // whole-crop-pooled (NOT mask-aware) L2-normalized companion,
                                 // same adaptive crop/forward — feeds the domain_projection head,
                                 // which can only be TRAINED the way historical curation PNGs allow
                                 // (whole-crop, no persisted mask); empty unless dino_embed is.
}

/// Everything needed to cheaply re-cut a Hierarchical run at a different K
/// without rerunning the O(n³) merge search — `cluster::labels_for_k` replays
/// a prefix of `merges` in O(n). `merges` indices are positions in `real_idx`
/// (every region's index, in order, at cluster time), NOT `AnomalyRegion`
/// indices in general — `real_idx[i]` maps a merge-sequence position back to
/// its actual region. `None` on a `PipeMsg::Clusters` where Dbscan (or the
/// closed-set argmax path) ran instead.
pub struct HierarchicalClusterState {
    pub merges:          Vec<(usize, usize, f32)>,
    pub real_idx:         Vec<usize>,
    pub min_cluster_size: usize,
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
        labels:   Vec<i32>,
        coords:   Vec<[f32; 2]>,
        names:    std::collections::HashMap<i32, String>,
        regions:  Vec<AnomalyRegion>,
        hcluster: Option<HierarchicalClusterState>,
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
    // What the extractor SETTLED on, which is not necessarily what we asked for:
    // a fixed-shape ort graph overrides the request (see DinoExtractor::load).
    // Everything downstream — the warmup log, the head-alignment check — has to
    // report this value, or it describes a run that did not happen.
    let dino_res = dino.res();
    // DINO GPU check: two warmup forwards on a dummy tile. The WARM time makes it
    // unambiguous whether DINO is actually on the GPU (~20ms) or silently on CPU
    // (~600ms) — regardless of what "CUDA loadable" says.
    {
        let dummy = image::RgbImage::new(256, 256);
        let _ = dino.features(&dummy);            // cold (CUDA/cuDNN autotune)
        let _ = dino.features(&dummy);            // warm
        // Report the backend this binary was BUILT for rather than guessing it
        // from wall time. The old heuristic ("under 150 ms must be a GPU") was
        // sound when a CPU forward meant ~4000 ms, but a CPU build running ort
        // at res 256 warms up in ~134 ms — so it confidently reported "GPU" on a
        // machine with no GPU involved. It also told a CPU user their run was
        // broken ("cuDNN not engaged"), which on the CPU variant is just untrue.
        // ASCII only: the bundled fonts have no U+2713 and render it as tofu.
        #[cfg(any(feature = "cuda", feature = "wgpu-gpu"))]
        let verdict = if dino.last_ms < 150.0 {
            "GPU engaged".to_string()
        } else {
            "SLOW - DINO is not on the GPU".to_string()
        };
        #[cfg(not(any(feature = "cuda", feature = "wgpu-gpu")))]
        let verdict = "CPU build - this is the expected path".to_string();
        log(tx, format!(
            "DINO warmup {:.0}ms/forward at {}px - {}",
            dino.last_ms, dino_res, verdict,
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
            if h.infer_resolution != dino_res {
                log(tx, format!(
                    "! head trained at {}px but DINO runs at {}px - detections may be sparser",
                    h.infer_resolution, dino_res,
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
        names.insert(NOVEL_FAMILY, "Novel (PatchCore)".to_string());
        let is_sentinel = |f: i32| f == NOVEL_FAMILY;
        let n_novel = all_regions.iter().filter(|r| r.family == NOVEL_FAMILY).count();
        // unsupervised_families lets DBSCAN assign the family/cluster label instead
        // of the head's closed-set argmax, even when a head IS configured — the
        // head's own defect-vs-healthy signal (used upstream for detection) is
        // untouched either way, only the FAMILY label's source changes. Requires
        // bank+meta to have actually loaded (the 8-D descriptor leans on PatchCore's
        // z-score channels); if they didn't, silently fall back to the head's own
        // families rather than clustering on all-zero descriptors.
        let unsupervised_ok = cfg.unsupervised_families && bank.is_some() && meta.is_some();
        let use_argmax = head.is_some() && !unsupervised_ok;
        // dino_embed is only ever populated inside process_leaf's `if let Some(head)
        // = head` branch — a head-absent PatchCore-only run can have unsupervised_ok
        // true (bank/meta always load when there's no head) while every region's
        // dino_embed is still empty. Must gate on head.is_some() too, or that run
        // would try to cluster on all-empty embedding vectors.
        let use_dino = unsupervised_ok && head.is_some();
        let (labels, coords, hcluster) = if use_argmax {
            let head = head.as_ref().expect("use_argmax implies head.is_some()");
            // Few-shot: the head already assigns each region a family. v1 clusters
            // ARE those families (labels = head family), with a per-family jittered
            // scatter so the plot reads as separated clouds. (Open-set discovery on
            // the dense embedding is a follow-up; this is the validated path.)
            // Novel regions already carry their sentinel family, so they flow
            // through unchanged and land in their own scatter cloud.
            let labels: Vec<i32> = all_regions.iter().map(|r| r.family).collect();
            let coords = family_scatter(&labels);
            for &l in &labels {
                if l >= 0 && !is_sentinel(l) {
                    names.entry(l).or_insert_with(|| head.family_name(l));
                }
            }
            log(tx, format!(
                "{} regions -> {} families (few-shot, {n_novel} novel/PatchCore)",
                all_regions.len(), names.len(),
            ));
            (labels, coords, None)
        } else {
            // Unsupervised: cluster the 8-D descriptors (StandardScaler + DBSCAN +
            // PCA-2) over every region. Novel (PatchCore) regions ARE folded in here (only excluded when this branch
            // is reached via head-argmax-off, i.e. never — see below): forcing every
            // novel-safety-net catch into one fixed "Novel (PatchCore)" bucket is the
            // same kind of forced-category problem this toggle exists to avoid for
            // the head's 5 classes. NOVEL_FAMILY regions only ever exist when a head
            // is configured (see process_leaf), and this branch only runs for a
            // configured head when unsupervised_families is actually on — so whenever
            // NOVEL_FAMILY regions are present here, folding them in is always what
            // was asked for, never a stray leftover from head-absent PatchCore-only
            // runs (which never produce NOVEL_FAMILY regions at all).
            if head.is_some() {
                log(tx, format!(
                    "unsupervised_families: clustering via {:?} instead of the head's closed-set families",
                    cfg.cluster_algo,
                ));
            }
            let real_idx: Vec<usize> = (0..all_regions.len()).collect();
            let (db_labels, db_coords, hc) = if use_dino {
                // Cluster on pooled DINO embeddings instead of the 8-D hand-crafted
                // descriptor — that descriptor is dominated by "how confidently
                // anomalous" dims (z_dino/z_res/z_color), which compress every region
                // (already past the anomaly threshold to exist at all) into one density
                // mode, swamping the dims that would separate actual defect types.
                // DINO's embedding is semantically richer (it's what already drives
                // PatchCore's own healthy-vs-anomalous distance), so it should actually
                // discriminate different visual appearances. PCA-reduce first — raw
                // 1536-D Euclidean distances are meaningless (they concentrate).
                let embeds: Vec<Vec<f32>> = real_idx.iter().map(|&i| all_regions[i].dino_embed.clone()).collect();

                // Optional domain-adapted projection, trained fresh (in-memory, no
                // persisted artifact) from this run's OWN curations if any exist —
                // see projection.rs. Falls back to the raw embeddings on any
                // failure (insufficient curated data, a training panic) rather than
                // erroring the whole run over an opt-in bonus feature.
                let (projected, used_projection): (Vec<Vec<f32>>, bool) = if cfg.domain_projection {
                    let whole_embeds: Vec<Vec<f32>> = real_idx.iter()
                        .map(|&i| all_regions[i].dino_embed_whole.clone()).collect();
                    let embed_res = embed_resolution(dino.res());
                    match projection::try_train_and_apply(&cfg.output_dir, &mut dino, embed_res, &whole_embeds, tx) {
                        Some(p) => (p, true),
                        None => (embeds.clone(), false),
                    }
                } else {
                    (embeds, false)
                };

                // Hybrid descriptor: concatenate the standardized 8-D hand-crafted
                // descriptor onto the raw DINO embedding, scaled dynamically to match
                // the embedding block's own total variance — a naive concatenation
                // would let the 8-D block dominate PCA by 1-2 orders of magnitude
                // (its total variance is exactly 8 after standardization; an
                // L2-normalized DINO block's is typically only 0.05-0.3), silently
                // undoing the whole point of adding it.
                //
                // SKIPPED when domain_projection actually trained: that projection is
                // already a purpose-built, curated-label-driven signal — concatenating
                // the SAME generic 8-D descriptor at equal weight (chosen for the raw,
                // unsupervised DINO case) dilutes exactly the direction it was trained
                // to separate with an unrelated, weaker one. Confirmed via real use:
                // even the curated training examples themselves failed to cluster
                // together with the hybrid concatenation active.
                let mut combined: Vec<Vec<f32>> = if used_projection {
                    projected
                } else {
                    let descs: Vec<[f32; 8]> = real_idx.iter().map(|&i| all_regions[i].descriptor).collect();
                    let std_desc = cluster::standardize(&descs);
                    let desc_weight = hybrid_descriptor_weight(&projected);
                    std_desc.iter().zip(&projected).map(|(d, e)| {
                        d.iter().map(|&v| v * desc_weight).chain(e.iter().copied()).collect()
                    }).collect()
                };
                // Last-line sanity check before PCA/clustering ever see
                // this data: a single non-finite component anywhere poisons
                // every downstream pairwise distance it touches, and
                // complete-linkage's own defenses can only cope with SO
                // much of that (a real production incident: a diverged
                // domain_projection model — now guarded separately at its
                // own source — produced an all-NaN feature block, and
                // clustering silently returned zero clusters instead of
                // failing loudly). Zero out just the offending region's
                // vector rather than let it corrupt the whole run — a
                // neutral, finite point that just won't cluster
                // meaningfully, instead of a landmine for every distance
                // computed against it.
                let mut n_bad = 0usize;
                for row in &mut combined {
                    if row.iter().any(|v| !v.is_finite()) {
                        n_bad += 1;
                        row.iter_mut().for_each(|v| *v = 0.0);
                    }
                }
                if n_bad > 0 {
                    log(tx, format!(
                        "unsupervised_families: {n_bad}/{} region embeddings were non-finite, zeroed before clustering",
                        combined.len(),
                    ));
                }

                let t_pca = std::time::Instant::now();
                let reduced = cluster::pca_k_var(&combined, DINO_CLUSTER_PCA_DIM);
                let std_var = cluster::standardize_var(&reduced);
                log(tx, format!(
                    "unsupervised_families: DINO-embedding clustering, {} regions -> {}-D PCA in {:.0}ms",
                    real_idx.len(), DINO_CLUSTER_PCA_DIM, t_pca.elapsed().as_secs_f64() * 1000.0,
                ));
                match cfg.cluster_algo {
                    ClusterAlgo::Dbscan => {
                        // Data-driven eps sanity check: log what the k-distance elbow
                        // heuristic would suggest alongside whatever eps is actually
                        // configured, so "everything merged into one blob" (eps too
                        // loose) or "99% noise" (eps too tight) has a concrete number
                        // to dial toward instead of blind guessing between extremes.
                        if let Some(sugg) = cluster::suggest_eps_var(&std_var, cfg.cluster_min_pts) {
                            log(tx, format!(
                                "cluster_eps: using {:.2} (min_pts={}) — k-distance elbow suggests ~{:.2}",
                                cfg.cluster_eps, cfg.cluster_min_pts, sugg,
                            ));
                        }
                        let labels = cluster::dbscan_var(&std_var, cfg.cluster_eps, cfg.cluster_min_pts);
                        let coords: Vec<[f32; 2]> = cluster::pca_k_var(&std_var, 2)
                            .iter().map(|p| [p[0], p[1]]).collect();
                        (labels, coords, None)
                    }
                    ClusterAlgo::Hierarchical => {
                        let merges = cluster::agglomerative_var(&std_var, cancel);
                        let labels = match cfg.cut_mode {
                            CutMode::FixedK => resolve_fixed_k(tx, &merges, real_idx.len(), cfg.target_k, cfg.cluster_min_pts).0,
                            CutMode::Adaptive => resolve_adaptive(tx, &merges, real_idx.len(), cfg.adaptive_threshold, cfg.cluster_min_pts),
                        };
                        let coords: Vec<[f32; 2]> = cluster::pca_k_var(&std_var, 2)
                            .iter().map(|p| [p[0], p[1]]).collect();
                        let hc = HierarchicalClusterState {
                            merges, real_idx: real_idx.clone(), min_cluster_size: cfg.cluster_min_pts,
                        };
                        (labels, coords, Some(hc))
                    }
                }
            } else {
                let descs: Vec<[f32; 8]> = real_idx.iter().map(|&i| all_regions[i].descriptor).collect();
                let std = cluster::standardize(&descs);
                match cfg.cluster_algo {
                    ClusterAlgo::Dbscan => {
                        if let Some(sugg) = cluster::suggest_eps(&std, cfg.cluster_min_pts) {
                            log(tx, format!(
                                "cluster_eps: using {:.2} (min_pts={}) — k-distance elbow suggests ~{:.2}",
                                cfg.cluster_eps, cfg.cluster_min_pts, sugg,
                            ));
                        }
                        let labels = cluster::dbscan(&std, cfg.cluster_eps, cfg.cluster_min_pts);
                        let coords = cluster::pca2(&std);
                        (labels, coords, None)
                    }
                    ClusterAlgo::Hierarchical => {
                        let merges = cluster::agglomerative(&std, cancel);
                        let labels = match cfg.cut_mode {
                            CutMode::FixedK => resolve_fixed_k(tx, &merges, real_idx.len(), cfg.target_k, cfg.cluster_min_pts).0,
                            CutMode::Adaptive => resolve_adaptive(tx, &merges, real_idx.len(), cfg.adaptive_threshold, cfg.cluster_min_pts),
                        };
                        let coords = cluster::pca2(&std);
                        let hc = HierarchicalClusterState {
                            merges, real_idx: real_idx.clone(), min_cluster_size: cfg.cluster_min_pts,
                        };
                        (labels, coords, Some(hc))
                    }
                }
            };
            // `real_idx` is every region index in order now (no more sentinel
            // exclusion), so `db_labels`/`db_coords` already line up 1:1 with
            // `all_regions` — no seed-then-overwrite indirection needed.
            let labels = db_labels;
            let coords = db_coords;
            // Count UNIQUE surviving ids, not max+1 — that assumption held for
            // plain DBSCAN (its ids are always dense from 0) but breaks for
            // labels_for_k/labels_adaptive: compact_and_fold assigns dense
            // sequential ids FIRST, then folds undersized groups to noise
            // afterward, which can leave gaps in the surviving id sequence.
            let n = labels.iter().copied().filter(|&l| l >= 0)
                .collect::<std::collections::HashSet<_>>().len();
            log(tx, format!("{} regions -> {} clusters ({n_novel} novel folded in)", all_regions.len(), n));
            (labels, coords, hc)
        };
        let _ = tx.send(PipeMsg::Clusters { labels, coords, names, regions: all_regions, hcluster });
    }

    let _ = tx.send(PipeMsg::Progress { done: total, total });
    stage(tx, "Done");
    Ok(())
}

/// Distance (in px, 4-connected) from every leaf pixel to the nearest EXTERIOR
/// pixel — the background OUTSIDE the leaf, not the transparent interior of a
/// hole. Exterior is defined by connectivity: transparent pixels reachable from
/// the image border. A hole's transparent pixels are unreachable, so they get a
/// large distance, exactly like the tissue around them.
///
/// This exists because appearance cannot separate a hole from the leaf margin.
/// `tile_leaf` fills every transparent pixel with the tile's mean valid colour
/// before DINO sees it, so a real hole and the leaf edge are both rendered as
/// "flat mean colour next to tissue" — the same picture. Eroding the alpha
/// (`margin_erode`) does not help either: it makes the eroded band transparent,
/// which mean-fills it too, relocating the edge rather than removing it. The
/// only signal that distinguishes the two is global geometry, i.e. this.
///
/// Returns `None` when the leaf has no exterior at all (it fills the frame), so
/// callers skip filtering rather than treating "no reference" as "distance 0".
fn exterior_distance(rgba: &[u8], w: usize, h: usize, alpha_thr: u8) -> Option<Vec<u32>> {
    let n = w * h;
    let transparent = |i: usize| rgba[i * 4 + 3] <= alpha_thr;

    // Flood-fill the exterior inward from the border, through transparent pixels.
    let mut dist = vec![u32::MAX; n];
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut seed = |i: usize, dist: &mut Vec<u32>, q: &mut VecDeque<usize>| {
        if transparent(i) && dist[i] != 0 {
            dist[i] = 0;
            q.push_back(i);
        }
    };
    for x in 0..w {
        seed(x, &mut dist, &mut queue);
        seed((h - 1) * w + x, &mut dist, &mut queue);
    }
    for y in 0..h {
        seed(y * w, &mut dist, &mut queue);
        seed(y * w + (w - 1), &mut dist, &mut queue);
    }
    if queue.is_empty() {
        return None;
    }
    // Pass 1: grow the exterior through transparent pixels only (stays 0).
    let mut head = 0usize;
    let mut frontier: Vec<usize> = queue.iter().copied().collect();
    while head < frontier.len() {
        let i = frontier[head];
        head += 1;
        let (x, y) = (i % w, i / w);
        let mut visit = |nx: usize, ny: usize, frontier: &mut Vec<usize>, dist: &mut Vec<u32>| {
            let j = ny * w + nx;
            if transparent(j) && dist[j] != 0 {
                dist[j] = 0;
                frontier.push(j);
            }
        };
        if x > 0 { visit(x - 1, y, &mut frontier, &mut dist); }
        if x + 1 < w { visit(x + 1, y, &mut frontier, &mut dist); }
        if y > 0 { visit(x, y - 1, &mut frontier, &mut dist); }
        if y + 1 < h { visit(x, y + 1, &mut frontier, &mut dist); }
    }
    // Pass 2: BFS outward from the exterior across EVERY pixel, giving each one
    // its distance to the outside world.
    let mut q: VecDeque<usize> = frontier.into_iter().collect();
    while let Some(i) = q.pop_front() {
        let d = dist[i];
        let (x, y) = (i % w, i / w);
        let mut step = |nx: usize, ny: usize, dist: &mut Vec<u32>, q: &mut VecDeque<usize>| {
            let j = ny * w + nx;
            if dist[j] == u32::MAX {
                dist[j] = d + 1;
                q.push_back(j);
            }
        };
        if x > 0 { step(x - 1, y, &mut dist, &mut q); }
        if x + 1 < w { step(x + 1, y, &mut dist, &mut q); }
        if y > 0 { step(x, y - 1, &mut dist, &mut q); }
        if y + 1 < h { step(x, y + 1, &mut dist, &mut q); }
    }
    Some(dist)
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
    let mut dino_embed_ms = 0f64; // unsupervised_families: per-region DINO forward time
    let mut dino_embed_n = 0usize;
    // Split out of dino_embed_ms for diagnosing where per-region cost
    // actually goes (CPU crop-building vs GPU forward-pass vs CPU pooling)
    // — added specifically because DINO batching (see features_batch)
    // barely moved per-region wall time despite running on CUDA, which is
    // surprising enough to want a real breakdown instead of another guess.
    let mut embed_crop_ms = 0f64;
    let mut embed_prep_ms = 0f64; // dino.rs's own resize+CHW-repack, was hidden in the "pool" bucket
    let mut embed_gpu_ms = 0f64;
    // Region embeddings run at a lower resolution than per-tile detection —
    // see `embed_resolution`'s doc comment.
    let embed_res = embed_resolution(dino.res());

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
        let (mut fs_mask, fs_regions) = fewshot::decide_global(
            &fs_prob, &fs_fam, &leaf_valid, lwu, lhu,
            cfg.head_tau, tau_lo, &head.hi_fam, fewshot::HEAD_MIN_REGION_PATCHES,
            params.region_close_px as usize, params.min_area,
        );

        // ── margin false positives on the "Holes" class ──
        // Filtered BEFORE the mask is OR'd into `anomaly` and before
        // `n_regions` counts them, so a dropped region leaves no trace in the
        // anomaly area, the region list, or the stats. Class is resolved by
        // NAME, matching the convention the geometric hole block below already
        // uses — rename the class and this stops applying to it.
        let fs_regions = if !cfg.filter_margin_holes {
            fs_regions
        } else {
            let holes_fam = head.families.iter()
                .find(|(_, name)| name.trim().eq_ignore_ascii_case("holes"))
                .and_then(|(idx, _)| idx.parse::<i32>().ok());
            match (holes_fam, exterior_distance(&leaf_rgba, lwu, lhu, HOLE_ALPHA_THR)) {
                (Some(fam), Some(dist)) => {
                    let mut kept = Vec::with_capacity(fs_regions.len());
                    let mut dropped = 0usize;
                    for rg in fs_regions {
                        let [bx, by, bw, bh] = rg.bbox;
                        let mut depth = 0u32;
                        if rg.family == fam {
                            for ly in 0..bh {
                                for lx in 0..bw {
                                    if !rg.mask[(ly * bw + lx) as usize] {
                                        continue;
                                    }
                                    let gi = (by + ly) as usize * lwu + (bx + lx) as usize;
                                    depth = depth.max(dist[gi]);
                                }
                            }
                        }
                        if rg.family == fam && depth < cfg.hole_margin_px {
                            // Clear its pixels so the leaf's anomaly area does not
                            // still include a region nobody will ever see listed.
                            for ly in 0..bh {
                                for lx in 0..bw {
                                    if rg.mask[(ly * bw + lx) as usize] {
                                        fs_mask[(by + ly) as usize * lwu + (bx + lx) as usize] = false;
                                    }
                                }
                            }
                            dropped += 1;
                            continue;
                        }
                        kept.push(rg);
                    }
                    if dropped > 0 {
                        log(tx, format!(
                            "leaf {}: -{dropped} margin 'Holes' region(s) (depth < {}px)",
                            *leaf_idx + 1, cfg.hole_margin_px,
                        ));
                    }
                    kept
                }
                _ => fs_regions, // no Holes class, or leaf fills the frame: nothing to do
            }
        };

        for i in 0..n_leaf {
            if fs_mask[i] { anomaly[i] = true; }
        }
        n_regions += fs_regions.len();
        let fs_start = all_regions.len();
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
                dino_embed: Vec::new(),
                dino_embed_whole: Vec::new(),
            });
        }

        if let (Some(_), Some(meta)) = (bank, meta) {
            // PatchCore runs alongside as an open-set safety net: only keep what
            // it flags that the few-shot head did NOT, so a known defect never
            // gets double-counted as two overlapping regions — its job here is
            // specifically to catch what the closed-set classifier misses.
            let pc = detect::decide_global(&pc_dino, &pc_a, &pc_b, &pc_res, &leaf_valid, lwu, lhu, meta, params);

            if cfg.unsupervised_families {
                // Score every few-shot-detected region against PatchCore's own
                // z-score channels so the final clustering pass has a real
                // descriptor to run DBSCAN on instead of the head's forced argmax
                // family. `rg.family` (already set above) becomes irrelevant for
                // these regions once DBSCAN overwrites the label downstream.
                for (rg, region) in fs_regions.iter().zip(&mut all_regions[fs_start..]) {
                    region.descriptor = detect::region_descriptor(
                        rg.bbox, &rg.mask, rg.area,
                        &pc.z_map, &pc.z_res, &pc.z_color, &pc_a, &pc_b, pc.med_a, pc.med_b, lwu,
                    );
                }
                // Also compute a DINO embedding of every region's own crop —
                // the richer feature the final clustering block actually
                // clusters on (the 8-D descriptor stays a harmless, unused
                // fallback). Batched (EMBED_BATCH regions per forward pass)
                // instead of one DINO call per region — see
                // `dino.rs::features_batch`'s doc comment for why that's the
                // dominant cost at this scale.
                let t_embed = std::time::Instant::now();
                let t_crop = std::time::Instant::now();
                let (crop_imgs, crop_wins): (Vec<_>, Vec<_>) = fs_regions.iter()
                    .map(|rg| embed_crop(&leaf_rgba, lw, lh, rg.centroid, rg.bbox))
                    .unzip();
                embed_crop_ms += t_crop.elapsed().as_secs_f64() * 1000.0;
                for start in (0..fs_regions.len()).step_by(EMBED_BATCH) {
                    let end = (start + EMBED_BATCH).min(fs_regions.len());
                    match dino.features_batch_at(&crop_imgs[start..end], embed_res) {
                        Ok(feats) => {
                            embed_gpu_ms += dino.last_ms as f64;
                            embed_prep_ms += dino.last_prep_ms as f64;
                            // Mask-aware pooling is pure per-region CPU work
                            // (~14% of embed time on real data) — compute the
                            // whole batch's results in parallel first, THEN
                            // write them back sequentially, so no two
                            // rayon tasks ever touch the same `all_regions`
                            // slot.
                            let pooled: Vec<(Vec<f32>, Vec<f32>)> = feats.par_iter().enumerate()
                                .map(|(k, f)| {
                                    let idx = start + k;
                                    let rg = &fs_regions[idx];
                                    pool_region_embed(f, crop_wins[idx], rg.centroid, rg.bbox, &rg.mask, lw, lh, cfg.domain_projection)
                                })
                                .collect();
                            for (k, (mask_aware, whole)) in pooled.into_iter().enumerate() {
                                let idx = start + k;
                                all_regions[fs_start + idx].dino_embed = mask_aware;
                                all_regions[fs_start + idx].dino_embed_whole = whole;
                            }
                        }
                        Err(e) => log(tx, format!("region DINO embed batch failed: {e}")),
                    }
                }
                dino_embed_ms += t_embed.elapsed().as_secs_f64() * 1000.0;
                dino_embed_n += fs_regions.len();
            }

            let mut novel_mask = vec![false; n_leaf];
            for i in 0..n_leaf {
                novel_mask[i] = pc.mask[i] && !fs_mask[i];
            }
            let novel_mask = detect::morph_close(&novel_mask, lwu, lhu, params.region_close_px as usize);
            let novel_regions = detect::extract_regions(&novel_mask, lwu, lhu, params.min_area);
            let dino_thr = meta.ch_threshold("dino") * NOVEL_CONFIDENCE_MULT;
            // Confidence gate FIRST (mean z_dino over the region must clear a
            // much higher bar than PatchCore's normal operating threshold —
            // see NOVEL_CONFIDENCE_MULT's doc comment for why a flat
            // pass-through would just be PatchCore's own noise floor), THEN
            // batch-embed only the survivors — no point spending a DINO
            // forward pass on a region that's about to be discarded anyway.
            let kept: Vec<usize> = (0..novel_regions.len())
                .filter(|&i| {
                    let rg = &novel_regions[i];
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
                    cnt > 0 && (sum / cnt as f32) >= dino_thr
                })
                .collect();

            let mut kept_embeds: Vec<(Vec<f32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); kept.len()];
            if cfg.unsupervised_families {
                let t_embed = std::time::Instant::now();
                let t_crop = std::time::Instant::now();
                let (crop_imgs, crop_wins): (Vec<_>, Vec<_>) = kept.iter()
                    .map(|&i| embed_crop(&leaf_rgba, lw, lh, novel_regions[i].centroid, novel_regions[i].bbox))
                    .unzip();
                embed_crop_ms += t_crop.elapsed().as_secs_f64() * 1000.0;
                for start in (0..kept.len()).step_by(EMBED_BATCH) {
                    let end = (start + EMBED_BATCH).min(kept.len());
                    match dino.features_batch_at(&crop_imgs[start..end], embed_res) {
                        Ok(feats) => {
                            embed_gpu_ms += dino.last_ms as f64;
                            embed_prep_ms += dino.last_prep_ms as f64;
                            let pooled: Vec<(Vec<f32>, Vec<f32>)> = feats.par_iter().enumerate()
                                .map(|(k, f)| {
                                    let idx = start + k;
                                    let rg = &novel_regions[kept[idx]];
                                    pool_region_embed(f, crop_wins[idx], rg.centroid, rg.bbox, &rg.mask, lw, lh, cfg.domain_projection)
                                })
                                .collect();
                            for (k, pair) in pooled.into_iter().enumerate() {
                                kept_embeds[start + k] = pair;
                            }
                        }
                        Err(e) => log(tx, format!("region DINO embed batch failed: {e}")),
                    }
                }
                dino_embed_ms += t_embed.elapsed().as_secs_f64() * 1000.0;
                dino_embed_n += kept.len();
            }

            // Score against the SAME PatchCore channels as the few-shot regions
            // above, so a novel-safety-net catch can be folded into unsupervised
            // clustering too instead of being force-bucketed as one fixed "Novel
            // (PatchCore)" catch-all — that catch-all is exactly the same kind of
            // forced-into-a-fixed-category problem this whole toggle exists to
            // avoid for the head's 5 classes.
            for (&i, (mask_aware, whole)) in kept.iter().zip(kept_embeds.into_iter()) {
                let rg = &novel_regions[i];
                let [bx, by, bw, bh] = rg.bbox;
                let crop = context_crop(&leaf_rgba, lw, lh, rg.centroid[0], rg.centroid[1], CROP_WIN);
                let (descriptor, dino_embed, dino_embed_whole) = if cfg.unsupervised_families {
                    let descriptor = detect::region_descriptor(
                        rg.bbox, &rg.mask, rg.area,
                        &pc.z_map, &pc.z_res, &pc.z_color, &pc_a, &pc_b, pc.med_a, pc.med_b, lwu,
                    );
                    (descriptor, mask_aware, whole)
                } else {
                    ([0.0; 8], Vec::new(), Vec::new())
                };
                all_regions.push(AnomalyRegion {
                    leaf: *leaf_idx,
                    bbox_leaf: rg.bbox,
                    mask: rg.mask.clone(),
                    descriptor,
                    family: NOVEL_FAMILY,
                    crop,
                    crop_size: CROP_WIN,
                    dino_embed,
                    dino_embed_whole,
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
                dino_embed: Vec::new(), // head-absent path never computes region embeddings
                dino_embed_whole: Vec::new(),
            });
        }
    }

    // ── interior holes: geometry, not appearance ──
    // A hole eaten clean through the leaf is TRANSPARENT after segmentation, so
    // `tile_leaf` marks it invalid and every detector above skips it — the model
    // never sees the one defect that is trivially identifiable. Only holes the
    // segmenter FAILED to cut out (still opaque, showing background) get
    // detected, which perversely means better segmentation finds fewer holes.
    //
    // Ported from `1Help/preprocessing.py::interior_holes`, whose own docstring
    // records the same failure ("the border\hole failures"): transparent
    // components NOT connected to the border are holes through the leaf, and are
    // flagged directly as defects rather than treated as missing data. Run on
    // the ORIGINAL alpha, not the `margin_erode`-eroded copy used for detection,
    // which would inflate every hole by the erosion radius.
    if cfg.detect_holes {
        let holes_class = head.and_then(|h| {
            h.families.iter()
                .find(|(_, name)| name.trim().eq_ignore_ascii_case("holes"))
                .and_then(|(idx, _)| idx.parse::<i32>().ok())
        });
        let transparent: Vec<bool> = (0..n_leaf)
            .map(|i| leaf_rgba[i * 4 + 3] <= HOLE_ALPHA_THR)
            .collect();
        // Every transparent blob, then discard the ones touching the image
        // border — those are the exterior background around the cutout.
        let blobs = detect::extract_regions(&transparent, lwu, lhu, cfg.min_hole_area);
        let mut n_holes = 0usize;
        for rg in &blobs {
            let [bx, by, bw, bh] = rg.bbox;
            if bx == 0 || by == 0 || bx + bw >= lw || by + bh >= lh {
                continue; // exterior background, not a hole through the leaf
            }
            // How much of this hole did an appearance detector already claim?
            //
            // This used to skip the hole entirely on ANY overlap, which was far
            // too strict: a small hole's discoloured RIM is usually flagged on
            // appearance, that rim touches the transparent blob, and the whole
            // hole was then discarded — leaving only the margin drawn and the
            // hole itself still invisible. Exactly the reported symptom, and it
            // spared big holes only because their rims happened not to fire.
            //
            // The mask is now always filled in (a hole IS damage, whoever found
            // it), and only the REGION record is suppressed when the hole is
            // mostly claimed already — that is the case where a second region
            // would double-count the same damage downstream.
            let mut n_px = 0usize;
            let mut n_already = 0usize;
            for ly in 0..bh {
                for lx in 0..bw {
                    if !rg.mask[(ly * bw + lx) as usize] {
                        continue;
                    }
                    n_px += 1;
                    let gi = (by + ly) as usize * lwu + (bx + lx) as usize;
                    if anomaly[gi] {
                        n_already += 1;
                    }
                    anomaly[gi] = true;
                }
            }
            // >50% already covered: an existing region is describing this hole,
            // so adding another would report the same damage twice.
            if n_px == 0 || n_already * 2 > n_px {
                continue;
            }
            let crop = context_crop(&leaf_rgba, lw, lh, rg.centroid[0], rg.centroid[1], CROP_WIN);
            all_regions.push(AnomalyRegion {
                leaf: *leaf_idx,
                bbox_leaf: rg.bbox,
                mask: rg.mask.clone(),
                descriptor: [0.0; 8],
                // No head, or no class named "Holes"? Fall back to the novel
                // bucket rather than guessing an id — a hole is certainly a
                // defect, but mislabelling it as some other family would
                // silently corrupt that family's statistics.
                family: holes_class.unwrap_or(NOVEL_FAMILY),
                crop,
                crop_size: CROP_WIN,
                dino_embed: Vec::new(),
                dino_embed_whole: Vec::new(),
            });
            n_holes += 1;
            n_regions += 1;
        }
        if n_holes > 0 {
            log(tx, format!(
                "leaf {}: +{n_holes} interior hole(s) flagged geometrically{}",
                *leaf_idx + 1,
                if holes_class.is_none() { " (no 'Holes' class in head — filed as novel)" } else { "" }
            ));
        }
    }

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
    let embed_note = if dino_embed_n > 0 {
        format!(
            " · {dino_embed_n} region embeds {dino_embed_ms:.0}ms ({:.1}ms/region — crop {:.0}ms, \
             prep {:.0}ms, gpu {:.0}ms, pool {:.0}ms)",
            dino_embed_ms / dino_embed_n as f64, embed_crop_ms, embed_prep_ms, embed_gpu_ms,
            (dino_embed_ms - embed_crop_ms - embed_prep_ms - embed_gpu_ms).max(0.0),
        )
    } else {
        String::new()
    };
    log(tx, format!(
        "[timing] leaf {}: {dino_tiles} tiles · detect {det_ms:.0}ms (dino {dino_ms:.0}ms = \
         {:.1}ms/tile, post {:.0}ms) · recon {rec_ms:.0}ms · morph {morph_ms:.0}ms{embed_note} · total {:.0}ms",
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

/// Returns `(added, whole, mask)` for a leaf cutout: `added` = reconstructed
/// lost tissue (predicted leaf where the cutout is missing), `whole` = the whole
/// intact leaf (predicted ∪ visible) — both in leaf-resolution px — `mask` = the
/// predicted intact-leaf mask (`sz²`) for the canvas preview (`show_recon`'s
/// cyan tint). Runs UNetSimple at `sz`×`sz`.
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

    // Pre-damage nudge: applied to a COPY fed to the model only — `r` (and the
    // `visible` derived from it below) stays the TRUE observed signal, used
    // for the area stat. See PRE_DAMAGE_PCT's doc comment.
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
            return (0, 0, Vec::new());
        }
        let unrotated = rotate_prob_cw_k(&raw, sz, (4 - k) % 4);
        for (s, v) in pred.iter_mut().zip(unrotated.iter()) { *s += v; }
    }
    for p in pred.iter_mut() { *p /= 4.0; }

    let visible: Vec<bool> = (0..n).map(|i| r[i * 4 + 3] > 128).collect();
    // "Shape only" cleanup: keep the silhouette connected to the visible leaf and
    // fill interior holes → one solid intact shape for the preview.
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

/// FixedK: resolve `target_k` (0 = auto) to a concrete cluster count and
/// compute labels for it, logging the auto-suggestion's top candidates
/// either way so the log is self-explanatory regardless of whether the user
/// picked K manually. Returns `(labels, resolved_k)`.
fn resolve_fixed_k(
    tx: &mpsc::Sender<PipeMsg>,
    merges: &[(usize, usize, f32)],
    n: usize,
    target_k: usize,
    min_cluster_size: usize,
) -> (Vec<i32>, usize) {
    let candidates = cluster::suggest_k(merges);
    let cand_str = candidates.iter()
        .map(|&(k, gap)| format!("K={k} (gap {gap:.2})"))
        .collect::<Vec<_>>().join(", ");
    let k = if target_k == 0 {
        let auto = candidates.first().map(|&(k, _)| k).unwrap_or(1);
        log(tx, format!("target_k=0 (auto): top candidates {cand_str} — using K={auto}"));
        auto
    } else {
        log(tx, format!("target_k={target_k} (manual): top candidates {cand_str}"));
        target_k
    };
    let k = k.clamp(1, n.max(1));
    let labels = cluster::labels_for_k(merges, n, k, min_cluster_size);
    (labels, k)
}

/// Adaptive: cut the tree at a per-branch depth via the inconsistency
/// statistic (see `cluster::labels_adaptive`'s doc comment) instead of one
/// flat K applied uniformly — added specifically because a flat cut left 2
/// of the biggest clusters merging two visually-distinct defect categories
/// no matter what K was tried, while everything else resolved fine.
fn resolve_adaptive(
    tx: &mpsc::Sender<PipeMsg>,
    merges: &[(usize, usize, f32)],
    n: usize,
    threshold: f32,
    min_cluster_size: usize,
) -> Vec<i32> {
    let labels = cluster::labels_adaptive(merges, n, threshold, min_cluster_size);
    let n_clusters = labels.iter().copied().filter(|&l| l >= 0).collect::<std::collections::HashSet<_>>().len();
    log(tx, format!("adaptive_threshold={threshold:.2}: {n_clusters} clusters"));
    labels
}

/// Scale factor for the standardized 8-D descriptor block so its total
/// variance matches the (possibly-projected) DINO embedding block's own
/// total variance, before concatenating the two for PCA — a defensible,
/// symmetric 50/50 starting point, recomputed fresh every run (like
/// `suggest_eps`/`suggest_k`, a data-driven number instead of a fixed
/// constant that would need re-tuning as the embedding's own variance
/// characteristics shift — e.g. domain_projection changes them a lot).
fn hybrid_descriptor_weight(embed_block: &[Vec<f32>]) -> f32 {
    if embed_block.is_empty() {
        return 1.0;
    }
    let dim = embed_block[0].len();
    let n = embed_block.len() as f32;
    let mean: Vec<f32> = (0..dim).map(|d| embed_block.iter().map(|p| p[d]).sum::<f32>() / n).collect();
    let embed_total_var: f32 = (0..dim)
        .map(|d| embed_block.iter().map(|p| (p[d] - mean[d]).powi(2)).sum::<f32>() / n)
        .sum();
    (embed_total_var / 8.0).sqrt().max(1e-3)
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
    // Zero-initialised, so anything we skip below stays RGBA(0,0,0,0).
    let mut out = vec![0u8; (win * win * 4) as usize];
    for oy in 0..win as i32 {
        let sy = cy - half + oy;
        // Out-of-image rows/columns are left TRANSPARENT, not clamped.
        //
        // This used to `.clamp()` the source coordinate, which meant every row
        // or column past the border re-sampled the same edge pixel — smearing
        // it into a streak. Reported from review as anomalies near a leaf edge
        // rendering with elongated pixels, visible in the region tiles but not
        // on the main canvas (which draws the leaf directly, not through here).
        //
        // It is not only cosmetic: `region.crop` is also the CURATION crop the
        // head retrains on, so clamping fed the model fabricated texture that
        // exists nowhere in a real leaf. Transparent padding matches what the
        // cutout already looks like just inside the border.
        if sy < 0 || sy >= h as i32 {
            continue;
        }
        for ox in 0..win as i32 {
            let sx = cx - half + ox;
            if sx < 0 || sx >= w as i32 {
                continue;
            }
            let si = ((sy as u32 * w + sx as u32) * 4) as usize;
            let oi = ((oy as u32 * win + ox as u32) * 4) as usize;
            out[oi..oi + 4].copy_from_slice(&rgba[si..si + 4]);
        }
    }
    out
}

// Bound how many region crops go into one batched DINO forward pass
// (`DinoExtractor::features_batch`) — a leaf can have hundreds of regions,
// and an unbounded batch risks a very large single GPU allocation. Small
// enough to keep memory bounded, large enough to meaningfully amortize the
// forward pass's fixed per-call overhead (the dominant per-region cost —
// see `dino.rs::features_batch`'s doc comment).
const EMBED_BATCH: usize = 16;

/// Model input resolution used for region embeddings (both the mask-aware
/// `dino_embed` and the `dino_embed_whole` companion `domain_projection`
/// trains on). Currently a no-op (returns `dino_res` unchanged) — TRIED
/// halving it (self-attention cost scales with patch count squared, so
/// this looked like a real, non-marginal lever after a crop/gpu/pool
/// timing breakdown confirmed GPU compute was ~86% of per-region cost),
/// but real-world timing showed NO measurable change in GPU ms/region —
/// meaning either the actual bottleneck isn't what the token-count-squared
/// reasoning assumed, or something about the JIT/CUDA kernel path
/// (untested hypothesis: a second, differently-shaped kernel needing
/// separate compilation) cancels out the expected savings. Reverted rather
/// than keep it as an unproven, unverifiable-from-here change — this repo
/// has no GPU here to test on, so a claim that isn't backed by the user's
/// own real timing shouldn't ship as if it were. `features_at`/
/// `features_batch_at` (the plumbing this called through) stay in place
/// since they're harmless and reusable if this gets revisited with a real
/// diagnosis instead of another guess.
///
/// If used for anything other than identity, MUST be applied consistently
/// to every embedding a projection is trained OR applied on —
/// `projection::try_train_and_apply`'s own training crops (historical
/// curation PNGs) go through this same call, so there's no train/inference
/// resolution mismatch (the exact kind of mismatch `dino_embed_whole`'s own
/// doc comment already had to reason about once, for masking).
pub(crate) fn embed_resolution(dino_res: u32) -> u32 {
    dino_res
}

/// Build the adaptive-size crop used for a region's DINO embedding — NEVER
/// `region.crop`, which stays fixed at `CROP_WIN` for the gallery thumbnail
/// AND the literal PNG persisted to curations (must not change size or
/// those diverge from what the curation flywheel expects). Returns the crop
/// alongside its own `win` size, which `pool_region_embed` needs to map
/// patch centers back to leaf coordinates. Split out from the old
/// `region_dino_embed` so the DINO forward itself can be batched across many
/// regions (`DinoExtractor::features_batch`) instead of one call per region.
// ── on-demand appearance ranking ────────────────────────────────────────────
//
// Measured on this project's GPU: one region embedding costs ~44 ms at res 512
// (22/second, compute-bound). Detection itself is ~24 tile passes for a whole
// leaf, so embedding every region of a busy leaf is roughly 250x the work of
// detecting it — 4.4 minutes for 6,000 regions. That rules out doing it on every
// run, and it is why this exists as an explicit action scoped to one family
// (~341 regions ≈ 15 s) rather than as a pipeline stage.

/// One region to embed. `mask` is bbox-local, as stored on `AnomalyRegion`.
pub struct RankRegion {
    pub idx:       usize,
    pub centroid:  [f32; 2],
    pub bbox_leaf: [u32; 4],
    pub mask:      Vec<bool>,
}

/// Regions grouped by their leaf, so each leaf's pixels are cloned once rather
/// than once per region.
pub struct RankLeaf {
    pub rgba:    Vec<u8>,
    pub w:       u32,
    pub h:       u32,
    pub regions: Vec<RankRegion>,
}

pub enum RankMsg {
    Progress { done: usize, total: usize },
    /// `(region index, L2-normalized mask-aware embedding)`
    Done(Vec<(usize, Vec<f32>)>),
    Error(String),
}

pub fn spawn_rank(
    dino_model: PathBuf,
    dino_res:   u32,
    leaves:     Vec<RankLeaf>,
    tx:         mpsc::Sender<RankMsg>,
    cancel:     Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        let total: usize = leaves.iter().map(|l| l.regions.len()).sum();
        let mut dino = match DinoExtractor::load(&dino_model, dino_res) {
            Ok(d) => d,
            Err(e) => { let _ = tx.send(RankMsg::Error(format!("DINO: {e}"))); return; }
        };
        let res = embed_resolution(dino.res());
        let mut out: Vec<(usize, Vec<f32>)> = Vec::with_capacity(total);
        let mut done = 0usize;

        for leaf in &leaves {
            // Crops are built here rather than on the UI thread: it is real CPU
            // work per region and the whole point is to keep the window alive.
            let prepared: Vec<(usize, image::RgbImage, u32)> = leaf.regions.iter()
                .map(|r| {
                    let (img, win) = embed_crop(&leaf.rgba, leaf.w, leaf.h, r.centroid, r.bbox_leaf);
                    (r.idx, img, win)
                })
                .collect();

            for chunk in prepared.chunks(EMBED_BATCH) {
                if cancel.load(Ordering::Relaxed) {
                    // Partial results are still useful — the ranking simply
                    // covers fewer regions — so send what we have.
                    let _ = tx.send(RankMsg::Done(out));
                    return;
                }
                let imgs: Vec<image::RgbImage> = chunk.iter().map(|(_, i, _)| i.clone()).collect();
                match dino.features_batch_at(&imgs, res) {
                    Ok(feats) => {
                        for (k, f) in feats.iter().enumerate() {
                            let (idx, _, win) = &chunk[k];
                            let Some(r) = leaf.regions.iter().find(|r| r.idx == *idx) else { continue };
                            let (mask_aware, _) = pool_region_embed(
                                f, *win, r.centroid, r.bbox_leaf, &r.mask,
                                leaf.w, leaf.h, false,
                            );
                            out.push((*idx, mask_aware));
                        }
                    }
                    Err(e) => { let _ = tx.send(RankMsg::Error(format!("embed batch: {e}"))); return; }
                }
                done += chunk.len();
                let _ = tx.send(RankMsg::Progress { done, total });
            }
        }
        let _ = tx.send(RankMsg::Done(out));
    });
}

fn embed_crop(
    leaf_rgba: &[u8], lw: u32, lh: u32,
    centroid: [f32; 2], bbox_leaf: [u32; 4],
) -> (image::RgbImage, u32) {
    let [_bx, _by, bw, bh] = bbox_leaf;
    let win = (bw.max(bh) + EMBED_PAD).clamp(MIN_EMBED_CROP, MAX_EMBED_CROP);
    let crop = context_crop(leaf_rgba, lw, lh, centroid[0], centroid[1], win);
    (crop_to_rgb_meanfill(&crop, win, 10), win)
}

/// Pool one region's already-computed DINO forward result (`f`, from a
/// batched `features_batch` call) into `(mask_aware, whole_crop)` — the
/// second half of what `region_dino_embed` used to do in one synchronous
/// per-region call.
///
/// - `mask_aware`: pools only the patches whose center lands on the
///   region's own mask (falls back to whole-crop pooling if literally zero
///   patches land on-mask — e.g. a region smaller than one patch's
///   footprint). This is what `dino_embed` stores and what plain
///   (non-projected) clustering uses — least diluted by surrounding
///   healthy tissue.
/// - `whole_crop`: plain whole-crop pooling, same forward pass, no mask.
///   Exists specifically to feed `domain_projection` (projection.rs): that
///   projection can only ever be TRAINED the way historical curation PNGs
///   allow (flat 64px images, no persisted mask array), so applying it to a
///   mask-aware vector at inference would be a train/inference mismatch.
///   Feeding it this whole-crop companion instead removes that mismatch's
///   masking dimension (a residual crop-SIZE mismatch, adaptive vs. fixed
///   64px, remains and is accepted as a reasonable approximation for a
///   coarse domain-adaptation signal, not exact classification). Only
///   actually computed when `need_whole` (pass `cfg.domain_projection`) —
///   otherwise returned as an empty `Vec`, since nothing reads it and the
///   whole-crop pooling pass is real, non-trivial CPU work.
///
/// Both outputs are L2-normalized.
fn pool_region_embed(
    f: &DinoFeatures, win: u32,
    centroid: [f32; 2], bbox_leaf: [u32; 4], mask_bbox_local: &[bool],
    lw: u32, lh: u32,
    need_whole: bool,
) -> (Vec<f32>, Vec<f32>) {
    let [bx, by, bw, bh] = bbox_leaf;
    let (grid, dim) = (f.grid, f.dim);

    // Mask-aware pooling: map each patch's center pixel through the SAME
    // clamp formula context_crop used (crop-local -> leaf-global), then to
    // bbox-local, and keep it only if it lands on the region's own mask.
    let half = (win / 2) as i32;
    let (cxr, cyr) = (centroid[0].round() as i32, centroid[1].round() as i32);
    let mut acc = vec![0f32; dim];
    let mut n_on = 0usize;
    for py in 0..grid {
        for px in 0..grid {
            let fx = (px as f32 + 0.5) * win as f32 / grid as f32;
            let fy = (py as f32 + 0.5) * win as f32 / grid as f32;
            let sx = (cxr - half + fx.round() as i32).clamp(0, lw as i32 - 1);
            let sy = (cyr - half + fy.round() as i32).clamp(0, lh as i32 - 1);
            if sx < bx as i32 || sy < by as i32 {
                continue;
            }
            let (lx, ly) = ((sx - bx as i32) as u32, (sy - by as i32) as u32);
            if lx >= bw || ly >= bh {
                continue;
            }
            if mask_bbox_local[(ly * bw + lx) as usize] {
                let t = py * grid + px;
                let row = &f.feat[t * dim..(t + 1) * dim];
                for (o, &v) in acc.iter_mut().zip(row) {
                    *o += v;
                }
                n_on += 1;
            }
        }
    }
    // `whole` (plain whole-crop mean pooling, a full unconditional
    // grid*grid summation — the more expensive of the two pooling passes)
    // is needed for two INDEPENDENT reasons: it's the required fallback
    // when literally zero patches land on the region's own mask (n_on ==
    // 0), and it's the companion vector `domain_projection` trains/applies
    // on. A redundancy audit found this running unconditionally even when
    // domain_projection is off and nothing ever reads the second return
    // value — compute it only when actually needed either way.
    let compute_whole = need_whole || n_on == 0;
    let whole_raw = if compute_whole { f.mean_pool() } else { Vec::new() };
    let mut mask_aware = if n_on > 0 {
        for v in &mut acc {
            *v /= n_on as f32;
        }
        acc
    } else {
        whole_raw.clone()
    };
    l2_normalize(&mut mask_aware);
    let mut whole_normed = if need_whole { whole_raw } else { Vec::new() };
    l2_normalize(&mut whole_normed);
    (mask_aware, whole_normed)
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for x in v.iter_mut() {
        *x /= norm;
    }
}
