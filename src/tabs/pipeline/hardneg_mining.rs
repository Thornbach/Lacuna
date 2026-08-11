//! In-app hard-negative mining for the few-shot head — the automated,
//! patch-level counterpart to the Curate tab's manual `stamp_hardneg`.
//!
//! Runs the CURRENT head over a folder of known-healthy tiles and harvests
//! individual patches where it's currently wrong (`defect_prob >= tau_mine`)
//! as new hard-negative curations, written through the exact same
//! crop+`labels.jsonl` interface `stamp_hardneg` (`pipeline/mod.rs`) already
//! produces — `train::head::retrain` needs zero changes to consume mined
//! examples, it already detects hard negatives via the `_hardneg_` filename
//! substring and gives them a weight boost.
//!
//! Mirrors `1Help/eval/mine_hardneg.py`'s approach (mine what's CURRENTLY
//! wrong on known-healthy tiles, at the PATCH level, not whole crops) but
//! natively in Rust — no Python dependency. Dense per-tile DINO features are
//! cached to disk (mirroring `bank.rs`'s manual binary format) so repeated
//! mining runs against the same tile pool, e.g. while tuning `tau_mine` or
//! after a retrain, don't re-pay the DINO forward-pass cost for tiles
//! already scanned.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use image::RgbaImage;
use rand::seq::SliceRandom;

use super::dino::DinoExtractor;
use super::fewshot::FewShotHead;
use super::json_escape;
use crate::tabs::leaf_seg::inference::list_images;

/// Matches the threshold `worker.rs` actually passes to `tile_leaf` for
/// "real leaf pixel" vs background/padding.
const ALPHA_THR: u8 = 10;
/// Cache-miss tiles per DINO forward-pass batch — `dino.rs`'s own doc
/// comment on `features_batch_at` explains batching exists because forward-
/// pass cost is dominated by fixed per-call overhead, exactly the "many
/// tiles" case mining targets. Hardcoded, not a UI setting.
const MINE_BATCH: usize = 16;
/// No single tile/leaf can contribute more than this many hard negatives,
/// no matter how many of its patches qualify — without this, one tile with
/// lots of qualifying patches (genuinely unusual, or just unlucky) can
/// absorb the ENTIRE `max_hardneg` budget alone before the scan ever
/// reaches the rest of the pool. Confirmed in practice: 2000 hard
/// negatives came from just 16 of 46,239 tiles — the scan stopped as soon
/// as those 16 alone hit the cap, never touching the other 46,223. Paired
/// with shuffling the scan order below (see `mine`/`mine_unmarked`), since
/// a per-item cap alone still wouldn't help if the scan always stops after
/// the same first N tiles in directory order.
const MAX_PER_ITEM: usize = 10;

pub struct MineConfig {
    pub healthy_dir:   PathBuf,
    pub head_path:     PathBuf,
    pub dino_model:    PathBuf,
    /// The curations folder DIRECTLY (the one containing `labels.jsonl` and
    /// `labels/`) — same convention `RetrainCfg::curations_dir` already
    /// uses, not its parent. Callers that only have an "output folder"
    /// (whose curations live in `<output>/curations/`) should join that on
    /// themselves before constructing this, exactly like `start_retrain`
    /// already does for `RetrainCfg`.
    pub curations_dir: PathBuf,
    pub tau_mine:      f32,
    pub hardneg_tile:  u32,
    pub max_hardneg:   usize,
}

pub enum MineMsg {
    Progress { done: usize, total: usize },
    Found { n_so_far: usize },
    Log(String),
    Error(String),
    Done(String),
}

pub fn spawn_mine(cfg: MineConfig, tx: mpsc::Sender<MineMsg>, cancel: Arc<AtomicBool>) {
    std::thread::spawn(move || match mine(&cfg, &tx, &cancel) {
        Ok(summary) => { let _ = tx.send(MineMsg::Done(summary)); }
        Err(e) => { let _ = tx.send(MineMsg::Error(e)); }
    });
}

// ── per-tile dense-feature cache (mirrors bank.rs's manual binary format) ──

struct TileFeatureCache {
    grid: usize,
    dim: usize,
    infer_resolution: u32,
    feat: Vec<f32>,
    valid: Vec<bool>,
}

impl TileFeatureCache {
    /// `<cache_dir>/<stem>_<hash8>.bin` — the hash (over the tile's
    /// canonicalized path) avoids collisions across different
    /// `healthy_dir` mining runs that happen to share a filename stem,
    /// since the cache directory is shared per `curations_dir`, not per
    /// `healthy_dir`.
    fn cache_path(cache_dir: &Path, tile_path: &Path) -> PathBuf {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let abs = std::fs::canonicalize(tile_path).unwrap_or_else(|_| tile_path.to_path_buf());
        let mut h = DefaultHasher::new();
        abs.hash(&mut h);
        let stem = tile_path.file_stem().and_then(|s| s.to_str()).unwrap_or("tile");
        cache_dir.join(format!("{stem}_{:08x}.bin", h.finish() as u32))
    }

    /// Any mismatch (missing file, truncated, wrong dim/resolution — e.g. a
    /// later model/resolution change) is a plain cache MISS, not an error —
    /// re-extraction is the normal, expected fallback, not a failure.
    fn load(path: &Path, expect_dim: usize, expect_res: u32) -> Option<Self> {
        let bytes = std::fs::read(path).ok()?;
        if bytes.len() < 12 {
            return None;
        }
        let grid = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
        let dim = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
        let res = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
        if dim != expect_dim || res != expect_res {
            return None;
        }
        let n = grid * grid;
        let need = 12 + n * dim * 4 + n;
        if bytes.len() < need {
            return None;
        }
        let mut feat = vec![0f32; n * dim];
        for (i, slot) in feat.iter_mut().enumerate() {
            let o = 12 + i * 4;
            *slot = f32::from_le_bytes(bytes[o..o + 4].try_into().ok()?);
        }
        let vo = 12 + n * dim * 4;
        let valid: Vec<bool> = bytes[vo..vo + n].iter().map(|&b| b != 0).collect();
        Some(Self { grid, dim, infer_resolution: res, feat, valid })
    }

    fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(p) = path.parent() {
            let _ = std::fs::create_dir_all(p);
        }
        let mut buf = Vec::with_capacity(12 + self.feat.len() * 4 + self.valid.len());
        buf.extend_from_slice(&(self.grid as u32).to_le_bytes());
        buf.extend_from_slice(&(self.dim as u32).to_le_bytes());
        buf.extend_from_slice(&self.infer_resolution.to_le_bytes());
        for v in &self.feat {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.valid {
            buf.push(if v { 1 } else { 0 });
        }
        std::fs::write(path, buf).map_err(|e| format!("write cache {}: {e}", path.display()))
    }
}

// ── patch-grid valid mask ───────────────────────────────────────────────────

/// Downsamples pixel-level alpha validity to `grid×grid` by majority vote
/// per cell — same `py*mh/grid`-style bucketing `head.rs::mean_feature_masked`
/// already uses for the reverse direction. Tiles with no alpha channel at
/// all read as alpha=255 everywhere via `to_rgba8()`, so this single code
/// path handles both real cutouts and plain RGB photos without branching.
fn patch_valid_mask(rgba: &RgbaImage, grid: usize) -> Vec<bool> {
    let (w, h) = (rgba.width() as usize, rgba.height() as usize);
    let mut out = vec![false; grid * grid];
    if w == 0 || h == 0 {
        return out;
    }
    for gy in 0..grid {
        let y0 = gy * h / grid;
        let y1 = ((gy + 1) * h / grid).max(y0 + 1).min(h);
        for gx in 0..grid {
            let x0 = gx * w / grid;
            let x1 = ((gx + 1) * w / grid).max(x0 + 1).min(w);
            let (mut on, mut tot) = (0usize, 0usize);
            for y in y0..y1 {
                for x in x0..x1 {
                    tot += 1;
                    if rgba.get_pixel(x as u32, y as u32).0[3] > ALPHA_THR {
                        on += 1;
                    }
                }
            }
            out[gy * grid + gx] = tot > 0 && on * 2 >= tot;
        }
    }
    out
}

/// Crops away the fully-transparent border around the actual leaf content.
/// Real cutout tile crops are often mostly padding — a small leaf fragment
/// in a much larger transparent frame (a tile from a bounding-box-based
/// crop step, not a tight leaf silhouette). Feeding that whole frame to
/// DINO matters more than it might look: a ViT's self-attention is GLOBAL,
/// so even the genuinely-valid patches' computed features are
/// contextualized against whatever ELSE is in the same image — a tile
/// that's ~90% transparent/black background puts every patch in a very
/// different visual context than a normal, mostly-leaf-filling detection
/// tile or curated crop ever appears in. Trimming first keeps the content
/// in a representative frame, closer to what the classifier actually
/// learned from. A small margin is kept so a patch right at the leaf's own
/// edge still has real surrounding context, not an abrupt crop boundary.
fn trim_to_content(rgba: RgbaImage) -> RgbaImage {
    const MARGIN: i64 = 8;
    let (w, h) = (rgba.width(), rgba.height());
    let (mut minx, mut miny, mut maxx, mut maxy) = (w, h, 0u32, 0u32);
    let mut any = false;
    for y in 0..h {
        for x in 0..w {
            if rgba.get_pixel(x, y).0[3] > ALPHA_THR {
                any = true;
                minx = minx.min(x);
                miny = miny.min(y);
                maxx = maxx.max(x + 1);
                maxy = maxy.max(y + 1);
            }
        }
    }
    if !any {
        // Fully transparent tile -- nothing to trim to. It'll mine nothing
        // (patch_valid_mask will be all-false), so the untrimmed image is
        // harmless here, just wasted work; not worth a special-cased error.
        return rgba;
    }
    let x0 = (minx as i64 - MARGIN).max(0) as u32;
    let y0 = (miny as i64 - MARGIN).max(0) as u32;
    let x1 = ((maxx as i64 + MARGIN).max(0) as u32).min(w);
    let y1 = ((maxy as i64 + MARGIN).max(0) as u32).min(h);
    image::imageops::crop_imm(&rgba, x0, y0, x1 - x0, y1 - y0).to_image()
}

// ── native-pixel coordinate mapping ─────────────────────────────────────────

/// `dino.rs::features_at` resizes with a plain non-aspect-preserving stretch
/// (`image::imageops::resize(img, res, res, Triangle)`) — verified by
/// reading it, not assumed — so mapping a patch back to the ORIGINAL
/// (pre-resize) tile is a simple independent-per-axis scale, the same
/// "fraction of grid -> fraction of extent" form `fewshot::upscale_family`
/// already uses for the reverse direction.
fn patch_center_native(gx: usize, gy: usize, grid: usize, w: u32, h: u32) -> (i32, i32) {
    let nx = ((gx as f32 + 0.5) / grid as f32 * w as f32).round() as i32;
    let ny = ((gy as f32 + 0.5) / grid as f32 * h as f32).round() as i32;
    (nx, ny)
}

/// Same in-bounds/transparent-pad crop loop `stamp_hardneg` (`pipeline/mod.rs`)
/// already uses for a manual click.
fn crop_square(rgba: &RgbaImage, cx: i32, cy: i32, tile: u32) -> RgbaImage {
    let t = tile as i32;
    let (x0, y0) = (cx - t / 2, cy - t / 2);
    let (w, h) = (rgba.width() as i32, rgba.height() as i32);
    let mut buf = vec![0u8; (tile * tile * 4) as usize];
    for row in 0..t {
        let sy = y0 + row;
        if sy < 0 || sy >= h {
            continue;
        }
        for col in 0..t {
            let sx = x0 + col;
            if sx < 0 || sx >= w {
                continue;
            }
            let sp = rgba.get_pixel(sx as u32, sy as u32).0;
            let di = ((row * t + col) * 4) as usize;
            buf[di..di + 4].copy_from_slice(&sp);
        }
    }
    RgbaImage::from_raw(tile, tile, buf).expect("crop_square: buffer size matches tile*tile*4")
}

// ── mining ───────────────────────────────────────────────────────────────

fn mine(cfg: &MineConfig, tx: &mpsc::Sender<MineMsg>, cancel: &AtomicBool) -> Result<String, String> {
    let _ = tx.send(MineMsg::Log("Loading head + DINO…".into()));
    let head = FewShotHead::load(&cfg.head_path)?;
    let mut dino = DinoExtractor::load(&cfg.dino_model, head.infer_resolution)?;
    let res = head.infer_resolution;

    let mut tiles = list_images(&cfg.healthy_dir);
    if tiles.is_empty() {
        return Err("no images found in the healthy tiles folder".into());
    }
    let total = tiles.len();
    // Shuffle so whichever tiles actually get scanned before `max_hardneg`
    // is hit are a random cross-section of the WHOLE pool, not whatever
    // happens to sort first in the folder (see MAX_PER_ITEM's doc comment).
    tiles.shuffle(&mut rand::thread_rng());

    let labels_dir = cfg.curations_dir.join("labels");
    std::fs::create_dir_all(&labels_dir).map_err(|e| format!("curations dir: {e}"))?;
    let cache_dir = cfg.curations_dir.join("healthy_feature_cache");
    let _ = std::fs::create_dir_all(&cache_dir);
    let jsonl_path = cfg.curations_dir.join("labels.jsonl");

    // One timestamp for the whole run — (tile_idx, native x, y) already
    // disambiguates every mined crop's filename, no need for per-crop clock
    // reads (stamp_hardneg reads it per-click; here that would just be
    // thousands of near-identical syscalls for no benefit).
    let run_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut n_found = 0usize;
    let mut done = 0usize;
    let mut batch: Vec<(usize, PathBuf, RgbaImage)> = Vec::new();

    for (idx, path) in tiles.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        if n_found >= cfg.max_hardneg {
            break;
        }

        let cp = TileFeatureCache::cache_path(&cache_dir, path);
        if let Some(c) = TileFeatureCache::load(&cp, head.dim, res) {
            // Cache hit: the classifier scan itself is cheap (a linear
            // model, not DINO) — only skip opening/decoding the actual
            // pixel image if nothing on this tile qualifies at all.
            let out = head.predict(&c.feat, c.grid, c.dim);
            let any = out.defect_prob.iter().zip(c.valid.iter())
                .any(|(&dp, &v)| v && dp >= cfg.tau_mine);
            if any {
                if let Ok(img) = image::open(path) {
                    // Same trim every time this tile is processed (a pure
                    // function of its own pixels), so patch indices map to
                    // the same native coordinates whether this run hit the
                    // cache or the one that originally populated it did.
                    let rgba = trim_to_content(img.to_rgba8());
                    let remaining = cfg.max_hardneg.saturating_sub(n_found).min(MAX_PER_ITEM);
                    let added = mine_one_tile(
                        &out.defect_prob, &c.valid, c.grid, &rgba, None, remaining, idx, path,
                        cfg.tau_mine, cfg.hardneg_tile, run_ts, &labels_dir, &jsonl_path, "mined",
                    )?;
                    n_found += added;
                    if added > 0 {
                        let _ = tx.send(MineMsg::Found { n_so_far: n_found });
                    }
                }
            }
        } else if let Ok(img) = image::open(path) {
            batch.push((idx, path.clone(), trim_to_content(img.to_rgba8())));
            if batch.len() >= MINE_BATCH {
                flush_batch(
                    &mut batch, &mut dino, res, &head, &cache_dir, &labels_dir, &jsonl_path,
                    cfg.tau_mine, cfg.hardneg_tile, cfg.max_hardneg, run_ts, &mut n_found, tx,
                )?;
            }
        } else {
            let _ = tx.send(MineMsg::Log(format!("skip {}: could not open", path.display())));
        }

        done += 1;
        if done % 5 == 0 || done == total {
            let _ = tx.send(MineMsg::Progress { done, total });
        }
    }
    flush_batch(
        &mut batch, &mut dino, res, &head, &cache_dir, &labels_dir, &jsonl_path,
        cfg.tau_mine, cfg.hardneg_tile, cfg.max_hardneg, run_ts, &mut n_found, tx,
    )?;

    Ok(format!(
        "Mined {n_found} hard negative(s) from {done}/{total} healthy tiles -> {}",
        jsonl_path.display()
    ))
}

/// Extracts + caches features for every tile still in `batch` (one DINO
/// forward pass for the whole batch, not one per tile), then scores and
/// mines each. Called both when a batch fills up mid-scan and once more
/// after the scan loop ends, for whatever partial batch is left.
fn flush_batch(
    batch: &mut Vec<(usize, PathBuf, RgbaImage)>,
    dino: &mut DinoExtractor,
    res: u32,
    head: &FewShotHead,
    cache_dir: &Path,
    labels_dir: &Path,
    jsonl_path: &Path,
    tau_mine: f32,
    hardneg_tile: u32,
    max_hardneg: usize,
    run_ts: u64,
    n_found: &mut usize,
    tx: &mpsc::Sender<MineMsg>,
) -> Result<(), String> {
    if batch.is_empty() {
        return Ok(());
    }
    let rgbs: Vec<image::RgbImage> = batch.iter()
        .map(|(_, _, rgba)| image::DynamicImage::ImageRgba8(rgba.clone()).to_rgb8())
        .collect();
    let feats = dino.features_batch_at(&rgbs, res)?;
    for ((idx, path, rgba), feat) in batch.drain(..).zip(feats.into_iter()) {
        let valid = patch_valid_mask(&rgba, feat.grid);
        let cache = TileFeatureCache {
            grid: feat.grid, dim: feat.dim, infer_resolution: res,
            feat: feat.feat.clone(), valid: valid.clone(),
        };
        if let Err(e) = cache.save(&TileFeatureCache::cache_path(cache_dir, &path)) {
            let _ = tx.send(MineMsg::Log(format!("cache save failed for {}: {e}", path.display())));
        }
        if *n_found >= max_hardneg {
            continue;
        }
        let out = head.predict(&feat.feat, feat.grid, feat.dim);
        let remaining = max_hardneg.saturating_sub(*n_found).min(MAX_PER_ITEM);
        let added = mine_one_tile(
            &out.defect_prob, &valid, feat.grid, &rgba, None, remaining, idx, &path,
            tau_mine, hardneg_tile, run_ts, labels_dir, jsonl_path, "mined",
        )?;
        *n_found += added;
        if added > 0 {
            let _ = tx.send(MineMsg::Found { n_so_far: *n_found });
        }
    }
    Ok(())
}

/// Cuts a `keep`-predicate mask over the SAME window `crop_square` cuts —
/// pixel-aligned so `train::head::crop_feature`'s mask-aware pooling (the
/// same machinery `persist_region`'s confirmed/rejected region crops
/// already rely on) restricts a mined crop's feature to exactly the pixels
/// that are actually valid+healthy, not the whole square around them.
fn crop_square_mask(w: u32, h: u32, cx: i32, cy: i32, tile: u32, keep: impl Fn(u32, u32) -> bool) -> image::GrayImage {
    let t = tile as i32;
    let (x0, y0) = (cx - t / 2, cy - t / 2);
    let mut buf = vec![0u8; (tile * tile) as usize];
    for row in 0..t {
        let sy = y0 + row;
        if sy < 0 || sy >= h as i32 {
            continue;
        }
        for col in 0..t {
            let sx = x0 + col;
            if sx < 0 || sx >= w as i32 {
                continue;
            }
            if keep(sx as u32, sy as u32) {
                buf[(row * t + col) as usize] = 255;
            }
        }
    }
    image::GrayImage::from_raw(tile, tile, buf).expect("crop_square_mask: buffer size matches tile*tile")
}

/// Scans one tile's per-patch predictions for `defect_prob >= tau_mine`
/// (and patch-valid), cuts a `hardneg_tile`-sized crop for each qualifying
/// patch, and writes it through the SAME crop+`labels.jsonl` interface
/// `stamp_hardneg` already produces. Stops once `remaining_cap` crops have
/// been written this call. Returns the number added. `source` is the
/// provenance tag written to `labels.jsonl` (`"mined"` for the healthy-tile
/// pool, `"unmarked"` for `mine_unmarked`'s leaf-complement pass) — purely
/// informational, `train::head::retrain` only keys off the `_hardneg_`
/// filename substring either way.
///
/// `marked` (leaf/tile-pixel-resolution, `w*h`, true = NOT healthy) is
/// `Some` only for `mine_unmarked`, which actually knows which pixels are
/// covered by a real anomaly region. EVERY mined crop still gets a saved
/// mask — even the `marked: None` (healthy-tile-pool) case — restricted to
/// alpha-valid pixels, so a crop near a tile's edge can't silently average
/// in transparent padding either. Without this, `crop_feature` had nothing
/// to mask against and fell back to averaging the WHOLE square uniformly —
/// the actual root cause of a real regression: hard-negative "Healthy"
/// crops only ever validated their OWN center patch, so neighboring pixels
/// within the same square (padding, or genuinely a different, adjacent
/// anomaly) got silently blended into "Healthy" training signal, which is
/// exactly what a class needs to converge to a low, washed-out coefficient
/// norm and over-predict Healthy at inference.
fn mine_one_tile(
    defect_prob: &[f32],
    valid: &[bool],
    grid: usize,
    rgba: &RgbaImage,
    marked: Option<&[bool]>,
    remaining_cap: usize,
    tile_idx: usize,
    tile_path: &Path,
    tau_mine: f32,
    hardneg_tile: u32,
    run_ts: u64,
    labels_dir: &Path,
    jsonl_path: &Path,
    source: &str,
) -> Result<usize, String> {
    let (w, h) = (rgba.width(), rgba.height());
    let leaf_src = tile_path.display().to_string();
    let mut added = 0usize;
    for p in 0..grid * grid {
        if added >= remaining_cap {
            break;
        }
        if !valid[p] || defect_prob[p] < tau_mine {
            continue;
        }
        let (gx, gy) = (p % grid, p / grid);
        let (cx, cy) = patch_center_native(gx, gy, grid, w, h);
        let crop = crop_square(rgba, cx, cy, hardneg_tile);
        let mask = crop_square_mask(w, h, cx, cy, hardneg_tile, |px, py| {
            rgba.get_pixel(px, py).0[3] > ALPHA_THR
                && marked.map_or(true, |m| !m[py as usize * w as usize + px as usize])
        });

        let fname = format!("{run_ts}_hardneg_{tile_idx}_{cx}_{cy}.png");
        let file = labels_dir.join(&fname);
        crop.save(&file).map_err(|e| format!("save crop {}: {e}", file.display()))?;
        let mask_fname = format!("{run_ts}_hardneg_{tile_idx}_{cx}_{cy}_mask.png");
        let mask_file = labels_dir.join(&mask_fname);
        mask.save(&mask_file).map_err(|e| format!("save mask {}: {e}", mask_file.display()))?;

        let line = format!(
            "{{\"crop\":\"{}\",\"mask\":\"{}\",\"family\":\"rejected\",\"source\":\"{}\",\"leaf_src\":\"{}\",\"ts\":{}}}\n",
            fname, mask_fname, source, json_escape(&leaf_src), run_ts,
        );
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().create(true).append(true)
            .open(jsonl_path)
            .map_err(|e| format!("labels.jsonl: {e}"))?;
        f.write_all(line.as_bytes()).map_err(|e| format!("labels.jsonl: {e}"))?;

        added += 1;
    }
    Ok(added)
}

// ── mining the UNMARKED area of already-curated leaves ─────────────────────
//
// A second producer for the same `_hardneg_` interface, feeding off Curate's
// OWN state instead of a separate healthy-tile folder: on any leaf where
// every detected candidate region has been acted on (confirmed or rejected —
// nothing left pending), the leaf area NOT covered by any of those regions
// is, by the curation contract itself, healthy — no separate "known-healthy"
// pool needed. Same patch-level/tau-gated harvest as `mine()`, just sourced
// from leaves already loaded in memory rather than files on disk, so there's
// no feature cache (a curated batch is dozens/hundreds of leaves, not tens
// of thousands of tiles — not worth it) and no `trim_to_content` (Pipeline's
// leaf cutouts are already tight crops, not padded tiles).

/// One already-loaded, fully-reviewed leaf, ready to mine. `marked` is a
/// `w*h` canvas the caller builds by OR-ing every region ever detected on
/// this leaf (visible AND removed/rejected — a rejected region already gets
/// its own `"source":"reject"` crop via `persist_region`, so re-mining that
/// same area here would just be a redundant duplicate) EXCEPT merged-away
/// ones, whose area the surviving merged region's own mask already covers.
pub struct LeafMineInput {
    pub leaf_idx: usize,
    pub src:      PathBuf,
    pub rgba:     RgbaImage,
    pub marked:   Vec<bool>,
}

pub struct MineUnmarkedConfig {
    pub head_path:     PathBuf,
    pub dino_model:    PathBuf,
    pub curations_dir: PathBuf,
    pub tau_mine:      f32,
    pub hardneg_tile:  u32,
    pub max_hardneg:   usize,
}

pub fn spawn_mine_unmarked(
    leaves: Vec<LeafMineInput>,
    cfg: MineUnmarkedConfig,
    tx: mpsc::Sender<MineMsg>,
    cancel: Arc<AtomicBool>,
) {
    std::thread::spawn(move || match mine_unmarked(leaves, &cfg, &tx, &cancel) {
        Ok(summary) => { let _ = tx.send(MineMsg::Done(summary)); }
        Err(e) => { let _ = tx.send(MineMsg::Error(e)); }
    });
}

/// Same majority-vote patch bucketing as `patch_valid_mask`, but a cell only
/// counts as minable if the MAJORITY of its pixels are both real leaf
/// pixels (alpha) AND outside every marked region — a patch straddling a
/// region boundary is conservatively excluded rather than harvested.
fn patch_valid_unmarked_mask(rgba: &RgbaImage, marked: &[bool], grid: usize) -> Vec<bool> {
    let (w, h) = (rgba.width() as usize, rgba.height() as usize);
    let mut out = vec![false; grid * grid];
    if w == 0 || h == 0 || marked.len() != w * h {
        return out;
    }
    for gy in 0..grid {
        let y0 = gy * h / grid;
        let y1 = ((gy + 1) * h / grid).max(y0 + 1).min(h);
        for gx in 0..grid {
            let x0 = gx * w / grid;
            let x1 = ((gx + 1) * w / grid).max(x0 + 1).min(w);
            let (mut on, mut tot) = (0usize, 0usize);
            for y in y0..y1 {
                for x in x0..x1 {
                    tot += 1;
                    let alpha_ok = rgba.get_pixel(x as u32, y as u32).0[3] > ALPHA_THR;
                    if alpha_ok && !marked[y * w + x] {
                        on += 1;
                    }
                }
            }
            out[gy * grid + gx] = tot > 0 && on * 2 >= tot;
        }
    }
    out
}

fn mine_unmarked(
    mut leaves: Vec<LeafMineInput>,
    cfg: &MineUnmarkedConfig,
    tx: &mpsc::Sender<MineMsg>,
    cancel: &AtomicBool,
) -> Result<String, String> {
    let _ = tx.send(MineMsg::Log("Loading head + DINO…".into()));
    let head = FewShotHead::load(&cfg.head_path)?;
    let mut dino = DinoExtractor::load(&cfg.dino_model, head.infer_resolution)?;
    let res = head.infer_resolution;

    if leaves.is_empty() {
        return Err("no fully-reviewed leaves with unmarked area to scan".into());
    }
    let total = leaves.len();
    // Same fix as `mine`'s tile scan: don't let scan order + no per-leaf
    // cap mean the budget gets exhausted by whichever leaves happen to be
    // loaded first (see MAX_PER_ITEM's doc comment).
    leaves.shuffle(&mut rand::thread_rng());

    let labels_dir = cfg.curations_dir.join("labels");
    std::fs::create_dir_all(&labels_dir).map_err(|e| format!("curations dir: {e}"))?;
    let jsonl_path = cfg.curations_dir.join("labels.jsonl");

    let run_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut n_found = 0usize;
    let mut done = 0usize;
    let mut batch: Vec<(usize, PathBuf, RgbaImage, Vec<bool>)> = Vec::new();

    for input in leaves {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        if n_found >= cfg.max_hardneg {
            break;
        }
        batch.push((input.leaf_idx, input.src, input.rgba, input.marked));
        if batch.len() >= MINE_BATCH {
            flush_unmarked_batch(
                &mut batch, &mut dino, res, &head, &labels_dir, &jsonl_path,
                cfg.tau_mine, cfg.hardneg_tile, cfg.max_hardneg, run_ts, &mut n_found, tx,
            )?;
        }
        done += 1;
        let _ = tx.send(MineMsg::Progress { done, total });
    }
    flush_unmarked_batch(
        &mut batch, &mut dino, res, &head, &labels_dir, &jsonl_path,
        cfg.tau_mine, cfg.hardneg_tile, cfg.max_hardneg, run_ts, &mut n_found, tx,
    )?;

    Ok(format!(
        "Mined {n_found} hard negative(s) from unmarked area on {done}/{total} reviewed leaves -> {}",
        jsonl_path.display()
    ))
}

// ── validation: scoring a healthy-tile folder without mining/writing ──────

/// Cap on how many tiles a validation pass scans — a random sample is
/// enough for a false-positive-rate sanity check, and this runs TWICE
/// (before + after) on EVERY retrain. Confirmed as a real regression:
/// against a 46,239-tile healthy pool with no cap and no progress
/// reporting, this made a whole retrain look permanently hung on "Loading
/// head + DINO" — the GPU was audibly working the entire time, just with
/// nothing surfaced to the UI.
const VALIDATE_MAX_TILES: usize = 1500;

/// Picks the (shuffled, capped) tile sample a validation pass will score —
/// call ONCE per retrain and reuse the SAME list for both the before and
/// after `compute_fpr_for` calls. Picking independently each time (an
/// earlier version did) compares two DIFFERENT random subsets of a large
/// pool, which can swing the reported delta by a point or more from pure
/// sampling noise alone, with no real model change behind it at all —
/// not a valid A/B comparison.
pub fn pick_validation_tiles(healthy_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut tiles = list_images(healthy_dir);
    if tiles.is_empty() {
        return Err("no images found in the healthy tiles folder".into());
    }
    tiles.shuffle(&mut rand::thread_rng());
    tiles.truncate(VALIDATE_MAX_TILES);
    Ok(tiles)
}

/// Fraction of valid (alpha, non-transparent) patches across `tiles` that
/// `head` calls a defect (`defect_prob >= tau`) — a false-positive rate on
/// independent, known-healthy material. Used by `train::head::retrain` as
/// a before/after validation gate around a candidate retrain (see
/// `pick_validation_tiles` — both calls MUST share the same tile list).
/// Purely a scoring pass, no disk feature cache (a validation run is a
/// one-off per retrain, not a repeated scan the way `mine()`'s tile pool is).
pub fn compute_fpr_for(
    head: &FewShotHead,
    dino: &mut DinoExtractor,
    tiles: &[PathBuf],
    tau: f32,
    mut on_progress: impl FnMut(usize, usize),
) -> Result<f32, String> {
    let total = tiles.len();
    let res = head.infer_resolution;
    let (mut n_valid, mut n_fired) = (0u64, 0u64);
    let mut batch: Vec<RgbaImage> = Vec::new();
    let mut done = 0usize;
    for path in tiles {
        let Ok(img) = image::open(path) else { continue };
        batch.push(img.to_rgba8());
        done += 1;
        if batch.len() >= MINE_BATCH {
            score_batch(&batch, dino, res, head, tau, &mut n_valid, &mut n_fired)?;
            batch.clear();
            on_progress(done, total);
        }
    }
    score_batch(&batch, dino, res, head, tau, &mut n_valid, &mut n_fired)?;
    on_progress(total, total);
    if n_valid == 0 {
        return Err("no valid (non-transparent) pixels found in the healthy tiles folder".into());
    }
    Ok(n_fired as f32 / n_valid as f32)
}

fn score_batch(
    batch: &[RgbaImage],
    dino: &mut DinoExtractor,
    res: u32,
    head: &FewShotHead,
    tau: f32,
    n_valid: &mut u64,
    n_fired: &mut u64,
) -> Result<(), String> {
    if batch.is_empty() {
        return Ok(());
    }
    let rgbs: Vec<image::RgbImage> = batch.iter()
        .map(|rgba| image::DynamicImage::ImageRgba8(rgba.clone()).to_rgb8())
        .collect();
    let feats = dino.features_batch_at(&rgbs, res)?;
    for (rgba, feat) in batch.iter().zip(feats.iter()) {
        let valid = patch_valid_mask(rgba, feat.grid);
        let out = head.predict(&feat.feat, feat.grid, feat.dim);
        for (p, &v) in valid.iter().enumerate() {
            if !v {
                continue;
            }
            *n_valid += 1;
            if out.defect_prob[p] >= tau {
                *n_fired += 1;
            }
        }
    }
    Ok(())
}

fn flush_unmarked_batch(
    batch: &mut Vec<(usize, PathBuf, RgbaImage, Vec<bool>)>,
    dino: &mut DinoExtractor,
    res: u32,
    head: &FewShotHead,
    labels_dir: &Path,
    jsonl_path: &Path,
    tau_mine: f32,
    hardneg_tile: u32,
    max_hardneg: usize,
    run_ts: u64,
    n_found: &mut usize,
    tx: &mpsc::Sender<MineMsg>,
) -> Result<(), String> {
    if batch.is_empty() {
        return Ok(());
    }
    let rgbs: Vec<image::RgbImage> = batch.iter()
        .map(|(_, _, rgba, _)| image::DynamicImage::ImageRgba8(rgba.clone()).to_rgb8())
        .collect();
    let feats = dino.features_batch_at(&rgbs, res)?;
    for ((leaf_idx, src, rgba, marked), feat) in batch.drain(..).zip(feats.into_iter()) {
        if *n_found >= max_hardneg {
            continue;
        }
        let valid = patch_valid_unmarked_mask(&rgba, &marked, feat.grid);
        let out = head.predict(&feat.feat, feat.grid, feat.dim);
        let remaining = max_hardneg.saturating_sub(*n_found).min(MAX_PER_ITEM);
        let added = mine_one_tile(
            &out.defect_prob, &valid, feat.grid, &rgba, Some(&marked), remaining, leaf_idx, &src,
            tau_mine, hardneg_tile, run_ts, labels_dir, jsonl_path, "unmarked",
        )?;
        *n_found += added;
        if added > 0 {
            let _ = tx.send(MineMsg::Found { n_so_far: *n_found });
        }
    }
    Ok(())
}
