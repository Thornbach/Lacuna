//! YOLO instance-segmentation inference via ONNX Runtime (`ort`).
//!
//! The exported model (YOLO26m-seg, NMS-free one-to-one head) has:
//!   input  `images`  [1, 3, 640, 640]   (BCHW, RGB, 0–1, letterboxed)
//!   output `output0` [1, 300, 38]        per row: x1,y1,x2,y2, score, cls, c0..c31
//!   output `output1` [1, 32, 160, 160]   32 mask prototypes
//! Instance mask = sigmoid(coeffs · proto) sampled back through the letterbox.
//! No NMS is needed — the head is one-to-one; we just filter by confidence.

use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
    sync::atomic::{AtomicBool, Ordering},
};

use image::{imageops::FilterType, RgbImage, RgbaImage};
#[cfg(feature = "ort-backend")]
use ort::value::Tensor as OrtTensor;

use crate::tabs::recon_train::model::{create_infer_device, InferBackend};
use crate::yolo_burn::{yolo_decode, YoloV26Burn};

type YoloBurnDevice = <InferBackend as burn::tensor::backend::Backend>::Device;

/// YOLO26-seg backend: the pure-Rust BURN reimplementation (default) or, with the
/// optional `ort-backend` feature + `LACUNA_USE_ORT=1`, ONNX Runtime. Both produce the
/// same `(output0, output1)` so `segment_one` is backend-agnostic. See `crate::yolo_burn`
/// (validated vs the fused model to ~1e-5; end-to-end mask IoU=1.0 vs ort).
pub enum YoloModel {
    #[cfg(feature = "ort-backend")]
    Ort(ort::session::Session),
    Burn(Box<YoloV26Burn<InferBackend>>, YoloBurnDevice),
}

#[cfg(feature = "ort-backend")]
fn use_ort() -> bool {
    // Stays OFF by default, unlike DINO. The exported YOLO ONNX has a FIXED
    // 640x640 input while the app segments at imgsz 1280, so the ort path would
    // fail outright ("Got 1280, Expected 640"). BURN has no fixed input shape,
    // which is the only reason 1280 works at all. Re-export dynamic or at 1280
    // before turning this on.
    std::env::var("LACUNA_USE_ORT").map(|v| v != "0" && !v.is_empty()).unwrap_or(false)
}

fn resolve_yolo_weights(model_path: &Path) -> PathBuf {
    crate::paths::resolve_weights(model_path, "yolo_weights.safetensors", "LACUNA_YOLO_WEIGHTS")
}

/// Build the YOLO backend — BURN by default; ort only with the `ort-backend` feature
/// AND `LACUNA_USE_ORT=1`.
pub fn build_yolo(model_path: &Path) -> Result<YoloModel, String> {
    #[cfg(feature = "ort-backend")]
    if use_ort() {
        eprintln!("[yolo] backend=ort (LACUNA_USE_ORT) {}", model_path.display());
        return Ok(YoloModel::Ort(crate::tabs::build_session(model_path)?));
    }
    let device = create_infer_device();
    let wpath = resolve_yolo_weights(model_path);
    eprintln!("[yolo] backend=BURN ({}) weights={}",
              crate::tabs::recon_train::model::backend_name(), wpath.display());
    let net = YoloV26Burn::<InferBackend>::load(&wpath.to_string_lossy(), &device)?;
    Ok(YoloModel::Burn(Box::new(net), device))
}

impl YoloModel {
    /// Run the network on a letterboxed `[1,3,target,target]` CHW [0,1] buffer.
    /// Returns `(d0, d1, [ndet, nattr, nproto, ph, pw])` — the ort output0/output1 layout.
    fn infer(&mut self, input: Vec<f32>, target: u32) -> Result<(Vec<f32>, Vec<f32>, [usize; 5]), String> {
        match self {
            #[cfg(feature = "ort-backend")]
            YoloModel::Ort(session) => {
                let t = OrtTensor::from_array(([1usize, 3, target as usize, target as usize], input))
                    .map_err(|e| format!("input tensor: {e}"))?;
                let outputs = session.run(ort::inputs!["images" => t]).map_err(|e| format!("run: {e}"))?;
                let (s0, d0) = outputs["output0"].try_extract_tensor::<f32>().map_err(|e| format!("output0: {e}"))?;
                let (s1, d1) = outputs["output1"].try_extract_tensor::<f32>().map_err(|e| format!("output1: {e}"))?;
                let dims = [s0[1] as usize, s0[2] as usize, s1[1] as usize, s1[2] as usize, s1[3] as usize];
                Ok((d0.to_vec(), d1.to_vec(), dims))
            }
            YoloModel::Burn(net, device) => {
                let x = burn::tensor::Tensor::<InferBackend, 4>::from_data(
                    burn::tensor::TensorData::new(input, [1, 3, target as usize, target as usize]),
                    device,
                );
                let raw = net.forward(x);
                Ok(yolo_decode(raw, target as usize))
            }
        }
    }
}

// ── Public types ────────────────────────────────────────────────────────────

pub struct SegInstance {
    pub bbox:        [u32; 4],   // x, y, w, h in ORIGINAL image coords
    pub score:       f32,
    pub mask:        Vec<u8>,    // bbox-local, w*h, 1 = leaf
    pub cutout_path: PathBuf,
}

pub struct SegItem {
    pub filename:  String,
    pub path:      PathBuf,
    pub size:      [u32; 2],     // original (w, h)
    pub instances: Vec<SegInstance>,
}

pub enum SegMsg {
    Progress { done: usize, total: usize },
    Item(SegItem),
    Log(String),
    Finished,
    Error(String),
}

pub struct SegConfig {
    pub model_path:  PathBuf,
    pub image_paths: Vec<PathBuf>,
    pub output_dir:  PathBuf,
    pub imgsz:       u32,
    pub conf:        f32,
    /// Cutout alpha feather start (0.50 = crisp at the mask boundary; higher = tighter cut).
    pub alpha_lo:    f32,
    /// Boundary pixels with chroma below this are treated as background and dropped
    /// (catches shadowed off-white / grey / black rims). ~28 default; 0 disables.
    pub chroma_min:  i32,
}

// ── Folder scan ─────────────────────────────────────────────────────────────

pub fn list_images(dir: &Path) -> Vec<PathBuf> {
    let exts = ["png", "jpg", "jpeg", "tif", "tiff", "bmp"];
    let mut out: Vec<PathBuf> = walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| exts.contains(&e.to_ascii_lowercase().as_str()))
                .unwrap_or(false)
        })
        .collect();
    out.sort();
    out
}

pub fn scan_image_count(dir: &Path) -> usize {
    list_images(dir).len()
}

/// Segment `img_path` with the given edge settings and return the FIRST leaf's RGBA
/// cutout `(rgba, w, h)` — for the "tune the edge before a full run" preview used by
/// the Segmentation and Pipeline tabs. Runs YOLO once; writes the cutout to a temp dir.
pub fn preview_cutout(
    yolo:       &Path,
    img_path:   &Path,
    alpha_lo:   f32,
    chroma_min: i32,
) -> Result<(Vec<u8>, u32, u32), String> {
    let mut model = build_yolo(yolo)?;
    let tmp = std::env::temp_dir().join("foliar_seg_preview");
    let _ = std::fs::create_dir_all(&tmp);
    let cfg = SegConfig {
        model_path:  yolo.to_path_buf(),
        image_paths: Vec::new(),
        output_dir:  tmp,
        imgsz:       1280, // see settings.rs::FieldReviewSettings::default
        conf:        0.25,
        alpha_lo,
        chroma_min,
    };
    let item = segment_one(&mut model, img_path, &cfg)?;
    let inst = item.instances.into_iter().next().ok_or_else(|| "no leaf detected".to_string())?;
    let cut = image::open(&inst.cutout_path).map_err(|e| e.to_string())?.to_rgba8();
    let (w, h) = cut.dimensions();
    Ok((cut.into_raw(), w, h))
}

// ── Entry point ─────────────────────────────────────────────────────────────

pub fn spawn_segmentation(cfg: SegConfig, tx: mpsc::Sender<SegMsg>, cancel: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        match run_segmentation(&cfg, &tx, &cancel) {
            Err(e) => { let _ = tx.send(SegMsg::Error(e)); }
            Ok(()) => {}
        }
        let _ = tx.send(SegMsg::Finished);
    });
}

fn log(tx: &mpsc::Sender<SegMsg>, msg: impl Into<String>) {
    let _ = tx.send(SegMsg::Log(msg.into()));
}

fn run_segmentation(
    cfg:    &SegConfig,
    tx:     &mpsc::Sender<SegMsg>,
    cancel: &AtomicBool,
) -> Result<(), String> {
    log(tx, format!("Loading model: {}", cfg.model_path.display()));
    let mut model = build_yolo(&cfg.model_path)?;
    log(tx, "Model loaded.");

    std::fs::create_dir_all(&cfg.output_dir).map_err(|e| e.to_string())?;

    let total = cfg.image_paths.len();
    for (idx, path) in cfg.image_paths.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            log(tx, "Segmentation cancelled.");
            break;
        }
        let _ = tx.send(SegMsg::Progress { done: idx, total });

        match segment_one(&mut model, path, cfg) {
            Ok(item) => {
                log(tx, format!("{}: {} leaves", item.filename, item.instances.len()));
                let _ = tx.send(SegMsg::Item(item));
            }
            Err(e) => log(tx, format!("[skip] {}: {e}", path.display())),
        }
    }
    let _ = tx.send(SegMsg::Progress { done: total, total });
    Ok(())
}

// ── Single-image segmentation ───────────────────────────────────────────────

/// A connected blob smaller than this fraction of an instance's LARGEST blob is
/// prototype noise, not part of the leaf (see the speckle filter in
/// `segment_one`). Relative rather than absolute so a genuinely split leaf keeps
/// both halves: 15% is well below any real occlusion split worth keeping, and
/// well above the confetti the prototype field scatters into neighbouring leaves.
const SPECKLE_FRAC: f32 = 0.15;
/// Absolute floor for the same test, so a tiny detection isn't judged against an
/// even tinier fraction.
const SPECKLE_MIN_PX: usize = 24;

pub fn segment_one(model: &mut YoloModel, path: &Path, cfg: &SegConfig) -> Result<SegItem, String> {
    let img = image::open(path).map_err(|e| format!("open: {e}"))?.to_rgb8();
    let (ow, oh) = img.dimensions();
    let target = cfg.imgsz;

    let (input, scale, padx, pady) = letterbox(&img, target);
    let (d0, d1, dims) = model.infer(input, target)?;

    let ndet  = dims[0];                 // 300
    let nattr = dims[1];                 // 38 = 4 bbox + score + cls + 32 coeffs
    let nproto = dims[2];                // 32
    let ph = dims[3];                    // 160
    let pw = dims[4];                    // 160
    let coeff_off = nattr - nproto;      // 6
    // grid cell -> letterbox pixel scale (target / proto side)
    let gscale = target as f32 / pw as f32;

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("img").to_string();
    let mut instances = Vec::new();

    for r in 0..ndet {
        let base = r * nattr;
        let score = d0[base + 4];
        if score < cfg.conf {
            continue;
        }
        // bbox in letterbox space -> original coords
        let lx1 = d0[base];
        let ly1 = d0[base + 1];
        let lx2 = d0[base + 2];
        let ly2 = d0[base + 3];
        let ox1 = (((lx1 - padx as f32) / scale).max(0.0)).min(ow as f32 - 1.0);
        let oy1 = (((ly1 - pady as f32) / scale).max(0.0)).min(oh as f32 - 1.0);
        let ox2 = (((lx2 - padx as f32) / scale).max(0.0)).min(ow as f32 - 1.0);
        let oy2 = (((ly2 - pady as f32) / scale).max(0.0)).min(oh as f32 - 1.0);
        let bx = ox1.floor() as u32;
        let by = oy1.floor() as u32;
        let bw = ((ox2 - ox1).round() as u32).max(1).min(ow - bx);
        let bh = ((oy2 - oy1).round() as u32).max(1).min(oh - by);

        // prototype mask coefficients for this instance
        let coeffs = &d0[base + coeff_off..base + coeff_off + nproto];

        // Compute the proto mask at grid resolution ONCE, then BILINEARLY sample
        // it back to original pixels -> smooth margins instead of blocky 4-px
        // cells. The boolean mask (>=0.5) drives detection; a soft alpha derived
        // from the same value anti-aliases the cutout edge.
        let mut pm = vec![0f32; ph * pw];
        for (c, slot) in pm.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for k in 0..nproto {
                acc += coeffs[k] * d1[k * ph * pw + c];
            }
            *slot = sigmoid(acc);
        }

        // ── Background-colour estimate (for edge cleanup) ─────────────────────
        // Scanner cutouts sit on a ~uniform background. Sample clearly-OUTSIDE
        // pixels (low proto value) to estimate that colour; boundary pixels that
        // match it are soft-mask bleed and get dropped below. Coarse stride keeps
        // it cheap even on multi-thousand-px leaves.
        let (mut sum_r, mut sum_g, mut sum_b, mut bg_n) = (0u64, 0u64, 0u64, 0u64);
        let step = (bw.max(bh) / 128).max(1);
        let mut sy = 0u32;
        while sy < bh {
            let oy = (by + sy) as f32;
            let gyf = (oy * scale + pady as f32) / gscale - 0.5;
            let mut sx = 0u32;
            while sx < bw {
                let ox = (bx + sx) as f32;
                let gxf = (ox * scale + padx as f32) / gscale - 0.5;
                if sample_bilinear(&pm, pw, ph, gxf, gyf) < 0.25 {
                    let p = img.get_pixel(bx + sx, by + sy);
                    sum_r += p[0] as u64; sum_g += p[1] as u64; sum_b += p[2] as u64; bg_n += 1;
                }
                sx += step;
            }
            sy += step;
        }
        let has_bg = bg_n > 16;
        let bg = if has_bg {
            [(sum_r / bg_n) as i32, (sum_g / bg_n) as i32, (sum_b / bg_n) as i32]
        } else { [0, 0, 0] };
        const BG_INTERIOR_V: f32 = 0.85;      // v above this = deep leaf, never colour-rejected
        const BG_COLOR_DIST2: i32 = 45 * 45;  // reject boundary px within RGB dist 45 of background

        let mut mask = vec![0u8; (bw * bh) as usize];
        let mut alpha = vec![0u8; (bw * bh) as usize];
        let mut v_grid = vec![0f32; (bw * bh) as usize];
        // Background-colour CANDIDATE: this pixel looks like background (either
        // truly outside the YOLO mask, or a boundary pixel matching the estimated
        // bg colour/chroma). Whether it's actually dropped is decided below, after
        // we know which candidates are connected to the real background — see the
        // flood-fill comment.
        let mut bg_candidate = vec![false; (bw * bh) as usize];

        for yy in 0..bh {
            let oy = (by + yy) as f32;
            let gyf = (oy * scale + pady as f32) / gscale - 0.5; // align_corners=false
            for xx in 0..bw {
                let ox = (bx + xx) as f32;
                let gxf = (ox * scale + padx as f32) / gscale - 0.5;
                let v = sample_bilinear(&pm, pw, ph, gxf, gyf);
                let i = (yy * bw + xx) as usize;
                v_grid[i] = v;
                if v >= 0.5 {
                    mask[i] = 1;
                } else {
                    bg_candidate[i] = true; // definitely non-leaf per YOLO's own mask
                }
                // Alpha feather sits ENTIRELY at/above the 0.5 mask boundary so no
                // sub-boundary background can bleed into the cutout edge. The old
                // [0.45,0.55] feather dipped 0.05 BELOW 0.5 → a faint background rim
                // (YOLO's soft mask edge). [0.50,0.60] keeps the interior crisp and
                // drops everything below the boundary. Raise CUTOUT_ALPHA_LO toward
                // 0.55–0.60 for an even tighter cut (at the risk of nibbling soft
                // leaf edges); lower it back toward 0.50 if it eats real margin.
                alpha[i] = (((v - cfg.alpha_lo) / 0.10).clamp(0.0, 1.0) * 255.0).round() as u8;

                // Background-colour edge cleanup: a boundary pixel (not deep leaf)
                // whose colour matches the estimated background is soft-mask bleed —
                // flag it as a CANDIDATE for dropping. Deep interior (v ≥ 0.85) is
                // never colour-rejected, so a pale leaf region that happens to match
                // the background is safe regardless.
                if alpha[i] > 0 && v < BG_INTERIOR_V {
                    let p = img.get_pixel(bx + xx, by + yy);
                    let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
                    // Chroma = achromatic-ness. ALL scanner backgrounds — white, grey,
                    // black, AND the shadowed off-white ring next to the leaf — have
                    // R≈G≈B (low chroma), while leaf tissue is green/brown (chromatic).
                    // So a low-chroma boundary pixel is background regardless of its
                    // brightness/shadow — this catches the shadowed rim a fixed-colour
                    // match misses, and works for any background shade.
                    let chroma = r.max(g).max(b) - r.min(g).min(b);
                    let near_bg = has_bg && {
                        let (dr, dg, db) = (r - bg[0], g - bg[1], b - bg[2]);
                        dr * dr + dg * dg + db * db < BG_COLOR_DIST2
                    };
                    if chroma < cfg.chroma_min || near_bg {
                        bg_candidate[i] = true;
                    }
                }
            }
        }

        // Only actually drop background-colour candidates that are CONNECTED to
        // the real background at the crop border (4-connected flood fill). A true
        // boundary-bleed pixel always touches the actual background; an isolated
        // interior pixel that merely resembles it in colour (e.g. deep shadow
        // along the midrib on a dark backdrop) does not — and must not be treated
        // as if it were background. Same topological-enclosure principle already
        // used for the Pipeline's hole detector (`worker.rs::reconstruct_area`).
        let reachable = flood_from_border(&bg_candidate, bw as usize, bh as usize);
        let mut leaf_px = 0u32;
        for i in 0..(bw * bh) as usize {
            if v_grid[i] < BG_INTERIOR_V && reachable[i] {
                alpha[i] = 0;
                mask[i] = 0;
            }
            if mask[i] == 1 { leaf_px += 1; }
        }
        if leaf_px == 0 {
            continue;
        }

        // ── drop speckle: keep only components comparable to the main body ──
        // An instance mask is sigmoid(coeffs · protos) CROPPED TO ITS BBOX. The
        // prototype field is global, so it can read high well away from the leaf
        // this detection is about — and in a crowded scene bounding boxes overlap
        // heavily, so that stray response lands INSIDE a neighbouring leaf. The
        // result is exactly what shows up on dense photos: confetti of one leaf's
        // colour sprinkled across another, plus straight-edged rectangular
        // patches where a high region was sliced off at the bbox border.
        //
        // The background-colour cleanup above cannot catch any of it: those
        // pixels ARE leaf-coloured (they sit on a real leaf), and they never
        // touch the crop border, so the flood fill never reaches them.
        //
        // A leaf is one blob. Anything much smaller than the main component is
        // prototype noise. Small components are dropped RELATIVE to the largest
        // rather than by absolute size, so a genuinely split leaf — one occluded
        // into two substantial halves — keeps both, while confetti disappears.
        {
            use crate::tabs::mask_tools::mask_connected_components;
            let bools: Vec<bool> = mask.iter().map(|&m| m == 1).collect();
            let comps = mask_connected_components(&bools, bw, bh);
            if comps.len() > 1 {
                let biggest = comps.iter().map(|c| c.len()).max().unwrap_or(0);
                let floor = ((biggest as f32 * SPECKLE_FRAC) as usize).max(SPECKLE_MIN_PX);
                let mut dropped = 0usize;
                for comp in &comps {
                    if comp.len() >= floor {
                        continue;
                    }
                    for &i in comp {
                        mask[i] = 0;
                        alpha[i] = 0;
                        dropped += 1;
                    }
                }
                leaf_px = leaf_px.saturating_sub(dropped as u32);
                if leaf_px == 0 {
                    continue;
                }
            }
        }

        // write cutout (RGBA, soft anti-aliased alpha)
        let cutout_path = cfg.output_dir.join(format!("{stem}_leaf{:02}.png", instances.len()));
        let cutout = build_cutout(&img, bx, by, bw, bh, &alpha);
        cutout.save(&cutout_path).map_err(|e| format!("save cutout: {e}"))?;

        instances.push(SegInstance {
            bbox: [bx, by, bw, bh],
            score,
            mask,
            cutout_path,
        });
    }

    Ok(SegItem {
        filename: path.file_name().and_then(|s| s.to_str()).unwrap_or("img").to_string(),
        path: path.to_path_buf(),
        size: [ow, oh],
        instances,
    })
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Aspect-preserving resize to `target`×`target` with centred grey padding.
/// Returns (CHW f32 RGB 0–1, scale, pad_x, pad_y).
fn letterbox(img: &RgbImage, target: u32) -> (Vec<f32>, f32, u32, u32) {
    let (w, h) = img.dimensions();
    let scale = (target as f32 / w as f32).min(target as f32 / h as f32);
    let nw = ((w as f32 * scale).round() as u32).max(1);
    let nh = ((h as f32 * scale).round() as u32).max(1);
    let resized = image::imageops::resize(img, nw, nh, FilterType::Triangle);
    let padx = (target - nw) / 2;
    let pady = (target - nh) / 2;

    let n = (target * target) as usize;
    let pad = 114.0f32 / 255.0;
    let mut data = vec![pad; 3 * n];
    for y in 0..nh {
        for x in 0..nw {
            let px = resized.get_pixel(x, y);
            let idx = ((pady + y) * target + (padx + x)) as usize;
            data[idx] = px[0] as f32 / 255.0;
            data[n + idx] = px[1] as f32 / 255.0;
            data[2 * n + idx] = px[2] as f32 / 255.0;
        }
    }
    (data, scale, padx, pady)
}

/// 4-connected flood fill seeded from every `passable` cell on the border of a
/// `w`×`h` grid, returning which `passable` cells are reachable from the
/// border. Used to tell real background bleed (touches the actual background
/// at the crop edge) apart from an isolated interior pixel that only
/// resembles background in colour but is fully surrounded by leaf tissue.
fn flood_from_border(passable: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut reachable = vec![false; w * h];
    if w == 0 || h == 0 { return reachable; }
    let mut queue: VecDeque<usize> = VecDeque::new();
    let idx = |x: usize, y: usize| y * w + x;

    let mut seed = |i: usize, queue: &mut VecDeque<usize>| {
        if passable[i] && !reachable[i] {
            reachable[i] = true;
            queue.push_back(i);
        }
    };
    for x in 0..w {
        seed(idx(x, 0), &mut queue);
        seed(idx(x, h - 1), &mut queue);
    }
    for y in 0..h {
        seed(idx(0, y), &mut queue);
        seed(idx(w - 1, y), &mut queue);
    }

    while let Some(i) = queue.pop_front() {
        let (x, y) = (i % w, i / w);
        if x > 0     { seed(idx(x - 1, y), &mut queue); }
        if x + 1 < w { seed(idx(x + 1, y), &mut queue); }
        if y > 0     { seed(idx(x, y - 1), &mut queue); }
        if y + 1 < h { seed(idx(x, y + 1), &mut queue); }
    }
    reachable
}

fn build_cutout(img: &RgbImage, bx: u32, by: u32, bw: u32, bh: u32, alpha: &[u8]) -> RgbaImage {
    let mut out = RgbaImage::new(bw, bh);
    for yy in 0..bh {
        for xx in 0..bw {
            let a = alpha[(yy * bw + xx) as usize];
            let src = img.get_pixel(bx + xx, by + yy);
            out.put_pixel(xx, yy, image::Rgba([src[0], src[1], src[2], a]));
        }
    }
    out
}

/// Bilinear sample of a `w`×`h` grid at fractional (x, y) (clamped to bounds).
fn sample_bilinear(m: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, w as f32 - 1.0);
    let y = y.clamp(0.0, h as f32 - 1.0);
    let x0 = x.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y0 = y.floor() as usize;
    let y1 = (y0 + 1).min(h - 1);
    let wx = x - x0 as f32;
    let wy = y - y0 as f32;
    let top = m[y0 * w + x0] * (1.0 - wx) + m[y0 * w + x1] * wx;
    let bot = m[y1 * w + x0] * (1.0 - wx) + m[y1 * w + x1] * wx;
    top * (1.0 - wy) + bot * wy
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Headless smoke test of the real ort path against the exported model.
    /// Run with: cargo test --no-default-features seg_smoke -- --ignored --nocapture
    #[test]
    #[ignore]
    fn seg_smoke() {
        let model = PathBuf::from(
            r"E:\PhD_TobiMu\02_code\02paper\leaf_segmentation\runs\segment\runs\segment\leaf_seg\weights\best.onnx",
        );
        let image = PathBuf::from(
            r"E:\PhD_TobiMu\02_code\02paper\leaf_segmentation\test\8f6a70ebb6ae0c10090b5ac3ec98e837.jpg",
        );
        let out = std::env::temp_dir().join("leaf_seg_smoke");
        assert!(model.exists(), "model missing: {}", model.display());
        assert!(image.exists(), "image missing: {}", image.display());

        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        spawn_segmentation(
            SegConfig {
                model_path: model,
                image_paths: vec![image],
                output_dir: out.clone(),
                imgsz: 1280, // see settings.rs::FieldReviewSettings::default for the measurement
                conf: 0.25,
                alpha_lo: 0.50,
                chroma_min: 28,
            },
            tx,
            cancel,
        );

        let (mut items, mut total) = (0usize, 0usize);
        while let Ok(msg) = rx.recv() {
            match msg {
                SegMsg::Item(it) => {
                    println!("  {} -> {} leaves (size {:?})", it.filename, it.instances.len(), it.size);
                    items += 1;
                    total += it.instances.len();
                }
                SegMsg::Log(l) => println!("  log: {l}"),
                SegMsg::Error(e) => panic!("segmentation error: {e}"),
                SegMsg::Finished => break,
                SegMsg::Progress { .. } => {}
            }
        }
        println!("SMOKE: {items} image(s), {total} instance(s); cutouts -> {}", out.display());
        assert!(items >= 1, "no items produced");
    }
}
