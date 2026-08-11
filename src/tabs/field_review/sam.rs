//! SAM (Segment Anything, ViT-B) via ONNX Runtime — Field Review's interactive
//! "click to segment" tool. Two-part inference, Meta's own documented split:
//! `sam_encoder.onnx` (image -> embedding, run ONCE per scene, expensive ViT
//! forward pass) and `sam_decoder.onnx` (embedding + point prompt -> mask,
//! run PER CLICK, cheap). That split is the whole point — encode a scene
//! once when it's opened, then every click is a fast decoder call.
//!
//! No BURN port exists for SAM (a windowed-attention ViT encoder with
//! relative position embeddings, plus a full prompt-encoder/mask-decoder
//! transformer — a bigger reimplementation project than DINOv3's own Burn
//! port was) — this is the one place in the app that unconditionally
//! depends on `ort` (see Cargo.toml's dependency comment). DINO/YOLO are
//! unaffected: they still default to BURN.
//!
//! Preprocessing verified directly against Meta's own `segment-anything`
//! Python source (not assumed — getting this silently wrong produces
//! garbage masks with no error):
//!   - `segment_anything/utils/transforms.py::ResizeLongestSide`: resize so
//!     the LONGEST side = 1024, preserving aspect ratio,
//!     `new = round(old * 1024/max(oldh,oldw))` (round-half-up).
//!   - `segment_anything/modeling/sam.py::Sam.preprocess`: normalize per RGB
//!     channel on the 0..255 scale (NOT 0..1): `(pixel - mean) / std`,
//!     `mean=[123.675,116.28,103.53]`, `std=[58.395,57.12,57.375]` — then
//!     pad to 1024×1024 with zeros in the BOTTOM/RIGHT only (image occupies
//!     the top-left corner, `F.pad(x, (0, padw, 0, padh))`).
//!   - Click coordinates get the SAME per-axis resize scale applied
//!     (`ResizeLongestSide.apply_coords`) — the resized-but-UNPADDED space,
//!     not raw photo pixels and not the padded 1024 canvas.
//!   - The exported decoder (built from `SamOnnxModel`) ALSO returns an
//!     already-postprocessed `masks` output (upscale, crop off padding,
//!     resize to `orig_im_size`, all baked into the ONNX graph) — but that
//!     path is NOT used here. `SamOnnxModel.mask_postprocessing` computes
//!     `prepadded_size` via `int(prepadded_size[0])`, a data-dependent
//!     Python `int()` cast on a tensor value. Under the legacy
//!     TorchScript-tracing exporter (`dynamo=False` — required at all,
//!     see `export_sam_onnx.py`'s comment, since the new exporter hard-
//!     errors on this exact pattern instead of silently miscompiling it),
//!     a trace-time `int()` cast freezes to whatever the EXPORT-TIME dummy
//!     `orig_im_size` was, not the real per-call value — confirmed as the
//!     actual root cause of a real, reported bug (mask visibly offset from
//!     the leaf for any photo whose aspect ratio differs from the export
//!     dummy's 1500x2250). Instead, this reads the decoder's `low_res_masks`
//!     output (the raw 256x256 mask logits, computed BEFORE
//!     `mask_postprocessing` — no `orig_im_size` dependency at all, so
//!     nothing here can be baked wrong) and reimplements
//!     `SamOnnxModel.mask_postprocessing` in plain Rust using the REAL
//!     per-photo `orig_im_size`/`resized` values computed at inference
//!     time: bilinear-upsample 256x256 -> the resized-but-unpadded crop
//!     size, then bilinear-resize that crop -> the native photo resolution
//!     (two `F.interpolate(..., align_corners=False)` calls, matched by
//!     `bilinear_sample` below).
//!   - A point-only prompt (no box) needs a padding point appended
//!     (coord (0,0), label -1) — Meta's own documented convention for
//!     `SamPredictor`-style point-only ONNX prompting. Applies once per
//!     call regardless of how many real points are given.

use std::path::Path;

use image::{imageops::FilterType, RgbaImage};
use ort::session::Session;
use ort::value::Tensor as OrtTensor;

const LOW_RES: u32 = 256;

const SAM_INPUT: u32 = 1024;
const PIXEL_MEAN: [f32; 3] = [123.675, 116.28, 103.53];
const PIXEL_STD: [f32; 3] = [58.395, 57.12, 57.375];

pub struct SamModel {
    encoder: Session,
    decoder: Session,
}

/// One scene's cached image embedding — computed once, reused for every
/// click on that scene until the user switches to a different one.
pub struct SamEmbedding {
    data: Vec<f32>, // 1*256*64*64, row-major
    /// The resized-but-UNPADDED dimensions (h, w) the embedding was computed
    /// from — needed to scale click coordinates into the same space the
    /// encoder actually saw (see `ResizeLongestSide.apply_coords` above).
    resized: (u32, u32),
    /// (h, w) of the source photo — the decoder's `orig_im_size`.
    orig: (u32, u32),
}

impl SamModel {
    pub fn load(encoder_path: &Path, decoder_path: &Path) -> Result<Self, String> {
        Ok(Self {
            encoder: crate::tabs::build_session(encoder_path)?,
            decoder: crate::tabs::build_session(decoder_path)?,
        })
    }

    /// Runs the (expensive) image encoder once. `rgba` is the scene's full
    /// native-resolution image.
    pub fn encode(&mut self, rgba: &RgbaImage) -> Result<SamEmbedding, String> {
        let (w, h) = (rgba.width(), rgba.height());
        let scale = SAM_INPUT as f32 / (w.max(h) as f32);
        let new_w = ((w as f32 * scale) + 0.5) as u32;
        let new_h = ((h as f32 * scale) + 0.5) as u32;

        let resized = image::imageops::resize(rgba, new_w.max(1), new_h.max(1), FilterType::Triangle);
        // 1024x1024 CHW canvas, zero-padded (image occupies the top-left corner).
        let plane = (SAM_INPUT * SAM_INPUT) as usize;
        let mut data = vec![0f32; 3 * plane];
        for y in 0..new_h.min(SAM_INPUT) {
            for x in 0..new_w.min(SAM_INPUT) {
                let px = resized.get_pixel(x, y).0;
                let idx = (y * SAM_INPUT + x) as usize;
                for c in 0..3 {
                    data[c * plane + idx] = (px[c] as f32 - PIXEL_MEAN[c]) / PIXEL_STD[c];
                }
            }
        }

        let input = OrtTensor::from_array(([1usize, 3, SAM_INPUT as usize, SAM_INPUT as usize], data))
            .map_err(|e| format!("sam encoder input tensor: {e}"))?;
        let outputs = self.encoder.run(ort::inputs!["input_image" => input])
            .map_err(|e| format!("sam encoder run: {e}"))?;
        let (_, emb) = outputs["image_embeddings"].try_extract_tensor::<f32>()
            .map_err(|e| format!("sam encoder extract: {e}"))?;

        Ok(SamEmbedding { data: emb.to_vec(), resized: (new_h, new_w), orig: (h, w) })
    }

    /// Runs the (cheap) decoder for a set of foreground/background point
    /// clicks — `points` is `(x, y, is_positive)` in the scene's own
    /// native-photo pixel coordinates (same space Field Review's other
    /// tools already use). All points refine the SAME mask (SAM's normal
    /// iterative-refinement convention), so callers accumulate clicks
    /// across a preview-then-commit cycle rather than calling this once
    /// per click in isolation. Returns a native-resolution `w*h`
    /// (row-major) boolean mask.
    pub fn click(&mut self, emb: &SamEmbedding, points: &[(f32, f32, bool)]) -> Result<Vec<bool>, String> {
        let (orig_h, orig_w) = emb.orig;
        let (new_h, new_w) = emb.resized;
        // Real points + one padding point (label -1) — required convention
        // for point-only (no box) prompting, see doc comment. Applies once
        // regardless of how many real points are given.
        let n = points.len() + 1;
        let mut coords = Vec::with_capacity(n * 2);
        let mut labels = Vec::with_capacity(n);
        for &(cx, cy, positive) in points {
            // Same per-axis scale the encoder's own resize used.
            coords.push(cx * (new_w as f32 / orig_w.max(1) as f32));
            coords.push(cy * (new_h as f32 / orig_h.max(1) as f32));
            labels.push(if positive { 1f32 } else { 0f32 });
        }
        coords.push(0.0);
        coords.push(0.0);
        labels.push(-1.0);

        let embed_t = OrtTensor::from_array(([1usize, 256, 64, 64], emb.data.clone()))
            .map_err(|e| format!("sam decoder embedding tensor: {e}"))?;
        let coords_t = OrtTensor::from_array(([1usize, n, 2], coords))
            .map_err(|e| format!("sam decoder coords tensor: {e}"))?;
        let labels_t = OrtTensor::from_array(([1usize, n], labels))
            .map_err(|e| format!("sam decoder labels tensor: {e}"))?;
        let mask_input_t = OrtTensor::from_array(([1usize, 1, 256, 256], vec![0f32; 256 * 256]))
            .map_err(|e| format!("sam decoder mask_input tensor: {e}"))?;
        let has_mask_t = OrtTensor::from_array(([1usize], vec![0f32]))
            .map_err(|e| format!("sam decoder has_mask tensor: {e}"))?;
        let orig_size_t = OrtTensor::from_array(([2usize], vec![orig_h as f32, orig_w as f32]))
            .map_err(|e| format!("sam decoder orig_size tensor: {e}"))?;

        let outputs = self.decoder.run(ort::inputs![
            "image_embeddings" => embed_t,
            "point_coords" => coords_t,
            "point_labels" => labels_t,
            "mask_input" => mask_input_t,
            "has_mask_input" => has_mask_t,
            "orig_im_size" => orig_size_t,
        ]).map_err(|e| format!("sam decoder run: {e}"))?;

        // Raw 256x256 mask logits, BEFORE the graph's own (export-time-
        // baked, see module doc comment) postprocessing — postprocessed by
        // us instead, correctly, using the real orig_h/orig_w/new_h/new_w.
        let (_, low_res) = outputs["low_res_masks"].try_extract_tensor::<f32>()
            .map_err(|e| format!("sam decoder extract: {e}"))?;
        Ok(postprocess_mask(low_res, new_h, new_w, orig_h, orig_w))
    }
}

/// Reimplements `SamOnnxModel.mask_postprocessing` (upsample 256x256 ->
/// SAM_INPUT canvas, crop to the resized-but-unpadded region, resize to the
/// native photo) directly in Rust with the real runtime sizes — see the
/// module doc comment for why this can't be trusted to the exported graph.
fn postprocess_mask(low_res: &[f32], new_h: u32, new_w: u32, orig_h: u32, orig_w: u32) -> Vec<bool> {
    // Stage 1: sample the 256x256 logits as if upsampled to the full
    // SAM_INPUT x SAM_INPUT canvas, but only for the pixels inside the
    // unpadded crop — equivalent to upsample-then-crop, without ever
    // materializing the full (and mostly-padding) 1024x1024 array.
    let mut cropped = vec![0f32; (new_h as usize) * (new_w as usize)];
    let scale_low = LOW_RES as f32 / SAM_INPUT as f32;
    for y in 0..new_h {
        let sy = (((y as f32 + 0.5) * scale_low) - 0.5).clamp(0.0, (LOW_RES - 1) as f32);
        for x in 0..new_w {
            let sx = (((x as f32 + 0.5) * scale_low) - 0.5).clamp(0.0, (LOW_RES - 1) as f32);
            cropped[(y * new_w + x) as usize] = bilinear_sample(low_res, LOW_RES, LOW_RES, sy, sx);
        }
    }
    // Stage 2: resize the unpadded crop to the native photo resolution.
    let mut out = vec![false; (orig_h as usize) * (orig_w as usize)];
    let scale_y = new_h as f32 / orig_h.max(1) as f32;
    let scale_x = new_w as f32 / orig_w.max(1) as f32;
    for y in 0..orig_h {
        let sy = (((y as f32 + 0.5) * scale_y) - 0.5).clamp(0.0, (new_h.max(1) - 1) as f32);
        for x in 0..orig_w {
            let sx = (((x as f32 + 0.5) * scale_x) - 0.5).clamp(0.0, (new_w.max(1) - 1) as f32);
            // SAM's own mask_threshold = 0.0 (Sam.mask_threshold) —
            // positive logit = foreground.
            out[(y * orig_w + x) as usize] = bilinear_sample(&cropped, new_h, new_w, sy, sx) > 0.0;
        }
    }
    out
}

/// `F.interpolate(..., mode="bilinear", align_corners=False)`-equivalent
/// single-sample lookup: `data` is `h*w` row-major, `(sy, sx)` already in
/// source-pixel-center coordinates (clamped to `[0, size-1]` by the caller).
fn bilinear_sample(data: &[f32], h: u32, w: u32, sy: f32, sx: f32) -> f32 {
    let y0 = sy.floor() as u32;
    let x0 = sx.floor() as u32;
    let y1 = (y0 + 1).min(h - 1);
    let x1 = (x0 + 1).min(w - 1);
    let (fy, fx) = (sy - y0 as f32, sx - x0 as f32);
    let v00 = data[(y0 * w + x0) as usize];
    let v01 = data[(y0 * w + x1) as usize];
    let v10 = data[(y1 * w + x0) as usize];
    let v11 = data[(y1 * w + x1) as usize];
    let top = v00 * (1.0 - fx) + v01 * fx;
    let bot = v10 * (1.0 - fx) + v11 * fx;
    top * (1.0 - fy) + bot * fy
}
