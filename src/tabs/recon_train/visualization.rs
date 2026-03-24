/// Build a 4-row × 5-col sample grid image (RGBA flat bytes, row-major).
///
/// Each row corresponds to one sample:
///   Col 0: Damaged input (RGBA composite on bg)
///   Col 1: Ground-truth mask  (white leaf on bg)
///   Col 2: Predicted mask     (thresholded at 0.5, white leaf on bg)
///   Col 3: Diagnostic overlay (TP/FP/FN/TN colours + damaged input @ 40% opacity)
///   Col 4: Reconstruction confidence heatmap
///          — In the reconstruction zone (GT=1, input alpha=0): jet heatmap of
///            model probability (blue=0→red=1), showing where the model is
///            confident vs. uncertain about the missing leaf area.
///          — Intact pixels (input alpha > 0): shown as normal damaged input.
///          — Background (GT=0): shown as bg colour.
///
/// Arguments:
///   `samples`  — up to 4 tuples of (damaged_rgba_bytes, gt_f32, pred_f32, saliency_f32)
///                at `src_size × src_size` resolution.
///                saliency_f32: |∂output/∂input| over RGBA channels, normalised 0..1.
///   `panel_px` — output panel size (each panel is this many pixels square)
///   `bg`       — RGB background colour
///   `overlay`  — opacity of damaged input overlay in col 3 (0.0–1.0)
pub fn build_sample_grid(
    samples:   &[(Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>)],
    panel_px:  usize,
    bg:        [u8; 3],
    overlay:   f32,
) -> (Vec<u8>, usize, usize) {
    let rows    = samples.len().min(4);
    let cols    = 6;
    let w       = cols * panel_px;
    let h       = rows * panel_px;
    let mut out = vec![0u8; w * h * 4];

    for (row, (damaged, gt, pred, saliency)) in samples.iter().take(4).enumerate() {
        // Infer source resolution from buffer lengths
        let n_px  = gt.len();
        let src   = (n_px as f64).sqrt() as usize;   // assume square

        for col in 0..6 {
            for py in 0..panel_px {
                for px in 0..panel_px {
                    // Map panel pixel → source pixel (nearest-neighbour)
                    let sx = px * src / panel_px;
                    let sy = py * src / panel_px;
                    let si = sy * src + sx;

                    let pixel = compose_pixel(
                        col, si,
                        damaged, gt, pred, saliency,
                        bg, overlay,
                    );

                    let dst = ((row * panel_px + py) * w + col * panel_px + px) * 4;
                    out[dst]     = pixel[0];
                    out[dst + 1] = pixel[1];
                    out[dst + 2] = pixel[2];
                    out[dst + 3] = pixel[3];
                }
            }
        }
    }

    (out, w, h)
}

#[inline]
fn compose_pixel(
    col:      usize,
    si:       usize,
    damaged:  &[u8],
    gt:       &[f32],
    pred:     &[f32],
    saliency: &[f32],
    bg:       [u8; 3],
    overlay:  f32,
) -> [u8; 4] {
    match col {
        // ── Col 0: Damaged input composite on background ──────────────────────
        0 => {
            let r = damaged[si * 4];
            let g = damaged[si * 4 + 1];
            let b = damaged[si * 4 + 2];
            let a = damaged[si * 4 + 3];
            let t = a as f32 / 255.0;
            [
                lerp_u8(bg[0], r, t),
                lerp_u8(bg[1], g, t),
                lerp_u8(bg[2], b, t),
                255,
            ]
        }

        // ── Col 1: Ground-truth mask ──────────────────────────────────────────
        1 => {
            let v = if gt[si] > 0.5 { 240u8 } else { 0 };
            let bg_part = if gt[si] > 0.5 { 1.0 } else { 0.0 };
            let r = lerp_u8(bg[0], v, bg_part);
            let g = lerp_u8(bg[1], v, bg_part);
            let b = lerp_u8(bg[2], v, bg_part);
            [r, g, b, 255]
        }

        // ── Col 2: Predicted mask (threshold 0.5) ─────────────────────────────
        2 => {
            let v = if pred[si] > 0.5 { 240u8 } else { 0 };
            let fg = if pred[si] > 0.5 { 1.0 } else { 0.0 };
            [
                lerp_u8(bg[0], v, fg),
                lerp_u8(bg[1], v, fg),
                lerp_u8(bg[2], v, fg),
                255,
            ]
        }

        // ── Col 5: Gradient saliency ──────────────────────────────────────────
        // Shows |∂output/∂input| for intact leaf pixels only.
        // High value (red) = model relied on this pixel as a reconstruction cue.
        // Eroded region shown as dim neutral (the area the model had to complete).
        // Background pixels shown as bg colour.
        5 => {
            let gt_leaf = gt[si]              > 0.5;
            let intact  = damaged[si * 4 + 3] > 128;

            if !gt_leaf {
                [bg[0], bg[1], bg[2], 255]
            } else if intact {
                // Intact leaf pixel: jet heatmap of saliency
                let [r, g, b] = jet_colormap(saliency[si]);
                [r, g, b, 255]
            } else {
                // Reconstruction zone: dark neutral — model had no info here
                [45, 45, 55, 255]
            }
        }

        // ── Col 4: Reconstruction confidence heatmap ──────────────────────────
        4 => {
            let gt_leaf    = gt[si]              > 0.5;
            let intact     = damaged[si * 4 + 3] > 128;

            if !gt_leaf {
                // Outside leaf: background
                [bg[0], bg[1], bg[2], 255]
            } else if intact {
                // Leaf pixel that was not eroded: show original input normally
                let r = damaged[si * 4];
                let g = damaged[si * 4 + 1];
                let b = damaged[si * 4 + 2];
                let a = damaged[si * 4 + 3];
                let t = a as f32 / 255.0;
                [lerp_u8(bg[0], r, t), lerp_u8(bg[1], g, t), lerp_u8(bg[2], b, t), 255]
            } else {
                // Reconstruction zone: jet heatmap of model probability
                let [r, g, b] = jet_colormap(pred[si]);
                [r, g, b, 255]
            }
        }

        // ── Col 3: Diagnostic (TP/FP/FN/TN) + damaged overlay ────────────────
        _ => {
            let p_bin = pred[si] > 0.5;
            let g_bin = gt[si]   > 0.5;

            // Base colour by category
            let base: [u8; 3] = match (p_bin, g_bin) {
                (true,  true)  => [0,   200,  80],   // TP – green
                (true,  false) => [220,  40,  40],   // FP – red
                (false, true)  => [230, 130,  20],   // FN – orange
                (false, false) => bg,                 // TN – background
            };

            // Blend damaged input RGBA on top at `overlay` opacity
            let dr = damaged[si * 4];
            let dg = damaged[si * 4 + 1];
            let db = damaged[si * 4 + 2];
            let da = damaged[si * 4 + 3];

            // Only show the damaged overlay where there's actual leaf content
            let alpha = (da as f32 / 255.0) * overlay;

            [
                lerp_u8(base[0], dr, alpha),
                lerp_u8(base[1], dg, alpha),
                lerp_u8(base[2], db, alpha),
                255,
            ]
        }
    }
}

#[inline]
fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    let t = t.clamp(0.0, 1.0);
    (a as f32 + (b as f32 - a as f32) * t) as u8
}

/// Jet colormap: blue (p=0) → cyan → green → yellow → red (p=1).
/// Used for the reconstruction confidence heatmap (col 4).
#[inline]
fn jet_colormap(p: f32) -> [u8; 3] {
    let p = p.clamp(0.0, 1.0);
    // Four segments: [0,0.25], [0.25,0.5], [0.5,0.75], [0.75,1.0]
    let (r, g, b) = if p < 0.25 {
        let t = p / 0.25;
        (0.0, t, 1.0)                      // blue → cyan
    } else if p < 0.5 {
        let t = (p - 0.25) / 0.25;
        (0.0, 1.0, 1.0 - t)               // cyan → green
    } else if p < 0.75 {
        let t = (p - 0.5) / 0.25;
        (t, 1.0, 0.0)                      // green → yellow
    } else {
        let t = (p - 0.75) / 0.25;
        (1.0, 1.0 - t, 0.0)               // yellow → red
    };
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}
