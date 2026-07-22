//! Split a segmented leaf (RGBA cutout) into square tiles for anomaly detection,
//! and restitch per-tile masks back into leaf coordinates.
//!
//! v1 uses non-overlapping tiles (overlap 0 -> trivial restitch). Each tile keeps
//! a `valid` mask (alpha > threshold) so transparent border/out-of-image pixels
//! are excluded from the per-tile robust-z statistics and the final mask. The
//! tile RGB has transparent pixels filled with the tile's mean valid colour (a
//! cheap stand-in for the inpainting the Python bank was built on, so DINO
//! features near the leaf edge aren't perturbed by hard transparency).

use image::RgbImage;

pub struct LeafTile {
    pub origin: [u32; 2], // (x, y) top-left in leaf coords
    pub size:   u32,      // tile side
    pub rgb:    RgbImage, // size×size, transparent filled with mean valid colour
    pub valid:  Vec<bool>, // size*size, alpha > threshold
}

/// Tile a leaf RGBA buffer into `tile`×`tile` cells covering the whole image
/// (the last row/col may extend past the edge -> those pixels are invalid).
pub fn tile_leaf(rgba: &[u8], w: u32, h: u32, tile: u32, alpha_thr: u8) -> Vec<LeafTile> {
    let cols = (w + tile - 1) / tile;
    let rows = (h + tile - 1) / tile;
    let mut out = Vec::with_capacity((cols * rows) as usize);
    for ry in 0..rows {
        for rx in 0..cols {
            let ox = rx * tile;
            let oy = ry * tile;
            let n = (tile * tile) as usize;
            let mut rgb = vec![0u8; n * 3];
            let mut valid = vec![false; n];
            // first pass: copy in-bounds pixels, record valid + colour sum
            let (mut sr, mut sg, mut sb, mut cnt) = (0u64, 0u64, 0u64, 0u64);
            for ty in 0..tile {
                for tx in 0..tile {
                    let gx = ox + tx;
                    let gy = oy + ty;
                    let ti = (ty * tile + tx) as usize;
                    if gx >= w || gy >= h {
                        continue;
                    }
                    let gi = ((gy * w + gx) * 4) as usize;
                    let (r, g, b, a) = (rgba[gi], rgba[gi + 1], rgba[gi + 2], rgba[gi + 3]);
                    rgb[ti * 3] = r;
                    rgb[ti * 3 + 1] = g;
                    rgb[ti * 3 + 2] = b;
                    if a > alpha_thr {
                        valid[ti] = true;
                        sr += r as u64;
                        sg += g as u64;
                        sb += b as u64;
                        cnt += 1;
                    }
                }
            }
            // second pass: fill invalid pixels with mean valid colour
            if cnt > 0 {
                let (mr, mg, mb) = ((sr / cnt) as u8, (sg / cnt) as u8, (sb / cnt) as u8);
                for ti in 0..n {
                    if !valid[ti] {
                        rgb[ti * 3] = mr;
                        rgb[ti * 3 + 1] = mg;
                        rgb[ti * 3 + 2] = mb;
                    }
                }
            }
            let img = RgbImage::from_raw(tile, tile, rgb).expect("tile rgb buffer");
            out.push(LeafTile { origin: [ox, oy], size: tile, rgb: img, valid });
        }
    }
    out
}

/// RGBA -> RGB for an already-square `win`×`win` buffer (e.g. a region context
/// crop from `worker::context_crop`), filling below-threshold-alpha pixels with
/// the mean colour of the above-threshold ones — same technique as `tile_leaf`'s
/// per-tile fill, but for a single complete buffer with no origin/partial-edge
/// bookkeeping (a context crop is always fully populated, just clamp-padded).
pub(crate) fn crop_to_rgb_meanfill(rgba: &[u8], win: u32, alpha_thr: u8) -> RgbImage {
    let n = (win * win) as usize;
    let mut rgb = vec![0u8; n * 3];
    let mut valid = vec![false; n];
    let (mut sr, mut sg, mut sb, mut cnt) = (0u64, 0u64, 0u64, 0u64);
    for i in 0..n {
        let (r, g, b, a) = (rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], rgba[i * 4 + 3]);
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
        if a > alpha_thr {
            valid[i] = true;
            sr += r as u64;
            sg += g as u64;
            sb += b as u64;
            cnt += 1;
        }
    }
    if cnt > 0 {
        let (mr, mg, mb) = ((sr / cnt) as u8, (sg / cnt) as u8, (sb / cnt) as u8);
        for i in 0..n {
            if !valid[i] {
                rgb[i * 3] = mr;
                rgb[i * 3 + 1] = mg;
                rgb[i * 3 + 2] = mb;
            }
        }
    }
    RgbImage::from_raw(win, win, rgb).expect("crop rgb buffer")
}

/// Erode the opaque region of an RGBA buffer inward by `radius` px (Chebyshev),
/// so the few-pixel background ring left by an imperfect cutout is excluded from
/// detection (mirrors the Python preprocessing's score-mask erosion). Separable:
/// two 1-D min passes -> O(w·h·radius). Sets alpha to 0 in the eroded band.
pub fn erode_alpha(rgba: &mut [u8], w: u32, h: u32, radius: u32, thr: u8) {
    if radius == 0 {
        return;
    }
    let (w, h, r) = (w as usize, h as usize, radius as usize);
    let mut valid: Vec<bool> = (0..w * h).map(|i| rgba[i * 4 + 3] > thr).collect();

    let mut tmp = vec![false; w * h]; // horizontal erosion
    for y in 0..h {
        for x in 0..w {
            let x0 = x.saturating_sub(r);
            let x1 = (x + r).min(w - 1);
            tmp[y * w + x] = (x0..=x1).all(|xx| valid[y * w + xx]);
        }
    }
    for x in 0..w {
        // vertical erosion of the horizontal result
        for y in 0..h {
            let y0 = y.saturating_sub(r);
            let y1 = (y + r).min(h - 1);
            valid[y * w + x] = (y0..=y1).all(|yy| tmp[yy * w + x]);
        }
    }
    for i in 0..w * h {
        if !valid[i] {
            rgba[i * 4 + 3] = 0;
        }
    }
}

/// Write per-tile boolean masks (tile-local, size*size) back into a leaf-sized
/// mask. `tiles` and `masks` are parallel.
pub fn restitch_mask(tiles: &[LeafTile], masks: &[Vec<bool>], w: u32, h: u32) -> Vec<bool> {
    let mut out = vec![false; (w * h) as usize];
    for (t, m) in tiles.iter().zip(masks.iter()) {
        let [ox, oy] = t.origin;
        let s = t.size;
        for ty in 0..s {
            for tx in 0..s {
                if !m[(ty * s + tx) as usize] {
                    continue;
                }
                let gx = ox + tx;
                let gy = oy + ty;
                if gx < w && gy < h {
                    out[(gy * w + gx) as usize] = true;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiling_grid_valid_and_restitch() {
        // 3×3 RGBA, all opaque except bottom-right pixel transparent
        let w = 3u32;
        let h = 3u32;
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            rgba[i * 4] = 10;
            rgba[i * 4 + 1] = 20;
            rgba[i * 4 + 2] = 30;
            rgba[i * 4 + 3] = 255;
        }
        rgba[(8 * 4 + 3) as usize] = 0; // pixel (2,2) transparent

        let tiles = tile_leaf(&rgba, w, h, 2, 10);
        assert_eq!(tiles.len(), 4, "ceil(3/2)^2 = 4 tiles");
        assert_eq!(tiles[0].origin, [0, 0]);
        assert_eq!(tiles[3].origin, [2, 2]);

        // tile 0 (covers 0,0..1,1) fully in-bounds + opaque -> all valid
        assert!(tiles[0].valid.iter().all(|&v| v));
        // tile 3 (origin 2,2): only pixel (2,2) in-bounds, and it's transparent -> none valid
        assert!(tiles[3].valid.iter().all(|&v| !v));

        // restitch: flag tile-0 local pixel (0,0) -> leaf (0,0)
        let mut masks: Vec<Vec<bool>> = tiles.iter().map(|t| vec![false; (t.size * t.size) as usize]).collect();
        masks[0][0] = true;
        let leaf = restitch_mask(&tiles, &masks, w, h);
        assert!(leaf[0]);
        assert_eq!(leaf.iter().filter(|&&b| b).count(), 1);
    }
}
