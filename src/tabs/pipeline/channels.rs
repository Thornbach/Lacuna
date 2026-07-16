//! Pixel-level anomaly channels (ports of 1Help/residual.py), pure Rust.
//!
//! * `residual_map`  — |pixel − local_median(ksize)|, max over RGB. Native-res
//!   spiky signal; the small-defect specialist. Median via a sliding 256-bin
//!   histogram (Huang) so it is O(w·h·k), not O(w·h·k²).
//! * `color_deviation_map` — CIELAB (a,b) chroma distance from the tile's median
//!   colour (lightness ignored). Catches necrosis / discolouration as solid
//!   blobs. Real CIELAB a,b units, so deviations match OpenCV's LAB up to
//!   quantization (the in-app trainer re-calibrates against this exact impl).

use image::RgbImage;
use rayon::prelude::*;

pub fn residual_map(rgb: &RgbImage, valid: &[bool], ksize: usize) -> Vec<f32> {
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let k = if ksize % 2 == 0 { ksize + 1 } else { ksize };
    let r = k / 2;
    let raw = rgb.as_raw(); // w*h*3
    let mut out = vec![0f32; w * h];
    for c in 0..3 {
        let ch: Vec<u8> = (0..w * h).map(|i| raw[i * 3 + c]).collect();
        let med = median_blur_u8(&ch, w, h, r);
        for i in 0..w * h {
            let d = (ch[i] as f32 - med[i] as f32).abs();
            if d > out[i] {
                out[i] = d;
            }
        }
    }
    for i in 0..w * h {
        if !valid[i] {
            out[i] = 0.0;
        }
    }
    out
}

fn median_blur_u8(ch: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    let k = 2 * r + 1;
    let half = (k * k) / 2; // window is odd -> middle index
    let rows: Vec<Vec<u8>> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut hist = [0u32; 256];
            // init window at x = 0 (columns -r..=r, rows y-r..=y+r, clamped)
            for dx in -(r as isize)..=(r as isize) {
                let xx = dx.clamp(0, w as isize - 1) as usize;
                for dy in -(r as isize)..=(r as isize) {
                    let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                    hist[ch[yy * w + xx] as usize] += 1;
                }
            }
            let mut row = vec![0u8; w];
            for x in 0..w {
                let mut cum = 0u32;
                for (v, &count) in hist.iter().enumerate() {
                    cum += count;
                    if cum as usize > half {
                        row[x] = v as u8;
                        break;
                    }
                }
                if x + 1 < w {
                    let xrem = (x as isize - r as isize).clamp(0, w as isize - 1) as usize;
                    let xadd = (x as isize + r as isize + 1).clamp(0, w as isize - 1) as usize;
                    for dy in -(r as isize)..=(r as isize) {
                        let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                        hist[ch[yy * w + xrem] as usize] -= 1;
                        hist[ch[yy * w + xadd] as usize] += 1;
                    }
                }
            }
            row
        })
        .collect();
    rows.concat()
}

/// CIELAB a,b channels + the per-tile median (a,b) over valid pixels. Reused by
/// both the color channel and the clustering descriptor (color direction).
pub fn lab_ab(rgb: &RgbImage, valid: &[bool]) -> (Vec<f32>, Vec<f32>, f32, f32) {
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let raw = rgb.as_raw();
    let mut a = vec![0f32; w * h];
    let mut b = vec![0f32; w * h];
    for i in 0..w * h {
        let (_l, ai, bi) = rgb_to_lab(raw[i * 3], raw[i * 3 + 1], raw[i * 3 + 2]);
        a[i] = ai;
        b[i] = bi;
    }
    let mut av: Vec<f32> = (0..w * h).filter(|&i| valid[i]).map(|i| a[i]).collect();
    let mut bv: Vec<f32> = (0..w * h).filter(|&i| valid[i]).map(|i| b[i]).collect();
    let (ma, mb) = if av.is_empty() { (0.0, 0.0) } else { (median(&mut av), median(&mut bv)) };
    (a, b, ma, mb)
}

pub fn color_deviation_map(rgb: &RgbImage, valid: &[bool]) -> Vec<f32> {
    let (a, b, ma, mb) = lab_ab(rgb, valid);
    let n = a.len();
    let mut out = vec![0f32; n];
    for i in 0..n {
        if valid[i] {
            out[i] = ((a[i] - ma).powi(2) + (b[i] - mb).powi(2)).sqrt();
        }
    }
    out
}

fn rgb_to_lab(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let lin = |c: u8| {
        let c = c as f32 / 255.0;
        if c > 0.04045 {
            ((c + 0.055) / 1.055).powf(2.4)
        } else {
            c / 12.92
        }
    };
    let (r, g, b) = (lin(r), lin(g), lin(b));
    let x = r * 0.4124 + g * 0.3576 + b * 0.1805;
    let y = r * 0.2126 + g * 0.7152 + b * 0.0722;
    let z = r * 0.0193 + g * 0.1192 + b * 0.9505;
    let (xn, yn, zn) = (0.95047f32, 1.0f32, 1.08883f32);
    let f = |t: f32| if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 };
    let (fx, fy, fz) = (f(x / xn), f(y / yn), f(z / zn));
    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

/// Exposed so callers can compute a median over a STITCHED (e.g. full-leaf,
/// rather than per-tile) set of values — see `detect::decide_global`.
pub(crate) fn median(v: &mut [f32]) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn residual_fires_on_speck_color_on_discolouration() {
        let (w, h) = (64u32, 64u32);
        let mut img = RgbImage::from_pixel(w, h, Rgb([60, 140, 70])); // uniform green
        // a bright speck (high local residual)
        img.put_pixel(32, 32, Rgb([255, 255, 255]));
        // a brown discoloured patch (high colour deviation, low local residual interior)
        for y in 10..16 {
            for x in 10..16 {
                img.put_pixel(x, y, Rgb([120, 80, 40]));
            }
        }
        let valid = vec![true; (w * h) as usize];

        let res = residual_map(&img, &valid, 9);
        let col = color_deviation_map(&img, &valid);
        let idx = |x: u32, y: u32| (y * w + x) as usize;

        // residual: speck >> uniform background
        assert!(res[idx(32, 32)] > 50.0, "speck residual {}", res[idx(32, 32)]);
        assert!(res[idx(50, 50)] < 5.0, "bg residual {}", res[idx(50, 50)]);
        // colour: brown patch centre >> green background
        assert!(col[idx(12, 12)] > 15.0, "brown color dev {}", col[idx(12, 12)]);
        assert!(col[idx(50, 50)] < 5.0, "bg color dev {}", col[idx(50, 50)]);
    }
}
