//! Mask → YOLO-format polygon export, replicating `generate_leaf_dataset.py`'s
//! `mask_to_yolo_polygon` exactly (same contour choice, same simplification
//! epsilon formula, same `%.5f`-normalized line format) so synthetic and
//! real-photo-corrected labels are numerically consistent with each other.

use imageproc::contours::{find_contours, BorderType};

/// Trace `mask` (bbox-local raster, `w*h`) to a single simplified polygon in
/// NORMALIZED (0..1) image coordinates, or `None` if no contour survives.
///
/// Mirrors the Python reference exactly: keep only the largest OUTER contour
/// (`cv2.RETR_EXTERNAL`'s equivalent — `BorderType::Outer`), reject areas
/// under 16px² (`cv2.contourArea(contour) < 16`), simplify with
/// Douglas-Peucker at `epsilon = 0.005 * perimeter` (`cv2.approxPolyDP`'s
/// exact formula), then normalize by `w`/`h`.
pub fn mask_to_polygon(mask: &[bool], w: u32, h: u32) -> Option<Vec<(f32, f32)>> {
    if w == 0 || h == 0 {
        return None;
    }
    let mut gray = image::GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            if mask[(y * w + x) as usize] {
                gray.put_pixel(x, y, image::Luma([255u8]));
            }
        }
    }

    let contours = find_contours::<i32>(&gray);
    let best = contours
        .into_iter()
        .filter(|c| c.border_type == BorderType::Outer && c.points.len() >= 3)
        .map(|c| {
            let pts: Vec<(f32, f32)> = c.points.iter().map(|p| (p.x as f32, p.y as f32)).collect();
            let area = shoelace_area(&pts);
            (area, pts)
        })
        .filter(|(area, _)| *area >= 16.0)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))?;

    let (_, pts) = best;
    let perimeter = closed_perimeter(&pts);
    let epsilon = (0.005 * perimeter).max(0.01);
    let simplified = douglas_peucker_closed(&pts, epsilon);
    if simplified.len() < 3 {
        return None;
    }

    Some(
        simplified
            .into_iter()
            .map(|(x, y)| (x / w as f32, y / h as f32))
            .collect(),
    )
}

/// `"{class_id} {x1:.5} {y1:.5} ..."` — one line, matching the Python
/// generator's exact `%.5f`-formatted output.
pub fn polygon_to_yolo_line(poly: &[(f32, f32)], class_id: u32) -> String {
    let coords: Vec<String> = poly.iter().flat_map(|(x, y)| [format!("{x:.5}"), format!("{y:.5}")]).collect();
    format!("{class_id} {}", coords.join(" "))
}

fn shoelace_area(pts: &[(f32, f32)]) -> f32 {
    if pts.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for i in 0..pts.len() {
        let (x1, y1) = pts[i];
        let (x2, y2) = pts[(i + 1) % pts.len()];
        sum += x1 * y2 - x2 * y1;
    }
    (sum / 2.0).abs()
}

fn closed_perimeter(pts: &[(f32, f32)]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..pts.len() {
        let (x1, y1) = pts[i];
        let (x2, y2) = pts[(i + 1) % pts.len()];
        sum += ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
    }
    sum
}

/// Douglas-Peucker for a CLOSED contour: split into two open chains at the
/// pair of points farthest apart, simplify each chain independently (both
/// share the two anchor points as endpoints), then merge — the standard
/// technique for applying RDP (an open-polyline algorithm) to a closed loop.
fn douglas_peucker_closed(pts: &[(f32, f32)], epsilon: f32) -> Vec<(f32, f32)> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    let n = pts.len();
    let (mut ia, mut ib, mut best) = (0usize, 1usize, -1.0f32);
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist2(pts[i], pts[j]);
            if d > best {
                best = d;
                ia = i;
                ib = j;
            }
        }
    }
    if ia > ib {
        std::mem::swap(&mut ia, &mut ib);
    }

    let chain_a: Vec<(f32, f32)> = pts[ia..=ib].to_vec();
    let chain_b: Vec<(f32, f32)> = pts[ib..].iter().chain(pts[..=ia].iter()).copied().collect();

    let mut simp_a = douglas_peucker_open(&chain_a, epsilon);
    let simp_b = douglas_peucker_open(&chain_b, epsilon);

    // simp_a runs anchor_a -> anchor_b, simp_b runs anchor_b -> anchor_a;
    // drop simp_b's duplicated endpoints when merging into one closed loop.
    if simp_b.len() > 2 {
        simp_a.extend_from_slice(&simp_b[1..simp_b.len() - 1]);
    }
    simp_a
}

/// Standard recursive Ramer-Douglas-Peucker over an open polyline.
fn douglas_peucker_open(pts: &[(f32, f32)], epsilon: f32) -> Vec<(f32, f32)> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    let (first, last) = (pts[0], pts[pts.len() - 1]);
    let mut idx_max = 0usize;
    let mut dist_max = 0.0f32;
    for (i, &p) in pts.iter().enumerate().take(pts.len() - 1).skip(1) {
        let d = dist_point_to_segment(p, first, last);
        if d > dist_max {
            dist_max = d;
            idx_max = i;
        }
    }
    if dist_max > epsilon {
        let mut left = douglas_peucker_open(&pts[..=idx_max], epsilon);
        let right = douglas_peucker_open(&pts[idx_max..], epsilon);
        left.pop(); // avoid duplicating the shared midpoint
        left.extend(right);
        left
    } else {
        vec![first, last]
    }
}

fn dist2(a: (f32, f32), b: (f32, f32)) -> f32 {
    (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)
}

fn dist_point_to_segment(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-6 {
        return dist2(p, a).sqrt();
    }
    let t = (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0);
    let c = (a.0 + t * dx, a.1 + t * dy);
    dist2(p, c).sqrt()
}
