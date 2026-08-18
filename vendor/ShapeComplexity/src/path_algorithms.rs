// src/path_algorithms.rs - Path analysis algorithms for geodesic calculations

use image::RgbaImage;
use bresenham::Bresenham;
use std::collections::{VecDeque, HashMap, BinaryHeap};
use std::cmp::Reverse;

/// Trace a straight line path between two points using Bresenham's algorithm
pub fn trace_straight_line(
    start: (u32, u32),
    end: (u32, u32),
) -> Vec<(u32, u32)> {
    let (start_x, start_y) = (start.0 as isize, start.1 as isize);
    let (end_x, end_y) = (end.0 as isize, end.1 as isize);
    let line = Bresenham::new((start_x, start_y), (end_x, end_y));
    line.map(|(x, y)| (x as u32, y as u32)).collect()
}

/// Check if a straight line path crosses any transparent pixels (excluding endpoints)
pub fn check_straight_line_transparency(
    line_points: &[(u32, u32)],
    image: &RgbaImage,
) -> bool {
    let (width, height) = image.dimensions();

    if line_points.len() <= 2 {
        return false;
    }

    for i in 1..(line_points.len() - 1) {
        let (x, y) = line_points[i];
        if x < width && y < height {
            if image.get_pixel(x, y)[3] == 0 {
                return true;
            }
        }
    }
    false
}

/// Check if a straight line passes through the 2-pixel boundary zone of the
/// leaf *before* reaching the endpoint vicinity.
///
/// This catches the rare case where the straight line from reference_point to
/// margin_point grazes a different part of the leaf edge mid-path — which
/// would otherwise silently inflate the pink count without Dijkstra being
/// triggered (since no transparent pixel is crossed).
///
/// The last `skip_tail` pixels of the line are excluded from the check:
/// those are the legitimate approach to the target tooth's own pink zone.
/// `skip_tail` = max(10, line.len() / 4) keeps the exclusion window
/// proportional to path length while ensuring a minimum safe margin.
pub fn check_straight_line_boundary_zone(
    line_points: &[(u32, u32)],
    image: &RgbaImage,
) -> bool {
    let n = line_points.len();
    if n <= 2 {
        return false;
    }

    let (width, height) = image.dimensions();

    // How many end-pixels to leave unchecked (they belong to the target tooth).
    let skip_tail = (n / 4).max(10);
    let check_end = if n > skip_tail + 1 { n - skip_tail } else { 1 };

    // Check pixels 1 .. check_end (skip start and the approach-to-target tail).
    for &(x, y) in &line_points[1..check_end] {
        for dy in -2i32..=2i32 {
            for dx in -2i32..=2i32 {
                if dx == 0 && dy == 0 { continue; }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                    return true;
                }
                if image.get_pixel(nx as u32, ny as u32)[3] == 0 {
                    return true;
                }
            }
        }
    }
    false
}

/// Euclidean distance between two points
pub fn calculate_straight_path_length(point1: (u32, u32), point2: (u32, u32)) -> f64 {
    let dx = point1.0 as f64 - point2.0 as f64;
    let dy = point1.1 as f64 - point2.1 as f64;
    (dx * dx + dy * dy).sqrt()
}

/// Calculate the geodesic (Diego) path from reference_point to margin_point,
/// staying within non-transparent pixels.
///
/// If the straight line is clear, it is returned directly.
/// Otherwise a graph search runs from **reference_point**:
///
/// - `boundary_penalty = 0.0` → plain BFS (true geodesic in hop count).
///   Used for MC where path *length* is the metric of interest.
///
/// - `boundary_penalty > 0.0` → Dijkstra where every pixel that is
///   4-adjacent to the transparent background pays an extra cost of
///   `boundary_penalty` on top of the normal step cost (1.0 cardinal,
///   √2 diagonal).  This steers the EC path through the leaf interior
///   instead of hugging the leaf margin around a lobe, which would
///   otherwise inflate pink-pixel counts with tooth material from
///   unrelated lobes.
pub fn calculate_diego_path(
    reference_point: (u32, u32),
    margin_point: (u32, u32),
    image: &RgbaImage,
    boundary_penalty: f64,
) -> Vec<(u32, u32)> {
    // Fast path: straight line is clear, no graph search needed.
    let straight_line = trace_straight_line(reference_point, margin_point);
    if !check_straight_line_transparency(&straight_line, image) {
        return straight_line;
    }

    let (width, height) = image.dimensions();

    // Reference point must itself be non-transparent.
    if image.get_pixel(reference_point.0, reference_point.1)[3] == 0 {
        return straight_line;
    }

    // Cardinal directions first so BFS prefers axis-aligned steps over diagonals
    // when path length is equal — preserves historical behaviour for clear paths.
    let directions: [(i32, i32); 8] = [
        (0, 1), (1, 0), (0, -1), (-1, 0),   // Cardinal
        (1, 1), (1, -1), (-1, 1), (-1, -1), // Diagonal
    ];

    // Predecessor map shared by both BFS and Dijkstra for path reconstruction.
    // reference_point points to itself as the loop-termination sentinel.
    let mut prev_map: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
    prev_map.insert(reference_point, reference_point);

    let found = if boundary_penalty <= 0.0 {
        // ----------------------------------------------------------------
        // Plain BFS — true geodesic (used for MC).
        // ----------------------------------------------------------------
        let max_iter = (width * height) as usize * 2;
        let mut queue: VecDeque<(u32, u32)> = VecDeque::new();
        queue.push_back(reference_point);
        let mut iter = 0_usize;
        let mut found = false;

        'bfs: while let Some(current) = queue.pop_front() {
            iter += 1;
            if iter > max_iter {
                println!("Warning: BFS exceeded max iterations ({}) — returning straight line",
                         max_iter);
                return straight_line;
            }
            for &(dx, dy) in &directions {
                let nx = current.0 as i32 + dx;
                let ny = current.1 as i32 + dy;
                if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 { continue; }
                let next = (nx as u32, ny as u32);
                if prev_map.contains_key(&next) { continue; }
                if image.get_pixel(next.0, next.1)[3] == 0 { continue; }
                prev_map.insert(next, current);
                if next == margin_point { found = true; break 'bfs; }
                queue.push_back(next);
            }
        }
        found
    } else {
        // ----------------------------------------------------------------
        // Dijkstra with boundary penalty (used for EC).
        //
        // Costs are integers scaled by 1000 to avoid floating-point heap
        // ordering issues:
        //   cardinal step  = 1000
        //   diagonal step  = 1414  (≈ √2 × 1000)
        //   boundary extra = boundary_penalty × 1000  (added to destination)
        //
        // A pixel is "boundary" if any of its 4-cardinal neighbours is
        // transparent (alpha == 0) in the navigation image.
        // ----------------------------------------------------------------
        let penalty_int = (boundary_penalty * 1000.0).round() as u64;

        // "Boundary" = any pixel whose 5×5 neighbourhood (radius 2) contains
        // a transparent pixel.  This covers the full 2-pixel-wide outer ring of
        // the leaf, which matches the inner layers of the pink zone produced by
        // the morphological opening.  Using only 4-cardinal neighbours at
        // distance 1 left inner pink layers un-penalised, allowing the path to
        // still hug the margin through the second or third pink layer.
        let is_boundary = |x: u32, y: u32| -> bool {
            for dy in -2i32..=2i32 {
                for dx in -2i32..=2i32 {
                    if dx == 0 && dy == 0 { continue; }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                        return true; // image edge treated as boundary
                    }
                    if image.get_pixel(nx as u32, ny as u32)[3] == 0 {
                        return true;
                    }
                }
            }
            false
        };

        // Min-heap entry: (Reverse(cost), x, y)
        let mut heap: BinaryHeap<(Reverse<u64>, u32, u32)> = BinaryHeap::new();
        let mut dist: HashMap<(u32, u32), u64> = HashMap::new();
        dist.insert(reference_point, 0);
        heap.push((Reverse(0), reference_point.0, reference_point.1));

        let mut found = false;

        'dijkstra: while let Some((Reverse(cost), cx, cy)) = heap.pop() {
            let current = (cx, cy);

            // Skip stale heap entries (a cheaper path was already found).
            if cost > *dist.get(&current).unwrap_or(&u64::MAX) {
                continue;
            }

            if current == margin_point {
                found = true;
                break 'dijkstra;
            }

            for &(dx, dy) in &directions {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;
                if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 { continue; }
                let next = (nx as u32, ny as u32);
                if image.get_pixel(next.0, next.1)[3] == 0 { continue; }

                let base  = if dx == 0 || dy == 0 { 1000u64 } else { 1414u64 };
                let extra = if is_boundary(next.0, next.1) { penalty_int } else { 0 };
                let new_cost = cost + base + extra;

                let best = dist.entry(next).or_insert(u64::MAX);
                if new_cost < *best {
                    *best = new_cost;
                    prev_map.insert(next, current);
                    heap.push((Reverse(new_cost), next.0, next.1));
                }
            }
        }
        found
    };

    if !found {
        println!("Path search could not reach target — returning straight line");
        return straight_line;
    }

    // Reconstruct path from margin_point back to reference_point, then reverse.
    let mut path = Vec::new();
    let mut cur = margin_point;
    loop {
        path.push(cur);
        let p = prev_map[&cur];
        if p == cur { break; } // reached reference_point sentinel
        cur = p;
    }
    path.reverse();
    path
}

/// Total length of a path in pixels (Euclidean step-by-step)
/// Single-source predecessor tree from `reference_point` to every reachable
/// non-transparent pixel — computed ONCE and reused by all margin points (they all
/// share the source). Uses the SAME BFS (boundary_penalty<=0) / boundary-penalised
/// Dijkstra (>0) as `calculate_diego_path`, with the SAME direction order and heap
/// tie-breaking, so paths reconstructed from it are byte-identical — but it's one
/// search instead of one-per-point (was the pipeline's 95% hotspot).
/// Returns a flat `Vec<u32>` of predecessor flat-indices (u32::MAX = unvisited;
/// the source points to itself as the reconstruction sentinel).
pub fn build_diego_prevmap(
    reference_point: (u32, u32),
    image: &RgbaImage,
    boundary_penalty: f64,
) -> Vec<u32> {
    let (width, height) = image.dimensions();
    let (w, h, wu) = (width as i32, height as i32, width as usize);
    let n = (width * height) as usize;
    let raw = image.as_raw();
    let opaque = |idx: usize| raw[idx * 4 + 3] != 0;
    let ref_idx = reference_point.1 as usize * wu + reference_point.0 as usize;

    let mut prev = vec![u32::MAX; n];
    if !opaque(ref_idx) {
        return prev; // reference transparent → callers fall back to the straight line
    }
    prev[ref_idx] = ref_idx as u32;

    // Cardinal first (matches historical tie preference in the original search).
    let directions: [(i32, i32); 8] = [
        (0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1),
    ];

    if boundary_penalty <= 0.0 {
        // Plain BFS (MC).
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(ref_idx);
        while let Some(cur) = queue.pop_front() {
            let (cx, cy) = ((cur % wu) as i32, (cur / wu) as i32);
            for &(dx, dy) in &directions {
                let (nx, ny) = (cx + dx, cy + dy);
                if nx < 0 || ny < 0 || nx >= w || ny >= h { continue; }
                let nidx = ny as usize * wu + nx as usize;
                if prev[nidx] != u32::MAX || !opaque(nidx) { continue; }
                prev[nidx] = cur as u32;
                queue.push_back(nidx);
            }
        }
    } else {
        // Boundary-penalised Dijkstra (EC). Precompute the boundary mask once
        // (was an O(25) get_pixel scan per edge relaxation in the original).
        let penalty_int = (boundary_penalty * 1000.0).round() as u64;
        let mut boundary = vec![false; n];
        for y in 0..h {
            for x in 0..w {
                let mut b = false;
                'scan: for dy in -2i32..=2 {
                    for dx in -2i32..=2 {
                        if dx == 0 && dy == 0 { continue; }
                        let (nx, ny) = (x + dx, y + dy);
                        if nx < 0 || ny < 0 || nx >= w || ny >= h { b = true; break 'scan; }
                        if !opaque(ny as usize * wu + nx as usize) { b = true; break 'scan; }
                    }
                }
                boundary[y as usize * wu + x as usize] = b;
            }
        }
        // Heap tuple (Reverse(cost), x, y) matches the original's tie-breaking exactly.
        let mut heap: BinaryHeap<(Reverse<u64>, u32, u32)> = BinaryHeap::new();
        let mut dist = vec![u64::MAX; n];
        dist[ref_idx] = 0;
        heap.push((Reverse(0), reference_point.0, reference_point.1));
        while let Some((Reverse(cost), cx, cy)) = heap.pop() {
            let cur = cy as usize * wu + cx as usize;
            if cost > dist[cur] { continue; }
            for &(dx, dy) in &directions {
                let (nx, ny) = (cx as i32 + dx, cy as i32 + dy);
                if nx < 0 || ny < 0 || nx >= w || ny >= h { continue; }
                let nidx = ny as usize * wu + nx as usize;
                if !opaque(nidx) { continue; }
                let base = if dx == 0 || dy == 0 { 1000u64 } else { 1414u64 };
                let extra = if boundary[nidx] { penalty_int } else { 0 };
                let new_cost = cost + base + extra;
                if new_cost < dist[nidx] {
                    dist[nidx] = new_cost;
                    prev[nidx] = cur as u32;
                    heap.push((Reverse(new_cost), nx as u32, ny as u32));
                }
            }
        }
    }
    prev
}

/// Reconstruct one margin point's path from a `build_diego_prevmap` result — the
/// O(path-length) replacement for a full `calculate_diego_path` call. Same fast-path
/// (clear straight line) and same unreachable fallback, so results are identical.
pub fn diego_path_from_prevmap(
    reference_point: (u32, u32),
    margin_point: (u32, u32),
    image: &RgbaImage,
    prev: &[u32],
) -> Vec<(u32, u32)> {
    let straight_line = trace_straight_line(reference_point, margin_point);
    if !check_straight_line_transparency(&straight_line, image) {
        return straight_line;
    }
    let width = image.width();
    let t_idx = (margin_point.1 * width + margin_point.0) as usize;
    if prev.get(t_idx).copied().unwrap_or(u32::MAX) == u32::MAX {
        return straight_line; // unreachable — matches the original's "not found" fallback
    }
    let mut path = Vec::new();
    let mut cur = t_idx as u32;
    loop {
        path.push((cur % width, cur / width));
        let p = prev[cur as usize];
        if p == cur { break; }
        cur = p;
    }
    path.reverse();
    path
}

pub fn calculate_diego_path_length(path: &[(u32, u32)]) -> f64 {
    if path.len() < 2 {
        return 0.0;
    }
    let mut length = 0.0;
    for i in 1..path.len() {
        let dx = path[i].0 as f64 - path[i - 1].0 as f64;
        let dy = path[i].1 as f64 - path[i - 1].1 as f64;
        length += (dx * dx + dy * dy).sqrt();
    }
    length
}

/// Count pink pixels at the margin using a local inward ray.
///
/// Instead of counting pink pixels along the full geodesic path (which inflates
/// the count whenever the path follows the leaf margin around a lobe), this
/// function shoots a short Bresenham ray from `margin_point` toward
/// `reference_point` (the COM) and returns the length of the **first contiguous
/// run of pink pixels** it encounters.
///
/// This is purely local: it measures the depth of the pink (eroded margin)
/// zone at the specific tooth/notch, unaffected by how the routing algorithm
/// chooses to navigate the leaf interior.
///
/// * `max_depth` — maximum number of pixels to walk inward (default: 150).
///   The ray stops earlier when it hits the first non-pink opaque pixel after
///   leaving the pink zone, or when it reaches `max_depth`.
pub fn calculate_local_pink_depth(
    margin_point: (u32, u32),
    reference_point: (u32, u32),
    marked_image: &RgbaImage,
    pink_color: [u8; 3],
    max_depth: u32,
) -> u32 {
    // Ray from margin_point toward reference_point.
    let ray = trace_straight_line(margin_point, reference_point);

    let limit = (max_depth as usize).min(ray.len());
    let mut pink_run = 0u32;
    let mut in_pink = false;

    for &(x, y) in &ray[..limit] {
        let (width, height) = marked_image.dimensions();
        if x >= width || y >= height {
            break;
        }
        let p = marked_image.get_pixel(x, y);
        // Skip fully transparent pixels (outside the leaf).
        if p[3] == 0 {
            continue;
        }
        let is_pink = p[0] == pink_color[0] && p[1] == pink_color[1] && p[2] == pink_color[2];

        if is_pink {
            in_pink = true;
            pink_run += 1;
        } else if in_pink {
            // First non-pink opaque pixel after entering the pink zone → done.
            break;
        }
    }
    pink_run
}

/// Measure the pink-zone depth at a contour point using the **inward contour normal**.
///
/// # Why this is the scientifically correct approach
///
/// Morphological opening erodes the leaf boundary *perpendicular* to the local
/// boundary direction.  The pink pixels created by `mark_opened_regions` therefore
/// form a zone whose depth, measured along the inward normal at each contour point,
/// exactly equals the erosion depth at that location.
///
/// Counting pink pixels along the full geodesic path (the previous approach)
/// accumulates material from unrelated teeth whenever the path travels laterally
/// along the margin — an artefact of routing, not of leaf shape.  The normal ray
/// is purely local and routing-independent.
///
/// # Algorithm
///
/// 1. Compute a smoothed tangent vector at `point_index` by finite-differencing
///    `contour[i + tangent_window]` and `contour[i - tangent_window]` (wrap-around).
///    A window of 10 suppresses pixel-level contour noise while remaining local
///    enough to resolve individual teeth.
/// 2. Rotate the tangent 90° to get two normal candidates; choose the one that
///    points toward `reference_point` (the COM, which is always interior).
/// 3. Step `max_depth` pixels along the inward normal (integer rounding, no
///    Bresenham needed for a unit-direction ray) and count the first contiguous
///    run of pink pixels, starting from the margin point itself.
///
/// # Parameters
/// * `point_index`      — index of this point in `contour`
/// * `contour`          — full ordered contour (used for tangent computation)
/// * `reference_point`  — COM / interior reference for normal disambiguation
/// * `marked_image`     — image with pink-marked erosion zones
/// * `pink_color`       — RGB of the pink marker
/// * `max_depth`        — maximum inward steps (150 px is a safe default)
/// * `tangent_window`   — contour-index offset for tangent finite difference (10 is recommended)
pub fn calculate_inward_normal_pink_depth(
    point_index: usize,
    contour: &[(u32, u32)],
    reference_point: (u32, u32),
    marked_image: &RgbaImage,
    pink_color: [u8; 3],
    max_depth: u32,
    tangent_window: usize,
) -> u32 {
    let n = contour.len();
    if n < 3 {
        return 0;
    }

    let margin_point = contour[point_index];
    let (width, height) = marked_image.dimensions();

    // --- Step 1: smoothed tangent via finite difference -------------------------
    let window = tangent_window.min(n / 2).max(1);
    let i_prev = (point_index + n - window) % n;
    let i_next = (point_index + window) % n;

    let prev = contour[i_prev];
    let next = contour[i_next];

    let tx = next.0 as f64 - prev.0 as f64;
    let ty = next.1 as f64 - prev.1 as f64;

    let len = (tx * tx + ty * ty).sqrt();
    if len < 1e-9 {
        return 0; // degenerate (duplicate contour points in window)
    }

    // --- Step 2: choose inward normal -------------------------------------------
    // Two candidates: rotate tangent ±90°.
    let nx1 = -ty / len;
    let ny1 =  tx / len;
    // nx2 = ty/len, ny2 = -tx/len  (opposite direction)

    // Select the candidate that points toward the COM (interior).
    let ref_dx = reference_point.0 as f64 - margin_point.0 as f64;
    let ref_dy = reference_point.1 as f64 - margin_point.1 as f64;

    let (inward_nx, inward_ny) = if nx1 * ref_dx + ny1 * ref_dy >= 0.0 {
        (nx1, ny1)
    } else {
        (-nx1, -ny1)
    };

    // --- Step 3: walk inward ray, count first contiguous pink run ---------------
    // Start at step 0 (the margin point itself) so tooth-tip points that are
    // already in the pink zone are counted correctly.
    let mut pink_run = 0u32;
    let mut in_pink = false;

    for step in 0..=max_depth {
        let x = (margin_point.0 as f64 + step as f64 * inward_nx).round() as i32;
        let y = (margin_point.1 as f64 + step as f64 * inward_ny).round() as i32;

        if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
            break;
        }
        let (x, y) = (x as u32, y as u32);

        let p = marked_image.get_pixel(x, y);
        if p[3] == 0 {
            break; // stepped outside the leaf
        }

        let is_pink = p[0] == pink_color[0] && p[1] == pink_color[1] && p[2] == pink_color[2];

        if is_pink {
            in_pink = true;
            pink_run += 1;
        } else if in_pink {
            break; // first non-pink interior pixel after the pink zone → done
        }
    }

    pink_run
}

/// Count pink (marked) pixels along a path.
/// Used for EC analysis to measure how much eroded margin material the path crosses.
pub fn calculate_diego_path_pink(
    path: &[(u32, u32)],
    marked_image: &RgbaImage,
    pink_color: [u8; 3],
) -> u32 {
    let mut count = 0;
    for &(x, y) in path {
        let p = marked_image.get_pixel(x, y);
        if p[0] == pink_color[0] && p[1] == pink_color[1] && p[2] == pink_color[2] {
            count += 1;
        }
    }
    count
}