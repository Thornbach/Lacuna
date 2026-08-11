//! Writes corrected `RealScene`s into the SAME `images/{train,val}` /
//! `labels/{train,val}` / `data.yaml` structure `generate_leaf_dataset.py`
//! already produces, distinguished by a `real_NNNNNN` filename prefix (vs.
//! `scene_NNNNNN`) so real and synthetic scenes coexist in one dataset with
//! no shared counter or multi-dataset YOLO config needed.

use std::collections::HashMap;
use std::path::Path;

use image::codecs::jpeg::JpegEncoder;
use image::ImageEncoder;
use serde::{Deserialize, Serialize};

use super::polygon::{mask_to_polygon, polygon_to_yolo_line};
use super::{RealInstance, RealScene};
use crate::tabs::leaf_seg::inference::list_images;
use crate::tabs::mask_tools::fill_polygon_mask;

const JPEG_QUALITY: u8 = 92;
/// Matches `generate_leaf_dataset.py`'s `CFG["val_split"]` default — a
/// re-export must land in the same split it started in, or the val metric
/// becomes hard to interpret, hence the deterministic hash below rather
/// than a fresh random draw every export.
const DEFAULT_VAL_PCT: u32 = 15;

#[derive(Clone, Serialize, Deserialize)]
struct ExportRecord {
    idx:   u32,
    split: String,
}

pub struct ExportSummary {
    pub exported: usize,
    pub skipped_unreviewed: usize,
    pub failed: Vec<(String, String)>, // (filename, error)
}

/// Export every `reviewed` scene in `scenes` (mutating each scene's
/// `exported_idx` on success) into `dataset_root`. Scenes with
/// `exported_idx` already set are upserted in place (overwrite, same index)
/// rather than assigned a new one.
pub fn export_scenes(dataset_root: &Path, scenes: &mut [RealScene]) -> ExportSummary {
    let mut summary = ExportSummary { exported: 0, skipped_unreviewed: 0, failed: Vec::new() };

    for split in ["train", "val"] {
        let _ = std::fs::create_dir_all(dataset_root.join("images").join(split));
        let _ = std::fs::create_dir_all(dataset_root.join("labels").join(split));
    }
    let _ = std::fs::create_dir_all(dataset_root.join("meta"));

    let mut export_map = load_export_map(dataset_root);
    let mut next_idx = find_next_real_idx(dataset_root);

    for scene in scenes.iter_mut() {
        if !scene.reviewed {
            summary.skipped_unreviewed += 1;
            continue;
        }

        let key = scene.path.to_string_lossy().to_string();
        let (idx, split) = if let Some(rec) = export_map.get(&key) {
            (rec.idx, rec.split.clone())
        } else if let Some(idx) = scene.exported_idx {
            // In-memory idx from an earlier export this session, sidecar not
            // yet caught up (shouldn't normally happen, but stay consistent).
            (idx, split_for(&scene.filename, DEFAULT_VAL_PCT).to_string())
        } else {
            let idx = next_idx;
            next_idx += 1;
            (idx, split_for(&scene.filename, DEFAULT_VAL_PCT).to_string())
        };

        match export_one(dataset_root, &split, idx, scene) {
            Ok(()) => {
                scene.exported_idx = Some(idx);
                export_map.insert(key, ExportRecord { idx, split });
                summary.exported += 1;
            }
            Err(e) => summary.failed.push((scene.filename.clone(), e)),
        }
    }

    if let Err(e) = save_export_map(dataset_root, &export_map) {
        summary.failed.push(("(export map)".into(), e));
    }
    ensure_data_yaml(dataset_root);

    summary
}

fn export_one(dataset_root: &Path, split: &str, idx: u32, scene: &RealScene) -> Result<(), String> {
    let stem = format!("real_{idx:06}");
    let img_path = dataset_root.join("images").join(split).join(format!("{stem}.jpg"));
    let lbl_path = dataset_root.join("labels").join(split).join(format!("{stem}.txt"));

    write_scene_image(&scene.path, &img_path)?;

    let mut lines = Vec::new();
    for inst in &scene.instances {
        let [_, _, bw, bh] = inst.bbox;
        if let Some(poly) = mask_to_polygon(&inst.mask, bw, bh) {
            // mask_to_polygon normalizes against the INSTANCE's own bbox —
            // re-express against the full scene before writing (YOLO
            // polygons are normalized to the whole image, not the crop).
            let [bx, by, ..] = inst.bbox;
            let (sw, sh) = (scene.size[0] as f32, scene.size[1] as f32);
            let scaled: Vec<(f32, f32)> = poly
                .into_iter()
                .map(|(x, y)| {
                    ((bx as f32 + x * bw as f32) / sw, (by as f32 + y * bh as f32) / sh)
                })
                .collect();
            lines.push(polygon_to_yolo_line(&scaled, inst.class_id));
        }
    }

    std::fs::write(&lbl_path, lines.join("\n")).map_err(|e| e.to_string())
}

fn write_scene_image(src_path: &Path, dest_path: &Path) -> Result<(), String> {
    let img = image::open(src_path).map_err(|e| format!("open {}: {e}", src_path.display()))?;
    let rgb = img.to_rgb8();
    let file = std::fs::File::create(dest_path).map_err(|e| e.to_string())?;
    let encoder = JpegEncoder::new_with_quality(file, JPEG_QUALITY);
    encoder
        .write_image(rgb.as_raw(), rgb.width(), rgb.height(), image::ExtendedColorType::Rgb8)
        .map_err(|e| e.to_string())
}

/// Mirrors `find_next_scene_idx` in `generate_leaf_dataset.py`, just scoped
/// to the `real_` prefix so synthetic and real scene indices never collide.
fn find_next_real_idx(dataset_root: &Path) -> u32 {
    let mut max_idx: i64 = -1;
    for split in ["train", "val"] {
        let dir = dataset_root.join("images").join(split);
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(rest) = name.strip_prefix("real_") {
                if let Some(num_str) = rest.split('.').next() {
                    if let Ok(idx) = num_str.parse::<i64>() {
                        max_idx = max_idx.max(idx);
                    }
                }
            }
        }
    }
    (max_idx + 1) as u32
}

fn load_export_map(dataset_root: &Path) -> HashMap<String, ExportRecord> {
    let path = dataset_root.join("meta").join("real_export_map.json");
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_export_map(dataset_root: &Path, map: &HashMap<String, ExportRecord>) -> Result<(), String> {
    let path = dataset_root.join("meta").join("real_export_map.json");
    let json = serde_json::to_string_pretty(map).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| e.to_string())
}

/// Never blind-overwrites an existing `data.yaml` (the common case — training
/// already ran against a synthetic one) — only writes one if missing.
fn ensure_data_yaml(dataset_root: &Path) {
    let path = dataset_root.join("data.yaml");
    if path.exists() {
        return;
    }
    let content = format!(
        "# Auto-generated YOLO dataset config\npath: {}\ntrain: images/train\nval:   images/val\n\nnc: 1\nnames: ['leaf']\n",
        dataset_root.display()
    );
    let _ = std::fs::write(&path, content);
}

/// Imports a folder of `images/`+`labels/` (YOLO-seg format — the exact
/// convention `export_scenes` itself writes, and what an offline batch
/// pre-labeling pass, e.g. YOLO+SAM, would produce) as fresh, UNREVIEWED
/// `RealScene`s — the counterpart to `export_scenes`, for loading
/// externally-generated pre-labels straight into the SAME correction UI.
/// `reviewed` starts false, same as a live "Run Segmentation" result, so
/// nothing here is ever treated as verified ground truth until a human
/// actually confirms it in Field Review.
pub fn import_prelabeled(dir: &Path) -> Result<Vec<RealScene>, String> {
    let images_dir = dir.join("images");
    let labels_dir = dir.join("labels");
    if !images_dir.is_dir() {
        return Err(format!("no images/ folder found in {}", dir.display()));
    }

    let mut scenes = Vec::new();
    for path in list_images(&images_dir) {
        let img = image::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let (w, h) = (img.width(), img.height());

        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
        let label_path = labels_dir.join(format!("{stem}.txt"));
        let mut instances = Vec::new();
        if let Ok(text) = std::fs::read_to_string(&label_path) {
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let mut parts = line.split_whitespace();
                let Some(class_str) = parts.next() else { continue };
                let Ok(class_id) = class_str.parse::<u32>() else { continue };
                let coords: Vec<f32> = parts.filter_map(|s| s.parse::<f32>().ok()).collect();
                if coords.len() < 6 || coords.len() % 2 != 0 {
                    continue; // need at least 3 (x,y) points
                }
                let poly: Vec<(f32, f32)> = coords
                    .chunks_exact(2)
                    .map(|c| (c[0] * w as f32, c[1] * h as f32))
                    .collect();
                if let Some((bbox, mask)) = fill_polygon_mask(&poly) {
                    instances.push(RealInstance { bbox, mask, class_id });
                }
            }
        }

        scenes.push(RealScene {
            filename: path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
            path,
            size: [w, h],
            instances,
            reviewed: false,
            exported_idx: None,
        });
    }
    Ok(scenes)
}

fn stable_hash(s: &str) -> u64 {
    // FNV-1a — deterministic across runs/platforms, unlike DefaultHasher
    // (whose algorithm isn't guaranteed stable), which matters here since a
    // re-export must land in the same split every time.
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn split_for(filename: &str, val_pct: u32) -> &'static str {
    if stable_hash(filename) % 100 < val_pct as u64 {
        "val"
    } else {
        "train"
    }
}

