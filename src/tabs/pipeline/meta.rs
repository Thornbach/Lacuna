//! Calibrated detector constants, loaded from `detector_meta.json`
//! (written by 1Help/export_assets.py at the ONNX resolution, in f32).

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub struct DetectorMeta {
    pub threshold:      f32,                 // legacy / display (max channel threshold)
    pub ch_thresholds:  HashMap<String, f32>, // per-channel z seed thresholds
    pub dino_threshold: f32,                 // absolute raw-DINO floor
    pub scale_floors:   HashMap<String, f32>, // per-channel robust-z scale floors
    pub dim:            usize,
    #[serde(default)]
    pub project_dim:    Option<usize>,
}

impl DetectorMeta {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("read meta: {e}"))?;
        serde_json::from_str(&text).map_err(|e| format!("parse detector_meta.json: {e}"))
    }

    pub fn has(&self, ch: &str) -> bool {
        self.ch_thresholds.contains_key(ch)
    }

    pub fn ch_threshold(&self, ch: &str) -> f32 {
        self.ch_thresholds.get(ch).copied().unwrap_or(f32::INFINITY)
    }

    pub fn scale_floor(&self, ch: &str) -> f32 {
        self.scale_floors.get(ch).copied().unwrap_or(0.0)
    }
}
