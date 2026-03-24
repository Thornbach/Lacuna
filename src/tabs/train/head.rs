//! Flywheel consume-side: fine-tune the few-shot head from saved curations.
//!
//! Reads `<curations>/labels.jsonl` (+ crops written by the Pipeline tab's "Save
//! curations"), runs the frozen DINO on each crop, and nudges the existing head's
//! weights toward the user's confirmed family / rejected labels — warm-started from
//! the current head and L2-anchored to it, so it improves on the curated cases
//! WITHOUT forgetting the base. A cluster NAME the user typed that isn't a known
//! family becomes a NEW class (the loop discovers types). Writes an updated
//! `fewshot_head.json` the app reloads. No Python, no DINO features stored — the
//! crop IMAGE is re-featurized, so this survives a backbone swap.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Deserialize;

use crate::tabs::pipeline::dino::DinoExtractor;
use crate::tabs::pipeline::fewshot::FewShotHead;

pub struct RetrainCfg {
    pub head_path:     PathBuf,
    pub dino_model:    PathBuf,
    pub curations_dir: PathBuf,
    pub out_path:      PathBuf,
    pub epochs:        usize,
    pub lr:            f32,
    pub l2_anchor:     f32, // pull weights toward the base head (anti-forgetting)
}

pub enum RetrainMsg {
    Stage(String),
    Log(String),
    Error(String),
    Done(String), // summary
}

#[derive(Deserialize)]
struct LabelRow {
    crop: String,
    family: String,
    #[serde(default)]
    source: String,
}

pub fn spawn_retrain(cfg: RetrainCfg, tx: mpsc::Sender<RetrainMsg>, cancel: Arc<AtomicBool>) {
    std::thread::spawn(move || match retrain(&cfg, &tx, &cancel) {
        Ok(summary) => { let _ = tx.send(RetrainMsg::Done(summary)); }
        Err(e) => { let _ = tx.send(RetrainMsg::Error(e)); }
    });
}

fn read_labels(path: &Path) -> Result<Vec<LabelRow>, String> {
    let txt = std::fs::read_to_string(path)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for line in txt.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(r) = serde_json::from_str::<LabelRow>(line) {
            out.push(r);
        }
    }
    Ok(out)
}

/// L2-normalised mean DINO feature over a crop (matches the unit-norm patch space
/// the head was trained on).
fn mean_feature(feat: &[f32], grid: usize, dim: usize) -> Vec<f32> {
    let n = grid * grid;
    let mut m = vec![0f32; dim];
    for p in 0..n {
        let fp = &feat[p * dim..p * dim + dim];
        for d in 0..dim {
            m[d] += fp[d];
        }
    }
    let mut nrm = 0f32;
    for d in 0..dim {
        m[d] /= n as f32;
        nrm += m[d] * m[d];
    }
    let nrm = nrm.sqrt().max(1e-8);
    for d in 0..dim {
        m[d] /= nrm;
    }
    m
}

fn retrain(
    cfg: &RetrainCfg,
    tx: &mpsc::Sender<RetrainMsg>,
    cancel: &AtomicBool,
) -> Result<String, String> {
    let _ = tx.send(RetrainMsg::Stage("Loading head + DINO".into()));
    let mut head = FewShotHead::load(&cfg.head_path)?;
    let dim = head.dim;
    let mut dino = DinoExtractor::load(&cfg.dino_model, head.infer_resolution)?;

    let rows = read_labels(&cfg.curations_dir.join("labels.jsonl"))?;
    if rows.is_empty() {
        return Err("no curation labels found".into());
    }

    // name -> class id, seeded from the head's families; new names get new ids.
    let mut name2class: HashMap<String, i32> = HashMap::new();
    for (idx, name) in &head.families {
        if let Ok(i) = idx.parse::<i32>() {
            name2class.insert(name.clone(), i);
        }
    }
    let healthy_class = 0i32;
    let mut next_id = head.classes.iter().copied().max().unwrap_or(0) + 1;
    let mut n_new = 0;

    // ── gather samples (mean feature per crop, one label each) ──
    let _ = tx.send(RetrainMsg::Stage("Extracting features from curations".into()));
    let crops_dir = cfg.curations_dir.join("labels");
    let mut samples: Vec<(Vec<f32>, i32)> = Vec::new();
    for (k, r) in rows.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let cls = if r.source == "reject" || r.family == "rejected" {
            healthy_class
        } else if let Some(&c) = name2class.get(&r.family) {
            c
        } else {
            // a new, user-named family -> grow the head with a fresh class
            let c = next_id;
            next_id += 1;
            n_new += 1;
            head.classes.push(c);
            head.intercept.push(0.0);
            head.coef.push(vec![0.0; dim]);
            head.families.insert(c.to_string(), r.family.clone());
            name2class.insert(r.family.clone(), c);
            c
        };
        let path = crops_dir.join(&r.crop);
        let img = match image::open(&path) {
            Ok(i) => i.to_rgb8(),
            Err(e) => {
                let _ = tx.send(RetrainMsg::Log(format!("skip {}: {e}", r.crop)));
                continue;
            }
        };
        let f = dino.features(&img)?;
        if f.dim != dim {
            return Err(format!("crop feature dim {} != head dim {dim}", f.dim));
        }
        samples.push((mean_feature(&f.feat, f.grid, dim), cls));
        if (k + 1) % 25 == 0 {
            let _ = tx.send(RetrainMsg::Log(format!("features {}/{}", k + 1, rows.len())));
        }
    }
    if samples.is_empty() {
        return Err("no usable curation crops".into());
    }

    // ── warm-started, anchored softmax fine-tune ──
    let _ = tx.send(RetrainMsg::Stage("Fine-tuning head".into()));
    let kk = head.classes.len();
    let class_row: HashMap<i32, usize> =
        head.classes.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let w0 = head.coef.clone(); // anchor toward the base head
    let b0 = head.intercept.clone();
    let n = samples.len() as f32;
    for ep in 0..cfg.epochs {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let mut gw = vec![vec![0f32; dim]; kk];
        let mut gb = vec![0f32; kk];
        let mut loss = 0f32;
        for (x, cls) in &samples {
            let row = class_row[cls];
            let mut logit = vec![0f32; kk];
            for k in 0..kk {
                let mut s = head.intercept[k];
                let wk = &head.coef[k];
                for d in 0..dim {
                    s += wk[d] * x[d];
                }
                logit[k] = s;
            }
            let m = logit.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for l in logit.iter_mut() {
                *l = (*l - m).exp();
                sum += *l;
            }
            for l in logit.iter_mut() {
                *l /= sum;
            }
            loss -= (logit[row] + 1e-9).ln();
            for k in 0..kk {
                let g = logit[k] - if k == row { 1.0 } else { 0.0 };
                gb[k] += g;
                let gwk = &mut gw[k];
                for d in 0..dim {
                    gwk[d] += g * x[d];
                }
            }
        }
        for k in 0..kk {
            head.intercept[k] -= cfg.lr * (gb[k] / n + cfg.l2_anchor * (head.intercept[k] - b0[k]));
            let wk = &mut head.coef[k];
            let w0k = &w0[k];
            let gwk = &gw[k];
            for d in 0..dim {
                wk[d] -= cfg.lr * (gwk[d] / n + cfg.l2_anchor * (wk[d] - w0k[d]));
            }
        }
        if (ep + 1) % 25 == 0 || ep == cfg.epochs - 1 {
            let _ = tx.send(RetrainMsg::Log(format!("epoch {}/{}  loss {:.4}", ep + 1, cfg.epochs, loss / n)));
        }
    }

    // ── write updated head ──
    head.onnx_parity = None; // weights changed; parity no longer the exported one
    let json = serde_json::to_string(&head).map_err(|e| format!("serialize head: {e}"))?;
    if let Some(p) = cfg.out_path.parent() {
        let _ = std::fs::create_dir_all(p);
    }
    std::fs::write(&cfg.out_path, json).map_err(|e| format!("write head: {e}"))?;
    Ok(format!(
        "Updated head: {} crops, {} classes (+{} new) -> {}",
        samples.len(), kk, n_new, cfg.out_path.display()
    ))
}
