//! Lightweight domain-adapted projection for unsupervised clustering
//! embeddings (`worker.rs`'s `domain_projection` config flag).
//!
//! Trains a small 2-layer MLP classifier on the user's OWN curation labels —
//! the same `curations/labels.jsonl` + `curations/labels/*.png` the existing
//! flywheel retrain (`train/head.rs`) already reads — then DISCARDS the
//! classification head and uses only the first layer's output as a
//! domain-adapted embedding for clustering (a standard "penultimate layer as
//! embedding" trick). This gives the unsupervised clustering path some of
//! the same-dataset advantage a partially-trained closed-set head has,
//! without giving up open-set flexibility: clustering still runs on the
//! resulting embedding via DBSCAN/Hierarchical, not a forced argmax, and new
//! curated families extend the training set immediately, unlike a fixed
//! class list.
//!
//! Trained fresh, in-memory, once per pipeline run — no persisted artifact,
//! no Train-tab UI. Falls back to raw embeddings (logged, not an error) on
//! insufficient curated data OR a training panic, since this is an opt-in
//! bonus feature and must never take down an otherwise-successful run.

use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc;

use burn::{
    grad_clipping::GradientClippingConfig,
    module::{AutodiffModule, Module},
    nn::{loss::CrossEntropyLossConfig, Linear, LinearConfig},
    optim::{AdamConfig, GradientsParams, Optimizer, decay::WeightDecayConfig},
    tensor::{activation::relu, backend::Backend, Int, Tensor, TensorData},
};
use serde::Deserialize;

use crate::tabs::recon_train::model::{create_infer_device, create_train_device, InferBackend, TrainBackend};

use super::dino::DinoExtractor;
use super::worker::PipeMsg;

// A real curations set is realistically tens of examples, not thousands —
// HIDDEN_DIM=128 (~197k params for a 1536-D input) badly overfits a 33-sample
// set: 200 epochs of full-batch descent memorizes the training points
// perfectly while learning nothing that generalizes to the thousands of
// regions never curated (confirmed via real use: trained fine, but every
// uncurated region still collapsed into one blob same as before training).
// A much smaller hidden layer (~49k params even at dim=32) plus real weight
// decay pushes back against this — mirrors why train/head.rs's own tiny-data
// retrain anchors to prior weights rather than training from scratch.
const HIDDEN_DIM: usize = 32;
const EPOCHS: usize = 200;
const LR: f64 = 1e-3;
const WEIGHT_DECAY: f32 = 0.05;
const MIN_CLASSES: usize = 2;
const MIN_PER_CLASS: usize = 3;
const MIN_TOTAL_SAMPLES: usize = 15;

#[derive(Module, Debug)]
pub struct ProjectionHead<B: Backend> {
    fc1: Linear<B>, // in_dim -> HIDDEN_DIM
    fc2: Linear<B>, // HIDDEN_DIM -> num_classes (training only, discarded after)
}

impl<B: Backend> ProjectionHead<B> {
    fn init(device: &B::Device, in_dim: usize, num_classes: usize) -> Self {
        Self {
            fc1: LinearConfig::new(in_dim, HIDDEN_DIM).init(device),
            fc2: LinearConfig::new(HIDDEN_DIM, num_classes).init(device),
        }
    }
    /// Penultimate-layer embedding — the ONLY thing used at clustering time.
    pub fn embed(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        relu(self.fc1.forward(x))
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.fc2.forward(self.embed(x))
    }
}

#[derive(Deserialize)]
struct LabelRow {
    crop: String,
    family: String,
    #[serde(default)]
    source: String,
}

fn read_labels(path: &Path) -> Result<Vec<LabelRow>, String> {
    let txt = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
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

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

fn log(tx: &mpsc::Sender<PipeMsg>, m: impl Into<String>) {
    let _ = tx.send(PipeMsg::Log(m.into()));
}

/// Try to train a domain-adapted projection from this run's curations and
/// apply it to `whole_embeds` (the whole-crop-pooled companion vectors from
/// `worker.rs::pool_region_embed` — NOT the mask-aware `dino_embed`; see
/// that function's doc comment for why this specific vector is the one fed
/// here, to match how the projection can only ever be TRAINED, given
/// historical curation crops have no persisted mask). `embed_res` MUST be
/// the same `worker.rs::embed_resolution(dino.res())` value `whole_embeds`
/// was itself computed at — training on curation crops resized to a
/// DIFFERENT resolution than the embeddings this gets applied to would
/// reintroduce a train/inference mismatch, the same class of bug the
/// mask-vs-whole-crop split above already exists to avoid. Returns `None`
/// (a clear log line, never a hard error) if there isn't enough curated
/// data, the feature dimensions don't line up, or training itself panics.
pub fn try_train_and_apply(
    output_dir: &Path,
    dino: &mut DinoExtractor,
    embed_res: u32,
    whole_embeds: &[Vec<f32>],
    tx: &mpsc::Sender<PipeMsg>,
) -> Option<Vec<Vec<f32>>> {
    let labels_path = output_dir.join("curations").join("labels.jsonl");
    let rows = match read_labels(&labels_path) {
        Ok(r) if !r.is_empty() => r,
        _ => {
            log(tx, "domain_projection: no curations found — using raw DINO embeddings");
            return None;
        }
    };

    let crops_dir = output_dir.join("curations").join("labels");
    let mut name2class: HashMap<String, i32> = HashMap::new();
    let mut class2name: Vec<String> = Vec::new();
    let mut raw_samples: Vec<(Vec<f32>, i32)> = Vec::new();
    for r in &rows {
        if r.source == "reject" || r.family == "rejected" {
            continue; // not a real anomaly type, just noise to skip
        }
        let cls = *name2class.entry(r.family.clone()).or_insert_with(|| {
            class2name.push(r.family.clone());
            (class2name.len() - 1) as i32
        });
        let path = crops_dir.join(&r.crop);
        let img = match image::open(&path) {
            Ok(i) => i.to_rgb8(),
            Err(_) => continue,
        };
        let f = match dino.features_at(&img, embed_res) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let mut m = f.mean_pool();
        l2_normalize(&mut m);
        raw_samples.push((m, cls));
    }

    // Drop under-populated classes (e.g. a single stray/leftover-named entry)
    // rather than letting them block training entirely — a curations file
    // accumulates across sessions and will realistically pick up the odd
    // one-off label; that shouldn't veto two otherwise-solid classes.
    let mut raw_counts = vec![0usize; class2name.len()];
    for &(_, c) in &raw_samples {
        raw_counts[c as usize] += 1;
    }
    let dropped: Vec<&str> = (0..class2name.len())
        .filter(|&i| raw_counts[i] < MIN_PER_CLASS)
        .map(|i| class2name[i].as_str())
        .collect();
    if !dropped.is_empty() {
        log(tx, format!(
            "domain_projection: dropping under-populated classes (< {MIN_PER_CLASS} samples): {}",
            dropped.join(", "),
        ));
    }
    let mut remap: HashMap<i32, i32> = HashMap::new();
    for old in 0..class2name.len() as i32 {
        if raw_counts[old as usize] >= MIN_PER_CLASS {
            let new_id = remap.len() as i32;
            remap.insert(old, new_id);
        }
    }
    let samples: Vec<(Vec<f32>, i32)> = raw_samples.into_iter()
        .filter_map(|(v, c)| remap.get(&c).map(|&nc| (v, nc)))
        .collect();
    let num_classes = remap.len();

    let enough = num_classes >= MIN_CLASSES && samples.len() >= MIN_TOTAL_SAMPLES;
    if !enough {
        log(tx, format!(
            "domain_projection: not enough curated data after filtering ({} samples, {} \
             usable classes — need >= {MIN_TOTAL_SAMPLES} samples, >= {MIN_CLASSES} classes, \
             >= {MIN_PER_CLASS}/class) — using raw DINO embeddings",
            samples.len(), num_classes,
        ));
        return None;
    }

    let in_dim = whole_embeds.first().map(|e| e.len()).unwrap_or(0);
    if in_dim == 0 || samples.iter().any(|(v, _)| v.len() != in_dim) {
        log(tx, "domain_projection: feature dimension mismatch — using raw DINO embeddings");
        return None;
    }

    let t0 = std::time::Instant::now();
    // Own catch_unwind, NOT relying on spawn_pipeline's outer one — that
    // wraps the whole run_pipeline call, so an uncaught panic here would
    // lose every leaf's already-completed results, not just fall back for
    // this one opt-in feature.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        train_projection(&samples, num_classes, in_dim)
    }));
    let model = match result {
        Ok(m) => m,
        Err(_) => {
            log(tx, "domain_projection: training panicked — using raw DINO embeddings");
            return None;
        }
    };
    log(tx, format!(
        "domain_projection: trained on {} samples / {} classes in {:.0}ms",
        samples.len(), num_classes, t0.elapsed().as_secs_f64() * 1000.0,
    ));

    Some(apply_batch(&model, whole_embeds, tx))
}

fn train_projection(
    samples: &[(Vec<f32>, i32)],
    num_classes: usize,
    in_dim: usize,
) -> ProjectionHead<InferBackend> {
    let device = create_train_device();
    let mut model = ProjectionHead::<TrainBackend>::init(&device, in_dim, num_classes);
    let mut optim = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_weight_decay(Some(WeightDecayConfig::new(WEIGHT_DECAY)))
        // This MLP has been observed to occasionally diverge (loss -> NaN)
        // as the curated dataset grows and class balance shifts — with
        // full-batch gradient descent and no dataloader/shuffling to smooth
        // out a bad batch, an unlucky gradient spike has nowhere to get
        // averaged away. A NaN model silently poisons EVERY region's
        // clustering feature at once (all regions share the same trained
        // weights, applied in one batched forward pass) — a real production
        // incident traced to exactly this ("7549 regions -> 0 clusters",
        // the clustering distance matrix effectively all-NaN). Clip by norm
        // as a preventive measure; the explicit finite-loss check in the
        // training loop below is the actual backstop.
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init::<TrainBackend, ProjectionHead<TrainBackend>>();

    let n = samples.len();
    let mut x_data = Vec::with_capacity(n * in_dim);
    let mut y_data: Vec<i32> = Vec::with_capacity(n);
    for (v, c) in samples {
        x_data.extend_from_slice(v);
        y_data.push(*c);
    }
    let x: Tensor<TrainBackend, 2> = Tensor::from_data(TensorData::new(x_data, [n, in_dim]), &device);
    let y: Tensor<TrainBackend, 1, Int> = Tensor::from_data(TensorData::new(y_data, [n]), &device);
    let loss_fn = CrossEntropyLossConfig::new().init::<TrainBackend>(&device);

    // Full-batch gradient descent per epoch — matches train/head.rs's own
    // precedent for curation-scale data (tens to low hundreds of samples),
    // no dataloader/shuffling infrastructure needed at this size.
    for epoch in 0..EPOCHS {
        let logits = model.forward(x.clone());
        let loss = loss_fn.forward(logits, y.clone());
        let loss_val: f32 = loss.clone().into_data().to_vec::<f32>().ok()
            .and_then(|v| v.first().copied())
            .unwrap_or(f32::NAN);
        // Panicking here is deliberate: `try_train_and_apply`'s own
        // `catch_unwind` already exists specifically to turn a training
        // failure into the same graceful "using raw DINO embeddings"
        // fallback as insufficient curated data, rather than shipping a
        // model whose weights have gone NaN.
        assert!(loss_val.is_finite(), "domain_projection training diverged at epoch {epoch} (loss={loss_val})");
        let grads = loss.backward();
        let params = GradientsParams::from_grads(grads, &model);
        model = optim.step(LR, model, params);
    }
    model.valid()
}

/// Apply the trained (inference-only) projection to every embedding in one
/// batched forward pass, L2-normalizing each result to match the rest of
/// the clustering pipeline's convention.
fn apply_batch(model: &ProjectionHead<InferBackend>, embeds: &[Vec<f32>], tx: &mpsc::Sender<PipeMsg>) -> Vec<Vec<f32>> {
    if embeds.is_empty() {
        return Vec::new();
    }
    let device = create_infer_device();
    let n = embeds.len();
    let in_dim = embeds[0].len();
    let mut flat = Vec::with_capacity(n * in_dim);
    for e in embeds {
        flat.extend_from_slice(e);
    }
    let x: Tensor<InferBackend, 2> = Tensor::from_data(TensorData::new(flat, [n, in_dim]), &device);
    let out = model.embed(x).into_data().to_vec::<f32>().unwrap_or_default();
    let mut n_bad = 0usize;
    let rows: Vec<Vec<f32>> = out.chunks(HIDDEN_DIM)
        .map(|c| {
            // `l2_normalize` alone does NOT sanitize a NaN/Infinite
            // component — Rust's `f32::max` ignores NaN in the norm's own
            // `.max(1e-8)` floor, so a poisoned row survives normalization
            // as a still-poisoned row, which would then feed NaN straight
            // into the clustering distance matrix (same class of incident
            // `train_projection`'s finite-loss check guards on the training
            // side — this is the inference-side counterpart, in case a
            // well-trained model still produces a bad output for some
            // specific input). A neutral zero vector is a safe fallback:
            // finite, well-defined distance to everything, just
            // uninformative for this one region rather than corrupting.
            if c.iter().any(|v| !v.is_finite()) {
                n_bad += 1;
                return vec![0.0; HIDDEN_DIM];
            }
            let mut v = c.to_vec();
            l2_normalize(&mut v);
            v
        })
        .collect();
    if n_bad > 0 {
        log(tx, format!("domain_projection: {n_bad}/{n} projected embeddings were non-finite, zeroed"));
    }
    rows
}
