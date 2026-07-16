//! Standalone reconstruction training entry point (headless CLI).
//!
//! Run:
//!   lacuna --recon-train <oak-intact-folder> [epochs] [max_images]
//!
//! One proper run with the CURRENT clean config: 4-channel RGBA input, no-delete loss,
//! boundary loss, boosted area head, margin-loss-only damage capped at 40%, trained at
//! 512px (matches the pipeline's RECON_SIZE). Saves checkpoint_best continuously + a
//! sample-grid PNG every 5 epochs so progress is visible. Build with the CUDA feature
//! (`cargo build --release`) to train on the GPU.
//!
//! The A/B benchmark against the old GAN trainer (`--recon-bench`) was removed along
//! with the GAN tab/trainer (legacy, unused) — this file now only drives the
//! `recon_simple` (UNet, no adversarial loss) trainer used in production.

use std::path::PathBuf;
use std::sync::{mpsc, Arc};
use std::sync::atomic::AtomicBool;

use rand::{rngs::SmallRng, SeedableRng, seq::SliceRandom};

use crate::tabs::leaf_seg::inference::list_images;
use crate::tabs::recon_train::training::DamageParams;
use crate::tabs::recon_simple::training::{
    SimpleTrainConfig, SimpleTrainMsg, spawn_simple_training,
};

const SEED: u64 = 0xA11CE5;

// Oak FiLM conditioning — Lobed shape (index 0). MUST match train + eval.
const OAK_SHAPE:  u32 = 0;          // LeafShape::Lobed
const OAK_MARGIN: u32 = 0;          // MarginType::default()

/// Margin-loss-only damage: marginal modes ON (coastal/snake/apex/clusters/lobe/
/// focal_sector), interior modes OFF (spots/ellipses). coastal/lobe/clusters/apex
/// up-weighted per the oak herbivory profile.
fn margin_loss_damage() -> DamageParams {
    DamageParams {
        min_pct:        6.0,
        max_pct:        40.0,   // realistic oak margin loss (matches the 40% cap)
        coastal:        true,  coastal_w:       1.00,  // edge nibble  (up-weighted)
        spots:          false, spots_w:         0.00,  // INTERIOR — off
        snake:          true,  snake_w:         0.50,  // margin notch
        ellipses:       false, ellipses_w:      0.00,  // INTERIOR — off
        apex:           true,  apex_w:          0.40,  // tip loss     (up-weighted)
        clusters:       true,  clusters_w:      1.00,  // margin bites (up-weighted)
        lobe:           true,  lobe_w:          1.00,  // whole-lobe   (up-weighted)
        focal_sector:   true,  focal_sector_w:  0.60,  // margin wedge
        zero_damage_prob: 0.40,  // ↑ suppress over-fill of intact margins / natural sinuses
        curriculum_max:   35.0,
    }
}

// ── Standalone clean Recon training run (GPU) ───────────────────────────────────
//
// `lacuna --recon-train <folder> [epochs] [max_images]`

/// Train resolution for the standalone run (matches pipeline RECON_SIZE=512).
const TRAIN_IMG: usize = 512;

pub fn run_training(folder: &str, epochs: usize, max_images: usize) {
    let epochs     = if epochs == 0 { 120 } else { epochs };
    let max_images = if max_images == 0 { 8000 } else { max_images };

    let mut all = list_images(&PathBuf::from(folder));
    if all.is_empty() { eprintln!("[recon-train] no images in {folder} — abort"); return; }

    let out = PathBuf::from(folder).parent()
        .map(|p| p.join("recon_train_out"))
        .unwrap_or_else(|| PathBuf::from("recon_train_out"));
    let _ = std::fs::create_dir_all(&out);

    // Durable log: the release build is a `windows`-subsystem app (no console), so
    // mirror every line to <out>/train.log. Watch with `Get-Content train.log -Wait`.
    let mut logf = std::fs::File::create(out.join("train.log")).ok();
    let mut logln = |s: String| {
        println!("{s}");
        if let Some(f) = logf.as_mut() { use std::io::Write; let _ = writeln!(f, "{s}"); let _ = f.flush(); }
    };

    logln(format!("[recon-train] clean Recon run  folder={folder}  epochs={epochs}  img={TRAIN_IMG}  batch={BATCH_TRAIN}(x{ACCUM} accum)"));
    logln(format!("[recon-train] output → {}", out.display()));

    // Pre-flight: the recon needs RGBA CUTOUTS (alpha carries the leaf silhouette).
    // Solid images (alpha all opaque) would make GT = the whole frame → meaningless.
    if let Ok(img) = image::open(&all[0]) {
        let rgba = img.to_rgba8();
        let (mut opaque, mut transparent) = (0u64, 0u64);
        for p in rgba.pixels() { if p[3] > 128 { opaque += 1 } else { transparent += 1 } }
        let total = (opaque + transparent).max(1);
        if transparent == 0 {
            logln("[recon-train] WARNING: first image has NO transparent pixels — these look like".into());
            logln("              SOLID images, not leaf cutouts. Recon needs RGBA cutouts (alpha =".into());
            logln("              leaf silhouette). Training on solid images will be meaningless.".into());
        } else {
            logln(format!("[recon-train] cutout check OK  ({:.0}% leaf / {:.0}% background)",
                     100.0 * opaque as f64 / total as f64, 100.0 * transparent as f64 / total as f64));
        }
    }

    let mut rng = SmallRng::seed_from_u64(SEED);
    all.shuffle(&mut rng);
    if all.len() > max_images { all.truncate(max_images); }
    let n_val = ((all.len() as f32 * 0.10).round() as usize).clamp(1, all.len().saturating_sub(1));
    let val_paths:   Vec<PathBuf> = all[..n_val].to_vec();
    let train_paths: Vec<PathBuf> = all[n_val..].to_vec();
    logln(format!("[recon-train] {} leaves → {} train / {} val  (image cache ~{} MB RAM)",
             all.len(), train_paths.len(), val_paths.len(),
             all.len() * TRAIN_IMG * TRAIN_IMG * 4 / 1_000_000));

    let cfg = SimpleTrainConfig {
        train_paths,
        val_paths,
        output_dir:         out.clone(),
        epochs,
        batch_size:         BATCH_TRAIN,
        lr:                 2e-4,
        l1_lambda:          10.0,
        tv_lambda:          0.05,
        conf_lambda:        0.5,
        recon_focus_weight: 4.0,
        tversky_alpha:      0.92,
        tversky_beta:       0.08,
        checkpoint_every:   5,          // best updates every 5 epochs → safe to stop early
        sample_every:       5,          // save a preview grid every 5 epochs
        image_size:         TRAIN_IMG,
        damage_params:      margin_loss_damage(),
        curriculum_epochs:  epochs / 4,
        resume_from:        None,
        start_epoch:        0,
        resume_best_iou:    0.0,
        lr_min_frac:        0.05,
        accum_steps:        ACCUM,
        leaf_shape:         OAK_SHAPE,
        margin_type:        OAK_MARGIN,
        pretrain_epochs:    0,
        d_lr_factor:        0.5,
        adv_lambda:         0.0,
        area_lambda:        3.0,
        boundary_lambda:    3.0,
        boundary_px:        3,
        hole_lambda:        3.0,
    };

    let (tx, rx) = mpsc::channel();
    spawn_simple_training(cfg, tx, Arc::new(AtomicBool::new(false)));
    let sdir = out.join("samples");
    let _ = std::fs::create_dir_all(&sdir);
    for msg in rx {
        match msg {
            SimpleTrainMsg::Log(s) => logln(format!("   {s}")),
            SimpleTrainMsg::EpochMetrics { epoch, metrics } =>
                logln(format!("[epoch {epoch}/{epochs}] IoU={:.4} Dice={:.4} Prec={:.4} Rec={:.4}",
                         metrics.iou, metrics.dice, metrics.precision, metrics.recall)),
            SimpleTrainMsg::SampleGrid { epoch, pixels, width, height } => {
                if let Some(img) = image::RgbaImage::from_raw(width as u32, height as u32, pixels) {
                    let _ = img.save(sdir.join(format!("epoch_{epoch:04}.png")));
                    logln(format!("   preview → samples/epoch_{epoch:04}.png"));
                }
            }
            SimpleTrainMsg::Checkpoint { path } => logln(format!("   ckpt: {path}")),
            SimpleTrainMsg::Error(e) => logln(format!("[recon-train ERROR] {e}")),
            SimpleTrainMsg::Finished => break,
            _ => {}
        }
    }
    logln(format!("\n[recon-train] DONE. Best model: {}\\checkpoint_best\\gen.mpk", out.display()));
    logln("[recon-train] Deploy it: copy that gen.mpk over".into());
    logln("              E:\\PhD_TobiMu\\02_code\\FoliarToolbox\\models\\recon\\gen.mpk".into());
}

const BATCH_TRAIN: usize = 1;   // batch 1 → half the activation VRAM (GroupNorm needs no batch stats)
const ACCUM:       usize = 4;   // effective batch 4, no extra VRAM
