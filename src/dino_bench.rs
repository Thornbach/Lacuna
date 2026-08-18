//! Headless measurement of DINO throughput, for costing per-region embeddings.
//!
//! The question this answers: "Rank by appearance" needs one DINO forward pass
//! per region, and a busy leaf can carry 6,000 regions. Detection itself only
//! runs one pass per TILE — roughly two dozen for a whole leaf. So the ranking
//! feature is potentially two orders of magnitude more GPU work than the
//! detection it is ranking, and that has to be measured rather than assumed
//! before it is offered as a button.
//!
//! Run: `lacuna --dino-bench [dino_weights.safetensors] [res] [n_regions]`

use std::time::Instant;

use image::RgbImage;

use crate::tabs::pipeline::dino::DinoExtractor;

pub fn run(model: &str, res: u32, n: usize) {
    println!("DINO throughput bench");
    println!("  model : {model}");
    println!("  res   : {res}");
    println!("  crops : {n}\n");

    let mut dino = match DinoExtractor::load(std::path::Path::new(model), res) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("could not load DINO: {e}");
            return;
        }
    };

    // Synthetic crops at the size `embed_crop` produces. Content does not affect
    // ViT cost — the patch count does, and that is fixed by `res`.
    let make = |seed: u8| -> RgbImage {
        RgbImage::from_fn(res, res, |x, y| {
            image::Rgb([
                ((x * 7 + seed as u32) % 255) as u8,
                ((y * 5 + seed as u32) % 255) as u8,
                ((x + y) % 255) as u8,
            ])
        })
    };

    // Warm up: the first calls pay CUDA/cuDNN autotune and allocator growth, and
    // including them would overstate the steady-state cost by a lot.
    let warm: Vec<RgbImage> = (0..4).map(make).collect();
    for _ in 0..2 {
        let _ = dino.features_batch_at(&warm, res);
    }

    // EMBED_BATCH in worker.rs. Batching matters because per-call overhead
    // dominates at this crop size.
    const BATCH: usize = 16;
    let crops: Vec<RgbImage> = (0..BATCH).map(|i| make(i as u8)).collect();

    let batches = n.div_ceil(BATCH);
    let t0 = Instant::now();
    let mut gpu_ms = 0f64;
    let mut prep_ms = 0f64;
    for _ in 0..batches {
        match dino.features_batch_at(&crops, res) {
            Ok(_) => {
                gpu_ms += dino.last_ms as f64;
                prep_ms += dino.last_prep_ms as f64;
            }
            Err(e) => {
                eprintln!("batch failed: {e}");
                return;
            }
        }
    }
    let wall = t0.elapsed().as_secs_f64();
    let done = batches * BATCH;

    println!("{done} crops in {:.2}s", wall);
    println!("  {:.2} ms/crop wall", wall * 1000.0 / done as f64);
    println!("  {:.2} ms/crop gpu forward", gpu_ms / done as f64);
    println!("  {:.2} ms/crop cpu prep (resize + CHW)", prep_ms / done as f64);
    println!("  {:.0} crops/second\n", done as f64 / wall);

    let per = wall / done as f64;
    println!("projected cost of ranking one family by appearance:");
    for regions in [100usize, 341, 1000, 3000, 6000] {
        println!("  {regions:>5} regions : {:>7}", fmt_secs(per * regions as f64));
    }
    println!("\nfor comparison, DETECTION on one leaf is ~24 tile passes at the");
    println!("same cost class, i.e. well under a second.");
}

fn fmt_secs(s: f64) -> String {
    if s < 1.0 { format!("{:.0} ms", s * 1000.0) }
    else if s < 90.0 { format!("{s:.1} s") }
    else { format!("{:.1} min", s / 60.0) }
}
