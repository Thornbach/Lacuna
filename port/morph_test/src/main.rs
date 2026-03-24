//! Measure the morphology pipeline per-step (set MORPH_TIMING=1) + report EC/MC
//! metrics — to find the real bottleneck and debug the EC-length issue.
//! Run: MORPH_TIMING=1 cargo run --release -- "<leaf.png>"

use leaf_complex_rust_lib::analyze_rgba;
use leaf_complex_rust_lib::config::Config;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        r"E:\PhD_TobiMu\01_data\processed\sorted\Healthy\july\2L1_R5_u.png".to_string()
    });
    let img = image::open(&path).expect("open image").to_rgba8();
    let (w, h) = img.dimensions();
    let raw = img.into_raw();
    let cfg = Config::default();
    println!("leaf {}x{}  resize={:?}", w, h, cfg.resize_dimensions);

    for i in 0..3 {
        let t = Instant::now();
        match analyze_rgba(&raw, w, h, &cfg) {
            Ok(r) => {
                let m = &r.metrics;
                println!(
                    "\nrun {i}: TOTAL {:.0}ms | EC_len={:.1} EC_out={} EC_area={} | MC_len={:.1} MC_out={} MC_area={}",
                    t.elapsed().as_secs_f64() * 1000.0,
                    m.ec_length, m.ec_outline_count, m.ec_area,
                    m.mc_length, m.mc_outline_count, m.mc_area
                );
            }
            Err(e) => println!("run {i}: ERROR {e}"),
        }
    }
}
