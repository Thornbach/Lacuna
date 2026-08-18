//! Headless end-to-end smoke test (`lacuna --smoke <image-folder>`).
//! Runs the REAL pipeline worker (segment -> tile -> few-shot detect -> cluster) on
//! a folder of full-leaf images, then exercises the flywheel (write curations ->
//! retrain head). Prints a report and exits. Not part of the GUI.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{mpsc, Arc};
use std::sync::atomic::AtomicBool;

use crate::tabs::leaf_seg::inference::list_images;
use crate::tabs::pipeline::worker::{spawn_pipeline, AnomalyRegion, PipeConfig, PipeMsg};
use crate::tabs::train::head::{spawn_retrain, RetrainCfg, RetrainMsg};

const YOLO: &str = r"E:\PhD_TobiMu\02_code\02paper\leaf_segmentation\runs\segment\runs\segment\leaf_seg\weights\best.onnx";
const DINO: &str = r"E:\PhD_TobiMu\02_code\02paper\anomaly\dinov3_vitb16_512.onnx";
/// 256-resolution export, for builds whose `default_dino_res()` is 256.
///
/// The ort path cannot resize: `1Help/export_dinov3.py` exports with
/// `dynamic_axes=None`, so the graph's input is fixed and feeding it any other
/// resolution fails outright. Picking the model to match the resolution is what
/// makes `--smoke` measure the configuration that actually ships instead of one
/// that cannot load.
const DINO_256: &str = r"E:\PhD_TobiMu\02_code\FoliarToolbox\models\cpu256\dino.onnx";
const HEAD: &str = r"E:\PhD_TobiMu\02_code\FoliarToolbox\models\fewshot_head.json";
const RECON: &str = r"E:\PhD_TobiMu\02_code\FoliarToolbox\models\recon";
const OUT:  &str = r"E:\PhD_TobiMu\02_code\FoliarToolbox\smoke_out";

pub fn run(folder: &str) {
    let images = list_images(&PathBuf::from(folder));
    println!("[smoke] {} images in {folder}", images.len());
    if images.is_empty() {
        eprintln!("[smoke] no images — abort");
        return;
    }
    let out = PathBuf::from(OUT);
    let dino_res = crate::tabs::pipeline::worker::default_dino_res();
    let dino_model = if dino_res == 256 { DINO_256 } else { DINO };
    println!("[smoke] dino_res={dino_res} model={dino_model}");
    let cfg = PipeConfig {
        image_paths: images,
        output_dir: out.clone(),
        yolo_model: PathBuf::from(YOLO),
        dino_model: PathBuf::from(dino_model),
        bank_path: PathBuf::new(),  // unused on the few-shot path
        meta_path: PathBuf::new(),
        tile_size: 256,
        margin_erode: 6,
        detect_holes: true,
        min_hole_area: 16,
        // Off here on purpose: the smoke test asserts the pipeline's DEFAULT
        // behaviour, and this filter is opt-in.
        filter_margin_holes: false,
        hole_margin_px: 16,
        // Follow the build's own default so --smoke measures the configuration
        // that actually ships, not a hardcoded 512 the CPU variant never uses.
        dino_res,
        conf: 0.25,
        recon_ckpt: Some(PathBuf::from(RECON)),
        head_path: Some(PathBuf::from(HEAD)),
        use_patchcore: false,
        unsupervised_families: false,
        domain_projection: false,
        // Also follow the build default: CPU relaxes tau to 0.60 to recover the
        // per-family recall that running DINO at 256 costs.
        head_tau: crate::settings::default_head_tau(),
        head_grow: 0.7f32.min(crate::settings::default_head_tau()),
        seg_alpha_lo: 0.50,
        seg_chroma_min: 28,
        cluster_eps: 1.5,
        cluster_min_pts: 5,
        cluster_algo: crate::settings::ClusterAlgo::Dbscan,
        target_k: 0,
        cut_mode: crate::settings::CutMode::FixedK,
        adaptive_threshold: 8.0,
    };

    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    spawn_pipeline(cfg, tx, cancel);

    let mut n_leaves = 0usize;
    let mut leaf_regions = Vec::new();
    let mut regions: Vec<AnomalyRegion> = Vec::new();
    let mut labels: Vec<i32> = Vec::new();
    let mut names: HashMap<i32, String> = HashMap::new();
    loop {
        match rx.recv() {
            Ok(PipeMsg::Stage(s)) => println!("[stage] {s}"),
            Ok(PipeMsg::Log(l)) => println!("   {l}"),
            Ok(PipeMsg::Leaf(l)) => {
                n_leaves += 1;
                leaf_regions.push(l.n_regions);
                println!("[leaf {n_leaves}] {}x{}  regions={}  recon_added={}  recon_whole={}  morph={}",
                         l.w, l.h, l.n_regions, l.recon_area, l.recon_whole, l.morph.is_some());
            }
            Ok(PipeMsg::Clusters { labels: lb, names: nm, regions: rg, .. }) => {
                labels = lb; names = nm; regions = rg;
            }
            Ok(PipeMsg::Error(e)) => println!("[ERROR] {e}"),
            Ok(PipeMsg::Progress { .. }) => {}
            Ok(PipeMsg::Finished) => break,
            Err(_) => break,
        }
    }

    let n_clusters = labels.iter().copied().filter(|&l| l >= 0)
        .collect::<std::collections::HashSet<_>>().len();
    println!("\n[smoke] PIPELINE: {n_leaves} leaves, {} regions, {n_clusters} families",
             regions.len());
    let mut by_fam: HashMap<String, usize> = HashMap::new();
    for (i, _) in regions.iter().enumerate() {
        let cid = labels.get(i).copied().unwrap_or(-1);
        let nm = names.get(&cid).cloned().unwrap_or_else(|| format!("Cluster {cid}"));
        *by_fam.entry(nm).or_insert(0) += 1;
    }
    for (k, v) in &by_fam {
        println!("   {k}: {v} regions");
    }

    if regions.is_empty() {
        println!("[smoke] no regions detected — skipping flywheel test");
        return;
    }

    // ── flywheel: write curations (crops + labels.jsonl), then retrain ──
    println!("\n[smoke] FLYWHEEL: writing curations + retraining head…");
    let cur = out.join("curations");
    let labels_dir = cur.join("labels");
    let _ = std::fs::create_dir_all(&labels_dir);
    let mut jsonl = String::new();
    for (i, r) in regions.iter().enumerate() {
        let cid = labels.get(i).copied().unwrap_or(-1);
        if cid < 0 { continue; }
        let fam = names.get(&cid).cloned().unwrap_or_else(|| format!("Cluster {cid}"));
        let fname = format!("r_{i}.png");
        if let Some(img) = image::RgbaImage::from_raw(r.crop_size, r.crop_size, r.crop.clone()) {
            let _ = img.save(labels_dir.join(&fname));
        }
        jsonl.push_str(&format!(
            "{{\"crop\":\"{fname}\",\"family\":\"{fam}\",\"source\":\"confirm\",\"leaf_src\":\"\",\"ts\":0}}\n"));
    }
    let _ = std::fs::write(cur.join("labels.jsonl"), jsonl);

    let (tx2, rx2) = mpsc::channel();
    spawn_retrain(
        RetrainCfg {
            head_path: PathBuf::from(HEAD),
            // Same model the pipeline just ran on. Using the 512 graph here
            // while detection ran at 256 would featurise the curations in a
            // different space than the head is about to be applied in.
            dino_model: PathBuf::from(dino_model),
            curations_dir: cur,
            out_path: out.join("fewshot_head_retrained.json"),
            max_iters: 60,
            l2_reg: 1.0, // sklearn's C — see RetrainCfg::l2_reg
            max_patches_per_crop: 8,
            validate_healthy_dir: None,
            validate_tau: 0.6,
            cold_start: false,
            dump_dir: None,
            base_set: None,
            base_rows: 0,
            anchor: 0.0, // no base head to anchor to in the smoke fixture
        },
        tx2,
        Arc::new(AtomicBool::new(false)),
    );
    loop {
        match rx2.recv() {
            Ok(RetrainMsg::Stage(s)) => println!("[retrain] {s}"),
            Ok(RetrainMsg::Log(l)) => println!("   {l}"),
            Ok(RetrainMsg::Error(e)) => { println!("[retrain ERROR] {e}"); break; }
            Ok(RetrainMsg::Done(s)) => { println!("[smoke] {s}"); break; }
            Err(_) => break,
        }
    }
    let outhead = out.join("fewshot_head_retrained.json");
    println!("\n[smoke] DONE. retrained head exists: {}", outhead.exists());
}
