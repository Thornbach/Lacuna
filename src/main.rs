#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod dino_burn;
mod dino_validate;
mod recon_bench;
mod settings;
mod yolo_burn;
mod yolo_validate;
mod smoke;
mod tabs;
mod theme;
mod ui_kit;
mod widgets;

// Leaf icon generated at compile time by build.rs (32×32 RGBA)
const LEAF_RGBA: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/leaf_rgba.bin"));

fn main() -> eframe::Result<()> {
    // headless end-to-end smoke test: `lacuna --smoke <image-folder>`
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--smoke") {
        let folder = args.get(pos + 1).cloned().unwrap_or_default();
        smoke::run(&folder);
        return Ok(());
    }
    // headless oak reconstruction A/B: `--recon-bench <oak-folder> [epochs] [max_images]`
    if let Some(pos) = args.iter().position(|a| a == "--recon-bench") {
        let folder     = args.get(pos + 1).cloned().unwrap_or_default();
        let epochs     = args.get(pos + 2).and_then(|s| s.parse().ok()).unwrap_or(0);
        let max_images = args.get(pos + 3).and_then(|s| s.parse().ok()).unwrap_or(0);
        recon_bench::run(&folder, epochs, max_images);
        return Ok(());
    }
    // headless BURN DINOv3 vs ONNX validation: `--dino-burn-validate [ref.st] [weights.st]`
    if let Some(pos) = args.iter().position(|a| a == "--dino-burn-validate") {
        let ref_path = args.get(pos + 1).cloned()
            .unwrap_or_else(|| "port/dino_ref.safetensors".into());
        let weights = args.get(pos + 2).cloned()
            .unwrap_or_else(|| "port/dino_weights.safetensors".into());
        dino_validate::run(&ref_path, &weights);
        return Ok(());
    }
    // headless BURN YOLO26 vs reference validation: `--yolo-burn-validate [ref.st] [weights.st]`
    if let Some(pos) = args.iter().position(|a| a == "--yolo-burn-validate") {
        let ref_path = args.get(pos + 1).cloned()
            .unwrap_or_else(|| r"e:\PhD_TobiMu\02_code\FoliarToolbox\port\yolo_ref.safetensors".into());
        let weights = args.get(pos + 2).cloned()
            .unwrap_or_else(|| r"e:\PhD_TobiMu\02_code\FoliarToolbox\port\yolo_weights.safetensors".into());
        yolo_validate::run(&ref_path, &weights);
        return Ok(());
    }
    // headless end-to-end YOLO ort-vs-BURN mask comparison on a real image:
    //   `--yolo-seg-compare [image] [onnx] [weights]`
    if let Some(pos) = args.iter().position(|a| a == "--yolo-seg-compare") {
        let image = args.get(pos + 1).cloned().unwrap_or_else(||
            r"E:\PhD_TobiMu\02_code\02paper\leaf_segmentation\test\8f6a70ebb6ae0c10090b5ac3ec98e837.jpg".into());
        let onnx = args.get(pos + 2).cloned().unwrap_or_else(||
            r"e:\PhD_TobiMu\02_code\FoliarToolbox\models\yolo.onnx".into());
        let weights = args.get(pos + 3).cloned().unwrap_or_else(||
            r"e:\PhD_TobiMu\02_code\FoliarToolbox\port\yolo_weights.safetensors".into());
        yolo_validate::compare_seg(&image, &onnx, &weights);
        return Ok(());
    }
    // headless clean Recon training run (GPU): `--recon-train <folder> [epochs] [max_images]`
    if let Some(pos) = args.iter().position(|a| a == "--recon-train") {
        let folder     = args.get(pos + 1).cloned().unwrap_or_default();
        let epochs     = args.get(pos + 2).and_then(|s| s.parse().ok()).unwrap_or(0);
        let max_images = args.get(pos + 3).and_then(|s| s.parse().ok()).unwrap_or(0);
        recon_bench::run_training(&folder, epochs, max_images);
        return Ok(());
    }

    let icon = egui::viewport::IconData {
        rgba:   LEAF_RGBA.to_vec(),
        width:  32,
        height: 32,
    };

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([960.0, 600.0])
            .with_title("Lacuna")
            .with_icon(std::sync::Arc::new(icon)),
        ..Default::default()
    };

    eframe::run_native(
        "Lacuna",
        native_options,
        Box::new(|cc| Ok(Box::new(app::LacunaApp::new(cc)))),
    )
}
