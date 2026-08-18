// main.rs - LeafComplexR GUI Application Entry Point

use eframe::egui;
use leaf_complex_gui::LeafComplexApp;

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("LeafComplexR - Edge & Margin Complexity Analysis"),
        ..Default::default()
    };
    
    eframe::run_native(
        "LeafComplexR",
        native_options,
        Box::new(|cc| Ok(Box::new(LeafComplexApp::new(cc)))),
    )
}