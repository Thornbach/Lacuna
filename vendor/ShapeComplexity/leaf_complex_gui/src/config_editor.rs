// config_editor.rs - Configuration Editor Window for LeafComplexR GUI

use eframe::egui;
use crate::config::Config;

pub struct ConfigEditor {
    config: Config,
    original_config: Config,
}

impl ConfigEditor {
    pub fn new(config: Config) -> Self {
        Self {
            config: config.clone(),
            original_config: config,
        }
    }
    
    pub fn get_config(&self) -> Config {
        self.config.clone()
    }
    
    /// Show the configuration editor window
    /// Returns true if the config was saved
    pub fn show(&mut self, ctx: &egui::Context, open: &mut bool) -> bool {
        let mut saved = false;
        
        egui::Window::new("⚙️ Configuration")
            .open(open)
            .default_size([500.0, 600.0])
            .resizable(true)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Adaptive Opening");
                    ui.separator();
                    
                    egui::Grid::new("adaptive_opening_grid")
                        .num_columns(2)
                        .spacing([10.0, 5.0])
                        .show(ui, |ui| {
                            ui.label("Max Density (%):");
                            ui.add(egui::DragValue::new(&mut self.config.adaptive_opening_max_density)
                                .speed(0.5)
                                .range(1.0..=100.0));
                            ui.end_row();
                            
                            ui.label("Max Opening (%):");
                            ui.add(egui::DragValue::new(&mut self.config.adaptive_opening_max_percentage)
                                .speed(0.1)
                                .range(1.0..=50.0));
                            ui.end_row();
                            
                            ui.label("Min Opening (%):");
                            ui.add(egui::DragValue::new(&mut self.config.adaptive_opening_min_percentage)
                                .speed(0.1)
                                .range(0.1..=20.0));
                            ui.end_row();
                        });
                    
                    ui.add_space(15.0);
                    ui.heading("Petiole Filtering");
                    ui.separator();
                    
                    ui.checkbox(&mut self.config.enable_petiole_filter_ec, "Enable EC Petiole Filter");
                    ui.checkbox(&mut self.config.enable_petiole_filter_mc, "Enable MC Petiole Filter");
                    ui.checkbox(&mut self.config.petiole_remove_completely, "Remove Petiole Completely");
                    
                    ui.add_space(15.0);
                    ui.heading("Pink Threshold Filtering");
                    ui.separator();
                    
                    ui.checkbox(&mut self.config.enable_pink_threshold_filter, "Enable Pink Threshold Filter");
                    
                    egui::Grid::new("pink_threshold_grid")
                        .num_columns(2)
                        .spacing([10.0, 5.0])
                        .show(ui, |ui| {
                            ui.label("Threshold Value:");
                            ui.add(egui::DragValue::new(&mut self.config.pink_threshold_value)
                                .speed(0.1)
                                .range(0.0..=100.0));
                            ui.end_row();
                        });
                    
                    ui.add_space(15.0);
                    ui.heading("Thornfiddle Parameters");
                    ui.separator();
                    
                    egui::Grid::new("thornfiddle_grid")
                        .num_columns(2)
                        .spacing([10.0, 5.0])
                        .show(ui, |ui| {
                            ui.label("Smoothing Strength:");
                            ui.add(egui::DragValue::new(&mut self.config.thornfiddle_smoothing_strength)
                                .speed(0.1)
                                .range(0.5..=10.0));
                            ui.end_row();
                            
                            ui.label("Max Opening (%):");
                            ui.add(egui::DragValue::new(&mut self.config.thornfiddle_max_opening_percentage)
                                .speed(0.5)
                                .range(5.0..=50.0));
                            ui.end_row();
                            
                            ui.label("Min Opening (%):");
                            ui.add(egui::DragValue::new(&mut self.config.thornfiddle_min_opening_percentage)
                                .speed(0.1)
                                .range(1.0..=20.0));
                            ui.end_row();
                            
                            ui.label("Pixel Threshold:");
                            ui.add(egui::DragValue::new(&mut self.config.thornfiddle_pixel_threshold)
                                .speed(1.0)
                                .range(1..=50));
                            ui.end_row();
                        });
                    
                    ui.add_space(15.0);
                    ui.heading("Harmonic Enhancement (MC only)");
                    ui.separator();
                    
                    egui::Grid::new("harmonic_grid")
                        .num_columns(2)
                        .spacing([10.0, 5.0])
                        .show(ui, |ui| {
                            ui.label("Max Harmonics:");
                            ui.add(egui::DragValue::new(&mut self.config.harmonic_max_harmonics)
                                .speed(1.0)
                                .range(1..=50));
                            ui.end_row();
                            
                            ui.label("Strength Multiplier:");
                            ui.add(egui::DragValue::new(&mut self.config.harmonic_strength_multiplier)
                                .speed(0.1)
                                .range(0.5..=5.0));
                            ui.end_row();
                            
                            ui.label("Min Chain Length:");
                            ui.add(egui::DragValue::new(&mut self.config.harmonic_min_chain_length)
                                .speed(1.0)
                                .range(5..=100));
                            ui.end_row();
                        });
                    
                    ui.add_space(15.0);
                    ui.heading("Entropy Parameters");
                    ui.separator();
                    
                    egui::Grid::new("entropy_grid")
                        .num_columns(2)
                        .spacing([10.0, 5.0])
                        .show(ui, |ui| {
                            ui.label("Approximate Entropy M:");
                            ui.add(egui::DragValue::new(&mut self.config.approximate_entropy_m)
                                .speed(1.0)
                                .range(1..=10));
                            ui.end_row();
                            
                            ui.label("Approximate Entropy R:");
                            ui.add(egui::DragValue::new(&mut self.config.approximate_entropy_r)
                                .speed(0.01)
                                .range(0.01..=1.0));
                            ui.end_row();
                            
                            ui.label("Spectral Sigmoid K:");
                            ui.add(egui::DragValue::new(&mut self.config.spectral_entropy_sigmoid_k)
                                .speed(1.0)
                                .range(1.0..=100.0));
                            ui.end_row();
                            
                            ui.label("Spectral Sigmoid C:");
                            ui.add(egui::DragValue::new(&mut self.config.spectral_entropy_sigmoid_c)
                                .speed(0.001)
                                .range(0.001..=0.5));
                            ui.end_row();
                        });
                    
                    ui.add_space(20.0);
                    ui.separator();
                    
                    ui.horizontal(|ui| {
                        if ui.button("Save").clicked() {
                            self.original_config = self.config.clone();
                            saved = true;
                        }
                        
                        if ui.button("Reset").clicked() {
                            self.config = self.original_config.clone();
                        }
                        
                        if ui.button("Defaults").clicked() {
                            self.config = Config::default();
                        }
                    });
                });
            });
        
        saved
    }
}