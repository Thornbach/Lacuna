// leaf_complex_gui/src/ui.rs
// Contains all UI rendering functions

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Points};
use std::sync::{Arc, Mutex};

use crate::state::{AppState, AnalysisStatus, PointType};

const PINK_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 0, 255);
const YELLOW_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 215, 0);
const RED_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 0, 0);
const BLUE_COLOR: egui::Color32 = egui::Color32::from_rgb(0, 120, 255);
const CYAN_COLOR: egui::Color32 = egui::Color32::from_rgb(0, 255, 255);

pub fn render_image_view(
    ui: &mut egui::Ui,
    state: &Arc<Mutex<AppState>>,
    _ctx: &egui::Context,
    analyze_clicked: &mut bool,
    batch_clicked: &mut bool,
) {
    // FIX 4: current_filename() does not exist on AppState.
    //         Use current_image() and access .filename instead.
    let filename = {
        let state_guard = state.lock().unwrap();
        state_guard
            .current_image()
            .map(|img| img.filename.clone())
            .unwrap_or_else(|| "No image selected".to_string())
    };

    ui.heading(format!("🖼️ Image View — {}", filename));
    ui.separator();

    ui.horizontal(|ui| {
        // Reset View button — must drop guard before relocking
        let zoom_label = {
            let state_guard = state.lock().unwrap();
            format!("Zoom: {:.0}%", state_guard.zoom_level * 100.0)
        };

        if ui.button("Reset View").clicked() {
            state.lock().unwrap().reset_view();
        }

        ui.label(zoom_label);
        ui.separator();

        // Overlay toggles
        let (mut show_ec, mut show_mc, mut show_path) = {
            let g = state.lock().unwrap();
            (g.show_ec_overlay, g.show_mc_overlay, g.show_path_overlay)
        };

        if ui.checkbox(&mut show_ec, "EC Overlay").changed() {
            state.lock().unwrap().show_ec_overlay = show_ec;
        }
        if ui.checkbox(&mut show_mc, "MC Overlay").changed() {
            state.lock().unwrap().show_mc_overlay = show_mc;
        }
        if ui.checkbox(&mut show_path, "Show Path").changed() {
            state.lock().unwrap().show_path_overlay = show_path;
        }

        ui.separator();

        let (has_current, analyzing, batch_processing, selected_count) = {
            let g = state.lock().unwrap();
            (
                g.current_image().is_some(),
                g.analysis_in_progress,
                g.batch_processing,
                g.selected_count(),
            )
        };

        ui.add_enabled_ui(has_current && !analyzing && !batch_processing, |ui| {
            if ui.button("▶ Analyze").clicked() {
                *analyze_clicked = true;
            }
        });

        let batch_label = if selected_count > 0 {
            format!("⏩ Analyze Selected ({})", selected_count)
        } else {
            "⏩ Analyze All".to_string()
        };

        ui.add_enabled_ui(!analyzing && !batch_processing, |ui| {
            if ui.button(batch_label).clicked() {
                *batch_clicked = true;
            }
        });
    });

    ui.separator();

    let response = ui.allocate_rect(
        ui.available_rect_before_wrap(),
        egui::Sense::click_and_drag(),
    );
    let rect = response.rect;

    let state_guard = state.lock().unwrap();

    if let Some(result) = state_guard.current_result() {
        let painter = ui.painter_at(rect);

        if let Some(original_texture) = &result.original_texture {
            let tex_size = original_texture.size_vec2();
            let zoom = state_guard.zoom_level;
            let pan = state_guard.pan_offset;

            let scaled_size = egui::vec2(tex_size.x * zoom, tex_size.y * zoom);
            let img_rect = egui::Rect::from_center_size(rect.center() + pan, scaled_size);

            painter.image(
                original_texture.id(),
                img_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );

            if state_guard.show_ec_overlay {
                if let Some(ec_texture) = &result.ec_image_texture {
                    painter.image(
                        ec_texture.id(),
                        img_rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                }
            }

            if state_guard.show_mc_overlay {
                if let Some(mc_texture) = &result.mc_image_texture {
                    painter.image(
                        mc_texture.id(),
                        img_rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                }
            }

            let pixel_to_screen = |pixel_x: u32, pixel_y: u32| -> egui::Pos2 {
                let img_origin = img_rect.min;
                let scale_x = img_rect.width() / tex_size.x;
                let scale_y = img_rect.height() / tex_size.y;
                egui::pos2(
                    img_origin.x + pixel_x as f32 * scale_x,
                    img_origin.y + pixel_y as f32 * scale_y,
                )
            };

            if state_guard.show_path_overlay {
                if let Some(point_idx) = state_guard.selected_point {
                    let (features, reference_point, contour) = match state_guard.selected_point_type {
                        PointType::EC => (
                            &result.ec_features,
                            result.ec_reference_point,
                            &result.ec_contour,
                        ),
                        PointType::MC => (
                            &result.mc_features,
                            result.mc_reference_point,
                            &result.mc_contour,
                        ),
                    };

                    if point_idx < features.len() && point_idx < contour.len() {
                        let (contour_x, contour_y) = contour[point_idx];
                        let contour_screen = pixel_to_screen(contour_x, contour_y);
                        let ref_screen = pixel_to_screen(reference_point.0, reference_point.1);

                        let straight_len = features[point_idx].straight_path_length;
                        let curved_len = features[point_idx].diego_path_length;

                        // If BFS was triggered, curved > straight (path navigated around a
                        // concave margin gap). Draw orange dashed-style line to signal this.
                        // If straight == curved, the geodesic IS the straight line — draw blue.
                        let bfs_used = (curved_len - straight_len).abs() > 0.5;
                        let line_color = if bfs_used {
                            egui::Color32::from_rgb(255, 165, 0) // orange = BFS detour
                        } else {
                            BLUE_COLOR // blue = straight path used
                        };

                        painter.line_segment(
                            [ref_screen, contour_screen],
                            egui::Stroke::new(2.0 * zoom, line_color),
                        );

                        painter.circle_filled(ref_screen, 4.0 * zoom, CYAN_COLOR);
                        painter.circle_filled(contour_screen, 5.0 * zoom, RED_COLOR);

                        let midpoint = egui::pos2(
                            (ref_screen.x + contour_screen.x) / 2.0,
                            (ref_screen.y + contour_screen.y) / 2.0,
                        );

                        let label_text = match state_guard.selected_point_type {
                            PointType::EC => {
                                let pink_pixels = features[point_idx].diego_path_pink.unwrap_or(0);
                                if bfs_used {
                                    // The leaf has a concave margin here — the geodesic detoured
                                    // around a gap. The straight line shown is an approximation;
                                    // the actual path is longer and crosses more pink margin pixels.
                                    format!("Straight: {:.1} (approx)\nGeodesic: {:.1} [BFS]\nMargin px: {}",
                                        straight_len, curved_len, pink_pixels)
                                } else {
                                    format!("Straight: {:.1}\nGeodesic: {:.1}\nMargin px: {}",
                                        straight_len, curved_len, pink_pixels)
                                }
                            }
                            PointType::MC => {
                                let harmonic = features[point_idx].thornfiddle_path_harmonic;
                                if bfs_used {
                                    format!("Straight: {:.1} (approx)\nGeodesic: {:.1} [BFS]\nHarmonic: {:.3}",
                                        straight_len, curved_len, harmonic)
                                } else {
                                    format!("Straight: {:.1}\nGeodesic: {:.1}\nHarmonic: {:.3}",
                                        straight_len, curved_len, harmonic)
                                }
                            }
                        };

                        painter.text(
                            midpoint,
                            egui::Align2::CENTER_CENTER,
                            label_text,
                            egui::FontId::proportional(12.0 * zoom),
                            egui::Color32::WHITE,
                        );
                    }
                }
            } else {
                if let Some(point_idx) = state_guard.selected_point {
                    let contour = match state_guard.selected_point_type {
                        PointType::EC => &result.ec_contour,
                        PointType::MC => &result.mc_contour,
                    };

                    if point_idx < contour.len() {
                        let (pixel_x, pixel_y) = contour[point_idx];
                        let screen_pos = pixel_to_screen(pixel_x, pixel_y);
                        painter.circle_filled(screen_pos, 5.0 * zoom, RED_COLOR);
                        painter.circle_stroke(
                            screen_pos,
                            5.0 * zoom,
                            egui::Stroke::new(2.0 * zoom, egui::Color32::WHITE),
                        );
                    }
                }
            }
        }
    } else {
        let painter = ui.painter_at(rect);
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            if state_guard.current_image().is_some() {
                "Click 'Analyze' to start analysis"
            } else {
                "Open a workspace folder to begin"
            },
            egui::FontId::proportional(16.0),
            ui.visuals().text_color(),
        );
    }

    drop(state_guard);

    if response.dragged() {
        let mut state_guard = state.lock().unwrap();
        state_guard.pan_offset += response.drag_delta();
    }

    if response.hovered() {
        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0 {
            let mut state_guard = state.lock().unwrap();
            let zoom_delta = scroll * 0.001;
            state_guard.zoom_level = (state_guard.zoom_level + zoom_delta).clamp(0.1, 5.0);
        }
    }
}

pub fn render_ec_graph(ui: &mut egui::Ui, state: &Arc<Mutex<AppState>>) {
    ui.heading("📊 Edge Complexity (EC)");

    let (data, selected_point, is_ec_selected) = {
        let state_guard = state.lock().unwrap();
        if let Some(result) = state_guard.current_result() {
            (
                result.ec_data.clone(),
                state_guard.selected_point,
                state_guard.selected_point_type == PointType::EC,
            )
        } else {
            ui.label("No analysis data available");
            return;
        }
    };

    ui.label(format!("Total points: {}", data.len()));

    let points: PlotPoints = data.iter().map(|&(x, y)| [x, y]).collect();

    let line = Line::new(points)
        .color(PINK_COLOR)
        .width(2.0)
        .name("Pink_Pixels");

    let plot = Plot::new("ec_plot")
        .height(200.0)
        .legend(egui_plot::Legend::default())
        .x_axis_label("Point Index")
        .y_axis_label("Pink Pixels Crossed");

    let response = plot.show(ui, |plot_ui| {
        plot_ui.line(line);

        if is_ec_selected {
            if let Some(idx) = selected_point {
                if let Some(&(x, y)) = data.get(idx) {
                    let highlight = Points::new(vec![[x, y]])
                        .color(RED_COLOR)
                        .radius(6.0)
                        .name("Selected");
                    plot_ui.points(highlight);
                }
            }
        }
    });

    if response.response.clicked() {
        if let Some(pos) = response.response.interact_pointer_pos() {
            let plot_pos = response.transform.value_from_position(pos);
            let idx = plot_pos.x.round() as usize;
            if idx < data.len() {
                let mut state_guard = state.lock().unwrap();
                state_guard.selected_point = Some(idx);
                state_guard.selected_point_type = PointType::EC;
            }
        }
    }
}

pub fn render_mc_graph(ui: &mut egui::Ui, state: &Arc<Mutex<AppState>>) {
    ui.heading("📊 Macro-shape Complexity (MC)");

    let (data, selected_point, is_mc_selected) = {
        let state_guard = state.lock().unwrap();
        if let Some(result) = state_guard.current_result() {
            (
                result.mc_data.clone(),
                state_guard.selected_point,
                state_guard.selected_point_type == PointType::MC,
            )
        } else {
            ui.label("No analysis data available");
            return;
        }
    };

    ui.label(format!("Total points: {}", data.len()));

    let points: PlotPoints = data.iter().map(|&(x, y)| [x, y]).collect();

    let line = Line::new(points)
        .color(YELLOW_COLOR)
        .width(2.0)
        .name("Harmonic_MC");

    let plot = Plot::new("mc_plot")
        .height(200.0)
        .legend(egui_plot::Legend::default())
        .x_axis_label("Point Index")
        .y_axis_label("Thornfiddle Harmonic");

    let response = plot.show(ui, |plot_ui| {
        plot_ui.line(line);

        if is_mc_selected {
            if let Some(idx) = selected_point {
                if let Some(&(x, y)) = data.get(idx) {
                    let highlight = Points::new(vec![[x, y]])
                        .color(RED_COLOR)
                        .radius(6.0)
                        .name("Selected");
                    plot_ui.points(highlight);
                }
            }
        }
    });

    if response.response.clicked() {
        if let Some(pos) = response.response.interact_pointer_pos() {
            let plot_pos = response.transform.value_from_position(pos);
            let idx = plot_pos.x.round() as usize;
            if idx < data.len() {
                let mut state_guard = state.lock().unwrap();
                state_guard.selected_point = Some(idx);
                state_guard.selected_point_type = PointType::MC;
            }
        }
    }
}

pub fn render_summary_panel(ui: &mut egui::Ui, state: &Arc<Mutex<AppState>>) {
    ui.heading("📋 Summary Statistics");

    let state_guard = state.lock().unwrap();
    if let Some(result) = state_guard.current_result() {
        let summary = &result.summary;

        egui::Grid::new("summary_grid")
            .num_columns(3)
            .spacing([20.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label("Metric");
                ui.label("EC");
                ui.label("MC");
                ui.end_row();

                ui.separator();
                ui.separator();
                ui.separator();
                ui.end_row();

                ui.label("Length:");
                ui.label(format!("{:.1} px", summary.ec_length));
                ui.label(format!("{:.1} px", summary.mc_length));
                ui.end_row();

                ui.label("Width:");
                ui.label(format!("{:.1} px", summary.ec_width));
                ui.label(format!("{:.1} px", summary.mc_width));
                ui.end_row();

                ui.label("Shape Index:");
                ui.label(format!("{:.3}", summary.ec_shape_index));
                ui.label(format!("{:.3}", summary.mc_shape_index));
                ui.end_row();

                ui.label("Circularity:");
                ui.label(format!("{:.3}", summary.ec_circularity));
                ui.label(format!("{:.3}", summary.mc_circularity));
                ui.end_row();

                ui.label("Entropy:");
                ui.label(format!("{:.4}", summary.ec_spectral_entropy));
                ui.label(format!("{:.4}", summary.mc_spectral_entropy));
                ui.end_row();

                ui.label("Area:");
                ui.label(format!("{} px²", summary.ec_area));
                ui.label(format!("{} px²", summary.mc_area));
                ui.end_row();

                ui.label("Outline Count:");
                ui.label(format!("{}", summary.ec_outline_count));
                ui.label(format!("{}", summary.mc_outline_count));
                ui.end_row();
            });
    } else {
        ui.label("No analysis data available");
    }
}

pub fn render_analysis_panel(ui: &mut egui::Ui, state: &Arc<Mutex<AppState>>, _ctx: &egui::Context) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        render_ec_graph(ui, state);
        ui.add_space(10.0);
        ui.separator();
        ui.add_space(10.0);

        render_mc_graph(ui, state);
        ui.add_space(10.0);
        ui.separator();
        ui.add_space(10.0);

        render_summary_panel(ui, state);
    });
}

pub fn render_thumbnail_strip(ui: &mut egui::Ui, state: &Arc<Mutex<AppState>>) {
    let mut selected_idx: Option<usize> = None;
    let mut selection_changes: Vec<(usize, bool)> = Vec::new();

    ui.horizontal(|ui| {
        let state_guard = state.lock().unwrap();
        ui.heading("📂 Workspace Images");
        ui.label(format!(
            "({} images, {} selected)",
            state_guard.images.len(),
            state_guard.selected_count()
        ));
        ui.separator();

        let select_all_clicked = ui.button("✓ Select All").clicked();
        let deselect_all_clicked = ui.button("✗ Deselect All").clicked();
        drop(state_guard);

        if select_all_clicked {
            state.lock().unwrap().select_all();
        }
        if deselect_all_clicked {
            state.lock().unwrap().deselect_all();
        }
    });

    ui.separator();

    egui::ScrollArea::horizontal()
        .auto_shrink([false, true])
        .show(ui, |ui| {
            let state_guard = state.lock().unwrap();

            ui.horizontal(|ui| {
                let current_idx = state_guard.current_image_index;

                for (idx, img_info) in state_guard.images.iter().enumerate() {
                    let is_current = current_idx == Some(idx);

                    let frame = if is_current {
                        egui::Frame::default()
                            .fill(ui.visuals().selection.bg_fill)
                            .stroke(ui.visuals().selection.stroke)
                            .rounding(4.0)
                            .inner_margin(4.0)
                    } else {
                        egui::Frame::default()
                            .fill(ui.visuals().window_fill())
                            .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
                            .rounding(4.0)
                            .inner_margin(4.0)
                    };

                    frame.show(ui, |ui| {
                        ui.set_width(130.0);
                        ui.set_height(130.0);

                        ui.vertical_centered(|ui| {
                            let mut selected = img_info.selected;
                            if ui.checkbox(&mut selected, "").changed() {
                                selection_changes.push((idx, selected));
                            }

                            let (rect, response) = ui.allocate_exact_size(
                                egui::vec2(120.0, 90.0),
                                egui::Sense::click(),
                            );

                            if let Some(thumbnail) = &img_info.thumbnail {
                                ui.painter().image(
                                    thumbnail.id(),
                                    rect,
                                    egui::Rect::from_min_max(
                                        egui::pos2(0.0, 0.0),
                                        egui::pos2(1.0, 1.0),
                                    ),
                                    egui::Color32::WHITE,
                                );
                            } else {
                                ui.painter().rect_filled(rect, 0.0, egui::Color32::DARK_GRAY);
                                ui.painter().text(
                                    rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    "Loading...",
                                    egui::FontId::proportional(10.0),
                                    egui::Color32::LIGHT_GRAY,
                                );
                            }

                            if response.clicked() {
                                selected_idx = Some(idx);
                            }

                            ui.add_space(2.0);
                            let filename_label = if img_info.filename.len() > 14 {
                                format!("{}...", &img_info.filename[..11])
                            } else {
                                img_info.filename.clone()
                            };
                            ui.label(egui::RichText::new(filename_label).small());

                            // FIX 6: AnalysisStatus::Pending does not exist — use Running
                            let status_text = match img_info.status {
                                AnalysisStatus::NotStarted => "",
                                AnalysisStatus::Running => "⏳",  // was: Pending
                                AnalysisStatus::Completed => "✓",
                                AnalysisStatus::Failed => "✗",
                            };
                            if !status_text.is_empty() {
                                ui.label(status_text);
                            }
                        });
                    });
                }
            });

            drop(state_guard);
        });

    // FIX 5: toggle_image_selection() does not exist on AppState.
    //         Apply the accumulated selection changes inline instead.
    if !selection_changes.is_empty() {
        let mut state_guard = state.lock().unwrap();
        for (idx, selected) in selection_changes {
            if let Some(img) = state_guard.images.get_mut(idx) {
                img.selected = selected;
            }
        }
    }

    if let Some(idx) = selected_idx {
        state.lock().unwrap().select_image(idx);
    }
}