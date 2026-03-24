//! Small shared UI helpers for a consistent, cohesive look across tabs.
//! Keep this dependency-light; it's the project's mini design system.

use egui::{Color32, Context, RichText, Ui};

/// Brand accent (leaf green). Used for active states, highlights, primary actions.
pub const ACCENT: Color32 = Color32::from_rgb(120, 200, 130);
/// Muted text for captions / secondary labels.
pub const MUTED: Color32 = Color32::from_gray(150);
/// Standard inter-element gap.
pub const GAP: f32 = 8.0;
/// Standard width for a tab's left control panel (keeps tabs aligned).
pub const CONTROL_W: f32 = 300.0;

/// A full-width, accent-filled primary action button.
pub fn primary_button(ui: &mut Ui, text: &str) -> egui::Response {
    ui.add(
        egui::Button::new(RichText::new(text).strong().color(Color32::from_gray(15)))
            .fill(ACCENT)
            .min_size(egui::vec2(ui.available_width(), 32.0)),
    )
}

/// An uppercase, muted section header with a little breathing room above it.
pub fn section_header(ui: &mut Ui, text: &str) {
    ui.add_space(GAP);
    ui.label(RichText::new(text.to_uppercase()).small().strong().color(MUTED));
    ui.add_space(2.0);
}

/// A muted caption line.
pub fn caption(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text).small().color(MUTED));
}

/// Inline "busy" indicator: spinner + muted label (for in-flight operations).
pub fn busy(ui: &mut Ui, text: &str) {
    ui.add(egui::Spinner::new().size(14.0));
    ui.label(RichText::new(text).small().color(MUTED));
}

/// Apply the project's cohesive spacing + rounding on top of the active theme's
/// colours. Call after `theme::apply` so theme colours are preserved.
pub fn apply_style(ctx: &Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(10.0, 5.0);
    let r = egui::Rounding::same(5.0);
    style.visuals.widgets.noninteractive.rounding = r;
    style.visuals.widgets.inactive.rounding = r;
    style.visuals.widgets.hovered.rounding = r;
    style.visuals.widgets.active.rounding = r;
    style.visuals.window_rounding = egui::Rounding::same(8.0);
    ctx.set_style(style);
}
