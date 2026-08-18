//! Small shared UI helpers for a consistent, cohesive look across tabs.
//! Keep this dependency-light; it's the project's mini design system.

use egui::{Color32, Context, RichText, Ui};

use std::sync::atomic::{AtomicU32, Ordering};

/// Fallback accent (leaf green) — used before a theme has been applied.
pub const ACCENT_FALLBACK: Color32 = Color32::from_rgb(120, 200, 130);

// The active theme's accent and muted colours, published here so call sites that
// have no `Context` (most of them) can still follow the theme.
//
// Previously this was a hardcoded const, so all 13 themes shared one leaf
// green: the primary buttons and the wordmark stayed green in Synthwave and
// Dracula, and on the light themes that green sat at roughly 1.9:1 against its
// own near-black label. Stored as packed RGB in an atomic rather than threaded
// through every signature, because this is read from ~40 call sites in draw code
// and written exactly once per theme change.
static ACCENT_RGB: AtomicU32 = AtomicU32::new(0x0078_C88A);
static MUTED_RGB:  AtomicU32 = AtomicU32::new(0x0096_9696);

fn pack(c: Color32) -> u32 {
    ((c.r() as u32) << 16) | ((c.g() as u32) << 8) | c.b() as u32
}
fn unpack(v: u32) -> Color32 {
    Color32::from_rgb((v >> 16) as u8, (v >> 8) as u8, v as u8)
}

/// Called by `theme::apply`.
pub fn set_theme_colors(accent: Color32, muted: Color32) {
    ACCENT_RGB.store(pack(accent), Ordering::Relaxed);
    MUTED_RGB.store(pack(muted), Ordering::Relaxed);
}

/// Brand accent for the ACTIVE theme. Active states, highlights, primary actions.
#[allow(non_snake_case)]
pub fn ACCENT() -> Color32 { unpack(ACCENT_RGB.load(Ordering::Relaxed)) }
/// Muted text for captions / secondary labels, for the ACTIVE theme.
#[allow(non_snake_case)]
pub fn MUTED() -> Color32 { unpack(MUTED_RGB.load(Ordering::Relaxed)) }
/// Standard inter-element gap.
pub const GAP: f32 = 8.0;
/// Standard width for a tab's left control panel (keeps tabs aligned).
pub const CONTROL_W: f32 = 300.0;

/// Readable foreground for text drawn ON the accent fill.
///
/// Was hardcoded near-black, which only worked because the accent was a fixed
/// pale green. Now that the accent follows the theme, a dark accent (several of
/// the 13 have one) would put near-black text on a near-black button. Rec. 601
/// luma, thresholded — crude, but exactly right for a two-way pick.
pub fn on_accent() -> Color32 {
    let c = ACCENT();
    let luma = 0.299 * c.r() as f32 + 0.587 * c.g() as f32 + 0.114 * c.b() as f32;
    if luma > 140.0 { Color32::from_gray(15) } else { Color32::from_gray(245) }
}

/// A full-width, accent-filled primary action button.
pub fn primary_button(ui: &mut Ui, text: &str) -> egui::Response {
    ui.add(
        egui::Button::new(RichText::new(text).strong().color(on_accent()))
            .fill(ACCENT())
            .min_size(egui::vec2(ui.available_width(), 32.0)),
    )
}

/// An uppercase, muted section header with a little breathing room above it.
pub fn section_header(ui: &mut Ui, text: &str) {
    ui.add_space(GAP);
    ui.label(RichText::new(text.to_uppercase()).small().strong().color(MUTED()));
    ui.add_space(2.0);
}

/// A muted caption line.
pub fn caption(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text).small().color(MUTED()));
}

/// Inline "busy" indicator: spinner + muted label (for in-flight operations).
pub fn busy(ui: &mut Ui, text: &str) {
    ui.add(egui::Spinner::new().size(14.0));
    ui.label(RichText::new(text).small().color(MUTED()));
}

/// Named text roles beyond egui's five. Use these instead of ad-hoc `.size(n)`
/// calls so the scale stays in one place.
pub mod text {
    /// Screen titles and empty-state headlines.
    pub const DISPLAY: &str = "Display";
    /// Sub-heading between Heading and Body.
    pub const SUBHEAD: &str = "Subhead";
    /// Measurements, counts, IDs. Monospace, so digits are tabular and columns
    /// line up — egui 0.29 has no OpenType feature API, and a monospace family is
    /// the reliable way to get tabular figures.
    pub const NUMERIC: &str = "Numeric";
}

pub fn display() -> egui::TextStyle { egui::TextStyle::Name(text::DISPLAY.into()) }
pub fn subhead() -> egui::TextStyle { egui::TextStyle::Name(text::SUBHEAD.into()) }
pub fn numeric() -> egui::TextStyle { egui::TextStyle::Name(text::NUMERIC.into()) }

/// Apply the project's cohesive typography, spacing and rounding on top of the
/// active theme's colours. Call after `theme::apply` so theme colours survive.
pub fn apply_style(ctx: &Context) {
    let mut style = (*ctx.style()).clone();

    // ── type scale ──────────────────────────────────────────────────────────
    // egui's defaults are Body 12.5, Button 12.5, Heading 18, Small 9 — body and
    // button identical, and nothing between body and heading. With no steps to
    // work with, every panel renders as one flat wall of same-sized text, which
    // is why density reads as clutter rather than as information. This is the
    // single change that most affects how finished the app looks.
    //
    // Small was 9px, which is genuinely unreadable on a 4K panel at 100% and is
    // the size `caption`/`section_header` use everywhere.
    use egui::{FontFamily::{Monospace, Proportional}, FontId, TextStyle};
    style.text_styles = [
        (display(),               FontId::new(26.0, Proportional)),
        (TextStyle::Heading,      FontId::new(18.5, Proportional)),
        (subhead(),               FontId::new(14.5, Proportional)),
        (TextStyle::Body,         FontId::new(13.5, Proportional)),
        (TextStyle::Button,       FontId::new(13.5, Proportional)),
        (TextStyle::Small,        FontId::new(11.0, Proportional)),
        (TextStyle::Monospace,    FontId::new(12.5, Monospace)),
        (numeric(),               FontId::new(13.0, Monospace)),
    ]
    .into();

    // ── spacing ─────────────────────────────────────────────────────────────
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(10.0, 5.0);

    // ── shape ───────────────────────────────────────────────────────────────
    // `open` is included deliberately: theme.rs sets all five states to 4.0 and
    // this used to overwrite only four, leaving an open ComboBox rounded 1px
    // differently from everything else.
    let r = egui::Rounding::same(5.0);
    for w in [
        &mut style.visuals.widgets.noninteractive,
        &mut style.visuals.widgets.inactive,
        &mut style.visuals.widgets.hovered,
        &mut style.visuals.widgets.active,
        &mut style.visuals.widgets.open,
    ] {
        w.rounding = r;
    }
    style.visuals.window_rounding = egui::Rounding::same(8.0);

    // ── flat surface ────────────────────────────────────────────────────────
    // egui's defaults draw a bevel-ish outline on every widget state, a visible
    // frame around every panel, and give hovered widgets a size "expansion" that
    // makes them jump. Together those read as generic toolkit chrome no amount
    // of layout work can overcome — the app can be perfectly organised and still
    // look unfinished. Removing them is what makes a flat, designed surface
    // possible; the interaction cues move to FILL changes instead, which is how
    // modern flat UIs signal state.
    // Outlines KEPT. Removing them globally made checkboxes and combo boxes stop
    // reading as controls at all — the outline is what says "this is operable",
    // and a flat surface is not worth losing that. Only the hover GROWTH goes:
    // widgets that change size under the pointer nudge their neighbours, which is
    // the part that actually felt unstable.

    // No size change on hover — widgets that grow under the pointer are the
    // single most "toolkit" behaviour egui has, and they nudge neighbours.
    style.visuals.widgets.hovered.expansion = 0.0;
    style.visuals.widgets.active.expansion  = 0.0;

    // Panels and windows sit flush; separators do the dividing.
    style.visuals.window_stroke = egui::Stroke::NONE;
    style.visuals.window_rounding = egui::Rounding::same(8.0);

    // Roomier hit targets and a calmer rhythm. The default 4px item spacing is
    // what made dense panels read as cramped rather than dense.
    style.spacing.item_spacing = egui::vec2(8.0, 7.0);
    style.spacing.button_padding = egui::vec2(11.0, 6.0);
    style.spacing.menu_margin = egui::Margin::same(8.0);
    style.spacing.indent = 18.0;
    style.spacing.scroll = egui::style::ScrollStyle::solid();

    ctx.set_style(style);
}
