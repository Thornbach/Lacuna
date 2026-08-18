//! Transient notifications, plus a persistent record of the ones that mattered.
//!
//! The previous version painted every toast as a fixed 360x48 rectangle with
//! `Painter::text`, which does no wrapping, and dropped all of them after four
//! seconds. Three consequences, all real:
//!
//!   * A long or multi-line message — the pipeline's own panic report is ~200
//!     characters across several lines — overflowed the box and was unreadable.
//!   * Errors vanished after four seconds. Step away from a long run and the
//!     only report of a failure is gone, with no history anywhere.
//!   * Nothing was clickable (a raw layer painter produces no `Response`), so
//!     they could not be dismissed or copied.
//!
//! Now: toasts are real widgets, they size to their text, errors and warnings
//! stay until dismissed, and everything is kept in a capped history the app can
//! show later.

use egui::{Color32, Context, RichText};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq)]
pub enum ToastKind {
    Info,
    Success,
    Warning,
    Error,
}

impl ToastKind {
    fn color(self) -> Color32 {
        match self {
            ToastKind::Info    => Color32::from_rgb(70, 130, 200),
            ToastKind::Success => Color32::from_rgb(70, 170, 100),
            // Was (210,150,50) with white text — about 2.3:1, below any
            // readability bar. Darkened so white text on it actually passes.
            ToastKind::Warning => Color32::from_rgb(150, 100, 20),
            ToastKind::Error   => Color32::from_rgb(165, 45, 45),
        }
    }
    fn tag(self) -> &'static str {
        match self {
            ToastKind::Info    => "Info",
            ToastKind::Success => "Done",
            ToastKind::Warning => "Warning",
            ToastKind::Error   => "Error",
        }
    }
    /// Routine confirmations disappear on their own; anything the user may need
    /// to act on waits to be dismissed. Shneiderman's rule — modest feedback for
    /// frequent actions, substantial for consequential ones.
    fn sticky(self) -> bool {
        matches!(self, ToastKind::Warning | ToastKind::Error)
    }
}

#[derive(Clone)]
pub struct ToastRecord {
    pub message: String,
    pub kind:    ToastKind,
    pub at:      String,
}

struct Toast {
    message:  String,
    kind:     ToastKind,
    born:     Instant,
    lifetime: Duration,
    dismissed: bool,
}

impl Toast {
    fn expired(&self) -> bool {
        self.dismissed || (!self.kind.sticky() && self.born.elapsed() >= self.lifetime)
    }
    fn alpha(&self) -> f32 {
        if self.kind.sticky() {
            return (self.born.elapsed().as_secs_f32() / 0.15).min(1.0);
        }
        let age = self.born.elapsed().as_secs_f32();
        let total = self.lifetime.as_secs_f32();
        (age / 0.15).min(1.0) * ((total - age) / 0.4).clamp(0.0, 1.0)
    }
}

pub struct ToastManager {
    toasts:  Vec<Toast>,
    history: Vec<ToastRecord>,
}

impl ToastManager {
    pub fn new() -> Self {
        Self { toasts: Vec::new(), history: Vec::new() }
    }

    pub fn push(&mut self, message: impl Into<String>, kind: ToastKind) {
        let message = message.into();
        // Keep a record even for toasts that self-dismiss, so "what was that
        // message that just flashed?" has an answer.
        const MAX_HISTORY: usize = 200;
        self.history.push(ToastRecord {
            message: message.clone(),
            kind,
            at: short_time(),
        });
        if self.history.len() > MAX_HISTORY {
            self.history.remove(0);
        }
        // Cap what is on screen. Without this a per-item failure in a 10,000-item
        // batch stacks 10,000 sticky error cards up the side of the window.
        const MAX_VISIBLE: usize = 4;
        if self.toasts.iter().filter(|t| !t.expired()).count() >= MAX_VISIBLE {
            if let Some(pos) = self.toasts.iter().position(|t| !t.kind.sticky()) {
                self.toasts.remove(pos);
            } else {
                self.toasts.remove(0);
            }
        }
        self.toasts.push(Toast {
            message,
            kind,
            born: Instant::now(),
            lifetime: Duration::from_secs(4),
            dismissed: false,
        });
    }

    pub fn success(&mut self, msg: impl Into<String>) { self.push(msg, ToastKind::Success); }
    pub fn info   (&mut self, msg: impl Into<String>) { self.push(msg, ToastKind::Info); }
    pub fn warning(&mut self, msg: impl Into<String>) { self.push(msg, ToastKind::Warning); }
    pub fn error  (&mut self, msg: impl Into<String>) { self.push(msg, ToastKind::Error); }

    /// Everything shown this session, newest last. For a "recent messages" view.
    pub fn history(&self) -> &[ToastRecord] { &self.history }

    pub fn show(&mut self, ctx: &Context) {
        self.toasts.retain(|t| !t.expired());
        if self.toasts.is_empty() { return; }

        let mut dismiss: Option<usize> = None;
        let mut copy: Option<String> = None;
        let screen = ctx.screen_rect();
        let mut y = screen.bottom() - 16.0;

        // Bottom-up so the newest sits closest to the corner and older ones push
        // upward — the order they arrived is preserved visually.
        for (i, toast) in self.toasts.iter().enumerate().rev() {
            let alpha = toast.alpha();
            if alpha <= 0.0 { continue; }
            let col = toast.kind.color();

            let area = egui::Area::new(egui::Id::new("toast").with(i))
                .order(egui::Order::Foreground)
                .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-16.0, y - screen.bottom()))
                .show(ctx, |ui| {
                    ui.set_max_width(420.0);
                    egui::Frame::popup(ui.style())
                        .fill(col.linear_multiply(alpha))
                        .stroke(egui::Stroke::new(1.0, Color32::from_white_alpha((90.0 * alpha) as u8)))
                        .rounding(egui::Rounding::same(6.0))
                        .inner_margin(egui::Margin::symmetric(12.0, 9.0))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(RichText::new(toast.kind.tag())
                                    .small().strong()
                                    .color(Color32::from_white_alpha((220.0 * alpha) as u8)));
                                if toast.kind.sticky() {
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            if ui.small_button("Dismiss").clicked() {
                                                dismiss = Some(i);
                                            }
                                            if ui.small_button("Copy").clicked() {
                                                copy = Some(toast.message.clone());
                                            }
                                        },
                                    );
                                }
                            });
                            // A real Label, so it WRAPS — the whole point.
                            ui.label(RichText::new(&toast.message)
                                .color(Color32::from_white_alpha((255.0 * alpha) as u8)));
                        });
                })
                .response;

            y -= area.rect.height() + 8.0;
        }

        if let Some(i) = dismiss {
            if let Some(t) = self.toasts.get_mut(i) { t.dismissed = true; }
        }
        if let Some(text) = copy {
            // Errors are the thing most likely to be pasted into a bug report.
            ctx.copy_text(text);
        }

        // Only keep animating while something is actually fading; sticky toasts
        // are static and must not hold the UI at full framerate indefinitely.
        if self.toasts.iter().any(|t| !t.kind.sticky()) {
            ctx.request_repaint();
        }
    }
}

/// "14:37" — enough to correlate a message with what you were doing.
fn short_time() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let mins = (secs / 60) % 60;
    let hours = (secs / 3600) % 24;
    format!("{hours:02}:{mins:02}")
}
