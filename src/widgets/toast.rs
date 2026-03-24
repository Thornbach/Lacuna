use egui::{Color32, Context, pos2, vec2, Align2, FontId, Rounding, Stroke};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq)]
pub enum ToastKind {
    Info,
    Success,
    Warning,
    Error,
}

struct Toast {
    message: String,
    kind: ToastKind,
    born: Instant,
    lifetime: Duration,
}

impl Toast {
    fn color(&self) -> Color32 {
        match self.kind {
            ToastKind::Info    => Color32::from_rgb(70, 130, 200),
            ToastKind::Success => Color32::from_rgb(70, 170, 100),
            ToastKind::Warning => Color32::from_rgb(210, 150, 50),
            ToastKind::Error   => Color32::from_rgb(200, 70, 70),
        }
    }

    fn icon(&self) -> &'static str {
        match self.kind {
            ToastKind::Info    => "ℹ",
            ToastKind::Success => "✓",
            ToastKind::Warning => "⚠",
            ToastKind::Error   => "✗",
        }
    }

    fn alpha(&self) -> f32 {
        let age = self.born.elapsed().as_secs_f32();
        let total = self.lifetime.as_secs_f32();
        // fade in first 0.15 s, fade out last 0.4 s
        let fade_in  = (age / 0.15).min(1.0);
        let fade_out = ((total - age) / 0.4).clamp(0.0, 1.0);
        fade_in * fade_out
    }
}

pub struct ToastManager {
    toasts: Vec<Toast>,
}

impl ToastManager {
    pub fn new() -> Self {
        Self { toasts: Vec::new() }
    }

    pub fn push(&mut self, message: impl Into<String>, kind: ToastKind) {
        self.toasts.push(Toast {
            message: message.into(),
            kind,
            born: Instant::now(),
            lifetime: Duration::from_secs(4),
        });
    }

    pub fn success(&mut self, msg: impl Into<String>) { self.push(msg, ToastKind::Success); }
    pub fn info   (&mut self, msg: impl Into<String>) { self.push(msg, ToastKind::Info); }
    pub fn warning(&mut self, msg: impl Into<String>) { self.push(msg, ToastKind::Warning); }
    pub fn error  (&mut self, msg: impl Into<String>) { self.push(msg, ToastKind::Error); }

    pub fn show(&mut self, ctx: &Context) {
        let now = Instant::now();
        self.toasts.retain(|t| now.duration_since(t.born) < t.lifetime);

        if self.toasts.is_empty() { return; }

        let screen = ctx.screen_rect();
        let width  = 360.0_f32;
        let height = 48.0_f32;
        let pad    = 10.0_f32;
        let margin = 16.0_f32;

        for (i, toast) in self.toasts.iter().enumerate() {
            let alpha = toast.alpha();
            if alpha <= 0.0 { continue; }

            let x = screen.right() - width - margin;
            let y = screen.bottom() - margin - (i as f32) * (height + pad) - height;

            let rect = egui::Rect::from_min_size(pos2(x, y), vec2(width, height));

            let bg   = toast.color().linear_multiply(alpha);
            let text = Color32::from_white_alpha((255.0 * alpha) as u8);

            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("toast").with(i),
            ));

            painter.rect_filled(rect, Rounding::same(6.0), bg);
            painter.rect_stroke(rect, Rounding::same(6.0), Stroke::new(1.0, text));

            let icon_pos = pos2(rect.left() + 14.0, rect.center().y);
            painter.text(icon_pos, Align2::LEFT_CENTER, toast.icon(),
                FontId::proportional(18.0), text);

            let text_pos = pos2(rect.left() + 34.0, rect.center().y);
            painter.text(text_pos, Align2::LEFT_CENTER, &toast.message,
                FontId::proportional(13.0), text);
        }

        // Keep repainting while any toast is visible
        if !self.toasts.is_empty() {
            ctx.request_repaint();
        }
    }
}
