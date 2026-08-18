//! Tile-Picker research tab.
//!
//! Hand-pick 256² tiles out of a folder of (possibly large, possibly RGBA)
//! images. Move the cursor over the image, a square follows it centred on the
//! cursor, click to "stamp" out that tile — it is cropped from the full-res
//! buffer and saved to the output folder immediately. A magnifying loupe in the
//! top-right aids precise placement; already-stamped tiles are outlined.
//!
//! Memory: only the *current* image is decoded into RAM (one at a time); the
//! filmstrip uses small thumbnails generated lazily on a worker thread, one at a
//! time, and bounded to a window around the current image.

use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::mpsc,
};

use egui::{
    Align2, Color32, ColorImage, Context, FontId, Pos2, Rect, RichText, Sense, Stroke,
    TextureHandle, TextureOptions, Ui, Vec2,
};

use crate::settings::{AppSettings, TilePickerSession};
use crate::tabs::leaf_seg::inference::list_images;
use crate::ui_kit;
use crate::widgets::ToastManager;

const THUMB: usize = 88; // filmstrip thumbnail box (logical px)
const MAX_TEX: u32 = 4096; // cap the display texture; full-res buffer kept for cropping

/// Amber used for already-stamped tiles.
const STAMP_COL: Color32 = Color32::from_rgb(255, 180, 60);

struct LoadedImage {
    rgba: Vec<u8>, // w*h*4, full resolution (the crop source)
    w: u32,
    h: u32,
    tex: TextureHandle, // possibly downscaled for display only
}

#[derive(Clone)]
struct Stamp {
    x: i32, // tile top-left in source px; may be negative (overhangs the edge)
    y: i32,
    file: PathBuf,
}

pub struct TilePickerTab {
    source_folder: Option<PathBuf>,
    output_folder: Option<PathBuf>,
    tile: u32,
    zoom: f32,            // loupe magnification relative to the on-screen image
    loupe_follow: bool,   // glue the loupe beside the cursor instead of a corner
    loupe_anchor: [f32; 2], // normalised top-left of the fixed loupe (default top-right)
    loupe_center: Option<[f32; 2]>, // last in-image hover, so the fixed loupe persists

    view_zoom: f32,         // canvas zoom (mouse-wheel), independent of the loupe
    view_pan: Vec2,         // canvas pan offset (drag)

    paths: Vec<PathBuf>,
    cur: usize,
    done: HashSet<String>,    // filenames marked done (persisted per folder)
    scroll_to_current: bool,  // request the strip to re-center on the current leaf
    vis_lo: usize,            // visible filmstrip range (drives thumbnail loading)
    vis_hi: usize,

    img: Option<LoadedImage>,
    img_rx: Option<mpsc::Receiver<Result<(Vec<u8>, u32, u32), String>>>,
    loading_idx: Option<usize>,
    img_for: Option<usize>, // which path index `img` currently holds

    stamps: HashMap<usize, Vec<Stamp>>, // per image index (undo stack + overlay)

    thumbs: HashMap<usize, TextureHandle>,
    thumb_rx: Option<(usize, mpsc::Receiver<Option<ColorImage>>)>,
    thumb_failed: HashSet<usize>,

    loupe_tex: Option<TextureHandle>,

    source_rx: Option<mpsc::Receiver<Option<PathBuf>>>,
    output_rx: Option<mpsc::Receiver<Option<PathBuf>>>,

    status: String,
    total_stamped: usize,
}

impl TilePickerTab {
    pub fn new() -> Self {
        Self {
            source_folder: None,
            output_folder: None,
            tile: 256,
            zoom: 3.0,
            loupe_follow: false,
            loupe_anchor: [1.0, 0.0],
            loupe_center: None,
            view_zoom: 1.0,
            view_pan: Vec2::ZERO,
            paths: Vec::new(),
            cur: 0,
            done: HashSet::new(),
            scroll_to_current: false,
            vis_lo: 0,
            vis_hi: 0,
            img: None,
            img_rx: None,
            loading_idx: None,
            img_for: None,
            stamps: HashMap::new(),
            thumbs: HashMap::new(),
            thumb_rx: None,
            thumb_failed: HashSet::new(),
            loupe_tex: None,
            source_rx: None,
            output_rx: None,
            status: String::new(),
            total_stamped: 0,
        }
    }

    pub fn needs_repaint(&self) -> bool {
        self.img_rx.is_some()
            || self.thumb_rx.is_some()
            || self.scroll_to_current
            || self.thumbs_pending()
    }

    /// Any still-missing thumbnail inside the visible filmstrip window?
    fn thumbs_pending(&self) -> bool {
        if self.paths.is_empty() {
            return false;
        }
        let hi = self.vis_hi.min(self.paths.len() - 1);
        (self.vis_lo..=hi).any(|i| !self.thumbs.contains_key(&i) && !self.thumb_failed.contains(&i))
    }

    pub fn save_settings(&self, s: &mut AppSettings) {
        s.tile_picker.source_folder = self.source_folder.clone();
        s.tile_picker.output_folder = self.output_folder.clone();
        s.tile_picker.tile = self.tile;
        s.tile_picker.zoom = self.zoom;
        s.tile_picker.loupe_follow = self.loupe_follow;
        s.tile_picker.loupe_anchor = self.loupe_anchor;
    }

    pub fn load_settings(&mut self, s: &AppSettings) {
        let t = &s.tile_picker;
        self.output_folder = t.output_folder.clone();
        self.tile = t.tile.max(8);
        self.zoom = t.zoom.clamp(1.5, 8.0);
        self.loupe_follow = t.loupe_follow;
        self.loupe_anchor = [t.loupe_anchor[0].clamp(0.0, 1.0), t.loupe_anchor[1].clamp(0.0, 1.0)];
        if let Some(src) = t.source_folder.clone() {
            // set_source loads the folder AND resumes the per-folder session
            // (done markers + last position) — so we pick up where we left off.
            self.set_source(src);
        }
    }

    pub fn show(&mut self, ui: &mut Ui, ctx: &Context, toasts: &mut ToastManager) {
        self.poll_dialogs();
        self.poll_image(ctx);
        self.pump_thumbs(ctx);
        self.request_image();

        // keyboard: arrows navigate, Ctrl+Z undoes
        let (prev, next, undo) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
                i.modifiers.ctrl && i.key_pressed(egui::Key::Z),
            )
        });
        if prev {
            self.nav(self.cur.saturating_sub(1));
        }
        if next {
            let n = self.paths.len();
            if n > 0 && self.cur + 1 >= n {
                toasts.info(format!("Last image ({}/{}) — nothing after this.", n, n));
            } else {
                self.nav(self.cur + 1);
            }
        }
        if undo {
            self.undo();
        }

        egui::SidePanel::left("tilepicker_controls")
            .exact_width(ui_kit::CONTROL_W)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("tp_ctrl")
                    .show(ui, |ui| self.show_controls(ui, toasts));
            });
        egui::TopBottomPanel::bottom("tilepicker_strip")
            .resizable(false)
            .min_height(THUMB as f32 + 30.0)
            .show_inside(ui, |ui| self.show_strip(ui));
        egui::CentralPanel::default().show_inside(ui, |ui| self.show_canvas(ui, ctx));
    }

    // ── controls ──────────────────────────────────────────────────────────────

    fn show_controls(&mut self, ui: &mut Ui, toasts: &mut ToastManager) {
        ui_kit::section_header(ui, "Folders");
        if ui.button("Source folder…").clicked() && self.source_rx.is_none() {
            self.source_rx = Some(spawn_folder_dialog());
        }
        ui.label(RichText::new(path_str(&self.source_folder)).small().color(Color32::GRAY));
        if !self.paths.is_empty() {
            ui.label(RichText::new(format!("{} images", self.paths.len())).small());
        }
        if ui.button("Output folder…").clicked() && self.output_rx.is_none() {
            self.output_rx = Some(spawn_folder_dialog());
        }
        ui.label(RichText::new(path_str(&self.output_folder)).small().color(Color32::GRAY));

        ui_kit::section_header(ui, "View");
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("Zoom {:.0}%", self.view_zoom * 100.0)).small());
            if ui.small_button("Reset").clicked() {
                self.view_zoom = 1.0;
                self.view_pan = Vec2::ZERO;
            }
        });
        ui_kit::caption(ui, "Mouse wheel = zoom · drag = pan");

        ui_kit::section_header(ui, "Navigate");
        let n = self.paths.len();
        if n > 0 {
            // prominent current-image indicator (the small strip caption is easy to miss)
            ui.label(RichText::new(format!("Image {} / {}", self.cur + 1, n)).size(16.0).strong());
            ui.label(RichText::new(self.filename(self.cur)).small().color(Color32::GRAY));
        }
        ui.horizontal(|ui| {
            if ui.add_enabled(self.cur > 0, egui::Button::new("< Prev")).clicked() {
                self.nav(self.cur.saturating_sub(1));
            }
            if ui.add_enabled(n > 0 && self.cur + 1 < n, egui::Button::new("Next >")).clicked() {
                self.nav(self.cur + 1);
            }
        });
        let cur_stamps = self.stamps.get(&self.cur).map(|v| v.len()).unwrap_or(0);
        ui.add_enabled_ui(cur_stamps > 0, |ui| {
            if ui.button(format!("Undo last  ({cur_stamps} on this image)")).clicked() {
                self.undo();
            }
        });

        ui_kit::section_header(ui, "Progress");
        ui.label(RichText::new(format!("{} / {} leaves done", self.done.len(), n)).small());
        ui.add_enabled_ui(n > 0, |ui| {
            if ui_kit::primary_button(ui, "Mark done -> next").clicked() {
                self.mark_done_next(toasts);
            }
        });
        let mut cur_done = self.is_done(self.cur);
        if ui.checkbox(&mut cur_done, "This leaf is done").changed() {
            self.set_done(self.cur, cur_done);
        }

        ui_kit::section_header(ui, "Session");
        ui.label(RichText::new(format!("Tiles stamped: {}", self.total_stamped)).small());
        if !self.status.is_empty() {
            ui.label(RichText::new(&self.status).small().color(ui_kit::ACCENT()));
        }

        ui_kit::section_header(ui, "Help");
        ui_kit::caption(
            ui,
            "The green square is the next tile, centred on the cursor. Near a border it overhangs \
             and the outside is saved as transparent pixels, so edge features aren't lost. \
             Left-click stamps (saved instantly); right-click removes a stamp; Ctrl+Z undoes the \
             last. Mouse wheel zooms, drag pans. ←/→ change image. 'Mark done' remembers finished \
             leaves per folder, so you resume where you left off.",
        );
    }

    pub fn show_settings_panel(&mut self, ui: &mut Ui) {
        ui_kit::section_header(ui, "Tile");
        egui::Grid::new("tp_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
            ui.label("Tile size:");
            ui.add(egui::DragValue::new(&mut self.tile).range(32..=2048).speed(8));
            ui.end_row();
        });

        ui_kit::section_header(ui, "Loupe");
        egui::Grid::new("tp_loupe").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
            ui.label("Zoom:");
            ui.add(egui::Slider::new(&mut self.zoom, 1.5..=8.0).fixed_decimals(1).suffix("x"));
            ui.end_row();
            ui.label("Position:");
            egui::ComboBox::from_id_salt("tp_loupe_pos")
                .selected_text(if self.loupe_follow { "Follow cursor" } else { "Fixed (drag)" })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.loupe_follow, false, "Fixed (drag)");
                    ui.selectable_value(&mut self.loupe_follow, true, "Follow cursor");
                });
            ui.end_row();
        });
        ui_kit::caption(
            ui,
            if self.loupe_follow {
                "Loupe sticks beside the tile square, near the cursor."
            } else {
                "Drag the loupe to move it; it stays pinned there."
            },
        );
    }

    // ── filmstrip ─────────────────────────────────────────────────────────────

    fn show_strip(&mut self, ui: &mut Ui) {
        if self.paths.is_empty() {
            ui.add_space(6.0);
            ui_kit::caption(ui, "No images — pick a source folder.");
            return;
        }
        ui.add_space(2.0);
        let n = self.paths.len();
        ui.horizontal(|ui| {
            ui_kit::caption(
                ui,
                &format!("Image {} / {}  ·  {} done", self.cur + 1, n, self.done.len()),
            );
            if ui.small_button("Jump to current").clicked() {
                self.scroll_to_current = true;
            }
            if let Some(name) = self.paths[self.cur].file_name() {
                ui.label(RichText::new(name.to_string_lossy()).small().color(Color32::GRAY));
            }
        });

        // Virtualized horizontal strip: only the visible thumbnails are laid out,
        // so the folder can hold thousands of images cheaply.
        let item_w = THUMB as f32 + 6.0;
        let want_scroll = std::mem::take(&mut self.scroll_to_current);
        egui::ScrollArea::horizontal().id_salt("tp_strip").show_viewport(ui, |ui, vp| {
            ui.set_width(n as f32 * item_w);
            ui.set_height(THUMB as f32);
            let origin = ui.min_rect().min;
            let first = (vp.min.x / item_w).floor().max(0.0) as usize;
            let last = (((vp.max.x / item_w).ceil() as usize) + 1).min(n - 1);
            self.vis_lo = first;
            self.vis_hi = last;
            for idx in first..=last {
                let x = origin.x + idx as f32 * item_w;
                let irect = Rect::from_min_size(Pos2::new(x, origin.y), Vec2::splat(THUMB as f32));
                let resp = ui.interact(irect, ui.id().with(("tp_thumb", idx)), Sense::click());
                self.paint_thumb(ui, irect, idx);
                if resp.clicked() {
                    self.goto(idx);
                }
            }
            if want_scroll {
                let cx = origin.x + self.cur as f32 * item_w;
                let crect = Rect::from_min_size(Pos2::new(cx, origin.y), Vec2::splat(THUMB as f32));
                ui.scroll_to_rect(crect, Some(egui::Align::Center));
            }
        });
    }

    /// Paint one filmstrip thumbnail: image/placeholder + done badge + highlight.
    fn paint_thumb(&self, ui: &Ui, rect: Rect, idx: usize) {
        let painter = ui.painter_at(rect);
        let selected = idx == self.cur;
        let done = self.is_done(idx);
        let stamped = self.stamps.get(&idx).map(|v| !v.is_empty()).unwrap_or(false);

        let bg = if selected { Color32::from_gray(72) } else { Color32::from_gray(38) };
        painter.rect_filled(rect, 4.0, bg);

        if let Some(tex) = self.thumbs.get(&idx) {
            let ts = tex.size_vec2();
            let s = (THUMB as f32 / ts.x).min(THUMB as f32 / ts.y);
            let dsz = ts * s;
            let drect = Rect::from_center_size(rect.center(), dsz);
            egui::Image::new((tex.id(), dsz)).paint_at(ui, drect);
        } else {
            painter.text(rect.center(), Align2::CENTER_CENTER, "…", FontId::proportional(16.0), Color32::GRAY);
        }

        // done: dim the thumb and stamp a green check badge in the corner
        if done {
            painter.rect_filled(rect, 4.0, Color32::from_black_alpha(120));
            let r = 9.0;
            let c = Pos2::new(rect.max.x - r - 3.0, rect.min.y + r + 3.0);
            painter.circle_filled(c, r, Color32::from_rgb(60, 170, 90));
            painter.line_segment(
                [c + Vec2::new(-4.0, 0.0), c + Vec2::new(-1.0, 3.0)],
                Stroke::new(1.8, Color32::WHITE),
            );
            painter.line_segment(
                [c + Vec2::new(-1.0, 3.0), c + Vec2::new(4.0, -3.5)],
                Stroke::new(1.8, Color32::WHITE),
            );
        }

        if selected {
            // big, unmistakable highlight for the leaf you're on
            painter.rect_stroke(rect.expand(1.5), 6.0, Stroke::new(3.0, ui_kit::ACCENT()));
        } else if stamped {
            painter.rect_stroke(rect, 4.0, Stroke::new(1.5, STAMP_COL));
        }
    }

    // ── main canvas ─────────────────────────────────────────────────────────────

    fn show_canvas(&mut self, ui: &mut Ui, ctx: &Context) {
        let avail = ui.available_rect_before_wrap();
        let resp = ui.allocate_rect(avail, Sense::click_and_drag());
        let painter = ui.painter_at(avail);
        painter.rect_filled(avail, 0.0, Color32::from_gray(24));

        if self.img.is_none() {
            let msg = if self.loading_idx.is_some() {
                "Loading image…"
            } else if self.paths.is_empty() {
                "Pick a source folder to begin."
            } else {
                "No image."
            };
            painter.text(
                avail.center(),
                Align2::CENTER_CENTER,
                msg,
                FontId::proportional(15.0),
                Color32::GRAY,
            );
            return;
        }

        let img = self.img.as_ref().unwrap();
        let (iw, ih) = (img.w as f32, img.h as f32);
        let fit = (avail.width() / iw).min(avail.height() / ih);
        let scale = fit * self.view_zoom; // wheel-zoom on top of the fit scale
        let disp = Vec2::new(iw * scale, ih * scale);
        let img_rect = Rect::from_center_size(avail.center() + self.view_pan, disp);

        draw_checker(&painter, img_rect, 14.0);
        egui::Image::new((img.tex.id(), disp)).paint_at(ui, img_rect);

        let t = self.tile as f32;
        let mut hover_src: Option<[f32; 2]> = None;
        let mut hover_screen: Option<Pos2> = None;
        let mut sq_tl: Option<[f32; 2]> = None; // tile top-left, source px (may be negative)
        if let Some(p) = resp.hover_pos() {
            if img_rect.contains(p) {
                hover_screen = Some(p);
                let sx = (p.x - img_rect.min.x) / scale;
                let sy = (p.y - img_rect.min.y) / scale;
                hover_src = Some([sx, sy]);
                // centre on the cursor WITHOUT clamping — the tile may overhang the
                // image edge; the outside is saved as transparent pixels.
                sq_tl = Some([sx - t / 2.0, sy - t / 2.0]);
            }
        }

        // already-stamped tiles on this image
        if let Some(list) = self.stamps.get(&self.cur) {
            for s in list {
                let r = Rect::from_min_size(
                    img_rect.min + Vec2::new(s.x as f32 * scale, s.y as f32 * scale),
                    Vec2::new(t * scale, t * scale),
                );
                painter.rect_filled(r, 0.0, Color32::from_rgba_unmultiplied(255, 180, 60, 28));
                painter.rect_stroke(r, 0.0, Stroke::new(1.5, STAMP_COL));
            }
        }
        // live placement square
        if let Some([tlx, tly]) = sq_tl {
            let r = Rect::from_min_size(
                img_rect.min + Vec2::new(tlx * scale, tly * scale),
                Vec2::new(t * scale, t * scale),
            );
            painter.rect_stroke(r, 0.0, Stroke::new(2.0, ui_kit::ACCENT()));
        }

        // ── loupe ──
        // The loupe always frames the tile square plus a small margin, so the
        // green box's edges stay visible no matter the image resolution. Its raw
        // on-screen size is *adaptive* (region × scale × zoom); the magnification
        // is the zoom level relative to the main view. (Previously the size was
        // fixed and the source region depended on resolution, so the square could
        // overflow the loupe on high-res images.)
        const LOUPE_MARGIN: f32 = 0.10; // show ~10% beyond the green box on each side
        let r_src = t * (1.0 + 2.0 * LOUPE_MARGIN); // source px spanned by the loupe
        // centre on the tile (whole square visible), falling back to the cursor
        let region_center = sq_tl.map(|[x, y]| [x + t / 2.0, y + t / 2.0]).or(hover_src);
        if region_center.is_some() {
            self.loupe_center = region_center;
        }
        let mut suppress_stamp = false;
        let draw_center = if self.loupe_follow { region_center } else { self.loupe_center };
        if let Some(center) = draw_center {
            let max_sz = (avail.width().min(avail.height()) - 24.0).clamp(160.0, 600.0);
            let loupe_sz = (r_src * scale * self.zoom).clamp(150.0, max_sz);
            let factor = loupe_sz / r_src; // source px → loupe px (region == r_src)
            let s = loupe_sz.round() as usize;
            let ci = build_loupe(img, center, factor, s);
            let h = self.loupe_tex.get_or_insert_with(|| {
                ctx.load_texture(
                    "tilepick_loupe",
                    ColorImage::new([1, 1], Color32::TRANSPARENT),
                    TextureOptions::NEAREST,
                )
            });
            h.set(ci, TextureOptions::NEAREST);
            let lid = h.id();

            let lrect = if self.loupe_follow {
                // glue the loupe beside the placement square, near the cursor
                let p = hover_screen.unwrap_or(avail.center());
                let sq_half = t * scale / 2.0;
                let gap = 14.0;
                let mut lx = p.x + sq_half + gap; // prefer the right of the square
                if lx + loupe_sz > avail.max.x {
                    lx = p.x - sq_half - gap - loupe_sz; // flip left near the edge
                }
                let ly = p.y - loupe_sz / 2.0;
                let lx = lx.clamp(avail.min.x, (avail.max.x - loupe_sz).max(avail.min.x));
                let ly = ly.clamp(avail.min.y, (avail.max.y - loupe_sz).max(avail.min.y));
                Rect::from_min_size(Pos2::new(lx, ly), Vec2::splat(loupe_sz))
            } else {
                let free = (avail.size() - Vec2::splat(loupe_sz)).max(Vec2::ZERO);
                let min = avail.min
                    + Vec2::new(self.loupe_anchor[0] * free.x, self.loupe_anchor[1] * free.y);
                Rect::from_min_size(min, Vec2::splat(loupe_sz))
            };

            // fixed mode: let the user drag the loupe (and don't stamp through it)
            if !self.loupe_follow {
                let drag = ui.interact(lrect, ui.id().with("tp_loupe_drag"), Sense::drag());
                if drag.hovered() || drag.dragged() {
                    suppress_stamp = true;
                    ctx.set_cursor_icon(if drag.dragged() {
                        egui::CursorIcon::Grabbing
                    } else {
                        egui::CursorIcon::Grab
                    });
                }
                if drag.dragged() {
                    let d = drag.drag_delta();
                    let free = (avail.size() - Vec2::splat(loupe_sz)).max(Vec2::splat(1.0));
                    self.loupe_anchor[0] = (self.loupe_anchor[0] + d.x / free.x).clamp(0.0, 1.0);
                    self.loupe_anchor[1] = (self.loupe_anchor[1] + d.y / free.y).clamp(0.0, 1.0);
                }
            }

            // dark mat behind the loupe so it separates from busy backgrounds
            painter.rect_filled(lrect.expand(4.0), egui::Rounding::same(6.0), Color32::from_black_alpha(160));
            let lp = ui.painter_at(lrect);
            lp.image(
                lid,
                lrect,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
            let to_loupe = |sx: f32, sy: f32| {
                Pos2::new(
                    lrect.center().x + (sx - center[0]) * factor,
                    lrect.center().y + (sy - center[1]) * factor,
                )
            };
            if let Some(list) = self.stamps.get(&self.cur) {
                for st in list {
                    let a = to_loupe(st.x as f32, st.y as f32);
                    let b = to_loupe(st.x as f32 + t, st.y as f32 + t);
                    lp.rect_stroke(Rect::from_two_pos(a, b), 0.0, Stroke::new(1.0, STAMP_COL));
                }
            }
            if let Some([tlx, tly]) = sq_tl {
                let a = to_loupe(tlx, tly);
                let b = to_loupe(tlx + t, tly + t);
                lp.rect_stroke(Rect::from_two_pos(a, b), 0.0, Stroke::new(1.5, ui_kit::ACCENT()));
            }
            let c = lrect.center();
            lp.line_segment(
                [Pos2::new(c.x - 7.0, c.y), Pos2::new(c.x + 7.0, c.y)],
                Stroke::new(1.0, Color32::WHITE),
            );
            lp.line_segment(
                [Pos2::new(c.x, c.y - 7.0), Pos2::new(c.x, c.y + 7.0)],
                Stroke::new(1.0, Color32::WHITE),
            );
            // bright frame on the unclipped painter so the full stroke shows
            painter.rect_stroke(
                lrect.expand(1.5),
                egui::Rounding::same(3.0),
                Stroke::new(2.0, Color32::from_gray(235)),
            );
            lp.text(
                lrect.left_top() + Vec2::new(5.0, 3.0),
                Align2::LEFT_TOP,
                format!("{:.1}x", self.zoom),
                FontId::proportional(11.0),
                Color32::WHITE,
            );
        }

        // ── input (after the immutable `img` borrow is done being used) ──
        // left-click = stamp; right-click = remove the stamp under the cursor;
        // left-drag = pan; wheel = zoom around the cursor.
        let stamp_at = if resp.clicked() && !suppress_stamp {
            sq_tl.map(|[x, y]| (x.round() as i32, y.round() as i32))
        } else {
            None
        };
        let remove_at = if resp.secondary_clicked() { hover_src } else { None };

        // pan
        if resp.dragged() {
            self.view_pan += resp.drag_delta();
        }
        // zoom around the cursor (keep the hovered source point fixed)
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if resp.hovered() && scroll.abs() > 0.1 {
            let factor = (scroll * 0.0015).exp();
            let new_zoom = (self.view_zoom * factor).clamp(0.25, 25.0);
            if let Some(p) = resp.hover_pos() {
                let q = Vec2::new(
                    (p.x - img_rect.min.x) / scale - iw / 2.0,
                    (p.y - img_rect.min.y) / scale - ih / 2.0,
                );
                let scale_new = fit * new_zoom;
                self.view_pan += q * (scale - scale_new);
            }
            self.view_zoom = new_zoom;
        }

        if let Some((x, y)) = stamp_at {
            self.stamp(x, y);
        }
        if let Some([sx, sy]) = remove_at {
            self.remove_stamp_at(sx, sy);
        }
    }

    /// Remove the topmost already-stamped tile containing source point (sx, sy).
    fn remove_stamp_at(&mut self, sx: f32, sy: f32) {
        let t = self.tile as f32;
        if let Some(list) = self.stamps.get_mut(&self.cur) {
            if let Some(pos) = list.iter().rposition(|s| {
                sx >= s.x as f32 && sx < s.x as f32 + t && sy >= s.y as f32 && sy < s.y as f32 + t
            }) {
                let s = list.remove(pos);
                let _ = std::fs::remove_file(&s.file);
                self.total_stamped = self.total_stamped.saturating_sub(1);
                let name = s.file.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
                self.status = format!("Removed {name}");
            }
        }
    }

    // ── actions ─────────────────────────────────────────────────────────────────

    fn stamp(&mut self, x: i32, y: i32) {
        let Some(img) = self.img.as_ref() else { return };
        let Some(out) = self.output_folder.clone() else {
            self.status = "Set an output folder first.".into();
            return;
        };
        let tu = self.tile;
        let t = tu as i32;
        let (iw, ih) = (img.w as i32, img.h as i32);
        // transparent-padded crop: pixels outside the image stay (0,0,0,0), so a
        // tile near the border captures the real edge features instead of snapping.
        let mut buf = vec![0u8; (tu * tu * 4) as usize];
        for row in 0..t {
            let sy = y + row;
            if sy < 0 || sy >= ih {
                continue;
            }
            for col in 0..t {
                let sx = x + col;
                if sx < 0 || sx >= iw {
                    continue;
                }
                let si = ((sy * iw + sx) * 4) as usize;
                let di = ((row * t + col) * 4) as usize;
                buf[di..di + 4].copy_from_slice(&img.rgba[si..si + 4]);
            }
        }
        let stem = self.paths[self.cur]
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "tile".into());
        let _ = std::fs::create_dir_all(&out);
        let file = out.join(format!("{stem}_x{x}_y{y}.png"));
        let res = image::RgbaImage::from_raw(tu, tu, buf)
            .ok_or_else(|| "bad crop".to_string())
            .and_then(|im| im.save(&file).map_err(|e| e.to_string()));
        match res {
            Ok(()) => {
                let name = file.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
                self.stamps.entry(self.cur).or_default().push(Stamp { x, y, file });
                self.total_stamped += 1;
                self.status = format!("Stamped {name}");
            }
            Err(e) => self.status = format!("Save failed: {e}"),
        }
    }

    fn undo(&mut self) {
        if let Some(list) = self.stamps.get_mut(&self.cur) {
            if let Some(s) = list.pop() {
                let _ = std::fs::remove_file(&s.file);
                self.total_stamped = self.total_stamped.saturating_sub(1);
                let name = s.file.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
                self.status = format!("Removed {name}");
            }
        }
    }

    /// Select an image (used by direct picks, e.g. filmstrip clicks).
    fn goto(&mut self, idx: usize) {
        if self.paths.is_empty() {
            return;
        }
        let idx = idx.min(self.paths.len() - 1);
        if idx != self.cur {
            self.cur = idx;
            self.loupe_tex = None;
            self.view_zoom = 1.0;
            self.view_pan = Vec2::ZERO;
            self.save_session();
        }
    }

    /// Navigate (Prev/Next/keyboard): like `goto` but also re-centers the strip.
    fn nav(&mut self, idx: usize) {
        self.goto(idx);
        self.scroll_to_current = true;
    }

    fn filename(&self, idx: usize) -> String {
        self.paths
            .get(idx)
            .and_then(|p| p.file_name())
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default()
    }

    fn is_done(&self, idx: usize) -> bool {
        let name = self.filename(idx);
        !name.is_empty() && self.done.contains(&name)
    }

    fn set_done(&mut self, idx: usize, done: bool) {
        let name = self.filename(idx);
        if name.is_empty() {
            return;
        }
        if done {
            self.done.insert(name);
        } else {
            self.done.remove(&name);
        }
        self.save_session();
    }

    /// First not-done image after the current one (wraps); None if all done.
    fn next_undone(&self) -> Option<usize> {
        let n = self.paths.len();
        if n == 0 {
            return None;
        }
        (1..=n).map(|off| (self.cur + off) % n).find(|&idx| !self.is_done(idx))
    }

    fn mark_done_next(&mut self, toasts: &mut ToastManager) {
        let from = self.cur;
        self.set_done(self.cur, true);
        match self.next_undone() {
            Some(idx) => {
                if idx <= from {
                    // next_undone wrapped past the end back to an earlier image
                    let remaining = self.paths.len().saturating_sub(self.done.len());
                    toasts.info(format!("Wrapped to the start — {remaining} image(s) still to do."));
                }
                self.nav(idx);
            }
            None => {
                toasts.success(format!("All {} images done!", self.paths.len()));
                self.status = "All images marked done.".into();
            }
        }
    }

    fn save_session(&self) {
        if let Some(src) = &self.source_folder {
            TilePickerSession {
                source_folder: src.clone(),
                done: self.done.clone(),
                last_index: self.cur,
            }
            .save();
        }
    }

    // ── async loading ─────────────────────────────────────────────────────────

    fn request_image(&mut self) {
        if self.paths.is_empty() || self.img_rx.is_some() {
            return;
        }
        if self.img_for == Some(self.cur) {
            return;
        }
        self.spawn_load(self.cur);
    }

    fn spawn_load(&mut self, idx: usize) {
        let path = self.paths[idx].clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let res = image::open(&path)
                .map(|im| {
                    let rgba = im.to_rgba8();
                    let (w, h) = rgba.dimensions();
                    (rgba.into_raw(), w, h)
                })
                .map_err(|e| e.to_string());
            let _ = tx.send(res);
        });
        self.img_rx = Some(rx);
        self.loading_idx = Some(idx);
    }

    fn poll_image(&mut self, ctx: &Context) {
        let Some(rx) = &self.img_rx else { return };
        let Ok(res) = rx.try_recv() else { return };
        self.img_rx = None;
        let idx = self.loading_idx.take();
        match res {
            Ok((rgba, w, h)) => {
                let tex = make_main_texture(ctx, &rgba, w, h);
                self.img = Some(LoadedImage { rgba, w, h, tex });
                self.img_for = idx;
            }
            Err(e) => {
                self.status = format!("Load failed: {e}");
                self.img = None;
                self.img_for = idx; // mark attempted so we don't spin
            }
        }
        self.loupe_tex = None;
    }

    fn pump_thumbs(&mut self, ctx: &Context) {
        // collect a finished thumbnail
        if let Some((idx, rx)) = &self.thumb_rx {
            if let Ok(opt) = rx.try_recv() {
                let idx = *idx;
                match opt {
                    Some(ci) => {
                        let tex =
                            ctx.load_texture(format!("tp_thumb_{idx}"), ci, TextureOptions::LINEAR);
                        self.thumbs.insert(idx, tex);
                    }
                    None => {
                        self.thumb_failed.insert(idx);
                    }
                }
                self.thumb_rx = None;
            }
        }
        // request the next missing thumbnail in the visible filmstrip window
        if self.paths.is_empty() {
            return;
        }
        let last = self.paths.len() - 1;
        let lo = self.vis_lo.min(last);
        let hi = self.vis_hi.min(last);
        if self.thumb_rx.is_none() {
            for idx in lo..=hi {
                if !self.thumbs.contains_key(&idx) && !self.thumb_failed.contains(&idx) {
                    self.spawn_thumb(idx);
                    break;
                }
            }
        }
        // bound memory: drop thumbnails far from the visible window
        let keep_lo = lo.saturating_sub(12);
        let keep_hi = hi + 12;
        self.thumbs.retain(|&k, _| k >= keep_lo && k <= keep_hi);
    }

    fn spawn_thumb(&mut self, idx: usize) {
        let path = self.paths[idx].clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let ci = image::open(&path).ok().map(|im| {
                let t = im.thumbnail(THUMB as u32, THUMB as u32).to_rgba8();
                let (w, h) = t.dimensions();
                ColorImage::from_rgba_unmultiplied([w as usize, h as usize], t.as_raw())
            });
            let _ = tx.send(ci);
        });
        self.thumb_rx = Some((idx, rx));
    }

    fn poll_dialogs(&mut self) {
        if let Some(rx) = &self.source_rx {
            if let Ok(res) = rx.try_recv() {
                if let Some(p) = res {
                    self.set_source(p);
                }
                self.source_rx = None;
            }
        }
        if let Some(rx) = &self.output_rx {
            if let Ok(res) = rx.try_recv() {
                if let Some(p) = res {
                    self.output_folder = Some(p);
                }
                self.output_rx = None;
            }
        }
    }

    fn set_source(&mut self, p: PathBuf) {
        let mut paths = list_images(&p);
        paths.sort();
        // resume per-folder progress (done markers + last position)
        let (done, start) = match TilePickerSession::load(&p) {
            Some(s) => {
                let start = if s.last_index < paths.len() { s.last_index } else { 0 };
                (s.done, start)
            }
            None => (HashSet::new(), 0),
        };
        self.paths = paths;
        self.source_folder = Some(p);
        self.done = done;
        self.cur = start;
        self.scroll_to_current = true;
        self.vis_lo = start.saturating_sub(2);
        self.vis_hi = start + 12;
        self.img = None;
        self.img_for = None;
        self.loading_idx = None;
        self.img_rx = None;
        self.thumbs.clear();
        self.thumb_failed.clear();
        self.thumb_rx = None;
        self.stamps.clear();
        self.total_stamped = 0;
        self.loupe_tex = None;
        self.view_zoom = 1.0;
        self.view_pan = Vec2::ZERO;
        self.status.clear();
    }
}

// ── free helpers ──────────────────────────────────────────────────────────────

/// Build the magnified loupe image (source pixels around `center`, composited
/// over a checkerboard so transparency shows; out-of-image is dark grey).
fn build_loupe(img: &LoadedImage, center: [f32; 2], factor: f32, s: usize) -> ColorImage {
    let half = s as f32 / 2.0;
    let mut pixels = Vec::with_capacity(s * s);
    for j in 0..s {
        for i in 0..s {
            let sx = center[0] + (i as f32 - half) / factor;
            let sy = center[1] + (j as f32 - half) / factor;
            let col = if sx >= 0.0 && sy >= 0.0 && sx < img.w as f32 && sy < img.h as f32 {
                let o = ((sy as u32 * img.w + sx as u32) * 4) as usize;
                let a = img.rgba[o + 3] as f32 / 255.0;
                let base = loupe_checker(i, j);
                let mix = |c: u8, b: u8| ((c as f32) * a + (b as f32) * (1.0 - a)) as u8;
                Color32::from_rgb(
                    mix(img.rgba[o], base.r()),
                    mix(img.rgba[o + 1], base.g()),
                    mix(img.rgba[o + 2], base.b()),
                )
            } else {
                Color32::from_gray(30)
            };
            pixels.push(col);
        }
    }
    ColorImage { size: [s, s], pixels }
}

fn loupe_checker(i: usize, j: usize) -> Color32 {
    if ((i / 8) + (j / 8)) % 2 == 0 {
        Color32::from_gray(55)
    } else {
        Color32::from_gray(80)
    }
}

/// Paint a checkerboard inside `rect` so a transparent image reads clearly.
fn draw_checker(painter: &egui::Painter, rect: Rect, cell: f32) {
    painter.rect_filled(rect, 0.0, Color32::from_gray(60));
    let cols = (rect.width() / cell).ceil() as i32;
    let rows = (rect.height() / cell).ceil() as i32;
    for r in 0..rows {
        for c in 0..cols {
            if (r + c) % 2 == 0 {
                continue;
            }
            let x = rect.min.x + c as f32 * cell;
            let y = rect.min.y + r as f32 * cell;
            let cr = Rect::from_min_size(Pos2::new(x, y), Vec2::splat(cell)).intersect(rect);
            painter.rect_filled(cr, 0.0, Color32::from_gray(82));
        }
    }
}

/// Build the display texture, downscaling if the image exceeds the GPU cap.
/// The full-resolution buffer is kept separately for cropping, so coordinate
/// math always uses the true dimensions regardless of this texture's size.
fn make_main_texture(ctx: &Context, rgba: &[u8], w: u32, h: u32) -> TextureHandle {
    let maxd = w.max(h);
    if maxd <= MAX_TEX {
        let ci = ColorImage::from_rgba_unmultiplied([w as usize, h as usize], rgba);
        return ctx.load_texture("tilepick_main", ci, TextureOptions::LINEAR);
    }
    let f = maxd.div_ceil(MAX_TEX);
    let tw = (w / f).max(1);
    let th = (h / f).max(1);
    let mut pixels = Vec::with_capacity((tw * th) as usize);
    for y in 0..th {
        let sy = (y * f).min(h - 1);
        for x in 0..tw {
            let sx = (x * f).min(w - 1);
            let o = ((sy * w + sx) * 4) as usize;
            pixels.push(Color32::from_rgba_unmultiplied(
                rgba[o],
                rgba[o + 1],
                rgba[o + 2],
                rgba[o + 3],
            ));
        }
    }
    ctx.load_texture(
        "tilepick_main",
        ColorImage { size: [tw as usize, th as usize], pixels },
        TextureOptions::LINEAR,
    )
}

fn spawn_folder_dialog() -> mpsc::Receiver<Option<PathBuf>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(rfd::FileDialog::new().pick_folder());
    });
    rx
}

fn path_str(p: &Option<PathBuf>) -> String {
    p.as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "- not set -".to_string())
}
