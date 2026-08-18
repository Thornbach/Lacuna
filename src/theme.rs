use egui::{Color32, Rounding, Stroke, Visuals, epaint::Shadow};
use egui::style::WidgetVisuals;

// ── Palette helper ────────────────────────────────────────────────────────────

macro_rules! rgb {
    ($r:expr, $g:expr, $b:expr) => { Color32::from_rgb($r, $g, $b) };
}

// ── Theme descriptor ──────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct ThemeDesc {
    pub name:     &'static str,
    pub dark:     bool,
    // backgrounds (darkest → lightest for dark themes, lightest → darkest for light)
    pub bg0:      Color32, // window / deepest bg
    pub bg1:      Color32, // panel fill
    pub bg2:      Color32, // widget inactive bg
    pub bg3:      Color32, // widget hover bg
    pub accent:   Color32, // primary accent / selection
    pub accent2:  Color32, // secondary accent (active / pressed)
    pub text:     Color32, // primary text
    pub text_dim: Color32, // secondary text / placeholder
    pub border:   Color32, // strokes / separators
    pub warn:     Color32,
    pub error:    Color32,
}

impl ThemeDesc {
    pub fn to_visuals(&self) -> Visuals {
        let base = if self.dark { Visuals::dark() } else { Visuals::light() };
        let r = Rounding::same(4.0);

        let noninteractive = WidgetVisuals {
            bg_fill:      self.bg1,
            weak_bg_fill: self.bg1,
            bg_stroke:    Stroke::new(1.0, self.border),
            fg_stroke:    Stroke::new(1.0, self.text_dim),
            rounding:     r,
            expansion:    0.0,
        };
        let inactive = WidgetVisuals {
            bg_fill:      self.bg2,
            weak_bg_fill: self.bg2,
            bg_stroke:    Stroke::new(1.0, self.border),
            fg_stroke:    Stroke::new(1.5, self.text),
            rounding:     r,
            expansion:    0.0,
        };
        let hovered = WidgetVisuals {
            bg_fill:      self.bg3,
            weak_bg_fill: self.bg3,
            bg_stroke:    Stroke::new(1.0, self.accent),
            fg_stroke:    Stroke::new(1.5, self.text),
            rounding:     r,
            expansion:    1.5,
        };
        let active = WidgetVisuals {
            bg_fill:      self.accent2.linear_multiply(0.4),
            weak_bg_fill: self.accent2.linear_multiply(0.2),
            bg_stroke:    Stroke::new(1.5, self.accent2),
            fg_stroke:    Stroke::new(2.0, self.text),
            rounding:     r,
            expansion:    1.5,
        };
        let open = WidgetVisuals {
            bg_fill:      self.bg3,
            weak_bg_fill: self.bg2,
            bg_stroke:    Stroke::new(1.0, self.accent),
            fg_stroke:    Stroke::new(1.5, self.text),
            rounding:     r,
            expansion:    0.0,
        };

        Visuals {
            dark_mode:           self.dark,
            override_text_color: None, // let widget visuals control text
            window_fill:         self.bg0,
            panel_fill:          self.bg1,
            window_stroke:       Stroke::new(1.0, self.border),
            extreme_bg_color:    if self.dark {
                self.bg0.linear_multiply(0.6)
            } else {
                self.bg0.linear_multiply(1.1)
            },
            faint_bg_color:      self.bg2.linear_multiply(0.5),
            code_bg_color:       self.bg2,
            hyperlink_color:     self.accent,
            warn_fg_color:       self.warn,
            error_fg_color:      self.error,
            selection: egui::style::Selection {
                bg_fill: self.accent.linear_multiply(0.35),
                stroke:  Stroke::new(1.0, self.accent),
            },
            widgets: egui::style::Widgets {
                noninteractive,
                inactive,
                hovered,
                active,
                open,
            },
            // Shadows were switched off on every theme, which removed the only
            // depth cue in the app — panels, popups, menus and dialogs all sat on
            // one flat plane with nothing indicating what floated above what.
            //
            // Tuned per polarity rather than shared: the same alpha that reads as
            // a soft shadow on a dark ground reads as a dirty smudge on a light
            // one, so light themes get roughly a third of the opacity.
            // NOTE: these fields are f32 in egui 0.29 (they become integers in
            // 0.31+, which is a compile error to watch for on any upgrade).
            window_shadow: Shadow {
                offset: egui::vec2(0.0, 4.0),
                blur:   16.0,
                spread: 0.0,
                color:  Color32::from_black_alpha(if self.dark { 110 } else { 38 }),
            },
            popup_shadow: Shadow {
                offset: egui::vec2(0.0, 2.0),
                blur:   10.0,
                spread: 0.0,
                color:  Color32::from_black_alpha(if self.dark { 90 } else { 30 }),
            },
            ..base
        }
    }
}

// ── Preset catalogue ──────────────────────────────────────────────────────────

pub fn all_themes() -> Vec<ThemeDesc> {
    vec![
        egui_dark(),
        egui_light(),
        catppuccin_mocha(),
        catppuccin_latte(),
        nord(),
        dracula(),
        solarized_dark(),
        gruvbox_dark(),
        tokyo_night(),
        monokai(),
        synthwave(),
        rose_pine(),
        everforest(),
        herbarium(),
        greenhouse(),
        sage_dark(),
        paper(),
    ]
}

// ── added for v0.5 ──────────────────────────────────────────────────────────
// Three light options and one more dark green. The catalogue was 11 dark to 2
// light, and the light pair were both cold blue-greys — for a tool whose subject
// is leaves, and which is used beside a window in daylight as often as in a dark
// office, that is a thin choice. Greens here are desaturated on purpose: an
// accent that competes with the leaf on the canvas is a bad accent.

/// Warm paper and ink, like a specimen sheet. Light.
fn herbarium() -> ThemeDesc {
    ThemeDesc {
        name:     "Herbarium",
        dark:     false,
        bg0:      rgb!(246, 244, 238),
        bg1:      rgb!(238, 235, 227),
        bg2:      rgb!(225, 221, 210),
        bg3:      rgb!(210, 205, 192),
        accent:   rgb!(58, 106, 74),
        accent2:  rgb!(76, 132, 94),
        text:     rgb!(32, 34, 30),
        text_dim: rgb!(104, 108, 98),
        border:   rgb!(198, 194, 182),
        warn:     rgb!(168, 118, 20),
        error:    rgb!(170, 56, 44),
    }
}

/// Cool, bright, slightly green-tinted white. Light.
fn greenhouse() -> ThemeDesc {
    ThemeDesc {
        name:     "Greenhouse",
        dark:     false,
        bg0:      rgb!(243, 248, 243),
        bg1:      rgb!(234, 241, 234),
        bg2:      rgb!(219, 229, 219),
        bg3:      rgb!(203, 215, 203),
        accent:   rgb!(42, 122, 92),
        accent2:  rgb!(56, 148, 112),
        text:     rgb!(24, 32, 27),
        text_dim: rgb!(96, 110, 100),
        border:   rgb!(192, 206, 193),
        warn:     rgb!(160, 116, 24),
        error:    rgb!(176, 58, 46),
    }
}

/// Muted green-grey, low contrast, for long sessions. Dark.
fn sage_dark() -> ThemeDesc {
    ThemeDesc {
        name:     "Sage",
        dark:     true,
        bg0:      rgb!(24, 30, 27),
        bg1:      rgb!(30, 38, 34),
        bg2:      rgb!(41, 51, 45),
        bg3:      rgb!(52, 64, 57),
        accent:   rgb!(126, 186, 142),
        accent2:  rgb!(150, 206, 166),
        text:     rgb!(220, 228, 222),
        text_dim: rgb!(140, 152, 144),
        border:   rgb!(58, 70, 63),
        warn:     rgb!(220, 176, 96),
        error:    rgb!(224, 118, 104),
    }
}

/// Near-white and neutral, maximum contrast for daylight. Light.
fn paper() -> ThemeDesc {
    ThemeDesc {
        name:     "Paper",
        dark:     false,
        bg0:      rgb!(252, 252, 251),
        bg1:      rgb!(245, 245, 244),
        bg2:      rgb!(232, 232, 230),
        bg3:      rgb!(216, 216, 213),
        accent:   rgb!(46, 98, 68),
        accent2:  rgb!(62, 126, 90),
        text:     rgb!(20, 20, 19),
        text_dim: rgb!(98, 100, 96),
        border:   rgb!(206, 206, 203),
        warn:     rgb!(150, 108, 16),
        error:    rgb!(166, 48, 38),
    }
}

pub fn theme_names() -> Vec<&'static str> {
    all_themes().iter().map(|t| t.name).collect()
}

pub fn apply(ctx: &egui::Context, name: &str) {
    if let Some(t) = all_themes().into_iter().find(|t| t.name == name) {
        ctx.set_visuals(t.to_visuals());
        // Publish the accent so `ui_kit::ACCENT()` follows the theme. Without
        // this the primary buttons and the wordmark stayed leaf green in all 13
        // themes — including the light ones, where that green sat at ~1.9:1
        // against its own near-black label.
        crate::ui_kit::set_theme_colors(t.accent, t.text_dim);
    }
}

// ── Definitions ───────────────────────────────────────────────────────────────

fn egui_dark() -> ThemeDesc {
    ThemeDesc {
        name:     "Egui Dark",
        dark:     true,
        bg0:      rgb!(27, 27, 27),
        bg1:      rgb!(33, 33, 33),
        bg2:      rgb!(55, 55, 55),
        bg3:      rgb!(70, 70, 70),
        accent:   rgb!(90, 170, 255),
        accent2:  rgb!(120, 200, 255),
        text:     rgb!(220, 220, 220),
        text_dim: rgb!(150, 150, 150),
        border:   rgb!(80, 80, 80),
        warn:     rgb!(255, 200, 0),
        error:    rgb!(255, 80, 80),
    }
}

fn egui_light() -> ThemeDesc {
    ThemeDesc {
        name:     "Egui Light",
        dark:     false,
        bg0:      rgb!(245, 245, 245),
        bg1:      rgb!(235, 235, 235),
        bg2:      rgb!(210, 210, 210),
        bg3:      rgb!(195, 195, 195),
        accent:   rgb!(0, 100, 200),
        accent2:  rgb!(0, 130, 255),
        text:     rgb!(20, 20, 20),
        text_dim: rgb!(100, 100, 100),
        border:   rgb!(170, 170, 170),
        warn:     rgb!(180, 120, 0),
        error:    rgb!(200, 30, 30),
    }
}

fn catppuccin_mocha() -> ThemeDesc {
    ThemeDesc {
        name:     "Catppuccin Mocha",
        dark:     true,
        bg0:      rgb!(0x11, 0x11, 0x1b), // crust
        bg1:      rgb!(0x18, 0x18, 0x25), // mantle
        bg2:      rgb!(0x31, 0x32, 0x44), // surface0
        bg3:      rgb!(0x45, 0x47, 0x5a), // surface1
        accent:   rgb!(0x89, 0xb4, 0xfa), // blue
        accent2:  rgb!(0xcb, 0xa6, 0xf7), // mauve
        text:     rgb!(0xcd, 0xd6, 0xf4),
        text_dim: rgb!(0xa6, 0xad, 0xc8), // subtext0
        border:   rgb!(0x58, 0x5b, 0x70), // surface2
        warn:     rgb!(0xf9, 0xe2, 0xaf), // yellow
        error:    rgb!(0xf3, 0x8b, 0xa8), // red
    }
}

fn catppuccin_latte() -> ThemeDesc {
    ThemeDesc {
        name:     "Catppuccin Latte",
        dark:     false,
        bg0:      rgb!(0xdc, 0xe0, 0xe8), // crust
        bg1:      rgb!(0xe6, 0xe9, 0xef), // mantle
        bg2:      rgb!(0xcc, 0xd0, 0xda), // surface0
        bg3:      rgb!(0xac, 0xb0, 0xbe), // surface1
        accent:   rgb!(0x1e, 0x66, 0xf5), // blue
        accent2:  rgb!(0x81, 0x60, 0xd0), // mauve
        text:     rgb!(0x4c, 0x4f, 0x69),
        text_dim: rgb!(0x6c, 0x6f, 0x85), // subtext1
        border:   rgb!(0x9c, 0xa0, 0xb0), // overlay1
        warn:     rgb!(0xdf, 0x8e, 0x1d), // yellow
        error:    rgb!(0xd2, 0x0f, 0x39), // red
    }
}

fn nord() -> ThemeDesc {
    ThemeDesc {
        name:     "Nord",
        dark:     true,
        bg0:      rgb!(0x2e, 0x34, 0x40), // nord0
        bg1:      rgb!(0x3b, 0x42, 0x52), // nord1
        bg2:      rgb!(0x43, 0x4c, 0x5e), // nord2
        bg3:      rgb!(0x4c, 0x56, 0x6a), // nord3
        accent:   rgb!(0x88, 0xc0, 0xd0), // nord8 (frost cyan)
        accent2:  rgb!(0x81, 0xa1, 0xc1), // nord9 (frost blue)
        text:     rgb!(0xec, 0xef, 0xf4), // nord6
        text_dim: rgb!(0xd8, 0xde, 0xe9), // nord4
        border:   rgb!(0x4c, 0x56, 0x6a), // nord3
        warn:     rgb!(0xeb, 0xcb, 0x8b), // nord13 (yellow)
        error:    rgb!(0xbf, 0x61, 0x6a), // nord11 (red)
    }
}

fn dracula() -> ThemeDesc {
    ThemeDesc {
        name:     "Dracula",
        dark:     true,
        bg0:      rgb!(0x21, 0x22, 0x2c),
        bg1:      rgb!(0x28, 0x2a, 0x36),
        bg2:      rgb!(0x44, 0x47, 0x5a),
        bg3:      rgb!(0x6a, 0x70, 0x8c),
        accent:   rgb!(0xbd, 0x93, 0xf9), // purple
        accent2:  rgb!(0xff, 0x79, 0xc6), // pink
        text:     rgb!(0xf8, 0xf8, 0xf2),
        text_dim: rgb!(0xc5, 0xc8, 0xd6),
        border:   rgb!(0x44, 0x47, 0x5a),
        warn:     rgb!(0xf1, 0xfa, 0x8c), // yellow
        error:    rgb!(0xff, 0x55, 0x55), // red
    }
}

fn solarized_dark() -> ThemeDesc {
    ThemeDesc {
        name:     "Solarized Dark",
        dark:     true,
        bg0:      rgb!(0x00, 0x2b, 0x36), // base03
        bg1:      rgb!(0x07, 0x36, 0x42), // base02
        bg2:      rgb!(0x58, 0x6e, 0x75), // base01
        bg3:      rgb!(0x65, 0x7b, 0x83), // base00
        accent:   rgb!(0x26, 0x8b, 0xd2), // blue
        accent2:  rgb!(0x2a, 0xa1, 0x98), // cyan
        text:     rgb!(0x83, 0x94, 0x96), // base0
        text_dim: rgb!(0x65, 0x7b, 0x83), // base00
        border:   rgb!(0x58, 0x6e, 0x75), // base01
        warn:     rgb!(0xb5, 0x89, 0x00), // yellow
        error:    rgb!(0xdc, 0x32, 0x2f), // red
    }
}

fn gruvbox_dark() -> ThemeDesc {
    ThemeDesc {
        name:     "Gruvbox Dark",
        dark:     true,
        bg0:      rgb!(0x1d, 0x20, 0x21),
        bg1:      rgb!(0x28, 0x28, 0x28),
        bg2:      rgb!(0x3c, 0x38, 0x36),
        bg3:      rgb!(0x50, 0x49, 0x45),
        accent:   rgb!(0x83, 0xa5, 0x98),
        accent2:  rgb!(0xd7, 0x99, 0x21),
        text:     rgb!(0xeb, 0xdb, 0xb2),
        text_dim: rgb!(0xa8, 0x99, 0x84),
        border:   rgb!(0x50, 0x49, 0x45),
        warn:     rgb!(0xfa, 0xbd, 0x2f),
        error:    rgb!(0xfb, 0x49, 0x34),
    }
}

fn tokyo_night() -> ThemeDesc {
    ThemeDesc {
        name:     "Tokyo Night",
        dark:     true,
        bg0:      rgb!(0x1a, 0x1b, 0x26), // dark navy
        bg1:      rgb!(0x1f, 0x23, 0x35), // panel
        bg2:      rgb!(0x24, 0x28, 0x3b), // widget
        bg3:      rgb!(0x29, 0x2e, 0x42), // hover
        accent:   rgb!(0x7a, 0xa2, 0xf7), // blue
        accent2:  rgb!(0xbb, 0x9a, 0xf7), // purple
        text:     rgb!(0xc0, 0xca, 0xf5),
        text_dim: rgb!(0x56, 0x5f, 0x89),
        border:   rgb!(0x3b, 0x42, 0x61),
        warn:     rgb!(0xe0, 0xaf, 0x68), // gold
        error:    rgb!(0xf7, 0x76, 0x8e), // rose
    }
}

fn monokai() -> ThemeDesc {
    ThemeDesc {
        name:     "Monokai",
        dark:     true,
        bg0:      rgb!(0x1e, 0x1e, 0x1e),
        bg1:      rgb!(0x27, 0x28, 0x22),
        bg2:      rgb!(0x3e, 0x3d, 0x32),
        bg3:      rgb!(0x52, 0x50, 0x44),
        accent:   rgb!(0xa6, 0xe2, 0x2e), // green
        accent2:  rgb!(0xfd, 0x97, 0x1f), // orange
        text:     rgb!(0xf8, 0xf8, 0xf2),
        text_dim: rgb!(0x75, 0x71, 0x5e),
        border:   rgb!(0x49, 0x48, 0x3e),
        warn:     rgb!(0xe6, 0xdb, 0x74), // yellow
        error:    rgb!(0xf9, 0x26, 0x72), // pink-red
    }
}

fn synthwave() -> ThemeDesc {
    ThemeDesc {
        name:     "Synthwave",
        dark:     true,
        bg0:      rgb!(0x0d, 0x0d, 0x1a), // deep space
        bg1:      rgb!(0x1a, 0x0a, 0x2e), // dark violet
        bg2:      rgb!(0x2d, 0x1b, 0x4e), // medium violet
        bg3:      rgb!(0x3d, 0x25, 0x64), // lighter violet
        accent:   rgb!(0xff, 0x00, 0xcc), // neon magenta
        accent2:  rgb!(0x00, 0xe5, 0xff), // neon cyan
        text:     rgb!(0xf0, 0xe6, 0xff),
        text_dim: rgb!(0x9b, 0x82, 0xc8),
        border:   rgb!(0x5a, 0x1f, 0x8c),
        warn:     rgb!(0xff, 0xea, 0x00), // neon yellow
        error:    rgb!(0xff, 0x44, 0x55), // neon red
    }
}

fn rose_pine() -> ThemeDesc {
    ThemeDesc {
        name:     "Rose Pine",
        dark:     true,
        bg0:      rgb!(0x19, 0x16, 0x24), // base
        bg1:      rgb!(0x1f, 0x1d, 0x2e), // surface
        bg2:      rgb!(0x26, 0x23, 0x3a), // overlay
        bg3:      rgb!(0x31, 0x2e, 0x44), // muted
        accent:   rgb!(0xc4, 0xa7, 0xe7), // iris (lavender)
        accent2:  rgb!(0xeb, 0xbc, 0xba), // rose
        text:     rgb!(0xe0, 0xde, 0xf4),
        text_dim: rgb!(0x6e, 0x6a, 0x86), // muted
        border:   rgb!(0x40, 0x3d, 0x52),
        warn:     rgb!(0xf6, 0xc1, 0x77), // gold
        error:    rgb!(0xeb, 0x6f, 0x92), // love
    }
}

fn everforest() -> ThemeDesc {
    ThemeDesc {
        name:     "Everforest",
        dark:     true,
        bg0:      rgb!(0x27, 0x2e, 0x33), // hard bg
        bg1:      rgb!(0x2d, 0x35, 0x3b), // bg0
        bg2:      rgb!(0x34, 0x3f, 0x44), // bg1
        bg3:      rgb!(0x3d, 0x48, 0x4d), // bg2
        accent:   rgb!(0x83, 0xc0, 0x92), // green
        accent2:  rgb!(0xa7, 0xc0, 0x80), // lime
        text:     rgb!(0xd3, 0xc6, 0xaa),
        text_dim: rgb!(0x85, 0x93, 0x8a), // grey2
        border:   rgb!(0x4f, 0x58, 0x5e),
        warn:     rgb!(0xdb, 0xc0, 0x74), // yellow
        error:    rgb!(0xe6, 0x75, 0x6b), // red
    }
}
