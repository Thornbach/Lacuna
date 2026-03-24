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
            window_shadow: Shadow::NONE,
            popup_shadow:  Shadow::NONE,
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
    ]
}

pub fn theme_names() -> Vec<&'static str> {
    all_themes().iter().map(|t| t.name).collect()
}

pub fn apply(ctx: &egui::Context, name: &str) {
    if let Some(t) = all_themes().into_iter().find(|t| t.name == name) {
        ctx.set_visuals(t.to_visuals());
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
