//! One registry of every keyboard action in the Pipeline tab.
//!
//! Before this existed, bindings were literals scattered across ~40 `key_pressed`
//! sites and documented — where they were documented at all — in code comments and
//! a handful of tooltips. Two consequences, both real:
//!
//!   * `Enter` was bound twice (confirm the selection, and commit a Scissor cut),
//!     and the Scissor site had no focus guard, so pressing Enter after typing a
//!     cluster name could trigger a cut. Nothing could have detected that, because
//!     nothing knew the full set of bindings.
//!   * The ten tool buttons render icon-only and their tooltips never state the
//!     letter, so the fast path stayed invisible. Measured shortcut adoption is
//!     ~10% even among experts; the binding has to be printed where the mouse
//!     user already is, or it is never learned.
//!
//! So this module owns the bindings, and the dispatch sites, the button labels,
//! the help overlay and the conflict check all read from here. Adding an action
//! without its documentation is now impossible by construction.

use std::collections::HashMap;

use egui::Key;

/// A bindable action. `id` is the stable key used on disk — renaming one silently
/// resets that user's binding, so treat them as permanent.
pub struct ActionDef {
    pub id:      &'static str,
    pub group:   &'static str,
    pub label:   &'static str,
    pub hint:    &'static str,
    pub default_key: Key,
}

/// Group order here is the order the help overlay renders, so it runs from
/// most-used to least rather than alphabetically.
pub const ACTIONS: &[ActionDef] = &[
    // ── moving between leaves ───────────────────────────────────────────────
    ActionDef { id: "leaf.prev", group: "Leaves", label: "Previous leaf",
        hint: "Clamps at the first leaf — it does not wrap.", default_key: Key::ArrowLeft },
    ActionDef { id: "leaf.next", group: "Leaves", label: "Next leaf",
        hint: "Clamps at the last leaf — it does not wrap.", default_key: Key::ArrowRight },
    ActionDef { id: "leaf.next_unreviewed", group: "Leaves", label: "Next unreviewed leaf",
        hint: "Skips leaves already marked reviewed or rejected. Wraps once, so it \
               finds work anywhere in the batch.", default_key: Key::N },

    // ── whole-leaf verdicts ─────────────────────────────────────────────────
    ActionDef { id: "leaf.reviewed", group: "Leaf verdict", label: "Mark reviewed",
        hint: "A bookmark: changes nothing about the export. Saved to disk so a long \
               batch survives closing the app.", default_key: Key::Y },
    ActionDef { id: "leaf.reject", group: "Leaf verdict", label: "Reject / restore leaf",
        hint: "Throws the leaf out of the run entirely — excluded from the CSV, the \
               counts and from mining. Reversible, and saved to disk.", default_key: Key::X },

    // ── judging detections ──────────────────────────────────────────────────
    ActionDef { id: "region.confirm", group: "Detections", label: "Confirm selected",
        hint: "Writes the selection to the curation set as correct examples.",
        default_key: Key::Enter },
    ActionDef { id: "region.reject", group: "Detections", label: "Reject selected",
        hint: "Rejects the selection. Undoable with Ctrl+Z.", default_key: Key::Delete },
    ActionDef { id: "region.reassign", group: "Detections", label: "Reassign to a family",
        hint: "Opens the family picker at the pointer.", default_key: Key::R },
    // The single largest interaction-cost item in the app. Selecting a detection
    // was mouse-only — a trip to a 48px thumbnail or a mask-exact click on the
    // canvas, often after zooming. At ~4 detections x 10,000 leaves that is
    // 40,000 mouse round-trips one keystroke could replace.
    ActionDef { id: "region.next", group: "Detections", label: "Next detection",
        hint: "Steps through this leaf's detections in the gallery's current order, \
               so with \"Unusual first\" you meet the doubtful ones immediately.",
        default_key: Key::ArrowDown },
    ActionDef { id: "region.prev", group: "Detections", label: "Previous detection",
        hint: "Back one detection.", default_key: Key::ArrowUp },
    ActionDef { id: "region.flag", group: "Detections", label: "Set aside for later",
        hint: "Defer a hard call without stalling on it, and come back via the \
               filter. Deciding under fatigue is where label quality goes.",
        default_key: Key::F },

    // ── tools ───────────────────────────────────────────────────────────────
    ActionDef { id: "tool.select", group: "Tools", label: "Select",
        hint: "Click to select · drag to box-select · ctrl+click to multi-select.",
        default_key: Key::V },
    ActionDef { id: "tool.mark_healthy", group: "Tools", label: "Mark healthy",
        hint: "Stamp a patch as a healthy training example.", default_key: Key::H },
    ActionDef { id: "tool.brush", group: "Tools", label: "Brush", hint: "Paint a region.",
        default_key: Key::B },
    ActionDef { id: "tool.eraser", group: "Tools", label: "Eraser",
        hint: "Erase from a region's mask.", default_key: Key::E },
    ActionDef { id: "tool.knife", group: "Tools", label: "Knife",
        hint: "Split a region with a straight cut.", default_key: Key::K },
    ActionDef { id: "tool.scissor", group: "Tools", label: "Scissor",
        hint: "Split a region along a drawn polyline.", default_key: Key::S },
    ActionDef { id: "tool.lasso", group: "Tools", label: "Lasso",
        hint: "Free-hand select an area.", default_key: Key::L },
    ActionDef { id: "tool.wand", group: "Tools", label: "Magic wand",
        hint: "Select by colour similarity.", default_key: Key::W },
    ActionDef { id: "tool.eyedropper", group: "Tools", label: "Eyedropper",
        hint: "Pick a family from a region on the canvas.", default_key: Key::I },
    ActionDef { id: "tool.polygon", group: "Tools", label: "Polygon",
        hint: "Draw a region vertex by vertex.", default_key: Key::P },

    // ── run / review / export ───────────────────────────────────────────────
    // Bound to nothing by default (Key::F35 is a real key nobody has), so they
    // cost no keyboard real estate — they exist so the command palette can reach
    // them by name and so a user can bind them if they want. An action that is
    // only reachable by hunting for its button is an action most people never
    // find.
    ActionDef { id: "run.start", group: "Run", label: "Run pipeline",
        hint: "Segment, detect and cluster the source folder.", default_key: Key::F35 },
    ActionDef { id: "run.cancel", group: "Run", label: "Cancel the run",
        hint: "Stop after the current leaf. Finished leaves are kept.", default_key: Key::F35 },
    ActionDef { id: "review.export", group: "Run", label: "Export results",
        hint: "Write results.csv (and images, if those are ticked).", default_key: Key::F35 },
    ActionDef { id: "flow.finish", group: "Run", label: "Finish and proceed to export",
        hint: "Leave review for the finish screen. Nothing is written yet.",
        default_key: Key::F35 },
    ActionDef { id: "review.confirm_family", group: "Detections",
        label: "Confirm the whole focused family",
        hint: "Accept every unreviewed region of the family currently filtered to.",
        default_key: Key::F35 },
    ActionDef { id: "review.undo", group: "Detections", label: "Undo last edit",
        hint: "Reverses a reject, a confirm or a knife cut. Also Ctrl+Z.",
        default_key: Key::F35 },
    ActionDef { id: "view.outline", group: "View", label: "Toggle outline mode",
        hint: "Contours instead of filled regions.", default_key: Key::O },
    ActionDef { id: "view.recon", group: "View", label: "Toggle reconstruction tint",
        hint: "Show where the model thinks tissue was lost.", default_key: Key::F35 },
    // Slash, not F: F is already "Set aside for later" (region.flag) and has the
    // muscle memory to prove it. Slash was completely unbound.
    ActionDef { id: "view.focus", group: "View", label: "Toggle family focus",
        hint: "Dim every family except the selected one.", default_key: Key::Slash },
    ActionDef { id: "view.clear_focus", group: "View", label: "Clear family focus",
        hint: "Stop dimming the other families.", default_key: Key::F35 },
    ActionDef { id: "view.fit", group: "View", label: "Fit leaf to window",
        hint: "Reset zoom and pan.", default_key: Key::F35 },
    ActionDef { id: "view.panel", group: "View", label: "Hide / show the settings panel",
        hint: "Collapses the folders-and-Run column so the leaf gets the width. \
               The tool rail stays.", default_key: Key::Tab },

    // ── everything else ─────────────────────────────────────────────────────
    ActionDef { id: "help", group: "General", label: "Keyboard shortcuts",
        hint: "This window.", default_key: Key::F1 },
    // SPACE, not K. The palette's real trigger was a hardcoded Ctrl+K while this
    // said `Key::K`, so the shortcuts window reported a permanent clash with the
    // Knife tool over a key the palette never actually listened for. Space is
    // free (the canvas pans with a middle-drag) and is a single keypress.
    ActionDef { id: "palette", group: "General", label: "Search commands",
        hint: "Find any action by name. Ctrl+K also works.", default_key: Key::Space },
];

/// Actions with no default binding — shown in the palette and rebindable, but
/// not listed as "press this" until the user gives them a key.
pub fn is_unbound(k: Key) -> bool {
    k == Key::F35
}

pub fn action(id: &str) -> Option<&'static ActionDef> {
    ACTIONS.iter().find(|a| a.id == id)
}

/// Only the DIFFERENCES from the defaults are stored, so a future release that
/// changes a default key reaches users who never rebound it, while still
/// honouring the choice of users who did.
#[derive(Default, Clone)]
pub struct Keymap {
    overrides: HashMap<String, Key>,
}

impl Keymap {
    pub fn key(&self, id: &str) -> Key {
        self.overrides
            .get(id)
            .copied()
            .or_else(|| action(id).map(|a| a.default_key))
            .unwrap_or(Key::F35) // unreachable for a registered id; never matches a real press
    }

    pub fn is_default(&self, id: &str) -> bool {
        !self.overrides.contains_key(id)
    }

    pub fn set(&mut self, id: &str, k: Key) {
        match action(id) {
            Some(a) if a.default_key == k => { self.overrides.remove(id); }
            Some(_) => { self.overrides.insert(id.to_string(), k); }
            None => {}
        }
    }

    pub fn reset_one(&mut self, id: &str) {
        self.overrides.remove(id);
    }

    /// "Generate the default state" — drop every override at once.
    pub fn reset_all(&mut self) {
        self.overrides.clear();
    }

    pub fn n_customised(&self) -> usize {
        self.overrides.len()
    }

    /// Keys bound to more than one action, with the ids that share them.
    ///
    /// Reported rather than prevented: two actions CAN legitimately share a key
    /// when they are never live at the same time (a tool-specific key versus a
    /// gallery key). The overlay shows the clash so the user decides, instead of
    /// the app quietly refusing a binding it cannot actually prove is wrong.
    pub fn conflicts(&self) -> Vec<(Key, Vec<&'static str>)> {
        let mut by_key: HashMap<Key, Vec<&'static str>> = HashMap::new();
        for a in ACTIONS {
            let k = self.key(a.id);
            // UNBOUND actions all share the F35 sentinel, which is not a key
            // anyone can press. Counting them produced one nonsense warning
            // listing every unbound action at once ("— is bound to 9 actions"),
            // which buried the one real clash underneath it.
            if is_unbound(k) {
                continue;
            }
            by_key.entry(k).or_default().push(a.id);
        }
        let mut out: Vec<(Key, Vec<&'static str>)> =
            by_key.into_iter().filter(|(_, v)| v.len() > 1).collect();
        out.sort_by_key(|(k, _)| k.name());
        out
    }

    /// Read inside a `ctx.input(|i| …)` closure.
    pub fn pressed(&self, i: &egui::InputState, id: &str) -> bool {
        i.key_pressed(self.key(id))
    }

    // ── persistence ─────────────────────────────────────────────────────────
    // Stored as `id -> key name` strings rather than a numeric discriminant, so
    // a settings file stays readable and survives egui reordering its Key enum.

    pub fn to_map(&self) -> HashMap<String, String> {
        self.overrides
            .iter()
            .map(|(id, k)| (id.clone(), k.name().to_string()))
            .collect()
    }

    pub fn from_map(m: &HashMap<String, String>) -> Self {
        let mut km = Keymap::default();
        for (id, name) in m {
            // Drop unknown ids and unparseable names rather than failing to load:
            // a settings file from a newer build must not break an older one.
            if let (Some(_), Some(k)) = (action(id), Key::from_name(name)) {
                km.overrides.insert(id.clone(), k);
            }
        }
        km
    }
}

/// What to print on a button or in a tooltip. `symbol_or_name` gives "←" rather
/// than "ArrowLeft", which is what a user expects to see on a key cap.
pub fn key_label(k: Key) -> &'static str {
    if is_unbound(k) { "—" } else { k.symbol_or_name() }
}

/// Subsequence match with a score, for the command palette.
///
/// Deliberately not a dependency: "cfm" should find "Confirm the whole focused
/// family", and ~30 lines of subsequence scoring does that well enough for a
/// list of twenty-odd actions. Consecutive matches and word-start matches score
/// higher, so the intuitive abbreviation wins.
///
/// Returns `None` when the query is not a subsequence at all.
pub fn fuzzy_score(haystack: &str, needle: &str) -> Option<i32> {
    if needle.is_empty() {
        return Some(0);
    }
    let h: Vec<char> = haystack.to_lowercase().chars().collect();
    let n: Vec<char> = needle.to_lowercase().chars().collect();
    let mut hi = 0usize;
    let mut score = 0i32;
    let mut last_hit: Option<usize> = None;
    for &nc in &n {
        let mut found = None;
        while hi < h.len() {
            if h[hi] == nc {
                found = Some(hi);
                hi += 1;
                break;
            }
            hi += 1;
        }
        let pos = found?;
        score += 1;
        if last_hit == Some(pos.wrapping_sub(1)) {
            score += 4; // consecutive
        }
        if pos == 0 || h.get(pos.wrapping_sub(1)).is_some_and(|c| *c == ' ' || *c == '.') {
            score += 3; // start of a word
        }
        last_hit = Some(pos);
    }
    // Prefer shorter labels when scores tie — "Run pipeline" over a long one that
    // merely contains the same letters.
    Some(score * 100 - haystack.len() as i32)
}
