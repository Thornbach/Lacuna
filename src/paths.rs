//! Portable resolution of bundled model weights.
//!
//! The weight lookups used to end in a hardcoded `E:\PhD_TobiMu\...` fallback.
//! On the developer machine that is invisible — the sibling-file check almost
//! always hits first — but it is not a path that can exist on a colleague's
//! laptop, on a conference machine, or on Linux/macOS at all. When the sibling
//! check missed, the user got "No such file: E:\PhD_TobiMu\02_code\..." naming a
//! drive they do not have, which reads as a broken program rather than as a
//! missing model file.
//!
//! What replaces it is the ordinary convention for a shipped desktop app: look
//! next to the executable. `scripts/package.ps1` already lays the zip out as
//! `lacuna.exe` + `models/`, so `<exe_dir>/models/<name>` is exactly where the
//! file is in every distributed build.

use std::path::{Path, PathBuf};

/// Directory containing the running executable, if it can be determined.
///
/// In a `cargo run` / `cargo test` build this is `target/<profile>/`, not the
/// repo root — which is why the dev-tree fallback below still exists.
pub fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
}

/// Locate a weights file, in decreasing order of specificity:
///
/// 1. `$env_var` — explicit override, always wins (used by the bench/validate
///    subcommands and by anyone testing a re-export).
/// 2. Next to `model_path` — keeps a hand-picked model directory self-contained,
///    which is how the app has always found weights in practice.
/// 3. `<exe_dir>/models/` and `<exe_dir>/` — the packaged layout.
/// 4. `./models/` and `./port/` relative to the working directory — the repo
///    layout, so `cargo run` from the source tree keeps working.
///
/// Returns the first path that exists. If none do, returns the packaged
/// location, so the resulting error names a path the user can actually act on
/// ("put the file here") instead of a drive letter from another machine.
pub fn resolve_weights(model_path: &Path, filename: &str, env_var: &str) -> PathBuf {
    if let Ok(p) = std::env::var(env_var) {
        return PathBuf::from(p);
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(dir) = model_path.parent() {
        candidates.push(dir.join(filename));
    }
    if let Some(exe) = exe_dir() {
        candidates.push(exe.join("models").join(filename));
        candidates.push(exe.join(filename));
    }
    candidates.push(PathBuf::from("models").join(filename));
    candidates.push(PathBuf::from("port").join(filename));

    if let Some(hit) = candidates.iter().find(|p| p.exists()) {
        return hit.clone();
    }

    // Nothing found. Point at where a packaged install expects it.
    exe_dir()
        .map(|e| e.join("models").join(filename))
        .unwrap_or_else(|| PathBuf::from("models").join(filename))
}
