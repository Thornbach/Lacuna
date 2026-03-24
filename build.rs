// build.rs — generates a leaf icon and embeds it in the Windows .exe
//
// The icon is synthesised mathematically (no asset files needed):
//   - a tilted green ellipse with a central vein
//   - 32×32 RGBA, wrapped in a BMP-DIB ICO (no external crates)
// The raw RGBA bytes are also written to $OUT_DIR/leaf_rgba.bin so that
// main.rs can set the same image as the runtime window/taskbar icon via
// eframe's icon_data.

fn main() {
    let out = std::env::var("OUT_DIR").unwrap();

    // Generate the 32×32 RGBA pixel data
    let rgba = leaf_rgba();

    // Write raw RGBA for use in main.rs (include_bytes!)
    let rgba_path = format!("{}/leaf_rgba.bin", out);
    std::fs::write(&rgba_path, &rgba).expect("write leaf_rgba.bin");

    // Wrap in a valid ICO file (BMP DIB format)
    let ico = make_ico(&rgba);
    let ico_path = format!("{}/leaf.ico", out);
    std::fs::write(&ico_path, &ico).expect("write leaf.ico");

    // Embed the icon into the Windows .exe via winres
    #[cfg(target_os = "windows")]
    {
        let mut res = winres::WindowsResource::new();
        res.set_icon(&ico_path);
        if let Err(e) = res.compile() {
            eprintln!("cargo:warning=Could not embed .exe icon: {e}");
        }
    }

    // Re-run only when this script changes
    println!("cargo:rerun-if-changed=build.rs");
}

// ── leaf shape ───────────────────────────────────────────────────────────────

/// Returns 32×32 RGBA pixels (top-to-bottom, left-to-right).
fn leaf_rgba() -> Vec<u8> {
    let mut out = vec![0u8; 32 * 32 * 4];

    // Rotation angle: 20° clockwise so the leaf tilts naturally
    let angle: f64 = 20.0_f64.to_radians();
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    for py in 0u32..32 {
        for px in 0u32..32 {
            // Centre the canvas
            let cx = px as f64 - 15.5;
            let cy = py as f64 - 15.5;

            // Rotate into the leaf's local frame
            let rx =  cx * cos_a + cy * sin_a;
            let ry = -cx * sin_a + cy * cos_a;

            // Ellipse: semi-axes (6.5 wide, 13 tall in local frame)
            let t = (rx / 6.5).powi(2) + (ry / 13.0).powi(2);
            if t > 1.0 { continue; } // transparent

            let i = (py * 32 + px) as usize * 4;

            // Darken slightly toward the edge for a subtle shading
            let brightness = 1.0 - t.sqrt() * 0.38;

            // Central vein: thin stripe along rx ≈ 0
            let on_vein = rx.abs() < 0.85 && ry.abs() < 11.5;

            if on_vein {
                out[i]     = 18;
                out[i + 1] = (100.0 * brightness) as u8;
                out[i + 2] = 18;
                out[i + 3] = 255;
            } else {
                out[i]     = (28.0  + 18.0 * brightness) as u8;
                out[i + 1] = (120.0 + 50.0 * brightness) as u8;
                out[i + 2] = (18.0  + 12.0 * brightness) as u8;
                out[i + 3] = 255;
            }
        }
    }
    out
}

// ── ICO file builder (BMP-DIB, no external crates) ───────────────────────────

/// Wraps 32×32 RGBA pixels into a minimal Windows ICO file (BMP-DIB, 32bpp).
fn make_ico(rgba: &[u8]) -> Vec<u8> {
    assert_eq!(rgba.len(), 32 * 32 * 4);

    // Convert RGBA (top-to-bottom) → BGRA (bottom-to-top) as BMP requires
    let mut bgra = vec![0u8; 32 * 32 * 4];
    for row in 0..32usize {
        for col in 0..32usize {
            let src = (row * 32 + col) * 4;
            let dst = ((31 - row) * 32 + col) * 4; // vertical flip
            bgra[dst]     = rgba[src + 2]; // B
            bgra[dst + 1] = rgba[src + 1]; // G
            bgra[dst + 2] = rgba[src];     // R
            bgra[dst + 3] = rgba[src + 3]; // A
        }
    }

    // AND mask: 32 pixels/row → 4 bytes/row → 32 rows = 128 bytes (all zero)
    // Modern Windows ignores this for 32bpp; we include it for spec compliance.
    let and_mask = vec![0u8; 128];

    // Total size of the BMP image data stored in the ICO
    let bmp_data_size: u32 = 40 + bgra.len() as u32 + and_mask.len() as u32;

    // ── BMP INFOHEADER ────────────────────────────────────────────────────────
    let mut bmp = Vec::new();
    bmp.extend_from_slice(&40u32.to_le_bytes());          // biSize
    bmp.extend_from_slice(&32i32.to_le_bytes());          // biWidth
    bmp.extend_from_slice(&64i32.to_le_bytes());          // biHeight (2× for ICO)
    bmp.extend_from_slice(&1u16.to_le_bytes());           // biPlanes
    bmp.extend_from_slice(&32u16.to_le_bytes());          // biBitCount
    bmp.extend_from_slice(&0u32.to_le_bytes());           // biCompression (BI_RGB)
    bmp.extend_from_slice(&(bgra.len() as u32).to_le_bytes()); // biSizeImage
    bmp.extend_from_slice(&0i32.to_le_bytes());           // biXPelsPerMeter
    bmp.extend_from_slice(&0i32.to_le_bytes());           // biYPelsPerMeter
    bmp.extend_from_slice(&0u32.to_le_bytes());           // biClrUsed
    bmp.extend_from_slice(&0u32.to_le_bytes());           // biClrImportant
    bmp.extend_from_slice(&bgra);
    bmp.extend_from_slice(&and_mask);

    // ── ICO file ──────────────────────────────────────────────────────────────
    let data_offset: u32 = 6 + 16; // ICO header + 1 directory entry = 22

    let mut ico = Vec::new();
    // Header
    ico.extend_from_slice(&[0x00, 0x00]);           // reserved
    ico.extend_from_slice(&1u16.to_le_bytes());     // type = icon
    ico.extend_from_slice(&1u16.to_le_bytes());     // image count = 1
    // ICONDIRENTRY
    ico.push(32);                                    // width
    ico.push(32);                                    // height
    ico.push(0);                                     // color count (0 = > 256)
    ico.push(0);                                     // reserved
    ico.extend_from_slice(&1u16.to_le_bytes());     // planes
    ico.extend_from_slice(&32u16.to_le_bytes());    // bit count
    ico.extend_from_slice(&bmp_data_size.to_le_bytes()); // bytes in resource
    ico.extend_from_slice(&data_offset.to_le_bytes());   // offset to image data
    // Image data
    ico.extend_from_slice(&bmp);

    ico
}
