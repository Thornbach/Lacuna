# ShapeComplexity v0.5

**A powerful tool for analyzing leaf edge and macro-shape complexity from PNG images**

ShapeComplexity provides both command-line and graphical interfaces for comprehensive leaf morphology analysis, measuring edge complexity (EC) and macro-shape complexity (MC) through advanced image processing algorithms.

---

##  Features

### Edge Complexity (EC) Analysis
- **Geodesic path analysis** along leaf edges with adaptive morphological opening
- **Approximate Entropy (ApEn)** calculation for edge irregularity
- **Adaptive petiole filtering** for accurate edge measurements

### Macro-shape Complexity (MC) Analysis
- **Harmonic enhancement** for capturing periodic patterns
- **Spectral entropy** using FFT-based frequency analysis

### Dual Interface
- 🖥️ **GUI Application** - Interactive visual analysis with real-time overlays
- ⌨️ **CLI Tool** - Batch processing for high-throughput analysis

### Advanced Capabilities
- Parallel processing for batch operations
- Configurable reference points (Center of Mass or Emerge Point)
- Comprehensive CSV output with 20+ metrics per image

---

## 📥 Installation

### Windows (Portable - No Installation Required)

**Download the latest release:**
- **GUI Version**: [LeafComplexR-GUI-v0.5-Windows.zip](https://github.com/Thornbach/ShapeComplexity/releases/download/v0.5/leaf_complex_gui.exe)

Simply extract and double-click the `.exe` file to start!


---

## 🚀 Quick Start

### GUI Application
1. Download and extract `LeafComplexR-GUI-v0.5-Windows.zip`
2. Double-click `leaf_complex_gui.exe`
3. Click **File → Open Workspace** and select your image folder
4. Select images and click **Analyze**
5. Export results via **File → Export Selected Analysis**


---

## 📊 Output Format

LeafComplexR generates:

```
output/
├── EC/                    # Edge Complexity visualizations (pink regions = opaque)
│   └── [filename]_EC.png
├── MC/                    # Margin Complexity visualizations (pink regions = transparent)
│   └── [filename]_MC.png
└── summary/
    ├── EC_summary.csv     # Edge complexity metrics
    └── MC_summary.csv     # Margin complexity metrics
```

**Key Metrics:**
- Geodesic Edge Complexity (length, curvature, entropy)
- Approximate Entropy (ApEn)
- Spectral Entropy (SpEn)
- Shape indices (length, width, aspect ratio)

---

## 🔧 Configuration

Edit `config.toml` to customize analysis parameters:

```toml
# Image processing
resize_dimensions = [512, 512]
opening_kernel_size = 9

# Reference point calculation
reference_point_choice = "COM"  # "COM" or "EP"

# Advanced parameters
adaptive_opening_max_density = 75.0
enable_petiole_filter_ec = true
thornfiddle_smoothing_strength = 2.0
```

See [config.toml](config.toml) for full documentation of all parameters.

---

## 📚 Documentation


- **[config.toml](config.toml)** - Parameter reference with explanations

---

## 🧪 Example Workflow

### Academic Research Pipeline
1. Collect leaf images with transparent backgrounds (PNG)
2. Organize images by species/treatment in folders
3. Run batch analysis with CLI tool
4. Import CSV results into R/Python for statistical analysis
5. Use GUI for interactive exploration of interesting cases

### Quality Control
1. Load image set in GUI
2. Visually inspect EC/MC overlays
3. Adjust configuration parameters interactively
4. Re-analyze flagged images
5. Export final results

---

##  Building from Source

If youre on Linux or MacOS you may build the tool from source

```bash
# Install Rust (https://rustup.rs/)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone repository
git clone https://github.com/Thornbach/leaf_complex_rust.git
cd leaf_complex_rust

# Build CLI
cargo build --release --bin leaf_complex_rust_cli

# Build GUI
cd leaf_complex_gui
cargo build --release

# Executables will be in target/release/
```

---

## 📖 Citation

If you use ShapeComplexity in your research, please cite:

```
[Your Citation Here]
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Tobias Müller**  
📧 tobimu@proton.me

---

## 🐛 Issues & Feedback

Found a bug or have a feature request? Please open an issue on [GitHub Issues](https://github.com/YOUR_USERNAME/leaf_complex_rust/issues).

---

## 🙏 Acknowledgments

Special thanks to contributors and the Rust community for excellent image processing libraries.

---

**Version:** 0.5  
**Last Updated:** January 2026  
**Status:** Stable Release

