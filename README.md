# Lacuna

**A pure-Rust desktop toolkit for reading leaves** — instance segmentation, label-free
anomaly/damage detection, defect clustering, intact-blade reconstruction, and
shape-complexity morphology, in one GPU-accelerated app.

> *lacuna* — a gap or missing part; in botany, an air-space in leaf tissue. The name
> also nods to **lacunarity**, a texture-complexity measure used in leaf and canopy
> analysis: Lacuna both *finds* the gaps (holes, lesions, missing blade) and *measures*
> the complexity of what remains.

Built with [egui](https://github.com/emilk/egui)/eframe and the
[Burn](https://burn.dev) deep-learning framework. Everything runs natively on the GPU —
**no Python, no ONNX Runtime, no cuDNN at inference time.**

---

## What it does

A streaming, per-leaf pipeline:

1. **Segment** — YOLO26-seg detects and cuts out individual leaves from a scan.
2. **Detect** — each leaf is tiled and scored for anomalies with a frozen **DINOv3**
   backbone + a PatchCore coreset bank, fused with residual and CIELAB-colour channels.
3. **Cluster** — the detected defect regions are described (mechanism / colour / shape)
   and grouped into families with DBSCAN.
4. **Reconstruct** — a U-Net estimates the intact leaf outline to recover the original
   blade area lost to damage.
5. **Morphology** — external- and marked-contour leaf-complexity metrics
   (EC/MC, margin complexity, entropy) via the embedded ShapeComplexity engine.

Standalone tabs are also provided for segmentation, tile picking, sorting, and each
reconstruction/morphology stage.

## The models

The two neural backbones were **hand-reimplemented in Burn** (not auto-converted) and
validated numerically against their original ONNX exports:

| Model | Role | Validation vs ONNX |
|---|---|---|
| **DINOv3 ViT-B/16** | patch features for anomaly detection | `max|Δ| ≈ 7.7e-6` |
| **YOLO26m-seg** | leaf instance segmentation | end-to-end mask **IoU = 1.0000** |

This is what makes Lacuna self-contained: one Burn backend drives DINOv3, YOLO26, and the
reconstruction U-Net, so the shipped binary needs no onnxruntime/cuDNN DLLs.

## Building

```bash
# NVIDIA GPU (default) — self-contained, no cuDNN
cargo build --release

# Any GPU via Vulkan/DX12/Metal
cargo build --release --no-default-features --features wgpu-gpu

# Portable CPU (slow, fully portable)
cargo build --release --no-default-features
```

Optional ONNX Runtime fallback (re-adds the onnxruntime dependency):

```bash
cargo build --release --features ort-cuda    # then set LACUNA_USE_ORT=1 at runtime
```

## Model weights

The network weights (DINOv3 ≈ 327 MB, YOLO26 ≈ 90 MB, reconstruction checkpoints) are
**not tracked in git** — they're too large. Place `dino_weights.safetensors` and
`yolo_weights.safetensors` in `models/` (next to the small config JSONs), or regenerate
them from the source checkpoints with the scripts in `port/` (`export_dino_weights.py`,
`yolo_export_and_ref.py`).

## Status

Research prototype (PhD project). Interfaces and defaults are still moving.

## License

_TBD._
