# Lacuna model weights

Lacuna auto-loads its models from this `models/` folder — found **next to
`lacuna.exe`** (packaged build) or **in the working directory** (dev / `cargo run`
from the project root). Any shared default you haven't set yourself in
Settings → Shared model defaults is filled in from here automatically.

## Runtime files (BURN — the default)

The pure-Rust backends load these directly; **no `.onnx`, no ONNX Runtime, no cuDNN.**

| File | What it is |
|---|---|
| `dino_weights.safetensors` | DINOv3 ViT-B/16 backbone (anomaly features). ~327 MB. |
| `yolo_weights.safetensors`  | YOLO26-seg leaf segmentation. ~90 MB. |
| `fewshot_head.json`         | Few-shot detection head (the detector; tiny, tracked in git). |
| `recon/gen.mpk`             | Reconstruction U-Net checkpoint. ~59 MB. |

Small config JSONs here are versioned in git; the multi-hundred-MB weights are **not**
(they're gitignored). A packaged release bundles all of the above — end users get them
automatically and never touch this folder.

## Getting the weights (developers building from source)

- **`yolo_weights.safetensors` + `recon/gen.mpk`** — download from the repo's GitHub
  Release and drop them here.
- **`dino_weights.safetensors`** — regenerate from the public DINOv3 model (its license
  requires you to accept Meta's terms once, so it isn't redistributed in the source repo):
  ```powershell
  # 1) accept the DINOv3 license on Hugging Face, create an access token
  $env:HF_TOKEN = "hf_..."
  # 2) convert the backbone to Lacuna's format → models\dino_weights.safetensors
  & E:\Program_Files\Conda\python.exe port\export_dino_weights.py
  ```

## Packaging a release for end users

```powershell
powershell -ExecutionPolicy Bypass -File scripts\package.ps1
```
Produces `dist\Lacuna-v<ver>.zip` = `lacuna.exe` + this `models\` folder (weights only,
no `.onnx`) + licenses. Users extract and double-click — that's the whole install.

## Optional: the ONNX Runtime fallback

The `.onnx` exports (`dino.onnx` + `dino.onnx.data`, `yolo.onnx`) are only needed if you
build with `--features ort-cuda`/`ort-backend` and set `LACUNA_USE_ORT=1`. The default
BURN build ignores them entirely.
