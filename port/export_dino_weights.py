"""Phase 1b: dump the HF DINOv3 ViT-B weights (clean semantic names) + config, so
the BURN reimplementation can load them by name. Same numbers the ONNX/tract use
(the ONNX was exported from this model), but with readable keys instead of val_*.

Run:  E:/Program_Files/Conda/python.exe port/export_dino_weights.py
"""
import sys, os, json
sys.path.insert(0, r"E:\PhD_TobiMu\02_code\02paper\anomaly\1Help")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from config import Config

cfg = Config()
print("model_id:", cfg.model_id, "| feature_layers:", cfg.feature_layers, "| patch:", cfg.patch_size)

from transformers import AutoModel
model = AutoModel.from_pretrained(cfg.model_id, attn_implementation="eager").eval()
c = model.config

# Config params the BURN module needs.
params = {}
for attr in ["hidden_size", "num_attention_heads", "num_hidden_layers", "patch_size",
             "num_register_tokens", "layerscale_value", "mlp_ratio", "rope_theta",
             "hidden_act", "image_size", "num_channels", "layer_norm_eps",
             "pos_embed_rope_base", "pos_embed_rope_min_period", "pos_embed_rope_max_period"]:
    params[attr] = getattr(c, attr, None)
params["feature_layers"] = list(cfg.feature_layers)
print("CONFIG:", json.dumps(params, indent=2, default=str))

sd = model.state_dict()
print(f"\nstate_dict: {len(sd)} tensors")
# Show the distinct per-block key patterns (layer.0.*) + embeddings.
for k in sd:
    if ".layer.0." in k or "embeddings" in k or "rope" in k.lower() or k.count(".") <= 2:
        print(f"  {k:60s} {tuple(sd[k].shape)}")

out_dir = r"E:\PhD_TobiMu\02_code\FoliarToolbox\port"
with open(os.path.join(out_dir, "dino_config.json"), "w") as f:
    json.dump(params, f, indent=2, default=str)

# Save weights: safetensors if available, else a flat .bin + json manifest.
tensors = {k: v.contiguous().to(torch.float32) for k, v in sd.items()}
try:
    from safetensors.torch import save_file
    save_file(tensors, os.path.join(out_dir, "dino_weights.safetensors"))
    print(f"\nsaved {len(tensors)} tensors -> dino_weights.safetensors")
except Exception as e:
    print(f"(safetensors unavailable: {e}) — writing flat .bin + manifest")
    manifest, offset = [], 0
    with open(os.path.join(out_dir, "dino_weights.bin"), "wb") as fb:
        for k, v in tensors.items():
            b = v.numpy().tobytes()
            manifest.append({"name": k, "shape": list(v.shape), "offset": offset, "bytes": len(b)})
            fb.write(b); offset += len(b)
    with open(os.path.join(out_dir, "dino_weights_manifest.json"), "w") as f:
        json.dump(manifest, f)
    print(f"saved {len(tensors)} tensors -> dino_weights.bin ({offset/1e6:.0f} MB)")
