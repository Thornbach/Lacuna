"""Phase 1c-pre: golden reference for validating the BURN DINOv3 reimplementation.

On a FIXED deterministic 512x512 input, dump:
  - the ONNX-baked mean/std + RoPE angle table (mul_2) from dino.onnx
  - the ONNX output [1,1024,1536]  (the deployed oracle, via onnxruntime CPU)
  - HF per-block hidden states: embeddings, after-block0, after-block6, after-block11
    (= hidden_states[0], [1], [7], [12]) for stage-by-stage debugging
All saved to port/dino_ref.npz so a Rust harness can assert parity per stage.

Run:  E:/Program_Files/Conda/python.exe port/dino_reference.py
"""
import sys, os, json
sys.path.insert(0, r"E:\PhD_TobiMu\02_code\02paper\anomaly\1Help")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import onnx
from onnx import numpy_helper
import torch
from config import Config

OUT = r"E:\PhD_TobiMu\02_code\FoliarToolbox\port"
ONNX_PATH = r"E:\PhD_TobiMu\02_code\FoliarToolbox\models\dino.onnx"
RES = 512

# --- pull the ONNX-baked constants (mean, std, RoPE angle table) ---
g = onnx.load(ONNX_PATH).graph
inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
mean = inits["mean"].reshape(-1)
std = inits["std"].reshape(-1)
mul_2 = inits["mul_2"]  # (1024, 2, 16) = 2*pi*coords*inv_freq  (pre flatten/tile)
print("mean:", mean, "std:", std, "| mul_2:", mul_2.shape, mul_2.dtype)

# --- deterministic input in [0,1] ---
rng = np.random.RandomState(0)
img01 = rng.rand(RES, RES, 3).astype(np.float32)  # HWC, [0,1]
x_chw = img01.transpose(2, 0, 1)[None]            # (1,3,512,512), raw [0,1] for ONNX

# --- ONNX oracle via onnxruntime (CPU) ---
onnx_out = None
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {iname: x_chw})[0]  # (1,1024,1536)
    print("ONNX out:", onnx_out.shape, "range", float(onnx_out.min()), float(onnx_out.max()),
          "| per-patch L2 of first 768:", float(np.linalg.norm(onnx_out[0, 0, :768])))
except Exception as e:
    print(f"(onnxruntime unavailable: {e}) — will rely on HF-derived reference")

# --- HF intermediates (normalized input) ---
cfg = Config()
from transformers import AutoModel
model = AutoModel.from_pretrained(cfg.model_id, attn_implementation="eager").eval()

x_norm = (torch.from_numpy(x_chw) - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
with torch.inference_mode():
    out = model(pixel_values=x_norm.float(), output_hidden_states=True)
hs = out.hidden_states  # tuple len 13
print("num hidden_states:", len(hs), "each", tuple(hs[0].shape))

# Replicate backbone._multilayer for (-1,-6): last 1024 tokens, L2-norm, concat [blk12, blk7]
def patches(h):
    return h[:, -1024:, :]
feat_hf = torch.cat([
    torch.nn.functional.normalize(patches(hs[-1].float()), dim=-1),
    torch.nn.functional.normalize(patches(hs[-6].float()), dim=-1),
], dim=-1).numpy()  # (1,1024,1536)
print("HF multilayer feat:", feat_hf.shape)

if onnx_out is not None:
    d = np.abs(onnx_out - feat_hf)
    print(f"ONNX vs HF multilayer: max|Δ|={d.max():.3e} mean|Δ|={d.mean():.3e}")

np.savez(os.path.join(OUT, "dino_ref.npz"),
         mean=mean, std=std, mul_2=mul_2,
         img01=img01, x_chw=x_chw.astype(np.float32),
         onnx_out=(onnx_out if onnx_out is not None else feat_hf).astype(np.float32),
         feat_hf=feat_hf.astype(np.float32),
         hs_embed=hs[0].float().numpy().astype(np.float32),   # input to block 0
         hs_block0=hs[1].float().numpy().astype(np.float32),  # after layer.0
         hs_block6=hs[7].float().numpy().astype(np.float32),  # after layer.6  (= -6)
         hs_block11=hs[12].float().numpy().astype(np.float32))# after layer.11 (= -1)
print("saved -> dino_ref.npz")

# Rust-loadable copy (safetensors) for the BURN validation harness.
from safetensors.numpy import save_file
save_file({
    "x_chw": np.ascontiguousarray(x_chw, dtype=np.float32),
    "onnx_out": np.ascontiguousarray(onnx_out if onnx_out is not None else feat_hf, dtype=np.float32),
    "hs_embed": np.ascontiguousarray(hs[0].float().numpy(), dtype=np.float32),
    "hs_block0": np.ascontiguousarray(hs[1].float().numpy(), dtype=np.float32),
    "hs_block6": np.ascontiguousarray(hs[7].float().numpy(), dtype=np.float32),
    "hs_block11": np.ascontiguousarray(hs[12].float().numpy(), dtype=np.float32),
}, os.path.join(OUT, "dino_ref.safetensors"))
print("saved -> dino_ref.safetensors")
