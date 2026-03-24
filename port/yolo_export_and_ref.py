"""Phase 2b+2c: export FUSED YOLO26m-seg weights + a golden reference for the BURN
reimplementation.

  * load best.pt, model.fuse() (fold conv+bn, drop one2many + semseg heads) — matches
    the deployed ONNX (ultralytics fuses on export)
  * dump fused state_dict -> yolo_weights.safetensors (clean model.N.* names)
  * dump per-layer type/from blueprint + head config -> yolo_blueprint.json
  * on a FIXED [1,3,640,640] input: capture per-layer feature maps (key checkpoints)
    + proto, and run onnxruntime on models/yolo.onnx for output0/output1 oracle
  * save Rust-loadable refs -> yolo_ref.safetensors, and report onnx-vs-torch diff

Run:  E:/Program_Files/Conda/python.exe port/yolo_export_and_ref.py
"""
import os, json
os.environ.setdefault("YOLO_VERBOSE", "False")
import numpy as np
import torch
from ultralytics import YOLO

PT = r"E:\PhD_TobiMu\02_code\02paper\leaf_segmentation\runs\segment\runs\segment\leaf_seg\weights\best.pt"
ONNX = r"E:\PhD_TobiMu\02_code\FoliarToolbox\models\yolo.onnx"
OUT = r"E:\PhD_TobiMu\02_code\FoliarToolbox\port"

yolo = YOLO(PT)
net = yolo.model
net.fuse()
net.eval()
head = net.model[-1]
print("fused. head:", type(head).__name__, "nc", head.nc, "nm", head.nm,
      "reg_max", head.reg_max, "stride", head.stride.tolist(), "max_det", head.max_det)

# ── weights ──
sd = net.state_dict()
tensors = {k: v.detach().contiguous().float() for k, v in sd.items()}
from safetensors.torch import save_file
save_file(tensors, os.path.join(OUT, "yolo_weights.safetensors"))
print(f"saved {len(tensors)} fused tensors -> yolo_weights.safetensors "
      f"({sum(v.numel() for v in tensors.values())/1e6:.1f}M params)")

# ── blueprint + config ──
blueprint = [{"i": i, "from": getattr(l, "f", None), "type": type(l).__name__}
             for i, l in enumerate(net.model)]
cfg = {"nc": head.nc, "nm": head.nm, "reg_max": head.reg_max,
       "stride": [float(s) for s in head.stride], "max_det": head.max_det,
       "npr": head.npr, "imgsz": 640}
json.dump({"config": cfg, "blueprint": blueprint},
          open(os.path.join(OUT, "yolo_blueprint.json"), "w"), indent=1)

# Print every fused weight name+shape (the map the BURN loader must satisfy).
with open(os.path.join(OUT, "yolo_weight_names.txt"), "w") as f:
    for k in sorted(tensors):
        f.write(f"{k}\t{tuple(tensors[k].shape)}\n")
print("weight names -> yolo_weight_names.txt")

# ── deterministic input ──
rng = np.random.RandomState(0)
x = rng.rand(1, 3, 640, 640).astype(np.float32)  # [0,1]
xt = torch.from_numpy(x)

# capture key layer outputs (avoid the huge 160² maps to keep the ref small)
CHECK = {0, 4, 9, 10, 16, 19, 22}
caps = {}
hooks = []
for i in CHECK:
    def mk(i):
        def h(m, inp, out):
            caps[f"layer_{i}"] = out.detach().float().cpu().numpy() if torch.is_tensor(out) else None
        return h
    hooks.append(net.model[i].register_forward_hook(mk(i)))
def proto_hook(m, inp, out):
    caps["proto"] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu().numpy()
hooks.append(head.proto.register_forward_hook(proto_hook))

# Raw pre-decode head branch outputs per scale (deterministic — unlike topk'd out0).
raw = {"box": [None] * 3, "cls": [None] * 3, "msk": [None] * 3}
def mk_raw(kind, i):
    def h(m, inp, out):
        raw[kind][i] = out.detach().float().cpu().numpy()
    return h
for i in range(3):
    hooks.append(head.one2one_cv2[i].register_forward_hook(mk_raw("box", i)))
    hooks.append(head.one2one_cv3[i].register_forward_hook(mk_raw("cls", i)))
    hooks.append(head.one2one_cv4[i].register_forward_hook(mk_raw("msk", i)))

with torch.no_grad():
    out = net(xt)
for h in hooks:
    h.remove()

# unwrap ((det,proto), preds) → det[1,300,38]
def find_det(o):
    if torch.is_tensor(o):
        return o if (o.ndim == 3 and o.shape[-1] == 38) else None
    if isinstance(o, (list, tuple)):
        for e in o:
            r = find_det(e)
            if r is not None:
                return r
    return None
det_torch = find_det(out)
print("torch det:", None if det_torch is None else tuple(det_torch.shape),
      "| captured:", {k: v.shape for k, v in caps.items()})

# ── onnxruntime oracle ──
onnx_out0 = onnx_out1 = None
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    inm = sess.get_inputs()[0].name
    outs = sess.run(None, {inm: x})
    # identify by shape
    for o in outs:
        if o.ndim == 3 and o.shape[-1] == 38:
            onnx_out0 = o
        elif o.ndim == 4 and o.shape[1] == 32:
            onnx_out1 = o
    print("onnx out0:", None if onnx_out0 is None else onnx_out0.shape,
          "out1:", None if onnx_out1 is None else onnx_out1.shape)
    if onnx_out0 is not None and det_torch is not None:
        d = np.abs(onnx_out0 - det_torch.cpu().numpy())
        print(f"  onnx vs torch out0: max|Δ|={d.max():.3e} mean|Δ|={d.mean():.3e}")
    if onnx_out1 is not None and "proto" in caps:
        d = np.abs(onnx_out1 - caps["proto"])
        print(f"  onnx vs torch proto: max|Δ|={d.max():.3e} mean|Δ|={d.mean():.3e}")
except Exception as e:
    print(f"(onnxruntime unavailable: {e})")

# ── save Rust-loadable refs ──
from safetensors.numpy import save_file as save_np
ref = {"input": x}
for k, v in caps.items():
    ref[k] = np.ascontiguousarray(v, dtype=np.float32)
# Raw head branches concatenated over scales [P3(6400),P4(1600),P5(400)] → [1,C,8400].
def cat_scales(lst, C):
    return np.concatenate([np.asarray(a).reshape(1, C, -1) for a in lst], axis=2)
if all(v is not None for v in raw["box"]):
    ref["head_box"] = np.ascontiguousarray(cat_scales(raw["box"], 4), dtype=np.float32)
    ref["head_cls"] = np.ascontiguousarray(cat_scales(raw["cls"], 1), dtype=np.float32)
    ref["head_msk"] = np.ascontiguousarray(cat_scales(raw["msk"], 32), dtype=np.float32)
    print("raw head:", {k: ref[k].shape for k in ("head_box", "head_cls", "head_msk")})
if det_torch is not None:
    ref["out0_torch"] = np.ascontiguousarray(det_torch.cpu().numpy(), dtype=np.float32)
if onnx_out0 is not None:
    ref["out0_onnx"] = np.ascontiguousarray(onnx_out0, dtype=np.float32)
if onnx_out1 is not None:
    ref["out1_onnx"] = np.ascontiguousarray(onnx_out1, dtype=np.float32)
save_np(ref, os.path.join(OUT, "yolo_ref.safetensors"))
print("saved refs -> yolo_ref.safetensors:", {k: v.shape for k, v in ref.items()})
