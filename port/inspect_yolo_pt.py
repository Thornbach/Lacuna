"""Phase 2a: characterize the trained YOLO26m-seg from its ultralytics .pt — the
authoritative architecture for the BURN reimplementation (cleaner than ONNX node
tracing). Dumps: the model yaml (backbone+head as [from,repeats,module,args]), the
per-layer routing (index, from, type, #params, out-shape), the Segment/one2one head
internals, and total params.

Run:  E:/Program_Files/Conda/python.exe port/inspect_yolo_pt.py
"""
import os, json
os.environ.setdefault("YOLO_VERBOSE", "False")
import torch
from ultralytics import YOLO

PT = r"E:\PhD_TobiMu\02_code\02paper\leaf_segmentation\runs\segment\runs\segment\leaf_seg\weights\best.pt"

yolo = YOLO(PT)
net = yolo.model  # SegmentationModel
net.eval()
print("=== model class:", type(net).__name__)
print("=== task:", getattr(yolo, "task", "?"))
print("=== names:", net.names)
print("=== total params:", sum(p.numel() for p in net.parameters()))

print("\n=== YAML (architecture spec) ===")
print(json.dumps(net.yaml, indent=1, default=str))

# Per-layer routing + type. Trace out-shapes with a dummy forward hook.
print("\n=== LAYERS (i | from | n | type | args | #params) ===")
shapes = {}
def mk_hook(i):
    def h(m, inp, out):
        if isinstance(out, (list, tuple)):
            shapes[i] = [tuple(o.shape) if hasattr(o, "shape") else type(o).__name__ for o in out]
        else:
            shapes[i] = tuple(out.shape)
    return h
for i, layer in enumerate(net.model):
    layer.register_forward_hook(mk_hook(i))
with torch.no_grad():
    _ = net(torch.zeros(1, 3, 640, 640))
for i, layer in enumerate(net.model):
    np_ = sum(p.numel() for p in layer.parameters())
    f = getattr(layer, "f", "?")
    t = type(layer).__name__
    print(f"  [{i:2d}] f={str(f):14s} {t:16s} params={np_:>9d}  out={shapes.get(i)}")

# Head detail (Segment / one2one) — the trickiest module.
head = net.model[-1]
print(f"\n=== HEAD: {type(head).__name__} ===")
for attr in ["nc", "no", "reg_max", "nm", "npr", "stride", "end2end", "max_det",
             "dynamic", "shape", "ch"]:
    if hasattr(head, attr):
        v = getattr(head, attr)
        print(f"  {attr} = {v.tolist() if torch.is_tensor(v) else v}")
print("  submodules:")
for name, sub in head.named_children():
    npn = sum(p.numel() for p in sub.parameters())
    print(f"    {name:14s} {type(sub).__name__:14s} params={npn}")
