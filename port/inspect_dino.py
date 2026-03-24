"""Characterize dino.onnx (DINOv3 ViT-B/16 multi-layer extractor) for the BURN port.

Prints: opset, I/O, op-type histogram, param count, EVERY initializer name+shape
(reveals patch-embed, per-block LN/attn/MLP/LayerScale, pos-embed vs RoPE, register
tokens), and the node sequence for ONE transformer block so we can mirror the exact
math in BURN.

Run:  E:/Program_Files/Conda/python.exe port/inspect_dino.py
"""
import onnx
from onnx import shape_inference, numpy_helper
from collections import Counter

PATH = r"E:\PhD_TobiMu\02_code\FoliarToolbox\models\dino.onnx"

m = onnx.load(PATH)  # loads external .data automatically if alongside
try:
    m = shape_inference.infer_shapes(m)
except Exception as e:
    print(f"(shape inference skipped: {e})")
g = m.graph

def tshape(t):
    return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
            for d in t.type.tensor_type.shape.dim]

print("=== OPSET ===")
for op in m.opset_import:
    print(f"  domain='{op.domain}' version={op.version}")

print("\n=== INPUTS / OUTPUTS ===")
for i in g.input:
    print(f"  in  {i.name} {tshape(i)}")
for o in g.output:
    print(f"  out {o.name} {tshape(o)}")

print("\n=== OP-TYPE HISTOGRAM ===")
for op, c in Counter(n.op_type for n in g.node).most_common():
    print(f"  {op:24s} {c}")

inits = {init.name: numpy_helper.to_array(init).shape for init in g.initializer}
total = 0
for init in g.initializer:
    a = numpy_helper.to_array(init)
    total += a.size
print(f"\n=== PARAMS: {total:,} ({total*4/1e6:.0f} MB f32) ===")

print("\n=== ALL INITIALIZERS (name -> shape) ===")
for name, shp in inits.items():
    print(f"  {name:60s} {shp}")

# One block's node sequence (block 0) to see the exact op graph (LN, attn, RoPE?, MLP)
print("\n=== NODE SEQUENCE (first ~90 nodes = embed + block 0) ===")
for idx, n in enumerate(g.node[:90]):
    attrs = ""
    for a in n.attribute:
        if a.name in ("axis", "axes", "perm", "epsilon", "keepdims"):
            attrs += f" {a.name}={onnx.helper.get_attribute_value(a)}"
    print(f"  [{idx:3d}] {n.op_type:14s} {n.name:40s}{attrs}")
