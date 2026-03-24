"""Characterize yolo.onnx so we can reimplement it exactly in BURN.

Prints: opset, I/O, op-type histogram, the full node sequence (op + name +
in/out), and every initializer (weight) name + shape. The conv weight shapes
identify the YOLOv8 variant and the exact channel schedule.

Run:  E:/Program_Files/Conda/python.exe port/inspect_yolo.py
"""
import onnx
from onnx import shape_inference
from collections import Counter

PATH = r"E:\PhD_TobiMu\02_code\FoliarToolbox\models\yolo.onnx"

m = onnx.load(PATH)
try:
    m = shape_inference.infer_shapes(m)
except Exception as e:
    print(f"(shape inference skipped: {e})")
g = m.graph

def tshape(t):
    dims = []
    for d in t.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.HasField("dim_value") else (d.dim_param or "?"))
    return dims

print("=== OPSET ===")
for op in m.opset_import:
    print(f"  domain='{op.domain}' version={op.version}")

print("\n=== INPUTS ===")
for i in g.input:
    print(f"  {i.name}  {tshape(i)}")
print("=== OUTPUTS ===")
for o in g.output:
    print(f"  {o.name}  {tshape(o)}")

print("\n=== OP-TYPE HISTOGRAM ===")
hist = Counter(n.op_type for n in g.node)
for op, c in hist.most_common():
    print(f"  {op:20s} {c}")

# total params from initializers
import numpy as np
total = 0
inits = {}
for init in g.initializer:
    arr = onnx.numpy_helper.to_array(init)
    inits[init.name] = arr.shape
    total += arr.size
print(f"\n=== PARAMS: {total:,} ({total*4/1e6:.1f} MB f32) ===")

print("\n=== CONV WEIGHT SHAPES (identify variant/channels) ===")
for name, shp in inits.items():
    if name.endswith(".weight") and len(shp) == 4:
        print(f"  {name:45s} {shp}")

print("\n=== FULL NODE SEQUENCE ===")
for idx, n in enumerate(g.node):
    ins = ",".join(n.input)
    outs = ",".join(n.output)
    attrs = ""
    for a in n.attribute:
        if a.name in ("axis", "axes", "starts", "ends", "mode", "to", "perm"):
            attrs += f" {a.name}={onnx.helper.get_attribute_value(a)}"
    print(f"  [{idx:3d}] {n.op_type:12s} {n.name:34s} in[{ins}] -> out[{outs}]{attrs}")

print("\n=== ALL INITIALIZERS ===")
for name, shp in inits.items():
    print(f"  {name:50s} {shp}")
