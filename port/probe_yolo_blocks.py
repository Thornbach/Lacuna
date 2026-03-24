import os
os.environ["YOLO_VERBOSE"] = "False"
from ultralytics import YOLO

net = YOLO(r"E:\PhD_TobiMu\02_code\02paper\leaf_segmentation\runs\segment\runs\segment\leaf_seg\weights\best.pt").model
net.fuse(); net.eval()
for i in [2, 4, 6, 8, 10, 13, 16, 19, 22]:
    L = net.model[i]
    m = getattr(L, "m", None)
    els = []
    if m is not None:
        for e in m:
            t = type(e).__name__
            if t == "C3k":
                els.append(f"C3k(inner={len(e.m)})")
            elif t == "Sequential":
                els.append("Seq(" + ",".join(type(s).__name__ for s in e) + ")")
            else:
                els.append(t)
    print(f"[{i:2d}] {type(L).__name__:8s} c={getattr(L,'c','?')} m=[{', '.join(els)}]")
