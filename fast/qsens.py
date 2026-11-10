"""Which single layer breaks INT8 RTMPose? Quantize one conv at a time, measure joint error.

Error = mean |joint shift| in crop pixels vs FP32, on real player boxes from clip 01.
usage: python qsens.py
"""
import json
import logging

import cv2
import nncf
import numpy as np
import openvino as ov

from pose import INPUT_H, INPUT_W, warp
from quantize import calibration_crops, to_input

nncf.set_log_level(logging.ERROR)
core = ov.Core()
frames = json.load(open("runs/shared-rtmpose-s.json"))["frames"]
cap = cv2.VideoCapture("../sample-videos/01-singles-rally.mp4")
tests = []
for i in range(0, 780, 65):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = cap.read()
    tests += [to_input(warp(f, p["bbox_xyxy"])[0]) for p in frames[i]["players"][:2]]
X = np.concatenate(tests)


def model(n=1):
    m = core.read_model("weights/rtmpose-s.onnx")
    m.reshape([n, 3, INPUT_H, INPUT_W])
    return m


def err(m):
    m.reshape([len(tests), 3, INPUT_H, INPUT_W])
    a, b = ref, core.compile_model(m, "CPU")(X)
    return float(((np.abs(a[0].argmax(-1) - b[0].argmax(-1)) + np.abs(a[1].argmax(-1) - b[1].argmax(-1))) / 2).mean())


ref = core.compile_model(model(len(tests)), "CPU")(X)
calib = [to_input(c) for c in calibration_crops(step=40)[:40]]
ops = [o for o in model().get_ordered_ops() if o.get_type_name() not in ("Parameter", "Result", "Constant")]
names = [o.get_friendly_name() for o in ops]
convs = [o.get_friendly_name() for o in ops if o.get_type_name() in ("Convolution", "GroupConvolution")]
print("test crops", len(tests), "convs", len(convs))
rows = []
for c in convs:
    q = nncf.quantize(model(), nncf.Dataset(calib), subset_size=len(calib),
                      ignored_scope=nncf.IgnoredScope(names=[n for n in names if n != c], validate=False))
    rows.append((err(q), c))
    print(f"{rows[-1][0]:7.2f}  {c}", flush=True)
print("worst", sorted(rows, reverse=True)[:8])
