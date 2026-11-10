"""Does 'most confident in-court person per side of the net' pick the labeled player?

usage: python selectbench.py [detector.onnx ...]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from court import court_from_video
from detect import Court, Detector
from pipeline import WEIGHTS, iou

HERE = Path(__file__).resolve().parent
ref = json.loads((HERE / "reference" / "consensus.json").read_text("utf-8"))["frames"]
video = HERE.parent / "sample-videos" / "01-singles-rally.mp4"
dets = [Path(p) for p in sys.argv[1:]] or [WEIGHTS / "yolo26n-384x640.onnx", WEIGHTS / "yolo26s-448x768.onnx"]
court = Court(court_from_video(video)[0], 1920, 1080)
models = {d.stem: Detector(d, hint="LATENCY") for d in dets}
cap = cv2.VideoCapture(str(video))
stats = {k: {"near": [0, 0], "far": [0, 0]} for k in models}
wrong = {k: [] for k in models}
for idx in range(780):
    ok, frame = cap.read()
    if idx % 4:
        continue
    roles = {p["role"]: p["bbox_xyxy"] for p in ref[idx]["people"] if p["role"]}
    for name, det in models.items():
        boxes = [(b, s) for b, s in det(frame, court.roi, min_conf=0.25) if court.contains_feet(b, slack=0.1)]
        for role, rb in roles.items():
            side = [(s, b) for b, s in boxes if court.side(b) == role]
            stats[name][role][1] += 1
            if side and iou(max(side)[1], rb) >= 0.5:
                stats[name][role][0] += 1
            else:
                wrong[name].append((idx, role, [round(s, 2) for s, _ in sorted(side, reverse=True)]))
for name in models:
    print(name, {r: f"{a}/{b}" for r, (a, b) in stats[name].items()})
    print("  wrong:", wrong[name][:30])
