"""Detector recall on the labeled players, per frame, plus latency.

usage: python detbench.py det1.onnx [det2.onnx ...]
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from court import find_court
from detect import Court, Detector

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "sample-videos" / "01-singles-rally.mp4"
REF = Path(__file__).resolve().parent / "reference" / "consensus.json"


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def main(models):
    ref = json.loads(REF.read_text("utf-8"))["frames"]
    cap = cv2.VideoCapture(str(VIDEO))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    court = Court(find_court(frames[0])[0], frames[0].shape[1], frames[0].shape[0])
    print("roi", court.roi)
    for m in models:
        det = Detector(m)
        det(frames[0], court.roi)
        times, hits, counts = [], {}, []
        for f, r in zip(frames, ref):
            t = time.perf_counter()
            boxes = det(f, court.roi)
            times.append(time.perf_counter() - t)
            boxes = [(b, s) for b, s in boxes if court.contains_feet(b, slack=0.1)]
            counts.append(sum(s >= 0.25 for _, s in boxes))
            for p in r["people"]:
                if not p["role"]:
                    continue
                best = max([(iou(b, p["bbox_xyxy"]), s) for b, s in boxes], default=(0, 0))
                for th in (0.1, 0.25):
                    key = f"{p['role']}@{th}"
                    hits.setdefault(key, []).append(best[0] >= 0.5 and best[1] >= th)
        print(json.dumps({"model": Path(m).name, "ms": round(1000 * float(np.median(times)), 1),
                          **{k: round(float(np.mean(v)), 4) for k, v in sorted(hits.items())},
                          "court_persons_per_frame@0.25": round(float(np.mean(counts)), 2)}))


if __name__ == "__main__":
    main(sys.argv[1:])
