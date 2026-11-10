"""Draw the reference and several runs' skeletons/boxes for one player in one frame.

usage: python look.py FRAME near|far OUT.png RUN.json [RUN.json ...]
white = consensus reference; runs get red, cyan, yellow, magenta in order.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from evaluate import REF, arr

COLORS = [(0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
VIDEO = Path(__file__).resolve().parent.parent / "sample-videos" / "01-singles-rally.mp4"


def draw(img, joints, color, r=4):
    for j in joints:
        if np.isfinite(j).all():
            cv2.circle(img, (int(j[0]), int(j[1])), r, color, -1, cv2.LINE_AA)


fi, role, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
cap = cv2.VideoCapture(str(VIDEO))
cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
img = cap.read()[1]
ref = [p for p in json.loads(REF.read_text("utf-8"))["frames"][fi]["people"] if p["role"] == role][0]
rj = arr(ref["joints"])
x0, y0, x1, y1 = ref["bbox_xyxy"]
h = y1 - y0
for path, color in zip(sys.argv[4:], COLORS):
    players = json.loads(Path(path).read_text("utf-8"))["frames"][fi]["players"]
    p = min(players, key=lambda p: np.nanmean(np.linalg.norm(arr(p["joints"]) - rj, axis=1)))
    b = [int(v) for v in p["bbox_xyxy"]]
    cv2.rectangle(img, b[:2], b[2:], color, 1)
    draw(img, arr(p["joints"]), color)
draw(img, rj, (255, 255, 255), 2)
m = int(0.6 * h)
crop = img[max(0, int(y0) - m):int(y1) + m, max(0, int(x0) - m):int(x1) + m]
scale = 700 / crop.shape[0]
cv2.imwrite(out, cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
