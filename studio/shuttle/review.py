import sys
from pathlib import Path

import cv2
import numpy as np

import shots

name, step = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3
first = float(sys.argv[3]) if len(sys.argv) > 3 else None
last = float(sys.argv[4]) if len(sys.argv) > 4 else None
xy = shots.track(name)
frames, _ = shots.poses(name, len(xy))
start = shots.SPANS[name][0]
cap = cv2.VideoCapture(str(shots.RUNS / f"{name}.mp4"))
tiles = []
for i in range(0, len(xy), step):
    t = start + i / shots.FPS
    if (first is not None and t < first) or (last is not None and t > last):
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, img = cap.read()
    if not ok:
        break
    img = cv2.resize(img, (1920, 1080))
    for role, colour in (("near", (60, 60, 255)), ("far", (255, 160, 60))):
        p = frames[i].get(role)
        if p:
            x1, y1, x2, y2 = (int(v) for v in p["bbox_xyxy"])
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
    trail = [k for k in range(max(0, i - 9), i + 1) if not np.isnan(xy[k, 0])]
    for a, b in zip(trail, trail[1:]):
        cv2.line(img, tuple(int(v) for v in xy[a]), tuple(int(v) for v in xy[b]), (0, 255, 255), 3)
    if not np.isnan(xy[i, 0]):
        cv2.circle(img, tuple(int(v) for v in xy[i]), 12, (0, 0, 255), 3)
    tile = cv2.resize(img[300:1080, 280:1640], (476, 273))
    cv2.rectangle(tile, (0, 0), (110, 22), (0, 0, 0), -1)
    cv2.putText(tile, f"{t:.2f}", (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    tiles.append(tile)
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]
out = shots.RUNS / f"review-{name}-{first or start:.1f}.jpg"
cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(out, len(tiles))
