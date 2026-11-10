"""Contact sheet of who got which role: near = green, far = orange, no role = grey.

usage: python roles.py RUN.json VIDEO OUT.jpg [FRAME ...]   (default: 12 evenly spaced frames)
"""
import json
import sys

import cv2
import numpy as np

COLORS = {"near": (80, 220, 80), "far": (0, 160, 255), None: (160, 160, 160)}

run, video, out = sys.argv[1:4]
frames = json.load(open(run, encoding="utf-8"))["frames"]
picks = [int(a) for a in sys.argv[4:]] or np.linspace(0, len(frames) - 1, 12).astype(int).tolist()
cap = cv2.VideoCapture(video)
tiles = []
for fi in picks:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    img = cap.read()[1]
    for p in frames[fi].get("players", frames[fi].get("people", [])):
        color = COLORS[p.get("role")]
        x0, y0, x1, y1 = (int(v) for v in p["bbox_xyxy"])
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)
        cv2.putText(img, f"t{p.get('track_id', '')} {p.get('role') or '-'}", (x0, max(30, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
    cv2.putText(img, f"frame {fi}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4, cv2.LINE_AA)
    tiles.append(cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA))
while len(tiles) % 3:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
