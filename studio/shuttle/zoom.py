import sys

import cv2
import numpy as np

import shots

name, x1, y1, x2, y2 = sys.argv[1], *(int(v) for v in sys.argv[2:6])
times = [float(t) for t in sys.argv[6:]]
cap = cv2.VideoCapture(str(shots.RUNS / f"{name}.mp4"))
tiles = []
for t in times:
    cap.set(cv2.CAP_PROP_POS_FRAMES, round((t - shots.SPANS[name][0]) * shots.FPS))
    ok, img = cap.read()
    img = cv2.resize(img, (1920, 1080))[y1:y2, x1:x2]
    scale = 420 / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * scale), 420))
    cv2.rectangle(img, (0, 0), (90, 22), (0, 0, 0), -1)
    cv2.putText(img, f"{t:.2f}", (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    tiles.append(img)
out = shots.RUNS / f"zoom-{name}-{times[0]:.2f}.jpg"
cv2.imwrite(str(out), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(out)
