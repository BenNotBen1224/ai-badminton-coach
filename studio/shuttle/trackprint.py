import sys

import numpy as np

import shots

name = sys.argv[1]
step = int(sys.argv[2]) if len(sys.argv) > 2 else 2
first = float(sys.argv[3]) if len(sys.argv) > 3 else -1e9
last = float(sys.argv[4]) if len(sys.argv) > 4 else 1e9
xy = shots.track(name)
frames, homography = shots.poses(name, len(xy))
start = shots.SPANS[name][0]
print("visible", int((~np.isnan(xy[:, 0])).sum()), "/", len(xy))
for i in range(0, len(xy), step):
    t = start + i / shots.FPS
    if not first <= t <= last:
        continue
    ball = "      -     " if np.isnan(xy[i, 0]) else f"{xy[i, 0]:5.0f},{xy[i, 1]:4.0f}"
    cells = []
    for role in ("near", "far"):
        p = frames[i].get(role)
        if p:
            fx, fy = shots.feet(p, homography)
            cells.append(f"{role[0]}=({fx:4.1f},{fy:5.1f}) top={p['bbox_xyxy'][1]:4.0f}")
        else:
            cells.append(f"{role[0]}=  -")
    print(f"{t:7.2f} {ball}  " + "  ".join(cells))
