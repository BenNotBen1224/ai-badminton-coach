"""Where do a run's crops go, and where does it miss the labeled players?

usage: python diagnose.py run.json
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from evaluate import arr, load_reference

reference = load_reference()
frames = json.loads(Path(sys.argv[1]).read_text("utf-8"))["frames"]
track_role = defaultdict(Counter)
track_feet = defaultdict(list)
misses = defaultdict(list)
low = defaultdict(list)
for f, roles in zip(frames, reference):
    cands = [(p, arr(p["joints"])) for p in f["players"]]
    taken = {}
    for role, ref in roles.items():
        best, best_n, best_d = None, -1, None
        for i, (p, c) in enumerate(cands):
            d = np.linalg.norm(c - ref["joints"], axis=1)
            n = int((d < 0.1 * ref["h"]).sum())
            if n > best_n:
                best, best_n, best_d = i, n, d
        valid = int(np.isfinite(ref["joints"]).all(1).sum())
        if best is None or best_n < 0.5 * valid:
            misses[role].append(f["frame_index"])
            continue
        taken[best] = role
        v = np.isfinite(ref["joints"]).all(1)
        pck05 = float((best_d[v] < 0.05 * ref["h"]).mean())
        if pck05 < 0.8:
            low[role].append((f["frame_index"], round(pck05, 2)))
    for i, (p, _) in enumerate(cands):
        track_role[p["track_id"]][taken.get(i, "extra")] += 1
        b = p["bbox_xyxy"]
        track_feet[p["track_id"]].append((round((b[0] + b[2]) / 2), round(b[3]), round(b[3] - b[1])))

print("track  frames  roles                      first_feet(x,y,h)")
for t in sorted(track_role):
    print(f"{t:5d}  {sum(track_role[t].values()):6d}  {dict(track_role[t])!s:26s} {track_feet[t][0]}")
for role in ("near", "far"):
    print(role, "misses", len(misses[role]), misses[role][:40])
    print(role, "low pck05<0.8", len(low[role]), low[role][:40])
