"""Light temporal smoothing of each tracked joint: weighted mean of the previous, current and
next frame of the same track. A neighbour is skipped when it is missing, low-confidence, or
far away (a real fast move or a detection glitch), so sharp moves are not blurred.

usage: python smooth.py in.json out.json [--w 0.25] [--max 0.08]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def smooth(doc, w=0.25, max_jump=0.08, min_conf=0.3):
    frames = doc["frames"]
    by_track = {}
    for fi, f in enumerate(frames):
        for p in f["players"]:
            by_track.setdefault(p["track_id"], {})[fi] = p

    def joints(p):
        return np.array([[np.nan, np.nan, 0.0] if j is None else j[:3] for j in p["joints"]], float)

    for rows in by_track.values():
        raw = {fi: joints(p) for fi, p in rows.items()}
        for fi, p in rows.items():
            k = raw[fi]
            b = p.get("bbox_xyxy")
            h = (b[3] - b[1]) if b else np.nanmax(k[:, 1]) - np.nanmin(k[:, 1])
            acc, wsum = k[:, :2].copy(), np.ones(17)
            for nf in (fi - 1, fi + 1):
                if nf not in raw:
                    continue
                n = raw[nf]
                ok = (n[:, 2] >= min_conf) & (k[:, 2] >= min_conf) & \
                     (np.linalg.norm(n[:, :2] - k[:, :2], axis=1) < max_jump * h)
                acc[ok] += w * n[ok, :2]
                wsum[ok] += w
            out = acc / wsum[:, None]
            p["joints"] = [None if j is None else [float(out[i, 0]), float(out[i, 1]), *j[2:]]
                           for i, j in enumerate(p["joints"])]
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--w", type=float, default=0.25)
    ap.add_argument("--max", type=float, default=0.08)
    a = ap.parse_args()
    doc = json.loads(Path(a.src).read_text("utf-8"))
    Path(a.dst).write_text(json.dumps(smooth(doc, a.w, a.max)), "utf-8")
