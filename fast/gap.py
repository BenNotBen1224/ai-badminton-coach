"""Where does run B lose PCK@0.05 to run A? Breaks the difference down by role, joint and time.

usage: python gap.py REF.json A.json B.json [--thr 0.1]
"""
import json
import sys
from pathlib import Path

import numpy as np

from evaluate import arr, load_reference

JOINTS = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sho", "r_sho", "l_elb", "r_elb", "l_wri", "r_wri",
          "l_hip", "r_hip", "l_knee", "r_knee", "l_ank", "r_ank"]


def hits(path, reference, thr):
    """(frame, role) -> bool[17] joint hits (valid reference joints only, NaN where invalid)."""
    frames = json.loads(Path(path).read_text("utf-8"))["frames"]
    out = {}
    for fi, (f, roles) in enumerate(zip(frames, reference)):
        cands = [arr(p["joints"]) for p in f["players"]]
        taken = set()
        for role, ref in sorted(roles.items(), key=lambda r: -r[1]["h"]):
            target = ref["joints"]
            best, best_key = None, (0, -np.inf)
            for i, c in enumerate(cands):
                if i in taken:
                    continue
                d = np.linalg.norm(c - target, axis=1)
                ok = np.isfinite(d)
                key = ((d[ok] < 0.1 * ref["h"]).sum(), -np.nanmean(d) if ok.any() else -np.inf)
                if key > best_key:
                    best, best_key = i, key
            valid = np.isfinite(target).all(1)
            matched = best is not None and best_key[0] >= 0.5 * valid.sum()
            if matched:
                taken.add(best)
            cand = cands[best] if matched else np.full((17, 2), np.nan)
            d = np.linalg.norm(cand - target, axis=1)
            h = np.where(valid, d < thr * ref["h"], np.nan)
            out[(fi, role)] = h
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    thr = float(sys.argv[sys.argv.index("--thr") + 1]) if "--thr" in sys.argv else 0.05
    args = [a for a in args if a != str(thr)]
    ref = load_reference(args[0])
    a, b = hits(args[1], ref, thr), hits(args[2], ref, thr)
    keys = sorted(a)
    for role in ("near", "far"):
        ka = [k for k in keys if k[1] == role]
        da = np.array([a[k] for k in ka])
        db = np.array([b[k] for k in ka])
        diff = np.nansum(db, 0) - np.nansum(da, 0)
        print(role, "frames", len(ka), "A", int(np.nansum(da)), "B", int(np.nansum(db)), "B-A", int(diff.sum()))
        print("   by joint", {JOINTS[j]: int(diff[j]) for j in np.argsort(diff)[:6]})
        per = np.nansum(db, 1) - np.nansum(da, 1)
        frames = np.array([k[0] for k in ka])
        bins = {}
        for f, v in zip(frames, per):
            bins[f // 60 * 60] = bins.get(f // 60 * 60, 0) + int(v)
        print("   by 60-frame bin", {k: v for k, v in bins.items() if v})


if __name__ == "__main__":
    main()
