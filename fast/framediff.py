"""Where does run A lose joints that run B gets? (PCK@0.05 vs the consensus joints)

usage: python framediff.py A.json B.json [bin=50] [--agreed] [--pck10]
--agreed: only joints all three references place within 5% of body height of each other
--pck10: count a joint correct within 10% of body height instead of 5%
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate import REFERENCES, arr, load_reference

AGREED = "--agreed" in sys.argv
THR = 0.1 if "--pck10" in sys.argv else 0.05


def per_frame(path, reference):
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
            v = np.isfinite(target).all(1)
            if AGREED:
                r = [ref[k] for k in REFERENCES]
                with np.errstate(invalid="ignore"):
                    v = np.all([np.linalg.norm(r[i] - r[j], axis=1) < 0.05 * ref["h"]
                                for i in range(3) for j in range(i + 1, 3)], axis=0)
            hit = 0
            if best is not None and best_key[0] >= 0.5 * v.sum():
                taken.add(best)
                d = np.linalg.norm(cands[best] - target, axis=1)
                hit = int((d[v] < THR * ref["h"]).sum())
            out[(fi, role)] = hit
    return out


if __name__ == "__main__":
    reference = load_reference()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    a, b = per_frame(args[0], reference), per_frame(args[1], reference)
    size = int(args[2]) if len(args) > 2 else 50
    bins = defaultdict(int)
    for key in a:
        bins[(key[0] // size * size, key[1])] += a[key] - b[key]
    print("A - B correct joints, by", size, "frame bin:")
    for role in ("near", "far"):
        print(role, {k[0]: v for k, v in sorted(bins.items()) if k[1] == role and v})
    worst = sorted(((a[k] - b[k], k) for k in a if a[k] != b[k]))[:15]
    print("total", sum(a.values()) - sum(b.values()), "worst", worst)
