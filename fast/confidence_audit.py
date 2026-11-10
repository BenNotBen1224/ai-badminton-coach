"""Dev-only offline diagnostic, not a timed candidate or a substitute for pipeline acceptance.

Estimate whether intrinsic tiny-pose confidence could select a useful second opinion.
The substituted skeletons come from a different saved run; real online refinement must
be measured separately because its tracking boxes and future predictions can change.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tiny", type=Path)
    ap.add_argument("large", type=Path)
    ap.add_argument("judge", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    tiny, large, judge = [json.loads(p.read_text("utf-8"))["frames"] for p in (a.tiny, a.large, a.judge)]
    rows = []
    for tf, sf, jf in zip(tiny, large, judge, strict=True):
        for ref in jf["people"]:
            role = ref["role"]
            t = next((p for p in tf["players"] if p.get("role") == role), None)
            s = next((p for p in sf["players"] if p.get("role") == role), None)
            if t is None or s is None:
                continue
            valid = np.isfinite(evaluate.arr(ref["joints"])).all(1)
            h = ref["bbox_xyxy"][3] - ref["bbox_xyxy"][1]
            hits = []
            for p in (t, s):
                d = np.linalg.norm(evaluate.arr(p["joints"]) - evaluate.arr(ref["joints"]), axis=1)
                hits.append(int((d[valid] < 0.05 * h).sum()))
            conf = np.array([j[2] if j else 0 for j in t["joints"]])
            rows.append({"frame": tf["frame_index"], "role": role, "tiny_hits": hits[0],
                         "large_hits": hits[1], "joints": int(valid.sum()),
                         "min": float(conf.min()), "mean": float(conf.mean()),
                         "limbs_min": float(conf[7:].min()), "limbs_mean": float(conf[7:].mean()),
                         "low_count": -int((conf < 0.5).sum())})
    total = sum(r["joints"] for r in rows)
    base = sum(r["tiny_hits"] for r in rows)
    diagnostic = {}
    for feature in ("min", "mean", "limbs_min", "limbs_mean", "low_count"):
        ordered = sorted(rows, key=lambda r: r[feature])
        diagnostic[feature] = []
        for frac in (0.1, 0.2, 0.3, 0.4, 0.5):
            n = round(len(rows) * frac)
            chosen = ordered[:n]
            hits = base + sum(r["large_hits"] - r["tiny_hits"] for r in chosen)
            diagnostic[feature].append({"selected_fraction": n / len(rows), "threshold": chosen[-1][feature],
                                        "hypothetical_pck05": hits / total})
    a.out.write_text(json.dumps({
        "warning": __doc__, "player_frames": len(rows), "tiny_pck05": base / total,
        "large_pck05": sum(r["large_hits"] for r in rows) / total,
        "features": diagnostic,
    }, indent=1), "utf-8")
    print(json.dumps(diagnostic, indent=1))


if __name__ == "__main__":
    main()
