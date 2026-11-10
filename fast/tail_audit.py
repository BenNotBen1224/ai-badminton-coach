"""Untimed dev-only audit: compare tiny and accurate predictions on identical fresh crops.

Extra strong-model calls are diagnostic only and never affect emitted predictions.
Threshold tables describe direct crop errors, not an accepted online candidate.
"""
import argparse
import json
import os
import sys
import threading
from pathlib import Path

sys.modules.setdefault("openvino.tools.ovc", None)
import numpy as np
import pipeline
import evaluate
from acceptance import sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--reference", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    rows = []
    lock = threading.Lock()
    original = pipeline.Segment.pose_crops

    def measured(self, crops, meta, small):
        result = original(self, crops, meta, small)
        chosen = [i for i, flag in enumerate(small) if flag]
        if chosen:
            strong = original(self, [crops[i] for i in chosen], [meta[i] for i in chosen],
                              [False] * len(chosen))
            fresh = [{"frame": max(self.out), "segment": int(self.first),
                      "tiny": result[i].tolist(), "strong": s.tolist(),
                      "center": meta[i][0].tolist(), "scale": meta[i][1].tolist()}
                     for i, s in zip(chosen, strong)]
            with lock:
                rows.extend(fresh)
        return result

    pipeline.Segment.pose_crops = measured
    internal = pipeline.run(a.video, a.out / "unchanged-output.json")
    output = json.loads((a.out / "unchanged-output.json").read_text())
    refs = evaluate.load_reference(a.reference)
    # load_reference applies the evaluator's last-person-per-role duplicate policy.
    frames = {f["frame_index"]: f for f in output["frames"]}
    matched = []
    for row in rows:
        idx = row["frame"]
        players = frames[idx]["players"]
        tiny = np.array(row["tiny"])
        p = min(players, key=lambda p: np.linalg.norm(
            np.array(p["bbox_xyxy"][:2]) + np.array(p["bbox_xyxy"][2:]) - 2 * np.array(row["center"])),
                default=None)
        if p is None or not p.get("role"):
            continue
        ref = refs[idx].get(p["role"])
        if ref is None:
            continue
        row["role"] = p["role"]
        row["mean"] = float(tiny[:, 2].mean())
        row["min"] = float(tiny[:, 2].min())
        row["tail3"] = float(np.sort(tiny[:, 2])[:3].mean())
        row["wrist_min"] = float(tiny[[9, 10], 2].min())
        row["arm_min"] = float(tiny[[7, 8, 9, 10], 2].min())
        row["emitted_tiny"] = any(np.allclose(np.array(
            [j if j is not None else [np.nan] * 3 for j in q["joints"]]), tiny, equal_nan=True)
                                  for q in players)
        for name in ("tiny", "strong"):
            pred = np.array(row[name])
            valid = np.isfinite(ref["joints"]).all(1)
            hit = np.linalg.norm(pred[:, :2] - ref["joints"], axis=1) < .05 * ref["h"]
            row[name + "_hits"] = int((hit & valid).sum())
        row["gain"] = row["strong_hits"] - row["tiny_hits"]
        matched.append(row)
    (a.out / "crops.json").write_text(json.dumps(rows, indent=1))
    eligible = [r for r in matched if r["mean"] >= 1.05]
    tables = {}
    for feature in ("min", "tail3", "wrist_min", "arm_min"):
        ordered = sorted(eligible, key=lambda r: r[feature])
        tables[feature] = []
        for fraction in (.025, .05, .1, .2):
            selected = ordered[:max(1, round(len(ordered) * fraction))]
            tables[feature].append({
                "fraction": fraction, "threshold": selected[-1][feature], "extra_crops": len(selected),
                "direct_hit_gain": sum(r["gain"] for r in selected),
                "better_crops": sum(r["gain"] > 0 for r in selected),
                "worse_crops": sum(r["gain"] < 0 for r in selected)})
    (a.out / "summary.json").write_text(json.dumps({
        "warning": __doc__, "reference": str(a.reference.resolve()),
        "reference_sha256": sha256(a.reference), "video_sha256": sha256(a.video),
        "total_tiny_probes": len(rows), "reference_matched": len(matched),
        "eligible_mean_pass": len(eligible), "tables": tables,
        "internal_instrumented_not_timing": internal}, indent=1))
    print(json.dumps({"eligible": len(eligible), "tables": tables}), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
