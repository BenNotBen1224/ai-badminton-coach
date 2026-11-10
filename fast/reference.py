"""Build the pseudo-ground-truth used to score fast pipelines.

Reference = the two heaviest models already run on clip 01 (Sapiens 0.3B and
PoseWarper HRNet-W48). A joint is kept only where both agree; its position is
their mean. Boxes are linked into tracklets so the two players can be labeled
once by eye (identity labels, not keypoint labels).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "comparison" / "results"
VIDEO = ROOT / "sample-videos" / "01-singles-rally.mp4"
OUT = Path(__file__).resolve().parent / "reference"
AGREE = 0.1  # both references within 10% of box height
# Labeled by eye from reference/tracklets.jpg and reference/early.jpg.
ROLES = {3: "near", 0: "far", 13: "far", 17: "far", 26: "far", 43: "far"}


def joints(person):
    return np.array([[np.nan] * 3 if j is None else j for j in person["joints"]], float)


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def build():
    sap = json.loads((RESULTS / "sapiens" / "keypoints.json").read_text("utf-8"))["frames"]
    pw = json.loads((RESULTS / "posewarper" / "keypoints.json").read_text("utf-8"))["frames"]
    frames, agreed, total = [], 0, 0
    for fs, fp in zip(sap, pw):
        assert fs["frame_index"] == fp["frame_index"]
        by_box = {tuple(p["bbox_xyxy"]): p for p in fp["players"]}
        people = []
        for ps in fs["players"]:
            pp = by_box[tuple(ps["bbox_xyxy"])]
            a, b = joints(ps)[:, :2], joints(pp)[:, :2]
            box = ps["bbox_xyxy"]
            h = box[3] - box[1]
            d = np.linalg.norm(a - b, axis=1)
            ok = np.isfinite(d) & (d < AGREE * h)
            total += int(np.isfinite(d).sum())
            agreed += int(ok.sum())
            mean = np.where(ok[:, None], (a + b) / 2, np.nan)
            xy = lambda arr: [None if not np.isfinite(p).all() else [round(float(p[0]), 2), round(float(p[1]), 2)]
                              for p in arr]
            people.append({"bbox_xyxy": box, "box_confidence": ps["box_confidence"],
                           "joints": xy(mean), "sapiens": xy(a), "posewarper": xy(b)})
        frames.append({"frame_index": fs["frame_index"], "people": people})
    link(frames)
    return frames, agreed / total


def link(frames):
    """Greedy IoU linking of consecutive boxes into tracklets."""
    next_id, prev = 0, []
    for f in frames:
        used = set()
        for p in sorted(f["people"], key=lambda p: -p["box_confidence"]):
            best, score = None, 0.3
            for q in prev:
                s = iou(p["bbox_xyxy"], q["bbox_xyxy"])
                if q["tracklet"] not in used and s > score:
                    best, score = q, s
            if best is None:
                p["tracklet"] = next_id
                next_id += 1
            else:
                p["tracklet"] = best["tracklet"]
            used.add(p["tracklet"])
            p["role"] = ROLES.get(p["tracklet"])
        prev = f["people"]


def sheet(frames):
    spans = {}
    for f in frames:
        for p in f["people"]:
            spans.setdefault(p["tracklet"], []).append((f["frame_index"], p["bbox_xyxy"]))
    keep = sorted((t for t, s in spans.items() if len(s) >= 5), key=lambda t: spans[t][0][0])
    cap = cv2.VideoCapture(str(VIDEO))
    tiles = []
    for t in keep:
        fi, box = spans[t][len(spans[t]) // 2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        _, img = cap.read()
        x0, y0, x1, y1 = map(int, box)
        pad = 40
        crop = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
        crop = cv2.resize(crop, (160, 240))
        cv2.putText(crop, f"T{t}", (4, 24), 0, 0.8, (0, 0, 255), 2)
        cv2.putText(crop, f"{spans[t][0][0]}-{spans[t][-1][0]}", (4, 232), 0, 0.55, (255, 255, 255), 2)
        tiles.append(crop)
    while len(tiles) % 10:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]
    cv2.imwrite(str(OUT / "tracklets.jpg"), np.vstack(rows))
    return {t: [spans[t][0][0], spans[t][-1][0], len(spans[t])] for t in keep}


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    frames, rate = build()
    (OUT / "consensus.json").write_text(json.dumps({"agreement_threshold": AGREE, "frames": frames}), "utf-8")
    spans = sheet(frames)
    print(f"joints where Sapiens and PoseWarper agree: {rate:.1%}")
    print(json.dumps(spans))
    sys.exit(0)
