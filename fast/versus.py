"""Head-to-head check of two keypoint runs on a clip with no ground truth.

Per role: the players A labeled itself (B's player = the person overlapping A's box);
for unlabeled runs, the lower feet of the two most confident people = near, and a lone
person is placed by NET_Y, the baseline's own far-court split.
Reports coverage, ID switches, and how often the two runs put each joint in the same place.
Writes a contact sheet of the frames where they disagree most (A = red, B = cyan)
so the disagreements can be judged by eye.

usage: python versus.py VIDEO A.json B.json SHEET.jpg [N]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

NET_Y = 600
LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


def arr(joints):
    return np.array([[np.nan, np.nan] if j is None else j[:2] for j in joints], float)


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def roles(frame, guide=None):
    """Use the run's own player labels; otherwise the person best overlapping the guide's
    labeled player; otherwise the two most confident people (lower feet = near)."""
    if any(p.get("role") for p in frame["players"]):
        return {p["role"]: p for p in frame["players"] if p.get("role")}
    if guide:
        out = {}
        for role, g in guide.items():
            score, best = max(((iou(p["bbox_xyxy"], g["bbox_xyxy"]), i) for i, p in enumerate(frame["players"])),
                              default=(0, None))
            if score >= 0.3:
                out[role] = frame["players"][best]
        return out
    people = sorted(frame["players"], key=lambda p: -p["box_confidence"])[:2]
    people.sort(key=lambda p: p["bbox_xyxy"][3])
    if len(people) == 2:
        return {"far": people[0], "near": people[1]}
    if people:
        return {"far" if people[0]["bbox_xyxy"][3] < NET_Y else "near": people[0]}
    return {}


def switches(ids):
    ids = [i for i in ids if i is not None]
    return sum(a != b for a, b in zip(ids, ids[1:]))


def compare(a_frames, b_frames):
    stats, rows = {}, []
    for role in ("near", "far"):
        s = {"frames": len(a_frames), "a_found": 0, "b_found": 0, "both": 0,
             "joints": 0, "same05": 0, "same10": 0, "a_only_joint": 0, "b_only_joint": 0}
        ida, idb = [], []
        for fi, (fa, fb) in enumerate(zip(a_frames, b_frames)):
            ra = roles(fa)
            pa, pb = ra.get(role), roles(fb, ra).get(role)
            ida.append(pa and pa["track_id"])
            idb.append(pb and pb["track_id"])
            s["a_found"] += pa is not None
            s["b_found"] += pb is not None
            if pa is None or pb is None:
                if pa is not None or pb is not None:
                    rows.append((99, fi, role, pa, pb))
                continue
            s["both"] += 1
            ja, jb = arr(pa["joints"]), arr(pb["joints"])
            h = (pa["bbox_xyxy"][3] - pa["bbox_xyxy"][1] + pb["bbox_xyxy"][3] - pb["bbox_xyxy"][1]) / 2
            va, vb = np.isfinite(ja).all(1), np.isfinite(jb).all(1)
            d = np.linalg.norm(ja - jb, axis=1)[va & vb]
            s["joints"] += len(d)
            s["same05"] += int((d < 0.05 * h).sum())
            s["same10"] += int((d < 0.1 * h).sum())
            s["a_only_joint"] += int((va & ~vb).sum())
            s["b_only_joint"] += int((vb & ~va).sum())
            bad = int((d >= 0.1 * h).sum())
            if bad:
                rows.append((bad, fi, role, pa, pb))
        s["pck05"] = round(s["same05"] / max(s["joints"], 1), 4)
        s["pck10"] = round(s["same10"] / max(s["joints"], 1), 4)
        s["a_switches"], s["b_switches"] = switches(ida), switches(idb)
        stats[role] = s
    return stats, rows


def draw(img, p, color):
    if p is None:
        return
    j = arr(p["joints"])
    for a, b in LIMBS:
        if np.isfinite(j[[a, b]]).all():
            cv2.line(img, tuple(map(int, j[a])), tuple(map(int, j[b])), color, 2, cv2.LINE_AA)
    for x, y in j[np.isfinite(j).all(1)]:
        cv2.circle(img, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)


def sheet(video, rows, out, n):
    picked, seen = [], set()
    for row in sorted(rows, key=lambda r: (-r[0], r[1])):
        key = (row[2], row[1] // 15)  # at most one tile per role per half second
        if key not in seen:
            seen.add(key)
            picked.append(row)
        if len(picked) == n:
            break
    cap = cv2.VideoCapture(str(video))
    tiles = []
    for bad, fi, role, pa, pb in sorted(picked, key=lambda r: r[1]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        img = cap.read()[1]
        draw(img, pa, (0, 0, 255))
        draw(img, pb, (255, 255, 0))
        boxes = [p["bbox_xyxy"] for p in (pa, pb) if p]
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        m = 0.35 * (y1 - y0)
        crop = img[max(0, int(y0 - m)):int(y1 + m), max(0, int(x0 - m)):int(x1 + m)]
        tile = cv2.resize(crop, (240, 320))
        label = f"{fi} {role} " + ("missing" if bad == 99 else f"{bad}j")
        cv2.putText(tile, label, (4, 20), 0, 0.6, (0, 0, 0), 3)
        cv2.putText(tile, label, (4, 20), 0, 0.6, (255, 255, 255), 1)
        tiles.append(tile)
    if not tiles:
        return
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))


if __name__ == "__main__":
    video, a, b, out = sys.argv[1:5]
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 18
    fa = json.loads(Path(a).read_text("utf-8"))["frames"]
    fb = json.loads(Path(b).read_text("utf-8"))["frames"]
    assert len(fa) == len(fb), (len(fa), len(fb))
    stats, rows = compare(fa, fb)
    sheet(video, rows, out, n)
    print(json.dumps({"A": a, "B": b, **stats, "disagreeing_frames": len(rows)}))
