"""Fill every frame from sparse pose anchors. No clip-specific constants."""
from __future__ import annotations

import numpy as np

JOINTS = 17
JOINT_MIN = 0.25


def player_row(track_id, box, conf, joints, frame_index, fps, src):
    row = {
        "track_id": int(track_id),
        "bbox_xyxy": [round(float(v), 2) for v in box],
        "box_confidence": round(float(conf), 4),
        "joints": joints_or_none(joints),
        "timestamp_seconds": round(frame_index / fps, 4),
    }
    # Measured rows match the dense document and carry no src. A guess is never called measured.
    if src == "interpolated":
        row["src"] = "interpolated"
    elif src != "measured":
        raise ValueError(src)
    return row


def joints_or_none(joints, threshold=JOINT_MIN):
    out = []
    for item in joints:
        if item is None:
            out.append(None)
            continue
        x, y, c = item[:3]
        c = float(c)
        if c < threshold:
            out.append(None)
        else:
            out.append([round(float(x), 2), round(float(y), 2), round(c, 4)])
    return out


def _lerp_joints(a, b, t, threshold=JOINT_MIN):
    """Position moves. Confidence is the lower anchor. A missing anchor stays missing."""
    out = []
    for pa, pb in zip(a, b):
        if pa is None or pb is None or pa[2] < threshold or pb[2] < threshold:
            out.append(None)
            continue
        x = pa[0] + (pb[0] - pa[0]) * t
        y = pa[1] + (pb[1] - pa[1]) * t
        out.append([round(float(x), 2), round(float(y), 2), round(float(min(pa[2], pb[2])), 4)])
    return out


def _lerp_box(a, b, t):
    return [round(float(a[i] + (b[i] - a[i]) * t), 2) for i in range(4)]


def tracks_from(measured):
    """measured: frame_index -> list of (track_id, box, conf, joints_list)."""
    by_id = {}
    for fi, rows in measured.items():
        for tid, box, conf, joints in rows:
            by_id.setdefault(int(tid), []).append((int(fi), box, float(conf), joints))
    for seq in by_id.values():
        seq.sort(key=lambda row: row[0])
    return by_id


def fill_frames(measured, n, fps, max_gap):
    """One record per frame. Measured rows stay measured. Gaps longer than max_gap stay empty."""
    by_id = tracks_from(measured)
    measured_at = {int(fi): rows for fi, rows in measured.items()}
    frames = []
    for i in range(n):
        if i in measured_at:
            players = [
                player_row(tid, box, conf, joints, i, fps, "measured")
                for tid, box, conf, joints in measured_at[i]
            ]
        else:
            players = []
            for tid, seq in by_id.items():
                hit = _between(seq, i, max_gap)
                if hit is None:
                    continue
                left, right, t = hit
                players.append(player_row(
                    tid, _lerp_box(left[1], right[1], t), min(left[2], right[2]),
                    _lerp_joints(left[3], right[3], t), i, fps, "interpolated"))
        frames.append({
            "frame_index": i,
            "timestamp_seconds": round(i / fps, 4),
            "players": players,
        })
    return frames


def _between(seq, i, max_gap):
    # seq is sorted by frame index. Find the closest anchors strictly around i.
    right = next((k for k, row in enumerate(seq) if row[0] > i), None)
    if right is None or right == 0:
        return None
    left, right = seq[right - 1], seq[right]
    gap = right[0] - left[0]
    if gap <= 0 or gap > max_gap:
        return None
    return left, right, (i - left[0]) / gap


def stitch_ids(gop_rows):
    """Map (gop, local_id) to a global id by box overlap at each GOP boundary."""
    gid, nxt = {}, 0
    previous = []
    for g, rows_by_fi in enumerate(gop_rows):
        order = sorted(rows_by_fi)
        first = rows_by_fi[order[0]] if order else []
        taken = set()
        for tid, box, *_ in first:
            best = (0.0, None)
            for ptid, pbox, *_ in previous:
                if gid.get((g - 1, ptid)) in taken:
                    continue
                overlap = _iou(box, pbox)
                if overlap > best[0]:
                    best = (overlap, ptid)
            if best[0] >= 0.3:
                gid[(g, tid)] = gid[(g - 1, best[1])]
                taken.add(gid[(g, tid)])
        for rows in rows_by_fi.values():
            for tid, *_ in rows:
                if (g, tid) not in gid:
                    gid[(g, tid)] = nxt
                    nxt += 1
        previous = rows_by_fi[order[-1]] if order else []
    return gid


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (area + 1e-9)


def apply_ids(gop_rows, gid):
    merged = {}
    for g, rows_by_fi in enumerate(gop_rows):
        for fi, rows in rows_by_fi.items():
            merged[fi] = [
                (gid[(g, tid)], box, conf, joints) for tid, box, conf, joints in rows
            ]
    return merged
