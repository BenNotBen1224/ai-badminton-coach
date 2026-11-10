"""Role-aware scoring. The old best-row metric is imported and reported, never gated here."""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from fast.evaluate import arr, load_reference, score as best_row_score

ROLES = ("near", "far")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def exclude_indexes(frame_hashes):
    """Frames the scorer must skip: consecutive repeats, phase-0 indexes, and their pixel copies."""
    n = len(frame_hashes)
    repeats = [i for i in range(1, n) if frame_hashes[i] == frame_hashes[i - 1]]
    phase0 = [i for i in range(n) if i % 3 == 0]
    seen = {frame_hashes[i] for i in phase0}
    copies = [i for i in range(n) if i not in set(phase0) and frame_hashes[i] in seen]
    exclude = sorted(set(repeats) | set(phase0) | set(copies))
    return {
        "frame_count": n,
        "repeat_frames": repeats,
        "phase0_frames": phase0,
        "phase0_pixel_duplicates": copies,
        "exclude": exclude,
        "repeat_share": (len(repeats) / n) if n else None,
    }


def repeat_mask(video, clip):
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    hashes = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hashes.append(hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest())
    cap.release()
    rec = exclude_indexes(hashes)
    rec.update(clip=clip, video=str(video), video_sha256=sha256(video))
    return rec


def _pack(hits, total):
    return {"hits": int(hits), "total": int(total), "rate": (hits / total) if total else None}


def _bucket():
    return {"pck05": [0, 0], "pck10": [0, 0], "found": [0, 0]}


def _reference_roles(reference):
    out = []
    for frame in reference.get("frames") or []:
        roles = {}
        for person in frame.get("people") or []:
            role = person.get("role")
            if role not in ROLES or role in roles:
                continue
            box = person.get("bbox_xyxy") or [0, 0, 0, 0]
            joints = person.get("joints")
            xy = arr(joints) if isinstance(joints, list) and len(joints) == 17 else np.full((17, 2), np.nan)
            roles[role] = {"h": float(box[3]) - float(box[1]), "joints": xy}
        out.append(roles)
    return out


def _prediction_roles(frame):
    roles = {}
    if not isinstance(frame, dict):
        return roles
    for person in frame.get("players") or []:
        role = person.get("role")
        if role not in ROLES or role in roles:
            continue
        joints = person.get("joints")
        xy = arr(joints) if isinstance(joints, list) and len(joints) == 17 else np.full((17, 2), np.nan)
        roles[role] = {"xy": xy, "src": person.get("src")}
    return roles


def _role_speed(here, there, i, j, fps):
    dt = abs(i - j) / fps
    if dt <= 0 or here["h"] <= 0:
        return 0.0
    dist = np.linalg.norm(here["joints"] - there["joints"], axis=1)
    ok = np.isfinite(dist)
    if not ok.any():
        return 0.0
    return float(np.max(dist[ok]) / here["h"] / dt)


def _speeds(ref_roles, fps):
    speeds = []
    n = len(ref_roles)
    for i, roles in enumerate(ref_roles):
        best = 0.0
        for role, here in roles.items():
            prev = next_i = None
            for j in range(i - 1, -1, -1):
                if role in ref_roles[j]:
                    prev = j
                    break
            for j in range(i + 1, n):
                if role in ref_roles[j]:
                    next_i = j
                    break
            for j in (prev, next_i):
                if j is not None:
                    best = max(best, _role_speed(here, ref_roles[j][role], i, j, fps))
        speeds.append(best)
    return speeds


def top_quarter(speeds, eligible):
    ranked = sorted(eligible, key=lambda i: (-speeds[i], i))
    k = math.ceil(len(eligible) / 4) if eligible else 0
    return ranked[:k]


def best_row(prediction, reference):
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pred = Path(tmp) / "pred.json"
            ref = Path(tmp) / "ref.json"
            pred.write_text(json.dumps(prediction), encoding="utf-8")
            ref.write_text(json.dumps(reference), encoding="utf-8")
            return best_row_score(pred, load_reference(ref))
    except Exception as err:
        return {"error": f"{type(err).__name__}: {err}"}


def score_pair(prediction, reference, *, exclude, fps):
    if not fps or fps <= 0:
        raise ValueError("fps must be positive")
    exclude = set(exclude or ())
    ref_roles = _reference_roles(reference)
    pred_frames = prediction.get("frames") or []
    speeds = _speeds(ref_roles, fps)
    eligible = [i for i, roles in enumerate(ref_roles) if i not in exclude and roles]
    fast = top_quarter(speeds, eligible)
    fast_set = set(fast)
    groups = ("all", "fast", "measured", "interpolated")
    raw = {name: _bucket() for name in groups}
    by_role = {role: {name: _bucket() for name in groups} for role in ROLES}
    for i, roles in enumerate(ref_roles):
        if i in exclude or not roles:
            continue
        pred = _prediction_roles(pred_frames[i]) if i < len(pred_frames) else {}
        for role, ref in roles.items():
            role_raw = by_role.get(role)
            valid = np.isfinite(ref["joints"]).all(axis=1)
            total = int(valid.sum())
            row = pred.get(role)
            raw["all"]["found"][1] += 1
            raw["all"]["found"][0] += int(row is not None)
            if role_raw is not None:
                role_raw["all"]["found"][1] += 1
                role_raw["all"]["found"][0] += int(row is not None)
            if i in fast_set:
                raw["fast"]["found"][1] += 1
                raw["fast"]["found"][0] += int(row is not None)
                if role_raw is not None:
                    role_raw["fast"]["found"][1] += 1
                    role_raw["fast"]["found"][0] += int(row is not None)
            if total == 0:
                continue
            if row is None:
                dist = np.full(17, np.inf)
                interpolated = False
            else:
                dist = np.linalg.norm(row["xy"] - ref["joints"], axis=1)
                interpolated = row.get("src") == "interpolated"
            hits = {}
            for alpha in (0.05, 0.1):
                hits[alpha] = int((dist[valid] < alpha * ref["h"]).sum()) if ref["h"] > 0 else 0
            for name in ("all",) + (("fast",) if i in fast_set else ()):
                for bucket in (raw, role_raw) if role_raw is not None else (raw,):
                    bucket[name]["pck05"][0] += hits[0.05]
                    bucket[name]["pck05"][1] += total
                    bucket[name]["pck10"][0] += hits[0.1]
                    bucket[name]["pck10"][1] += total
            split = "interpolated" if interpolated else "measured"
            for bucket in (raw, role_raw) if role_raw is not None else (raw,):
                bucket[split]["pck05"][0] += hits[0.05]
                bucket[split]["pck05"][1] += total
                bucket[split]["pck10"][0] += hits[0.1]
                bucket[split]["pck10"][1] += total
    packed = {}
    for name, bucket in raw.items():
        packed[name] = {key: _pack(*bucket[key]) for key in ("pck05", "pck10", "found")}
    packed_roles = {}
    for role, buckets in by_role.items():
        packed_roles[role] = {
            name: {key: _pack(*bucket[key]) for key in ("pck05", "pck10", "found")}
            for name, bucket in buckets.items()
        }
    return {
        "all": packed["all"],
        "fast": packed["fast"],
        "measured": packed["measured"],
        "interpolated": packed["interpolated"],
        "by_role": packed_roles,
        "fast_frames": fast,
        "eligible_frames": len(eligible),
        "best_row": best_row(prediction, reference),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5 or sys.argv[1] != "repeats":
        sys.exit("usage: python -m turbo.score repeats VIDEO OUT.json CLIP")
    rec = repeat_mask(sys.argv[2], sys.argv[4])
    Path(sys.argv[3]).write_text(json.dumps(rec), encoding="utf-8")
    print(json.dumps({"clip": rec["clip"], "frames": rec["frame_count"], "repeat_share": rec["repeat_share"]}))
