"""Acceptance harness for the under-one-second preview.

The clock is perf_counter_ns around Runner.run only. Runner timings are stored
and never gated. Final holds cpu.lock from runner construction through scoring,
refuses a lock it does not own, a replay runner, and a 2-second background CPU
average above 5% before any sealed reference is opened.

dev:
  python -m turbo.accept dev --runner replay=turbo.replay:ReplayRunner --models-dir fast\\weights --runs 1 --out turbo\\runs\\dev-replay

final pose (S-A, not the headline):
  python -m turbo.accept final --runner preview=turbo.service:PoseRunner --models-dir fast\\weights --out turbo\\runs\\final-pose

final end to end (the only headline):
  python -m turbo.accept final --runner coach=coach.app.pipeline:E2ERunner --models-dir fast\\weights --out turbo\\runs\\final-e2e
"""
from __future__ import annotations

import argparse
import array
import base64
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from coach.schema import SCHEMA_VERSION, report_from_dict
from turbo.fresh import evidence, fresh_copy
from turbo.replay import dense_path
from turbo.runner import load_class
from turbo.score import score_pair

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
LOCK = PROJECT / "work" / "cpu.lock"
FFMPEG = PROJECT / "tools" / "ffmpeg.exe"
REFS = HERE / "refs"
SEALED = HERE / "sealed"
LIMIT_NS = 1_000_000_000
POSE_SCHEMA_VERSION = 3
REPORT_CURRENT_VERSION = 3
REPORT_LEGACY_VERSION = 2
FINAL_RUNS = 5
CPU_LIMIT = 5.0
CPU_SAMPLE_S = 2.0
DEV_CLIPS = ("01", "02", "03", "04", "05", "08")
GATING = ("01", "02", "03", "04", "05", "08", "09", "10")
EDGE = "11"
WARMUP = "07"
OWNED = ("runner.py", "fresh.py", "replay.py", "score.py", "judge_phase.py", "accept.py")
MARGINS = {"pck10": (3, 100), "fast_pck10": (6, 100), "found": (2, 100)}
REFINED_PCK05 = (2, 1000)


class Refusal(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def claim_for(cls):
    qual = f"{cls.__module__}:{cls.__name__}"
    if qual == "coach.app.pipeline:E2ERunner":
        return "S-A2"
    if qual == "turbo.service:PoseRunner":
        return "S-A"
    if getattr(cls, "IS_REPLAY", False):
        return "replay"
    return "unknown"


def schedule(clips, repeats):
    order = []
    n = len(clips)
    for repeat in range(repeats):
        start = repeat % n
        order.extend(clips[(start + i) % n] for i in range(n))
    return order


def reference_path(mode, clip):
    if mode == "dev":
        return REFS / f"validation-{clip}.json"
    if mode == "final":
        return SEALED / f"phase2-{clip}.json"
    raise ValueError(mode)


def refuse_final(runner_cls, runner, lock_path, cpu_percent, owned_token=None):
    if getattr(runner_cls, "IS_REPLAY", False) or getattr(runner, "IS_REPLAY", False):
        raise Refusal("final refuses a runner with IS_REPLAY")
    path = Path(lock_path)
    if owned_token is None:
        if path.exists():
            raise Refusal("cpu.lock exists")
    else:
        try:
            held = path.read_text(encoding="utf-8")
        except OSError:
            held = None
        if held != owned_token:
            raise Refusal("cpu.lock exists")
    if cpu_percent > CPU_LIMIT:
        raise Refusal(f"background CPU {cpu_percent} is above {CPU_LIMIT}%")


def new_lock_token():
    return f"H {os.getpid()} {uuid.uuid4().hex} accept-final"


def _drop_owned_lock(path, token):
    """Remove path only when its text is exactly token. Leave partial, foreign, or unreadable locks."""
    try:
        held = path.read_text(encoding="utf-8")
    except OSError:
        return
    if held != token:
        return
    try:
        path.unlink()
    except OSError:
        return


def acquire_cpu_lock(path, token):
    path = Path(path)
    try:
        handle = open(path, "x", encoding="utf-8")
    except FileExistsError:
        raise Refusal("cpu.lock exists") from None
    try:
        handle.write(token)
        handle.flush()
    except Exception:
        try:
            handle.close()
        except Exception:
            pass
        _drop_owned_lock(path, token)
        raise
    handle.close()


def release_cpu_lock(path, token):
    path = Path(path)
    try:
        held = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    if held != token:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return


@contextmanager
def cpu_lock_held(path, cpu_percent, before_acquire=None):
    path = Path(path)
    if path.exists():
        raise Refusal("cpu.lock exists")
    if cpu_percent > CPU_LIMIT:
        raise Refusal(f"background CPU {cpu_percent} is above {CPU_LIMIT}%")
    if before_acquire is not None:
        before_acquire()
    token = new_lock_token()
    acquire_cpu_lock(path, token)
    try:
        yield token
    finally:
        release_cpu_lock(path, token)


def both_role_frames(doc):
    count = 0
    for frame in doc.get("frames") or []:
        if not isinstance(frame, dict):
            continue
        roles = {p.get("role") for p in frame.get("players") or [] if isinstance(p, dict)}
        if "near" in roles and "far" in roles:
            count += 1
    return count


def validate_pose(doc, frame_count, fps, size, baseline_both):
    problems = []
    if not isinstance(doc, dict):
        return ["pose document is not an object"]
    if doc.get("schema_version") != POSE_SCHEMA_VERSION:
        problems.append(f"pose schema_version {doc.get('schema_version')} != {POSE_SCHEMA_VERSION}")
    if doc.get("tier") not in ("preview", "refined"):
        problems.append(f"tier {doc.get('tier')!r} is not preview or refined")
    src = doc.get("source") if isinstance(doc.get("source"), dict) else {}
    if not isinstance(doc.get("source"), dict):
        problems.append("source is not an object")
    if src.get("frame_count") != frame_count:
        problems.append(f"header frame_count {src.get('frame_count')} != {frame_count}")
    if size and (src.get("width"), src.get("height")) != tuple(size):
        problems.append(f"source size does not match {tuple(size)}")
    if not _finite(src.get("fps")) or abs(src["fps"] - fps) > 0.01:
        problems.append(f"header fps {src.get('fps')} != {fps}")
    frames = doc.get("frames")
    if not isinstance(frames, list):
        return problems + ["frames is not a list"]
    if len(frames) != frame_count:
        problems.append(f"{len(frames)} frames saved, expected {frame_count}")
    both = 0
    shown = 0
    for pos, frame in enumerate(frames):
        errs = []
        if not isinstance(frame, dict):
            errs.append("frame is not an object")
        else:
            if type(frame.get("frame_index")) is not int or frame["frame_index"] != pos:
                errs.append(f"frame_index {frame.get('frame_index')} at position {pos}")
            players = frame.get("players")
            if not isinstance(players, list):
                errs.append("players is not a list")
                players = []
            roles = set()
            for player in players:
                if not isinstance(player, dict):
                    errs.append("player is not an object")
                    continue
                if player.get("role") in ("near", "far"):
                    roles.add(player["role"])
                if "src" in player and player.get("src") != "interpolated":
                    errs.append(f"src {player.get('src')!r} is not interpolated")
                joints = player.get("joints")
                if not (isinstance(joints, list) and len(joints) == 17):
                    errs.append("joints are not 17 entries")
                    continue
                for joint in joints:
                    if joint is None:
                        continue
                    if not (isinstance(joint, list) and len(joint) == 3 and all(_finite(v) for v in joint)):
                        errs.append("non-finite or malformed joint")
                        break
            if roles == {"near", "far"}:
                both += 1
        if errs:
            shown += 1
            if shown <= 8:
                problems.append(f"frame {pos}: {'; '.join(errs)}")
    if shown > 8:
        problems.append(f"{shown} frames with problems")
    if not isinstance(baseline_both, int) or baseline_both < 0:
        problems.append("baseline both-role count is missing")
    elif both * 100 < baseline_both * 98:
        problems.append(f"both roles in {both} frames, baseline {baseline_both}, need at least 98%")
    return problems


def _inside(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    return path == root or root in path.parents


# Published by C2. There is no pose file. The bundle dict is the pose.
BUNDLE_CONTENT_BOUND = True
BUNDLE_KEYS = (
    "schema_version", "n_frames", "fps", "tier", "players", "source", "style",
    "timestamps", "findings", "poses", "series", "events", "windows", "warnings",
    "video_url", "court", "joints",
)
JOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
_XY_UNKNOWN = -32768
_FINDING_EXTRA = ("timelines", "exercise_ids")
_SEGMENT_KEYS = ("kind", "start_frame", "end_frame", "speed", "hold_s", "zoom", "show")


def _finding_identity(doc):
    found = []
    for item in doc.get("findings") or []:
        if not isinstance(item, dict):
            return None
        found.append((item.get("id"), item.get("player"), item.get("severity")))
    return sorted(found)


def _real(value):
    return _finite(value) and not isinstance(value, bool)


def _close(left, right):
    return _real(left) and _real(right) and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)


def _q_xy(value):
    if not _real(value):
        return None
    tenths = round(float(value) * 10.0)
    if tenths < -32767 or tenths > 32767:
        return "overflow"
    return tenths


def _raw_component(value):
    if not _real(value):
        return None
    return float(value)


def _raw_joints(player):
    """Unrounded x, y, and confidence. None marks a missing component."""
    missing = (None, None, None)
    joints = player.get("joints") if isinstance(player, dict) else None
    if not isinstance(joints, list) or len(joints) != 17:
        return (missing,) * 17
    rows = []
    for item in joints:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            rows.append(missing)
            continue
        rows.append((_raw_component(item[0]), _raw_component(item[1]), _raw_component(item[2])))
    return tuple(rows)


def _q_conf(value):
    if not _real(value) or float(value) < 0.0:
        return 0
    scaled = round(float(value) * 100.0)
    if float(value) < 0.5:
        return min(scaled, 49)
    return min(max(scaled, 50), 255)


def _decode_packed(packed, shape, dtype, scale, unknown):
    if not isinstance(packed, dict):
        return None, "is not an object"
    if packed.get("endian") != "little" or packed.get("dtype") != dtype or packed.get("shape") != list(shape):
        return None, "has the wrong dtype, shape, or endian"
    if scale is None and ("scale" in packed or "unknown" in packed):
        return None, "must not carry a scale or unknown"
    if scale is not None and (packed.get("scale") != scale or packed.get("unknown") != unknown):
        return None, "has the wrong scale or unknown"
    try:
        raw = base64.b64decode(packed.get("b64"), validate=True)
    except Exception:
        return None, "has bytes that are not base64"
    count = 1
    for dim in shape:
        count *= dim
    width = 1 if dtype == "uint8" else 2
    if len(raw) != count * width:
        return None, "has the wrong number of bytes"
    if dtype == "uint8":
        return list(raw), None
    values = array.array("h")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return list(values), None


def _frame_index(frame, pos, n):
    idx = frame.get("frame_index", pos) if isinstance(frame, dict) else None
    if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0 or idx >= n:
        return None
    return idx


def _best_player(frame, role):
    chosen = None
    score = None
    for player in (frame.get("players") or []) if isinstance(frame, dict) else []:
        if not isinstance(player, dict) or player.get("role") != role:
            continue
        raw = player.get("box_confidence")
        value = float(raw) if _real(raw) else 0.0
        if chosen is None or value > score:
            chosen, score = player, value
    return chosen


def _shot_starts(doc, n):
    starts = set()
    for cut in doc.get("cuts") or []:
        if isinstance(cut, bool) or not isinstance(cut, (int, float)) or isinstance(cut, bool):
            continue
        number = float(cut)
        if not math.isfinite(number) or not number.is_integer():
            continue
        cut = int(number)
        if 0 < cut < n:
            starts.add(cut)
    return starts


def _expected_pose(doc, n, players):
    slots = {
        role: {
            "xy": [None] * (n * 17 * 2),
            "conf": [0] * (n * 17),
            "bbox": [_XY_UNKNOWN] * (n * 4),
            "present": [False] * n,
            "interpolated": [False] * n,
            "raw": [None] * n,
        }
        for role in players
    }
    frames = doc.get("frames") or []
    for pos, frame in enumerate(frames):
        idx = _frame_index(frame, pos, n)
        if idx is None:
            continue
        for role in players:
            player = _best_player(frame, role)
            if player is None:
                continue
            slot = slots[role]
            slot["present"][idx] = True
            slot["interpolated"][idx] = player.get("src") == "interpolated"
            slot["raw"][idx] = _raw_joints(player)
            joints = player.get("joints")
            if isinstance(joints, list) and len(joints) == 17:
                for joint, item in enumerate(joints):
                    if not isinstance(item, (list, tuple)) or len(item) < 3:
                        continue
                    x, y = _q_xy(item[0]), _q_xy(item[1])
                    if x == "overflow" or y == "overflow":
                        slot["xy"][(idx * 17 + joint) * 2] = "overflow"
                        slot["xy"][(idx * 17 + joint) * 2 + 1] = "overflow"
                        continue
                    if x is None or y is None:
                        continue
                    slot["xy"][(idx * 17 + joint) * 2] = x
                    slot["xy"][(idx * 17 + joint) * 2 + 1] = y
                    slot["conf"][idx * 17 + joint] = _q_conf(item[2])
            box = player.get("bbox_xyxy")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                quantized = [_q_xy(value) for value in box]
                if any(item == "overflow" for item in quantized):
                    slot["bbox"][idx * 4:idx * 4 + 4] = ["overflow"] * 4
                elif all(item is not None for item in quantized):
                    slot["bbox"][idx * 4:idx * 4 + 4] = quantized
    starts = _shot_starts(doc, n)
    measured = {}
    for role, slot in slots.items():
        flags = []
        for idx in range(n):
            repeat = False
            if idx > 0 and slot["present"][idx] and slot["present"][idx - 1] and idx not in starts:
                # Raw joint equality, including missingness. Packed values can hide a real change.
                repeat = slot["raw"][idx] == slot["raw"][idx - 1]
            flags.append(int(slot["present"][idx] and not slot["interpolated"][idx] and not repeat))
        measured[role] = flags
    return slots, measured


def _published_style():
    path = PROJECT / "coach" / "style.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _bind_bundle(bundle, doc, report, frame_count, fps):
    problems = []
    extra = [key for key in bundle if key not in BUNDLE_KEYS]
    if extra:
        problems.append("bundle has unexpected keys " + ", ".join(sorted(extra)))
    missing = [key for key in BUNDLE_KEYS if key not in bundle]
    if missing:
        problems.append("bundle missing " + ", ".join(missing))
        return problems
    if not isinstance(bundle.get("schema_version"), int) or isinstance(bundle.get("schema_version"), bool):
        problems.append("bundle schema_version must be an int")
    if bundle.get("n_frames") != frame_count or not isinstance(bundle.get("n_frames"), int) or isinstance(bundle.get("n_frames"), bool):
        problems.append(f"bundle n_frames {bundle.get('n_frames')} != {frame_count}")
    bundle_fps = bundle.get("fps")
    report_fps = report.get("fps") if isinstance(report, dict) else None
    if not isinstance(bundle_fps, float) or not math.isfinite(bundle_fps) or bundle_fps <= 0:
        problems.append("bundle fps is not a positive finite number")
    elif report_fps is not None and not _close(bundle_fps, report_fps):
        problems.append(f"bundle fps {bundle_fps} != report fps {report_fps}")
    elif fps is not None and not _close(bundle_fps, fps):
        problems.append(f"bundle fps {bundle_fps} != clip fps {fps}")
    if not isinstance(bundle.get("tier"), str) or (isinstance(report, dict) and bundle.get("tier") != report.get("tier")):
        problems.append("bundle tier does not match the report")
    players = report.get("players") if isinstance(report, dict) else None
    if not isinstance(bundle.get("players"), list) or not all(isinstance(role, str) for role in bundle.get("players") or []):
        problems.append("bundle players must be a list of roles")
        players = []
    elif players is not None and bundle.get("players") != list(players):
        problems.append("bundle players do not match the report")
    else:
        players = list(bundle.get("players") or [])
    if not isinstance(bundle.get("source"), dict):
        problems.append("bundle source must be an object")
    elif isinstance(report, dict):
        expected = dict(report.get("source") or {})
        doc_source = doc.get("source") or {}
        for key in ("width", "height"):
            if expected.get(key) is None and key in doc_source:
                expected[key] = doc_source[key]
        if bundle.get("source") != expected:
            problems.append("bundle source does not match the report")
    try:
        if bundle.get("style") != _published_style():
            problems.append("bundle style does not match the published style")
    except Exception as err:
        problems.append(f"published style could not be read: {err}")
    stamps = bundle.get("timestamps")
    report_stamps = report.get("timestamps") if isinstance(report, dict) else None
    if report_stamps:
        expected_stamps = list(report_stamps)
    else:
        rate = float(bundle_fps) if isinstance(bundle_fps, float) and bundle_fps > 0 else None
        expected_stamps = [i / rate for i in range(frame_count)] if rate else None
    if not isinstance(stamps, list) or len(stamps) != frame_count:
        problems.append("bundle timestamps must cover every frame")
    elif any(not _real(item) or float(item) < 0 for item in stamps):
        problems.append("bundle timestamps must be finite and non-negative")
    elif any(float(b) < float(a) for a, b in zip(stamps, stamps[1:])):
        problems.append("bundle timestamps go backwards")
    elif expected_stamps is not None and any(not _close(got, want) for got, want in zip(stamps, expected_stamps)):
        problems.append("bundle timestamps do not match the report")
    if bundle.get("joints") != list(JOINTS):
        problems.append("bundle joints are not the published 17 names")
    if not isinstance(bundle.get("video_url"), str):
        problems.append("bundle video_url must be a string")
    _bind_findings(bundle, report, frame_count, problems)
    _bind_lists(bundle, report, problems)
    _bind_court(bundle, doc, problems)
    _bind_poses(bundle, doc, frame_count, players, problems)
    return problems


def _bind_findings(bundle, report, frame_count, problems):
    found = bundle.get("findings")
    report_findings = report.get("findings") if isinstance(report, dict) else None
    if not isinstance(found, list) or not isinstance(report_findings, list):
        problems.append("bundle findings do not match the report")
        return
    if len(found) != len(report_findings):
        problems.append("bundle findings do not match the report")
        return
    for index, (item, plain) in enumerate(zip(found, report_findings)):
        if not isinstance(item, dict) or not isinstance(plain, dict):
            problems.append("bundle findings do not match the report")
            return
        extra = [key for key in item if key not in plain and key not in _FINDING_EXTRA]
        missing = [key for key in plain if key not in item]
        body = {key: item.get(key) for key in plain}
        if extra or missing or body != plain:
            problems.append(f"bundle finding {index} text does not match the report")
        timelines = item.get("timelines")
        moments = plain.get("moments") or []
        if not isinstance(timelines, list) or len(timelines) != len(moments):
            problems.append(f"bundle finding {index} timelines do not match its moments")
        else:
            for timeline in timelines:
                if not isinstance(timeline, dict) or not _real(timeline.get("duration_s")) or float(timeline.get("duration_s")) < 0:
                    problems.append(f"bundle finding {index} timeline is malformed")
                    break
                segments = timeline.get("segments")
                if not isinstance(segments, list) or not segments and any(
                        not isinstance(segment, dict) or any(key not in segment for key in _SEGMENT_KEYS)
                        or isinstance(segment.get("start_frame"), bool) or not isinstance(segment.get("start_frame"), int)
                        or isinstance(segment.get("end_frame"), bool) or not isinstance(segment.get("end_frame"), int)
                        or segment["start_frame"] < 0 or segment["end_frame"] >= frame_count
                        or segment["end_frame"] < segment["start_frame"]
                        for segment in segments):
                    problems.append(f"bundle finding {index} timeline is malformed")
                    break
        exercises = item.get("exercise_ids")
        if not isinstance(exercises, list) or any(not isinstance(name, str) for name in exercises):
            problems.append(f"bundle finding {index} exercise_ids must be a list of strings")


def _bind_lists(bundle, report, problems):
    if not isinstance(report, dict):
        return
    for key, empty in (("series", {}), ("events", []), ("windows", []), ("warnings", [])):
        if bundle.get(key) != report.get(key, empty):
            problems.append(f"bundle {key} does not match the report")


def _bind_court(bundle, doc, problems):
    court = bundle.get("court")
    lines = doc.get("court_lines_normalized")
    fit = (doc.get("court_fit") or {}).get("fit")
    source = doc.get("source") or {}
    usable = (
        isinstance(lines, list) and len(lines) == 4
        and source.get("width") and source.get("height")
        and not (_real(fit) and float(fit) < 0.85)
    )
    if court is None:
        if usable:
            problems.append("bundle court is missing for a document that has court lines")
        return
    matrix = court.get("to_court") if isinstance(court, dict) else None
    width = court.get("width_m") if isinstance(court, dict) else None
    length = court.get("length_m") if isinstance(court, dict) else None
    if not isinstance(court, dict) or not _close(width, 6.10) or not _close(length, 13.40) or not (
            isinstance(matrix, list) and len(matrix) == 3 and all(
                isinstance(row, list) and len(row) == 3 and all(_real(item) for item in row) for row in matrix)):
        problems.append("bundle court must be null or width_m 6.10, length_m 13.40, and a 3x3 to_court")
    elif not usable:
        problems.append("bundle court does not match the document")


def _bind_poses(bundle, doc, frame_count, players, problems):
    poses = bundle.get("poses")
    if not isinstance(poses, dict) or set(poses) != set(players):
        problems.append("bundle poses are missing a per-role payload")
        return
    expected, measured = _expected_pose(doc, frame_count, players)
    specs = {
        "xy": ((frame_count, 17, 2), "int16", 0.1, _XY_UNKNOWN),
        "conf": ((frame_count, 17), "uint8", 0.01, 0),
        "measured": ((frame_count,), "uint8", None, None),
        "bbox": ((frame_count, 4), "int16", 0.1, _XY_UNKNOWN),
    }
    for role in players:
        block = poses.get(role)
        if not isinstance(block, dict) or set(block) != set(specs):
            problems.append(f"bundle pose {role} does not have xy, conf, measured, and bbox")
            continue
        decoded = {}
        for name, (shape, dtype, scale, unknown) in specs.items():
            values, err = _decode_packed(block[name], shape, dtype, scale, unknown)
            if err:
                problems.append(f"bundle pose {role} {name} {err}")
            decoded[name] = values
        if any(values is None for values in decoded.values()):
            continue
        if any(item not in (0, 1) for item in decoded["measured"]):
            problems.append(f"bundle pose {role} measured must be 0 or 1")
        if decoded["measured"] != measured[role]:
            problems.append(f"bundle pose {role} measured does not match the document")
        slot = expected[role]
        xy = [(_XY_UNKNOWN if item is None else item) for item in slot["xy"]]
        if "overflow" in xy or "overflow" in slot["bbox"]:
            problems.append(f"bundle pose {role} has a joint outside -32767 to 32767 tenths")
        elif decoded["xy"] != xy or decoded["conf"] != slot["conf"] or decoded["bbox"] != slot["bbox"]:
            problems.append(f"bundle pose {role} does not match the document")


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _metric_availability_problems(metric, where):
    if not isinstance(metric, dict):
        return [f"{where} is not an object"]
    if "value" not in metric or "n_obs" not in metric or "confidence" not in metric or "best" not in metric:
        return [f"{where} needs value, n_obs, confidence, and best"]
    value = metric.get("value")
    n_obs = metric.get("n_obs")
    confidence = metric.get("confidence")
    best = metric.get("best")
    if value is None:
        problems = []
        if type(n_obs) is not int or n_obs != 0:
            problems.append(f"{where} null value requires n_obs 0")
        if not _finite_number(confidence) or confidence != 0:
            problems.append(f"{where} null value requires confidence 0")
        if best is not None:
            problems.append(f"{where} null value requires best null")
        return problems
    problems = []
    if not _finite_number(value):
        problems.append(f"{where} value is not a finite number or null")
    if type(n_obs) is not int or n_obs <= 0:
        problems.append(f"{where} numeric value requires n_obs > 0")
    if not _finite_number(confidence) or not 0 <= confidence <= 1:
        problems.append(f"{where} confidence must be in [0, 1]")
    if best is not None and not _finite_number(best):
        problems.append(f"{where} best is not a finite number or null")
    return problems


def _finding_availability_problems(finding, where):
    if not isinstance(finding, dict):
        return [f"{where} is not an object"]
    metrics = finding.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        return [f"{where} metrics must be a non-empty list"]
    problems = []
    unavailable = True
    for index, metric in enumerate(metrics):
        problems.extend(_metric_availability_problems(metric, f"{where} metric {index}"))
        if not isinstance(metric, dict) or metric.get("value") is not None:
            unavailable = False
    if problems or not unavailable:
        return problems
    if finding.get("severity") != "info":
        problems.append(f"{where} unavailable finding must stay info")
    score = finding.get("score")
    if not _finite_number(score) or score != 0:
        problems.append(f"{where} unavailable finding score must be 0")
    if finding.get("moments") != []:
        problems.append(f"{where} unavailable finding must have no moments")
    return problems


def report_contract_problems(doc):
    """Fresh coaching-report checks. Does not load or normalize legacy files."""
    if not isinstance(doc, dict):
        return ["report is not an object"]
    version = doc.get("schema_version")
    if version != REPORT_CURRENT_VERSION:
        return [f"coaching report schema_version {version} != {REPORT_CURRENT_VERSION}"]
    problems = []
    metrics = doc.get("metrics")
    if not isinstance(metrics, dict):
        problems.append("report metrics must be an object")
    else:
        for role, rows in metrics.items():
            if role not in ("near", "far"):
                problems.append(f"report metrics role {role!r} is not near or far")
            if not isinstance(rows, list):
                problems.append(f"report metrics {role} is not a list")
                continue
            for index, metric in enumerate(rows):
                problems.extend(_metric_availability_problems(metric, f"report metric {role}[{index}]"))
    findings = doc.get("findings")
    if not isinstance(findings, list):
        problems.append("report findings must be a list")
    else:
        for index, finding in enumerate(findings):
            problems.extend(_finding_availability_problems(finding, f"finding {index}"))
    return problems


def _report_loader_ready(version):
    return SCHEMA_VERSION == REPORT_CURRENT_VERSION and version in (REPORT_LEGACY_VERSION, REPORT_CURRENT_VERSION)


def validate_artifacts(result, frame_count, work_dir, claim, created=None, fps=None):
    problems = []
    if not isinstance(result, dict) or not isinstance(result.get("doc"), dict):
        return ["runner did not return a pose document"]
    report = result.get("report_path")
    bundle = result.get("bundle_path")
    pose = result.get("pose_path")
    needed = claim == "S-A2" or report is not None or bundle is not None or pose is not None
    if not needed:
        return problems
    if report is None or bundle is None or (claim == "S-A2" and pose is None):
        problems.append("end-to-end needs pose_path, report_path, and bundle_path")
        return problems
    if claim == "S-A2" and Path(pose).resolve() != (Path(work_dir) / "pose.json").resolve():
        problems.append("pose_path must be work_dir/pose.json")
        return problems
    created_set = {str(Path(path).resolve()) for path in created} if created is not None else None
    paths = {}
    labels = [("report", report), ("bundle", bundle)]
    if pose is not None:
        labels.insert(0, ("pose", pose))
    for label, raw in labels:
        if not _inside(raw, work_dir):
            problems.append(f"{label} path is outside the run directory")
            continue
        path = Path(raw)
        if not path.is_file():
            problems.append(f"{label} file was not written")
            continue
        if claim == "S-A2" and (created_set is None or str(path.resolve()) not in created_set):
            problems.append(f"{label} was not written inside the call")
            continue
        paths[label] = path
    if "report" not in paths or "bundle" not in paths:
        return problems
    pose_doc = None
    if "pose" in paths:
        try:
            pose_doc = json.loads(paths["pose"].read_text(encoding="utf-8"))
        except Exception as err:
            problems.append(f"pose rejected: {err}")
        if pose_doc != result["doc"]:
            problems.append("persisted pose does not match the returned document")
        saved = len(pose_doc.get("frames") or []) if isinstance(pose_doc, dict) else -1
        if saved != frame_count:
            problems.append(f"pose frames {saved} != {frame_count}")
    report_doc = None
    try:
        report_doc = json.loads(paths["report"].read_text(encoding="utf-8"))
    except Exception as err:
        problems.append(f"report rejected: {err}")
    if isinstance(report_doc, dict):
        problems.extend(report_contract_problems(report_doc))
        if _report_loader_ready(report_doc.get("schema_version")):
            try:
                report_from_dict(report_doc)
            except Exception as err:
                problems.append(f"report rejected: {err}")
    elif report_doc is not None:
        problems.append("report rejected: not an object")
    if isinstance(report_doc, dict) and report_doc.get("n_frames") != frame_count:
        problems.append(f"report n_frames {report_doc.get('n_frames')} != {frame_count}")
    try:
        bundle_doc = json.loads(paths["bundle"].read_text(encoding="utf-8"))
    except Exception as err:
        problems.append(f"bundle rejected: {err}")
        bundle_doc = None
    if not isinstance(bundle_doc, dict):
        problems.append("bundle must be a JSON object")
        return problems
    problems.extend(_bind_bundle(bundle_doc, result["doc"], report_doc, frame_count, fps))
    return problems


def call_timed(runner, video, work_dir):
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    before = {p.resolve() for p in work.rglob("*") if p.is_file()}
    t0 = time.perf_counter_ns()
    error = None
    try:
        result = runner.run(video, work)
    except Exception as err:
        result = None
        error = f"{type(err).__name__}: {err}"
    wall_ns = time.perf_counter_ns() - t0
    created = [str(p.resolve()) for p in work.rglob("*") if p.is_file() and p.resolve() not in before]
    timings = result.get("timings") if isinstance(result, dict) else None
    return {"wall_ns": wall_ns, "result": result, "error": error, "runner_timings": timings, "created": created}


def _project_py(path):
    if not path:
        return None
    path = Path(path).resolve()
    if path.suffix != ".py" or not path.is_file() or not _inside(path, PROJECT):
        return None
    return path


def _runner_models(runner):
    held = getattr(runner, "svc", None)
    raw = getattr(held, "models_dir", None) if held is not None else None
    if raw is None:
        raw = getattr(runner, "models_dir", None)
    return None if raw is None else Path(raw).resolve()


# Explicit coaching inputs. Verified by path. Not inferred from source text.
COACH_ASSETS = (
    "coach/style.json",
    "coach/exercises/library.json",
    "coach/kb/exercise_aliases.json",
    "coach/kb/fix_guides.json",
    "coach/kb/glossary.json",
    "coach/kb/lexicon.json",
    "coach/kb/limits.json",
    "coach/kb/safety.json",
    "coach/kb/tactics.json",
)
CACHE = HERE / "cache"
_WALK_LIMIT = 8


def _producer_manifest(runner):
    declared = getattr(runner, "binding_manifest", None)
    if declared is None:
        held = getattr(runner, "svc", None)
        declared = getattr(held, "binding_manifest", None) if held is not None else None
    if callable(declared):
        declared = declared()
    paths = []
    if isinstance(declared, dict):
        for key in ("sources", "assets"):
            for raw in declared.get(key) or []:
                paths.append(Path(raw))
    elif isinstance(declared, (list, tuple)):
        paths.extend(Path(raw) for raw in declared)
    return paths


def _declared_files(runner):
    paths = _producer_manifest(runner)
    paths.extend(PROJECT / rel for rel in COACH_ASSETS)
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise Refusal("binding manifest path is missing: " + "; ".join(missing))
    return [Path(path).resolve() for path in paths]


def _walk_runtime(obj, seen, modules, files, depth=0):
    """Follow the live object graph, including helpers stored inside other objects."""
    if obj is None or depth > _WALK_LIMIT or id(obj) in seen:
        return
    if inspect.ismodule(obj):
        seen.add(id(obj))
        path = _project_py(getattr(obj, "__file__", None))
        if path is None:
            return
        modules.add(path)
        for value in list(vars(obj).values()):
            if inspect.ismodule(value) or inspect.isclass(value) or inspect.isfunction(value):
                _walk_runtime(value, seen, modules, files, depth + 1)
        return
    if isinstance(obj, (str, Path)):
        seen.add(id(obj))
        path = Path(obj)
        try:
            if path.is_file():
                files.add(path.resolve())
        except OSError:
            return
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        seen.add(id(obj))
        for item in obj:
            _walk_runtime(item, seen, modules, files, depth + 1)
        return
    if isinstance(obj, dict):
        seen.add(id(obj))
        for key, value in obj.items():
            _walk_runtime(key, seen, modules, files, depth + 1)
            _walk_runtime(value, seen, modules, files, depth + 1)
        return
    if isinstance(obj, (int, float, bool, bytes, bytearray)):
        return
    seen.add(id(obj))
    if inspect.isclass(obj) or inspect.isfunction(obj):
        _walk_runtime(inspect.getmodule(obj), seen, modules, files, depth + 1)
        return
    mod = inspect.getmodule(type(obj))
    project_mod = _project_py(getattr(mod, "__file__", None)) if mod is not None else None
    if project_mod is not None:
        _walk_runtime(mod, seen, modules, files, depth + 1)
        try:
            values = vars(obj)
        except TypeError:
            return
        for value in values.values():
            _walk_runtime(value, seen, modules, files, depth + 1)


def _compiled_sources():
    """ONNX files OpenVINO actually compiled, including a fast/weights fallback."""
    mod = sys.modules.get("ovcache")
    found = set()
    if mod is None:
        return found
    for key in getattr(mod, "_source_hashes", {}):
        if isinstance(key, tuple) and key:
            path = Path(key[0])
            try:
                if path.is_file():
                    found.add(path.resolve())
            except OSError:
                continue
    return found


def _cache_inputs(model_files):
    if not CACHE.is_dir():
        return []
    stems = {path.stem for path in model_files if path.suffix.lower() == ".onnx"}
    if not stems and sys.modules.get("ovcache") is None:
        return []
    found = []
    for asset in CACHE.iterdir():
        if not asset.is_file():
            continue
        name = asset.name
        if name == "prepared-models.zip" or asset.suffix.lower() == ".blob" or any(
                name.startswith(stem + "-") for stem in stems):
            found.append(asset.resolve())
    return found


def execution_binding(runner, models_dir):
    """Hash the files the warmed runner actually holds, plus its declared manifest.

    Paths come from the live object graph and from binding_manifest(). A filename
    that merely appears in source text is not an input. A model OpenVINO compiled
    is kept even when it sits outside models_dir. A video or output path found on
    the object is not.
    """
    models = Path(models_dir).resolve()
    held = _runner_models(runner)
    if held is not None and held != models:
        raise Refusal(f"runner models_dir {held} does not match {models}")
    modules, files = set(), set()
    seen = set()
    _walk_runtime(runner, seen, modules, files)
    held_obj = getattr(runner, "svc", None)
    if held_obj is not None:
        _walk_runtime(held_obj, seen, modules, files)
    frozen = HERE / "frozen"
    for mod in list(sys.modules.values()):
        path = _project_py(getattr(mod, "__file__", None))
        if path is not None and _inside(path, frozen):
            modules.add(path)
    compiled = _compiled_sources()
    files.update(compiled)
    declared = set(_declared_files(runner))
    files.update(path for path in declared if path.suffix != ".py")
    modules.update(path for path in declared if path.suffix == ".py")
    files.update(_cache_inputs(files))
    keep = set(compiled)
    for path in files:
        if path in declared or _inside(path, models) or _inside(path, CACHE):
            keep.add(path)
    bound = {}
    for path in sorted(modules | keep, key=lambda item: str(item).lower()):
        try:
            key = str(path.relative_to(PROJECT)).replace("\\", "/")
        except ValueError:
            key = str(path)
        bound[key] = sha256(path)
    return bound


def s_a2_eligible(run):
    """Replay and every other non-S-A2 clock cannot fill the five-run counter."""
    if run.get("is_replay") or run.get("claim") == "replay":
        return False
    if run.get("s_a2_eligible") is False:
        return False
    return run.get("claim") == "S-A2"


def time_gate(runs, *, required_runs=FINAL_RUNS, limit_ns=LIMIT_NS):
    grouped = {}
    ineligible = 0
    for run in runs:
        if not run.get("gating"):
            continue
        if not s_a2_eligible(run):
            ineligible += 1
            continue
        grouped.setdefault(run["clip"], []).append(run["wall_ns"])
    failures = []
    if not grouped:
        failures.append("no S-A2 eligible timed runs")
    for clip, walls in grouped.items():
        if len(walls) < required_runs:
            failures.append(f"{clip}: {len(walls)} timed runs, need {required_runs}")
        for i, wall in enumerate(walls, start=1):
            if wall >= limit_ns:
                failures.append(f"{clip} run {i}: {wall} ns is not under {limit_ns}")
    return {
        "passed": bool(grouped) and not failures,
        "failures": failures,
        "ineligible_runs": ineligible,
        "replay_clocks_eligible": False,
        "limit_ns": limit_ns,
        "required_runs": required_runs,
    }


def _counts(doc, group, key):
    pack = (doc.get(group) or {}).get(key) or {}
    return pack.get("hits"), pack.get("total")


def meets_margin(new, base, num, den):
    nh, nt = new
    bh, bt = base
    if nt <= 0 or bt <= 0:
        return False
    return (nh * den + num * nt) * bt >= bh * nt * den


def within(new, base, num, den):
    nh, nt = new
    bh, bt = base
    if nt <= 0 or bt <= 0:
        return False
    return abs(nh * bt - bh * nt) * den <= num * nt * bt


def quality_gate(score, baseline, tier):
    """S-B and S-D are per role. Pooled counts stay on the score as diagnostics."""
    failures, insufficient = [], []
    roles = score.get("by_role") if isinstance(score, dict) else None
    base_roles = baseline.get("by_role") if isinstance(baseline, dict) else None
    if not isinstance(roles, dict) or not isinstance(base_roles, dict):
        insufficient.append("missing per-role scores")
    elif tier == "preview":
        checks = (
            ("all", "pck10", "pck10", MARGINS["pck10"]),
            ("fast", "pck10", "fast_pck10", MARGINS["fast_pck10"]),
            ("all", "found", "found", MARGINS["found"]),
        )
        for role in ("near", "far"):
            for group, key, name, margin in checks:
                label = f"{role} {name}"
                new = _counts(roles.get(role) or {}, group, key)
                base = _counts(base_roles.get(role) or {}, group, key)
                if None in new or None in base or new[1] in (0, None) or base[1] in (0, None):
                    insufficient.append(label)
                elif not meets_margin(new, base, *margin):
                    failures.append(f"{label}: {new} vs baseline {base}")
    elif tier == "refined":
        for role in ("near", "far"):
            new = _counts(roles.get(role) or {}, "all", "pck05")
            base = _counts(base_roles.get(role) or {}, "all", "pck05")
            label = f"{role} pck05"
            if None in new or None in base or new[1] in (0, None) or base[1] in (0, None):
                insufficient.append(label)
            elif not within(new, base, *REFINED_PCK05):
                failures.append(f"{label}: {new} vs baseline {base}")
    else:
        insufficient.append(f"tier {tier}")
    return {"passed": not failures and not insufficient, "failures": failures, "insufficient": insufficient,
            "tier": tier, "gated": "per-role",
            "margins": {"preview_pck10": 0.03, "preview_fast_pck10": 0.06, "preview_found": 0.02,
                        "refined_pck05": 0.002}}


def headline(claim, time_ok, valid_ok, quality_ok):
    if claim != "S-A2":
        return {"verdict": "NOT_MET", "reason": "the headline is only ever S-A2"}
    if time_ok and valid_ok and quality_ok:
        return {"verdict": "MET", "reason": None}
    return {"verdict": "NOT_MET", "reason": "S-A2 time, validation, or quality failed"}


def summarize(walls):
    ordered = sorted(walls)
    if not ordered:
        return {}
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {"n": len(ordered), "min_ns": ordered[0], "median_ns": median, "max_ns": ordered[-1]}


def host_info():
    import psutil
    cpu = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Processor | Select -First 1).Name"],
                         capture_output=True, text=True).stdout.strip()
    return {"cpu": cpu, "logical_cores": psutil.cpu_count(), "physical_cores": psutil.cpu_count(logical=False),
            "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1), "platform": platform.platform()}


def background_cpu(seconds=CPU_SAMPLE_S):
    import psutil
    procs, unavailable = [], 0
    for proc in psutil.process_iter(["name"]):
        if proc.pid == 0:
            continue
        try:
            proc.cpu_percent()
            procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            unavailable += 1
    cpu = psutil.cpu_percent(interval=seconds)
    busy = []
    for proc in procs:
        try:
            usage = proc.cpu_percent()
            if usage:
                busy.append((usage, proc.info["name"], proc.pid))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            unavailable += 1
    return {"cpu_percent": cpu, "sample_seconds": seconds, "loadavg": list(psutil.getloadavg()),
            "top_processes": sorted(busy, reverse=True)[:5], "unavailable_processes": unavailable}


def read_reference(mode, clip):
    path = reference_path(mode, clip)
    if mode != "final" and "sealed" in path.parts:
        raise Refusal("only accept.py final may read sealed references")
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8")), sha256(path), str(path)


def _write(path, doc):
    Path(path).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def warmup_video(out, clips):
    info = clips["clips"][WARMUP]
    src = PROJECT / info["file"]
    dest = Path(out) / "warmup-07.mp4"
    cmd = [str(FFMPEG), "-y", "-i", str(src), "-t", "3", "-an", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "23", "-r", "30", str(dest)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dest.is_file():
        raise RuntimeError(f"warm-up trim failed: {proc.stderr[-500:]}")
    return dest


def load_clip_table():
    path = HERE / "clips.json"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _selected(mode, table):
    if mode == "dev":
        return list(DEV_CLIPS)
    selected = list(GATING)
    if EDGE in table["clips"]:
        selected.append(EDGE)
    return selected


def run_acceptance(args):
    specs = args.runner
    if args.mode == "final" and len(specs) != 1:
        raise SystemExit("final takes exactly one runner")
    if args.mode == "dev" and not 1 <= len(specs) <= 2:
        raise SystemExit("dev takes one or two runners")
    repeats = FINAL_RUNS if args.mode == "final" else args.runs
    if args.mode == "dev" and repeats < 1:
        raise SystemExit("dev runs must be at least 1")
    loaded = []
    for spec in specs:
        try:
            label, cls = load_class(spec)
        except Exception as err:
            raise SystemExit(f"cannot load runner {spec}: {err}") from err
        if args.mode == "final" and getattr(cls, "IS_REPLAY", False):
            raise Refusal("final refuses a runner with IS_REPLAY")
        loaded.append((label, spec, cls))
    if args.mode == "final" and LOCK.exists():
        raise Refusal("cpu.lock exists")
    if args.mode == "final":
        sample = background_cpu(CPU_SAMPLE_S)
        if sample["cpu_percent"] > CPU_LIMIT:
            raise Refusal(f"background CPU {sample['cpu_percent']} is above {CPU_LIMIT}%")
        with cpu_lock_held(LOCK, sample["cpu_percent"]) as token:
            return _accept_body(args, loaded, sample, repeats, token)
    return _accept_body(args, loaded, None, repeats, None)


def _accept_body(args, loaded, sample, repeats, token):
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; results are never overwritten")
    table = load_clip_table()
    required = DEV_CLIPS if args.mode == "dev" else GATING
    missing = [clip for clip in required if clip not in table["clips"] or not (PROJECT / table["clips"][clip]["file"]).is_file()]
    if missing:
        raise SystemExit(f"missing clips: {', '.join(missing)}")
    if WARMUP not in table["clips"]:
        raise SystemExit("clip 07 is required for the untimed warm-up")
    out.mkdir(parents=True, exist_ok=True)
    runners = []
    for label, spec, cls in loaded:
        runner = cls(args.models_dir)
        if args.mode == "final":
            refuse_final(cls, runner, LOCK, sample["cpu_percent"], owned_token=token)
        runners.append({"label": label, "spec": spec, "claim": claim_for(cls), "runner": runner, "cls": cls})
    warmup = warmup_video(out, table)
    for item in runners:
        item["runner"].warm_up(warmup)
        item["binding"] = execution_binding(item["runner"], args.models_dir)
    selected = _selected(args.mode, table)
    order = schedule(selected, repeats)
    aggregate = {
        "mode": args.mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "timer": "perf_counter_ns around Runner.run only",
        "limit_ns": LIMIT_NS,
        "claims": [item["claim"] for item in runners],
        "replay_clocks_eligible_for_s_a2": False,
        "host": host_info(),
        "preflight_cpu": sample,
        "code_sha256": {
            **{name: sha256(HERE / name) for name in OWNED if (HERE / name).is_file()},
            "executed": {item["label"]: item["binding"] for item in runners},
        },
        "bundle_content_bound": BUNDLE_CONTENT_BOUND,
        "envelope": table.get("envelope"),
        "clips": {clip: table["clips"][clip] for clip in selected},
        "order": [],
        "runs": [],
    }
    seen_hash = set()
    (out / "outputs").mkdir()
    (out / "runs").mkdir()
    for slot, clip in enumerate(order):
        info = table["clips"][clip]
        video = PROJECT / info["file"]
        want_hash = (info.get("video_sha256") or "").lower()
        if want_hash and sha256(video) != want_hash:
            raise SystemExit(f"clip {clip} file hash does not match clips.json")
        turn = runners if slot % 2 == 0 else list(reversed(runners))
        for item in turn:
            tag = f"{clip}-{item['label']}-r{order[:slot].count(clip) + 1}"
            aggregate["order"].append(tag)
            fresh_dir = out / "fresh" / tag
            copied = fresh_copy(video, fresh_dir)
            proof = evidence(copied)
            if proof["copy_sha256"] in seen_hash:
                raise RuntimeError(f"fresh copy hash repeated for {tag}")
            seen_hash.add(proof["copy_sha256"])
            work = out / "work" / tag
            work.mkdir(parents=True)
            if args.mode == "final":
                load = background_cpu(CPU_SAMPLE_S)
                if load["cpu_percent"] > CPU_LIMIT:
                    raise Refusal(f"background CPU {load['cpu_percent']} is above {CPU_LIMIT}% before {tag}")
            else:
                load = background_cpu(CPU_SAMPLE_S)
            before_binding = execution_binding(item["runner"], args.models_dir)
            if before_binding != item["binding"]:
                raise Refusal(f"mixed inputs: {item['label']} sources or assets changed between runs")
            timed = call_timed(item["runner"], copied, work)
            after_binding = execution_binding(item["runner"], args.models_dir)
            if after_binding != before_binding:
                raise Refusal(f"executed sources or assets changed during {tag}")
            result = timed["result"] if isinstance(timed["result"], dict) else {}
            doc = result.get("doc") if isinstance(result, dict) else None
            output = out / "outputs" / f"{tag}.json"
            if isinstance(doc, dict):
                _write(output, doc)
            dense = dense_path(clip)
            baseline_both = both_role_frames(json.loads(dense.read_text(encoding="utf-8"))) if dense.is_file() else -1
            problems = []
            if timed["error"]:
                problems.append(timed["error"])
            if proof["frame_count"] != info["probe"]["frame_count"]:
                problems.append("decoded frame count does not match clips.json")
            problems += validate_pose(doc, info["probe"]["frame_count"], info["probe"]["fps"],
                                      (info["probe"]["width"], info["probe"]["height"]), baseline_both)
            problems += validate_artifacts(result, info["probe"]["frame_count"], work, item["claim"],
                                           created=timed["created"], fps=info["probe"]["fps"])
            is_replay = bool(
                getattr(item["cls"], "IS_REPLAY", False)
                or getattr(item["runner"], "IS_REPLAY", False)
                or item["claim"] == "replay"
            )
            rec = {
                "tag": tag,
                "clip": clip,
                "runner": item["label"],
                "claim": item["claim"],
                "is_replay": is_replay,
                "s_a2_eligible": item["claim"] == "S-A2" and not is_replay,
                "gating": clip in GATING,
                "edge": clip == EDGE,
                "wall_ns": timed["wall_ns"],
                "binding_sha256": before_binding,
                "runner_timings": timed["runner_timings"],
                "fresh": {k: proof[k] for k in ("source_sha256", "copy_sha256", "frame_count", "frames_match")},
                "load_before": load,
                "output": str(output),
                "validation_problems": problems,
                "valid": not problems,
            }
            aggregate["runs"].append(rec)
            _write(out / "runs" / f"{tag}.json", rec)
            _write(out / "acceptance.json", aggregate)
            print(json.dumps({"tag": tag, "wall_s": timed["wall_ns"] / 1e9, "valid": rec["valid"]}), flush=True)
    _score(args.mode, aggregate)
    gating = [run for run in aggregate["runs"] if run["gating"]]
    valid_ok = bool(gating) and all(run["valid"] for run in gating)
    quality_ok = bool(gating) and all(run.get("quality", {}).get("passed") for run in gating)
    timed_ok = time_gate(aggregate["runs"])["passed"]
    claims = {item["claim"] for item in runners}
    claim = "S-A2" if claims == {"S-A2"} else ("S-A" if claims == {"S-A"} else next(iter(claims)))
    aggregate["time_gate"] = time_gate(aggregate["runs"])
    aggregate["validation_passed"] = valid_ok
    aggregate["quality_passed"] = quality_ok
    aggregate["headline"] = headline(claim, timed_ok, valid_ok, quality_ok)
    aggregate["timing_by_clip"] = {
        clip: summarize([run["wall_ns"] for run in aggregate["runs"] if run["clip"] == clip and run["runner"] == item["label"]])
        for item in runners for clip in selected
    }
    aggregate["edge"] = [run["tag"] for run in aggregate["runs"] if run["edge"]]
    aggregate["cold"] = _cold(runners[0]["spec"], args.models_dir, warmup)
    _write(out / "acceptance.json", aggregate)
    if args.mode == "final":
        passed = timed_ok and valid_ok and quality_ok
    else:
        passed = valid_ok and quality_ok
    print(json.dumps({"passed": passed, "headline": aggregate["headline"]["verdict"]}), flush=True)
    return 0 if passed else 1


def _unavailable_reference(info):
    return isinstance(info, dict) and info.get("pose_quality_reference") == "unavailable"


def _score(mode, aggregate):
    for run in aggregate["runs"]:
        info = aggregate.get("clips", {}).get(run["clip"], {})
        if _unavailable_reference(info):
            notes = ["pose-quality reference unavailable"]
            if not run.get("valid"):
                notes.insert(0, "invalid output")
            run["quality"] = {
                "passed": False,
                "failures": [],
                "insufficient": notes,
                "unavailable": True,
            }
            run["score"] = None
            continue
        if not run["valid"]:
            run["quality"] = {"passed": False, "failures": [], "insufficient": ["invalid output"]}
            continue
        try:
            reference, ref_hash, ref_path = read_reference(mode, run["clip"])
        except FileNotFoundError as err:
            run["quality"] = {"passed": False, "failures": [], "insufficient": [f"missing reference {err}"]}
            continue
        mask_path = REFS / f"repeats-{run['clip']}.json"
        dense = dense_path(run["clip"])
        if not mask_path.is_file() or not dense.is_file():
            run["quality"] = {"passed": False, "failures": [], "insufficient": ["missing repeat mask or dense baseline"]}
            continue
        mask = json.loads(mask_path.read_text(encoding="utf-8"))
        prediction = json.loads(Path(run["output"]).read_text(encoding="utf-8"))
        baseline = json.loads(dense.read_text(encoding="utf-8"))
        fps = aggregate["clips"][run["clip"]]["probe"]["fps"]
        got = score_pair(prediction, reference, exclude=mask["exclude"], fps=fps)
        base = score_pair(baseline, reference, exclude=mask["exclude"], fps=fps)
        run["reference_sha256"] = ref_hash
        run["reference_path"] = ref_path
        kept = ("all", "fast", "measured", "interpolated", "fast_frames", "by_role")
        run["score"] = {k: got[k] for k in kept}
        run["baseline_score"] = {k: base[k] for k in kept}
        run["best_row"] = got.get("best_row")
        run["quality"] = quality_gate(got, base, prediction.get("tier"))


def _cold(spec, models_dir, video):
    cmd = [sys.executable, "-m", "turbo.accept", "cold", "--runner", spec, "--models-dir", str(models_dir), "--video", str(video)]
    t0 = time.perf_counter_ns()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {"label": "cold", "wall_ns": time.perf_counter_ns() - t0, "returncode": proc.returncode,
            "stdout": proc.stdout[-1000:], "stderr": proc.stderr[-1000:]}


def run_cold(args):
    label, cls = load_class(args.runner[0])
    runner = cls(args.models_dir)
    work = Path(tempfile.mkdtemp(prefix="turbo-cold-"))
    try:
        result = runner.run(args.video, work)
        ok = isinstance(result, dict) and isinstance(result.get("doc"), dict)
    except Exception as err:
        print(json.dumps({"label": "cold", "ok": False, "runner": label, "error": f"{type(err).__name__}: {err}"}))
        return 1
    print(json.dumps({"label": "cold", "ok": ok, "runner": label}))
    return 0 if ok else 1


def parse_args(argv):
    ap = argparse.ArgumentParser(prog="turbo.accept")
    sub = ap.add_subparsers(dest="mode", required=True)
    for mode in ("dev", "final"):
        parser = sub.add_parser(mode)
        parser.add_argument("--runner", action="append", required=True)
        parser.add_argument("--models-dir", required=True)
        parser.add_argument("--out", required=True)
        if mode == "dev":
            parser.add_argument("--runs", type=int, default=1)
    cold = sub.add_parser("cold")
    cold.add_argument("--runner", action="append", required=True)
    cold.add_argument("--models-dir", required=True)
    cold.add_argument("--video", required=True)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.mode == "cold":
            return run_cold(args)
        return run_acceptance(args)
    except Refusal as err:
        print(str(err), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
