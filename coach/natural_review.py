"""Render and audit the frozen learner-paced natural-review storyboard."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .app.backend import PROJECT, sha256_file
from .analyze import analyze
from .complete_review import VOICE
from .export import _tools
from .schema import Role
from .short_review import _srt_time
from .tracks import JOINT_INDEX, load_tracks
from .whole_clip_review import verify_pose_binding

ROOT = Path(__file__).resolve().parents[1]
NATURAL = ROOT / "work" / "NATURAL"
DEFAULT_VIDEO = ROOT / "sample-videos" / "04-china-open-ws-a.mp4"
DEFAULT_OUT = ROOT / "outputs" / "natural-three-moments-20261103-v6"
POSE_DIR = ROOT / "results" / "tiktok-coaching-editorial-20261103-v4"
FPS = 30
WIDTH, HEIGHT = 1080, 1920
VIEW = (64, 385, 1016, 1215)
VIEW_W, VIEW_H = VIEW[2] - VIEW[0], VIEW[3] - VIEW[1]
SOURCE_W, SOURCE_H = 1920, 1080
OVERLAY_MIN_SCORE = 0.70
PERSON_PAD_FRACTION = 0.08
DUPLICATE_MANIFEST = NATURAL / "audit" / "source_frame_diffs.json"
DUPLICATE_MANIFEST_SHA256 = "0440d7223655970b5cca49061fbf749a098288e5e1194023b94e8214befc7f00"
COLORS = {
    "near": {
        "left": (255, 190, 60), "right": (70, 235, 220),
        "body": (70, 235, 220), "arm": (70, 235, 220),
    },
    "far": {
        "left": (255, 120, 190), "right": (100, 190, 255),
        "body": (255, 120, 190), "arm": (100, 190, 255),
    },
}
JOINT_CHAINS = {
    "legs": {
        "near": [
            (("left_hip", "left_knee", "left_ankle"), "left"),
            (("right_hip", "right_knee", "right_ankle"), "right"),
        ],
    },
    "long_exchange": {
        role: [
            (("left_hip", "left_knee", "left_ankle"), "left"),
            (("right_hip", "right_knee", "right_ankle"), "right"),
        ]
        for role in ("near", "far")
    },
    "attack_dive": {
        "near": [
            (("left_hip", "right_hip"), "body"),
            (("left_shoulder", "right_shoulder"), "body"),
            (("right_hip", "right_shoulder"), "body"),
            (("right_shoulder", "right_elbow", "right_wrist"), "arm"),
        ],
    },
}
MOMENT_ROLES = {
    "legs": ("near",),
    "long_exchange": ("near", "far"),
    "attack_dive": ("near", "far"),
}
TITLE = {
    "legs": "跨一步，再回得来",
    "long_exchange": "你来我往",
    "attack_dive": "一挥之后，她扑出去",
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", "utf-8")


def _sha(path: Path) -> str:
    return sha256_file(path)


def _srt(seconds: float) -> str:
    return _srt_time(seconds)


def _blend_frames(first: np.ndarray, second: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0.0:
        return first.copy()
    if weight >= 1.0:
        return second.copy()
    return cv2.addWeighted(first, 1.0 - weight, second, weight, 0.0)


class SourceFrames:
    def __init__(self, path: Path):
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open source video: {path}")
        self.cache: dict[int, np.ndarray] = {}

    def get(self, index: int) -> np.ndarray:
        if index not in self.cache:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode source frame {index}")
            self.cache[index] = frame
            while len(self.cache) > 4:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[index]

    def close(self):
        self.capture.release()


def _position_parts(position: float, frame_count: int) -> tuple[int, int, float]:
    position = min(max(float(position), 0.0), frame_count - 1.0)
    first = math.floor(position + 1e-9)
    second = min(first + 1, frame_count - 1)
    return first, second, position - first


def _is_integer_position(position: float) -> bool:
    return abs(float(position) - round(float(position))) < 1e-6


def _load_unique_knots(video: Path, expected_source_sha256: str,
                       frame_count: int, selected_ranges: list[list[int]]) -> tuple[list[int], dict]:
    raw = DUPLICATE_MANIFEST.read_bytes()
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    if manifest_sha256 != DUPLICATE_MANIFEST_SHA256:
        raise ValueError(f"Duplicate manifest hash mismatch: {manifest_sha256}")
    manifest = json.loads(raw)
    if (Path(manifest["source"]).resolve() != video.resolve()
            or manifest["frames"] != frame_count):
        raise ValueError("Duplicate manifest is not bound to this source video and frame count")
    if _sha(video) != expected_source_sha256:
        raise ValueError("Source video hash does not match verified pose evidence")
    duplicates = manifest["duplicates"]
    if duplicates != sorted(set(duplicates)) or any(
        not isinstance(frame, int) or frame <= 0 or frame >= frame_count
        for frame in duplicates
    ):
        raise ValueError("Duplicate manifest frame list is not unique, ordered, and in range")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot independently inspect duplicate frames in {video}")
    measured = []
    previous = None
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (480, 270),
                interpolation=cv2.INTER_AREA,
            )
            if previous is not None:
                difference = cv2.absdiff(gray, previous)
                if float(difference.mean()) < 0.35 and float(np.percentile(difference, 99)) < 4.0:
                    measured.append(frame_index)
            previous = gray
            frame_index += 1
    finally:
        capture.release()
    if frame_index != frame_count:
        raise ValueError(f"Duplicate scan decoded {frame_index} frames, expected {frame_count}")

    manifest_duplicates = set(duplicates)
    measured_duplicates = set(measured)
    range_checks = []
    for start, end in selected_ranges:
        expected = sorted(frame for frame in manifest_duplicates if start <= frame <= end)
        actual = sorted(frame for frame in measured_duplicates if start <= frame <= end)
        range_checks.append({
            "source_frames": [start, end],
            "manifest_duplicates": expected,
            "independently_detected_duplicates": actual,
            "matched": expected == actual,
        })
    if not all(row["matched"] for row in range_checks):
        raise ValueError(f"Duplicate detection disagrees inside a selected moment: {range_checks}")
    knots = [frame for frame in range(frame_count) if frame not in manifest_duplicates]
    if len(knots) < 2:
        raise ValueError("Duplicate manifest leaves too few unique source frames")
    audit = {
        "manifest_path": str(DUPLICATE_MANIFEST.resolve()),
        "manifest_sha256": manifest_sha256,
        "manifest_source_sha256": expected_source_sha256,
        "source_sha256": _sha(video),
        "frame_count": frame_index,
        "rule": "480x270 grayscale; mean absolute difference < 0.35 and p99 < 4",
        "manifest_duplicate_count": len(duplicates),
        "independently_detected_duplicate_count": len(measured),
        "full_manifest_match": measured == duplicates,
        "selected_moment_ranges": range_checks,
    }
    return knots, audit


def _unique_sample(position: float, knots: list[int], control: bool = False,
                   shots: tuple[tuple[int, int], ...] | None = None):
    if _is_integer_position(position):
        frame = int(round(float(position)))
        sample = (frame, frame, 0.0, None)
    else:
        coordinates = np.arange(len(knots), dtype=np.float64)
        q = float(np.interp(float(position), np.asarray(knots, dtype=np.float64), coordinates))
        lower = min(math.floor(q), len(knots) - 1)
        upper = min(lower + 1, len(knots) - 1)
        weight = 0.0 if control else q - lower
        if control:
            nearest = min(len(knots) - 1, math.floor(q + 0.5))
            lower = upper = nearest
        sample = (knots[lower], knots[upper], weight, q)
    if shots is not None:
        raw_first = math.floor(float(position) + 1e-9)
        raw_second = raw_first if _is_integer_position(position) else raw_first + 1
        if not any(
            start <= raw_first <= raw_second <= end
            and start <= sample[0] <= sample[1] <= end
            for start, end in shots
        ):
            raise ValueError(f"Unique-frame sample crosses a camera cut at {position}")
    return sample


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _segment_for_index(segments: list[dict], index: int) -> dict:
    for segment in segments:
        start = round(float(segment["start_s"]) * FPS)
        end = round(float(segment["end_s"]) * FPS)
        if start <= index < end:
            return segment
    raise ValueError(f"No planned segment owns output frame {index}")


def _pose_moment_for_segment(name: str) -> str | None:
    moment_id = name.split(".", 1)[0]
    return moment_id if name.endswith((".slow_in", ".hold", ".follow")) else None


def _player_box(player, first: int, second: int, weight: float):
    if not player.present[first] or not player.present[second]:
        return None
    a = np.asarray(player.bbox[first], dtype=np.float64)
    b = np.asarray(player.bbox[second], dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    return a * (1.0 - weight) + b * weight


def _angle(player, frame: int, first_name: str, middle_name: str,
           last_name: str) -> float:
    a = player.joint(first_name)[frame].astype(np.float64)
    b = player.joint(middle_name)[frame].astype(np.float64)
    c = player.joint(last_name)[frame].astype(np.float64)
    u, v = a - b, c - b
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom <= 0 or not np.isfinite(denom):
        raise ValueError(f"Cannot measure {first_name}-{middle_name}-{last_name} at {frame}")
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / denom, -1.0, 1.0))))


def _person_roles_for_moment(moment: dict) -> tuple[str, ...]:
    return MOMENT_ROLES[moment["id"]]


def _hold_crop(tracks, moment: dict) -> dict:
    frame = int(moment["key_frame"])
    boxes = []
    for role in _person_roles_for_moment(moment):
        player = tracks.players[Role(role)]
        box = _player_box(player, frame, frame, 0.0)
        if box is None:
            raise ValueError(f"No measured {role} player box at {moment['id']} hold frame {frame}")
        boxes.append((role, box))
    joined = np.asarray([
        min(box[0] for _, box in boxes),
        min(box[1] for _, box in boxes),
        max(box[2] for _, box in boxes),
        max(box[3] for _, box in boxes),
    ], dtype=np.float64)
    pad_x = max(18.0, (joined[2] - joined[0]) * PERSON_PAD_FRACTION)
    pad_y = max(18.0, (joined[3] - joined[1]) * PERSON_PAD_FRACTION)
    needed_w = joined[2] - joined[0] + 2.0 * pad_x
    needed_h = joined[3] - joined[1] + 2.0 * pad_y
    zoom = min(SOURCE_W / needed_w, SOURCE_H / needed_h, 2.2)
    crop_w, crop_h = SOURCE_W / zoom, SOURCE_H / zoom
    center_x, center_y = (joined[0] + joined[2]) / 2, (joined[1] + joined[3]) / 2
    x1 = min(max(center_x - crop_w / 2, 0.0), SOURCE_W - crop_w)
    y1 = min(max(center_y - crop_h / 2, 0.0), SOURCE_H - crop_h)
    crop = [x1, y1, x1 + crop_w, y1 + crop_h]
    containment = []
    for role, box in boxes:
        margin = min(box[0] - crop[0], box[1] - crop[1],
                     crop[2] - box[2], crop[3] - box[3])
        containment.append({
            "role": role, "source_bbox": box.tolist(),
            "crop": crop, "minimum_source_margin_px": float(margin),
            "inside_with_margin": bool(
                box[0] - pad_x >= crop[0] - 1.0
                and box[1] - pad_y >= crop[1] - 1.0
                and box[2] + pad_x <= crop[2] + 1.0
                and box[3] + pad_y <= crop[3] + 1.0
            ),
        })
    if not all(row["inside_with_margin"] for row in containment):
        raise ValueError(f"Hold crop loses a named player: {moment['id']}")
    return {
        "crop": crop, "zoom": float(zoom), "center": [center_x, center_y],
        "named_players": list(_person_roles_for_moment(moment)),
        "hold_frame": frame, "containment": containment,
    }


def _crop_for_state(segment: dict, index: int, positions: list[float],
                    moments: dict[str, dict], crops: dict[str, dict]) -> tuple[float, ...]:
    name = segment["name"]
    moment_id = name.split(".", 1)[0]
    if moment_id not in moments:
        return (0.0, 0.0, float(SOURCE_W), float(SOURCE_H))
    moment = moments[moment_id]
    base = crops[moment_id]
    first = round(float(segment["start_s"]) * FPS)
    last = max(first, round(float(segment["end_s"]) * FPS) - 1)
    if name.endswith(".slow_in"):
        amount = _smoothstep((index - first) / max(1, last - first))
        zoom = 1.0 + (base["zoom"] - 1.0) * amount
    elif name.endswith(".hold"):
        zoom = base["zoom"]
    elif name.endswith(".follow"):
        amount = _smoothstep((index - first) / max(1, last - first))
        zoom = base["zoom"] + (1.0 - base["zoom"]) * amount
    else:
        zoom = 1.0
    crop_w, crop_h = SOURCE_W / zoom, SOURCE_H / zoom
    center_x, center_y = base["center"]
    x1 = min(max(center_x - crop_w / 2, 0.0), SOURCE_W - crop_w)
    y1 = min(max(center_y - crop_h / 2, 0.0), SOURCE_H - crop_h)
    return (x1, y1, x1 + crop_w, y1 + crop_h)


def _fit_crop(source: np.ndarray, crop: tuple[float, ...]) -> tuple[np.ndarray, float, tuple[float, float]]:
    x1, y1, x2, y2 = crop
    x1i, y1i = max(0, math.floor(x1)), max(0, math.floor(y1))
    x2i, y2i = min(SOURCE_W, math.ceil(x2)), min(SOURCE_H, math.ceil(y2))
    area = source[y1i:y2i, x1i:x2i]
    scale = min(VIEW_W / area.shape[1], VIEW_H / area.shape[0])
    width = max(1, round(area.shape[1] * scale))
    height = max(1, round(area.shape[0] * scale))
    resized = cv2.resize(area, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized, scale, (float(x1i), float(y1i))


def _transform_box(box: np.ndarray, crop: tuple[float, ...], scale: float,
                   offset: tuple[float, float]) -> list[float] | None:
    x1, y1, x2, y2 = crop
    if box[2] < x1 or box[0] > x2 or box[3] < y1 or box[1] > y2:
        return None
    ox = VIEW[0] + (VIEW_W - (x2 - x1) * scale) / 2 - offset[0] * scale
    oy = VIEW[1] + (VIEW_H - (y2 - y1) * scale) / 2 - offset[1] * scale
    return [
        max(VIEW[0], float(box[0] * scale + ox)),
        max(VIEW[1], float(box[1] * scale + oy)),
        min(VIEW[2], float(box[2] * scale + ox)),
        min(VIEW[3], float(box[3] * scale + oy)),
    ]


def _screen_person_boxes(tracks, position: float, crop: tuple[float, ...]) -> list[dict]:
    first, second, weight = _position_parts(position, tracks.n)
    boxes = []
    for role, player in ((role.value, player) for role, player in tracks.players.items()):
        raw = _player_box(player, first, second, weight)
        if raw is None:
            continue
        scale = min(VIEW_W / (crop[2] - crop[0]), VIEW_H / (crop[3] - crop[1]))
        offset = (math.floor(crop[0]), math.floor(crop[1]))
        screen = _transform_box(raw, crop, scale, offset)
        if screen is not None:
            boxes.append({"role": role, "bbox": screen})
    return boxes


def _inside(rect: list[float], box: list[float]) -> bool:
    return rect[0] >= box[2] or rect[2] <= box[0] or rect[1] >= box[3] or rect[3] <= box[1]


def _clearance(rect: list[float], box: list[float]) -> float:
    dx = max(box[0] - rect[2], rect[0] - box[2], 0.0)
    dy = max(box[1] - rect[3], rect[1] - box[3], 0.0)
    return math.hypot(dx, dy)


def _find_recap_card_rect(frame_map: list[float], segments: list[dict], tracks) -> list[int]:
    card_segments = [
        row for row in segments
        if row["name"] == "recap.card"
    ]
    if len(card_segments) != 1:
        raise ValueError("The plan must contain one recap.card segment")
    segment = card_segments[0]
    start, end = round(segment["start_s"] * FPS), round(segment["end_s"] * FPS)
    all_boxes = []
    full_crop = (0.0, 0.0, float(SOURCE_W), float(SOURCE_H))
    for position in frame_map[start:end]:
        all_boxes.extend(_screen_person_boxes(tracks, position, full_crop))
    width, height = 430, 300
    candidates = [
        (x, y, x + width, y + height)
        for y in (60, 535, 570, 620, 680, 735)
        for x in (74, 574)
    ]
    for rect in candidates:
        if all(_inside(list(rect), row["bbox"]) for row in all_boxes):
            return list(rect)
    raise ValueError("No fixed recap-card position stays clear of all detected players")


def _supported_point(player, name: str, first: int, second: int, weight: float):
    idx = JOINT_INDEX[name]
    a, b = player.joint(name)[first], player.joint(name)[second]
    if (not np.isfinite(a).all() or not np.isfinite(b).all()
            or float(player.conf[first, idx]) < OVERLAY_MIN_SCORE
            or float(player.conf[second, idx]) < OVERLAY_MIN_SCORE
            or not player.present[first] or not player.present[second]):
        return None, "raw-score-or-presence"
    for point, frame in ((a, first), (b, second)):
        box = player.bbox[frame]
        pad_x = max(8.0, float(box[2] - box[0]) * 0.08)
        pad_y = max(8.0, float(box[3] - box[1]) * 0.08)
        if not (box[0] - pad_x <= point[0] <= box[2] + pad_x
                and box[1] - pad_y <= point[1] <= box[3] + pad_y):
            return None, "outside-detected-player-box"
    point = a * (1.0 - weight) + b * weight
    return tuple(np.rint(point).astype(int)), None


def _draw_pose(frame: np.ndarray, tracks, moment_id: str | None,
               first: int, second: int, weight: float) -> list[dict]:
    if moment_id not in JOINT_CHAINS:
        return []
    checks = []
    for role, chains in JOINT_CHAINS[moment_id].items():
        player = tracks.players[Role(role)]
        width = max(3, round(float(player.bbox[first, 3] - player.bbox[first, 1]) / 55))
        drawn, rejected = set(), {}
        accepted = {}
        for names, color_key in chains:
            color = COLORS[role][color_key]
            for name in names:
                if name in accepted or name in rejected:
                    continue
                point, reason = _supported_point(player, name, first, second, weight)
                if reason:
                    rejected[name] = reason
                else:
                    accepted[name] = point
            for left, right in zip(names, names[1:]):
                if left in accepted and right in accepted:
                    cv2.line(frame, accepted[left], accepted[right], (10, 20, 25),
                             width + 4, cv2.LINE_AA)
                    cv2.line(frame, accepted[left], accepted[right], color, width, cv2.LINE_AA)
            for name in names:
                if name in accepted:
                    point = accepted[name]
                    cv2.circle(frame, point, width + 2, (10, 20, 25), -1, cv2.LINE_AA)
                    cv2.circle(frame, point, width, color, -1, cv2.LINE_AA)
                    drawn.add(name)
        checks.append({
            "role": role, "moment": moment_id, "source_frames": [first, second],
            "blend_weight": weight, "minimum_raw_score": OVERLAY_MIN_SCORE,
            "drawn_joints": sorted(drawn), "rejected": rejected,
        })
    return checks


def _text_boxes(draw, text: str, xy: tuple[int, int], font,
                fill="#ffffff", width=900, line_h=58) -> list[list[float]]:
    words = []
    for char in text:
        if words and draw.textlength(words[-1] + char, font=font) > width:
            words.append(char)
        elif words:
            words[-1] += char
        else:
            words.append(char)
    boxes = []
    x, y = xy
    for line in words:
        b = draw.textbbox((x, y), line, font=font)
        rect = [float(b[0] - 12), float(b[1] - 7), float(b[2] + 12), float(b[3] + 9)]
        draw.rounded_rectangle(rect, radius=12, fill="#07121f")
        draw.text((x, y), line, font=font, fill=fill,
                  stroke_width=1, stroke_fill="#07121f")
        boxes.append(rect)
        y += line_h
    return boxes


def _caption_for(schedule: dict, index: int) -> dict | None:
    at = index / FPS
    rows = [row for row in schedule["captions"]
            if row["speech_start_s"] <= at < row["speech_end_s"]]
    if len(rows) > 1:
        raise ValueError(f"More than one caption is active at output frame {index}")
    return rows[0] if rows else None


def _render_frame(source: np.ndarray, tracks, position: float, segment: dict,
                  index: int, schedule: dict, script: dict, fonts,
                  moments: dict, crops: dict, recap_rect: list[int]) -> tuple[np.ndarray, dict]:
    first, second, weight = _position_parts(position, tracks.n)
    image = source.copy()
    moment_id = segment["name"].split(".", 1)[0]
    active_moment = moment_id if moment_id in moments else None
    crop = _crop_for_state(segment, index, schedule["source_positions"],
                           moments, crops)
    pose_checks = _draw_pose(
        image, tracks, _pose_moment_for_segment(segment["name"]), first, second, weight,
    )
    pane, scale, offset = _fit_crop(image, crop)
    canvas = np.full((HEIGHT, WIDTH, 3), (17, 26, 31), np.uint8)
    px = VIEW[0] + (VIEW_W - pane.shape[1]) // 2
    py = VIEW[1] + (VIEW_H - pane.shape[0]) // 2
    canvas[py:py + pane.shape[0], px:px + pane.shape[1]] = pane
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    title_font, tag_font, caption_font, small_font, point_font = fonts
    boxes = []
    name = segment["name"]
    if active_moment:
        boxes += _text_boxes(
            draw, TITLE[active_moment], (78, 118), title_font, width=900, line_h=72,
        )
        tag = script["labels"][active_moment]["tag"]
        boxes += _text_boxes(draw, tag, (80, 252), tag_font, "#92d9d4", width=850)
        if name.endswith(".hold"):
            for row, text in enumerate(script["labels"][active_moment]["hold"]):
                boxes += _text_boxes(
                    draw, text, (78, 1240 + row * 60), point_font,
                    "#f2f6ef", width=875, line_h=44,
                )
    elif name == "hook":
        boxes += _text_boxes(draw, script["title"], (78, 118), title_font, width=900)
    elif name == "recap.card":
        x1, y1, x2, y2 = recap_rect
        draw.rounded_rectangle(recap_rect, radius=18, fill="#07121f")
        boxes.append([float(x1), float(y1), float(x2), float(y2)])
        recap = script["labels"]["recap"]
        boxes += _text_boxes(draw, recap["title"], (x1 + 18, y1 + 14),
                             tag_font, "#92d9d4", width=x2 - x1 - 36, line_h=55)
        for item, text in enumerate(recap["bullets"]):
            boxes += _text_boxes(
                draw, text, (x1 + 18, y1 + 72 + item * 58), point_font,
                "#f2f6ef", width=x2 - x1 - 36, line_h=48,
            )
    elif name == "recap.end":
        attribution = script.get("attribution_display", [])
        for item, text in enumerate(attribution):
            boxes += _text_boxes(draw, text, (78, 1128 + item * 42), small_font,
                                 "#c8d6dd", width=900, line_h=34)

    caption = _caption_for(schedule, index)
    if caption:
        boxes += _text_boxes(draw, caption["text"], (78, 1460), caption_font,
                             "#ffffff", width=900, line_h=58)
    screen_people = _screen_person_boxes(tracks, position, crop)
    clearances = []
    collisions = []
    for text_box in boxes:
        for person in screen_people:
            if not _inside(text_box, person["bbox"]):
                collisions.append({
                    "output_frame": index, "text_box": text_box,
                    "role": person["role"], "person_box": person["bbox"],
                })
            else:
                clearances.append(_clearance(text_box, person["bbox"]))
    image_out = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return image_out, {
        "text_boxes": boxes, "person_boxes": screen_people,
        "collisions": collisions, "clearances": clearances,
        "pose_checks": pose_checks, "crop": list(crop), "crop_scale": scale,
        "source_frames": [first, second], "blend_weight": weight,
    }


def _event_pose_evidence(script: dict, tracks, report) -> dict:
    near, far = tracks.players[Role.NEAR], tracks.players[Role.FAR]
    right_shift = near.joint("right_ankle")[86] - near.joint("right_ankle")[60]
    left_shift = near.joint("left_ankle")[86] - near.joint("left_ankle")[60]
    knee_before = _angle(near, 60, "right_hip", "right_knee", "right_ankle")
    knee_after = _angle(near, 89, "right_hip", "right_knee", "right_ankle")
    if not (right_shift[1] < -30 and np.linalg.norm(right_shift) > np.linalg.norm(left_shift)):
        raise ValueError("The named right-foot step is not supported by source frames 60–86")
    if not knee_after < knee_before - 15:
        raise ValueError("The front-knee bend is not stronger at frame 89 than frame 60")

    bottoms = [
        {"player": e.player.value, "frame": int(e.data["bottom_frame"])}
        for e in report.events
        if e.kind == "lunge" and 459 <= e.frame <= 545
        and "bottom_frame" in e.data
    ]
    bottoms.sort(key=lambda row: row["frame"])
    alternations = sum(a["player"] != b["player"]
                       for a, b in zip(bottoms, bottoms[1:]))
    if alternations < 3:
        raise ValueError("The exchange does not support at least three alternating lunge bottoms")

    wrist_candidates = [
        (frame, float(far.joint("right_wrist")[frame, 1]))
        for frame in range(620, 636)
        if far.measured[frame]
        and np.isfinite(far.joint("right_wrist")[frame]).all()
        and float(far.conf[frame, JOINT_INDEX["right_wrist"]]) >= OVERLAY_MIN_SCORE
    ]
    swing_peak = min(wrist_candidates, key=lambda row: row[1])[0]
    hip_rows = [
        (frame, float(np.mean([
            near.joint("left_hip")[frame, 1],
            near.joint("right_hip")[frame, 1],
        ])))
        for frame in range(612, 665)
        if near.measured[frame]
        and all(np.isfinite(near.joint(name)[frame]).all()
                and float(near.conf[frame, JOINT_INDEX[name]]) >= OVERLAY_MIN_SCORE
                for name in ("left_hip", "right_hip"))
    ]
    lowest_frame, lowest_y = max(hip_rows, key=lambda row: row[1])
    key_y = dict(hip_rows)[642]
    excursion = max(value for _, value in hip_rows) - min(value for _, value in hip_rows)
    low_delta_fraction = (lowest_y - key_y) / max(1.0, excursion)
    lag_s = (lowest_frame - swing_peak) / tracks.fps
    arm_extension = _angle(near, 642, "right_shoulder", "right_elbow", "right_wrist")
    if not (0.2 <= lag_s <= 0.8):
        raise ValueError("The overhead-to-low-pose sequence is outside the planned timing window")
    if low_delta_fraction > 0.10:
        raise ValueError("The dive hold is not within 10 percent of the measured lowest hip position")
    if arm_extension < 135:
        raise ValueError("The arm-extension hold label is not supported at source frame 642")

    motion = {}
    for name, start, end in (("hook", 240, 343), ("outro", 700, 890)):
        rows = []
        for role, player in (("near", near), ("far", far)):
            ids = [frame for frame in range(start, end + 1)
                   if player.measured[frame]
                   and np.isfinite(player.bbox[frame]).all()]
            if len(ids) < 2:
                continue
            centers = np.asarray([
                [(player.bbox[frame, 0] + player.bbox[frame, 2]) / 2,
                 (player.bbox[frame, 1] + player.bbox[frame, 3]) / 2]
                for frame in ids
            ])
            rows.append({
                "role": role, "measured_frames": len(ids),
                "bbox_center_path_px": float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()),
            })
        motion[name] = rows
        if not rows or max(row["bbox_center_path_px"] for row in rows) < 80:
            raise ValueError(f"The {name} footage does not show enough measured player motion")
    return {
        "m1_right_foot_step": {
            "rule": "right ankle from frame 60 to 86 moves forward in image and farther than the left ankle",
            "right_ankle_delta_px": right_shift.tolist(),
            "left_ankle_delta_px": left_shift.tolist(),
            "passed": True,
        },
        "m1_front_knee_bend": {
            "rule": "right hip-knee-ankle angle is smaller at 89 than 60",
            "angle_deg_at_60": knee_before, "angle_deg_at_89": knee_after,
            "passed": True,
        },
        "m1_removed_label": {
            "text": "上身低下来",
            "reason": "Not supported: the hip midpoint is higher in the image at frame 89 than at frame 60.",
            "hip_mid_y_px_at_60": float(np.mean([
                near.joint("left_hip")[60, 1], near.joint("right_hip")[60, 1],
            ])),
            "hip_mid_y_px_at_89": float(np.mean([
                near.joint("left_hip")[89, 1], near.joint("right_hip")[89, 1],
            ])),
        },
        "m2_alternating_lunges": {
            "rule": "at least three alternations in lunge-bottom player order",
            "bottoms": bottoms, "alternations": int(alternations), "passed": True,
        },
        "m3_overhead_precedes_dive": {
            "rule": "far player's right-wrist overhead peak precedes near player's lowest measured hip pose by 0.2-0.8s",
            "far_right_wrist_peak_frame": int(swing_peak),
            "near_lowest_hip_frame": int(lowest_frame),
            "lag_s": lag_s, "passed": True,
        },
        "m3_arm_out": {
            "rule": "near right shoulder-elbow-wrist angle at frame 642 is at least 135 degrees",
            "elbow_angle_deg": arm_extension, "passed": True,
        },
        "m3_body_low": {
            "rule": "frame 642 hip midpoint is within 10 percent of the full source-window vertical hip range from its lowest position",
            "frame_642_hip_y_px": key_y, "lowest_hip_y_px": lowest_y,
            "lowest_frame": int(lowest_frame),
            "range_px": excursion, "difference_fraction": low_delta_fraction,
            "passed": True,
        },
        "live_play_motion": motion,
    }


def _camera_cut_evidence(video: Path, tracks) -> dict:
    reader = SourceFrames(video)
    cuts = []
    try:
        for start, _ in tracks.shots[1:]:
            before, after = reader.get(start - 1), reader.get(start)
            cut_mae = float(np.abs(before.astype(np.int16) - after.astype(np.int16)).mean())
            nearby = []
            for first in range(max(0, start - 5), min(tracks.n - 1, start + 4)):
                if first in (start - 1,):
                    continue
                a, b = reader.get(first), reader.get(first + 1)
                nearby.append(float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()))
            baseline = float(np.median(nearby)) if nearby else 0.0
            cuts.append({
                "first_frame_after_cut": int(start),
                "source_shot_ranges": [list(x) for x in tracks.shots],
                "boundary_mean_absolute_pixel_difference": cut_mae,
                "nearby_within_shot_median_difference": baseline,
                "boundary_to_local_ratio": cut_mae / max(0.001, baseline),
                "verified": cut_mae > baseline * 3.0,
            })
    finally:
        reader.close()
    if not all(row["verified"] for row in cuts):
        raise ValueError("A camera cut was not verified against adjacent within-shot changes")
    return {"shots": [list(x) for x in tracks.shots], "cuts": cuts}


def _sync_evidence(script: dict, plan: dict, planned_map: dict) -> list[dict]:
    captions = {row["id"]: row for row in plan["captions"]}
    positions = planned_map["source_position_per_output_frame"]
    segments = plan["segments"]
    results = []
    for moment in script["timeline"]["moments"]:
        prefix = moment["id"] + "."
        indexes = [
            index for index, row in enumerate(segments)
            if row["name"].startswith(prefix)
        ]
        start_index = min(round(segments[index]["start_s"] * FPS) for index in indexes)
        end_index = max(round(segments[index]["end_s"] * FPS) for index in indexes)
        for event in moment.get("sync", []):
            output_index = next(
                (index for index in range(start_index, end_index)
                 if positions[index] + 1e-9 >= event["frame"]),
                None,
            )
            if output_index is None:
                raise ValueError(f"No output position reaches source event {event}")
            at = output_index / FPS
            caption = captions[event["line"]]
            start, end = caption["speech_start_s"], caption["speech_end_s"]
            distance = max(start - at, 0.0, at - end)
            row = {
                "moment": moment["id"], "line": event["line"],
                "event": event["what"], "source_frame": event["frame"],
                "output_frame": output_index, "output_time_s": at,
                "speech_window_s": [start, end],
                "margin_to_speech_window_s": (
                    at - start if at < start else at - end if at > end else 0.0
                ),
                "absolute_margin_s": distance,
                "within_tolerance": distance <= 0.5,
            }
            if not row["within_tolerance"]:
                raise ValueError(f"Narration event is out of sync: {row}")
            results.append(row)
    return results


PIXEL_AUDIT_GATES = (
        "picture_model_declared", "frame_binding", "blend_weight",
        "binding_control_rejects_wrong_map", "smoothness",
)


def _pixel_audit_results(lines: list[str]) -> dict[str, str]:
    results = {}
    for line in lines:
        parts = line.split(maxsplit=2)
        if len(parts) >= 2 and parts[1] in PIXEL_AUDIT_GATES:
            results[parts[1]] = parts[0]
    return results


def _pixel_audit_passed(lines: list[str]) -> bool:
    results = _pixel_audit_results(lines)
    return all(results.get(gate) == "VERIFIED" for gate in PIXEL_AUDIT_GATES)


def _run_pixel_audit(folder: Path) -> dict:
    command = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"), "-B",
        str(NATURAL / "audit" / "pixel_audit.py"),
        str(folder), "natural.mp4",
        "--plan", str(folder / "plan" / "script_check.json"),
        "--strict",
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    output = result.stdout + result.stderr
    log = folder / "pixel-audit.log"
    log.write_text(output, "utf-8")
    lines = result.stdout.splitlines()
    return {
        "status": "VERIFIED" if result.returncode == 0 and _pixel_audit_passed(lines)
        else "NOT VERIFIED",
        "returncode": result.returncode,
        "command": command,
        "log": str(log.resolve()),
        "lines": lines,
        "output": result.stdout,
    }


def _write_analysis(folder: Path, data: dict):
    _write_json(folder / "analysis.json", data)
    cards = []
    for row in data["moments"]:
        points = "".join(
            f'<li><b>{html.escape(item["kind"])}：</b>{html.escape(item["text"])}</li>'
            for item in row["analysis_points"]
        )
        cards.append(
            f'<article><h2>{row["rank"]}. {html.escape(row["title"])}</h2>'
            f'<p><b>源帧：</b>{row["source_frames"][0]}–{row["source_frames"][2]}，'
            f'关键帧 {row["source_frames"][1]}。'
            f'<b>输出：</b>{row["output_range_s"][0]:.2f}–{row["output_range_s"][1]:.2f} 秒。</p>'
            f'<ul>{points}</ul>'
            f'<p><b>练习提示：</b>{html.escape(row["practice_tip"])}</p>'
            f'<p><b>证据限制：</b>{html.escape(row["evidence_limit"])}</p>'
            f'</article>'
        )
    corpus = data["research_scope"]
    legal = data["attribution"]["full"].replace("\n", "<br>")
    (folder / "analysis.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>三个值得看的羽毛球瞬间</title>'
        '<style>body{font:18px/1.8 "Microsoft YaHei",sans-serif;background:#101a1f;color:#edf6f1;'
        'max-width:980px;margin:26px auto;padding:24px}article{padding:22px;background:#1a2a2c;'
        'margin:18px 0;border-radius:14px}li{margin:.5em 0}code{overflow-wrap:anywhere}</style>'
        '<h1>三个值得看的瞬间</h1>'
        '<p>只看这段30秒剪辑，不代表整场比赛。动作观察与给观众的练习提示分开。</p>'
        + "".join(cards)
        + '<h2>怎么选出的</h2>'
        '<p>候选来自整段900帧和四个镜头。按画面动作醒目程度、可供学习的观察价值、'
        '以及模型证据是否可读排序。单摄像机关键点只用于二维可见线索，不给球员打技术分。</p>'
        '<h2>研究范围与不足</h2>'
        f'<p>机器处理了{corpus["documents_read"]}篇文档，其中约{corpus["article_like"]}篇像文章、'
        f'{corpus["learner_posts"]}篇学习者提问或帖子、{corpus["answers"]}篇回答。'
        f'用户提出的1000篇文章没有达成：只处理了{corpus["documents_read"]}篇文档，'
        '其中大多数不是文章。用户提出的100场比赛也没有达成：只采样分析了'
        f'{corpus["games_machine_analysed"]}场，按计数规则统计为{corpus["games_counted"]}场，'
        '且不是人工逐场观看。语料分类是规则统计、单一阅读者核验，不能代表所有学习者。</p>'
        '<p>不判断球、触球、球路、得失分、胜负、拍型、三维动作、受力或伤病；'
        '不作动作正确/错误或标准评判。单机位二维观察不等同真实三维生物力学。</p>'
        '<h2>声音说明</h2><p>旁白由本机 Kokoro 生成，没有上传。没有听众评分或语音识别，'
        '自然度只能由实际收听者判断；自动音频指标只是已知问题的提示门槛。'
        'Kokoro 模型标注 Apache-2.0；其训练数据来源未经审计。</p>'
        '<h2>来源与许可</h2>'
        f'<p>{html.escape(legal)}</p>'
        '<p>Wikimedia Commons 上的 CC BY 3.0 标注不代表上游赛事转播权已获清理。</p>'
        '</html>',
        "utf-8",
    )


def _prepare_schedule(script: dict, plan_report: dict, plan_schedule: dict,
                      planned_map: dict, tracks, crops: dict) -> tuple[dict, list[dict]]:
    positions = planned_map["source_position_per_output_frame"]
    frames = planned_map["source_frame_per_output_frame"]
    if len(positions) != len(frames) or not positions:
        raise ValueError("Planned frame map is empty or inconsistent")
    floored = [math.floor(float(value) + 1e-9) for value in positions]
    if floored != frames:
        raise ValueError("Planned integer frame map does not match planned source positions")
    if any(value is None for value in frames):
        raise ValueError("The plan contains a source-less frame")
    for position in positions:
        first, second, _ = _position_parts(position, tracks.n)
        if not any(start <= first <= second <= end for start, end in tracks.shots):
            raise ValueError(f"Planned source blend crosses a camera cut at {position}")
    segments = plan_report["segments"]
    captions = [{
        "chapter": row["chapter"], "line_id": row["id"], "text": row["text"],
        "speech_start_s": float(row["speech_start_s"]),
        "speech_end_s": float(row["speech_end_s"]),
        "start_s": float(row["speech_start_s"]),
        "end_s": float(row["speech_end_s"]),
    } for row in plan_schedule["captions"]]
    for index, caption in enumerate(captions):
        for other in captions[index + 1:]:
            if max(caption["speech_start_s"], other["speech_start_s"]) < min(
                caption["speech_end_s"], other["speech_end_s"]
            ):
                raise ValueError("Caption speech windows overlap")
    moments = []
    for moment in script["timeline"]["moments"]:
        segments_for = [row for row in segments if row["name"].startswith(moment["id"] + ".")]
        first = min(round(row["start_s"] * FPS) for row in segments_for)
        end = max(round(row["end_s"] * FPS) for row in segments_for)
        moments.append({
            **moment, "title": TITLE[moment["id"]],
            "source_frames": [
                min(row["src_first"] for row in segments_for),
                moment["key_frame"],
                max(row["src_last"] for row in segments_for),
            ],
            "output_frame_range": [first, end],
            "output_range_s": [first / FPS, end / FPS],
            "hold_output_frame": next(
                round(row["start_s"] * FPS) for row in segments_for
                if row["kind"] == "freeze"
            ),
            "crop": crops[moment["id"]],
            "labels": script["labels"][moment["id"]],
        })
    schedule = {
        "fps": FPS,
        "planned_duration_s": len(positions) / FPS,
        "planned_output_frames": len(positions),
        "captions": captions,
        "segments": segments,
        "moments": moments,
        "source_positions": positions,
        "voice": script["voice"],
    }
    return schedule, segments


def _render_picture(video: Path, folder: Path, script: dict, schedule: dict,
                    tracks, planned_map: dict, recap_rect: list[int],
                    knots: list[int], duplicate_audit: dict) -> dict:
    ffmpeg, ffprobe = _tools()
    positions = planned_map["source_position_per_output_frame"]
    segments = schedule["segments"]
    moments = {row["id"]: row for row in script["timeline"]["moments"]}
    crops = {row["id"]: row["crop"] for row in schedule["moments"]}
    font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    fonts = tuple(ImageFont.truetype(str(font_dir / "msyh.ttc"), size)
                  for size in (58, 44, 44, 30, 42))
    source = SourceFrames(video)
    picture = folder / "picture.mp4"
    total_boxes, total_clearances, all_collisions = [], [], []
    overlay_checks, player_heights, frame_meta = [], [], {}
    started = time.perf_counter()
    samples = set()
    for moment in schedule["moments"]:
        start, end = moment["output_frame_range"]
        samples.update((start + min(5, end - start - 1),
                        moment["hold_output_frame"], end - 1))
    for row in ("hook", "recap.card", "recap.end"):
        segment = next(item for item in segments if item["name"] == row)
        samples.add(round(segment["start_s"] * FPS) + 3)
    try:
        with tempfile.TemporaryDirectory(prefix="natural-review-", dir=folder) as temporary:
            encoder_log = Path(temporary) / "encoder.log"
            with encoder_log.open("wb") as error_log:
                proc = subprocess.Popen([
                    ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                    "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0",
                    "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "19",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(picture),
                ], stdin=subprocess.PIPE, stderr=error_log)
                try:
                    for index, position in enumerate(positions):
                        segment = _segment_for_index(segments, index)
                        first, second, weight, q = _unique_sample(
                            position, knots, shots=tracks.shots,
                        )
                        raw = source.get(first)
                        if second != first:
                            raw = _blend_frames(raw, source.get(second), weight)
                        rendered, meta = _render_frame(
                            raw, tracks, position, segment, index, schedule, script,
                            fonts, moments, crops, recap_rect,
                        )
                        if index in samples:
                            frame_meta[index] = {
                                "position": position, "segment": segment["name"],
                                "unique_position": q,
                                "picture_frames": [first, second],
                                "picture_weight": weight,
                                "expected": rendered.copy(), "meta": meta,
                            }
                        total_boxes.extend(meta["text_boxes"])
                        total_clearances.extend(meta["clearances"])
                        all_collisions.extend(meta["collisions"])
                        overlay_checks.extend(
                            {"output_frame": index, **row}
                            for row in meta["pose_checks"]
                        )
                        for person in meta["person_boxes"]:
                            player_heights.append(
                                (person["bbox"][3] - person["bbox"][1]) / HEIGHT
                            )
                        proc.stdin.write(rendered.tobytes())
                    proc.stdin.close()
                    if proc.wait() != 0:
                        raise RuntimeError(encoder_log.read_text("utf-8", errors="replace"))
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
    finally:
        source.close()

    probe = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-count_frames", "-show_streams", "-show_format",
        "-of", "json", str(picture),
    ], text=True))
    stream = next(row for row in probe["streams"] if row["codec_type"] == "video")
    if (stream.get("codec_name") != "h264" or stream["width"] != WIDTH
            or stream["height"] != HEIGHT or int(stream["nb_read_frames"]) != len(positions)
            or float(stream["r_frame_rate"].split("/")[0]) / float(stream["r_frame_rate"].split("/")[1]) != FPS):
        raise RuntimeError("Encoded picture does not match required H.264 portrait frame format")
    decoded = cv2.VideoCapture(str(picture))
    sample_checks = []
    try:
        for index in sorted(samples):
            decoded.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, actual = decoded.read()
            if not ok:
                raise RuntimeError(f"Could not decode picture output frame {index}")
            expected = frame_meta[index]["expected"]
            error = float(cv2.absdiff(actual, expected).mean())
            sample_checks.append({
                "output_frame": index,
                "source_position": frame_meta[index]["position"],
                "segment": frame_meta[index]["segment"],
                "mean_absolute_pixel_error": error,
                "max_allowed": 22.0,
                "passed": error <= 22.0,
            })
    finally:
        decoded.release()
    hold_checks = []
    for moment in schedule["moments"]:
        index = moment["hold_output_frame"]
        meta = frame_meta.get(index, {}).get("meta")
        if meta is None:
            raise RuntimeError(f"Hold frame {index} was not decoded for skeleton verification")
        relevant = [row for row in overlay_checks if row["output_frame"] == index]
        required_roles = list(JOINT_CHAINS[moment["id"]])
        if not all(any(row["role"] == role and row["drawn_joints"]
                       for row in relevant) for role in required_roles):
            raise RuntimeError(f"Pose overlay is not visibly supported at hold {moment['id']}")
        hold_checks.append({
            "moment": moment["id"], "output_frame": index,
            "source_position": planned_map["source_position_per_output_frame"][index],
            "required_players": required_roles,
            "overlay_bindings": relevant,
            "verified": True,
        })
    if all_collisions:
        raise RuntimeError(f"Rendered text intersects a player box: {all_collisions[0]}")
    return {
        "picture": str(picture.resolve()),
        "sha256": _sha(picture),
        "codec": stream["codec_name"], "pixel_format": stream["pix_fmt"],
        "shape": [stream["width"], stream["height"]],
        "fps": FPS, "frames": int(stream["nb_read_frames"]),
        "duration_s": float(stream["duration"]),
        "decoded_sample_frames": sample_checks,
        "decoded_sample_count": len(sample_checks),
        "sample_bindings_passed": all(row["passed"] for row in sample_checks),
        "skeleton_binding_at_every_hold": hold_checks,
        "crop_containment_at_every_hold": [
            {"moment": row["id"], **row["crop"]} for row in schedule["moments"]
        ],
        "tallest_player_height_fraction_of_output": max(player_heights, default=0.0),
        "text_person_box_collisions": len(all_collisions),
        "minimum_text_to_person_clearance_px": min(total_clearances, default=None),
        "text_box_count": len(total_boxes),
        "overlay_check_count": len(overlay_checks),
        "overlay_checks": overlay_checks,
        "picture_model": "unique_blend_v1",
        "source_duplicates_sha256": duplicate_audit["manifest_sha256"],
        "duplicate_frame_audit": duplicate_audit,
        "render_elapsed_s": time.perf_counter() - started,
    }


def _audit_existing_picture(video: Path, folder: Path, script: dict, schedule: dict,
                            tracks, planned_map: dict, recap_rect: list[int],
                            knots: list[int], duplicate_audit: dict) -> dict:
    ffprobe = _tools()[1]
    positions = planned_map["source_position_per_output_frame"]
    segments = schedule["segments"]
    moments = {row["id"]: row for row in script["timeline"]["moments"]}
    crops = {row["id"]: row["crop"] for row in schedule["moments"]}
    font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    fonts = tuple(ImageFont.truetype(str(font_dir / "msyh.ttc"), size)
                  for size in (58, 44, 44, 30, 42))
    picture = folder / "picture.mp4"
    probe = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-count_frames", "-show_streams", "-show_format",
        "-of", "json", str(picture),
    ], text=True))
    stream = next(row for row in probe["streams"] if row["codec_type"] == "video")
    if (stream.get("codec_name") != "h264" or int(stream["width"]) != WIDTH
            or int(stream["height"]) != HEIGHT
            or int(stream["nb_read_frames"]) != len(positions)
            or float(stream["r_frame_rate"].split("/")[0])
            / float(stream["r_frame_rate"].split("/")[1]) != FPS):
        raise RuntimeError("Existing picture does not match the planned H.264 portrait format")

    samples = set()
    for moment in schedule["moments"]:
        start, end = moment["output_frame_range"]
        samples.update((start + min(5, end - start - 1),
                        moment["hold_output_frame"], end - 1))
    for name in ("hook", "recap.card", "recap.end"):
        segment = next(item for item in segments if item["name"] == name)
        samples.add(round(segment["start_s"] * FPS) + 3)

    sample_checks, overlay_checks = [], []
    total_boxes, total_clearances, all_collisions, player_heights = [], [], [], []
    source = SourceFrames(video)
    decoded = cv2.VideoCapture(str(picture))
    started = time.perf_counter()
    try:
        for index, position in enumerate(positions):
            segment = _segment_for_index(segments, index)
            first, second, weight, _ = _unique_sample(
                position, knots, shots=tracks.shots,
            )
            raw = source.get(first)
            if second != first:
                raw = _blend_frames(raw, source.get(second), weight)
            rendered, meta = _render_frame(
                raw, tracks, position, segment, index, schedule, script,
                fonts, moments, crops, recap_rect,
            )
            if index in samples:
                ok, actual = decoded.read()
                if not ok:
                    raise RuntimeError(f"Could not decode existing picture frame {index}")
                error = float(cv2.absdiff(actual, rendered).mean())
                sample_checks.append({
                    "output_frame": index,
                    "source_position": position,
                    "segment": segment["name"],
                    "mean_absolute_pixel_error": error,
                    "max_allowed": 22.0,
                    "passed": error <= 22.0,
                })
            elif not decoded.grab():
                raise RuntimeError(f"Could not advance existing picture through frame {index}")
            total_boxes.extend(meta["text_boxes"])
            total_clearances.extend(meta["clearances"])
            all_collisions.extend(meta["collisions"])
            overlay_checks.extend(
                {"output_frame": index, **row} for row in meta["pose_checks"]
            )
            player_heights.extend(
                (person["bbox"][3] - person["bbox"][1]) / HEIGHT
                for person in meta["person_boxes"]
            )
    finally:
        decoded.release()
        source.close()

    hold_checks = []
    for moment in schedule["moments"]:
        index = moment["hold_output_frame"]
        relevant = [row for row in overlay_checks if row["output_frame"] == index]
        required_roles = list(JOINT_CHAINS[moment["id"]])
        verified = all(any(row["role"] == role and row["drawn_joints"]
                           for row in relevant) for role in required_roles)
        hold_checks.append({
            "moment": moment["id"], "output_frame": index,
            "source_position": planned_map["source_position_per_output_frame"][index],
            "required_players": required_roles,
            "overlay_bindings": relevant, "verified": verified,
        })
    return {
        "picture": str(picture.resolve()), "sha256": _sha(picture),
        "codec": stream["codec_name"], "pixel_format": stream["pix_fmt"],
        "shape": [int(stream["width"]), int(stream["height"])],
        "fps": FPS, "frames": int(stream["nb_read_frames"]),
        "duration_s": float(stream["duration"]),
        "decoded_sample_frames": sample_checks,
        "decoded_sample_count": len(sample_checks),
        "sample_bindings_passed": all(row["passed"] for row in sample_checks),
        "skeleton_binding_at_every_hold": hold_checks,
        "crop_containment_at_every_hold": [
            {"moment": row["id"], **row["crop"]} for row in schedule["moments"]
        ],
        "tallest_player_height_fraction_of_output": max(player_heights, default=0.0),
        "text_person_box_collisions": len(all_collisions),
        "minimum_text_to_person_clearance_px": min(total_clearances, default=None),
        "text_box_count": len(total_boxes), "overlay_check_count": len(overlay_checks),
        "overlay_checks": overlay_checks,
        "picture_model": "unique_blend_v1",
        "source_duplicates_sha256": duplicate_audit["manifest_sha256"],
        "duplicate_frame_audit": duplicate_audit,
        "render_elapsed_s": time.perf_counter() - started,
    }


def _write_reports(folder: Path, video: Path, pose_verification: dict,
                   tracks, report, script: dict, schedule: dict,
                   evidence: dict, cuts: dict, sync: list[dict],
                   pixel_audit: dict, render: dict, attribution: dict):
    research_file = NATURAL / "design" / "design_findings.json"
    findings = _read_json(research_file)
    counts = findings["honest_counts"]
    research = {
        "goal": findings["goal"],
        "documents_read": counts["articles"]["total_docs"],
        "article_like": counts["articles"]["article_like"],
        "learner_posts": counts["articles"]["learner_question_or_post"],
        "answers": counts["articles"]["answer"],
        "articles_asked": counts["articles"]["asked"],
        "games_requested": counts["games"]["asked"],
        "games_machine_analysed": counts["games"]["analysed"],
        "games_counted": counts["games"]["counted"],
        "games_sampled_hours": counts["games"]["sampled_hours"],
        "evidence_path": str(research_file.resolve()),
        "limits": [
            "997 documents were processed, not 1000 articles.",
            "44 games were counted from 50 machine-analysed open-licensed samples, not 100 games.",
            "The game/video analysis was sampled and automated, not human full-match viewing.",
        ],
    }
    selected = []
    for index, moment in enumerate(schedule["moments"], 1):
        lines = [
            row for row in schedule["captions"]
            if row["chapter"] == moment["chapter"]
        ]
        selected.append({
            "rank": index, "id": moment["id"], "title": moment["title"],
            "source_frames": moment["source_frames"],
            "source_window": moment["shot"],
            "output_range_s": moment["output_range_s"],
            "key_frame_output": moment["hold_output_frame"],
            "spoken_lines": [
                {"line_id": row["line_id"], "text": row["text"],
                 "speech_window_s": [row["speech_start_s"], row["speech_end_s"]]}
                for row in lines
            ],
            "sync_events": [row for row in sync if row["moment"] == moment["id"]],
            "analysis_points": [
                {
                    "kind": "画面观察",
                    "text": _analysis_observation(moment["id"], evidence),
                },
            ],
            "practice_tip": _practice_tip(moment["id"]),
            "evidence_limit": _evidence_limit(moment["id"]),
            "crop": moment["crop"],
        })
    data = {
        "title": script["title"],
        "source": {
            "path": str(video.resolve()), "sha256": pose_verification["source_sha256"],
            "fps": tracks.fps, "frame_count": tracks.n,
            "scope": "仅分析所提供的30秒剪辑片段，不代表整场比赛。",
            "pose_reused": True, "pose_verification": pose_verification,
            "camera_shots": cuts,
        },
        "ranking_basis": [
            "按动作画面是否醒目、是否提供对学习者有用的观察线索、证据是否清楚排序。",
            "模型置信分只作可见性限制，不作为球员水平或教学价值的分数。",
        ],
        "moments": selected,
        "pose_word_and_label_evidence": evidence,
        "render_evidence": {
            "minimum_overlay_score": OVERLAY_MIN_SCORE,
            "overlay": render["skeleton_binding_at_every_hold"],
            "crop_containment": render["crop_containment_at_every_hold"],
            "tallest_player_height_fraction_of_output":
                render["tallest_player_height_fraction_of_output"],
            "minimum_text_person_clearance_px":
                render["minimum_text_to_person_clearance_px"],
            "picture_sampling": {
                "model": render["picture_model"],
                "duplicate_manifest_sha256": render["source_duplicates_sha256"],
                "duplicate_scan": render["duplicate_frame_audit"],
            },
            "smoothness": pixel_audit,
        },
        "sync_events": sync,
        "research_scope": research,
        "voice": {
            "engine": script["voice"]["engine"], "speaker_id": script["voice"]["sid"],
            "speed": script["voice"]["speed"],
            "execution": "local, no text or audio uploaded",
            "human_naturalness_rating": "not available",
            "model_license": "Apache-2.0",
            "model_training_data_provenance": "not audited",
        },
        "attribution": attribution,
        "claims_not_made": [
            "No shuttle path, contact, shot type, result, score, winner or loser claim.",
            "No 3D biomechanics, force, medical, injury, or right/wrong technique verdict.",
            "Pose overlays show only the saved 2D joints that pass the raw-score and body-box gates.",
        ],
        "limitations": [
            "The source is an edited 30-second excerpt containing four camera shots, not a full match.",
            "Pose and court observations are single-camera 2D estimates; they do not establish 3D motion or force.",
            "No shuttle/contact tracking is available; match outcome and shot result are unknown.",
            "Naturalness audit gates are proxy tripwires; no listener or ASR judged this voice.",
            "The Commons CC BY 3.0 source label does not clear upstream broadcast rights.",
        ],
        "narration": schedule["captions"],
    }
    _write_analysis(folder, data)
    _write_json(folder / "media-render.json", render)


def _analysis_observation(moment_id: str, evidence: dict) -> str:
    if moment_id == "legs":
        row = evidence["m1_front_knee_bend"]
        foot = evidence["m1_right_foot_step"]
        return (
            f'近端球员右脚从60到86帧向前移动约{np.linalg.norm(foot["right_ankle_delta_px"]):.0f}px；'
            f'右膝二维夹角从约{row["angle_deg_at_60"]:.0f}°变到{row["angle_deg_at_89"]:.0f}°。'
            '只描述画面中的关键点变化。'
        )
    if moment_id == "long_exchange":
        order = evidence["m2_alternating_lunges"]["bottoms"]
        return "可见两人交替下沉，低点顺序为" + "、".join(
            f'{item["player"]}{item["frame"]}帧' for item in order
        ) + "。"
    return (
        f'远端右腕高点在{evidence["m3_overhead_precedes_dive"]["far_right_wrist_peak_frame"]}帧，'
        f'近端髋部低点在{evidence["m3_overhead_precedes_dive"]["near_lowest_hip_frame"]}帧；'
        f'相差约{evidence["m3_overhead_precedes_dive"]["lag_s"]:.2f}秒。'
    )


def _practice_tip(moment_id: str) -> str:
    return {
        "legs": "练习发力时，从脚到腰连起来，别光靠手。",
        "long_exchange": "每跨一步，都要想着回到中间。",
        "attack_dive": "扑完以后，先想怎么起身。",
    }[moment_id]


def _evidence_limit(moment_id: str) -> str:
    return {
        "legs": "右脚和右膝只按未镜像的2D关键点观察；角度不是技术评分。",
        "long_exchange": "跨步低点来自模型事件与2D姿态；没有球路和触球证据。",
        "attack_dive": "上挥以右腕最高点作画面代理，不等于确认挥拍触球；地面阶段不推断受力或结果。",
    }[moment_id]


def _attribution() -> dict:
    path = ROOT / "sample-videos" / "china-open-source-attribution.json"
    data = _read_json(path)
    source = data["source"]
    credit = (
        "Badminton Sphere, Wikimedia Commons, CC BY 3.0, "
        "China Open 2025 women's singles final.\n"
        f'Commons page: {source["commons_page"]}\n'
        "The Commons licence label does not establish that upstream broadcast rights are cleared."
    )
    return {
        "credit": "Badminton Sphere, Wikimedia Commons, CC BY 3.0, China Open 2025 women's singles final",
        "commons_page": source["commons_page"],
        "license_as_listed": source["license"],
        "license_note": "The Commons CC BY 3.0 label does not clear upstream broadcast rights.",
        "full": credit,
        "last_frames_text": [
            "Badminton Sphere · Wikimedia Commons · CC BY 3.0",
            "许可标注不代表上游转播权已清理",
        ],
    }


def _run_plan_check(folder: Path) -> tuple[dict, dict, dict, str]:
    command = [
        str(NATURAL / ".venv" / "Scripts" / "python.exe"), "-B",
        str(NATURAL / "design" / "check_script.py"),
        "--script", str(folder / "script.json"),
        "--out-dir", str(folder / "plan"),
        "--cache-dir", str(folder / "voice"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    log = result.stdout + result.stderr
    (folder / "plan" / "check_script.log").write_text(log, "utf-8")
    if result.returncode != 0 or "gates 67 failed 0" not in log:
        raise RuntimeError("Natural script gate did not pass 67/0:\n" + log[-4000:])
    report = _read_json(folder / "plan" / "script_check.json")
    schedule = _read_json(folder / "plan" / "planned_schedule.json")
    frame_map = _read_json(folder / "plan" / "planned_frame_map.json")
    rows = list(_gate_rows(report.get("gates", {})))
    failed = sum(not row["pass"] for row in rows)
    if failed or len(rows) != 67:
        raise RuntimeError(f"Plan gate JSON mismatch: {len(rows)}, {failed} failed")
    return report, schedule, frame_map, log


def _gate_rows(value):
    if isinstance(value, dict):
        if isinstance(value.get("pass"), bool):
            yield value
            return
        for child in value.values():
            yield from _gate_rows(child)
    elif isinstance(value, list):
        for child in value:
            yield from _gate_rows(child)


def _mix_audio(folder: Path, expected_duration: float) -> dict:
    command = [
        str(NATURAL / ".venv" / "Scripts" / "python.exe"), "-B",
        str(NATURAL / "design" / "mock_mix.py"),
        "--report", str(folder / "plan" / "script_check.json"),
        "--clips", str(folder / "voice" / "sid60_x1.0"),
        "--out-dir", str(folder), "--out-name", "audio.m4a",
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    (folder / "audio-mix.log").write_text(result.stdout + result.stderr, "utf-8")
    if result.returncode != 0 or not (folder / "audio.m4a").is_file():
        raise RuntimeError("Natural audio mixer failed:\n" + result.stderr[-4000:])
    ffmpeg, ffprobe = _tools()
    audio_path = folder / "audio.m4a"
    probe = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-show_streams", "-of", "json", str(audio_path),
    ], text=True))
    audio_stream = next(row for row in probe["streams"] if row["codec_type"] == "audio")
    original_duration = float(audio_stream["duration"])
    padding = {
        "original_duration_s": original_duration,
        "planned_duration_s": expected_duration,
        "reencoded": False,
        "added_s": 0.0,
    }
    if original_duration < expected_duration - 1 / FPS:
        target_duration = expected_duration + 1 / (2 * FPS)
        with tempfile.TemporaryDirectory(prefix="audio-sync-", dir=folder) as temporary:
            padded_path = Path(temporary) / "padded.m4a"
            command = [
                ffmpeg, "-v", "error", "-y", "-i", str(audio_path),
                "-af", f"apad=whole_dur={target_duration:.6f}",
                "-t", f"{target_duration:.6f}", "-ar", "48000", "-ac", "1",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                str(padded_path),
            ]
            padded = subprocess.run(command, capture_output=True, text=True)
            (folder / "audio-sync.log").write_text(
                padded.stdout + padded.stderr, "utf-8",
            )
            if padded.returncode:
                raise RuntimeError("Could not align mixed AAC duration:\n" + padded.stderr[-4000:])
            os.replace(padded_path, audio_path)
        final_probe = json.loads(subprocess.check_output([
            ffprobe, "-v", "error", "-show_streams", "-of", "json", str(audio_path),
        ], text=True))
        final_stream = next(row for row in final_probe["streams"]
                            if row["codec_type"] == "audio")
        final_duration = float(final_stream["duration"])
        if abs(final_duration - expected_duration) > 1 / FPS:
            raise RuntimeError(
                f"Mixed AAC remains outside one-frame sync tolerance: "
                f"{final_duration:.6f}s vs {expected_duration:.6f}s"
            )
        padding.update({
            "reencoded": True,
            "target_duration_s": target_duration,
            "final_duration_s": final_duration,
            "added_s": final_duration - original_duration,
        })
    mix_folder = folder / "mix"
    if mix_folder.exists():
        shutil.rmtree(mix_folder)
    return padding


def _mux(picture: Path, audio: Path, output: Path):
    ffmpeg, _ = _tools()
    subprocess.run([
        ffmpeg, "-v", "error", "-y", "-i", str(picture), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0", "-c", "copy",
        "-movflags", "+faststart", str(output),
    ], check=True, capture_output=True)


def _final_media_check(folder: Path, expected_duration: float) -> dict:
    ffmpeg, ffprobe = _tools()
    media = folder / "natural.mp4"
    probe = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-count_frames", "-show_streams", "-show_format",
        "-of", "json", str(media),
    ], text=True))
    video = next(row for row in probe["streams"] if row["codec_type"] == "video")
    audio = next(row for row in probe["streams"] if row["codec_type"] == "audio")
    subprocess.run([ffmpeg, "-v", "error", "-i", str(media), "-f", "null", "-"],
                   check=True, capture_output=True)
    frames = int(video["nb_read_frames"])
    video_duration = float(video["duration"])
    audio_duration = float(audio["duration"])
    result = {
        "path": str(media.resolve()), "sha256": _sha(media),
        "video_codec": video["codec_name"], "audio_codec": audio["codec_name"],
        "pixel_format": video["pix_fmt"],
        "shape": [int(video["width"]), int(video["height"])],
        "fps": float(video["r_frame_rate"].split("/")[0])
        / float(video["r_frame_rate"].split("/")[1]),
        "frames": frames, "video_duration_s": video_duration,
        "audio_duration_s": audio_duration,
        "audio_video_duration_delta_s": abs(video_duration - audio_duration),
        "planned_duration_s": expected_duration,
        "full_decode": True,
        "faststart": _has_faststart_atom(media),
        "audio_video_within_one_frame": abs(video_duration - audio_duration) <= 1 / FPS,
    }
    if (result["video_codec"] != "h264" or result["audio_codec"] != "aac"
            or result["pixel_format"] != "yuv420p"
            or result["shape"] != [WIDTH, HEIGHT] or result["fps"] != FPS
            or frames != round(expected_duration * FPS)
            or abs(video_duration - expected_duration) > 1 / FPS
            or not result["faststart"]
            or not result["audio_video_within_one_frame"]):
        raise RuntimeError(f"Final media format/decode checks failed: {result}")
    return result


def _has_faststart_atom(media: Path) -> bool:
    with media.open("rb") as stream:
        head = stream.read(1024 * 1024)
    moov = head.find(b"moov")
    mdat = head.find(b"mdat")
    return moov >= 0 and (mdat < 0 or moov < mdat)


def _run_independent_media_check(folder: Path, expected_duration: float) -> dict:
    command = [
        str(NATURAL / ".venv" / "Scripts" / "python.exe"), "-B",
        str(ROOT / "scripts" / "audit_narrated_media.py"),
        str(folder / "natural.mp4"), "--ffmpeg",
        str(ROOT / "tools" / "ffmpeg.exe"),
        "--expected-duration", str(expected_duration),
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    output = result.stdout + result.stderr
    log = folder / "independent-media-audit.log"
    log.write_text(output, "utf-8")
    return {
        "status": "VERIFIED" if result.returncode == 0 else "NOT VERIFIED",
        "returncode": result.returncode,
        "command": command,
        "log": str(log.resolve()),
        "output": output,
        "lines": output.splitlines(),
    }


def _run_audit(folder: Path):
    command = [
        str(NATURAL / ".venv" / "Scripts" / "python.exe"), "-B",
        str(NATURAL / "analysis" / "audit_natural.py"),
        "--media", str(folder / "natural.mp4"),
        "--schedule", str(folder / "schedule.json"),
        "--frame-map", str(folder / "frame-map.json"),
        "--out", str(folder / "natural-audit.json"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    (folder / "natural-audit.log").write_text(result.stdout + result.stderr, "utf-8")
    if result.returncode:
        return _read_json(folder / "natural-audit.json"), result.stdout + result.stderr
    return _read_json(folder / "natural-audit.json"), result.stdout + result.stderr


def _write_captions(folder: Path, schedule: dict):
    rows = schedule["captions"]
    (folder / "captions.srt").write_text(
        "\n\n".join(
            f'{i}\n{_srt(row["speech_start_s"])} --> {_srt(row["speech_end_s"])}\n'
            f'{row["text"]}'
            for i, row in enumerate(rows, 1)
        ) + "\n", "utf-8",
    )
    (folder / "narration.txt").write_text(
        "\n".join(f'{row["line_id"]}：{row["text"]}' for row in rows) + "\n",
        "utf-8",
    )


def _write_final_maps(folder: Path, plan_schedule: dict,
                      planned_map: dict, segments: list[dict], script: dict,
                      schedule: dict, source_frame_count: int,
                      knots: list[int], duplicate_manifest_sha256: str,
                      shots: tuple[tuple[int, int], ...]):
    captions = [{
        "chapter": row["chapter"], "line_id": row["id"], "text": row["text"],
        "speech_start_s": float(row["speech_start_s"]),
        "speech_end_s": float(row["speech_end_s"]),
        "start_s": float(row["speech_start_s"]), "end_s": float(row["speech_end_s"]),
    } for row in plan_schedule["captions"]]
    schedule_data = dict(schedule)
    schedule_data.update({
        "fps": FPS,
        "planned_duration_s": len(planned_map["source_position_per_output_frame"]) / FPS,
        "planned_output_frames": len(planned_map["source_position_per_output_frame"]),
        "captions": captions, "segments": segments,
        "script_title": script["title"], "voice": script["voice"],
    })
    _write_json(folder / "schedule.json", schedule_data)
    states = []
    unique_positions = []
    for index, position in enumerate(planned_map["source_position_per_output_frame"]):
        segment = _segment_for_index(segments, index)
        first, second, weight = _position_parts(position, source_frame_count)
        picture_first, picture_second, picture_weight, q = _unique_sample(
            position, knots, shots=shots,
        )
        unique_positions.append(q)
        states.append({
            "output_frame": index, "segment": segment["name"],
            "chapter": segment["name"].split(".", 1)[0],
            "kind": segment["kind"], "phase": segment["name"].split(".", 1)[-1],
            "source_frame": planned_map["source_frame_per_output_frame"][index],
            "source_position": position, "blend_frames": [first, second],
            "blend_weight": weight,
            "picture_frames": [picture_first, picture_second],
            "picture_weight": picture_weight,
            "unique_position": q,
        })
    _write_json(folder / "frame-map.json", {
        "fps": planned_map["fps"],
        "source_frame_per_output_frame": planned_map["source_frame_per_output_frame"],
        "source_position_per_output_frame": planned_map["source_position_per_output_frame"],
        "picture_model": "unique_blend_v1",
        "source_duplicates_sha256": duplicate_manifest_sha256,
        "unique_position_per_output_frame": unique_positions,
        "states": states,
    })
    return _read_json(folder / "schedule.json"), states


def _verification(folder: Path, render: dict, media: dict, audit: dict,
                  plan_hash_checks: dict, pose_evidence: dict, sync: list[dict],
                  pixel_audit: dict, cut_evidence: dict, tests: dict | None = None) -> dict:
    plan_report = _read_json(folder / "plan" / "script_check.json")
    plan_gates = list(_gate_rows(plan_report["gates"]))
    frame_map = _read_json(folder / "frame-map.json")
    planned_map = _read_json(folder / "plan" / "planned_frame_map.json")
    unique_positions = frame_map.get("unique_position_per_output_frame", [])
    manifest = _read_json(DUPLICATE_MANIFEST)
    duplicate_set = set(manifest["duplicates"])
    verified_knots = [
        frame for frame in range(manifest["frames"])
        if frame not in duplicate_set
    ]
    expected_samples = [
        _unique_sample(position, verified_knots)
        for position in planned_map["source_position_per_output_frame"]
    ]
    states = frame_map.get("states", [])
    frame_binding = (
        frame_map.get("picture_model") == "unique_blend_v1"
        and frame_map.get("source_duplicates_sha256") == DUPLICATE_MANIFEST_SHA256
        and frame_map.get("source_position_per_output_frame")
        == planned_map["source_position_per_output_frame"]
        and frame_map.get("source_frame_per_output_frame")
        == planned_map["source_frame_per_output_frame"]
        and len(unique_positions) == len(planned_map["source_position_per_output_frame"])
        and all(
            (q is None) == _is_integer_position(position)
            for q, position in zip(
                unique_positions, planned_map["source_position_per_output_frame"],
            )
        )
        and len(states) == len(expected_samples)
        and all(
            (q is None and expected_q is None)
            or (q is not None and expected_q is not None
                and abs(q - expected_q) <= 1e-12)
            for q, (_, _, _, expected_q) in zip(unique_positions, expected_samples)
        )
        and all(
            row.get("picture_frames") == [first, second]
            and abs(float(row.get("picture_weight", 0.0)) - weight) <= 1e-12
            and row.get("unique_position") == q
            for row, (first, second, weight, q) in zip(states, expected_samples)
        )
    )
    checks = {
        "inputs_before": {
            "status": "VERIFIED", "log": str((folder / "inputs-before.log").resolve()),
            "measured": "ALL OK: 34 files",
        },
        "plan_checker": {
            "status": "VERIFIED" if len(plan_gates) == 67
            and all(row["pass"] for row in plan_gates) else "NOT VERIFIED",
            "measured": {"gates": len(plan_gates),
                         "failed": sum(not row["pass"] for row in plan_gates)},
            "log": str((folder / "plan" / "check_script.log").resolve()),
            "planned_files": plan_hash_checks,
        },
        "media_decode": {
            "status": "VERIFIED" if media["full_decode"] else "NOT VERIFIED",
            "measured": media,
        },
        "independent_media_check": {
            "status": media.get("independent_media_audit", {}).get(
                "status", "NOT VERIFIED",
            ),
            "measured": media.get("independent_media_audit", {}),
            "log": media.get("independent_media_audit", {}).get("log"),
        },
        "source_frame_binding": {
            "status": "VERIFIED" if render["sample_bindings_passed"]
            and render["decoded_sample_count"] >= 9 else "NOT VERIFIED",
            "measured": render["decoded_sample_frames"],
        },
        "unique_blend_frame_map": {
            "status": "VERIFIED" if frame_binding else "NOT VERIFIED",
            "measured": {
                "picture_model": frame_map.get("picture_model"),
                "manifest_sha256": frame_map.get("source_duplicates_sha256"),
                "plan_arrays_preserved": (
                    frame_map.get("source_position_per_output_frame")
                    == planned_map["source_position_per_output_frame"]
                    and frame_map.get("source_frame_per_output_frame")
                    == planned_map["source_frame_per_output_frame"]
                ),
                "frame_count": len(unique_positions),
            },
        },
        "duplicate_frame_detection": {
            "status": "VERIFIED" if all(
                row["matched"] for row in render["duplicate_frame_audit"]["selected_moment_ranges"]
            ) else "NOT VERIFIED",
            "measured": render["duplicate_frame_audit"],
        },
        "skeleton_binding_at_every_hold": {
            "status": "VERIFIED" if all(row["verified"]
                                         for row in render["skeleton_binding_at_every_hold"])
            else "NOT VERIFIED",
            "measured": render["skeleton_binding_at_every_hold"],
        },
        "hold_crop_containment": {
            "status": "VERIFIED" if all(
                all(person["inside_with_margin"] for person in row["containment"])
                for row in render["crop_containment_at_every_hold"]
            ) else "NOT VERIFIED",
            "measured": render["crop_containment_at_every_hold"],
        },
        "text_person_clearance": {
            "status": "VERIFIED" if render["text_person_box_collisions"] == 0 else "NOT VERIFIED",
            "measured": {
                "collisions": render["text_person_box_collisions"],
                "minimum_clearance_px": render["minimum_text_to_person_clearance_px"],
            },
        },
        "pixel_audit": {
            "status": pixel_audit["status"],
            "log": pixel_audit["log"],
            "lines": pixel_audit["lines"],
            "output": pixel_audit["output"],
            "required_gates": {
                gate: _pixel_audit_results(pixel_audit["lines"]).get(
                    gate, "NOT REPORTED",
                )
                for gate in PIXEL_AUDIT_GATES
            },
        },
        "sync_events": {
            "status": "VERIFIED" if all(row["within_tolerance"] for row in sync)
            else "NOT VERIFIED",
            "measured": sync,
        },
        "pose_claims_and_labels": {
            "status": "VERIFIED" if all(
                row["passed"] for key, row in pose_evidence.items()
                if key.startswith("m") and "passed" in row
            ) else "NOT VERIFIED",
            "measured": pose_evidence,
        },
        "naturalness_audit": {
            "status": "VERIFIED" if audit.get("pass") is True else "NOT VERIFIED",
            "measured": audit,
            "log": str((folder / "natural-audit.log").resolve()),
        },
        "repeat_motion": {
            "status": "VERIFIED" if audit.get("pacing", {}).get("gates", {}).get(
                "visits_per_source_frame", {}
            ).get("pass") and audit.get("pacing", {}).get("gates", {}).get(
                "hold_max_s", {}
            ).get("pass") else "NOT VERIFIED",
            "measured": audit.get("pacing", {}),
        },
        "camera_cuts": {
            "status": "VERIFIED" if all(row["verified"] for row in cut_evidence["cuts"])
            else "NOT VERIFIED",
            "measured": cut_evidence,
        },
        "tests": {
            "status": "VERIFIED" if tests and tests["status"] == "passed" else "NOT VERIFIED",
            "measured": tests or {},
        },
        "human_naturalness_rating": {
            "status": "NOT VERIFIED",
            "measured": "No human listener rating or ASR is available; only proxy tripwires ran.",
        },
    }
    result = {
        "status": "VERIFIED" if all(
            row["status"] == "VERIFIED"
            for key, row in checks.items() if key != "human_naturalness_rating"
        ) else "NOT VERIFIED",
        "checks": checks,
    }
    _write_json(folder / "verification.json", result)
    return result


def _hash_manifest(folder: Path) -> dict:
    hashes = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.name != "handoff.json":
            hashes[str(path.relative_to(folder))] = {
                "path": str(path.resolve()), "sha256": _sha(path),
                "bytes": path.stat().st_size,
            }
    return hashes


def _finalize(folder: Path, tests: dict | None = None) -> dict:
    analysis = _read_json(folder / "analysis.json")
    verification = _read_json(folder / "verification.json")
    if tests is None and (folder / "tests.json").is_file():
        tests = _read_json(folder / "tests.json")
    tests = tests or {"status": "pending"}
    verification["checks"]["tests"] = {
        "status": "VERIFIED" if tests.get("status") == "passed" else "NOT VERIFIED",
        "measured": tests,
    }
    verification["status"] = "VERIFIED" if all(
        row["status"] == "VERIFIED"
        for key, row in verification["checks"].items()
        if key != "human_naturalness_rating"
    ) else "NOT VERIFIED"
    _write_json(folder / "verification.json", verification)
    source_paths = [
        ROOT / "coach" / "natural_review.py",
        ROOT / "coach" / "tests" / "test_natural_review.py",
    ]
    changed = {
        str(path.relative_to(ROOT)).replace("\\", "/"): {
            "path": str(path.resolve()), "sha256": _sha(path),
        } for path in source_paths
    }
    model_dir = ROOT / "models" / "kokoro-multi-lang-v1_1"
    voice_files = {
        str(path.relative_to(NATURAL)): {"sha256": _sha(path), "bytes": path.stat().st_size}
        for path in sorted(model_dir.iterdir()) if path.is_file()
    }
    clip_dir = folder / "voice" / "sid60_x1.0"
    voice_clips = {
        path.name: {"sha256": _sha(path), "bytes": path.stat().st_size}
        for path in sorted(clip_dir.iterdir()) if path.is_file()
    }
    handoff = {
        "status": verification["status"],
        "output_folder": str(folder.resolve()),
        "video": str((folder / "natural.mp4").resolve()),
        "video_sha256": _sha(folder / "natural.mp4"),
        "source": analysis["source"],
        "outputs": _hash_manifest(folder),
        "changed_source_files": changed,
        "voice": {
            **analysis["voice"],
            "model_files": voice_files,
            "reference_clip_files": voice_clips,
        },
        "attribution": analysis["attribution"],
        "research_scope": analysis["research_scope"],
        "verification": verification,
        "tests": tests,
        "open_risks": [
            "No human listener or ASR rated the naturalness of the rendered narration.",
            "Only 44 games were counted and 997 documents were read; neither research request was fully met.",
            "The Commons CC BY 3.0 attribution does not clear upstream broadcast rights.",
        ],
    }
    _write_json(folder / "handoff.json", handoff)
    return handoff


def _recover_existing_outputs(video: Path, folder: Path) -> dict:
    from turbo.accept import acquire_cpu_lock, release_cpu_lock
    lock = ROOT / "work" / "cpu.lock"
    token = f"natural-review {os.getpid()} {uuid.uuid4().hex}"
    acquire_cpu_lock(lock, token)
    try:
        if not (folder / "natural.mp4").is_file() or not (folder / "picture.mp4").is_file():
            raise FileNotFoundError("Recovery requires the already-rendered picture and muxed video")
        verifier = NATURAL / "design" / "verify_inputs.py"
        before = subprocess.run([
            str(NATURAL / ".venv" / "Scripts" / "python.exe"), "-B", str(verifier),
        ], capture_output=True, text=True, encoding="utf-8")
        (folder / "inputs-before.log").write_text(before.stdout + before.stderr, "utf-8")
        if before.returncode or "ALL OK: 34 files" not in before.stdout:
            raise RuntimeError("Frozen inputs did not verify before recovery")

        report = _read_json(folder / "plan" / "script_check.json")
        plan_schedule = _read_json(folder / "plan" / "planned_schedule.json")
        planned_map = _read_json(folder / "plan" / "planned_frame_map.json")
        plan_gates = list(_gate_rows(report["gates"]))
        frozen_schedule = NATURAL / "design" / "planned_schedule.json"
        frozen_frame_map = NATURAL / "design" / "planned_frame_map.json"
        plan_hashes = {
            "schedule_identical_to_frozen":
                (folder / "plan" / "planned_schedule.json").read_bytes()
                == frozen_schedule.read_bytes(),
            "frame_map_identical_to_frozen":
                (folder / "plan" / "planned_frame_map.json").read_bytes()
                == frozen_frame_map.read_bytes(),
            "schedule_sha256": _sha(folder / "plan" / "planned_schedule.json"),
            "frozen_schedule_sha256": _sha(frozen_schedule),
            "frame_map_sha256": _sha(folder / "plan" / "planned_frame_map.json"),
            "frozen_frame_map_sha256": _sha(frozen_frame_map),
            "script_sha256": _sha(folder / "script.json"),
        }
        if (len(plan_gates) != 67 or not all(row["pass"] for row in plan_gates)
                or not all(plan_hashes[key] for key in (
                    "schedule_identical_to_frozen", "frame_map_identical_to_frozen",
                ))):
            raise RuntimeError("Existing plan does not satisfy its frozen contract")

        pose_paths = (
            POSE_DIR / "pose.json", POSE_DIR / "points.json",
            POSE_DIR / "analysis.json", POSE_DIR / "handoff.json",
        )
        pose_verification = verify_pose_binding(video, *pose_paths)
        tracks = load_tracks(_read_json(pose_paths[0]))
        pose_report = analyze(_read_json(pose_paths[0]))
        cuts = _camera_cut_evidence(video, tracks)
        script = _read_json(folder / "script.json")
        attribution = _attribution()
        script["attribution_display"] = attribution["last_frames_text"]
        evidence = _event_pose_evidence(script, tracks, pose_report)
        schedule = _read_json(folder / "schedule.json")
        sync = _sync_evidence(script, report, planned_map)
        selected_ranges = [
            [int(moment["source_frames"][0]), int(moment["source_frames"][2])]
            for moment in schedule["moments"]
        ]
        knots, duplicate_audit = _load_unique_knots(
            video, pose_verification["source_sha256"], tracks.n, selected_ranges,
        )
        previous_duplicate_audit = _read_json(folder / "duplicate-frame-audit.json")
        if duplicate_audit != previous_duplicate_audit:
            raise RuntimeError("Existing duplicate audit differs from current verified source scan")

        pixel_log = folder / "pixel-audit.log"
        pixel_output = pixel_log.read_text("utf-8")
        pixel_lines = pixel_output.splitlines()
        pixel_audit = {
            "status": "VERIFIED" if _pixel_audit_passed(pixel_lines) else "NOT VERIFIED",
            "returncode": 0 if _pixel_audit_passed(pixel_lines) else 1,
            "command": [
                str(ROOT / ".venv" / "Scripts" / "python.exe"), "-B",
                str(NATURAL / "audit" / "pixel_audit.py"),
                str(folder), "natural.mp4", "--plan",
                str(folder / "plan" / "script_check.json"), "--strict",
            ],
            "log": str(pixel_log.resolve()), "lines": pixel_lines,
            "output": pixel_output,
        }
        _write_json(folder / "pixel-audit.json", pixel_audit)
        cached_render = folder / "media-render.json"
        if cached_render.is_file():
            candidate = _read_json(cached_render)
        else:
            candidate = {}
        render = candidate if candidate.get("sha256") == _sha(folder / "picture.mp4") else (
            _audit_existing_picture(
                video, folder, script, schedule, tracks, planned_map,
                schedule["recap_card_rect"], knots, duplicate_audit,
            )
        )
        media = _final_media_check(folder, schedule["planned_duration_s"])
        independent_media = _run_independent_media_check(
            folder, schedule["planned_duration_s"],
        )
        media["independent_media_audit"] = independent_media
        _write_reports(
            folder, video, pose_verification, tracks, pose_report, script, schedule,
            evidence, cuts, sync, pixel_audit, render, attribution,
        )
        _write_captions(folder, schedule)
        audit, audit_log = _run_audit(folder)
        _write_json(folder / "media-audit.json", {
            "independent_media_check": independent_media,
            "internal_decode_check": media,
            "pixel_audit": pixel_audit,
            "naturalness_audit": audit,
            "naturalness_audit_log": audit_log,
        })
        _write_json(folder / "media-render.json", render)
        _write_json(folder / "run.json", {
            "command": "python -B -m coach.natural_review --finalize",
            "recovered_existing_render_without_rerender": True,
            "source_sha256": pose_verification["source_sha256"],
            "source_frames": tracks.n, "source_fps": tracks.fps,
            "full_source_inference": False,
            "pose_reused_after_hash_and_full_coverage_verification": True,
            "camera_cuts_reverified": cuts,
            "research_shortfalls": _read_json(
                NATURAL / "design" / "design_findings.json"
            )["honest_counts"],
            "script_sha256": plan_hashes["script_sha256"],
            "plan_hash_checks": plan_hashes,
            "plan_gate_count": len(plan_gates),
            "plan_failed_gates": sum(not row["pass"] for row in plan_gates),
            "planned_frames": len(planned_map["source_frame_per_output_frame"]),
            "planned_duration_s": schedule["planned_duration_s"],
            "picture_model": "unique_blend_v1",
            "duplicate_frame_audit": duplicate_audit,
            "render": render, "final_media": media,
            "sync_events": sync, "pixel_audit": pixel_audit,
            "naturalness_audit": audit,
        })
        after = subprocess.run([
            str(NATURAL / ".venv" / "Scripts" / "python.exe"), "-B", str(verifier),
        ], capture_output=True, text=True, encoding="utf-8")
        (folder / "inputs-after.log").write_text(after.stdout + after.stderr, "utf-8")
        if after.returncode or "ALL OK: 34 files" not in after.stdout:
            raise RuntimeError("Frozen inputs did not verify after recovery")
        tests = _read_json(folder / "tests.json") if (folder / "tests.json").is_file() else None
        verification = _verification(
            folder, render, media, audit, plan_hashes, evidence, sync, pixel_audit,
            cuts, tests,
        )
        verification["checks"]["inputs_after"] = {
            "status": "VERIFIED", "log": str((folder / "inputs-after.log").resolve()),
            "measured": "ALL OK: 34 files",
        }
        verification["status"] = "VERIFIED" if all(
            row["status"] == "VERIFIED" for key, row in verification["checks"].items()
            if key != "human_naturalness_rating"
        ) else "NOT VERIFIED"
        _write_json(folder / "verification.json", verification)
        return verification
    finally:
        release_cpu_lock(lock, token)


def export_natural(video: Path, folder: Path) -> dict:
    if not folder.is_dir():
        raise FileNotFoundError(f"Stage the new output folder first: {folder}")
    if (folder / "natural.mp4").exists():
        raise FileExistsError(f"Refusing to overwrite an existing natural.mp4: {folder}")
    required = [
        folder / "script.json", folder / "voice" / "sid60_x1.0",
    ]
    if any(not path.exists() for path in required):
        raise FileNotFoundError("Output staging is missing script.json or copied voice clips")

    from turbo.accept import acquire_cpu_lock, release_cpu_lock
    lock = ROOT / "work" / "cpu.lock"
    token = f"natural-review {os.getpid()} {uuid.uuid4().hex}"
    acquire_cpu_lock(lock, token)
    try:
        verifier = NATURAL / "design" / "verify_inputs.py"
        before = subprocess.run([
            str(NATURAL / ".venv" / "Scripts" / "python.exe"), "-B", str(verifier),
        ], capture_output=True, text=True, encoding="utf-8")
        (folder / "inputs-before.log").write_text(before.stdout + before.stderr, "utf-8")
        if before.returncode or "ALL OK: 34 files" not in before.stdout:
            raise RuntimeError("Frozen inputs did not verify before render")

        report, plan_schedule, planned_map, plan_log = _run_plan_check(folder)
        frozen_schedule = NATURAL / "design" / "planned_schedule.json"
        frozen_frame_map = NATURAL / "design" / "planned_frame_map.json"
        plan_hashes = {
            "schedule_identical_to_frozen": (
                (folder / "plan" / "planned_schedule.json").read_bytes()
                == frozen_schedule.read_bytes()
            ),
            "frame_map_identical_to_frozen": (
                (folder / "plan" / "planned_frame_map.json").read_bytes()
                == frozen_frame_map.read_bytes()
            ),
            "schedule_sha256": _sha(folder / "plan" / "planned_schedule.json"),
            "frozen_schedule_sha256": _sha(frozen_schedule),
            "frame_map_sha256": _sha(folder / "plan" / "planned_frame_map.json"),
            "frozen_frame_map_sha256": _sha(frozen_frame_map),
            "script_sha256": _sha(folder / "script.json"),
        }
        if not (plan_hashes["schedule_identical_to_frozen"]
                and plan_hashes["frame_map_identical_to_frozen"]):
            raise RuntimeError("Final script changed frozen plan output bytes")

        pose_paths = (
            POSE_DIR / "pose.json", POSE_DIR / "points.json",
            POSE_DIR / "analysis.json", POSE_DIR / "handoff.json",
        )
        pose_verification = verify_pose_binding(video, *pose_paths)
        pose_doc = _read_json(pose_paths[0])
        tracks = load_tracks(pose_doc)
        pose_report = analyze(pose_doc)
        cuts = _camera_cut_evidence(video, tracks)
        script = _read_json(folder / "script.json")
        attribution = _attribution()
        script["attribution_display"] = attribution["last_frames_text"]
        evidence = _event_pose_evidence(script, tracks, pose_report)
        crops = {
            moment["id"]: _hold_crop(tracks, moment)
            for moment in script["timeline"]["moments"]
        }
        schedule, segments = _prepare_schedule(
            script, report, plan_schedule, planned_map, tracks, crops,
        )
        schedule["source_positions"] = planned_map["source_position_per_output_frame"]
        sync = _sync_evidence(script, report, planned_map)
        recap_rect = _find_recap_card_rect(
            planned_map["source_position_per_output_frame"], segments, tracks,
        )
        schedule["recap_card_rect"] = recap_rect
        selected_ranges = [
            [int(moment["source_frames"][0]), int(moment["source_frames"][2])]
            for moment in schedule["moments"]
        ]
        knots, duplicate_audit = _load_unique_knots(
            video, pose_verification["source_sha256"], tracks.n, selected_ranges,
        )
        _write_json(folder / "duplicate-frame-audit.json", duplicate_audit)
        _write_json(folder / "plan" / "plan-hash-checks.json", plan_hashes)
        _write_final_maps(
            folder, plan_schedule, planned_map, segments, script, schedule, tracks.n,
            knots, duplicate_audit["manifest_sha256"], tracks.shots,
        )
        schedule_path = folder / "schedule.json"
        schedule = _read_json(schedule_path)
        audio_sync = _mix_audio(folder, len(planned_map["source_frame_per_output_frame"]) / FPS)
        render = _render_picture(
            video, folder, script, schedule, tracks, planned_map, recap_rect,
            knots, duplicate_audit,
        )
        _mux(folder / "picture.mp4", folder / "audio.m4a", folder / "natural.mp4")
        media = _final_media_check(folder, schedule["planned_duration_s"])
        pixel_audit = _run_pixel_audit(folder)
        _write_json(folder / "pixel-audit.json", pixel_audit)
        _write_reports(
            folder, video, pose_verification, tracks, pose_report, script, schedule,
            evidence, cuts, sync, pixel_audit, render, attribution,
        )
        independent_media = _run_independent_media_check(
            folder, schedule["planned_duration_s"],
        )
        media["independent_media_audit"] = independent_media
        audit, audit_log = _run_audit(folder)
        _write_json(folder / "media-audit.json", {
            "independent_media_check": independent_media,
            "internal_decode_check": media,
            "audio_duration_alignment": audio_sync,
            "pixel_audit": pixel_audit,
            "naturalness_audit": audit,
            "naturalness_audit_log": audit_log,
        })
        _write_json(folder / "run.json", {
            "command": "python -B -m coach.natural_review <video> <output>",
            "source_sha256": pose_verification["source_sha256"],
            "source_frames": tracks.n, "source_fps": tracks.fps,
            "full_source_inference": False,
            "pose_reused_after_hash_and_full_coverage_verification": True,
            "camera_cuts_reverified": cuts,
            "research_shortfalls": _read_json(
                NATURAL / "design" / "design_findings.json"
            )["honest_counts"],
            "script_sha256": plan_hashes["script_sha256"],
            "plan_hash_checks": plan_hashes,
            "plan_gate_count": len(list(_gate_rows(report["gates"]))),
            "plan_failed_gates": 0,
            "planned_frames": len(planned_map["source_frame_per_output_frame"]),
            "planned_duration_s": schedule["planned_duration_s"],
            "picture_model": "unique_blend_v1",
            "duplicate_frame_audit": duplicate_audit,
            "render": render,
            "audio_sync": audio_sync,
            "final_media": media,
            "sync_events": sync,
            "pixel_audit": pixel_audit,
            "render_elapsed_s": render["render_elapsed_s"],
        })
        after = subprocess.run([
            str(NATURAL / ".venv" / "Scripts" / "python.exe"), "-B", str(verifier),
        ], capture_output=True, text=True, encoding="utf-8")
        (folder / "inputs-after.log").write_text(after.stdout + after.stderr, "utf-8")
        if after.returncode or "ALL OK: 34 files" not in after.stdout:
            raise RuntimeError("Frozen inputs did not verify after render")
        verification = _verification(
            folder, render, media, audit, plan_hashes, evidence, sync, pixel_audit, cuts,
        )
        verification["checks"]["inputs_after"] = {
            "status": "VERIFIED", "log": str((folder / "inputs-after.log").resolve()),
            "measured": "ALL OK: 34 files",
        }
        _write_json(folder / "verification.json", verification)
        return {
            "output": str(folder.resolve()), "video": str((folder / "natural.mp4").resolve()),
            "sha256": _sha(folder / "natural.mp4"),
            "verification": verification["status"],
            "plan_log": plan_log,
        }
    finally:
        release_cpu_lock(lock, token)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, nargs="?", default=DEFAULT_VIDEO)
    parser.add_argument("out", type=Path, nargs="?", default=DEFAULT_OUT)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args(argv)
    if args.finalize:
        _recover_existing_outputs(args.video, args.out)
        print(json.dumps(_finalize(args.out), ensure_ascii=True, indent=2))
        return
    print(json.dumps(export_natural(args.video, args.out), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
