"""Export an evidence-bound, chaptered portrait coaching short."""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import subprocess
import tempfile
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .app.backend import PROJECT, FastBackend, sha256_file
from .app.pipeline import analyze_video
from .coaching_corpus.corpus_learn import HEDGE_ZH
from .complete_review import VOICE, _voice, _wrap_zh
from .export import _tools
from .player_input import player_inputs
from .schema import Role
from .short_review import SAFE, _dump, _srt_time, source_credit
from .tracks import JOINT_INDEX, load_tracks

FPS = 30
WIDTH, HEIGHT = 1080, 1920
VIEW = (64, 385, 1016, 1215)
TACTICS_VIEW = (64, 385, 636, 1215)
CAPTION_XY = (78, 1310)
CAPTION_WIDTH = 810
MAX_FREEZE_S = 1.0
MAX_PAUSED_SHARE = 0.35
SPEECH_SPEED = 1.08
COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
HAND_GATE = 0.50
CHANGED_SOURCES = (
    "coach/README.md",
    "coach/editorial_short.py",
    "coach/tests/test_editorial_short.py",
)

VOICE_TEXT = {
    "legs": "右脚跨出，前膝弯下。",
    "core": "上身随跨步倾，蹬回时回正。",
    "tactics": "回到半场中部，再准备四边移动。",
    "arms": "左手先抬再收，右臂后挥。",
    "recap": "右脚跨，前膝弯；两臂接，上身回正，再回到中部。",
}

RECAP_LINES = (
    "腿｜右脚跨出，前膝屈曲",
    "手臂｜左手先收，右臂上挥",
    "核心｜上身倾斜，再回正",
    "战术｜回到中部，准备下一步",
)


@dataclass(frozen=True)
class Clip:
    start: int
    end: int
    speed: float = 1.0
    focus: str | None = None
    overlay: str | None = None
    phase: str = "motion"
    hold_s: float = 0.0
    recap_line: int = 0


@dataclass(frozen=True)
class Chapter:
    key: str
    title: str
    narration: str
    shot: tuple[int, int]
    clips: tuple[Clip, ...]
    evidence: dict


def _role(tracks, name):
    return tracks.players[Role(name)]


def _visible_conf(player, frame, names):
    if not player.measured[frame] or not all(player.joint_visible(name)[frame] for name in names):
        return None
    return min(float(player.conf[frame, JOINT_INDEX[name]]) for name in names)


def _safe_point(player, frame):
    names = ("left_ankle", "right_ankle")
    confidence = _visible_conf(player, frame, names)
    if confidence is None:
        return None
    return {
        "xy_px": ((player.joint(names[0])[frame] + player.joint(names[1])[frame]) / 2).astype(float),
        "confidence": confidence,
        "source_frame": int(frame),
    }


def court_projection(doc, tracks):
    lines = np.asarray(doc.get("court_lines_normalized"), dtype=np.float32)
    if lines.shape != (4, 2) or not np.isfinite(lines).all():
        raise ValueError("Court minimap needs four measured court-corner points")
    corners = (lines * np.asarray([tracks.width, tracks.height], dtype=np.float32)).reshape(4, 1, 2)
    court = np.asarray([
        [0.0, 0.0], [COURT_WIDTH_M, 0.0],
        [COURT_WIDTH_M, COURT_LENGTH_M], [0.0, COURT_LENGTH_M],
    ], dtype=np.float32).reshape(4, 1, 2)
    transform = cv2.getPerspectiveTransform(corners, court)
    if not np.isfinite(transform).all():
        raise ValueError("Court homography is invalid")

    def project(xy_px):
        mapped = cv2.perspectiveTransform(
            np.asarray(xy_px, dtype=np.float32).reshape(-1, 1, 2), transform
        ).reshape(-1, 2)
        return mapped / np.asarray([COURT_WIDTH_M, COURT_LENGTH_M], dtype=np.float32)

    return corners.reshape(4, 2), transform, project


def court_samples(doc, tracks, start_frame, end_frame):
    corners, transform, project = court_projection(doc, tracks)
    positions = {}
    for role in ("near", "far"):
        player = _role(tracks, role)
        rows = []
        for frame in range(start_frame, end_frame + 1):
            point = _safe_point(player, frame)
            if point is None:
                continue
            xy = project([point["xy_px"]])[0]
            row = {
                "frame": frame,
                "xy": [float(xy[0]), float(xy[1])],
                "confidence": float(point["confidence"]),
            }
            if np.isfinite(xy).all():
                positions[(role, frame)] = row
                rows.append(row)
        if not rows:
            raise RuntimeError(f"No two-ankle court measurements for {role} in {start_frame}-{end_frame}")
    return corners, transform, positions


def _median_position(positions, role, start, end):
    rows = [row for (candidate, frame), row in positions.items()
            if candidate == role and start <= frame <= end]
    if not rows:
        raise RuntimeError(f"No measured court positions for {role} in {start}-{end}")
    return {
        "xy": np.median(np.asarray([row["xy"] for row in rows]), axis=0).tolist(),
        "confidence": float(np.median([row["confidence"] for row in rows])),
        "n": len(rows),
        "source_start_frame": start,
        "source_end_frame": end,
    }


def core_measurement(tracks, role, frame):
    player = _role(tracks, role)
    names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    confidence = _visible_conf(player, frame, names)
    if confidence is None:
        raise ValueError(f"Core landmarks are not measured at frame {frame}")
    left_shoulder, right_shoulder, left_hip, right_hip = (
        player.joint(name)[frame].astype(float) for name in names
    )
    shoulders = (left_shoulder + right_shoulder) / 2
    hips = (left_hip + right_hip) / 2
    trunk = shoulders - hips
    if np.linalg.norm(trunk) <= 1e-6:
        raise ValueError(f"Degenerate trunk vector at frame {frame}")
    trunk_tilt = abs(math.degrees(math.atan2(float(trunk[0]), float(-trunk[1]))))
    shoulder_tilt = abs(math.degrees(math.atan2(
        float(right_shoulder[1] - left_shoulder[1]),
        float(right_shoulder[0] - left_shoulder[0]),
    )))
    hip_tilt = abs(math.degrees(math.atan2(
        float(right_hip[1] - left_hip[1]),
        float(right_hip[0] - left_hip[0]),
    )))
    return {
        "frame": int(frame),
        "trunk_tilt_2d_deg": trunk_tilt,
        "shoulder_line_tilt_2d_deg": shoulder_tilt,
        "hip_line_tilt_2d_deg": hip_tilt,
        "confidence": confidence,
        "keypoints": {
            name: player.joint(name)[frame].astype(float).tolist() for name in names
        },
    }


def _nearest_measured(player, frame, shot):
    choices = [
        candidate for candidate in range(max(shot[0], frame - 3), min(shot[1], frame + 3) + 1)
        if player.measured[candidate] and np.isfinite(player.bbox[candidate]).all()
    ]
    if not choices:
        raise ValueError(f"No measured player box near source frame {frame}")
    return min(choices, key=lambda candidate: abs(candidate - frame))


def crop_geometry(bbox, width, height, target_aspect=(VIEW[2] - VIEW[0]) / (VIEW[3] - VIEW[1])):
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if not np.isfinite(bbox).all() or x2 <= x1 or y2 <= y1:
        raise ValueError("Focus crop needs a finite player box")
    crop_h = (y2 - y1) * 1.75
    crop_w = crop_h * target_aspect
    if crop_w < (x2 - x1) * 1.15:
        crop_w = (x2 - x1) * 1.15
        crop_h = crop_w / target_aspect
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(center_x - crop_w / 2, 0), width - crop_w)
    top = min(max(center_y - crop_h / 2, 0), height - crop_h)
    rect = (int(math.floor(left)), int(math.floor(top)),
            int(math.ceil(left + crop_w)), int(math.ceil(top + crop_h)))
    if not (rect[0] <= x1 <= x2 <= rect[2] and rect[1] <= y1 <= y2 <= rect[3]):
        raise ValueError("Portrait crop cannot contain the full focus-player box")
    scale = min((VIEW[2] - VIEW[0]) / (rect[2] - rect[0]),
                (VIEW[3] - VIEW[1]) / (rect[3] - rect[1]))
    return rect, float(scale)


def _shot_for(tracks, frame):
    found = [shot for shot in tracks.shots if shot[0] <= frame <= shot[1]]
    if len(found) != 1:
        raise ValueError(f"Frame {frame} is not in exactly one camera shot")
    return found[0]


def _min_conf(player, frame, names):
    confidence = _visible_conf(player, frame, names)
    if confidence is None:
        raise ValueError(f"Required pose joints are missing at frame {frame}")
    return confidence


def _select_arm_sequence(report, tracks):
    player = _role(tracks, "far")
    if player.hand != "right":
        raise RuntimeError("The arm chapter needs evidence-backed right-handedness for the far player")
    events = sorted(
        (event for event in report.events
         if event.kind == "swing" and event.player == Role.FAR and event.data.get("overhead")),
        key=lambda event: event.frame,
    )
    for event in events:
        start = max(_shot_for(tracks, event.frame)[0], int(event.frame) - 20)
        end = min(_shot_for(tracks, event.frame)[1], int(event.frame) + 15)
        head = player.joint("nose")
        left_candidates = [
            frame for frame in range(start, max(start, event.frame - 3))
            if player.measured[frame]
            and player.joint_visible("left_wrist")[frame]
            and player.joint_visible("nose")[frame]
            and abs(float(player.joint("left_wrist")[frame, 1] - head[frame, 1]))
            <= 0.08 * float(player.bbox[frame, 3] - player.bbox[frame, 1])
        ]
        right_candidates = [
            frame for frame in range(max(start, event.frame - 2), event.frame + 1)
            if player.measured[frame]
            and player.joint_visible("right_wrist")[frame]
            and player.joint_visible("nose")[frame]
            and player.joint("right_wrist")[frame, 1] < player.joint("nose")[frame, 1]
        ]
        if not left_candidates or not right_candidates:
            continue
        left_peak = min(left_candidates, key=lambda frame: float(player.joint("left_wrist")[frame, 1]))
        right_peak = min(right_candidates, key=lambda frame: float(player.joint("right_wrist")[frame, 1]))
        if right_peak <= left_peak:
            continue
        left_return_candidates = []
        for frame in range(left_peak + 1, right_peak):
            confidence = _visible_conf(
                player, frame, ("left_shoulder", "right_shoulder", "left_elbow", "left_wrist")
            )
            if confidence is None:
                continue
            shoulder_y = float(np.mean([
                player.joint("left_shoulder")[frame, 1],
                player.joint("right_shoulder")[frame, 1],
            ]))
            height = float(player.bbox[frame, 3] - player.bbox[frame, 1])
            if float(player.joint("left_wrist")[frame, 1] - shoulder_y) >= 0.08 * height:
                left_return_candidates.append(frame)
        if not left_return_candidates:
            continue
        left_return = left_return_candidates[0]
        left_conf = _min_conf(player, left_peak, ("left_shoulder", "left_elbow", "left_wrist"))
        left_return_conf = _min_conf(
            player, left_return, ("left_shoulder", "left_elbow", "left_wrist")
        )
        right_conf = _min_conf(player, right_peak, ("right_shoulder", "right_elbow", "right_wrist"))
        if left_conf < HAND_GATE or left_return_conf < HAND_GATE or right_conf < HAND_GATE:
            continue
        return {
            "start": start, "key": int(event.frame), "end": end,
            "left_peak": left_peak, "left_return": left_return, "right_peak": right_peak,
            "left_confidence": left_conf, "left_return_confidence": left_return_conf,
            "right_confidence": right_conf,
            "swing_event": event.to_dict() if hasattr(event, "to_dict") else {
                "frame": event.frame, "data": dict(event.data),
            },
            "shot": list(_shot_for(tracks, event.frame)),
        }
    raise RuntimeError("No visible, confidence-gated left-then-right overhead sequence was found")


def build_story(doc, report):
    tracks = load_tracks(doc)
    near, far = _role(tracks, "near"), _role(tracks, "far")
    if near.hand != "right" or far.hand != "right":
        raise RuntimeError("Clip 04 handedness must be explicit and evidence-backed for both players")
    cycles = sorted(
        (event for event in report.events
         if event.kind == "lunge" and event.player == Role.NEAR and event.data.get("knee_stride")
         and event.data.get("lead_right")),
        key=lambda event: (-float(event.data.get("lead_knee_flexion_deg", 0)), event.frame),
    )
    if not cycles:
        raise RuntimeError("No measured right-led lunge cycle supports the leg and core chapters")
    cycle = cycles[0]
    cycle_data = cycle.data
    plant = int(cycle_data["plant_frame"])
    bottom = int(cycle_data["bottom_frame"])
    return_frame = int(cycle_data["end_frame"])
    lunge_shot = _shot_for(tracks, bottom)
    next_near_swing = sorted(
        (event for event in report.events
         if event.kind == "swing" and event.player == Role.NEAR and event.frame > return_frame
         and event.frame <= min(lunge_shot[1], return_frame + round(1.6 * tracks.fps))),
        key=lambda event: event.frame,
    )
    settle_end = min(lunge_shot[1], int(next_near_swing[0].data["end"])) if next_near_swing else return_frame
    leg_start = max(lunge_shot[0], plant - round(0.6 * tracks.fps))
    leg_end = min(lunge_shot[1], bottom + 2)
    if not leg_start < bottom < leg_end:
        raise RuntimeError("The selected lunge lacks action on both sides of its bottom frame")
    knee_flexion = float(cycle_data["lead_knee_flexion_deg"])
    foot_conf = _min_conf(near, plant, ("left_ankle", "right_ankle"))
    knee_conf = _min_conf(near, bottom, ("right_hip", "right_knee", "right_ankle"))
    leg_evidence = {
        "claim": "右脚跨出，前膝弯下。",
        "teaching_principle": "练习时把跨步、屈膝、上身下降和回动连成一个动作。",
        "plant_frame": plant, "bottom_frame": bottom, "return_frame": return_frame,
        "next_move_end_frame": settle_end,
        "lead_side": "right", "knee_flexion_2d_deg": knee_flexion,
        "stride_ratio_of_stature": float(cycle_data["stride_ratio"]),
        "plant_confidence": foot_conf, "bottom_confidence": knee_conf,
        "n": 1, "shot": list(lunge_shot),
    }

    core_candidates = []
    for frame in range(max(lunge_shot[0], bottom - 12), min(lunge_shot[1], return_frame + 12) + 1):
        try:
            core_candidates.append(core_measurement(tracks, "near", frame))
        except ValueError:
            continue
    baseline = [row for row in core_candidates if row["frame"] < bottom and row["trunk_tilt_2d_deg"] < 10]
    leaning = [row for row in core_candidates if bottom <= row["frame"] <= return_frame]
    returned = [row for row in core_candidates
                if row["frame"] > return_frame and row["trunk_tilt_2d_deg"] < 10]
    if not baseline or not leaning or not returned:
        raise RuntimeError("The core chapter needs measured baseline, lunge, and return poses")
    core = [
        min(baseline, key=lambda row: (abs(row["trunk_tilt_2d_deg"]), -row["frame"])),
        max(leaning, key=lambda row: row["trunk_tilt_2d_deg"]),
        min(returned, key=lambda row: (row["trunk_tilt_2d_deg"], row["frame"])),
    ]
    core_frames = [row["frame"] for row in core]
    if not (core[0]["trunk_tilt_2d_deg"] < 10
            and core[1]["trunk_tilt_2d_deg"] >= 25
            and core[1]["shoulder_line_tilt_2d_deg"] > core[1]["hip_line_tilt_2d_deg"] + 10
            and core[2]["trunk_tilt_2d_deg"] < 10):
        raise RuntimeError("The selected 2D trunk sequence does not support a visible lean-and-return story")
    core_evidence = {
        "claim": "跨到底时上身朝跨步一侧倾，肩线比髋线更斜；回位时上身逐渐回正。",
        "teaching_principle": "观察肩线、髋线与上身在动作前后的相对变化；不据此推断肌肉、受力或三维旋转。",
        "frames": core, "shot": list(lunge_shot),
    }

    arm = _select_arm_sequence(report, tracks)
    arm["claim"] = "另一回合中，非持拍手先抬起并开始回收，持拍臂顺势上挥。"
    arm["teaching_principle"] = "观察两只手臂在上举动作中的先后关系；不据此判断触球或羽毛球轨迹。"

    tactics_start = min(lunge_shot[1], return_frame + round(0.7 * tracks.fps))
    tactics_end = min(lunge_shot[1], tactics_start + round(3.0 * tracks.fps))
    if tactics_end - tactics_start < round(1.0 * tracks.fps):
        raise RuntimeError("The same-shot court-position sequence is too short to show a recovery")
    position_scan_start = max(lunge_shot[0], int(cycle.frame) - round(0.8 * tracks.fps))
    before_swing = sorted(
        (event for event in report.events
         if event.kind == "swing" and event.player == Role.NEAR and event.frame < cycle.frame
         and event.frame >= lunge_shot[0]),
        key=lambda event: event.frame,
    )
    if not before_swing:
        raise RuntimeError("No earlier near-player action supports the court-path starting point")
    initial_frame = min(tactics_start, int(before_swing[-1].frame) + round(0.3 * tracks.fps))
    corners, homography, positions = court_samples(
        doc, tracks, min(position_scan_start, initial_frame), tactics_end,
    )
    initial = positions.get(("near", initial_frame))
    lunge_point = positions.get(("near", bottom))
    near_settled = _median_position(positions, "near", tactics_start, tactics_end)
    far_settled = _median_position(positions, "far", tactics_start, tactics_end)
    if initial is None or lunge_point is None:
        raise RuntimeError("Court path anchors need measured two-ankle positions")
    if not (near_settled["xy"][1] >= 0.65 and far_settled["xy"][1] <= 0.20):
        raise RuntimeError("Court positions do not support the stated middle-versus-back-court observation")
    if not (0.25 < near_settled["xy"][0] < 0.5):
        raise RuntimeError("The near player is not measured around the backhand-side half-court middle")
    tactics = {
        "claim": "她回到半场中部、稍偏反手；对手在后场。",
        "teaching_principle": "单打训练时，把回位和下一次向四边移动连起来。",
        "source_start_frame": tactics_start, "source_end_frame": tactics_end,
        "initial_frame": initial_frame, "lunge_frame": bottom,
        "initial_position": initial, "lunge_position": lunge_point,
        "near_settled": near_settled, "far_settled": far_settled,
        "court_corners_px": corners.astype(float).tolist(),
        "homography_to_normalized_court": homography.astype(float).tolist(),
        "court_position_by_frame": {
            f"{role}:{frame}": row for (role, frame), row in positions.items()
            if tactics_start <= frame <= tactics_end
        },
        "shot": list(lunge_shot),
    }

    arm_shot = tuple(arm["shot"])
    core_start = core_frames[1]
    core_end = tactics_start - 1
    if core_end <= core_start or leg_end >= core_start:
        raise RuntimeError("The lunge-to-core edit needs consecutive, non-overlapping source windows")
    return tracks, {
        "evidence": {"legs": leg_evidence, "core": core_evidence, "arms": arm, "tactics": tactics},
        "chapters": [
            Chapter(
                "legs", "腿", VOICE_TEXT["legs"], lunge_shot,
                (Clip(leg_start, leg_end, 0.35, "near", "legs"),),
                leg_evidence,
            ),
            Chapter(
                "core", "核心", VOICE_TEXT["core"], lunge_shot,
                (Clip(core_start, core_end, 0.32, "near", "core"),),
                core_evidence,
            ),
            Chapter(
                "tactics", "战术", VOICE_TEXT["tactics"], lunge_shot,
                (Clip(tactics_start, tactics_end, 1.0),),
                tactics,
            ),
            Chapter(
                "arms", "手臂", VOICE_TEXT["arms"], arm_shot,
                (
                    Clip(max(arm_shot[0], arm["left_peak"] - round(0.5 * tracks.fps)),
                         arm["left_return"], 0.50, "far", "arms-free"),
                    Clip(arm["left_return"] + 1, arm["right_peak"] - 1, 0.50, "far"),
                    Clip(arm["right_peak"], arm["end"], 0.50, "far", "arms-racket"),
                ),
                arm,
            ),
            Chapter(
                "recap", "收尾总结", VOICE_TEXT["recap"], arm_shot,
                (Clip(arm["end"], arm["end"], phase="summary", hold_s=5.5, recap_line=4),),
                {"claim": "用四行收束腿、手臂、核心与战术观察；不重播动作。",
                 "lines": list(RECAP_LINES)},
            ),
        ],
        "tracks": tracks,
        "court_positions": positions,
        "tactics": tactics,
    }


def _clip_frames(clip):
    if clip.phase in ("hold", "summary"):
        return [int(clip.start)] * max(1, round(clip.hold_s * FPS))
    count = max(1, math.ceil((clip.end - clip.start + 1) / clip.speed))
    return [min(clip.end, clip.start + math.floor(index * clip.speed)) for index in range(count)]


def make_schedule(story, speech_durations, tracks):
    chapters = story["chapters"]
    if len(chapters) != 5 or [chapter.key for chapter in chapters] != [
            "legs", "core", "tactics", "arms", "recap"]:
        raise ValueError("Editorial short needs one forward passage, another-rally arms, and a recap")
    frame_map, states, captions, chapter_rows = [], [], [], []
    positions = story["court_positions"]
    for chapter, speech_s in zip(chapters, speech_durations):
        if HEDGE_ZH.search(chapter.narration):
            raise ValueError(f"Spoken line contains a hedge: {chapter.key}")
        planned = []
        for clip_index, clip in enumerate(chapter.clips):
            if clip.speed <= 0:
                raise ValueError("Playback speed must be positive")
            if clip.phase not in ("motion", "hold", "summary"):
                raise ValueError(f"Unsupported editorial phase: {clip.phase}")
            if clip.phase == "motion" and not clip.start < clip.end:
                raise ValueError(f"{chapter.key} motion clip needs source action on both sides")
            if clip.phase == "hold" and clip.hold_s > MAX_FREEZE_S:
                raise ValueError("Freeze exceeds the one-second editorial limit")
            if clip.phase == "summary" and chapter.key != "recap":
                raise ValueError("Only the closing recap may use the static summary card")
            if _shot_for(tracks, clip.start) != _shot_for(tracks, clip.end):
                raise ValueError(f"{chapter.key} clip crosses a camera cut")
            planned.extend((clip, frame, clip_index) for frame in _clip_frames(clip))
        target_frames = math.ceil((speech_s + 0.24) * FPS)
        if len(planned) < target_frames:
            raise ValueError(
                f"{chapter.key} narration does not fit its selected footage; "
                "shorten the line or choose a longer forward window"
            )
        motion_frames = [
            frame for clip, frame, _ in planned if clip.phase == "motion"
        ]
        if any(right < left for left, right in zip(motion_frames, motion_frames[1:])):
            raise ValueError(f"{chapter.key} replays or reverses its source motion")
        start_s = len(frame_map) / FPS
        clip_offsets = {}
        visible_recap_lines = 0
        for clip, source_frame, clip_index in planned:
            output_index = len(frame_map)
            phase = clip.phase
            frame_map.append(int(source_frame))
            if clip.recap_line:
                visible_recap_lines = max(visible_recap_lines, clip.recap_line)
            if phase == "summary":
                visible_recap_lines = len(RECAP_LINES)
            pos = {
                role: positions.get((role, int(source_frame)))
                for role in ("near", "far")
            } if chapter.key == "tactics" else {}
            states.append({
                "chapter": chapter.key,
                "title": chapter.title,
                "phase": phase,
                "shot": list(_shot_for(tracks, source_frame)),
                "focus": clip.focus,
                "overlay": clip.overlay,
                "speed": clip.speed,
                "clip_index": clip_index,
                "recap_lines_visible": visible_recap_lines,
                "court_positions": pos,
                "source_frame": int(source_frame),
            })
            clip_offsets.setdefault(clip_index, [output_index, output_index + 1])
            clip_offsets[clip_index][1] = output_index + 1
        end_s = len(frame_map) / FPS
        speech_start = start_s + 0.10
        captions.append({
            "chapter": chapter.key, "title": chapter.title, "text": chapter.narration,
            "start_s": speech_start, "end_s": speech_start + speech_s,
            "speech_start_s": speech_start, "speech_end_s": speech_start + speech_s,
        })
        source_ranges = []
        for clip_index, clip in enumerate(chapter.clips):
            span = clip_offsets.get(clip_index)
            if span is None:
                continue
            source_ranges.append({
                "start_frame": clip.start, "end_frame": clip.end,
                "source_frame": clip.start if clip.phase in ("hold", "summary") else None,
                "speed": clip.speed, "phase": clip.phase,
                "focus": clip.focus, "overlay": clip.overlay,
                "output_start_s": span[0] / FPS,
                "output_end_s": span[1] / FPS,
                "source_shot": list(_shot_for(tracks, clip.start)),
            })
        chapter_rows.append({
            "key": chapter.key, "title": chapter.title,
            "source_shot": list(chapter.shot),
            "source_shots": [list(shot) for shot in sorted({
                _shot_for(tracks, clip.start) for clip, _, _ in planned
            })],
            "source_ranges": source_ranges,
            "output_start_s": start_s, "output_end_s": end_s,
            "speech_start_s": speech_start, "speech_end_s": speech_start + speech_s,
            "evidence": chapter.evidence,
        })
    paused_frames = sum(state["phase"] in ("hold", "summary") for state in states)
    summary_frames = sum(state["phase"] == "summary" for state in states)
    longest_hold_frames = 0
    current_hold_frames = 0
    for state in states:
        current_hold_frames = current_hold_frames + 1 if state["phase"] == "hold" else 0
        longest_hold_frames = max(longest_hold_frames, current_hold_frames)
    longest_hold = longest_hold_frames / FPS
    duration = len(frame_map) / FPS
    paused_s = paused_frames / FPS
    if duration > 45:
        raise RuntimeError(f"Editorial short duration {duration:.2f}s exceeds 45s")
    if paused_s / duration > MAX_PAUSED_SHARE:
        raise RuntimeError("Editorial short exceeds the 35% still-frame and summary-card limit")
    if longest_hold > MAX_FREEZE_S:
        raise RuntimeError("Editorial short exceeds the one-second per-chapter hold limit")
    for row in captions:
        if row["end_s"] > chapter_rows[
                [chapter["key"] for chapter in chapter_rows].index(row["chapter"])
        ]["output_end_s"] - 0.10:
            raise RuntimeError(f"{row['chapter']} narration does not fit its moving edit")
    return {
        "fps": FPS, "width": WIDTH, "height": HEIGHT,
        "planned_duration_s": duration, "planned_output_frames": len(frame_map),
        "moving_s": duration - paused_s, "paused_s": paused_s,
        "summary_s": summary_frames / FPS,
        "paused_share": paused_s / duration, "longest_hold_s": longest_hold,
        "chapters": chapter_rows, "captions": captions,
        "recap_lines": list(RECAP_LINES), "frame_map": frame_map, "states": states,
        "safe_rect": list(SAFE), "caption_rect": [CAPTION_XY[0], CAPTION_XY[1], 892, 1495],
    }


def _crop_for_state(raw, tracks, frame, focus, shot, full_crop):
    if focus is None:
        return full_crop, 1.0
    player = _role(tracks, focus)
    measured = _nearest_measured(player, frame, shot)
    return crop_geometry(player.bbox[measured], tracks.width, tracks.height)


def _full_crop(tracks, shot):
    corners = np.asarray(tracks.court.corners_px, dtype=float) if tracks.court else None
    if corners is None:
        return (0, 0, tracks.width, tracks.height)
    boxes = []
    for player in tracks.players.values():
        boxes.extend(player.bbox[shot[0]:shot[1] + 1][player.measured[shot[0]:shot[1] + 1]])
    boxes = np.asarray(boxes, dtype=float)
    valid = boxes.ndim == 2 and boxes.shape[1] == 4 and len(boxes)
    min_x = min(float(corners[:, 0].min()), float(boxes[:, 0].min()) if valid else tracks.width)
    max_x = max(float(corners[:, 0].max()), float(boxes[:, 2].max()) if valid else 0)
    min_y = min(float(corners[:, 1].min()) - 175, float(boxes[:, 1].min()) if valid else tracks.height)
    max_y = max(float(corners[:, 1].max()) + 55, float(boxes[:, 3].max()) if valid else 0)
    left = max(0, int(math.floor(min_x - 55)))
    right = min(tracks.width, int(math.ceil(max_x + 55)))
    top = max(0, int(math.floor(min_y - 20)))
    bottom = min(tracks.height, int(math.ceil(max_y + 20)))
    return left, top, right, bottom


def _draw_joint_chain(image, player, frame, names, color, width):
    for left, right in zip(names, names[1:]):
        if player.joint_visible(left)[frame] and player.joint_visible(right)[frame]:
            a, b = [tuple(np.rint(player.joint(name)[frame]).astype(int)) for name in (left, right)]
            cv2.line(image, a, b, (10, 20, 25), width + 4, cv2.LINE_AA)
            cv2.line(image, a, b, color, width, cv2.LINE_AA)
    for name in names:
        if player.joint_visible(name)[frame]:
            point = tuple(np.rint(player.joint(name)[frame]).astype(int))
            cv2.circle(image, point, width + 2, (10, 20, 25), -1, cv2.LINE_AA)
            cv2.circle(image, point, width, color, -1, cv2.LINE_AA)


def _overlay(raw, tracks, frame, role, kind):
    if not role or not kind:
        return raw
    player = _role(tracks, role)
    if not player.measured[frame]:
        return raw
    image = raw.copy()
    box_h = max(1.0, float(player.bbox[frame, 3] - player.bbox[frame, 1]))
    width = max(3, round(box_h / 55))
    if kind == "legs":
        for side, color in (("left", (255, 190, 60)), ("right", (70, 235, 220))):
            names = tuple(f"{side}_{joint}" for joint in ("hip", "knee", "ankle"))
            _draw_joint_chain(image, player, frame, names, color, width)
        if all(player.joint_visible(f"{side}_ankle")[frame] for side in ("left", "right")):
            a, b = [tuple(np.rint(player.joint(f"{side}_ankle")[frame]).astype(int))
                    for side in ("left", "right")]
            cv2.line(image, a, b, (255, 245, 210), max(2, width // 2), cv2.LINE_AA)
    elif kind == "core":
        for left, right, color in (
            ("left_shoulder", "right_shoulder", (70, 235, 220)),
            ("left_hip", "right_hip", (255, 190, 60)),
        ):
            if player.joint_visible(left)[frame] and player.joint_visible(right)[frame]:
                a, b = [tuple(np.rint(player.joint(name)[frame]).astype(int)) for name in (left, right)]
                cv2.line(image, a, b, (10, 20, 25), width + 4, cv2.LINE_AA)
                cv2.line(image, a, b, color, width, cv2.LINE_AA)
                cv2.circle(image, a, width, color, -1, cv2.LINE_AA)
                cv2.circle(image, b, width, color, -1, cv2.LINE_AA)
    elif kind in ("arms", "arms-free", "arms-racket"):
        sides = {
            "arms": ("left", "right"),
            "arms-free": ("left",),
            "arms-racket": ("right",),
        }[kind]
        for side in sides:
            color = (255, 190, 60) if side == "left" else (70, 235, 220)
            names = tuple(f"{side}_{joint}" for joint in ("shoulder", "elbow", "wrist"))
            confidence = _visible_conf(player, frame, names)
            if confidence is not None and confidence >= HAND_GATE:
                _draw_joint_chain(image, player, frame, names, color, width)
    else:
        raise ValueError(f"Unknown editorial overlay: {kind}")
    return image


def _fit_source(raw, crop, viewport=VIEW):
    left, top, right, bottom = crop
    view_w, view_h = viewport[2] - viewport[0], viewport[3] - viewport[1]
    source = raw[top:bottom, left:right]
    scale = min(view_w / source.shape[1], view_h / source.shape[0])
    size = (max(1, round(source.shape[1] * scale)), max(1, round(source.shape[0] * scale)))
    resized = cv2.resize(source, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    pane = np.zeros((view_h, view_w, 3), np.uint8)
    x, y = (view_w - size[0]) // 2, (view_h - size[1]) // 2
    pane[y:y + size[1], x:x + size[0]] = resized
    return pane, scale


def _draw_minimap(draw, state, story, chapter_row):
    if state["chapter"] != "tactics":
        return
    x0, y0, x1, y1 = 672, 590, 890, 1112
    draw.rounded_rectangle((x0 - 12, y0 - 20, x1 + 12, y1 + 18), radius=20, fill="#07121f")
    court_x, court_y, court_w, court_h = x0 + 34, y0 + 26, 142, 480
    draw.rectangle((court_x, court_y, court_x + court_w, court_y + court_h),
                   fill="#175f54", outline="#e5f0d7", width=4)
    draw.line((court_x, court_y + court_h / 2, court_x + court_w, court_y + court_h / 2),
              fill="#f3f5e8", width=3)
    current = state.get("court_positions") or {}

    def point_xy(row):
        if not row:
            return None
        nx, ny = row["xy"]
        return (court_x + nx * court_w, court_y + ny * court_h)

    initial = point_xy(story["tactics"]["initial_position"])
    lunge = point_xy(story["tactics"]["lunge_position"])
    settled = point_xy(story["tactics"]["near_settled"])
    trail = [point for point in (initial, lunge, settled) if point is not None]
    if len(trail) >= 2:
        draw.line(trail, fill="#ff7b78", width=8, joint="curve")
    far = point_xy(current.get("far") or story["tactics"]["far_settled"])
    near = point_xy(current.get("near"))
    if far:
        draw.ellipse((far[0] - 12, far[1] - 12, far[0] + 12, far[1] + 12),
                     fill="#ffc857", outline="#14212b", width=3)
    if near:
        draw.ellipse((near[0] - 13, near[1] - 13, near[0] + 13, near[1] + 13),
                     fill="#ff7b78", outline="#14212b", width=3)
    title_font = story["fonts"][3]
    draw.text((court_x, y0), "场上位置", font=title_font, fill="#ffffff")
    draw.text((court_x, court_y + court_h + 18), "对手", font=title_font, fill="#ffc857")
    draw.text((court_x + 78, court_y + court_h + 18), "近端路径", font=title_font, fill="#ffaaa1")


def _wrap_editorial(draw, text, font, width):
    lines = _wrap_zh(draw, text, font, max(1, width - 64))
    for index in range(1, len(lines)):
        while lines[index] and lines[index][0] in "，。、；：？！":
            lines[index - 1] += lines[index][0]
            lines[index] = lines[index][1:]
    return [line for line in lines if line]


def _draw_text(draw, text, xy, font, fill, width=CAPTION_WIDTH, line_h=62):
    x, y = xy
    boxes = []
    for line in _wrap_editorial(draw, text, font, width):
        box = draw.textbbox((x, y), line, font=font)
        if not (SAFE[0] <= box[0] and SAFE[1] <= box[1]
                and box[2] <= SAFE[2] and box[3] <= SAFE[3]):
            raise RuntimeError(f"On-screen text enters a platform UI zone: {line!r} {box}")
        draw.text((x, y), line, font=font, fill=fill, stroke_width=1, stroke_fill="#0b1014")
        boxes.append(list(box))
        y += line_h
    return boxes


def render_frame(raw, tracks, state, schedule, story, fonts, credit_text, source_time):
    chapter = next(row for row in schedule["chapters"] if row["key"] == state["chapter"])
    source_frame = state["source_frame"]
    heading, subhead, caption_font, small = fonts[:4]
    if state["phase"] == "summary":
        image = Image.new("RGB", (WIDTH, HEIGHT), "#111d22")
        draw = ImageDraw.Draw(image)
        boxes = _draw_text(draw, "这一拍，记住四件事", (78, 285), heading, "#f5f8ff")
        draw.rounded_rectangle((78, 405, 170, 415), radius=5, fill="#58e5dc")
        recap_y = 520
        for line in RECAP_LINES:
            row_boxes = _draw_text(
                draw, line, (78, recap_y), fonts[4], "#f2f6ef",
                width=810, line_h=60,
            )
            boxes += row_boxes
            recap_y += max(110, len(row_boxes) * 60 + 35)
        caption = next((row for row in schedule["captions"]
                        if row["chapter"] == state["chapter"]
                        and row["start_s"] <= source_time < row["end_s"]), None)
        if caption:
            boxes += _draw_text(draw, caption["text"], CAPTION_XY, caption_font,
                                "#ffffff", width=CAPTION_WIDTH, line_h=58)
        if source_time >= chapter["output_end_s"] - 0.7:
            boxes += _draw_text(draw, credit_text, (78, 1410), fonts[3], "#c8d6dd",
                                width=810, line_h=32)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), boxes, {
            "source_crop": [0, 0, tracks.width, tracks.height],
            "crop_scale": 1.0, "viewport_scale": 1.0,
        }
    annotated = _overlay(raw, tracks, source_frame, state["focus"], state["overlay"])
    shot = tuple(state["shot"])
    crop, scale = _crop_for_state(
        annotated, tracks, source_frame, state["focus"], shot,
        _full_crop(tracks, shot),
    )
    viewport = TACTICS_VIEW if state["chapter"] == "tactics" else VIEW
    pane, output_scale = _fit_source(annotated, crop, viewport)
    canvas = np.full((HEIGHT, WIDTH, 3), (17, 26, 31), np.uint8)
    canvas[viewport[1]:viewport[3], viewport[0]:viewport[2]] = pane
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    boxes = []
    title = chapter["title"]
    kicker = "另一回合 · 远端球员" if state["chapter"] == "arms" else ""
    boxes += _draw_text(draw, title, (78, 218), heading, "#f5f8ff")
    if kicker:
        boxes += _draw_text(draw, kicker, (80, 313), subhead, "#92d9d4", line_h=45)

    if state["chapter"] == "legs" and state["phase"] == "hold":
        knee = story["evidence"]["legs"]["knee_flexion_2d_deg"]
        boxes += _draw_text(draw, f"前膝 {knee:.0f}° · 2D估计", (78, 1150), small, "#fbd47c", line_h=42)
    if state["chapter"] == "core":
        boxes += _draw_text(draw, "肩线", (78, 1146), small, "#58e5dc", line_h=42)
        boxes += _draw_text(draw, "髋线", (226, 1146), small, "#ffd071", line_h=42)
        boxes += _draw_text(draw, "画面中的二维投影", (375, 1146), small, "#d5dfe5", line_h=42)
    _draw_minimap(draw, state, story, chapter)

    caption = next((row for row in schedule["captions"]
                    if row["chapter"] == state["chapter"]
                    and row["start_s"] <= source_time < row["end_s"]), None)
    if caption:
        boxes += _draw_text(draw, caption["text"], CAPTION_XY, caption_font,
                            "#ffffff", width=CAPTION_WIDTH, line_h=58)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), boxes, {
        "source_crop": list(crop), "crop_scale": scale, "viewport_scale": output_scale,
    }


def _synthesize(texts, model_dir):
    tts = _voice(model_dir)
    entries = []
    for key, text in texts:
        generated = tts.generate(text, sid=0, speed=SPEECH_SPEED)
        samples = np.asarray(generated.samples, dtype=np.float32)
        rate = int(generated.sample_rate)
        if not samples.size or not np.isfinite(samples).all() or np.max(np.abs(samples)) < 0.01:
            raise RuntimeError(f"Local neural voice returned silent or invalid speech for {key}")
        voiced = np.flatnonzero(np.abs(samples) > 0.003)
        start = max(0, int(voiced[0]) - round(0.03 * rate))
        end = min(len(samples), int(voiced[-1]) + round(0.06 * rate))
        entries.append({"key": key, "text": text, "samples": samples[start:end], "rate": rate})
    if len({entry["rate"] for entry in entries}) != 1:
        raise RuntimeError("Neural voice sample rates do not agree")
    return entries


def _duration_text(seconds):
    return f"{seconds:.2f}s"


def _write_analysis(folder, schedule, evidence, attribution, source_info, court_info):
    chapters = []
    output_time_by_source_frame = {}
    for output_index, (source_frame, state) in enumerate(
            zip(schedule["frame_map"], schedule["states"])):
        if state["phase"] in ("motion", "hold"):
            output_time_by_source_frame.setdefault(int(source_frame), output_index / FPS)
    for row in schedule["chapters"]:
        claim = row["evidence"].get("claim", "")
        evidence_rows = row["evidence"]
        frames = sorted({
            int(value) for key, value in evidence_rows.items()
            if key.endswith("_frame") and isinstance(value, (int, float))
        })
        if row["key"] == "core":
            frames = [item["frame"] for item in evidence_rows["frames"]]
        if row["key"] == "arms":
            frames = [
                evidence_rows["left_peak"], evidence_rows["left_return"],
                evidence_rows["right_peak"],
            ]
        if row["key"] == "tactics":
            frames = [
                evidence_rows["initial_frame"], evidence_rows["lunge_frame"],
                evidence_rows["near_settled"]["source_start_frame"],
                evidence_rows["near_settled"]["source_end_frame"],
                evidence_rows["far_settled"]["source_start_frame"],
                evidence_rows["far_settled"]["source_end_frame"],
            ]
        confidence = []
        if row["key"] == "legs":
            confidence = [
                {"frame": evidence_rows["plant_frame"], "value": evidence_rows["plant_confidence"]},
                {"frame": evidence_rows["bottom_frame"], "value": evidence_rows["bottom_confidence"]},
            ]
        elif row["key"] == "core":
            confidence = [
                {"frame": item["frame"], "value": item["confidence"]}
                for item in evidence_rows["frames"]
            ]
        elif row["key"] == "arms":
            confidence = [
                {"frame": evidence_rows["left_peak"], "value": evidence_rows["left_confidence"]},
                {"frame": evidence_rows["left_return"],
                 "value": evidence_rows["left_return_confidence"]},
                {"frame": evidence_rows["right_peak"], "value": evidence_rows["right_confidence"]},
            ]
        elif row["key"] == "tactics":
            confidence = [
                {"frame": evidence_rows["initial_frame"],
                 "value": evidence_rows["initial_position"]["confidence"]},
                {"frame": evidence_rows["lunge_frame"],
                 "value": evidence_rows["lunge_position"]["confidence"]},
                {"range": [evidence_rows["near_settled"]["source_start_frame"],
                           evidence_rows["near_settled"]["source_end_frame"]],
                 "value": evidence_rows["near_settled"]["confidence"],
                 "n": evidence_rows["near_settled"]["n"]},
                {"range": [evidence_rows["far_settled"]["source_start_frame"],
                           evidence_rows["far_settled"]["source_end_frame"]],
                 "value": evidence_rows["far_settled"]["confidence"],
                 "n": evidence_rows["far_settled"]["n"]},
            ]
        source_bindings = [{
            "start_frame": source_range["start_frame"],
            "end_frame": source_range["end_frame"],
            "source_frame": source_range["source_frame"],
            "phase": source_range["phase"],
            "output_time_s": [
                source_range["output_start_s"], source_range["output_end_s"],
            ],
        } for source_range in row["source_ranges"]]
        evidence_bindings = []
        for frame in frames:
            representation = "source_video"
            output_time = output_time_by_source_frame.get(frame)
            if row["key"] == "tactics" and frame == evidence_rows["initial_frame"]:
                representation = "court_map"
                output_time = row["output_start_s"]
            evidence_bindings.append({
                "source_frame": frame, "representation": representation,
                "output_time_s": output_time,
            })
        chapters.append({
            "chapter": row["title"], "claim": claim, "source_frames": frames,
            "source_range_bindings": source_bindings,
            "evidence_frame_output_times": evidence_bindings,
            "output_time_s": [row["output_start_s"], row["output_end_s"]],
            "speech_time_s": [row["speech_start_s"], row["speech_end_s"]],
            "confidence": confidence,
            "teaching_principle": evidence_rows.get("teaching_principle", ""),
            "measurements": evidence_rows,
        })
    analysis = {
        "title": "一拍之间：腿、手臂、核心与战术",
        "source": source_info,
        "chapters": chapters,
        "court_map": court_info,
        "editorial_limits": [
            "原片为多回合剪辑，不是整场比赛；每个片段都留在同一镜头内。",
            "所有角度和场地位置都是单摄像机二维投影，不能推出三维旋转、肌肉活动或受力。",
            "没有羽毛球轨迹或触球检测；不判断击球接触、落点、回合胜负或对手意图。",
            "人物动作是可见观察；训练建议是一般单打练习原则，不是对职业球员的错误判决。",
            "握拍、落点和前臂/手腕细节不在这段模糊且未追踪羽毛球的画面中验证。",
            "模型置信度反映关键点估计，不是教学建议正确率。",
        ],
        "attribution": attribution,
    }
    _dump(folder / "analysis.json", analysis)
    cards = []
    for chapter in chapters:
        evidence_text = json.dumps(chapter["measurements"], ensure_ascii=False, indent=2)
        ranges_text = "；".join(
            (f'静态总结卡 {binding["output_time_s"][0]:.2f}–'
             f'{binding["output_time_s"][1]:.2f}s（原片锚点帧不显示）'
             if binding["phase"] == "summary" else
             f'源帧 {binding["start_frame"]}–{binding["end_frame"]} → '
             f'{binding["output_time_s"][0]:.2f}–{binding["output_time_s"][1]:.2f}s')
            for binding in chapter["source_range_bindings"]
        )
        mapped_evidence = "；".join(
            (f'{"地图" if binding["representation"] == "court_map" else "画面"}帧 '
             f'{binding["source_frame"]} → {binding["output_time_s"]:.2f}s'
             if binding["output_time_s"] is not None else
             f'帧 {binding["source_frame"]}：本片未显示')
            for binding in chapter["evidence_frame_output_times"]
        )
        confidence_text = "；".join(
            (f'帧 {row["frame"]}：{row["value"]:.3f}'
             if "frame" in row else
             f'帧 {row["range"][0]}–{row["range"][1]}：中位置信度 {row["value"]:.3f}，n={row["n"]}')
            for row in chapter["confidence"]
        )
        cards.append(
            f'<article><h2>{html.escape(chapter["chapter"])}</h2>'
            f'<p><b>画面观察：</b>{html.escape(chapter["claim"])}</p>'
            f'<p><b>一般练习提示：</b>{html.escape(chapter["teaching_principle"])}</p>'
            f'<p><b>片段映射：</b>{html.escape(ranges_text)}'
            f' · <b>证据帧映射：</b>{html.escape(mapped_evidence)}'
            f' · <b>旁白：</b>{chapter["speech_time_s"][0]:.2f}–{chapter["speech_time_s"][1]:.2f}s'
            f'</p>'
            f'<p><b>关键点置信度：</b>{html.escape(confidence_text)}</p>'
            f'<details><summary>测量与帧绑定</summary><pre>{html.escape(evidence_text)}</pre></details>'
            '</article>'
        )
    limits = "".join(f"<li>{html.escape(text)}</li>" for text in analysis["editorial_limits"])
    (folder / "analysis.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>一拍之间｜羽毛球技术拆解</title>'
        '<style>body{font:18px/1.75 "Microsoft YaHei",sans-serif;background:#101a1f;color:#edf6f1;'
        'max-width:980px;margin:28px auto;padding:24px}video{width:100%;background:#000}'
        'article{padding:22px;background:#1a2a2c;margin:18px 0;border-radius:14px}'
        'a{color:#7ee5d0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}'
        'summary{cursor:pointer;color:#a6d9ce}</style>'
        '<h1>一拍之间｜腿、手臂、核心与战术</h1>'
        '<video controls preload="metadata" poster="cover.jpg" src="tiktok.mp4"></video>'
        '<p><a download href="tiktok.mp4">下载视频</a> · '
        '<a download href="captions.srt">字幕</a> · '
        '<a download href="analysis.json">证据数据</a> · '
        '<a download href="post.txt">发布文案与署名</a> · '
        '<a download href="points.json">章节与测量</a></p>'
        + "".join(cards)
        + '<h2>怎么读这份分析</h2><p>每章先写画面里能直接看到的动作，再把一般练习提示分开。'
        '画面没有显示的击球结果，不拿来补全故事。转录语料只作为选题背景，不证明某个动作有效，也没有声称读过1000份转录。</p>'
        '<h2>局限</h2><ul>' + limits + '</ul>'
        + '<h2>来源与许可</h2><pre>' + html.escape(attribution["full"]) + '</pre></html>',
        "utf-8",
    )
    return analysis


def export_editorial(video, doc, report, out_mp4, *, model_dir=VOICE, attribution_file=None):
    out = Path(out_mp4)
    folder = out.parent
    folder.mkdir(parents=True, exist_ok=True)
    credit = source_credit(video, attribution_file)
    tracks, story = build_story(doc, report)
    chapter_texts = [(chapter.key, chapter.narration) for chapter in story["chapters"]]
    speeches = _synthesize(chapter_texts, model_dir)
    lengths = [len(entry["samples"]) / entry["rate"] for entry in speeches]
    schedule = make_schedule(story, lengths, tracks)
    schedule["credit"] = credit
    schedule["voice"] = {
        "engine": "sherpa-onnx", "model": "MeloTTS-Chinese",
        "speed": SPEECH_SPEED, "sample_rate": speeches[0]["rate"],
        "model_sha256": sha256_file(Path(model_dir) / "model.onnx"),
    }
    schedule_for_json = {key: value for key, value in schedule.items()
                         if key not in ("frame_map", "states")}
    source_sha = sha256_file(Path(video))
    _dump(folder / "schedule.json", schedule_for_json)
    _dump(folder / "frame-map.json", {
        "fps": FPS,
        "source_frame_per_output_frame": schedule["frame_map"],
        "states": schedule["states"],
    })
    _dump(folder / "points.json", {
        "source_video": str(Path(video).resolve()), "source_sha256": source_sha,
        "chapters": schedule["chapters"], "recap_lines": list(RECAP_LINES),
        "priority_corpus_note": (
            "An open-dataset sample of 100 coaching videos is background only; "
            "concept frequency is not proof of coaching quality."
        ),
        "limits": [
            "Single-camera 2D pose and court-plane estimates only.",
            "No shuttle/contact tracking, force, medical, or 3D conclusions.",
            "A cut separates the arm chapter from the lunge rally.",
        ],
    })
    tactics = story["evidence"]["tactics"]
    court_info = {
        "coordinate_definition": "court-normalized; x left-to-right, y far-baseline-to-near-baseline; both ankles averaged",
        "homography_method": "recomputed with cv2.getPerspectiveTransform from this pose run's four detected court corners",
        "near_initial": tactics["initial_position"],
        "near_lunge": tactics["lunge_position"],
        "near_settled_median": tactics["near_settled"],
        "far_settled_median": tactics["far_settled"],
        "court_corners_px": tactics["court_corners_px"],
        "homography_to_normalized_court": tactics["homography_to_normalized_court"],
    }
    source_info = {
        "video": str(Path(video).resolve()), "sha256": source_sha,
        "fps": tracks.fps, "frame_count": tracks.n,
        "camera_shots": [list(shot) for shot in tracks.shots],
        "pose_backend": report.source,
    }
    analysis = _write_analysis(folder, schedule, story["evidence"], credit, source_info, court_info)
    (folder / "captions.srt").write_text(
        "\n\n".join(
            f'{index}\n{_srt_time(row["start_s"])} --> {_srt_time(row["end_s"])}\n{row["text"]}'
            for index, row in enumerate(schedule["captions"], 1)
        ) + "\n", "utf-8",
    )
    (folder / "narration.txt").write_text(
        "\n".join(f'{row["chapter"]}：{row["text"]}' for row in schedule["captions"]) + "\n",
        "utf-8",
    )
    (folder / "post.txt").write_text(
        "一拍之间，看腿、手臂、躯干和回位怎样接起来。\n"
        "动作观察与一般练习提示分开；单镜头二维估计，不推断触球、落点、受力或三维动作。\n"
        "#羽毛球 #技术分析 #步法 #动作拆解\n\n" + credit["full"], "utf-8",
    )
    _dump(folder / "court-map.json", court_info)

    font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    fonts = tuple(ImageFont.truetype(str(font_dir / "msyh.ttc"), size)
                  for size in (72, 42, 44, 28, 52))
    story["fonts"] = fonts
    ffmpeg, ffprobe = _tools()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot read source {video}")
    text_boxes, crop_checks = [], []
    started = time.perf_counter()
    sound_rate = speeches[0]["rate"]
    sound = np.zeros(math.ceil(schedule["planned_duration_s"] * sound_rate), np.float32)
    for caption, speech in zip(schedule["captions"], speeches):
        start = round(caption["start_s"] * sound_rate)
        samples = speech["samples"]
        sound[start:start + len(samples)] = samples

    with tempfile.TemporaryDirectory(prefix="editorial-", dir=folder) as temp_dir:
        temp_dir = Path(temp_dir)
        voice_path = temp_dir / "voice.wav"
        with wave.open(str(voice_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sound_rate)
            handle.writeframes((np.clip(sound, -1, 1) * 32767).astype("<i2").tobytes())
        with (temp_dir / "encoder.log").open("wb") as errors:
            proc = subprocess.Popen([
                str(ffmpeg), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0",
                "-i", str(voice_path), "-c:v", "libx264", "-preset", "fast",
                "-crf", "19", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                "-movflags", "+faststart", str(out),
            ], stdin=subprocess.PIPE, stderr=errors)
            previous = None
            try:
                for index, (source_frame, state) in enumerate(zip(schedule["frame_map"], schedule["states"])):
                    if source_frame != previous:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
                        ok, raw = cap.read()
                        if not ok:
                            raise RuntimeError(f"Could not decode source frame {source_frame}")
                        previous = source_frame
                    rendered, boxes, crop_info = render_frame(
                        raw, tracks, state, schedule, story, fonts, credit["endcard"], index / FPS,
                    )
                    if index == 0 and not cv2.imwrite(str(folder / "cover.jpg"), rendered):
                        raise RuntimeError("Cannot write the cover image")
                    text_boxes.extend(boxes)
                    if state["focus"]:
                        player = _role(tracks, state["focus"])
                        measured = _nearest_measured(player, source_frame, tuple(state["shot"]))
                        bbox = player.bbox[measured]
                        left, top, right, bottom = crop_info["source_crop"]
                        crop_checks.append({
                            "chapter": state["chapter"], "source_frame": source_frame,
                            "focus": state["focus"],
                            "bbox": bbox.astype(float).tolist(),
                            "player_height_px": float(bbox[3] - bbox[1]),
                            "crop": crop_info["source_crop"],
                            "scale": crop_info["crop_scale"],
                            "contains_player": bool(left <= bbox[0] <= bbox[2] <= right
                                                   and top <= bbox[1] <= bbox[3] <= bottom),
                        })
                    proc.stdin.write(rendered.tobytes())
                proc.stdin.close()
                if proc.wait() != 0:
                    raise RuntimeError((temp_dir / "encoder.log").read_text("utf-8", errors="replace"))
            finally:
                cap.release()
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        subprocess.run([str(ffmpeg), "-v", "error", "-i", str(out), "-f", "null", "-"],
                       check=True, capture_output=True)
        probe = json.loads(subprocess.check_output([
            str(ffprobe), "-v", "error", "-count_frames", "-show_streams", "-show_format",
            "-of", "json", str(out),
        ]))
        decoded = subprocess.check_output([
            str(ffmpeg), "-v", "error", "-i", str(out), "-map", "0:a:0",
            "-f", "f32le", "-ac", "1", "pipe:1",
        ])
    audio = np.frombuffer(decoded, dtype="<f4")
    video_stream = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    audio_stream = next(stream for stream in probe["streams"] if stream["codec_type"] == "audio")
    checks = {
        "full_decode": True,
        "non_silent_audio": bool(audio.size and np.max(np.abs(audio)) > 0.01),
        "output_shape": [video_stream["width"], video_stream["height"]],
        "video_frames": int(video_stream["nb_read_frames"]),
        "video_duration_s": float(video_stream["duration"]),
        "audio_duration_s": float(audio_stream["duration"]),
        "caption_windows_match_measured_speech": all(
            row["start_s"] == row["speech_start_s"] and row["end_s"] == row["speech_end_s"]
            for row in schedule["captions"]
        ),
        "all_text_inside_safe_rect": all(
            SAFE[0] <= box[0] <= box[2] <= SAFE[2]
            and SAFE[1] <= box[1] <= box[3] <= SAFE[3] for box in text_boxes
        ),
        "all_focus_crops_contain_player": bool(crop_checks) and all(
            row["contains_player"]
            and row["scale"] <= (2.4 if row["focus"] == "far" else 2.7)
            and (row["focus"] != "far" or row["player_height_px"] >= 190)
            for row in crop_checks
        ),
        "crop_checks": crop_checks,
        "moving_s": schedule["moving_s"],
        "paused_s": schedule["paused_s"],
        "summary_s": schedule["summary_s"],
        "paused_share": schedule["paused_share"],
        "longest_hold_s": schedule["longest_hold_s"],
        "no_clip_spans_camera_cut": all(
            clip["source_shot"][0] <= (
                clip["source_frame"] if clip["phase"] in ("hold", "summary") else clip["start_frame"]
            ) <= (clip["source_frame"] if clip["phase"] == "hold" else clip["end_frame"])
            <= clip["source_shot"][1]
            for row in schedule["chapters"] for clip in row["source_ranges"]
        ),
        "text_safe_rect": list(SAFE),
        "render_s": time.perf_counter() - started,
        "sha256": sha256_file(out),
        "voice": schedule["voice"],
        "court_homography_recomputed": True,
        "free_arm_returns_before_racket_raise": (
            story["evidence"]["arms"]["left_peak"]
            < story["evidence"]["arms"]["left_return"]
            < story["evidence"]["arms"]["right_peak"]
        ),
        "speech_naturalness": "Not independently rated by a human",
    }
    if (checks["output_shape"] != [WIDTH, HEIGHT]
            or checks["video_frames"] != schedule["planned_output_frames"]
            or abs(checks["audio_duration_s"] - schedule["planned_duration_s"]) > 0.1
            or not checks["non_silent_audio"]
            or not checks["all_text_inside_safe_rect"]
            or not checks["all_focus_crops_contain_player"]
            or not checks["no_clip_spans_camera_cut"]):
        failed = [
            name for name in (
                "non_silent_audio", "all_text_inside_safe_rect",
                "all_focus_crops_contain_player", "no_clip_spans_camera_cut",
            ) if not checks[name]
        ]
        raise RuntimeError(
            "Rendered editorial short failed media, layout, or source-alignment checks: "
            + ", ".join(failed or ["shape, frame count, or duration"])
        )
    _dump(folder / "verification.json", checks)
    return schedule, checks, analysis


def _handoff(folder, video, schedule, checks, run, tests=None):
    artifacts = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.name != "handoff.json":
            artifacts[path.name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    changed = {
        name: {"path": str((PROJECT / name).resolve()), "sha256": sha256_file(PROJECT / name)}
        for name in CHANGED_SOURCES
    }
    value = {
        "status": "rendered",
        "video": str(Path(video).resolve()),
        "video_sha256": sha256_file(Path(video)),
        "planned_duration_s": schedule["planned_duration_s"],
        "moving_s": checks["moving_s"], "paused_s": checks["paused_s"],
        "summary_s": checks["summary_s"],
        "paused_share": checks["paused_share"], "longest_hold_s": checks["longest_hold_s"],
        "chapters": [{
            "title": row["title"], "source_shots": row["source_shots"],
            "source_ranges": row["source_ranges"],
            "source_frames": row["evidence"],
            "output_start_s": row["output_start_s"], "output_end_s": row["output_end_s"],
            "speech_start_s": row["speech_start_s"], "speech_end_s": row["speech_end_s"],
        } for row in schedule["chapters"]],
        "media": {key: checks[key] for key in (
            "output_shape", "video_frames", "video_duration_s", "audio_duration_s",
            "non_silent_audio", "full_decode", "court_homography_recomputed",
        )},
        "tests": tests or {"status": "pending"},
        "artifacts": artifacts, "changed_source_files": changed,
        "limits": [
            "Single-camera 2D keypoint and court projection; no 3D, force, or medical conclusions.",
            "No shuttle/contact tracking; no shot-placement, result, or opponent-intent claims.",
            "The clip contains edits and multiple rallies, not a continuous full match.",
            "Neural speech was generated locally; human naturalness and upstream broadcast rights are not independently rated.",
        ],
        "run": run,
    }
    _dump(folder / "handoff.json", value)
    return value


def export_command(video, out_dir, attribution_file=None, voice_model=VOICE):
    if out_dir.exists():
        raise FileExistsError(f"Use a fresh output folder; refusing to overwrite {out_dir}")
    credit = source_credit(video.resolve(), attribution_file)
    from turbo.accept import acquire_cpu_lock, release_cpu_lock
    lock = PROJECT / "work" / "cpu.lock"
    token = f"editorial-short {os.getpid()} {uuid.uuid4().hex}"
    acquire_cpu_lock(lock, token)
    started = time.perf_counter()
    try:
        out_dir.mkdir(parents=True)
        player_metadata = player_inputs(video.resolve(), attribution_file)
        session = analyze_video(video.resolve(), FastBackend(), out_dir, video.stem,
                                players=player_metadata)
        if session.error:
            raise RuntimeError(session.error.message)
        schedule, checks, _analysis = export_editorial(
            video.resolve(), session.doc, session.report, out_dir / "tiktok.mp4",
            model_dir=voice_model.resolve(), attribution_file=attribution_file,
        )
        run = {
            "fresh_inference": True, "backend": "fast",
            "timings": session.timings, "total_elapsed_s": time.perf_counter() - started,
        }
        _dump(out_dir / "run.json", run)
        handoff = _handoff(out_dir, out_dir / "tiktok.mp4", schedule, checks, run)
        print(json.dumps({
            "output": str(out_dir.resolve()),
            "video": handoff["video"],
            "sha256": handoff["video_sha256"],
            "duration_s": schedule["planned_duration_s"],
        }))
    finally:
        release_cpu_lock(lock, token)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--attribution-json", type=Path)
    parser.add_argument("--voice-model", type=Path, default=VOICE)
    args = parser.parse_args(argv)
    export_command(args.video, args.out, args.attribution_json, args.voice_model)


if __name__ == "__main__":
    main()
