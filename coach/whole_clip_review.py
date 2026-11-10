"""Rank and export three evidence-bound moments from a complete supplied clip."""
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
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .app.backend import PROJECT, sha256_file
from .analyze import analyze
from .complete_review import VOICE
from .editorial_short import (
    FPS,
    HEIGHT,
    VIEW,
    WIDTH,
    _draw_text,
    _full_crop,
    _nearest_measured,
    _overlay,
    _role,
    _select_arm_sequence,
    _synthesize,
    _wrap_zh,
    _fit_source,
    _crop_for_state,
)
from .export import _tools
from .schema import Role
from .short_review import _dump, _srt_time, source_credit
from .tracks import JOINT_INDEX, load_tracks

INTRO_FRAMES = 30
SUMMARY_MIN_S = 4.0
KEY_HOLD_S = 2.0
SPEECH_SPEED = 1.08
MOMENT_TEXT = {
    "attack_dive": (
        "远端先上挥，近端右臂向外伸、侧跨落地，躯干随之降低；练习把落地保护接到起身。"
    ),
    "long_exchange": (
        "近端与远端先后跨步下沉；练习时每次下沉后接回动。"
    ),
    "legs": "右脚跨出、前膝弯曲，上身随下沉倾斜，再继续回动；练习把跨步和回收接上。",
    "recap": "复盘三处：攻守后回动、连续跨步、单次跨步回收。",
}
SUMMARY_LINES = (
    "双人攻防后的低位回动",
    "连续跨步后接回动",
    "单次跨步后蹬回",
)


def verify_pose_binding(video: Path, pose_path: Path, points_path: Path,
                        source_analysis_path: Path, pose_handoff_path: Path) -> dict:
    source_sha = sha256_file(video)
    pose_sha = sha256_file(pose_path)
    points = json.loads(points_path.read_text("utf-8"))
    source_analysis = json.loads(source_analysis_path.read_text("utf-8"))
    handoff = json.loads(pose_handoff_path.read_text("utf-8"))
    pose = json.loads(pose_path.read_text("utf-8"))
    expected_pose_sha = handoff.get("artifacts", {}).get("pose.json", {}).get("sha256")
    if (points.get("source_sha256") != source_sha
            or source_analysis.get("source", {}).get("sha256") != source_sha):
        raise ValueError("Pose provenance does not bind to the exact supplied video hash")
    if expected_pose_sha != pose_sha:
        raise ValueError("Pose hash does not match its saved handoff manifest")
    source = pose.get("source", {})
    frames = pose.get("frames", [])
    count = source.get("frame_count")
    fps = source.get("fps")
    if (not isinstance(count, int) or count != 900 or fps != FPS
            or len(frames) != count):
        raise ValueError("Reusable pose must cover all 900 frames at 30 fps")
    max_timestamp_error = 0.0
    for index, row in enumerate(frames):
        if row.get("frame_index") != index:
            raise ValueError(f"Pose frame coverage is not contiguous at index {index}")
        error = abs(float(row.get("timestamp_seconds", -1)) - index / fps)
        max_timestamp_error = max(max_timestamp_error, error)
        if error > 0.001:
            raise ValueError(f"Pose timestamp is not aligned at frame {index}")
    return {
        "source_sha256": source_sha,
        "pose_sha256": pose_sha,
        "source_frames": count,
        "source_fps": fps,
        "first_frame": 0,
        "last_frame": count - 1,
        "timestamps_aligned": True,
        "max_timestamp_error_s": max_timestamp_error,
        "provenance_points": str(points_path.resolve()),
        "provenance_analysis": str(source_analysis_path.resolve()),
        "provenance_handoff": str(pose_handoff_path.resolve()),
        "pose_manifest_sha256": expected_pose_sha,
    }


def _event_window(event) -> tuple[int, int, int]:
    data = event.data
    if event.kind == "lunge":
        return (
            int(data.get("plant_frame", event.frame)),
            int(data.get("bottom_frame", event.frame)),
            int(data.get("end_frame", event.frame)),
        )
    if event.kind == "swing":
        return (
            int(data.get("start", event.frame)),
            int(data.get("peak_frame", event.frame)),
            int(data.get("end", event.frame)),
        )
    if event.kind == "base_return":
        return (
            int(data.get("from_frame", event.frame)),
            int(event.frame),
            int(event.frame),
        )
    if event.kind == "split_step":
        return (
            int(data.get("onset_frame", event.frame)),
            int(data.get("landing_frame", event.frame)),
            int(data.get("landing_frame", event.frame)),
        )
    return (int(event.frame), int(event.frame), int(event.frame))


def _event_dict(event) -> dict:
    return {
        "kind": event.kind,
        "player": event.player.value,
        "frame": int(event.frame),
        "raw_event_model_score": float(event.confidence),
        "data": dict(event.data),
    }


def candidate_inventory(report, tracks) -> dict:
    candidate_kinds = {"lunge", "swing", "split_step", "base_return"}
    rows = []
    counts_by_shot = []
    for shot_index, shot in enumerate(tracks.shots):
        shot_rows = []
        shot_steps = sum(
            event.kind == "step" and shot[0] <= event.frame <= shot[1]
            for event in report.events
        )
        for event in report.events:
            if event.kind not in candidate_kinds or not shot[0] <= event.frame <= shot[1]:
                continue
            start, peak, end = _event_window(event)
            row = {
                "candidate_id": f"{event.kind}-{event.player.value}-{event.frame}",
                "kind": event.kind,
                "player": event.player.value,
                "shot_index": shot_index,
                "shot_frames": list(shot),
                "start_frame": start,
                "peak_frame": peak,
                "end_frame": end,
                "raw_event_model_score": float(event.confidence),
                "event_data": dict(event.data),
            }
            rows.append(row)
            shot_rows.append(row)
        counts_by_shot.append({
            "shot_index": shot_index,
            "source_frames": list(shot),
            "step_events": int(shot_steps),
            "candidate_event_count": len(shot_rows),
            "candidate_ids": [row["candidate_id"] for row in shot_rows],
        })
    return {
        "timeline_frames_reviewed": tracks.n,
        "camera_shots": [list(shot) for shot in tracks.shots],
        "event_counts": {
            kind: sum(event.kind == kind for event in report.events)
            for kind in sorted({event.kind for event in report.events})
        },
        "candidate_event_kinds": sorted(candidate_kinds),
        "candidate_event_count": len(rows),
        "supporting_events": rows,
        "shots": counts_by_shot,
        "ranking_basis": [
            "Ranked by visible action, completeness of the action-to-recovery passage, and distinct analysis value.",
            "Camera framing and whether the relevant body parts remain readable are part of the review.",
            "Raw pose/event model scores are evidence notes, not probabilities, ranking inputs, or player-skill scores.",
            "All windows are checked against the four source camera shots and none crosses a cut.",
        ],
        "score_definition": {
            "scale": "编辑评分：由分析代理依据画面给出，未经人工复核；1–10分",
            "weights": {
                "highlight_score": 0.40,
                "coaching_value_score": 0.35,
                "evidence_quality_score": 0.25,
            },
            "formula": (
                "round(0.40*highlight_score + 0.35*coaching_value_score "
                "+ 0.25*evidence_quality_score, 2)"
            ),
            "raw_model_score_policy": (
                "Raw model scores do not contribute to any of the three ratings or the combined rank."
            ),
        },
    }


def _raw_pose_quality(track, frame: int, names: tuple[str, ...]) -> dict:
    visible = [name for name in names if track.joint_visible(name)[frame]]
    scores = [float(track.conf[frame, JOINT_INDEX[name]]) for name in visible]
    return {
        "frame": int(frame),
        "measured": bool(track.measured[frame]),
        "visible_joint_count": len(visible),
        "raw_joint_model_scores": {
            name: float(track.conf[frame, JOINT_INDEX[name]]) for name in visible
        },
        "minimum_raw_joint_model_score": min(scores) if scores else None,
        "joints_below_overlay_threshold": [
            name for name in visible
            if track.conf[frame, JOINT_INDEX[name]] < 0.70
        ],
    }


def _score_candidate(candidate: dict) -> dict:
    score = round(
        0.40 * candidate["highlight_score"]
        + 0.35 * candidate["coaching_value_score"]
        + 0.25 * candidate["evidence_quality_score"],
        2,
    )
    return {**candidate, "combined_score": score}


def _court_position_evidence(tracks, window):
    if tracks.court is None:
        raise ValueError("Court-position analysis needs measured court corners")
    image_points = np.asarray(tracks.court.corners_px, dtype=np.float32).reshape(4, 1, 2)
    court_points = np.asarray([
        [0.0, 0.0], [6.10, 0.0], [6.10, 13.40], [0.0, 13.40],
    ], dtype=np.float32).reshape(4, 1, 2)
    transform = cv2.getPerspectiveTransform(image_points, court_points)

    def position(role, frame):
        player = tracks.players[role]
        ankles = ("left_ankle", "right_ankle")
        if not player.measured[frame] or not all(
                player.joint_visible(name)[frame] for name in ankles):
            return None
        image_xy = np.mean(
            [player.joint(name)[frame] for name in ankles], axis=0,
        ).astype(np.float32).reshape(1, 1, 2)
        court_xy = cv2.perspectiveTransform(image_xy, transform).reshape(2)
        if not np.isfinite(court_xy).all():
            return None
        return court_xy.astype(float)

    result = {}
    for role in (Role.NEAR, Role.FAR):
        usable = [
            (frame, position(role, frame))
            for frame in range(window[0], window[1] + 1)
        ]
        usable = [(frame, xy) for frame, xy in usable if xy is not None]
        if not usable:
            raise ValueError(f"No court-position evidence for {role.value} in {window}")
        start_frame, start_xy = usable[0]
        end_frame, end_xy = usable[-1]
        result[role.value] = {
            "start_frame": start_frame,
            "start_xy_m": [round(float(value), 1) for value in start_xy],
            "end_frame": end_frame,
            "end_xy_m": [round(float(value), 1) for value in end_xy],
            "net_displacement_2d_m": round(float(np.linalg.norm(end_xy - start_xy)), 1),
            "measurement": (
                "2D court-homography estimate from midpoint of both ankles; "
                "straight start-to-end displacement, not path length."
            ),
        }
    return {
        "source_window": list(window),
        "court_dimensions_m": [6.10, 13.40],
        "players": result,
    }


def choose_moments(report, tracks, inventory: dict) -> list[dict]:
    events = list(report.events)
    specs = [
        {
            "candidate_id": "opening-racket-exchange",
            "source_shot": [0, 211], "window": [44, 57],
            "key": "opening_exchange", "title": "近端与远端的短交换",
            "highlight_score": 6.4, "coaching_value_score": 6.0,
            "evidence_quality_score": 7.0,
            "reason": "双人挥拍顺序清楚，但动作短，能展开的脚步与回动分析有限。",
        },
        {
            "candidate_id": "opening-near-lunge",
            "source_shot": [0, 211], "window": [60, 109],
            "key": "legs", "title": "近端跨步与回收",
            "highlight_score": 8.3, "coaching_value_score": 8.7,
            "evidence_quality_score": 9.0,
            "reason": "完整呈现跨步、前膝屈曲和回动，腿部关键点与动作阶段齐全。",
        },
        {
            "candidate_id": "opening-later-exchange",
            "source_shot": [0, 211], "window": [112, 140],
            "key": "later_exchange", "title": "后续跨步片段",
            "highlight_score": 7.4, "coaching_value_score": 7.0,
            "evidence_quality_score": 7.5,
            "reason": "有挥拍与跨步下沉，但阶段较短，完整性不及前一处。",
        },
        {
            "candidate_id": "second-shot-step-only",
            "source_shot": [212, 366], "window": [212, 366],
            "key": "step_only", "title": "第二镜头步伐候选",
            "highlight_score": 1.5, "coaching_value_score": 1.8,
            "evidence_quality_score": 2.4,
            "reason": "全镜头均纳入扫描，但只有零散步伐事件，没有清楚、独立的跨步或挥拍高光。",
        },
        {
            "candidate_id": "third-shot-early-exchange",
            "source_shot": [367, 664], "window": [427, 490],
            "key": "early_exchange", "title": "前段步伐与跨步",
            "highlight_score": 7.8, "coaching_value_score": 7.6,
            "evidence_quality_score": 7.2,
            "reason": "有垫步、挥拍及连续跨步，但与更长的双人跨步回合部分重叠。",
        },
        {
            "candidate_id": "third-shot-long-lunge-exchange",
            "source_shot": [367, 664], "window": [459, 545],
            "key": "long_exchange", "title": "双人连续跨步回合",
            "highlight_score": 9.2, "coaching_value_score": 9.0,
            "evidence_quality_score": 7.8,
            "reason": "近端与远端都出现跨步下沉和回动，可比较连续移动；固定裁切保留同镜头内两人。",
        },
        {
            "candidate_id": "third-shot-mid-exchange",
            "source_shot": [367, 664], "window": [551, 585],
            "key": "mid_exchange", "title": "中段挥拍与步伐",
            "highlight_score": 6.1, "coaching_value_score": 6.2,
            "evidence_quality_score": 6.7,
            "reason": "动作在画面内，但缺少连续的伸展—回动片段，教学线索较少。",
        },
        {
            "candidate_id": "third-shot-attack-and-dive",
            "source_shot": [367, 664], "window": [612, 664],
            "key": "attack_dive", "title": "远端上挥与近端低位伸展",
            "highlight_score": 10.0, "coaching_value_score": 9.4,
            "evidence_quality_score": 7.2,
            "reason": "双人先后动作与地面阶段最醒目；伏地时逐关节检查分数与身体框，仅弱点位不绘制。",
        },
        {
            "candidate_id": "fourth-shot-swing-exchange",
            "source_shot": [665, 899], "window": [731, 757],
            "key": "fourth_exchange", "title": "后段挥拍交换",
            "highlight_score": 6.5, "coaching_value_score": 5.9,
            "evidence_quality_score": 6.3,
            "reason": "可见两次挥拍，但缺少明显的伸展与回动链条。",
        },
        {
            "candidate_id": "fourth-shot-low-reach-return",
            "source_shot": [665, 899], "window": [779, 821],
            "key": "low_reach", "title": "低位移动与回位",
            "highlight_score": 8.4, "coaching_value_score": 8.0,
            "evidence_quality_score": 6.7,
            "reason": "低位伸展后能看到回到中段附近，但侧向细节的姿态证据偏弱。",
        },
    ]
    scored = [_score_candidate(spec) for spec in specs]
    scored.sort(key=lambda row: (-row["combined_score"], row["window"][0]))
    for rank, row in enumerate(scored, 1):
        row["combined_rank"] = rank
        row["selected"] = rank <= 3
        row["score_scale"] = (
            "编辑评分由分析代理依据画面给出，未经人工复核；1–10分衡量高光感、教学价值、画面/动作证据完整度；"
            "综合分=0.40×高光+0.35×教学价值+0.25×证据质量。不是球员水平或模型置信度。"
        )
        related = [
            event for event in events
            if row["window"][0] <= event.frame <= row["window"][1]
            and event.kind in {"lunge", "swing", "split_step", "base_return"}
        ]
        row["supporting_candidate_ids"] = [
            f"{event.kind}-{event.player.value}-{event.frame}" for event in related
        ]
        row["supporting_event_scores_raw"] = [
            {"event_id": f"{event.kind}-{event.player.value}-{event.frame}",
             "raw_event_model_score": float(event.confidence)}
            for event in related
        ]
    inventory["ranked_candidates"] = scored

    lunge = next(event for event in events if event.kind == "lunge"
                 and event.player == Role.NEAR and event.frame == 84)
    long_lunges = [
        event for event in events
        if event.kind == "lunge" and 459 <= event.frame <= 545
    ]
    if {event.player for event in long_lunges} != {Role.NEAR, Role.FAR}:
        raise RuntimeError("The long exchange does not include both players' lunge events")
    far_overhead = next(
        event for event in events
        if event.kind == "swing" and event.player == Role.FAR
        and event.data.get("overhead") and 612 <= event.frame <= 664
    )
    near_swing = next(event for event in events if event.kind == "swing"
                      and event.player == Role.NEAR and 636 <= event.frame <= 664)
    near = tracks.players[Role.NEAR]
    floor_joint_names = (
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    )
    floor_quality = [
        _raw_pose_quality(near, frame, floor_joint_names)
        for frame in (642, 644, 648, 652, 656, 660, 664)
    ]
    exchange_positions = _court_position_evidence(tracks, (459, 545))
    near_position = exchange_positions["players"]["near"]
    far_position = exchange_positions["players"]["far"]

    def format_xy(values):
        return f'({values[0]:.1f},{values[1]:.1f})'

    exchange_narration = (
        f'近端二维位移约{near_position["net_displacement_2d_m"]:.1f}米，'
        f'远端约{far_position["net_displacement_2d_m"]:.1f}米。'
        '近端先跨步下沉，远端随后移动；练习下沉后主动回位。'
    )
    actions = {
        "third-shot-attack-and-dive": {
            "key": "attack_dive", "rank": 1, "title": "双人上挥与低位伸展",
            "window": [612, 664], "action_frames": [612, 642, 664],
            "action_labels": ["镜头动作开始", "近端伏地时刻", "撑身起步阶段"],
            "evidence_frames": [629, 636, 642, 648, 660, 664],
            "source_shot": [367, 664], "focus": None,
            "narration": MOMENT_TEXT["attack_dive"],
            "hold_points": ["右臂向外伸展", "侧跨落地后开始撑起", "躯干随低位动作倾斜"],
            "claim": "远端球员上挥后，近端球员侧向伸展并落到地面；镜头继续记录她伏地并开始撑身起步。只作画面顺序描述。",
            "practice_cue": "观察侧向低位伸展之后的身体保护与起身节奏；不判断是否触球、救球或得分。",
            "evidence": {
                "visible_observation_frames": [629, 636, 642, 648, 660, 664],
                "far_overhead_swing": _event_dict(far_overhead),
                "near_swing_event": _event_dict(near_swing),
                "near_ground_pose_raw_scores": floor_quality,
                "ground_phase_overlay": (
                    "Per-joint 0.70 model-score and body-box gates; draw supported trunk/legs/right-arm points "
                    "on measured frames, omit only unsupported joints or unmeasured frames."
                ),
                "ground_pose_limit": (
                    "At frame 642, joints below the 0.70 threshold are omitted; unmeasured frames such as 644 "
                    "receive no overlay. From 648 onward, selected right-arm, trunk, and leg joints pass the "
                    "score gate. A score pass does not "
                    "prove anatomical accuracy, so overlay placement was visually checked on the encoded crop."
                ),
                "claim_excludes_contact_outcome_and_intent": True,
            },
            "analysis_points": [
                {"area": "战术", "text": "远端先上挥，近端随后低位伸展；缺少球轨迹，不推断两者的战术因果或结果。"},
                {"area": "腿部", "text": "近端侧向伸展后落地，镜头末段可见开始撑身；只描述可见顺序。"},
                {"area": "手臂", "text": "伸展阶段右臂向画面左侧延长；叠加仅保留过阈值且落在身体框内的点。"},
                {"area": "核心", "text": "躯干随低位动作倾斜；地面阶段不据关键点给出角度或稳定性判断。"},
            ],
        },
        "third-shot-long-lunge-exchange": {
            "key": "long_exchange", "rank": 2, "title": "双人连续跨步",
            "window": [459, 545], "action_frames": [459, 497, 545],
            "action_labels": ["连续移动开始", "近端下沉最低点", "远端回动阶段"],
            "evidence_frames": [463, 473, 490, 497, 525, 538, 545],
            "source_shot": [367, 664], "focus": None,
            "narration": exchange_narration,
            "hold_points": [
                f'近端二维净位移约{near_position["net_displacement_2d_m"]:.1f}米',
                f'远端二维净位移约{far_position["net_displacement_2d_m"]:.1f}米',
                "两人先后跨步下沉后接回动",
            ],
            "claim": "这段连续变化里，近端与远端都出现跨步下沉，再起身移动；不从画面推测球路或触球。",
            "practice_cue": "把伸展后的回动纳入单打多球练习，留意每次重心下沉后如何再次移动。",
            "evidence": {
                "lunge_events": [_event_dict(event) for event in long_lunges],
                "visible_observation_frames": [463, 473, 490, 497, 525, 538, 545],
                "players": ["near", "far"],
                "single_camera_shot": [367, 664],
                "court_position_evidence_2d": exchange_positions,
            },
            "analysis_points": [
                {"area": "战术-近端", "text": f'球场坐标从{format_xy(near_position["start_xy_m"])}m到{format_xy(near_position["end_xy_m"])}m；二维净位移约{near_position["net_displacement_2d_m"]:.1f}m。'},
                {"area": "战术-远端", "text": f'球场坐标从{format_xy(far_position["start_xy_m"])}m到{format_xy(far_position["end_xy_m"])}m；二维净位移约{far_position["net_displacement_2d_m"]:.1f}m。'},
                {"area": "腿部", "text": "近端与远端都出现低位跨步，随后可见起身或继续移动；可练习跨步后的连续回动。"},
                {"area": "核心", "text": "可见下沉与起身伴随躯干姿态变化；仅作二维趋势观察，不作精确角度结论。"},
            ],
        },
        "opening-near-lunge": {
            "key": "legs", "rank": 3, "title": "跨步与回收",
            "window": [60, 109], "action_frames": [84, 89, 99],
            "action_labels": ["跨步开始", "下沉最低点", "回动"],
            "evidence_frames": [86, 89, 99],
            "source_shot": [0, 211], "focus": "near",
            "narration": MOMENT_TEXT["legs"],
            "hold_points": ["右脚跨出、前膝屈曲", "二维膝角估计约114°", "起身后接回收"],
            "claim": "近端球员右脚跨出、前膝屈曲，随后起身回动；这是单摄像机画面中的二维动作观察。",
            "practice_cue": "练习时把跨步、屈膝与回收连成一个连续动作；不把单次角度当成技术好坏标准。",
            "evidence": {
                "event": _event_dict(lunge),
                "plant_frame": int(lunge.data["plant_frame"]),
                "bottom_frame": int(lunge.data["bottom_frame"]),
                "return_frame": int(lunge.data["end_frame"]),
                "lead_side": "right" if lunge.data.get("lead_right") else "not specified",
                "knee_flexion_2d_deg": float(lunge.data["lead_knee_flexion_deg"]),
                "stride_ratio": float(lunge.data["stride_ratio"]),
                "raw_event_model_score": float(lunge.confidence),
            },
            "analysis_points": [
                {"area": "战术", "text": "动作后近端球员向后回收；单个片段不足以评价整体站位策略。"},
                {"area": "腿部", "text": "右脚跨出、前膝屈曲后再起身回动；约114°为单帧二维估计，不作为技术好坏标准。"},
                {"area": "核心", "text": "二维画面里下沉阶段可见躯干倾斜；单视角不足以判断三维姿态或技术好坏。"},
            ],
        },
    }
    selected = []
    for candidate in scored[:3]:
        moment = actions[candidate["candidate_id"]]
        moment["candidate_id"] = candidate["candidate_id"]
        moment["rank"] = candidate["combined_rank"]
        moment["scores"] = {
            key: candidate[key] for key in (
                "highlight_score", "coaching_value_score",
                "evidence_quality_score", "combined_score", "combined_rank",
            )
        }
        moment["rank_reason"] = candidate["reason"]
        moment["window"] = list(candidate["window"])
        moment["source_shot"] = list(candidate["source_shot"])
        selected.append(moment)
    for moment in selected:
        start, end = moment["window"]
        if start > moment["action_frames"][0] or end < moment["action_frames"][-1]:
            raise ValueError(f"{moment['key']} evidence falls outside its selected source window")
        cuts = [shot for shot in tracks.shots if shot[0] <= start and end <= shot[1]]
        if len(cuts) != 1 or cuts[0] != tuple(moment["source_shot"]):
            raise ValueError(f"{moment['key']} selection crosses or mismatches a camera cut")
    selected_by_start = sorted(selected, key=lambda row: row["window"][0])
    for left, right in zip(selected_by_start, selected_by_start[1:]):
        if left["window"][1] >= right["window"][0]:
            raise ValueError("The top-three source moments must be distinct and non-overlapping")
    return selected


def _timecode(frame: int, fps: float) -> str:
    seconds = frame / fps
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - 60 * minutes:05.2f}"


def _frame_sequence(start: int, end: int, speed: float) -> list[int]:
    count = math.ceil((end - start + 1) / speed)
    return [min(end, start + math.floor(index * speed)) for index in range(count)]


def make_schedule(moments: list[dict], speech_durations: list[float],
                  fps: float = FPS, tracks=None) -> dict:
    if len(moments) != 3 or len(speech_durations) != 4:
        raise ValueError("A whole-clip review needs exactly three moments and one recap")
    frame_map: list[int | None] = [None] * INTRO_FRAMES
    states: list[dict] = [{
        "phase": "intro", "source_frame": None, "chapter": "intro",
        "focus": None, "overlay": None, "speed": None, "source_shot": None,
    } for _ in range(INTRO_FRAMES)]
    captions = []
    chapters = []
    for moment, speech_s in zip(moments, speech_durations[:3]):
        start, end = moment["window"]
        key_frame = moment["action_frames"][1]
        if not start < key_frame < end:
            raise ValueError(f"{moment['key']} needs a key frame inside its action window")
        if moment["key"] == "long_exchange" and tracks is None:
            raise ValueError("The two-player court crop needs measured player tracks")
        moment["shared_court_crop"] = (
            _shared_court_crop(tracks, moment["window"])
            if moment["key"] == "long_exchange" else None
        )
        slow_sequence = _frame_sequence(start, key_frame, 0.5)
        follow_sequence = list(range(key_frame + 1, end + 1))
        hold_count = max(1, round(KEY_HOLD_S * fps))
        required_count = math.ceil((speech_s + 0.4) * fps)
        required_slow_count = max(
            len(slow_sequence), required_count - hold_count - len(follow_sequence),
        )
        slow_speed = min(
            0.5, (key_frame - start + 1) / required_slow_count,
        )
        if slow_speed < 0.25:
            raise ValueError(
                f"{moment['key']} narration is too long for its slow-in/hold/follow-through; "
                "shorten the line"
            )
        slow_sequence = _frame_sequence(start, key_frame, slow_speed)
        output_start = len(frame_map) / fps
        def append_source_frame(source_frame, phase, speed, playback_phase,
                                hold_points=None):
            output_index = len(frame_map)
            frame_map.append(source_frame)
            focus, crop_mode, overlays = _visual_state(moment, source_frame)
            states.append({
                "phase": phase,
                "source_frame": source_frame,
                "chapter": moment["key"],
                "focus": focus,
                "crop_mode": crop_mode,
                "overlay": overlays[0]["kind"] if overlays else None,
                "overlays": overlays,
                "speed": speed,
                "playback_phase": playback_phase,
                "hold_points": hold_points or [],
                "source_shot": moment["source_shot"],
                "shared_court_crop": moment["shared_court_crop"],
            })
            return output_index

        slow_output_start = len(frame_map)
        for source_frame in slow_sequence:
            append_source_frame(source_frame, "motion", slow_speed, "slow-in")
        slow_output_end = len(frame_map)
        hold_start = len(frame_map)
        for _ in range(hold_count):
            append_source_frame(
                key_frame, "hold", 0.0, "key-frame-hold", moment["hold_points"],
            )
        hold_end = len(frame_map)
        follow_start = len(frame_map)
        for source_frame in follow_sequence:
            append_source_frame(source_frame, "motion", 1.0, "follow-through")
        follow_end = len(frame_map)
        output_end = len(frame_map) / fps
        speech_start = output_start + 0.1
        moment["playback_speed"] = slow_speed
        moment["output_start_s"] = output_start
        moment["output_end_s"] = output_end
        moment["speech_start_s"] = speech_start
        moment["speech_end_s"] = speech_start + speech_s
        moment["playback"] = {
            "key_frame": key_frame,
            "slow_in_speed": slow_speed,
            "hold_s": hold_count / fps,
            "hold_output_frame": hold_start,
            "hold_output_range_s": [hold_start / fps, hold_end / fps],
            "slow_in": {
                "source_frames": [start, key_frame],
                "output_range_s": [slow_output_start / fps, slow_output_end / fps],
                "speed": slow_speed,
            },
            "follow_through": {
                "source_frames": [key_frame + 1, end],
                "output_range_s": [follow_start / fps, follow_end / fps],
                "speed": 1.0,
            },
            "hold_points": list(moment["hold_points"]),
        }
        moment["source_ranges"] = [
            {
                "phase": "slow-in", "start_frame": start, "end_frame": key_frame,
                "speed": slow_speed,
                "output_start_s": slow_output_start / fps,
                "output_end_s": slow_output_end / fps,
                "source_shot": moment["source_shot"],
            },
            {
                "phase": "key-frame-hold", "start_frame": key_frame,
                "end_frame": key_frame, "speed": 0.0,
                "output_start_s": hold_start / fps,
                "output_end_s": hold_end / fps,
                "source_shot": moment["source_shot"],
            },
            {
                "phase": "follow-through", "start_frame": key_frame + 1,
                "end_frame": end, "speed": 1.0,
                "output_start_s": follow_start / fps,
                "output_end_s": follow_end / fps,
                "source_shot": moment["source_shot"],
            },
        ]
        captions.append({
            "chapter": moment["key"],
            "text": moment["narration"],
            "start_s": speech_start,
            "end_s": speech_start + speech_s,
            "speech_start_s": speech_start,
            "speech_end_s": speech_start + speech_s,
        })
        moment["source_range"] = {
            "start_frame": start, "end_frame": end,
            "phase": "slow-in, key-frame-hold, follow-through",
            "output_start_s": output_start, "output_end_s": output_end,
            "source_shot": moment["source_shot"],
        }
        if speech_start + speech_s > output_end - 0.1:
            raise ValueError(f"{moment['key']} speech does not fit its moving source passage")
        output_source = [frame for frame, state in zip(frame_map, states)
                         if state["chapter"] == moment["key"]]
        if any(right < left for left, right in zip(output_source, output_source[1:])):
            raise ValueError(f"{moment['key']} includes a replay or reversed source motion")
        if not (start <= min(output_source) and max(output_source) <= end):
            raise ValueError(f"{moment['key']} frame map escapes its selected source range")
        for frame in moment["action_frames"]:
            if frame not in output_source:
                raise ValueError(f"Evidence frame {frame} is absent from {moment['key']} output")
    recap_speech_s = speech_durations[3]
    summary_s = max(SUMMARY_MIN_S, recap_speech_s + 0.5)
    summary_frames = math.ceil(summary_s * fps)
    output_start = len(frame_map) / fps
    frame_map.extend([None] * summary_frames)
    states.extend([{
        "phase": "summary", "source_frame": None, "chapter": "recap",
        "focus": None, "overlay": None, "speed": None, "source_shot": None,
    } for _ in range(summary_frames)])
    captions.append({
        "chapter": "recap",
        "text": MOMENT_TEXT["recap"],
        "start_s": output_start + 0.1,
        "end_s": output_start + 0.1 + recap_speech_s,
        "speech_start_s": output_start + 0.1,
        "speech_end_s": output_start + 0.1 + recap_speech_s,
    })
    duration = len(frame_map) / fps
    chapters = [{
        "key": moment["key"], "title": moment["title"],
        "source_shot": moment["source_shot"],
        "source_ranges": moment["source_ranges"],
        "source_start_frame": moment["window"][0],
        "source_end_frame": moment["window"][1],
        "action_frames": moment["action_frames"],
        "output_start_s": moment["output_start_s"],
        "output_end_s": moment["output_end_s"],
        "speech_start_s": moment["speech_start_s"],
        "speech_end_s": moment["speech_end_s"],
        "rank": moment["rank"],
        "claim": moment["claim"],
        "evidence": moment["evidence"],
        "playback": moment["playback"],
    } for moment in moments]
    chapters.append({
        "key": "recap", "title": "三处要点",
        "source_shot": None, "source_ranges": [],
        "source_start_frame": None, "source_end_frame": None,
        "action_frames": [], "output_start_s": output_start,
        "output_end_s": duration,
        "speech_start_s": output_start + 0.1,
        "speech_end_s": output_start + 0.1 + recap_speech_s,
        "rank": None, "claim": "跨步回收、两臂先后、低位后回位。",
        "evidence": {"lines": list(SUMMARY_LINES)},
    })
    return {
        "fps": fps, "width": WIDTH, "height": HEIGHT,
        "planned_duration_s": duration,
        "planned_output_frames": len(frame_map),
        "intro_s": INTRO_FRAMES / fps,
        "summary_s": summary_frames / fps,
        "chapters": chapters,
        "captions": captions,
        "frame_map": frame_map,
        "states": states,
        "summary_lines": list(SUMMARY_LINES),
    }


def _shared_court_crop(tracks, window):
    if not any(start <= window[0] and window[1] <= end for start, end in tracks.shots):
        raise ValueError(f"Shared court crop crosses a camera cut: {window}")
    boxes = []
    for player in tracks.players.values():
        for frame in range(window[0], window[1] + 1):
            if player.measured[frame] and np.isfinite(player.bbox[frame]).all():
                boxes.append(player.bbox[frame].astype(float))
    if not boxes:
        raise ValueError(f"No measured player boxes in shared court window {window}")
    all_boxes = np.asarray(boxes)
    left, top = all_boxes[:, 0].min(), all_boxes[:, 1].min()
    right, bottom = all_boxes[:, 2].max(), all_boxes[:, 3].max()
    aspect = (VIEW[2] - VIEW[0]) / (VIEW[3] - VIEW[1])
    crop_h = max((bottom - top) * 1.25, (right - left) / aspect * 1.12)
    crop_w = crop_h * aspect
    if crop_w > tracks.width:
        crop_w = float(tracks.width)
        crop_h = crop_w / aspect
    if crop_h > tracks.height:
        crop_h = float(tracks.height)
        crop_w = crop_h * aspect
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    x1 = min(max(center_x - crop_w / 2, 0), tracks.width - crop_w)
    y1 = min(max(center_y - crop_h / 2, 0), tracks.height - crop_h)
    rect = (
        int(math.floor(x1)), int(math.floor(y1)),
        int(math.ceil(x1 + crop_w)), int(math.ceil(y1 + crop_h)),
    )
    if not all(
        rect[0] <= box[0] <= box[2] <= rect[2]
        and rect[1] <= box[1] <= box[3] <= rect[3]
        for box in boxes
    ):
        raise ValueError("Shared court crop does not contain all measured players")
    return rect


def _visual_state(moment, source_frame):
    key = moment["key"]
    if key == "attack_dive":
        if source_frame <= 632:
            focus, crop_mode = None, "full"
            overlays = ([{"role": "far", "kind": "right-arm"}]
                        if 625 <= source_frame <= 630 else [])
        else:
            focus, crop_mode = "near", "focus"
            overlays = [{"role": "near", "kind": "dive"}]
    elif key == "long_exchange":
        focus, crop_mode = None, "shared-court"
        overlays = [
            {"role": "near", "kind": "lunge-core"},
            {"role": "far", "kind": "lunge-core"},
        ]
    else:
        focus, crop_mode = moment["focus"], "focus" if moment["focus"] else "full"
        overlays = ([{"role": "near", "kind": "lunge-core"}]
                    if key == "legs" and 80 <= source_frame <= 101 else [])
    return focus, crop_mode, overlays


def _draw_wrapped(draw, text, xy, font, fill, width=880, line_h=60):
    lines = _wrap_zh(draw, text, font, width)
    boxes = []
    x, y = xy
    for line in lines:
        box = draw.textbbox((x, y), line, font=font)
        draw.rounded_rectangle(
            (box[0] - 14, box[1] - 7, box[2] + 14, box[3] + 9),
            radius=12, fill="#07121f",
        )
        draw.text((x, y), line, font=font, fill=fill,
                  stroke_width=1, stroke_fill="#07121f")
        boxes.append([box[0], box[1], box[2], box[3]])
        y += line_h
    return boxes


OVERLAY_THRESHOLD = 0.70


def _supported_joint(player, frame, name):
    if not player.measured[frame]:
        return None, "unmeasured"
    index = JOINT_INDEX[name]
    point = player.joint(name)[frame]
    if not np.isfinite(point).all():
        return None, "non-finite"
    if float(player.conf[frame, index]) < OVERLAY_THRESHOLD:
        return None, "below-score-threshold"
    left, top, right, bottom = (float(value) for value in player.bbox[frame])
    pad_x = max(8.0, (right - left) * 0.08)
    pad_y = max(8.0, (bottom - top) * 0.08)
    if not (left - pad_x <= point[0] <= right + pad_x
            and top - pad_y <= point[1] <= bottom + pad_y):
        return None, "outside-player-box"
    return tuple(np.rint(point).astype(int)), None


def _draw_supported_chain(image, player, frame, names, color, width):
    accepted = {}
    rejected = {}
    for name in names:
        point, reason = _supported_joint(player, frame, name)
        if reason is None:
            accepted[name] = point
        else:
            rejected[name] = reason
    for left, right in zip(names, names[1:]):
        if left in accepted and right in accepted:
            cv2.line(image, accepted[left], accepted[right], (10, 20, 25),
                     width + 4, cv2.LINE_AA)
            cv2.line(image, accepted[left], accepted[right], color, width, cv2.LINE_AA)
    for point in accepted.values():
        cv2.circle(image, point, width + 2, (10, 20, 25), -1, cv2.LINE_AA)
        cv2.circle(image, point, width, color, -1, cv2.LINE_AA)
    return sorted(accepted), rejected


def _overlay_supported(image, tracks, frame, overlay):
    role, kind = overlay["role"], overlay["kind"]
    player = _role(tracks, role)
    if not player.measured[frame]:
        return {
            "role": role, "kind": kind, "drawn_joints": [],
            "rejected": {"frame": "unmeasured"},
        }
    box_h = max(1.0, float(player.bbox[frame, 3] - player.bbox[frame, 1]))
    width = max(3, round(box_h / 55))
    palette = {
        "near": {
            "left": (255, 190, 60), "right": (70, 235, 220),
            "core": (70, 235, 220), "arm": (70, 235, 220),
        },
        "far": {
            "left": (255, 120, 190), "right": (100, 190, 255),
            "core": (255, 120, 190), "arm": (100, 190, 255),
        },
    }[role]
    if kind in ("lunge-core", "dive"):
        chains = [
            (("left_shoulder", "right_shoulder"), palette["core"]),
            (("left_hip", "right_hip"), palette["core"]),
            (("left_hip", "left_knee", "left_ankle"), palette["left"]),
            (("right_hip", "right_knee", "right_ankle"), palette["right"]),
        ]
        if kind == "dive":
            chains.append((("right_shoulder", "right_elbow", "right_wrist"),
                           palette["arm"]))
    elif kind == "right-arm":
        chains = [
            (("right_shoulder", "right_elbow", "right_wrist"), palette["arm"]),
        ]
    else:
        raise ValueError(f"Unsupported evidence overlay: {kind}")
    drawn, rejected = set(), {}
    for names, color in chains:
        accepted, failed = _draw_supported_chain(
            image, player, frame, names, color, width,
        )
        drawn.update(accepted)
        rejected.update(failed)
    return {
        "role": role, "kind": kind, "drawn_joints": sorted(drawn),
        "rejected": rejected,
    }


def _render_card(state, schedule, fonts, credit):
    image = Image.new("RGB", (WIDTH, HEIGHT), "#111d22")
    draw = ImageDraw.Draw(image)
    heading, subhead, caption_font, small, recap_font = fonts[:5]
    boxes = []
    if state["phase"] == "intro":
        boxes += _draw_wrapped(draw, "30秒片段｜三个值得看的瞬间",
                               (78, 410), heading, "#f5f8ff")
        boxes += _draw_wrapped(draw, "全900帧扫描 · 四个镜头 · 仅讨论本段",
                               (82, 545), subhead, "#92d9d4")
        draw.rounded_rectangle((80, 650, 950, 660), radius=5, fill="#58e5dc")
    else:
        boxes += _draw_wrapped(draw, "三处复盘",
                               (78, 245), heading, "#f5f8ff")
        y = 425
        for line in SUMMARY_LINES:
            boxes += _draw_wrapped(draw, line, (82, y), recap_font, "#f2f6ef",
                                   width=875, line_h=68)
            y += 132
        boxes += _draw_wrapped(draw, credit["endcard"].splitlines()[0],
                               (78, 1085), small, "#c8d6dd", width=900, line_h=42)
        boxes += _draw_wrapped(
            draw, "Commons许可标注不代表上游赛事转播权已清理。",
            (78, 1140), small, "#c8d6dd", width=900, line_h=42,
        )
        caption = next(
            row for row in schedule["captions"]
            if row["chapter"] == "recap"
        )
        boxes += _draw_wrapped(draw, caption["text"], (78, 1305), caption_font,
                               "#ffffff", width=875, line_h=62)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), boxes


def _render_motion(raw, tracks, state, moment, schedule, fonts, source_time):
    source_frame = int(state["source_frame"])
    annotated = raw.copy()
    overlay_checks = [
        _overlay_supported(annotated, tracks, source_frame, overlay)
        for overlay in state.get("overlays", [])
    ]
    shot = tuple(state["source_shot"])
    if state.get("crop_mode") == "shared-court":
        crop = tuple(state["shared_court_crop"])
        crop_scale = min(
            (VIEW[2] - VIEW[0]) / (crop[2] - crop[0]),
            (VIEW[3] - VIEW[1]) / (crop[3] - crop[1]),
        )
    else:
        crop, crop_scale = _crop_for_state(
            annotated, tracks, source_frame, state["focus"], shot, _full_crop(tracks, shot)
        )
    pane, output_scale = _fit_source(annotated, crop, VIEW)
    canvas = np.full((HEIGHT, WIDTH, 3), (17, 26, 31), np.uint8)
    canvas[VIEW[1]:VIEW[3], VIEW[0]:VIEW[2]] = pane
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    heading, _, caption_font, small, _, point_font = fonts
    rank = moment["rank"]
    title = f'第{rank}处｜{moment["title"]}'
    boxes = _draw_wrapped(draw, title, (78, 205), heading, "#f5f8ff", width=900)
    if state["phase"] == "hold":
        for index, point in enumerate(state.get("hold_points", [])):
            boxes += _draw_wrapped(
                draw, f'· {point}', (78, 1260 + index * 64), point_font,
                "#f2f6ef", width=875, line_h=46,
            )
    caption = next(
        row for row in schedule["captions"]
        if row["chapter"] == moment["key"]
        and row["start_s"] <= source_time < row["end_s"]
    ) if any(row["chapter"] == moment["key"]
             and row["start_s"] <= source_time < row["end_s"]
             for row in schedule["captions"]) else None
    if caption:
        caption_y = 1540 if state["phase"] == "hold" else 1305
        boxes += _draw_wrapped(draw, caption["text"], (78, caption_y), caption_font,
                               "#ffffff", width=875, line_h=62)
    source_label = (
        f'原片 {source_frame / tracks.fps:05.2f}s · {source_frame}帧'
    )
    source_y = 1815 if state["phase"] == "hold" else 1510
    boxes += _draw_wrapped(draw, source_label, (78, source_y), small,
                           "#c8d6dd", width=870, line_h=38)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), boxes, {
        "source_crop": list(crop), "crop_scale": float(crop_scale),
        "crop_mode": state.get("crop_mode"),
        "overlay_checks": overlay_checks,
        "viewport_scale": float(output_scale),
    }


def verify_output_frame_bindings(video: Path, output: Path, tracks, schedule: dict,
                                 moments: list[dict], fonts) -> list[dict]:
    source = cv2.VideoCapture(str(video))
    rendered = cv2.VideoCapture(str(output))
    if not source.isOpened() or not rendered.isOpened():
        raise RuntimeError("Cannot reopen source and rendered videos for alignment checks")
    checks = []
    try:
        for moment in moments:
            for source_frame in moment["action_frames"]:
                if source_frame == moment["playback"]["key_frame"]:
                    output_index = moment["playback"]["hold_output_frame"]
                else:
                    output_index = next(
                        index for index, (mapped, state) in enumerate(
                            zip(schedule["frame_map"], schedule["states"])
                        )
                        if mapped == source_frame and state["chapter"] == moment["key"]
                        and state["phase"] == "motion"
                    )
                source.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
                source_ok, raw = source.read()
                rendered.set(cv2.CAP_PROP_POS_FRAMES, output_index)
                output_ok, actual = rendered.read()
                if not source_ok or not output_ok:
                    raise RuntimeError(
                        f"Cannot decode bound frame {source_frame} -> {output_index}"
                    )
                state = schedule["states"][output_index]
                expected, _, _ = _render_motion(
                    raw, tracks, state, moment, schedule, fonts, output_index / FPS,
                )
                x1, y1, x2, y2 = VIEW
                delta = cv2.absdiff(expected[y1:y2, x1:x2], actual[y1:y2, x1:x2])
                error = float(delta.mean())
                row = {
                    "chapter": moment["key"],
                    "source_frame": int(source_frame),
                    "output_frame": int(output_index),
                    "output_time_s": output_index / FPS,
                    "viewport_mean_absolute_error": error,
                    "threshold": 20.0,
                    "passed": error <= 20.0,
                }
                checks.append(row)
                if not row["passed"]:
                    raise RuntimeError(
                        f"Source/output alignment failed for frame {source_frame}: "
                        f"viewport MAE {error:.2f}"
                    )
    finally:
        source.release()
        rendered.release()
    return checks


def _analysis_data(video, pose_verification, tracks, inventory, moments, schedule, credit):
    for moment in moments:
        moment["action_timecodes"] = [
            _timecode(frame, tracks.fps) for frame in moment["action_frames"]
        ]
        chapter = next(row for row in schedule["chapters"]
                       if row["key"] == moment["key"])
        moment["output_frame_bindings"] = []
        for source_frame in sorted(set(
                moment["action_frames"] + moment.get("evidence_frames", []))):
            output_index = next(
                index for index, state in enumerate(schedule["states"])
                if state["chapter"] == moment["key"]
                and state["source_frame"] == source_frame
            )
            moment["output_frame_bindings"].append({
                "source_frame": source_frame,
                "source_timecode": _timecode(source_frame, tracks.fps),
                "output_frame": output_index,
                "output_time_s": output_index / FPS,
            })
        action_set = set(moment["action_frames"])
        moment["action_frame_bindings"] = [
            row for row in moment["output_frame_bindings"]
            if row["source_frame"] in action_set
        ]
        moment["key_frame_output_binding"] = {
            "source_frame": moment["playback"]["key_frame"],
            "source_timecode": _timecode(moment["playback"]["key_frame"], tracks.fps),
            "output_frame": moment["playback"]["hold_output_frame"],
            "output_time_s": moment["playback"]["hold_output_frame"] / FPS,
            "hold_output_range_s": moment["playback"]["hold_output_range_s"],
        }
        moment["output_range_s"] = [chapter["output_start_s"], chapter["output_end_s"]]
    source = {
        "video": str(video.resolve()),
        "sha256": pose_verification["source_sha256"],
        "fps": tracks.fps,
        "frame_count": tracks.n,
        "duration_s": tracks.n / tracks.fps,
        "scope": "分析仅覆盖这个30秒剪辑片段，不代表整场比赛。",
        "camera_shots": [list(shot) for shot in tracks.shots],
        "pose_reused": True,
        "pose_verification": pose_verification,
    }
    return {
        "title": "全片精选三处｜羽毛球动作分析",
        "source": source,
        "whole_timeline_inventory": inventory,
        "selected_moments": moments,
        "ranking_method": inventory["ranking_basis"],
        "limitations": [
            "输入只有30秒，且由四个镜头剪接组成；不能代表整场比赛。",
            "单摄像机二维关键点与球场投影；不推断三维旋转、肌肉、受力或医疗结论。",
            "没有羽毛球轨迹或触球检测；不判断触球、球路、落点、胜负或对手意图。",
            "动作观察与一般练习提示分开；原始模型分数不是概率，也不代表球员水平或建议正确率。",
            "画面来源标注为Wikimedia Commons CC BY 3.0；该许可标注不代表上游赛事转播权已清理。",
        ],
        "attribution": credit,
        "narration": [{"chapter": row["chapter"], "text": row["text"],
                       "start_s": row["start_s"], "end_s": row["end_s"]}
                      for row in schedule["captions"]],
    }


def _write_analysis(folder: Path, data: dict):
    _dump(folder / "analysis.json", data)
    cards = []
    for moment in data["selected_moments"]:
        measured = json.dumps({
            "evidence": moment["evidence"],
            "analysis_points": moment["analysis_points"],
            "output_frame_bindings": moment["output_frame_bindings"],
            "playback": moment["playback"],
        }, ensure_ascii=False, indent=2)
        frame_text = " → ".join(
            f'{label} {frame}帧（{tc}）'
            for label, frame, tc in zip(
                moment["action_labels"], moment["action_frames"],
                moment["action_timecodes"],
            )
        )
        analysis_points = "".join(
            f'<li><b>{html.escape(point["area"])}：</b>'
            f'{html.escape(point["text"])}</li>'
            for point in moment["analysis_points"]
        )
        cards.append(
            f'<article><h2>第{moment["rank"]}处｜{html.escape(moment["title"])}</h2>'
            f'<p><b>动作时间：</b>{html.escape(frame_text)} · '
            f'<b>输出时间：</b>{moment["output_range_s"][0]:.2f}–'
            f'{moment["output_range_s"][1]:.2f}s</p>'
            f'<p><b>关键帧：</b>源帧{moment["key_frame_output_binding"]["source_frame"]} · '
            f'输出冻结{moment["playback"]["hold_output_range_s"][0]:.2f}–'
            f'{moment["playback"]["hold_output_range_s"][1]:.2f}s'
            f'（{moment["playback"]["hold_s"]:.1f}秒） · '
            f'<b>慢入：</b>{moment["playback"]["slow_in_speed"]:.2f}×</p>'
            f'<ul class="points">{analysis_points}</ul>'
            f'<p><b>练习提示：</b>{html.escape(moment["practice_cue"])}</p>'
            f'<p><b>旁白：</b>{html.escape(moment["narration"])}</p>'
            f'<details><summary>展开原始关键点、事件与帧绑定 JSON</summary><pre>'
            f'{html.escape(measured)}</pre></details></article>'
        )
    shot_rows = "".join(
        f'<tr><td>镜头 {row["shot_index"] + 1}</td>'
        f'<td>{row["source_frames"][0]}–{row["source_frames"][1]}</td>'
        f'<td>{row["step_events"]}</td>'
        f'<td>{row["candidate_event_count"]}</td>'
        f'<td>{html.escape("、".join(row["candidate_ids"]) or "无显著动作事件")}</td></tr>'
        for row in data["whole_timeline_inventory"]["shots"]
    )
    candidate_rows = "".join(
        f'<tr><td>{row["combined_rank"]}{" ✓" if row["selected"] else ""}</td>'
        f'<td>{html.escape(row["candidate_id"])} · '
        f'{row["window"][0]}–{row["window"][1]}</td>'
        f'<td>{row["highlight_score"]:.1f}</td>'
        f'<td>{row["coaching_value_score"]:.1f}</td>'
        f'<td>{row["evidence_quality_score"]:.1f}</td>'
        f'<td>{row["combined_score"]:.2f}</td>'
        f'<td>{html.escape(row["reason"])}</td></tr>'
        for row in data["whole_timeline_inventory"]["ranked_candidates"]
    )
    event_json = json.dumps(
        data["whole_timeline_inventory"]["supporting_events"],
        ensure_ascii=False, indent=2,
    )
    criteria = "".join(
        f'<li>{html.escape(value)}</li>' for value in data["ranking_method"]
    )
    limitations = "".join(
        f'<li>{html.escape(value)}</li>' for value in data["limitations"]
    )
    attribution = html.escape(data["attribution"]["full"])
    (folder / "analysis.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>全片精选三处｜羽毛球动作分析</title>'
        '<style>body{font:18px/1.75 "Microsoft YaHei",sans-serif;background:#101a1f;color:#edf6f1;'
        'max-width:980px;margin:28px auto;padding:24px}video{width:100%;background:#000}'
        'article{padding:22px;background:#1a2a2c;margin:18px 0;border-radius:14px}'
        '.points li{margin:.5em 0}'
        'table{border-collapse:collapse;width:100%;font-size:15px}td,th{border-bottom:1px solid #47605e;'
        'padding:8px;vertical-align:top}a{color:#7ee5d0}pre{white-space:pre-wrap;overflow-wrap:anywhere;'
        'font-size:14px}summary{cursor:pointer;color:#a6d9ce}</style>'
        '<h1>全片精选三处</h1>'
        '<p>900帧、四个镜头全部纳入候选审阅；结论只针对这段30秒剪辑。</p>'
        '<video controls preload="metadata" poster="cover.jpg" src="three-moments.mp4"></video>'
        '<p><a download href="three-moments.mp4">下载视频</a> · '
        '<a download href="captions.srt">字幕</a> · '
        '<a download href="analysis.json">机器可读分析</a> · '
        '<a download href="frame-map.json">逐帧来源映射</a></p>'
        + "".join(cards)
        + '<h2>全时间线候选审阅</h2>'
        '<p>候选覆盖四个镜头，并包括459–545帧的双人跨步长回合及612–664帧的上挥—低位伸展片段。'
        '1–10分为编辑评分：由分析代理依据画面给出，未经人工复核，衡量高光感、教学价值与证据可读性；综合分按0.40/0.35/0.25加权。'
        '第二镜头保留了低分候选，不会因没有高光事件而从全片扫描中消失。</p>'
        '<table><thead><tr><th>镜头</th><th>源帧</th><th>步伐事件</th>'
        '<th>分析候选</th><th>候选清单</th></tr></thead><tbody>'
        + shot_rows + '</tbody></table>'
        '<h3>候选评分与排名</h3><table><thead><tr><th>名次</th><th>候选与源帧</th>'
        '<th>高光</th><th>教学</th><th>证据</th><th>综合</th><th>理由</th></tr></thead><tbody>'
        + candidate_rows + '</tbody></table>'
        '<details><summary>展开全时间线原始候选事件 JSON（含原始模型分数）</summary><pre>'
        + html.escape(event_json) + '</pre></details>'
        '<h2>排名依据</h2><ul>' + criteria + '</ul>'
        '<h2>证据与局限</h2><ul>' + limitations + '</ul>'
        '<h2>来源与许可</h2><pre>' + attribution + '</pre></html>',
        "utf-8",
    )


def finalize_handoff(folder: Path, tests: dict | None = None) -> dict:
    analysis = json.loads((folder / "analysis.json").read_text("utf-8"))
    schedule = json.loads((folder / "schedule.json").read_text("utf-8"))
    media = json.loads((folder / "media-render.json").read_text("utf-8"))
    run = json.loads((folder / "run.json").read_text("utf-8"))
    video = folder / "three-moments.mp4"
    audit_path = folder / "media-audit.json"
    if audit_path.is_file():
        media["independent_media_audit"] = json.loads(audit_path.read_text("utf-8"))
    root = Path(__file__).resolve().parents[1]
    changed_paths = (
        "coach/whole_clip_review.py",
        "coach/tests/test_whole_clip_review.py",
        "coach/README.md",
    )
    changed_sources = {
        relative: {
            "path": str((root / relative).resolve()),
            "sha256": sha256_file(root / relative),
        }
        for relative in changed_paths
    }
    artifacts = {
        path.name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for path in sorted(folder.iterdir())
        if path.is_file() and path.name != "handoff.json"
    }
    handoff = {
        "status": "rendered",
        "source": analysis["source"],
        "video": str(video.resolve()),
        "video_sha256": sha256_file(video),
        "planned_duration_s": schedule["planned_duration_s"],
        "candidate_ranking": analysis["whole_timeline_inventory"]["ranked_candidates"],
        "score_definition": analysis["whole_timeline_inventory"]["score_definition"],
        "selected_moments": [{
            "rank": row["rank"], "title": row["title"],
            "source_frames": row["action_frames"],
            "source_timecodes": row["action_timecodes"],
            "source_range": row["source_range"],
            "output_range_s": row["output_range_s"],
            "speech_range_s": [row["speech_start_s"], row["speech_end_s"]],
            "claim": row["claim"],
            "evidence": row["evidence"],
            "analysis_points": row["analysis_points"],
            "playback": row["playback"],
            "key_frame_output_binding": row["key_frame_output_binding"],
        } for row in analysis["selected_moments"]],
        "media": media,
        "attribution": analysis["attribution"],
        "tests": tests or {"status": "pending"},
        "artifacts": artifacts,
        "changed_source_files": changed_sources,
        "limitations": analysis["limitations"],
        "run": run,
    }
    _dump(folder / "handoff.json", handoff)
    return handoff


def export_review(video: Path, out_dir: Path, pose_path: Path,
                  points_path: Path, source_analysis_path: Path, pose_handoff_path: Path,
                  model_dir: Path = VOICE) -> dict:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output folder: {out_dir}")
    pose_verification = verify_pose_binding(
        video, pose_path, points_path, source_analysis_path, pose_handoff_path,
    )
    doc = json.loads(pose_path.read_text("utf-8"))
    report = analyze(doc)
    tracks = load_tracks(doc)
    inventory = candidate_inventory(report, tracks)
    moments = choose_moments(report, tracks, inventory)
    if len(moments) != 3 or len({tuple(row["source_shot"]) for row in moments}) < 2:
        raise RuntimeError("The full timeline did not yield three distinct, multi-shot moments")
    credit = source_credit(video)
    speeches = _synthesize(
        [(moment["key"], moment["narration"]) for moment in moments]
        + [("recap", MOMENT_TEXT["recap"])],
        model_dir,
    )
    durations = [len(entry["samples"]) / entry["rate"] for entry in speeches]
    schedule = make_schedule(moments, durations, tracks.fps, tracks)
    sound_rate = speeches[0]["rate"]
    for entry in speeches:
        if entry["rate"] != sound_rate:
            raise RuntimeError("Local neural voice returned inconsistent sample rates")

    out_dir.mkdir(parents=True)
    _dump(out_dir / "schedule.json", {
        key: value for key, value in schedule.items() if key not in ("frame_map", "states")
    })
    _dump(out_dir / "frame-map.json", {
        "fps": FPS,
        "source_frame_per_output_frame": schedule["frame_map"],
        "states": schedule["states"],
    })
    data = _analysis_data(
        video, pose_verification, tracks, inventory, moments, schedule, credit,
    )
    _write_analysis(out_dir, data)
    (out_dir / "narration.txt").write_text(
        "\n".join(f'{row["chapter"]}：{row["text"]}'
                  for row in schedule["captions"]) + "\n",
        "utf-8",
    )
    (out_dir / "captions.srt").write_text(
        "\n\n".join(
            f'{index}\n{_srt_time(row["start_s"])} --> {_srt_time(row["end_s"])}\n{row["text"]}'
            for index, row in enumerate(schedule["captions"], 1)
        ) + "\n", "utf-8",
    )
    font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    fonts = tuple(ImageFont.truetype(str(font_dir / "msyh.ttc"), size)
                  for size in (64, 42, 44, 28, 52, 36))
    ffmpeg, ffprobe = _tools()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot read source video: {video}")
    sound = np.zeros(math.ceil(schedule["planned_duration_s"] * sound_rate), np.float32)
    for caption, speech in zip(schedule["captions"], speeches):
        offset = round(caption["start_s"] * sound_rate)
        samples = speech["samples"]
        sound[offset:offset + len(samples)] = samples
    text_boxes, crop_checks, overlay_checks = [], [], []
    started = time.perf_counter()
    output = out_dir / "three-moments.mp4"
    with tempfile.TemporaryDirectory(prefix="whole-clip-", dir=out_dir) as temporary:
        temp = Path(temporary)
        voice_path = temp / "voice.wav"
        with wave.open(str(voice_path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(sound_rate)
            stream.writeframes((np.clip(sound, -1, 1) * 32767).astype("<i2").tobytes())
        with (temp / "encoder.log").open("wb") as error_log:
            proc = subprocess.Popen([
                str(ffmpeg), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0",
                "-i", str(voice_path), "-c:v", "libx264", "-preset", "fast",
                "-crf", "19", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                "-movflags", "+faststart", str(output),
            ], stdin=subprocess.PIPE, stderr=error_log)
            previous_frame = None
            raw = None
            try:
                for index, state in enumerate(schedule["states"]):
                    if state["phase"] in ("intro", "summary"):
                        rendered, boxes = _render_card(state, schedule, fonts, credit)
                        crop_info = None
                    else:
                        source_frame = int(state["source_frame"])
                        if source_frame != previous_frame:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
                            ok, raw = cap.read()
                            if not ok:
                                raise RuntimeError(f"Could not decode source frame {source_frame}")
                            previous_frame = source_frame
                        moment = next(row for row in moments if row["key"] == state["chapter"])
                        rendered, boxes, crop_info = _render_motion(
                            raw, tracks, state, moment, schedule, fonts, index / FPS,
                        )
                        overlay_checks.extend(
                            {"chapter": state["chapter"], **row}
                            for row in crop_info["overlay_checks"]
                        )
                        if state["focus"]:
                            player = _role(tracks, state["focus"])
                            measured = _nearest_measured(
                                player, source_frame, tuple(state["source_shot"])
                            )
                            bbox = player.bbox[measured]
                            left, top, right, bottom = crop_info["source_crop"]
                            crop_checks.append({
                                "chapter": state["chapter"],
                                "source_frame": source_frame,
                                "focus": state["focus"],
                                "player_height_px": float(bbox[3] - bbox[1]),
                                "crop": crop_info["source_crop"],
                                "contains_player": bool(
                                    left <= bbox[0] <= bbox[2] <= right
                                    and top <= bbox[1] <= bbox[3] <= bottom
                                ),
                            })
                        elif state.get("crop_mode") == "shared-court":
                            left, top, right, bottom = crop_info["source_crop"]
                            for role in ("near", "far"):
                                player = _role(tracks, role)
                                if not player.measured[source_frame]:
                                    continue
                                bbox = player.bbox[source_frame]
                                crop_checks.append({
                                    "chapter": state["chapter"],
                                    "source_frame": source_frame,
                                    "focus": role,
                                    "player_height_px": float(bbox[3] - bbox[1]),
                                    "crop": crop_info["source_crop"],
                                    "contains_player": bool(
                                        left <= bbox[0] <= bbox[2] <= right
                                        and top <= bbox[1] <= bbox[3] <= bottom
                                    ),
                                })
                    if index == 0 and not cv2.imwrite(str(out_dir / "cover.jpg"), rendered):
                        raise RuntimeError("Cannot write the rendered cover")
                    text_boxes.extend(boxes)
                    proc.stdin.write(rendered.tobytes())
                proc.stdin.close()
                if proc.wait() != 0:
                    raise RuntimeError((temp / "encoder.log").read_text(
                        "utf-8", errors="replace"
                    ))
            finally:
                cap.release()
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
    subprocess.run([str(ffmpeg), "-v", "error", "-i", str(output), "-f", "null", "-"],
                   check=True, capture_output=True)
    alignment_checks = verify_output_frame_bindings(
        video, output, tracks, schedule, moments, fonts,
    )
    media = {
        "full_decode": True,
        "source_fps": tracks.fps,
        "source_frames": tracks.n,
        "source_sha256": pose_verification["source_sha256"],
        "pose_sha256": pose_verification["pose_sha256"],
        "output_frames": len(schedule["frame_map"]),
        "output_duration_s": schedule["planned_duration_s"],
        "audio_sample_rate": sound_rate,
        "all_audio_samples_nonzero": bool(np.any(sound != 0)),
        "render_elapsed_s": time.perf_counter() - started,
        "crop_checks": {
            "focused_frames": len(crop_checks),
            "contains_player": all(row["contains_player"] for row in crop_checks),
            "minimum_player_height_px": min(
                (row["player_height_px"] for row in crop_checks), default=None
            ),
            "players_checked_in_shared_crops": sum(
                row["focus"] in ("near", "far") for row in crop_checks
            ),
        },
        "overlay_validation": {
            "minimum_raw_model_score": OVERLAY_THRESHOLD,
            "on_player_box_check": True,
            "frames_with_overlay": sum(bool(state.get("overlays"))
                                       for state in schedule["states"]),
            "drawn_joint_observations": sum(
                len(row["drawn_joints"]) for row in overlay_checks
            ),
            "rejected_joint_observations": sum(
                len(row["rejected"]) for row in overlay_checks
            ),
            "by_chapter": {
                chapter: {
                    "overlay_frames": sum(row["chapter"] == chapter
                                          for row in overlay_checks),
                    "drawn_joint_observations": sum(
                        len(row["drawn_joints"]) for row in overlay_checks
                        if row["chapter"] == chapter
                    ),
                    "rejected_joint_observations": sum(
                        len(row["rejected"]) for row in overlay_checks
                        if row["chapter"] == chapter
                    ),
                }
                for chapter in sorted({row["chapter"] for row in overlay_checks})
            },
        },
        "source_frame_alignment": {
            "checked_action_frames": len(alignment_checks),
            "passed": all(row["passed"] for row in alignment_checks),
            "checks": alignment_checks,
        },
        "text_box_count": len(text_boxes),
        "camera_cut_windows": [list(shot) for shot in tracks.shots],
    }
    _dump(out_dir / "media-render.json", media)
    run = {
        "full_source_inference": False,
        "pose_reused_after_hash_and_full_coverage_verification": True,
        "timeline_frames_reviewed": tracks.n,
        "candidate_counts": inventory["event_counts"],
        "candidate_events_ranked": inventory["candidate_event_count"],
        "candidate_windows_checked_per_camera_shot": inventory["shots"],
        "render_elapsed_s": media["render_elapsed_s"],
    }
    _dump(out_dir / "run.json", run)
    handoff = finalize_handoff(out_dir)
    return {
        "output": str(out_dir.resolve()),
        "video": str(output.resolve()),
        "duration_s": schedule["planned_duration_s"],
        "sha256": sha256_file(output),
        "handoff": handoff,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--pose-points", type=Path, required=True)
    parser.add_argument("--pose-analysis", type=Path, required=True)
    parser.add_argument("--pose-handoff", type=Path, required=True)
    parser.add_argument("--voice-model", type=Path, default=VOICE)
    args = parser.parse_args(argv)
    from turbo.accept import acquire_cpu_lock, release_cpu_lock
    lock = PROJECT / "work" / "cpu.lock"
    token = f"whole-clip-review {os.getpid()} {uuid.uuid4().hex}"
    acquire_cpu_lock(lock, token)
    try:
        result = export_review(
            args.video, args.out, args.pose, args.pose_points,
            args.pose_analysis, args.pose_handoff, args.voice_model,
        )
        print(json.dumps(result, ensure_ascii=True))
    finally:
        release_cpu_lock(lock, token)


if __name__ == "__main__":
    main()
