from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from coach.events import CONTACT_DELAY_S, MAX_SWING_S, detect_events
from coach.tracks import Tracks, load_tracks


FPS = 30.0
COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4
NET_Y_M = 6.7
SINGLES_LEFT_M = 0.46
SINGLES_RIGHT_M = 5.64
DEPTH_FRONT_M = 1.98
DEPTH_REAR_M = 4.467
POSE_EVENT_KINDS = {"swing"}


@dataclass(frozen=True)
class SegmentRun:
    number: int
    start_s: float
    end_s: float
    pose_path: Path
    cuts: tuple[int, ...]
    click_offset_s: float
    clicks: tuple[dict[str, float], ...]
    source_chunks: tuple[int, ...] = ()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_player_attribution(video: Path, attribution_path: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if attribution_path is None:
        return (
            {role: {"hand": "unknown", "evidence": "no player-hand attribution supplied"}
             for role in ("near", "far")},
            {"status": "unavailable"},
        )
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
    source_name = Path(str(attribution.get("source", {}).get("file", ""))).name
    if source_name.casefold() != video.name.casefold():
        raise ValueError(
            f"Player attribution source {source_name!r} does not match input video {video.name!r}"
        )
    clip = next(
        (item for item in attribution.get("clips", []) if isinstance(item.get("players"), dict)),
        None,
    )
    if clip is None:
        raise ValueError(f"No player-hand evidence is listed for {source_name}")
    players = {}
    for role in ("near", "far"):
        evidence = clip["players"].get(role, {})
        hand = evidence.get("hand", "unknown")
        if hand not in {"left", "right"} or not evidence.get("evidence"):
            players[role] = {"hand": "unknown", "evidence": "no validated hand evidence"}
        else:
            players[role] = {"hand": hand, "evidence": str(evidence["evidence"])}
    return players, {
        "status": "matched_source_attribution",
        "path": str(attribution_path.resolve()),
        "sha256": _sha256(attribution_path),
        "source_file": source_name,
        "clip": clip["file"],
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if hasattr(value, "value"):
        return _jsonable(value.value)
    return value


def score_pair(text: str) -> tuple[int, int] | None:
    try:
        left, right = text.split("-", maxsplit=1)
        return int(left), int(right)
    except (AttributeError, ValueError):
        return None


def group_score_states(states: list[dict[str, Any]], gap_tolerance_s: float = 0.26) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for state in states:
        score = score_pair(str(state.get("score_han_wang", "")))
        if score is None:
            continue
        current = {
            "game": int(state["game"]),
            "score_han_wang": str(state["score_han_wang"]),
            "score": score,
            "start_s": float(state["start_s"]),
            "end_s": float(state["end_s"]),
            "state_indices": [int(state["k"])],
            "serve_icon": state.get("serve_icon"),
        }
        if (
            groups
            and groups[-1]["game"] == current["game"]
            and groups[-1]["score"] == current["score"]
            and current["start_s"] - groups[-1]["end_s"] <= gap_tolerance_s
        ):
            groups[-1]["end_s"] = current["end_s"]
            groups[-1]["state_indices"].extend(current["state_indices"])
            if current["serve_icon"] is not None:
                groups[-1]["serve_icon"] = current["serve_icon"]
        else:
            groups.append(current)
    for index, group in enumerate(groups):
        next_group = groups[index + 1] if index + 1 < len(groups) else None
        winner = None
        winner_reason = "no_next_score"
        if next_group and next_group["game"] == group["game"]:
            before, after = group["score"], next_group["score"]
            changed = (after[0] - before[0], after[1] - before[1])
            if changed == (1, 0):
                winner, winner_reason = "Han Yue", "one_point_change"
            elif changed == (0, 1):
                winner, winner_reason = "Wang Zhiyi", "one_point_change"
            else:
                winner_reason = "points_skipped_or_nonstandard_change"
        group["winner"] = winner
        group["winner_reason"] = winner_reason
    return groups


def infer_landing_zone(x_m: float | None, y_m: float | None, receiver_role: str | None) -> dict[str, Any]:
    if x_m is None or y_m is None or receiver_role not in {"near", "far"}:
        return {"label": "unknown", "depth": None, "side": None, "method": "inferred_from_next_contact"}
    depth_from_net = y_m - NET_Y_M if receiver_role == "near" else NET_Y_M - y_m
    if not (0.0 <= depth_from_net <= 6.7) or not (SINGLES_LEFT_M <= x_m <= SINGLES_RIGHT_M):
        return {"label": "unknown", "depth": None, "side": None, "method": "inferred_from_next_contact"}
    depth = "front" if depth_from_net < DEPTH_FRONT_M else "mid" if depth_from_net < DEPTH_REAR_M else "rear"
    width = SINGLES_RIGHT_M - SINGLES_LEFT_M
    fraction = (x_m - SINGLES_LEFT_M) / width
    side = "left" if fraction < 1 / 3 else "center" if fraction < 2 / 3 else "right"
    return {
        "label": f"{depth}-{side}",
        "depth": depth,
        "side": side,
        "method": "inferred_from_next_contact",
        "estimated_depth_from_net_m": round(depth_from_net, 3),
        "estimated_screen_x_m": round(x_m, 3),
    }


def infer_against(movement: list[float] | None, next_move: list[float] | None) -> dict[str, Any]:
    if movement is None or next_move is None:
        return {"against": None, "direction_dot": None}
    m = np.asarray(movement, dtype=float)
    n = np.asarray(next_move, dtype=float)
    m_norm, n_norm = float(np.linalg.norm(m)), float(np.linalg.norm(n))
    if m_norm < 0.15 or n_norm < 0.15:
        return {"against": None, "direction_dot": None}
    dot = float(np.dot(m / m_norm, n / n_norm))
    return {"against": dot < -0.15, "direction_dot": round(dot, 3)}


def is_receiver_next_contact(hitter: str | None, next_hitter: str | None) -> bool:
    return hitter in {"near", "far"} and next_hitter == ("far" if hitter == "near" else "near")


def infer_shot_family(
    *,
    contact_height: str | None,
    hitter_depth: str | None,
    receiver_zone: dict[str, Any] | None,
    time_to_next_s: float | None,
    is_serve: bool,
    serve_supported: bool,
) -> dict[str, Any]:
    if is_serve:
        if serve_supported:
            return {"family": "serve", "confidence": 0.55, "rule": "first contact with serve evidence"}
        return {"family": "unknown", "confidence": 0.0, "rule": "serve role or service evidence unavailable"}
    if contact_height not in {"overhead", "side", "underarm"} or not receiver_zone or receiver_zone.get("depth") is None:
        return {"family": "unknown", "confidence": 0.0, "rule": "required contact or next-contact evidence unavailable"}
    if time_to_next_s is None or not 0.0 < time_to_next_s <= 3.0:
        return {"family": "unknown", "confidence": 0.0, "rule": "next-contact timing unavailable"}
    depth = receiver_zone["depth"]
    if contact_height == "overhead":
        if time_to_next_s <= 0.48 and depth in {"mid", "rear"}:
            return {"family": "smash", "confidence": 0.4, "rule": "overhead proxy; short next-contact interval; receiver reached mid/rear"}
        if hitter_depth == "rear" and depth == "rear" and time_to_next_s >= 0.7:
            return {"family": "clear", "confidence": 0.35, "rule": "rear-court overhead proxy; long next-contact interval"}
        if depth == "front" and time_to_next_s >= 0.35:
            return {"family": "drop", "confidence": 0.35, "rule": "overhead proxy; receiver next contacts in front zone"}
    elif contact_height == "underarm":
        if hitter_depth == "front" and depth == "front":
            return {"family": "net", "confidence": 0.35, "rule": "front-court underarm proxy; receiver next contacts in front zone"}
        if depth == "rear":
            return {"family": "lift", "confidence": 0.35, "rule": "underarm proxy; receiver next contacts in rear zone"}
        if time_to_next_s <= 0.45:
            return {"family": "drive/push", "confidence": 0.3, "rule": "underarm proxy; short next-contact interval"}
    elif contact_height == "side":
        if time_to_next_s <= 0.42 and depth in {"mid", "rear"}:
            return {"family": "drive/push", "confidence": 0.3, "rule": "side-height proxy; short next-contact interval"}
        if depth == "front" and time_to_next_s >= 0.45:
            return {"family": "drop", "confidence": 0.3, "rule": "side-height proxy; receiver next contacts in front zone"}
    return {"family": "unknown", "confidence": 0.0, "rule": "available evidence does not distinguish a shot family"}


def detect_same_angle_cuts(tracks: Tracks, threshold_m: float = 2.0) -> list[int]:
    cuts: list[int] = []
    if tracks.n < 2:
        return cuts
    for frame in range(1, tracks.n):
        distances: list[float] = []
        for role in ("near", "far"):
            player = tracks.players[role]
            if not (player.measured[frame - 1] and player.measured[frame]):
                break
            a, b = player.foot_court[frame - 1], player.foot_court[frame]
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                break
            distances.append(float(np.linalg.norm(b - a)))
        if len(distances) == 2 and min(distances) >= threshold_m:
            cuts.append(frame)
    return cuts


def detect_audio_clicks(pcm_bytes: bytes, sample_rate: int = 16000, window_s: float = 0.02) -> list[dict[str, float]]:
    samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32)
    if samples.size < 2:
        return []
    diff = np.diff(samples)
    width = max(1, round(sample_rate * window_s))
    count = diff.size // width
    if count == 0:
        return []
    windows = diff[: count * width].reshape(count, width)
    rms = np.sqrt(np.mean(windows * windows, axis=1))
    baseline = float(np.median(rms))
    threshold = max(baseline * 4.0, 1.0)
    peaks = []
    last_s = -1.0
    for index, level in enumerate(rms):
        if level < threshold:
            continue
        left = rms[index - 1] if index > 0 else -math.inf
        right = rms[index + 1] if index + 1 < count else -math.inf
        if level < left or level < right:
            continue
        time_s = (index + 0.5) * window_s
        if time_s - last_s < 0.10:
            continue
        peaks.append({"time_s": round(time_s, 4), "rms": round(float(level), 3),
                      "baseline_rms": round(baseline, 3),
                      "ratio": round(float(level / baseline), 3) if baseline > 0 else None})
        last_s = time_s
    return peaks


def _role_value(role: Any) -> str:
    return str(getattr(role, "value", role))


def _pose_events(tracks: Tracks) -> list[dict[str, Any]]:
    result = []
    for event in detect_events(tracks):
        kind = str(event.kind)
        if kind not in POSE_EVENT_KINDS:
            continue
        result.append({
            "time_s": float(tracks.t[event.frame]),
            "frame": int(event.frame),
            "role": _role_value(event.player),
            "confidence": float(event.confidence),
            "kind": kind,
            "data": _jsonable(dict(event.data)),
        })
    return result


def _nearest_measured(tracks: Tracks, role: str, frame: int, max_offset: int = 8) -> dict[str, Any]:
    player = tracks.players[role]
    start = max(0, frame - max_offset)
    end = min(tracks.n, frame + max_offset + 1)
    eligible = [
        i for i in range(start, end)
        if player.measured[i] and np.isfinite(player.foot_court[i]).all()
    ]
    if not eligible:
        return {"xy_m": None, "frame": None, "offset_frames": None, "measured": False, "court_ok": tracks.court is not None}
    found = min(eligible, key=lambda i: (abs(i - frame), i))
    xy = player.foot_court[found]
    return {
        "xy_m": [round(float(xy[0]), 3), round(float(xy[1]), 3)],
        "frame": int(found),
        "offset_frames": int(found - frame),
        "measured": True,
        "court_ok": tracks.court is not None,
    }


def _movement_before(tracks: Tracks, role: str, frame: int, seconds: float = 0.2) -> dict[str, Any]:
    player = tracks.players[role]
    start_t = tracks.t[frame] - seconds
    indices = [
        i for i in range(max(0, frame - math.ceil(seconds * tracks.fps) - 2), frame + 1)
        if player.measured[i] and np.isfinite(player.foot_court[i]).all()
        and start_t <= tracks.t[i] <= tracks.t[frame]
        and any(a <= i <= b for a, b in tracks.shots)
        and any(a <= frame <= b and a <= i for a, b in tracks.shots)
    ]
    if len(indices) < 2:
        return {"vector_m": None, "speed_m_s": None, "direction": None, "samples": len(indices)}
    first, last = indices[0], indices[-1]
    elapsed = float(tracks.t[last] - tracks.t[first])
    if elapsed <= 0:
        return {"vector_m": None, "speed_m_s": None, "direction": None, "samples": len(indices)}
    vector = tracks.players[role].foot_court[last] - tracks.players[role].foot_court[first]
    speed = float(np.linalg.norm(vector) / elapsed)
    direction = None
    if speed >= 0.15:
        direction = (
            ("right" if vector[0] > 0 else "left")
            + "-"
            + ("near" if vector[1] > 0 else "far")
        )
    return {
        "vector_m": [round(float(vector[0]), 3), round(float(vector[1]), 3)],
        "speed_m_s": round(speed, 3),
        "direction": direction,
        "samples": len(indices),
        "from_frame": first,
        "to_frame": last,
    }


def _height_at_contact(tracks: Tracks, role: str, frame: int) -> dict[str, Any]:
    player = tracks.players[role]
    if not player.measured[frame]:
        return {"label": None, "evidence": "contact frame not measured"}
    hand = player.hand
    if hand not in {"left", "right"}:
        return {"label": None, "evidence": "hitting-hand attribution is unknown"}
    joints = player.xy[frame]
    conf = player.conf[frame]
    wrist_index = 9 if hand == "left" else 10
    indices = (0, wrist_index, 11, 12)
    if any(not np.isfinite(joints[i]).all() or conf[i] < 0.5 for i in indices):
        return {"label": None, "evidence": f"head, {hand} wrist or hip keypoint confidence below 0.5"}
    nose_y, wrist_y = float(joints[0][1]), float(joints[wrist_index][1])
    hip_y = float((joints[11][1] + joints[12][1]) / 2)
    if wrist_y < nose_y - 0.03 * tracks.height:
        label = "overhead"
    elif wrist_y > hip_y + 0.03 * tracks.height:
        label = "underarm"
    else:
        label = "side"
    return {
        "label": label,
        "evidence": f"attributed {hand} wrist relative to nose and hip mid-point in one camera view",
        "wrist_to_head_px": round(wrist_y - nose_y, 2),
        "wrist_to_hip_px": round(wrist_y - hip_y, 2),
        "2d_only": True,
    }


def _depth_from_net(point: dict[str, Any] | None, role: str) -> str | None:
    if not point or point.get("xy_m") is None:
        return None
    y = float(point["xy_m"][1])
    distance = y - NET_Y_M if role == "near" else NET_Y_M - y
    if not 0 <= distance <= 6.7:
        return None
    return "front" if distance < DEPTH_FRONT_M else "mid" if distance < DEPTH_REAR_M else "rear"


def _balance_state(tracks: Tracks, role: str, frame: int, lunge_events: list[dict[str, Any]]) -> dict[str, Any]:
    lunge = min(
        (e for e in lunge_events if e["role"] == role),
        key=lambda e: abs(int(e["data"].get("bottom_frame", e["frame"])) - frame),
        default=None,
    )
    bottom = int(lunge["data"].get("bottom_frame", lunge["frame"])) if lunge else None
    recovery = int(lunge["data"].get("end_frame", lunge["frame"])) if lunge else None
    if lunge and bottom - round(0.08 * tracks.fps) <= frame <= recovery:
        return {"state": "stretched-lunge", "evidence": "pose-derived lunge event within 0.25 s",
                "event_frame": lunge["frame"], "bottom_frame": bottom, "recovery_frame": recovery,
                "confidence": lunge["confidence"]}
    movement = _movement_before(tracks, role, frame)
    if movement["speed_m_s"] is None:
        return {"state": "unknown", "evidence": "insufficient measured positions"}
    if movement["speed_m_s"] >= 1.35:
        return {"state": "moving", "evidence": "measured court-position speed >= 1.35 m/s",
                "speed_m_s": movement["speed_m_s"]}
    return {"state": "settled", "evidence": "measured court-position speed < 1.35 m/s",
            "speed_m_s": movement["speed_m_s"]}


def _serve_icon_name(icon: Any) -> str | None:
    text = str(icon or "").strip().upper()
    return {"W": "Wang Zhiyi", "H": "Han Yue"}.get(text)


def _service_score(icon_name: str | None, score: tuple[int, int]) -> int | None:
    if icon_name == "Han Yue":
        return score[0]
    if icon_name == "Wang Zhiyi":
        return score[1]
    return None


def _service_parity_ok(role: str | None, position: dict[str, Any], score: int | None) -> bool | None:
    if role not in {"near", "far"} or score is None or not position or position.get("xy_m") is None:
        return None
    x = float(position["xy_m"][0])
    near_right = score % 2 == 0
    expected_right = near_right if role == "near" else not near_right
    return x > COURT_WIDTH_M / 2 if expected_right else x < COURT_WIDTH_M / 2


def _group_swing_phases(pose_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for event in sorted(pose_events, key=lambda item: (item["time_s"], item["role"])):
        start = event.get("swing_start_s")
        end = event.get("swing_end_s")
        if start is None or end is None:
            groups.append({
                "events": [event],
                "role": event["role"],
                "shot_index": event.get("shot_index"),
                "start_s": None,
                "end_s": None,
            })
            continue
        if (
            groups
            and groups[-1]["role"] == event["role"]
            and groups[-1]["shot_index"] == event.get("shot_index")
            and groups[-1]["start_s"] is not None
            and float(start) - float(groups[-1]["end_s"]) <= MAX_SWING_S
        ):
            groups[-1]["events"].append(event)
            groups[-1]["end_s"] = max(float(groups[-1]["end_s"]), float(end))
        else:
            groups.append({
                "events": [event],
                "role": event["role"],
                "shot_index": event.get("shot_index"),
                "start_s": float(start),
                "end_s": float(end),
            })
    return groups


def _click_candidates(
    pose_events: list[dict[str, Any]],
    clicks: tuple[dict[str, float], ...],
    fps: float,
    time_origin_s: float = 0.0,
) -> tuple[list[dict[str, Any]], float]:
    groups = _group_swing_phases(pose_events)
    click_groups: dict[int, list[int]] = {}
    group_clicks: dict[int, list[int]] = {}
    frame_tolerance_s = 0.5 / fps + 0.01
    for click_index, click in enumerate(clicks):
        click_time = float(click["time_s"])
        eligible = [
            group_index
            for group_index, group in enumerate(groups)
            if group["start_s"] is not None
            and float(group["start_s"]) - frame_tolerance_s <= click_time
            <= float(group["end_s"]) + CONTACT_DELAY_S + frame_tolerance_s
        ]
        for group_index in eligible:
            click_groups.setdefault(group_index, []).append(click_index)
        group_clicks[click_index] = eligible

    offsets = []
    matched: dict[int, int] = {}
    for group_index, click_indices in click_groups.items():
        if len(click_indices) != 1:
            continue
        click_index = click_indices[0]
        if len(group_clicks[click_index]) != 1:
            continue
        matched[group_index] = click_index
        representative = max(
            groups[group_index]["events"],
            key=lambda event: float(event["data"].get("peak_speed", 0.0)),
        )
        offsets.append(float(clicks[click_index]["time_s"]) - float(representative["time_s"]))
    offset = float(np.median(offsets)) if offsets else 0.0
    candidates = []
    for group_index, group in enumerate(groups):
        events = group["events"]
        event = max(events, key=lambda item: float(item["data"].get("peak_speed", 0.0)))
        click_index = matched.get(group_index)
        click = clicks[click_index] if click_index is not None else None
        pose_time = float(event["time_s"])
        contact_time = float(click["time_s"]) if click is not None else None
        frame = int(event["frame"])
        if contact_time is not None:
            frame = max(0, round((contact_time - time_origin_s) * fps))
        window = (
            [float(group["start_s"]), float(group["end_s"]) + CONTACT_DELAY_S]
            if group["start_s"] is not None else None
        )
        candidates.append({
            "time_s": contact_time if contact_time is not None else pose_time,
            "pose_event_time_s": pose_time,
            "contact_time_s": contact_time,
            "contact_timing_status": (
                "audio_supported_pose_phase" if contact_time is not None else "pose_phase_unresolved"
            ),
            "swing_window_s": window,
            "frame": frame,
            "role": group["role"],
            "source": ["pose_swing"] + (["audio_click"] if click is not None else []),
            "pose_confidence": max(float(item["confidence"]) for item in events),
            "confidence": (
                min(0.68, 0.4 + 0.2 * max(float(item["confidence"]) for item in events))
                if click is not None
                else min(0.5, 0.25 + 0.25 * max(float(item["confidence"]) for item in events))
            ),
            "pose_event": {
                **event,
                "phase_events": events,
                "swing_window_s": window,
            },
            "click": click,
            "click_only": False,
        })
    candidates.sort(key=lambda hit: (hit["time_s"], hit["frame"]))
    deduped = []
    for hit in candidates:
        if (
            deduped
            and hit["time_s"] - deduped[-1]["time_s"] < 0.12
            and hit["role"] == deduped[-1]["role"]
        ):
            if (
                deduped[-1]["contact_time_s"] is None
                and hit["contact_time_s"] is not None
            ):
                deduped[-1] = hit
            continue
        deduped.append(hit)
    return deduped, offset


def _candidate_cut_frame(tracks: Tracks, original_cuts: list[int]) -> list[int]:
    found = detect_same_angle_cuts(tracks)
    return sorted(set(original_cuts) | set(found))


def _source_frame(global_time_s: float) -> int:
    return max(0, int(round(global_time_s * 60.0)))


def _rally_spans(group: dict[str, Any], segments: list[dict[str, float]]) -> list[tuple[int, float, float]]:
    spans = []
    for index, segment in enumerate(segments):
        start = max(group["start_s"], float(segment["start_s"]))
        end = min(group["end_s"], float(segment["end_s"]))
        if end - start >= 0.1:
            spans.append((index, start, end))
    return spans


def split_camera_intervals(
    main_segments: list[dict[str, float]],
    states: list[dict[str, Any]],
    max_chunk_s: float = 36.0,
) -> list[dict[str, float]]:
    boundaries = sorted({
        float(value)
        for state in states
        for value in (state["start_s"], state["end_s"])
    })
    chunks = []
    for segment in main_segments:
        cursor, end = float(segment["start_s"]), float(segment["end_s"])
        while end - cursor > max_chunk_s:
            target = cursor + max_chunk_s
            eligible = [b for b in boundaries if cursor + 18.0 <= b <= min(end, target)]
            boundary = min(eligible, key=lambda b: abs(b - target)) if eligible else target
            if boundary <= cursor or boundary >= end:
                boundary = target
            chunks.append({"start_s": round(cursor, 6), "end_s": round(boundary, 6)})
            cursor = boundary
        if end > cursor:
            chunks.append({"start_s": round(cursor, 6), "end_s": round(end, 6)})
    return chunks


def _run_command(command: list[str], *, cwd: Path, timeout: int | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout[-3000:]}\nstderr:\n{result.stderr[-5000:]}"
        )
    return result


def audio_stream_listed(probe_text: str) -> bool:
    return any(line.strip() == "audio" for line in probe_text.splitlines())


def _read_clicks(ffmpeg: Path, video: Path, start_s: float, duration_s: float) -> list[dict[str, float]]:
    ffprobe = ffmpeg.with_name("ffprobe.exe" if ffmpeg.suffix.lower() == ".exe" else "ffprobe")
    if ffprobe.is_file():
        probe = subprocess.run(
            [str(ffprobe), "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, timeout=60,
        )
        if probe.returncode == 0 and not audio_stream_listed(probe.stdout):
            return []
    command = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-ss", f"{start_s:.6f}", "-i", str(video),
        "-t", f"{duration_s:.6f}", "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, timeout=max(120, int(duration_s * 4)))
    if result.returncode:
        raise RuntimeError(f"Audio decode failed for {start_s:.3f}-{start_s + duration_s:.3f}s: {result.stderr[-2000:]!r}")
    return detect_audio_clicks(result.stdout)


def _run_pose_segments(
    video: Path,
    segments: list[dict[str, float]],
    output: Path,
    ffmpeg: Path,
    pose_python: Path,
    worker_count: int,
    detector_every: int,
    player_hands: dict[str, Any],
    hand_attribution: dict[str, Any],
) -> tuple[list[SegmentRun], list[dict[str, Any]]]:
    work = output / ".work"
    pose_dir = output / "pose_cache"
    work.mkdir(parents=True)
    pose_dir.mkdir(parents=True)
    pipeline = Path(__file__).resolve().parents[1] / "fast" / "pipeline.py"
    results: list[SegmentRun] = []
    summary: list[dict[str, Any]] = []
    try:
        for number, segment in enumerate(segments, start=1):
            start_s, end_s = float(segment["start_s"]), float(segment["end_s"])
            duration_s = end_s - start_s
            clip = work / f"segment-{number:02d}.mp4"
            pose_path = pose_dir / f"segment-{number:02d}.json"
            extract = [
                str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{start_s:.6f}", "-i", str(video), "-t", f"{duration_s:.6f}",
                "-vf", "fps=30", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "8",
                "-pix_fmt", "yuv420p", str(clip),
            ]
            _run_command(extract, cwd=Path(__file__).resolve().parents[1], timeout=max(120, int(duration_s * 5)))
            pipeline_command = [
                str(pose_python), "-B", str(pipeline), str(clip), str(pose_path),
                "--segments", str(worker_count), "--every", str(detector_every), "--reuse", "0", "--decoders", "2",
            ]
            started = time.perf_counter()
            completed = _run_command(pipeline_command, cwd=Path(__file__).resolve().parents[1],
                                     timeout=max(300, int(duration_s * 20)))
            elapsed = time.perf_counter() - started
            try:
                pipeline_summary = json.loads(completed.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                pipeline_summary = {"stdout_tail": completed.stdout[-2000:]}
            document = json.loads(pose_path.read_text(encoding="utf-8"))
            document["source"]["original_file"] = str(video.resolve())
            document["source"]["reel_start_s"] = start_s
            document["source"]["reel_end_s"] = end_s
            document["source"]["reel_fps"] = 60.0
            document["source"]["players"] = player_hands
            document["source"]["player_handedness_attribution"] = hand_attribution
            document["source"]["timeline_mapping"] = "reel_time_s = reel_start_s + timestamp_seconds"
            for frame in document.get("frames", []):
                reel_time = start_s + float(frame.get("timestamp_seconds", 0.0))
                frame["reel_time_seconds"] = round(reel_time, 4)
                frame["source_frame_60fps"] = _source_frame(reel_time)
            _write_json(pose_path, document)
            tracks = load_tracks(document)
            cuts = _candidate_cut_frame(tracks, [int(x) for x in document.get("cuts", [])])
            document["cuts"] = cuts
            _write_json(pose_path, document)
            clicks = _read_clicks(ffmpeg, video, start_s, duration_s)
            results.append(SegmentRun(number, start_s, end_s, pose_path, tuple(cuts), 0.0, tuple(clicks)))
            summary.append({
                "number": number,
                "reel_start_s": start_s,
                "reel_end_s": end_s,
                "pose_path": str(pose_path.resolve()),
                "pose_frames": len(document.get("frames", [])),
                "fps": document.get("source", {}).get("fps"),
                "court_fit": document.get("court_fit"),
                "court_lines_normalized": document.get("court_lines_normalized"),
                "models": document.get("model"),
                "cuts_doc_frames": cuts,
                "audio_click_count": len(clicks),
                "pipeline_elapsed_s": round(elapsed, 3),
                "pipeline": pipeline_summary,
            })
    finally:
        if work.exists():
            shutil.rmtree(work)
    _write_json(pose_dir / "index.json", {"segments": summary})
    return results, summary


def _reuse_pose_cache(
    cache: Path,
    video: Path,
    output: Path,
    chunks: list[dict[str, float]],
    ffmpeg: Path,
    player_hands: dict[str, Any],
    hand_attribution: dict[str, Any],
) -> tuple[list[SegmentRun], list[dict[str, Any]]]:
    cache_index_path = cache / "index.json"
    if not cache_index_path.is_file():
        raise FileNotFoundError(cache_index_path)
    cache_index = json.loads(cache_index_path.read_text(encoding="utf-8"))
    cached_segments = cache_index.get("segments", [])
    if len(cached_segments) != len(chunks):
        raise ValueError(
            f"Pose cache has {len(cached_segments)} chunks; expected {len(chunks)}"
        )
    pose_dir = output / "pose_cache"
    pose_dir.mkdir(parents=True, exist_ok=True)
    results: list[SegmentRun] = []
    summary: list[dict[str, Any]] = []
    for number, (chunk, cached) in enumerate(zip(chunks, cached_segments), start=1):
        start_s, end_s = float(chunk["start_s"]), float(chunk["end_s"])
        if (
            abs(float(cached["reel_start_s"]) - start_s) > 1e-4
            or abs(float(cached["reel_end_s"]) - end_s) > 1e-4
        ):
            raise ValueError(
                f"Pose cache chunk {number} does not match requested interval "
                f"{start_s:.3f}-{end_s:.3f}s"
            )
        source_path = Path(cached["pose_path"])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        destination = pose_dir / f"segment-{number:02d}.json"
        shutil.copy2(source_path, destination)
        document = json.loads(destination.read_text(encoding="utf-8"))
        original_file = Path(str(document.get("source", {}).get("original_file", ""))).name
        if original_file and original_file.casefold() != video.name.casefold():
            raise ValueError(
                f"Pose cache chunk {number} belongs to {original_file!r}, not {video.name!r}"
            )
        document["source"]["players"] = player_hands
        document["source"]["player_handedness_attribution"] = hand_attribution
        _write_json(destination, document)
        clicks = tuple(_read_clicks(ffmpeg, video, start_s, end_s - start_s))
        cuts = tuple(int(value) for value in document.get("cuts", []))
        results.append(SegmentRun(number, start_s, end_s, destination, cuts, 0.0, clicks))
        summary.append({
            **cached,
            "number": number,
            "pose_path": str(destination.resolve()),
            "pose_frames": len(document.get("frames", [])),
            "audio_click_count": len(clicks),
            "pipeline_elapsed_s": 0.0,
            "cache_source_path": str(source_path.resolve()),
            "cache_source_sha256": _sha256(source_path),
            "pipeline": {"reused_pose_cache": True, "inference_run": False},
        })
    _write_json(pose_dir / "index.json", {
        "segments": summary,
        "reused_from": str(cache.resolve()),
        "source_index_sha256": _sha256(cache_index_path),
    })
    return results, summary


def _segment_data_from_document(
    run: SegmentRun,
    document: dict[str, Any],
    source_frames: list[dict[str, int]] | None = None,
) -> tuple[dict[str, Any], Tracks, list[dict[str, Any]], list[dict[str, Any]], float]:
    tracks = load_tracks(document)
    pose_events_all = detect_events(tracks)
    swing_events = []
    for event in pose_events_all:
        if str(event.kind) not in POSE_EVENT_KINDS:
            continue
        start = max(0, min(tracks.n - 1, int(event.data["start"])))
        end = max(0, min(tracks.n - 1, int(event.data["end"])))
        shot_index = next(
            (index for index, (shot_start, shot_end) in enumerate(tracks.shots)
             if shot_start <= int(event.frame) <= shot_end),
            None,
        )
        swing_events.append({
            "time_s": float(tracks.t[event.frame]),
            "frame": int(event.frame),
            "role": _role_value(event.player),
            "confidence": float(event.confidence),
            "kind": str(event.kind),
            "data": _jsonable(dict(event.data)),
            "swing_start_s": float(tracks.t[start]),
            "swing_end_s": float(tracks.t[end]),
            "shot_index": shot_index,
        })
    lunge_events = [
        {"frame": int(e.frame), "role": _role_value(e.player), "confidence": float(e.confidence),
         "data": _jsonable(dict(e.data))}
        for e in pose_events_all if "lunge" in str(e.kind)
    ]
    candidates, offset = _click_candidates(
        swing_events, run.clicks, tracks.fps, time_origin_s=float(tracks.t[0])
    )
    for candidate in candidates:
        candidate["reel_time_s"] = run.start_s + candidate["time_s"]
        candidate["pose_event_reel_time_s"] = run.start_s + candidate["pose_event_time_s"]
        candidate["contact_time_reel_s"] = (
            run.start_s + candidate["contact_time_s"]
            if candidate["contact_time_s"] is not None else None
        )
        if source_frames is not None:
            origin = source_frames[max(0, min(len(source_frames) - 1, int(candidate["frame"])))]
            candidate["source_segment_number"] = origin["segment_number"]
            candidate["source_frame_30fps"] = origin["frame"]
            pose_origin = source_frames[
                max(0, min(len(source_frames) - 1, int(candidate["pose_event"]["frame"])))
            ]
            candidate["pose_event_source_segment_number"] = pose_origin["segment_number"]
            candidate["pose_event_source_frame_30fps"] = pose_origin["frame"]
        candidate["evidence_status"] = "provisional_pose_swing"
    return document, tracks, candidates, lunge_events, offset


def _segment_data(run: SegmentRun) -> tuple[dict[str, Any], Tracks, list[dict[str, Any]], list[dict[str, Any]], float]:
    return _segment_data_from_document(run, json.loads(run.pose_path.read_text(encoding="utf-8")))


def _join_camera_segments(
    main_segments: list[dict[str, float]],
    segment_runs: list[SegmentRun],
    segment_data: list[tuple[dict[str, Any], Tracks, list[dict[str, Any]], list[dict[str, Any]], float]],
) -> tuple[list[SegmentRun], list[tuple[dict[str, Any], Tracks, list[dict[str, Any]], list[dict[str, Any]], float]]]:
    joined_runs = []
    joined_data = []
    for camera_index, camera in enumerate(main_segments, start=1):
        start_s, end_s = float(camera["start_s"]), float(camera["end_s"])
        chunks = [
            (run, data)
            for run, data in zip(segment_runs, segment_data)
            if run.start_s >= start_s - 1e-4 and run.end_s <= end_s + 1e-4
        ]
        chunks.sort(key=lambda item: item[0].start_s)
        if not chunks or abs(chunks[0][0].start_s - start_s) > 1e-4 or abs(chunks[-1][0].end_s - end_s) > 1e-4:
            raise ValueError(f"Pose-cache chunks do not cover approved camera interval {start_s:.3f}-{end_s:.3f}s")
        for (previous, _), (current, _) in zip(chunks, chunks[1:]):
            if abs(previous.end_s - current.start_s) > 1e-4:
                raise ValueError(f"Gap between storage chunks at {previous.end_s:.3f}-{current.start_s:.3f}s")

        first_document = chunks[0][1][0]
        document = dict(first_document)
        source = dict(first_document["source"])
        frames = []
        source_frames = []
        cuts = []
        clicks = []
        for run, data in chunks:
            chunk_document, _, _, _, _ = data
            frame_offset = len(frames)
            for frame_position, raw_frame in enumerate(chunk_document["frames"]):
                frame = dict(raw_frame)
                local_frame = int(frame.get("frame_index", frame_position))
                local_time = float(frame.get("timestamp_seconds", local_frame / float(source["fps"])))
                frame["frame_index"] = len(frames)
                frame["timestamp_seconds"] = run.start_s - start_s + local_time
                frame["reel_time_seconds"] = run.start_s + local_time
                frame["source_frame_60fps"] = _source_frame(frame["reel_time_seconds"])
                frames.append(frame)
                source_frames.append({"segment_number": run.number, "frame": local_frame})
            cuts.extend(frame_offset + int(cut) for cut in chunk_document.get("cuts", []) if int(cut) > 0)
            offset_s = run.start_s - start_s
            clicks.extend({**click, "time_s": float(click["time_s"]) + offset_s} for click in run.clicks)

        source.update({
            "frame_count": len(frames),
            "reel_start_s": start_s,
            "reel_end_s": end_s,
            "original_file": str(source.get("original_file", "")),
        })
        document["source"] = source
        document["frames"] = frames
        document["cuts"] = sorted(set(cuts))
        document.pop("windows_refined", None)
        joined_run = SegmentRun(
            camera_index,
            start_s,
            end_s,
            chunks[0][0].pose_path,
            tuple(document["cuts"]),
            0.0,
            tuple(clicks),
            tuple(run.number for run, _ in chunks),
        )
        preliminary_tracks = load_tracks(document)
        document["cuts"] = _candidate_cut_frame(preliminary_tracks, document["cuts"])
        joined_run = SegmentRun(
            joined_run.number,
            joined_run.start_s,
            joined_run.end_s,
            joined_run.pose_path,
            tuple(document["cuts"]),
            joined_run.click_offset_s,
            joined_run.clicks,
            joined_run.source_chunks,
        )
        joined_runs.append(joined_run)
        joined_data.append(_segment_data_from_document(joined_run, document, source_frames))
    return joined_runs, joined_data


def _select_span(
    group: dict[str, Any],
    possible: list[tuple[int, float, float]],
    segment_runs: list[SegmentRun],
    segment_data: list[tuple[dict[str, Any], Tracks, list[dict[str, Any]], list[dict[str, Any]], float]],
) -> tuple[int, float, float, list[dict[str, Any]]]:
    ranked = []
    for segment_index, start, end in possible:
        run = segment_runs[segment_index]
        _, tracks, candidates, _, _ = segment_data[segment_index]
        in_span = [c for c in candidates if start <= run.start_s + c["time_s"] <= end]
        local_start = max(0, int(round((start - run.start_s) * tracks.fps)))
        local_end = min(tracks.n - 1, int(round((end - run.start_s) * tracks.fps)))
        cut_frames = [c for c in run.cuts if local_start < c < local_end]
        boundaries = [local_start] + cut_frames + [local_end + 1]
        for left, right in zip(boundaries, boundaries[1:]):
            if right - left < max(3, round(0.1 * tracks.fps)):
                continue
            piece_start = run.start_s + left / tracks.fps
            piece_end = min(end, run.start_s + (right - 1) / tracks.fps)
            piece_hits = [c for c in in_span if piece_start <= run.start_s + c["time_s"] <= piece_end]
            motion_evidence = sum(
                bool(tracks.players[role].measured[max(0, left):min(tracks.n, right)].sum())
                for role in ("near", "far")
            )
            rank = (len(piece_hits), motion_evidence, piece_end)
            ranked.append((rank, segment_index, piece_start, piece_end, piece_hits))
    if not ranked:
        raise ValueError(f"No main-camera span for score state group {group['state_indices']}")
    with_hits = [row for row in ranked if row[0][0] > 0]
    candidates = with_hits or ranked
    _, segment_index, start, end, hits = max(candidates, key=lambda x: (x[2], x[0][0], x[0][1]))
    return segment_index, start, end, hits


def _annotate_candidates(
    candidates: list[dict[str, Any]],
    tracks: Tracks,
    group: dict[str, Any],
    span_start: float,
    segment_start: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = [dict(c) for c in candidates if span_start <= segment_start + c["time_s"] <= group["end_s"]]
    selected.sort(key=lambda c: c["reel_time_s"])
    serve_name = _serve_icon_name(group.get("serve_icon"))
    server_score = _service_score(serve_name, group["score"])
    last_role = None
    for index, candidate in enumerate(selected):
        confirmed = candidate.get("contact_time_reel_s") is not None
        if (
            confirmed
            and candidate["role"] in {"near", "far"}
            and last_role == candidate["role"]
            and index > 0
        ):
            candidate["alternation_warning"] = "pose swings do not alternate with the preceding candidate"
            candidate["confidence"] = min(candidate["confidence"], 0.25)
        if confirmed and candidate["role"] in {"near", "far"}:
            last_role = candidate["role"]
    first = selected[0] if selected else None
    first_confirmed = bool(first and first.get("contact_time_reel_s") is not None)
    mapping = {"near": None, "far": None}
    parity_check = None
    if first_confirmed and first and first["role"] in {"near", "far"} and first["reel_time_s"] - span_start <= 1.2:
        mapping[first["role"]] = serve_name
        mapping["far" if first["role"] == "near" else "near"] = (
            "Wang Zhiyi" if serve_name == "Han Yue" else "Han Yue" if serve_name == "Wang Zhiyi" else None
        )
        position = _nearest_measured(tracks, first["role"], int(first["frame"]))
        parity_check = _service_parity_ok(first["role"], position, server_score)
        if parity_check is False:
            mapping = {"near": None, "far": None}
    for index, candidate in enumerate(selected):
        candidate["hit_index"] = index + 1
        candidate["is_serve"] = index == 0 and candidate.get("contact_time_reel_s") is not None
        candidate["hitter_name"] = (
            mapping.get(candidate["role"])
            if candidate.get("contact_time_reel_s") is not None else None
        )
        candidate["serve_name_from_scorebox"] = serve_name if index == 0 else None
        candidate["serve_parity_check"] = parity_check if index == 0 else None
    return selected, {
        "serve_icon": group.get("serve_icon"),
        "serve_name": serve_name,
        "serve_score": server_score,
        "first_hit_role": first["role"] if first else None,
        "service_parity_check": parity_check,
        "role_to_player": mapping,
        "player_mapping_confidence": "per_rally_pose_and_scorebox" if any(mapping.values()) else "unknown",
    }


def _build_hit_rows(
    group: dict[str, Any],
    rally_id: str,
    segment_index: int,
    segment_run: SegmentRun,
    document: dict[str, Any],
    tracks: Tracks,
    selected: list[dict[str, Any]],
    serve_info: dict[str, Any],
    click_offset_s: float,
    lunge_events: list[dict[str, Any]],
    base_positions: dict[tuple[int, str], list[list[float]]],
) -> list[dict[str, Any]]:
    rows = []
    for index, candidate in enumerate(selected):
        role = candidate["role"]
        contact_confirmed = candidate.get("contact_time_reel_s") is not None
        receiver = (
            "far" if role == "near" else "near" if role == "far" else None
        ) if contact_confirmed else None
        frame = max(0, min(tracks.n - 1, int(candidate["frame"])))
        next_candidate = selected[index + 1] if index + 1 < len(selected) else None
        next_role = next_candidate["role"] if next_candidate else None
        receiver_next_supported = bool(
            contact_confirmed
            and next_candidate
            and next_candidate.get("contact_time_reel_s") is not None
            and is_receiver_next_contact(role, next_role)
        )
        next_frame = int(next_candidate["frame"]) if receiver_next_supported else None
        time_to_next_candidate = (
            max(0.0, float(next_candidate["reel_time_s"] - candidate["reel_time_s"]))
            if next_candidate else None
        )
        time_to_next = time_to_next_candidate if receiver_next_supported else None
        pose_event_positions = {r: _nearest_measured(tracks, r, frame) for r in ("near", "far")}
        pose_event_movement = {r: _movement_before(tracks, r, frame) for r in ("near", "far")}
        current_positions = pose_event_positions if contact_confirmed else {
            r: {"xy_m": None, "frame": None, "offset_frames": None, "measured": False,
                "court_ok": tracks.court is not None}
            for r in ("near", "far")
        }
        movement = pose_event_movement if contact_confirmed else {
            r: {"vector_m": None, "speed_m_s": None, "direction": None, "samples": 0}
            for r in ("near", "far")
        }
        contact_height = (
            _height_at_contact(tracks, role, frame)
            if role and contact_confirmed
            else {"label": None, "evidence": "pose phase does not identify a contact frame"}
        )
        receiver_next = _nearest_measured(tracks, receiver, next_frame) if receiver and next_frame is not None else None
        receiver_current = current_positions.get(receiver) if receiver else None
        next_vector = None
        pull_m = None
        zone = {"label": "unknown", "depth": None, "side": None, "method": "inferred_from_next_contact"}
        if receiver_current and receiver_next and receiver_current.get("xy_m") and receiver_next.get("xy_m"):
            a = np.asarray(receiver_current["xy_m"], dtype=float)
            b = np.asarray(receiver_next["xy_m"], dtype=float)
            next_vector = (b - a).tolist()
            pull_m = round(float(np.linalg.norm(b - a)), 3)
            zone = infer_landing_zone(float(b[0]), float(b[1]), receiver)
        base_detail = {}
        hitter_depth = None
        for player_role in ("near", "far"):
            base_samples = base_positions.get((group["game"], player_role), [])
            if len(base_samples) >= 2:
                base = np.median(np.asarray(base_samples, dtype=float), axis=0)
                base_source = "median measured position at opponent contacts, same game"
            else:
                base = np.array([3.05, 9.7 if player_role == "near" else 3.7], dtype=float)
                base_source = "fixed fallback at court centre, 3 m from net"
            xy = current_positions[player_role].get("xy_m")
            base_detail[player_role] = {
                "position_m": [round(float(base[0]), 3), round(float(base[1]), 3)],
                "source": base_source,
                "distance_m": (
                    round(float(np.linalg.norm(np.asarray(xy) - base)), 3)
                    if contact_confirmed and xy else None
                ),
                "pose_event_distance_m": (
                    round(float(np.linalg.norm(np.asarray(pose_event_positions[player_role]["xy_m"]) - base)), 3)
                    if pose_event_positions[player_role].get("xy_m") else None
                ),
            }
            if player_role == role and contact_confirmed:
                hitter_depth = _depth_from_net(current_positions[player_role], player_role)
        against_info = infer_against(movement[receiver]["vector_m"] if receiver else None, next_vector)
        family = infer_shot_family(
            contact_height=contact_height["label"],
            hitter_depth=hitter_depth,
            receiver_zone=zone,
            time_to_next_s=time_to_next,
            is_serve=bool(candidate["is_serve"]),
            serve_supported=bool(candidate["is_serve"] and serve_info["service_parity_check"] is True),
        )
        rows.append({
            "hit_id": f"{rally_id}-H{index + 1:02d}",
            "rally_id": rally_id,
            "time_s": round(float(candidate["reel_time_s"]), 4),
            "time_basis": (
                "audio_click_within_pose_swing_phase"
                if contact_confirmed else "pose_swing_estimate_not_contact_time"
            ),
            "contact_time_s": (
                round(float(candidate["contact_time_reel_s"]), 4) if contact_confirmed else None
            ),
            "contact_timing_status": candidate["contact_timing_status"],
            "pose_event_time_s": round(float(candidate["pose_event_reel_time_s"]), 4),
            "pose_swing_window_s": (
                [round(segment_run.start_s + float(x), 4) for x in candidate["swing_window_s"]]
                if candidate.get("swing_window_s") else None
            ),
            "source_frame_60fps": _source_frame(candidate["reel_time_s"]),
            "contact_source_frame_60fps": (
                _source_frame(candidate["contact_time_reel_s"]) if contact_confirmed else None
            ),
            "pose_event_source_frame_60fps": _source_frame(candidate["pose_event_reel_time_s"]),
            "pose_frame_30fps": candidate.get("source_frame_30fps", frame),
            "pose_event_frame_30fps": candidate.get("pose_event_source_frame_30fps", frame),
            "analysis_frame_30fps": frame,
            "segment_number": candidate.get("source_segment_number", segment_run.number),
            "hitter": role if contact_confirmed else None,
            "pose_role_candidate": role,
            "hitter_name": candidate.get("hitter_name"),
            "hitter_name_evidence": (
                "scorebox server plus audio-supported pose phase"
                if candidate.get("hitter_name") else "unknown"
            ),
            "hitter_assignment_status": (
                "audio_supported_pose_phase" if contact_confirmed else "pose_role_only_not_confirmed_hitter"
            ),
            "receiver": receiver,
            "evidence_status": "provisional_pose_swing",
            "confidence": round(float(candidate["confidence"]), 3),
            "confidence_basis": (
                "calibrated candidate score; pose event confidence is not contact certainty"
            ),
            "sources": candidate["source"],
            "contact_rule": (
                "unique audio click within the measured pose-swing phase; contact remains unverified"
                if contact_confirmed
                else "pose-swing observation; hitter and contact time unresolved"
            ),
            "contact_evidence": {
                "pose_event": candidate.get("pose_event"),
                "audio_click": candidate.get("click"),
                "audio_minus_pose_median_offset_s": round(click_offset_s, 4),
                "alternation_warning": candidate.get("alternation_warning"),
            },
            "contact_frame_measured": (
                bool(tracks.players[role].measured[frame]) if role and contact_confirmed else False
            ),
            "court_ok": tracks.court is not None,
            "is_serve": bool(candidate["is_serve"]),
            "serve_evidence": serve_info if candidate["is_serve"] else None,
            "positions_at_contact_m": current_positions if contact_confirmed else None,
            "positions_at_pose_event_m": pose_event_positions,
            "opponent_movement_before_contact": movement if contact_confirmed else None,
            "opponent_movement_before_pose_event": pose_event_movement,
            "distance_from_base": base_detail,
            "balance": (
                {r: _balance_state(tracks, r, frame, lunge_events) for r in ("near", "far")}
                if contact_confirmed else None
            ),
            "balance_at_pose_event": {
                r: _balance_state(tracks, r, frame, lunge_events) for r in ("near", "far")
            },
            "contact_height": contact_height,
            "time_to_next_contact_s": round(time_to_next, 4) if time_to_next is not None else None,
            "time_to_next_candidate_s": round(time_to_next_candidate, 4) if time_to_next_candidate is not None else None,
            "next_candidate_role": next_role,
            "next_contact_alternation_supported": receiver_next_supported,
            "receiver_position_at_next_contact_m": receiver_next,
            "landing_zone": zone,
            "landing_zone_basis": "receiver's next measured contact; no shuttle-flight tracking",
            "shot_family": family,
            "pull_m": pull_m,
            "against": against_info["against"],
            "against_direction_dot": against_info["direction_dot"],
            "limits": ["single-camera 2D estimate", "contact is a pose/audio proxy, not confirmed shuttle contact"],
        })
    return rows


def _make_rallies(
    score_groups: list[dict[str, Any]],
    main_segments: list[dict[str, float]],
    segment_runs: list[SegmentRun],
    segment_data: list[tuple[dict[str, Any], Tracks, list[dict[str, Any]], list[dict[str, Any]], float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rallies = []
    hits = []
    excluded_groups = []
    base_positions: dict[tuple[int, str], list[list[float]]] = {}
    plans = []
    for group in score_groups:
        possible = _rally_spans(group, main_segments)
        if not possible:
            excluded_groups.append({
                "score_state_indices": group["state_indices"],
                "game": group["game"],
                "score_han_wang": group["score_han_wang"],
                "start_s": group["start_s"],
                "end_s": group["end_s"],
                "reason": "no approved main-camera footage overlaps this score interval",
            })
            continue
        try:
            segment_index, span_start, span_end, candidates = _select_span(
                group, possible, segment_runs, segment_data
            )
        except ValueError:
            excluded_groups.append({
                "score_state_indices": group["state_indices"],
                "game": group["game"],
                "score_han_wang": group["score_han_wang"],
                "start_s": group["start_s"],
                "end_s": group["end_s"],
                "reason": "main-camera overlap contains no continuous pose interval of at least three frames",
            })
            continue
        run = segment_runs[segment_index]
        document, tracks, _, lunge_events, click_offset = segment_data[segment_index]
        group_with_span = dict(group)
        group_with_span["selected_start_s"] = span_start
        candidates_in_span = [
            c for c in candidates
            if span_start <= run.start_s + c["time_s"] <= span_end
        ]
        selected, serve_info = _annotate_candidates(
            candidates_in_span, tracks, group_with_span, span_start, run.start_s
        )
        rally_id = f"G{group['game']}-R{len(plans) + 1:03d}"
        for candidate_index, candidate in enumerate(selected):
            role = candidate["role"]
            previous_role = selected[candidate_index - 1]["role"] if candidate_index > 0 else None
            if role in {"near", "far"} and previous_role in {"near", "far"}:
                opponent = "far" if role == "near" else "near"
                if previous_role != opponent:
                    continue
                position = _nearest_measured(tracks, opponent, int(candidate["frame"]))
                if position["xy_m"] is not None:
                    base_positions.setdefault((group["game"], opponent), []).append(position["xy_m"])
        plans.append({
            "group": group_with_span,
            "rally_id": rally_id,
            "segment_index": segment_index,
            "run": run,
            "document": document,
            "tracks": tracks,
            "selected": selected,
            "serve_info": serve_info,
            "lunge_events": lunge_events,
            "click_offset": click_offset,
            "span_start": span_start,
            "span_end": span_end,
            "source_pose_segments": sorted({
                int(candidate.get("source_segment_number", run.number))
                for candidate in selected
            }) or list(run.source_chunks),
        })
    for plan in plans:
        group = plan["group"]
        rally_id = plan["rally_id"]
        run = plan["run"]
        tracks = plan["tracks"]
        click_offset = plan["click_offset"]
        hit_rows = _build_hit_rows(
            group, rally_id, plan["segment_index"], run, plan["document"], tracks,
            plan["selected"], plan["serve_info"], click_offset, plan["lunge_events"], base_positions,
        )
        if hit_rows:
            hit_rows[-1]["landing_zone"] = {
                "label": "unknown", "depth": None, "side": None,
                "method": "inferred_from_next_contact", "reason": "last hit in the visible rally",
            }
            hit_rows[-1]["landing_zone_basis"] = "unknown for the last visible contact; shuttle flight is not tracked"
        hits.extend(hit_rows)
        rallies.append({
            "rally_id": rally_id,
            "game": group["game"],
            "score_before_han_wang": group["score_han_wang"],
            "scorebox_state_indices": group["state_indices"],
            "serve_icon": group.get("serve_icon"),
            "player_mapping": plan["serve_info"],
            "start_s": round(float(hit_rows[0]["time_s"] if hit_rows else plan["span_start"]), 4),
            "end_s": round(float(hit_rows[-1]["time_s"] if hit_rows else plan["span_end"]), 4),
            "selected_live_span_s": [round(plan["span_start"], 4), round(plan["span_end"], 4)],
            "start_basis": "first retained pose/audio contact; otherwise selected live-span start",
            "end_basis": "last retained pose/audio contact; otherwise selected live-span end",
            "hit_count": len(hit_rows),
            "winner_inferred": group["winner"],
            "winner_reason": group["winner_reason"],
            "source_segment_number": (
                plan["source_pose_segments"][0] if plan["source_pose_segments"] else run.number
            ),
            "source_pose_segments": plan["source_pose_segments"],
            "pose_fps": tracks.fps,
            "court_available": tracks.court is not None,
            "audio_video_offset_median_s": round(click_offset, 4),
            "limits": ["highlight reel may omit rallies", "scoreboard intervals are coarse", "no shuttle tracking"],
        })
    return rallies, hits, excluded_groups


def _file_manifest(root: Path, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = exclude or set()
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = str(path.relative_to(root))
        if rel in excluded:
            continue
        files.append({"path": str(path.resolve()), "relative_path": rel, "size_bytes": path.stat().st_size,
                      "sha256": _sha256(path)})
    return files


def run(
    video: Path,
    output: Path,
    segments_path: Path,
    states_path: Path,
    ffmpeg: Path,
    pose_python: Path | None,
    workers: int = 8,
    detector_every: int = 4,
    pose_cache: Path | None = None,
    player_attribution: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    video, output = video.resolve(), output.resolve()
    segments_path, states_path = segments_path.resolve(), states_path.resolve()
    ffmpeg = ffmpeg.resolve()
    pose_python = pose_python.resolve() if pose_python else None
    pose_cache = pose_cache.resolve() if pose_cache else None
    player_attribution = player_attribution.resolve() if player_attribution else None
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    for required in (video, segments_path, states_path, ffmpeg):
        if not required.is_file():
            raise FileNotFoundError(required)
    if pose_python is not None and not pose_python.is_file():
        raise FileNotFoundError(pose_python)
    if pose_cache is None and pose_python is None:
        raise ValueError("--pose-python is required when no --pose-cache is supplied")
    if player_attribution is not None and not player_attribution.is_file():
        raise FileNotFoundError(player_attribution)
    if not (workers > 0 and detector_every > 0):
        raise ValueError("workers and detector_every must be positive")
    design = json.loads(segments_path.read_text(encoding="utf-8"))
    state_doc = json.loads(states_path.read_text(encoding="utf-8"))
    main_segments = design["main_segments"]
    states = state_doc["states"]
    processing_chunks = split_camera_intervals(main_segments, states)
    score_groups = group_score_states(states)
    player_hands, hand_attribution = _read_player_attribution(video, player_attribution)
    if pose_cache is not None:
        if not pose_cache.is_dir():
            raise FileNotFoundError(pose_cache)
        segment_runs, segment_summaries = _reuse_pose_cache(
            pose_cache, video, output, processing_chunks, ffmpeg, player_hands, hand_attribution
        )
    else:
        segment_runs, segment_summaries = _run_pose_segments(
            video, processing_chunks, output, ffmpeg, pose_python, workers, detector_every,
            player_hands, hand_attribution,
        )
    chunk_data = [_segment_data(run_item) for run_item in segment_runs]
    rally_runs, rally_data = _join_camera_segments(main_segments, segment_runs, chunk_data)
    rallies, hits, excluded_groups = _make_rallies(score_groups, main_segments, rally_runs, rally_data)
    _write_json(output / "hits.json", {
        "schema_version": 1,
        "source_video": str(video),
        "hit_count": len(hits),
        "rallies_with_hits": sum(r["hit_count"] > 0 for r in rallies),
        "hits": hits,
    })
    _write_json(output / "rallies.json", {
        "schema_version": 1,
        "source_video": str(video),
        "rally_count": len(rallies),
        "score_state_group_count": len(score_groups),
        "excluded_score_groups": excluded_groups,
        "rallies": rallies,
    })
    project_root = Path(__file__).resolve().parents[1]
    code_files = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parent / "tests" / "test_tactics.py",
        project_root / "coach" / "README.md",
    ]
    cache_bundle = Path(__file__).resolve().parents[1] / "fast" / "cache" / "prepared-models.zip"
    model_assets = [
        project_root / "fast" / "weights" / "rtmpose-s-ft1.onnx",
        project_root / "fast" / "weights" / "rtmpose-t-ft3.onnx",
        project_root / "fast" / "weights" / "yolo26s-448x768.onnx",
        project_root / "fast" / "weights" / "yolo26n-384x640.onnx",
    ]
    rule_tables = {
        "landing_zone": {
            "columns": "thirds of singles width x=[0.46,5.64] m, screen-left to screen-right",
            "depth_from_net_m": {"front": "[0,1.98)", "mid": "[1.98,4.467)", "rear": "[4.467,6.70]"},
            "evidence": "receiver's next measured foot position; never tracked shuttle flight",
        },
        "shot_family": [
            {"when": "first contact and serve role plus service-parity evidence agree", "family": "serve"},
            {"when": "overhead; hitter rear; receiver next zone rear; next-contact gap >= 0.70 s", "family": "clear"},
            {"when": "overhead; next-contact gap <= 0.48 s; receiver next zone mid/rear", "family": "smash"},
            {"when": "overhead; receiver next zone front; next-contact gap >= 0.35 s", "family": "drop"},
            {"when": "underarm; hitter front; receiver next zone front", "family": "net"},
            {"when": "underarm; receiver next zone rear", "family": "lift"},
            {"when": "side; next-contact gap <= 0.42 s; receiver next zone mid/rear", "family": "drive/push"},
            {"when": "evidence is incomplete or does not meet a rule", "family": "unknown"},
            {"confidence_limit": "heuristic family confidence <= 0.40; not a verified shot label"},
        ],
    }
    run_doc = {
        "source_video": str(video),
        "source_sha256": _sha256(video),
        "source_bytes": video.stat().st_size,
        "source_clock_fps": 60,
        "processing_fps": FPS,
        "main_camera_segments": main_segments,
        "processing_chunks": processing_chunks,
        "main_camera_seconds": round(sum(float(s["end_s"]) - float(s["start_s"]) for s in main_segments), 3),
        "score_state_count": len(states),
        "merged_score_groups": len(score_groups),
        "score_groups_without_main_camera": excluded_groups,
        "rally_count": len(rallies),
        "hit_count": len(hits),
        "segments": segment_summaries,
        "audio_click_rule": "20 ms first-difference RMS windows, local peaks > 4x median, 100 ms refractory",
        "pose_hit_rule": (
            "pose swing phase observation; only an unambiguous click inside its measured phase "
            "provides a contact-time estimate"
        ),
        "pose_cache_reused": pose_cache is not None,
        "pose_cache_source": str(pose_cache) if pose_cache else None,
        "pose_cache_source_index_sha256": (
            _sha256(pose_cache / "index.json")
            if pose_cache is not None and (pose_cache / "index.json").is_file() else None
        ),
        "player_handedness_attribution": hand_attribution,
        "click_pose_alignment": [round(d[4], 4) for d in chunk_data],
        "rally_rule": "merge consecutive same-score states, intersect approved main-camera spans, split at detected cuts and retain latest live stretch with evidence",
        "pose_pipeline": str((Path(__file__).resolve().parents[1] / "fast" / "pipeline.py")),
        "pose_interpreter": str(pose_python) if pose_python else None,
        "pose_command_settings": (
            {"workers": workers, "detector_every": detector_every, "reuse": 0, "decoders": 2}
            if pose_cache is None else None
        ),
        "prepared_model_bundle": {
            "path": str(cache_bundle.resolve()),
            "sha256": _sha256(cache_bundle) if cache_bundle.is_file() else None,
        },
        "model_assets": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in model_assets if path.is_file()
        ],
        "inference_rules": rule_tables,
        "changed_source_files": [
            {"path": str(p.resolve()), "sha256": _sha256(p)} for p in code_files if p.is_file()
        ],
        "input_sha256": {
            str(segments_path): _sha256(segments_path),
            str(states_path): _sha256(states_path),
            **(
                {str(player_attribution): _sha256(player_attribution)}
                if player_attribution is not None else {}
            ),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    _write_json(output / "run.json", run_doc)
    test_record = {
        "status": "pending",
        "note": "Run coach/tests/test_tactics.py and applicable existing tests after this export; replace this record with measured results.",
    }
    _write_json(output / "tests.json", test_record)
    handoff = {
        "output_directory": str(output),
        "outputs": _file_manifest(output, {"handoff.json"}),
        "source_video": {"path": str(video), "sha256": _sha256(video)},
        "changed_source_files": run_doc["changed_source_files"],
        "counts": {"rallies": len(rallies), "hits": len(hits),
                   "hits_per_rally": {r["rally_id"]: r["hit_count"] for r in rallies},
                   "score_groups_without_main_camera": len(excluded_groups)},
        "elapsed_seconds": run_doc["elapsed_seconds"],
        "inference_rules": rule_tables,
        "pose_engine": {
            "name": "reused saved pose cache" if pose_cache else "fast OpenVINO RTMPose + YOLO",
            "interpreter": str(pose_python) if pose_python else None,
            "cache_source": str(pose_cache) if pose_cache else None,
            "bundle_path": run_doc["prepared_model_bundle"]["path"],
            "bundle_sha256": run_doc["prepared_model_bundle"]["sha256"],
            "inference_run": pose_cache is None,
        },
        "attribution": {
            "credit": "Badminton Sphere, Wikimedia Commons, CC BY 3.0, China Open 2025 women's singles final",
            "licence_note": "CC BY 3.0 source attribution does not clear upstream broadcast rights.",
        },
        "limitations": [
            "The source is an edited highlights reel, not a complete match.",
            "No shuttle-flight or racket-contact tracking; pose-only swing timestamps are not contact times.",
            "A contact-time estimate is emitted only when one audio click falls inside one measured pose-swing phase.",
            "Court coordinates, distances, direction and landing zones are single-camera 2D estimates.",
            "Landing zone is inferred from the receiver's next measured contact, never a tracked shuttle landing.",
            "Scoreboard intervals can skip points or include edits; winner is null unless one point changed.",
            "Handedness is unknown unless explicit source attribution is supplied.",
            "No human coaching review or label-based evaluation was performed.",
        ],
        "hash_scope_note": "handoff.json does not hash itself; its hash can be computed externally without a self-referential manifest.",
    }
    _write_json(output / "handoff.json", handoff)
    return run_doc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a cautious per-hit tactics table from pose and score evidence.")
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--segments", required=True, type=Path, help="Approved main-camera interval JSON")
    parser.add_argument("--states", required=True, type=Path, help="Score-state JSON")
    parser.add_argument("--ffmpeg", required=True, type=Path)
    parser.add_argument("--pose-python", type=Path, help="Interpreter with OpenVINO for fast/pipeline.py")
    parser.add_argument("--pose-cache", type=Path, help="Reuse a saved pose_cache directory without inference")
    parser.add_argument("--player-attribution", type=Path, help="JSON with explicit near/far handedness evidence")
    parser.add_argument("--workers", type=int, default=8, help="Pipeline model streams; use 8 for the prepared model cache")
    parser.add_argument("--detector-every", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(
        args.video, args.output, args.segments, args.states, args.ffmpeg,
        args.pose_python, args.workers, args.detector_every, args.pose_cache, args.player_attribution,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "rallies": result["rally_count"],
        "hits": result["hit_count"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
