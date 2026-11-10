"""Scripted pose documents with exact truth for the analysis units.

Call `script` to build a document and its truth. The document matches a schema-3
pose JSON: source, court_lines_normalized, court_fit, cuts, and frames with
timestamp_seconds and players. Each player has role, track_id, bbox_xyxy,
box_confidence, and 17 joints as [x, y, conf]. A preview row adds "src".

    doc, truth = script(n=60, fps=30, near=PlayerPlan(...), far=PlayerPlan(...), cuts=(), noise=None)

PlayerPlan fields:
  court_xy            base feet, metres, (x across the court, y from the far baseline)
  posture             Posture
  path                Path or None. Feet move from court_xy to end_xy.
  lunge               Lunge or None. The right foot reaches toward the net and returns.
  swing               Swing or None. Overhead right-arm swing.
  split_step          SplitStep or None. A small hop, then a hip dip on landing.
  crouch_fraction     share of frames with a deep knee bend, spread across the clip

Posture fields:
  knee_flexion_deg    0 is straight. When set, this wins over hip_height_ratio.
  hip_height_ratio    hip height divided by the straight-leg hip height. Used only
                      when knee_flexion_deg is None.
  stance_width_m      distance between the ankles
  trunk_lean_deg      0 is vertical. Positive rotates the shoulders clockwise
                      in the image.

Path fields: start_frame, stop_frame, end_xy. Speed is distance divided by the
time from start to stop, and truth stores that speed.

Lunge fields: start_frame, depth_m, recovery_s. The reach takes 0.2 seconds.
truth stores bottom_frame and end_frame. end_frame is back at the base.

Swing fields: contact_frame, contact_height_m, elbow_deg. Wrist speed peaks
three frames before contact. truth stores peak_frame and contact_frame.
contact_height_m is metres above the feet. Bones are not stretched. If the arm
cannot reach, the wrist stops at full reach and truth["achieved_contact_height_m"]
is the height that was drawn.

SplitStep field: frame. The hop apex is that frame. The hip dip is the next frame.

Noise fields:
  jitter_px           gaussian pixel noise, seed controls it
  dropout             chance a joint is kept but scored 0.1, so it is not visible
  missing_role        "near" or "far"
  missing_frames      frames where that role is omitted
  seed

truth fields:
  fps, width, height, width_m (6.10), length_m (13.40), corners_px, cuts
  repeated_frames     filled by repeat_frames
  players[role].foot_court     (n, 2) metres. Mean of the ankles on the floor.
  players[role].stature_m      standing nose-to-ankle length, metres
  players[role].stature_px     standing nose-to-ankle pixels at that frame's depth
  players[role].px_per_m
  players[role].crouch_frames
  players[role].posture, path, lunge, swing, split_step

The court outline is the doubles court, the same rectangle load_tracks uses.
Near defaults to (3.05, 10.5). Far defaults to (3.05, 2.9).

make_preview(doc, stride) returns a new document. Every stride-th frame stays
measured. The frames between are linear blends of the surrounding measured
frames, marked src "interpolated". An interpolated joint uses the smaller of
the two neighbours' confidences. A joint that is missing on either neighbour
stays missing.

repeat_frames(doc, every, truth=None) copies the previous player rows onto
every every-th frame (indices 5, 11, 17 when every is 6) and returns those
indices. When truth is passed, it sets truth["repeated_frames"]. The copy is
exact, so load_tracks marks those rows as repeats rather than new measurements.

Bone lengths stay constant. Knee and elbow angles stay inside human limits.
Feet stay planted unless a path, lunge, split step, or swing is moving them.
A hop lifts the drawn ankles off the floor, so foot_court in truth stays the
floor position and will not match the lifted pixels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
SHIN = 0.42
THIGH = 0.44
TORSO = 0.50
NECK = 0.24
HIP_W = 0.26
SHOULDER_W = 0.38
UPPER_ARM = 0.28
FOREARM = 0.25
CONF = 0.95


@dataclass
class Posture:
    knee_flexion_deg: float | None = None
    hip_height_ratio: float | None = None
    stance_width_m: float = 0.36
    trunk_lean_deg: float = 0.0


@dataclass
class Path:
    start_frame: int
    stop_frame: int
    end_xy: tuple[float, float]


@dataclass
class Lunge:
    start_frame: int
    depth_m: float
    recovery_s: float


@dataclass
class Swing:
    contact_frame: int
    contact_height_m: float
    elbow_deg: float


@dataclass
class SplitStep:
    frame: int


@dataclass
class Noise:
    jitter_px: float = 0.0
    dropout: float = 0.0
    missing_role: str | None = None
    missing_frames: tuple[int, ...] = ()
    seed: int = 0


@dataclass
class PlayerPlan:
    court_xy: tuple[float, float] = (3.05, 10.5)
    posture: Posture = field(default_factory=Posture)
    path: Path | None = None
    lunge: Lunge | None = None
    swing: Swing | None = None
    split_step: SplitStep | None = None
    crouch_fraction: float = 0.0


def script(
    n: int = 60,
    fps: float = 30.0,
    width: int = 1280,
    height: int = 720,
    near: PlayerPlan | None = None,
    far: PlayerPlan | None = None,
    cuts: tuple[int, ...] = (),
    noise: Noise | None = None,
) -> tuple[dict, dict]:
    if n <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("n, fps, width and height must be positive")
    near = near or PlayerPlan(court_xy=(3.05, 10.5))
    far = far or PlayerPlan(court_xy=(3.05, 2.9))
    noise = noise or Noise()
    corners = _corners(width, height)
    to_court, from_court = _homographies(corners)
    rng = np.random.default_rng(noise.seed)
    plans = {"near": near, "far": far}
    crouch = {role: _crouch_set(n, plan.crouch_fraction) for role, plan in plans.items()}
    foot = {role: np.full((n, 2), np.nan) for role in plans}
    stature_px = {role: np.full(n, np.nan) for role in plans}
    px_per_m = {role: np.full(n, np.nan) for role in plans}
    frames = []
    for i in range(n):
        players = []
        for role, plan in plans.items():
            if noise.missing_role == role and i in noise.missing_frames:
                continue
            joints, ankles_m, scale, span = _frame_joints(
                role, plan, i, n, fps, from_court, to_court, i in crouch[role],
            )
            foot[role][i] = ankles_m.mean(axis=0)
            stature_px[role][i] = span
            px_per_m[role][i] = scale
            if noise.jitter_px:
                joints[:, :2] += rng.normal(0.0, noise.jitter_px, joints[:, :2].shape)
            conf = np.full(17, CONF)
            if noise.dropout >= 1.0:
                conf[:] = 0.1
            elif noise.dropout > 0.0:
                conf[rng.random(17) < noise.dropout] = 0.1
            players.append(_player_row(role, joints, conf))
        frames.append({
            "frame_index": i,
            "timestamp_seconds": i / float(fps),
            "players": players,
        })
    doc = {
        "schema_version": 3,
        "source": {
            "file": "synthetic.mp4",
            "fps": float(fps),
            "width": int(width),
            "height": int(height),
            "frame_count": int(n),
            "players": {"near": {"hand": "right"}, "far": {"hand": "right"}},
        },
        "court_lines_normalized": (corners / np.array([width, height])).tolist(),
        "court_fit": {"fit": 0.99},
        "cuts": [int(c) for c in cuts],
        "frames": frames,
    }
    truth = {
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "width_m": COURT_WIDTH_M,
        "length_m": COURT_LENGTH_M,
        "corners_px": corners.tolist(),
        "cuts": [int(c) for c in cuts],
        "repeated_frames": [],
        "players": {},
    }
    for role, plan in plans.items():
        truth["players"][role] = _player_truth(plan, fps, foot[role], stature_px[role], px_per_m[role], crouch[role])
        _record_contact_height(truth["players"][role], role, plan, frames, px_per_m[role])
    return doc, truth


def make_preview(doc: dict, stride: int) -> dict:
    if stride < 2:
        raise ValueError("stride must be at least 2")
    out = {
        "schema_version": doc.get("schema_version", 3),
        "tier": "preview",
        "source": dict(doc["source"]),
        "court_lines_normalized": [list(p) for p in doc["court_lines_normalized"]],
        "court_fit": dict(doc.get("court_fit") or {}),
        "cuts": list(doc.get("cuts") or []),
        "frames": [],
    }
    frames = doc["frames"]
    n = len(frames)
    keys = list(range(0, n, stride))
    for i, frame in enumerate(frames):
        if i in keys or not keys:
            out["frames"].append(_copy_frame(frame, interpolated=False))
            continue
        prev = max((k for k in keys if k < i), default=keys[0])
        nxt = min((k for k in keys if k > i), default=keys[-1])
        if prev == nxt:
            copied = _copy_frame(frames[prev], interpolated=True)
            copied["frame_index"] = frame["frame_index"]
            copied["timestamp_seconds"] = frame["timestamp_seconds"]
            out["frames"].append(copied)
            continue
        alpha = (i - prev) / (nxt - prev)
        out["frames"].append(_blend_frame(frames[prev], frames[nxt], frame, alpha))
    return out


def repeat_frames(doc: dict, every: int, truth: dict | None = None) -> list[int]:
    if every < 2:
        raise ValueError("every must be at least 2")
    indices = []
    frames = doc["frames"]
    for i in range(every - 1, len(frames), every):
        if i == 0:
            continue
        frames[i]["players"] = _copy_frame(frames[i - 1])["players"]
        for player in frames[i]["players"]:
            player.pop("src", None)
        indices.append(i)
    if truth is not None:
        truth["repeated_frames"] = list(indices)
    return indices


def _corners(width, height):
    base = np.array([[460.0, 80.0], [820.0, 80.0], [1180.0, 640.0], [100.0, 640.0]])
    return base * np.array([width / 1280.0, height / 720.0])


def _homographies(corners):
    dst = np.array(
        [[0.0, 0.0], [COURT_WIDTH_M, 0.0], [COURT_WIDTH_M, COURT_LENGTH_M], [0.0, COURT_LENGTH_M]],
        np.float32,
    )
    src = np.ascontiguousarray(corners, np.float32)
    return cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src)


def _map_points(xy, matrix):
    raw = np.ascontiguousarray(np.asarray(xy, np.float64), np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(raw, np.ascontiguousarray(matrix, np.float32)).reshape(-1, 2)


def _px_per_m(pixel, to_court):
    up = _map_points(pixel + np.array([0.0, 0.5]), to_court)[0]
    down = _map_points(pixel + np.array([0.0, -0.5]), to_court)[0]
    dist = float(np.linalg.norm(up - down))
    if dist < 1e-9:
        return float("nan")
    return 1.0 / dist


def _crouch_set(n, fraction):
    if fraction <= 0:
        return set()
    count = int(round(float(fraction) * n))
    if count <= 0:
        return set()
    if count >= n:
        return set(range(n))
    return {int(i) for i in np.linspace(0, n - 1, count)}


def _court_center(plan, frame):
    base = np.array(plan.court_xy, dtype=np.float64)
    path = plan.path
    if path is None or path.stop_frame <= path.start_frame:
        return base
    if frame <= path.start_frame:
        return base
    end = np.array(path.end_xy, dtype=np.float64)
    if frame >= path.stop_frame:
        return end
    alpha = (frame - path.start_frame) / (path.stop_frame - path.start_frame)
    return base + alpha * (end - base)


def _lunge_window(lunge, fps):
    reach_n = max(4, int(round(0.2 * fps)))
    recover_n = max(1, int(round(lunge.recovery_s * fps)))
    bottom = lunge.start_frame + reach_n
    end = bottom + recover_n
    return bottom, end


def _reach(frame, lunge, fps):
    if lunge is None:
        return 0.0
    bottom, end = _lunge_window(lunge, fps)
    if frame <= lunge.start_frame or frame >= end:
        return 0.0
    if frame <= bottom:
        return (frame - lunge.start_frame) / (bottom - lunge.start_frame)
    return 1.0 - (frame - bottom) / (end - bottom)


def _toward_net(role):
    return -1.0 if role == "near" else 1.0


def _flexion(posture, crouch, stance):
    if crouch:
        return 65.0
    if posture.knee_flexion_deg is not None:
        return float(posture.knee_flexion_deg)
    if posture.hip_height_ratio is not None:
        return _flexion_for_ratio(float(posture.hip_height_ratio), stance)
    return 0.0


def _ah_metres(flexion_deg):
    angle = math.radians(180.0 - flexion_deg)
    return math.sqrt(max(SHIN ** 2 + THIGH ** 2 - 2 * SHIN * THIGH * math.cos(angle), 1e-8))


def _standing_hip_height(stance):
    d = abs(stance - HIP_W) / 2
    return math.sqrt(max((SHIN + THIGH) ** 2 - d ** 2, 1e-8))


def _flexion_for_ratio(ratio, stance):
    target_h = max(0.45, min(1.0, ratio)) * _standing_hip_height(stance)
    d = abs(stance - HIP_W) / 2
    target_ah = math.sqrt(target_h ** 2 + d ** 2)
    cos = (SHIN ** 2 + THIGH ** 2 - target_ah ** 2) / (2 * SHIN * THIGH)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    return max(0.0, 180.0 - angle)


def _split_offsets(frame, split):
    if split is None:
        return 0.0, 0.0, 0.0
    f = split.frame
    if frame == f - 1:
        return 0.02, 0.0, 0.0
    if frame == f:
        return 0.04, 0.0, 0.02
    if frame == f + 1:
        return 0.0, 0.10, 0.06
    if frame == f + 2:
        return 0.0, 0.02, 0.03
    return 0.0, 0.0, 0.0


def _frame_joints(role, plan, frame, n, fps, from_court, to_court, crouch):
    lift, hip_drop, stance_extra = _split_offsets(frame, plan.split_step)
    stance = plan.posture.stance_width_m + stance_extra
    center = _court_center(plan, frame)
    reach = _reach(frame, plan.lunge, fps)
    left_m = center + np.array([-stance / 2, 0.0])
    right_m = center + np.array([stance / 2, 0.0])
    if reach:
        right_m = right_m + np.array([0.0, _toward_net(role) * reach * plan.lunge.depth_m])
    pixels = _map_points(np.vstack([left_m, right_m]), from_court)
    scale = _px_per_m(pixels.mean(axis=0), to_court)
    flexion = _flexion(plan.posture, crouch, stance)
    if hip_drop:
        flexion = max(flexion, _flexion_for_ratio(1.0 - hip_drop / _standing_hip_height(stance), stance))
    joints = _skeleton(pixels[0], pixels[1], scale, flexion, plan.posture.trunk_lean_deg)
    upright = _skeleton(pixels[0], pixels[1], scale, 0.0, 0.0)
    span = float((upright[15, 1] + upright[16, 1]) / 2 - upright[0, 1])
    if plan.swing is not None:
        _apply_swing(joints, plan.swing, scale, frame)
    if lift:
        joints[:, 1] -= lift * scale
    return joints, np.vstack([left_m, right_m]), scale, span


def _skeleton(ankle_l, ankle_r, scale, flexion, lean_deg):
    joints = np.zeros((17, 2), dtype=np.float64)
    joints[15] = ankle_l
    joints[16] = ankle_r
    hip_center = _hip_center(ankle_l, ankle_r, scale, flexion)
    joints[11] = hip_center + np.array([-HIP_W * scale / 2, 0.0])
    joints[12] = hip_center + np.array([HIP_W * scale / 2, 0.0])
    outward_l = -1.0 if ankle_l[0] <= ankle_r[0] else 1.0
    joints[13] = _knee_point(ankle_l, joints[11], SHIN * scale, THIGH * scale, outward_l)
    joints[14] = _knee_point(ankle_r, joints[12], SHIN * scale, THIGH * scale, -outward_l)
    shoulder_mid = _rotate(hip_center + np.array([0.0, -TORSO * scale]), hip_center, lean_deg)
    across = _rotate(np.array([SHOULDER_W * scale / 2, 0.0]), np.zeros(2), lean_deg)
    joints[5] = shoulder_mid - across
    joints[6] = shoulder_mid + across
    nose = _rotate(shoulder_mid + np.array([0.0, -NECK * scale]), hip_center, lean_deg)
    joints[0] = nose
    joints[1] = nose + np.array([-0.03 * scale, 0.02 * scale])
    joints[2] = nose + np.array([0.03 * scale, 0.02 * scale])
    joints[3] = nose + np.array([-0.07 * scale, 0.04 * scale])
    joints[4] = nose + np.array([0.07 * scale, 0.04 * scale])
    elbow, wrist = _arm(joints[5], scale, 8.0, 170.0, -1.0)
    joints[7] = elbow
    joints[9] = wrist
    elbow, wrist = _arm(joints[6], scale, 8.0, 170.0, 1.0)
    joints[8] = elbow
    joints[10] = wrist
    return joints


def _hip_center(ankle_l, ankle_r, scale, flexion):
    ah = _ah_metres(flexion) * scale
    hip_w = HIP_W * scale / 2
    mid_x = (float(ankle_l[0]) + float(ankle_r[0])) / 2
    dx = max(abs((mid_x - hip_w) - float(ankle_l[0])), abs((mid_x + hip_w) - float(ankle_r[0])))
    vertical = math.sqrt(max(ah * ah - dx * dx, 1e-8))
    floor_y = max(float(ankle_l[1]), float(ankle_r[1]))
    hip = np.array([mid_x, floor_y - vertical])
    max_leg = (SHIN + THIGH) * scale
    for _ in range(8):
        far = max(
            np.linalg.norm(hip + np.array([-hip_w, 0.0]) - ankle_l),
            np.linalg.norm(hip + np.array([hip_w, 0.0]) - ankle_r),
        )
        if far <= max_leg + 1e-4:
            return hip
        hip[1] += far - max_leg
    return hip


def _knee_point(ankle, hip, shin_px, thigh_px, outward):
    ah = hip - ankle
    dist = float(np.linalg.norm(ah))
    dist = min(max(dist, abs(shin_px - thigh_px) + 1e-3), shin_px + thigh_px - 1e-3)
    direction = ah / max(float(np.linalg.norm(ah)), 1e-6)
    a = (shin_px ** 2 - thigh_px ** 2 + dist ** 2) / (2 * dist)
    h = math.sqrt(max(shin_px ** 2 - a ** 2, 0.0))
    base = ankle + direction * a
    perp = np.array([-direction[1], direction[0]])
    first = base + h * perp
    second = base - h * perp
    if outward >= 0:
        return first if first[0] >= second[0] else second
    return first if first[0] <= second[0] else second


def _rotate(point, origin, deg):
    if deg == 0:
        return point
    rad = math.radians(deg)
    delta = point - origin
    rot = np.array([
        [math.cos(rad), -math.sin(rad)],
        [math.sin(rad), math.cos(rad)],
    ])
    return origin + rot @ delta


def _arm(shoulder, scale, raise_deg, elbow_deg, side):
    rad = math.radians(raise_deg)
    direction = np.array([math.sin(rad) * side, math.cos(rad)])
    elbow = np.asarray(shoulder, dtype=np.float64) + direction * (UPPER_ARM * scale)
    bend = math.radians(180.0 - elbow_deg) * side
    rot = np.array([[math.cos(bend), -math.sin(bend)], [math.sin(bend), math.cos(bend)]])
    wrist = elbow + (rot @ direction) * (FOREARM * scale)
    return elbow, wrist


def _apply_swing(joints, swing, scale, frame):
    peak = swing.contact_frame - 3
    start = max(0, peak - 8)
    if frame < start:
        return
    shoulder = joints[6]
    contact_raise = _solve_raise(shoulder, float(joints[16, 1]), scale, swing.elbow_deg, swing.contact_height_m)
    raise_deg, elbow_deg = _swing_angles(frame, swing, contact_raise, shoulder, scale, start)
    joints[8], joints[10] = _arm(shoulder, scale, raise_deg, elbow_deg, 1.0)


def _swing_angles(frame, swing, contact_raise, shoulder, scale, start):
    if frame >= swing.contact_frame:
        return contact_raise, float(swing.elbow_deg)
    end = swing.contact_frame
    frames = np.arange(start, end + 1)
    samples = 160
    mix = np.linspace(0.0, 1.0, samples)
    raises = 8.0 + mix * (contact_raise - 8.0)
    elbows = 170.0 + mix * (float(swing.elbow_deg) - 170.0)
    wrists = np.array([
        _arm(shoulder, scale, float(r), float(e), 1.0)[1] for r, e in zip(raises, elbows)
    ])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(wrists, axis=0), axis=1))])
    total = float(arc[-1])
    if total < 1e-6:
        return contact_raise, float(swing.elbow_deg)
    peak = swing.contact_frame - 3
    omega = np.exp(-0.5 * ((frames.astype(np.float64) - peak) / 0.55) ** 2)
    progress = np.cumsum(omega / omega.sum())
    target = float(progress[int(frame - start)]) * total
    k = int(np.searchsorted(arc, target, side="left"))
    k = min(max(k, 1), len(arc) - 1)
    span = float(arc[k] - arc[k - 1])
    alpha = 0.0 if span < 1e-9 else (target - float(arc[k - 1])) / span
    alpha = min(max(alpha, 0.0), 1.0)
    raise_deg = float(raises[k - 1] + alpha * (raises[k] - raises[k - 1]))
    elbow_deg = float(elbows[k - 1] + alpha * (elbows[k] - elbows[k - 1]))
    return raise_deg, elbow_deg


def _record_contact_height(info, role, plan, frames, px_per_m):
    swing = plan.swing
    if swing is None or not info.get("swing"):
        return
    frame = swing.contact_frame
    if frame < 0 or frame >= len(frames):
        return
    row = next((p for p in frames[frame]["players"] if p["role"] == role), None)
    if row is None or px_per_m[frame] <= 0:
        return
    wrist_y = row["joints"][10][1]
    foot_y = (row["joints"][15][1] + row["joints"][16][1]) / 2
    info["swing"]["achieved_contact_height_m"] = (foot_y - wrist_y) / float(px_per_m[frame])


def _solve_raise(shoulder, foot_y, scale, elbow_deg, target_m):
    target_y = foot_y - target_m * scale
    lo, hi = 0.0, 180.0
    for _ in range(24):
        mid = (lo + hi) / 2
        _, wrist = _arm(shoulder, scale, mid, elbow_deg, 1.0)
        if wrist[1] > target_y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _player_row(role, joints, conf):
    xs = joints[:, 0]
    ys = joints[:, 1]
    track_id = 1 if role == "near" else 2
    return {
        "role": role,
        "track_id": track_id,
        "bbox_xyxy": [float(xs.min() - 4), float(ys.min() - 4), float(xs.max() + 4), float(ys.max() + 4)],
        "box_confidence": 0.9,
        "joints": [[float(joints[j, 0]), float(joints[j, 1]), float(conf[j])] for j in range(17)],
    }


def _player_truth(plan, fps, foot, stature_px, px_per_m, crouch):
    posture = plan.posture
    out = {
        "foot_court": foot,
        "stature_m": float(np.nanmedian(stature_px / px_per_m)),
        "stature_px": stature_px,
        "px_per_m": px_per_m,
        "crouch_frames": sorted(crouch),
        "posture": {
            "knee_flexion_deg": posture.knee_flexion_deg,
            "hip_height_ratio": posture.hip_height_ratio,
            "stance_width_m": posture.stance_width_m,
            "trunk_lean_deg": posture.trunk_lean_deg,
        },
        "path": None,
        "lunge": None,
        "swing": None,
        "split_step": None,
    }
    if plan.path is not None and plan.path.stop_frame > plan.path.start_frame:
        start = np.array(plan.court_xy, dtype=np.float64)
        end = np.array(plan.path.end_xy, dtype=np.float64)
        seconds = (plan.path.stop_frame - plan.path.start_frame) / float(fps)
        out["path"] = {
            "start_frame": plan.path.start_frame,
            "stop_frame": plan.path.stop_frame,
            "end_xy": [float(end[0]), float(end[1])],
            "speed_m_s": float(np.linalg.norm(end - start) / seconds),
        }
    if plan.lunge is not None:
        bottom, end = _lunge_window(plan.lunge, fps)
        out["lunge"] = {
            "start_frame": plan.lunge.start_frame,
            "depth_m": plan.lunge.depth_m,
            "recovery_s": plan.lunge.recovery_s,
            "bottom_frame": bottom,
            "end_frame": end,
        }
    if plan.swing is not None:
        out["swing"] = {
            "contact_frame": plan.swing.contact_frame,
            "peak_frame": plan.swing.contact_frame - 3,
            "contact_height_m": plan.swing.contact_height_m,
            "elbow_deg": plan.swing.elbow_deg,
        }
    if plan.split_step is not None:
        out["split_step"] = {"frame": plan.split_step.frame}
    return out


def _copy_frame(frame, interpolated=False):
    players = []
    for player in frame["players"]:
        row = {
            "role": player["role"],
            "track_id": player["track_id"],
            "bbox_xyxy": list(player["bbox_xyxy"]),
            "box_confidence": player["box_confidence"],
            "joints": [None if joint is None else list(joint) for joint in player["joints"]],
        }
        if interpolated or player.get("src") == "interpolated":
            row["src"] = "interpolated"
        players.append(row)
    return {
        "frame_index": frame["frame_index"],
        "timestamp_seconds": frame["timestamp_seconds"],
        "players": players,
    }


def _blend_frame(prev, nxt, current, alpha):
    prev_by = {p["role"]: p for p in prev["players"]}
    next_by = {p["role"]: p for p in nxt["players"]}
    players = []
    for role in ("near", "far"):
        a = prev_by.get(role)
        b = next_by.get(role)
        if a is None or b is None:
            continue
        joints = []
        for ja, jb in zip(a["joints"], b["joints"]):
            if ja is None or jb is None:
                joints.append(None)
                continue
            joints.append([
                (1 - alpha) * ja[0] + alpha * jb[0],
                (1 - alpha) * ja[1] + alpha * jb[1],
                min(ja[2], jb[2]),
            ])
        xs = [j[0] for j in joints if j is not None]
        ys = [j[1] for j in joints if j is not None]
        players.append({
            "role": role,
            "track_id": a["track_id"],
            "bbox_xyxy": [min(xs) - 4, min(ys) - 4, max(xs) + 4, max(ys) + 4],
            "box_confidence": min(a["box_confidence"], b["box_confidence"]),
            "joints": joints,
            "src": "interpolated",
        })
    return {
        "frame_index": current["frame_index"],
        "timestamp_seconds": current["timestamp_seconds"],
        "players": players,
    }
