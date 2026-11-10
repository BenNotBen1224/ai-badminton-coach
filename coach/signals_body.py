"""Body signals for technique and contact.

Distances are fractions of stature_px so they do not grow with camera zoom.
Speeds come from coach.signals, which differences measured samples in real time.
A joint is used only when it is visible. NaN means unknown.
"""
from __future__ import annotations

import weakref

import numpy as np

from .signals import speed

# A ready knee is at least this open, and the trunk is within this many degrees
# of vertical, before a frame can set the player's own upright hip reference.
UPRIGHT_KNEE_DEG = 150.0
UPRIGHT_LEAN_DEG = 20.0


def both_hand_speeds(player, tracks) -> dict[str, np.ndarray]:
    return {
        "right_wrist": relative_hand_speed(player, tracks, "right_wrist"),
        "left_wrist": relative_hand_speed(player, tracks, "left_wrist"),
    }


def relative_hand_speed(player, tracks, hand: str) -> np.ndarray:
    """Racket-hand speed in stature heights per second, relative to the shoulders.

    Shoulder motion is removed so running does not look like a swing.
    """
    rel, valid = _wrist_relative(player, hand)
    return _stature_speed(player, tracks, rel, valid)


def foot_speed_ratio(player, tracks) -> np.ndarray:
    foot = player.foot_px
    valid = player.measured & np.isfinite(foot).all(axis=1)
    spd = speed(np.where(valid[:, None], foot, np.nan), tracks.t, valid, tracks.shots)
    return _divide_stature(spd, player.stature_px)


def hip_height_px(player) -> np.ndarray:
    """Hip midpoint height above the lowest visible ankle, in source pixels."""
    hip, hip_ok = midpoint(player, "left_hip", "right_hip")
    ank_y, ank_ok = lowest_ankle_y(player)
    out = np.full(player.n, np.nan)
    ok = hip_ok & ank_ok
    out[ok] = ank_y[ok] - hip[ok, 1]
    out[~(out > 1.0)] = np.nan
    return out


def hip_height_ratio(player) -> np.ndarray:
    return _divide_stature(hip_height_px(player), player.stature_px)


def stance_width_ratio(player) -> np.ndarray:
    left = player.joint("left_ankle")
    right = player.joint("right_ankle")
    both = player.joint_visible("left_ankle") & player.joint_visible("right_ankle")
    dist = np.full(player.n, np.nan)
    dist[both] = np.linalg.norm(left[both] - right[both], axis=1)
    return _divide_stature(dist, player.stature_px)


def trunk_lean_deg(player) -> np.ndarray:
    """Sideways lean from vertical. Positive is clockwise in the image."""
    hip, hip_ok = midpoint(player, "left_hip", "right_hip")
    shoulder, shoulder_ok = midpoint(player, "left_shoulder", "right_shoulder")
    delta = shoulder - hip
    out = np.full(player.n, np.nan)
    ok = hip_ok & shoulder_ok & (np.linalg.norm(delta, axis=1) > 1.0)
    out[ok] = np.degrees(np.arctan2(delta[ok, 0], -delta[ok, 1]))
    return out


def knee_angle_deg(player, side: str) -> np.ndarray:
    return joint_angle(player, f"{side}_hip", f"{side}_knee", f"{side}_ankle")


def knee_flexion_deg(player) -> np.ndarray:
    """Degrees away from a straight knee. Mean of the visible knees."""
    left = knee_angle_deg(player, "left")
    right = knee_angle_deg(player, "right")
    out = np.full(player.n, np.nan)
    both = np.isfinite(left) & np.isfinite(right)
    out[both] = 180.0 - (left[both] + right[both]) * 0.5
    left_only = np.isfinite(left) & ~np.isfinite(right)
    right_only = np.isfinite(right) & ~np.isfinite(left)
    out[left_only] = 180.0 - left[left_only]
    out[right_only] = 180.0 - right[right_only]
    return out


def elbow_deg(player, hand: str) -> np.ndarray:
    side = "right" if hand.startswith("right") else "left"
    return joint_angle(player, f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")


def wrist_height_ratio(player, hand: str) -> np.ndarray:
    wrist = player.joint(hand)
    ank_y, ank_ok = lowest_ankle_y(player)
    vis = player.joint_visible(hand) & ank_ok
    height = np.full(player.n, np.nan)
    height[vis] = ank_y[vis] - wrist[vis, 1]
    return _divide_stature(height, player.stature_px)


def standing_hip_ratio(player) -> float:
    """Upright hip height as a fraction of stature_px.

    stature_px grows when the player is closer to the camera and does not
    shrink in a squat, so this ratio stays put when only the apparent size
    changes. A single pixel median does not: walking away looks like a drop.
    A frame counts as upright when it is measured, at least one knee is
    visible, every visible knee is at least 150 degrees, and sideways trunk
    lean is within 20 degrees of vertical. The reference is the median of
    those frames. With fewer than five, it is the 90th percentile of the
    ratios we do have.
    """
    ratio = hip_height_ratio(player)
    upright = upright_mask(player) & np.isfinite(ratio)
    if int(upright.sum()) >= 5:
        return float(np.median(ratio[upright]))
    finite = ratio[np.isfinite(ratio)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, 90))


_RATIO_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def clear_ratio_cache() -> None:
    """Drop cached ratios. Editing a live player's pose is not a new object."""
    _RATIO_CACHE.clear()


def upright_hip_ratio_at(player) -> np.ndarray:
    cached = _RATIO_CACHE.get(player)
    if cached is not None:
        return cached
    computed = _upright_hip_ratio_at(player)
    _RATIO_CACHE[player] = computed
    return computed


def _upright_hip_ratio_at(player) -> np.ndarray:
    """Upright hip ratio matched to each frame's stature.

    One clip-wide ratio still mixes a close pose with a far one when the
    image shape changes with distance. Each frame uses upright samples within
    8% of its own stature_px. A fixed count of neighbors can still span a
    large size change on a fast walk.
    """
    ratio = hip_height_ratio(player)
    stature = np.asarray(player.stature_px, dtype=np.float64)
    upright = upright_mask(player) & np.isfinite(ratio) & np.isfinite(stature) & (stature > 1.0)
    out = np.full(player.n, np.nan)
    idx = np.flatnonzero(upright)
    finite = ratio[np.isfinite(ratio)]
    fallback = float(np.percentile(finite, 90)) if finite.size else float("nan")
    if idx.size < 5:
        out[:] = fallback
        return out
    order = np.argsort(stature[idx])
    u_stat = stature[idx][order]
    u_ratio = ratio[idx][order]
    out[:] = float(np.median(u_ratio))
    valid = np.isfinite(stature) & (stature > 1.0)
    query = np.where(valid, stature, u_stat[0])
    lo = np.searchsorted(u_stat, query * 0.92, side="left")
    hi = np.searchsorted(u_stat, query * 1.08, side="right")
    keys = np.where(valid, lo.astype(np.int64) * (u_stat.size + 1) + hi, -1)
    for key in np.unique(keys):
        if key < 0:
            continue
        members = np.flatnonzero(keys == key)
        start = int(lo[members[0]])
        stop = int(hi[members[0]])
        if stop - start >= 5:
            out[members] = _median(u_ratio[start:stop])
            continue
        for i in members:
            out[i] = _thin_upright_ratio(u_stat, u_ratio, float(stature[i]), bool(upright[i]), float(ratio[i]))
    return out


def _median(values: np.ndarray) -> float:
    count = int(values.size)
    if count == 0:
        return float("nan")
    mid = count // 2
    part = np.partition(values, mid)
    high = float(part[mid])
    if count % 2:
        return high
    return 0.5 * (float(np.max(part[:mid])) + high)


def _thin_upright_ratio(u_stat, u_ratio, target, is_upright, own) -> float:
    k = int(np.searchsorted(u_stat, target))
    k = min(max(k, 0), u_stat.size - 1)
    start = max(0, min(k - 2, u_stat.size - 5))
    near = u_stat[start:start + 5]
    if float(np.max(np.abs(near - target))) <= 0.15 * target:
        return _median(u_ratio[start:start + 5])
    if is_upright:
        return own
    return float(u_ratio[k])


def standing_hip_px(player, frame: int) -> float:
    """Upright hip line in pixels at one frame's stature.

    The upright ratio at that frame's apparent size, times stature_px.
    Not a clip-wide pixel median.
    """
    ratio = float(upright_hip_ratio_at(player)[int(frame)])
    stature = float(player.stature_px[int(frame)])
    if not (np.isfinite(ratio) and np.isfinite(stature) and stature > 1.0):
        return float("nan")
    return float(ratio * stature)


def upright_mask(player) -> np.ndarray:
    left = knee_angle_deg(player, "left")
    right = knee_angle_deg(player, "right")
    lean = np.abs(trunk_lean_deg(player))
    have_knee = np.isfinite(left) | np.isfinite(right)
    bent = (np.isfinite(left) & (left < UPRIGHT_KNEE_DEG)) | (np.isfinite(right) & (right < UPRIGHT_KNEE_DEG))
    leaning = np.isfinite(lean) & (lean > UPRIGHT_LEAN_DEG)
    return player.measured & have_knee & ~bent & ~leaning


def base_stance_ratio(player) -> float:
    """Narrow ready width used to tell a lunge step from a split-step dip."""
    stance = stance_width_ratio(player)
    upright = upright_mask(player) & np.isfinite(stance)
    if int(upright.sum()) >= 5:
        return float(np.median(stance[upright]))
    finite = stance[np.isfinite(stance)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, 20))


def ready_bouts(mask: np.ndarray, shots, t: np.ndarray, min_s: float = 0.30) -> list[tuple[int, int]]:
    """Contiguous ready runs, split on camera cuts, each at least min_s long."""
    runs = []
    n = mask.shape[0]
    for start, end in shots:
        start = max(0, int(start))
        end = min(n - 1, int(end))
        i = start
        while i <= end:
            if not mask[i]:
                i += 1
                continue
            j = i + 1
            while j <= end and mask[j]:
                j += 1
            last = j - 1
            if float(t[last] - t[i]) >= min_s - 1e-9:
                runs.append((i, last))
            i = j
    return runs


def joint_angle(player, a: str, b: str, c: str) -> np.ndarray:
    pa, pb, pc = player.joint(a), player.joint(b), player.joint(c)
    valid = player.joint_visible(a) & player.joint_visible(b) & player.joint_visible(c)
    ba = pa - pb
    bc = pc - pb
    denom = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    cos = np.divide(np.sum(ba * bc, axis=1), denom, out=np.full(player.n, np.nan), where=denom > 1e-6)
    angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    angle[~valid | ~np.isfinite(angle)] = np.nan
    return angle


def midpoint(player, left_name: str, right_name: str):
    left = player.joint(left_name)
    right = player.joint(right_name)
    both = player.joint_visible(left_name) & player.joint_visible(right_name)
    xy = np.full((player.n, 2), np.nan)
    xy[both] = (left[both] + right[both]) * 0.5
    return xy, both


def lowest_ankle_y(player):
    left = player.joint("left_ankle")
    right = player.joint("right_ankle")
    left_on = player.joint_visible("left_ankle")
    right_on = player.joint_visible("right_ankle")
    y = np.full(player.n, np.nan)
    both = left_on & right_on
    y[both] = np.maximum(left[both, 1], right[both, 1])
    y[left_on & ~right_on] = left[left_on & ~right_on, 1]
    y[right_on & ~left_on] = right[right_on & ~left_on, 1]
    return y, left_on | right_on


def head_y(player) -> np.ndarray:
    """Nose, else the eyes, else the ears. NaN when the head is not visible."""
    y = np.full(player.n, np.nan)
    nose = player.joint_visible("nose")
    y[nose] = player.joint("nose")[nose, 1]
    missing = ~np.isfinite(y)
    for pair in (("left_eye", "right_eye"), ("left_ear", "right_ear")):
        left = player.joint(pair[0])
        right = player.joint(pair[1])
        left_on = player.joint_visible(pair[0])
        right_on = player.joint_visible(pair[1])
        both = missing & left_on & right_on
        y[both] = (left[both, 1] + right[both, 1]) * 0.5
        one_left = missing & left_on & ~right_on
        one_right = missing & right_on & ~left_on
        y[one_left] = left[one_left, 1]
        y[one_right] = right[one_right, 1]
        missing = ~np.isfinite(y)
    return y


def shot_bounds(frame: int, shots) -> tuple[int, int]:
    for start, end in shots:
        if start <= frame <= end:
            return int(start), int(end)
    return 0, frame


def _wrist_relative(player, hand: str):
    side = "right_shoulder" if hand.startswith("right") else "left_shoulder"
    other = "left_shoulder" if side == "right_shoulder" else "right_shoulder"
    both = player.joint_visible(side) & player.joint_visible(other)
    origin = np.where(both[:, None], (player.joint(side) + player.joint(other)) * 0.5, player.joint(side))
    origin_ok = both | player.joint_visible(side)
    wrist = player.joint(hand)
    valid = player.measured & player.joint_visible(hand) & origin_ok & np.isfinite(wrist).all(axis=1)
    rel = np.where(valid[:, None], wrist - origin, np.nan)
    return rel, valid


def _stature_speed(player, tracks, points, valid) -> np.ndarray:
    spd = speed(points, tracks.t, valid, tracks.shots)
    return _divide_stature(spd, player.stature_px)


def _divide_stature(values, stature) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    ok = np.isfinite(values) & np.isfinite(stature) & (stature > 1.0)
    out[ok] = values[ok] / stature[ok]
    return out

