"""Swing and lunge events for both players.

Swing speed is the racket hand relative to the shoulders, in stature heights
per second, from measured samples and real time. The racket arm is explicitly
provided by the caller; unknown handedness withholds swings. Start and end are the frames where that speed crosses
30% of the peak, inside the same camera shot. End extends to the contact frame
so contact stays inside the swing.

Contact for an overhead hit is 0.10 s after the speed peak when the wrist is
still near its highest point in that window. Otherwise contact is the highest
wrist at or after the peak. Overhead means the wrist is above the head.
That frame is a geometric estimate. The four clip 01 labels did not confirm
it, so findings withhold a hit claim rather than treating the peak as impact.

A peak is rejected when it is an isolated spike, when the hand barely moves,
when the wrist is sitting on the hip, or when a dangling arm swings during a
run or a lunge. That rejection happens before spacing. A louder peak that
will never be emitted cannot erase a quieter eligible neighbor. The 0.40 s
gap still keeps the stronger of two eligible peaks. A dangling arm is a bent
elbow with the wrist close to the hip. An extended below-shoulder reach is
not dangling, so a low stroke is kept.

A lunge is a hip drop of at least 6% of the player's standing hip height
together with a step that widens the ankles by at least 0.10 stature. Depth
and the ready comparison use hip height divided by stature_px, so walking
toward the camera does not look like a drop. The upright ratio is the depth
yardstick. The cycle itself is delimited by the observed ready hip just
before the drop, not by a return to full standing height. frame is the last
frame of that ready spell. bottom_frame is the lowest hip. end_frame is the
first frame where the hips have recovered 90% of that local drop and the step
has closed to the pre-lunge width. A missing hip or a shot that ends before
that return is not a measured end: the event is omitted and recovery_windows
records it. Unknown stance does not count as a closed step. That end_frame
is the recovery frame unit C1b reads.
"""
from __future__ import annotations

import numpy as np

from .schema import Event, Tier, Window, lunge_event, swing_event
from .signals_body import (
    base_stance_ratio,
    both_hand_speeds,
    clear_ratio_cache,
    elbow_deg,
    foot_speed_ratio,
    head_y,
    hip_height_ratio,
    knee_angle_deg,
    shot_bounds,
    stance_width_ratio,
    upright_hip_ratio_at,
)
from .tracks import JOINT_INDEX, Tracks

MIN_SWING_SPEED = 3.0
PEAK_FRACTION = 0.30
SWING_GAP_S = 0.40
MAX_SWING_S = 0.50
CONTACT_DELAY_S = 0.10
MIN_TRAVEL = 0.12
RUN_FOOT_SPEED = 1.5
MIN_LUNGE_DEPTH = 0.06
MAX_LUNGE_DEPTH = 0.70
MIN_STEP_GAIN = 0.10
MIN_LUNGE_S = 0.15
LUNGE_GAP_S = 0.50
READY_WINDOW_S = 0.15
READY_STABLE = 0.04
LOCAL_DROP = 0.06
LOOKBACK_S = 0.80
RECOVER_FRACTION = 0.90
DANGLING_ELBOW = 150.0
DANGLING_REACH = 0.35
HIP_GLUED = 0.10
DEEP_KNEE_FLEXION = 75.0
STRAIGHT_RETURN_FLEXION = 20.0
UNILATERAL_FLEXION_GAP = 40.0
MIN_LUNGE_STRIDE = 0.40


def detect_events(tracks: Tracks) -> tuple[Event, ...]:
    clear_ratio_cache()
    found = []
    for player in tracks.players.values():
        if int(player.present.sum()) < 3:
            continue
        lunges, _windows = _lunges(player, tracks)
        found.extend(lunges)
        found.extend(_swings(player, tracks, lunges))
    return tuple(sorted(found, key=lambda e: (e.frame, e.kind, e.player.value)))


def recovery_windows(tracks: Tracks) -> tuple[Window, ...]:
    """Drops that never returned to the pre-lunge ready height. Not measured lunges."""
    found = []
    for player in tracks.players.values():
        if int(player.present.sum()) < 3:
            continue
        _events, windows = _lunges(player, tracks)
        found.extend(windows)
    return tuple(sorted(found, key=lambda w: (w.start_frame, w.end_frame, w.reason)))


def _swings(player, tracks, lunges):
    if player.hand == "unknown":
        return []
    speeds = both_hand_speeds(player, tracks)
    hand = f"{player.hand}_wrist"
    spd = speeds[hand]
    if player.n < 3 or not np.isfinite(spd).any():
        return []
    peaks = _peak_indices(spd)
    visible = player.joint_visible(hand)
    peaks = [int(peak) for peak in peaks if _supported(spd, int(peak)) and visible[int(peak)]]
    foot = foot_speed_ratio(player, tracks)
    eligible = []
    for peak in peaks:
        event = _swing_from_peak(player, tracks, hand, spd, int(peak), foot, lunges)
        if event is not None:
            eligible.append(event)
    kept = _keep_spaced([int(event.data["peak_frame"]) for event in eligible], spd, tracks.t, SWING_GAP_S)
    events = [event for event in eligible if int(event.data["peak_frame"]) in set(kept)]
    return _drop_overlaps(events)


def _swing_from_peak(player, tracks, hand, spd, peak, foot, lunges):
    if not _supported(spd, peak) or not player.joint_visible(hand)[peak]:
        return None
    shot_start, shot_end = shot_bounds(peak, tracks.shots)
    peak_speed = float(spd[peak])
    start = _cross(spd, peak, peak_speed, shot_start, shot_end, -1)
    speed_end = _cross(spd, peak, peak_speed, shot_start, shot_end, 1)
    if _travel(player, hand, start, peak) < MIN_TRAVEL:
        return None
    if _at_hip(player, hand, peak):
        return None
    if _dangling(player, hand, peak) and (_running(foot, peak) or _in_lunge(peak, lunges)):
        return None
    contact, overhead = _contact(player, tracks, hand, peak, speed_end, shot_end)
    end = int(min(shot_end, max(speed_end, contact)))
    start = int(max(shot_start, min(start, peak)))
    if tracks.t[end] - tracks.t[start] > MAX_SWING_S:
        end = _frame_at(tracks, start, MAX_SWING_S, shot_end)
        contact = int(min(contact, end))
    if not (start <= peak <= end and start <= contact <= end):
        return None
    conf = _pose_confidence(player, hand, peak, spd)
    if tracks.tier == Tier.PREVIEW:
        conf = min(conf, 0.55)
    return swing_event(
        player.role,
        contact,
        start=start,
        end=end,
        peak_frame=peak,
        peak_speed=peak_speed,
        overhead=overhead,
        right_hand=hand.startswith("right"),
        confidence=conf,
    )


def _contact(player, tracks, hand, peak, speed_end, shot_end):
    delay = int(round(CONTACT_DELAY_S * tracks.fps))
    window_end = min(shot_end, max(speed_end, peak + max(delay, 1)))
    wrist_y = player.joint(hand)[:, 1]
    visible = player.joint_visible(hand)
    frames = [i for i in range(peak, window_end + 1) if visible[i] and np.isfinite(wrist_y[i])]
    if not frames:
        return peak, False
    high = min(frames, key=lambda i: wrist_y[i])
    delayed = min(peak + delay, window_end)
    stature = player.stature_px[peak]
    near_top = np.isfinite(stature) and visible[delayed] and wrist_y[delayed] <= wrist_y[high] + 0.04 * stature
    overhead = _above_head(player, hand, frames)
    contact = delayed if overhead and near_top else high
    return int(contact), overhead


def _above_head(player, hand, frames) -> bool:
    wrist_y = player.joint(hand)[:, 1]
    heads = head_y(player)
    return any(np.isfinite(heads[i]) and wrist_y[i] < heads[i] for i in frames)


def _cross(spd, peak, peak_speed, shot_start, shot_end, step) -> int:
    """Frame where speed crosses 30% of the peak, walking out from the peak.

    A short hole is skipped. A longer hole is unknown, so the walk stops.
    """
    limit = peak_speed * PEAK_FRACTION
    crossed = peak
    i = peak + step
    gap = 0
    while shot_start <= i <= shot_end:
        if not np.isfinite(spd[i]):
            gap += 1
            if gap > 4:
                break
            i += step
            continue
        gap = 0
        if spd[i] < limit:
            return int(i)
        crossed = i
        i += step
    return int(crossed)


def _travel(player, hand, start, peak) -> float:
    if not player.joint_visible(hand)[start] or not player.joint_visible(hand)[peak]:
        return 0.0
    dist = float(np.linalg.norm(player.joint(hand)[peak] - player.joint(hand)[start]))
    stature = player.stature_px[peak]
    if not np.isfinite(stature) or stature <= 1.0:
        return 0.0
    return dist / float(stature)


def _at_hip(player, hand, frame) -> bool:
    """Wrist sitting on the hip. A straight arm can do this without looking bent."""
    side = "right" if hand.startswith("right") else "left"
    hip_name = f"{side}_hip"
    if not player.joint_visible(hand)[frame] or not player.joint_visible(hip_name)[frame]:
        return False
    stature = player.stature_px[frame]
    if not np.isfinite(stature) or stature <= 1.0:
        return False
    dist = float(np.linalg.norm(player.joint(hand)[frame] - player.joint(hip_name)[frame]))
    return dist <= HIP_GLUED * float(stature)


def _dangling(player, hand, frame) -> bool:
    """Bent elbow and wrist close to the hip. An extended reach is not dangling.

    Missing elbow or hip is unknown, not evidence of a dangling arm.
    """
    side = "right" if hand.startswith("right") else "left"
    shoulder = f"{side}_shoulder"
    hip_name = f"{side}_hip"
    elbow_name = f"{side}_elbow"
    needed = (hand, shoulder, hip_name, elbow_name)
    if not all(player.joint_visible(name)[frame] for name in needed):
        return False
    stature = player.stature_px[frame]
    if not np.isfinite(stature) or stature <= 1.0:
        return False
    below = float(player.joint(hand)[frame, 1]) > float(player.joint(shoulder)[frame, 1]) + 0.02 * float(stature)
    if not below:
        return False
    angle = float(elbow_deg(player, hand)[frame])
    if not np.isfinite(angle) or angle >= DANGLING_ELBOW:
        return False
    dist = float(np.linalg.norm(player.joint(hand)[frame] - player.joint(hip_name)[frame]))
    return dist <= DANGLING_REACH * float(stature)


def _running(foot, frame) -> bool:
    return bool(np.isfinite(foot[frame]) and foot[frame] > RUN_FOOT_SPEED)


def _in_lunge(frame, lunges) -> bool:
    return any(int(event.frame) <= frame <= int(event.data["end_frame"]) for event in lunges)


def _nearest_speed(spd, peak, step) -> float:
    """Nearest measured speed. A missing frame is a gap, not a quiet sample."""
    i = peak + step
    for _ in range(4):
        if i < 0 or i >= spd.shape[0]:
            return 0.0
        if np.isfinite(spd[i]):
            return float(spd[i])
        i += step
    return 0.0


def _supported(spd, peak) -> bool:
    if not np.isfinite(spd[peak]) or float(spd[peak]) <= 0.0:
        return False
    return _nearest_speed(spd, peak, -1) + _nearest_speed(spd, peak, 1) >= 0.15 * float(spd[peak])


def _pose_confidence(player, hand, peak, spd) -> float:
    side = "right" if hand.startswith("right") else "left"
    names = (hand, f"{side}_elbow", f"{side}_shoulder")
    confs = []
    for name in names:
        if player.joint_visible(name)[peak]:
            confs.append(min(1.0, float(player.conf[peak, JOINT_INDEX[name]])))
    pose = float(np.median(confs)) if confs else 0.4
    window = spd[max(0, peak - 5):peak + 6]
    window = window[np.isfinite(window)]
    sharp = 1.0
    if window.size:
        sharp = float(np.clip((spd[peak] - np.median(window)) / max(float(spd[peak]), 1e-6), 0.0, 1.0))
    return float(np.clip(0.6 * pose + 0.4 * sharp, 0.0, 1.0))


def _peak_indices(spd) -> list[int]:
    finite = np.flatnonzero(np.isfinite(spd))
    peaks = []
    for pos, i in enumerate(finite):
        if pos == 0 or pos + 1 == finite.size:
            continue
        prev = int(finite[pos - 1])
        nxt = int(finite[pos + 1])
        if i - prev > 4 or nxt - i > 4:
            continue
        if spd[i] < MIN_SWING_SPEED:
            continue
        if spd[i] >= spd[prev] and spd[i] > spd[nxt]:
            peaks.append(int(i))
    return peaks


def _keep_spaced(peaks, values, t, gap_s) -> list[int]:
    kept = []
    for peak in sorted(peaks, key=lambda i: -float(values[i])):
        if any(abs(float(t[peak] - t[other])) < gap_s for other in kept):
            continue
        kept.append(peak)
    return sorted(kept)


def _drop_overlaps(events):
    kept = []
    for event in sorted(events, key=lambda e: -float(e.data["peak_speed"])):
        if any(_overlap(event, other) for other in kept):
            continue
        kept.append(event)
    return kept


def _overlap(a, b) -> bool:
    return int(a.data["start"]) <= int(b.data["end"]) and int(b.data["start"]) <= int(a.data["end"])


def _frame_at(tracks, start, seconds, shot_end) -> int:
    goal = float(tracks.t[start]) + seconds
    frames = np.flatnonzero(tracks.t <= goal)
    if frames.size == 0:
        return start
    return int(min(int(frames[-1]), shot_end))


def _lunges(player, tracks):
    ratio = hip_height_ratio(player)
    stand = upright_hip_ratio_at(player)
    stance = stance_width_ratio(player)
    base = base_stance_ratio(player)
    if not np.isfinite(stand).any() or not np.isfinite(base):
        return [], []
    events = []
    windows = []
    for start, end in tracks.shots:
        found, gaps = _lunges_in_shot(
            player, tracks, ratio, stand, stance, base, int(start), int(end),
        )
        events.extend(found)
        windows.extend(gaps)
    supplemental = _knee_stride_lunges(player, tracks, ratio, stand, stance)
    for event in supplemental:
        events = [other for other in events if not _lunge_overlap(event, other)]
        events.append(event)
    return _keep_deeper(events, tracks.t), _merge_gap_windows(windows)


def _knee_stride_lunges(player, tracks, ratio, stand, stance):
    """Unilateral deep knee cycle with a travelling lead foot and visible return.

    Hip height can barely change when a high rear camera foreshortens the thigh.
    This independent signal does not relabel knee flexion as a hip-height drop.
    """
    flexions = {side: 180.0 - knee_angle_deg(player, side) for side in ("left", "right")}
    found = []
    for shot_start, shot_end in tracks.shots:
        for side, flexion in flexions.items():
            other = "left" if side == "right" else "right"
            smooth = np.full(player.n, np.nan)
            for frame in range(shot_start + 2, shot_end - 1):
                window = flexion[frame - 2:frame + 3]
                if np.isfinite(window).sum() >= 4 and player.present[frame]:
                    smooth[frame] = float(np.nanmedian(window))
            mask = (smooth >= DEEP_KNEE_FLEXION) & (stance >= MIN_LUNGE_STRIDE)
            transitions = np.flatnonzero(np.diff(np.r_[False, mask, False]))
            for first, stop in zip(transitions[::2], transitions[1::2]):
                frames = np.arange(first, stop)
                if player.measured[frames].sum() < 3 or tracks.t[frames[-1]] - tracks.t[first] < 0.10:
                    continue
                bottom = int(max(frames, key=lambda f: smooth[f]))
                plateau = frames[(smooth[frames] >= smooth[bottom] - 3) & player.measured[frames]]
                bottom = int(max(plateau, key=lambda f: flexion[f]))
                if not np.isfinite(flexions[other][bottom]) or flexion[bottom] - flexions[other][bottom] < UNILATERAL_FLEXION_GAP:
                    continue
                before = max(shot_start, bottom - round(tracks.fps * 0.5))
                ankle = player.joint(f"{side}_ankle")
                trailing = player.joint(f"{other}_ankle")
                needed = (f"{side}_ankle", f"{other}_ankle")
                if not all(player.joint_visible(name)[before] and player.joint_visible(name)[bottom] for name in needed):
                    continue
                travel = ankle[bottom] - ankle[before]
                trailing_travel = trailing[bottom] - trailing[before]
                stature = player.stature_px[bottom]
                if not np.isfinite(stature) or np.linalg.norm(travel) < 0.12 * stature:
                    continue
                if np.dot(travel - trailing_travel, travel) <= 0.05 * stature * np.linalg.norm(travel):
                    continue
                start = first
                while start > before and np.isfinite(smooth[start - 1]) and smooth[start - 1] > 40:
                    start -= 1
                start = max(before, start - 1)
                end = None
                for frame in range(bottom + 1, min(shot_end, bottom + round(1.5 * tracks.fps)) + 1):
                    if not np.isfinite(smooth[frame]):
                        break
                    if smooth[frame] <= STRAIGHT_RETURN_FLEXION and player.measured[frame]:
                        end = frame
                        break
                if end is None or not np.isfinite(ratio[bottom]) or not np.isfinite(stand[bottom]):
                    continue
                plant = int(first)
                depth = max(0.0, float((stand[bottom] - ratio[bottom]) / stand[bottom]))
                confidence = min(_lunge_confidence(player, bottom),
                                 *(min(1.0, float(player.conf[bottom, JOINT_INDEX[f"{side}_{joint}"]]))
                                   for joint in ("hip", "knee", "ankle")))
                found.append(Event("lunge", player.role, int(start), confidence, {
                    "depth": depth, "bottom_frame": bottom, "end_frame": end,
                    "knee_stride": 1, "lead_right": int(side == "right"),
                    "plant_frame": plant, "lead_knee_flexion_deg": float(flexion[bottom]),
                    "stride_ratio": float(stance[plant]),
                    "knee_return_s": float(tracks.t[end] - tracks.t[bottom]),
                }))
    return found


def _lunges_in_shot(player, tracks, ratio, stand, stance, base, start, end):
    bottoms = []
    for i in range(start + 1, end):
        if not (np.isfinite(ratio[i]) and np.isfinite(stand[i]) and stand[i] > 0.05):
            continue
        if not (np.isfinite(ratio[i - 1]) and np.isfinite(ratio[i + 1])):
            continue
        if ratio[i] > ratio[i - 1] or ratio[i] > ratio[i + 1]:
            continue
        depth = (float(stand[i]) - float(ratio[i])) / float(stand[i])
        gain = float(stance[i] - base) if np.isfinite(stance[i]) else 0.0
        if MIN_LUNGE_DEPTH <= depth <= MAX_LUNGE_DEPTH and gain >= MIN_STEP_GAIN:
            bottoms.append(i)
    bottoms = _keep_spaced(bottoms, -ratio, tracks.t, LUNGE_GAP_S)
    events = []
    windows = []
    for bottom in bottoms:
        event, window = _lunge_cycle(player, tracks, ratio, stand, stance, bottom, start, end)
        if event is not None:
            events.append(event)
        elif window is not None:
            windows.append(window)
    return events, windows


def _lunge_cycle(player, tracks, ratio, stand, stance, bottom, shot_start, shot_end):
    """A finished lunge, or a window when the return was not observed."""
    reason = f"{player.role.value} unresolved recovery"
    yard = float(stand[bottom])
    ready = _local_ready(ratio, stance, tracks.t, bottom, shot_start, yard)
    if ready is None:
        return None, Window(int(bottom), int(bottom), reason)
    level, start, pre_stance = ready
    end = _return_frame(ratio, stance, level, pre_stance, bottom, shot_end)
    if end is None:
        last = _last_seen(ratio, bottom, shot_end)
        return None, Window(int(start), int(max(start, last)), reason)
    if float(tracks.t[end] - tracks.t[start]) < MIN_LUNGE_S or not (start <= bottom <= end):
        return None, None
    depth = (yard - float(ratio[bottom])) / yard
    event = lunge_event(
        player.role,
        int(start),
        depth=float(depth),
        bottom_frame=int(bottom),
        end_frame=int(end),
        confidence=_lunge_confidence(player, bottom),
    )
    return event, None


def _tiny_median(values) -> float:
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    mid = count // 2
    if count % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _local_ready(ratio, stance, t, bottom, shot_start, stand):
    """Stable hip ratio just before the drop, plus the step width in that spell."""
    limit = float(t[bottom]) - LOOKBACK_S
    i = bottom - 1
    while i > shot_start and float(t[i]) >= limit:
        if not np.isfinite(ratio[i]):
            return None
        window_t = float(t[i]) - READY_WINDOW_S
        frames = []
        j = i
        while j >= shot_start and float(t[j]) >= window_t:
            if np.isfinite(ratio[j]):
                frames.append(j)
            j -= 1
        if frames and float(t[i] - t[frames[-1]]) >= READY_WINDOW_S * 0.6:
            chosen = np.array(frames)
            vals = ratio[chosen]
            level = _tiny_median(vals)
            stable = float(np.max(vals) - np.min(vals)) <= READY_STABLE * stand
            above = level >= float(ratio[bottom]) + LOCAL_DROP * stand
            widths = stance[chosen]
            widths = widths[np.isfinite(widths)]
            if stable and above and widths.size:
                return level, int(i), _tiny_median(widths)
        i -= 1
    return None


def _return_frame(ratio, stance, level, pre_stance, bottom, shot_end):
    """First frame back at the pre-lunge hips with the step closed.

    A missing hip ends the search. The shot end is not a recovery. Unknown
    stance is not a closed step.
    """
    drop = level - float(ratio[bottom])
    if drop <= 1e-6:
        return None
    line = level - (1.0 - RECOVER_FRACTION) * drop
    gain = float(stance[bottom] - pre_stance) if np.isfinite(stance[bottom]) else 0.0
    step_left = max(0.02, 0.20 * max(gain, 0.0))
    for i in range(bottom + 1, shot_end + 1):
        if not np.isfinite(ratio[i]):
            return None
        hip_back = float(ratio[i]) >= line
        step_back = np.isfinite(stance[i]) and float(stance[i] - pre_stance) <= step_left
        if hip_back and step_back:
            return int(i)
    return None


def _last_seen(ratio, bottom, shot_end) -> int:
    last = int(bottom)
    for i in range(bottom, shot_end + 1):
        if not np.isfinite(ratio[i]):
            break
        last = int(i)
    return last


def _merge_gap_windows(windows):
    merged = []
    for window in sorted(windows, key=lambda item: (item.reason, item.start_frame, item.end_frame)):
        if (
            merged
            and merged[-1].reason == window.reason
            and window.start_frame <= merged[-1].end_frame + 1
        ):
            last = merged[-1]
            merged[-1] = Window(last.start_frame, max(last.end_frame, window.end_frame), last.reason)
        else:
            merged.append(window)
    return merged


def _lunge_confidence(player, frame) -> float:
    names = ("left_hip", "right_hip", "left_ankle", "right_ankle")
    confs = [
        min(1.0, float(player.conf[frame, JOINT_INDEX[name]]))
        for name in names
        if player.joint_visible(name)[frame]
    ]
    if not confs:
        return 0.4
    return float(np.clip(np.median(confs), 0.0, 1.0))


def _keep_deeper(events, t):
    kept = []
    for event in sorted(events, key=lambda e: -float(e.data["depth"])):
        close = any(abs(float(t[int(event.frame)] - t[int(other.frame)])) < LUNGE_GAP_S for other in kept)
        piled = any(_lunge_overlap(event, other) for other in kept)
        if close or piled:
            continue
        kept.append(event)
    return kept


def _lunge_overlap(a, b) -> bool:
    return int(a.frame) <= int(b.data["end_frame"]) and int(b.frame) <= int(a.data["end_frame"])
