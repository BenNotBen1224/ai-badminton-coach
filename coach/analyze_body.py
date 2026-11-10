"""Technique and contact findings for both players.

Hip height is projected hip height: the image gap from the hips down to the
lowest visible ankle, divided by stature. It is not a measured centre of mass.
Knee, elbow, lean and foot width are projected estimates. Nothing here is
validated against amateur players. The ranges a player sees are practice guides.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .catalog_body import BODY_SPECS
from .events import recovery_windows
from .play_state import exchange_mask
from .schema import (
    FINDING_CATALOG, Event, Finding, Mark, Metric, Moment, Partial, Severity, Tier, Window,
)
from .signals_body import (
    both_hand_speeds,
    clear_ratio_cache,
    elbow_deg,
    foot_speed_ratio,
    hip_height_px,
    hip_height_ratio,
    knee_angle_deg,
    knee_flexion_deg,
    ready_bouts,
    shot_bounds,
    stance_width_ratio,
    standing_hip_px,
    trunk_lean_deg,
    wrist_height_ratio,
)
from .tracks import JOINT_INDEX, Tracks

PRACTICE = "These ranges are practice guides, not validated personal norms."
READY_FOOT_SPEED = 0.8
STANCE_CONFIDENCE_CAP = 0.50
PREVIEW_CONFIDENCE_CAP = 0.55
CONTACT_CONFIDENCE_CAP = 0.55
WRIST_MOVE_FLOOR = 0.20
HIP_MOVE_FLOOR = 0.06
NOT_A_HIT = "The hit itself was not confirmed, so this is not a contact score."
MIN_REVIEW_CYCLE_S = 0.30
_SPECS = {spec.finding_id: spec for spec in BODY_SPECS}


def body_analysis(tracks: Tracks, events: Sequence[Event]) -> Partial:
    clear_ratio_cache()
    metrics: dict[str, tuple] = {}
    series: dict[str, dict] = {}
    findings = []
    windows = []
    swings = [event for event in events if event.kind == "swing"]
    gaps = recovery_windows(tracks)
    for role in sorted(tracks.players, key=lambda item: item.value):
        player = tracks.players[role]
        if int(player.present.sum()) < 3:
            continue
        built = _player(player, tracks, events, gaps)
        metrics[role.value] = built["metrics"]
        series[role.value] = built["series"]
        findings.extend(built["findings"])
        windows.extend(built["windows"])
    warnings = []
    if len(swings) < 3:
        warnings.append("few_swings")
    if gaps:
        warnings.append("unresolved_recovery")
    return Partial(
        metrics, series, (), tuple(findings), tuple(windows), tuple(warnings),
    )


def _player(player, tracks, events, gaps):
    speeds = both_hand_speeds(player, tracks)
    hand = f"{player.hand}_wrist" if player.hand != "unknown" else None
    hip_ratio = hip_height_ratio(player)
    stance = stance_width_ratio(player)
    lean = trunk_lean_deg(player)
    knees = knee_flexion_deg(player)
    unknown = np.full(player.n, np.nan)
    wrist = wrist_height_ratio(player, hand) if hand else unknown
    elbow = elbow_deg(player, hand) if hand else unknown
    named = {
        "hip_height_ratio": hip_ratio,
        "stance_width_ratio": stance,
        "trunk_lean_deg": lean,
        "hand_speed": speeds[hand] if hand else unknown,
        "knee_flexion_deg": knees,
        "wrist_height_ratio": wrist,
        "elbow_deg": elbow,
    }
    ready = _ready_mask(player, tracks, events)
    ready_reps = {
        "hip_height_ratio": _signal_reps(player, tracks, ready, hip_ratio),
        "knee_flexion_deg": _signal_reps(player, tracks, ready, knees),
        "stance_width_ratio": _signal_reps(player, tracks, ready, stance),
        "trunk_lean_deg": _signal_reps(player, tracks, ready, np.abs(lean)),
    }
    mine = [event for event in events if event.player == player.role
            and (event.kind != "swing" or (hand and bool(event.data["hand"]) == (player.hand == "right")))]
    mine_gaps = tuple(window for window in gaps if window.reason.startswith(player.role.value))
    findings = (
        _stay_lower(player, tracks, ready_reps["hip_height_ratio"]),
        _knee_bend(player, tracks, ready_reps["knee_flexion_deg"]),
        _lunge_depth(player, tracks, mine, mine_gaps),
        _stance_width(player, tracks, ready_reps["stance_width_ratio"]),
        _trunk_lean(player, tracks, ready_reps["trunk_lean_deg"]),
        *(() if hand is None else (
            _reach(player, tracks, mine, wrist, hand),
            _arm(player, tracks, mine, elbow, hand),
            _hand_speed(player, tracks, mine, hand),
        )),
    )
    return {
        "metrics": tuple(finding.metrics[0] for finding in findings),
        "series": named,
        "findings": findings,
        "windows": (
            *(_preview_windows(player, tracks, hand, hip_height_px(player)) if hand else ()),
            *mine_gaps,
        ),
    }


def _ready_mask(player, tracks, events):
    """Quiet frames supported by an observed exchange.

    A still player repeats the same pose, so measured is true on only the first
    frame of that pose. Those copies are still the ready position. Foot speed is
    unknown there; a fast measured foot is what takes a frame out of the wait.
    Standing around before the rally, or a pickup after it, is not that wait.
    No exchange means no ready-stance measurement, not a static-pose fallback.
    """
    blocked = np.zeros(player.n, dtype=bool)
    for event in events:
        if event.player != player.role:
            continue
        if event.kind == "swing":
            start, end = int(event.data["start"]), int(event.data["end"])
        elif event.kind == "lunge":
            start, end = int(event.frame), int(event.data["end_frame"])
        else:
            continue
        blocked[max(0, start):min(player.n, end + 1)] = True
    foot = foot_speed_ratio(player, tracks)
    quiet = ~np.isfinite(foot) | (foot < READY_FOOT_SPEED)
    ready = player.present & quiet & ~blocked
    near = _near_play(tracks, events)
    return ready & near


def _near_play(tracks, events):
    return exchange_mask(tracks, events)


def _signal_reps(player, tracks, mask, signal):
    usable = mask & np.isfinite(signal)
    reps = []
    for start, end in ready_bouts(usable, tracks.shots, tracks.t):
        frames = np.flatnonzero(usable[start:end + 1]) + start
        if frames.size == 0:
            continue
        value = float(np.median(signal[frames]))
        frame = _visible_frame(player, frames, signal, value)
        reps.append(_Rep(value, frame, int(start), int(end), frames=frames))
    return reps


def _visible_frame(player, frames, signal, value) -> int:
    measured = frames[player.measured[frames]]
    pool = measured if measured.size else frames
    order = np.argsort(np.abs(signal[pool] - value))
    return int(pool[order[0]])


class _Rep:
    def __init__(self, value, frame, start=0, end=0, hand="", frames=None):
        self.value = float(value)
        self.frame = int(frame)
        self.start = int(start)
        self.end = int(end)
        self.hand = hand
        self.frames = tuple(int(item) for item in (frames if frames is not None else (frame,)))


def _event_reps(events, accept, value_at, hand_of):
    reps = []
    for event in events:
        if event.kind != "swing" or not accept(event):
            continue
        frame = int(value_at(event)[0])
        value = value_at(event)[1]
        if value is None or not np.isfinite(value):
            continue
        reps.append(_Rep(value, frame, int(event.data["start"]), int(event.data["end"]), hand_of(event)))
    return reps


def _hand_of(event) -> str:
    return "right_wrist" if event.data["hand"] else "left_wrist"


def _stay_lower(player, tracks, reps):
    spec = _SPECS["technique.stay_lower"]
    value = _value(spec, reps)
    if not reps:
        headline = "Not enough upright time to judge hip height"
        explain = "There were not enough upright frames to read projected hip height, so this is only a note."
    elif value > spec.target[1]:
        headline = "Your hips stay high while you wait"
        explain = (
            f"Your projected hip height is {value:.0%} of your height while you wait, "
            "above the practice range. Sitting a little lower may help you move."
        )
    else:
        headline = "Your hips sit in a ready height"
        explain = (
            f"Your projected hip height is {value:.0%} of your height while you wait, "
            "inside the practice range."
        )
    return _finding(
        player, tracks, spec, reps, value, headline, explain,
        ("left_hip", "right_hip", "left_ankle", "right_ankle"),
        "hip_height_ratio", "lower_stance", _stay_marks,
    )


def _knee_bend(player, tracks, reps):
    spec = _SPECS["technique.knee_bend"]
    value = _value(spec, reps)
    if not reps:
        headline = "Not enough ready time to judge knee bend"
        explain = "There was not enough ready time to read the knees. This angle would be a projected estimate."
    elif value < spec.target[0]:
        headline = "Your knees stay too straight while you wait"
        explain = (
            f"Your ready knees bend about {value:.0f} degrees. A small bend may help you push off. "
            "This angle is projected."
        )
    else:
        headline = "Your knees stay softly bent while you wait"
        explain = (
            f"Your ready knees bend about {value:.0f} degrees, inside the practice range. "
            "This angle is projected."
        )
    return _finding(
        player, tracks, spec, reps, value, headline, explain,
        ("left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"),
        "knee_flexion_deg", "lower_stance", _knee_marks,
    )


def _lunge_depth(player, tracks, events, gaps=()):
    spec = _SPECS["technique.lunge_depth"]
    reps = []
    for event in events:
        if event.kind != "lunge":
            continue
        if tracks.t[int(event.data["end_frame"])] - tracks.t[int(event.frame)] < MIN_REVIEW_CYCLE_S:
            continue
        reps.append(_Rep(float(event.data["depth"]), int(event.data["bottom_frame"]), int(event.frame), int(event.data["end_frame"])))
    value = _value(spec, reps)
    left_out = " A drop that did not return to the ready hips was left out." if gaps else ""
    if not reps:
        headline = "No lunge was clear enough to measure"
        explain = "No lunge was clear enough to read. Depth would be the projected hip height lost in the step." + left_out
    elif value < spec.target[0]:
        headline = "Your lunges do not drop the hips much"
        explain = (
            f"Projected hip height drops about {value:.0%} in the lunge, less than the practice range. "
            "A deeper step may help you reach wide shots." + left_out
        )
    else:
        headline = "Your lunges drop the hips into the shot"
        explain = f"Projected hip height drops about {value:.0%} in the lunge, inside the practice range." + left_out
    return _finding(
        player, tracks, spec, reps, value, headline, explain,
        ("left_hip", "right_hip", "left_ankle", "right_ankle"),
        "hip_height_ratio", "lunge_depth", _lunge_marks,
    )


def _stance_width(player, tracks, reps):
    spec = _SPECS["technique.stance_width"]
    value = _value(spec, reps)
    low, high = spec.target
    if not reps:
        headline = "Not enough ready time to judge foot width"
        explain = "There was not enough ready time to read foot width. Width from this camera is a projected estimate."
    elif value < low:
        headline = "Your ready feet look narrow in this view"
        explain = (
            f"Ready feet look about {value:.0%} of your height apart, under the practice range. "
            "Width from this camera is only a projected estimate."
        )
    elif value > high:
        headline = "Your ready feet look very wide in this view"
        explain = (
            f"Ready feet look about {value:.0%} of your height apart, over the practice range. "
            "Width from this camera is only a projected estimate."
        )
    else:
        headline = "Your ready foot width is in the practice range"
        explain = (
            f"Ready feet look about {value:.0%} of your height apart, inside the practice range. "
            "Width from this camera is only a projected estimate."
        )
    return _finding(
        player, tracks, spec, reps, value, headline, explain,
        ("left_ankle", "right_ankle"), "stance_width_ratio", "balance", _stance_marks,
        confidence_cap=STANCE_CONFIDENCE_CAP,
    )


def _trunk_lean(player, tracks, reps):
    spec = _SPECS["technique.trunk_lean"]
    value = _value(spec, reps)
    if not reps:
        headline = "Not enough ready time to judge trunk lean"
        explain = "There was not enough ready time to read sideways lean. This angle is projected, not a 3D measure."
    elif value > spec.target[1]:
        headline = "Your trunk leans sideways while you wait"
        explain = (
            f"Your trunk leans about {value:.0f} degrees sideways while you wait. "
            "A taller trunk may help balance. This angle is projected, not a 3D measure."
        )
    else:
        headline = "Your trunk stays upright while you wait"
        explain = (
            f"Your trunk leans about {value:.0f} degrees sideways, inside the practice range. "
            "This angle is projected, not a 3D measure."
        )
    return _finding(
        player, tracks, spec, reps, value, headline, explain,
        ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
        "trunk_lean_deg", "core_stability", _lean_marks,
    )


def _reach(player, tracks, events, wrist, hand):
    spec = _SPECS["contact.reach_height"]
    reps = _event_reps(
        events,
        lambda event: event.data["overhead"] == 1,
        lambda event: (int(event.frame), _at(wrist, int(event.frame))),
        _hand_of,
    )
    value = _value(spec, reps)
    if not reps:
        headline = "No overhead shot was found to judge reach"
        explain = "No overhead shot was clear enough to read reach. Reach would be projected hand height, not a 3D measure. " + NOT_A_HIT
    elif value < spec.target[0]:
        headline = "Overhead wrist height is under the practice range"
        explain = (
            f"At the estimated overhead frame the wrist is about {value:.2f} times your height. "
            f"{NOT_A_HIT} This is projected height, not a 3D reach."
        )
    else:
        headline = "Overhead wrist height is inside the practice range"
        explain = (
            f"At the estimated overhead frame the wrist is about {value:.2f} times your height, "
            f"inside the practice range. {NOT_A_HIT} This is projected height, not a 3D reach."
        )
    return _finding(
        player, tracks, spec, reps, value, headline, explain,
        _arm_joints(hand), "wrist_height_ratio", "contact_height",
        lambda p, frame, got, _rep: _reach_marks(p, frame, got, hand),
        CONTACT_CONFIDENCE_CAP,
    )


def _arm(player, tracks, events, elbow, hand):
    spec = _SPECS["contact.arm_extension"]
    reps = _event_reps(
        events,
        lambda event: event.data["overhead"] == 1,
        lambda event: (int(event.frame), _at(elbow, int(event.frame))),
        _hand_of,
    )
    value = _value(spec, reps)
    if not reps:
        headline = "No overhead shot was found to judge the arm"
        explain = "No overhead shot was clear enough to read the elbow. This angle would be projected. " + NOT_A_HIT
    elif value < spec.target[0]:
        headline = "The elbow at the estimated frame stays bent"
        explain = (
            f"The elbow is about {value:.0f} degrees at the estimated overhead frame. "
            f"{NOT_A_HIT} This angle is projected."
        )
    else:
        headline = "The elbow at the estimated frame is long"
        explain = (
            f"The elbow is about {value:.0f} degrees at the estimated overhead frame, inside the practice range. "
            f"{NOT_A_HIT} This angle is projected."
        )
    return _finding(
        player, tracks, spec, reps, value, headline, explain,
        _arm_joints(hand), "elbow_deg", "arm_extension",
        lambda p, frame, got, rep: _elbow_marks(p, frame, got, rep.hand or hand),
        CONTACT_CONFIDENCE_CAP,
    )


def _hand_speed(player, tracks, events, hand):
    spec = _SPECS["contact.hand_speed"]
    reps = _event_reps(
        events,
        lambda event: True,
        lambda event: (int(event.data["peak_frame"]), float(event.data["peak_speed"])),
        _hand_of,
    )
    value = _value(spec, reps)
    if not reps:
        headline = "No swing was found to judge hand speed"
        explain = "No swing was clear enough to read racket-hand speed. " + NOT_A_HIT
    elif value < spec.target[0]:
        headline = "Arm motion is under the practice speed"
        explain = (
            f"The fastest arm motion is about {value:.1f} times your height per second, under the practice range. "
            + NOT_A_HIT
        )
    else:
        headline = "Arm motion is inside the practice speed"
        explain = (
            f"The fastest arm motion is about {value:.1f} times your height per second, inside the practice range. "
            + NOT_A_HIT
        )
    return _finding(
        player, tracks, spec, reps, value, headline, explain,
        (hand,), "hand_speed", "core_rotation",
        lambda p, frame, got, rep: _speed_marks(p, tracks, frame, got, rep.hand or hand),
        CONTACT_CONFIDENCE_CAP,
    )


def _finding(player, tracks, spec, reps, value, headline, explain, focus, overlay, fix, marks_for, confidence_cap=1.0):
    if not reps or value is None:
        return Finding(
            spec.finding_id, FINDING_CATALOG[spec.finding_id][0], player.role,
            Severity.INFO, headline, _explain(explain),
            (_metric(spec, None, 0.0, overlay, 0, None),),
            (), _evidence_tier(player, tracks, ()), focus, (overlay,), (), 0.0,
        )
    evidence = _evidence_tier(player, tracks, reps)
    conf = _confidence(player, reps, focus) * confidence_cap
    if evidence == Tier.PREVIEW:
        conf = min(conf, PREVIEW_CONFIDENCE_CAP)
    conf = float(np.clip(conf, 0.0, 1.0))
    n_obs = len(reps)
    # Pose confidence validates visibility, not a personal technique norm.
    headline = f"{spec.label}: an observed 2D estimate"
    explain = (
        f"Observed {spec.label.lower()}: {value:.3g} {spec.unit} across {n_obs} samples. "
        "This does not establish good or bad technique or a need to change posture."
    )
    if spec.finding_id.startswith("contact."):
        explain += " " + NOT_A_HIT
    representative = min(reps, key=lambda rep: (abs(rep.value - value), rep.frame))
    marks = tuple(
        mark for mark in marks_for(player, representative.frame, representative.value, representative)
        if mark.style != "target"
    )
    moments = (_moment(player, tracks, representative, "issue", marks, headline),)
    return Finding(
        spec.finding_id, FINDING_CATALOG[spec.finding_id][0], player.role,
        Severity.INFO, headline, _explain(explain),
        (_metric(spec, value, conf, overlay, n_obs, None),),
        moments, evidence, focus, (overlay,), (), 0.0,
    )


def _evidence_tier(player, tracks, reps):
    """Refined only when every frame that entered the value was sharpened.

    A listed window means those rows were replaced. A later identical frame is
    a repeat of that row, not a preview guess, so presence counts. The document
    tier is left as loaded.
    """
    if tracks.tier == Tier.REFINED:
        return Tier.REFINED
    frames = [frame for rep in reps for frame in rep.frames]
    if frames and _frames_in_refined_windows(player, frames, tracks.refined_windows):
        return Tier.REFINED
    return Tier.PREVIEW


def _frames_in_refined_windows(player, frames, spans) -> bool:
    if not spans:
        return False
    for frame in frames:
        if frame < 0 or frame >= player.n or not player.present[frame]:
            return False
        if not any(start <= frame <= end for start, end in spans):
            return False
    return True


def _value(spec, reps):
    if not reps:
        return None
    return float(np.median([rep.value for rep in reps]))


def _metric(spec, value, conf, overlay, n_obs, best):
    return Metric(
        spec.name, spec.label, value, spec.unit, spec.basis, conf, spec.target, overlay, n_obs, best,
    )


def _explain(body: str) -> str:
    text = f"{body} {PRACTICE}"
    if len(text) > 400:
        raise ValueError(f"explanation is {len(text)} characters")
    return text


def _confidence(player, reps, names) -> float:
    if not reps:
        return 0.0
    vals = []
    for rep in reps:
        for name in names:
            if name not in JOINT_INDEX or not player.joint_visible(name)[rep.frame]:
                continue
            vals.append(min(1.0, float(player.conf[rep.frame, JOINT_INDEX[name]])))
    if not vals:
        return 0.0
    return float(np.median(vals))


def _moment(player, tracks, rep, kind, marks, label):
    start, end = _bounds(tracks, rep.frame)
    kept = tuple(mark for mark in marks if start <= mark.frame <= end)
    text = label if len(label) <= 80 else label[:80]
    return Moment(player.role, start, rep.frame, end, text or "Ready position", kind, kept)


def _bounds(tracks, key):
    shot_a, shot_b = shot_bounds(key, tracks.shots)
    start = key
    while start > shot_a and float(tracks.t[key] - tracks.t[start - 1]) <= 1.0:
        start -= 1
    end = key
    while end < shot_b and float(tracks.t[end + 1] - tracks.t[key]) <= 0.50:
        end += 1
    return start, end


def _at(signal, frame):
    if frame < 0 or frame >= signal.shape[0] or not np.isfinite(signal[frame]):
        return None
    return float(signal[frame])


def _arm_joints(hand):
    side = "right" if hand.startswith("right") else "left"
    return (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")


def _xy(player, name, frame):
    if name not in JOINT_INDEX or not player.joint_visible(name)[frame]:
        return None
    x, y = player.joint(name)[frame]
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return float(x), float(y)


def _mid(player, left, right, frame):
    a = _xy(player, left, frame)
    b = _xy(player, right, frame)
    if a and b:
        return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    return a or b


def _span(points, pad=12.0):
    xs = [point[0] for point in points]
    lo, hi = min(xs), max(xs)
    if hi - lo < 8:
        mid = (lo + hi) / 2
        lo, hi = mid - pad, mid + pad
    return lo, hi


def _stay_marks(player, frame, value, _rep):
    marks = []
    hips = [point for point in (_xy(player, "left_hip", frame), _xy(player, "right_hip", frame)) if point]
    ankles = [point for point in (_xy(player, "left_ankle", frame), _xy(player, "right_ankle", frame)) if point]
    if len(hips) == 2:
        marks.append(Mark("line", frame, hips, f"hips {value:.0%} of height", value, "actual"))
    if len(ankles) == 2:
        marks.append(Mark("line", frame, ankles, "ankles", None, "neutral"))
    stature = float(player.stature_px[frame])
    if hips and ankles and np.isfinite(stature) and stature > 1:
        ankle_y = max(point[1] for point in ankles)
        lo, hi = _span(hips)
        top = ankle_y - 0.50 * stature
        bottom = ankle_y - 0.40 * stature
        marks.append(Mark("band", frame, ((lo, top), (hi, bottom)), "practice hip band", None, "target"))
    return tuple(marks)


def _knee_marks(player, frame, value, _rep):
    side = _straighter_knee(player, frame)
    if side is None:
        return ()
    points = [_xy(player, f"{side}_{name}", frame) for name in ("hip", "knee", "ankle")]
    if any(point is None for point in points):
        return ()
    return (Mark("angle", frame, points, f"projected {value:.0f} deg", value, "actual"),)


def _straighter_knee(player, frame):
    best = None
    for side in ("left", "right"):
        angle = knee_angle_deg(player, side)[frame]
        if np.isfinite(angle) and (best is None or angle > best[0]):
            best = (float(angle), side)
    return None if best is None else best[1]


def _lunge_marks(player, frame, value, _rep):
    marks = []
    hips = [point for point in (_xy(player, "left_hip", frame), _xy(player, "right_hip", frame)) if point]
    ankles = [point for point in (_xy(player, "left_ankle", frame), _xy(player, "right_ankle", frame)) if point]
    if len(hips) == 2:
        marks.append(Mark("line", frame, hips, "hips at the bottom", value, "actual"))
    stand = standing_hip_px(player, frame)
    if hips and ankles and np.isfinite(stand) and stand > 1:
        ankle_y = max(point[1] for point in ankles)
        lo, hi = _span(hips)
        standing_y = ankle_y - stand
        hip_y = sum(point[1] for point in hips) / len(hips)
        hip_x = sum(point[0] for point in hips) / len(hips)
        marks.append(Mark("line", frame, ((lo, standing_y), (hi, standing_y)), "standing hip line", None, "neutral"))
        marks.append(Mark("bracket", frame, ((hip_x, standing_y), (hip_x, hip_y)), f"drop {value:.0%}", value, "actual"))
    return tuple(marks)


def _stance_marks(player, frame, value, _rep):
    left = _xy(player, "left_ankle", frame)
    right = _xy(player, "right_ankle", frame)
    if left is None or right is None:
        return ()
    marks = [Mark("bracket", frame, (left, right), f"feet {value:.2f} heights", value, "actual")]
    stature = float(player.stature_px[frame])
    if np.isfinite(stature) and stature > 1:
        mid = (left[0] + right[0]) / 2
        y = max(left[1], right[1])
        half = 0.40 * stature
        marks.append(Mark("band", frame, ((mid - half, y - 8), (mid + half, y + 8)), "practice width", None, "target"))
    return tuple(marks)


def _lean_marks(player, frame, value, _rep):
    hip = _mid(player, "left_hip", "right_hip", frame)
    shoulder = _mid(player, "left_shoulder", "right_shoulder", frame)
    if hip is None or shoulder is None:
        return ()
    top = (hip[0], min(hip[1], shoulder[1]) - 10)
    bottom = (hip[0], hip[1] + 15)
    return (
        Mark("line", frame, (hip, shoulder), f"lean {value:.0f} deg", value, "actual"),
        Mark("line", frame, (top, bottom), "vertical", None, "neutral"),
    )


def _reach_marks(player, frame, value, hand):
    wrist = _xy(player, hand, frame)
    if wrist is None:
        return ()
    marks = [Mark("line", frame, ((wrist[0] - 24, wrist[1]), (wrist[0] + 24, wrist[1])), f"reach {value:.2f} heights", value, "actual")]
    ankles = [point for point in (_xy(player, "left_ankle", frame), _xy(player, "right_ankle", frame)) if point]
    stature = float(player.stature_px[frame])
    if ankles and np.isfinite(stature) and stature > 1:
        ankle_y = max(point[1] for point in ankles)
        lo, hi = _span([wrist, *ankles])
        top = ankle_y - 1.40 * stature
        bottom = ankle_y - 1.00 * stature
        marks.append(Mark("band", frame, ((lo, top), (hi, bottom)), "practice reach", None, "target"))
    return tuple(marks)


def _elbow_marks(player, frame, value, hand):
    side = "right" if hand.startswith("right") else "left"
    points = [_xy(player, f"{side}_{name}", frame) for name in ("shoulder", "elbow", "wrist")]
    if any(point is None for point in points):
        return ()
    return (Mark("angle", frame, points, f"projected {value:.0f} deg", value, "actual"),)


def _speed_marks(player, tracks, frame, value, hand):
    start, _end = _bounds(tracks, frame)
    points = []
    for index in range(max(start, frame - 8), frame + 1):
        point = _xy(player, hand, index)
        if point is not None:
            points.append(point)
    label = f"{value:.1f} heights/s"
    if len(points) >= 2:
        return (Mark("trail", frame, points, label, value, "actual"),)
    if points:
        return (Mark("dot", frame, (points[-1],), label, value, "actual"),)
    return ()


def _preview_windows(player, tracks, hand, hip):
    """Ask for a denser pose where a preview gap leaves the move unsettled.

    A regular sample interval is still unmeasured. Request it when the wrist
    moved at least WRIST_MOVE_FLOOR stature or the hip moved at least
    HIP_MOVE_FLOOR. A gap alone does not. An emitted swing is not a request:
    denser pose cannot confirm the hit.
    """
    if tracks.tier != Tier.PREVIEW:
        return []
    measured = np.flatnonzero(player.measured)
    if measured.size < 2:
        return []
    wrist = player.joint(hand)
    visible = player.joint_visible(hand)
    found = []
    for left, right in zip(measured[:-1], measured[1:]):
        left = int(left)
        right = int(right)
        if right - left < 3:
            continue
        shot_start, shot_end = shot_bounds(left, tracks.shots)
        if right > shot_end or right < shot_start:
            continue
        stature = float(player.stature_px[left])
        if not np.isfinite(stature) or stature <= 1:
            continue
        moved = False
        if visible[left] and visible[right] and np.isfinite(wrist[left]).all() and np.isfinite(wrist[right]).all():
            moved = float(np.linalg.norm(wrist[right] - wrist[left])) / stature >= WRIST_MOVE_FLOOR
        if np.isfinite(hip[left]) and np.isfinite(hip[right]):
            moved = moved or abs(float(hip[right] - hip[left])) / stature >= HIP_MOVE_FLOOR
        if moved:
            found.append(Window(left, right, "unresolved body move"))
    return found
