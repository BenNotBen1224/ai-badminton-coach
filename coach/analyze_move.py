"""Footwork and positional tactics.

A step is a local peak in smoothed foot speed of at least 1.7 m/s. The raw
speed at that frame must be finite, so a repeated video frame cannot become a
step. Peaks closer than 0.18 s are one step.

The shuttle is not tracked. A hop's onset and landing are reported as what we
measured, not as a timing to aim for. Coverage, distance from a centre spot,
rally gaps, who stands nearer the net, and a late first step stay at info.
Those clocks start at an arm swing we marked, not a confirmed hit. Recovery
after a measured lunge can become advice. A return timed from an arm swing
stays info, and a stop does not prove a hit. A look we could not judge still
asks for a closer pose. Expected sample spacing is not a reason to drop that
ask. Denser pose cannot confirm the arm swing, so the contact stays
unconfirmed either way.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .catalog_move import MOVE_SPECS
from .schema import (
    FINDING_CATALOG, SEVERITY_RANK, Event, Finding, Mark, Metric, Moment,
    Partial, Role, Severity, Tier, Window,
)
from .severity import score_for, severity_for
from .signals_move import (
    SPLIT_AFTER_S, SPLIT_BEFORE_S, ZONE_M, cell_centre, confidence, coverage_cells,
    distances, fine_motion_reliable, first_return, fixed_base, foot_speed,
    habitual_base, hop_and_dip, hop_onset, measured_gap, median, pair_splits,
    rally_bounds, split_landings, step_peaks, waiting_base,
)
from .tracks import Tracks

SPECS = {spec.name: spec for spec in MOVE_SPECS}
DESCRIBE = {
    "footwork.split_step",
    "footwork.first_step",
    "footwork.coverage",
    "tactics.base_discipline",
    "tactics.tempo",
    "tactics.initiative",
}
DESCRIBE_CONF = 0.55
FEW_SWINGS = 3
PARTIAL_SHARE = 0.80
RECOVERY_LOOK_S = 3.0
FIRST_LOOK_S = 0.80
STEPS_BEFORE_S = 0.80
SMALL_STEP_MAX = 3.2
TRIGGER_MERGE_S = 0.40
CLUB = "Not checked on club players."
SHUTTLE = "The shuttle is not tracked."
GUIDE = "These numbers are practice guides, not validated personal norms."


def movement_analysis(tracks: Tracks, events: Sequence[Event]) -> Partial:
    """Court position and vertical foot signals to a Partial. Never reads the shuttle."""
    warnings: list[str] = []
    windows: list[Window] = []
    if tracks.court is None:
        warnings.append("no_court")
    findings: list[Finding] = []
    extra: list[Event] = []
    series: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, list[Metric]] = {}
    for role, track in tracks.players.items():
        if not np.asarray(track.present).any():
            continue
        pack = _player(tracks, track, role, events, windows)
        findings.extend(pack["findings"])
        extra.extend(pack["events"])
        series[role.value] = pack["series"]
        metrics[role.value] = [m for finding in pack["findings"] for m in finding.metrics]
        if pack["partial"]:
            warnings.append("partial_track")
        if pack["few"]:
            warnings.append("few_swings")
    return Partial(
        {role: tuple(found) for role, found in metrics.items()},
        series,
        tuple(extra),
        tuple(findings),
        tuple(windows),
        tuple(dict.fromkeys(warnings)),
    )


def _player(tracks: Tracks, track, role: Role, events, windows: list[Window]) -> dict:
    court = tracks.court
    foot = track.foot_court if court is not None else np.full((track.n, 2), np.nan)
    raw, smooth = foot_speed(foot, tracks.t, track.measured, tracks.shots)
    hop, dip = hop_and_dip(track, court, tracks.shots)
    own_swings = _kind(events, "swing", role)
    own_lunges = _kind(events, "lunge", role)
    opp = _other(role)
    opp_swings = _kind(events, "swing", opp)
    opp_track = tracks.players.get(opp)
    contacts = [int(e.frame) for e in opp_swings if 0 <= int(e.frame) < track.n]
    base, base_name = _choose_base(foot, smooth, role, court, tracks.t, contacts)
    fixed = fixed_base(role, court) if court is not None else None
    dist = distances(foot, base) if base is not None else np.full(track.n, np.nan)
    landings = split_landings(hop, dip, tracks.shots)
    peaks = step_peaks(raw, smooth, tracks.t, tracks.shots)
    fine_ok = fine_motion_reliable(track) if court is not None else False
    preview = tracks.tier == Tier.PREVIEW
    pose = _pose(track, contacts or [0])
    ctx = {
        "tracks": tracks, "track": track, "role": role, "court": court, "foot": foot,
        "raw": raw, "smooth": smooth, "hop": hop, "dip": dip, "dist": dist,
        "base": base, "base_name": base_name, "fixed": fixed, "landings": landings,
        "peaks": peaks, "contacts": contacts, "own_swings": own_swings,
        "own_lunges": own_lunges, "opp_track": opp_track, "fine_ok": fine_ok,
        "preview": preview, "pose": pose,
        "t": tracks.t,
    }
    made: list[Event] = []
    findings: list[Finding] = []
    made.extend(_step_events(ctx))
    split_finding, split_events = _split(ctx, windows)
    made.extend(split_events)
    if split_finding is not None:
        findings.append(split_finding)
    returns = _returns(ctx)
    made.extend(returns["events"])
    for builder in (_recovery, _first_step, _step_pattern, _coverage, _base_discipline, _tempo, _initiative):
        finding = builder(ctx, returns, windows)
        if finding is not None:
            findings.append(finding)
    return {
        "findings": findings,
        "events": made,
        "series": _series(ctx),
        "partial": _is_partial(track, tracks.shots),
        "few": len(contacts) < FEW_SWINGS,
    }


def _series(ctx) -> dict[str, np.ndarray]:
    foot = ctx["foot"]
    return {
        "court_x": foot[:, 0].astype(np.float64),
        "court_y": foot[:, 1].astype(np.float64),
        "foot_speed": ctx["smooth"].astype(np.float64),
        "base_distance": ctx["dist"].astype(np.float64),
        "hop_height_ratio": ctx["hop"].astype(np.float64),
        "hip_dip_ratio": ctx["dip"].astype(np.float64),
    }


def _step_events(ctx) -> list[Event]:
    out = []
    smooth = ctx["smooth"]
    conf = 0.70 if ctx["fine_ok"] else 0.40
    for frame in ctx["peaks"]:
        speed = float(smooth[frame])
        if not np.isfinite(speed):
            continue
        out.append(Event("step", ctx["role"], int(frame), conf, {"speed_m_s": speed}))
    return out


def _split(ctx, windows: list[Window]):
    t = ctx["t"]
    track = ctx["track"]
    judgeable = []
    for contact in ctx["contacts"]:
        ok, start, end = _signal_window(ctx, contact, SPLIT_BEFORE_S, SPLIT_AFTER_S)
        if not ok:
            windows.append(_window(start, end, "split_step unjudgeable"))
            continue
        judgeable.append(contact)
    pairs = pair_splits(ctx["landings"], judgeable, t)
    by_contact = {contact: landing for landing, contact in pairs}
    events = []
    onsets = []
    land_times = []
    reps = []
    for contact in judgeable:
        landing = by_contact.get(contact)
        if landing is None:
            continue
        shot_start, _ = _shot_of(ctx["tracks"].shots, landing)
        onset = hop_onset(ctx["hop"], landing, shot_start)
        onset_s = float(t[onset] - t[contact])
        landing_s = float(t[landing] - t[contact])
        piece = ctx["hop"][onset:landing + 1]
        hop_h = float(np.nanmax(piece)) if np.isfinite(piece).any() else 0.0
        conf = 0.70 if ctx["fine_ok"] else 0.40
        events.append(Event("split_step", ctx["role"], int(landing), conf, {
            "onset_s": onset_s, "landing_s": landing_s, "hop_height": hop_h,
            "onset_frame": float(onset), "landing_frame": float(landing),
        }))
        onsets.append(onset_s)
        land_times.append(landing_s)
        reps.append({"contact": contact, "onset": onset, "landing": landing, "onset_s": onset_s, "landing_s": landing_s})
    if not judgeable:
        return None, events
    share = len(reps) / len(judgeable)
    conf = _conf(ctx, len(judgeable), fine=True)
    metrics = [_metric("split_step_share", share, conf, len(judgeable), series="hop_height_ratio")]
    if land_times:
        metrics.append(_metric(
            "split_step_timing_s", median(land_times), _conf(ctx, len(land_times), fine=True),
            len(land_times), best=_closest_to_zero(land_times), series="hop_height_ratio",
        ))
    finding = _finding(
        "footwork.split_step", ctx, metrics,
        *_split_text(len(reps), len(judgeable), onsets, land_times, not ctx["fine_ok"]),
        _split_moments(ctx, reps, judgeable),
        ("left_ankle", "right_ankle", "left_hip", "right_hip"),
        ("hop_height_ratio",),
        (),
    )
    return finding, events


def _returns(ctx) -> dict:
    t = ctx["t"]
    triggers = _triggers(ctx["own_swings"], ctx["own_lunges"], t)
    events = []
    completed = []
    for trigger, kind in triggers:
        if ctx["base"] is None:
            break
        start_shot, shot_end = _shot_of(ctx["tracks"].shots, trigger)
        stop = min(shot_end, trigger + _ahead(t, trigger, RECOVERY_LOOK_S))
        nxt = [f for f, _ in triggers if f > trigger]
        if nxt:
            stop = min(stop, nxt[0] - 1)
        if stop <= trigger:
            continue
        if not np.isfinite(ctx["dist"][trigger]) or ctx["dist"][trigger] <= ZONE_M:
            continue
        if _pose_gap(ctx["track"], t, trigger, stop):
            continue
        found = first_return(ctx["dist"], ctx["raw"], t, trigger, stop, ctx["track"].measured)
        if found is None:
            continue
        since = float(t[found] - t[trigger])
        frame = int(found)
        distance = float(ctx["dist"][found])
        events.append(Event("base_return", ctx["role"], frame, 0.75, {
            "since_s": since, "distance_m": distance, "from_frame": float(trigger),
        }))
        completed.append({
            "trigger": trigger, "frame": frame, "since": since, "distance": distance, "kind": kind,
        })
    return {"events": events, "completed": completed}


def _recovery(ctx, returns, windows):
    reps = returns["completed"]
    if not reps or ctx["court"] is None:
        return None
    movement = [r for r in reps if r["kind"] == "lunge"]
    scored = movement or [r for r in reps if r["kind"] != "lunge"]
    shot_anchored = not movement
    values = [r["since"] for r in scored]
    conf = _conf(ctx, len(values), fine=False)
    if shot_anchored:
        conf = min(conf, DESCRIBE_CONF)
    best = min(values)
    metrics = [_metric("recovery_s", median(values), conf, len(values), best=best, series="base_distance")]
    worst = max(scored, key=lambda r: r["since"])
    best_rep = min(scored, key=lambda r: r["since"])
    moments = [_recovery_moment(ctx, worst, "issue")]
    if len(scored) >= 3 and best_rep["frame"] != worst["frame"]:
        moments.append(_recovery_moment(ctx, best_rep, "best"))
    spot = "the spot you usually wait" if ctx["base_name"] == "usual" else "the spot you stood still in"
    return _finding(
        "footwork.recovery", ctx, metrics,
        *_recovery_text(median(values), best, spot, shot_anchored, bool(movement and len(scored) != len(reps))),
        moments, ("left_ankle", "right_ankle"), ("base_distance", "foot_speed"), ("base_recovery",),
    )


def _first_step(ctx, returns, windows):
    if ctx["court"] is None or not ctx["contacts"]:
        return None
    t = ctx["t"]
    peaks = ctx["peaks"]
    delays = []
    reps = []
    missed = 0
    for contact in ctx["contacts"]:
        ok, start, end = _signal_window(ctx, contact, 0.0, FIRST_LOOK_S)
        if not ok:
            windows.append(_window(start, end, "first_step unjudgeable"))
            continue
        later = [int(p) for p in peaks if t[contact] < t[p] <= t[contact] + FIRST_LOOK_S and _same_shot(ctx["tracks"].shots, contact, int(p))]
        if not later:
            missed += 1
            continue
        frame = later[0]
        delay = float(t[frame] - t[contact])
        delays.append(delay)
        reps.append({"contact": contact, "frame": frame, "delay": delay})
    if not delays:
        return None
    conf = _conf(ctx, len(delays), fine=True)
    metrics = [_metric("first_step_s", median(delays), conf, len(delays), best=min(delays), series="foot_speed")]
    worst = max(reps, key=lambda r: r["delay"])
    best_rep = min(reps, key=lambda r: r["delay"])
    moments = [_first_moment(ctx, worst, "issue")]
    if len(reps) >= 3 and best_rep["frame"] != worst["frame"]:
        moments.append(_first_moment(ctx, best_rep, "best"))
    return _finding(
        "footwork.first_step", ctx, metrics,
        *_first_text(median(delays), min(delays), missed, not ctx["fine_ok"]),
        moments, ("left_ankle", "right_ankle"), ("foot_speed",), ("first_step",),
    )


def _step_pattern(ctx, returns, windows):
    if ctx["court"] is None or not ctx["own_lunges"]:
        return None
    t = ctx["t"]
    counts = []
    launches = []
    reps = []
    for lunge in ctx["own_lunges"]:
        frame = int(lunge.frame)
        if frame < 0 or frame >= ctx["track"].n:
            continue
        start = _back(t, frame, STEPS_BEFORE_S)
        shot_start, _ = _shot_of(ctx["tracks"].shots, frame)
        start = max(start, shot_start)
        plants = [
            int(p) for p in ctx["peaks"]
            if start <= int(p) < frame and np.isfinite(ctx["smooth"][int(p)]) and ctx["smooth"][int(p)] <= SMALL_STEP_MAX
        ]
        launch = float(ctx["dist"][frame]) if np.isfinite(ctx["dist"][frame]) else float("nan")
        counts.append(float(len(plants)))
        if np.isfinite(launch):
            launches.append(launch)
        reps.append({"frame": frame, "plants": plants, "count": len(plants), "launch": launch})
    if not counts:
        return None
    metrics = [_metric("steps_before_lunge", median(counts), _conf(ctx, len(counts), fine=False), len(counts), best=min(counts), series="foot_speed")]
    if launches:
        metrics.append(_metric("lunge_launch_m", median(launches), _conf(ctx, len(launches), fine=False), len(launches), best=min(launches), series="base_distance"))
    worst = max(reps, key=lambda r: (r["count"], r["launch"] if np.isfinite(r["launch"]) else -1))
    best_rep = min(reps, key=lambda r: (r["count"], r["launch"] if np.isfinite(r["launch"]) else 99))
    moments = [_pattern_moment(ctx, worst, "issue")]
    if len(reps) >= 3 and best_rep["frame"] != worst["frame"]:
        moments.append(_pattern_moment(ctx, best_rep, "best"))
    return _finding(
        "footwork.step_pattern", ctx, metrics,
        *_pattern_text(median(counts), median(launches) if launches else None),
        moments, ("left_ankle", "right_ankle", "left_knee", "right_knee"),
        ("foot_speed", "base_distance"), ("lateral_movement",),
    )


def _coverage(ctx, returns, windows):
    if ctx["court"] is None:
        return None
    cells = coverage_cells(ctx["foot"], ctx["role"], ctx["court"])
    known = cells >= 0
    if int(known.sum()) < 5:
        return None
    visited = {int(c) for c in cells[known]}
    share = len(visited) / 9.0
    left = int(np.isin(cells, (0, 3, 6)).sum())
    right = int(np.isin(cells, (2, 5, 8)).sum())
    side = left + right
    quieter = min(left, right) / side if side else 0.5
    side_name = "left" if left <= right else "right"
    n_obs = max(1, int(known.sum() // max(1, round(ctx["tracks"].fps))))
    conf = _conf(ctx, n_obs, fine=False)
    metrics = [
        _metric("half_coverage", share, conf, n_obs, series="court_x"),
        _metric("weaker_side_share", quieter, conf, n_obs, series="court_y"),
    ]
    corner = _quiet_corner(cells)
    moments = [_coverage_moment(ctx, corner, side_name)]
    return _finding(
        "footwork.coverage", ctx, metrics,
        *_coverage_text(len(visited), quieter, side_name),
        moments, ("left_ankle", "right_ankle"), ("court_x", "court_y"), (),
    )


def _base_discipline(ctx, returns, windows):
    if ctx["court"] is None or ctx["fixed"] is None or not ctx["contacts"]:
        return None
    foot = ctx["foot"]
    habitual = []
    fixed_gaps = []
    reps = []
    for contact in ctx["contacts"]:
        if not np.isfinite(foot[contact]).all():
            continue
        fixed_gap = float(np.linalg.norm(foot[contact] - ctx["fixed"]))
        fixed_gaps.append(fixed_gap)
        hab = float(ctx["dist"][contact]) if ctx["base"] is not None and np.isfinite(ctx["dist"][contact]) else float("nan")
        if np.isfinite(hab):
            habitual.append(hab)
        reps.append({"contact": contact, "habitual": hab, "fixed": fixed_gap})
    if not fixed_gaps:
        return None
    conf = _conf(ctx, len(fixed_gaps), fine=False)
    metrics = []
    if habitual:
        metrics.append(_metric("habitual_base_gap_m", median(habitual), conf, len(habitual), best=min(habitual), series="base_distance"))
    metrics.append(_metric("fixed_base_gap_m", median(fixed_gaps), conf, len(fixed_gaps), best=min(fixed_gaps), series="court_x"))
    sample = min(reps, key=lambda r: r["fixed"])
    return _finding(
        "tactics.base_discipline", ctx, metrics,
        *_base_text(median(habitual) if habitual else None, median(fixed_gaps), ctx["base_name"]),
        [_base_moment(ctx, sample)],
        ("left_ankle", "right_ankle"), ("base_distance", "court_x", "court_y"), (),
    )


def _tempo(ctx, returns, windows):
    frames = sorted({int(e.frame) for e in ctx["own_swings"]} | set(ctx["contacts"]))
    frames = [f for f in frames if 0 <= f < len(ctx["t"])]
    if len(frames) < 2:
        return None
    t = ctx["t"]
    groups = rally_bounds(frames, t)
    gaps = []
    lengths = []
    for start, end in groups:
        rally = [f for f in frames if start <= f <= end]
        lengths.append(len(rally))
        for a, b in zip(rally, rally[1:]):
            gaps.append(float(t[b] - t[a]))
    if not gaps:
        return None
    conf = _conf(ctx, len(gaps), fine=False)
    metrics = [_metric("exchange_gap_s", median(gaps), conf, len(gaps), series=None)]
    key = frames[min(1, len(frames) - 1)]
    return _finding(
        "tactics.tempo", ctx, metrics,
        *_tempo_text(median(gaps), len(groups), median(lengths)),
        [_label_moment(ctx, key, "issue", f"gap {_sec(median(gaps))} s")],
        (), (), (),
    )


def _initiative(ctx, returns, windows):
    opp = ctx["opp_track"]
    if ctx["court"] is None or opp is None or not ctx["contacts"]:
        return None
    net = ctx["court"].net_y
    ahead = []
    margins = []
    reps = []
    for contact in ctx["contacts"]:
        if contact >= opp.n or not np.isfinite(ctx["foot"][contact]).all() or not np.isfinite(opp.foot_court[contact]).all():
            continue
        mine = abs(float(ctx["foot"][contact, 1]) - net)
        theirs = abs(float(opp.foot_court[contact, 1]) - net)
        ahead.append(1.0 if mine + 0.05 < theirs else 0.0)
        margins.append(theirs - mine)
        reps.append({"contact": contact, "ahead": ahead[-1], "margin": margins[-1]})
    if not ahead:
        return None
    conf = _conf(ctx, len(ahead), fine=False)
    share = float(np.mean(ahead))
    metrics = [_metric("net_ahead_share", share, conf, len(ahead), series="court_y")]
    sample = max(reps, key=lambda r: abs(r["margin"]))
    return _finding(
        "tactics.initiative", ctx, metrics,
        *_initiative_text(int(round(share * len(ahead))), len(ahead)),
        [_initiative_moment(ctx, sample)],
        ("left_ankle", "right_ankle"), ("court_y",), (),
    )


def _finding(fid, ctx, metrics, headline, explain, moments, joints, overlay, tags):
    metrics = [m for m in metrics if m is not None and np.isfinite(m.value)]
    if not metrics:
        return None
    severity, score = _rate(metrics)
    use_tags = tags if severity in (Severity.IMPROVE, Severity.PRIORITY) else ()
    use_moments = [m for m in moments if m is not None]
    if severity in (Severity.IMPROVE, Severity.PRIORITY) and not any(m.kind == "issue" for m in use_moments):
        return None
    headline, explain = _fit(fid, headline, explain)
    return Finding(
        fid, FINDING_CATALOG[fid][0], ctx["role"], severity, headline, explain,
        tuple(metrics), tuple(use_moments), ctx["tracks"].tier,
        tuple(joints), tuple(overlay), tuple(use_tags), score,
    )


def _rate(metrics: Sequence[Metric]):
    ranked = []
    for item in metrics:
        spec = SPECS[item.name]
        ranked.append((
            severity_for(spec, item.value, item.confidence, item.n_obs),
            score_for(spec, item.value, item.confidence),
        ))
    severity = max((item[0] for item in ranked), key=SEVERITY_RANK.__getitem__)
    score = max(item[1] for item in ranked)
    return severity, score


def _metric(name, value, conf, n_obs, best=None, series=None) -> Optional[Metric]:
    if not np.isfinite(value):
        return None
    spec = SPECS[name]
    if spec.finding_id in DESCRIBE:
        conf = min(conf, DESCRIBE_CONF)
    if best is not None and not np.isfinite(best):
        best = None
    return Metric(
        spec.name, spec.label, float(value), spec.unit, spec.basis, float(conf),
        spec.target, series, int(n_obs), None if best is None else float(best),
    )


def _conf(ctx, n_obs, fine: bool) -> float:
    return confidence(
        n_obs, pose=ctx["pose"], fine=fine, fine_ok=ctx["fine_ok"],
        preview=ctx["preview"],
    )


def _split_text(matched, judgeable, onsets, landings, far):
    if matched == 0:
        headline = f"No clear hop near {judgeable} marked swings"
        timing = ""
    else:
        headline = f"A small hop showed up near {matched} of {judgeable} marked swings"
        timing = " " + _onset_landing(median(onsets), median(landings))
    far_bit = " Small far-player feet lower confidence." if far else ""
    explain = (
        f"Near {matched} of {judgeable} marked arm swings, a small hop showed up.{timing} "
        f"That mark is not a confirmed hit.{far_bit} "
        "Some shots need no hop, so this is not a fault. "
        "The range is not a timing target. "
        f"{GUIDE} {CLUB}"
    )
    return headline, explain


def _onset_landing(onset, landing) -> str:
    if abs(onset - landing) < 0.02:
        return f"The hop and the landing were both {_when(landing)}, too close to separate."
    return f"The hop started {_when(onset)} and landed {_when(landing)}."


def _when(dt) -> str:
    if dt < -0.02:
        return f"{abs(dt):.2f} s before that mark"
    if dt > 0.02:
        return f"{dt:.2f} s after that mark"
    return "about the same time as that mark"


def _recovery_text(value, best, spot, shot_anchored, left_out):
    headline = f"Getting back to your waiting spot took {_sec(value)} s"
    if shot_anchored:
        lead = "After an arm swing we marked"
        proof = "That mark is not a confirmed hit, and a stop does not prove one."
        advice = ""
    else:
        lead = "After the lunge"
        proof = "This is a measured stop after the lunge, not proof of a hit."
        advice = " Getting back sooner may help." if value > 0.90 else " That is a quick return."
    left = " Times from an arm swing were left out." if left_out else ""
    explain = (
        f"{lead}, the median time back to {spot} was {_sec(value)} s. "
        f"A practice guide is under 0.90 s. Your best was {_sec(best)} s.{advice} "
        f"{proof}{left} {GUIDE} {CLUB}"
    )
    return headline, explain


def _first_text(value, best, missed, far):
    headline = f"The first step came {_sec(value)} s after an arm swing"
    if far:
        explain = (
            f"From a marked arm swing to your first step, the median wait was {_sec(value)} s. "
            "Far feet are small, so this timing is not sure enough to coach. "
            f"That swing is not a confirmed hit. {GUIDE} {CLUB}"
        )
        return headline, explain
    missed_bit = ""
    if missed:
        missed_bit = f" On {missed} marked swings no step showed up. Those are left out."
    explain = (
        f"From a marked arm swing to your first step, the median wait was {_sec(value)} s. "
        f"A practice guide is under 0.35 s. Your best was {_sec(best)} s.{missed_bit} "
        "That swing is not a confirmed hit, so this wait is not a timing fault. "
        f"{GUIDE} {CLUB}"
    )
    return headline, explain


def _pattern_text(count, launch):
    headline = f"{_count(count)} small steps before the lunge"
    launch_bit = ""
    slow = count > 4 or (launch is not None and np.isfinite(launch) and launch > 1.30)
    if launch is not None and np.isfinite(launch):
        launch_bit = f" The lunge started {_m(launch)} m from your waiting spot."
    advice = " A shorter push may help." if slow else ""
    explain = (
        f"Before a lunge, the median count of small steps was {_count(count)}.{launch_bit} "
        "A practice guide is 4 small steps or fewer, starting within 1.3 m of the spot you wait from."
        f"{advice} This is not a call to stand in the centre. {GUIDE} {CLUB}"
    )
    return headline, explain


def _coverage_text(used, quieter, side_name):
    headline = f"You used {used} of 9 squares on your half"
    explain = (
        f"You used {used} of 9 squares. The quieter side ({side_name}) had {quieter:.0%} of your side-to-side time. "
        "The squares are the singles court, 0.46 m in from the doubles lines. "
        f"An empty corner is not a fault without the shuttle. {GUIDE} {CLUB}"
    )
    return headline, explain


def _base_text(habitual, fixed_gap, base_name):
    if habitual is None:
        headline = f"{_m(fixed_gap)} m from a centre spot at their arm swing"
        usual = "The waiting spot could not be estimated."
    else:
        headline = f"{_m(habitual)} m from your usual spot at their arm swing"
        name = "the spot you usually wait" if base_name == "usual" else "the spot you stood still in"
        usual = f"You were a median of {_m(habitual)} m from {name}."
    explain = (
        f"{usual} You were {_m(fixed_gap)} m from a centre spot 3.0 m behind the net. "
        f"Without the shuttle, neither distance is a tactics fault. {SHUTTLE} {GUIDE} {CLUB}"
    )
    return headline, explain


def _tempo_text(gap, rallies, length):
    headline = f"Marked swings inside rallies were {_sec(gap)} s apart"
    explain = (
        f"The median gap between marked arm swings inside a rally was {_sec(gap)} s across {rallies} rallies. "
        f"A typical rally here had {_swings(length)}. A gap over 2.5 s starts a new rally. "
        "This is time between arm motions, not shuttle flight, and not a judgment of pressure. "
        f"{SHUTTLE} {GUIDE} {CLUB}"
    )
    return headline, explain


def _initiative_text(count, total):
    headline = f"Nearer the net on {count} of {total} marked swings"
    explain = (
        f"On {count} of {total} marked arm swings you stood closer to the net than they did. "
        f"This is court position only. {SHUTTLE} It does not show who attacked. {GUIDE} {CLUB}"
    )
    return headline, explain


def _split_moments(ctx, reps, judgeable):
    if reps:
        sample = reps[len(reps) // 2]
        label = f"hop landed {_when(sample['landing_s'])}"
        return [_split_moment(ctx, sample, label)]
    if not judgeable:
        return []
    return [_label_moment(ctx, judgeable[0], "issue", "feet stayed flat")]


def _split_moment(ctx, rep, label):
    contact = rep["contact"]
    marks = []
    before = _foot(ctx["track"], rep["onset"])
    at = _foot(ctx["track"], contact)
    if before is not None:
        marks.append(_mark("line", rep["onset"], _hline(before), style="neutral"))
    if at is not None:
        marks.append(_mark("line", contact, _hline(at)))
        marks.append(_mark("label", contact, (at,), label[:40]))
    return _moment(ctx, contact, label[:80], "issue", marks, (rep["onset"], rep["landing"], contact))


def _recovery_moment(ctx, rep, kind):
    key = rep["frame"]
    start_pt = _foot(ctx["track"], rep["trigger"])
    home = _court_px(ctx["court"], ctx["base"])
    marks = []
    if start_pt is not None:
        marks.append(_mark("dot", rep["trigger"], (start_pt,), "start of return"))
    if home is not None:
        marks.append(_mark("dot", rep["trigger"], (home,), "waiting spot", style="target"))
    if start_pt is not None and home is not None:
        marks.append(_mark("line", rep["trigger"], (start_pt, home), f"{_m(rep['distance'])} m", value=rep["distance"]))
    trail = _trail(ctx["track"], rep["trigger"], key)
    if trail is not None:
        marks.append(_mark("trail", key, trail))
    label = f"back in {_sec(rep['since'])} s" if kind == "issue" else f"fastest return {_sec(rep['since'])} s"
    return _moment(ctx, key, label, kind, marks, (rep["trigger"], key))


def _first_moment(ctx, rep, kind):
    end = rep["frame"]
    trail = _trail(ctx["track"], rep["contact"], end)
    marks = []
    if trail is not None:
        marks.append(_mark("trail", end, trail))
    foot = _foot(ctx["track"], end)
    label = f"first step {_sec(rep['delay'])} s"
    if foot is not None:
        marks.append(_mark("label", end, (foot,), label))
    title = label if kind == "issue" else f"quickest step {_sec(rep['delay'])} s"
    return _moment(ctx, end, title, kind, marks, (rep["contact"], end))


def _pattern_moment(ctx, rep, kind):
    marks = []
    for plant in rep["plants"]:
        pt = _foot(ctx["track"], plant)
        if pt is not None:
            marks.append(_mark("dot", plant, (pt,)))
    foot = _foot(ctx["track"], rep["frame"])
    label = f"{rep['count']} small steps"
    if foot is not None:
        marks.append(_mark("label", rep["frame"], (foot,), label))
    title = label if kind == "issue" else f"shortest push, {rep['count']} steps"
    return _moment(ctx, rep["frame"], title, kind, marks, (*rep["plants"], rep["frame"]))


def _coverage_moment(ctx, corner, side_name):
    key = _present_frame(ctx["track"])
    marks = []
    trail = _trail(ctx["track"], *(_shot_of(ctx["tracks"].shots, key)))
    if trail is not None:
        marks.append(_mark("trail", key, trail))
    if corner is not None:
        px = _court_px(ctx["court"], cell_centre(corner, ctx["role"], ctx["court"]))
        if px is not None:
            marks.append(_mark("dot", key, (px,), f"{side_name} side quieter", style="target"))
    return _moment(ctx, key, f"{side_name} side quieter", "issue", marks, (key,))


def _base_moment(ctx, rep):
    key = rep["contact"]
    foot = _foot(ctx["track"], key)
    home = _court_px(ctx["court"], ctx["base"] if ctx["base"] is not None else ctx["fixed"])
    marks = []
    gap = rep["habitual"] if np.isfinite(rep["habitual"]) else rep["fixed"]
    if foot is not None:
        marks.append(_mark("dot", key, (foot,)))
    if home is not None:
        marks.append(_mark("dot", key, (home,), "usual spot", style="target"))
    if foot is not None and home is not None:
        marks.append(_mark("line", key, (foot, home), f"{_m(gap)} m from usual spot", value=gap))
    return _moment(ctx, key, f"{_m(gap)} m from usual spot", "issue", marks, (key,))


def _initiative_moment(ctx, rep):
    key = rep["contact"]
    mine = _foot(ctx["track"], key)
    theirs = _foot(ctx["opp_track"], key)
    marks = []
    if mine is not None:
        marks.append(_mark("dot", key, (mine,), "you"))
    if theirs is not None:
        marks.append(_mark("dot", key, (theirs,), "opponent", style="neutral"))
    if mine is not None and theirs is not None:
        nearer = "you" if rep["margin"] > 0 else "opponent"
        marks.append(_mark("line", key, (mine, theirs), f"{nearer} {_m(abs(rep['margin']))} m nearer the net", value=rep["margin"]))
    return _moment(ctx, key, f"{_m(abs(rep['margin']))} m net gap", "issue", marks, (key,))


def _label_moment(ctx, key, kind, label):
    foot = _foot(ctx["track"], key)
    marks = []
    if foot is not None:
        marks.append(_mark("label", key, (foot,), label[:40]))
    return _moment(ctx, key, label, kind, marks, (key,))


def _moment(ctx, key, label, kind, marks, cover):
    start, end = _span(ctx["tracks"], key, cover)
    kept = [m for m in marks if m is not None and start <= m.frame <= end]
    text = label.strip() or "moment"
    return Moment(ctx["role"], start, int(key), end, text[:120], kind, tuple(kept))


def _span(tracks, key, cover):
    t = tracks.t
    key = int(np.clip(key, 0, tracks.n - 1))
    start = int(np.searchsorted(t, t[key] - 1.0, side="left"))
    end = int(np.searchsorted(t, t[key] + 0.5, side="right") - 1)
    shot_start, shot_end = _shot_of(tracks.shots, key)
    for frame in cover:
        if frame is None:
            continue
        frame = int(frame)
        if shot_start <= frame <= shot_end:
            start = min(start, frame)
            end = max(end, frame)
    start = max(0, shot_start, start)
    end = min(tracks.n - 1, shot_end, max(end, key))
    start = min(start, key)
    return start, end


def _mark(kind, frame, points, label="", value=None, style="actual"):
    if points is None:
        return None
    clean = []
    for point in points:
        if point is None:
            return None
        arr = np.asarray(point, dtype=np.float64)
        if arr.shape != (2,) or not np.isfinite(arr).all():
            return None
        clean.append((float(arr[0]), float(arr[1])))
    if value is not None and not np.isfinite(value):
        value = None
    return Mark(kind, int(frame), tuple(clean), label[:40], None if value is None else float(value), style)


def _foot(track, frame):
    frame = int(frame)
    if track is None or frame < 0 or frame >= track.n:
        return None
    if not (bool(track.visible[frame, 15]) or bool(track.visible[frame, 16])):
        return None
    xy = track.foot_px[frame]
    if not np.isfinite(xy).all():
        return None
    return (float(xy[0]), float(xy[1]))


def _court_px(court, xy):
    if court is None or xy is None or not np.isfinite(xy).all():
        return None
    px = court.unproject(np.asarray(xy, dtype=np.float64).reshape(1, 2))[0]
    if not np.isfinite(px).all():
        return None
    return (float(px[0]), float(px[1]))


def _trail(track, start, end):
    start, end = int(start), int(end)
    if end < start:
        start, end = end, start
    step = max(1, (end - start) // 40)
    points = []
    for frame in range(start, end + 1, step):
        pt = _foot(track, frame)
        if pt is not None:
            points.append(pt)
    last = _foot(track, end)
    if last is not None and (not points or points[-1] != last):
        points.append(last)
    if len(points) < 2:
        return None
    return points[:90]


def _hline(pt, half=24.0):
    return ((pt[0] - half, pt[1]), (pt[0] + half, pt[1]))


def _choose_base(foot, speed, role, court, t, contacts):
    if court is None:
        return None, "none"
    habitual = habitual_base(foot, t, contacts)
    if habitual is not None:
        return habitual, "usual"
    return waiting_base(foot, speed, role, court), "still"


def _triggers(swings, lunges, t):
    covered = []
    raw = []
    for lunge in lunges:
        bottom = int(lunge.data["bottom_frame"])
        end = int(lunge.data["end_frame"])
        raw.append((bottom, "lunge"))
        covered.append((int(lunge.frame), end))
    for swing in swings:
        end = int(swing.data["end"])
        if any(a <= end <= b for a, b in covered):
            continue
        raw.append((end, "swing"))
    raw = [(frame, kind) for frame, kind in raw if 0 <= frame < len(t)]
    raw.sort()
    merged = []
    for frame, kind in raw:
        if merged and t[frame] - t[merged[-1][0]] <= TRIGGER_MERGE_S:
            continue
        merged.append((frame, kind))
    return merged


def _pose_gap(track, t, start, end) -> bool:
    # A repeated pose is still on screen. Only a missing player row is a hole.
    return measured_gap(track.present, t, start, end, 0.25)


def _signal_window(ctx, frame, before, after):
    t = ctx["t"]
    start = _back(t, frame, before)
    end = frame + _ahead(t, frame, after)
    shot_start, shot_end = _shot_of(ctx["tracks"].shots, frame)
    start = max(start, shot_start)
    end = min(end, shot_end, ctx["track"].n - 1)
    if end <= start:
        return False, start, end
    gap = _pose_gap(ctx["track"], t, start, end)
    piece_h = ctx["hop"][start:end + 1]
    piece_d = ctx["dip"][start:end + 1]
    signal = np.isfinite(piece_h).any() or np.isfinite(piece_d).any()
    return (not gap and signal), start, end


def _window(start, end, reason) -> Window:
    start, end = int(start), int(end)
    if end < start:
        end = start
    return Window(max(0, start), end, reason)


def _quiet_corner(cells):
    corners = (0, 2, 6, 8)
    counts = {cell: int(np.sum(cells == cell)) for cell in corners}
    if max(counts.values()) - min(counts.values()) < 3:
        return None
    return min(counts, key=counts.get)


def _kind(events, kind, role):
    return [e for e in events if e.kind == kind and e.player == role]


def _other(role: Role) -> Role:
    return Role.FAR if role == Role.NEAR else Role.NEAR


def _shot_of(shots, frame):
    frame = int(frame)
    for start, end in shots:
        if start <= frame <= end:
            return start, end
    return 0, frame


def _same_shot(shots, a, b) -> bool:
    for start, end in shots:
        if start <= a <= end and start <= b <= end:
            return True
    return False


def _back(t, frame, seconds) -> int:
    t0 = t[frame] - seconds
    idx = np.flatnonzero(t >= t0)
    return int(idx[0]) if idx.size else 0


def _ahead(t, frame, seconds) -> int:
    t1 = t[frame] + seconds
    idx = np.flatnonzero(t <= t1)
    if not idx.size:
        return 0
    return int(idx[-1] - frame)


def _pose(track, frames) -> float:
    idx = np.asarray(frames, dtype=int)
    idx = idx[(idx >= 0) & (idx < track.n)]
    if idx.size == 0:
        idx = np.flatnonzero(track.present)[:1]
    if idx.size == 0:
        return 0.5
    vals = track.conf[idx][:, 15:17]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.5
    return float(np.clip(vals.mean(), 0.0, 1.0))


def _is_partial(track, shots) -> bool:
    present = np.asarray(track.present)
    if not present.any():
        return False
    for start, end in shots:
        piece = present[start:end + 1]
        if piece.any() and float(piece.mean()) < PARTIAL_SHARE:
            return True
    return False


def _present_frame(track) -> int:
    idx = np.flatnonzero(track.present)
    return int(idx[len(idx) // 2]) if idx.size else 0


def _closest_to_zero(values) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr[np.argmin(np.abs(arr))])


def _fit(fid, headline, explain):
    headline = " ".join(headline.split())
    explain = " ".join(explain.split())
    if not headline or not explain or len(headline) > 90 or len(explain) > 400:
        raise ValueError(f"{fid} text {len(headline)}/{len(explain)}: {headline} || {explain}")
    return headline, explain


def _sec(value) -> str:
    return f"{float(value):.2f}"


def _m(value) -> str:
    return f"{float(value):.1f}"


def _count(value) -> str:
    return str(int(round(float(value))))


def _swings(value) -> str:
    count = int(round(float(value)))
    return f"{count} swing" if count == 1 else f"{count} swings"
