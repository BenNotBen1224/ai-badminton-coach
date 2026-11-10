from __future__ import annotations

import time
from typing import Any, Mapping, Optional, Sequence

from .analyze_body import body_analysis
from .analyze_move import movement_analysis
from .catalog import METRIC_CATALOG
from .events import detect_events
from .play_state import exchange_events
from .schema import (
    SCHEMA_VERSION, Event, Metric, Moment, Partial, Report, Severity, Tier, Window,
)
from .severity import MetricSpec, check_findings
from .tracks import Tracks, load_tracks

WINDOW_PAD_FRAMES = 3


STAGES = ("load_tracks", "events", "body", "move", "assemble")


def analyze(
    doc: Mapping[str, Any],
    video_id: str = "",
    revision: int = 0,
    timings: Optional[dict[str, float]] = None,
) -> Report:
    """Pose document to Report. When `timings` is a dict it receives the seconds spent in each of STAGES."""
    clock = time.perf_counter
    t0 = clock()
    tracks = load_tracks(doc)
    t1 = clock()
    events = exchange_events(tracks, detect_events(tracks))
    t2 = clock()
    body = body_analysis(tracks, events)
    t3 = clock()
    move = movement_analysis(tracks, events)
    t4 = clock()
    report = assemble(tracks, events, (body, move), METRIC_CATALOG, video_id, revision)
    if timings is not None:
        timings.update(zip(STAGES, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, clock() - t4)))
    return report


def assemble(
    tracks: Tracks,
    events: Sequence[Event],
    parts: Sequence[Partial],
    catalog: Optional[Mapping[str, MetricSpec]] = None,
    video_id: str = "",
    revision: int = 0,
) -> Report:
    metrics: dict[str, tuple[Metric, ...]] = {}
    series: dict[str, dict[str, Sequence[Optional[float]]]] = {}
    for part in parts:
        for role, found in part.metrics.items():
            metrics[role] = metrics.get(role, ()) + tuple(found)
        for role, named in part.series.items():
            have = series.setdefault(role, {})
            clash = sorted(set(have) & set(named))
            if clash:
                raise ValueError(f"series {role}.{clash} produced by two analysis units")
            have.update(named)
    all_events = tuple(sorted(
        (*events, *(e for part in parts for e in part.events)),
        key=lambda e: (e.frame, e.kind, e.player.value),
    ))
    findings = tuple(f for part in parts for f in part.findings)
    if catalog is not None:
        check_findings(findings, catalog)
    windows: tuple[Window, ...] = ()
    if tracks.tier == Tier.PREVIEW:
        wanted = [w for part in parts for w in part.windows]
        for finding in findings:
            if finding.severity in (Severity.IMPROVE, Severity.PRIORITY):
                wanted.extend(_moment_window(m, finding.id, tracks.n) for m in finding.issue_moments)
        windows = merge_windows(wanted)
    codes = [*tracks.warnings, *(w for part in parts for w in part.warnings)]
    if tracks.tier == Tier.PREVIEW:
        codes.append("preview_estimate")
    return Report(
        SCHEMA_VERSION, dict(tracks.source), tracks.tier, tracks.n, tracks.fps,
        tuple(tracks.players), metrics, series, all_events, findings, windows,
        video_id, revision, tuple(dict.fromkeys(codes)), tuple(tracks.t),
    )


def _moment_window(moment: Moment, reason: str, n_frames: int) -> Window:
    return Window(
        max(0, moment.start_frame - WINDOW_PAD_FRAMES),
        min(n_frames - 1, moment.end_frame + WINDOW_PAD_FRAMES),
        reason,
    )


def merge_windows(windows: Sequence[Window]) -> tuple[Window, ...]:
    merged: list[Window] = []
    for w in sorted(windows, key=lambda w: (w.start_frame, w.end_frame)):
        if merged and w.start_frame <= merged[-1].end_frame + 1:
            last = merged[-1]
            reasons = last.reason.split(", ")
            if w.reason not in reasons:
                reasons.append(w.reason)
            merged[-1] = Window(last.start_frame, max(last.end_frame, w.end_frame), ", ".join(reasons))
        else:
            merged.append(w)
    return tuple(merged)
