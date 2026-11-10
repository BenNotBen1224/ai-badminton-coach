from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

from .schema import SEVERITY_RANK

PARAMS = {
    "lead_s": 0.6,
    "slow": 0.25,
    "ramp_s": 0.45,
    "ramp_parts": 4,
    "freeze_s": 1.5,
    "zoom_peak": 1.6,
    "compare_freeze_s": 1.5,
    "review_s": 30.0,
}


class Segment(NamedTuple):
    kind: str
    start_frame: int
    end_frame: int
    speed: float
    hold_s: float
    zoom: float
    show: str


class _Part(NamedTuple):
    desired_source: float
    speed: float
    zoom: float


class Timeline:
    def __init__(self, segments: Sequence[Segment], timestamps: Sequence[float]):
        self.segments = tuple(segments)
        self._stamps = timestamps if isinstance(timestamps, tuple) else tuple(timestamps)
        self.duration_s = sum(_wall(segment, self._stamps) for segment in self.segments)

    def frame_at(self, t: float) -> float:
        segment, local = self._locate(t)
        if segment is None:
            return 0.0
        return self._mapped(segment, local)

    def sample_at(self, t: float) -> dict:
        segment, local = self._locate(t)
        if segment is None:
            return {"frame": 0.0, "speed": 0.0, "zoom": 1.0, "show": "issue"}
        return {
            "frame": self._mapped(segment, local),
            "speed": segment.speed,
            "zoom": segment.zoom,
            "show": segment.show,
        }

    def to_dict(self) -> dict:
        return {
            "segments": [segment._asdict() for segment in self.segments],
            "duration_s": self.duration_s,
        }

    def _locate(self, t: float):
        segments = self.segments
        if not segments:
            return None, 0.0
        if self.duration_s <= 0 or t >= self.duration_s:
            last = segments[-1]
            return last, _wall(last, self._stamps)
        if t < 0.0:
            t = 0.0
        cursor = 0.0
        for segment in segments:
            wall = _wall(segment, self._stamps)
            nxt = cursor + wall
            if wall > 0.0 and t < nxt:
                return segment, t - cursor
            cursor = nxt
        last = segments[-1]
        return last, _wall(last, self._stamps)

    def _mapped(self, segment: Segment, local: float) -> float:
        if segment.kind == "freeze":
            return float(segment.start_frame)
        stamps = self._stamps
        start = segment.start_frame
        end = segment.end_frame
        span = stamps[end] - stamps[start]
        if span <= 0.0 or segment.speed == 0.0:
            return float(start)
        wall = span / segment.speed
        if local <= 0.0:
            progress = 0.0
        elif local >= wall:
            progress = 1.0
        else:
            progress = local / wall
        return _invert(stamps, stamps[start] + progress * span, start, end)


def build_timeline(moment, fps, n_frames, *, timestamps, params=None, compare=None, show="issue") -> Timeline:
    merged = _merged(params)
    if timestamps is None:
        stamps = tuple(i / fps for i in range(n_frames))
    elif isinstance(timestamps, tuple):
        stamps = timestamps
    else:
        stamps = tuple(timestamps)
    parts = int(merged["ramp_parts"])
    part_wall = float(merged["ramp_s"]) / parts if parts else 0.0
    speeds = _speeds(parts, float(merged["slow"]))
    zooms = _zooms(parts, float(merged["zoom_peak"]))
    near_key = [
        _Part(speeds[i] * part_wall, speeds[i], zooms[i])
        for i in range(parts - 1, -1, -1)
    ]
    inbound = near_key + [_Part(float(merged["lead_s"]), 1.0, 1.0)]
    segments: list[Segment] = []
    if moment.key_frame > moment.start_frame:
        taken, _cursor = _allocate(moment.key_frame, moment.start_frame, stamps, inbound, show)
        segments.extend(taken)
    segments.append(Segment(
        "freeze",
        moment.key_frame,
        moment.key_frame,
        0.0,
        float(merged["freeze_s"]),
        float(merged["zoom_peak"]),
        show,
    ))
    if moment.end_frame > moment.key_frame:
        taken, cursor = _allocate(moment.key_frame, moment.end_frame, stamps, near_key, show)
        segments.extend(taken)
        if cursor < moment.end_frame:
            tail = _play(cursor, moment.end_frame, 1.0, 1.0, show, stamps)
            if tail is not None:
                segments.append(tail)
    if compare is not None:
        segments.append(Segment(
            "freeze",
            compare.key_frame,
            compare.key_frame,
            0.0,
            float(merged["compare_freeze_s"]),
            float(merged["zoom_peak"]),
            "best",
        ))
    return Timeline(segments, stamps)


def review_picks(report):
    # top_findings stops at improve and priority, so a short clip ends before the cap.
    pool = [finding for finding in report.findings if finding.issue_moments or finding.best_moment]
    pool.sort(key=lambda finding: (
        -SEVERITY_RANK[finding.severity],
        -finding.score,
        -sum(metric.confidence for metric in finding.metrics) / len(finding.metrics),
        finding.id,
    ))
    return tuple(pool)


def build_review(report, picks, fps, *, limit_s=None) -> list[Timeline]:
    cap = PARAMS["review_s"] if limit_s is None else float(limit_s)
    chosen = review_picks(report) if picks is None else picks
    stamps = tuple(report.timestamps) if report.timestamps else None
    out: list[Timeline] = []
    used = 0.0
    for finding in chosen:
        issues = finding.issue_moments
        if issues:
            moment = issues[0]
            full = build_timeline(
                moment, fps, report.n_frames,
                timestamps=stamps, compare=finding.best_moment, show="issue",
            )
            if used + full.duration_s <= cap:
                out.append(full)
                used += full.duration_s
                continue
            if finding.best_moment is not None:
                bare = build_timeline(
                    moment, fps, report.n_frames,
                    timestamps=stamps, compare=None, show="issue",
                )
                if used + bare.duration_s <= cap:
                    out.append(bare)
                    used += bare.duration_s
                    continue
            break
        best = finding.best_moment
        if best is None:
            continue
        timeline = build_timeline(
            best, fps, report.n_frames,
            timestamps=stamps, compare=None, show="best",
        )
        if used + timeline.duration_s > cap:
            break
        out.append(timeline)
        used += timeline.duration_s
    return out


def _merged(overrides) -> dict:
    if not overrides:
        return PARAMS
    merged = dict(PARAMS)
    for key, value in overrides.items():
        if key in merged:
            merged[key] = value
    return merged


def _speeds(parts: int, slow: float) -> list[float]:
    # One constant speed per slice. The midpoint is not pinned to either edge of the ease.
    speeds = []
    for i in range(parts):
        u = (i + 0.5) / parts
        eased = u * u * (3.0 - 2.0 * u)
        speeds.append(1.0 + (slow - 1.0) * eased)
    return speeds


def _zooms(parts: int, peak: float) -> list[float]:
    return [1.0 + (peak - 1.0) * ((i + 0.5) / parts) for i in range(parts)]


def _wall(segment: Segment, stamps: Sequence[float]) -> float:
    if segment.kind == "freeze":
        return float(segment.hold_s)
    span = stamps[segment.end_frame] - stamps[segment.start_frame]
    if segment.start_frame == segment.end_frame or span <= 0.0 or segment.speed == 0.0:
        return 0.0
    return span / segment.speed


def _play(start: int, end: int, speed: float, zoom: float, show: str, stamps: Sequence[float]) -> Optional[Segment]:
    if start >= end or speed <= 0.0 or stamps[end] - stamps[start] <= 0.0:
        return None
    return Segment("play", int(start), int(end), float(speed), 0.0, float(zoom), show)


def _take(origin: int, bound: int, desired: float, stamps: Sequence[float]) -> int:
    step = 1 if bound > origin else -1
    if step == 0 or desired <= 0.0:
        return origin
    edge = origin
    while edge != bound:
        nxt = edge + step
        span = stamps[nxt] - stamps[origin] if step > 0 else stamps[origin] - stamps[nxt]
        if span > desired + 1e-9:
            break
        edge = nxt
    if edge == origin and edge != bound:
        edge = origin + step
    return edge


def _allocate(origin: int, bound: int, stamps: Sequence[float], parts: Sequence[_Part], show: str):
    forward = bound >= origin
    cursor = origin
    chunks: list[Segment] = []
    for part in parts:
        if cursor == bound:
            break
        remaining = stamps[bound] - stamps[cursor] if forward else stamps[cursor] - stamps[bound]
        if remaining <= 1e-15:
            break
        shrunk = remaining + 1e-9 < part.desired_source
        edge = bound if shrunk else _take(cursor, bound, part.desired_source, stamps)
        if edge == cursor:
            break
        if forward:
            segment = _play(cursor, edge, part.speed, part.zoom, show, stamps)
        else:
            segment = _play(edge, cursor, part.speed, part.zoom, show, stamps)
        if segment is not None:
            chunks.append(segment)
        cursor = edge
        if shrunk:
            break
    if not forward:
        chunks.reverse()
    return chunks, cursor


def _invert(stamps: Sequence[float], source: float, start: int, end: int) -> float:
    if source <= stamps[start]:
        return float(start)
    if source >= stamps[end]:
        return float(end)
    lo = start
    hi = end
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if stamps[mid] <= source:
            lo = mid
        else:
            hi = mid
    gap = stamps[lo + 1] - stamps[lo]
    if gap <= 0.0:
        return float(lo)
    return lo + (source - stamps[lo]) / gap
