"""Contract for the coaching layer: vocabulary and data types only.

No analysis logic lives here. Workers import these types and never edit this
file. Requests for changes go to the parent session.

A coaching report uses schema version 3. An absent measurement is Metric.value
None, never a stand-in number. The loader also accepts a version 2 report and
normalizes those old placeholders in memory. It does not rewrite the file.

Input documents are the pose JSON (schema_version 3) written by the fast
pipeline. That version number belongs to the pose file, not the coaching report.
A preview document also carries:
  doc["tier"] == "preview"            (refined documents say "refined" or omit it)
  player row["src"] == "interpolated" for frames that were not measured
Rows without "src" were not interpolated. A source video can also repeat a
frame (25 to 30 fps conversion, duplicated frames) and the pose JSON then
repeats the previous row exactly. Those rows are not fresh measurements
either, so Tracks.measured is False for them too.

Numbers: every numeric field accepts any real number, including numpy scalars,
and is stored as a plain Python float or int. A series accepts a list, tuple
or numpy array; NaN becomes None (unknown). Infinity is rejected.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

SCHEMA_VERSION = 3
LEGACY_SCHEMA_VERSION = 2

COCO_JOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


class Role(str, Enum):
    NEAR = "near"
    FAR = "far"


class Category(str, Enum):
    TECHNIQUE = "technique"
    CONTACT = "contact"
    FOOTWORK = "footwork"
    TACTICS = "tactics"


class Severity(str, Enum):
    STRENGTH = "strength"
    INFO = "info"
    IMPROVE = "improve"
    PRIORITY = "priority"


class Basis(str, Enum):
    PROJECTED_2D = "projected_2d"
    COURT_PLANE = "court_plane"
    TIMING = "timing"
    COUNT = "count"


class Tier(str, Enum):
    PREVIEW = "preview"
    REFINED = "refined"


SEVERITY_RANK = {Severity.PRIORITY: 3, Severity.IMPROVE: 2, Severity.INFO: 1, Severity.STRENGTH: 0}

EVENT_KINDS = frozenset({"swing", "step", "split_step", "lunge", "base_return"})

FIX_TAGS: Mapping[str, str] = {
    "lower_stance": "sit lower with bent knees in the ready position",
    "lunge_depth": "reach net and corner shots with a deeper, controlled lunge",
    "lunge_recovery": "push back out of a lunge quickly",
    "split_step": "time the small hop just before the opponent hits",
    "first_step": "explosive first step after the split step",
    "base_recovery": "return to the base position between shots",
    "lateral_movement": "side-to-side court movement and agility",
    "core_stability": "trunk control and posture",
    "core_rotation": "trunk rotation for overhead and forehand strokes",
    "shoulder_health": "shoulder stability and overhead mobility",
    "contact_height": "hit higher and earlier, with reach and jump",
    "arm_extension": "extend the hitting arm at contact",
    "balance": "single-leg balance and landing control",
    "ankle_calf": "calf and ankle strength and bounce",
    "hip_mobility": "hip flexibility for deep lunges",
    "jump_power": "explosive leg power for jump shots",
    "endurance": "rally stamina and tempo",
}

FINDING_CATALOG: Mapping[str, tuple[Category, str]] = {
    "technique.stay_lower": (Category.TECHNIQUE, "hips stay high, so the center of mass proxy sits above the target range"),
    "technique.knee_bend": (Category.TECHNIQUE, "knees are too straight in the ready stance"),
    "technique.lunge_depth": (Category.TECHNIQUE, "lunges to the net or corners are shallow"),
    "technique.stance_width": (Category.TECHNIQUE, "feet are too close together or too wide in the ready stance"),
    "technique.trunk_lean": (Category.TECHNIQUE, "upper body leans too far forward or back"),
    "contact.reach_height": (Category.CONTACT, "overhead contact point is low relative to the player's standing reach"),
    "contact.arm_extension": (Category.CONTACT, "hitting arm is bent at the contact moment (projected elbow angle)"),
    "contact.hand_speed": (Category.CONTACT, "racket-hand speed relative to body size, and how consistent it is across swings"),
    "footwork.split_step": (Category.FOOTWORK, "split step is missing or mistimed against the opponent's swing"),
    "footwork.recovery": (Category.FOOTWORK, "slow return to base after the player's own shots"),
    "footwork.first_step": (Category.FOOTWORK, "slow first move after the opponent's swing"),
    "footwork.step_pattern": (Category.FOOTWORK, "too many small steps, or lunging from too far away"),
    "footwork.coverage": (Category.FOOTWORK, "how much of the court the player covers and where"),
    "tactics.base_discipline": (Category.TACTICS, "distance from the base position at the moment the opponent hits"),
    "tactics.tempo": (Category.TACTICS, "rhythm of the exchanges: time between swings"),
    "tactics.initiative": (Category.TACTICS, "share of the rally spent ahead of or behind the opponent in court depth"),
}


EVENT_DATA_KEYS: Mapping[str, tuple[str, ...]] = {
    "swing": ("start", "end", "peak_frame", "peak_speed", "overhead", "hand"),
    "lunge": ("depth", "bottom_frame", "end_frame"),
}

MOMENT_KINDS = ("issue", "best")

MARK_STYLES = ("actual", "target", "neutral")

# Geometry per mark kind, in source-video pixels (x right, y down), as (min, max) point counts.
#   dot      one point: a joint or a spot to highlight
#   label    one point: where the text sits
#   line     two points: a segment such as a hip-height line
#   bracket  two points: a dimension line with end ticks, such as stance width
#   band     two opposite corners of a rectangle, such as a target zone
#   angle    three points (a, vertex, b): the arc at the vertex between a and b
#   trail    two or more points: a path over time, ending at the mark frame
MARK_KINDS: Mapping[str, tuple[int, int]] = {
    "dot": (1, 1),
    "label": (1, 1),
    "line": (2, 2),
    "bracket": (2, 2),
    "band": (2, 2),
    "angle": (3, 3),
    "trail": (2, 90),
}

WARNINGS: Mapping[str, str] = {
    "doubles": "This looks like a doubles game. Coaching follows two players only, so some numbers may be off.",
    "high_fps": "This video has a high frame rate, so analysis can take longer than usual.",
    "no_court": "The court lines were not found, so court positions and distances are skipped.",
    "preview_estimate": "This is the fast preview. Numbers are estimates and the refined pass can change them.",
    "repeated_frames": "Some pose estimates repeat, so speed estimates skip those repeats.",
    "few_swings": "Only a few arm swings were found, so averages are rough.",
    "partial_track": "A player was visible for only part of the clip, so some numbers use fewer frames.",
    "unresolved_recovery": "A hip drop had no measured return to the ready hips from before the drop, so that unfinished move was not scored.",
}


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _num(value: Any, name: str) -> int | float:
    ok = isinstance(value, numbers.Real) and not isinstance(value, bool)
    _ensure(ok and math.isfinite(value), f"{name} must be a finite number, got {value!r}")
    return int(value) if isinstance(value, numbers.Integral) else float(value)


def _int(value: Any, name: str) -> int:
    ok = isinstance(value, numbers.Integral) and not isinstance(value, bool)
    _ensure(ok, f"{name} must be an integer, got {value!r}")
    return int(value)


def _put(obj: Any, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


def as_series(values: Any, ndigits: int = 3) -> tuple[Optional[float], ...]:
    """Turn a list, tuple or numpy array into a plain tuple of rounded floats.

    NaN and None become None (unknown). Infinity is a bug upstream and raises.
    """
    items = values.tolist() if hasattr(values, "tolist") else list(values)
    out: list[Optional[float]] = []
    for v in items:
        if v is None:
            out.append(None)
            continue
        _ensure(isinstance(v, numbers.Real) and not isinstance(v, bool), f"series values must be numbers, got {v!r}")
        f = float(v)
        if math.isnan(f):
            out.append(None)
            continue
        _ensure(math.isfinite(f), f"non-finite value {v!r}")
        out.append(round(f, ndigits))
    return tuple(out)


def _snake(name: str, what: str) -> None:
    _ensure(
        bool(name) and name == name.lower() and all(c.isalnum() or c in "_." for c in name),
        f"{what} must be lowercase snake_case, got {name!r}",
    )


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    value: Optional[float]
    unit: str
    basis: Basis
    confidence: float
    target: Optional[tuple[float, float]] = None
    series: Optional[str] = None
    n_obs: int = 1
    best: Optional[float] = None

    def __post_init__(self) -> None:
        _snake(self.name, "metric name")
        _ensure(bool(self.label) and bool(self.unit), f"metric {self.name} needs a label and a unit")
        _put(self, "confidence", float(_num(self.confidence, f"metric {self.name} confidence")))
        _ensure(0.0 <= self.confidence <= 1.0, f"metric {self.name} confidence must be in [0, 1]")
        if self.target is not None:
            _ensure(len(self.target) == 2, f"metric {self.name} target must be (low, high)")
            low, high = (float(_num(t, f"metric {self.name} target")) for t in self.target)
            _ensure(low <= high, f"metric {self.name} target must be (low, high)")
            _put(self, "target", (low, high))
        if self.series is not None:
            _snake(self.series, "series name")
        _put(self, "n_obs", _int(self.n_obs, f"metric {self.name} n_obs"))
        _ensure(self.n_obs >= 0, f"metric {self.name} n_obs must be >= 0")
        if self.best is not None:
            _put(self, "best", float(_num(self.best, f"metric {self.name} best")))
        if self.value is None:
            _ensure(self.n_obs == 0, f"metric {self.name} value is missing but n_obs is {self.n_obs}")
            _ensure(self.confidence == 0.0, f"metric {self.name} value is missing but confidence is {self.confidence}")
            _ensure(self.best is None, f"metric {self.name} value is missing but best is set")
            return
        _put(self, "value", float(_num(self.value, f"metric {self.name} value")))
        _ensure(self.n_obs > 0, f"metric {self.name} has a value but n_obs is {self.n_obs}")


@dataclass(frozen=True)
class Mark:
    """One drawing primitive. The analysis units decide the geometry, the viewer only draws it."""

    kind: str
    frame: int
    points: tuple[tuple[float, float], ...]
    label: str = ""
    value: Optional[float] = None
    style: str = "actual"

    def __post_init__(self) -> None:
        _ensure(self.kind in MARK_KINDS, f"unknown mark kind {self.kind!r}")
        _ensure(self.style in MARK_STYLES, f"unknown mark style {self.style!r}")
        _put(self, "frame", _int(self.frame, "mark frame"))
        _ensure(self.frame >= 0, "mark frame must be >= 0")
        points = tuple((float(_num(x, "mark x")), float(_num(y, "mark y"))) for x, y in self.points)
        low, high = MARK_KINDS[self.kind]
        _ensure(low <= len(points) <= high, f"mark {self.kind} needs {low} to {high} points, got {len(points)}")
        _put(self, "points", points)
        _ensure(len(self.label) <= 40, "mark label must be at most 40 characters")
        if self.value is not None:
            _put(self, "value", float(_num(self.value, "mark value")))


@dataclass(frozen=True)
class Moment:
    player: Role
    start_frame: int
    key_frame: int
    end_frame: int
    label: str
    kind: str = "issue"
    marks: tuple[Mark, ...] = ()

    def __post_init__(self) -> None:
        for name in ("start_frame", "key_frame", "end_frame"):
            _put(self, name, _int(getattr(self, name), f"moment {name}"))
        _ensure(0 <= self.start_frame <= self.key_frame <= self.end_frame, f"moment frames must satisfy 0 <= start <= key <= end, got {self.start_frame}, {self.key_frame}, {self.end_frame}")
        _ensure(bool(self.label), "moment needs a label")
        _ensure(self.kind in MOMENT_KINDS, f"unknown moment kind {self.kind!r}")
        _put(self, "marks", tuple(self.marks))
        _ensure(all(self.start_frame <= m.frame <= self.end_frame for m in self.marks), "moment marks must fall between the start and end frames")


@dataclass(frozen=True)
class Event:
    """A detected happening.

    For a swing, frame is the estimated contact frame. The hand-speed peak comes a few
    frames earlier (data["peak_frame"]) and that gap is the timing signal.
    """

    kind: str
    player: Role
    frame: int
    confidence: float
    data: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure(self.kind in EVENT_KINDS, f"unknown event kind {self.kind!r}")
        _put(self, "frame", _int(self.frame, "event frame"))
        _ensure(self.frame >= 0, "event frame must be >= 0")
        _put(self, "confidence", float(_num(self.confidence, "event confidence")))
        _ensure(0.0 <= self.confidence <= 1.0, "event confidence must be in [0, 1]")
        _put(self, "data", {str(k): _num(v, f"event data {k}") for k, v in self.data.items()})
        missing = [k for k in EVENT_DATA_KEYS.get(self.kind, ()) if k not in self.data]
        _ensure(not missing, f"{self.kind} event is missing data {missing}")
        d = self.data
        if self.kind == "swing":
            _ensure(d["start"] <= self.frame <= d["end"], "swing contact frame must lie between its start and end")
            _ensure(d["start"] <= d["peak_frame"] <= d["end"], "swing peak frame must lie between its start and end")
            _ensure(d["overhead"] in (0, 1) and d["hand"] in (0, 1), "swing overhead and hand must be 0 or 1")
        if self.kind == "lunge":
            _ensure(self.frame <= d["bottom_frame"] <= d["end_frame"], "lunge needs frame <= bottom_frame <= end_frame")


def swing_event(player: Role, frame: int, *, start: int, end: int, peak_frame: int, peak_speed: float,
                overhead: bool, right_hand: bool = True, confidence: float = 0.8) -> Event:
    data = {"start": start, "end": end, "peak_frame": peak_frame, "peak_speed": peak_speed, "overhead": int(overhead), "hand": int(right_hand)}
    return Event("swing", player, frame, confidence, data)


def lunge_event(player: Role, frame: int, *, depth: float, bottom_frame: int, end_frame: int, confidence: float = 0.8) -> Event:
    return Event("lunge", player, frame, confidence, {"depth": depth, "bottom_frame": bottom_frame, "end_frame": end_frame})


@dataclass(frozen=True)
class Window:
    start_frame: int
    end_frame: int
    reason: str

    def __post_init__(self) -> None:
        _put(self, "start_frame", _int(self.start_frame, "window start_frame"))
        _put(self, "end_frame", _int(self.end_frame, "window end_frame"))
        _ensure(0 <= self.start_frame <= self.end_frame, "window needs 0 <= start <= end")


@dataclass(frozen=True)
class Finding:
    id: str
    category: Category
    player: Role
    severity: Severity
    headline: str
    explain: str
    metrics: tuple[Metric, ...]
    moments: tuple[Moment, ...]
    tier: Tier
    focus_joints: tuple[str, ...] = ()
    overlay_series: tuple[str, ...] = ()
    fix_tags: tuple[str, ...] = ()
    score: float = 0.0

    def __post_init__(self) -> None:
        _ensure(self.id in FINDING_CATALOG, f"finding id {self.id!r} is not in FINDING_CATALOG")
        _ensure(FINDING_CATALOG[self.id][0] == self.category, f"finding {self.id} belongs to category {FINDING_CATALOG[self.id][0].value}")
        _ensure(0 < len(self.headline) <= 90, f"finding {self.id} headline must be 1 to 90 characters")
        _ensure(0 < len(self.explain) <= 400, f"finding {self.id} explanation must be 1 to 400 characters")
        for name in ("metrics", "moments", "focus_joints", "overlay_series", "fix_tags"):
            _put(self, name, tuple(getattr(self, name)))
        _put(self, "score", float(_num(self.score, f"finding {self.id} score")))
        _ensure(self.score >= 0.0, f"finding {self.id} score must be >= 0")
        _ensure(len(self.metrics) >= 1, f"finding {self.id} needs at least one metric")
        if all(m.value is None for m in self.metrics):
            _ensure(self.severity == Severity.INFO, f"finding {self.id} is unavailable, so it must be info")
            _ensure(self.score == 0, f"finding {self.id} is unavailable, so its score must be 0")
            _ensure(not self.moments, f"finding {self.id} is unavailable, so it must not have a moment")
        _ensure(all(m.player == self.player for m in self.moments), f"finding {self.id} has a moment for another player")
        _ensure(sum(m.kind == "best" for m in self.moments) <= 1, f"finding {self.id} has more than one best moment")
        unknown = [t for t in self.fix_tags if t not in FIX_TAGS]
        _ensure(not unknown, f"finding {self.id} has unknown fix tags {unknown}")
        bad = [j for j in self.focus_joints if j not in COCO_JOINTS]
        _ensure(not bad, f"finding {self.id} has unknown joints {bad}")
        if self.severity in (Severity.IMPROVE, Severity.PRIORITY):
            _ensure(len(self.fix_tags) >= 1, f"finding {self.id} needs a fix tag so exercises can be suggested")
            _ensure(len(self.issue_moments) >= 1, f"finding {self.id} needs a moment so the viewer can show it")

    @property
    def issue_moments(self) -> tuple[Moment, ...]:
        return tuple(m for m in self.moments if m.kind == "issue")

    @property
    def best_moment(self) -> Optional[Moment]:
        return next((m for m in self.moments if m.kind == "best"), None)


@dataclass(frozen=True)
class Exercise:
    id: str
    name: str
    fix_tags: tuple[str, ...]
    equipment: tuple[str, ...]
    sets_reps: str
    cues: tuple[str, ...]
    mistakes: tuple[str, ...]
    safety: str
    difficulty: int
    gif: str
    duration_s: float
    why: str
    easier: Optional[str] = None
    harder: Optional[str] = None

    def __post_init__(self) -> None:
        _snake(self.id, "exercise id")
        for name in ("fix_tags", "equipment", "cues", "mistakes"):
            _put(self, name, tuple(getattr(self, name)))
        _ensure(len(self.fix_tags) >= 1 and all(t in FIX_TAGS for t in self.fix_tags), f"exercise {self.id} needs known fix tags")
        _ensure(3 <= len(self.cues) <= 6, f"exercise {self.id} needs 3 to 6 cues")
        _ensure(len(self.mistakes) >= 2, f"exercise {self.id} needs at least 2 common mistakes")
        _ensure(bool(self.safety), f"exercise {self.id} needs a safety note")
        _ensure(0 < len(self.why) <= 160, f"exercise {self.id} why must be 1 to 160 characters")
        _put(self, "difficulty", _int(self.difficulty, f"exercise {self.id} difficulty"))
        _ensure(self.difficulty in (1, 2, 3), f"exercise {self.id} difficulty must be 1, 2 or 3")
        _ensure(not self.gif.startswith(("/", "\\")) and ":" not in self.gif, f"exercise {self.id} gif path must be relative")
        _put(self, "duration_s", float(_num(self.duration_s, f"exercise {self.id} duration")))
        for name in ("easier", "harder"):
            other = getattr(self, name)
            if other is not None:
                _snake(other, f"exercise {self.id} {name}")
                _ensure(other != self.id, f"exercise {self.id} cannot be its own {name} variant")


class ActionKind(str, Enum):
    SEEK = "seek"
    EXERCISE = "exercise"
    FINDING = "finding"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    ref: str
    arg: int = 0


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str

    def __post_init__(self) -> None:
        _ensure(self.speaker in ("user", "coach"), "turn speaker must be user or coach")


@dataclass(frozen=True)
class Reply:
    text: str
    actions: tuple[Action, ...]
    intent: str
    confidence: float
    suggestions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _put(self, "actions", tuple(self.actions))
        _put(self, "suggestions", tuple(self.suggestions))
        _put(self, "confidence", float(_num(self.confidence, "reply confidence")))
        _ensure(0.0 <= self.confidence <= 1.0, "reply confidence must be in [0, 1]")
        _ensure(len(self.suggestions) <= 4 and all(0 < len(s) <= 48 for s in self.suggestions), "reply suggestions are at most 4 chips of 1 to 48 characters")


Series = Mapping[str, Sequence[Optional[float]]]


@dataclass(frozen=True)
class Partial:
    metrics: Mapping[str, tuple[Metric, ...]] = field(default_factory=dict)
    series: Mapping[str, Series] = field(default_factory=dict)
    events: tuple[Event, ...] = ()
    findings: tuple[Finding, ...] = ()
    windows: tuple[Window, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Report:
    schema_version: int
    source: Mapping[str, Any]
    tier: Tier
    n_frames: int
    fps: float
    players: tuple[Role, ...]
    metrics: Mapping[str, tuple[Metric, ...]]
    series: Mapping[str, Series]
    events: tuple[Event, ...]
    findings: tuple[Finding, ...]
    windows: tuple[Window, ...] = ()
    video_id: str = ""
    revision: int = 0
    warnings: tuple[str, ...] = ()
    timestamps: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _ensure(self.schema_version == SCHEMA_VERSION, f"schema_version must be {SCHEMA_VERSION}")
        _put(self, "n_frames", _int(self.n_frames, "n_frames"))
        _put(self, "fps", float(_num(self.fps, "fps")))
        _put(self, "revision", _int(self.revision, "revision"))
        _ensure(self.n_frames > 0 and self.fps > 0, "report needs n_frames > 0 and fps > 0")
        for name in ("players", "events", "findings", "windows", "warnings"):
            _put(self, name, tuple(getattr(self, name)))
        _ensure(len(self.players) >= 1 and len(set(self.players)) == len(self.players), "report needs at least one distinct player")
        roles = {p.value for p in self.players}
        _ensure(set(self.metrics) <= roles, "metrics reference a player that is not in the report")
        _ensure(set(self.series) <= roles, "series reference a player that is not in the report")
        _put(self, "metrics", {role: tuple(found) for role, found in self.metrics.items()})
        for role, found in self.metrics.items():
            names = [m.name for m in found]
            _ensure(len(names) == len(set(names)), f"metrics for {role} repeat a name")
        clean: dict[str, dict[str, tuple[Optional[float], ...]]] = {}
        for role, named in self.series.items():
            clean[role] = {}
            for name, values in named.items():
                _snake(name, "series name")
                try:
                    got = as_series(values)
                except ValueError as err:
                    raise ValueError(f"series {role}.{name}: {err}") from None
                _ensure(len(got) == self.n_frames, f"series {role}.{name} has {len(got)} values, expected {self.n_frames}")
                clean[role][name] = got
        _put(self, "series", clean)
        stamps = tuple(float(_num(t, "timestamp")) for t in self.timestamps)
        _ensure(not stamps or len(stamps) == self.n_frames, f"timestamps has {len(stamps)} values, expected {self.n_frames}")
        _ensure(all(b >= a for a, b in zip(stamps, stamps[1:])), "timestamps must not go backwards")
        _put(self, "timestamps", stamps)
        unknown = [w for w in self.warnings if w not in WARNINGS]
        _ensure(not unknown, f"unknown warning codes {unknown}")
        for event in self.events:
            _ensure(event.player in self.players and event.frame < self.n_frames, f"event {event.kind}@{event.frame} is outside the report")
        for window in self.windows:
            _ensure(window.end_frame < self.n_frames, "window ends after the last frame")
        seen: set[tuple[str, str]] = set()
        for finding in self.findings:
            key = (finding.id, finding.player.value)
            _ensure(key not in seen, f"duplicate finding {key}")
            seen.add(key)
            _ensure(finding.player in self.players, f"finding {finding.id} is for a player that is not in the report")
            for moment in finding.moments:
                _ensure(moment.end_frame < self.n_frames, f"finding {finding.id} has a moment past the last frame")
            have = self.series.get(finding.player.value, {})
            missing = [s for s in finding.overlay_series if s not in have]
            _ensure(not missing, f"finding {finding.id} overlays unknown series {missing}")

    def time_of(self, frame: int) -> float:
        return self.timestamps[frame] if self.timestamps else frame / self.fps

    def top_findings(self, n: int = 3, player: Optional[Role] = None) -> tuple[Finding, ...]:
        pool = [f for f in self.findings if f.severity in (Severity.IMPROVE, Severity.PRIORITY) and (player is None or f.player == player)]
        pool.sort(key=lambda f: (-SEVERITY_RANK[f.severity], -f.score, -sum(m.confidence for m in f.metrics) / len(f.metrics), f.id))
        return tuple(pool[:n])

    def strengths(self, n: int = 3, player: Optional[Role] = None) -> tuple[Finding, ...]:
        pool = [f for f in self.findings if f.severity == Severity.STRENGTH and (player is None or f.player == player)]
        pool.sort(key=lambda f: (-sum(m.confidence for m in f.metrics) / len(f.metrics), f.id))
        return tuple(pool[:n])

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


def to_plain(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Mapping):
        return {(k.value if isinstance(k, Enum) else str(k)): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if hasattr(obj, "tolist"):
        return to_plain(obj.tolist())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _is_zero_int(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool) and int(value) == 0


def _is_zero_real(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value) and float(value) == 0.0


def _is_finite_real(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def normalize_metric_record(record: Mapping[str, Any], schema_version: int) -> dict[str, Any]:
    """Copy a metric record onto the current observation rule. Does not change `record`.

    Pass the coaching-report version, not the pose or bundle version. Version 3 is
    copied unchanged. Version 2 turns a zero-observation reading with confidence 0
    and no best into value None. Any other old shape is rejected instead of guessed.
    """
    version = _int(schema_version, "schema_version")
    _ensure(
        version in (LEGACY_SCHEMA_VERSION, SCHEMA_VERSION),
        f"schema_version must be {LEGACY_SCHEMA_VERSION} or {SCHEMA_VERSION}, got {version}",
    )
    out = dict(record)
    if version == SCHEMA_VERSION:
        return out
    name = out.get("name", "?")
    if "value" not in out:
        raise ValueError(f"metric {name} is missing value")
    value = out["value"]
    n_obs = out.get("n_obs", 1)
    if not _is_zero_int(n_obs):
        if value is None:
            raise ValueError(f"metric {name} value is null but n_obs is not 0")
        return out
    confidence = out.get("confidence")
    best = out.get("best")
    if "confidence" not in out or not _is_zero_real(confidence) or best is not None:
        raise ValueError(f"metric {name} has n_obs 0 but confidence or best is set; refusing to guess")
    if value is not None and not _is_finite_real(value):
        raise ValueError(f"metric {name} has a contradictory legacy value {value!r}")
    out["value"] = None
    return out


def _metric(d: Mapping[str, Any], schema_version: int = SCHEMA_VERSION) -> Metric:
    fields = normalize_metric_record(d, schema_version)
    target = fields.get("target")
    return Metric(
        fields["name"], fields["label"], fields["value"], fields["unit"], Basis(fields["basis"]), fields["confidence"],
        tuple(target) if target else None, fields.get("series"), fields.get("n_obs", 1), fields.get("best"),
    )


def _mark(d: Mapping[str, Any]) -> Mark:
    return Mark(d["kind"], d["frame"], tuple(tuple(p) for p in d["points"]), d.get("label", ""), d.get("value"), d.get("style", "actual"))


def _moment(d: Mapping[str, Any]) -> Moment:
    return Moment(
        Role(d["player"]), d["start_frame"], d["key_frame"], d["end_frame"], d["label"],
        d.get("kind", "issue"), tuple(_mark(m) for m in d.get("marks", ())),
    )


def _event(d: Mapping[str, Any]) -> Event:
    return Event(d["kind"], Role(d["player"]), d["frame"], d["confidence"], dict(d.get("data", {})))


def _finding(d: Mapping[str, Any], schema_version: int = SCHEMA_VERSION) -> Finding:
    return Finding(
        d["id"], Category(d["category"]), Role(d["player"]), Severity(d["severity"]), d["headline"], d["explain"],
        tuple(_metric(m, schema_version) for m in d["metrics"]), tuple(_moment(m) for m in d["moments"]), Tier(d["tier"]),
        tuple(d.get("focus_joints", ())), tuple(d.get("overlay_series", ())), tuple(d.get("fix_tags", ())),
        d.get("score", 0.0),
    )


def report_from_dict(d: Mapping[str, Any]) -> Report:
    version = _int(d["schema_version"], "schema_version")
    _ensure(
        version in (LEGACY_SCHEMA_VERSION, SCHEMA_VERSION),
        f"schema_version must be {LEGACY_SCHEMA_VERSION} or {SCHEMA_VERSION}, got {version}",
    )
    return Report(
        SCHEMA_VERSION, dict(d["source"]), Tier(d["tier"]), d["n_frames"], d["fps"],
        tuple(Role(p) for p in d["players"]),
        {role: tuple(_metric(m, version) for m in ms) for role, ms in d["metrics"].items()},
        {role: {name: tuple(vals) for name, vals in named.items()} for role, named in d["series"].items()},
        tuple(_event(e) for e in d["events"]), tuple(_finding(f, version) for f in d["findings"]),
        tuple(Window(w["start_frame"], w["end_frame"], w["reason"]) for w in d.get("windows", ())),
        d.get("video_id", ""), d.get("revision", 0), tuple(d.get("warnings", ())), tuple(d.get("timestamps", ())),
    )
