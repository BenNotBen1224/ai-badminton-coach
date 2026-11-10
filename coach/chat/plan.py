"""Build a short practice plan from a report's top findings.

Pure: a report and an exercise library come in, a frozen Plan comes out.
`fits` and `step` are shared with the chat engine so that "easier",
"harder" and equipment changes always keep the finding's goal, level and gear.
`avoid` names exercises the player found too hard; `easier_dose` is the only way one comes back.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Collection, Mapping, Optional, Sequence

from coach.exercises.library import EQUIPMENT, for_tag, variant
from coach.schema import FIX_TAGS, Exercise, Finding, Report, Role

WARM_UP = (
    "Warm up for five to ten minutes first: easy skipping or jogging, leg swings, "
    "arm circles and a few slow lunges."
)
SAFETY = (
    "Stop any exercise that causes sharp pain, dizziness or chest pain, and see a doctor or "
    "physiotherapist before you train through pain. Keep every rep slow and controlled before adding speed, "
    "and rest a day between sessions."
)
NO_PLAN = ("There is nothing to plan yet: either nothing in this clip clearly needs work for this player, or no "
           "exercise for it fits this level and equipment.")
FOCUS = (
    "Start easy and learn the movements.",
    "Step up where the first week felt easy.",
    "Keep the harder versions smooth and controlled.",
)
REPEAT = "Repeat last week and aim for smoother, steadier reps."
EASED = " - easier dose, because the full one was too hard"
_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")
_DOSE = re.compile(r"(\d+) x (\d+)((?: .*)?)")


def number_word(n: int) -> str:
    return _WORDS[n] if 0 <= n < len(_WORDS) else "several"


def easier_dose(dose: str) -> str:
    """One set fewer and about half the reps or seconds: "3 x 30 seconds" becomes "2 x 15 seconds"."""
    found = _DOSE.fullmatch(dose)
    if found is not None:
        sets, amount, rest = int(found[1]), int(found[2]), found[3]
        half = max(5, amount // 10 * 5) if rest.lstrip().startswith("second") else max(1, (amount + 1) // 2)
        eased = f"{max(1, sets - 1)} x {min(half, amount)}{rest}"
        if eased != dose:
            return eased
    return f"{dose}, with shorter holds or fewer reps"


def fits(exercise: Exercise, tag: str, max_difficulty: int = 3, equipment: Optional[Sequence[str]] = None) -> bool:
    """True when the exercise trains `tag` within the level and the gear on hand (None = any gear)."""
    return (
        tag in exercise.fix_tags
        and exercise.difficulty <= max_difficulty
        and (equipment is None or set(exercise.equipment) <= set(equipment))
    )


def step(
    library: Mapping[str, Exercise],
    exercise: Exercise,
    which: str,
    tag: str,
    max_difficulty: int = 3,
    equipment: Optional[Sequence[str]] = None,
    avoid: Collection[str] = (),
) -> Optional[Exercise]:
    """The nearest easier or harder exercise that still fits the goal, level and gear, never one in `avoid`.

    A library edge alone is not trusted: the chain is walked until a variant fits,
    then any other exercise for the tag at a strictly lower or higher difficulty is tried.
    """
    if which not in ("easier", "harder"):
        raise ValueError(f"step must be easier or harder, got {which!r}")
    seen = {exercise.id}
    current = exercise
    while getattr(current, which) in library and getattr(current, which) not in seen:
        current = variant(library, current, which)
        seen.add(current.id)
        if current.id not in avoid and fits(current, tag, max_difficulty, equipment):
            return current
    if tag not in FIX_TAGS:
        return None
    pool = [e for e in for_tag(library, tag, len(library), max_difficulty, equipment)
            if e.id not in seen and e.id not in avoid]
    if which == "easier":
        pool = sorted((e for e in pool if e.difficulty < exercise.difficulty), key=lambda e: -e.difficulty)
    else:
        pool = sorted((e for e in pool if e.difficulty > exercise.difficulty), key=lambda e: e.difficulty)
    return pool[0] if pool else None


@dataclass(frozen=True)
class Block:
    exercise_id: str
    name: str
    dose: str
    finding_id: str
    reason: str


@dataclass(frozen=True)
class Session:
    blocks: tuple[Block, ...]


@dataclass(frozen=True)
class Week:
    focus: str
    sessions: tuple[Session, ...]


@dataclass(frozen=True)
class Plan:
    player: Role
    level: int
    equipment: Optional[tuple[str, ...]]
    findings: tuple[str, ...]
    warm_up: str
    weeks: tuple[Week, ...]
    safety: str

    def exercise_ids(self) -> tuple[str, ...]:
        found = (b.exercise_id for w in self.weeks for s in w.sessions for b in s.blocks)
        return tuple(dict.fromkeys(found))

    def to_dict(self) -> dict[str, Any]:
        return {
            "player": self.player.value,
            "level": self.level,
            "equipment": None if self.equipment is None else list(self.equipment),
            "findings": list(self.findings),
            "warm_up": self.warm_up,
            "weeks": [
                {
                    "week": w,
                    "focus": week.focus,
                    "sessions": [
                        {"session": s, "blocks": [asdict(b) for b in session.blocks]}
                        for s, session in enumerate(week.sessions, 1)
                    ],
                }
                for w, week in enumerate(self.weeks, 1)
            ],
            "safety": self.safety,
        }

    def to_text(self) -> str:
        if not self.weeks:
            return f"{NO_PLAN}\nSafety: {self.safety}"
        lines = [f"Practice plan for the {self.player.value} player", f"Warm-up: {self.warm_up}"]
        for w, week in enumerate(self.weeks, 1):
            lines.append(f"Week {w}: {week.focus}")
            for s, session in enumerate(week.sessions, 1):
                lines.append(f"  Session {s}:")
                lines.extend(f"    - {b.name}, {b.dose}. {b.reason}" for b in session.blocks)
        lines.append(f"Safety: {self.safety}")
        return "\n".join(lines)


def _count(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be a whole number from {low} to {high}, got {value!r}")
    return value


def _role(report: Report, player: Any) -> Role:
    if player is None:
        return report.players[0]
    try:
        role = Role(player)
    except ValueError:
        raise ValueError(f"unknown player {player!r}") from None
    if role not in report.players:
        raise ValueError(f"player {role.value} is not in this report")
    return role


def _gear(equipment: Any) -> Optional[tuple[str, ...]]:
    if equipment is None:
        return None
    if isinstance(equipment, str):
        equipment = (equipment,)
    try:
        items = tuple(dict.fromkeys(equipment))
    except TypeError:
        raise ValueError(f"equipment must be a list of names, got {equipment!r}") from None
    unknown = [item for item in items if item not in EQUIPMENT]
    if unknown:
        raise ValueError(f"unknown equipment {unknown}")
    return tuple(sorted(items))


def _avoid(avoid: Any) -> frozenset[str]:
    if avoid is None:
        return frozenset()
    if isinstance(avoid, str):
        avoid = (avoid,)
    try:
        items = frozenset(avoid)
    except TypeError:
        raise ValueError(f"avoid must be a list of exercise ids, got {avoid!r}") from None
    if not all(isinstance(item, str) for item in items):
        raise ValueError(f"avoid must be a list of exercise ids, got {avoid!r}")
    return items


def _choose(top: list[tuple], limit: int = 3) -> list[tuple]:
    """(finding, options) pairs in rank order, at most two per category while others are left."""
    picked: list[tuple] = []
    per_category: Counter = Counter()
    for pair in top:
        if len(picked) < limit and per_category[pair[0].category] < 2:
            picked.append(pair)
            per_category[pair[0].category] += 1
    for pair in top:
        if len(picked) < limit and pair not in picked:
            picked.append(pair)
    return picked


def _candidates(
    library: Mapping[str, Exercise], finding: Finding, level: int, gear, avoid: Collection[str] = ()
) -> list[tuple[Exercise, str]]:
    """Exercises for any of the finding's tags, easiest first, each with the tag it trains.

    Avoided exercises are left out while anything else fits; when nothing else does, only they remain.
    """
    found: dict[str, tuple] = {}
    for index, tag in enumerate(finding.fix_tags):
        if tag not in FIX_TAGS:
            continue
        for exercise in for_tag(library, tag, len(library), level, gear):
            found.setdefault(exercise.id, (exercise.difficulty, index, len(found), exercise, tag))
    ranked = [(row[3], row[4]) for row in sorted(found.values(), key=lambda row: row[:3])]
    return [pair for pair in ranked if pair[0].id not in avoid] or ranked


def _order(rows: list[tuple]) -> list[Block]:
    """Spread categories so that two blocks for the same kind of work do not sit side by side."""
    left = list(rows)
    out: list[tuple] = []
    while left:
        counts = Counter(row[0] for row in left)
        prev = out[-1] if out else None

        def rank(i: int) -> tuple:
            category, first_tag, _ = left[i]
            clash = prev is not None and (category == prev[0] or first_tag == prev[1])
            return (clash, -counts[category], i)

        out.append(left.pop(min(range(len(left)), key=rank)))
    return [row[2] for row in out]


def build_plan(
    report: Report,
    library: Mapping[str, Exercise],
    *,
    player: Any,
    weeks: int = 2,
    sessions_per_week: int = 3,
    level: int = 3,
    equipment: Optional[Sequence[str]] = None,
    avoid: Optional[Collection[str]] = None,
) -> Plan:
    """A plan over the player's top findings. player=None means the report's first player.

    `avoid` lists exercise ids the player found too hard. They are replaced where another exercise fits;
    otherwise they stay at an easier dose and do not step up in later weeks.
    """
    role = _role(report, player)
    weeks = _count(weeks, "weeks", 1, 12)
    sessions_per_week = _count(sessions_per_week, "sessions_per_week", 1, 7)
    level = _count(level, "level", 1, 3)
    gear = _gear(equipment)
    avoid = _avoid(avoid)
    top = [(f, _candidates(library, f, level, gear, avoid))
           for f in report.top_findings(len(report.findings), role) if f.fix_tags]
    chosen = _choose([(f, options) for f, options in top if options])
    built: list[Week] = []
    for w in range(weeks if chosen else 0):
        sessions = []
        for s in range(sessions_per_week):
            used: set[str] = set()
            rows = []
            for finding, options in chosen:
                exercise, tag = options[:2][s % min(2, len(options))]
                for _ in range(0 if exercise.id in avoid else w):
                    harder = step(library, exercise, "harder", tag, level, gear, avoid)
                    if harder is None:
                        break
                    exercise = harder
                if exercise.id in used:
                    spare = [(e, t) for e, t in options if e.id not in used]
                    if not spare:
                        continue
                    exercise, tag = spare[0]
                used.add(exercise.id)
                eased = exercise.id in avoid
                reason = f"{finding.headline} - goal: {FIX_TAGS[tag]}{EASED if eased else ''}"
                dose = easier_dose(exercise.sets_reps) if eased else exercise.sets_reps
                block = Block(exercise.id, exercise.name, dose, finding.id, reason)
                rows.append((finding.category, exercise.fix_tags[0], block))
            sessions.append(Session(tuple(_order(rows))))
        same = bool(built) and built[-1].sessions == tuple(sessions)
        built.append(Week(REPEAT if same else FOCUS[min(w, len(FOCUS) - 1)], tuple(sessions)))
    return Plan(role, level, gear, tuple(f.id for f, _ in chosen), WARM_UP, tuple(built), SAFETY)
