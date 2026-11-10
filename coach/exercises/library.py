"""Load the exercise library and pick exercises for a fix tag.

The JSON file is the boundary. `for_tag` and `variant` are pure lookups over
an already loaded mapping, so a caller can pass a stub.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional

from coach.schema import FIX_TAGS, Exercise

ROOT = Path(__file__).resolve().parent
LIBRARY_PATH = ROOT / "library.json"

# Bodyweight is the empty tuple, not a name in this set.
EQUIPMENT = frozenset({
    "wall",
    "box",
    "mat",
    "dumbbell",
    "kettlebell",
    "band",
    "rope",
    "cable",
    "bench",
})


def _exercise(row: Mapping) -> Exercise:
    equipment = tuple(row.get("equipment") or ())
    unknown = [item for item in equipment if item not in EQUIPMENT]
    if unknown:
        raise ValueError(f"exercise {row.get('id')} has unknown equipment {unknown}")
    return Exercise(
        id=row["id"],
        name=row["name"],
        fix_tags=tuple(row["fix_tags"]),
        equipment=equipment,
        sets_reps=row["sets_reps"],
        cues=tuple(row["cues"]),
        mistakes=tuple(row["mistakes"]),
        safety=row["safety"],
        difficulty=row["difficulty"],
        gif=row["gif"],
        duration_s=row["duration_s"],
        why=row["why"],
        easier=row.get("easier"),
        harder=row.get("harder"),
    )


def check_links(library: Mapping[str, Exercise]) -> None:
    for exercise in library.values():
        for which in ("easier", "harder"):
            other_id = getattr(exercise, which)
            if other_id is None:
                continue
            other = library.get(other_id)
            if other is None:
                raise ValueError(f"exercise {exercise.id} {which} {other_id} is not in the library")
            if which == "easier" and other.difficulty > exercise.difficulty:
                raise ValueError(f"exercise {exercise.id} easier {other_id} is a higher difficulty")
            if which == "harder" and other.difficulty < exercise.difficulty:
                raise ValueError(f"exercise {exercise.id} harder {other_id} is a lower difficulty")
            if not set(exercise.fix_tags) & set(other.fix_tags):
                raise ValueError(f"exercise {exercise.id} {which} {other_id} shares no fix tag")


def load_library(path: Path = LIBRARY_PATH) -> dict[str, Exercise]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    library: dict[str, Exercise] = {}
    for row in rows:
        exercise = _exercise(row)
        if exercise.id in library:
            raise ValueError(f"duplicate exercise id {exercise.id}")
        library[exercise.id] = exercise
    check_links(library)
    return library


def for_tag(
    library: Mapping[str, Exercise],
    tag: str,
    limit: int = 3,
    max_difficulty: int = 3,
    equipment_ok=None,
) -> list[Exercise]:
    if tag not in FIX_TAGS:
        raise ValueError(f"unknown fix tag {tag}")
    picked = [
        exercise
        for exercise in library.values()
        if tag in exercise.fix_tags
        and exercise.difficulty <= max_difficulty
        and (equipment_ok is None or all(item in equipment_ok for item in exercise.equipment))
    ]
    picked.sort(key=lambda exercise: (exercise.fix_tags[0] != tag, exercise.difficulty, exercise.id))
    return picked[:limit]


def variant(library: Mapping[str, Exercise], exercise: Exercise, which: str) -> Optional[Exercise]:
    if which not in ("easier", "harder"):
        raise ValueError(f"variant must be easier or harder, got {which}")
    other_id = getattr(exercise, which)
    if other_id is None:
        return None
    return library[other_id]


def gif_path(exercise: Exercise) -> Path:
    path = ROOT / exercise.gif
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
