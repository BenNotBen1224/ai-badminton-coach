"""Load and check the chat knowledge base in coach/kb/*.json.

Every file is plain JSON so the coach improves by editing data. `load_kb` fails
loudly on a missing key, an unknown finding id, fix tag or intent, or a number
in reply text (numbers must come from the report or the library).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from coach.exercises.library import EQUIPMENT
from coach.schema import FINDING_CATALOG, FIX_TAGS, Category, Exercise

KB_DIR = Path(__file__).resolve().parent.parent / "kb"

INTENTS = (
    "greeting", "thanks", "help", "summary", "strengths", "explain", "how_to_fix", "exercise_request",
    "exercise_detail", "exercise_why", "exercise_adjust", "show_moment", "compare_players", "metric_question",
    "practice_plan", "terms", "tactics_tip", "limits", "progress", "safety", "set_level", "affirm", "deny", "next",
    "simplify", "out_of_scope", "unknown",
)

POSITIONAL = frozenset({Category.FOOTWORK, Category.TACTICS})


@dataclass(frozen=True)
class KnowledgeBase:
    glossary: Mapping[str, Mapping[str, Any]]
    fix_guides: Mapping[str, Mapping[str, Any]]
    tactics: Mapping[str, Mapping[str, Any]]
    limits: Mapping[str, Any]
    safety: Mapping[str, Any]
    lexicon: Mapping[str, Any]
    exercise_aliases: Mapping[str, Any]


def _fail(where: str, message: str) -> None:
    raise ValueError(f"kb {where}: {message}")


def _need(where: str, row: Any, keys: tuple[str, ...]) -> None:
    if not isinstance(row, Mapping):
        _fail(where, "must be an object")
    missing = [k for k in keys if k not in row]
    if missing:
        _fail(where, f"missing {missing}")


def _text(where: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        _fail(where, "must be non-empty text")
    if re.search(r"\d", re.sub(r"\b[23]D\b", "", value)):
        _fail(where, f"reply text must not hold a number: {value!r}")


def _words(where: str, value: Any) -> None:
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        _fail(where, "must be a list of non-empty strings")


def _tags(where: str, tags: Any) -> None:
    if not isinstance(tags, list) or not tags or any(t not in FIX_TAGS for t in tags):
        _fail(where, f"unknown fix tags {tags!r}")


def _check(kb: KnowledgeBase) -> None:
    for key, row in kb.glossary.items():
        _need(f"glossary.{key}", row, ("name", "aliases", "define", "simple"))
        _words(f"glossary.{key}.aliases", row["aliases"])
        for name in ("define", "simple"):
            _text(f"glossary.{key}.{name}", row[name])

    if set(kb.fix_guides) != set(FINDING_CATALOG):
        _fail("fix_guides", f"ids must equal FINDING_CATALOG, diff {sorted(set(kb.fix_guides) ^ set(FINDING_CATALOG))}")
    for key, row in kb.fix_guides.items():
        where = f"fix_guides.{key}"
        _need(where, row, ("name", "topics", "why", "cues", "feel", "reasons", "progression", "fix_tags", "simple"))
        if not row["topics"] or not all(isinstance(w, (int, float)) and w > 0 for w in row["topics"].values()):
            _fail(where, "topics need positive weights")
        _words(f"{where}.cues", row["cues"])
        if not 3 <= len(row["cues"]) <= 5:
            _fail(where, "needs 3 to 5 cues")
        _words(f"{where}.reasons", row["reasons"])
        for text in (row["name"], row["why"], row["feel"], row["progression"], row["simple"], *row["cues"], *row["reasons"]):
            _text(where, text)
        _tags(where, row["fix_tags"])

    for key, row in kb.tactics.items():
        where = f"tactics.{key}"
        _need(where, row, ("findings", "title", "text"))
        if not row["findings"] or any(FINDING_CATALOG.get(f, (None,))[0] not in POSITIONAL for f in row["findings"]):
            _fail(where, f"findings must be footwork or tactics ids, got {row['findings']}")
        _text(where, row["title"])
        _text(where, row["text"])
        if "shuttle" not in row["text"] or "not tracked" not in row["text"]:
            _fail(where, "text must say plainly that the shuttle is not tracked")

    _need("limits", kb.limits, ("can", "cannot", "preview", "general"))
    _text("limits.can", kb.limits["can"])
    for key, row in kb.limits["cannot"].items():
        _need(f"limits.cannot.{key}", row, ("keys", "text"))
        _words(f"limits.cannot.{key}.keys", row["keys"])
        if "asks" in row:
            # Cues that ask for this quantity directly ("how fast was my smash"); they always get the limits reply.
            _words(f"limits.cannot.{key}.asks", row["asks"])
            if any(len(p.split()) < 2 or p.split()[0] == "*" for p in row["asks"]):
                _fail(f"limits.cannot.{key}.asks", "each cue needs two or more words and must not start with *")
        _text(f"limits.cannot.{key}", row["text"])
    _need("limits.preview", kb.limits["preview"], ("keys", "text_preview", "text_refined"))
    _text("limits.preview", kb.limits["preview"]["text_preview"])
    _text("limits.preview", kb.limits["preview"]["text_refined"])
    _need("limits.general", kb.limits["general"], ("keys", "text"))
    _words("limits.general.keys", kb.limits["general"]["keys"])
    _text("limits.general", kb.limits["general"]["text"])

    lists = ("always", "stems", "phrases", "joint_concern", "conditional_any", "conditional_joint", "body_parts",
             "joint_parts", "safe_phrases", "non_english", "non_english_tokens")
    _need("safety", kb.safety, ("text", "suggestions") + lists)
    _text("safety.text", kb.safety["text"])
    if not isinstance(kb.safety["suggestions"], list):
        _fail("safety.suggestions", "must be a list")
    for key in lists:
        _words(f"safety.{key}", kb.safety[key])

    lex = kb.lexicon
    _need("lexicon", lex, ("token_map", "fillers", "contractions", "equipment", "equipment_groups", "equipment_ops",
                           "levels", "players", "followups", "aspects", "adjust", "quantity", "bare", "best", "judge",
                           "known_words", "stop_words", "intents"))
    unknown = set(lex["intents"]) - set(INTENTS)
    if unknown:
        _fail("lexicon.intents", f"unknown intents {sorted(unknown)}")
    # The classifier skips contractions for text without an apostrophe, so every pattern must contain one.
    for pattern, _ in lex["contractions"]:
        if "'" not in pattern:
            _fail("lexicon.contractions", f"pattern without an apostrophe {pattern!r}")
    for name, cues in (*((f"intents.{i}", c) for i, c in lex["intents"].items()), ("judge", dict.fromkeys(lex["judge"], 1))):
        for pattern, weight in cues.items():
            parts = pattern.strip().lstrip("^").rstrip("$").split()
            if pattern != pattern.strip() or not parts or "*" in (parts[0], parts[-1]):
                _fail(f"lexicon.{name}", f"bad cue {pattern!r}")
            if not isinstance(weight, (int, float)) or weight <= 0:
                _fail(f"lexicon.{name}", f"bad weight for {pattern!r}: {weight!r}")
    if not set(lex["equipment"]) <= EQUIPMENT:
        _fail("lexicon.equipment", f"unknown equipment {sorted(set(lex['equipment']) - EQUIPMENT)}")
    for group, items in lex["equipment_groups"].items():
        if not set(items) <= EQUIPMENT:
            _fail(f"lexicon.equipment_groups.{group}", f"unknown equipment {items}")
    for key, want in (("equipment_ops", {"none", "all", "only", "remove"}), ("levels", {"1", "2", "3"}),
                      ("adjust", {"easier", "harder", "another"}), ("followups", {"flip", "self", "near", "far"}),
                      ("bare", {"how", "why", "huh"})):
        if set(lex[key]) != want:
            _fail(f"lexicon.{key}", f"needs exactly {sorted(want)}")

    al = kb.exercise_aliases
    _need("exercise_aliases", al, ("library", "weak", "weak_before", "weak_block", "missing"))
    for key, row in al["missing"].items():
        _need(f"exercise_aliases.missing.{key}", row, ("name", "aliases", "fix_tags"))
        _tags(f"exercise_aliases.missing.{key}", row["fix_tags"])
    overlap = set(al["missing"]) & set(al["library"])
    if overlap:
        _fail("exercise_aliases", f"ids both in library and missing {sorted(overlap)}")


def load_kb(directory: Path = KB_DIR) -> KnowledgeBase:
    data = {}
    for name in ("glossary", "fix_guides", "tactics", "limits", "safety", "lexicon", "exercise_aliases"):
        path = Path(directory) / f"{name}.json"
        try:
            data[name] = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ValueError(f"kb file missing: {path}") from None
    kb = KnowledgeBase(**data)
    _check(kb)
    return kb


def check_kb_against_library(kb: KnowledgeBase, library: Mapping[str, Exercise]) -> None:
    """The alias file must cover exactly the real library: every id named, no alias for an unknown id."""
    named = kb.exercise_aliases["library"]
    no_alias = sorted(i for i in library if not named.get(i))
    if no_alias:
        raise ValueError(f"library ids without aliases: {no_alias}")
    unknown = sorted(i for i in named if i not in library)
    weak = sorted({i for ids in kb.exercise_aliases["weak"].values() for i in ids if i not in library})
    if unknown or weak:
        raise ValueError(f"aliases point at unknown ids: {unknown + weak}")
    clash = sorted(i for i in kb.exercise_aliases["missing"] if i in library)
    if clash:
        raise ValueError(f"missing-exercise aliases name library ids: {clash}")
