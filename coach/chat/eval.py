"""Run a question file through the chat coach; print intent and entity accuracy and reply times.

    python -m coach.chat.eval QUESTIONS [--export OUT] [--split all|tune|check] [--report full|preview|real]

QUESTIONS uses the held-out format: a JSON list of {"id", "text", "intent", "context", "history", "entities",
"note"}. The engine sees only "text", "context" and "history". "intent", "entities" and "note" are read after
every reply has been made, for scoring. Items that share a "dialogue" key run in file order as one conversation:
the first uses its own context and history, later ones get the context and history the engine itself produced.

--export writes what the coach actually said, one row per question, so an outside grader can check the replies.
--split runs a fixed 70% ("tune") or 30% ("check") part of the file, chosen by a hash of the dialogue or id.
--report picks the report the coach answers about: the fixture with every finding (the held-out set assumes it),
its preview copy, or the real analysed dev clip.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
REAL_DOC = ROOT / "fast" / "runs" / "step4-rebalanced-dev" / "outputs" / "01-r1.json"
SOURCES = ("coach/__init__.py", "coach/chat/*.py", "coach/kb/*.json", "coach/exercises/__init__.py",
           "coach/exercises/library.py", "coach/exercises/library.json", "coach/schema.py", "coach/tests/__init__.py",
           "coach/tests/fixture_report.py")
ROW_KEYS = ("id", "intent", "entities", "text", "actions", "suggestions", "elapsed_ms", "error")
ENTITIES = ("finding_id", "exercise", "player")
ENTITY_SOURCE = (
    "Predicted intent and entities come from the Parse that Coach.reply_traced returned in the same call that "
    "made the reply: intent = Reply.intent (equal to Parse.intent), finding_id = Parse.finding_id, exercise = "
    "Parse.exercise_id (a library id) or else Parse.missing (the exercise_aliases id of a known exercise the "
    "library does not ship), player = Parse.player ('me' and 'my' resolve to the selected player), topic = "
    "Parse.topic_word. Parse holds only what the message itself names, not values carried in the context. "
    "Expected labels and notes are never passed to the engine.")
TIMING_NOTE = (
    "elapsed_ms times one Coach.reply_traced call with time.perf_counter on an engine already built in this "
    "process. construction_ms times Coach(report, library) cold: loading the knowledge base and building the "
    "classifier. Imports, library loading and report building are timed separately.")


def split_of(item: Mapping[str, Any]) -> str:
    """'tune' for a fixed 70% of the file, 'check' for the rest. A dialogue stays together."""
    key = str(item.get("dialogue") or item.get("id"))
    return "tune" if zlib.crc32(key.encode("utf-8")) % 10 < 7 else "check"


def load_questions(path: Path) -> tuple[str, list[dict]]:
    """The SHA256 of the file's exact bytes, taken before anything runs, and its items."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    items = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError(f"{path}: expected a JSON list of question objects")
    return digest, items


def question_id(item: Mapping[str, Any], n: int) -> str:
    return str(item.get("id", f"#{n + 1}"))


def _value(x: Any) -> Any:
    return getattr(x, "value", x)


def run(coach: Any, items: Sequence[Mapping[str, Any]]) -> list[dict]:
    """One export row per item, in order. Only text, context and history reach the engine."""
    from coach.chat.engine import context_to_dict

    rows, state = [], {}
    for n, item in enumerate(items):
        text = item.get("text")
        dialogue = item.get("dialogue")
        if dialogue is not None and dialogue in state:
            context, history = state[dialogue]
        else:
            context, history = copy.deepcopy(item.get("context")), copy.deepcopy(item.get("history"))
        row = dict.fromkeys(ROW_KEYS)
        row.update(id=question_id(item, n), actions=[], suggestions=[])
        started = time.perf_counter()
        try:
            reply, ctx, parse = coach.reply_traced(history, text, context)
            row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
            next_context = context_to_dict(ctx)
        except Exception as e:  # a failed reply is evidence too: keep the row with its error
            if row["elapsed_ms"] is None:
                row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
            row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
            continue
        row.update(
            intent=reply.intent,
            entities={"finding_id": parse.finding_id, "exercise": parse.exercise_id or parse.missing,
                      "player": _value(parse.player), "topic": parse.topic_word},
            text=reply.text,
            actions=[{"kind": _value(a.kind), "ref": a.ref, "arg": a.arg} for a in reply.actions],
            suggestions=list(reply.suggestions))
        rows.append(row)
        if dialogue is not None:
            said = text if isinstance(text, str) else ""
            before = list(history) if isinstance(history, list) else []
            state[dialogue] = (next_context, [*before, {"speaker": "user", "text": said},
                                              {"speaker": "coach", "text": reply.text}])
    return rows


def _norm(name: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", name.lower()).split())


def exercise_names(library: Mapping[str, Any], kb: Any) -> dict[str, str]:
    """Normalised exercise id, name or alias -> id, to read the natural names a question file expects."""
    names: dict[str, str] = {}
    aliases = kb.exercise_aliases
    for ex_id, ex in library.items():
        names[_norm(ex_id)] = ex_id
        names[_norm(ex.name)] = ex_id
    for ex_id, row in aliases["missing"].items():
        names.setdefault(_norm(ex_id), ex_id)
        names.setdefault(_norm(row["name"]), ex_id)
    for ex_id, words in aliases["library"].items():
        for word in words:
            names.setdefault(_norm(word), ex_id)
    for ex_id, row in aliases["missing"].items():
        for word in row["aliases"]:
            names.setdefault(_norm(word), ex_id)
    return names


def _expected(item: Mapping[str, Any], names: Mapping[str, str]) -> dict[str, Optional[str]]:
    want = item.get("entities") if isinstance(item.get("entities"), Mapping) else {}
    exercise = want.get("exercise")
    if isinstance(exercise, str):
        exercise = names.get(_norm(exercise), "?" + exercise)
    return {"finding_id": want.get("finding_id"), "exercise": exercise, "player": want.get("player")}


def percentile(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)] if ordered else float("nan")


def score(items: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], names: Mapping[str, str]) -> dict:
    confusion: dict[str, Counter] = defaultdict(Counter)
    entities = {k: {"expected": 0, "right": 0, "spurious": 0} for k in ENTITIES}
    misses = []
    for item, row in zip(items, rows):
        want, got = item.get("intent"), row["intent"]
        confusion[str(want)][str(got)] += 1
        expected = _expected(item, names)
        predicted = row["entities"] or {}
        wrong = [] if want == got else ["intent"]
        for k in ENTITIES:
            if expected[k] is not None:
                entities[k]["expected"] += 1
                if predicted.get(k) == expected[k]:
                    entities[k]["right"] += 1
                else:
                    wrong.append(k)
            elif predicted.get(k) is not None:
                entities[k]["spurious"] += 1
        if wrong or row["error"]:
            misses.append({"id": row["id"], "wrong": wrong, "text": item.get("text"), "want": want, "got": got,
                           "expected": expected, "predicted": predicted, "error": row["error"]})
    right = sum(c[w] for w, c in confusion.items())
    times = [r["elapsed_ms"] for r in rows if r["error"] is None and r["elapsed_ms"] is not None]
    return {"n": len(rows), "right": right, "confusion": confusion, "entities": entities, "misses": misses,
            "errors": [r for r in rows if r["error"]],
            "ms": {"p50": percentile(times, 50), "p95": percentile(times, 95), "max": max(times, default=float("nan"))}}


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.1f}%)" if b else "0/0"


def show(summary: dict, out=None) -> None:
    out = out or sys.stdout
    w = lambda s="": print(s, file=out)
    w(f"intent: {_pct(summary['right'], summary['n'])}")
    w("by expected intent (right/n, then what it got instead):")
    for want in sorted(summary["confusion"]):
        c = summary["confusion"][want]
        others = "  ".join(f"{g}:{k}" for g, k in c.most_common() if g != want)
        w(f"  {want:<18} {c[want]:>3}/{sum(c.values()):<3} {others}")
    e = summary["entities"]
    both = (e["finding_id"]["right"] + e["exercise"]["right"], e["finding_id"]["expected"] + e["exercise"]["expected"])
    w("entities where the question names one: " + ", ".join(
        f"{k} {_pct(e[k]['right'], e[k]['expected'])}" for k in ENTITIES) + f"; finding+exercise {_pct(*both)}")
    w("named by the engine but not in the label: " + ", ".join(f"{k} {e[k]['spurious']}" for k in ENTITIES))
    w(f"errors: {len(summary['errors'])}")
    for r in summary["errors"]:
        w(f"  {r['id']}: {r['error']}")
    ms = summary["ms"]
    w(f"reply ms (engine built): p50 {ms['p50']:.2f}, p95 {ms['p95']:.2f}, max {ms['max']:.2f}")
    w(f"misses ({len(summary['misses'])}):")
    for m in summary["misses"]:
        if m["error"]:
            continue
        w(f"  {m['id']} [{','.join(m['wrong'])}] {json.dumps(m['text'], ensure_ascii=False)}")
        w(f"      want {m['want']} {_brief(m['expected'])}")
        w(f"      got  {m['got']} {_brief(m['predicted'])}")


def _brief(entities: Mapping[str, Any]) -> str:
    return " ".join(f"{k}={entities.get(k)}" for k in ("finding_id", "exercise", "player", "topic")
                    if entities.get(k) is not None)


def source_hashes(extra: Sequence[Path] = ()) -> dict[str, str]:
    files = sorted({p for pattern in SOURCES for p in ROOT.glob(pattern)} | set(extra))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m coach.chat.eval", description=__doc__.split("\n")[0])
    ap.add_argument("questions", type=Path)
    ap.add_argument("--export", type=Path, help="write the actual replies, one row per question, as JSON")
    ap.add_argument("--split", choices=("all", "tune", "check"), default="all")
    ap.add_argument("--report", choices=("full", "preview", "real"), default="full")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    digest, items = load_questions(args.questions)
    if args.split != "all":
        items = [item for item in items if split_of(item) == args.split]

    clock = time.perf_counter
    t0 = clock()
    from coach.chat.engine import Coach
    from coach.exercises.library import load_library
    from coach.tests import fixture_report
    t1 = clock()
    library = load_library()
    t2 = clock()
    if args.report == "real":
        from coach.analyze import analyze
        report = analyze(json.loads(REAL_DOC.read_text(encoding="utf-8")))
    else:
        report = fixture_report.full_report() if args.report == "full" else fixture_report.preview_report()
    t3 = clock()
    coach = Coach(report, library)
    t4 = clock()
    ms = lambda a, b: round((b - a) * 1000, 3)
    timing = {"import_ms": ms(t0, t1), "library_load_ms": ms(t1, t2), "report_build_ms": ms(t2, t3),
              "construction_ms": ms(t3, t4)}

    rows = run(coach, items)
    summary = score(items, rows, exercise_names(library, coach.kb))
    print(f"questions: {args.questions} sha256 {digest[:16]}  split {args.split}  report {args.report}  "
          f"rows {len(rows)}")
    show(summary)
    print(f"construction ms (cold): {timing['construction_ms']:.1f}  (imports {timing['import_ms']:.1f}, "
          f"library {timing['library_load_ms']:.1f}, report {timing['report_build_ms']:.1f})")

    if args.export:
        doc = {
            "format": "coach-chat-observations/1",
            "question_file": str(args.questions),
            "question_sha256": digest,
            "split": args.split,
            "report": args.report,
            "construction_ms": timing["construction_ms"],
            "timing": {**timing, "note": TIMING_NOTE, "python": platform.python_version(),
                       "platform": platform.platform()},
            "entity_source": ENTITY_SOURCE,
            "source_sha256": source_hashes([REAL_DOC] if args.report == "real" else []),
            "rows": rows,
        }
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(doc, ensure_ascii=False, allow_nan=False, indent=1) + "\n",
                               encoding="utf-8")
        print(f"export: {args.export} ({len(rows)} rows)")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
