"""Deterministic chat coach: one reply per player message, built from the report, the knowledge base and
the exercise library. No network and no language model.

`Coach.reply` keeps no state between calls. The conversation state lives in the returned ChatContext,
which the app stores (context_to_dict / context_from_dict) and sends back with the next message.

Context conventions:
  player      the user's own player ("me"); the engine never changes it. None means the report's first player.
  topic       "kind:ref@role", for example "finding:technique.stay_lower" or "exercise:wall_sit@far".
              "@role" is only present when the discussed player is not `player`.
  last_offer  the same shape, naming what a plain "yes" accepts, for example "exercise_detail:wall_sit".
  too_hard    exercises the player called too hard. They come back only when nothing else fits, and then at an
              easier dose, until the player asks to make that exercise harder.
  pain_stop   true once the coach has stopped for pain. The engine never clears it, so it lasts for the rest of the
              conversation; the app must keep it with the conversation. While it is set, anything but a purely
              informational question gets the fixed safety reply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Sequence

from coach.chat.intents import Classifier, Parse
from coach.chat.kb import INTENTS, POSITIONAL, KnowledgeBase, load_kb
from coach.chat.plan import build_plan, easier_dose, fits, number_word, step
from coach.exercises.library import EQUIPMENT, for_tag
from coach.schema import (FINDING_CATALOG, FIX_TAGS, Action, ActionKind, Basis, Category, Exercise, Finding, Metric,
                          Moment, Report, Reply, Role, Severity, Tier, Turn)


@dataclass(frozen=True)
class ChatContext:
    finding_id: Optional[str] = None
    player: Optional[Role] = None
    frame: int = 0
    exercise_id: Optional[str] = None
    shown_exercises: tuple[str, ...] = ()
    topic: Optional[str] = None
    last_offer: Optional[str] = None
    level: int = 3
    equipment: Optional[tuple[str, ...]] = None
    too_hard: tuple[str, ...] = ()
    pain_stop: bool = False
    history_reports: tuple[Report, ...] = ()


def context_to_dict(ctx: ChatContext) -> dict[str, Any]:
    """JSON-safe copy of the context. history_reports is left out: the app supplies it per request."""
    return {
        "finding_id": ctx.finding_id,
        "player": None if ctx.player is None else Role(ctx.player).value,
        "frame": ctx.frame,
        "exercise_id": ctx.exercise_id,
        "shown_exercises": list(ctx.shown_exercises),
        "topic": ctx.topic,
        "last_offer": ctx.last_offer,
        "level": ctx.level,
        "equipment": None if ctx.equipment is None else list(ctx.equipment),
        "too_hard": list(ctx.too_hard),
        "pain_stop": ctx.pain_stop is True,
    }


def _whole(value: Any, name: str, default: int, low: int, high: Optional[int]) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < low or (high is not None and value > high):
        raise ValueError(f"context {name} must be a whole number from {low}{'' if high is None else f' to {high}'}")
    return value


def _text_or_none(value: Any, name: str) -> Optional[str]:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"context {name} must be text or null")
    return value


def _names(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"context {name} must be a list of names")
    return tuple(dict.fromkeys(value))


def context_from_dict(data: Mapping[str, Any], history_reports: Sequence[Report] = ()) -> ChatContext:
    """Parse the app's JSON context. Missing keys take defaults, unknown keys are ignored, bad values raise ValueError."""
    if not isinstance(data, Mapping):
        raise ValueError("context must be a JSON object")
    finding_id = _text_or_none(data.get("finding_id"), "finding_id")
    if finding_id is not None and finding_id not in FINDING_CATALOG:
        raise ValueError(f"context finding_id {finding_id!r} is not a known finding")
    player = data.get("player")
    if player is not None:
        if not isinstance(player, str) or player not in ("near", "far"):
            raise ValueError("context player must be 'near', 'far' or null")
        player = Role(player)
    equipment = data.get("equipment")
    if equipment is not None:
        equipment = _names(equipment, "equipment")
        unknown = [e for e in equipment if e not in EQUIPMENT]
        if unknown:
            raise ValueError(f"context equipment has unknown items {unknown}")
    shown = data.get("shown_exercises")
    too_hard = data.get("too_hard")
    pain_stop = data.get("pain_stop")
    if pain_stop is not None and not isinstance(pain_stop, bool):
        raise ValueError("context pain_stop must be true, false or null")
    return ChatContext(
        finding_id=finding_id,
        player=player,
        frame=_whole(data.get("frame"), "frame", 0, 0, None),
        exercise_id=_text_or_none(data.get("exercise_id"), "exercise_id"),
        shown_exercises=() if shown is None else _names(shown, "shown_exercises"),
        topic=_text_or_none(data.get("topic"), "topic"),
        last_offer=_text_or_none(data.get("last_offer"), "last_offer"),
        level=_whole(data.get("level"), "level", 3, 1, 3),
        equipment=equipment,
        too_hard=() if too_hard is None else _names(too_hard, "too_hard"),
        pain_stop=pain_stop is True,
        history_reports=tuple(history_reports),
    )


# ---- fixed wording --------------------------------------------------------------------------------------

CHIP_FIX = "How do I fix it?"
CHIP_FIX_THEIRS = "How would they fix it?"
CHIP_SHOW = "Show me the moment"
CHIP_BEST = "Show me my best one"
CHIP_BEST_THEIRS = "Show me their best one"
CHIP_WHY = "Why does it matter?"
CHIP_EXERCISES = "What exercises help?"
CHIP_HOW_EX = "How do I do it?"
CHIP_WHY_EX = "Why this exercise?"
CHIP_EASIER = "Make it easier"
CHIP_HARDER = "Make it harder"
CHIP_ANOTHER = "Give me another exercise"
CHIP_NO_GEAR = "I have no equipment"
CHIP_ANY_GEAR = "Any equipment is fine"
CHIP_PLAN = "Make me a practice plan"
CHIP_FIRST = "What should I work on first?"
CHIP_WELL = "What did I do well?"
CHIP_NEXT = "What else?"
CHIP_LIMITS = "What can't you measure?"
CHIP_COMPARE = "How do I compare with the other player?"
CHIP_LEVELS = ("I am a beginner", "I am intermediate", "I am advanced")

SHUTTLE_NOTE = "The shuttle is not tracked, so this comes from movement and timing only."
PLAN_SHUTTLE_NOTE = "The shuttle is not tracked, so these goals come from movement and timing only."
DESCRIPTIVE_NOTE = "On its own this is a description, not proof the tactic was right or wrong."
SHUTTLE_SHORT = "The shuttle is not tracked."
DESCRIPTIVE_SHORT = "It describes the play; it does not judge the tactic."
PREVIEW_NOTE = "This is a quick estimate that may change after the full analysis."
DESCRIPTIVE = frozenset({"tactics.base_discipline", "tactics.initiative", "tactics.tempo", "footwork.coverage"})
# A report text that already says a description is no verdict on the play, so the description note repeats it.
VERDICT_SAID = re.compile(r"\bnot (?:a (?:judgment|verdict|fault)|proof)\b|\btactics fault\b|\bposition (?:share )?only\b"
                          r"|\bdoes not (?:say whether|show who)\b", re.IGNORECASE)
# The analysis marks swings from arm motion alone. A finding whose text says so was read at a marked swing, not at
# a confirmed hit, so replies must not turn it into a hit or a personal timing fault.
UNCONFIRMED = re.compile(r"not a confirmed hit|hit itself was not confirmed|arm swing|marked swing", re.IGNORECASE)
CONFIRM_SAID = re.compile(r"not (?:a )?confirmed|was not confirmed", re.IGNORECASE)
CONTACT_NOTE = "The hit itself was not confirmed, so this is not a contact score."
TIMED_NOTE = "It is timed from marked arm swings, not confirmed hits, so it is not a timing fault."
SWING_NOTE = "It is read at marked arm swings, not confirmed hits, so it is not a fault on its own."
BASIS_NOTE = "It is read at marked arm swings, not confirmed hits."
TIMED = frozenset({"footwork.split_step", "footwork.first_step"})
TIMING_WORDS = frozenset("late later early earlier timing timed slow slower quick quicker sooner".split())
NO_TIMING = "I cannot judge its timing, as the hit is not confirmed."
GENERAL = "In general, not from this clip: "
SOFT_ASK = "Want some practice ideas for it?"
SEVERITY_WORDS = {Severity.PRIORITY: "a top priority", Severity.IMPROVE: "worth improving",
                  Severity.INFO: "a plain reading", Severity.STRENGTH: "a strength"}
RANK_WORDS = {Severity.PRIORITY: "The analysis ranks it as a top thing to work on",
              Severity.IMPROVE: "The analysis marks it as worth improving",
              Severity.INFO: "The analysis keeps it as a plain reading, not a fault or a strength",
              Severity.STRENGTH: "The analysis counts it as a strength"}
LEVEL_TEXT = {1: "From now on I will stick to beginner exercises.",
              2: "From now on I will suggest beginner and intermediate exercises.",
              3: "From now on I will include the harder exercises too."}
GEAR_WORDS = {"cable": "a cable machine", "kettlebell": "a kettlebell"}
WORD_LIMIT = 85
LEVEL_NAMES = {1: "beginner", 2: "intermediate", 3: "advanced"}
PLAN_SAME = {"easier": "With the exercises I have for these goals, I cannot make this plan any easier. Instead, do fewer "
                       "reps and rest longer between sets.",
             "harder": "With the exercises I have for these goals, I cannot make this plan any harder. Instead, slow "
                       "each rep down and pause at the hardest point."}
# Each closing question has a second wording; the exchange count picks one, so a reply is repeatable for the same
# history but the coach does not say the same thing twice in a row.
ASKS = {
    "Want to know how to do the first one?": "Shall I walk you through the first one?",
    "Want to know how to do it?": "Shall I walk you through it?",
    "Want to know how to fix it?": "Shall I give you a fix for it?",
    "Want some practice ideas for it?": "Shall I suggest some practice ideas for it?",
    "Want to hear what to work on first?": "Shall I tell you what to work on first?",
    "Want to hear what went well?": "Shall I tell you what went well?",
    "Want another exercise for the same goal?": "Shall I find another exercise for the same goal?",
    "Want to see it?": "Shall I play it?",
    "Want to see the best moment?": "Shall I play the best moment?",
    "Want me to show the first one?": "Shall I play the first one?",
    "Want to see one on video?": "Shall I play one for you?",
}
NO_REPEAT_PREFIX = frozenset(("greeting", "thanks", "affirm", "deny", "unknown"))
FAREWELL = re.compile(r"\b(?:bye|goodbye|see you|see ya|later|got to go|night)\b")
# While a stop for pain lasts, only these purely informational intents get their own answer, and then with no offer,
# chip or exercise that leads to training. Everything else gets the fixed safety reply. Word definitions are not here
# because several describe how to move, such as the split step's hop.
AFTER_STOP = frozenset(("limits", "strengths", "show_moment"))
ASK_END = re.compile(r"(?<=[.!:])\s+[^.!?\n]*\?$")
LAST_ASK = re.compile(r"(?:(?<=[.!:])\s+|\s*\n)[^.!?\n]*\?$")  # ASK_END, or a question on its own last line
ABSENT_ASK = " Ask me what to work on first instead."
SAFE_LINE = "Warm up first, and stop any exercise that causes sharp pain"
OBJECT_BEFORE = frozenset((
    "to at for with from past behind beside near toward towards of around than about into onto over under on in by "
    "let lets help helps make makes keep keeps give gives put puts force forces pull pulls push pushes take takes "
    "slow slows leave leaves send sends catch catches beat beats stop stops move moves").split())
SECOND_PERSON = re.compile(r"\b(you(?:['\u2019](?:re|ll|ve|d))?|yourself|yours|your)\b", re.IGNORECASE)
THIRD_PERSON = {"yourself": "themselves", "yours": "theirs", "your": "their"}


# ---- small text helpers ---------------------------------------------------------------------------------

UNIT_WORDS = {"deg": " degrees", "s": " seconds", "m": " metres", "stature/s": " body heights a second",
              "heights/s": " body heights a second", "heights": " of body height", "share": "", "fraction": "",
              "count": "", "ratio": ""}
UNIT_ONE = {"deg": " degree", "s": " second", "m": " metre"}
PERCENT = frozenset({"share", "fraction", "heights"})


def number(value: float, unit: str) -> str:
    """The bare number as quoted. Values are only rounded, never invented."""
    if unit in PERCENT:
        return f"{round(value * 100)}%"
    text = f"{value:.{1 if unit in ('deg', 'count', 'heights/s') else 2}f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def fmt(value: float, unit: str) -> str:
    text = number(value, unit)
    return text + (UNIT_ONE.get(unit) if text == "1" and unit in UNIT_ONE else UNIT_WORDS.get(unit, f" {unit}"))


def span(low: float, high: float, unit: str) -> str:
    return f"{number(low, unit)} to {fmt(high, unit)}"


def shown(value: float, unit: str) -> float:
    """The value as number() rounds it, so a comparison agrees with the quoted numbers."""
    return round(value, 2 if unit in PERCENT else 1 if unit in ("deg", "count", "heights/s") else 2)


def sure_word(confidence: float) -> str:
    return "fairly sure" if confidence >= 0.75 else "somewhat sure" if confidence >= 0.5 else "not very sure"


def split_tag(tag: Optional[str]) -> tuple[str, str, Optional[Role]]:
    """'kind:ref@role' -> (kind, ref, role)."""
    if not tag:
        return "", "", None
    body, _, role = tag.partition("@")
    kind, _, ref = body.partition(":")
    return kind, ref, Role(role) if role in ("near", "far") else None


def make_tag(kind: str, ref: str = "", role: Optional[Role] = None, me: Optional[Role] = None) -> str:
    text = f"{kind}:{ref}" if ref else kind
    return f"{text}@{role.value}" if role is not None and role != me else text


def lower_first(text: str) -> str:
    """'Sit-low hold' -> 'sit-low hold', but '2D view' and 'ACL' stay as they are."""
    return text if text[1:2].isupper() else text[:1].lower() + text[1:]


def sentence(text: str) -> str:
    text = text.strip()
    return text if not text or text[-1] in ".!?" else text + "."


def word_count(text: str) -> int:
    return len(text.split())


def compose(required: Sequence[str], optional: Sequence[str] = (), limit: int = WORD_LIMIT) -> str:
    """Required sentences always; optional ones in order while the reply stays near the word limit."""
    parts = [sentence(p) for p in required if p]
    total = sum(word_count(p) for p in parts)
    for extra in optional:
        if extra and total + word_count(extra) <= limit:
            parts.append(sentence(extra))
            total += word_count(extra)
    return " ".join(parts)


def ask_words(ask: str) -> int:
    """The words a closing question needs in whichever of its two wordings vary() picks."""
    return max(word_count(ask), word_count(ASKS.get(ask, "")))


def gear_words(equipment: Optional[Sequence[str]]) -> str:
    if equipment is None:
        return "any equipment"
    if not equipment:
        return "no equipment"
    items = [GEAR_WORDS.get(e, f"a {e}") for e in equipment]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def agree(ex: Exercise, one: str, many: str) -> str:
    """The verb form for an exercise name: "the squat jumps need", "the wall sit needs"."""
    return many if ex.name.endswith("s") else one


def harder_tip(ex: Exercise) -> str:
    """How to make the hardest exercise left harder, by the kind of dose it has."""
    if "second" in ex.sets_reps:
        return "Make each set a little longer while your form stays good."
    if re.search(r"\b(?:hop|jump)", ex.name.casefold()):
        return "Add a few more to each set, keeping every landing soft and quiet."
    return "Slow each rep down and pause at the hardest point."


def third_person(text: str) -> str:
    """'when you hit' -> 'when they hit', for report text about the player who is not the user."""
    def swap(m: re.Match) -> str:
        word = m.group(1)
        low = word.lower().replace("\u2019", "'")
        if low in THIRD_PERSON:
            out = THIRD_PERSON[low]
        else:
            before = text[:m.start()].split()
            obj = low == "you" and bool(before) and before[-1].lower().strip(",.;:") in OBJECT_BEFORE
            out = "them" if obj else "they" + low[3:]
        return out[:1].upper() + out[1:] if word[:1].isupper() else out
    return SECOND_PERSON.sub(swap, text)


def vary(text: str, exchange: int, said: set[str]) -> str:
    """Word the closing question by the exchange count, and never repeat the coach's own earlier reply."""
    for ask, other in ASKS.items():
        if text.endswith(ask):
            head = text[:-len(ask)]
            first, second = (ask, other) if exchange % 2 == 0 else (other, ask)
            return head + (second if head + first in said else first)
    return text


def fresh(extra: str, said: str) -> str:
    """The sentences of extra that do not restate a sentence already in said."""
    def content(s: str) -> set[str]:
        return {w for w in re.findall(r"[a-z]+", s.lower()) if len(w) >= 4}
    old = [content(s) for s in re.split(r"(?<=[.!?])\s+", said)]
    keep = []
    for s in re.split(r"(?<=[.!?])\s+", extra.strip()):
        new = content(s)
        if s and not any(new and o and len(new & o) >= 0.5 * min(len(new), len(o)) for o in old):
            keep.append(s)
    return " ".join(keep)


def with_note(text: str, note: str) -> str:
    """Add a note while keeping a closing question last."""
    if not text.endswith("?"):
        return f"{text} {note}"
    dot, line = text.rfind(". "), text.rfind("\n")
    if line > dot:
        return f"{text[:line]}\n{note}{text[line:]}"
    if dot >= 0:
        return f"{text[:dot + 1]} {note}{text[dot + 1:]}"
    return f"{note} {text}"


@dataclass(frozen=True)
class _Out:
    text: str
    ctx: ChatContext
    actions: tuple[Action, ...] = ()
    suggestions: tuple[str, ...] = ()
    quoted: bool = False
    fixed: bool = False  # approved wording, sent exactly as written


SPEAKERS = {"user": "user", "player": "user", "coach": "coach"}  # question files call the user "player"


def _turns(history: Any) -> tuple[Turn, ...]:
    """Turns from a list of Turn objects or {"speaker", "text"} objects. Anything malformed is skipped."""
    if not isinstance(history, (list, tuple)):
        return ()
    out = []
    for item in history:
        if isinstance(item, Turn):
            out.append(item)
        elif isinstance(item, Mapping) and item.get("speaker") in SPEAKERS and isinstance(item.get("text"), str):
            out.append(Turn(SPEAKERS[item["speaker"]], item["text"]))
    return tuple(out)


def _as_context(context: Any) -> ChatContext:
    if context is None:
        return ChatContext()
    if isinstance(context, ChatContext):
        return context
    return context_from_dict(context)


OFFERS = {"explain": "explain", "how_to_fix": "how_to_fix", "show_moment": "show_moment", "show_best": "show_moment",
          "exercise_detail": "exercise_detail", "exercise_request": "exercise_request", "another": "exercise_adjust",
          "practice_plan": "practice_plan", "next": "next", "summary": "summary", "strengths": "strengths"}


class Coach:
    """Answers one player message at a time from the report, the knowledge base and the exercise library."""

    def __init__(self, report: Report, library: Mapping[str, Exercise], kb: Optional[KnowledgeBase] = None):
        if not isinstance(report, Report):
            raise TypeError("Coach needs a Report")
        self.report = report
        self.library = library
        self.kb = kb if kb is not None else load_kb()
        self.classifier = Classifier(self.kb, library, report)
        self.findings = {(f.id, f.player): f for f in report.findings}
        self.preview = report.tier == Tier.PREVIEW or "preview_estimate" in report.warnings
        self.handlers = {intent: getattr(self, "_" + intent) for intent in INTENTS}

    def reply(self, history: Sequence[Turn], user_text: str, context: Any = None) -> tuple[Reply, ChatContext]:
        reply, ctx, _ = self.reply_traced(history, user_text, context)
        return reply, ctx

    def reply_traced(self, history: Sequence[Turn], user_text: str, context: Any = None) -> tuple[Reply, ChatContext, Parse]:
        """reply() plus the parse it acted on, for evaluation."""
        turns = _turns(history)
        ctx = self._clean(_as_context(context))
        parse = self.classifier.parse(user_text if isinstance(user_text, str) else "", ctx, turns)
        # A stop for pain lasts for the rest of the conversation, whether it comes from this message, the context or
        # an earlier stop reply still in the history.
        stopped = (parse.intent == "safety" or ctx.pain_stop
                   or any(t.speaker == "coach" and t.text == self.kb.safety["text"] for t in turns))
        wanted = OFFERS.get(split_tag(ctx.last_offer)[0]) if parse.intent == "affirm" else parse.intent
        if stopped and wanted not in AFTER_STOP:
            out = self._safety(parse, ctx, turns)
        else:
            out = self.handlers[parse.intent](parse, self._prefs(parse, ctx), turns)
            if stopped:
                out = self._informational(out, turns)
        reply, ctx = self._finish(out, parse, turns)
        return reply, replace(ctx, pain_stop=stopped), parse

    # ---- boundary -------------------------------------------------------------------------------------

    def _clean(self, ctx: ChatContext) -> ChatContext:
        """Drop stale or unknown values so a context from another clip or app version cannot break a reply."""
        lib = self.library
        role = Role(ctx.player) if isinstance(ctx.player, str) and ctx.player in ("near", "far") else None
        player = role if role in self.report.players else None
        whole = lambda v: isinstance(v, int) and not isinstance(v, bool)
        shown = ctx.shown_exercises if isinstance(ctx.shown_exercises, (list, tuple)) else ()
        hard = ctx.too_hard if isinstance(ctx.too_hard, (list, tuple)) else ()
        gear = ctx.equipment
        if gear is not None:
            gear = (tuple(dict.fromkeys(e for e in gear if isinstance(e, str) and e in EQUIPMENT))
                    if isinstance(gear, (list, tuple)) else None)
        reports = ctx.history_reports if isinstance(ctx.history_reports, (list, tuple)) else ()
        return ChatContext(
            finding_id=ctx.finding_id if isinstance(ctx.finding_id, str) and ctx.finding_id in FINDING_CATALOG else None,
            player=player,
            frame=ctx.frame if whole(ctx.frame) and 0 <= ctx.frame < self.report.n_frames else 0,
            exercise_id=ctx.exercise_id if isinstance(ctx.exercise_id, str) and ctx.exercise_id in lib else None,
            shown_exercises=tuple(dict.fromkeys(e for e in shown if isinstance(e, str) and e in lib)),
            topic=ctx.topic if isinstance(ctx.topic, str) else None,
            last_offer=ctx.last_offer if isinstance(ctx.last_offer, str) else None,
            level=min(3, max(1, ctx.level)) if whole(ctx.level) else 3,
            equipment=gear,
            too_hard=tuple(dict.fromkeys(e for e in hard if isinstance(e, str) and e in lib)),
            pain_stop=ctx.pain_stop is True,
            history_reports=tuple(r for r in reports if isinstance(r, Report)),
        )

    @staticmethod
    def _prefs(p: Parse, ctx: ChatContext) -> ChatContext:
        """Remember a level or equipment the message states, whatever else it asks."""
        if p.level is not None:
            ctx = replace(ctx, level=p.level)
        if p.equipment:
            op, items = p.equipment
            if op == "none":
                gear: Optional[tuple[str, ...]] = ()
            elif op == "all":
                gear = None
            elif op == "only":
                gear = tuple(items)
            elif op == "remove":
                gear = tuple(e for e in (ctx.equipment if ctx.equipment is not None else sorted(EQUIPMENT)) if e not in items)
            else:
                gear = tuple(items) if ctx.equipment is None else tuple(dict.fromkeys((*ctx.equipment, *items)))
            ctx = replace(ctx, equipment=gear)
        return ctx

    def _finish(self, out: _Out, parse: Parse, turns: Sequence[Turn]) -> tuple[Reply, ChatContext]:
        text, ctx = out.text, out.ctx
        said = {t.text for t in turns if t.speaker == "coach"}
        if not out.fixed:
            text = vary(text, len(turns) // 2, said)
        if out.quoted and self._note_due(turns):
            # Handlers leave room for the note; if one did not, the note stays and the question goes.
            noted = with_note(text, PREVIEW_NOTE)
            text = noted if word_count(noted) <= WORD_LIMIT else with_note(ASK_END.sub("", text), PREVIEW_NOTE)
        if out.text.endswith("?") and not text.endswith("?"):
            ctx = replace(ctx, last_offer=None)  # never leave an offer the player cannot see
        if (not out.fixed and text in said and parse.intent not in NO_REPEAT_PREFIX
                and word_count(text) + 3 <= WORD_LIMIT):
            text = "As I said, " + (text if text.startswith(("I ", "I'")) else lower_first(text))
        chips = tuple(dict.fromkeys(s for s in out.suggestions if 0 < len(s) <= 48))[:4]
        reply = Reply(text, tuple(out.actions), parse.intent, min(1.0, max(0.0, float(parse.confidence))), chips)
        return reply, ctx

    def _note_due(self, turns: Sequence[Turn]) -> bool:
        """True when a reply quoting numbers must still carry the preview note."""
        return self.preview and not any(PREVIEW_NOTE in t.text for t in turns if t.speaker == "coach")

    def _room(self, turns: Sequence[Turn], ask: str = "", keep: int = 8, note: bool = True) -> int:
        """Words a handler may use and still fit its closing question and, when due, the preview note."""
        return WORD_LIMIT - max(keep, ask_words(ask)) - (word_count(PREVIEW_NOTE) if note and self._note_due(turns) else 0)

    def _fits(self, turns: Sequence[Turn], text: str, ask: str, quoted: bool) -> bool:
        """True when _finish() will keep the closing question: text with the question as vary() will word it, plus
        the preview note when due, stays within the limit."""
        note = word_count(PREVIEW_NOTE) if quoted and self._note_due(turns) else 0
        said = {t.text for t in turns if t.speaker == "coach"}
        return word_count(vary(f"{text} {ask}".strip(), len(turns) // 2, said)) + note <= WORD_LIMIT

    def _informational(self, out: _Out, turns: Sequence[Turn]) -> _Out:
        """An informational answer during a stop for pain: no exercise, and no offer or chip that leads to training.
        A chip is a message sent with one tap, so it passes the same gate as a typed one."""
        text, offer = out.text, out.ctx.last_offer
        if offer is not None and OFFERS.get(split_tag(offer)[0]) not in AFTER_STOP:
            text, offer = LAST_ASK.sub("", text).removesuffix(ABSENT_ASK), None
        chips = tuple(c for c in out.suggestions if self.classifier.parse(c, out.ctx, turns).intent in AFTER_STOP)
        actions = tuple(a for a in out.actions if a.kind != ActionKind.EXERCISE)
        return replace(out, text=text, ctx=replace(out.ctx, last_offer=offer), actions=actions, suggestions=chips)

    # ---- who and what ---------------------------------------------------------------------------------

    def _me(self, ctx: ChatContext) -> Role:
        return ctx.player or self.report.players[0]

    def _role(self, p: Parse, ctx: ChatContext) -> Role:
        """The player a message is about: the one it names, else the one under discussion, else the user."""
        if p.player in self.report.players:
            return p.player
        role = split_tag(ctx.topic)[2]
        return role if role in self.report.players else self._me(ctx)

    def _who(self, role: Role, ctx: ChatContext) -> str:
        return "you" if role == self._me(ctx) else "the other player"

    def _whose(self, role: Role, ctx: ChatContext) -> str:
        return "your" if role == self._me(ctx) else "the other player's"

    def _voice(self, text: str, role: Role, ctx: ChatContext) -> str:
        return text if role == self._me(ctx) else third_person(text)

    def _tag(self, kind: str, ref: str, role: Role, ctx: ChatContext) -> str:
        return make_tag(kind, ref, role, self._me(ctx))

    def _target(self, p: Parse, ctx: ChatContext, role: Role) -> tuple[Optional[Finding], Optional[str]]:
        """(finding, absent id): the finding the message names, else the one in focus, else the top one."""
        if p.finding_id:
            found = self.findings.get((p.finding_id, role))
            return (found, None) if found else (None, p.finding_id)
        if ctx.finding_id:
            found = self.findings.get((ctx.finding_id, role))
            return (found, None) if found else (None, ctx.finding_id)
        pool = self.report.top_findings(1, role) or self._plain(role)[:1] or self.report.strengths(1, role)
        return (pool[0] if pool else None), None

    def _absent(self, finding_id: Optional[str], role: Role, ctx: ChatContext) -> _Out:
        who = self._who(role, ctx)
        if finding_id:
            text = f"I have no reading for {self._guide(finding_id)['name']} for {who} in this clip, so I cannot say."
        else:
            text = f"I have no findings for {who} in this clip."
        return _Out(text + ABSENT_ASK, replace(ctx, last_offer="summary"),
                    suggestions=(CHIP_FIRST, CHIP_WELL, CHIP_LIMITS))

    def _guide(self, finding_id: str) -> Mapping[str, Any]:
        return self.kb.fix_guides[finding_id]

    def _tags(self, f: Optional[Finding]) -> tuple[str, ...]:
        return self.classifier.finding_tags(f.id, f.player) if f else ()

    @staticmethod
    def _observed(m: Metric) -> bool:
        """A value of None means nothing was observed; the schema gives every present value its observations."""
        return m.value is not None

    @staticmethod
    def _measured(m: Metric) -> bool:
        """A reading chat may quote or compare: observed, with positive confidence."""
        return m.value is not None and m.confidence > 0

    @staticmethod
    def _moment(f: Optional[Finding], best: bool = False) -> Optional[Moment]:
        """The moment "show me" plays for f: the best one for a strength or when the best is asked for, else the
        first that needs work. None when f has no such moment, and then no reply offers to play one."""
        if f is None:
            return None
        if best or f.severity == Severity.STRENGTH:
            return f.best_moment
        return next(iter(f.issue_moments), None)

    def _show_chip(self, f: Optional[Finding], instead: str = "") -> str:
        """"Show me the moment" when f has a moment to play, else `instead`; an empty chip is dropped."""
        return CHIP_SHOW if self._moment(f) else instead

    @staticmethod
    def _soft(f: Finding) -> bool:
        """A plain reading or a position on its own is not a fault: it gets general practice, never a fix."""
        return f.severity != Severity.STRENGTH and (f.severity == Severity.INFO or f.id in DESCRIPTIVE)

    def _plain(self, role: Role) -> list[Finding]:
        """Plain readings worth a look when nothing is ranked as needing work: scored and actually measured."""
        rows = [f for f in self.report.findings if f.player == role and f.severity == Severity.INFO and f.score > 0
                and f.metrics and self._measured(f.metrics[0])]
        return sorted(rows, key=lambda f: (-f.score, f.id))

    @staticmethod
    def _anchor(f: Finding) -> str:
        """What a reading taken at a marked arm swing does not show. Descriptions carry their own note."""
        text = " ".join([f.explain, f.headline, *(m.label for m in f.metrics)])
        if f.id in DESCRIPTIVE or not UNCONFIRMED.search(text):
            return ""
        if f.severity != Severity.INFO:
            return BASIS_NOTE  # the analysis ranked it, so the basis is stated without overruling that rank
        if f.category == Category.CONTACT:
            return CONTACT_NOTE
        return TIMED_NOTE if f.metrics and f.metrics[0].unit == "s" else SWING_NOTE

    def _fix_ask(self, f: Finding) -> str:
        return SOFT_ASK if self._soft(f) else "Want to know how to fix it?"

    def _work_chip(self, f: Finding, role: Role, ctx: ChatContext) -> str:
        return CHIP_EXERCISES if self._soft(f) else self._fix_chip(role, ctx)

    def _notes(self, f: Finding, said: str = "", anchor: bool = True, room: int = WORD_LIMIT - 8) -> list[str]:
        """Shuttle, description and marked-swing notes, leaving out any that `said` already makes.

        Short forms are used when the full ones would push `said` past `room`, the words left once the ask fits."""
        def pick(short: bool) -> list[str]:
            notes = [SHUTTLE_SHORT if short else SHUTTLE_NOTE] if f.category in POSITIONAL else []
            notes += [DESCRIPTIVE_SHORT if short else DESCRIPTIVE_NOTE] if f.id in DESCRIPTIVE else []
            note = self._anchor(f) if anchor else ""
            notes += [note] if note and not CONFIRM_SAID.search(said) else []
            return [n for n in notes if fresh(n, said)] if said else notes
        notes = pick(False)
        return pick(True) if word_count(" ".join([said, *notes])) > room else notes

    def _show(self, f: Finding) -> Action:
        return Action(ActionKind.FINDING, f.id, self.report.players.index(f.player))

    @staticmethod
    def _label(m: Metric) -> str:
        label = lower_first(m.label)
        if m.unit == "count" and "count" not in label.lower():
            label = f"count of {label}"
        projected = m.unit == "deg" or m.basis == Basis.PROJECTED_2D
        return f"projected {label}" if projected and "projected" not in label.lower() else label

    @staticmethod
    def _guide_range(f: Finding, m: Metric) -> str:
        """The practice guide as words, or "" when the range guides nothing: an open share or a description."""
        if not m.target:
            return ""
        low, high = m.target
        if low <= 0 and (f.id in DESCRIPTIVE or (m.unit in ("share", "fraction") and high >= 1)):
            return ""
        if low <= 0:
            return f"{number(high, m.unit)} or fewer" if m.unit == "count" else f"{fmt(high, m.unit)} or less"
        return f"about {span(low, high, m.unit)}"

    def _reading(self, f: Finding, m: Metric, ctx: ChatContext) -> list[str]:
        """The number, unit, practice guide and how sure, each as plain sentences."""
        label = self._voice(self._label(m), f.player, ctx)
        count = f" over {m.n_obs} readings" if m.n_obs >= 2 else ""
        out = [f"{self._whose(f.player, ctx).capitalize()} {label} was {fmt(m.value, m.unit)}{count}"]
        guide = self._guide_range(f, m)
        if guide:
            out.append(self._voice(f"A practice guide is {guide}, a coaching aim rather than a tested norm for you",
                                   f.player, ctx))
        out.append(f"I am {sure_word(m.confidence)} about this reading")
        return out

    @staticmethod
    def _metric(f: Finding, p: Parse, turns: Sequence[Turn]) -> Metric:
        """The metric the message names, else the one the last coach reply named, else the main one."""
        def words(text: str) -> set[str]:
            return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 4}
        asked = words(" ".join(p.tokens))
        if not any(asked & words(m.label) for m in f.metrics):
            asked = words(next((t.text for t in reversed(turns) if t.speaker == "coach"), ""))
        return max(f.metrics, key=lambda m: (len(asked & words(m.label)), -f.metrics.index(m)))

    def _verdict(self, f: Finding, m: Metric, ctx: ChatContext, rank: bool = True) -> list[str]:
        """Whether a reading looks good: the number against the practice guide, how the analysis ranked it, how sure."""
        label = self._voice(self._label(m), f.player, ctx)
        count = f" over {m.n_obs} readings" if m.n_obs >= 2 else ""
        out = [f"{self._whose(f.player, ctx).capitalize()} {label} was {fmt(m.value, m.unit)}{count}"]
        guide = self._guide_range(f, m)
        if guide:
            low, high = (shown(v, m.unit) for v in m.target)
            value = shown(m.value, m.unit)
            where = "within" if low <= value <= high else "below" if value < low else "above"
            out.append(self._voice(f"That is {where} the practice guide of {guide}, a coaching aim rather than a tested "
                                   "norm for you", f.player, ctx))
        if rank and f.id not in DESCRIPTIVE:
            plain = f.severity == Severity.INFO and self._anchor(f)
            out.append("The analysis keeps it as a plain reading" if plain else RANK_WORDS[f.severity])
        out.append(f"I am {sure_word(m.confidence)} about this reading")
        return out

    # ---- conversation -----------------------------------------------------------------------------------

    def _fix_chip(self, role: Role, ctx: ChatContext) -> str:
        return CHIP_FIX if role == self._me(ctx) else CHIP_FIX_THEIRS

    def _best_chip(self, f: Finding, role: Role, ctx: ChatContext, instead: str = "") -> str:
        """"Show me my best one" (or theirs) when f has a best moment to play, else `instead`."""
        if f.best_moment is None:
            return instead
        return CHIP_BEST if role == self._me(ctx) else CHIP_BEST_THEIRS

    def _greeting(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        return _Out("Hi, I am your coach for this clip. Shall we start with what to work on first?",
                    replace(ctx, last_offer="summary"), suggestions=(CHIP_FIRST, CHIP_WELL, CHIP_PLAN))

    def _thanks(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        if FAREWELL.search(" ".join(p.tokens)):
            return _Out("Bye for now, and good luck with your practice. Come back any time with a question.",
                        replace(ctx, last_offer=None), suggestions=(CHIP_PLAN, CHIP_FIRST))
        if split_tag(ctx.topic)[0] == "plan":
            return _Out("You are welcome. Want to look at the next thing to work on?", replace(ctx, last_offer="next"),
                        suggestions=(CHIP_NEXT, CHIP_WELL))
        return _Out("You are welcome. Want a two-week practice plan next?", replace(ctx, last_offer="practice_plan"),
                    suggestions=(CHIP_PLAN, CHIP_NEXT))

    def _help(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        text = ("I can tell you what to work on first, explain why it matters, play the moment in the video, "
                "suggest gym exercises and show how to do them, and build a two-week plan. Try \"What should I work on first?\", "
                "\"How do I fix it?\" or \"Make me a practice plan\".")
        return _Out(text, replace(ctx, last_offer=None), suggestions=(CHIP_FIRST, CHIP_FIX, CHIP_PLAN, CHIP_LIMITS))

    def _summary(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        role = self._role(p, ctx)
        mine = role == self._me(ctx)
        top = self.report.top_findings(3, role)
        if not top:
            good = self.report.strengths(1, role)
            plain = self._plain(role)
            text = f"I found nothing that clearly needs work for {self._who(role, ctx)} in this clip."
            if plain:
                f = plain[0]
                line = f"One reading worth a look: {sentence(lower_first(self._voice(f.headline, role, ctx)))}"
                text = compose([text, line, *self._notes(f, f.headline)]) + " " + SOFT_ASK
                new = replace(ctx, finding_id=f.id, topic=self._tag("finding", f.id, role, ctx),
                              last_offer=self._tag("how_to_fix", f.id, role, ctx))
                return _Out(text, new, (self._show(f),), (CHIP_EXERCISES, self._show_chip(f), CHIP_NEXT))
            if good and good[0].best_moment:
                text += f" One thing that went well: {sentence(lower_first(self._voice(good[0].headline, role, ctx)))} Want to see it?"
                new = replace(ctx, finding_id=good[0].id, topic=self._tag("finding", good[0].id, role, ctx),
                              last_offer=self._tag("show_best", good[0].id, role, ctx))
                return _Out(text, new, (self._show(good[0]),), (self._best_chip(good[0], role, ctx), CHIP_LIMITS))
            return _Out(text, replace(ctx, last_offer=None), suggestions=(CHIP_LIMITS, CHIP_COMPARE))
        lead = "Most important first:" if mine else "For the other player, most important first:"
        lines = [f"- {self._voice(f.headline, role, ctx)} ({SEVERITY_WORDS[f.severity]})" for f in top]
        notes = [SHUTTLE_NOTE] if any(f.category in POSITIONAL for f in top) else []
        text = "\n".join([lead, *lines, *notes, "Want me to show the first one?"])
        first = top[0]
        new = replace(ctx, finding_id=first.id, topic=self._tag("finding", first.id, role, ctx),
                      last_offer=self._tag("show_moment", first.id, role, ctx))
        return _Out(text, new, (self._show(first),), (self._show_chip(first), self._work_chip(first, role, ctx), CHIP_WELL))

    def _strengths(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        role = self._role(p, ctx)
        good = self.report.strengths(3, role)
        if not good:
            text = (f"I did not find a clear strength for {self._who(role, ctx)} in this clip. That can just mean the "
                    "camera could not see it well. Want to hear what to work on first?")
            return _Out(text, replace(ctx, last_offer="summary"), suggestions=(CHIP_FIRST, CHIP_LIMITS))
        lead = "What went well:" if role == self._me(ctx) else "What went well for the other player:"
        best = next((f for f in good if f.best_moment), None)
        notes = [SHUTTLE_NOTE] if any(f.category in POSITIONAL for f in good) else []
        ask = "Want to see one on video?" if best else "Want to hear what to work on first?"
        text = "\n".join([lead, *(f"- {self._voice(f.headline, role, ctx)}" for f in good), *notes, ask])
        focus = best or good[0]
        offer = self._tag("show_best", best.id, role, ctx) if best else "summary"
        new = replace(ctx, finding_id=focus.id, topic=self._tag("finding", focus.id, role, ctx), last_offer=offer)
        return _Out(text, new, (self._show(focus),), (self._best_chip(focus, role, ctx, CHIP_FIRST), CHIP_FIRST, CHIP_PLAN))

    def _explain(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        role = self._role(p, ctx)
        f, absent = self._target(p, ctx, role)
        if f is None:
            return self._absent(absent, role, ctx)
        guide = self._guide(f.id)
        strong = f.severity == Severity.STRENGTH
        soft = self._soft(f)
        body = ("This is a strength. " if strong else "") + (f.explain if role == self._me(ctx)
                                                               else f"For the other player: {third_person(f.explain)}")
        # A timing question about a reading at a marked swing gets a plain "cannot judge" first.
        timing = f.id in TIMED and self._anchor(f) and TIMING_WORDS.intersection(p.tokens)
        if strong:
            ask, offer = ("Want to see the best moment?", "show_best") if f.best_moment else ("", None)
            chips = (self._best_chip(f, role, ctx), CHIP_NEXT, CHIP_FIRST)
        else:
            ask, offer = (self._fix_ask(f), "how_to_fix")
            chips = (self._work_chip(f, role, ctx), self._show_chip(f), CHIP_EXERCISES if not soft else CHIP_NEXT)
        # General advice is about the finding in general; for a plain reading it must not read as this clip's verdict.
        lead = GENERAL if soft else ""
        why = fresh(self._voice(guide["why"], role, ctx), body)
        m = f.metrics[0]
        said = [sentence(s) for s in self._reading(f, m, ctx)] if self._measured(m) else []
        reading = fresh(" ".join(said), body)
        number = fresh(" ".join(said[:-1]), body)
        sure = fresh(said[-1], body) if said else ""

        def build(body: str, note: bool, plain: bool, how: str) -> str:
            """The reply before its question. Keeping `how` leaves only the number and guide optional."""
            room = self._room(turns, ask, note=note) - word_count(how)
            head = f"{NO_TIMING} {body}" if timing else body
            notes = self._notes(f, head, room=room)
            required = [head, *(n for n in notes if not (plain and n in (DESCRIPTIVE_NOTE, DESCRIPTIVE_SHORT)))]
            spare = room - word_count(" ".join(required)) - word_count(lead)
            general = why if word_count(why) <= spare else self._voice(guide["simple"], role, ctx)
            general = f"{lead}{lower_first(general)}" if general and lead else general
            extra = number if how else reading
            if soft:
                return f"{compose(required, [general, extra], room)} {how}".strip()
            return f"{compose([*required, general], [extra], room)} {how}".strip()

        main = build(body, False, False, "")
        # Once this reply says how sure the reading is, only the number and guide give way to the note and question.
        keep = sure if sure and main.endswith(sure) else ""
        # When space is short, a sentence that repeats what another one already says goes first.
        bodies = [(body, False, keep)]
        if timing:
            # NO_TIMING already says the hit is not confirmed, so the body's own saying of it goes.
            bodies.append((" ".join(s for s in re.split(r"(?<=[.!?])\s+", body) if not CONFIRM_SAID.search(s)), False,
                           keep))
        if f.id in DESCRIPTIVE and VERDICT_SAID.search(body):
            # The body already says this is no verdict, so the description note goes.
            bodies.append((bodies[-1][0], True, keep))
        for i, (text, plain, how) in enumerate(bodies):
            if i:
                main = build(text, False, plain, how)
            if self._note_due(turns) and re.search(r"\d", main):
                main = build(text, True, plain, how)
            if self._fits(turns, main, ask, bool(re.search(r"\d", main))):
                break
        else:
            ask, offer = "", None
        new = replace(ctx, finding_id=f.id, topic=self._tag("finding", f.id, role, ctx),
                      last_offer=self._tag(offer, f.id, role, ctx) if offer else None)
        return _Out(f"{main} {ask}".strip(), new, (self._show(f),), chips, quoted=bool(re.search(r"\d", main)))

    def _simplify(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        kind, ref, role = split_tag(ctx.topic)
        role = role if role in self.report.players else self._me(ctx)
        if kind == "exercise" and ref in self.library:
            ex = self.library[ref]
            text = f"In short: {lower_first(ex.name)}, {self._dose(ex, ctx)}. Key point: {ex.cues[0]}."
            return _Out(text, ctx, (Action(ActionKind.EXERCISE, ex.id),), (CHIP_HOW_EX, CHIP_EASIER, CHIP_ANOTHER))
        if kind == "term" and ref in self.kb.glossary:
            return _Out(f"In short: {lower_first(self.kb.glossary[ref]['simple'])}", ctx,
                        suggestions=(CHIP_FIRST, CHIP_FIX))
        if kind == "plan":
            text = "In short: three short gym sessions a week for two weeks, easy at first and a little harder in week two."
            return _Out(text, ctx, suggestions=(CHIP_HOW_EX, CHIP_EASIER, CHIP_NO_GEAR))
        if kind == "tactic" and ref in self.kb.tactics:
            first = self.kb.tactics[ref]["findings"][0]
            return _Out(f"In short: {self._guide(first)['simple']} {SHUTTLE_NOTE}", ctx, suggestions=(CHIP_FIRST, CHIP_FIX))
        f, _ = self._target(replace(p, finding_id=None), ctx, role)
        if f is None:
            return _Out("In short: ask me what to work on first, and I will take it one step at a time.", ctx,
                        suggestions=(CHIP_FIRST, CHIP_PLAN))
        lead = "In short, as general practice: " if self._soft(f) else "In short: "
        text = " ".join([f"{lead}{self._voice(self._guide(f.id)['simple'], role, ctx)}", *self._notes(f)])
        return _Out(text, replace(ctx, finding_id=f.id), (self._show(f),), (self._work_chip(f, role, ctx), self._show_chip(f)))

    def _next(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        if split_tag(ctx.topic)[0] == "exercise":
            return self._exercise_adjust(replace(p, adjust="another", exercise_id=None), ctx, turns)
        role = self._role(p, ctx)
        top = list(self.report.top_findings(len(self.report.findings), role))
        pool = top or self._plain(role)
        ids = [f.id for f in pool]
        rest = pool[ids.index(ctx.finding_id) + 1:] if ctx.finding_id in ids else pool
        if not rest:
            what = "everything to work on" if top else "the readings worth a look"
            text = f"That covers {what} for {self._who(role, ctx)}."
            if self.report.strengths(1, role):
                return _Out(text + " Want to hear what went well?",
                            replace(ctx, last_offer=self._tag("strengths", "", role, ctx)), suggestions=(CHIP_WELL, CHIP_PLAN))
            return _Out(text, replace(ctx, last_offer=None), suggestions=(CHIP_EXERCISES, CHIP_LIMITS, CHIP_COMPARE))
        f = rest[0]
        lead = "Next: " if role == self._me(ctx) else "Next for the other player: "
        guide = self._guide(f.id)
        headline = lead + sentence(lower_first(self._voice(f.headline, role, ctx)))
        why = self._voice(guide["why"], role, ctx)
        main = compose([headline, *self._notes(f, headline)],
                       [f"{GENERAL}{lower_first(why)}" if self._soft(f) else why], WORD_LIMIT - 8)
        new = replace(ctx, finding_id=f.id, topic=self._tag("finding", f.id, role, ctx),
                      last_offer=self._tag("how_to_fix", f.id, role, ctx))
        return _Out(f"{main} {self._fix_ask(f)}", new, (self._show(f),),
                    (self._work_chip(f, role, ctx), self._show_chip(f), CHIP_WHY))

    # ---- exercises ------------------------------------------------------------------------------------

    def _options(self, tags: Sequence[str], level: int, equipment: Optional[Sequence[str]],
                 skip: Sequence[str] = ()) -> list[tuple[Exercise, str]]:
        """Exercises for any of the tags within the level and gear, easiest first, then earlier tags first."""
        found: dict[str, tuple] = {}
        for index, tag in enumerate(tags):
            if tag not in FIX_TAGS:
                continue
            for ex in for_tag(self.library, tag, len(self.library), level, equipment):
                if ex.id not in skip:
                    found.setdefault(ex.id, (ex.difficulty, index, len(found), ex, tag))
        return [(row[3], row[4]) for row in sorted(found.values(), key=lambda row: row[:3])]

    def _base(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> Optional[Exercise]:
        """The exercise a message is about: named, in focus, just offered, last shown, or last named by the coach."""
        offered = split_tag(ctx.last_offer)[1]
        for ex_id in (p.exercise_id, ctx.exercise_id, offered, *reversed(ctx.shown_exercises)):
            if ex_id in self.library:
                return self.library[ex_id]
        last = next((t.text.casefold() for t in reversed(turns) if t.speaker == "coach"), "")
        named = sorted((last.find(ex.name.casefold()), ex.id) for ex in self.library.values() if ex.name.casefold() in last)
        return self.library[named[0][1]] if named else None

    @staticmethod
    def _the(ex: Exercise) -> str:
        return f"the {lower_first(ex.name)}"

    @staticmethod
    def _act(*exercises: Exercise) -> tuple[Action, ...]:
        return tuple(Action(ActionKind.EXERCISE, ex.id) for ex in exercises)

    @staticmethod
    def _dose(ex: Exercise, ctx: ChatContext) -> str:
        """The library dose, or an easier one for an exercise the player found too hard."""
        if ex.id in ctx.too_hard:
            return f"{easier_dose(ex.sets_reps)}, an easier dose as you found it too hard"
        return ex.sets_reps

    def _named(self, ex: Exercise, ctx: ChatContext) -> str:
        return f"{self._the(ex)} ({self._dose(ex, ctx)})"

    @staticmethod
    def _prefer(options: Sequence[Exercise], ctx: ChatContext, n: int = 2, fresh: bool = False) -> list[Exercise]:
        """Up to n options the player has not found too hard, ones not shown yet first when `fresh`.

        When every option was too hard, only the first comes back, and callers give it at an easier dose.
        """
        ok = [ex for ex in options if ex.id not in ctx.too_hard]
        if fresh:
            ok = [ex for ex in ok if ex.id not in ctx.shown_exercises] or ok
        return ok[:n] if ok else list(options[:1])

    def _focus_ex(self, ctx: ChatContext, picks: Sequence[Exercise], role: Role, f: Optional[Finding], offer: str,
                  topic: Optional[str] = None) -> ChatContext:
        first = picks[0]
        return replace(ctx, finding_id=f.id if f else ctx.finding_id, exercise_id=first.id,
                       shown_exercises=tuple(dict.fromkeys((*ctx.shown_exercises, *(e.id for e in picks)))),
                       topic=topic or self._tag("exercise", first.id, role, ctx),
                       last_offer=self._tag(offer, first.id, role, ctx))

    def _no_fit(self, ctx: ChatContext, what: str, tags: Sequence[str], skip: Sequence[str] = ()) -> _Out:
        """Nothing fits the level and equipment: say so, and offer only the changes that would help."""
        other = "other " if skip else ""
        if not self._options(tags, 3, None, skip):
            return _Out(f"I have no {other}gym exercise for {what} in my library yet.", replace(ctx, last_offer=None),
                        suggestions=(CHIP_NEXT, CHIP_FIRST, CHIP_PLAN))
        level = f"{LEVEL_NAMES[ctx.level]} " if ctx.level < 3 else ""
        gear = "" if ctx.equipment is None else f" that works with {gear_words(ctx.equipment)}"
        ways = [("say \"Any equipment is fine\"", CHIP_ANY_GEAR)] if ctx.equipment is not None else []
        ways += [("tell me a higher level", CHIP_LEVELS[ctx.level])] if ctx.level < 3 else []
        text = f"I have no {other}{level}exercise for {what}{gear}. To widen the choice, {' or '.join(w for w, _ in ways)}."
        return _Out(text, replace(ctx, last_offer=None), suggestions=(*(c for _, c in ways), CHIP_NEXT))

    def _how_to_fix(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        role = self._role(p, ctx)
        f, absent = self._target(p, ctx, role)
        if f is None:
            return self._absent(absent, role, ctx)
        guide = self._guide(f.id)
        name = guide["name"]
        if f.severity == Severity.STRENGTH:
            text = (f"{name[:1].upper() + name[1:]} is already a strength for {self._who(role, ctx)}, so keep doing "
                    f"what works. {self._voice(guide['feel'], role, ctx)}")
            offer = self._tag("show_best", f.id, role, ctx) if f.best_moment else "next"
            new = replace(ctx, finding_id=f.id, topic=self._tag("finding", f.id, role, ctx), last_offer=offer)
            return _Out(text, new, (self._show(f),), (self._best_chip(f, role, ctx), CHIP_FIRST, CHIP_NEXT))
        soft = self._soft(f)
        mine = role == self._me(ctx)
        cues = guide["cues"][:2 if f.category in POSITIONAL or soft else 3]
        if soft:
            # Practice for a plain reading or a position is general advice, not a fix for something the clip showed.
            m, who = f.metrics[0], "" if mine else " for the other player"
            if self._measured(m):
                lead = f"This clip does not show a clear fault in {name}{who}, so this is general practice:"
            elif self._observed(m):
                label = self._voice(self._label(m), role, ctx)
                lead = (f"I saw {self._whose(role, ctx)} {label} in this clip, but not clearly enough to judge, so this "
                        "is general practice:")
            else:
                lead = f"This clip has no reading for {name}{who}, so this is general practice:"

            def notes(short: bool) -> list[str]:
                return [SHUTTLE_SHORT if short else SHUTTLE_NOTE] if f.category in POSITIONAL else []
        else:
            lead = f"To work on {name}:" if mine else f"For the other player to work on {name}:"

            def notes(short: bool) -> list[str]:
                return self._notes(f, room=0 if short else WORD_LIMIT - 8)
        # To fit the limit, cues go one at a time, then the notes take their short forms, then the question goes.
        tries = [(k, False) for k in range(len(cues), 0, -1)] + [(1, True)]
        picks = self._prefer([ex for ex, _ in self._options(self._tags(f), ctx.level, ctx.equipment)], ctx)
        if not picks:
            out = self._no_fit(ctx, name, self._tags(f))
            for k, short in tries:
                text = " ".join([lead, *cues[:k], *notes(short), out.text])
                if word_count(text) <= WORD_LIMIT:
                    break
            return replace(out, text=text, ctx=replace(out.ctx, finding_id=f.id), actions=(self._show(f),))
        gym = sentence("Gym exercises: " + " and ".join(self._named(ex, ctx) for ex in picks))
        ask = "Want to know how to do the first one?"
        for k, short in tries:
            text = " ".join([lead, *cues[:k], gym, *notes(short)])
            if self._fits(turns, text, ask, False):
                break
        else:
            ask = ""
        new = self._focus_ex(ctx, picks, role, f, "exercise_detail", topic=self._tag("finding", f.id, role, ctx))
        new = new if ask else replace(new, last_offer=None)
        return _Out(f"{text} {ask}".strip(), new, (self._show(f), *self._act(*picks)),
                    (CHIP_HOW_EX, CHIP_ANOTHER, self._show_chip(f), CHIP_PLAN))

    def _exercise_request(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        if p.missing:
            return self._missing(p, ctx, turns)
        role = self._role(p, ctx)
        f, absent = self._target(p, ctx, role)
        if f is None:
            return self._absent(absent, role, ctx)
        name = self._guide(f.id)["name"]
        options = [ex for ex, _ in self._options(self._tags(f), ctx.level, ctx.equipment)]
        picks = self._prefer(options, ctx, fresh=True)
        if not picks:
            return self._no_fit(ctx, name, self._tags(f))
        one = len(picks) == 1
        these = "try this:" if one else "try these:"
        lead = f"For {name}, {these}" if role == self._me(ctx) else f"For the other player to work on {name}, {these}"
        lines = [f"- {ex.name} ({self._dose(ex, ctx)}): {ex.why}" for ex in picks]
        ask = "Want to know how to do it?" if one else "Want to know how to do the first one?"
        text = "\n".join([lead, *lines, ask])
        new = self._focus_ex(ctx, picks, role, f, "exercise_detail")
        return _Out(text, new, self._act(*picks), (CHIP_HOW_EX, CHIP_ANOTHER, CHIP_EASIER, CHIP_PLAN))

    def _exercise_detail(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        if p.missing:
            return self._missing(p, ctx, turns)
        ex = self._base(p, ctx, turns)
        if ex is None:
            return self._exercise_request(p, ctx, turns)
        role = self._role(p, ctx)
        the = self._the(ex)
        if p.aspect == "dose":
            eased = ", an easier dose as you found it too hard" if ex.id in ctx.too_hard else ""
            text = f"For {the}, do {easier_dose(ex.sets_reps) if eased else ex.sets_reps}{eased}. Key point: {ex.cues[0]}."
        elif p.aspect == "equipment":
            text = (f"{the[:1].upper() + the[1:]} {agree(ex, 'needs', 'need')} {gear_words(ex.equipment)}."
                    + ("" if ex.equipment else f" You can do {agree(ex, 'it', 'them')} anywhere."))
        elif p.aspect == "mistakes":
            text = f"Common mistakes with {the}: {'; '.join(ex.mistakes)}."
        elif p.aspect == "safety":
            text = f"Safety for {the}: {ex.safety}"
        else:
            text = compose([f"{ex.name}: {self._dose(ex, ctx)}", f"Cues: {'; '.join(ex.cues[:2])}",
                            f"Watch out for: {ex.mistakes[0]}", f"Safety: {ex.safety}"],
                           [f"Also: {'; '.join(ex.cues[2:])}"] if ex.cues[2:] else [], WORD_LIMIT - 10)
        new = self._focus_ex(ctx, [ex], role, None, "another")
        return _Out(f"{text} Want another exercise for the same goal?", new, self._act(ex),
                    (CHIP_WHY_EX, CHIP_EASIER, CHIP_HARDER, CHIP_ANOTHER))

    def _exercise_why(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        if p.missing:
            return self._missing(p, ctx, turns)
        ex = self._base(p, ctx, turns)
        if ex is None:
            return self._exercise_request(p, ctx, turns)
        role = self._role(p, ctx)
        f, _ = self._target(p, ctx, role)
        shared = [t for t in self._tags(f) if t in ex.fix_tags]
        text = f"Why {self._the(ex)}? It {lower_first(ex.why)}"
        if f is not None and shared:
            name, goal = self._guide(f.id)["name"], FIX_TAGS[shared[0]]
            text += (f" It is general practice for {name}: {goal}." if self._soft(f)
                     else f" It works on {self._whose(role, ctx)} goal for {name}: {goal}.")
        new = self._focus_ex(ctx, [ex], role, None, "exercise_detail")
        return _Out(f"{text} Want to know how to do it?", new, self._act(ex), (CHIP_HOW_EX, CHIP_ANOTHER, CHIP_PLAN))

    def _missing(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        row = self.kb.exercise_aliases["missing"][p.missing]
        role = self._role(p, ctx)
        f, _ = self._target(p, ctx, role)
        tags = [t for t in self._tags(f) if t in row["fix_tags"]] or list(row["fix_tags"])
        found = [(t, [ex for ex, _ in self._options([t], ctx.level, ctx.equipment)]) for t in tags]
        usable = [(t, options) for t, options in found if options]
        fresh = [(t, options) for t, options in usable if any(ex.id not in ctx.too_hard for ex in options)]
        tag, options = (fresh or usable or [(tags[0], [])])[0]
        picks = self._prefer(options, ctx)
        text = f"I do not have {lower_first(row['name'])} in my library, so I have no how-to clip for it."
        if not picks:
            widen = ((CHIP_ANY_GEAR,) if ctx.equipment is not None else ()) + ((CHIP_LEVELS[ctx.level],) if ctx.level < 3 else ())
            return _Out(f"{text} I have nothing similar that fits your equipment and level.", replace(ctx, last_offer=None),
                        suggestions=(*widen, CHIP_EXERCISES))
        similar = " or ".join(self._named(ex, ctx) for ex in picks)
        ask = "Want to know how to do the first one?" if len(picks) > 1 else "Want to know how to do it?"
        text += f" For a similar goal ({FIX_TAGS[tag]}), try {similar}. {ask}"
        new = self._focus_ex(ctx, picks, role, f, "exercise_detail")
        return _Out(text, new, self._act(*picks), (CHIP_HOW_EX, CHIP_ANOTHER, CHIP_EASIER))

    # ---- easier, harder, another and gear changes -----------------------------------------------------

    def _exercise_adjust(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        if split_tag(ctx.topic)[0] == "plan" and not p.exercise_id and p.adjust != "another":
            return self._adjust_plan(p, ctx)
        if p.missing:
            return self._missing(p, ctx, turns)
        base = self._base(p, ctx, turns)
        if base is None:
            return self._exercise_request(replace(p, adjust=None), ctx, turns)
        role = self._role(p, ctx)
        f, _ = self._target(p, ctx, role)
        # The goal is the active finding's, re-applied to every candidate; a library edge alone is never trusted.
        ftags = list(self._tags(f))
        shared = [t for t in ftags if t in base.fix_tags]
        goal = ftags if ftags and (shared or not p.exercise_id) else list(base.fix_tags)
        tag = shared[0] if shared else goal[0]
        tags = [tag, *(t for t in goal if t != tag)]
        gear = gear_words(ctx.equipment)
        the = self._the(base)
        mode = p.adjust if p.adjust in ("easier", "harder", "another") else None
        if mode is None and not p.equipment:
            text = (f"{the[:1].upper()}{the[1:]} {agree(base, 'needs', 'need')} {gear_words(base.equipment)}. "
                    "Tell me what equipment you have, or ask for an easier or harder one.")
            new = self._focus_ex(ctx, [base], role, f, "another")
            return _Out(text, new, self._act(base), (CHIP_NO_GEAR, CHIP_EASIER, CHIP_HARDER, CHIP_ANOTHER))
        was_hard = base.id in ctx.too_hard
        if mode == "easier":
            # Asking for an easier one means this one was too hard: it only comes back at an easier dose.
            ctx = replace(ctx, too_hard=tuple(dict.fromkeys((*ctx.too_hard, base.id))))
        elif mode == "harder" and was_hard and fits(base, tag, ctx.level, ctx.equipment):
            ctx = replace(ctx, too_hard=tuple(e for e in ctx.too_hard if e != base.id))
            text = (f"Harder option: the full dose of {the} ({base.sets_reps}). It works on the same goal: "
                    f"{FIX_TAGS[tag]}. Want to know how to do it?")
            return _Out(text, self._focus_ex(ctx, [base], role, f, "exercise_detail"), self._act(base),
                        (CHIP_HOW_EX, CHIP_EASIER, CHIP_HARDER, CHIP_ANOTHER))
        pick, trains = None, tag
        if mode in ("easier", "harder"):
            low, high = (1, min(ctx.level, base.difficulty - 1)) if mode == "easier" else (ctx.level, min(3, base.difficulty + 1))
            ctx = replace(ctx, level=max(low, high))
            pick = step(self.library, base, mode, tag, ctx.level, ctx.equipment, ctx.too_hard)
            if pick is None and fits(base, tag, ctx.level, ctx.equipment):
                eased = easier_dose(base.sets_reps)
                tip = (harder_tip(base) if mode == "harder"
                       else f"Go lighter than {eased}: shorter holds or fewer reps, and longer rests." if was_hard
                       else f"Try an easier dose instead: {eased}, with longer rests between sets.")
                most = "easiest" if mode == "easier" else "hardest"
                text = (f"{the[:1].upper()}{the[1:]} {agree(base, 'is', 'are')} already the {most} exercise I have "
                        f"for this goal with {gear}. {tip} Want another exercise for the same goal?")
                return _Out(text, self._focus_ex(ctx, [base], role, f, "another"), self._act(base),
                            (CHIP_HOW_EX, CHIP_ANOTHER, CHIP_PLAN))
            lead = "Easier option:" if mode == "easier" else "Harder option:"
        elif mode is None and fits(base, tag, ctx.level, ctx.equipment):
            keep = f" at the easier dose ({easier_dose(base.sets_reps)})" if was_hard else ""
            it = agree(base, "it", "them")
            text = (f"Good news: {the} {agree(base, 'needs', 'need')} {gear_words(base.equipment)}, so you can keep "
                    f"{it}{keep}. Want to know how to do {it}?")
            return _Out(text, self._focus_ex(ctx, [base], role, f, "exercise_detail"), self._act(base),
                        (CHIP_HOW_EX, CHIP_ANOTHER, CHIP_EASIER, CHIP_PLAN))
        else:
            lead = "Another one:" if mode == "another" else f"That one needs {gear_words(base.equipment)}, so with {gear} try"
        if pick is None:
            options = self._options(tags, ctx.level, ctx.equipment, skip=(base.id,))
            chosen = self._prefer([ex for ex, _ in options], ctx, 1, fresh=True)
            if not chosen:
                return self._no_fit(ctx, "this goal", tags, skip=(base.id,))
            pick = chosen[0]
            trains = dict((ex.id, t) for ex, t in options)[pick.id]
        new = self._focus_ex(ctx, [pick], role, f, "exercise_detail")
        if pick.id in ctx.too_hard:
            # Only exercises the player already found too hard are left: say so and give an easier dose.
            head = f"That one needs {gear_words(base.equipment)}. " if mode is None else ""
            level = f"{LEVEL_NAMES[ctx.level]} " if ctx.level < 3 else ""
            where = "" if ctx.equipment is None else f" with {gear}"
            text = (f"{head}The only other {level}exercise I have for this goal{where} is {self._the(pick)}, which you "
                    f"found too hard. Try it at an easier dose: {easier_dose(pick.sets_reps)}. Want to know how to do it?")
            widen = ((CHIP_ANY_GEAR,) if ctx.equipment is not None else ()) + ((CHIP_LEVELS[ctx.level],) if ctx.level < 3 else ())
            return _Out(text, new, self._act(pick), (CHIP_HOW_EX, *widen, CHIP_PLAN))
        same = (f"It works on the same goal: {FIX_TAGS[trains]}." if trains == tag
                else f"It works on a related goal: {FIX_TAGS[trains]}.")
        text = f"{lead} {self._the(pick)} ({pick.sets_reps}). {same} It {lower_first(pick.why)} Want to know how to do it?"
        return _Out(text, new, self._act(pick), (CHIP_HOW_EX, CHIP_EASIER, CHIP_HARDER, CHIP_ANOTHER))

    # ---- plans ----------------------------------------------------------------------------------------

    def _plan(self, role: Role, level: int, ctx: ChatContext) -> tuple:
        """What a plan contains, without its level and equipment fields; () when nothing fits."""
        plan = build_plan(self.report, self.library, player=role, level=level, equipment=ctx.equipment,
                          avoid=ctx.too_hard)
        return (plan.findings, plan.weeks) if plan.weeks else ()

    def _plan_out(self, role: Role, ctx: ChatContext, kind: str = "") -> _Out:
        plan = build_plan(self.report, self.library, player=role, level=ctx.level, equipment=ctx.equipment,
                          avoid=ctx.too_hard)
        if not plan.weeks:
            top = self.report.top_findings(len(self.report.findings), role)
            if not top:
                # Plans are built only from things that need work; a plain reading gets practice ideas, not a plan.
                text = (f"I found nothing that clearly needs work for {self._who(role, ctx)} in this clip, so there is "
                        "nothing to plan yet.")
                plain = self._plain(role)
                if plain:
                    f = plain[0]
                    name = self._voice(self._guide(f.id)["name"], role, ctx)
                    new = replace(ctx, finding_id=f.id, topic=self._tag("finding", f.id, role, ctx),
                                  last_offer=self._tag("how_to_fix", f.id, role, ctx))
                    return _Out(f"{text} One reading worth a look is {name}. {SOFT_ASK}", new, (self._show(f),),
                                (CHIP_EXERCISES, self._show_chip(f), CHIP_LIMITS))
                if self.report.strengths(1, role):
                    return _Out(text + " Want to hear what went well?",
                                replace(ctx, last_offer=self._tag("strengths", "", role, ctx)),
                                suggestions=(CHIP_WELL, CHIP_LIMITS))
                return _Out(text, replace(ctx, last_offer=None), suggestions=(CHIP_LIMITS, CHIP_COMPARE))
            return self._no_fit(ctx, "a practice plan", list(dict.fromkeys(t for f in top for t in self._tags(f))))
        names = [self._voice(self._guide(fid)["name"], role, ctx) for fid in plan.findings]
        goals = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        blocks = plan.weeks[0].sessions[0].blocks
        first = ", ".join(f"{lower_first(b.name)} ({b.dose}{', easier dose' if b.exercise_id in ctx.too_hard else ''})"
                          for b in blocks)
        weeks, per = number_word(len(plan.weeks)), number_word(len(plan.weeks[0].sessions))
        whom = "" if role == self._me(ctx) else " for the other player"
        gear = ("" if ctx.equipment is None else " and no equipment" if not ctx.equipment
                else f" and nothing beyond {gear_words(ctx.equipment)}")
        what = {"easier": "an easier", "harder": "a harder"}.get(kind, "a")
        intro = f"Here is {what} {weeks}-week plan{whom} to work on {goals}, with {per} sessions a week"
        parts = [intro + gear, f"Session one: {first}"]
        if any(self.findings[(fid, role)].category in POSITIONAL for fid in plan.findings):
            parts.append(PLAN_SHUTTLE_NOTE)
        tail = [SAFE_LINE, "Want to know how to do the first one?"]
        if word_count(" ".join([*parts, *tail])) > WORD_LIMIT:
            parts[0] = intro  # over the word limit, the gear phrase goes first: the listed exercises already respect it
        if len(plan.weeks) > 1:
            two = f"Week two: {lower_first(plan.weeks[1].focus)}"
            if word_count(" ".join([*parts, two, *tail])) <= WORD_LIMIT:
                parts.append(two)
        picks = [self.library[b.exercise_id] for b in blocks]
        new = replace(self._focus_ex(ctx, picks, role, None, "exercise_detail", topic=self._tag("plan", "", role, ctx)),
                      finding_id=blocks[0].finding_id)
        return _Out(compose([*parts, *tail]), new, self._act(*picks), (CHIP_HOW_EX, CHIP_EASIER, CHIP_HARDER, CHIP_NO_GEAR))

    def _practice_plan(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        return self._plan_out(p.player if p.player in self.report.players else self._me(ctx), ctx)

    def _adjust_plan(self, p: Parse, ctx: ChatContext) -> _Out:
        role = split_tag(ctx.topic)[2]
        role = role if role in self.report.players else self._me(ctx)
        if p.adjust not in ("easier", "harder"):
            if p.equipment:
                return self._plan_out(role, ctx)
            text = "I can make the plan easier or harder, or fit it to the equipment you have. Which would you like?"
            return _Out(text, replace(ctx, last_offer=None), suggestions=(CHIP_EASIER, CHIP_HARDER, CHIP_NO_GEAR))
        move = -1 if p.adjust == "easier" else 1
        now = self._plan(role, ctx.level, ctx)
        if not now:
            return self._plan_out(role, replace(ctx, level=min(3, max(1, ctx.level + move))), p.adjust)
        level = ctx.level
        # Walk the levels until the plan really changes: a higher level can still start with the same easy exercises.
        for nxt in range(ctx.level + move, 0 if move < 0 else 4, move):
            plan = self._plan(role, nxt, ctx)
            if not plan:
                break
            if plan != now:
                return self._plan_out(role, replace(ctx, level=nxt), p.adjust)
            level = nxt
        text = PLAN_SAME[p.adjust]
        other = CHIP_HARDER if move < 0 else CHIP_EASIER
        return _Out(text, replace(ctx, level=level, last_offer=None), suggestions=(other, CHIP_NO_GEAR, CHIP_FIRST))

    # ---- video, numbers and comparisons ---------------------------------------------------------------

    def _show_moment(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        role = self._role(p, ctx)
        f, absent = self._target(p, ctx, role)
        if f is None:
            return self._absent(absent, role, ctx)
        name = self._guide(f.id)["name"]
        strong = f.severity == Severity.STRENGTH
        here = replace(ctx, finding_id=f.id, topic=self._tag("finding", f.id, role, ctx))
        fix_offer = None if strong else self._tag("how_to_fix", f.id, role, ctx)
        if p.best or strong:
            m = f.best_moment
            if m is None:
                again = bool(f.issue_moments) and not strong
                which = "the moment I looked at" if self._soft(f) else "the moment that needs work"
                text = f"I have no standout good moment for {name} in this clip." + (
                    f" Want to see {which} instead?" if again else "")
                offer = self._tag("show_moment", f.id, role, ctx) if again else "summary"
                return _Out(text, replace(here, last_offer=offer), (self._show(f),),
                            (self._show_chip(f), CHIP_FIRST, CHIP_WELL))
            ask = ((SOFT_ASK if self._soft(f) else "Want to know how to fix the rest?") if fix_offer
                   else "Want to hear what to work on first?")
            text = " ".join([f"Playing it now: {lower_first(self._voice(m.label, role, ctx))}.", *self._notes(f), ask])
            new = replace(here, frame=m.key_frame, last_offer=fix_offer or "summary")
            seek = Action(ActionKind.SEEK, f.id, f.moments.index(m))
            return _Out(text, new, (self._show(f), seek), (self._work_chip(f, role, ctx), CHIP_NEXT, CHIP_FIRST))
        issues = f.issue_moments
        if not issues and not any(self._observed(x) for x in f.metrics):
            return self._absent(f.id, role, ctx)
        if not issues:
            best = f.best_moment is not None
            text = " ".join([f"There is no single moment to show for {name}, because it comes from the whole video.",
                             *self._notes(f), "Want to see the best moment instead?" if best else self._fix_ask(f)])
            offer = self._tag("show_best", f.id, role, ctx) if best else fix_offer
            return _Out(text, replace(here, last_offer=offer), (self._show(f),),
                        (self._best_chip(f, role, ctx, self._work_chip(f, role, ctx)), CHIP_NEXT))
        keys = [m.key_frame for m in issues]
        m = issues[(keys.index(ctx.frame) + 1) % len(issues)] if ctx.finding_id == f.id and ctx.frame in keys else issues[0]
        more = (f"It is one of {number_word(len(issues))} moments like this, so ask again to see the next."
                if len(issues) > 1 else "")
        # Once an exercise for this finding is on screen, the fix was already given: offer its how-to instead.
        given = self.library.get(ctx.exercise_id) if ctx.finding_id == f.id and ctx.exercise_id else None
        given = given if given is not None and set(given.fix_tags) & set(f.fix_tags) else None
        ask = f"Want to know how to do {self._the(given)}?" if given else self._fix_ask(f)
        text = " ".join(s for s in [f"Playing it now: {lower_first(self._voice(m.label, role, ctx))}.", more, *self._notes(f),
                                    ask] if s)
        offer = self._tag("exercise_detail", given.id, role, ctx) if given else fix_offer
        new = replace(here, frame=m.key_frame, last_offer=offer)
        seek = Action(ActionKind.SEEK, f.id, f.moments.index(m))
        chips = ((CHIP_HOW_EX, CHIP_SHOW if more else CHIP_WHY, CHIP_ANOTHER) if given else
                 (self._work_chip(f, role, ctx), CHIP_SHOW if more else CHIP_WHY,
                  CHIP_NEXT if self._soft(f) else CHIP_EXERCISES))
        return _Out(text, new, (self._show(f), seek, *(self._act(given) if given else ())), chips)

    def _uncompared(self, name: str, mine: Metric, theirs: Metric, other: Role, ctx: ChatContext) -> str:
        """Why two players' readings are not compared, telling a reading never seen apart from one seen too unclearly
        to use. A confidence-0 value is never quoted."""
        def unclear(x: Metric) -> bool:
            return self._observed(x) and not self._measured(x)
        label = self._label(mine)
        if unclear(mine) and unclear(theirs):
            return f"I saw your {label} and the other player's in this clip, but neither clearly enough to compare."
        if unclear(mine):
            rest = (" it with the other player's" if self._observed(theirs) else
                    ", and I have no reading for the other player")
            return f"I saw your {label} in this clip, but not clearly enough to compare{rest}."
        if unclear(theirs):
            rest = " it with yours" if self._observed(mine) else ", and I have no reading for you"
            return (f"I saw the other player's {self._voice(label, other, ctx)} in this clip, but not clearly enough to "
                    f"compare{rest}.")
        who = "either of you" if not (self._observed(mine) or self._observed(theirs)) else (
            "you" if not self._observed(mine) else "the other player")
        return f"I cannot compare {name}, because I have no reading for {who} in this clip."

    def _compare_players(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        me = self._me(ctx)
        if len(self.report.players) < 2:
            return _Out("Only one player is in this clip, so there is nobody to compare with. Want to hear what to "
                        "work on first?", replace(ctx, last_offer="summary"), suggestions=(CHIP_FIRST, CHIP_WELL))
        other = next(r for r in self.report.players if r != me)
        fid = p.finding_id or ctx.finding_id
        pair = (self.findings.get((fid, me)), self.findings.get((fid, other))) if fid else (None, None)
        if all(pair):
            a, b = pair
            m = a.metrics[0]
            n = next((x for x in b.metrics if x.name == m.name), None)
            if n is not None:
                name = self._guide(fid)["name"]
                if not (self._measured(m) and self._measured(n)):
                    text = self._uncompared(name, m, n, other, ctx) + " Want to hear what to work on first?"
                    return _Out(text, replace(ctx, finding_id=fid, last_offer="summary"), (self._show(a), self._show(b)),
                                (CHIP_FIRST, CHIP_WELL, CHIP_LIMITS))
                line = (f"For {name}, your {self._label(m)} was {fmt(m.value, m.unit)} and the "
                        f"other player's was {fmt(n.value, n.unit)}")
                rough = ("Timing comes from video frames, so treat a small gap as rough" if m.unit == "s" else
                         "You stand at different distances from the camera, so treat this as a rough comparison")
                g = self._guide_range(a, m)
                guide = f"A practice guide is {g}, a coaching aim rather than a tested norm" if g else ""
                if a.severity == Severity.STRENGTH:
                    ask, offer, chip = "Want to hear what to work on first?", "summary", CHIP_FIRST
                elif self._soft(a):
                    ask, offer, chip = SOFT_ASK, self._tag("how_to_fix", fid, me, ctx), CHIP_EXERCISES
                else:
                    ask, offer, chip = "Want to know how to fix yours?", self._tag("how_to_fix", fid, me, ctx), CHIP_FIX
                room = self._room(turns, ask)
                text = compose([line, rough, *self._notes(a, f"{sentence(line)} {sentence(rough)}", room=room)],
                               [guide], room)
                if not self._fits(turns, text, ask, True):
                    ask, offer = "", None
                new = replace(ctx, finding_id=fid, topic=self._tag("finding", fid, me, ctx), last_offer=offer)
                return _Out(f"{text} {ask}".strip(), new, (self._show(a), self._show(b)),
                            (chip, self._show_chip(a), CHIP_FIRST), quoted=True)
        counts = [(len(self.report.top_findings(len(self.report.findings), r)),
                   len(self.report.strengths(len(self.report.findings), r))) for r in (me, other)]

        def tally(work: int, good: int) -> str:
            things = ("nothing clear to work on" if not work else
                      f"{number_word(work)} thing{'' if work == 1 else 's'} to work on")
            return things + " and " + ("no clear strengths" if not good else
                                       f"{number_word(good)} strength{'' if good == 1 else 's'}")
        text = (f"You have {tally(*counts[0])}. The other player has {tally(*counts[1])}. "
                "Want to hear what to work on first?")
        return _Out(text, replace(ctx, last_offer="summary"), suggestions=(CHIP_FIRST, CHIP_WELL, CHIP_PLAN))

    def _metric_question(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        role = self._role(p, ctx)
        f, absent = self._target(p, ctx, role)
        if f is None:
            return self._absent(absent, role, ctx)
        m = self._metric(f, p, turns)
        strong = f.severity == Severity.STRENGTH
        ask = "" if strong else self._fix_ask(f)
        if self._measured(m):
            # A verdict loses its rank sentence before the closing question or the preview note would not fit.
            options = ([self._verdict(f, m, ctx), self._verdict(f, m, ctx, rank=False)] if p.judge
                       else [self._reading(f, m, ctx)])
            options = [(lines, " ".join(lines)) for lines in options]
        else:
            whose, label = self._whose(role, ctx), self._voice(self._label(m), role, ctx)
            said = (f"I saw {whose} {label} in this clip, but not clearly enough to give a number" if self._observed(m)
                    else f"I have no reading of {whose} {label} in this clip")
            # A producer headline can carry a value too unclear to quote, so it is repeated only when no reading is.
            unclear = any(self._observed(x) and not self._measured(x) for x in f.metrics)
            head = "" if unclear else self._voice(f.headline, role, ctx)
            options = [([said, head], head or said)]

        def build(lines: list[str], heard: str, note: bool) -> str:
            room = self._room(turns, ask, note=note)
            return compose([*lines, *self._notes(f, heard, room=room)], (), room)

        for lines, heard in options:
            text = build(lines, heard, note=False)
            if self._note_due(turns) and re.search(r"\d", text):
                text = build(lines, heard, note=True)
            if self._fits(turns, text, ask, bool(re.search(r"\d", text))):
                break
        else:
            ask = ""
        text = f"{text} {ask}".strip()
        new = replace(ctx, finding_id=f.id, topic=self._tag("finding", f.id, role, ctx),
                      last_offer=self._tag("how_to_fix", f.id, role, ctx) if ask else None)
        first = CHIP_FIRST if strong else self._work_chip(f, role, ctx)
        return _Out(text, new, (self._show(f),), (first, self._show_chip(f), CHIP_LIMITS),
                    quoted=any(c.isdigit() for c in text))

    def _progress(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        if not ctx.history_reports:
            text = ("This is the only session I have, so I cannot compare yet. Record another clip later and I can "
                    "show what changed.")
            return _Out(text, replace(ctx, last_offer="summary"), suggestions=(CHIP_FIRST, CHIP_PLAN))
        role = self._role(p, ctx)
        old = {(f.id, f.player): f for f in ctx.history_reports[-1].findings}
        f, absent = self._target(p, ctx, role)
        if absent:
            return self._absent(absent, role, ctx)
        # A named reading answers for itself. Otherwise the reading in focus comes first, then the ranked ones; a
        # reading not measured in both clips is never compared.
        pool = [f] if f is not None else []
        if not p.finding_id:
            pool += [x for x in self.report.top_findings(len(self.report.findings), role) if x not in pool]
        whose = self._whose(role, ctx)

        def earlier(x: Finding) -> Optional[Metric]:
            before = old.get((x.id, x.player))
            return next((y for y in before.metrics if y.name == x.metrics[0].name), None) if before else None
        for now in pool:
            m, prev = now.metrics[0], earlier(now)
            if prev is None or not (self._measured(m) and self._measured(prev)):
                continue
            change = f"{whose.capitalize()} {self._voice(self._label(m), role, ctx)} went from {fmt(prev.value, m.unit)} to {fmt(m.value, m.unit)}"
            verdict = ""
            guide = self._guide_range(now, m)
            if guide:
                lo, hi = m.target
                gap = lambda v: 0.0 if lo <= v <= hi else min(abs(v - lo), abs(v - hi))
                if abs(m.value - prev.value) <= 0.01 * max(abs(prev.value), abs(m.value)):
                    verdict = "That is about the same as last time"
                elif gap(m.value) == gap(prev.value) == 0.0:
                    verdict = "Both are inside the practice guide"
                else:
                    verdict = (f"That is {'closer to' if gap(m.value) < gap(prev.value) else 'further from'} the practice "
                               f"guide of {guide}")
            room = self._room(turns, keep=0)
            lines = [change, verdict, "Different clips and camera angles can change readings, so treat it as a rough trend"]
            heard = " ".join(sentence(s) for s in lines if s)
            text = compose([*lines, *self._notes(now, heard, room=room)], (), room)
            if now is not f:
                lead = self._unchanged(f, earlier(f), role, ctx)
                text = f"{lead} {text}" if word_count(lead) + word_count(text) <= room else text
            new = replace(ctx, finding_id=now.id, topic=self._tag("finding", now.id, role, ctx), last_offer=None)
            return _Out(text, new, (self._show(now),), (self._work_chip(now, role, ctx), CHIP_FIRST, CHIP_PLAN), quoted=True)
        if f is not None:
            text = self._unchanged(f, earlier(f), role, ctx) + " Want to hear what to work on first?"
            new = replace(ctx, finding_id=f.id, topic=self._tag("finding", f.id, role, ctx), last_offer="summary")
            return _Out(text, new, suggestions=(CHIP_FIRST, CHIP_PLAN))
        text = "I have nothing clear enough to compare between the two clips. Want to hear what to work on first?"
        return _Out(text, replace(ctx, last_offer="summary"), suggestions=(CHIP_FIRST, CHIP_PLAN))

    def _unchanged(self, f: Finding, prev: Optional[Metric], role: Role, ctx: ChatContext) -> str:
        """Why f's main reading is not compared between clips, clip by clip: never seen, or seen too unclearly to
        use. A confidence-0 value is never quoted."""
        def state(x: Optional[Metric]) -> str:
            if x is None or not self._observed(x):
                return "none"
            return "clear" if self._measured(x) else "unclear"
        now, before = state(f.metrics[0]), state(prev)
        if now == before == "none":
            why = "I have no reading of it in either clip"
        elif now == before == "unclear":
            why = "I saw it in both clips, but not clearly enough to compare"
        else:
            said = {"none": "I have no reading of it in {}", "unclear": "I saw it in {}, but not clearly enough to compare"}
            why = ", and ".join(said[s].format(clip) for s, clip in ((now, "this clip"), (before, "the earlier clip"))
                                if s != "clear")
        label = self._voice(self._label(f.metrics[0]), role, ctx)
        return f"I cannot say whether {self._whose(role, ctx)} {label} changed: {why}."

    # ---- words, tactics, limits, safety and level -----------------------------------------------------

    def _terms(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        if p.term not in self.kb.glossary:
            text = "Which word should I explain? For example the split step, the base position or a projected angle."
            return _Out(text, replace(ctx, last_offer=None), suggestions=("What is a split step?", "What is the base?"))
        entry = self.kb.glossary[p.term]
        text = f"{entry['name'][:1].upper()}{entry['name'][1:]}: {entry['define']}"
        return _Out(text, replace(ctx, topic=f"term:{p.term}", last_offer=None), suggestions=(CHIP_FIRST, CHIP_WELL, CHIP_PLAN))

    def _tactics_tip(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        role = self._role(p, ctx)
        fid = p.finding_id or ctx.finding_id
        tid = next((t for t, row in self.kb.tactics.items() if fid in row["findings"]), None)
        if tid is None:
            top = [f for f in self.report.top_findings(len(self.report.findings), role) if f.category in POSITIONAL]
            tid = next((t for f in top for t, row in self.kb.tactics.items() if f.id in row["findings"]), "base_spot")
        row = self.kb.tactics[tid]
        f = next((self.findings[(x, role)] for x in row["findings"] if (x, role) in self.findings), None)
        required = [f"{row['title'][:1].upper()}{row['title'][1:]}: {row['text']}"]
        if f is not None and f.id in DESCRIPTIVE and "description" not in row["text"]:
            required.append(DESCRIPTIVE_NOTE)
        extra = [f"In this clip: {lower_first(self._voice(f.headline, f.player, ctx))}"] if f is not None else []
        new = replace(ctx, finding_id=f.id if f else ctx.finding_id, topic=self._tag("tactic", tid, role, ctx), last_offer=None)
        chips = (self._show_chip(f, CHIP_FIRST), CHIP_EXERCISES, CHIP_LIMITS)
        return _Out(compose(required, extra), new, (self._show(f),) if f else (), chips)

    def _limits(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        lim = self.kb.limits
        if p.limit == "preview":
            text = lim["preview"]["text_preview" if self.preview else "text_refined"]
        elif p.limit in lim["cannot"]:
            text = lim["cannot"][p.limit]["text"]
        else:
            text = lim["general"]["text"]
        return _Out(text, replace(ctx, last_offer=None), suggestions=(CHIP_FIRST, CHIP_PLAN, CHIP_WELL))

    def _safety(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        return _Out(self.kb.safety["text"], replace(ctx, last_offer=None), (), tuple(self.kb.safety["suggestions"]),
                    fixed=True)

    def _set_level(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        if p.level is None:
            return _Out("Which level suits you best: beginner, intermediate or advanced?", replace(ctx, last_offer=None),
                        suggestions=CHIP_LEVELS)
        ctx = replace(ctx, level=p.level)
        if split_tag(ctx.topic)[0] == "plan":
            return _Out(f"{LEVEL_TEXT[p.level]} Want me to redo the plan at this level?",
                        replace(ctx, last_offer="practice_plan"), suggestions=(CHIP_PLAN, CHIP_EXERCISES))
        return _Out(f"{LEVEL_TEXT[p.level]} Want some exercises for what to work on first?",
                    replace(ctx, last_offer="exercise_request"), suggestions=(CHIP_EXERCISES, CHIP_PLAN, CHIP_FIRST))

    # ---- yes, no and everything else ------------------------------------------------------------------

    def _affirm(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        kind, ref, role = split_tag(ctx.last_offer)
        intent = OFFERS.get(kind)
        if intent is None:
            return _Out("Great. What would you like next?", ctx, suggestions=(CHIP_FIRST, CHIP_PLAN, CHIP_WELL))
        follow = replace(p, intent=intent, player=role or self._me(ctx),
                         finding_id=ref if ref in FINDING_CATALOG else None,
                         exercise_id=ref if ref in self.library else None, missing=None, equipment=None,
                         aspect=None, best=kind == "show_best", adjust="another" if kind == "another" else None)
        return self.handlers[intent](follow, replace(ctx, last_offer=None), turns)

    def _deny(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        kind = split_tag(ctx.last_offer)[0]
        if kind in ("how_to_fix", "show_moment", "show_best", "explain"):
            return _Out("No problem. Want to look at the next thing instead?", replace(ctx, last_offer="next"),
                        suggestions=(CHIP_NEXT, CHIP_PLAN, CHIP_WELL))
        if kind in ("exercise_detail", "exercise_request", "another"):
            return _Out("No problem. Want a two-week practice plan instead?", replace(ctx, last_offer="practice_plan"),
                        suggestions=(CHIP_PLAN, CHIP_NEXT, CHIP_WELL))
        if kind == "practice_plan":
            return _Out("No problem. Want to hear what went well?", replace(ctx, last_offer="strengths"),
                        suggestions=(CHIP_WELL, CHIP_FIRST))
        return _Out("No problem. What would you like to know?", replace(ctx, last_offer=None),
                    suggestions=(CHIP_FIRST, CHIP_PLAN, CHIP_WELL))

    def _out_of_scope(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        text = ("I can only help with this clip, how you move in it and gym exercises for badminton, so I cannot "
                "answer that. You could ask what to work on first or for a practice plan.")
        return _Out(text, replace(ctx, last_offer=None), suggestions=(CHIP_FIRST, CHIP_PLAN, CHIP_LIMITS))

    def _unknown(self, p: Parse, ctx: ChatContext, turns: Sequence[Turn]) -> _Out:
        if p.judge:
            top = self.report.top_findings(1, self._me(ctx)) or self._plain(self._me(ctx))[:1]
            example = f", for example {self._guide(top[0].id)['name']}," if top else ""
            text = (f"Do you mean one of the readings from your clip? Name it{example} and I will say how it compares "
                    "with the practice guide and how sure I am.")
            return _Out(text, ctx, suggestions=(CHIP_FIRST, CHIP_WELL, CHIP_LIMITS))
        text = (f"Sorry, I did not catch that. You could ask \"{CHIP_FIRST}\", \"{CHIP_FIX}\" or "
                f"\"{CHIP_PLAN}\".")
        return _Out(text, ctx, suggestions=(CHIP_FIRST, CHIP_FIX, CHIP_PLAN))
