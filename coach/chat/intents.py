"""Deterministic intent classifier for the chat coach.

Words live in coach/kb: lexicon.json (intent cues and entity words), fix_guides.json (finding topic
words), glossary.json, limits.json, safety.json and exercise_aliases.json. This module normalises
text, matches token patterns against those tables and applies a few context rules, so improving
the classifier mostly means editing JSON.

Cue syntax: tokens separated by spaces; "*" skips 0 to 3 tokens; a leading "^" anchors at the start
and a trailing "$" at the end; a token ending in "~" matches any word with that prefix. An intent
scores its best cue plus 0.3 times the sum of its other matched cues.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Optional, Sequence

from coach.chat.kb import INTENTS, KnowledgeBase
from coach.schema import FINDING_CATALOG, SEVERITY_RANK, Exercise, Report, Role, Turn

MAX_TOKENS = 300
MAX_CHARS = 4000
FLOOR = 2.5        # a best score below this asks the player to rephrase
MARGIN = 0.5       # top scores closer than this are decided by context, with lower confidence
FULL = 7.0         # a score of FULL or more is full confidence
GAP = 3            # "*" in a cue skips up to this many tokens
FINDING_MIN = 2.0  # topic-word weight needed to name a finding
JUDGE = 5.0        # weight of "is that good?" for the reading or exercise in focus
RESCUE = 5.0       # weight of an intent read from word classes when no listed phrase claims the message

ORDER = {intent: i for i, intent in enumerate(INTENTS)}
CATALOG_ORDER = {fid: i for i, fid in enumerate(FINDING_CATALOG)}
APOSTROPHES = re.compile("[\u2018\u2019\u201b\u02bc\u0060\u00b4]")
SEPARATORS = re.compile(r"[_\-/\\]+")
PUNCTUATION = re.compile(r"[^\w\s]")
LONG_REPEATS = re.compile(r"(.)\1{2,}")
REPEATS = re.compile(r"(.)\1+")
REUSABLE = frozenset({"summary", "strengths", "explain", "how_to_fix", "show_moment", "metric_question",
                      "compare_players", "practice_plan", "exercise_request", "tactics_tip", "progress"})
EXERCISE_WORDS = frozenset({"exercise", "exercises", "drill", "drills", "workout", "workouts"})
WHY_WORDS = frozenset({"why", "how come"})
HOW_WORDS = frozenset({"how", "where", "should", "improve", "fix", "better"})
OPPONENT_WORDS = frozenset({"opponent", "opponents", "him", "her", "his", "hers", "he", "she", "them", "guy", "girl"})
TRAILING = frozenset({"then", "too", "now", "instead", "though", "also"})
JOINERS = frozenset({"a", "an", "any", "the", "my", "or", "and", "nor", "some"})
LEVEL_LEAD = frozenset({"for", "as", "am", "a", "an"})
STEM_ENDINGS = ("", "s", "e", "es", "ed", "ing", "y", "ies", "ied", "ion", "ions", "ment", "ness", "ful")
# One inflection dropped from both an exercise name and the message, so "wall sitting", "single-legged calf
# raises" and "single leg deadlifts" meet their names.
INFLECTIONS = ("ings", "ing", "ers", "er", "es", "ed", "s", "e")
JOINED_MIN = 6     # a joined window ("wallsits", "wal sit") shorter than this is never typo-matched
JOINED_SPAN = 4    # tokens a joined window may cover
VOWELS = frozenset("aeiou")  # a four-letter token with none of these is read as a dropped-letter typo ("splt")
POSSESSIVES = frozenset({"my", "your", "his", "her", "their", "our"})
# A bare term that is also a strong cue for one of these ("reps?", "slow motion please") asks for that, not a definition.
TERM_RIVALS = ("exercise_detail", "show_moment", "exercise_adjust")
ARITHMETIC = re.compile(r"\d\s*[+÷^]\s*\d")  # "2+2"; not x, * or - which also write doses ("3x10", "10-15")


@dataclass(frozen=True)
class Parse:
    """What one message asks. Entity fields hold only what the text itself names."""

    intent: str
    confidence: float
    tokens: tuple[str, ...] = ()
    finding_id: Optional[str] = None
    exercise_id: Optional[str] = None      # a library id
    missing: Optional[str] = None          # an exercise_aliases["missing"] id
    player: Optional[Role] = None          # "me" and "my" resolve to the context player
    level: Optional[int] = None
    equipment: Optional[tuple[str, tuple[str, ...]]] = None   # (none | all | only | remove, items)
    term: Optional[str] = None
    limit: Optional[str] = None
    aspect: Optional[str] = None
    adjust: Optional[str] = None
    bare: Optional[str] = None
    best: bool = False
    judge: bool = False                    # asks whether something is good, e.g. "is that good?"
    followup: Optional[str] = None
    topic_word: Optional[str] = None
    scores: tuple[tuple[str, float], ...] = ()


class Phrases:
    """Token-sequence lookup: find every phrase occurrence, or the longest non-overlapping ones."""

    def __init__(self, pairs: Iterable[tuple[Sequence[str], Any]]):
        self.index: dict[str, list[tuple[tuple[str, ...], Any]]] = {}
        self.words: set[str] = set()
        seen = set()
        for tokens, value in pairs:
            key = (tuple(tokens), value)
            if not tokens or key in seen:
                continue
            seen.add(key)
            self.index.setdefault(tokens[0], []).append(key)
            self.words.update(tokens)
        for entries in self.index.values():
            entries.sort(key=lambda entry: -len(entry[0]))

    def find(self, tokens: Sequence[str]) -> list[tuple[int, int, Any]]:
        found = []
        for i, token in enumerate(tokens):
            for phrase, value in self.index.get(token, ()):
                if tuple(tokens[i:i + len(phrase)]) == phrase:
                    found.append((i, i + len(phrase), value))
        return found

    def longest(self, tokens: Sequence[str]) -> list[tuple[int, int, Any]]:
        return _longest(self.find(tokens))


def _longest(found: Iterable[tuple[int, int, Any]]) -> list[tuple[int, int, Any]]:
    """The longest non-overlapping matches, earliest first among equals, in text order."""
    taken: set[int] = set()
    kept = []
    for start, end, value in sorted(found, key=lambda m: (m[0] - m[1], m[0])):
        if taken.isdisjoint(range(start, end)):
            kept.append((start, end, value))
            taken.update(range(start, end))
    return sorted(kept, key=lambda m: m[0])


def _stem(token: str) -> str:
    for ending in INFLECTIONS:
        if token.endswith(ending) and len(token) - len(ending) >= 3:
            token = token[:-len(ending)]
            break
    if len(token) > 3 and token[-1] == token[-2] and token[-1] not in VOWELS:
        token = token[:-1]  # "sitting" -> "sit", "legged" -> "leg"
    return token


@dataclass(frozen=True)
class _Cue:
    intent: str
    parts: tuple[str, ...]
    start: bool
    end: bool
    weight: float
    needs: frozenset[str]  # literal words the message must contain for the cue to match


def _cue(intent: str, pattern: str, weight: float) -> _Cue:
    parts = tuple(pattern.lstrip("^").rstrip("$").split())
    return _Cue(intent, parts, pattern.startswith("^"), pattern.endswith("$"), weight,
                frozenset(p for p in parts if p != "*" and not p.endswith("~")))


def _close(a: str, b: str) -> bool:
    """True when a and b differ by at most one insert, delete, substitution or adjacent swap."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1
                                  and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    i = 0
    while i < len(short) and short[i] == long_[i]:
        i += 1
    return short[i:] == long_[i + 1:]


def _deletes(word: str) -> set[str]:
    return {word[:i] + word[i + 1:] for i in range(len(word))}


def _same_start(a: str, b: str) -> bool:
    return a[0] == b[0] or (len(a) > 1 and len(b) > 1 and a[0] == b[1] and a[1] == b[0])


def _match(parts: Sequence[str], tokens: Sequence[str], i: int, need_end: bool) -> bool:
    if not parts:
        return not need_end or i == len(tokens)
    head = parts[0]
    if head == "*":
        return any(_match(parts[1:], tokens, i + gap, need_end) for gap in range(GAP + 1) if i + gap <= len(tokens))
    if i >= len(tokens):
        return False
    token = tokens[i]
    ok = token.startswith(head[:-1]) if head.endswith("~") and len(head) > 1 else token == head
    return ok and _match(parts[1:], tokens, i + 1, need_end)


def _gap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(b[0] - (a[1] - 1), a[0] - (b[1] - 1), 0)


class Classifier:
    def __init__(self, kb: KnowledgeBase, library: Mapping[str, Exercise], report: Report):
        lex = kb.lexicon
        self.kb = kb
        self.library = library
        self.players = report.players
        self.contractions = [(re.compile(pattern), repl) for pattern, repl in lex["contractions"]]
        self.token_map = {key: value.split() for key, value in lex["token_map"].items()}
        self.fillers = frozenset(lex["fillers"])
        self.severity = {(f.id, f.player): SEVERITY_RANK[f.severity] for f in report.findings}
        self.tags = {(f.id, f.player): tuple(f.fix_tags) for f in report.findings}

        phrase = self._words
        aliases = kb.exercise_aliases
        lib_pairs = []
        for ex_id, names in aliases["library"].items():
            if ex_id in library:
                for name in (*names, library[ex_id].name, ex_id.replace("_", " ")):
                    lib_pairs.append((phrase(name), ("lib", ex_id)))
        for ex_id, row in aliases["missing"].items():
            for name in (*row["aliases"], row["name"]):
                lib_pairs.append((phrase(name), ("missing", ex_id)))
        # Several-word names match on stems. A one-word name ("plank", "shadow") matches only itself or its plural,
        # since its stem alone is too loose ("running" would name "run").
        self.exercises = Phrases((tuple(_stem(t) for t in tokens), value) for tokens, value in lib_pairs if len(tokens) > 1)
        self.exercise_single: dict[str, tuple[str, str]] = {}
        for tokens, value in lib_pairs:
            if len(tokens) == 1:
                word = tokens[0]
                for form in (word, word + "s", word + "es"):
                    self.exercise_single.setdefault(form, value)
        self.exercise_words = frozenset(t for tokens, _ in lib_pairs for t in tokens)
        self.joined: dict[str, tuple[str, str]] = {}
        self.joined_index: dict[str, list[str]] = defaultdict(list)
        for tokens, value in lib_pairs:
            key = "".join(tokens)
            if len(key) >= JOINED_MIN + 1 and key not in self.joined:
                self.joined[key] = value
                for part in {key} | _deletes(key):
                    self.joined_index[part].append(key)
        self.weak = {word: [i for i in ids if i in library] for word, ids in aliases["weak"].items()}
        self.weak = {word: ids for word, ids in self.weak.items() if ids}
        self.weak_before = frozenset(aliases["weak_before"])
        self.weak_block = frozenset(aliases["weak_block"])

        self.topics = Phrases((phrase(p), (fid, float(w))) for fid, guide in kb.fix_guides.items()
                              for p, w in guide["topics"].items())
        self.terms = Phrases((phrase(p), tid) for tid, row in kb.glossary.items() for p in (*row["aliases"], row["name"]))
        limit_pairs = [(phrase(p), key) for key, row in kb.limits["cannot"].items() for p in row["keys"]]
        limit_pairs += [(phrase(p), "preview") for p in kb.limits["preview"]["keys"]]
        limit_pairs += [(phrase(p), "general") for p in kb.limits["general"]["keys"]]
        self.limits = Phrases(limit_pairs)
        self.asks = [_cue(key, p, 0.0) for key, row in kb.limits["cannot"].items() for p in row.get("asks", ())]
        self.player_tables = {kind: Phrases((phrase(p), kind) for p in lex["players"][kind]) for kind in ("near", "far", "other")}
        self.self_words = frozenset(lex["players"]["self"])
        self.me_exempt = frozenset(lex["players"]["me_exempt_before"])
        self.followups = {" ".join(phrase(p)): kind for kind, ps in lex["followups"].items() for p in ps}
        self.levels = Phrases((phrase(p), int(level)) for level, ps in lex["levels"].items() for p in ps)
        eq_pairs = [(phrase(p), (item,)) for item, ps in lex["equipment"].items() for p in ps]
        eq_pairs += [(phrase(p), tuple(items)) for p, items in lex["equipment_groups"].items()]
        self.equipment = Phrases(eq_pairs)
        self.eq_ops = {op: Phrases((phrase(p), op) for p in ps) for op, ps in lex["equipment_ops"].items()}
        self.aspects = Phrases((phrase(p), kind) for kind, ps in lex["aspects"].items() for p in ps)
        self.adjust = Phrases((phrase(p), kind) for kind, ps in lex["adjust"].items() for p in ps)
        self.quantity = Phrases((phrase(p), True) for p in lex["quantity"])
        self.bare = {" ".join(phrase(p)): kind for kind, ps in lex["bare"].items() for p in ps}
        self.best = Phrases((phrase(p), True) for p in lex["best"])

        safety = kb.safety
        self.safe_phrases = sorted({" ".join(phrase(p)) for p in safety["safe_phrases"]}, key=len, reverse=True)
        always = [phrase(p) for p in safety["always"]]
        self.s_always = frozenset(t[0] for t in always if len(t) == 1)
        self.s_stems = tuple(safety["stems"])
        self.s_phrases = Phrases([(phrase(p), True) for p in safety["phrases"]] + [(t, True) for t in always if len(t) > 1])
        self.s_concern = Phrases((phrase(p), True) for p in safety["joint_concern"])
        self.s_any = Phrases((phrase(p), True) for p in safety["conditional_any"])
        self.s_joint_words = Phrases((phrase(p), True) for p in safety["conditional_joint"])
        self.s_body = Phrases((phrase(p), True) for p in safety["body_parts"])
        self.s_joints = Phrases((phrase(p), True) for p in safety["joint_parts"])
        self.s_foreign = tuple(s.casefold() for s in safety["non_english"])
        self.s_foreign_tokens = frozenset(safety["non_english_tokens"])
        self.s_typo = sorted({w for w in self.s_always if len(w) >= 5 and w.isalpha()}
                             | {stem + end for stem in self.s_stems for end in STEM_ENDINGS if len(stem + end) >= 5})

        self.cues: dict[str, list[_Cue]] = defaultdict(list)
        self.prefix_cues: list[_Cue] = []
        cue_words: set[str] = set()
        for intent, table in lex["intents"].items():
            for pattern, weight in table.items():
                cue = _cue(intent, pattern, float(weight))
                (self.prefix_cues if cue.parts[0].endswith("~") else self.cues[cue.parts[0]]).append(cue)
                cue_words.update(cue.needs)
        self.judge = [_cue("judge", p, 0.0) for p in lex["judge"]]
        cue_words.update(p for cue in self.judge for p in cue.parts if p != "*")
        cue_words.update(p.rstrip("~") for cue in self.asks for p in cue.parts if p != "*")

        vocab = set(cue_words) | set(lex["known_words"]) | set(lex["stop_words"]) | set(self.fillers)
        vocab |= {t for words in self.token_map.values() for t in words}
        for table in (self.topics, self.terms, self.limits, *self.player_tables.values(), self.levels,
                      self.equipment, *self.eq_ops.values(), self.aspects, self.adjust, self.quantity, self.best,
                      self.s_phrases, self.s_concern, self.s_any, self.s_joint_words, self.s_body, self.s_joints):
            vocab |= table.words
        vocab |= self.s_always | self.self_words | self.me_exempt | self.exercise_words | RULE_WORDS
        vocab |= {t for key in (*self.followups, *self.bare) for t in key.split()}
        self.vocab = frozenset(vocab)
        # Coaching words a four-letter typo may stand for when one letter was dropped, e.g. "splt" -> "split".
        self.domain = frozenset(w for words in (self.exercise_words, self.topics.words, self.terms.words) for w in words
                                if len(w) == 5 and w.isalpha()) - frozenset(lex["stop_words"])
        self.typo_index: dict[str, list[str]] = defaultdict(list)
        for word in sorted(self.vocab):
            if len(word) >= 4 and word.isalpha():
                for key in {word} | _deletes(word):
                    self.typo_index[key].append(word)

    # ---- text -------------------------------------------------------------------------------------

    def _basic(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text or "")[:MAX_CHARS].casefold()
        text = APOSTROPHES.sub("'", text)
        if "'" in text:
            for pattern, repl in self.contractions:
                text = pattern.sub(repl, text)
            text = text.replace("'", "")
        return PUNCTUATION.sub(" ", SEPARATORS.sub(" ", text))

    def _words(self, text: str) -> list[str]:
        out: list[str] = []
        for token in self._basic(text).split()[:MAX_TOKENS]:
            out.extend(self.token_map.get(token, (token,)))
        return out[:MAX_TOKENS]

    def _fix(self, token: str) -> str:
        if token in self.vocab or len(token) < 4 or not token.isalpha():
            return token
        if len(token) == 4:
            if VOWELS.intersection(token):
                return token
            dropped = sorted(w for w in self.typo_index.get(token, ()) if w in self.domain and _same_start(token, w))
            return dropped[0] if dropped else token
        for squeezed in (LONG_REPEATS.sub(r"\1\1", token), REPEATS.sub(r"\1", token)):
            if squeezed in self.vocab:
                return squeezed
        found = {w for key in {token} | _deletes(token) for w in self.typo_index.get(key, ())}
        good = sorted((w for w in found if _same_start(token, w) and _close(token, w)), key=lambda w: (len(w) != len(token), w))
        return good[0] if good else token

    def tokens(self, text: str) -> tuple[list[str], list[str]]:
        """(words after the token map, the same words after typo fixes)."""
        words = self._words(text)
        return words, [self._fix(t) for t in words]

    # ---- safety -------------------------------------------------------------------------------------

    def unsafe(self, text: str) -> bool:
        words, fixed = self.tokens(text)
        return self._unsafe(text, words, fixed)

    def _unsafe(self, text: str, words: list[str], fixed: list[str]) -> bool:
        basic = " " + " ".join(self._basic(text).split()) + " "
        if any(s in basic for s in self.s_foreign) or any(t in self.s_foreign_tokens for t in words):
            return True
        for raw in (words, fixed):
            joined = " " + " ".join(raw) + " "
            for safe in self.safe_phrases:
                joined = joined.replace(" " + safe + " ", " _ ")
            tokens = joined.split()
            if any(t in self.s_always or t.startswith(self.s_stems) for t in tokens):
                return True
            if self.s_phrases.find(tokens):
                return True
            body = [(s, e) for s, e, _ in self.s_body.find(tokens)]
            if body and any(_gap((s, e), b) <= 4 for s, e, _ in self.s_any.find(tokens) for b in body):
                return True
            joints = [(s, e) for s, e, _ in self.s_joints.find(tokens)]
            if joints and any(_gap((s, e), j) <= 3 for s, e, _ in self.s_joint_words.find(tokens) for j in joints):
                return True
            # "is it hard on my knees": a worry about a joint, unless it goes on "... my hips to stay low" (technique).
            free = {s for s, e in joints if e == len(tokens) or tokens[e] != "to"}
            for _, end, _ in self.s_concern.find(tokens):
                if end in free or (end + 1 in free and tokens[end] in POSSESSIVES | ARTICLES):
                    return True
        for token, fixed_token in zip(words, fixed):
            if fixed_token == token and len(token) >= 5 and token.isalpha() and token not in self.vocab:
                if any(_same_start(token, w) and _close(token, w) for w in self.s_typo):
                    return True
        return False

    # ---- cues ---------------------------------------------------------------------------------------

    def _scores(self, tokens: Sequence[str]) -> dict[str, float]:
        hits: dict[str, list[float]] = defaultdict(list)
        done: set[int] = set()
        present = frozenset(tokens)
        for i, token in enumerate(tokens):
            for cue in (*self.cues.get(token, ()), *self.prefix_cues):
                if id(cue) in done or (cue.start and i) or not cue.needs <= present:
                    continue
                if _match(cue.parts, tokens, i, cue.end):
                    done.add(id(cue))
                    hits[cue.intent].append(cue.weight)
        return {intent: max(ws) + 0.3 * (sum(ws) - max(ws)) for intent, ws in hits.items()}

    # ---- entities -----------------------------------------------------------------------------------

    def me(self, context: Any) -> Optional[Role]:
        return getattr(context, "player", None) or (self.players[0] if self.players else None)

    def other(self, me: Optional[Role]) -> Optional[Role]:
        if me is None:
            return None
        return next((p for p in self.players if p != me), Role.FAR if me == Role.NEAR else Role.NEAR)

    def finding_tags(self, finding_id: Optional[str], player: Optional[Role]) -> tuple[str, ...]:
        if not finding_id:
            return ()
        guide = self.kb.fix_guides.get(finding_id, {})
        return self.tags.get((finding_id, player)) or tuple(guide.get("fix_tags", ()))

    def _named(self, tokens: Sequence[str]) -> list[tuple[int, int, tuple[str, str]]]:
        """Exercise names in the text: several-word names on stems, one-word names exactly, then a window holding an
        unknown word whose letters, run together, are one typo from a long name ("wallsits", "side plnak")."""
        found = self.exercises.find([_stem(t) for t in tokens])
        found += [(i, i + 1, self.exercise_single[t]) for i, t in enumerate(tokens) if t in self.exercise_single]
        kept = _longest(found)
        taken = {i for s, e, _ in kept for i in range(s, e)}
        for start in range(len(tokens)):
            for end in range(min(len(tokens), start + JOINED_SPAN), start, -1):
                window = tokens[start:end]
                if taken.intersection(range(start, end)) or all(t in self.vocab for t in window):
                    continue
                key = "".join(window)
                if len(key) < JOINED_MIN or not key.isalpha():
                    continue
                near = {k for part in {key} | _deletes(key) for k in self.joined_index.get(part, ())}
                hits = sorted(k for k in near if k[0] == key[0] and _close(key, k))
                if hits:
                    kept.append((start, end, self.joined[hits[0]]))
                    taken.update(range(start, end))
                    break
        return sorted(kept)

    def _exercise(self, tokens: list[str], ctx: Any, tags: Sequence[str]):
        found: list[tuple[int, str, str]] = []
        spans: list[tuple[int, int]] = []
        for start, end, (kind, ex_id) in self._named(tokens):
            found.append((start, kind, ex_id))
            spans.append((start, end))
        covered = {i for s, e in spans for i in range(s, e)}
        exercise_talk = bool(EXERCISE_WORDS.intersection(tokens))
        for i, token in enumerate(tokens):
            ids = self.weak.get(token)
            if not ids or i == 0 or i in covered or tokens[i - 1] not in self.weak_before:
                continue
            if self.weak_block.intersection(tokens[max(0, i - 2):i]):
                continue
            known = [x for x in ids if x == ctx.exercise_id] or [x for x in ids if x in ctx.shown_exercises]
            if not (known or exercise_talk or tokens[i - 1] in DO_WORDS or token not in self.topics.words):
                continue
            pick = (known or [x for x in ids if set(self.library[x].fix_tags) & set(tags)] or ids)[0]
            found.append((i, "lib", pick))
            spans.append((i, i + 1))
        found.sort()
        lib = [x for _, kind, x in found if kind == "lib"]
        missing = [x for _, kind, x in found if kind == "missing"]
        fresh = [x for x in lib if x != ctx.exercise_id]
        if fresh:
            return fresh[0], None, spans
        if missing:
            return None, missing[0], spans
        return (lib[0] if lib else None), None, spans

    def _finding(self, tokens: list[str], spans, ctx: Any, player: Optional[Role]):
        masked = {i for s, e in spans for i in range(s, e)}
        total: dict[str, float] = defaultdict(float)
        best: dict[str, tuple[float, int, str]] = {}
        seen = set()
        for start, end, (fid, weight) in self.topics.find(tokens):
            phrase = " ".join(tokens[start:end])
            if (fid, phrase) in seen or masked.intersection(range(start, end)):
                continue
            seen.add((fid, phrase))
            total[fid] += weight
            best[fid] = max(best.get(fid, (0.0, 0, "")), (weight, end - start, phrase))
        ok = [fid for fid, value in total.items() if value >= FINDING_MIN]
        if not ok:
            return None, None
        top = max(total[fid] for fid in ok)
        fid = min((f for f in ok if total[f] >= top - 1e-9),
                  key=lambda f: (f != ctx.finding_id, -self.severity.get((f, player), -1), CATALOG_ORDER.get(f, 99)))
        return fid, best[fid][2]

    def _player(self, tokens: list[str], me: Optional[Role]) -> Optional[Role]:
        named = sorted((s, kind) for kind in ("near", "far") for s, _, _ in self.player_tables[kind].longest(tokens))
        if named:
            return Role.NEAR if named[0][1] == "near" else Role.FAR
        spans = self.player_tables["other"].longest(tokens)
        inside = {i for s, e, _ in spans for i in range(s, e)}
        mine = any(t in self.self_words and i not in inside and not (t == "me" and i and tokens[i - 1] in self.me_exempt)
                   for i, t in enumerate(tokens))
        if spans and not mine:
            return self.other(me)
        return me if (mine or spans) else None

    def _equipment(self, tokens: list[str], ex_spans) -> Optional[tuple[str, tuple[str, ...]]]:
        if self.eq_ops["none"].find(tokens):
            return ("none", ())
        if self.eq_ops["all"].find(tokens):
            return ("all", ())
        in_exercise = {i for s, e in ex_spans for i in range(s, e)}
        # "with" inside "rotation with a band" is part of an exercise name, not a statement about equipment.
        only = [m for m in self.eq_ops["only"].find(tokens) if m[0] not in in_exercise]
        removes = {end for start, end, _ in self.eq_ops["remove"].find(tokens) if start not in in_exercise}
        op_ends = removes | {end for _, end, _ in only}
        kept: list[str] = []
        removed: list[str] = []
        prev_end, prev_removed = None, False
        for start, end, items in self.equipment.longest(tokens):
            lead = start - 1 if start and tokens[start - 1] in ARTICLES else start
            if start in in_exercise and start not in op_ends and lead not in op_ends:
                continue
            between = set(tokens[prev_end:start]) if prev_end is not None else None
            chained = prev_removed and between is not None and between <= JOINERS and bool(between & {"or", "and", "nor"})
            prev_removed = start in removes or lead in removes or chained
            (removed if prev_removed else kept).extend(items)
            prev_end = end
        kept = list(dict.fromkeys(kept))
        removed = list(dict.fromkeys(removed))
        if only and kept:
            return ("only", tuple(kept))
        if removed:
            return ("remove", tuple(removed))
        if kept and HAVE_WORDS.intersection(tokens):
            return ("have", tuple(kept))
        return None

    @staticmethod
    def _first(hits, order: Optional[Mapping[str, int]] = None):
        if not hits:
            return None, None
        start, end, value = min(hits, key=lambda h: (h[0] - h[1], (order or {}).get(h[2], 0), h[0]))
        return value, (start, end)

    # ---- decision -----------------------------------------------------------------------------------

    def parse(self, text: str, context: Any = None, history: Sequence[Turn] = (), depth: int = 0) -> Parse:
        ctx = context if context is not None else _EMPTY
        words, fixed = self.tokens(text)
        me = self.me(ctx)
        if self._unsafe(text, words, fixed):
            # The reply is always the fixed safety text; the entities only record what the message named.
            ex_id, missing, ex_spans = self._exercise(fixed, ctx, self.finding_tags(ctx.finding_id, me))
            player = self._player(fixed, me)
            fid, _ = self._finding(fixed, ex_spans, ctx, player or me)
            return Parse("safety", 1.0, tuple(fixed), finding_id=fid, exercise_id=ex_id, missing=missing, player=player)
        content = [t for t in fixed if t not in self.fillers] or fixed
        if not content:
            return Parse("unknown", 0.9, ())
        follow = self._followup(content)
        if follow and depth < 3:
            return self._follow(follow, content, ctx, history, depth, me)

        ex_id, missing, ex_spans = self._exercise(fixed, ctx, self.finding_tags(ctx.finding_id, me))
        player = self._player(fixed, me)
        fid, finding_phrase = self._finding(fixed, ex_spans, ctx, player or me)
        equipment = self._equipment(fixed, ex_spans)
        term, term_span = self._first(self.terms.longest(fixed))
        limit, _ = self._first(self.limits.longest(fixed), {"accuracy": 1})
        aspect, _ = self._first(self.aspects.longest(fixed), ASPECT_ORDER)
        adjust, _ = self._first(self.adjust.longest(fixed))
        level, _ = self._first(self.levels.longest(fixed))
        bare = self.bare.get(" ".join(content))
        present = frozenset(content)
        judge = any(_match(c.parts, content, i, c.end) for c in self.judge if c.needs <= present
                    for i in range(len(content)) if not (c.start and i))
        # "how fast was my smash" asks directly for something the video cannot give.
        asked = next((c.intent for c in self.asks if c.needs <= present
                      and any(_match(c.parts, content, i, c.end) for i in range(len(content)))), None)
        limit = asked or limit
        topic = " ".join(fixed[term_span[0]:term_span[1]]) if term_span else finding_phrase or limit
        found = dict(tokens=tuple(fixed), finding_id=fid, exercise_id=ex_id, missing=missing, player=player,
                     equipment=equipment, term=term, limit=limit, aspect=aspect, adjust=adjust, bare=bare,
                     best=bool(self.best.find(fixed)), judge=judge, topic_word=topic)
        joined = " " + " ".join(fixed) + " "
        level_ok = any(lead in joined for lead in LEVEL_LEADS)

        exercise_focus = (ctx.last_offer or "").startswith("exercise_detail") or (ctx.topic or "").startswith("exercise:")
        if bare in ("how", "why") and not (exercise_focus or ctx.topic or ctx.last_offer):
            # A context with no focus at all: the coach's last turn is the only evidence, as in Coach._base.
            exercise_focus = self._coach_named_exercise(history)
        if bare == "how":
            return Parse("exercise_detail" if exercise_focus else "how_to_fix", 0.8, **found)
        if bare == "why":
            return Parse("exercise_why" if exercise_focus else "explain", 0.8, **found)
        if bare == "huh":
            heard = any(turn.speaker == "coach" for turn in history)
            return Parse("simplify" if heard else "unknown", 0.7, **found)
        if [t for t in words] == ["y"] and ctx.last_offer:
            return Parse("affirm", 0.8, **found)

        s = self._scores(content)

        def bump(intent: str, amount: float, only_if_cued: bool = False) -> None:
            if not only_if_cued or intent in s:
                s[intent] = s.get(intent, 0.0) + amount

        if "affirm" in s and not ctx.last_offer:
            s["affirm"] *= 0.6
        if ex_id or missing:
            if PURPOSE_WORDS.intersection(content):
                bump("exercise_why", 3.0)  # "is the wall sit worth it" asks what the named exercise is for
            bump("exercise_detail", 3.0)
            bump("exercise_why", 2.0, True)
            bump("exercise_adjust", 2.0, True)
            bump("show_moment", -2.0, True)
            bump("strengths", -3.0, True)
            if not fid:
                bump("explain", -3.0, True)  # "tell me about skater hops" asks about the exercise
        if equipment:
            in_focus = bool(ctx.exercise_id or ctx.shown_exercises or ex_id)
            if equipment[0] != "have" or in_focus:
                bump("exercise_adjust", 3.0)
            else:
                bump("exercise_request", 2.0)
        if fid and self.quantity.find(content):
            bump("metric_question", 2.5)
        if (exercise_focus and not fid and "metric_question" in s
                and (PRESCRIBE_WORDS & present or not CLIP_WORDS & present)):
            # During an exercise, "how deep should I go" asks how to do it, not for a reading from the clip.
            s["exercise_detail"] = max(s.get("exercise_detail", 0.0), s.pop("metric_question"))
        defined = False  # the message names a glossary word to define
        if term:
            # "my lunges" and "the far player" name the player's own movement or a player, not a word to define.
            owned = term_span[0] > 0 and fixed[term_span[0] - 1] in POSSESSIVES
            # "the coverage finding" names a reading, not a word to define.
            labels = term_span[1] < len(fixed) and fixed[term_span[1]] in FINDING_WORDS
            players = [(s, e) for table in self.player_tables.values() for s, e, _ in table.longest(fixed)]
            defined = not owned and not labels and not any(s < term_span[1] and term_span[0] < e for s, e in players)
            if defined:
                bump("terms", 3.0, True)
                whole = term_span[1] - term_span[0] == len([t for t in content if t not in ARTICLES])
                rivals = self._scores(fixed[term_span[0]:term_span[1]])
                if whole and not any(rivals.get(i, 0.0) >= 4.5 for i in TERM_RIVALS):
                    bump("terms", FULL)
        if fid and not (ex_id or missing) and {"why", "wrong"} & set(content):
            bump("explain", 3.0)
        if fid and fid.startswith("tactics.") and ({"how", "where"} & set(content) or "what should" in joined):
            bump("tactics_tip", 2.0)
        if (ctx.finding_id and not (fid or ex_id or missing or exercise_focus)
                and EXPLAIN_FRAME & present and not ADVICE_FRAME & present):
            # With a reading in focus, "what's wrong with my position" asks about that reading, not for a tip.
            bump("explain", 2.0, True)
            bump("tactics_tip", -2.0, True)
        if (PAIR_WORDS & present or (self.self_words & present and OPPONENT_WORDS & present)) and CONTRAST_WORDS & present:
            bump("compare_players", 3.0)  # "how are we different", "is my opponent faster"
        if judge and not (ex_id or missing):
            # "is that good?" judges the reading in focus, or the exercise in focus; with neither it stays unknown
            # and the reply asks which reading is meant.
            focus = (ctx.topic or "").split("@")[0].split(":")[0]
            if fid or (ctx.finding_id and focus not in ("exercise", "plan", "term")):
                bump("metric_question", JUDGE)
            elif focus == "exercise":
                bump("exercise_why", JUDGE)
        if ARITHMETIC.search(text or ""):
            bump("out_of_scope", FULL)
        if exercise_focus and not (fid or ex_id or missing) and PURPOSE_WORDS & present:
            # During an exercise, "what's the reason for this one" asks what the exercise is for.
            s["exercise_why"] = max(s.get("exercise_why", 0.0), RESCUE)
        if player and not (judge or ex_id or missing) and max(s.values(), default=0.0) < RESCUE and _asks_praise(content):
            s["strengths"] = RESCUE  # "what did I nail", "where do I shine"

        ranked = sorted(((i, v) for i, v in s.items() if v > 0), key=lambda kv: (-kv[1], ORDER[kv[0]]))
        if asked and not (ex_id or missing):
            # It gets the limits reply even when a word in it ("smash") also names a reading.
            return Parse("limits", 0.9, scores=tuple((i, round(v, 2)) for i, v in ranked[:5]), **found)
        if not ranked or ranked[0][1] < FLOOR:
            intent = ("explain" if fid else "terms" if defined else "exercise_detail" if (ex_id or missing)
                      else "limits" if limit else "unknown")
            return Parse(intent, 0.4 if intent != "unknown" else 0.5, scores=tuple(ranked[:5]),
                         level=level if level_ok else None, **found)
        top, best_score = ranked[0]
        confidence = min(1.0, best_score / FULL)
        close = [i for i, v in ranked if best_score - v < MARGIN]
        if len(close) > 1:
            offered = OFFER_INTENT.get((ctx.last_offer or "").split(":")[0])

            def prefer(intent: str) -> float:
                bonus = 1.0 if intent == offered else 0.0
                if (ctx.topic or "").startswith("exercise:") and intent.startswith("exercise_"):
                    bonus += 0.5
                if ctx.finding_id and intent in ("explain", "how_to_fix", "show_moment", "metric_question"):
                    bonus += 0.5
                return bonus

            top = min(close, key=lambda i: (-(s[i] + prefer(i)), ORDER[i]))
            confidence *= 0.8
        if adjust == "another" and top in ("exercise_adjust", "next", "exercise_request"):
            if "next" in content or FINDING_WORDS.intersection(content):
                top = "next"
            elif EXERCISE_WORDS.intersection(content):
                top = "exercise_adjust" if ctx.shown_exercises else "exercise_request"
            elif ex_id or missing or equipment:
                top = "exercise_adjust"  # "something instead of jump rope, I don't have a rope"
            else:
                on_exercise = (ctx.topic or "").startswith("exercise:")
                exercise_offer = (ctx.last_offer or "").startswith(("exercise_detail", "another"))
                top = "exercise_adjust" if on_exercise or (exercise_offer and "else" not in content) else "next"
        if top == "set_level":
            rivals = [i for i, v in ranked if i != "set_level" and v >= 5.5]
            top = rivals[0] if rivals else top
        if top == "simplify" and not any(turn.speaker == "coach" for turn in history):
            top = "explain" if (fid or ctx.finding_id) else "help"
        if top == "how_to_fix" and fid and fid.startswith("tactics."):
            top = "tactics_tip"  # "how do I stop being on the back foot" names a tactics reading
        return Parse(top, round(confidence, 3), scores=tuple((i, round(v, 2)) for i, v in ranked[:5]),
                     level=level if (top == "set_level" or level_ok) else None, **found)

    def _coach_named_exercise(self, history: Sequence[Turn]) -> bool:
        last = next((turn.text for turn in reversed(history) if turn.speaker == "coach"), "")
        return bool(last) and any(kind == "lib" for _, _, (kind, _) in self._named(self.tokens(last)[1]))

    def _followup(self, content: list[str]) -> Optional[str]:
        words = list(content)
        while len(words) > 1 and words[-1] in TRAILING:
            words.pop()
        return self.followups.get(" ".join(words))

    def _follow(self, kind: str, content: list[str], ctx: Any, history: Sequence[Turn], depth: int, me) -> Parse:
        users = [i for i, turn in enumerate(history) if turn.speaker == "user"]
        prev = self.parse(history[users[-1]].text, ctx, history[:users[-1]], depth + 1) if users else None
        if kind == "near":
            player = Role.NEAR
        elif kind == "far":
            player = Role.FAR
        elif kind == "self":
            player = me
        elif OPPONENT_WORDS.intersection(content):
            player = self.other(me)
        else:
            player = self.other((prev.player if prev else None) or me)
        if prev is None or prev.intent not in REUSABLE:
            return Parse("summary", 0.75, tuple(content), player=player, followup=kind)
        return replace(prev, confidence=0.75, tokens=tuple(content), player=player, followup=kind, scores=())


ARTICLES = frozenset({"a", "an", "any", "the", "my", "some"})
ASPECT_ORDER = {"safety": 0, "mistakes": 1, "equipment": 2, "dose": 3, "cues": 4}
# Possessives and past tense point a measurement question at the clip rather than at the exercise.
CLIP_WORDS = POSSESSIVES | frozenset({"was", "were", "did", "had", "am"})
DO_WORDS = frozenset({"do", "doing", "did", "try", "trying"})
FINDING_WORDS = frozenset({"finding", "findings", "problem", "problems", "issue", "issues", "thing", "area", "point"})
HAVE_WORDS = frozenset({"have", "got", "own", "use", "using", "with"})
LEVEL_LEADS = (" i am ", " as a ", " for a ", " for an ", " im ")
PURPOSE_WORDS = frozenset({"why", "purpose", "benefit", "benefits", "useful", "worth", "reason"})
# "how deep am I meant to go" asks what to do, even with a word that otherwise points at the clip.
PRESCRIBE_WORDS = frozenset({"should", "meant", "supposed", "need", "must"})
# Word classes for questions that no listed phrase covers.
PRAISE_WORDS = frozenset({"good", "well", "great", "best", "strong", "strongest", "strength", "strengths", "solid",
                          "nail", "nailed", "shine", "shined", "shone", "excel", "excelled", "positive", "positives",
                          "proud", "decent"})
NOT_PRAISE = EXERCISE_WORDS | frozenset({
    "not", "no", "never", "wrong", "bad", "worse", "worst", "weak", "weaker", "weakness", "weaknesses", "problem",
    "problems", "issue", "issues", "fix", "improve", "better", "enough", "way", "ways", "how", "who", "plan", "plans",
    "routine", "program", "programme", "schedule", "morning", "afternoon", "evening", "night"})
MODALS = frozenset({"can", "could", "should", "would", "will", "need", "must"})
COPULAS = frozenset({"is", "are", "was", "were"})
POINTERS = frozenset({"the", "that", "this", "it", "those", "these"})
WH_ASKS = frozenset({"what", "which", "where", "anything"})
ASK_LEADS = frozenset({"did", "do", "does", "is", "are", "was", "were", "have", "has", "any", "tell", "list", "name",
                       "give"})
PAIR_WORDS = frozenset({"we", "us", "our", "ourselves"})
CONTRAST_WORDS = frozenset({"differ", "differs", "differed", "different", "difference", "differences", "compare",
                            "compared", "comparing", "comparison", "similar", "same", "alike", "versus", "vs", "better",
                            "worse", "faster", "slower", "quicker", "lower", "higher", "stronger", "weaker"})
EXPLAIN_FRAME = frozenset({"why", "wrong", "problem", "problems", "issue", "issues", "matter", "matters", "worry",
                           "worried", "bad", "mean", "means", "meaning", "concern", "serious"})
ADVICE_FRAME = EXERCISE_WORDS | frozenset({"how", "where", "should", "tip", "tips", "advice", "fix", "improve", "show"})
# Words the rules in this module test for; typo fixing must never turn one into another word ("shine" -> "shins").
RULE_WORDS = frozenset().union(CLIP_WORDS, FINDING_WORDS, PURPOSE_WORDS, PRESCRIBE_WORDS, PRAISE_WORDS, NOT_PRAISE,
                               MODALS, COPULAS, POINTERS, WH_ASKS, ASK_LEADS, PAIR_WORDS, CONTRAST_WORDS, EXPLAIN_FRAME,
                               ADVICE_FRAME)
OFFER_INTENT = {"show_moment": "show_moment", "show_best": "show_moment", "exercise_request": "exercise_request",
                "exercise_detail": "exercise_detail", "practice_plan": "practice_plan", "next": "next",
                "how_to_fix": "how_to_fix", "explain": "explain", "another": "exercise_adjust", "summary": "summary",
                "strengths": "strengths"}


class _Empty:
    finding_id = None
    player = None
    frame = 0
    exercise_id = None
    shown_exercises: tuple = ()
    topic = None
    last_offer = None
    level = 3
    equipment = None


_EMPTY = _Empty()


def _asks_praise(content: Sequence[str]) -> bool:
    """A question with a word of praise and no word of fault, advice, plan or exercise, as in "what did I nail",
    "where do I shine" or "is there anything I do really well"."""
    praise = [i for i, t in enumerate(content) if t in PRAISE_WORDS and (i or t != "well")]  # "well, ..." opens
    if not praise or NOT_PRAISE.intersection(content):
        return False
    after = {content[i + 1] for i in praise if i + 1 < len(content)}
    before = {content[i - 1] for i in praise if i}
    if after & {"for", "to"} or before & {"as", "am", "im"}:
        return False  # "good for my knees", "as well", "I am good"
    if any(t in MODALS and content[i + 1:i + 2] != ["you"] for i, t in enumerate(content)):
        return False  # "what can I do best" asks for advice; "can you tell me where I shine" does not
    if content[0] in COPULAS and len(content) > 1 and content[1] in POSSESSIVES | POINTERS:
        return False  # "is my split step good" judges one reading
    return bool(WH_ASKS.intersection(content)) or content[0] in ASK_LEADS
