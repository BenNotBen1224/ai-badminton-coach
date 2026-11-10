import hashlib
import io
import json
import random
import re
import statistics
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

from coach.chat import eval as chat_eval
from coach.chat import intents
from coach.chat.engine import (CONTACT_NOTE, DESCRIPTIVE_NOTE, DESCRIPTIVE_SHORT, NO_TIMING, OFFERS, PREVIEW_NOTE,
                               SHUTTLE_NOTE, SHUTTLE_SHORT, SOFT_ASK, SWING_NOTE, TIMED_NOTE, WORD_LIMIT, ChatContext, Coach,
                               context_from_dict, context_to_dict, number, span, split_tag)
from coach.chat.kb import INTENTS, check_kb_against_library, load_kb
from coach.chat.plan import EASED, build_plan, easier_dose
from coach.exercises.library import load_library
from coach.schema import Action, ActionKind, Reply, Role, Severity, Turn, report_from_dict
from coach.tests.fixture_report import earlier_report, full_report, preview_report, stub_library

KB = load_kb()
LIBRARY = load_library()
REPORT = full_report()
COACH = Coach(REPORT, LIBRARY, KB)

APPROVED_SAFETY = (
    "I will stop the coaching here. Please stop and rest. If you have pain, an injury or feel unwell, check with a "
    "doctor or physiotherapist before you train again. If it is sudden or severe, such as chest pain, trouble "
    "breathing or fainting, get urgent medical help. I cannot give medical advice, so I will not suggest exercises "
    "for this.")

CHIPS_KEEP = ("How do I do it?", "Give me another exercise", "Make me a practice plan")
CHIPS_SWAP = ("How do I do it?", "Make it easier", "Make it harder", "Give me another exercise")
CHIPS_EASED = ("How do I do it?", "Any equipment is fine", "I am intermediate", "Make me a practice plan")
SIT_LOW_WHY = "It trains the ready stance so your hips stay down while you wait for the next shot."
SAME_GOAL = "It works on the same goal: sit lower with bent knees in the ready position."
NONE_TEXT = re.compile(r"\b(?:None|null|nan|NaN|inf)\b")


def ex(ref):
    return Action(ActionKind.EXERCISE, ref, 0)


def talk(coach, questions, context):
    """Play a conversation the way the app does: JSON context in and out, history growing each turn."""
    history, out = [], []
    for q in questions:
        reply, ctx = coach.reply(history, q, json.loads(json.dumps(context)))
        context = context_to_dict(ctx)
        history += [Turn("user", q), Turn("coach", reply.text)]
        out.append((reply, context))
    return out


def assert_sane(test, coach, reply, limit=WORD_LIMIT):
    test.assertIsInstance(reply, Reply)
    test.assertIn(reply.intent, INTENTS)
    test.assertLessEqual(len(reply.text.split()), limit, reply.text)
    test.assertTrue(reply.text.strip())
    test.assertLessEqual(len(reply.suggestions), 4)
    test.assertTrue(all(0 < len(s) <= 48 for s in reply.suggestions))
    test.assertNotIn("you must", reply.text.lower())
    test.assertNotRegex(reply.text, NONE_TEXT)
    findings = {(f.id, f.player): f for f in coach.report.findings}
    focus = None
    for a in reply.actions:
        if a.kind == ActionKind.EXERCISE:
            test.assertIn(a.ref, coach.library)
        elif a.kind == ActionKind.FINDING:
            focus = findings[(a.ref, coach.report.players[a.arg])]
        else:
            test.assertEqual(a.ref, focus.id, "a seek follows the finding it plays")
            test.assertTrue(0 <= a.arg < len(focus.moments))


class ContextTest(unittest.TestCase):
    def test_round_trip_through_json(self):
        for ctx in (ChatContext(),
                    ChatContext(finding_id="technique.stay_lower", player=Role.FAR, frame=12, exercise_id="wall_sit",
                                shown_exercises=("wall_sit", "sit_low_hold"), topic="exercise:wall_sit",
                                last_offer="exercise_detail:wall_sit", level=2, equipment=("band",),
                                too_hard=("sit_low_hold",)),
                    ChatContext(equipment=()), ChatContext(pain_stop=True)):
            data = json.loads(json.dumps(context_to_dict(ctx), allow_nan=False))
            self.assertEqual(context_from_dict(data), ctx)

    def test_defaults_and_unknown_keys(self):
        self.assertEqual(context_from_dict({}), ChatContext())
        self.assertEqual(context_from_dict({"colour": "red", "level": None, "too_hard": None, "pain_stop": None}),
                         ChatContext())
        self.assertEqual(context_to_dict(ChatContext()), {
            "finding_id": None, "player": None, "frame": 0, "exercise_id": None, "shown_exercises": [], "topic": None,
            "last_offer": None, "level": 3, "equipment": None, "too_hard": [], "pain_stop": False})

    def test_bad_values_raise_value_error(self):
        cases = [
            ([1], "context must be a JSON object"),
            ({"finding_id": "nope"}, "context finding_id 'nope' is not a known finding"),
            ({"player": "middle"}, "context player must be 'near', 'far' or null"),
            ({"frame": -1}, "context frame must be a whole number from 0"),
            ({"frame": True}, "context frame must be a whole number from 0"),
            ({"level": 7}, "context level must be a whole number from 1 to 3"),
            ({"equipment": ["car"]}, "context equipment has unknown items ['car']"),
            ({"equipment": "band"}, "context equipment must be a list of names"),
            ({"topic": 5}, "context topic must be text or null"),
            ({"shown_exercises": [1]}, "context shown_exercises must be a list of names"),
            ({"too_hard": "wall_sit"}, "context too_hard must be a list of names"),
            ({"too_hard": [3]}, "context too_hard must be a list of names"),
            ({"pain_stop": "yes"}, "context pain_stop must be true, false or null"),
            ({"pain_stop": 1}, "context pain_stop must be true, false or null"),
        ]
        for data, message in cases:
            with self.subTest(data=data):
                with self.assertRaises(ValueError) as caught:
                    context_from_dict(data)
                self.assertEqual(str(caught.exception), message)
        for bad in ({"level": 7}, [1, 2], "near"):
            with self.subTest(reply_context=bad):
                with self.assertRaises(ValueError):
                    COACH.reply([], "how do I fix it?", bad)

    def test_stale_context_is_cleaned_not_fatal(self):
        stale = ChatContext(finding_id="nope", player="middle", frame=10 ** 9, exercise_id="nope",
                            shown_exercises=("nope", "wall_sit"), level=9, equipment=("car",),
                            too_hard=("nope", "wall_sit", "wall_sit"))
        reply, ctx = COACH.reply([], "thanks", stale)
        assert_sane(self, COACH, reply)
        self.assertEqual((ctx.finding_id, ctx.player, ctx.frame, ctx.exercise_id, ctx.shown_exercises, ctx.level,
                          ctx.equipment, ctx.too_hard), (None, None, 0, None, ("wall_sit",), 3, (), ("wall_sit",)))


class UserExampleTest(unittest.TestCase):
    """The user's own example: the player asks how, then gets gym exercises, each one with a GIF to show how."""

    def test_how_then_exercises_then_plan(self):
        turns = talk(COACH, ["how?", "show me", "too hard", "I only have a band", "make a plan"],
                     {"finding_id": "technique.stay_lower", "player": "near"})
        (how, c1), (show, c2), (easy, c3), (band, c4), (plan, c5) = turns

        self.assertEqual((how.intent, how.confidence), ("how_to_fix", 0.8))
        self.assertEqual(how.text, (
            "To work on staying low: Bend your knees and sit your hips back a little, as if a chair is behind you. "
            "Keep your weight on the balls of your feet with light heels. Keep your chest up and your racket in front "
            "of you. Gym exercises: the sit-low hold (3 x 30 seconds) and the wall sit (3 x 30 seconds). Want to know "
            "how to do the first one?"))
        self.assertEqual(how.actions, (Action(ActionKind.FINDING, "technique.stay_lower", 0), ex("sit_low_hold"),
                                       ex("wall_sit")))
        self.assertEqual(how.suggestions, ("How do I do it?", "Give me another exercise", "Show me the moment",
                                           "Make me a practice plan"))
        self.assertEqual((c1["exercise_id"], c1["shown_exercises"], c1["topic"], c1["last_offer"], c1["level"]),
                         ("sit_low_hold", ["sit_low_hold", "wall_sit"], "finding:technique.stay_lower",
                          "exercise_detail:sit_low_hold", 3))

        self.assertEqual((show.intent, show.confidence), ("show_moment", 1.0))
        self.assertEqual(show.text, "Playing it now: hips high while waiting. It is one of two moments like this, so "
                                    "ask again to see the next. Want to know how to do the sit-low hold?")
        # SEEK's arg indexes finding.moments; the context keeps the video frame of that moment.
        self.assertEqual(show.actions, (Action(ActionKind.FINDING, "technique.stay_lower", 0),
                                        Action(ActionKind.SEEK, "technique.stay_lower", 0), ex("sit_low_hold")))
        self.assertEqual(show.suggestions, ("How do I do it?", "Show me the moment", "Give me another exercise"))
        self.assertEqual((c2["frame"], c2["last_offer"]), (372, "exercise_detail:sit_low_hold"))

        self.assertEqual((easy.intent, easy.confidence), ("exercise_adjust", 1.0))
        self.assertEqual(easy.text, "Easier option: the wall sit (3 x 30 seconds). " + SAME_GOAL + " It holds your legs "
                                    "in the low ready shape so you can stay down between shots. Want to know how to "
                                    "do it?")
        self.assertEqual((easy.actions, easy.suggestions), ((ex("wall_sit"),), CHIPS_SWAP))
        self.assertEqual((c3["exercise_id"], c3["topic"], c3["last_offer"], c3["level"], c3["too_hard"]),
                         ("wall_sit", "exercise:wall_sit", "exercise_detail:wall_sit", 1, ["sit_low_hold"]))

        # The band rules out the wall sit, and the only other fit is the exercise just called too hard.
        self.assertEqual(band.intent, "exercise_adjust")
        self.assertEqual(band.text, "That one needs a wall. The only other beginner exercise I have for this goal with "
                                    "a band is the sit-low hold, which you found too hard. Try it at an easier dose: "
                                    "2 x 15 seconds. Shall I walk you through it?")
        self.assertEqual((band.actions, band.suggestions), ((ex("sit_low_hold"),), CHIPS_EASED))
        self.assertEqual((c4["exercise_id"], c4["equipment"], c4["level"], c4["too_hard"]),
                         ("sit_low_hold", ["band"], 1, ["sit_low_hold"]))

        self.assertEqual((plan.intent, plan.confidence), ("practice_plan", 1.0))
        self.assertEqual(plan.text, (
            "Here is a two-week plan to work on staying low, the split step and arm extension, with three sessions a "
            "week and nothing beyond a band. Session one: sit-low hold (2 x 15 seconds, easier dose), split-step hops "
            "(3 x 20 hops), forward lunge with reach (3 x 6 each leg). The shuttle is not tracked, so these goals come "
            "from movement and timing only. Warm up first, and stop any exercise that causes sharp pain. Want to know "
            "how to do the first one?"))
        self.assertEqual(plan.actions, (ex("sit_low_hold"), ex("split_step_hops"), ex("forward_lunge_reach")))
        self.assertEqual(plan.suggestions, ("How do I do it?", "Make it easier", "Make it harder",
                                            "I have no equipment"))
        self.assertEqual((c5["topic"], c5["shown_exercises"], c5["last_offer"]),
                         ("plan", ["sit_low_hold", "wall_sit", "split_step_hops", "forward_lunge_reach"],
                          "exercise_detail:sit_low_hold"))

        for reply, _ in turns:
            assert_sane(self, COACH, reply)
            for a in reply.actions:
                if a.kind == ActionKind.EXERCISE:
                    self.assertRegex(LIBRARY[a.ref].gif, r"^gifs/[a-z0-9_]+\.gif$")

    def test_every_chip_is_understood(self):
        turns = talk(COACH, ["how?", "show me", "too hard", "I only have a band", "make a plan"],
                     {"finding_id": "technique.stay_lower", "player": "near"})
        history, context = [], {"finding_id": "technique.stay_lower", "player": "near"}
        for reply, after in turns:
            for chip in reply.suggestions:
                with self.subTest(after=reply.text[:30], chip=chip):
                    answer, _ = COACH.reply(history, chip, after)
                    self.assertNotIn(answer.intent, ("unknown", "out_of_scope", "safety"))
            history += [Turn("user", "x"), Turn("coach", reply.text)]


class AdjustTest(unittest.TestCase):
    """A swap must keep the active goal, the stated equipment and the level, whatever the library edges say."""

    def adjust(self, start, gear, request):
        ctx = ChatContext(finding_id="technique.stay_lower", exercise_id=start, shown_exercises=(start,),
                          topic=f"exercise:{start}", equipment=gear)
        return COACH.reply([], request, ctx)

    def test_literal_cases(self):
        swap_low = "the sit-low hold (3 x 30 seconds). " + SAME_GOAL + " " + SIT_LOW_WHY + " Want to know how to do it?"
        cases = [
            ("goblet_squat", None, "make it harder", "goblet_squat", None, CHIPS_KEEP,
             "The goblet squat is already the hardest exercise I have for this goal with any equipment. Slow each rep "
             "down and pause at the hardest point. Want another exercise for the same goal?"),
            ("wall_sit", None, "no equipment", "sit_low_hold", (), CHIPS_SWAP,
             "That one needs a wall, so with no equipment try " + swap_low),
            ("goblet_squat", None, "I have no equipment", "sit_low_hold", (), CHIPS_SWAP,
             "That one needs a dumbbell, so with no equipment try " + swap_low),
            ("sit_low_hold", (), "make it harder", "sit_low_hold", (), CHIPS_KEEP,
             "The sit-low hold is already the hardest exercise I have for this goal with no equipment. Make each set "
             "a little longer while your form stays good. Want another exercise for the same goal?"),
            ("sit_low_hold", None, "make it harder", "goblet_squat", None, CHIPS_SWAP,
             "Harder option: the goblet squat (3 x 8). " + SAME_GOAL + " It loads the squat so you can sit lower and "
             "push back up into a jump. Want to know how to do it?"),
            ("sit_low_hold", None, "make it easier", "wall_sit", None, CHIPS_SWAP,
             "Easier option: the wall sit (3 x 30 seconds). " + SAME_GOAL + " It holds your legs in the low ready "
             "shape so you can stay down between shots. Want to know how to do it?"),
            ("sit_low_hold", (), "make it easier", "sit_low_hold", (), CHIPS_KEEP,
             "The sit-low hold is already the easiest exercise I have for this goal with no equipment. Try an easier "
             "dose instead: 2 x 15 seconds, with longer rests between sets. Want another exercise for the same goal?"),
        ]
        for start, gear, request, end, end_gear, chips, text in cases:
            with self.subTest(start=start, gear=gear, request=request):
                reply, ctx = self.adjust(start, gear, request)
                self.assertEqual(reply.intent, "exercise_adjust")
                self.assertEqual(reply.text, text)
                self.assertEqual(reply.actions, (ex(end),))
                self.assertEqual(reply.suggestions, chips)
                self.assertEqual((ctx.exercise_id, ctx.equipment), (end, end_gear))

    def test_harder_never_drops_the_stance_goal(self):
        # goblet_squat's harder edge in the library is squat_jumps, which trains jump power but not a lower stance.
        self.assertEqual(LIBRARY["goblet_squat"].harder, "squat_jumps")
        self.assertNotIn("lower_stance", LIBRARY["squat_jumps"].fix_tags)
        reply, ctx = self.adjust("goblet_squat", None, "harder please")
        self.assertNotIn("squat_jumps", [a.ref for a in reply.actions])
        self.assertEqual(ctx.exercise_id, "goblet_squat")

    def test_no_equipment_never_keeps_gear(self):
        for start in ("wall_sit", "goblet_squat", "sit_low_hold"):
            for request in ("no equipment", "I have no equipment", "make it easier, I have nothing"):
                with self.subTest(start=start, request=request):
                    reply, ctx = self.adjust(start, None, request)
                    self.assertEqual(ctx.equipment, ())
                    for a in reply.actions:
                        if a.kind == ActionKind.EXERCISE:
                            self.assertEqual(LIBRARY[a.ref].equipment, ())
                            self.assertIn("lower_stance", LIBRARY[a.ref].fix_tags)


SMOKE_FLOW = ["hi", "what should I work on first?", "yes", "how do I fix it?", "how do I do it?", "too hard",
              "I only have a band", "give me another one", "why this exercise?", "make me a practice plan",
              "make it easier", "I have no equipment", "thanks"]
FULL_SIT_LOW = re.compile(r"sit-low hold\W+3 x 30|3 x 30 seconds of the sit-low hold", re.I)


def plan_text(gear):
    return ("Here is a two-week plan to work on staying low, the split step and arm extension, with three sessions a "
            f"week and {gear}. Session one: sit-low hold (2 x 15 seconds, easier dose), split-step hops (3 x 20 "
            "hops), forward lunge with reach (3 x 6 each leg). The shuttle is not tracked, so these goals come from "
            "movement and timing only. Warm up first, and stop any exercise that causes sharp pain. ")


class TooHardTest(unittest.TestCase):
    """An exercise the player called too hard never comes back unchanged: not after a gear change, not in a plan."""

    def test_smoke_flow_keeps_the_rejection(self):
        turns = talk(COACH, SMOKE_FLOW, {})
        said = dict(zip(SMOKE_FLOW, turns))
        for reply, _ in turns:
            assert_sane(self, COACH, reply)

        detail, c = said["how do I do it?"]
        self.assertTrue(detail.text.startswith("Sit-low hold: 3 x 30 seconds. Cues:"), detail.text)
        self.assertEqual(c["too_hard"], [])

        easier, c = said["too hard"]
        self.assertEqual((easier.actions, c["exercise_id"], c["level"], c["too_hard"]),
                         ((ex("wall_sit"),), "wall_sit", 1, ["sit_low_hold"]))

        band, c = said["I only have a band"]
        self.assertEqual(band.text, "That one needs a wall. The only other beginner exercise I have for this goal with "
                                    "a band is the sit-low hold, which you found too hard. Try it at an easier dose: "
                                    "2 x 15 seconds. Want to know how to do it?")
        self.assertEqual((band.actions, band.suggestions), ((ex("sit_low_hold"),), CHIPS_EASED))

        another, _ = said["give me another one"]
        self.assertEqual(another.text, 'I have no other beginner exercise for this goal that works with a band. To '
                                       'widen the choice, say "Any equipment is fine" or tell me a higher level.')
        self.assertEqual(another.actions, ())

        plan, c = said["make me a practice plan"]
        self.assertEqual(plan.text, plan_text("nothing beyond a band") + "Shall I walk you through the first one?")
        app_plan = build_plan(REPORT, LIBRARY, player="near", level=c["level"], equipment=c["equipment"],
                              avoid=c["too_hard"])
        eased = [b for w in app_plan.weeks for s in w.sessions for b in s.blocks if b.exercise_id == "sit_low_hold"]
        self.assertEqual(len(eased), 6)
        self.assertEqual({(b.dose, b.reason.endswith(EASED)) for b in eased}, {("2 x 15 seconds", True)})

        same, _ = said["make it easier"]
        self.assertEqual(same.text, "With the exercises I have for these goals, I cannot make this plan any easier. "
                                    "Instead, do fewer reps and rest longer between sets.")
        none, c = said["I have no equipment"]
        self.assertEqual(none.text, plan_text("no equipment") + "Shall I walk you through the first one?")

        for (reply, c), question in list(zip(turns, SMOKE_FLOW))[SMOKE_FLOW.index("too hard"):]:
            with self.subTest(question=question):
                self.assertEqual(c["too_hard"], ["sit_low_hold"])
                self.assertIsNone(FULL_SIT_LOW.search(reply.text), reply.text)

    def test_harder_brings_back_the_full_dose(self):
        flow = SMOKE_FLOW[:-1] + ["how do I do it?", "how many?", "make it harder", "make me a practice plan"]
        (detail, _), (dose, _), (harder, context), (plan, _) = talk(COACH, flow, {})[-4:]
        self.assertTrue(detail.text.startswith("Sit-low hold: 2 x 15 seconds, an easier dose as you found it too "
                                               "hard. Cues:"), detail.text)
        self.assertEqual(dose.text, "For the sit-low hold, do 2 x 15 seconds, an easier dose as you found it too hard. "
                                    "Key point: feet wider than shoulders. Shall I find another exercise for the same "
                                    "goal?")
        self.assertEqual(harder.text, "Harder option: the full dose of the sit-low hold (3 x 30 seconds). " + SAME_GOAL
                         + " Want to know how to do it?")
        self.assertEqual((harder.actions, context["too_hard"]), ((ex("sit_low_hold"),), []))
        self.assertIn("Session one: sit-low hold (3 x 30 seconds), split-step hops", plan.text)

    def test_both_stance_exercises_too_hard(self):
        turns = talk(COACH, ["too hard", "too hard", "give me another one", "make me a practice plan"],
                     {"finding_id": "technique.stay_lower", "player": "near", "exercise_id": "sit_low_hold"})
        (first, _), (second, c2), (another, c3), (plan, c4) = turns
        self.assertEqual(first.actions, (ex("wall_sit"),))
        self.assertEqual(second.text, "The wall sit is already the easiest exercise I have for this goal with any "
                                      "equipment. Try an easier dose instead: 2 x 15 seconds, with longer rests "
                                      "between sets. Shall I find another exercise for the same goal?")
        self.assertEqual(c2["too_hard"], ["sit_low_hold", "wall_sit"])
        self.assertEqual(another.text, "The only other beginner exercise I have for this goal is the sit-low hold, "
                                       "which you found too hard. Try it at an easier dose: 2 x 15 seconds. Want to "
                                       "know how to do it?")
        self.assertEqual(another.suggestions, ("How do I do it?", "I am intermediate", "Make me a practice plan"))
        self.assertIn("sit-low hold (2 x 15 seconds, easier dose)", plan.text)
        self.assertNotIn("(3 x 30 seconds)", plan.text)
        self.assertEqual(c4["too_hard"], ["sit_low_hold", "wall_sit"])
        for reply, _ in turns:
            assert_sane(self, COACH, reply)


SAFETY_QUESTIONS = [
    "my knee hurts when I lunge", "I felt a sharp pain in my ankle", "my shoulder is sore after smashing",
    "I think I sprained my ankle", "I hurt my back yesterday", "is it safe to train with a sore knee?", "I feel dizzy",
    "I have chest pain when I play", "my wrist clicks and it's painful", "should I see a doctor about my knee",
    "i got injured last week", "I pulled a muscle in my calf", "my achilles is tight and hurts",
    "I can't breathe properly after rallies", "I fainted during practice", "my elbow is swollen", "I have tennis elbow",
    "my knee gave way when I landed", "I tore my ACL last year, can I still lunge?", "I'm recovering from surgery",
    "my hamstring feels strained", "my lower back aches after training", "I rolled my ankle", "I have shin splints",
    "my foot has been numb since the match", "I keep getting cramps in my legs", "my knees feel weak and unstable",
    "me duele la rodilla", "膝が痛い", "MY KNEE HURTZ", "ouch my wrist", "is this exercise bad for my knees?",
    "my heart is racing and I feel faint", "I fell and hit my head", "yes but my knee hurts",
    "make it harder, my back is killing me", "my calf popped during a lunge", "i think i broke my toe",
    "ankle twisted landing from a jump", "can i play with a stress fracture", "the pain in my hip is agony",
    "my shin is tender after jumping",
]

NOT_SAFETY = [
    "x" * 5000, "how do I keep my chest up?", "what is a split step?", "how far was I from base?", "make it harder",
    "my backhand is weak", "how do I get back to base faster?", "I have no equipment",
    "show me the moment where my hips are high", "how do I bend my knees more?", "is it safe to say I improved?",
    "how do I hit a sharp angle?", "my smash is weak", "how do I warm up?", "what does knee bend mean?",
    "which arm should be straight?", "my legs are slow", "I am a beginner",
    "is it ok for my hips to stay that low", "is it ok for my knee to go past my toes", "go easy on the back foot",
    "how hard should I push on the lunge", "how many reps of the side plank", "is wall sit good for my knees",
]

# A worry about the body stops coaching even inside an exercise question; the parse still records the exercise named.
SAFETY_IN_EXERCISE = [
    ("how many sets of wall sits can I do if my knee keeps acting up", "wall_sit"),
    ("is the goblet squat ok for my lower back", "goblet_squat"),
    ("are squat jumps hard on the knees?", "squat_jumps"),
    ("my ankle gives out on the skater hops", "skater_hops"),
    ("I threw my back out last week, can I still do planks", "front_plank"),
    ("got a niggle in my calf, how many calf raises", "calf_raises"),
    ("my shoulder has been bothering me during wall angels", "wall_angels"),
    ("is jump rope easy on the ankles", "jump_rope"),
    ("I tweaked something in my hip doing lateral lunges", "lateral_lunge"),
    ("my wrist is taped, can I still do band rotations", "band_trunk_rotation"),
    ("will the side plank be too much for my back", "side_plank"),
    ("can I do single leg deadlifts with a dodgy knee", "single_leg_romanian_deadlift"),
    ("I've had a knee replacement, which exercises are fine", None),
    ("should I wear a knee brace for the lunges", None),
]


class SafetyTest(unittest.TestCase):
    def test_kb_wording_is_the_approved_text(self):
        self.assertEqual((KB.safety["text"], KB.safety["suggestions"]), (APPROVED_SAFETY, []))

    def test_pain_and_injury_always_stop_coaching(self):
        self.assertGreaterEqual(len(SAFETY_QUESTIONS), 25)
        open_offer = ChatContext(finding_id="technique.stay_lower", exercise_id="wall_sit", topic="exercise:wall_sit",
                                 last_offer="exercise_detail:wall_sit")
        for context in (None, open_offer):
            for q in SAFETY_QUESTIONS:
                with self.subTest(q=q, context=context is not None):
                    reply, ctx = COACH.reply([], q, context)
                    self.assertEqual(reply, Reply(APPROVED_SAFETY, (), "safety", 1.0, ()))
                    self.assertIsNone(ctx.last_offer)

    def test_safety_text_is_never_reworded(self):
        history = [Turn("user", "my knee hurts"), Turn("coach", APPROVED_SAFETY)]
        reply, _ = COACH.reply(history, "my knee still hurts", None)
        self.assertEqual(reply.text, APPROVED_SAFETY)

    def test_coaching_words_are_not_mistaken_for_injury(self):
        for q in NOT_SAFETY:
            with self.subTest(q=q[:40]):
                self.assertNotEqual(COACH.reply([], q, None)[0].intent, "safety")

    def test_a_worry_inside_an_exercise_question_still_stops_coaching(self):
        offer = {"finding_id": "technique.stay_lower", "player": "near"}
        for q, named in SAFETY_IN_EXERCISE:
            with self.subTest(q=q):
                reply, ctx, parse = COACH.reply_traced([], q, offer)
                self.assertEqual(reply, Reply(APPROVED_SAFETY, (), "safety", 1.0, ()))
                self.assertEqual(parse.exercise_id or parse.missing, named)
                self.assertIsNone(ctx.exercise_id)


# What the player sees and the offer the next turn inherits, for the fixed stop reply.
BLOCKED = (APPROVED_SAFETY, (), (), None)
LIST_REPLY = (
    "For staying low, try these:\n- Sit-low hold (3 x 30 seconds): Trains the ready stance so your hips stay down while "
    "you wait for the next shot.\n- Wall sit (3 x 30 seconds): Holds your legs in the low ready shape so you can stay "
    "down between shots.\nWant to know how to do the first one?", (ex("sit_low_hold"), ex("wall_sit")),
    ("How do I do it?", "Give me another exercise", "Make it easier", "Make me a practice plan"),
    "exercise_detail:sit_low_hold")
LIMITS_REPLY = (
    "I only see body movement in one flat video. I cannot track the shuttle, the racket, your heart rate or true 3D "
    "angles, so angles are called projected and every number is an estimate with a confidence word. Targets are "
    "practice guides, not norms tested for you.", (), ("What did I do well?",), None)
STANCE_WIDTH = Action(ActionKind.FINDING, "technique.stance_width", 0)


def shown(reply, context):
    return reply.text, reply.actions, reply.suggestions, context.get("last_offer")


def stops(turns):
    return [context.get("pain_stop") for _, context in turns]


class PainStopTest(unittest.TestCase):
    """Once the coach stops for pain it stays stopped for the rest of the chat. Only replies with no training in them
    (what went well, the video moments, what the tool cannot measure) still get their own answer."""

    def test_the_inspected_pain_sequence_stays_stopped(self):
        turns = talk(COACH, ["my knee hurts", "how do I fix it?", "yes"],
                     {"player": "far", "finding_id": "footwork.split_step"})
        self.assertEqual([shown(r, c) for r, c in turns], [BLOCKED] * 3)
        self.assertEqual([r.intent for r, _ in turns], ["safety", "how_to_fix", "affirm"])
        self.assertEqual(stops(turns), [True] * 3)

    def test_an_exercise_offer_made_before_the_pain_is_dropped(self):
        turns = talk(COACH, ["what exercises help?", "my knee hurts", "yes"], {})
        self.assertEqual([shown(r, c) for r, c in turns], [LIST_REPLY, BLOCKED, BLOCKED])
        self.assertEqual([r.intent for r, _ in turns], ["exercise_request", "safety", "affirm"])
        self.assertEqual(stops(turns), [False, True, True])

    def test_every_later_training_request_gets_the_stop(self):
        turns = talk(COACH, ["what exercises help?", "my knee hurts", "yes", "what's next?", "make me a practice plan",
                             "make it easier", "make it harder", "give me a drill"], {})
        self.assertEqual([shown(r, c) for r, c in turns], [LIST_REPLY] + [BLOCKED] * 7)
        self.assertEqual([r.intent for r, _ in turns[1:]], ["safety", "affirm", "next", "practice_plan",
                                                             "exercise_adjust", "exercise_adjust", "exercise_request"])
        self.assertEqual(stops(turns), [False] + [True] * 7)

    def test_a_stale_offer_cannot_restart_training(self):
        for offer in ("practice_plan", "exercise_detail:wall_sit", "another:sit_low_hold",
                      "how_to_fix:technique.stay_lower", "exercise_request"):
            with self.subTest(offer=offer):
                self.assertNotEqual(COACH.reply([], "yes", {"last_offer": offer})[0].text, APPROVED_SAFETY)
                reply, ctx = COACH.reply([], "yes", {"pain_stop": True, "last_offer": offer})
                after = context_to_dict(ctx)
                self.assertEqual(shown(reply, after), BLOCKED)
                self.assertEqual((reply.intent, after.get("pain_stop")), ("affirm", True))

    def test_questions_with_no_training_in_them_still_get_answers(self):
        turns = talk(COACH, ["my knee hurts", "what is a split step?", "what did I do well?", "yes",
                             "what can't you measure?", "give me a drill"], {})
        self.assertEqual([shown(r, c) for r, c in turns], [
            BLOCKED,
            BLOCKED,  # a word definition can describe a movement, such as the split step's hop
            ("What went well:\n- Good stance width while you wait\n- Fast, steady racket-hand speed\nWant to see one on "
             "video?", (STANCE_WIDTH,), ("Show me my best one",), "show_best:technique.stance_width"),
            ("Playing it now: good stance width.", (STANCE_WIDTH, Action(ActionKind.SEEK, "technique.stance_width", 0)),
             (), None),
            LIMITS_REPLY,
            BLOCKED,
        ])
        self.assertEqual([r.intent for r, _ in turns],
                         ["safety", "terms", "strengths", "affirm", "limits", "exercise_request"])
        self.assertEqual(stops(turns), [True] * 6)

    def test_a_moment_after_the_stop_drops_its_exercise_offer(self):
        moment = ("Playing it now: hips high while waiting. It is one of two moments like this, so ask again to see the "
                  "next.")
        seek = (Action(ActionKind.FINDING, "technique.stay_lower", 0), Action(ActionKind.SEEK, "technique.stay_lower", 0))
        plain = talk(COACH, ["what exercises help?", "show me the moment"], {})[-1]
        self.assertEqual(shown(*plain), (f"{moment} Want to know how to do the sit-low hold?", (*seek, ex("sit_low_hold")),
                                         ("How do I do it?", "Show me the moment", "Give me another exercise"),
                                         "exercise_detail:sit_low_hold"))
        stopped = talk(COACH, ["what exercises help?", "my knee hurts", "show me the moment"], {})[-1]
        self.assertEqual(shown(*stopped), (moment, seek, ("Show me the moment",), None))
        self.assertEqual((stopped[0].intent, stopped[1].get("pain_stop")), ("show_moment", True))

    def test_the_stop_survives_json_a_missing_flag_and_trimmed_history(self):
        [(_, data)] = talk(COACH, ["my knee hurts"], {})
        reply, ctx = COACH.reply([], "what exercises help?", data)
        self.assertEqual((shown(reply, context_to_dict(ctx)), reply.intent), (BLOCKED, "exercise_request"))
        self.assertEqual((data.get("pain_stop"), context_to_dict(ctx).get("pain_stop")), (True, True))
        history = [Turn("user", "my knee hurts"), Turn("coach", APPROVED_SAFETY)]
        for context in (None, {}, {"pain_stop": False}, {"pain_stop": None}):
            with self.subTest(context=context):
                reply, ctx = COACH.reply(history, "make it harder", context)
                after = context_to_dict(ctx)
                self.assertEqual((shown(reply, after), reply.intent), (BLOCKED, "exercise_adjust"))
                self.assertIs(after.get("pain_stop"), True)
        turns = talk(COACH, ["my knee hurts", "what did I do well?", "yes", "what can't you measure?",
                             "what did the far player do well?", "show me the moment"], {})
        self.assertEqual([shown(r, c) for r, c in turns[-2:]], [
            ("What went well for the other player:\n- They stay low and ready between shots\n- Quick return to base "
             "after their shots\n- Good knee bend in the ready stance\nThe shuttle is not tracked, so this comes from "
             "movement and timing only.\nWant to see one on video?",
             (Action(ActionKind.FINDING, "technique.stay_lower", 1),), ("Show me their best one",),
             "show_best:technique.stay_lower@far"),
            ("Playing it now: low ready stance.", (Action(ActionKind.FINDING, "technique.stay_lower", 1),
                                                   Action(ActionKind.SEEK, "technique.stay_lower", 0)), (), None)])
        trimmed = [Turn("user", "show me the moment"), Turn("coach", turns[-1][0].text)]
        reply, ctx = COACH.reply(trimmed, "make me a practice plan", turns[-1][1])
        self.assertEqual((shown(reply, context_to_dict(ctx)), reply.intent), (BLOCKED, "practice_plan"))
        self.assertEqual(stops(turns) + [context_to_dict(ctx).get("pain_stop")], [True] * 7)

    def test_a_fresh_chat_is_not_stopped(self):
        talk(COACH, ["my knee hurts", "yes"], {})
        turns = talk(COACH, ["what exercises help?", "yes"], {})
        self.assertEqual([shown(r, c) for r, c in turns], [LIST_REPLY, (
            "Sit-low hold: 3 x 30 seconds. Cues: feet wider than shoulders; knees bent and out. Watch out for: standing "
            "tall. Safety: Stop if you feel sharp knee pain. Bend only as far as you can control. Also: chest up; "
            "weight on the whole foot. Shall I find another exercise for the same goal?", (ex("sit_low_hold"),),
            ("Why this exercise?", "Make it easier", "Make it harder", "Give me another exercise"),
            "another:sit_low_hold")])
        self.assertEqual(stops(turns), [False, False])

    def test_no_dev_question_gets_training_after_a_stop(self):
        items = json.loads(Path(__file__).with_name("dev_questions.json").read_text(encoding="utf-8"))
        training = [s for e in LIBRARY.values() for s in (e.name, e.sets_reps, e.why, e.safety, *e.cues, *e.mistakes)]
        training += [s for g in KB.fix_guides.values()
                     for s in (g["simple"], g["why"], g["feel"], g["progression"], *g["cues"])]
        training += [row["text"] for row in KB.tactics.values()]
        training = sorted({s.lower().rstrip(".") for s in training})
        answered = set()
        for item in items:
            reply, ctx = COACH.reply([], item["text"], dict(item["context"], pain_stop=True))
            after = context_to_dict(ctx)
            with self.subTest(id=item["id"]):
                if shown(reply, after) != BLOCKED:
                    answered.add(reply.intent)
                    self.assertEqual([s for s in training if s in reply.text.lower()], [])
                    self.assertNotRegex(reply.text, r"\d+ x \d+")
                    self.assertNotIn(ActionKind.EXERCISE, [a.kind for a in reply.actions])
                    self.assertIn(split_tag(after["last_offer"])[0], ("", "show_best", "show_moment"))
                    for chip in reply.suggestions:
                        self.assertNotEqual(COACH.reply([], chip, after)[0].text, APPROVED_SAFETY, chip)
                self.assertIs(after.get("pain_stop"), True)
        self.assertEqual(answered, {"limits", "strengths", "show_moment"})


EXERCISE_NAMES = [
    ("how many wall sits should I do", "wall_sit"), ("is wall sitting good for my legs", "wall_sit"),
    ("how do I do the wal sit", "wall_sit"), ("wallsits every day?", "wall_sit"),
    ("single legged calf raises how many", "single_leg_calf_raise"), ("calve raises how", "calf_raises"),
    ("can I do single leg deadlifts instead", "single_leg_romanian_deadlift"),
    ("1 leg deadlift ok?", "single_leg_romanian_deadlift"), ("how about doing romanian deadlifts", "single_leg_romanian_deadlift"),
    ("what about pullaparts", "band_pull_apart"), ("side plnak for how long", "side_plank"),
    ("sideplank form?", "side_plank"), ("how about box jumping", "box_jumps"), ("jump onto a box instead?", "box_jumps"),
    ("6 corner shadow reps", "six_corner_shadow"), ("do split step jumps help", "split_step_hops"),
    ("standing turn with arms?", "standing_trunk_turn"), ("how do I do the rotation with a band", "band_trunk_rotation"),
    ("wall angles are hard", "wall_angels"), ("stance hold how long", "sit_low_hold"), ("jumpropes every day", "jump_rope"),
    ("I need to sprint faster", None), ("how do I step up my footwork", None), ("is a run good for stamina", None),
]


class ExerciseNameTest(unittest.TestCase):
    def test_names_are_read_through_plurals_typos_and_joined_words(self):
        for q, named in EXERCISE_NAMES:
            with self.subTest(q=q):
                parse = COACH.classifier.parse(q)
                self.assertEqual(parse.exercise_id or parse.missing, named)

    def test_an_exercise_name_holding_equipment_words_is_not_an_equipment_change(self):
        self.assertIsNone(COACH.classifier.parse("how do I do the rotation with a band").equipment)

    def test_player_turns_in_a_question_file_count_as_the_user(self):
        said = COACH.reply([], "what am I good at?", {"player": "near"})[0].text
        for speaker in ("user", "player"):
            with self.subTest(speaker=speaker):
                history = [{"speaker": speaker, "text": "what am I good at?"}, {"speaker": "coach", "text": said}]
                reply, _ = COACH.reply(history, "what about the other player?", {"player": "near"})
                self.assertEqual(reply.intent, "strengths")


NEAR = {"player": "near"}
LOW = {"finding_id": "technique.stay_lower", "player": "near"}
BASE = {"finding_id": "tactics.base_discipline", "player": "near"}
WALL = "wall"  # the context after "how do I do the wall sit" in LOW
INTENT_RULES = [
    # Praise words about a player ask for strengths; fault, advice, show and statement frames do not.
    (LOW, "what did I nail", "strengths"), (NEAR, "where do I shine", "strengths"),
    (NEAR, "is there anything I do really well", "strengths"), (LOW, "what parts of my footwork look good", "strengths"),
    (NEAR, "can you tell me where I shine", "strengths"), (NEAR, "nah im good", "deny"),
    (NEAR, "good morning, just watched my clip", "greeting"), (NEAR, "show me my best one", "show_moment"),
    (LOW, "where was my best rep", "show_moment"), (NEAR, "what's the best way to fix my lunge", "how_to_fix"),
    (NEAR, "who is the best badminton player ever", "out_of_scope"),
    # A reading in focus: fault and meaning words ask about it; whole-game and advice words do not.
    (LOW, "is that something to worry about", "explain"), (LOW, "what's the reasoning behind that", "explain"),
    (BASE, "what's wrong with my position", "explain"), (LOW, "what's wrong with me", "summary"),
    (LOW, "what am I doing wrong", "summary"), (BASE, "where should I stand instead", "tactics_tip"),
    # Both players plus a contrast word compare them.
    (NEAR, "show me how we differ", "compare_players"), (NEAR, "are our lunges different", "compare_players"),
    (NEAR, "who recovered faster", "compare_players"),
    # Purpose questions about an exercise; "meant to" prescribes rather than asking for a clip reading.
    (WALL, "what does this exercise do for me", "exercise_why"), (LOW, "what is the goblet squat meant to fix", "exercise_why"),
    (WALL, "what's the reason for this one", "exercise_why"), (NEAR, "what is good about the wall sit", "exercise_why"),
    (WALL, "is it worth it", "exercise_why"), (WALL, "how deep am I meant to go", "exercise_detail"),
    (WALL, "how fast should I do it", "exercise_detail"), (WALL, "how low was my stance", "metric_question"),
    (NEAR, "how should I play against someone faster", "tactics_tip"),
    (NEAR, "how do I beat a faster opponent", "tactics_tip"), (NEAR, "what does coverage mean", "terms"),
]


class IntentRuleTest(unittest.TestCase):
    def test_word_class_rules_and_their_controls(self):
        wall = COACH.reply([], "how do I do the wall sit", LOW)[1]
        for ctx, q, intent in INTENT_RULES:
            with self.subTest(q=q):
                context = wall if ctx is WALL else context_from_dict(ctx)
                self.assertEqual(COACH.classifier.parse(q, context).intent, intent)

    def test_a_past_tense_topic_word_still_names_its_finding(self):
        self.assertEqual(COACH.classifier.parse("who recovered faster").finding_id, "footwork.recovery")

    def test_typo_fixing_never_rewrites_a_word_the_rules_test_for(self):
        self.assertLessEqual(intents.RULE_WORDS, COACH.classifier.vocab)
        self.assertEqual(COACH.classifier.tokens("where do I shine")[1], ["where", "do", "i", "shine"])


class RobustnessTest(unittest.TestCase):
    def test_odd_inputs_get_a_sane_reply(self):
        inputs = ["", "   ", "x" * 5000, "how " * 2500, "¿Cómo puedo mejorar mi juego?", "如何提高我的步法", "🏸🏸🔥",
                  "'; DROP TABLE users; --", "<script>alert(1)</script>", "\x00\x07\u200b", "?", "how how how how",
                  "make it harder " * 50, "what what", "asdfghjkl qwerty"]
        for text in inputs:
            with self.subTest(text=text[:30]):
                reply, ctx = COACH.reply([], text, None)
                assert_sane(self, COACH, reply)
                json.dumps(context_to_dict(ctx), allow_nan=False)
        for user_text, history in ((None, None), (42, "not a list"), ("hi", [1, "x", {"speaker": "bot", "text": 3}])):
            with self.subTest(user_text=user_text):
                assert_sane(self, COACH, COACH.reply(history, user_text, None)[0])

    def test_repeating_a_question_never_loops_the_same_words(self):
        history, seen = [], []
        for _ in range(4):
            reply, _ = COACH.reply(history, "what should I work on first?", None)
            assert_sane(self, COACH, reply)
            self.assertNotIn(reply.text, seen[-1:])
            seen.append(reply.text)
            history += [Turn("user", "what should I work on first?"), Turn("coach", reply.text)]

    def test_same_input_same_reply(self):
        ctx = {"finding_id": "footwork.split_step", "player": "near", "level": 2}
        self.assertEqual(COACH.reply([], "what exercises help?", ctx), COACH.reply([], "what exercises help?", ctx))


FIRST = ("Most important first:\n- Hips stay high while you wait for the shot (a top priority)\n- No split step seen at "
         "many of the opponent's hits (a top priority)\n- Slow return to base after your own shots (worth improving)\n"
         "The shuttle is not tracked, so this comes from movement and timing only.\n")


class WordingTest(unittest.TestCase):
    def test_closing_question_varies_by_exchange(self):
        q = "what should I work on first?"
        first, _ = COACH.reply([], q, None)
        self.assertEqual(first.text, FIRST + "Want me to show the first one?")
        later, _ = COACH.reply([Turn("user", "hi"), Turn("coach", "Hello.")], q, None)
        self.assertEqual(later.text, FIRST + "Shall I play the first one?")
        again, _ = COACH.reply([Turn("user", q), Turn("coach", first.text), Turn("user", "ok"), Turn("coach", "Sure.")],
                               q, None)
        self.assertEqual(again.text, FIRST + "Shall I play the first one?")

    def test_other_player_is_spoken_of_in_third_person(self):
        ctx = ChatContext(finding_id="footwork.first_step", topic="finding:footwork.first_step@far", player=Role.NEAR)
        reply, after = COACH.reply([], "what else?", ctx)
        self.assertEqual(reply.intent, "next")
        self.assertEqual(reply.text, (
            "Next for the other player: overhead hits are low. Hitting the shuttle higher and earlier gives them "
            "steeper shots and more time. A low contact point lets the shuttle drop, so their shots are flatter and "
            "the opponent gets more time. Want to know how to fix it?"))
        self.assertEqual(reply.actions, (Action(ActionKind.FINDING, "contact.reach_height", 1),))
        self.assertEqual(reply.suggestions, ("How would they fix it?", "Show me the moment", "Why does it matter?"))
        self.assertEqual(after.last_offer, "how_to_fix:contact.reach_height@far")

    def test_position_is_described_not_judged(self):
        reply, ctx = COACH.reply([], "how far was I from base?", None)
        self.assertIn(DESCRIPTIVE_NOTE, reply.text)
        self.assertIn("a coaching aim rather than a tested norm for you", reply.text)
        self.assertTrue(reply.text.endswith("Want some practice ideas for it?"))
        yes, _ = COACH.reply([Turn("user", "how far was I from base?"), Turn("coach", reply.text)], "yes", ctx)
        self.assertEqual(yes.intent, "affirm")
        self.assertTrue(yes.text.startswith(
            "This clip does not show a clear fault in distance from base, so this is general practice: "), yes.text)

    def test_preview_note_is_said_once(self):
        coach = Coach(preview_report(), LIBRARY, KB)
        turns = talk(coach, ["how far was I from base?", "how high are my hips?"], {})
        (first, _), (second, _) = turns
        # The note and the longer wording of the question both count against the limit, so the shuttle and
        # description notes take their short forms here.
        self.assertEqual(first.text, (
            "Your distance from base when they hit was 0.9 metres over 12 readings. A practice guide is about 0.3 to "
            "0.8 metres, a coaching aim rather than a tested norm for you. I am somewhat sure about this reading. "
            + SHUTTLE_SHORT + " " + DESCRIPTIVE_SHORT + " " + PREVIEW_NOTE + " Want some practice ideas for it?"))
        self.assertEqual(second.text, (
            "Your projected hip height while waiting was 53% of body height over 14 readings. A practice guide is "
            "about 42% to 48% of body height, a coaching aim rather than a tested norm for you. I am fairly sure about "
            "this reading. Shall I give you a fix for it?"))

    def test_is_that_good_judges_the_reading_in_focus(self):
        (_, _), (mine, ctx) = talk(COACH, ["how high are my hips?", "is that good?"], {})
        self.assertEqual(mine.intent, "metric_question")
        self.assertEqual(mine.text, (
            "Your projected hip height while waiting was 53% of body height over 14 readings. That is above the "
            "practice guide of about 42% to 48% of body height, a coaching aim rather than a tested norm for you. The "
            "analysis ranks it as a top thing to work on. I am fairly sure about this reading. Shall I give you a fix "
            "for it?"))
        self.assertEqual(ctx["last_offer"], "how_to_fix:technique.stay_lower")
        (_, _), (theirs, ctx) = talk(COACH, ["what does the far player do well?", "is that good?"], {})
        self.assertEqual(theirs.text, (
            "The other player's projected hip height while waiting was 45% of body height over 12 readings. That is "
            "within the practice guide of about 42% to 48% of body height, a coaching aim rather than a tested norm "
            "for them. The analysis counts it as a strength. I am somewhat sure about this reading."))
        self.assertEqual(theirs.suggestions, ("What should I work on first?", "Show me the moment",
                                              "What can't you measure?"))
        self.assertIsNone(ctx["last_offer"])

    def test_is_that_good_without_a_reading_asks_which_one(self):
        reply, _ = COACH.reply([], "is that good?", None)
        self.assertEqual((reply.intent, reply.actions), ("unknown", ()))
        self.assertEqual(reply.text, "Do you mean one of the readings from your clip? Name it, for example staying low, "
                                     "and I will say how it compares with the practice guide and how sure I am.")
        (_, _), (why, _) = talk(COACH, ["what exercises help?", "is that good?"], {})
        self.assertEqual(why.intent, "exercise_why")
        self.assertTrue(why.text.startswith("Why the sit-low hold? " + SIT_LOW_WHY), why.text)

    def test_no_unearned_certainty(self):
        for q in ("how do I fix it?", "what exercises help?", "make me a plan", "how far was I from base?"):
            text = COACH.reply([], q, None)[0].text.lower()
            self.assertIsNone(re.search(r"\b(you must|guaranteed|always works|will fix)\b", text), text)

    def test_unmeasured_speed_and_force_are_limits(self):
        shuttle = ("The shuttle is not tracked, so I cannot tell shot speed, shot type, placement, or whether it landed "
                   "in. The advice uses player movement and timing only.")
        force = "I cannot measure force, power or how hard you hit. I only see body positions and timing in the video."
        racket = ("I cannot see the racket face, grip or strings clearly enough to measure them. I follow the hitting "
                  "hand instead.")
        for q, text in (("how fast was my smash?", shuttle), ("what was my smash speed", shuttle),
                        ("how hard do i hit?", force), ("how powerful is my smash", force),
                        ("how fast is my racket", racket)):
            with self.subTest(q=q):
                reply, _ = COACH.reply([], q, None)
                self.assertEqual((reply.intent, reply.text, reply.actions), ("limits", text, ()))
        swing, _ = COACH.reply([], "how fast was my swing?", None)
        self.assertEqual((swing.intent, swing.actions), ("metric_question",
                                                         (Action(ActionKind.FINDING, "contact.hand_speed", 0),)))
        self.assertTrue(swing.text.startswith("Your projected racket-hand speed at the swing peak was 7.4 body heights "
                                              "a second over 9 readings."), swing.text)

    def test_text_speak_and_farewell(self):
        self.assertEqual(COACH.reply([], "wat shud i wrk on", None)[0].intent, "summary")
        for q in ("bye", "ok thanks bye", "see you later"):
            with self.subTest(q=q):
                reply, _ = COACH.reply([], q, None)
                self.assertEqual((reply.intent, reply.text, reply.actions, reply.suggestions),
                                 ("thanks", "Bye for now, and good luck with your practice. Come back any time with a "
                                  "question.", (), ("Make me a practice plan", "What should I work on first?")))

    def test_plural_names_and_single_picks_read_naturally(self):
        ctx = ChatContext(finding_id="footwork.split_step", exercise_id="split_step_hops",
                          shown_exercises=("split_step_hops",), topic="exercise:split_step_hops", equipment=())
        harder, _ = COACH.reply([], "make it harder", ctx)
        self.assertEqual(harder.text, "The split-step hops are already the hardest exercise I have for this goal with "
                                      "no equipment. Add a few more to each set, keeping every landing soft and quiet. "
                                      "Want another exercise for the same goal?")
        gear, _ = COACH.reply([], "what equipment do I need?", ctx)
        self.assertTrue(gear.text.startswith("The split-step hops need no equipment. You can do them anywhere."),
                        gear.text)
        keep, _ = COACH.reply([], "no equipment", replace(ctx, equipment=None))
        self.assertEqual(keep.text, "Good news: the split-step hops need no equipment, so you can keep them. Want to "
                                    "know how to do them?")
        one, _ = COACH.reply([], "what exercises help?", ChatContext(finding_id="contact.hand_speed", equipment=()))
        self.assertEqual(one.text, "For racket-hand speed, try this:\n- Standing trunk turn (3 x 8 each side): Trains "
                                   "the trunk turn you use to load a clear and a smash.\nWant to know how to do it?")
        self.assertEqual(one.actions, (ex("standing_trunk_turn"),))


REAL_DOC = Path(__file__).resolve().parents[2] / "fast" / "runs" / "step4-rebalanced-dev" / "outputs" / "01-r1.json"


def real_hand_reports():
    from coach.analyze import analyze
    from coach.player_input import player_inputs
    unknown = json.loads(REAL_DOC.read_text(encoding="utf-8"))
    unknown["source"].pop("players", None)
    declared = json.loads(REAL_DOC.with_name("04-r1.json").read_text(encoding="utf-8"))
    declared["source"]["players"] = player_inputs(
        Path(__file__).resolve().parents[2] / "sample-videos" / "04-china-open-ws-a.mp4")
    return analyze(unknown), analyze(declared)


# Words that would turn a marked arm swing into a hit, and phrasing that would make a reading a personal timing fault.
HIT_WORDS = re.compile(r"confirmed hit|contact point|at contact|at the hit|when you hit", re.IGNORECASE)
TIMING_FAULT = re.compile(r"\b(?:your|their) (?:split step|first step|hop) (?:was|is|came) (?:too )?(?:late|early|slow)",
                          re.IGNORECASE)
CLIP_INTENTS = ("summary", "explain", "simplify", "next", "how_to_fix", "practice_plan", "show_moment",
                "compare_players", "metric_question", "progress")


@unittest.skipUnless(REAL_DOC.exists(), "the real 01-r1 analysis output is not on this machine")
class RealReportTest(unittest.TestCase):
    """The real clip marks swings from arm motion only, so replies must not turn its readings into hits or faults."""

    @classmethod
    def setUpClass(cls):
        cls.unknown_report, cls.report = real_hand_reports()
        cls.unknown_coach = Coach(cls.unknown_report, LIBRARY, KB)
        cls.coach = Coach(cls.report, LIBRARY, KB)

    def ask(self, text, **context):
        reply, ctx = self.coach.reply([], text, {"player": "near", **context})
        assert_sane(self, self.coach, reply)
        return reply, context_to_dict(ctx)

    def test_nothing_clear_offers_a_plain_reading(self):
        reply, ctx = self.ask("what should I work on first?")
        self.assertTrue(reply.text.startswith("I found nothing that clearly needs work for you in this clip. "
                                             "One reading worth a look:"), reply.text)
        self.assertIn(TIMED_NOTE, reply.text)
        self.assertEqual(ctx["last_offer"], "how_to_fix:footwork.recovery")
        self.assertEqual(reply.actions, (Action(ActionKind.FINDING, "footwork.recovery", 0),))
        self.assertEqual(reply.suggestions, ("What exercises help?", "Show me the moment", "What else?"))
        unknown, _ = self.unknown_coach.reply([], "what should I work on first?", {"player": "near"})
        self.assertEqual(unknown.text, "I found nothing that clearly needs work for you in this clip.")
        self.assertEqual(unknown.actions, ())

    def test_split_step_timing_is_not_judged(self):
        reply, _ = self.ask("is my split step late?")
        self.assertTrue(reply.text.startswith(NO_TIMING), reply.text)
        self.assertIn("marked arm swings", reply.text)
        self.assertEqual(reply.actions, (Action(ActionKind.FINDING, "footwork.split_step", 0),))
        self.assertNotRegex(reply.text, TIMING_FAULT)
        self.assertTrue(reply.text.endswith(SOFT_ASK), reply.text)
        unknown, _ = self.unknown_coach.reply([], "is my split step late?", {"player": "near"})
        self.assertIn("I have no reading for the split step", unknown.text)
        self.assertEqual(unknown.actions, ())

    def test_first_step_number_is_not_a_timing_fault(self):
        reply, _ = self.ask("what was the number?", finding_id="footwork.first_step")
        self.assertTrue(reply.text.startswith("Your delay before the first step was "), reply.text)
        self.assertIn(TIMED_NOTE, reply.text)
        self.assertNotRegex(reply.text, TIMING_FAULT)
        self.assertEqual(reply.actions, (Action(ActionKind.FINDING, "footwork.first_step", 0),))
        synthetic, _ = COACH.reply([], "what was the number?", {"player": "near", "finding_id": "footwork.first_step"})
        self.assertIn("0.29 seconds over 12 readings", synthetic.text)

    def test_hand_speed_is_not_a_contact_score(self):
        reply, _ = self.ask("how fast was my hand?")
        self.assertTrue(reply.text.startswith("Your projected racket-hand speed was "), reply.text)
        self.assertIn(CONTACT_NOTE, reply.text)
        self.assertEqual(reply.actions, (Action(ActionKind.FINDING, "contact.hand_speed", 0),))
        unknown, _ = self.unknown_coach.reply([], "how fast was my hand?", {"player": "near"})
        self.assertIn("I have no reading for racket-hand speed", unknown.text)
        self.assertEqual(unknown.actions, ())
        self.assertNotRegex(unknown.text, NUMBER)
        synthetic, _ = COACH.reply([], "how fast was my hand?", {"player": "near"})
        self.assertIn("7.4 body heights a second over 9 readings", synthetic.text)

    def test_fix_for_a_plain_reading_is_general_practice(self):
        reply, _ = self.ask("how do I fix this?", finding_id="footwork.split_step")
        self.assertTrue(reply.text.startswith("This clip does not show a clear fault in the split step, so this is "
                                              "general practice: "), reply.text)
        self.assertIn(ex("split_step_hops"), reply.actions)
        self.assertNotIn("How do I fix it?", reply.suggestions)
        unknown, _ = self.unknown_coach.reply([], "how do I fix this?",
                                            {"player": "near", "finding_id": "footwork.split_step"})
        self.assertIn("I have no reading for the split step", unknown.text)
        self.assertEqual(unknown.actions, ())

    def test_plans_follow_only_findings_that_need_work(self):
        for coach in (self.coach, self.unknown_coach):
            for role in ("near", "far"):
                plan = build_plan(coach.report, LIBRARY, player=role, weeks=2)
                self.assertEqual((plan.findings, plan.weeks), ((), ()))
                reply, _ = coach.reply([], "make me a practice plan", {"player": role})
                self.assertTrue(reply.text.startswith("I found nothing that clearly needs work for you in this clip, "
                                                     "so there is nothing to plan yet."), reply.text)
                self.assertFalse(any(a.kind == ActionKind.EXERCISE for a in reply.actions))
        synthetic = build_plan(REPORT, LIBRARY, player="near", weeks=2)
        self.assertIn("technique.stay_lower", synthetic.findings)
        self.assertEqual(len(synthetic.weeks), 2)

    def test_is_that_good_keeps_a_plain_reading_plain(self):
        reply, _ = self.ask("is that good?", finding_id="footwork.first_step")
        self.assertEqual(reply.intent, "metric_question")
        self.assertIn("The analysis keeps it as a plain reading.", reply.text)
        self.assertIn(TIMED_NOTE, reply.text)
        reply, _ = self.ask("is that good?", finding_id="footwork.split_step")
        self.assertIn("The analysis keeps it as a plain reading.", reply.text)
        self.assertIn(SWING_NOTE, reply.text)
        reply, _ = self.ask("is that good?", finding_id="tactics.base_discipline")
        self.assertIn(DESCRIPTIVE_NOTE, reply.text)
        for role in ("near", "far"):
            reply, _ = self.ask("is that good?", player=role, finding_id="technique.knee_bend")
            self.assertIn("a coaching aim rather than a tested norm for you", reply.text)
            self.assertIn("plain reading, not a fault or a strength", reply.text)
            self.assertNotRegex(reply.text, r"too straight|too high|worth improving")
        for f in self.report.findings:
            reply, _ = self.ask("is that good?", player=f.player.value, finding_id=f.id)
            self.assertEqual(reply.intent, "metric_question", f.id)
            self.assertNotRegex(reply.text, TIMING_FAULT)
            if f.severity == Severity.INFO:
                rest = re.sub(r"not a (?:timing )?fault(?: or a strength)?", "", reply.text)
                self.assertNotRegex(rest, r"fault|strength|top thing|worth improving", reply.text)

    def test_compare_counts_only_clear_findings(self):
        reply, _ = self.ask("how do I compare with the other player?")
        self.assertTrue(reply.text.startswith("You have nothing clear to work on"), reply.text)
        self.assertIn("The other player has nothing clear to work on and no clear strengths.", reply.text)
        self.assertEqual(reply.actions, ())
        for finding in self.report.findings:
            if finding.id.startswith("technique."):
                self.assertEqual(finding.severity, Severity.INFO)
                self.assertEqual(finding.fix_tags, ())
        unknown, _ = self.unknown_coach.reply([], "how do I compare with the other player?", {"player": "near"})
        self.assertEqual(unknown.text, "You have nothing clear to work on and no clear strengths. The other player has "
                                      "nothing clear to work on and no clear strengths. Want to hear what to work on first?")

    def test_progress_never_compares_an_unavailable_reading(self):
        allowed = allowed_numbers([self.report])
        held = {(f.id, f.player) for f in self.report.findings if not measured(f.metrics[0])}
        self.assertTrue(held)
        for f in self.report.findings:
            ctx = context_from_dict({"player": f.player.value, "finding_id": f.id}, (self.report,))
            reply, _ = self.coach.reply([], "is it better than last time?", ctx)
            assert_sane(self, self.coach, reply)
            self.assertEqual(reply.intent, "progress")
            self.assertEqual(set(NUMBER.findall(reply.text)) - allowed, set(), reply.text)
            if (f.id, f.player) in held:
                self.assertRegex(reply.text, r"^I cannot say whether ", reply.text)
        ctx = context_from_dict({"player": "near", "finding_id": "contact.reach_height"}, (self.report,))
        reply, _ = self.coach.reply([], "is it better than last time?", ctx)
        self.assertEqual(reply.text, "I cannot say whether your projected overhead reach changed: I have no reading of it "
                                     "in either clip. Want to hear what to work on first?")

    def test_unknown_hand_focus_does_not_fall_back_to_another_finding(self):
        labels = {"contact.hand_speed": "racket-hand speed", "footwork.first_step": "the first step",
                  "footwork.split_step": "the split step"}
        for finding_id, label in labels.items():
            for question in ("what was the number?", "is that good?", "how do I fix this?", "show me the moment",
                             "is it better than last time?"):
                with self.subTest(finding=finding_id, question=question):
                    context = context_from_dict({"player": "near", "finding_id": finding_id},
                                                (self.unknown_report,))
                    reply, _ = self.unknown_coach.reply([], question, context)
                    assert_sane(self, self.unknown_coach, reply)
                    self.assertIn(f"I have no reading for {label} for you in this clip", reply.text)
                    self.assertEqual(reply.actions, ())
                    self.assertNotRegex(reply.text, NUMBER)
        present, _ = self.coach.reply([], "what was the number?",
                                     {"player": "near", "finding_id": "contact.hand_speed"})
        self.assertEqual(present.actions, (Action(ActionKind.FINDING, "contact.hand_speed", 0),))
        self.assertIn("Your projected racket-hand speed was ", present.text)

    def test_far_player_timing_explanation_preserves_caveats_within_word_limit(self):
        reply, _ = self.ask("is my split step late?", player="far")
        self.assertTrue(reply.text.startswith("I cannot judge its timing"), reply.text)
        self.assertIn("Small far-player feet lower confidence.", reply.text)
        self.assertIn("not validated personal norms", reply.text)
        self.assertIn("The shuttle is not tracked.", reply.text)
        self.assertLessEqual(len(reply.text.split()), 85)

    def test_no_reply_turns_a_marked_swing_into_a_hit_or_a_fault(self):
        severity = {(f.id, f.player): f.severity for f in self.report.findings}
        unsure = {number(m.value, m.unit) for f in self.report.findings for m in f.metrics
                  if m.value is not None and not measured(m)}
        allowed = allowed_numbers([self.report])
        starts = [{"player": "near"}, {"player": "far"}]
        starts += [{"player": f.player.value, "finding_id": f.id} for f in self.report.findings]
        questions = ("what should I work on first?", "what does this mean?", "is my split step late?",
                     "is my first step slow?", "how do I fix this?", "show me my best one", "what was the number?",
                     "how do I compare with the other player on this?", "what else?", "make me a practice plan")
        checked = 0
        for start in starts:
            for q in questions:
                for reply, _ in talk(self.coach, [q, "say that more simply"], start):
                    assert_sane(self, self.coach, reply)
                    self.assertNotRegex(reply.text, TIMING_FAULT)
                    self.assertEqual(set(NUMBER.findall(reply.text)) - allowed, set(), reply.text)
                    for value in unsure:
                        self.assertNotIn(value, reply.text)
                    focus = {severity[(a.ref, self.report.players[a.arg])] for a in reply.actions
                             if a.kind == ActionKind.FINDING}
                    if focus == {Severity.INFO}:
                        self.assertNotIn("how to fix", reply.text, reply.text)
                        self.assertNotIn("How do I fix it?", reply.suggestions)
                    if reply.intent not in CLIP_INTENTS:
                        continue
                    # General practice may talk about contact; what the clip shows may not.
                    clip_part = re.split(r"In general, not from this clip:|general practice", reply.text)[0]
                    for m in HIT_WORDS.finditer(clip_part):
                        self.assertIn("not", clip_part[max(0, m.start() - 12):m.start()], reply.text)
                    checked += 1
        self.assertGreater(checked, 500)


NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?!\w)")
GROUNDING_POOL = (
    "hi", "what should I work on first?", "what am I doing well?", "yes", "no", "what else?", "how do I fix it?",
    "what exercises help?", "how do I do it?", "why that exercise?", "make it easier", "make it harder", "too hard",
    "give me another exercise", "I have no equipment", "I only have a band", "any equipment is fine",
    "I only have dumbbells", "I am a beginner", "I am advanced", "make me a 2 week plan", "show me the moment",
    "what's my number?", "is that good?", "why does that matter?", "explain it simpler",
    "how does the other player compare?", "what about the far player?", "am I improving?", "what is a split step?",
    "where should I stand?", "what can't you see?", "my knee hurts", "can I do burpees instead?",
    "how do I do a wall sit?", "a harder version of squat jumps", "how fast is my smash?", "thanks", "asdf", "",
)
EXERCISE_HANDLERS = ("exercise_detail", "exercise_why", "exercise_adjust")
MISSING_NAMES = tuple(row["name"] for row in KB.exercise_aliases["missing"].values())


def measured(m):
    """A reading chat may quote or compare: observed, with confidence behind it."""
    return m.value is not None and m.confidence > 0


def unavailable(report, finding_id, role, index=0):
    """The report with one reading of a finding unobserved, as the producer writes it: value None, no observations,
    no confidence and no best. A finding left with no reading at all becomes an INFO note, as the schema requires."""
    target = next(f for f in report.findings if (f.id, f.player) == (finding_id, role))
    name = target.metrics[index].name

    def blank(m):
        return replace(m, value=None, n_obs=0, confidence=0.0, best=None) if m.name == name else m

    metrics = tuple(blank(m) for m in target.metrics)
    change = {"metrics": metrics}
    if all(m.value is None for m in metrics):
        change.update(severity=Severity.INFO, score=0.0, moments=(), fix_tags=(),
                      headline=f"Not enough clear frames to read {metrics[0].label.lower()}",
                      explain="There were not enough clear frames to read this, so this is only a note.")
    return replace(report, findings=tuple(replace(f, **change) if f is target else f for f in report.findings),
                   metrics={r: tuple(blank(m) if r == role.value else m for m in ms) for r, ms in report.metrics.items()})


def unconfident(report, finding_id, role, value, n_obs=6):
    """The report with one finding's main reading observed but at confidence 0, which the schema allows. The finding
    keeps its severity, score and moments, so only the confidence rule decides what chat may say."""
    target = next(f for f in report.findings if (f.id, f.player) == (finding_id, role))
    name = target.metrics[0].name

    def low(m):
        return replace(m, value=value, n_obs=n_obs, confidence=0.0) if m.name == name else m

    return replace(report, findings=tuple(replace(f, metrics=tuple(low(m) for m in f.metrics)) if f is target else f
                                          for f in report.findings),
                   metrics={r: tuple(low(m) if r == role.value else m for m in ms) for r, ms in report.metrics.items()})


def allowed_numbers(reports):
    """Every number a reply may print: measured report values, guides, counts, finding and moment texts, and the
    library's doses and texts. An unavailable or confidence-0 value is not grounded, so it is not allowed."""
    found = set()
    for report in reports:
        for f in report.findings:
            found.update(NUMBER.findall(" ".join([f.headline, f.explain, *(mo.label for mo in f.moments)])))
            for m in f.metrics:
                if m.target:
                    found.update(NUMBER.findall(span(*m.target, m.unit)))
                if measured(m):
                    found.update(NUMBER.findall(number(m.value, m.unit)))
                    if m.n_obs >= 2:
                        found.add(str(m.n_obs))
    for e in LIBRARY.values():
        found.update(NUMBER.findall(" ".join([e.sets_reps, easier_dose(e.sets_reps), e.safety, e.why, *e.cues,
                                              *e.mistakes])))
    return found


def exercise_in_focus(named, before, history):
    """The exercise a message is about: named, in focus, offered, last shown, or first named in the coach's last turn."""
    for ex_id in (named, before.exercise_id, split_tag(before.last_offer)[1], *reversed(before.shown_exercises)):
        if ex_id in LIBRARY:
            return ex_id
    last = next((t.text.casefold() for t in reversed(history) if t.speaker == "coach"), "")
    named_in_text = sorted((last.find(e.name.casefold()), e.id) for e in LIBRARY.values() if e.name.casefold() in last)
    return named_in_text[0][1] if named_in_text else None


class GroundingTest(unittest.TestCase):
    """Generated conversations: every number comes from the report or the library, every exercise from the library,
    and every exercise shares a fix tag with the goal under discussion."""

    @classmethod
    def setUpClass(cls):
        earlier = earlier_report()
        preview = preview_report()
        unsure = unconfident(unconfident(REPORT, "technique.stay_lower", Role.NEAR, 0.47),
                             "footwork.first_step", Role.FAR, 0.27)
        unsure_before = unconfident(earlier, "footwork.split_step", Role.NEAR, 0.29)
        gone = unavailable(unavailable(unavailable(REPORT, "technique.stay_lower", Role.NEAR),
                                       "footwork.first_step", Role.FAR), "footwork.split_step", Role.NEAR, 1)
        gone_before = unavailable(earlier, "footwork.split_step", Role.NEAR)
        cls.setups = [("full", COACH, (), allowed_numbers([REPORT])),
                      ("full+history", COACH, (earlier,), allowed_numbers([REPORT, earlier])),
                      ("preview", Coach(preview, LIBRARY, KB), (), allowed_numbers([preview])),
                      ("unconfident+history", Coach(unsure, LIBRARY, KB), (unsure_before,),
                       allowed_numbers([unsure, unsure_before])),
                      ("unavailable+history", Coach(gone, LIBRARY, KB), (gone_before,),
                       allowed_numbers([gone, gone_before]))]
        if REAL_DOC.exists():
            from coach.analyze import analyze
            real = analyze(json.loads(REAL_DOC.read_text(encoding="utf-8")))
            cls.setups.append(("real-01", Coach(real, LIBRARY, KB), (), allowed_numbers([real])))

    def goal(self, coach, reply, before, after, parse, history):
        """The fix tags the reply's exercises must share."""
        tags = coach.classifier.finding_tags
        role = split_tag(after.topic)[2] or after.player or coach.report.players[0]
        if split_tag(after.topic)[0] == "plan":
            return {t for f in coach.report.findings if f.player == role for t in tags(f.id, role)}
        goal = set(tags(after.finding_id, role))
        kind, offered, _ = split_tag(before.last_offer)
        handler = OFFERS.get(kind) if reply.intent == "affirm" else reply.intent
        if parse.missing and reply.intent != "affirm":
            goal |= set(KB.exercise_aliases["missing"][parse.missing]["fix_tags"])
        if handler in EXERCISE_HANDLERS:
            # An exercise the player asked about directly keeps its own goal when it does not train the finding's.
            base = exercise_in_focus(offered if reply.intent == "affirm" else parse.exercise_id, before, history)
            if base and not goal & set(LIBRARY[base].fix_tags):
                goal |= set(LIBRARY[base].fix_tags)
        return goal

    def check(self, coach, allowed, text, before, reply, after, parse, history):
        said = f"{text!r} -> {reply.text!r}"
        assert_sane(self, coach, reply)
        self.assertEqual(set(NUMBER.findall(reply.text)) - allowed, set(), said)
        for name in MISSING_NAMES:
            if re.search(rf"\b{re.escape(name)}\b", reply.text, re.IGNORECASE):
                self.assertIn("I do not have", reply.text, said)
        refs = [a.ref for a in reply.actions if a.kind == ActionKind.EXERCISE]
        if refs:
            goal = self.goal(coach, reply, before, after, parse, history)
            for ref in refs:
                self.assertTrue(goal & set(LIBRARY[ref].fix_tags), f"{ref} shares no fix tag with {sorted(goal)}: {said}")
        return len(refs)

    def test_two_hundred_generated_conversations_stay_grounded(self):
        rng = random.Random(20261025)
        replies = exercises = 0
        for n in range(200):
            name, coach, history_reports, allowed = self.setups[n % len(self.setups)]
            start = {"player": rng.choice(["near", "far", None]), "level": rng.choice([1, 2, 3]),
                     "equipment": rng.choice([None, [], ["band"], ["wall"], ["dumbbell"], ["band", "mat", "wall"]])}
            ctx, history, last = context_from_dict(start, history_reports), [], None
            for _ in range(rng.randint(2, 7)):
                use_chip = last is not None and last.suggestions and rng.random() < 0.5
                text = rng.choice(last.suggestions) if use_chip else rng.choice(GROUNDING_POOL)
                reply, after, parse = coach.reply_traced(history, text, ctx)
                with self.subTest(setup=name, conversation=n, turn=len(history) // 2):
                    exercises += self.check(coach, allowed, text, ctx, reply, after, parse, history)
                history = [*history, Turn("user", text), Turn("coach", reply.text)]
                ctx = context_from_dict(json.loads(json.dumps(context_to_dict(after))), history_reports)
                last, replies = reply, replies + 1
        self.assertGreater(replies, 700)
        self.assertGreater(exercises, 200)

    def test_numbers_are_checked_against_a_closed_set(self):
        allowed = self.setups[0][3]
        reply, _ = COACH.reply([], "what's my hip height?", {"player": "near"})
        self.assertEqual(NUMBER.findall(reply.text), ["53", "14", "42", "48"])
        self.assertTrue({"53", "14", "42", "48"} <= allowed)
        self.assertFalse({"47", "50", "0.35", "99"} & allowed)
        self.assertEqual(NUMBER.findall("Use a 2D view: 3 x 30 seconds, 53% and 0.35 s."), ["3", "30", "53", "0.35"])

    def test_unconfident_and_unavailable_values_are_outside_the_closed_set(self):
        allowed = {name: numbers for name, _, _, numbers in self.setups}
        self.assertEqual({"47", "0.27", "29"} & allowed["unconfident+history"], set())
        self.assertEqual({"53", "0.41", "31"} & allowed["unavailable+history"], set())
        self.assertTrue({"53", "0.41", "31"} <= allowed["full+history"])


class MeasuredReadingTest(unittest.TestCase):
    """Progress compares only readings measured in both clips: an unavailable or confidence-0 reading in either clip
    is never quoted as a change, and the reply says which clip had no reading and which had one too unclear to use."""

    HELD = {("unconfident", "now"): (unconfident(REPORT, "technique.stay_lower", Role.NEAR, 0.47), earlier_report()),
            ("unconfident", "before"): (REPORT, unconfident(earlier_report(), "technique.stay_lower", Role.NEAR, 0.47)),
            ("unavailable", "now"): (unavailable(REPORT, "technique.stay_lower", Role.NEAR), earlier_report()),
            ("unavailable", "before"): (REPORT, unavailable(earlier_report(), "technique.stay_lower", Role.NEAR))}
    WHY = {("unconfident", "now"): "I saw it in this clip, but not clearly enough to compare.",
           ("unconfident", "before"): "I saw it in the earlier clip, but not clearly enough to compare.",
           ("unavailable", "now"): "I have no reading of it in this clip.",
           ("unavailable", "before"): "I have no reading of it in the earlier clip."}
    HELD_LEAD = "I cannot say whether your projected hip height while waiting changed: "

    def ask(self, report, earlier, text):
        coach = COACH if report is REPORT else Coach(report, LIBRARY, KB)
        reply, ctx = coach.reply([], text, context_from_dict({"player": "near"}, (earlier,)))
        assert_sane(self, coach, reply)
        return reply, context_to_dict(ctx)

    def test_progress_moves_past_a_held_reading_in_either_clip(self):
        trend = ("Your opponent hits with a split step went from 31% to 38%. Different clips and camera angles can change "
                 f"readings, so treat it as a rough trend. {SHUTTLE_NOTE}")
        for (kind, clip), (report, earlier) in self.HELD.items():
            with self.subTest(kind=kind, clip=clip):
                reply, ctx = self.ask(report, earlier, "am I improving?")
                # With no reading now, staying low is an unranked note, so the split step comes first and needs no lead.
                lead = "" if (kind, clip) == ("unavailable", "now") else f"{self.HELD_LEAD}{self.WHY[kind, clip]} "
                self.assertEqual(reply.text, lead + trend)
                self.assertNotIn("47", reply.text)
                self.assertEqual(reply.actions, (Action(ActionKind.FINDING, "footwork.split_step", 0),))
                self.assertEqual(ctx["finding_id"], "footwork.split_step")
        reply, _ = self.ask(REPORT, earlier_report(), "am I improving?")
        self.assertEqual(reply.text, "Your projected hip height while waiting went from 56% of body height to 53% of body "
                                     "height. That is closer to the practice guide of about 42% to 48% of body height. "
                                     "Different clips and camera angles can change readings, so treat it as a rough trend.")

    def test_a_named_reading_without_both_measurements_is_not_compared(self):
        for key, (report, earlier) in self.HELD.items():
            with self.subTest(kind=key[0], clip=key[1]):
                reply, ctx = self.ask(report, earlier, "is my hip height better than last time?")
                self.assertEqual((reply.intent, reply.actions), ("progress", ()))
                self.assertEqual(reply.text, f"{self.HELD_LEAD}{self.WHY[key]} Want to hear what to work on first?")
                self.assertEqual(reply.suggestions, ("What should I work on first?", "Make me a practice plan"))
                self.assertEqual((ctx["finding_id"], ctx["last_offer"]), ("technique.stay_lower", "summary"))
        reply, _ = self.ask(REPORT, earlier_report(), "is my hip height better than last time?")
        self.assertTrue(reply.text.startswith("Your projected hip height while waiting went from 56% of body height to "
                                              "53% of body height."), reply.text)

    def test_both_clips_without_a_clear_reading_name_each_clip(self):
        cases = (("unconfident", "unconfident", "I saw it in both clips, but not clearly enough to compare."),
                 ("unavailable", "unavailable", "I have no reading of it in either clip."),
                 ("unavailable", "unconfident", "I have no reading of it in this clip, and I saw it in the earlier clip, "
                                                "but not clearly enough to compare."),
                 ("unconfident", "unavailable", "I saw it in this clip, but not clearly enough to compare, and I have no "
                                                "reading of it in the earlier clip."))
        hold = {"unconfident": lambda r: unconfident(r, "technique.stay_lower", Role.NEAR, 0.47),
                "unavailable": lambda r: unavailable(r, "technique.stay_lower", Role.NEAR)}
        for now, before, why in cases:
            with self.subTest(now=now, before=before):
                reply, _ = self.ask(hold[now](REPORT), hold[before](earlier_report()),
                                    "is my hip height better than last time?")
                self.assertEqual(reply.text, f"{self.HELD_LEAD}{why} Want to hear what to work on first?")
                self.assertNotIn("47", reply.text)


class AvailabilityTest(unittest.TestCase):
    """A reading with no observations is not a number: chat says so and never quotes, compares, plays or plans it. An
    observed reading at confidence 0 is not quoted or compared either, but its moments still play."""

    SL, SS = "technique.stay_lower", "footwork.split_step"
    GONE = unavailable(REPORT, "technique.stay_lower", Role.NEAR)
    UNSURE = unconfident(REPORT, "technique.stay_lower", Role.NEAR, 0.47)
    MIXED = unavailable(REPORT, "footwork.split_step", Role.NEAR, 1)
    ABSENT = ("I have no reading for staying low for you in this clip, so I cannot say. Ask me what to work on first "
              "instead.")

    def ask(self, report, text, history=(), **context):
        coach = Coach(report, LIBRARY, KB)
        reply, ctx = coach.reply([], text, context_from_dict({"player": "near", **context}, history))
        assert_sane(self, coach, reply)
        return reply, context_to_dict(ctx)

    def test_direct_questions_say_there_is_no_reading(self):
        for text, context in (("what's my hip height?", {}), ("is that good?", {"finding_id": self.SL})):
            reply, ctx = self.ask(self.GONE, text, **context)
            self.assertEqual(reply.intent, "metric_question")
            self.assertEqual(reply.text, "I have no reading of your projected hip height while waiting in this clip. Not "
                                         f"enough clear frames to read hip height while waiting. {SOFT_ASK}")
            self.assertEqual(reply.actions, (Action(ActionKind.FINDING, self.SL, 0),))
            self.assertEqual(reply.suggestions, ("What exercises help?", "What can't you measure?"))
            self.assertEqual(ctx["last_offer"], "how_to_fix:technique.stay_lower")

    def test_direct_questions_say_an_unclear_reading_was_seen_but_give_no_number(self):
        for text, context in (("what's my hip height?", {}), ("is that good?", {"finding_id": self.SL}),
                              ("what was the number?", {"finding_id": self.SL})):
            with self.subTest(text=text):
                reply, ctx = self.ask(self.UNSURE, text, **context)
                self.assertEqual(reply.intent, "metric_question")
                self.assertEqual(reply.text, "I saw your projected hip height while waiting in this clip, but not clearly "
                                             "enough to give a number. Want to know how to fix it?")
                self.assertNotIn("47", reply.text)
                self.assertEqual(reply.actions, (Action(ActionKind.FINDING, self.SL, 0),))
                # Its moments are real, so the moment chip stays.
                self.assertEqual(reply.suggestions, ("How do I fix it?", "Show me the moment", "What can't you measure?"))
                self.assertEqual(ctx["last_offer"], "how_to_fix:technique.stay_lower")

    def test_general_practice_says_whether_the_reading_was_missing_unclear_or_clear(self):
        step = "footwork.step_pattern"
        cue1 = "Use a few larger steps to cover the distance, then lunge. "
        cue2 = "Start the lunge once you are close enough to reach. "
        forward, lateral = "the forward lunge with reach (3 x 6 each leg)", "the lateral lunge (3 x 6 each side)"
        tail = ". The shuttle is not tracked, so this comes from movement and timing only. Want to know how to do the first one?"
        cases = ((REPORT, "This clip does not show a clear fault in step pattern, so this is general practice: "
                  + cue1 + cue2 + f"Gym exercises: {forward} and {lateral}" + tail, "forward_lunge_reach"),
                 # The longer lead leaves room for one cue: with both, the reply would be 87 words, over the limit.
                 (unconfident(REPORT, step, Role.NEAR, 4.0), "I saw your count of steps before a lunge in this clip, "
                  "but not clearly enough to judge, so this is general practice: " + cue1
                  + f"Gym exercises: {forward} and {lateral}" + tail, "forward_lunge_reach"),
                 # With no reading, the report gives no fix tags, so the guide's own order picks the exercises.
                 (unavailable(REPORT, step, Role.NEAR), "This clip has no reading for step pattern, so this is general "
                  "practice: " + cue1 + cue2 + f"Gym exercises: {lateral} and {forward}" + tail, "lateral_lunge"))
        for report, expected, first in cases:
            with self.subTest(expected=expected[:60]):
                reply, ctx = self.ask(report, "how do I fix it?", finding_id=step)
                self.assertEqual(reply.text, expected)
                self.assertLessEqual(len(reply.text.split()), WORD_LIMIT)
                self.assertEqual(ctx["last_offer"], f"exercise_detail:{first}")
                # The step pattern has no moment in any of these reports, so none is offered.
                self.assertEqual(reply.suggestions, CHIPS_KEEP)

    def test_an_unavailable_reading_has_no_moment_to_play(self):
        _, (shown, ctx) = talk(Coach(self.GONE, LIBRARY, KB), ["what's my hip height?", "Show me the moment"],
                               {"player": "near"})
        self.assertEqual((shown.intent, shown.text, shown.actions), ("show_moment", self.ABSENT, ()))
        self.assertEqual(shown.suggestions, ("What should I work on first?", "What did I do well?",
                                             "What can't you measure?"))
        self.assertEqual(ctx["last_offer"], "summary")
        best, _ = self.ask(self.GONE, "show me my best one", finding_id=self.SL)
        self.assertEqual(best.text, "I have no standout good moment for staying low in this clip.")
        self.assertNotIn(ActionKind.SEEK, {a.kind for a in best.actions})
        played, _ = self.ask(self.UNSURE, "show me the moment", finding_id=self.SL)
        self.assertEqual(played.text, "Playing it now: hips high while waiting. It is one of two moments like this, so ask "
                                      "again to see the next. Want to know how to fix it?")
        self.assertEqual(played.actions, (Action(ActionKind.FINDING, self.SL, 0), Action(ActionKind.SEEK, self.SL, 0)))

    def test_fix_and_why_are_general_practice(self):
        reply, _ = self.ask(self.GONE, "how do I fix it?", finding_id=self.SL)
        self.assertTrue(reply.text.startswith("This clip has no reading for staying low, so this is general practice: "),
                        reply.text)
        self.assertEqual([a.ref for a in reply.actions if a.kind == ActionKind.EXERCISE], ["sit_low_hold", "wall_sit"])
        reply, _ = self.ask(self.GONE, "why does that matter?", finding_id=self.SL)
        self.assertTrue(reply.text.startswith("There were not enough clear frames to read this, so this is only a note. "
                                              "In general, not from this clip: "), reply.text)

    def test_compare_declines_without_both_readings(self):
        both = unavailable(self.GONE, self.SL, Role.FAR)
        cases = ((self.GONE, "near", "you", (0, 1)), (self.GONE, "far", "the other player", (1, 0)),
                 (both, "near", "either of you", (0, 1)))
        for report, player, whom, order in cases:
            with self.subTest(player=player, whom=whom):
                reply, ctx = self.ask(report, "how do I compare with the other player?", player=player,
                                      finding_id=self.SL)
                self.assertEqual(reply.intent, "compare_players")
                self.assertEqual(reply.text, f"I cannot compare staying low, because I have no reading for {whom} in "
                                             "this clip. Want to hear what to work on first?")
                self.assertEqual(reply.actions, tuple(Action(ActionKind.FINDING, self.SL, i) for i in order))
                self.assertEqual(ctx["last_offer"], "summary")

    def test_compare_tells_an_unclear_reading_apart_from_a_missing_one(self):
        far_unsure = unconfident(REPORT, self.SL, Role.FAR, 0.44)
        label = "projected hip height while waiting"
        cases = ((self.UNSURE, "near", f"I saw your {label} in this clip, but not clearly enough to compare it with the "
                                       "other player's."),
                 (self.UNSURE, "far", f"I saw the other player's {label} in this clip, but not clearly enough to compare "
                                      "it with yours."),
                 (far_unsure, "near", f"I saw the other player's {label} in this clip, but not clearly enough to compare "
                                      "it with yours."),
                 (unconfident(self.UNSURE, self.SL, Role.FAR, 0.44), "near",
                  f"I saw your {label} and the other player's in this clip, but neither clearly enough to compare."),
                 (unavailable(self.UNSURE, self.SL, Role.FAR), "near",
                  f"I saw your {label} in this clip, but not clearly enough to compare, and I have no reading for the "
                  "other player."),
                 (unavailable(self.UNSURE, self.SL, Role.FAR), "far",
                  f"I saw the other player's {label} in this clip, but not clearly enough to compare, and I have no "
                  "reading for you."))
        for report, player, said in cases:
            with self.subTest(player=player, said=said):
                reply, ctx = self.ask(report, "how do I compare with the other player?", player=player,
                                      finding_id=self.SL)
                self.assertEqual(reply.intent, "compare_players")
                self.assertEqual(reply.text, said + " Want to hear what to work on first?")
                self.assertNotRegex(reply.text, r"4[47]")
                self.assertEqual({a.kind for a in reply.actions}, {ActionKind.FINDING})
                self.assertEqual(reply.suggestions, ("What should I work on first?", "What did I do well?",
                                                     "What can't you measure?"))
                self.assertEqual(ctx["last_offer"], "summary")

    def test_mixed_availability_quotes_only_the_observed_reading(self):
        landing = next(f for f in REPORT.findings if (f.id, f.player) == (self.SS, Role.NEAR)).metrics[1]
        quoted = number(landing.value, landing.unit)
        full, _ = self.ask(REPORT, "what's my split-step landing time?")
        self.assertIn(quoted, full.text)
        reply, _ = self.ask(self.MIXED, "what's my split-step landing time?")
        self.assertEqual(reply.text, "I have no reading of your split-step landing after their hit in this clip. No split "
                                     f"step seen at many of the opponent's hits. {SHUTTLE_NOTE} Want to know how to fix it?")
        reply, _ = self.ask(self.MIXED, "is that good?", finding_id=self.SS)
        self.assertEqual(reply.text, "Your opponent hits with a split step was 38% over 13 readings. The analysis ranks it "
                                     "as a top thing to work on. I am somewhat sure about this reading. "
                                     f"{SHUTTLE_NOTE} Want to know how to fix it?")
        reply, _ = self.ask(self.MIXED, "how do I compare with the other player?", finding_id=self.SS)
        self.assertTrue(reply.text.startswith("For the split step, your opponent hits with a split step was 38% and the "
                                              "other player's was 55%. "), reply.text)
        for text in ("what's my number?", "show me the moment", "why does that matter?", "how do I fix it?"):
            reply, _ = self.ask(self.MIXED, text, finding_id=self.SS)
            self.assertNotIn(quoted, reply.text)
        before = unavailable(earlier_report(), self.SS, Role.NEAR)
        reply, ctx = self.ask(self.MIXED, "am I improving?", (before,), finding_id=self.SS)
        self.assertEqual(reply.text, "I cannot say whether your opponent hits with a split step changed: I have no "
                                     "reading of it in the earlier clip. Your projected hip height while waiting went "
                                     "from 56% of body height to 53% of body height. That is closer to the practice guide "
                                     "of about 42% to 48% of body height. Different clips and camera angles can change "
                                     "readings, so treat it as a rough trend.")
        self.assertEqual(ctx["finding_id"], self.SL)

    def test_summary_and_plan_skip_an_unavailable_finding(self):
        self.assertEqual(build_plan(REPORT, LIBRARY, player="near", weeks=2).findings,
                         ("technique.stay_lower", "footwork.split_step", "footwork.recovery"))
        self.assertEqual(build_plan(self.GONE, LIBRARY, player="near", weeks=2).findings,
                         ("footwork.split_step", "footwork.recovery", "contact.arm_extension"))
        reply, _ = self.ask(self.GONE, "make me a practice plan")
        self.assertTrue(reply.text.startswith("Here is a two-week plan to work on the split step, getting back to base "
                                              "and arm extension, with three sessions a week."), reply.text)
        reply, _ = self.ask(self.GONE, "what should I work on first?")
        self.assertEqual(reply.actions, (Action(ActionKind.FINDING, self.SS, 0),))
        self.assertNotIn("Hips stay high", reply.text)


READY = Path(__file__).resolve().parent / "data" / "ready-stance"


@unittest.skipUnless((READY / "r1" / "report.json").exists(), "the saved ready-stance session is not on this machine")
class ReadyStanceTest(unittest.TestCase):
    """Two saved version 2 reports of one session. The later one wrote a hip-height placeholder with no observations
    for the near player, which loads as no reading; the far player's hip height is a real reading."""

    @classmethod
    def setUpClass(cls):
        cls.raw = {r: json.loads((READY / r / "report.json").read_text(encoding="utf-8")) for r in ("r0", "r1")}
        cls.before, cls.now = report_from_dict(cls.raw["r0"]), report_from_dict(cls.raw["r1"])
        cls.coach = Coach(cls.now, LIBRARY, KB)

    def ask(self, text, **context):
        reply, ctx = self.coach.reply([], text, context_from_dict(context, (self.before,)))
        assert_sane(self, self.coach, reply)
        return reply, context_to_dict(ctx)

    def test_the_reply_carrying_the_preview_note_keeps_its_question_within_the_word_limit(self):
        # Before this fix the note took the question's place and the reply ran to 88 words.
        reply, ctx = self.ask("is that good?", player="near", finding_id="footwork.first_step")
        self.assertEqual(reply.text, (
            "Your delay before the first step was 0.27 seconds. That is within the practice guide of 0.35 seconds or "
            "less, a coaching aim rather than a tested norm for you. I am not very sure about this reading. "
            + SHUTTLE_SHORT + " " + TIMED_NOTE + " " + PREVIEW_NOTE + " Want some practice ideas for it?"))
        self.assertEqual(len(reply.text.split()), 78)
        self.assertEqual(ctx["last_offer"], "how_to_fix:footwork.first_step")

    def test_a_preview_reply_with_no_room_for_a_question_keeps_its_caveats_and_offers_nothing(self):
        reply, ctx = self.ask("was my split step late?", player="far")
        self.assertEqual(reply.intent, "explain")
        # The report's "That mark is not a confirmed hit." is left out, as the opening sentence already says it.
        self.assertEqual(reply.text, (
            NO_TIMING + " Near 0 of 3 marked arm swings, a small hop showed up. Far feet are small, so this is less "
            "sure. Some shots need no hop, so this is not a fault. The stored range is not a timing to aim for. These "
            "numbers are practice guides, not validated personal norms. Not checked on club players. " + SHUTTLE_SHORT
            + " " + PREVIEW_NOTE))
        self.assertEqual(len(reply.text.split()), 84)
        self.assertIsNone(ctx["last_offer"])
        # Chips stay: the far split step has a real moment.
        self.assertEqual(reply.suggestions, ("What exercises help?", "Show me the moment", "What else?"))
        self.assertEqual(reply.actions, (Action(ActionKind.FINDING, "footwork.split_step", 1),))

    def test_a_saved_placeholder_loads_as_no_reading(self):
        saved = next(f for f in self.raw["r1"]["findings"] if (f["id"], f["player"]) == ("technique.stay_lower", "near"))
        self.assertEqual((self.raw["r1"]["schema_version"], saved["metrics"][0]["value"], saved["metrics"][0]["n_obs"]),
                         (2, 0.45, 0))
        loaded = next(f for f in self.now.findings if (f.id, f.player) == ("technique.stay_lower", Role.NEAR))
        m = loaded.metrics[0]
        self.assertEqual((m.value, m.n_obs, m.confidence, m.best), (None, 0, 0.0, None))
        self.assertEqual((loaded.severity, loaded.score, loaded.moments), (Severity.INFO, 0.0, ()))

    def test_progress_compares_only_the_player_with_a_reading(self):
        reply, ctx = self.ask("is my hip height better than last time?", player="near")
        self.assertEqual(reply.text, "I cannot say whether your projected hip height changed: I have no reading of it in "
                                     "this clip. Want to hear what to work on first?")
        self.assertEqual((reply.actions, ctx["last_offer"]), ((), "summary"))
        reply, _ = self.ask("is my hip height better than last time?", player="far")
        self.assertEqual(reply.text, "Your projected hip height went from 53% of body height to 52% of body height. That "
                                     "is closer to the practice guide of about 40% to 50% of body height. Different clips "
                                     f"and camera angles can change readings, so treat it as a rough trend. {PREVIEW_NOTE}")
        reply, _ = self.ask("am I improving?", player="near")
        self.assertEqual(reply.text, "Your projected lunge hip drop went from 11% to 10%. That is further from the practice "
                                     "guide of about 15% to 45%. Different clips and camera angles can change readings, so "
                                     f"treat it as a rough trend. {PREVIEW_NOTE}")

    def test_direct_questions_compare_and_moment_follow_each_players_reading(self):
        reply, _ = self.ask("what's my hip height?", player="near")
        self.assertEqual(reply.text, "I have no reading of your projected hip height in this clip. Not enough upright time "
                                     f"to judge hip height. {SOFT_ASK}")
        self.assertEqual(reply.suggestions, ("What exercises help?", "What can't you measure?"))
        reply, _ = self.ask("what's my hip height?", player="far")
        self.assertTrue(reply.text.startswith("Your projected hip height was 52% of body height over 2 readings."),
                        reply.text)
        self.assertEqual(reply.suggestions, ("What exercises help?", "Show me the moment", "What can't you measure?"))
        for player, whom in (("near", "you"), ("far", "the other player")):
            reply, _ = self.ask("how do I compare with the other player?", player=player,
                                finding_id="technique.stay_lower")
            self.assertEqual(reply.text, f"I cannot compare staying low, because I have no reading for {whom} in this "
                                         "clip. Want to hear what to work on first?")
        reply, _ = self.ask("show me the moment", player="near", finding_id="technique.stay_lower")
        self.assertEqual(reply.text, "I have no reading for staying low for you in this clip, so I cannot say. Ask me "
                                     "what to work on first instead.")

    def test_summary_and_plan_find_nothing_to_work_on(self):
        reply, _ = self.ask("what should I work on first?", player="near")
        self.assertEqual(reply.text, "I found nothing that clearly needs work for you in this clip. One reading worth a "
                                     "look: your lunges do not drop the hips much. Want some practice ideas for it?")
        reply, _ = self.ask("make me a practice plan", player="near")
        self.assertEqual(reply.text, "I found nothing that clearly needs work for you in this clip, so there is nothing "
                                     "to plan yet. One reading worth a look is lunge depth. Want some practice ideas for it?")
        for player in ("near", "far"):
            self.assertEqual(build_plan(self.now, LIBRARY, player=player, weeks=2).findings, ())

    def test_every_unavailable_finding_declines_and_numbers_stay_grounded(self):
        gone = [f for f in self.now.findings if all(m.value is None for m in f.metrics)]
        self.assertEqual(len(gone), 8)
        declines = (("what was the number?", "I have no reading of your "),
                    ("is it better than last time?", "I cannot say whether your "),
                    ("how do I compare with the other player on this?", "I cannot compare "),
                    ("show me the moment", "I have no reading for "))
        for f in gone:
            for text, start in declines:
                with self.subTest(finding=f.id, player=f.player.value, question=text):
                    reply, _ = self.ask(text, player=f.player.value, finding_id=f.id)
                    self.assertTrue(reply.text.startswith(start), reply.text)
        allowed = allowed_numbers([self.now, self.before])
        starts = [{"player": "near"}, {"player": "far"}]
        starts += [{"player": f.player.value, "finding_id": f.id} for f in self.now.findings]
        questions = ("what should I work on first?", "what was the number?", "is that good?", "am I improving?",
                     "how do I compare with the other player on this?", "show me the moment", "how do I fix this?",
                     "make me a practice plan")
        for start in starts:
            for q in questions:
                reply, _ = self.ask(q, **start)
                self.assertEqual(set(NUMBER.findall(reply.text)) - allowed, set(), reply.text)
                self.assertNotIn("45% of body height", reply.text)


SHOW_CHIPS = ("Show me the moment", "Show me my best one", "Show me their best one")
MOMENT_QUESTIONS = ("what should I work on first?", "what did I do well?", "why does that matter?", "explain it simply",
                    "what else?", "how do I fix it?", "what was the number?", "how do I compare with the other player?",
                    "am I improving?", "show me the moment", "show me my best one", "make me a practice plan",
                    "what tactics should I use?")


class MomentOfferTest(unittest.TestCase):
    """A reply offers to play a moment only when its finding has one: following an offered moment chip, or saying yes to
    an offer to show one, always plays a real moment, and nothing ever plays for a finding with no reading."""

    def test_moment_chips_follow_the_moments_each_finding_has(self):
        cases = ((("far", "technique.knee_bend"), "why does that matter?", ("What else?", "What should I work on first?")),
                 (("far", "technique.knee_bend"), "how do I fix it?", ("What should I work on first?", "What else?")),
                 (("far", "technique.knee_bend"), "show me my best one",
                  ("What should I work on first?", "What did I do well?")),
                 (("near", "technique.stance_width"), "how do I fix it?",
                  ("Show me my best one", "What should I work on first?", "What else?")),
                 (("near", "footwork.step_pattern"), "why does that matter?", ("What exercises help?", "What else?")),
                 (("near", "tactics.base_discipline"), "why does that matter?",
                  ("What exercises help?", "Show me the moment", "What else?")),
                 (("near", "tactics.tempo"), "what tactics should I use?",
                  ("What should I work on first?", "What exercises help?", "What can't you measure?")),
                 (("near", "tactics.base_discipline"), "what tactics should I use?",
                  ("Show me the moment", "What exercises help?", "What can't you measure?")))
        for (player, fid), text, chips in cases:
            with self.subTest(player=player, finding=fid, text=text):
                reply, _ = COACH.reply([], text, context_from_dict({"player": player, "finding_id": fid}))
                self.assertEqual(reply.suggestions, chips)
                self.assertEqual({a.kind for a in reply.actions}, {ActionKind.FINDING})
        # A strength with no best moment is not offered one, so "how do I fix it?" moves on instead.
        _, ctx = COACH.reply([], "how do I fix it?", context_from_dict({"player": "far", "finding_id": "technique.knee_bend"}))
        self.assertEqual(ctx.last_offer, "next")

    def sweep(self, report, before):
        """Ask every moment-related question from every start, follow each moment chip and moment offer, and return how
        many were followed."""
        coach = Coach(report, LIBRARY, KB)
        empty = {(f.id, f.player) for f in report.findings if all(m.value is None for m in f.metrics)}
        starts = [{"player": r.value} for r in report.players]
        starts += [{"player": f.player.value, "finding_id": f.id} for f in report.findings]
        followed = 0
        for start in starts:
            for q in MOMENT_QUESTIONS:
                reply, ctx = coach.reply([], q, context_from_dict(start, (before,)))
                self.assert_no_seek_for(empty, coach, reply)
                follow = [c for c in reply.suggestions if c in SHOW_CHIPS]
                follow += ["yes"] if split_tag(ctx.last_offer)[0] in ("show_moment", "show_best") else []
                for said in follow:
                    then, _ = coach.reply([Turn("user", q), Turn("coach", reply.text)], said, ctx)
                    followed += 1
                    with self.subTest(start=start, question=q, then=said):
                        self.assertIn(ActionKind.SEEK, {a.kind for a in then.actions}, then.text)
                        self.assert_no_seek_for(empty, coach, then)
        return followed

    def assert_no_seek_for(self, empty, coach, reply):
        focus = None
        for a in reply.actions:
            if a.kind == ActionKind.FINDING:
                focus = (a.ref, coach.report.players[a.arg])
            elif a.kind == ActionKind.SEEK:
                self.assertNotIn(focus, empty, reply.text)

    def test_every_offered_moment_plays_in_the_fixture_and_availability_reports(self):
        reports = {"fixture": REPORT, "unavailable": AvailabilityTest.GONE, "unconfident": AvailabilityTest.UNSURE,
                   "mixed": AvailabilityTest.MIXED, "far knee unavailable": unavailable(REPORT, "technique.knee_bend",
                                                                                        Role.FAR)}
        followed = {name: self.sweep(report, earlier_report()) for name, report in reports.items()}
        self.assertTrue(all(n > 100 for n in followed.values()), followed)

    @unittest.skipUnless(REAL_DOC.exists(), "the real 01-r1 analysis output is not on this machine")
    def test_every_offered_moment_plays_in_the_real_report(self):
        unknown, report = real_hand_reports()
        self.assertGreater(self.sweep(report, report), 100)
        self.sweep(unknown, unknown)
        coach = Coach(unknown, LIBRARY, KB)
        for finding in unknown.findings:
            if not finding.moments:
                continue
            with self.subTest(player=finding.player, finding=finding.id):
                reply, _ = coach.reply([], "show me the moment",
                                       {"player": finding.player.value, "finding_id": finding.id})
                self.assertEqual(reply.actions[0].ref, finding.id)
                self.assertIn(ActionKind.SEEK, {action.kind for action in reply.actions})
                assert_sane(self, coach, reply)
        absent, _ = coach.reply([], "show me the moment",
                                {"player": "near", "finding_id": "contact.hand_speed"})
        self.assertIn("I have no reading for racket-hand speed", absent.text)
        self.assertEqual(absent.actions, ())

    @unittest.skipUnless((READY / "r1" / "report.json").exists(), "the saved ready-stance session is not on this machine")
    def test_every_offered_moment_plays_in_the_ready_stance_reports(self):
        before, now = (report_from_dict(json.loads((READY / r / "report.json").read_text(encoding="utf-8")))
                       for r in ("r0", "r1"))
        self.assertGreater(self.sweep(now, before), 100)


class LatencyTest(unittest.TestCase):
    def test_construction_and_replies_are_fast(self):
        built = []
        for _ in range(5):
            started = time.perf_counter()
            Coach(REPORT, LIBRARY)
            built.append((time.perf_counter() - started) * 1000)
        self.assertLess(statistics.median(built), 300, built)
        coach = Coach(REPORT, LIBRARY)
        history = [Turn("user", "hi"), Turn("coach", "Hi! Ask me what to work on first.")]
        contexts = ({}, {"player": "near"}, {"player": "far", "level": 1, "equipment": []},
                    {"finding_id": "technique.stay_lower", "exercise_id": "wall_sit", "player": "near"})
        times = []
        for context in contexts:
            for text in GROUNDING_POOL:
                started = time.perf_counter()
                coach.reply(history, text, context)
                times.append((time.perf_counter() - started) * 1000)
        self.assertEqual(len(times), 4 * len(GROUNDING_POOL))
        self.assertLessEqual(chat_eval.percentile(times, 95), 200, sorted(times)[-5:])


EVAL_ITEMS = [
    {"id": "d1", "dialogue": "talk", "text": "what should I work on first?", "intent": "summary",
     "context": {"player": "near"}, "history": [], "entities": {"player": "near"}, "note": "never shown to the engine"},
    {"id": "d2", "dialogue": "talk", "text": "how do I fix it?", "intent": "how_to_fix", "context": {"player": "far"},
     "history": [], "entities": {}, "note": "runs on the context the engine produced for d1"},
    {"id": "s1", "text": "how do I do a wall sit?", "intent": "exercise_detail", "context": {"level": 1},
     "history": [{"speaker": "user", "text": "hi"}], "entities": {"exercise": "wall sit"}},
]


class Spy:
    """Records exactly what the evaluation passes to the engine."""

    def __init__(self, coach):
        self.coach, self.calls = coach, []

    def reply_traced(self, history, text, context):
        self.calls.append((json.loads(json.dumps(history)), text, json.loads(json.dumps(context))))
        return self.coach.reply_traced(history, text, context)


class EvalTest(unittest.TestCase):
    def test_rows_hold_what_the_coach_said(self):
        spy = Spy(COACH)
        rows = chat_eval.run(spy, EVAL_ITEMS)
        self.assertEqual([r["id"] for r in rows], ["d1", "d2", "s1"])
        for row in rows:
            self.assertEqual(tuple(row), chat_eval.ROW_KEYS)
            self.assertIsNone(row["error"])
            self.assertGreaterEqual(row["elapsed_ms"], 0)
        first, ctx = COACH.reply([], "what should I work on first?", {"player": "near"})
        self.assertEqual((rows[0]["intent"], rows[0]["text"], rows[0]["suggestions"]),
                         (first.intent, first.text, list(first.suggestions)))
        self.assertEqual(rows[0]["actions"], [{"kind": a.kind.value, "ref": a.ref, "arg": a.arg} for a in first.actions])
        # "I" names the selected player, here the report's first, as ENTITY_SOURCE declares.
        self.assertEqual(rows[2]["entities"], {"finding_id": None, "exercise": "wall_sit", "player": "near", "topic": None})
        # The engine gets text, context and history only; a dialogue's later turn gets what the engine produced.
        self.assertEqual(spy.calls, [
            ([], "what should I work on first?", {"player": "near"}),
            ([{"speaker": "user", "text": "what should I work on first?"}, {"speaker": "coach", "text": first.text}],
             "how do I fix it?", context_to_dict(ctx)),
            ([{"speaker": "user", "text": "hi"}], "how do I do a wall sit?", {"level": 1}),
        ])

    def test_a_failed_reply_keeps_its_row(self):
        class Broken:
            def reply_traced(self, history, text, context):
                raise RuntimeError("engine down")

        rows = chat_eval.run(Broken(), EVAL_ITEMS)
        self.assertEqual([r["id"] for r in rows], ["d1", "d2", "s1"])
        for row in rows:
            self.assertEqual(row["error"], "RuntimeError: engine down")
            self.assertEqual((row["intent"], row["text"], row["actions"], row["suggestions"]), (None, None, [], []))
            self.assertIsNotNone(row["elapsed_ms"])

    def test_export_hashes_the_exact_question_bytes(self):
        raw = json.dumps(EVAL_ITEMS, indent=2).replace("\n", "\r\n").encode("utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            questions, out = Path(tmp) / "questions.json", Path(tmp) / "obs" / "export.json"
            questions.write_bytes(raw)
            with redirect_stdout(io.StringIO()) as printed:
                code = chat_eval.main([str(questions), "--export", str(out)])
            doc = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(doc["question_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual((doc["format"], doc["split"], doc["report"]), ("coach-chat-observations/1", "all", "full"))
        self.assertEqual([r["id"] for r in doc["rows"]], ["d1", "d2", "s1"])
        self.assertEqual(doc["rows"][0]["text"], COACH.reply([], "what should I work on first?", {"player": "near"})[0].text)
        self.assertGreater(doc["construction_ms"], 0)
        self.assertEqual(doc["entity_source"], chat_eval.ENTITY_SOURCE)
        self.assertEqual(doc["source_sha256"], chat_eval.source_hashes())
        self.assertIn("intent: 3/3 (100.0%)", printed.getvalue())


class KbTest(unittest.TestCase):
    def test_kb_text_has_no_unearned_certainty(self):
        def strings(node):
            if isinstance(node, str):
                yield node
            elif isinstance(node, dict):
                for value in node.values():
                    yield from strings(value)
            elif isinstance(node, list):
                for value in node:
                    yield from strings(value)
        kb_dir = Path(__file__).resolve().parents[1] / "kb"
        texts = [t for path in sorted(kb_dir.glob("*.json")) for t in strings(json.loads(path.read_text("utf-8")))]
        self.assertGreater(len(texts), 500)
        for text in texts:
            self.assertIsNone(re.search(r"\b(you must|guaranteed|always works|will fix)\b", text, re.IGNORECASE), text)

    def test_kb_covers_the_real_library(self):
        check_kb_against_library(KB, LIBRARY)

    def test_kb_check_names_missing_aliases(self):
        with self.assertRaises(ValueError) as caught:
            check_kb_against_library(KB, stub_library())
        self.assertTrue(str(caught.exception).startswith("library ids without aliases: "))

    def test_engine_runs_on_a_small_library(self):
        coach = Coach(REPORT, stub_library(), KB)
        reply, ctx = coach.reply([], "how do I fix it?", {"finding_id": "technique.stay_lower", "player": "near"})
        self.assertEqual(reply.actions, (Action(ActionKind.FINDING, "technique.stay_lower", 0), ex("low_hold"),
                                         ex("low_squat")))
        self.assertIn("Gym exercises: the low hold (3 x 20 seconds) and the low squat (3 x 8).", reply.text)
        harder, ctx = coach.reply([], "make it harder", context_to_dict(ctx).copy() | {"exercise_id": "low_squat"})
        # low_squat's harder edge is jump_up, which does not train a lower stance, so the coach keeps low_squat.
        self.assertEqual(harder.actions, (ex("low_squat"),))


if __name__ == "__main__":
    unittest.main()
