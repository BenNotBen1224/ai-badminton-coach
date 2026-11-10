import dataclasses
import json
import unittest

from coach.chat.plan import EASED, NO_PLAN, REPEAT, SAFETY, WARM_UP, build_plan, easier_dose, fits, step
from coach.exercises.library import load_library
from coach.schema import FIX_TAGS, Role, Severity
from coach.tests.fixture_report import full_report, stub_library

REPORT = full_report()
LIBRARY = load_library()
STUB = stub_library()
GOAL_TAG = {words: tag for tag, words in FIX_TAGS.items()}


def ids(plan):
    return [[[b.exercise_id for b in s.blocks] for s in w.sessions] for w in plan.weeks]


class StubPlanTest(unittest.TestCase):
    def test_week_two_steps_up_but_never_leaves_the_goal(self):
        plan = build_plan(REPORT, STUB, player="near")
        self.assertEqual(plan.findings, ("technique.stay_lower", "footwork.split_step"))
        # low_squat's harder edge is jump_up, which drops the low-stance goal, so week two stops at low_squat.
        self.assertEqual(ids(plan), [
            [["low_hold", "hop_drill"], ["low_squat", "rope_skip"], ["low_hold", "hop_drill"]],
            [["low_squat", "rope_skip"], ["low_squat", "rope_skip"], ["low_squat", "rope_skip"]],
        ])
        self.assertEqual([w.focus for w in plan.weeks],
                         ["Start easy and learn the movements.", "Step up where the first week felt easy."])

    def test_no_equipment_keeps_bodyweight_and_repeats_week_one(self):
        plan = build_plan(REPORT, STUB, player="near", equipment=())
        self.assertEqual(plan.equipment, ())
        self.assertEqual(ids(plan), [[["low_hold", "hop_drill"]] * 3] * 2)
        self.assertEqual(plan.weeks[1].focus, "Repeat last week and aim for smoother, steadier reps.")

    def test_level_one_caps_difficulty(self):
        plan = build_plan(REPORT, STUB, player="near", level=1)
        self.assertEqual(plan.level, 1)
        self.assertEqual(ids(plan), [[["low_hold", "hop_drill"]] * 3] * 2)

    def test_to_dict_and_to_text(self):
        plan = build_plan(REPORT, STUB, player="near", weeks=1, sessions_per_week=1, equipment=())
        low = "Hips stay high while you wait for the shot - goal: sit lower with bent knees in the ready position"
        hop = "No split step seen at many of the opponent's hits - goal: time the small hop just before the opponent hits"
        expected = {
            "player": "near",
            "level": 3,
            "equipment": [],
            "findings": ["technique.stay_lower", "footwork.split_step"],
            "warm_up": WARM_UP,
            "weeks": [{"week": 1, "focus": "Start easy and learn the movements.", "sessions": [{"session": 1, "blocks": [
                {"exercise_id": "low_hold", "name": "Low hold", "dose": "3 x 20 seconds",
                 "finding_id": "technique.stay_lower", "reason": low},
                {"exercise_id": "hop_drill", "name": "Hop drill", "dose": "3 x 20 hops",
                 "finding_id": "footwork.split_step", "reason": hop},
            ]}]}],
            "safety": SAFETY,
        }
        self.assertEqual(plan.to_dict(), expected)
        self.assertEqual(json.loads(json.dumps(plan.to_dict(), allow_nan=False)), expected)
        self.assertEqual(plan.to_text(), "\n".join([
            "Practice plan for the near player",
            f"Warm-up: {WARM_UP}",
            "Week 1: Start easy and learn the movements.",
            "  Session 1:",
            f"    - Low hold, 3 x 20 seconds. {low}",
            f"    - Hop drill, 3 x 20 hops. {hop}",
            f"Safety: {SAFETY}",
        ]))
        self.assertEqual(plan.exercise_ids(), ("low_hold", "hop_drill"))

    def test_no_fitting_exercise_gives_an_empty_plan(self):
        plan = build_plan(REPORT, {}, player="near")
        self.assertEqual((plan.findings, plan.weeks), ((), ()))
        self.assertEqual(plan.to_dict()["weeks"], [])
        self.assertEqual(plan.to_text(), f"{NO_PLAN}\nSafety: {SAFETY}")
        self.assertEqual(NO_PLAN, "There is nothing to plan yet: either nothing in this clip clearly needs work for "
                                  "this player, or no exercise for it fits this level and equipment.")

    def test_step_rejects_a_variant_that_drops_the_goal(self):
        self.assertIsNone(step(STUB, STUB["low_squat"], "harder", "lower_stance"))
        self.assertEqual(step(STUB, STUB["low_squat"], "harder", "jump_power").id, "jump_up")
        self.assertEqual(step(STUB, STUB["low_squat"], "easier", "lower_stance").id, "low_hold")


class RealLibraryPlanTest(unittest.TestCase):
    def test_beginner_without_equipment(self):
        plan = build_plan(REPORT, LIBRARY, player="near", level=1, equipment=())
        self.assertEqual(plan.findings, ("technique.stay_lower", "footwork.split_step", "contact.arm_extension"))
        self.assertEqual(ids(plan), [[["sit_low_hold", "split_step_hops", "forward_lunge_reach"]] * 3] * 2)
        self.assertEqual(plan.weeks[1].focus, REPEAT)

    def test_every_block_fits_goal_level_and_gear(self):
        gear_options = (None, (), ("band",), ("wall", "mat"), ("dumbbell",), ("rope",))
        for player in (Role.NEAR, Role.FAR):
            top = {f.id: f for f in REPORT.top_findings(len(REPORT.findings), player)}
            for level in (1, 2, 3):
                for gear in gear_options:
                    with self.subTest(player=player.value, level=level, gear=gear):
                        plan = build_plan(REPORT, LIBRARY, player=player.value, level=level, equipment=gear)
                        self.assertTrue(plan.weeks, "every fixture setting has at least one fitting exercise")
                        self.assertEqual((len(plan.weeks), plan.warm_up, plan.safety), (2, WARM_UP, SAFETY))
                        self.assertLessEqual(len(plan.findings), 3)
                        self.assertTrue(set(plan.findings) <= set(top))
                        categories = [top[fid].category for fid in plan.findings]
                        self.assertTrue(all(categories.count(c) <= 2 for c in categories))
                        for week in plan.weeks:
                            self.assertEqual(len(week.sessions), 3)
                            for session in week.sessions:
                                used = [b.exercise_id for b in session.blocks]
                                self.assertEqual(len(used), len(set(used)))
                                for b in session.blocks:
                                    ex = LIBRARY[b.exercise_id]
                                    tag = GOAL_TAG[b.reason.rsplit(" - goal: ", 1)[1]]
                                    self.assertIn(b.finding_id, plan.findings)
                                    self.assertIn(tag, ex.fix_tags)
                                    self.assertIn(tag, top[b.finding_id].fix_tags)
                                    self.assertEqual((b.name, b.dose), (ex.name, ex.sets_reps))
                                    self.assertLessEqual(ex.difficulty, level)
                                    if gear is not None:
                                        self.assertTrue(set(ex.equipment) <= set(gear))

    def test_week_two_is_never_easier_for_a_goal(self):
        plan = build_plan(REPORT, LIBRARY, player="near")
        for one, two in zip(plan.weeks[0].sessions, plan.weeks[1].sessions):
            first = {b.finding_id: LIBRARY[b.exercise_id].difficulty for b in one.blocks}
            for b in two.blocks:
                if b.finding_id in first:
                    self.assertGreaterEqual(LIBRARY[b.exercise_id].difficulty, first[b.finding_id], b)

    def test_same_inputs_same_plan(self):
        self.assertEqual(build_plan(REPORT, LIBRARY, player="far", level=2),
                         build_plan(REPORT, LIBRARY, player="far", level=2))

    def test_player_none_means_first_player(self):
        self.assertEqual(build_plan(REPORT, LIBRARY, player=None).player, Role.NEAR)

    def test_step_keeps_the_low_stance_goal(self):
        self.assertIsNone(step(LIBRARY, LIBRARY["goblet_squat"], "harder", "lower_stance"))
        self.assertEqual(step(LIBRARY, LIBRARY["sit_low_hold"], "harder", "lower_stance").id, "goblet_squat")
        self.assertIsNone(step(LIBRARY, LIBRARY["sit_low_hold"], "harder", "lower_stance", 3, ()))
        self.assertEqual(step(LIBRARY, LIBRARY["sit_low_hold"], "easier", "lower_stance").id, "wall_sit")
        self.assertIsNone(step(LIBRARY, LIBRARY["sit_low_hold"], "easier", "lower_stance", 3, ()))
        self.assertTrue(fits(LIBRARY["sit_low_hold"], "lower_stance", 1, ()))
        self.assertFalse(fits(LIBRARY["wall_sit"], "lower_stance", 1, ()))
        self.assertFalse(fits(LIBRARY["goblet_squat"], "lower_stance", 1))
        self.assertFalse(fits(LIBRARY["squat_jumps"], "lower_stance"))

    def test_bad_inputs_raise_value_error(self):
        one_player = dataclasses.replace(
            REPORT, players=(Role.NEAR,), findings=tuple(f for f in REPORT.findings if f.player == Role.NEAR),
            metrics={"near": REPORT.metrics["near"]}, series={"near": REPORT.series["near"]},
            events=tuple(e for e in REPORT.events if e.player == Role.NEAR))
        cases = [
            ((REPORT, {"player": "middle"}), "unknown player 'middle'"),
            ((one_player, {"player": "far"}), "player far is not in this report"),
            ((REPORT, {"player": "near", "weeks": 0}), "weeks must be a whole number from 1 to 12, got 0"),
            ((REPORT, {"player": "near", "weeks": True}), "weeks must be a whole number from 1 to 12, got True"),
            ((REPORT, {"player": "near", "sessions_per_week": 8}),
             "sessions_per_week must be a whole number from 1 to 7, got 8"),
            ((REPORT, {"player": "near", "level": 4}), "level must be a whole number from 1 to 3, got 4"),
            ((REPORT, {"player": "near", "level": "2"}), "level must be a whole number from 1 to 3, got '2'"),
            ((REPORT, {"player": "near", "equipment": ["car"]}), "unknown equipment ['car']"),
            ((REPORT, {"player": "near", "equipment": 5}), "equipment must be a list of names, got 5"),
        ]
        for (report, kwargs), message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError) as caught:
                    build_plan(report, LIBRARY, **kwargs)
                self.assertEqual(str(caught.exception), message)
        with self.assertRaises(ValueError) as caught:
            step(LIBRARY, LIBRARY["wall_sit"], "sideways", "lower_stance")
        self.assertEqual(str(caught.exception), "step must be easier or harder, got 'sideways'")

    def test_single_equipment_name_is_accepted(self):
        self.assertEqual(build_plan(REPORT, LIBRARY, player="near", equipment="band").equipment, ("band",))

    def test_strengths_never_get_planned(self):
        for player in (Role.NEAR, Role.FAR):
            plan = build_plan(REPORT, LIBRARY, player=player.value)
            severities = {f.severity for f in REPORT.findings if f.player == player and f.id in plan.findings}
            self.assertTrue(severities <= {Severity.PRIORITY, Severity.IMPROVE})


def blocks_of(plan, exercise_id):
    return [b for w in plan.weeks for s in w.sessions for b in s.blocks if b.exercise_id == exercise_id]


class TooHardPlanTest(unittest.TestCase):
    """An exercise the player found too hard is replaced where another fits, otherwise kept only at an easier dose."""

    def test_easier_dose(self):
        cases = {"3 x 30 seconds": "2 x 15 seconds", "3 x 45 seconds": "2 x 20 seconds", "3 x 8": "2 x 4",
                 "3 x 6 each leg": "2 x 3 each leg", "3 x 20 hops": "2 x 10 hops", "3 x 12": "2 x 6",
                 "1 x 8 seconds": "1 x 5 seconds", "2 x 5 seconds": "1 x 5 seconds",
                 "1 x 5 seconds": "1 x 5 seconds, with shorter holds or fewer reps",
                 "1 x 1": "1 x 1, with shorter holds or fewer reps",
                 "3 x 8-10": "3 x 8-10, with shorter holds or fewer reps",
                 "until tired": "until tired, with shorter holds or fewer reps"}
        for dose, eased in cases.items():
            with self.subTest(dose=dose):
                self.assertEqual(easier_dose(dose), eased)
        for ex in LIBRARY.values():
            with self.subTest(exercise=ex.id):
                self.assertNotEqual(easier_dose(ex.sets_reps), ex.sets_reps)

    def test_replaced_where_another_exercise_fits(self):
        for level, gear in ((1, None), (3, None), (2, ("wall",))):
            with self.subTest(level=level, gear=gear):
                self.assertIn("sit_low_hold", build_plan(REPORT, LIBRARY, player="near", level=level,
                                                         equipment=gear).exercise_ids())
                plan = build_plan(REPORT, LIBRARY, player="near", level=level, equipment=gear, avoid=["sit_low_hold"])
                self.assertNotIn("sit_low_hold", plan.exercise_ids())
                self.assertIn("wall_sit", plan.exercise_ids())
                for w in plan.weeks:
                    for s in w.sessions:
                        for b in s.blocks:
                            self.assertEqual(b.dose, LIBRARY[b.exercise_id].sets_reps)
                            self.assertFalse(b.reason.endswith(EASED))

    def test_kept_only_at_an_easier_dose_when_nothing_else_fits(self):
        plain = build_plan(REPORT, LIBRARY, player="near", level=1, equipment=["band"])
        plan = build_plan(REPORT, LIBRARY, player="near", level=1, equipment=["band"], avoid=["sit_low_hold"])
        self.assertEqual(ids(plan), ids(plain))
        eased = blocks_of(plan, "sit_low_hold")
        self.assertEqual(len(eased), 6)
        self.assertEqual({(b.dose, b.reason) for b in eased}, {(
            "2 x 15 seconds", "Hips stay high while you wait for the shot - goal: sit lower with bent knees in the "
                              "ready position - easier dose, because the full one was too hard")})
        self.assertEqual({b.dose for b in blocks_of(plain, "sit_low_hold")}, {"3 x 30 seconds"})

    def test_week_two_never_steps_onto_an_avoided_exercise(self):
        # wall_sit's harder edge is sit_low_hold; with it avoided, week two goes on to the goblet squat.
        plan = build_plan(REPORT, LIBRARY, player="near", avoid=["sit_low_hold"])
        self.assertEqual([[b.exercise_id for b in s.blocks if b.finding_id == "technique.stay_lower"]
                          for w in plan.weeks for s in w.sessions],
                         [["wall_sit"], ["goblet_squat"], ["wall_sit"], ["goblet_squat"], ["goblet_squat"],
                          ["goblet_squat"]])

    def test_step_skips_avoided_variants(self):
        wall, goblet = LIBRARY["wall_sit"], LIBRARY["goblet_squat"]
        self.assertEqual(step(LIBRARY, wall, "harder", "lower_stance").id, "sit_low_hold")
        self.assertEqual(step(LIBRARY, wall, "harder", "lower_stance", avoid=("sit_low_hold",)).id, "goblet_squat")
        self.assertEqual(step(LIBRARY, goblet, "easier", "lower_stance").id, "sit_low_hold")
        self.assertEqual(step(LIBRARY, goblet, "easier", "lower_stance", avoid=("sit_low_hold",)).id, "wall_sit")
        self.assertIsNone(step(LIBRARY, goblet, "easier", "lower_stance", avoid=("sit_low_hold", "wall_sit")))

    def test_avoid_inputs(self):
        args = dict(player="near", level=1, equipment=["band"])
        plain = build_plan(REPORT, LIBRARY, **args)
        self.assertEqual(build_plan(REPORT, LIBRARY, avoid=None, **args), plain)
        self.assertEqual(build_plan(REPORT, LIBRARY, avoid=["no_such_exercise"], **args), plain)
        self.assertEqual(build_plan(REPORT, LIBRARY, avoid="sit_low_hold", **args),
                         build_plan(REPORT, LIBRARY, avoid=("sit_low_hold",), **args))
        for bad, message in ((5, "avoid must be a list of exercise ids, got 5"),
                             ([1], "avoid must be a list of exercise ids, got [1]"),
                             ([["a"]], "avoid must be a list of exercise ids, got [['a']]")):
            with self.subTest(avoid=bad):
                with self.assertRaises(ValueError) as caught:
                    build_plan(REPORT, LIBRARY, avoid=bad, **args)
                self.assertEqual(str(caught.exception), message)


if __name__ == "__main__":
    unittest.main()
