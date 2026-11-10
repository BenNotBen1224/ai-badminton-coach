import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from coach import schema as S
from coach.analyze import assemble
from coach.tracks import PlayerTrack, Tracks


def metric(**kw):
    base = dict(
        name="hip_height_ratio", label="Hip height", value=0.52, unit="ratio",
        basis=S.Basis.PROJECTED_2D, confidence=0.8, target=(0.4, 0.5), series="hip_height_ratio",
    )
    base.update(kw)
    return S.Metric(**base)


def moment(player=S.Role.NEAR, start=2, key=5, end=8):
    return S.Moment(player, start, key, end, "ready stance")


def finding(**kw):
    base = dict(
        id="technique.stay_lower", category=S.Category.TECHNIQUE, player=S.Role.NEAR,
        severity=S.Severity.IMPROVE, headline="Sit lower in the ready stance",
        explain="Your hips stay high between shots.", metrics=(metric(),), moments=(moment(),),
        tier=S.Tier.PREVIEW, focus_joints=("left_hip", "right_hip"),
        overlay_series=("hip_height_ratio",), fix_tags=("lower_stance",),
    )
    base.update(kw)
    return S.Finding(**base)


def report(**kw):
    n = 12
    base = dict(
        schema_version=S.SCHEMA_VERSION, source={"file": "x.mp4"}, tier=S.Tier.PREVIEW,
        n_frames=n, fps=30.0, players=(S.Role.NEAR, S.Role.FAR),
        metrics={"near": (metric(),)},
        series={"near": {"hip_height_ratio": tuple([0.5] * n)}},
        events=(S.swing_event(S.Role.NEAR, 4, start=2, end=6, peak_frame=3, peak_speed=2.0, overhead=True),),
        findings=(finding(),), windows=(S.Window(2, 8, "swing"),),
    )
    base.update(kw)
    return S.Report(**base)


class RoundTrip(unittest.TestCase):
    def test_json_round_trip_gives_an_equal_report(self):
        original = report()
        wire = json.loads(json.dumps(original.to_dict()))
        self.assertEqual(S.report_from_dict(wire), original)

    def test_series_gaps_survive_the_round_trip(self):
        values = tuple([None, 0.5] * 6)
        original = report(series={"near": {"hip_height_ratio": values}})
        wire = json.loads(json.dumps(original.to_dict()))
        self.assertEqual(S.report_from_dict(wire).series["near"]["hip_height_ratio"], values)


class Rejections(unittest.TestCase):
    def assertRejected(self, fn, fragment):
        with self.assertRaises(ValueError) as caught:
            fn()
        self.assertIn(fragment, str(caught.exception))

    def test_unknown_finding_id(self):
        self.assertRejected(lambda: finding(id="technique.made_up"), "not in FINDING_CATALOG")

    def test_category_must_match_catalog(self):
        self.assertRejected(lambda: finding(category=S.Category.TACTICS), "belongs to category technique")

    def test_improve_needs_a_fix_tag(self):
        self.assertRejected(lambda: finding(fix_tags=()), "needs a fix tag")

    def test_improve_needs_a_moment(self):
        self.assertRejected(lambda: finding(moments=()), "needs a moment")

    def test_strength_needs_neither_moment_nor_fix_tag(self):
        self.assertEqual(finding(severity=S.Severity.STRENGTH, fix_tags=(), moments=()).severity, S.Severity.STRENGTH)

    def test_unknown_fix_tag(self):
        self.assertRejected(lambda: finding(fix_tags=("lower_stance", "stronger_everything")), "unknown fix tags")

    def test_unknown_joint(self):
        self.assertRejected(lambda: finding(focus_joints=("left_hip", "tail")), "unknown joints")

    def test_headline_length(self):
        self.assertRejected(lambda: finding(headline="x" * 91), "headline")

    def test_nan_metric(self):
        self.assertRejected(lambda: metric(value=float("nan")), "finite number")

    def test_metric_confidence_range(self):
        self.assertRejected(lambda: metric(confidence=1.2), "confidence")

    def test_moment_order(self):
        self.assertRejected(lambda: S.Moment(S.Role.NEAR, 5, 4, 9, "x"), "start <= key <= end")

    def test_finding_moment_must_belong_to_the_same_player(self):
        self.assertRejected(lambda: finding(moments=(moment(player=S.Role.FAR),)), "another player")

    def test_series_length_must_match_frames(self):
        self.assertRejected(lambda: report(series={"near": {"hip_height_ratio": (0.5,) * 3}}), "expected 12")

    def test_series_value_must_be_finite(self):
        values = (0.5,) * 11 + (float("inf"),)
        self.assertRejected(lambda: report(series={"near": {"hip_height_ratio": values}}), "non-finite")

    def test_duplicate_finding(self):
        self.assertRejected(lambda: report(findings=(finding(), finding())), "duplicate finding")

    def test_event_outside_report(self):
        late = S.swing_event(S.Role.NEAR, 12, start=10, end=13, peak_frame=11, peak_speed=2.0, overhead=True)
        self.assertRejected(lambda: report(events=(late,)), "outside the report")

    def test_moment_past_the_last_frame(self):
        self.assertRejected(lambda: report(findings=(finding(moments=(moment(end=12),)),)), "past the last frame")

    def test_overlay_series_must_exist(self):
        self.assertRejected(lambda: report(findings=(finding(overlay_series=("missing_series",)),)), "unknown series")

    def test_finding_player_must_be_present(self):
        self.assertRejected(
            lambda: report(players=(S.Role.NEAR,), findings=(finding(player=S.Role.FAR, moments=(moment(player=S.Role.FAR),)),)),
            "not in the report",
        )

    def test_unknown_event_kind(self):
        self.assertRejected(lambda: S.Event("dance", S.Role.NEAR, 1, 0.5), "unknown event kind")


class Ranking(unittest.TestCase):
    def test_priority_beats_improve_and_strengths_are_excluded(self):
        improve = finding()
        priority = finding(id="technique.knee_bend", severity=S.Severity.PRIORITY, fix_tags=("lower_stance",))
        strength = finding(id="contact.arm_extension", category=S.Category.CONTACT, severity=S.Severity.STRENGTH, fix_tags=(), moments=())
        rep = report(findings=(improve, priority, strength))
        self.assertEqual([f.id for f in rep.top_findings(5)], ["technique.knee_bend", "technique.stay_lower"])
        self.assertEqual([f.id for f in rep.strengths()], ["contact.arm_extension"])


class Vocabulary(unittest.TestCase):
    def test_catalog_ids_are_prefixed_with_their_category(self):
        for fid, (category, blurb) in S.FINDING_CATALOG.items():
            self.assertTrue(fid.startswith(category.value + "."), fid)
            self.assertTrue(blurb, fid)

    def test_every_fix_tag_is_snake_case_with_a_description(self):
        for tag, text in S.FIX_TAGS.items():
            self.assertEqual(tag, tag.lower())
            self.assertNotIn(" ", tag)
            self.assertTrue(text, tag)

    def test_every_category_has_findings(self):
        used = {category for category, _ in S.FINDING_CATALOG.values()}
        self.assertEqual(used, set(S.Category))


class ExerciseRules(unittest.TestCase):
    def exercise(self, **kw):
        base = dict(
            id="wall_sit", name="Wall sit", fix_tags=("lower_stance",), equipment=(), sets_reps="3 x 30 s",
            cues=("back flat on the wall", "knees over ankles", "breathe steadily"),
            mistakes=("knees past toes", "holding breath"), safety="Stop if you feel sharp knee pain.",
            difficulty=1, gif="gifs/wall_sit.gif", duration_s=2.4,
            why="Builds the leg endurance that keeps you low in the ready stance.",
        )
        base.update(kw)
        return S.Exercise(**base)

    def test_valid_exercise(self):
        self.assertEqual(self.exercise().id, "wall_sit")

    def test_rejects_unknown_fix_tag(self):
        with self.assertRaises(ValueError):
            self.exercise(fix_tags=("nope",))

    def test_rejects_absolute_gif_path(self):
        with self.assertRaises(ValueError):
            self.exercise(gif="C:\\gifs\\wall_sit.gif")

    def test_needs_a_safety_note(self):
        with self.assertRaises(ValueError):
            self.exercise(safety="")

    def test_needs_a_short_reason(self):
        with self.assertRaises(ValueError):
            self.exercise(why="")
        with self.assertRaises(ValueError):
            self.exercise(why="x" * 161)

    def test_easier_and_harder_variants_are_ids_and_not_itself(self):
        ex = self.exercise(easier="wall_sit_short", harder="split_squat_hold")
        self.assertEqual((ex.easier, ex.harder), ("wall_sit_short", "split_squat_hold"))
        with self.assertRaises(ValueError):
            self.exercise(easier="wall_sit")
        with self.assertRaises(ValueError):
            self.exercise(harder="Not An Id")


class ReplyRules(unittest.TestCase):
    def test_suggestions_are_at_most_four_short_chips(self):
        reply = S.Reply("ok", (), "greet", 0.9, ("How?", "Show me"))
        self.assertEqual(reply.suggestions, ("How?", "Show me"))
        with self.assertRaises(ValueError):
            S.Reply("ok", (), "greet", 0.9, ("a", "b", "c", "d", "e"))
        with self.assertRaises(ValueError):
            S.Reply("ok", (), "greet", 0.9, ("x" * 49,))

    def test_confidence_is_a_probability(self):
        with self.assertRaises(ValueError):
            S.Reply("ok", (), "greet", 1.5)


class Numbers(unittest.TestCase):
    def test_numpy_scalars_become_plain_numbers(self):
        m = metric(value=np.float32(0.5), confidence=np.float64(0.75), n_obs=np.int64(4))
        self.assertIs(type(m.value), float)
        self.assertIs(type(m.confidence), float)
        self.assertIs(type(m.n_obs), int)
        self.assertEqual((m.value, m.confidence, m.n_obs), (0.5, 0.75, 4))

    def test_booleans_are_not_numbers(self):
        with self.assertRaises(ValueError):
            metric(value=True)

    def test_a_series_takes_arrays_and_maps_nan_to_unknown(self):
        self.assertEqual(S.as_series(np.array([0.12345, np.nan, 1.0])), (0.123, None, 1.0))

    def test_a_series_rejects_infinity(self):
        with self.assertRaises(ValueError) as caught:
            S.as_series([1.0, float("inf")])
        self.assertIn("non-finite", str(caught.exception))

    def test_to_plain_gives_json_types_for_numpy(self):
        wire = S.to_plain({"a": np.arange(3), "b": np.float32(1.5)})
        self.assertEqual(json.loads(json.dumps(wire)), {"a": [0, 1, 2], "b": 1.5})


class Marks(unittest.TestCase):
    def test_every_kind_checks_its_point_count(self):
        S.Mark("dot", 5, ((100, 200),))
        S.Mark("label", 5, ((100, 200),), "knee")
        S.Mark("line", 5, ((0, 0), (10, 10)))
        S.Mark("bracket", 5, ((0, 0), (10, 0)), "stance 0.9 m", 0.9)
        S.Mark("band", 5, ((0, 0), (100, 50)), style="target")
        S.Mark("angle", 5, ((0, 0), (10, 0), (10, 10)), "elbow 142", 142.0)
        S.Mark("trail", 5, tuple((i, i) for i in range(90)), style="neutral")
        for kind, points in (("dot", ()), ("line", ((0, 0),)), ("angle", ((0, 0), (1, 1))), ("trail", tuple((i, i) for i in range(91)))):
            with self.assertRaises(ValueError, msg=kind):
                S.Mark(kind, 5, points)

    def test_unknown_kind_style_and_long_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            S.Mark("arrow", 5, ((0, 0),))
        with self.assertRaises(ValueError):
            S.Mark("dot", 5, ((0, 0),), style="red")
        with self.assertRaises(ValueError):
            S.Mark("label", 5, ((0, 0),), "x" * 41)

    def test_a_moment_only_holds_marks_that_fall_inside_it(self):
        inside = S.Mark("dot", 5, ((1, 1),))
        self.assertEqual(S.Moment(S.Role.NEAR, 2, 5, 8, "x", marks=(inside,)).marks, (inside,))
        with self.assertRaises(ValueError):
            S.Moment(S.Role.NEAR, 2, 5, 8, "x", marks=(S.Mark("dot", 9, ((1, 1),)),))

    def test_marks_survive_the_round_trip(self):
        marks = (
            S.Mark("angle", 5, ((10, 20), (30, 40), (50, 20)), "knee 150", 150.0),
            S.Mark("band", 5, ((0, 0), (100, 50)), "", None, "target"),
        )
        original = report(findings=(finding(moments=(S.Moment(S.Role.NEAR, 2, 5, 8, "ready stance", marks=marks),)),))
        restored = S.report_from_dict(json.loads(json.dumps(original.to_dict())))
        self.assertEqual(restored, original)
        self.assertEqual(restored.findings[0].moments[0].marks, marks)


class BestMoments(unittest.TestCase):
    def best(self, start=20, key=24, end=28):
        return S.Moment(S.Role.NEAR, start, key, end, "your best rep", kind="best")

    def test_a_finding_carries_one_best_moment_next_to_its_issue_moment(self):
        f = finding(moments=(moment(), self.best()))
        self.assertEqual(f.best_moment.key_frame, 24)
        self.assertEqual([m.label for m in f.issue_moments], ["ready stance"])

    def test_two_best_moments_are_rejected(self):
        with self.assertRaises(ValueError):
            finding(moments=(moment(), self.best(), self.best(30, 34, 38)))

    def test_a_best_moment_cannot_stand_in_for_the_issue_moment(self):
        with self.assertRaises(ValueError):
            finding(moments=(self.best(),))

    def test_unknown_moment_kind(self):
        with self.assertRaises(ValueError):
            S.Moment(S.Role.NEAR, 2, 5, 8, "x", kind="worst")


class EventRules(unittest.TestCase):
    def test_a_swing_needs_its_data(self):
        with self.assertRaises(ValueError) as caught:
            S.Event("swing", S.Role.NEAR, 4, 0.9, {"speed": 2.0})
        self.assertIn("missing data", str(caught.exception))

    def test_swing_frames_must_nest(self):
        ok = dict(start=10, end=20, peak_frame=13, peak_speed=3.0, overhead=True)
        self.assertEqual(S.swing_event(S.Role.NEAR, 15, **ok).data["peak_frame"], 13)
        with self.assertRaises(ValueError):
            S.swing_event(S.Role.NEAR, 25, **ok)
        with self.assertRaises(ValueError):
            S.swing_event(S.Role.NEAR, 15, **{**ok, "peak_frame": 9})

    def test_swing_flags_are_stored_as_zero_or_one(self):
        e = S.swing_event(S.Role.FAR, 15, start=10, end=20, peak_frame=13, peak_speed=3.0, overhead=False, right_hand=False)
        self.assertEqual((e.data["overhead"], e.data["hand"]), (0, 0))

    def test_lunge_frames_must_be_in_order(self):
        e = S.lunge_event(S.Role.NEAR, 30, depth=0.3, bottom_frame=34, end_frame=38)
        self.assertEqual(e.data["depth"], 0.3)
        with self.assertRaises(ValueError):
            S.lunge_event(S.Role.NEAR, 30, depth=0.3, bottom_frame=28, end_frame=38)

    def test_other_events_need_no_data(self):
        self.assertEqual(S.Event("step", S.Role.NEAR, 3, 0.7).kind, "step")


class ReportExtras(unittest.TestCase):
    def test_timestamps_match_the_frame_count_and_never_go_backwards(self):
        with self.assertRaises(ValueError) as short:
            report(timestamps=(0.0, 0.1))
        self.assertIn("expected 12", str(short.exception))
        stamps = [i / 30 for i in range(12)]
        stamps[5] = 0.01
        with self.assertRaises(ValueError) as back:
            report(timestamps=tuple(stamps))
        self.assertIn("backwards", str(back.exception))

    def test_time_of_uses_real_timestamps_when_present(self):
        stamps = tuple(i / 30 + 0.5 for i in range(12))
        self.assertAlmostEqual(report(timestamps=stamps).time_of(3), 0.6)
        self.assertAlmostEqual(report().time_of(3), 0.1)

    def test_warning_codes_come_from_the_vocabulary(self):
        self.assertEqual(report(warnings=("no_court",)).warnings, ("no_court",))
        with self.assertRaises(ValueError):
            report(warnings=("made_up",))

    def test_every_warning_has_plain_text(self):
        for code, text in S.WARNINGS.items():
            self.assertEqual(code, code.lower(), code)
            self.assertTrue(text.endswith("."), code)

    def test_repeat_and_swing_warnings_name_what_was_measured(self):
        self.assertEqual(
            S.WARNINGS["repeated_frames"],
            "Some pose estimates repeat, so speed estimates skip those repeats.",
        )
        self.assertEqual(
            S.WARNINGS["few_swings"],
            "Only a few arm swings were found, so averages are rough.",
        )
        self.assertNotIn("video", S.WARNINGS["repeated_frames"].lower())
        self.assertNotIn("shot", S.WARNINGS["few_swings"].lower())
        self.assertEqual(tuple(S.WARNINGS), (
            "doubles", "high_fps", "no_court", "preview_estimate",
            "repeated_frames", "few_swings", "partial_track", "unresolved_recovery",
        ))
        original = report(warnings=("repeated_frames", "few_swings"))
        restored = S.report_from_dict(json.loads(json.dumps(original.to_dict())))
        self.assertEqual(restored.warnings, ("repeated_frames", "few_swings"))
        self.assertEqual(restored, original)

    def test_numpy_series_are_stored_as_rounded_tuples(self):
        rep = report(series={"near": {"hip_height_ratio": np.linspace(0.0, 1.0, 12)}})
        got = rep.series["near"]["hip_height_ratio"]
        self.assertIsInstance(got, tuple)
        self.assertEqual(got[1], 0.091)

    def test_series_error_names_the_player_and_series(self):
        with self.assertRaises(ValueError) as caught:
            report(series={"near": {"hip_height_ratio": (0.5,) * 11 + (float("inf"),)}})
        self.assertIn("series near.hip_height_ratio", str(caught.exception))

    def test_top_findings_break_severity_ties_by_score(self):
        low = finding(score=1.2)
        high = finding(id="technique.knee_bend", score=2.7)
        rep = report(findings=(low, high))
        self.assertEqual([f.id for f in rep.top_findings()], ["technique.knee_bend", "technique.stay_lower"])

    def test_extra_fields_survive_the_round_trip(self):
        stamps = tuple(i / 30 for i in range(12))
        original = report(video_id="abc", revision=3, warnings=("preview_estimate",), timestamps=stamps)
        restored = S.report_from_dict(json.loads(json.dumps(original.to_dict())))
        self.assertEqual((restored.video_id, restored.revision, restored.warnings), ("abc", 3, ("preview_estimate",)))
        self.assertEqual(restored, original)


class UnresolvedRecovery(unittest.TestCase):
    def _tracks(self, tier):
        n = 12
        player = PlayerTrack(
            role=S.Role.NEAR, n=n, present=np.ones(n, bool), measured=np.ones(n, bool),
            visible=np.ones((n, 17), bool), xy=np.full((n, 17, 2), np.nan, np.float32),
            conf=np.zeros((n, 17), np.float32), bbox=np.full((n, 4), np.nan, np.float32),
            foot_px=np.full((n, 2), np.nan, np.float32), foot_court=np.full((n, 2), np.nan, np.float32),
            stature_px=np.full(n, np.nan, np.float32), px_per_m=np.full(n, np.nan, np.float32),
        )
        return Tracks(
            tier=tier, n=n, fps=30.0, t=np.arange(n) / 30.0, width=1920, height=1080,
            players={S.Role.NEAR: player}, court=None, shots=((0, n - 1),), source={"file": "x.mp4"},
        )

    def test_warning_round_trips_on_both_tiers_without_a_made_up_lunge(self):
        gap = S.Window(2, 8, "near unresolved recovery")
        part = S.Partial(windows=(gap,), warnings=("unresolved_recovery",))
        self.assertIn("ready hips from before the drop", S.WARNINGS["unresolved_recovery"])
        for tier in (S.Tier.PREVIEW, S.Tier.REFINED):
            report = assemble(self._tracks(tier), (), (part,))
            self.assertIn("unresolved_recovery", report.warnings)
            self.assertFalse(any(event.kind == "lunge" for event in report.events))
            self.assertFalse(any("end_frame" in event.data for event in report.events))
            restored = S.report_from_dict(json.loads(json.dumps(report.to_dict())))
            self.assertEqual(restored, report)
            self.assertIn("unresolved_recovery", restored.warnings)
            if tier == S.Tier.REFINED:
                self.assertEqual(restored.windows, ())
                self.assertNotIn("preview_estimate", restored.warnings)
            else:
                self.assertEqual(restored.windows, (gap,))
                self.assertIn("preview_estimate", restored.warnings)


class MetricAvailability(unittest.TestCase):
    def assertRejected(self, fn, fragment):
        with self.assertRaises(ValueError) as caught:
            fn()
        self.assertIn(fragment, str(caught.exception))

    def absent(self, **kw):
        base = dict(value=None, confidence=0.0, n_obs=0, best=None, series=None)
        base.update(kw)
        return metric(**base)

    def wire(self, version=S.LEGACY_SCHEMA_VERSION):
        raw = json.loads(json.dumps(report().to_dict()))
        raw["schema_version"] = version
        raw["metrics"]["near"][0].update(value=0.41, n_obs=0, confidence=0, best=None)
        raw["metrics"]["far"] = [dict(raw["metrics"]["near"][0], value=99.0)]
        found = raw["findings"][0]
        found.update(severity="info", score=0, moments=[], fix_tags=[])
        value = 0.62 if version == S.LEGACY_SCHEMA_VERSION else None
        found["metrics"][0].update(value=value, n_obs=0, confidence=0, best=None)
        return raw

    def test_a_missing_reading_is_none_only_with_no_observations(self):
        self.assertEqual(S.SCHEMA_VERSION, 3)
        gone = self.absent(target=(0.4, 0.5))
        self.assertIsNone(gone.value)
        self.assertEqual((gone.n_obs, gone.confidence, gone.best, gone.target), (0, 0.0, None, (0.4, 0.5)))
        kept = metric(value=0.2, confidence=0.0, n_obs=4, best=0.1)
        self.assertEqual((kept.value, kept.confidence, kept.n_obs, kept.best), (0.2, 0.0, 4, 0.1))

    def test_numpy_zero_counts_as_an_absent_reading(self):
        gone = self.absent(n_obs=np.int64(0), confidence=np.float64(0.0))
        self.assertIsNone(gone.value)
        self.assertIs(type(gone.n_obs), int)
        self.assertIs(type(gone.confidence), float)

    def test_illegal_absence_combinations_are_rejected(self):
        cases = (
            (dict(value=None), "n_obs"),
            (dict(value=None, n_obs=0, confidence=0.4), "confidence"),
            (dict(value=None, n_obs=0, confidence=0, best=0.2), "best"),
            (dict(value=None, n_obs=5, confidence=0, best=None), "n_obs"),
            (dict(value=0.5, n_obs=0, confidence=0, best=None), "n_obs"),
            (dict(value=float("inf"), n_obs=2), "finite number"),
            (dict(confidence=-0.1), "confidence"),
            (dict(best=float("nan")), "finite number"),
            (dict(n_obs=-1), "n_obs"),
        )
        for kw, fragment in cases:
            self.assertRejected(lambda kw=kw: metric(**kw), fragment)

    def test_absent_metric_round_trips_as_json_null(self):
        gone = finding(
            severity=S.Severity.INFO, score=0.0, moments=(), fix_tags=(),
            metrics=(self.absent(),), overlay_series=(),
        )
        original = report(metrics={"near": (self.absent(),)}, findings=(gone,), series={})
        wire = json.loads(json.dumps(original.to_dict()))
        self.assertIsNone(wire["metrics"]["near"][0]["value"])
        self.assertIsNone(wire["findings"][0]["metrics"][0]["value"])
        self.assertEqual(wire["schema_version"], 3)
        self.assertEqual(S.report_from_dict(wire), original)

    def test_version_2_placeholders_become_none_without_rewriting_the_file(self):
        raw = self.wire()
        snapshot = json.dumps(raw)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old-report.json"
            path.write_text(snapshot, encoding="utf-8")
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded = S.report_from_dict(json.loads(path.read_text(encoding="utf-8")))
            after = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(before, after)
        self.assertEqual(snapshot, json.dumps(raw))
        self.assertEqual(loaded.schema_version, 3)
        self.assertIsNone(loaded.metrics["near"][0].value)
        self.assertIsNone(loaded.metrics["far"][0].value)
        self.assertEqual(loaded.metrics["far"][0].n_obs, 0)
        self.assertIsNone(loaded.findings[0].metrics[0].value)
        self.assertEqual(loaded.findings[0].severity, S.Severity.INFO)
        self.assertEqual(loaded.findings[0].score, 0)
        self.assertEqual(loaded.findings[0].moments, ())

    def test_missing_legacy_n_obs_stays_an_observation(self):
        raw = self.wire()
        item = raw["metrics"]["near"][0]
        item.pop("n_obs")
        item.update(value=0.45, confidence=0, best=None)
        loaded = S.report_from_dict(raw)
        self.assertEqual((loaded.metrics["near"][0].value, loaded.metrics["near"][0].n_obs), (0.45, 1))

    def test_legacy_confidence_zero_with_observations_stays_numeric(self):
        raw = self.wire()
        raw["metrics"]["near"][0].update(value=0.2, n_obs=4, confidence=0, best=0.1)
        loaded = S.report_from_dict(raw).metrics["near"][0]
        self.assertEqual((loaded.value, loaded.n_obs, loaded.confidence, loaded.best), (0.2, 4, 0.0, 0.1))

    def test_contradictory_legacy_values_are_rejected(self):
        cases = (
            dict(value=0.45, n_obs=0, confidence=0.2, best=None),
            dict(value=0.45, n_obs=0, confidence=0, best=0.4),
            dict(value=0.45, n_obs=0, confidence=0, best=0),
            dict(value=float("nan"), n_obs=0, confidence=0, best=None),
            dict(value=None, n_obs=2, confidence=0, best=None),
        )
        for fields in cases:
            raw = self.wire()
            raw["metrics"]["near"][0].update(fields)
            snapshot = json.dumps(raw)
            with self.assertRaises(ValueError, msg=str(fields)):
                S.report_from_dict(raw)
            self.assertEqual(json.dumps(raw), snapshot)

    def test_a_legacy_null_with_no_observations_loads(self):
        raw = self.wire()
        raw["metrics"]["near"][0].update(value=None, n_obs=0, confidence=0, best=None)
        self.assertIsNone(S.report_from_dict(raw).metrics["near"][0].value)

    def test_current_version_does_not_treat_a_placeholder_as_legacy(self):
        raw = self.wire()
        raw["schema_version"] = 3
        raw["metrics"]["near"][0].update(value=0.45, n_obs=0, confidence=0, best=None)
        snapshot = json.dumps(raw)
        self.assertRejected(lambda: S.report_from_dict(raw), "n_obs")
        self.assertEqual(json.dumps(raw), snapshot)

    def test_a_float_zero_observation_count_is_not_guessed(self):
        raw = self.wire()
        raw["metrics"]["near"][0].update(value=0.45, n_obs=0.0, confidence=0, best=None)
        self.assertRejected(lambda: S.report_from_dict(raw), "integer")

    def test_other_report_versions_are_rejected(self):
        raw = self.wire()
        raw["schema_version"] = 1
        self.assertRejected(lambda: S.report_from_dict(raw), "schema_version")
        self.assertRejected(lambda: report(schema_version=2), "schema_version must be 3")

    def test_loader_rejects_an_all_missing_finding_that_overclaims(self):
        moment = json.loads(json.dumps(report().to_dict()))["findings"][0]["moments"]
        cases = (
            (dict(severity="strength"), "must be info"),
            (dict(moments=moment), "must not have a moment"),
            (dict(score=0.0005), "score must be 0"),
        )
        for version in (S.LEGACY_SCHEMA_VERSION, S.SCHEMA_VERSION):
            for patch, fragment in cases:
                raw = self.wire(version)
                if version == S.SCHEMA_VERSION:
                    for item in (*raw["metrics"]["near"], *raw["metrics"]["far"]):
                        item["value"] = None
                raw["findings"][0].update(patch)
                snapshot = json.dumps(raw)
                self.assertRejected(lambda raw=raw: S.report_from_dict(raw), fragment)
                self.assertEqual(json.dumps(raw), snapshot)

    def test_measured_and_mixed_findings_still_load(self):
        measured = json.loads(json.dumps(report().to_dict()))
        measured["schema_version"] = S.LEGACY_SCHEMA_VERSION
        measured["findings"][0]["metrics"][0].update(value=0.2, n_obs=4, confidence=0, best=0.1)
        loaded = S.report_from_dict(measured).findings[0]
        self.assertEqual((loaded.metrics[0].value, loaded.metrics[0].n_obs, loaded.metrics[0].confidence), (0.2, 4, 0.0))
        self.assertEqual(loaded.severity, S.Severity.IMPROVE)
        self.assertTrue(loaded.moments)

        mixed = json.loads(json.dumps(report().to_dict()))
        seen = mixed["findings"][0]["metrics"][0]
        absent = dict(seen, name="ready_trunk_lean", label="Lean", value=0.5, n_obs=0, confidence=0, best=None)
        mixed["schema_version"] = S.LEGACY_SCHEMA_VERSION
        mixed["findings"][0]["metrics"] = [absent, seen]
        mixed["findings"][0].update(severity="strength", score=0, moments=[], fix_tags=[])
        loaded = S.report_from_dict(mixed).findings[0]
        self.assertIsNone(loaded.metrics[0].value)
        self.assertEqual(loaded.metrics[1].value, seen["value"])
        self.assertEqual(loaded.severity, S.Severity.STRENGTH)
        self.assertEqual(loaded.score, 0)

        current = json.loads(json.dumps(mixed))
        current["schema_version"] = S.SCHEMA_VERSION
        current["findings"][0]["metrics"][0]["value"] = None
        loaded = S.report_from_dict(current).findings[0]
        self.assertIsNone(loaded.metrics[0].value)
        self.assertEqual(loaded.severity, S.Severity.STRENGTH)


if __name__ == "__main__":
    unittest.main()
