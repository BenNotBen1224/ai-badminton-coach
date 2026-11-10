import unittest

import numpy as np

from coach import schema as S
from coach.analyze import assemble, merge_windows
from coach.catalog import build_catalog
from coach.severity import MetricSpec
from coach.tracks import PlayerTrack, Tracks


def fake_track(role, n):
    nan = np.full
    return PlayerTrack(
        role=role, n=n, present=np.ones(n, bool), measured=np.ones(n, bool), visible=np.ones((n, 17), bool),
        xy=nan((n, 17, 2), np.nan, np.float32), conf=np.zeros((n, 17), np.float32),
        bbox=nan((n, 4), np.nan, np.float32), foot_px=nan((n, 2), np.nan, np.float32),
        foot_court=nan((n, 2), np.nan, np.float32), stature_px=nan(n, np.nan, np.float32),
        px_per_m=nan(n, np.nan, np.float32),
    )


def fake_tracks(n=60, tier=S.Tier.PREVIEW, warnings=()):
    return Tracks(
        tier=tier, n=n, fps=30.0, t=np.arange(n) / 30.0, width=1920, height=1080,
        players={S.Role.NEAR: fake_track(S.Role.NEAR, n), S.Role.FAR: fake_track(S.Role.FAR, n)},
        court=None, shots=((0, n - 1),), source={"file": "x.mp4"}, warnings=warnings,
    )


def swing(frame, role=S.Role.NEAR):
    return S.swing_event(role, frame, start=frame - 3, end=frame + 3, peak_frame=frame - 1, peak_speed=2.0, overhead=True)


def metric(name, **kw):
    base = dict(name=name, label=name, value=1.0, unit="ratio", basis=S.Basis.PROJECTED_2D, confidence=0.7)
    base.update(kw)
    return S.Metric(**base)


def improve(key, series, fid="technique.stay_lower", cat=S.Category.TECHNIQUE, start=10, end=20, **kw):
    base = dict(
        id=fid, category=cat, player=S.Role.NEAR, severity=S.Severity.IMPROVE, headline="h", explain="e",
        metrics=(metric("m_" + fid.replace(".", "_")),), moments=(S.Moment(S.Role.NEAR, start, key, end, "m"),),
        tier=S.Tier.PREVIEW, fix_tags=("lower_stance",), overlay_series=(series,),
    )
    base.update(kw)
    return S.Finding(**base)


class Assemble(unittest.TestCase):
    def test_merges_metrics_series_events_and_findings_from_both_units(self):
        n = 60
        body = S.Partial(
            metrics={"near": (metric("hip_height_ratio"),)},
            series={"near": {"hip_height_ratio": (0.5,) * n}},
            findings=(improve(15, "hip_height_ratio"),),
        )
        move = S.Partial(
            metrics={"near": (metric("recovery_time"),), "far": (metric("recovery_time"),)},
            series={"far": {"base_distance": (1.0,) * n}},
            events=(S.Event("step", S.Role.FAR, 30, 0.8),),
        )
        hit = swing(15)
        report = assemble(fake_tracks(n), (hit,), (body, move))
        self.assertEqual([m.name for m in report.metrics["near"]], ["hip_height_ratio", "recovery_time"])
        self.assertEqual(sorted(report.series), ["far", "near"])
        self.assertEqual([(e.kind, e.frame) for e in report.events], [("swing", 15), ("step", 30)])
        self.assertEqual(len(report.findings), 1)

    def test_two_units_cannot_write_the_same_series(self):
        part = S.Partial(series={"near": {"hip_height_ratio": (0.5,) * 60}})
        with self.assertRaises(ValueError) as caught:
            assemble(fake_tracks(), (), (part, part))
        self.assertIn("two analysis units", str(caught.exception))

    def test_repeated_metric_name_is_rejected(self):
        part = S.Partial(metrics={"near": (metric("x"),)})
        with self.assertRaises(ValueError):
            assemble(fake_tracks(), (), (part, part))

    def test_preview_windows_cover_finding_moments_with_padding_and_clamp(self):
        part = S.Partial(
            series={"near": {"hip_height_ratio": (0.5,) * 60}},
            findings=(improve(2, "hip_height_ratio", start=1, end=4), improve(40, "hip_height_ratio", fid="technique.knee_bend", start=38, end=58)),
        )
        report = assemble(fake_tracks(60), (), (part,))
        self.assertEqual([(w.start_frame, w.end_frame) for w in report.windows], [(0, 7), (35, 59)])

    def test_refined_reports_have_no_windows(self):
        part = S.Partial(
            series={"near": {"hip_height_ratio": (0.5,) * 60}},
            findings=(improve(15, "hip_height_ratio"),),
        )
        self.assertEqual(assemble(fake_tracks(60, S.Tier.REFINED), (), (part,)).windows, ())

    def test_best_moments_do_not_ask_for_a_refine_window(self):
        best = S.Moment(S.Role.NEAR, 40, 44, 48, "your best rep", kind="best")
        part = S.Partial(
            series={"near": {"hip_height_ratio": (0.5,) * 60}},
            findings=(improve(15, "hip_height_ratio", moments=(S.Moment(S.Role.NEAR, 10, 15, 20, "m"), best)),),
        )
        report = assemble(fake_tracks(60), (), (part,))
        self.assertEqual([(w.start_frame, w.end_frame) for w in report.windows], [(7, 23)])

    def test_video_id_revision_and_timestamps_are_carried_into_the_report(self):
        tracks = fake_tracks(60)
        report = assemble(tracks, (), (), None, "clip-a", 2)
        self.assertEqual((report.video_id, report.revision), ("clip-a", 2))
        self.assertEqual(len(report.timestamps), 60)
        self.assertAlmostEqual(report.time_of(30), 1.0)

    def test_warnings_from_tracks_and_units_merge_without_repeats(self):
        part = S.Partial(warnings=("few_swings", "repeated_frames"))
        report = assemble(fake_tracks(60, S.Tier.REFINED, ("repeated_frames", "no_court")), (), (part,))
        self.assertEqual(report.warnings, ("repeated_frames", "no_court", "few_swings"))

    def test_only_the_preview_tier_warns_that_numbers_are_estimates(self):
        self.assertIn("preview_estimate", assemble(fake_tracks(60), (), ()).warnings)
        self.assertNotIn("preview_estimate", assemble(fake_tracks(60, S.Tier.REFINED), (), ()).warnings)


class AnalyzeTimings(unittest.TestCase):
    def run_analyze(self, timings):
        from unittest import mock
        from coach import analyze as A
        with mock.patch.object(A, "load_tracks", return_value=fake_tracks(60)), \
                mock.patch.object(A, "detect_events", return_value=()), \
                mock.patch.object(A, "body_analysis", return_value=S.Partial()), \
                mock.patch.object(A, "movement_analysis", return_value=S.Partial()):
            return A.analyze({}, "clip-a", 2, timings)

    def test_every_stage_is_timed_when_a_dict_is_given(self):
        from coach.analyze import STAGES
        timings = {}
        report = self.run_analyze(timings)
        self.assertEqual(tuple(timings), STAGES)
        self.assertTrue(all(seconds >= 0 for seconds in timings.values()))
        self.assertEqual((report.video_id, report.revision), ("clip-a", 2))

    def test_no_dict_means_no_timing_and_the_same_report(self):
        self.assertEqual(self.run_analyze(None).to_dict(), self.run_analyze({}).to_dict())


class CatalogCheck(unittest.TestCase):
    SPEC =  MetricSpec("hip_height_ratio", "Hip height", "ratio", S.Basis.PROJECTED_2D, "lower_better", (0.0, 0.5), 0.1, "technique.stay_lower")

    def part(self, severity, score):
        m = metric("hip_height_ratio", value=0.8, confidence=0.8, n_obs=6, target=(0.0, 0.5))
        return S.Partial(
            series={"near": {"hip_height_ratio": (0.5,) * 60}},
            findings=(improve(15, "hip_height_ratio", severity=severity, score=score, metrics=(m,)),),
        )

    def test_a_finding_that_follows_the_rule_is_accepted(self):
        catalog = build_catalog((self.SPEC,))
        report = assemble(fake_tracks(60), (), (self.part(S.Severity.PRIORITY, 2.4),), catalog)
        self.assertEqual(report.findings[0].severity, S.Severity.PRIORITY)

    def test_a_hand_picked_severity_is_rejected(self):
        catalog = build_catalog((self.SPEC,))
        with self.assertRaises(ValueError) as caught:
            assemble(fake_tracks(60), (), (self.part(S.Severity.IMPROVE, 2.4),), catalog)
        self.assertIn("catalog rule gives priority", str(caught.exception))

    def test_a_hand_picked_score_is_rejected(self):
        catalog = build_catalog((self.SPEC,))
        with self.assertRaises(ValueError) as caught:
            assemble(fake_tracks(60), (), (self.part(S.Severity.PRIORITY, 9.0),), catalog)
        self.assertIn("score", str(caught.exception))


class MergeWindows(unittest.TestCase):
    def test_overlapping_and_adjacent_windows_merge_and_keep_both_reasons(self):
        merged = merge_windows([S.Window(10, 20, "a"), S.Window(21, 30, "b"), S.Window(50, 60, "c"), S.Window(55, 58, "a")])
        self.assertEqual(
            [(w.start_frame, w.end_frame, w.reason) for w in merged],
            [(10, 30, "a, b"), (50, 60, "c, a")],
        )


if __name__ == "__main__":
    unittest.main()
