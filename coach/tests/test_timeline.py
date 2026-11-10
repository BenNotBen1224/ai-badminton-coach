from __future__ import annotations

import unittest

from coach.schema import (
    SCHEMA_VERSION,
    Basis,
    Category,
    Finding,
    Metric,
    Moment,
    Report,
    Role,
    Severity,
    Tier,
)
from coach.timeline import PARAMS, Segment, Timeline, build_review, build_timeline, review_picks


def _metric():
    return Metric(
        "hip_height_ratio",
        "Hip height",
        0.53,
        "ratio",
        Basis.PROJECTED_2D,
        0.8,
        (0.42, 0.48),
    )


def _moment(start, key, end, kind="issue"):
    return Moment(Role.NEAR, start, key, end, "Ready stance", kind)


def _finding(moments, finding_id="technique.stay_lower", severity=Severity.IMPROVE, category=Category.TECHNIQUE):
    tags = ("lower_stance",) if severity in (Severity.IMPROVE, Severity.PRIORITY) else ()
    return Finding(
        finding_id,
        category,
        Role.NEAR,
        severity,
        "Hips stay high while you wait",
        "Your hips sit above the practice guide range while you wait.",
        (_metric(),),
        moments,
        Tier.REFINED,
        fix_tags=tags,
    )


def _report(n, findings, fps=30.0):
    return Report(
        schema_version=SCHEMA_VERSION,
        source={"width": 1280, "height": 720, "frame_count": n, "fps": fps},
        tier=Tier.REFINED,
        n_frames=n,
        fps=fps,
        players=(Role.NEAR,),
        metrics={},
        series={},
        events=(),
        findings=tuple(findings),
        timestamps=tuple(i / fps for i in range(n)),
    )


def _walls(timeline, stamps):
    total = 0.0
    for seg in timeline.to_dict()["segments"]:
        if seg["kind"] == "freeze":
            total += seg["hold_s"]
        else:
            total += (stamps[seg["end_frame"]] - stamps[seg["start_frame"]]) / seg["speed"]
    return total


class TimelineTests(unittest.TestCase):
    def test_unequal_gaps_do_not_use_fps(self):
        stamps = (0.0, 0.01, 0.50, 0.51, 0.52)
        timeline = Timeline((Segment("play", 0, 2, 1.0, 0.0, 1.0, "issue"),), stamps)
        self.assertEqual(timeline.duration_s, 0.5)
        got = timeline.frame_at(0.01)
        self.assertAlmostEqual(got, 1.0, delta=1e-9)
        self.assertGreater(abs(got - 0.3), 0.1)
        self.assertEqual(timeline.frame_at(timeline.duration_s), 2.0)

    def test_freeze_holds_the_key_frame(self):
        n = 120
        fps = 30.0
        key = 45
        stamps = tuple(i / fps for i in range(n))
        timeline = build_timeline(_moment(0, key, 90), fps, n, timestamps=stamps)
        cursor = 0.0
        freeze = None
        for seg in timeline.to_dict()["segments"]:
            if seg["kind"] == "freeze":
                wall = seg["hold_s"]
            else:
                wall = (stamps[seg["end_frame"]] - stamps[seg["start_frame"]]) / seg["speed"]
            if seg["kind"] == "freeze":
                freeze = (cursor, cursor + wall, seg)
                break
            cursor += wall
        start, end, seg = freeze
        self.assertEqual(seg["hold_s"], 1.5)
        self.assertEqual(seg["start_frame"], key)
        self.assertEqual(seg["end_frame"], key)
        t = start
        while t < end - 1e-9:
            self.assertEqual(timeline.frame_at(t), float(key))
            t += 0.02
        self.assertEqual(
            timeline.sample_at(start),
            {"frame": float(key), "speed": 0.0, "zoom": 1.6, "show": "issue"},
        )

    def test_issue_samples_stay_continuous(self):
        n = 160
        fps = 30.0
        issue = _moment(0, 40, 80)
        stamps = tuple(i / fps for i in range(n))
        timeline = build_timeline(issue, fps, n, timestamps=stamps)
        frames = _samples(timeline)
        jumps = [abs(b - a) for a, b in zip(frames, frames[1:]) if abs(b - a) > 1.5]
        self.assertEqual(jumps, [])
        for seg in timeline.to_dict()["segments"]:
            self.assertGreaterEqual(seg["start_frame"], issue.start_frame)
            self.assertLessEqual(seg["end_frame"], issue.end_frame)

    def test_compare_has_one_jump_to_the_best_key(self):
        n = 160
        fps = 30.0
        issue = _moment(0, 40, 80)
        best = _moment(100, 120, 140, kind="best")
        stamps = tuple(i / fps for i in range(n))
        timeline = build_timeline(issue, fps, n, timestamps=stamps, compare=best)
        frames = _samples(timeline)
        big = [(a, b) for a, b in zip(frames, frames[1:]) if abs(b - a) > 1.5]
        self.assertEqual(len(big), 1)
        self.assertEqual(big[0][1], float(best.key_frame))
        last = timeline.to_dict()["segments"][-1]
        self.assertEqual(last["show"], "best")
        self.assertEqual(last["kind"], "freeze")
        hold_start = timeline.duration_s - last["hold_s"]
        self.assertEqual(timeline.frame_at(hold_start), float(best.key_frame))
        self.assertEqual(timeline.frame_at(hold_start + 0.5), float(best.key_frame))
        self.assertEqual(timeline.frame_at(timeline.duration_s), float(best.key_frame))

    def test_duration_is_the_sum_of_segment_walls(self):
        n = 120
        fps = 30.0
        stamps = tuple(i / fps for i in range(n))
        timeline = build_timeline(_moment(0, 45, 90), fps, n, timestamps=stamps)
        self.assertAlmostEqual(timeline.duration_s, _walls(timeline, stamps), delta=1e-9)

    def test_review_drops_compare_before_the_issue(self):
        n = 160
        fps = 30.0
        issue = _moment(0, 40, 80)
        best = _moment(100, 120, 140, kind="best")
        finding = _finding((issue, best))
        report = _report(n, [finding], fps)
        stamps = report.timestamps
        bare = build_timeline(issue, fps, n, timestamps=stamps, compare=None)
        full = build_timeline(issue, fps, n, timestamps=stamps, compare=best)
        self.assertLess(bare.duration_s + 1.0, full.duration_s)
        mid = (bare.duration_s + full.duration_s) / 2.0
        tight = build_review(report, [finding], fps, limit_s=mid)
        self.assertEqual(len(tight), 1)
        self.assertEqual([seg.show for seg in tight[0].segments if seg.show == "best"], [])
        self.assertAlmostEqual(tight[0].duration_s, bare.duration_s, delta=1e-9)
        wide = build_review(report, [finding], fps, limit_s=full.duration_s + 1.0)
        self.assertEqual(len(wide), 1)
        self.assertEqual(
            [seg.start_frame for seg in wide[0].segments if seg.show == "best"],
            [best.key_frame],
        )
        self.assertAlmostEqual(wide[0].duration_s, full.duration_s, delta=1e-9)

    def test_review_keeps_later_real_moments_until_the_cap(self):
        n = 120
        fps = 30.0
        improve = _finding((_moment(8, 16, 30),))
        later = _finding(
            (_moment(40, 50, 70),),
            finding_id="contact.arm_extension",
            severity=Severity.INFO,
            category=Category.CONTACT,
        )
        report = _report(n, [later, improve], fps)
        self.assertEqual(
            [(item.id, item.severity) for item in review_picks(report)],
            [("technique.stay_lower", Severity.IMPROVE), ("contact.arm_extension", Severity.INFO)],
        )
        filled = build_review(report, None, fps, limit_s=30)
        self.assertEqual(
            [next(seg.start_frame for seg in item.segments if seg.kind == "freeze") for item in filled],
            [16, 50],
        )
        self.assertEqual(
            [seg.hold_s for item in filled for seg in item.segments if seg.kind == "freeze"],
            [1.5, 1.5],
        )
        stopped = build_review(report, None, fps, limit_s=filled[0].duration_s)
        self.assertEqual(len(stopped), 1)
        self.assertEqual(stopped[0].segments[0].start_frame, 8)

    def test_params_replace_only_the_given_freeze(self):
        stamps = tuple(i / 30.0 for i in range(100))
        timeline = build_timeline(
            _moment(0, 40, 80), 30.0, 100, timestamps=stamps, params={"freeze_s": 0.25},
        )
        holds = [seg["hold_s"] for seg in timeline.to_dict()["segments"] if seg["kind"] == "freeze"]
        self.assertEqual(holds, [0.25])
        self.assertEqual(PARAMS["freeze_s"], 1.5)


def _samples(timeline):
    frames = []
    t = 0.0
    while t < timeline.duration_s - 1e-12:
        frames.append(timeline.frame_at(t))
        t += 0.02
    frames.append(timeline.frame_at(timeline.duration_s))
    return frames


if __name__ == "__main__":
    unittest.main()
