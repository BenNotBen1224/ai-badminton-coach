from __future__ import annotations

import copy
from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from coach.analyze import analyze
from coach.short_review import _crop, make_schedule, select_points, source_credit, FPS, MAX_FREEZE_S
from coach.tracks import load_tracks
from coach.play_state import exchange_mask
from coach.player_input import player_inputs
from coach.coaching_corpus.corpus_learn import HEDGE_ZH


class ShortReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.doc = json.loads((root / "fast" / "runs" / "step4-rebalanced-dev" / "outputs" / "04-r1.json").read_text("utf-8"))
        cls.doc["source"]["players"] = player_inputs(root / "sample-videos/04-china-open-ws-a.mp4")
        cls.report = analyze(cls.doc)
        cls.tracks = load_tracks(cls.doc)
        cls.points = select_points(cls.doc, cls.report)

    def schedule(self, lengths=None):
        return make_schedule(self.points, lengths or [[2.3, 3.2, 3.0]] * 3, 30, 900)

    def test_freeze_and_paused_time_caps(self):
        schedule = self.schedule()
        self.assertGreaterEqual(schedule["planned_duration_s"], 30)
        self.assertLessEqual(schedule["planned_duration_s"], 45)
        self.assertLessEqual(schedule["paused_share"], 0.35)
        for point in schedule["points"]:
            self.assertLessEqual(point["freeze_end_s"] - point["freeze_start_s"], MAX_FREEZE_S)
        self.assertLessEqual(sum(state["phase"] == "hook" for state in schedule["states"]) / FPS, 1.5)

    def test_long_speech_is_rejected_not_silently_truncated(self):
        with self.assertRaisesRegex(RuntimeError, "outside 30-45s"):
            self.schedule([[6.0, 6.0, 6.0]] * 3)

    def test_captions_are_inside_actual_clause_and_point_windows(self):
        lengths = [[1.923, 2.178, 3.428], [2.002, 2.522, 3.222], [1.213, 3.444, 3.777]]
        schedule = self.schedule(lengths)
        for row, duration in zip(schedule["captions"], [v for parts in lengths for v in parts]):
            self.assertAlmostEqual(row["end_s"] - row["start_s"], duration)
            self.assertEqual(row["start_s"], row["speech_start_s"])
            self.assertEqual(row["end_s"], row["speech_end_s"])
            point = schedule["points"][row["point"]]
            self.assertGreaterEqual(row["start_s"], point["output_start_s"])
            self.assertLessEqual(row["end_s"], point["output_end_s"])

    def test_freeze_crop_contains_complete_focus_box(self):
        schedule = self.schedule()
        for point in schedule["points"]:
            x1, y1, x2, y2 = point["bbox"]
            left, top, right, bottom = point["freeze_crop"]
            self.assertLessEqual(left, x1)
            self.assertLessEqual(top, y1)
            self.assertGreaterEqual(right, x2)
            self.assertGreaterEqual(bottom, y2)
            for frame, state in zip(schedule["frame_map"], schedule["states"]):
                if state["phase"] == "freeze" and state["point"] == schedule["points"].index(point):
                    self.assertEqual(frame, point["source_frame"])

    def test_no_point_or_displayed_excerpt_spans_camera_cut(self):
        schedule = self.schedule()
        active = exchange_mask(self.tracks, self.report.events)
        for point in schedule["points"]:
            self.assertNotIn(point["source_frame"], (668, 680, 385, 704))
            self.assertTrue(active[point["source_frame"]])
            self.assertLessEqual(point["source_start_frame"], point["cycle"]["start_frame"])
            self.assertGreaterEqual(point["source_end_frame"], point["cycle"]["return_frame"])
        for frame, state in zip(schedule["frame_map"], schedule["states"]):
            point = schedule["points"][state["point"]]
            if state["phase"] != "credits":
                self.assertLessEqual(point["shot"][0], frame)
                self.assertLessEqual(frame, point["shot"][1])

    def test_cross_cut_window_is_an_error(self):
        points = copy.deepcopy(self.points)
        points[0]["source_end_frame"] = points[0]["shot"][1] + 1
        with self.assertRaisesRegex(ValueError, "one camera shot"):
            make_schedule(points, [[2, 3, 3]] * 3, 30, 900)

    def test_frame_measurements_are_not_aggregate_values(self):
        step, knee, recovery = self.points
        self.assertEqual([p["n"] for p in self.points], [1, 1, 1])
        self.assertAlmostEqual(step["value"], 0.607519578124478)
        self.assertAlmostEqual(knee["value"], 113.86665479734488)
        self.assertAlmostEqual(recovery["value"], 0.3333, places=4)
        self.assertNotIn("错误", "".join(p["title"] for p in self.points))

    def test_incomplete_measurement_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "bounding box"):
            _crop([float("nan"), 4, 10, 20], 1920, 1080)

    def test_no_unsupported_replacement_for_missing_evidence(self):
        report = replace(self.report, events=(), findings=())
        with self.assertRaisesRegex(RuntimeError, "Only .* supported coaching concepts"):
            select_points(self.doc, report)

    def test_one_plain_instruction_per_point_has_no_hedges(self):
        for point in self.points:
            self.assertTrue(point["cue"].startswith("关键是"))
            self.assertEqual(point["cue"], "".join(point["clauses"]))
            self.assertIsNone(HEDGE_ZH.search(point["cue"]))
            self.assertEqual(point["cue"].count("。"), 1)
        recovery = next(point for point in self.points if point["concept"] == "recovery")
        self.assertIn("回位", recovery["title"])
        self.assertIn("回位", recovery["clauses"][0])

    def test_unknown_video_does_not_inherit_sample_authorship(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "Source credit unavailable"):
                source_credit(Path(folder) / "other.mp4")

    def test_explicit_credit_is_used_without_inheriting_sample_author(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "credit.json"
            file.write_text(json.dumps({"author": "Local creator", "license": "Permission",
                                        "source_url": "https://example.com/my-video",
                                        "license_url": "https://example.com/permission"}), "utf-8")
            credit = source_credit(Path(folder) / "other.mp4", file)
            self.assertIn("Local creator", credit["endcard"])
            self.assertNotIn("Badminton Sphere", credit["full"])


if __name__ == "__main__":
    unittest.main()
