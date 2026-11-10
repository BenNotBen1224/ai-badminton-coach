from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from coach.analyze import analyze
from coach.complete_review import _wrap_zh
from coach.schema import Role
from coach.whole_clip_review import (
    _analysis_data,
    _overlay_supported,
    _shared_court_crop,
    _supported_joint,
    _write_analysis,
    candidate_inventory,
    choose_moments,
    make_schedule,
    verify_pose_binding,
)
from coach.app.backend import sha256_file
from coach.tracks import load_tracks


ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "sample-videos" / "04-china-open-ws-a.mp4"
V4 = ROOT / "results" / "tiktok-coaching-editorial-20261103-v4"
POSE = V4 / "pose.json"
POINTS = V4 / "points.json"
ANALYSIS = V4 / "analysis.json"
HANDOFF = V4 / "handoff.json"
ORIGINAL_SHA = "93bd090cf314400ed3df53ceea7c181b52337fc809054a79e4c6008b3c13c31f"


class WholeClipReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(POSE.read_text("utf-8"))
        cls.tracks = load_tracks(cls.doc)
        cls.report = analyze(cls.doc)
        cls.inventory = candidate_inventory(cls.report, cls.tracks)

    @unittest.skipUnless(VIDEO.is_file() and sha256_file(VIDEO) == ORIGINAL_SHA,
                         "needs the original bytes of sample 04; a recut from scripts/get_footage.py differs")
    def test_exact_input_binding_covers_every_aligned_source_frame(self):
        verified = verify_pose_binding(VIDEO, POSE, POINTS, ANALYSIS, HANDOFF)
        self.assertEqual(verified["source_sha256"],
                         "93bd090cf314400ed3df53ceea7c181b52337fc809054a79e4c6008b3c13c31f")
        self.assertEqual((verified["first_frame"], verified["last_frame"]), (0, 899))
        self.assertTrue(verified["timestamps_aligned"])
        self.assertLessEqual(verified["max_timestamp_error_s"], 0.001)

    def test_candidate_inventory_includes_all_four_camera_shots(self):
        self.assertEqual(
            self.inventory["camera_shots"],
            [[0, 211], [212, 366], [367, 664], [665, 899]],
        )
        self.assertIn("raw_model_score_policy", self.inventory["score_definition"])
        self.assertNotIn("confidence_policy", self.inventory["score_definition"])
        self.assertEqual(
            [row["candidate_event_count"] for row in self.inventory["shots"]],
            [6, 0, 13, 5],
        )
        self.assertEqual(self.inventory["timeline_frames_reviewed"], 900)
        self.assertEqual(self.inventory["event_counts"]["step"], 119)

    def test_top_three_are_distinct_evidence_bound_action_sequences(self):
        moments = choose_moments(self.report, self.tracks, self.inventory)
        self.assertEqual(
            [(row["rank"], row["key"], row["action_frames"], row["window"])
             for row in moments],
            [
                (1, "attack_dive", [612, 642, 664], [612, 664]),
                (2, "long_exchange", [459, 497, 545], [459, 545]),
                (3, "legs", [84, 89, 99], [60, 109]),
            ],
        )
        self.assertEqual([row["source_shot"] for row in moments],
                         [[367, 664], [367, 664], [0, 211]])
        self.assertTrue(any(row["candidate_id"] == "third-shot-attack-and-dive"
                            for row in moments))
        self.assertEqual(
            [row["candidate_id"] for row in self.inventory["ranked_candidates"][:3]],
            [
                "third-shot-attack-and-dive",
                "third-shot-long-lunge-exchange",
                "opening-near-lunge",
            ],
        )
        self.assertEqual(len(self.inventory["ranked_candidates"]), 10)
        for expected_rank, candidate in enumerate(
                self.inventory["ranked_candidates"], 1):
            self.assertEqual(candidate["combined_rank"], expected_rank)
            self.assertAlmostEqual(
                candidate["combined_score"],
                round(
                    0.40 * candidate["highlight_score"]
                    + 0.35 * candidate["coaching_value_score"]
                    + 0.25 * candidate["evidence_quality_score"],
                    2,
                ),
            )
            self.assertTrue(candidate["reason"])
        self.assertTrue(all(isinstance(row["highlight_score"], (int, float))
                            and isinstance(row["coaching_value_score"], (int, float))
                            and isinstance(row["evidence_quality_score"], (int, float))
                            and isinstance(row["combined_rank"], int)
                            for row in self.inventory["ranked_candidates"]))
        quiet = next(row for row in self.inventory["ranked_candidates"]
                     if row["candidate_id"] == "second-shot-step-only")
        self.assertEqual(quiet["window"], [212, 366])
        self.assertLess(quiet["combined_score"], 3.0)
        floor = moments[0]["evidence"]["near_ground_pose_raw_scores"]
        self.assertEqual([row["frame"] for row in floor], [642, 644, 648, 652, 656, 660, 664])
        self.assertTrue(all("raw_joint_model_scores" in row for row in floor))

    def test_schedule_maps_action_frames_without_replay_or_cut_crossing(self):
        moments = choose_moments(self.report, self.tracks, self.inventory)
        schedule = make_schedule(moments, [1.8, 2.2, 2.0, 1.2], tracks=self.tracks)
        self.assertEqual(len([row for row in schedule["chapters"]
                              if row["rank"] is not None]), 3)
        for moment in moments:
            mapped = [
                frame for frame, state in zip(schedule["frame_map"], schedule["states"])
                if state["chapter"] == moment["key"]
            ]
            self.assertEqual(mapped, sorted(mapped))
            self.assertEqual((min(mapped), max(mapped)), tuple(moment["window"]))
            self.assertTrue(set(moment["action_frames"]).issubset(mapped))
            start, end = moment["window"]
            self.assertEqual(
                sum(shot[0] <= start and end <= shot[1] for shot in self.tracks.shots),
                1,
            )
            hold = moment["playback"]
            self.assertEqual(hold["key_frame"], moment["action_frames"][1])
            self.assertGreaterEqual(hold["hold_s"], 1.5)
            self.assertLessEqual(hold["hold_s"], 2.5)
            self.assertGreaterEqual(hold["slow_in_speed"], 0.25)
            self.assertLess(hold["slow_in_speed"], 1.0)
            self.assertEqual(len(hold["hold_points"]), 3)
            hold_index = hold["hold_output_frame"]
            self.assertEqual(schedule["frame_map"][hold_index], hold["key_frame"])
            self.assertEqual(schedule["states"][hold_index]["phase"], "hold")
            self.assertEqual(schedule["states"][hold_index]["hold_points"], hold["hold_points"])
            follow_through = [
                state for state in schedule["states"]
                if state["chapter"] == moment["key"]
                and state["playback_phase"] == "follow-through"
            ]
            self.assertTrue(follow_through)
            self.assertTrue(all(
                state["phase"] == "motion" and state["speed"] == 1.0
                for state in follow_through
            ))
            self.assertLessEqual(hold["hold_output_range_s"][1],
                                 schedule["chapters"][moment["rank"] - 1]["output_end_s"])
        self.assertEqual(schedule["states"][0]["phase"], "intro")
        self.assertEqual(schedule["states"][-1]["phase"], "summary")

    def test_v3_crops_and_draws_only_supported_on_body_joints(self):
        moments = choose_moments(self.report, self.tracks, self.inventory)
        schedule = make_schedule(moments, [1.8, 2.2, 2.0, 1.2], tracks=self.tracks)
        states = schedule["states"]
        moment_one = [state for state in states if state["chapter"] == "attack_dive"]
        wide = next(state for state in moment_one if state["source_frame"] == 612)
        overhead = next(state for state in moment_one if state["source_frame"] == 629)
        dive = next(state for state in moment_one if state["source_frame"] == 636)
        self.assertEqual((wide["crop_mode"], wide["focus"]), ("full", None))
        self.assertEqual(overhead["overlays"], [{"role": "far", "kind": "right-arm"}])
        self.assertEqual((dive["crop_mode"], dive["focus"]), ("focus", "near"))
        self.assertEqual(dive["overlays"], [{"role": "near", "kind": "dive"}])

        long_exchange = next(row for row in moments if row["key"] == "long_exchange")
        crop = _shared_court_crop(self.tracks, long_exchange["window"])
        self.assertEqual(crop, long_exchange["shared_court_crop"])
        self.assertLess(crop[0], crop[2])
        self.assertLess(crop[1], crop[3])
        for role in self.tracks.players.values():
            for frame in range(459, 546):
                if role.measured[frame]:
                    box = role.bbox[frame]
                    self.assertTrue(
                        crop[0] <= box[0] <= box[2] <= crop[2]
                        and crop[1] <= box[1] <= box[3] <= crop[3]
                    )

        near = self.tracks.players[Role.NEAR]
        self.assertEqual(_supported_joint(near, 642, "left_elbow"),
                         (None, "below-score-threshold"))
        self.assertIsNotNone(_supported_joint(near, 642, "right_elbow")[0])
        self.assertEqual(_supported_joint(near, 644, "right_elbow"),
                         (None, "unmeasured"))
        image = Image.new("RGB", (1920, 1080))
        import numpy as np
        import cv2
        frame = np.asarray(image)[:, :, ::-1].copy()
        overlay = _overlay_supported(frame, self.tracks, 648,
                                     {"role": "near", "kind": "dive"})
        self.assertIn("right_wrist", overlay["drawn_joints"])
        self.assertNotIn("left_wrist", overlay["rejected"])
        self.assertIn("left_shoulder", overlay["drawn_joints"])

    def test_report_has_three_or_four_short_analysis_points_and_agent_scoring_label(self):
        moments = choose_moments(self.report, self.tracks, self.inventory)
        self.assertEqual(
            [len(moment["analysis_points"]) for moment in moments],
            [4, 4, 3],
        )
        self.assertNotIn("手臂", [point["area"] for point in moments[1]["analysis_points"]])
        exchange = moments[1]["evidence"]["court_position_evidence_2d"]["players"]
        self.assertAlmostEqual(exchange["near"]["net_displacement_2d_m"], 3.2, places=1)
        self.assertAlmostEqual(exchange["far"]["net_displacement_2d_m"], 2.7, places=1)
        self.assertIn("二维位移约", moments[1]["narration"])
        self.assertNotIn("不判断", moments[1]["narration"])
        self.assertIn("未经人工复核", self.inventory["score_definition"]["scale"])
        self.assertTrue(all("未经人工复核" in candidate["score_scale"]
                            for candidate in self.inventory["ranked_candidates"]))
        font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 64)
        draw = ImageDraw.Draw(Image.new("RGB", (1080, 100)))
        for moment in moments:
            title = f'第{moment["rank"]}处｜{moment["title"]}'
            self.assertEqual(len(_wrap_zh(draw, title, font, 900)), 1, title)

    def test_raw_json_evidence_stays_in_collapsed_html_details(self):
        moments = choose_moments(self.report, self.tracks, self.inventory)
        schedule = make_schedule(moments, [1.8, 2.2, 2.0, 1.2], tracks=self.tracks)
        data = _analysis_data(
            VIDEO,
            {"source_sha256": "test"},
            self.tracks,
            self.inventory,
            moments,
            schedule,
            {"full": "source credit", "endcard": "source credit"},
        )
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            _write_analysis(folder, data)
            page = (folder / "analysis.html").read_text("utf-8")
        self.assertIn("<details>", page)
        self.assertIn("展开原始关键点、事件与帧绑定 JSON", page)
        self.assertIn("展开全时间线原始候选事件 JSON", page)
        self.assertIn("未经人工复核", page)
        self.assertNotIn("人工编辑评分", page)


if __name__ == "__main__":
    unittest.main()
