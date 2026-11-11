from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from coach.analyze import analyze
from coach.coaching_corpus.corpus_learn import HEDGE_ZH
from coach.editorial_short import (
    FPS,
    TACTICS_VIEW,
    _draw_text,
    _nearest_measured,
    _overlay,
    _role,
    _synthesize,
    _wrap_editorial,
    _write_analysis,
    build_story,
    crop_geometry,
    make_schedule,
    render_frame,
)
from coach.complete_review import VOICE
from coach.player_input import player_inputs
from coach.schema import Role

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "sample-videos" / "04-china-open-ws-a.mp4"
POSE = ROOT / "fast" / "runs" / "step4-rebalanced-dev" / "outputs" / "04-r1.json"


class EditorialShortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(POSE.read_text("utf-8"))
        cls.doc["source"]["players"] = player_inputs(VIDEO)
        cls.report = analyze(cls.doc)
        cls.tracks, cls.story = build_story(cls.doc, cls.report)
        speeches = _synthesize(
            [(chapter.key, chapter.narration) for chapter in cls.story["chapters"]],
            VOICE,
        )
        cls.schedule = make_schedule(
            cls.story,
            [len(entry["samples"]) / entry["rate"] for entry in speeches],
            cls.tracks,
        )

    def test_story_uses_one_pass_through_four_topics_and_a_four_line_recap(self):
        self.assertEqual(
            [(chapter.key, chapter.title) for chapter in self.story["chapters"]],
            [
                ("legs", "腿"),
                ("core", "核心"),
                ("tactics", "战术"),
                ("arms", "手臂"),
                ("recap", "收尾总结"),
            ],
        )
        legs = self.story["evidence"]["legs"]
        self.assertEqual(
            (legs["plant_frame"], legs["bottom_frame"], legs["return_frame"]),
            (86, 89, 99),
        )
        self.assertAlmostEqual(legs["knee_flexion_2d_deg"], 113.86665479734488)
        core = self.story["evidence"]["core"]["frames"]
        self.assertEqual([row["frame"] for row in core], [80, 92, 107])
        self.assertLess(core[0]["trunk_tilt_2d_deg"], 10)
        self.assertGreater(core[1]["trunk_tilt_2d_deg"], 30)
        self.assertGreater(
            core[1]["shoulder_line_tilt_2d_deg"],
            core[1]["hip_line_tilt_2d_deg"] + 10,
        )
        self.assertLess(core[2]["trunk_tilt_2d_deg"], 10)
        self.assertEqual(
            self.story["chapters"][-1].evidence["lines"],
            [
                "腿｜右脚跨出，前膝屈曲",
                "手臂｜左手先收，右臂上挥",
                "核心｜上身倾斜，再回正",
                "战术｜回到中部，准备下一步",
            ],
        )
        chapters = self.story["chapters"]
        for chapter in chapters[:3]:
            self.assertEqual(sum(clip.phase == "motion" for clip in chapter.clips), 1)
        self.assertEqual(sum(clip.phase == "motion" for clip in chapters[3].clips), 3)
        ranges = [
            (chapter.clips[0].start, chapter.clips[0].end)
            for chapter in chapters[:3]
        ]
        self.assertEqual(ranges, [(68, 91), (92, 119), (120, 210)])

    def test_arm_order_is_another_shot_and_both_arms_are_measured(self):
        evidence = self.story["evidence"]["arms"]
        self.assertEqual(evidence["shot"], [367, 664])
        self.assertEqual(
            (evidence["left_peak"], evidence["left_return"], evidence["right_peak"]),
            (617, 624, 629),
        )
        self.assertGreaterEqual(evidence["left_confidence"], 0.5)
        self.assertGreaterEqual(evidence["left_return_confidence"], 0.5)
        self.assertGreaterEqual(evidence["right_confidence"], 0.5)
        self.assertIn("左手先抬再收", self.story["chapters"][3].narration)
        self.assertIn("顺势上挥", evidence["claim"])
        self.assertEqual(self.story["chapters"][3].shot, (367, 664))
        self.assertLess(
            max(clip.end for clip in self.story["chapters"][2].clips),
            min(clip.start for clip in self.story["chapters"][3].clips),
        )

    def test_tactics_positions_use_the_recomputed_court_projection(self):
        tactics = self.story["evidence"]["tactics"]
        self.assertAlmostEqual(tactics["near_settled"]["xy"][0], 0.4112, delta=0.02)
        self.assertAlmostEqual(tactics["near_settled"]["xy"][1], 0.7517, delta=0.03)
        self.assertLess(tactics["far_settled"]["xy"][1], 0.10)
        self.assertEqual(tactics["near_settled"]["n"], 76)
        self.assertEqual(tactics["far_settled"]["n"], 78)

    def test_arm_overlays_show_only_the_measured_arm_phase(self):
        far = self.tracks.players[Role.FAR]
        frame = np.zeros((self.tracks.height, self.tracks.width, 3), dtype=np.uint8)
        free = _overlay(frame, self.tracks, 617, "far", "arms-free")
        racket = _overlay(frame, self.tracks, 629, "far", "arms-racket")
        left_wrist_617 = np.rint(far.joint("left_wrist")[617]).astype(int)
        right_wrist_617 = np.rint(far.joint("right_wrist")[617]).astype(int)
        left_wrist_629 = np.rint(far.joint("left_wrist")[629]).astype(int)
        right_wrist_629 = np.rint(far.joint("right_wrist")[629]).astype(int)
        self.assertEqual(tuple(free[left_wrist_617[1], left_wrist_617[0]]), (255, 190, 60))
        self.assertEqual(tuple(free[right_wrist_617[1], right_wrist_617[0]]), (0, 0, 0))
        self.assertEqual(tuple(racket[right_wrist_629[1], right_wrist_629[0]]), (70, 235, 220))
        self.assertEqual(tuple(racket[left_wrist_629[1], left_wrist_629[0]]), (0, 0, 0))

    def test_tactics_map_is_outside_the_footage_and_wrapping_avoids_leading_punctuation(self):
        self.assertLessEqual(TACTICS_VIEW[2], 672 - 12)
        image = Image.new("RGB", (1080, 1920))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 44)
        text = "她回到半场中部、稍偏反手；对手在后场。练单打时，把回位和下一次向四边移动连起来。"
        lines = _wrap_editorial(draw, text, font, 810)
        self.assertEqual("".join(lines), text)
        self.assertTrue(all(not line[0] in "，。、；：？！" for line in lines if line))
        _draw_text(draw, text, (78, 1310), font, "#ffffff", width=810, line_h=58)

    def test_all_spoken_lines_are_direct_and_do_not_claim_unseen_shuttle_events(self):
        for chapter in self.story["chapters"]:
            self.assertIsNone(HEDGE_ZH.search(chapter.narration))
            self.assertNotRegex(chapter.narration, r"触球|落点|得分|对手想")

    def test_schedule_binds_chapter_ranges_captions_and_source_frames(self):
        schedule = self.schedule
        self.assertLessEqual(schedule["planned_duration_s"], 45)
        self.assertLessEqual(schedule["paused_share"], 0.35)
        self.assertLessEqual(schedule["longest_hold_s"], 1.0)
        self.assertEqual(schedule["summary_s"], 5.5)
        self.assertLess(schedule["planned_duration_s"], 20)
        for row in schedule["chapters"]:
            caption = next(item for item in schedule["captions"]
                           if item["chapter"] == row["key"])
            self.assertEqual(caption["start_s"], caption["speech_start_s"])
            self.assertEqual(caption["end_s"], caption["speech_end_s"])
            self.assertGreaterEqual(caption["start_s"], row["output_start_s"])
            self.assertLessEqual(caption["end_s"], row["output_end_s"])
            for source_range in row["source_ranges"]:
                begin = round(source_range["output_start_s"] * FPS)
                end = round(source_range["output_end_s"] * FPS)
                states = schedule["states"][begin:end]
                frames = schedule["frame_map"][begin:end]
                self.assertTrue(states)
                self.assertTrue(all(state["chapter"] == row["key"] for state in states))
                self.assertTrue(all(
                    source_range["source_shot"][0] <= frame <= source_range["source_shot"][1]
                    for frame in frames
                ))
                if source_range["phase"] in ("hold", "summary"):
                    self.assertTrue(all(frame == source_range["source_frame"] for frame in frames))
                else:
                    self.assertTrue(all(
                        source_range["start_frame"] <= frame <= source_range["end_frame"]
                        for frame in frames
                    ))
        recap = schedule["chapters"][-1]
        self.assertEqual(recap["source_ranges"][0]["phase"], "summary")
        recap_states = schedule["states"][
            round(recap["output_start_s"] * FPS):round(recap["output_end_s"] * FPS)
        ]
        self.assertTrue(recap_states)
        self.assertTrue(all(state["phase"] == "summary" for state in recap_states))
        self.assertTrue(all(state["recap_lines_visible"] == 4 for state in recap_states))
        motion = [frame for state, frame in zip(schedule["states"], schedule["frame_map"])
                  if state["phase"] == "motion"]
        self.assertTrue(all(right >= left for left, right in zip(motion, motion[1:])))

    def test_summary_card_replaces_footage_and_shows_all_four_lines(self):
        state = next(state for state in self.schedule["states"] if state["phase"] == "summary")
        fonts = tuple(ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", size)
                      for size in (72, 42, 44, 28, 52))
        source = np.full((self.tracks.height, self.tracks.width, 3), 255, dtype=np.uint8)
        rendered, boxes, _ = render_frame(
            source, self.tracks, state, self.schedule, self.story, fonts, "来源信息",
            self.schedule["chapters"][-1]["output_start_s"] + 0.5,
        )
        self.assertFalse(np.all(rendered == 255))
        self.assertEqual(fonts[4].size, 52)
        recap_boxes = boxes[1:5]
        self.assertEqual(len(recap_boxes), 4)
        self.assertTrue(all(78 <= box[0] and box[2] <= 900 for box in recap_boxes))
        self.assertTrue(all(box[3] < 1200 for box in recap_boxes))
        self.assertEqual([box[1] for box in recap_boxes], [532, 642, 752, 862])

    def test_analysis_maps_each_claim_and_court_map_anchor_to_output_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            analysis = _write_analysis(
                Path(temporary), self.schedule, self.story["evidence"],
                {"full": "test attribution"}, {}, {},
            )
        chapters = {chapter["chapter"]: chapter for chapter in analysis["chapters"]}
        self.assertEqual(
            chapters["腿"]["source_range_bindings"][0]["output_time_s"],
            [0.0, 2.3],
        )
        core_frames = chapters["核心"]["evidence_frame_output_times"]
        self.assertEqual(core_frames[0]["output_time_s"], 1.1666666666666667)
        map_frame = next(
            row for row in chapters["战术"]["evidence_frame_output_times"]
            if row["source_frame"] == 60
        )
        self.assertEqual(map_frame["representation"], "court_map")
        self.assertEqual(map_frame["output_time_s"], 5.233333333333333)

    def test_focus_crop_contains_sharp_enough_boxes_for_each_player_overlay(self):
        checked = set()
        for state in self.schedule["states"]:
            if not state["focus"]:
                continue
            player = _role(self.tracks, state["focus"])
            source_frame = state["source_frame"]
            measured = _nearest_measured(player, source_frame, tuple(state["shot"]))
            crop, scale = crop_geometry(
                player.bbox[measured], self.tracks.width, self.tracks.height,
            )
            x1, y1, x2, y2 = player.bbox[measured]
            left, top, right, bottom = crop
            self.assertLessEqual(left, x1)
            self.assertLessEqual(top, y1)
            self.assertGreaterEqual(right, x2)
            self.assertGreaterEqual(bottom, y2)
            if state["focus"] == "far":
                self.assertGreaterEqual(y2 - y1, 190)
                self.assertLessEqual(scale, 2.4)
            else:
                self.assertLessEqual(scale, 2.7)
            checked.add(state["focus"])
        self.assertEqual(checked, {"near", "far"})

    def test_no_source_range_crosses_a_camera_cut(self):
        schedule = self.schedule
        for row in schedule["chapters"]:
            for source_range in row["source_ranges"]:
                self.assertLessEqual(source_range["source_shot"][0], source_range["start_frame"])
                self.assertLessEqual(source_range["start_frame"], source_range["end_frame"])
                self.assertLessEqual(source_range["end_frame"], source_range["source_shot"][1])
        self.assertEqual(schedule["chapters"][3]["source_shots"], [[367, 664]])
        self.assertEqual(schedule["chapters"][1]["source_shots"], [[0, 211]])


class EditorialAnchorTests(unittest.TestCase):
    def test_tactics_samples_the_initial_anchor_before_the_lunge_scan(self):
        doc = json.loads(POSE.read_text("utf-8"))
        doc["source"]["players"] = player_inputs(VIDEO)
        report = analyze(doc)
        events = [
            replace(event, frame=44)
            if event.kind == "swing" and event.player == Role.NEAR and event.frame < 80
            else event
            for event in report.events
        ]
        _, story = build_story(doc, replace(report, events=events))
        tactics = story["evidence"]["tactics"]
        self.assertEqual(tactics["initial_frame"], 53)
        self.assertEqual(tactics["initial_position"]["frame"], 53)


if __name__ == "__main__":
    unittest.main()
