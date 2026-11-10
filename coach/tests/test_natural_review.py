from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from coach.natural_review import (
    _gate_rows,
    _has_faststart_atom,
    _inside,
    _pixel_audit_passed,
    _position_parts,
    _pose_moment_for_segment,
    _prepare_schedule,
    _unique_sample,
    _write_final_maps,
)


class NaturalReviewTests(unittest.TestCase):
    def test_fractional_source_position_maps_to_two_adjacent_frames(self):
        self.assertEqual(_position_parts(4.0, 10), (4, 5, 0.0))
        self.assertEqual(_position_parts(4.25, 10), (4, 5, 0.25))
        self.assertEqual(_position_parts(9.75, 10), (9, 9, 0.0))

    def test_final_frame_map_preserves_plan_positions_including_recap_frames(self):
        planned = {
            "fps": 30,
            "source_position_per_output_frame": [2.0, 2.25, 8.0],
            "source_frame_per_output_frame": [2, 2, 8],
        }
        segments = [{
            "name": "recap.card",
            "start_s": 0.0,
            "end_s": 0.1,
            "kind": "card",
        }] * 3
        schedule = {
            "moments": [{"id": "legs"}],
            "source_positions": planned["source_position_per_output_frame"],
            "recap_card_rect": [1, 2, 3, 4],
            "captions": [{"line_id": "recap", "text": "记住三点"}],
        }
        script = {"title": "三个瞬间", "voice": {"engine": "kokoro"}}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            saved_schedule, states = _write_final_maps(
                folder, {"captions": []}, planned, segments, script, schedule, 10,
                list(range(10)), "a" * 64, ((0, 9),),
            )
            frame_map = json.loads((folder / "frame-map.json").read_text("utf-8"))
        self.assertEqual(frame_map["source_position_per_output_frame"],
                         planned["source_position_per_output_frame"])
        self.assertEqual(frame_map["source_frame_per_output_frame"],
                         planned["source_frame_per_output_frame"])
        self.assertTrue(all(row["source_frame"] is not None for row in states))
        self.assertEqual(states[1]["blend_frames"], [2, 3])
        self.assertEqual(states[1]["blend_weight"], 0.25)
        self.assertEqual(frame_map["unique_position_per_output_frame"],
                         [None, 2.25, None])
        self.assertEqual(frame_map["picture_model"], "unique_blend_v1")
        self.assertEqual(frame_map["source_duplicates_sha256"], "a" * 64)
        self.assertEqual(saved_schedule["recap_card_rect"], [1, 2, 3, 4])
        self.assertEqual(saved_schedule["moments"], [{"id": "legs"}])

    def test_fractional_picture_samples_interpolate_over_unique_knots(self):
        knots = [0, 1, 3, 4, 6]
        first, second, weight, q = _unique_sample(2.0, knots)
        self.assertEqual((first, second, weight, q), (2, 2, 0.0, None))
        first, second, weight, q = _unique_sample(3.5, knots)
        self.assertEqual((first, second), (3, 4))
        self.assertEqual(weight, 0.5)
        self.assertEqual(q, 2.5)

    def test_integer_picture_position_remains_raw_even_if_duplicate(self):
        self.assertEqual(_unique_sample(2.0, [0, 1, 3, 4]), (2, 2, 0.0, None))

    def test_unique_picture_samples_must_stay_inside_camera_shot(self):
        with self.assertRaisesRegex(ValueError, "crosses a camera cut"):
            _unique_sample(3.5, [0, 1, 3, 4, 6], shots=((0, 3), (4, 6)))

    def test_pixel_audit_gate_requires_all_picture_and_smoothness_checks(self):
        lines = [
            "VERIFIED picture_model_declared model declared",
            "VERIFIED frame_binding bound",
            "VERIFIED blend_weight fitted",
            "VERIFIED binding_control_rejects_wrong_map controlled",
            "VERIFIED smoothness each segment passes",
        ]
        self.assertTrue(_pixel_audit_passed(lines))
        self.assertFalse(_pixel_audit_passed(lines[:-1]))
        self.assertFalse(_pixel_audit_passed([
            line.replace("VERIFIED blend_weight", "NOT VERIFIED blend_weight")
            for line in lines
        ]))

    def test_plan_rejects_blended_frame_that_crosses_camera_cut(self):
        tracks = SimpleNamespace(n=8, shots=((0, 3), (4, 7)))
        planned = {
            "source_position_per_output_frame": [3.5],
            "source_frame_per_output_frame": [3],
        }
        with self.assertRaisesRegex(ValueError, "crosses a camera cut"):
            _prepare_schedule(
                {"timeline": {"moments": []}},
                {"segments": []},
                {"captions": []},
                planned,
                tracks,
                {},
            )

    def test_faststart_requires_moov_before_media_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            fast = Path(temporary) / "fast.mp4"
            slow = Path(temporary) / "slow.mp4"
            fast.write_bytes(b"ftypxxxxmoovxxxxmdat")
            slow.write_bytes(b"ftypxxxxmdatxxxxmoov")
            self.assertTrue(_has_faststart_atom(fast))
            self.assertFalse(_has_faststart_atom(slow))

    def test_text_clearance_accepts_separation_and_rejects_overlap(self):
        person = [100.0, 100.0, 200.0, 220.0]
        self.assertTrue(_inside([10.0, 10.0, 90.0, 90.0], person))
        self.assertFalse(_inside([90.0, 90.0, 120.0, 120.0], person))

    def test_nested_script_gate_report_counts_leaf_gates(self):
        gates = {
            "voice": {"hook": {"pass": True}, "line": {"pass": True}},
            "plan": {"window": {"pass": False}},
        }
        rows = list(_gate_rows(gates))
        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(not row["pass"] for row in rows), 1)

    def test_skeleton_is_limited_to_slow_hold_and_follow_phases(self):
        self.assertIsNone(_pose_moment_for_segment("legs.lead"))
        for phase in ("slow_in", "hold", "follow"):
            self.assertEqual(_pose_moment_for_segment(f"legs.{phase}"), "legs")
        self.assertIsNone(_pose_moment_for_segment("recap.card"))


if __name__ == "__main__":
    unittest.main()
