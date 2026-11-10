from io import StringIO
import json
import unittest

import numpy as np

from badminton_coach.__main__ import (
    COCO17_NAMES,
    joint_record,
    point_in_polygon,
    validate_polygon,
    write_frames_json_end,
    write_frames_json_start,
)
from badminton_coach.overlay import COCO17_EDGES


class AnalysisContractTests(unittest.TestCase):
    def test_coco_layout_has_17_named_points_and_valid_edges(self):
        self.assertEqual(17, len(COCO17_NAMES))
        self.assertTrue(all(0 <= joint < 17 for edge in COCO17_EDGES for joint in edge))

    def test_low_confidence_joints_are_missing_not_filled(self):
        raw = np.asarray([[12.0, 34.0, 0.9]] * 17, dtype=np.float32)
        raw[3] = [0.0, 0.0, 0.1]
        joints = joint_record(raw, 0.25)
        self.assertEqual([12.0, 34.0, 0.9], joints[0])
        self.assertIsNone(joints[3])
        self.assertEqual(17, len(joints))

    def test_streamed_keypoint_header_is_valid_json(self):
        output = StringIO()
        write_frames_json_start(output, {"schema_version": 1, "source": "clip.mp4"})
        output.write('{"frame_index": 0, "players": []}\n')
        write_frames_json_end(output)
        parsed = json.loads(output.getvalue())
        self.assertEqual(1, parsed["schema_version"])
        self.assertEqual([{"frame_index": 0, "players": []}], parsed["frames"])

    def test_court_contains_inside_point_and_rejects_adjacent_court(self):
        court = [(100.0, 100.0), (500.0, 100.0), (550.0, 500.0), (50.0, 500.0)]
        self.assertTrue(point_in_polygon((300.0, 400.0), court))
        self.assertFalse(point_in_polygon((650.0, 400.0), court))

    def test_court_rejects_crossed_or_too_small_points(self):
        with self.assertRaisesRegex(ValueError, "convex"):
            validate_polygon(
                [(50, 50), (500, 400), (500, 50), (50, 400)], 640, 480,
            )
        with self.assertRaisesRegex(ValueError, "too small"):
            validate_polygon(
                [(1, 1), (10, 1), (10, 10), (1, 10)], 640, 480,
            )


if __name__ == "__main__":
    unittest.main()
