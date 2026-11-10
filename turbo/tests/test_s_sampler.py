"""Interpolation and stitch rules. No video, no models."""
import unittest

import sampler


def joints(conf, x=10.0):
    return [[x, 20.0, conf] for _ in range(17)]


class FillTests(unittest.TestCase):
    def test_measured_stays_measured_and_gap_uses_lower_confidence(self):
        measured = {
            0: [(1, [0, 0, 10, 20], 0.9, joints(0.8, 0))],
            4: [(1, [4, 0, 14, 20], 0.4, joints(0.5, 4))],
        }
        frames = sampler.fill_frames(measured, 5, 30, max_gap=48)
        self.assertEqual([f["frame_index"] for f in frames], list(range(5)))
        self.assertNotIn("src", frames[0]["players"][0])
        mid = frames[2]["players"][0]
        self.assertEqual(mid["src"], "interpolated")
        self.assertEqual(mid["joints"][0][2], 0.5)
        self.assertEqual(mid["joints"][0][0], 2.0)
        self.assertEqual(mid["box_confidence"], 0.4)
        self.assertEqual(mid["timestamp_seconds"], round(2 / 30, 4))

    def test_missing_anchor_joint_stays_missing(self):
        left = joints(0.9)
        right = joints(0.9)
        left[3] = None
        right[4] = [1, 2, 0.1]
        out = sampler._lerp_joints(left, right, 0.5)
        self.assertIsNone(out[3])
        self.assertIsNone(out[4])
        self.assertIsNotNone(out[0])

    def test_long_gap_and_edges_are_not_guessed(self):
        measured = {2: [(1, [0, 0, 10, 20], 0.9, joints(0.8))],
                    20: [(1, [8, 0, 18, 20], 0.9, joints(0.8))]}
        frames = sampler.fill_frames(measured, 24, 30, max_gap=8)
        self.assertEqual(frames[0]["players"], [])
        self.assertEqual(frames[10]["players"], [])
        self.assertEqual(frames[23]["players"], [])
        self.assertNotIn("src", frames[2]["players"][0])

    def test_low_confidence_joint_is_absent(self):
        row = sampler.player_row(1, [0, 0, 1, 1], 0.5, [[1, 2, 0.1]], 0, 30, "measured")
        self.assertIsNone(row["joints"][0])
        self.assertNotIn("src", row)


class StitchTests(unittest.TestCase):
    def test_boundary_overlap_keeps_one_id(self):
        box = [0, 0, 10, 20]
        gop_rows = [
            {0: [(7, box, 0.9, joints(0.8))]},
            {10: [(3, [1, 0, 11, 20], 0.9, joints(0.8))]},
        ]
        gid = sampler.stitch_ids(gop_rows)
        self.assertEqual(gid[(0, 7)], gid[(1, 3)])
        merged = sampler.apply_ids(gop_rows, gid)
        self.assertEqual(merged[0][0][0], merged[10][0][0])

    def test_far_boxes_stay_apart(self):
        gop_rows = [
            {0: [(1, [0, 0, 10, 10], 0.9, joints(0.8))]},
            {8: [(1, [100, 100, 110, 110], 0.9, joints(0.8))]},
        ]
        gid = sampler.stitch_ids(gop_rows)
        self.assertNotEqual(gid[(0, 1)], gid[(1, 1)])


if __name__ == "__main__":
    unittest.main()
