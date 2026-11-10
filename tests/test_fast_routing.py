import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fast"))
from pipeline import Models, Segment


class Routing(unittest.TestCase):
    def setUp(self):
        self.segment = Segment.__new__(Segment)
        self.segment.pose_small_req = object()
        self.segment.small_confidence = 1.0
        self.segment.small_h = 100
        self.segment.calls = {"fallback": 0, "tail_fallback": 0}
        self.flags = []
        self.tiny_confidence = 1.2
        self.weak_joint = None
        self.large = np.tile([60.0, 80.0, 1.3], (17, 1))

        def inference(crops, meta, flags):
            self.flags.append(flags)
            out = [np.tile([20.0, 40.0, self.tiny_confidence], (17, 1)) if flag else self.large.copy()
                   for flag in flags]
            if self.weak_joint is not None:
                for flag, k in zip(flags, out):
                    if flag:
                        k[9, 2] = self.weak_joint
            return out
        self.segment.pose_crops = inference

    def route(self, track):
        return self.segment.route_poses([None], [None], [[0, 0, 50, 150]], [track])[0]

    def test_confident_tiny_pose_is_used_without_second_inference(self):
        result = self.route({})
        self.assertEqual(self.flags, [[True]])
        np.testing.assert_array_equal(result, np.tile([20.0, 40.0, 1.2], (17, 1)))

    def test_uncertain_tiny_pose_is_replaced_on_the_same_frame(self):
        self.tiny_confidence = 0.7
        track = {}
        result = self.route(track)
        self.assertEqual(self.flags, [[True], [False]])
        np.testing.assert_array_equal(result, self.large)
        self.assertEqual(track["large_left"], 8)
        self.assertEqual(self.segment.calls["fallback"], 1)
        self.assertEqual(self.segment.calls["tail_fallback"], 0)

    def test_isolated_weak_joint_retries_now_without_holding_the_track(self):
        self.weak_joint = 0.64
        track = {}
        np.testing.assert_array_equal(self.route(track), self.large)
        self.assertEqual(self.flags, [[True], [False]])
        self.assertNotIn("large_left", track)
        self.assertEqual(self.segment.calls["tail_fallback"], 1)
        self.route(track)
        self.assertEqual(self.flags[-2:], [[True], [False]])

    def test_joint_at_threshold_does_not_add_a_strong_pass(self):
        self.weak_joint = 0.65
        self.route({})
        self.assertEqual(self.flags, [[True]])
        self.assertEqual(self.segment.calls["fallback"], 0)

    def test_difficult_track_uses_large_for_eight_inferences_then_reprobes(self):
        track = {"large_left": 8}
        for _ in range(8):
            np.testing.assert_array_equal(self.route(track), self.large)
        self.assertEqual(self.flags, [[False]] * 8)
        self.assertEqual(track["large_left"], 0)
        self.route(track)
        self.assertEqual(self.flags[-1], [True])

    def test_height_routing_is_preserved_when_confidence_routing_is_off(self):
        self.segment.small_confidence = 0
        np.testing.assert_array_equal(self.route({}), self.large)
        self.assertEqual(self.flags, [[False]])
        self.assertTrue(self.segment.small([0, 0, 20, 50]))


class LazyFollowers(unittest.TestCase):
    def test_followers_are_not_compiled_until_needed_and_then_are_cached(self):
        with patch("pipeline.ov.Core"), patch("pipeline.Pose") as pose, patch("pipeline.Detector") as det:
            models = Models("s", "start", "det", streams=8, pose_small_onnx="t")
            self.assertEqual(pose.call_count, 2)
            self.assertEqual(det.call_count, 2)
            first = models.one
            self.assertEqual(pose.call_count, 4)
            self.assertEqual(det.call_count, 3)
            self.assertIs(models.one, first)
            self.assertEqual(pose.call_count, 4)
            self.assertEqual(det.call_count, 3)


if __name__ == "__main__":
    unittest.main()
