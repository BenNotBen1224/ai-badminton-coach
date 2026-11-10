import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fast"))
from tail_candidate import route_tail
from risk_features import crop_features, features


class Candidate(unittest.TestCase):
    def setUp(self):
        self.calls = {"fallback": 0}
        self.small_confidence = 1.05
        self.flags = []
        self.tiny = np.tile([20., 40., 1.2], (17, 1))
        self.tiny[9, 2] = .64
        self.strong = np.tile([60., 80., 1.3], (17, 1))

    def pose_crops(self, crops, meta, flags):
        self.flags.append(flags)
        return [self.tiny.copy() if flag else self.strong.copy() for flag in flags]

    def run_route(self, track, risk=None):
        return route_tail(self, [None], [(np.array([96., 128.]), np.array([192., 256.]))],
                          [None], [track], .65, risk=risk)[0]

    def test_isolated_weak_joint_only_retries_the_current_crop(self):
        track = {}
        np.testing.assert_array_equal(self.run_route(track), self.strong)
        self.assertNotIn("large_left", track)
        self.assertEqual(self.flags, [[True], [False]])

    def test_mean_weak_crop_still_gets_existing_eight_inference_hold(self):
        self.tiny[:, 2] = .7
        track = {}
        np.testing.assert_array_equal(self.run_route(track, risk={"value": -1}), self.strong)
        self.assertEqual(track["large_left"], 8)

    def test_negative_risk_leaf_avoids_an_extra_tail_pass(self):
        np.testing.assert_array_equal(self.run_route({}, risk={"value": -1}), self.tiny)
        self.assertEqual(self.flags, [[True]])
        self.assertEqual(self.calls["fallback"], 0)

    def test_runtime_features_match_training_crop_features(self):
        center = np.array([400., 600.])
        scale = np.array([300., 400.])
        image = self.tiny.copy()
        image[:, :2] = (image[:, :2] / [192, 256] - .5) * scale + center
        np.testing.assert_allclose(crop_features(image, center, scale), features(self.tiny[None])[0],
                                   atol=1e-12)


if __name__ == "__main__":
    unittest.main()
