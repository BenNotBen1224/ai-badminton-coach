import importlib.util
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SPEC = importlib.util.spec_from_file_location("comparison_runner", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RecordTests(unittest.TestCase):
    def test_coordinates_and_unknown_id_remain_honest(self):
        raw = np.tile([1000., 500., .8], (17, 1))
        raw[9] = [1010., 540., .1]
        result = runner.player([900, 400, 1100, 600], raw, .3)
        self.assertEqual(result["track_id"], None)
        self.assertEqual(result["joints"][0], [1000., 500., .8])
        self.assertIsNone(result["joints"][9])
        self.assertEqual(result["raw_joints"][9], [1010., 540., .1])

    def test_court_filter_uses_footpoint(self):
        raw = np.tile([20., 20., .8], (17, 1))
        self.assertIsNone(runner.player([0, 0, 50, 50], raw, .8))

    def test_box_overlap(self):
        self.assertEqual(runner.overlap([0, 0, 10, 10], [0, 0, 10, 10]), 1.)
        self.assertEqual(runner.overlap([0, 0, 10, 10], [20, 20, 30, 30]), 0.)
        self.assertAlmostEqual(runner.overlap([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)

    def test_complete_frame_sequence_required(self):
        with self.assertRaisesRegex(ValueError, "780"):
            runner.validate_data({"frames": [{"frame_index": 0, "players": []}]})
        runner.validate_data({"frames": [{"frame_index": i, "players": []} for i in range(780)]})

    def test_joint_layout_must_be_coco17(self):
        with self.assertRaisesRegex(ValueError, "17 COCO"):
            runner.player([900, 400, 1100, 600], np.ones((16, 3)), .8)

    def test_near_player_cannot_suppress_far_retry(self):
        near = {"bbox_xyxy": [432.59, 409.8, 536.09, 652.84], "box_confidence": .89835}
        self.assertTrue(runner.needs_far_retry([near]))
        far = {"bbox_xyxy": [740, 380, 810, 550], "box_confidence": .2185}
        self.assertTrue(runner.needs_far_retry([near, far]))
        far["box_confidence"] = .7959
        self.assertFalse(runner.needs_far_retry([near, far]))

    def test_gallery_versions_video_and_inspection_images(self):
        methods = {name: {**deepcopy(config), "status": "pending"} for name, config in runner.METHODS.items()}
        methods["crop"].update(status="complete", video_sha256="1234567890abcdef1234567890abcdef")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner.save(root / "manifest.json", {"methods": methods})
            with patch.object(runner, "HERE", root):
                runner.gallery()
            text = (root / "index.html").read_text(encoding="utf-8")
        self.assertIn('src="results/crop/annotated.mp4?v=1234567890abcdef1234"', text)
        self.assertIn('poster="results/crop/frame-8s.jpg?v=1234567890abcdef1234"', text)
        self.assertIn('src="results/crop/frame-22s.jpg?v=1234567890abcdef1234"', text)
        self.assertNotIn('src="results/crop/annotated.mp4"', text)


if __name__ == "__main__":
    unittest.main()
