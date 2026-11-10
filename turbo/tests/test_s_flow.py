"""Box flow follows a shifted pattern and ignores a jump that is too large."""
import unittest

import numpy as np

import service


def _square(shift):
    img = np.full((180, 240), 30, np.uint8)
    img[40 + shift:100 + shift, 50 + shift:110 + shift] = 220
    img[55 + shift:70 + shift, 65 + shift:80 + shift] = 40
    return img


class FlowShiftTest(unittest.TestCase):
    def test_small_shift_moves_the_box(self):
        tracks = {0: {"box": [50, 40, 110, 100]}}
        service._flow_shift(_square(0), _square(8), tracks, 240, 180)
        moved = tracks[0]["box"]
        self.assertAlmostEqual(moved[0], 58, delta=2)
        self.assertAlmostEqual(moved[1], 48, delta=2)
        self.assertAlmostEqual(moved[2] - moved[0], 60, delta=0.1)

    def test_wild_jump_keeps_the_box(self):
        tracks = {0: {"box": [50, 40, 90, 80]}}
        before = list(tracks[0]["box"])
        service._flow_shift(_square(0), _square(40), tracks, 240, 180)
        self.assertEqual(tracks[0]["box"], before)


if __name__ == "__main__":
    unittest.main()
