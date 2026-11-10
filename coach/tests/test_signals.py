"""Behavior of velocity and speed. Callers pass arrays and read the result."""
from __future__ import annotations

import unittest

import numpy as np

from coach.signals import speed, velocity


def _track(n, metres_per_second, fps=30.0):
    t = np.arange(n, dtype=np.float64) / fps
    values = np.column_stack([metres_per_second * t, np.zeros(n)])
    valid = np.ones(n, dtype=bool)
    return values, t, valid


class SpeedTests(unittest.TestCase):
    def test_every_sixth_frame_repeated_stays_at_true_speed(self):
        values, t, valid = _track(30, 2.4)
        for i in range(5, 30, 6):
            values[i] = values[i - 1]
            valid[i] = False
        got = speed(values, t, valid)
        paired = np.array([0, 4, 5, 6, 11, 12, 28])
        self.assertTrue(np.all(np.isfinite(got[paired])))
        self.assertTrue(np.all(np.abs(got[paired] - 2.4) <= 0.03 * 2.4))
        self.assertTrue(np.isnan(got[29]))
        naive = np.linalg.norm(np.diff(values, axis=0), axis=1) * 30.0
        self.assertAlmostEqual(naive[4], 0.0, places=6)
        self.assertAlmostEqual(naive[5], 4.8, places=6)
        self.assertAlmostEqual(got[5], 2.4, places=6)
        self.assertAlmostEqual(got[6], 2.4, places=6)

    def test_seven_frame_repeat_run_does_not_invent_a_peak(self):
        values, t, valid = _track(24, 2.4)
        for i in range(10, 17):
            values[i] = values[9]
            valid[i] = False
        values[17] = values[9]
        for i in range(18, 24):
            values[i] = values[17] + np.array([2.4 * (t[i] - t[17]), 0.0])
        got = speed(values, t, valid)
        self.assertAlmostEqual(got[9], 2.4, places=6)
        self.assertTrue(np.isnan(got[10]))
        self.assertTrue(np.isnan(got[16]))
        self.assertAlmostEqual(got[17], 2.4, places=6)
        self.assertAlmostEqual(got[18], 2.4, places=6)

    def test_shot_boundary_is_not_a_speed(self):
        values, t, valid = _track(10, 3.0)
        values[5:, 0] = 50.0 + np.arange(5) * 0.1
        got = speed(values, t, valid, ((0, 4), (5, 9)))
        self.assertAlmostEqual(got[4], 3.0, places=6)
        self.assertAlmostEqual(got[5], 3.0, places=6)
        self.assertAlmostEqual(got[9], 3.0, places=6)

    def test_gap_longer_than_max_gap_is_unknown(self):
        t = np.array([0.0, 0.10, 0.40])
        values = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        valid = np.array([True, False, True])
        got = speed(values, t, valid, max_gap_s=0.25)
        self.assertTrue(np.all(np.isnan(got)))

    def test_gap_equal_to_max_gap_is_kept(self):
        t = np.array([0.0, 0.25])
        values = np.array([[0.0, 0.0], [1.0, 0.0]])
        valid = np.array([True, True])
        got = speed(values, t, valid, max_gap_s=0.25)
        self.assertAlmostEqual(got[0], 4.0, places=6)
        self.assertAlmostEqual(got[1], 4.0, places=6)

    def test_long_backward_gap_uses_the_closer_forward_neighbour(self):
        t = np.array([0.0, 0.40, 0.50])
        values = np.array([[0.0, 0.0], [0.0, 0.0], [0.3, 0.0]])
        valid = np.array([True, True, True])
        got = speed(values, t, valid, max_gap_s=0.25)
        self.assertTrue(np.isnan(got[0]))
        self.assertAlmostEqual(got[1], 3.0, places=6)
        self.assertAlmostEqual(got[2], 3.0, places=6)

    def test_empty_shots_is_one_span(self):
        values, t, valid = _track(4, 1.5)
        self.assertTrue(np.allclose(speed(values, t, valid), speed(values, t, valid, ())))

    def test_one_dimensional_speed_is_absolute_velocity(self):
        got = speed(np.array([0.0, -1.0]), np.array([0.0, 0.5]), np.array([True, True]), max_gap_s=0.5)
        self.assertAlmostEqual(got[0], 2.0, places=6)
        self.assertAlmostEqual(got[1], 2.0, places=6)

    def test_infinity_raises(self):
        with self.assertRaises(ValueError):
            velocity(np.array([0.0, np.inf]), np.array([0.0, 0.1]), np.array([True, True]))
        with self.assertRaises(ValueError):
            velocity(np.array([0.0, 1.0]), np.array([0.0, np.nan]), np.array([True, True]))


if __name__ == "__main__":
    unittest.main()
