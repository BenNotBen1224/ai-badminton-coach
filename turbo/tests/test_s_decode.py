"""Crop geometry. These checks do not open a video."""
import unittest

import numpy as np

import decode


class BoxTests(unittest.TestCase):
    def test_even_box_stays_inside_and_even(self):
        x0, y0, x1, y1 = decode.even_box(1.2, 3.8, 10.1, 20.0, 1920, 1080)
        self.assertEqual((x0, y0, x1, y1), (0, 2, 12, 20))
        self.assertEqual(x0 % 2, 0)
        self.assertEqual(y0 % 2, 0)
        self.assertEqual(x1 % 2, 0)
        self.assertEqual(y1 % 2, 0)

    def test_even_box_clamps(self):
        self.assertEqual(decode.even_box(-4, -2, 5, 5, 8, 6), (0, 0, 6, 6))

    def test_shift_warp_matches_full_frame_point(self):
        matrix = np.array([[2.0, 0.0, -30.0], [0.0, 2.0, -10.0]], np.float32)
        moved = decode.shift_warp(matrix, 8, 4)
        x, y = 20.0, 15.0
        full = matrix @ [x, y, 1]
        local = moved @ [x - 8, y - 4, 1]
        np.testing.assert_allclose(full, local)


class ConversionThreadsTests(unittest.TestCase):
    """Varying crops through the production helper. Does not open a video."""

    def test_changing_sizes_match_plain_bgr24(self):
        rng = np.random.RandomState(0)
        y = rng.randint(0, 256, (48, 64), np.uint8)
        u = rng.randint(0, 256, (24, 32), np.uint8)
        v = rng.randint(0, 256, (24, 32), np.uint8)
        rects = (
            (0, 0, 16, 12),
            (2, 2, 28, 20),
            (4, 0, 18, 24),
            (0, 4, 32, 14),
            (6, 2, 40, 30),
            (0, 0, 64, 48),
            (10, 8, 22, 18),
        )
        shapes = []
        for x0, y0, x1, y1 in rects:
            got = decode.yuv_region_bgr(y, u, v, x0, y0, x1, y1)
            ref = _plain_bgr24(y, u, v, x0, y0, x1, y1)
            self.assertEqual(got.shape, ref.shape)
            self.assertEqual(got.dtype, np.uint8)
            self.assertEqual(got.tobytes(), ref.tobytes())
            shapes.append(got.shape)
        self.assertGreaterEqual(len(set(shapes)), 4)
        self.assertTrue(all(a != b for a, b in zip(shapes, shapes[1:])))


def _plain_bgr24(y, u, v, x0, y0, x1, y1):
    """Same crop blit, plain to_ndarray('bgr24'), no threads argument."""
    yy = y[y0:y1, x0:x1]
    uu = u[y0 // 2:y1 // 2, x0 // 2:x1 // 2]
    vv = v[y0 // 2:y1 // 2, x0 // 2:x1 // 2]
    h, w = yy.shape
    fr = decode.av().VideoFrame(w, h, "yuv420p")
    decode._blit(fr.planes[0], yy)
    decode._blit(fr.planes[1], uu)
    decode._blit(fr.planes[2], vv)
    return fr.to_ndarray(format="bgr24")


class ServiceContractTests(unittest.TestCase):
    def test_runner_is_not_a_replay(self):
        import service
        self.assertFalse(service.PoseRunner.IS_REPLAY)
        self.assertTrue(hasattr(service.PoseRunner, "warm_up"))
        self.assertTrue(hasattr(service.PoseRunner, "run"))


if __name__ == "__main__":
    unittest.main()
