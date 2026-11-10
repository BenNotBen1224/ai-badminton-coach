"""Production _infer_crops must raise the worker error, not a None row.

The rejected async probe does not cover this helper. No video and no timing.
"""
import threading
import unittest

import numpy as np

import service


class _Gate:
    def __init__(self, need):
        self.need = need
        self.lock = threading.Lock()
        self.started = []
        self.finished = []
        self.ready = threading.Event()

    def enter(self, req):
        with self.lock:
            self.started.append(req)
            if len(self.started) >= self.need:
                self.ready.set()
        if not self.ready.wait(2):
            raise TimeoutError("second request did not start")

    def leave(self, req):
        with self.lock:
            self.finished.append(req)


class _Ov:
    def __init__(self, gate, fail_req=None):
        self.gate = gate
        self.fail_req = fail_req
        self.calls = []
        self.lock = threading.Lock()

    def infer(self, req, crop):
        with self.lock:
            self.calls.append(req)
        self.gate.enter(req)
        try:
            if req == self.fail_req:
                raise RuntimeError("infer failed")
            return ([[np.array([float(req)])]], [[np.array([1.0])]])
        finally:
            self.gate.leave(req)


class _Pose:
    def __init__(self, fail_center=None):
        self.fail_center = fail_center

    def decode(self, simcc_x, simcc_y, center, scale):
        if center == self.fail_center:
            raise RuntimeError("decode failed")
        return np.array([float(center[0]), float(scale), float(np.reshape(simcc_x, -1)[0])])


def _jobs(n):
    crop = np.zeros((4, 4, 3), np.uint8)
    return [(crop, (i + 1, 0), float(i + 1)) for i in range(n)]


def _waves(calls, reqs):
    """The next wave cannot append until the previous wave joins."""
    size = len(reqs)
    return [calls[i:i + size] for i in range(0, len(calls), size)]


def _call(reqs, jobs, pose, ov):
    try:
        rows = service._infer_crops(reqs, jobs, pose, ov)
    except Exception as exc:
        return exc
    return rows


class InferErrorTest(unittest.TestCase):
    def test_infer_failure_waits_for_the_started_peer(self):
        gate = _Gate(2)
        ov = _Ov(gate, fail_req=1)
        got = _call([0, 1], _jobs(2), _Pose(), ov)
        self.assertIsInstance(got, RuntimeError, f"returned partial rows {got}")
        self.assertEqual(str(got), "infer failed")
        self.assertEqual(sorted(gate.started), [0, 1])
        self.assertIn(0, gate.finished)
        self.assertIn(1, gate.finished)

    def test_decode_failure_waits_for_the_started_peer(self):
        gate = _Gate(2)
        ov = _Ov(gate)
        got = _call([0, 1], _jobs(2), _Pose(fail_center=(1, 0)), ov)
        self.assertIsInstance(got, RuntimeError, f"returned partial rows {got}")
        self.assertEqual(str(got), "decode failed")
        self.assertEqual(sorted(gate.started), [0, 1])
        self.assertEqual(sorted(gate.finished), [0, 1])
        self.assertEqual(len(_waves(ov.calls, [0, 1])), 1)
        self.assertCountEqual(_waves(ov.calls, [0, 1])[0], [0, 1])

    def test_failure_does_not_start_the_next_wave(self):
        gate = _Gate(2)
        ov = _Ov(gate, fail_req=0)
        got = _call([0, 1], _jobs(3), _Pose(), ov)
        self.assertIsInstance(got, RuntimeError, f"returned partial rows {got}")
        self.assertEqual(str(got), "infer failed")
        self.assertEqual(len(_waves(ov.calls, [0, 1])), 1)
        self.assertCountEqual(_waves(ov.calls, [0, 1])[0], [0, 1])

    def test_one_request_still_raises_the_real_error(self):
        gate = _Gate(1)
        ov = _Ov(gate, fail_req=0)
        got = _call([0], _jobs(1), _Pose(), ov)
        self.assertIsInstance(got, RuntimeError, f"returned partial rows {got}")
        self.assertEqual(str(got), "infer failed")
        self.assertEqual(ov.calls, [0])

    def test_normal_path_keeps_order_and_count(self):
        gate = _Gate(2)
        ov = _Ov(gate)
        rows = service._infer_crops([0, 1], _jobs(3), _Pose(), ov)
        waves = _waves(ov.calls, [0, 1])
        self.assertEqual(len(waves), 2)
        self.assertCountEqual(waves[0], [0, 1])
        self.assertCountEqual(waves[1], [0])
        self.assertEqual(len(rows), 3)
        self.assertEqual([row[0] for row in rows], [1.0, 2.0, 3.0])
        self.assertEqual([row[2] for row in rows], [0.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
