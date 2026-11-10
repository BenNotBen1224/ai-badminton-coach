"""Behavior of load_tracks and CourtGeometry. Expected numbers are literals."""
from __future__ import annotations

import time
import unittest

import cv2
import numpy as np

from coach.schema import Role, Tier
from coach.tracks import COURT_LENGTH_M, COURT_WIDTH_M, load_tracks

WIDTH = 1280
HEIGHT = 720
CORNERS_PX = np.array(
    [[460.0, 80.0], [820.0, 80.0], [1180.0, 640.0], [100.0, 640.0]],
    dtype=np.float64,
)


def _source(n, fps=30.0):
    return {"file": "synthetic", "fps": fps, "width": WIDTH, "height": HEIGHT, "frame_count": n}


def _court_lines():
    return (CORNERS_PX / np.array([WIDTH, HEIGHT])).tolist()


def _blank_joints():
    return [[0.0, 0.0, 0.9] for _ in range(17)]


def _put(joints, index, x, y, conf=0.9):
    joints[index] = [float(x), float(y), float(conf)]


def _standing(cx, foot_y, height, conf=0.9):
    joints = _blank_joints()
    ankle_y = foot_y
    knee_y = foot_y - 0.28 * height
    hip_y = foot_y - 0.52 * height
    shoulder_y = foot_y - 0.82 * height
    nose_y = foot_y - height
    for index, x in ((15, cx - 12), (16, cx + 12)):
        _put(joints, index, x, ankle_y, conf)
    for index, x in ((13, cx - 12), (14, cx + 12)):
        _put(joints, index, x, knee_y, conf)
    for index, x in ((11, cx - 10), (12, cx + 10)):
        _put(joints, index, x, hip_y, conf)
    for index, x in ((5, cx - 16), (6, cx + 16)):
        _put(joints, index, x, shoulder_y, conf)
    _put(joints, 0, cx, nose_y, conf)
    _put(joints, 1, cx - 4, nose_y + 2, conf)
    _put(joints, 2, cx + 4, nose_y + 2, conf)
    return joints


def _crouched(cx, foot_y):
    joints = _standing(cx, foot_y, 280.0)
    _put(joints, 13, cx - 40, foot_y - 70)
    _put(joints, 14, cx + 40, foot_y - 70)
    _put(joints, 11, cx - 10, foot_y - 130)
    _put(joints, 12, cx + 10, foot_y - 130)
    _put(joints, 5, cx - 16, foot_y - 210)
    _put(joints, 6, cx + 16, foot_y - 210)
    _put(joints, 0, cx, foot_y - 280)
    return joints


def _player(role, joints, box=0.9, src=None, track_id=1):
    xs = [j[0] for j in joints if j is not None]
    ys = [j[1] for j in joints if j is not None]
    row = {
        "role": role,
        "track_id": track_id,
        "bbox_xyxy": [min(xs) - 4, min(ys) - 4, max(xs) + 4, max(ys) + 4],
        "box_confidence": box,
        "joints": joints,
    }
    if src is not None:
        row["src"] = src
    return row


def _doc(frames, fps=30.0, cuts=None, court=True, fit=0.98, tier=None):
    n = len(frames)
    doc = {"source": _source(n, fps), "frames": frames}
    if court:
        doc["court_lines_normalized"] = _court_lines()
        doc["court_fit"] = {"fit": fit}
    if cuts is not None:
        doc["cuts"] = cuts
    if tier is not None:
        doc["tier"] = tier
    return doc


def _frames(n, builder):
    frames = []
    for i in range(n):
        players = builder(i)
        frames.append({
            "frame_index": i,
            "timestamp_seconds": i / 30.0,
            "players": players,
        })
    return frames


def _pixel_for_metres(x_m, y_m):
    dst = np.array(
        [[0.0, 0.0], [COURT_WIDTH_M, 0.0], [COURT_WIDTH_M, COURT_LENGTH_M], [0.0, COURT_LENGTH_M]],
        np.float32,
    )
    src = np.ascontiguousarray(CORNERS_PX, np.float32)
    matrix = cv2.getPerspectiveTransform(dst, src)
    raw = np.array([[[x_m, y_m]]], np.float32)
    return cv2.perspectiveTransform(raw, matrix).reshape(2)


class CourtTests(unittest.TestCase):
    def test_doubles_corners_roundtrip_to_court_metres(self):
        frames = _frames(4, lambda i: [_player("near", _standing(640, 500, 400))])
        tracks = load_tracks(_doc(frames))
        got = tracks.court.project(CORNERS_PX)
        expect = np.array([
            [0.0, 0.0],
            [6.10, 0.0],
            [6.10, 13.40],
            [0.0, 13.40],
        ])
        self.assertTrue(np.allclose(got, expect, atol=0.01))
        back = tracks.court.unproject(expect)
        self.assertTrue(np.allclose(back, CORNERS_PX, atol=1.0))
        self.assertEqual(tracks.court.width_m, 6.10)
        self.assertEqual(tracks.court.length_m, 13.40)

    def test_zero_noise_foot_is_within_5cm(self):
        pixel = _pixel_for_metres(3.05, 10.0)
        joints = _standing(pixel[0], pixel[1], 180)
        _put(joints, 15, pixel[0] - 2, pixel[1])
        _put(joints, 16, pixel[0] + 2, pixel[1])
        frames = _frames(4, lambda i: [_player("near", joints)])
        foot = load_tracks(_doc(frames)).players[Role.NEAR].foot_court[0]
        self.assertLess(abs(foot[0] - 3.05), 0.05)
        self.assertLess(abs(foot[1] - 10.0), 0.05)

    def test_missing_outline_raises_no_court(self):
        frames = _frames(1, lambda i: [_player("near", _standing(640, 500, 400))])
        tracks = load_tracks(_doc(frames, court=False))
        self.assertIsNone(tracks.court)
        self.assertEqual(tracks.warnings, ("no_court",))
        self.assertTrue(np.all(np.isnan(tracks.players[Role.NEAR].foot_court)))
        self.assertTrue(np.all(np.isnan(tracks.players[Role.NEAR].px_per_m)))

    def test_low_fit_raises_no_court(self):
        frames = _frames(2, lambda i: [_player("near", _standing(640, 500, 400))])
        tracks = load_tracks(_doc(frames, fit=0.84))
        self.assertIsNone(tracks.court)
        self.assertIn("no_court", tracks.warnings)


class MeasuredTests(unittest.TestCase):
    def test_exact_repeat_is_not_measured_and_warns_above_5_percent(self):
        frames = _frames(20, lambda i: [_player("near", _standing(640 + i, 500, 400))])
        frames[1]["players"][0]["joints"] = [list(j) for j in frames[0]["players"][0]["joints"]]
        frames[3]["players"][0]["joints"] = [list(j) for j in frames[2]["players"][0]["joints"]]
        tracks = load_tracks(_doc(frames))
        near = tracks.players[Role.NEAR]
        self.assertTrue(near.measured[0])
        self.assertFalse(near.measured[1])
        self.assertFalse(near.measured[3])
        self.assertTrue(near.measured[2])
        self.assertIn("repeated_frames", tracks.warnings)

    def test_five_percent_repeats_do_not_warn(self):
        frames = _frames(20, lambda i: [_player("near", _standing(640 + i, 500, 400))])
        frames[1]["players"][0]["joints"] = [list(j) for j in frames[0]["players"][0]["joints"]]
        tracks = load_tracks(_doc(frames))
        self.assertNotIn("repeated_frames", tracks.warnings)
        self.assertFalse(tracks.players[Role.NEAR].measured[1])

    def test_interpolated_row_is_visible_but_not_measured(self):
        frames = _frames(3, lambda i: [_player("near", _standing(640 + i * 5, 500, 400))])
        frames[1]["players"][0]["src"] = "interpolated"
        tracks = load_tracks(_doc(frames, tier="preview"))
        near = tracks.players[Role.NEAR]
        self.assertEqual(tracks.tier, Tier.PREVIEW)
        self.assertFalse(near.measured[1])
        self.assertTrue(near.visible[1, 0])
        self.assertNotIn("repeated_frames", tracks.warnings)

    def test_confidence_gate_is_half_and_keeps_scores_above_one(self):
        joints = _standing(640, 500, 400, conf=1.4)
        joints[15][2] = 0.5
        joints[16] = None
        joints[9] = [100.0, 100.0, 0.49]
        frames = _frames(1, lambda i: [_player("near", joints)])
        near = load_tracks(_doc(frames)).players[Role.NEAR]
        self.assertTrue(near.visible[0, 0])
        self.assertTrue(near.visible[0, 15])
        self.assertFalse(near.visible[0, 16])
        self.assertFalse(near.visible[0, 9])
        self.assertTrue(np.isnan(near.xy[0, 16, 0]))
        self.assertAlmostEqual(near.foot_px[0, 0], 628.0, places=4)
        self.assertAlmostEqual(near.foot_px[0, 1], 500.0, places=4)

    def test_nan_joint_matches_nan_joint_as_a_repeat(self):
        frames = _frames(2, lambda i: [_player("near", _standing(640, 500, 400))])
        frames[0]["players"][0]["joints"][7] = None
        frames[1]["players"][0]["joints"] = [list(j) if j is not None else None for j in frames[0]["players"][0]["joints"]]
        near = load_tracks(_doc(frames)).players[Role.NEAR]
        self.assertFalse(near.measured[1])

    def test_highest_box_confidence_wins_a_shared_role(self):
        low = _player("near", _standing(100, 500, 400), box=0.2, track_id=7)
        high = _player("near", _standing(400, 500, 400), box=0.8, track_id=9)
        frames = [{"frame_index": 0, "timestamp_seconds": 0.0, "players": [low, high]}]
        near = load_tracks(_doc(frames)).players[Role.NEAR]
        self.assertAlmostEqual(near.foot_px[0, 0], 400.0, places=4)

    def test_timestamp_falls_back_to_frame_over_fps(self):
        frames = _frames(3, lambda i: [_player("near", _standing(640, 500, 400))])
        frames[0]["timestamp_seconds"] = 0.05
        del frames[1]["timestamp_seconds"]
        frames[2]["timestamp_seconds"] = 0.30
        tracks = load_tracks(_doc(frames, fps=10.0))
        self.assertAlmostEqual(tracks.t[0], 0.05, places=6)
        self.assertAlmostEqual(tracks.t[1], 0.1, places=6)
        self.assertAlmostEqual(tracks.t[2], 0.30, places=6)

    def test_backward_or_negative_timestamp_raises(self):
        frames = _frames(2, lambda i: [_player("near", _standing(640, 500, 400))])
        frames[0]["timestamp_seconds"] = 1.5
        del frames[1]["timestamp_seconds"]
        with self.assertRaises(ValueError) as backward:
            load_tracks(_doc(frames, fps=10.0))
        self.assertIn("backwards", str(backward.exception))
        frames[0]["timestamp_seconds"] = -0.1
        frames[1]["timestamp_seconds"] = 0.2
        with self.assertRaises(ValueError) as negative:
            load_tracks(_doc(frames, fps=10.0))
        self.assertIn("negative", str(negative.exception))

    def test_reacquired_pose_is_measured_and_shot_start_is_fresh(self):
        pose = _standing(640, 500, 400)
        frames = [
            {"frame_index": 0, "timestamp_seconds": 0.0, "players": [_player("near", pose)]},
            {"frame_index": 1, "timestamp_seconds": 1 / 30, "players": []},
            {"frame_index": 2, "timestamp_seconds": 2 / 30, "players": [_player("near", pose)]},
        ]
        near = load_tracks(_doc(frames)).players[Role.NEAR]
        self.assertTrue(near.measured[0])
        self.assertFalse(near.present[1])
        self.assertTrue(near.measured[2])
        held = _frames(6, lambda i: [_player("near", pose)])
        shot = load_tracks(_doc(held, cuts=[3])).players[Role.NEAR]
        self.assertTrue(shot.measured[0])
        self.assertFalse(shot.measured[2])
        self.assertTrue(shot.measured[3])
        self.assertFalse(shot.measured[4])

    def test_nan_coordinates_are_not_visible(self):
        joints = _standing(640, 500, 400)
        joints[9] = [float("nan"), 100.0, 0.9]
        near = load_tracks(_doc(_frames(1, lambda i: [_player("near", joints)]))).players[Role.NEAR]
        self.assertFalse(near.visible[0, 9])
        self.assertTrue(np.isnan(near.xy[0, 9, 0]))
        self.assertTrue(near.visible[0, 0])

    def test_unknown_tier_raises_and_explicit_refined_is_kept(self):
        frames = _frames(1, lambda i: [_player("near", _standing(640, 500, 400))])
        with self.assertRaises(ValueError) as bad:
            load_tracks(_doc(frames, tier="fast"))
        self.assertIn("tier", str(bad.exception))
        self.assertEqual(load_tracks(_doc(frames, tier="refined")).tier, Tier.REFINED)

    def test_infinity_raises(self):
        joints = _standing(640, 500, 400)
        joints[0][0] = float("inf")
        frames = _frames(1, lambda i: [_player("near", joints)])
        with self.assertRaises(ValueError):
            load_tracks(_doc(frames))


class WarningTests(unittest.TestCase):
    def test_partial_track_and_high_fps(self):
        def builder(i):
            if i < 7:
                return [_player("near", _standing(640, 500, 400))]
            return []
        tracks = load_tracks(_doc(_frames(10, builder), fps=41.0))
        self.assertIn("partial_track", tracks.warnings)
        self.assertIn("high_fps", tracks.warnings)
        self.assertEqual(tracks.shots, ((0, 9),))

    def test_eighty_percent_present_is_not_partial(self):
        def builder(i):
            if i < 8:
                return [_player("near", _standing(640, 500, 400))]
            return []
        tracks = load_tracks(_doc(_frames(10, builder), fps=40.0))
        self.assertNotIn("partial_track", tracks.warnings)
        self.assertNotIn("high_fps", tracks.warnings)

    def test_cuts_split_inclusive_shots(self):
        frames = _frames(20, lambda i: [_player("near", _standing(640, 500, 400))])
        tracks = load_tracks(_doc(frames, cuts=[10]))
        self.assertEqual(tracks.shots, ((0, 9), (10, 19)))

    def test_foreign_shot_blanks_court_positions(self):
        pixel = _pixel_for_metres(3.05, 10.0)

        def builder(i):
            if i < 20:
                joints = _standing(pixel[0], pixel[1], 160)
                _put(joints, 15, pixel[0] - 1, pixel[1])
                _put(joints, 16, pixel[0] + 1, pixel[1])
            else:
                joints = _standing(8, 8, 40)
                _put(joints, 15, 6, 8)
                _put(joints, 16, 10, 8)
            return [_player("near", joints)]

        tracks = load_tracks(_doc(_frames(40, builder), cuts=[20]))
        near = tracks.players[Role.NEAR]
        self.assertLess(abs(near.foot_court[0, 0] - 3.05), 0.05)
        self.assertTrue(np.all(np.isnan(near.foot_court[20:])))
        self.assertTrue(np.all(np.isnan(near.px_per_m[20:])))
        self.assertTrue(np.isfinite(near.foot_court[0, 1]))

    def test_omitted_tier_is_refined_and_absent_role_is_empty(self):
        frames = _frames(2, lambda i: [_player("near", _standing(640, 500, 400))])
        tracks = load_tracks(_doc(frames))
        self.assertEqual(tracks.tier, Tier.REFINED)
        self.assertFalse(tracks.players[Role.FAR].present.any())
        self.assertEqual(tracks.n, 2)


class StatureTests(unittest.TestCase):
    def test_crouch_does_not_shrink_the_yardstick(self):
        def builder(i):
            if i < 6:
                return [_player("near", _crouched(640, 500))]
            return [_player("near", _standing(640, 500, 400))]
        near = load_tracks(_doc(_frames(20, builder))).players[Role.NEAR]
        self.assertAlmostEqual(near.stature_px[0], 400.0, places=3)
        self.assertAlmostEqual(near.stature_px[10], 400.0, places=3)

    def test_depth_bins_ignore_crouched_samples(self):
        def builder(i):
            if i < 30:
                return [_player("near", _standing(500 + i * 0.02, 400, 250))]
            if i < 35:
                return [_player("near", _standing(700 + i * 0.02, 800, 500))]
            return [_player("near", _crouched(700 + i * 0.02, 800))]
        near = load_tracks(_doc(_frames(50, builder))).players[Role.NEAR]
        self.assertLess(abs(near.stature_px[10] - 250.0) / 250.0, 0.03)
        self.assertLess(abs(near.stature_px[32] - 500.0) / 500.0, 0.03)
        self.assertLess(abs(near.stature_px[40] - 500.0) / 500.0, 0.03)

    def test_eyes_stand_in_when_the_nose_is_missing(self):
        joints = _standing(640, 500, 400)
        joints[0] = None
        _put(joints, 1, 636, 102)
        _put(joints, 2, 644, 102)
        frames = _frames(4, lambda i: [_player("near", joints)])
        near = load_tracks(_doc(frames)).players[Role.NEAR]
        self.assertAlmostEqual(near.stature_px[0], 398.0, places=3)


class TimingTests(unittest.TestCase):
    def test_900_frames_best_of_five_under_60ms(self):
        frames = _frames(900, lambda i: [
            _player("near", _standing(700, 560, 420)),
            _player("far", _standing(640, 180, 180)),
        ])
        doc = _doc(frames)
        times = []
        for _ in range(5):
            start = time.perf_counter()
            load_tracks(doc)
            times.append((time.perf_counter() - start) * 1000.0)
        self.assertLessEqual(max(times), 150.0)
        self.assertLess(min(times), 60.0)


class RefinedWindowTests(unittest.TestCase):
    def test_sharpened_windows_do_not_change_tier_or_measured(self):
        def builder(i):
            src = "interpolated" if i == 1 else None
            return [_player("near", _standing(640, 500, 400), src=src)]

        frames = _frames(4, builder)
        plain = load_tracks(_doc(frames, tier="preview"))
        doc = _doc(frames, tier="preview")
        doc["windows_refined"] = [
            {"start": 1, "end": 2, "s": 0.01},
            {"start": 2, "end": 2},
            {"start": 0, "end": 3, "s": 0.2},
        ]
        tracks = load_tracks(doc)
        near = tracks.players[Role.NEAR]
        self.assertEqual(tracks.refined_windows, ((1, 2), (2, 2), (0, 3)))
        self.assertEqual(tracks.tier, Tier.PREVIEW)
        self.assertTrue(np.array_equal(near.measured, plain.players[Role.NEAR].measured))
        self.assertFalse(near.measured[1])
        self.assertEqual(tracks.warnings, plain.warnings)
        self.assertEqual(plain.refined_windows, ())
        empty = _doc(frames, tier="preview")
        empty["windows_refined"] = []
        self.assertEqual(load_tracks(empty).refined_windows, ())

    def test_malformed_refined_windows_raise(self):
        frames = _frames(4, lambda i: [_player("near", _standing(640, 500, 400))])
        bad = [
            ("not-a-list", "must be a list"),
            ({"start": 0, "end": 1}, "must be a list"),
            (None, "must be a list"),
            ([{"start": 1, "end": 2}, {"start": 1.0, "end": 2}], "must be an integer"),
            ([{"start": True, "end": 1}], "must be an integer"),
            ([{"start": -1, "end": 1}], "0 <= start <= end"),
            ([{"start": 2, "end": 1}], "0 <= start <= end"),
            ([{"start": 3, "end": 4}], "0 <= start <= end"),
            ([{"end": 1}], "start and end"),
            (["1-2"], "start and end"),
            ([None], "start and end"),
        ]
        for raw, fragment in bad:
            doc = _doc(frames, tier="preview")
            doc["windows_refined"] = raw
            with self.assertRaises(ValueError, msg=repr(raw)) as caught:
                load_tracks(doc)
            self.assertIn(fragment, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
