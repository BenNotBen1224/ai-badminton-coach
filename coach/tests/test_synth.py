"""Synthetic documents against load_tracks. Expected numbers are literals."""
from __future__ import annotations

import unittest

import numpy as np

from coach.schema import Role
from coach.signals import speed
from coach.synth import (
    Lunge,
    Noise,
    Path,
    PlayerPlan,
    Posture,
    SplitStep,
    Swing,
    make_preview,
    repeat_frames,
    script,
)
from coach.tracks import load_tracks


def _angle(a, b, c):
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _near(doc, truth):
    tracks = load_tracks(doc)
    return tracks, tracks.players[Role.NEAR], truth["players"]["near"]


class TruthTests(unittest.TestCase):
    def test_zero_noise_feet_are_within_5cm(self):
        doc, truth = script(n=8)
        _, near, info = _near(doc, truth)
        self.assertLess(abs(near.foot_court[3, 0] - 3.05), 0.05)
        self.assertLess(abs(near.foot_court[3, 1] - 10.5), 0.05)
        self.assertTrue(np.allclose(near.foot_court, info["foot_court"], atol=0.05))

    def test_path_ends_at_the_chosen_point(self):
        plan = PlayerPlan(path=Path(0, 10, (4.40, 9.20)))
        doc, _ = script(n=12, near=plan)
        near = load_tracks(doc).players[Role.NEAR]
        self.assertLess(abs(near.foot_court[10, 0] - 4.40), 0.05)
        self.assertLess(abs(near.foot_court[10, 1] - 9.20), 0.05)
        self.assertLess(abs(near.foot_court[0, 0] - 3.05), 0.05)

    def test_stature_holds_with_30_percent_crouch(self):
        plan = PlayerPlan(crouch_fraction=0.3)
        doc, truth = script(n=40, near=plan)
        _, near, info = _near(doc, truth)
        self.assertEqual(len(info["crouch_frames"]), 12)
        rel = np.abs(near.stature_px - info["stature_px"]) / info["stature_px"]
        self.assertLess(float(np.nanmax(rel)), 0.03)
        self.assertGreater(float(np.mean(near.stature_px)), float(np.mean(load_tracks(doc).players[Role.FAR].stature_px)))

    def test_ready_posture_uses_the_chosen_angles(self):
        plan = PlayerPlan(posture=Posture(knee_flexion_deg=40, stance_width_m=0.50, trunk_lean_deg=15))
        doc, _ = script(n=2, near=plan)
        tracks = load_tracks(doc)
        near = tracks.players[Role.NEAR]
        knee = _angle(near.xy[0, 11], near.xy[0, 13], near.xy[0, 15])
        self.assertLess(abs(knee - 140.0), 8.0)
        ankles = tracks.court.project(np.vstack([near.xy[0, 15], near.xy[0, 16]]))
        stance = abs(float(ankles[0, 0] - ankles[1, 0]))
        self.assertLess(abs(stance - 0.50), 0.02)
        trunk = near.xy[0, 5] * 0.5 + near.xy[0, 6] * 0.5 - (near.xy[0, 11] * 0.5 + near.xy[0, 12] * 0.5)
        lean = abs(float(np.degrees(np.arctan2(trunk[0], -trunk[1]))))
        self.assertLess(abs(lean - 15.0), 3.0)

    def test_hip_height_ratio_drops_the_hips(self):
        standing, _ = script(n=1)
        ready, _ = script(n=1, near=PlayerPlan(posture=Posture(hip_height_ratio=0.85)))
        stand = load_tracks(standing).players[Role.NEAR]
        low = load_tracks(ready).players[Role.NEAR]
        stand_h = stand.foot_px[0, 1] - stand.xy[0, 11, 1]
        low_h = low.foot_px[0, 1] - low.xy[0, 11, 1]
        self.assertLess(abs(low_h / stand_h - 0.85), 0.03)

    def test_lunge_reaches_the_chosen_depth_and_returns(self):
        plan = PlayerPlan(lunge=Lunge(10, 0.80, 0.4))
        doc, truth = script(n=36, near=plan)
        info = truth["players"]["near"]["lunge"]
        self.assertEqual(info["bottom_frame"], 16)
        self.assertEqual(info["end_frame"], 28)
        near = load_tracks(doc).players[Role.NEAR]
        lead = near.xy[16, 16]
        lead_m = load_tracks(doc).court.project(lead)
        self.assertLess(abs(lead_m[1] - 9.70), 0.05)
        self.assertLess(abs(near.foot_court[28, 1] - 10.5), 0.05)

    def test_swing_peaks_before_contact(self):
        plan = PlayerPlan(swing=Swing(30, 1.75, 150))
        doc, truth = script(n=40, near=plan)
        info = truth["players"]["near"]["swing"]
        self.assertEqual(info["peak_frame"], 27)
        self.assertEqual(info["contact_frame"], 30)
        tracks = load_tracks(doc)
        near = tracks.players[Role.NEAR]
        wrist = near.joint("right_wrist")
        valid = near.measured & near.joint_visible("right_wrist")
        spd = speed(wrist, tracks.t, valid, tracks.shots)
        self.assertEqual(int(np.nanargmax(spd)), 27)
        elbow = _angle(near.xy[30, 6], near.xy[30, 8], near.xy[30, 10])
        self.assertLess(abs(elbow - 150.0), 5.0)
        height = (near.foot_px[30, 1] - near.xy[30, 10, 1]) / near.px_per_m[30]
        self.assertLess(abs(height - 1.75), 0.05)

    def test_split_step_hops_then_dips(self):
        doc, _ = script(n=12, near=PlayerPlan(split_step=SplitStep(6)))
        near = load_tracks(doc).players[Role.NEAR]
        self.assertLess(near.foot_px[6, 1], near.foot_px[4, 1] - 2.0)
        hip = near.xy[:, 11, 1]
        self.assertGreater(hip[7], hip[4] + 2.0)


class AnatomyTests(unittest.TestCase):
    def test_bones_stay_constant_and_feet_stay_planted(self):
        doc, _ = script(n=6)
        near = load_tracks(doc).players[Role.NEAR]
        shin = []
        elbow = []
        for i in range(6):
            scale = near.px_per_m[i]
            shin.append(np.linalg.norm(near.xy[i, 13] - near.xy[i, 15]) / scale)
            upper = np.linalg.norm(near.xy[i, 6] - near.xy[i, 8]) / scale
            fore = np.linalg.norm(near.xy[i, 8] - near.xy[i, 10]) / scale
            knee = _angle(near.xy[i, 11], near.xy[i, 13], near.xy[i, 15])
            arm = _angle(near.xy[i, 6], near.xy[i, 8], near.xy[i, 10])
            self.assertGreater(knee, 30.0)
            self.assertLess(knee, 185.0)
            self.assertGreater(arm, 20.0)
            self.assertLess(arm, 185.0)
            self.assertLess(abs(upper - fore), 0.2)
            elbow.append(upper)
        self.assertLess(max(shin) - min(shin), 0.005)
        self.assertLess(max(elbow) - min(elbow), 0.005)
        self.assertTrue(np.allclose(near.xy[0, 15], near.xy[5, 15]))
        self.assertTrue(np.allclose(near.xy[0, 16], near.xy[5, 16]))


class PreviewTests(unittest.TestCase):
    def test_preview_interpolates_between_measured_frames(self):
        doc, _ = script(n=7, near=PlayerPlan(path=Path(0, 6, (4.2, 10.5))))
        preview = make_preview(doc, 3)
        self.assertEqual(preview["tier"], "preview")
        self.assertNotIn("src", doc["frames"][1]["players"][0])
        row = preview["frames"][1]["players"][0]
        self.assertEqual(row["src"], "interpolated")
        j0 = doc["frames"][0]["players"][0]["joints"][15]
        j3 = doc["frames"][3]["players"][0]["joints"][15]
        got = row["joints"][15]
        self.assertAlmostEqual(got[0], j0[0] + (j3[0] - j0[0]) / 3, places=4)
        self.assertAlmostEqual(got[1], j0[1] + (j3[1] - j0[1]) / 3, places=4)
        self.assertAlmostEqual(got[2], min(j0[2], j3[2]), places=4)
        tracks = load_tracks(preview)
        self.assertFalse(tracks.players[Role.NEAR].measured[1])
        self.assertTrue(tracks.players[Role.NEAR].measured[0])

    def test_repeat_frames_keep_constant_speed(self):
        doc, truth = script(n=18, near=PlayerPlan(path=Path(0, 17, (4.75, 10.5))))
        indices = repeat_frames(doc, 6, truth)
        self.assertEqual(indices, [5, 11, 17])
        self.assertEqual(truth["repeated_frames"], [5, 11, 17])
        self.assertEqual(
            doc["frames"][5]["players"][0]["joints"],
            doc["frames"][4]["players"][0]["joints"],
        )
        tracks = load_tracks(doc)
        near = tracks.players[Role.NEAR]
        self.assertFalse(near.measured[5])
        spd = speed(near.foot_px, tracks.t, near.measured, tracks.shots)
        true = truth["players"]["near"]["path"]["speed_m_s"]
        court_speed = speed(near.foot_court, tracks.t, near.measured, tracks.shots)
        paired = np.isfinite(court_speed)
        self.assertTrue(paired[6])
        self.assertTrue(np.all(np.abs(court_speed[paired] - true) <= 0.03 * true))
        self.assertGreater(spd[6], 0.0)


class NoiseTests(unittest.TestCase):
    def test_dropout_and_missing_frames(self):
        noise = Noise(dropout=1.0, missing_role="far", missing_frames=(2, 3), seed=1)
        doc, _ = script(n=6, noise=noise, cuts=(4,))
        self.assertEqual(doc["cuts"], [4])
        far_roles = [p["role"] for p in doc["frames"][2]["players"]]
        self.assertNotIn("far", far_roles)
        conf = doc["frames"][0]["players"][0]["joints"][0][2]
        self.assertEqual(conf, 0.1)
        near = load_tracks(doc).players[Role.NEAR]
        self.assertFalse(near.visible[0].any())
        self.assertFalse(load_tracks(doc).players[Role.FAR].present[2])


if __name__ == "__main__":
    unittest.main()
