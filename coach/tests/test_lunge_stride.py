import json
import unittest
from pathlib import Path

from coach.analyze import analyze
from coach.events import detect_events
from coach.player_input import player_inputs
from coach.schema import Role
from coach.tracks import load_tracks

ROOT = Path(__file__).resolve().parents[2]


class LungeStrideTests(unittest.TestCase):
    def pose(self):
        doc = json.loads((ROOT / "fast/runs/step4-rebalanced-dev/outputs/04-r1.json").read_text("utf-8"))
        doc["source"]["players"] = player_inputs(ROOT / "sample-videos/04-china-open-ws-a.mp4")
        return doc

    def test_exactly_one_right_led_lunge_in_forehand_sequence(self):
        events = detect_events(load_tracks(self.pose()))
        found = [e for e in events if e.kind == "lunge" and e.player == Role.NEAR
                 and 86 <= e.data["bottom_frame"] <= 92]
        self.assertEqual(len(found), 1)
        event = found[0]
        self.assertEqual(event.data["lead_right"], 1)
        self.assertEqual(event.data["plant_frame"], 86)
        self.assertEqual(event.data["bottom_frame"], 89)
        self.assertEqual(event.data["end_frame"], 99)
        self.assertAlmostEqual(event.data["lead_knee_flexion_deg"], 113.86665479734488)
        self.assertAlmostEqual(event.data["knee_return_s"], 0.3333, places=4)
        self.assertGreater(event.data["stride_ratio"], 0.60)

    def test_report_keeps_direct_movement_evidence_without_false_balance_arm_swing(self):
        report = analyze(self.pose())
        self.assertTrue(any(e.kind == "lunge" and e.data.get("plant_frame") == 86 for e in report.events))
        self.assertFalse(any(e.kind == "swing" and e.data["hand"] == 0 for e in report.events))

    def test_camera_cut_prevents_finishing_the_cycle(self):
        doc = self.pose()
        doc["cuts"] = sorted(set(doc["cuts"] + [95]))
        events = detect_events(load_tracks(doc))
        self.assertFalse(any(e.data.get("knee_stride") and e.frame < 95 <= e.data["end_frame"] for e in events))

    def test_static_front_knee_is_not_a_lunge_without_travelling_foot(self):
        tracks = load_tracks(self.pose())
        player = tracks.players[Role.NEAR]
        for name in ("left_ankle", "right_ankle"):
            player.joint(name)[65:102] = player.joint(name)[89]
        events = detect_events(tracks)
        self.assertFalse(any(e.data.get("knee_stride") and 86 <= e.data["bottom_frame"] <= 92 for e in events))


if __name__ == "__main__":
    unittest.main()
