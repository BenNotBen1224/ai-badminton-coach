import copy
import json
import unittest
from pathlib import Path

from coach.analyze import analyze
from coach.events import detect_events
from coach.player_input import player_inputs
from coach.tracks import load_tracks

ROOT = Path(__file__).resolve().parents[2]


class PlayerInputTests(unittest.TestCase):
    def pose(self, known=True):
        doc = json.loads((ROOT / "fast/runs/step4-rebalanced-dev/outputs/04-r1.json").read_text("utf-8"))
        if known:
            doc["source"]["players"] = player_inputs(ROOT / "sample-videos/04-china-open-ws-a.mp4")
        return doc

    def test_explicit_right_hand_removes_frame79_left_overhead(self):
        report = analyze(self.pose())
        swings = [e for e in report.events if e.kind == "swing"]
        self.assertTrue(swings)
        self.assertTrue(all(e.data["hand"] == 1 for e in swings))
        arm = [f for f in report.findings if f.player.value == "near" and f.id.startswith("contact.")]
        self.assertTrue(arm)
        self.assertFalse(any(m.key_frame == 79 for f in arm for m in f.moments))
        self.assertTrue(all("left_wrist" not in f.focus_joints for f in arm))

    def test_unknown_hand_withholds_swings_and_arm_findings(self):
        doc = self.pose(False)
        doc["source"].pop("players", None)
        self.assertFalse(any(e.kind == "swing" for e in detect_events(load_tracks(doc))))
        self.assertFalse(any(f.id.startswith("contact.") for f in analyze(doc).findings))

    def test_explicit_left_and_unknown_overrides(self):
        players = player_inputs(ROOT / "sample-videos/04-china-open-ws-a.mp4",
                               overrides={"near": "left", "far": "unknown"})
        self.assertEqual(players["near"]["hand"], "left")
        self.assertEqual(players["far"]["hand"], "unknown")

    def test_invalid_hand_is_rejected(self):
        doc = self.pose()
        doc["source"]["players"]["near"]["hand"] = "fastest"
        with self.assertRaisesRegex(ValueError, "left, right or unknown"):
            load_tracks(doc)


if __name__ == "__main__":
    unittest.main()
