import json
import unittest
from pathlib import Path

from coach.analyze import analyze
from coach.coaching_points import evidence_candidates, priorities, rank_candidates
from coach.player_input import player_inputs
from coach.short_review import select_points
from coach.coaching_corpus.describe_corpus import describe

ROOT = Path(__file__).resolve().parents[2]


class CoachingPriorityTests(unittest.TestCase):
    def test_missing_channel_videos_are_groups_not_verified_channels(self):
        counts = describe([{"video_id": "a", "channel": "Coach"},
                           {"video_id": "b", "channel": "Coach"},
                           {"video_id": "c", "channel": ""},
                           {"video_id": "d"}])
        self.assertEqual(counts["coaching_videos"], 4)
        self.assertEqual(counts["known_channel_names"], 1)
        self.assertEqual(counts["videos_without_channel_metadata"], 2)
        self.assertEqual(counts["weighting_groups"], 3)

    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((ROOT / "fast/runs/step4-rebalanced-dev/outputs/04-r1.json").read_text("utf-8"))
        cls.doc["source"]["players"] = player_inputs(ROOT / "sample-videos/04-china-open-ws-a.mp4")
        cls.report = analyze(cls.doc)
        cls.candidates = evidence_candidates(cls.doc, cls.report)

    def test_real_pipeline_picks_priority_concepts_not_upright_static_metrics(self):
        points = select_points(self.doc, self.report)
        self.assertEqual([p["concept"] for p in points], ["footwork", "knee_bend", "recovery"])
        self.assertEqual([p["priority"]["rank"] for p in points], [4, 7, 8])
        self.assertEqual([p["source_frame"] for p in points], [86, 89, 99])
        self.assertTrue(all(p["shot"] == [0, 211] for p in points))
        self.assertEqual(points[1]["balance_arm"]["side"], "left")
        self.assertGreater(points[1]["balance_arm"]["elbow_deg"], 159)
        self.assertGreaterEqual(points[2]["cycle"]["next_move_end_frame"], 136)

    def test_priority_file_order_controls_selection(self):
        ranking = priorities()
        ranking["visible_order"] = ["recovery", "knee_bend", "footwork"]
        picked = rank_candidates(self.candidates, ranking)
        self.assertEqual([p["concept"] for p in picked], ranking["visible_order"])

    def test_concepts_without_evidence_are_skipped(self):
        ranking = priorities()
        ranking["visible_order"] = ["contact_point", "non_racket_arm", "footwork", "knee_bend", "recovery"]
        self.assertEqual([p["concept"] for p in rank_candidates(self.candidates, ranking)],
                         ["footwork", "knee_bend", "recovery"])

    def test_lower_ranked_static_points_never_displace_supported_movement(self):
        lower = [{"concept": concept, "confidence": 1.0, "source_frame": 10}
                 for concept in ("ready_position", "stance_width")]
        points = rank_candidates(lower + self.candidates)
        self.assertEqual([p["concept"] for p in points], ["footwork", "knee_bend", "recovery"])

    def test_insufficient_evidence_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "Only 1 supported coaching concepts"):
            rank_candidates([c for c in self.candidates if c["concept"] == "footwork"])


if __name__ == "__main__":
    unittest.main()
