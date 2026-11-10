import unittest

from eval_strict import score


class StrictSubsetTests(unittest.TestCase):
    def hit(self, time, role, tier="strict"):
        return {
            "t": time, "hitter": role, "eval_tier": tier, "evidence": "visible_contact",
            "conf": {"t": "med", "hitter": "high", "zone": "med", "family": "low"},
            "next_zone": {"depth": "rear", "side": "left"}, "family": "clear",
        }

    def prediction(self, time, role):
        return {"t": time, "hitter": role, "zone": ("rear", "left"), "family": "clear"}

    def test_incomplete_labels_never_produce_precision(self):
        result = score(
            [{"id": "sample", "hits": [self.hit(1, "near"), self.hit(2, "far", "low_or_inferred")]}],
            [self.prediction(1.1, "near"), self.prediction(2, "far"), self.prediction(3, "near")], 0.25,
        )
        self.assertEqual((result["strict_annotations"], result["matched"]), (1, 1))
        self.assertEqual(result["strict_subset_recall"], 1)
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["full_rally_recall"])
        self.assertIsNone(result["rallies"][0]["matches"][0]["family_agrees"])

    def test_one_candidate_cannot_cover_two_labels(self):
        result = score(
            [{"id": "sample", "hits": [self.hit(1, "near"), self.hit(1.2, "near")]}],
            [self.prediction(1.1, "near")], 0.25,
        )
        self.assertEqual((result["strict_annotations"], result["matched"]), (2, 1))
        self.assertEqual(result["strict_subset_recall"], 0.5)

    def test_roles_and_tolerance_are_required(self):
        result = score(
            [{"id": "sample", "hits": [self.hit(1, "near"), self.hit(3, "near")]}],
            [self.prediction(1, "far"), self.prediction(3.3, "near")], 0.25,
        )
        self.assertEqual(result["matched"], 0)

    def test_mislabeled_strict_annotation_is_an_error(self):
        hit = self.hit(1, "near")
        hit["conf"]["t"] = "low"
        with self.assertRaisesRegex(ValueError, "inconsistent strict"):
            score([{"id": "sample", "hits": [hit]}], [], 0.25)


if __name__ == "__main__":
    unittest.main()
