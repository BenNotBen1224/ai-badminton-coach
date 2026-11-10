import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fast"))
import evaluate


class ModelAgreement(unittest.TestCase):
    def test_missing_people_and_joints_reduce_integer_counts(self):
        people = [
            {"role": role, "bbox_xyxy": [x, 0, x + 20, 100],
             **{k: [[x, 0]] * 17 for k in evaluate.REFERENCES}}
            for role, x in (("near", 0), ("far", 100))]
        reference = {"frames": [{"people": people}, {"people": people}]}
        output = {"frames": [
            {"players": [{"track_id": 1, "joints": [[0, 0, 0.9]] * 16 + [None]}]},
            {"players": []},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            ref, pred = Path(tmp) / "ref.json", Path(tmp) / "pred.json"
            ref.write_text(json.dumps(reference), "utf-8")
            pred.write_text(json.dumps(output), "utf-8")
            score = evaluate.score(pred, evaluate.load_reference(ref))
        self.assertEqual(score["counts"], {
            "pck05": [16, 68], "pck10": [16, 68], "player_frames_found": [1, 4]})
        self.assertEqual(score["pck05_exact"], 16 / 68)
        self.assertEqual(score["player_frames_found_exact"], 0.25)


if __name__ == "__main__":
    unittest.main()
