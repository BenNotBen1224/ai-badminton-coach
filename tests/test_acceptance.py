import json
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fast"))
import acceptance  # noqa: E402

FPS, N = 30.0, 4


def good_doc(n=N):
    joints = [[1.0, 2.0, 0.9]] * 17
    return {"source": {"frame_count": n, "fps": FPS},
            "frames": [{"frame_index": i, "timestamp_seconds": round(i / FPS, 4),
                        "players": [{"track_id": 1, "box_confidence": 0.9,
                                     "bbox_xyxy": [0, 0, 5, 9], "joints": joints}]} for i in range(n)]}


class ValidateOutput(unittest.TestCase):
    def test_good_output_has_no_problems(self):
        self.assertEqual(acceptance.validate_output(good_doc(), N, FPS), [])

    def test_missing_frame_fails(self):
        d = good_doc()
        del d["frames"][2]
        self.assertTrue(acceptance.validate_output(d, N, FPS))

    def test_wrong_frame_index_fails(self):
        d = good_doc()
        d["frames"][1]["frame_index"] = 5
        self.assertTrue(acceptance.validate_output(d, N, FPS))

    def test_bad_timestamp_fails(self):
        d = good_doc()
        d["frames"][3]["timestamp_seconds"] = 9.0
        self.assertTrue(acceptance.validate_output(d, N, FPS))

    def test_nan_joint_fails(self):
        d = good_doc()
        d["frames"][0]["players"][0]["joints"][3] = [float("nan"), 1.0, 0.5]
        self.assertTrue(acceptance.validate_output(d, N, FPS))

    def test_wrong_joint_count_fails(self):
        d = good_doc()
        d["frames"][0]["players"][0]["joints"] = d["frames"][0]["players"][0]["joints"][:16]
        self.assertTrue(acceptance.validate_output(d, N, FPS))

    def test_no_players_anywhere_fails(self):
        d = good_doc()
        for f in d["frames"]:
            f["players"] = []
        self.assertTrue(acceptance.validate_output(d, N, FPS))

    def test_header_count_mismatch_fails(self):
        d = good_doc()
        d["source"]["frame_count"] = N + 1
        self.assertTrue(acceptance.validate_output(d, N, FPS))

    def test_null_joint_is_allowed(self):
        d = good_doc()
        d["frames"][0]["players"][0]["joints"][0] = None
        self.assertEqual(acceptance.validate_output(d, N, FPS), [])

    def test_invalid_frame_object_fails(self):
        d = good_doc()
        d["frames"][0] = None
        self.assertTrue(acceptance.validate_output(d, N, FPS))

    def test_invalid_player_object_fails(self):
        d = good_doc()
        d["frames"][0]["players"] = [None]
        self.assertTrue(acceptance.validate_output(d, N, FPS))

    def test_source_dimensions_are_checked(self):
        d = good_doc()
        d["source"].update(width=1920, height=1080)
        self.assertEqual(acceptance.validate_output(d, N, FPS, (1920, 1080)), [])
        self.assertTrue(acceptance.validate_output(d, N, FPS, (960, 540)))

    def test_duplicate_track_id_fails(self):
        d = good_doc()
        d["frames"][0]["players"] *= 2
        self.assertTrue(acceptance.validate_output(d, N, FPS))

    def test_joint_without_confidence_fails(self):
        d = good_doc()
        d["frames"][0]["players"][0]["joints"][0] = [1.0, 2.0]
        self.assertTrue(acceptance.validate_output(d, N, FPS))


class RunOne(unittest.TestCase):
    def run_child(self, body, limit=10.0, timeout=30):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            code = f"import json,sys\nout=sys.argv[1]\n{body}"
            rec = acceptance.run_one([sys.executable, "-c", code, str(out)], out, Path(tmp) / "logs", "t", N, FPS,
                                     limit, timeout)
            rec["_out_kept"] = out.exists()
            return rec

    def test_complete_output_passes_and_is_kept(self):
        rec = self.run_child(f"open(out,'w').write(json.dumps({good_doc()!r}))")
        self.assertTrue(rec["pass"], rec["validation_problems"])
        self.assertIsInstance(rec["wall_ns"], int)
        self.assertTrue(rec["_out_kept"])

    def test_nonzero_exit_fails_even_with_valid_output(self):
        rec = self.run_child(f"open(out,'w').write(json.dumps({good_doc()!r}))\nsys.exit(3)")
        self.assertFalse(rec["pass"])

    def test_exit_zero_without_output_fails(self):
        self.assertFalse(self.run_child("pass")["pass"])

    def test_truncated_output_fails(self):
        self.assertFalse(self.run_child("open(out,'w').write('{\"frames\": [')")["pass"])

    def test_incomplete_frames_fail(self):
        d = good_doc(N - 1)
        self.assertFalse(self.run_child(f"open(out,'w').write(json.dumps({d!r}))")["pass"])

    def test_too_slow_fails(self):
        rec = self.run_child(f"import time; time.sleep(0.3)\nopen(out,'w').write(json.dumps({good_doc()!r}))",
                             limit=0.1)
        self.assertTrue(rec["valid"])
        self.assertFalse(rec["pass"])

    def test_timeout_fails(self):
        self.assertFalse(self.run_child("import time; time.sleep(30)", timeout=1)["pass"])


class QualityGate(unittest.TestCase):
    base = {"01": {"pck05_exact": 0.9, "pck10_exact": 0.95, "player_frames_found_exact": 0.99}}

    def run_with(self, **over):
        s = {"pck05_exact": 0.9, "pck10_exact": 0.95, "player_frames_found_exact": 0.99, **over}
        return acceptance.quality_gate([{"tag": "01-r1", "clip": "01", "vs_judge": s}], self.base)

    def test_equal_passes(self):
        self.assertTrue(self.run_with()["passed"])

    def test_tiny_unrounded_deficit_fails(self):
        self.assertFalse(self.run_with(pck05_exact=0.9 - 1e-9)["passed"])

    def test_found_rate_deficit_fails(self):
        self.assertFalse(self.run_with(player_frames_found_exact=0.98)["passed"])

    def test_nan_score_fails(self):
        self.assertFalse(self.run_with(pck05_exact=float("nan"))["passed"])

    def test_integer_counts_take_precedence_over_rounded_ratio(self):
        score = {**self.base["01"], "counts": {"pck05": [8999, 10000]}}
        base = {"01": {**self.base["01"], "counts": {"pck05": [9000, 10000]}}}
        g = acceptance.quality_gate([{"tag": "x", "clip": "01", "vs_judge": score}], base)
        self.assertFalse(g["passed"])

    def test_missing_score_fails(self):
        g = acceptance.quality_gate([{"tag": "x", "clip": "01", "vs_judge": {"error": "boom"}}], self.base)
        self.assertFalse(g["passed"])


class Gate(unittest.TestCase):
    def test_one_failure_fails_the_gate(self):
        g = acceptance.gate([{"tag": "a", "pass": True}, {"tag": "b", "pass": False}])
        self.assertFalse(g["passed"])
        self.assertEqual(g["failed_tags"], ["b"])

    def test_empty_run_list_fails(self):
        self.assertFalse(acceptance.gate([])["passed"])

    def test_all_pass(self):
        self.assertTrue(acceptance.gate([{"tag": "a", "pass": True}])["passed"])


class Merge(unittest.TestCase):
    def doc(self, clip="01"):
        score = QualityGate.base["01"]
        return {"suite": clip, "setup": {
            "code_sha256": {"pipeline.py": "frozen"}, "model_sha256": {"s": "s", "t": "t"},
            "production_config": {"threshold": 1.05},
            "cache_unchanged": True, "code_unchanged": True, "models_unchanged": True},
            "clips": {clip: {"baseline_vs_judge": score}},
            "runs": [{"tag": f"{clip}-r{i}", "clip": clip, "pass": True, "vs_judge": copy.deepcopy(score)}
                     for i in range(5)]}

    def test_same_revision_suites_pass(self):
        self.assertTrue(all(g["passed"] for g in acceptance.merge_checks([self.doc(), self.doc("07")]).values()))

    def test_mixed_revision_is_rejected(self):
        other = self.doc("07")
        other["setup"]["model_sha256"]["t"] = "different"
        with self.assertRaisesRegex(ValueError, "different"):
            acceptance.merge_checks([self.doc(), other])

    def test_duplicate_run_is_rejected(self):
        other = self.doc("07")
        other["runs"][0]["tag"] = "01-r0"
        with self.assertRaisesRegex(ValueError, "duplicate run"):
            acceptance.merge_checks([self.doc(), other])

    def test_quality_failure_cannot_be_hidden_by_saved_pass(self):
        doc = self.doc()
        doc["runs"][0]["vs_judge"]["pck05_exact"] = 0.1
        self.assertFalse(acceptance.merge_checks([doc])["01"]["passed"])

    def test_incomplete_repetition_fails(self):
        doc = self.doc()
        doc["runs"].pop()
        self.assertFalse(acceptance.merge_checks([doc])["01"]["passed"])

    def test_secondary_regression_fails_even_when_primary_passes(self):
        doc = self.doc()
        secondary = copy.deepcopy(doc["runs"])
        secondary[0]["vs_judge"]["pck05_exact"] = .1
        doc["secondary_consensus"] = {"runs": secondary, "baseline": QualityGate.base["01"]}
        result = acceptance.merge_checks([doc])["01"]
        self.assertTrue(result["quality"]["passed"])
        self.assertFalse(result["secondary_passed"])
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
