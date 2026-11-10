import copy
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from coach.app.backend import (
    SAMPLES, STORED_DENSE, FastBackend, NoCourt, NotApplicable, ReplayBackend, Unreadable, classify, sha256_file,
)
from coach.synth import script
from coach.tracks import load_tracks

COURT_LINE = ("RuntimeError: no court found in clip.mp4 (best fit 0.12); the camera must be fixed behind a "
              "baseline, or pass --court x,y,... (8 numbers)")
TRACEBACK = 'Traceback (most recent call last):\n  File "fast\\pipeline.py", line 790, in <module>\n'


class ClassifyTest(unittest.TestCase):
    def test_frozen_pose_errors(self):
        self.assertIsInstance(classify(TRACEBACK + COURT_LINE + "\n"), NoCourt)
        self.assertIsInstance(classify("no court found in clip.mp4 (best fit 0.00); the camera"), NoCourt)
        self.assertIsInstance(classify(TRACEBACK + "RuntimeError: cannot read clip.mp4"), Unreadable)
        self.assertIsInstance(classify("RuntimeError: could not read frame 12"), Unreadable)
        self.assertIsInstance(classify("RuntimeError: decoded 10 of 20 frames"), Unreadable)

    def test_other_text_is_not_mapped(self):
        self.assertIsNone(classify("ValueError: bad thing"))
        self.assertIsNone(classify(COURT_LINE + "\nValueError: later"))
        self.assertIsNone(classify("the model says no court found"))
        self.assertIsNone(classify(""))


class FastBackendTest(unittest.TestCase):
    """The real subprocess path, with stand-in CLI scripts instead of the frozen pose models."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.video = self.dir / "clip.mp4"
        self.video.write_bytes(b"not decoded by the stand-in")

    def tearDown(self):
        self.tmp.cleanup()

    def cli(self, body: str) -> Path:
        path = self.dir / "stand_in.py"
        path.write_text("import json, sys, time\nvideo, out = sys.argv[1], sys.argv[2]\n" + textwrap.dedent(body),
                        "utf-8")
        return path

    def test_success_reads_the_written_document_and_stats(self):
        cli = self.cli("""
            json.dump({"schema_version": 3, "frames": [], "source": {"file": video}}, open(out, "w"))
            print("loading")
            print(json.dumps({"total_s": 1.5, "frames": 9, "crops": 4, "workers": 8}))
        """)
        out = self.dir / "pose.json"
        doc, timings = FastBackend(cli=cli).preview(self.video, out)
        self.assertEqual(doc, json.loads(out.read_text("utf-8")))
        self.assertEqual(doc["source"]["file"], str(self.video))
        self.assertEqual(timings["backend"], "fast")
        self.assertEqual((timings["total_s"], timings["frames"], timings["crops"]), (1.5, 9, 4))
        self.assertNotIn("workers", timings)
        self.assertGreater(timings["process_s"], 0.0)

    def test_without_out_path_uses_a_temporary_file(self):
        cli = self.cli("""
            json.dump({"schema_version": 3, "frames": []}, open(out, "w"))
            print(json.dumps({"total_s": 0.1}))
        """)
        doc, _ = FastBackend(cli=cli).refined(self.video)
        self.assertEqual(doc, {"schema_version": 3, "frames": []})

    def test_errors_map_to_types(self):
        cases = [(COURT_LINE, NoCourt), ("RuntimeError: decoded 3 of 9 frames", Unreadable)]
        for line, kind in cases:
            cli = self.cli(f"sys.stderr.write({TRACEBACK + line!r})\nsys.exit(1)\n")
            with self.subTest(kind=kind.__name__), self.assertRaises(kind):
                FastBackend(cli=cli).preview(self.video, self.dir / "pose.json")

    def test_other_failure_names_the_last_line(self):
        cli = self.cli("1 / 0\n")
        with self.assertRaisesRegex(RuntimeError, r"^fast pose failed: ZeroDivisionError: division by zero$"):
            FastBackend(cli=cli).preview(self.video, self.dir / "pose.json")

    def test_timeout(self):
        cli = self.cli("time.sleep(30)\n")
        with self.assertRaisesRegex(RuntimeError, "longer than 1 s"):
            FastBackend(cli=cli, timeout_s=1.0).preview(self.video, self.dir / "pose.json")

    def test_dense_documents_have_no_windows_to_sharpen(self):
        backend = FastBackend()
        self.assertFalse(backend.sharpens_windows)
        self.assertEqual(backend.cli.name, "pipeline.py")
        with self.assertRaises(NotApplicable):
            backend.refine(self.video, {}, [(0, 5)])


class ReplayBackendTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.video = self.dir / "clip.mp4"
        self.video.write_bytes(b"stored pose stands in for this file")
        self.dense, _ = script(n=40)
        self.backend = ReplayBackend({sha256_file(self.video): self.dense}, preview_stride=4)

    def tearDown(self):
        self.tmp.cleanup()

    def test_preview_is_a_sparse_preview_written_to_disk(self):
        out = self.dir / "pose.json"
        doc, timings = self.backend.preview(self.video, out)
        self.assertEqual(doc["tier"], "preview")
        self.assertEqual(doc["windows_refined"], [])
        self.assertEqual(json.loads(out.read_text("utf-8")), doc)
        self.assertEqual(timings["backend"], "replay")
        sources = {player.get("src") for frame in doc["frames"] for player in frame["players"]}
        self.assertIn("interpolated", sources)
        self.assertEqual(load_tracks(doc).tier.value, "preview")

    def test_refine_follows_the_window_contract(self):
        base, _ = self.backend.preview(self.video)
        before = copy.deepcopy(base)
        doc, _ = self.backend.refine(self.video, base, [(10, 14)])
        self.assertEqual(base, before)
        self.assertEqual(doc["tier"], "preview")
        for frame in doc["frames"]:
            index = frame["frame_index"]
            if 10 <= index <= 14:
                self.assertEqual(frame["players"], self.dense["frames"][index]["players"])
                self.assertTrue(all("src" not in player for player in frame["players"]))
            else:
                self.assertEqual(frame, base["frames"][index])
        for key in ("source", "court_lines_normalized", "court_fit", "cuts"):
            self.assertEqual(doc[key], base[key])
        self.assertEqual([(w["start"], w["end"]) for w in doc["windows_refined"]], [(10, 14)])
        again, _ = self.backend.refine(self.video, doc, [(20, 20)])
        self.assertEqual([(w["start"], w["end"]) for w in again["windows_refined"]], [(10, 14), (20, 20)])
        self.assertEqual(load_tracks(again).tier.value, "preview")

    def test_refine_rejects_a_reversed_window(self):
        base, _ = self.backend.preview(self.video)
        with self.assertRaises(ValueError):
            self.backend.refine(self.video, base, [(14, 10)])

    def test_refined_is_the_stored_dense_document(self):
        doc, _ = self.backend.refined(self.video)
        self.assertEqual(doc, self.dense)
        doc["frames"].clear()
        self.assertEqual(len(self.backend.refined(self.video)[0]["frames"]), 40)

    def test_unknown_video_is_an_error_not_a_fallback(self):
        other = self.dir / "other.mp4"
        other.write_bytes(b"different bytes")
        with self.assertRaisesRegex(RuntimeError, "replay has no stored pose"):
            self.backend.preview(other)

    def test_without_stride_the_document_is_dense(self):
        backend = ReplayBackend({sha256_file(self.video): self.dense})
        self.assertFalse(backend.sharpens_windows)
        self.assertEqual(backend.preview(self.video)[0], self.dense)
        with self.assertRaises(NotApplicable):
            backend.refine(self.video, self.dense, [(0, 1)])

    @unittest.skipUnless(STORED_DENSE["01-singles-rally"].is_file() and (SAMPLES / "01-singles-rally.mp4").is_file(),
                         "sample 01 or its stored pose is missing")
    def test_for_samples_reads_the_stored_fast_outputs(self):
        backend = ReplayBackend.for_samples()
        doc, _ = backend.refined(SAMPLES / "01-singles-rally.mp4")
        stored = json.loads(STORED_DENSE["01-singles-rally"].read_text("utf-8"))
        self.assertEqual(doc["source"]["frame_count"], stored["source"]["frame_count"])
        self.assertEqual(len(doc["frames"]), len(stored["frames"]))


if __name__ == "__main__":
    unittest.main()
