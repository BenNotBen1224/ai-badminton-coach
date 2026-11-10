import json
import tempfile
import time
import unittest
from pathlib import Path

import cv2

from coach.app import bench_e2e
from coach.app.backend import PROJECT
from coach.tests.test_app_pipeline import write_video


class TimedRunner:
    """Stands in for E2ERunner: tells clips apart by frame count, waits as told, writes the three files."""

    IS_REPLAY = False

    def __init__(self, plan: dict[int, object]):
        self.plan = plan
        self.calls: list[tuple[str, Path]] = []

    def warm_up(self, video):
        self.calls.append(("warm_up", Path(video)))

    def run(self, video, work_dir):
        self.calls.append(("run", Path(video)))
        capture = cv2.VideoCapture(str(video))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        action = self.plan[frames]
        if isinstance(action, Exception):
            raise action
        time.sleep(action)
        doc = {"frames": frames}
        paths = {key: Path(work_dir) / name for key, name in
                 (("pose_path", "pose.json"), ("report_path", "report.json"), ("bundle_path", "bundle.json"))}
        for path in paths.values():
            path.write_text(json.dumps(doc), "utf-8")
        if frames == 35:
            paths["pose_path"].write_text(json.dumps({"frames": 0}), "utf-8")
        timings = {stage: 0.001 for stage in bench_e2e.STAGES}
        return {"doc": doc, **{key: str(path) for key, path in paths.items()}, "timings": timings}


class BenchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.clips = {name: write_video(self.dir / f"{name}.mp4", frames, 30.0)
                      for name, frames in (("a", 20), ("b", 25), ("c", 30), ("d", 35))}
        self.warmup = write_video(self.dir / "warm.mp4", 15, 30.0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_each_clip_runs_in_turn_on_a_fresh_copy_and_slow_failed_or_wrong_runs_count_over_the_limit(self):
        runner = TimedRunner({20: 0.0, 25: 0.08, 30: RuntimeError("no court found in x"), 35: 0.0})
        scratch = self.dir / "scratch"
        result = bench_e2e.bench(runner, self.warmup, self.clips, 2, scratch, quiet=lambda: 4.0)

        self.assertEqual(runner.calls[0], ("warm_up", self.warmup))
        self.assertEqual([call[0] for call in runner.calls[1:]], ["run"] * 8)
        self.assertEqual([run["clip"] for run in result["runs"]], ["a", "b", "c", "d"] * 2)
        copies = [path for kind, path in runner.calls[1:]]
        self.assertEqual(len(set(copies)), 8)
        self.assertFalse(set(copies) & set(self.clips.values()))
        self.assertFalse([path for path in copies if path.exists()])
        self.assertEqual(list(scratch.glob("*-*")), [])

        summary = bench_e2e.summarise(result["runs"], limit_s=0.05)
        self.assertEqual({name: (clip["runs"], clip["failed"], clip["over"]) for name, clip in summary["clips"].items()},
                         {"a": (2, 0, 0), "b": (2, 0, 2), "c": (2, 2, 2), "d": (2, 2, 2)})
        self.assertEqual((summary["runs"], summary["over"], summary["verdict"]), (8, 6, "FAIL"))
        self.assertGreaterEqual(summary["clips"]["b"]["median_s"], 0.08)
        self.assertIsNone(summary["clips"]["c"]["median_s"])

        result.update(backend="test runner", machine=bench_e2e.machine(), summary=summary)
        text = bench_e2e.table(result)
        self.assertIn("Failed run: no court found in x", text)
        self.assertIn("Failed run: pose.json is not the document run() returned", text)
        self.assertIn("Runs not under 0.050 s (failed runs included): 6 of 8", text)
        self.assertIn("Diagnostic verdict, every run under 0.050 s: FAIL", text)
        self.assertIn("Not acceptance: this bench deletes each run's files", text)
        self.assertIn("Background CPU before each run: median 4.0 %, max 4.0 %", text)

    def test_every_run_under_the_limit_passes(self):
        runner = TimedRunner({20: 0.0, 25: 0.0})
        clips = {name: self.clips[name] for name in ("a", "b")}
        result = bench_e2e.bench(runner, self.warmup, clips, 3, self.dir / "scratch", quiet=lambda: 0.0)
        summary = bench_e2e.summarise(result["runs"], limit_s=5.0)
        self.assertEqual((summary["runs"], summary["over"], summary["verdict"]), (6, 0, "PASS"))
        self.assertEqual(summary["clips"]["a"]["stages"]["write"], 0.001)

    def test_no_runs_is_not_a_pass(self):
        self.assertEqual(bench_e2e.summarise([])["verdict"], "FAIL")

    def test_a_run_of_exactly_one_second_is_not_under_one_second(self):
        at = bench_e2e.summarise([{"clip": "a", "wall_s": 1.0, "timings": {}}])
        below = bench_e2e.summarise([{"clip": "a", "wall_s": 0.999, "timings": {}}])
        self.assertEqual((at["limit_s"], at["over"], at["verdict"]), (1.0, 1, "FAIL"))
        self.assertEqual((below["over"], below["verdict"]), (0, "PASS"))

    def test_the_clip_roles_come_from_turbo_clips_json(self):
        warmup, clips = bench_e2e.clip_table()
        self.assertEqual(warmup, PROJECT / "sample-videos" / "07-china-open-ws-d.mp4")
        self.assertEqual(list(clips), ["01", "02", "03", "04", "05", "08", "09", "10"])


if __name__ == "__main__":
    unittest.main()
