import functools
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from coach.app import pipeline
from coach.app.backend import (
    DEFAULT_MODELS, PROJECT, SAMPLES, FastBackend, NoCourt, ReplayBackend, Unreadable, sha256_file,
)
from coach.app.pipeline import (
    E2ERunner, analyze_video, exercise_ids, finding_exercises, players_found, probe, revise,
)
from coach.bundle import build_bundle
from coach.exercises.library import for_tag, load_library
from coach.schema import report_from_dict
from coach.synth import Noise, script
from coach.tests.fixture_report import full_report

N = 90
FPS = 30.0
STAGES = {"probe", "pose", "pose_stages", "load_tracks", "events", "body", "move", "assemble", "bundle", "write",
          "total"}


def write_video(path: Path, frames: int, fps: float, size=(64, 36)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    try:
        for i in range(frames):
            writer.write(np.full((size[1], size[0], 3), (i * 7) % 255, np.uint8))
    finally:
        writer.release()
    return path


def opencv_numbers(path: Path) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()


def box(kind: bytes, body: bytes = b"", large: bool = False) -> bytes:
    if large:
        return struct.pack(">I4sQ", 1, kind, 16 + len(body)) + body
    return struct.pack(">I4s", 8 + len(body), kind) + body


def mp4(stts=((90, 512),), timescale=15360, size=(64, 36), codec=b"avc1", entries=1, handler=b"vide", tracks=1,
        mdhd_version=0, moov_extra=b"", first=b"ftyp", large=False) -> bytes:
    """Only the boxes probe reads; there are no frames, so OpenCV cannot open it."""
    b = functools.partial(box, large=large)
    before, after = (20, 12) if mdhd_version == 1 else (12, 8)
    mdhd = bytes([mdhd_version]) + bytes(before - 1) + struct.pack(">I", timescale) + bytes(after)
    entry = b(codec, bytes(24) + struct.pack(">HH", *size) + bytes(50))
    table = struct.pack(">II", 0, len(stts)) + b"".join(struct.pack(">II", *row) for row in stts)
    stbl = b(b"stbl", b(b"stsd", struct.pack(">II", 0, entries) + entry * entries) + b(b"stts", table))
    trak = b(b"trak", b(b"mdia", b(b"mdhd", mdhd) + b(b"hdlr", bytes(8) + handler + bytes(13)) + b(b"minf", stbl)))
    return b(first, b"isom" + bytes(4) + b"isomavc1") + b(b"moov", trak * tracks + moov_extra)


class StubBackend:
    name = "stub"

    def __init__(self, error: Exception):
        self.error = error

    def preview(self, video, out_json=None):
        raise self.error


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.video = write_video(cls.dir / "clip.mp4", N, FPS)
        cls.digest = sha256_file(cls.video)
        cls.dense, _ = script(n=N, fps=FPS)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def out(self, name: str) -> Path:
        return self.dir / name

    def test_probe_reads_the_header(self):
        self.assertEqual(probe(self.video), (N, FPS))
        self.assertEqual(probe(write_video(self.out("three-min.mp4"), 180, 1.0)), (180, 1.0))
        self.assertEqual(probe(write_video(self.out("too-long.mp4"), 182, 1.0)).code, "too_long")

    def test_frames_too_large_for_the_bundle_are_refused_before_pose(self):
        self.assertEqual(probe(write_video(self.out("widest.mp4"), 2, FPS, size=(3276, 16))), (2, FPS))
        wide = write_video(self.out("wide.mp4"), 2, FPS, size=(3278, 16))
        session = analyze_video(wide, StubBackend(AssertionError("pose must not run")), self.out("wide"), "wide")
        self.assertEqual(session.error.code, "too_large")
        self.assertEqual(session.error.message, "this video is more than 3276 pixels across; save it at 1080p and try again")
        tall = write_video(self.out("tall.mp4"), 2, FPS, size=(16, 3278))
        self.assertEqual(probe(tall).code, "too_large")

    def test_plain_h264_clips_are_probed_without_opening_a_decoder(self):
        clips = sorted(SAMPLES.glob("*.mp4"))
        expected = {clip.name: opencv_numbers(clip) for clip in clips}
        with mock.patch.object(pipeline.cv2, "VideoCapture", side_effect=AssertionError("a decoder was opened")):
            self.assertEqual({clip.name: probe(clip) for clip in clips}, expected)
            self.assertEqual(probe(self.header("plain.mp4", mp4())), (90, 30.0))
            self.assertEqual(probe(self.header("64-bit-sizes.mp4", mp4(large=True))), (90, 30.0))
            self.assertEqual(probe(self.header("mdhd-v1.mp4", mp4(mdhd_version=1))), (90, 30.0))
            self.assertEqual(probe(self.header("long.mp4", mp4(stts=((182, 15360),)))).code, "too_long")
            self.assertEqual(probe(self.header("wide.mp4", mp4(size=(3278, 16)))).code, "too_large")

    def test_any_other_layout_is_left_to_opencv(self):
        plain = mp4()
        layouts = {
            "two-frame-durations": mp4(stts=((89, 512), (1, 1024))),
            "fragmented": mp4(moov_extra=box(b"mvex", box(b"trex", bytes(24)))),
            "two-tracks": mp4(tracks=2),
            "sound-only": mp4(handler=b"soun"),
            "mpeg4-part2": mp4(codec=b"mp4v"),
            "two-sample-entries": mp4(entries=2),
            "zero-timescale": mp4(timescale=0),
            "no-ftyp-first": mp4(first=b"free"),
            "cut-short": plain[:len(plain) // 2],
            "empty": b"",
        }
        for name, data in layouts.items():
            with self.subTest(name), mock.patch.object(pipeline.cv2, "VideoCapture") as opened:
                opened.return_value.isOpened.return_value = False
                self.assertEqual(probe(self.header(f"{name}.mp4", data)).code, "not_a_video")
                opened.assert_called_once()

    def header(self, name: str, data: bytes) -> Path:
        path = self.out(name)
        path.write_bytes(data)
        return path

    def test_a_clip_becomes_three_agreeing_files(self):
        session = analyze_video(self.video, ReplayBackend({self.digest: self.dense}), self.out("ok"), "abc")
        self.assertIsNone(session.error)
        self.assertEqual((session.video_id, session.revision, session.duration_s), ("abc", 0, N / FPS))
        self.assertEqual(json.loads(session.pose_path.read_text("utf-8")), session.doc)
        report = report_from_dict(json.loads(session.report_path.read_text("utf-8")))
        self.assertEqual((report.n_frames, report.video_id, report.revision), (N, "abc", 0))
        bundle = json.loads(session.bundle_path.read_text("utf-8"))
        self.assertEqual(bundle, session.bundle)
        self.assertEqual((bundle["n_frames"], bundle["video_url"]), (N, "/media/abc"))
        for path in (session.pose_path, session.report_path, session.bundle_path):
            self.assertEqual(path.parent, self.out("ok"))
        self.assertEqual(set(session.timings), STAGES)
        self.assertEqual(session.timings["pose_stages"]["backend"], "replay")
        parts = sum(session.timings[k] for k in STAGES - {"probe", "pose_stages", "total"})
        self.assertGreaterEqual(session.timings["total"], parts)

    def test_files_that_are_not_videos(self):
        text = self.out("notes.mp4")
        text.write_text("hello", "utf-8")
        empty = self.out("empty.mp4")
        empty.write_bytes(b"")
        for path in (text, empty, self.out("missing.mp4")):
            with self.subTest(path=path.name):
                session = analyze_video(path, ReplayBackend({}), self.out("bad-" + path.stem), "bad")
                self.assertEqual(session.error.code, "not_a_video")
                self.assertEqual(session.error.message, "this file is not a video I can read")
                self.assertFalse(self.out("bad-" + path.stem).exists())
                self.assertIn("total", session.timings)

    def test_too_long(self):
        long_video = write_video(self.out("long.mp4"), 182, 1.0)
        session = analyze_video(long_video, ReplayBackend({}), self.out("long"), "long")
        self.assertEqual(session.error.code, "too_long")
        self.assertEqual(session.error.message, "this clip is longer than 3 minutes")

    def test_one_player_missing(self):
        doc, _ = script(n=N, fps=FPS, noise=Noise(missing_role="far", missing_frames=tuple(range(75))))
        session = analyze_video(self.video, ReplayBackend({self.digest: doc}), self.out("one"), "one")
        self.assertEqual(session.error.code, "no_players")
        self.assertTrue(session.pose_path.is_file())
        self.assertIsNone(session.report_path)
        self.assertFalse(self.out("one/report.json").exists())

    def test_players_found_threshold(self):
        for missing, found in ((72, True), (73, False)):
            doc, _ = script(n=N, fps=FPS, noise=Noise(missing_role="near", missing_frames=tuple(range(missing))))
            with self.subTest(present=N - missing):
                self.assertIs(players_found(doc), found)

    def test_backend_errors_become_analysis_errors(self):
        cases = [(NoCourt("no court found in x"), "no_court"), (Unreadable("decoded 3 of 9 frames"), "not_a_video")]
        for error, code in cases:
            with self.subTest(code=code):
                session = analyze_video(self.video, StubBackend(error), self.out(code), code)
                self.assertEqual(session.error.code, code)
        with self.assertLogs("coach.app", "ERROR") as logs:
            session = analyze_video(self.video, StubBackend(ValueError("boom")), self.out("boom"), "boom")
        self.assertEqual(session.error.code, "failed")
        self.assertEqual(session.error.message, "something went wrong while analysing this clip")
        self.assertIn("ValueError: boom", "\n".join(logs.output))

    def test_a_sharpened_document_becomes_a_new_revision_folder(self):
        backend = ReplayBackend({self.digest: self.dense}, preview_stride=3)
        first = analyze_video(self.video, backend, self.out("rev/r0"), "rev")
        self.assertIsNone(first.error)
        self.assertEqual(first.report.tier.value, "preview")
        before = {p.name: p.read_bytes() for p in self.out("rev/r0").iterdir()}
        r1 = self.out("rev/r1")
        r1.mkdir()
        doc, stages = backend.refine(self.video, first.doc, [(10, 20)], r1 / "pose.json")
        second = revise(doc, stages, 0.25, "rev", 1, r1)
        self.assertIsNone(second.error)
        self.assertEqual((second.revision, second.report.revision), (1, 1))
        self.assertEqual(second.report.tier.value, "preview")
        self.assertEqual(json.loads(second.pose_path.read_text("utf-8")), doc)
        self.assertEqual(report_from_dict(json.loads(second.report_path.read_text("utf-8"))).revision, 1)
        self.assertEqual(second.bundle_path.parent, r1)
        self.assertEqual({p.name: p.read_bytes() for p in self.out("rev/r0").iterdir()}, before)
        self.assertGreaterEqual(second.timings["total"], 0.25)


class ExerciseIdsTest(unittest.TestCase):
    def test_each_finding_gets_its_own_picks(self):
        report = full_report()
        library = load_library()
        chosen = exercise_ids(report)
        self.assertEqual(set(chosen), {(f.id, f.player.value) for f in report.findings})
        for finding in report.findings:
            ids = chosen[(finding.id, finding.player.value)]
            with self.subTest(finding=finding.id, player=finding.player.value):
                self.assertLessEqual(len(ids), 3)
                self.assertEqual(len(ids), len(set(ids)))
                for exercise_id in ids:
                    self.assertTrue(set(library[exercise_id].fix_tags) & set(finding.fix_tags))
                if not finding.fix_tags:
                    self.assertEqual(ids, [])

    def test_two_players_with_the_same_finding_id_get_their_own_cards(self):
        report = full_report()
        library = load_library()
        lean = {f.player.value: f for f in report.findings if f.id == "technique.trunk_lean"}
        self.assertNotEqual(lean["near"].fix_tags, lean["far"].fix_tags)
        bundle = build_bundle({}, report, video_url="/media/x", exercises=exercise_ids(report))
        cards = {(f["id"], f["player"]): f["exercise_ids"] for f in bundle["findings"]}
        near = cards[("technique.trunk_lean", "near")]
        far = cards[("technique.trunk_lean", "far")]
        self.assertEqual(near, finding_exercises(library, lean["near"].fix_tags))
        self.assertEqual(far, finding_exercises(library, lean["far"].fix_tags))
        self.assertNotEqual(near, far)

    def test_tags_take_turns(self):
        library = load_library()
        tags = ("core_stability", "core_rotation")
        firsts = [for_tag(library, tag, limit=1)[0].id for tag in tags]
        self.assertEqual(finding_exercises(library, tags)[:2], firsts)
        self.assertEqual(finding_exercises(library, ()), [])


class E2ERunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.video = write_video(cls.dir / "clip.mp4", N, FPS)
        dense, _ = script(n=N, fps=FPS)
        cls.backend = ReplayBackend({sha256_file(cls.video): dense})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_replay_runner_is_marked_and_returns_the_runner_protocol(self):
        runner = E2ERunner(DEFAULT_MODELS, backend=self.backend)
        self.assertTrue(runner.IS_REPLAY)
        self.assertFalse(E2ERunner.IS_REPLAY)
        work = self.dir / "work"
        work.mkdir()
        result = runner.run(self.video, work)
        self.assertEqual(set(result), {"doc", "pose_path", "report_path", "bundle_path", "timings"})
        for key in ("pose_path", "report_path", "bundle_path"):
            self.assertEqual(Path(result[key]).parent, work)
        self.assertEqual(json.loads(Path(result["pose_path"]).read_text("utf-8")), result["doc"])
        self.assertEqual(report_from_dict(json.loads(Path(result["report_path"]).read_text("utf-8"))).n_frames, N)

    def test_errors_raise(self):
        bad = self.dir / "bad.mp4"
        bad.write_text("not a video", "utf-8")
        with self.assertRaisesRegex(RuntimeError, "^this file is not a video I can read$"):
            E2ERunner(DEFAULT_MODELS, backend=self.backend).run(bad, self.dir / "bad")

    def test_default_runner_is_the_live_turbo_service(self):
        from turbo.service import TurboService

        runner = E2ERunner(DEFAULT_MODELS)
        self.assertIsInstance(runner.svc, TurboService)
        self.assertFalse(runner.IS_REPLAY)
        self.assertEqual(runner.svc.models_dir.resolve(), DEFAULT_MODELS.resolve())
        self.assertEqual(runner.models_dir.resolve(), DEFAULT_MODELS.resolve())

    def test_fast_runner_only_uses_the_fast_weights(self):
        runner = E2ERunner(DEFAULT_MODELS, backend=FastBackend())
        self.assertIsNone(runner.svc)
        self.assertFalse(runner.IS_REPLAY)
        with self.assertRaises(ValueError):
            E2ERunner(self.dir, backend=FastBackend())

    def test_binding_manifest_names_real_sources_and_assets(self):
        import coach.analyze
        import coach.bundle

        manifest = E2ERunner(DEFAULT_MODELS, backend=self.backend).binding_manifest()
        for path in manifest["sources"] + manifest["assets"]:
            self.assertTrue(Path(path).is_file(), path)
        for module in (pipeline, coach.analyze, coach.bundle):
            self.assertIn(str(Path(module.__file__).resolve()), manifest["sources"])
        self.assertFalse([p for p in manifest["sources"] if "tests" in Path(p).parts])
        self.assertIn(str((PROJECT / "coach" / "exercises" / "library.json").resolve()), manifest["assets"])
        fast = E2ERunner(DEFAULT_MODELS, backend=FastBackend()).binding_manifest()
        self.assertIn(str((PROJECT / "fast" / "pipeline.py").resolve()), fast["sources"])


if __name__ == "__main__":
    unittest.main()
