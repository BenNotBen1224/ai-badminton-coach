from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from badminton_coach.__main__ import require_ffmpeg
from coach.analyze import analyze
import coach.export as export
from coach.export import export_review
from coach.schema import (
    SCHEMA_VERSION,
    Basis,
    Category,
    Finding,
    Mark,
    Metric,
    Moment,
    Report,
    Role,
    Severity,
    Tier,
)
from coach.synth import script
from coach.timeline import build_review, review_picks

PROJECT = Path(__file__).resolve().parents[2]
LOCK = PROJECT / "work" / "cpu.lock"
DOC04 = PROJECT / "fast" / "runs" / "step4-rebalanced-dev" / "outputs" / "04-r1.json"
VIDEO04 = PROJECT / "sample-videos" / "04-china-open-ws-a.mp4"
REAL_OUT = PROJECT / "work" / "C2" / "real"
OUT_W = 1920
OUT_H = 1080


def _joint(doc, frame, role, index):
    row = next(item for item in doc["frames"][frame]["players"] if item["role"] == role)
    x, y, _conf = row["joints"][index]
    return (float(x), float(y))


def _report(doc):
    n = int(doc["source"]["frame_count"])
    fps = float(doc["source"]["fps"])
    metric = Metric(
        "recovery_time_s",
        "Time back to base after your shot",
        1.12,
        "s",
        Basis.PROJECTED_2D,
        0.74,
        (0.6, 0.9),
        "base_distance",
        11,
        0.78,
    )
    hip = _joint(doc, 16, "near", 11)
    ankle_l = _joint(doc, 16, "near", 15)
    ankle_r = _joint(doc, 16, "near", 16)
    shoulder_l = _joint(doc, 16, "near", 5)
    shoulder_r = _joint(doc, 16, "near", 6)
    issue = Moment(
        Role.NEAR, 8, 16, 30, "Slow way back to base", "issue",
        (
            Mark("dot", 16, (ankle_l,), "Left ankle"),
            Mark("label", 16, ((12.0, 12.0),), "Corner"),
            Mark("label", 16, (hip,), "Hip"),
            Mark("line", 16, (ankle_l, ankle_r)),
            Mark("bracket", 16, (shoulder_l, shoulder_r)),
            Mark("band", 16, ((40.0, 40.0), (120.0, 90.0)), style="target"),
            Mark("angle", 16, (shoulder_l, hip, ankle_l)),
            Mark("trail", 16, (ankle_l, hip, shoulder_l, (hip[0] + 12.0, hip[1]))),
        ),
    )
    best_hip = _joint(doc, 36, "near", 11)
    best = Moment(
        Role.NEAR, 32, 36, 44, "A faster return", "best",
        (Mark("dot", 36, (best_hip,), "Best rep"),),
    )
    finding = Finding(
        "footwork.recovery",
        Category.FOOTWORK,
        Role.NEAR,
        Severity.IMPROVE,
        "Slow return to base after your own shots",
        "After you hit, it takes you longer than the practice guide to get back near the middle of your half.",
        (metric,),
        (issue, best),
        Tier.REFINED,
        focus_joints=("left_ankle", "right_ankle"),
        overlay_series=("base_distance",),
        fix_tags=("base_recovery",),
        score=0.66,
    )
    return Report(
        schema_version=SCHEMA_VERSION,
        source={
            "width": doc["source"]["width"],
            "height": doc["source"]["height"],
            "frame_count": n,
            "fps": fps,
        },
        tier=Tier.REFINED,
        n_frames=n,
        fps=fps,
        players=(Role.NEAR, Role.FAR),
        metrics={},
        series={"near": {"base_distance": tuple(0.4 + (i % 5) * 0.05 for i in range(n))}},
        events=(),
        findings=(finding,),
        warnings=("preview_estimate",),
        timestamps=tuple(i / fps for i in range(n)),
    )


def _write_clip(path, n, width, height, fps):
    ffmpeg, _ffprobe = require_ffmpeg()
    proc = subprocess.Popen(
        [
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-preset", "ultrafast", "-g", "1",
            "-pix_fmt", "yuv420p", str(path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        for index in range(n):
            frame = np.zeros((height, width, 3), np.uint8)
            frame[:] = (40, 90, 40)
            x = 8 + (index * 11) % (width - 80)
            cv2.rectangle(frame, (x, 40), (x + 70, 180), (20, 20, 200), -1)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.stdin = None
        err = proc.stderr.read()
        code = proc.wait()
        if code != 0:
            raise RuntimeError(err.decode("utf-8", "replace"))
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if proc.stderr is not None:
            proc.stderr.close()
        if proc.stdin is not None:
            proc.stdin.close()


def _probe(path):
    _ffmpeg, ffprobe = require_ffmpeg()
    completed = subprocess.run(
        [
            str(ffprobe), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,codec_name:format=duration",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(completed.stdout)
    stream = data["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "codec": stream["codec_name"],
        "duration": float(data["format"]["duration"]),
    }


def _decoded_frames(path):
    ffmpeg, _ffprobe = require_ffmpeg()
    with tempfile.TemporaryDirectory() as raw_dir:
        raw = Path(raw_dir) / "frames.raw"
        subprocess.run(
            [
                str(ffmpeg), "-hide_banner", "-v", "error", "-i", str(path),
                "-f", "rawvideo", "-pix_fmt", "bgr24", str(raw),
            ],
            check=True,
            capture_output=True,
        )
        frame_bytes = OUT_W * OUT_H * 3
        size = raw.stat().st_size
        if size % frame_bytes != 0:
            raise AssertionError(f"raw decode size {size} is not a whole frame")
        return size // frame_bytes


class ExportTests(unittest.TestCase):
    def test_short_synth_decodes_to_the_timeline_frame_count(self):
        doc, _truth = script(n=48, fps=30.0, width=640, height=360, cuts=(22,))
        report = _report(doc)
        timelines = build_review(report, None, report.fps, limit_s=30)
        timeline_s = sum(item.duration_s for item in timelines)
        expected = max(1, round(timeline_s * report.fps))
        lock_before = LOCK.read_text(encoding="utf-8") if LOCK.is_file() else None
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "synth.mp4"
            out = root / "review.mp4"
            _write_clip(video, 48, 640, 360, 30.0)
            print(
                f"export_review starting expected_frames={expected} timeline_s={timeline_s:.3f}",
                file=sys.stderr,
            )
            result = export_review(video, doc, report, out)
            print(
                f"render_s={result['render_s']:.3f} "
                f"seconds_per_output_second={result['seconds_per_output_second']:.3f} "
                f"frames={result['frames']}",
                file=sys.stderr,
            )
            self.assertEqual(result["frames"], expected)
            self.assertAlmostEqual(result["duration_s"], timeline_s, delta=1e-9)
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 0)
            probed = _probe(out)
            self.assertEqual(probed["width"], OUT_W)
            self.assertEqual(probed["height"], OUT_H)
            self.assertEqual(probed["codec"], "h264")
            self.assertLessEqual(abs(probed["duration"] - timeline_s), 0.5)
            decoded = _decoded_frames(out)
            self.assertEqual(decoded, expected)
            self.assertEqual(decoded, result["frames"])
        lock_after = LOCK.read_text(encoding="utf-8") if LOCK.is_file() else None
        self.assertEqual(lock_after, lock_before)

    def test_clip04_full_export_is_opt_in(self):
        if os.environ.get("C2_FULL_EXPORT") != "1":
            self.skipTest("C2_FULL_EXPORT is not 1")
        self.assertTrue(DOC04.is_file(), str(DOC04))
        self.assertTrue(VIDEO04.is_file(), str(VIDEO04))
        doc = json.loads(DOC04.read_text(encoding="utf-8"))
        report = analyze(doc)
        self.assertGreater(len(report.findings), 0)
        self.assertEqual(report.n_frames, int(doc["source"]["frame_count"]))
        picks = review_picks(report)
        timelines = build_review(report, None, report.fps, limit_s=30)
        timeline_s = sum(item.duration_s for item in timelines)
        expected = max(1, round(timeline_s * report.fps))
        self.assertGreaterEqual(timeline_s, 25.0)
        self.assertLessEqual(timeline_s, 30.0)
        self.assertEqual(
            [(item.id, item.player.value) for item in picks[:3]],
            [(item.id, item.player.value) for item in report.top_findings(3)],
        )
        REAL_OUT.mkdir(parents=True, exist_ok=True)
        out = REAL_OUT / "review.mp4"
        print(
            f"real export starting findings={len(report.findings)} "
            f"included={len(timelines)} expected_frames={expected} timeline_s={timeline_s:.3f}",
            file=sys.stderr,
        )
        result = export_review(VIDEO04, doc, report, out)
        probed = _probe(out)
        decoded = _decoded_frames(out)
        log = {
            "source_doc": str(DOC04),
            "video": str(VIDEO04),
            "findings": len(report.findings),
            "included": [
                {
                    "id": finding.id,
                    "player": finding.player.value,
                    "severity": finding.severity.value,
                    "headline": finding.headline,
                    "duration_s": timeline.duration_s,
                    "freeze_frame": next(
                        segment.start_frame
                        for segment in timeline.segments
                        if segment.kind == "freeze"
                    ),
                }
                for finding, timeline in zip(picks, timelines)
            ],
            "expected_frames": expected,
            "timeline_s": timeline_s,
            "result": result,
            "probed": probed,
            "decoded": decoded,
        }
        (REAL_OUT / "export.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        self.assertEqual(result["frames"], expected)
        self.assertAlmostEqual(result["duration_s"], timeline_s, delta=1e-9)
        self.assertEqual(probed["width"], OUT_W)
        self.assertEqual(probed["height"], OUT_H)
        self.assertEqual(probed["codec"], "h264")
        self.assertLessEqual(abs(probed["duration"] - timeline_s), 0.5)
        self.assertEqual(decoded, expected)
        self.assertEqual(decoded, result["frames"])


_PANEL_STYLE = {"panel": {"background": "#F6F4F1", "text": "#1C1C1C", "muted": "#57534E"}}


def _reading(value, n_obs, confidence, best, label="Hip height", target=(0.42, 0.48)):
    return Metric(
        "hip_height_ratio",
        label,
        value,
        "ratio",
        Basis.PROJECTED_2D,
        confidence,
        target,
        "hip_height_ratio",
        n_obs,
        best,
    )


def _panel_finding(metrics):
    return SimpleNamespace(
        headline="Sit deeper between shots",
        explain="The clip has no measured reading.",
        category="technique",
        overlay_series=(),
        metrics=metrics,
    )


def _paint(finding):
    texts = []
    valued = []
    confidences = []
    real_write = export._write
    real_value = export._value
    real_confidence = export._confidence

    def capture_write(draw, text, font, fill, x, y, width, floor, line_h):
        texts.append(text)
        return real_write(draw, text, font, fill, x, y, width, floor, line_h)

    def capture_value(metric):
        valued.append(metric.value)
        return real_value(metric)

    def capture_confidence(value):
        confidences.append(value)
        return real_confidence(value)

    with (
        patch.object(export, "_write", capture_write),
        patch.object(export, "_value", capture_value),
        patch.object(export, "_confidence", capture_confidence),
    ):
        image = export._panel(
            finding,
            None,
            "issue",
            [],
            _PANEL_STYLE,
            SimpleNamespace(warnings=()),
            {},
            {},
            0,
            0.0,
        )
    return image, texts, valued, confidences


class PanelReadingTests(unittest.TestCase):
    def test_missing_reading_omits_your_number_and_keeps_the_guide(self):
        image, texts, valued, confidences = _paint(_panel_finding((
            _reading(None, 0, 0.0, None),
            Metric(
                "knee_bend_ratio",
                "Second reading",
                None,
                "ratio",
                Basis.PROJECTED_2D,
                0.0,
                None,
                None,
                0,
                None,
            ),
        )))
        self.assertIsNotNone(image)
        joined = "\n".join(texts)
        self.assertIn("Sit deeper between shots", texts)
        self.assertIn("Practice guide", texts)
        self.assertNotIn("Your number, clip summary", texts)
        self.assertNotIn("1.2", joined)
        self.assertNotIn("Second reading", texts)
        self.assertFalse(any(text in ("low", "medium", "high") for text in texts))
        self.assertEqual(valued, [])
        self.assertEqual(confidences, [])

    def test_finite_reading_still_paints_your_number(self):
        _image, texts, valued, confidences = _paint(_panel_finding((
            _reading(0.53, 4, 0.0, None),
        )))
        self.assertIn("Your number, clip summary", texts)
        self.assertIn("0.53 ratio", texts)
        self.assertIn("low", texts)
        self.assertEqual(valued, [0.53])
        self.assertEqual(confidences, [0.0])
