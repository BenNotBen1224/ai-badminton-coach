from __future__ import annotations

import json
import unittest
import tempfile
from pathlib import Path

import numpy as np

from coach.analyze import analyze
from coach.analyze_body import _near_play
from coach.complete_review import make_schedule, select_observations, _wrap_zh, _voice, _number, METRIC_LABELS
from coach.play_state import exchange_events
from coach.schema import Role, Severity, swing_event
from coach.synth import script
from coach.tracks import load_tracks


def swing(role, frame):
    return swing_event(role, frame, start=frame, end=frame, peak_frame=frame,
                       peak_speed=4, overhead=False)


class PlayStateTests(unittest.TestCase):
    def tracks(self, cuts=()):
        doc, _ = script(n=150, cuts=cuts)
        return load_tracks(doc)

    def test_cut_cannot_connect_players_or_pad_into_pre_serve(self):
        tracks = self.tracks((60,))
        events = [swing(Role.NEAR, 55), swing(Role.FAR, 65),
                  swing(Role.NEAR, 100), swing(Role.FAR, 120)]
        mask = _near_play(tracks, events)
        self.assertEqual(np.flatnonzero(mask).tolist(), list(range(65, 121)))
        self.assertFalse(mask[:65].any())
        self.assertFalse(mask[121:].any())

    def test_only_alternating_neighbors_support_play(self):
        tracks = self.tracks()
        events = [swing(Role.FAR, 40), swing(Role.FAR, 60),
                  swing(Role.FAR, 80), swing(Role.NEAR, 100)]
        self.assertEqual(np.flatnonzero(_near_play(tracks, events)).tolist(), list(range(80, 101)))
        self.assertEqual([e.frame for e in exchange_events(tracks, events)], [80, 100])

    def test_no_events_is_unknown_not_ready(self):
        self.assertFalse(_near_play(self.tracks(), []).any())
        doc, _ = script(n=150)
        report = analyze(doc)
        for finding in report.findings:
            if finding.id in ("technique.stay_lower", "technique.knee_bend"):
                self.assertIsNone(finding.metrics[0].value)
                self.assertEqual(finding.moments, ())
                self.assertEqual(finding.severity, Severity.INFO)

    def test_large_gap_is_not_play(self):
        tracks = self.tracks()
        self.assertFalse(_near_play(tracks, [swing(Role.NEAR, 10), swing(Role.FAR, 140)]).any())

    def test_real_pause_defect_and_no_unvalidated_posture_judgments(self):
        path = Path(__file__).resolve().parents[2] / "fast" / "runs" / "step4-rebalanced-dev" / "outputs" / "04-r1.json"
        from coach.player_input import player_inputs
        doc = json.loads(path.read_text("utf-8"))
        doc["source"]["players"] = player_inputs(path.parents[4] / "sample-videos" / "04-china-open-ws-a.mp4")
        report = analyze(doc)
        for finding in report.findings:
            if finding.id.startswith("technique."):
                self.assertEqual(finding.severity, Severity.INFO)
                self.assertEqual(finding.fix_tags, ())
                self.assertIsNone(finding.best_moment)
                for moment in finding.moments:
                    self.assertNotIn(moment.key_frame, (668, 680, 385, 704))
                    self.assertFalse(any(mark.style == "target" for mark in moment.marks))
        rows = select_observations(doc, report)
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["player"] for row in rows}, {"near", "far"})


class ScheduleTests(unittest.TestCase):
    def test_written_report_distinguishes_counts_distances_and_shares(self):
        from types import SimpleNamespace
        self.assertEqual(_number(SimpleNamespace(value=0.779, unit="m")), "0.78 米")
        self.assertEqual(_number(SimpleNamespace(value=0.8, unit="share")), "80%")
        self.assertEqual(_number(SimpleNamespace(value=2, unit="count")), "2 次")
        self.assertNotEqual(METRIC_LABELS["steps_before_lunge"], METRIC_LABELS["lunge_launch_m"])

    def test_missing_neural_voice_is_an_explicit_error_not_silent_video(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "Local Chinese neural TTS is unavailable"):
                _voice(folder)

    def test_narration_fits_and_every_original_frame_survives(self):
        rows = [{"source_start_frame": 20, "source_end_frame": 40, "key_frame": 30,
                 "finding_id": "technique.lunge_depth", "player": "near"}]
        frames, entries = make_schedule(90, 30, rows, [3.1, 6.8, 2.9])
        self.assertEqual(sorted(set(frames)), list(range(90)))
        self.assertTrue(all(a <= b for a, b in zip(frames, frames[1:])))
        self.assertGreaterEqual(rows[0]["output_end_s"] - rows[0]["output_start_s"], 6.8)
        self.assertEqual(frames[round(rows[0]["output_freeze_s"] * 30)], 30)
        self.assertEqual([entry["kind"] for entry in entries], ["intro", "observation", "outro"])

    def test_chinese_wrapping_never_exceeds_panel(self):
        from PIL import Image, ImageDraw, ImageFont
        draw = ImageDraw.Draw(Image.new("RGB", (540, 1080)))
        font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 28)
        text = "近端球员的投影膝角不代表三维关节角度，也不能直接证明技术好坏。"
        lines = _wrap_zh(draw, text, font, 464)
        self.assertEqual("".join(lines), text)
        self.assertTrue(all(draw.textlength(line, font=font) <= 464 for line in lines))


if __name__ == "__main__":
    unittest.main()
