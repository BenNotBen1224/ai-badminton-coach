"""Synthetic checks for footwork and positional tactics.

Truth comes from coach.synth. Most checks build swing and lunge events from
that truth. One check calls analyze(), which runs detect_events, so a late
first step is judged on the public path.
"""
from __future__ import annotations

import json
import time
import unittest

import numpy as np

from coach.analyze import analyze, assemble
from coach.analyze_move import movement_analysis
from coach.catalog import METRIC_CATALOG
from coach.schema import Role, Severity, lunge_event, swing_event
from coach.signals_move import measured_gap
from coach.synth import Lunge, Noise, Path, PlayerPlan, SplitStep, Swing, make_preview, repeat_frames, script
from coach.tracks import load_tracks

FPS = 30.0
FRAME = 1.0 / FPS
DESCRIBE = {
    "footwork.split_step",
    "footwork.coverage",
    "tactics.base_discipline",
    "tactics.tempo",
    "tactics.initiative",
}


def _contact_doc():
    doc, _truth = script(
        n=90, fps=FPS,
        near=PlayerPlan(court_xy=(3.05, 9.7)),
        far=PlayerPlan(court_xy=(3.05, 2.9), swing=Swing(46, 2.2, 160)),
    )
    return doc, [_swing(Role.FAR, 46)]


def _blank_ankles(doc, role):
    for frame in doc["frames"]:
        for player in frame.get("players") or []:
            if player.get("role") != role:
                continue
            for index in (15, 16):
                player["joints"][index] = None


def _hide_samples(doc, role, start, end):
    for frame in doc["frames"]:
        if not start <= int(frame["frame_index"]) <= end:
            continue
        for player in frame.get("players") or []:
            if player.get("role") == role:
                player["src"] = "interpolated"


def _swing(role, frame, peak=None):
    peak = frame - 3 if peak is None else peak
    return swing_event(role, frame, start=max(0, peak), end=frame + 4, peak_frame=peak, peak_speed=2.0, overhead=True)


def _lunge_from(truth, role, offset=0):
    lg = truth["players"][role.value]["lunge"]
    return lunge_event(
        role, lg["start_frame"] + offset, depth=lg["depth_m"],
        bottom_frame=lg["bottom_frame"] + offset, end_frame=lg["end_frame"] + offset,
    )


def _stitch(docs):
    frames = []
    cuts = []
    fps = docs[0]["source"]["fps"]
    t_shift = 0.0
    for doc in docs:
        if frames:
            cuts.append(len(frames))
        for fr in doc["frames"]:
            item = {
                "frame_index": len(frames),
                "timestamp_seconds": t_shift + fr["timestamp_seconds"],
                "players": fr["players"],
            }
            frames.append(item)
        t_shift = frames[-1]["timestamp_seconds"] + 1.0 / fps
    out = {
        "schema_version": docs[0].get("schema_version", 3),
        "source": dict(docs[0]["source"]),
        "court_lines_normalized": docs[0]["court_lines_normalized"],
        "court_fit": docs[0].get("court_fit") or {},
        "cuts": cuts,
        "frames": frames,
    }
    out["source"]["frame_count"] = len(frames)
    return out


def _returns(part, role=Role.NEAR):
    return [e for e in part.events if e.kind == "base_return" and e.player == role]


def _home_contacts():
    return [_swing(Role.FAR, frame) for frame in (30, 50, 70)]


def _jitter(seed):
    return Noise(jitter_px=1.5, seed=seed)


def _leave_and_stay():
    doc, _ = script(
        n=240, fps=FPS, noise=_jitter(3),
        near=PlayerPlan(court_xy=(3.05, 9.7), path=Path(80, 95, (1.2, 12.4))),
        far=PlayerPlan(court_xy=(3.05, 2.9)),
    )
    events = _home_contacts() + [
        lunge_event(Role.NEAR, 102, depth=1.2, bottom_frame=110, end_frame=130),
    ]
    return movement_analysis(load_tracks(doc), events)


def _false_plant():
    doc, _ = script(
        n=180, fps=FPS, noise=_jitter(5),
        near=PlayerPlan(court_xy=(3.05, 9.7), path=Path(40, 55, (1.2, 12.4))),
        far=PlayerPlan(court_xy=(3.05, 2.9)),
    )
    tracks = load_tracks(doc)
    near = tracks.players[Role.NEAR]
    home = near.foot_court[10].copy()
    near.measured[90] = False
    near.foot_court[90] = home
    events = [_swing(Role.FAR, frame) for frame in (8, 16, 24)]
    events.append(lunge_event(Role.NEAR, 58, depth=1.2, bottom_frame=66, end_frame=80))
    return movement_analysis(tracks, events)


def _dropout_edge():
    doc, _ = script(
        n=180, fps=FPS, noise=_jitter(7),
        near=PlayerPlan(court_xy=(3.05, 9.7), path=Path(40, 55, (1.2, 12.4))),
        far=PlayerPlan(court_xy=(3.05, 2.9)),
    )
    tracks = load_tracks(doc)
    near = tracks.players[Role.NEAR]
    near.present[145:157] = False
    near.measured[145:157] = True
    near.foot_court[145:157] = near.foot_court[10]
    events = [_swing(Role.FAR, frame) for frame in (8, 16, 24)]
    events.append(lunge_event(Role.NEAR, 58, depth=1.2, bottom_frame=66, end_frame=80))
    return movement_analysis(tracks, events)


def _cut_before_return():
    doc, _ = script(
        n=180, fps=FPS, cuts=(100,), noise=_jitter(4),
        near=PlayerPlan(court_xy=(3.05, 9.7), path=Path(20, 35, (1.2, 12.4))),
        far=PlayerPlan(court_xy=(3.05, 2.9)),
    )
    tracks = load_tracks(doc)
    near = tracks.players[Role.NEAR]
    near.foot_court[100:] = near.foot_court[5]
    events = [_swing(Role.FAR, frame) for frame in (8, 12, 16)]
    events.append(lunge_event(Role.NEAR, 32, depth=1.2, bottom_frame=40, end_frame=90))
    return movement_analysis(tracks, events)


def _finding(part, fid, role=Role.NEAR):
    return next((f for f in part.findings if f.id == fid and f.player == role), None)


def _metric(finding, name):
    return next(m for m in finding.metrics if m.name == name)


class MoveSyntheticTests(unittest.TestCase):
    def test_positions_within_10cm(self):
        doc, truth = script(n=40, fps=FPS, near=PlayerPlan(court_xy=(3.05, 9.7)), far=PlayerPlan(court_xy=(2.4, 2.9)))
        part = movement_analysis(load_tracks(doc), [])
        for role in ("near", "far"):
            got = np.column_stack([part.series[role]["court_x"], part.series[role]["court_y"]])
            err = np.linalg.norm(got - truth["players"][role]["foot_court"], axis=1)
            self.assertLess(float(np.nanmax(err)), 0.10, role)

    def test_base_return_matches_lunge_truth(self):
        doc, truth = script(
            n=120, fps=FPS, noise=Noise(jitter_px=0.4, seed=1),
            near=PlayerPlan(court_xy=(3.05, 9.7), lunge=Lunge(30, 1.8, 0.50)),
            far=PlayerPlan(court_xy=(3.05, 2.9), swing=Swing(16, 2.2, 160)),
        )
        lg = truth["players"]["near"]["lunge"]
        part = movement_analysis(load_tracks(doc), [_lunge_from(truth, Role.NEAR), _swing(Role.FAR, 16)])
        got = [e for e in part.events if e.kind == "base_return" and e.player == Role.NEAR]
        self.assertEqual(len(got), 1)
        self.assertLessEqual(abs(got[0].frame - lg["end_frame"]), 3)
        self.assertLessEqual(abs(got[0].data["since_s"] - lg["recovery_s"]), 3 * FRAME + 1e-6)
        finding = _finding(part, "footwork.recovery")
        self.assertIsNotNone(finding)
        self.assertLessEqual(abs(_metric(finding, "recovery_s").value - lg["recovery_s"]), 3 * FRAME + 1e-6)

    def test_split_precision_and_recall(self):
        matched = 0
        truths = 0
        detected = 0
        for i in range(8):
            hop = 36 + i
            contact = hop + 6
            doc, truth = script(
                n=80, fps=FPS,
                near=PlayerPlan(court_xy=(3.05, 9.7), split_step=SplitStep(hop)),
                far=PlayerPlan(court_xy=(3.05, 2.9), swing=Swing(contact, 2.2, 160)),
            )
            part = movement_analysis(load_tracks(doc), [_swing(Role.FAR, contact, truth["players"]["far"]["swing"]["peak_frame"])])
            found = [e.frame for e in part.events if e.kind == "split_step" and e.player == Role.NEAR]
            landing = truth["players"]["near"]["split_step"]["frame"] + 1
            detected += len(found)
            truths += 1
            matched += int(any(abs(frame - landing) <= 3 for frame in found))
            self.assertTrue(found, i)
            event = next(e for e in part.events if e.kind == "split_step" and e.player == Role.NEAR)
            self.assertIn("onset_s", event.data)
            self.assertIn("landing_s", event.data)
            self.assertLess(event.data["onset_s"], event.data["landing_s"] + 1e-9)
        for i in range(4):
            contact = 50
            doc, truth = script(
                n=80, fps=FPS,
                near=PlayerPlan(court_xy=(3.05, 9.7)),
                far=PlayerPlan(court_xy=(3.05, 2.9), swing=Swing(contact, 2.2, 160)),
            )
            part = movement_analysis(load_tracks(doc), [_swing(Role.FAR, contact, truth["players"]["far"]["swing"]["peak_frame"])])
            found = [e for e in part.events if e.kind == "split_step" and e.player == Role.NEAR]
            detected += len(found)
        self.assertGreaterEqual(matched / truths, 0.85)
        self.assertGreaterEqual(matched / detected, 0.85)

    def test_no_return_omits_recovery(self):
        part = _leave_and_stay()
        self.assertEqual(_returns(part), [])
        self.assertIsNone(_finding(part, "footwork.recovery"))

    def test_repeat_or_dropout_is_not_a_plant(self):
        for part in (_false_plant(), _dropout_edge()):
            self.assertEqual(_returns(part), [])
            self.assertIsNone(_finding(part, "footwork.recovery"))

    def test_truncated_look_omits_recovery(self):
        part = _cut_before_return()
        self.assertEqual(_returns(part), [])
        self.assertIsNone(_finding(part, "footwork.recovery"))

    def test_observed_return_is_kept(self):
        doc, truth = script(
            n=120, fps=FPS, noise=Noise(jitter_px=0.4, seed=1),
            near=PlayerPlan(court_xy=(3.05, 9.7), lunge=Lunge(30, 1.8, 0.50)),
            far=PlayerPlan(court_xy=(3.05, 2.9), swing=Swing(16, 2.2, 160)),
        )
        lg = truth["players"]["near"]["lunge"]
        part = movement_analysis(load_tracks(doc), [_lunge_from(truth, Role.NEAR), _swing(Role.FAR, 16)])
        got = _returns(part)
        self.assertEqual(len(got), 1)
        self.assertNotEqual(got[0].frame, lg["bottom_frame"])
        self.assertLess(got[0].data["since_s"], 1.5)
        finding = _finding(part, "footwork.recovery")
        self.assertIsNotNone(finding)
        self.assertEqual(_metric(finding, "recovery_s").n_obs, 1)

    def test_unresolved_watch_does_not_enter_the_median(self):
        doc, truth = script(
            n=240, fps=FPS,
            near=PlayerPlan(
                court_xy=(3.05, 9.7),
                lunge=Lunge(30, 1.8, 0.50),
                path=Path(100, 115, (1.2, 12.4)),
            ),
            far=PlayerPlan(court_xy=(3.05, 2.9)),
            noise=_jitter(6),
        )
        lg = truth["players"]["near"]["lunge"]
        events = [
            _lunge_from(truth, Role.NEAR),
            lunge_event(Role.NEAR, 118, depth=1.0, bottom_frame=125, end_frame=140),
            _swing(Role.FAR, 16),
            _swing(Role.FAR, 50),
            _swing(Role.FAR, 75),
        ]
        part = movement_analysis(load_tracks(doc), events)
        got = _returns(part)
        self.assertEqual(len(got), 1)
        self.assertLessEqual(abs(got[0].data["since_s"] - lg["recovery_s"]), 3 * FRAME + 1e-6)
        finding = _finding(part, "footwork.recovery")
        self.assertEqual(_metric(finding, "recovery_s").n_obs, 1)
        self.assertLess(_metric(finding, "recovery_s").value, 1.0)

    def test_measured_gap_covers_edges(self):
        t = np.arange(40) / FPS
        interior = np.zeros(40, dtype=bool)
        interior[10:13] = True
        self.assertTrue(measured_gap(interior, t, 0, 39, 0.25))
        covered = np.ones(40, dtype=bool)
        self.assertFalse(measured_gap(covered, t, 0, 39, 0.25))
        leading = covered.copy()
        leading[:10] = False
        self.assertTrue(measured_gap(leading, t, 0, 39, 0.25))
        trailing = covered.copy()
        trailing[-10:] = False
        self.assertTrue(measured_gap(trailing, t, 0, 39, 0.25))
        one = covered.copy()
        one[0] = False
        self.assertFalse(measured_gap(one, t, 0, 39, 0.25))

    def test_severity_rises_when_recovery_is_slower(self):
        fast = _recovery_severity(0.30)
        slow = _recovery_severity(1.80)
        self.assertGreater(slow.rank, fast.rank)
        self.assertGreater(slow.value, fast.value)

    def test_public_late_step_stays_info(self):
        """A late first step on analyze() must not become advice. The swing is not a hit."""
        report = analyze(_late_step_doc(), video_id="late-step")
        finding = _finding(report, "footwork.first_step")
        self.assertIsNone(finding)
        self.assertFalse(any(e.kind == "swing" for e in report.events))
        # This synthetic clip has only one player's swings, not an exchange.
        from coach.tracks import load_tracks
        from coach.events import detect_events
        from coach.analyze_move import movement_analysis
        tracks = load_tracks(_late_step_doc())
        findings = movement_analysis(tracks, detect_events(tracks)).findings
        finding = next(f for f in findings if f.id == "footwork.first_step")
        metric = _metric(finding, "first_step_s")
        self.assertGreater(metric.value, 0.55, metric.value)
        self.assertGreaterEqual(metric.n_obs, 3)
        self.assertEqual(metric.target, (0.0, 0.35))
        self.assertEqual(finding.severity, Severity.INFO, finding.severity)
        self.assertLess(metric.confidence, 0.6)
        text = _movement_text(report.findings)
        move = [row for row in report.to_dict()["findings"] if row["category"] in ("footwork", "tactics")]
        blob = json.dumps(move)
        self.assertTrue(move)
        for banned in ("0.07", "two frames", "your shot", "their hit"):
            self.assertNotIn(banned, text, banned)
            self.assertNotIn(banned, blob, banned)
        self.assertNotIn(" hits", " " + text)
        self.assertIn("not a confirmed hit", finding.explain)
        self.assertIn("not validated personal norms", finding.explain)
        self.assertIn("arm", finding.explain)

    def test_swing_anchored_return_is_not_a_hit(self):
        part = _swing_anchored_recovery()
        got = _returns(part)
        self.assertGreaterEqual(len(got), 3, "measured return was dropped")
        finding = _finding(part, "footwork.recovery")
        self.assertIsNotNone(finding)
        metric = _metric(finding, "recovery_s")
        self.assertGreater(metric.value, 0.90)
        self.assertGreaterEqual(metric.n_obs, 3)
        self.assertEqual(finding.severity, Severity.INFO, finding.severity)
        self.assertLess(metric.confidence, 0.6)
        self.assertNotIn("your shot", finding.explain)
        self.assertIn("not a confirmed hit", finding.explain)
        self.assertIn("does not prove", finding.explain)
        self.assertIn("not validated personal norms", finding.explain)

    def test_swing_return_does_not_enter_the_lunge_median(self):
        part = _mixed_returns()
        finding = _finding(part, "footwork.recovery")
        self.assertIsNotNone(finding)
        metric = _metric(finding, "recovery_s")
        self.assertEqual(metric.n_obs, 1)
        self.assertLess(metric.value, 0.80, metric.value)
        self.assertIn("left out", finding.explain)

    def test_descriptive_findings_stay_info(self):
        events = [_swing(Role.FAR, frame) for frame in (20, 40, 60, 80)]
        events += [_swing(Role.NEAR, frame) for frame in (30, 50, 70)]
        doc, _ = script(n=100, fps=FPS, near=PlayerPlan(court_xy=(3.05, 9.7)), far=PlayerPlan(court_xy=(3.05, 2.9)))
        part = movement_analysis(load_tracks(doc), events)
        seen = {f.id for f in part.findings}
        self.assertTrue(DESCRIBE & seen, seen)
        for finding in part.findings:
            if finding.id in DESCRIBE:
                self.assertEqual(finding.severity, Severity.INFO, finding.id)
                self.assertNotIn("0.10 to 0.35", finding.explain)
                self.assertNotIn("0.10-0.35", finding.explain)
            self.assertIn("club players", finding.explain)
            self.assertIn("not validated personal norms", finding.explain)
            self.assertNotIn("0.07", finding.explain)
            self.assertNotIn("two frames", finding.explain)
            self.assertNotIn("their hit", finding.headline + finding.explain)
            if finding.category.value == "tactics":
                self.assertIn("shuttle", finding.explain)
            self.assertLessEqual(len(finding.headline), 90)
            self.assertLessEqual(len(finding.explain), 400)
            for metric in finding.metrics:
                self.assertEqual(metric.target, METRIC_CATALOG[metric.name].target)

    def test_repeat_frames_add_no_phantom_steps(self):
        plan = PlayerPlan(court_xy=(3.05, 9.7), path=Path(8, 70, (4.8, 9.7)))
        clean, _ = script(n=90, fps=FPS, near=plan, far=PlayerPlan(court_xy=(3.05, 2.9)))
        dirty, truth = script(n=90, fps=FPS, near=plan, far=PlayerPlan(court_xy=(3.05, 2.9)))
        repeated = set(repeat_frames(dirty, 6, truth))
        clean_steps = {e.frame for e in movement_analysis(load_tracks(clean), []).events if e.kind == "step"}
        part = movement_analysis(load_tracks(dirty), [])
        dirty_steps = [e for e in part.events if e.kind == "step" and e.player == Role.NEAR]
        self.assertTrue(repeated)
        self.assertFalse(repeated & {e.frame for e in dirty_steps})
        self.assertLessEqual(len(dirty_steps), len(clean_steps))
        clean_speed = movement_analysis(load_tracks(clean), []).series["near"]["foot_speed"]
        speed = part.series["near"]["foot_speed"]
        for frame in repeated:
            if not np.isfinite(speed[frame]):
                continue
            self.assertTrue(np.isfinite(clean_speed[frame]))
            self.assertLess(abs(float(speed[frame] - clean_speed[frame])), 0.35, frame)

    def test_noise_and_dropout_do_not_crash(self):
        doc, truth = script(
            n=80, fps=FPS,
            near=PlayerPlan(court_xy=(3.05, 9.7), split_step=SplitStep(40)),
            far=PlayerPlan(court_xy=(3.05, 2.9), swing=Swing(46, 2.2, 160)),
            noise=Noise(jitter_px=2.5, dropout=0.08, seed=3),
        )
        part = movement_analysis(load_tracks(doc), [_swing(Role.FAR, 46, truth["players"]["far"]["swing"]["peak_frame"])])
        self.assertIsInstance(part.findings, tuple)
        for finding in part.findings:
            self.assertGreaterEqual(finding.score, 0.0)
            self.assertLessEqual(max(m.confidence for m in finding.metrics), 0.95)

    def test_preview_does_not_raise_confidence(self):
        doc, truth = script(
            n=80, fps=FPS,
            near=PlayerPlan(court_xy=(3.05, 9.7), split_step=SplitStep(40), lunge=Lunge(18, 1.8, 0.40)),
            far=PlayerPlan(court_xy=(3.05, 2.9), swing=Swing(46, 2.2, 160)),
        )
        events = [_swing(Role.FAR, 46, truth["players"]["far"]["swing"]["peak_frame"]), _lunge_from(truth, Role.NEAR)]
        refined = movement_analysis(load_tracks(doc), events)
        preview = movement_analysis(load_tracks(make_preview(doc, 4)), events)
        refined_conf = {m.name: m.confidence for f in refined.findings for m in f.metrics if f.player == Role.NEAR}
        preview_conf = {m.name: m.confidence for f in preview.findings for m in f.metrics if f.player == Role.NEAR}
        shared = set(refined_conf) & set(preview_conf)
        self.assertTrue(shared)
        for name in shared:
            self.assertLessEqual(preview_conf[name], refined_conf[name] + 1e-9, name)

    def test_sparse_unjudgeable_look_still_asks(self):
        for plan in (
            PlayerPlan(court_xy=(3.05, 9.7)),
            PlayerPlan(court_xy=(3.05, 9.7), path=Path(0, 89, (4.8, 12.4))),
        ):
            doc, _truth = script(
                n=90, fps=FPS,
                near=plan,
                far=PlayerPlan(court_xy=(3.05, 2.9), swing=Swing(46, 2.2, 160)),
            )
            preview = make_preview(doc, 8)
            _blank_ankles(preview, "near")
            part = movement_analysis(load_tracks(preview), [_swing(Role.FAR, 46)])
            reasons = [window.reason for window in part.windows]
            self.assertTrue(any("unjudgeable" in reason for reason in reasons))
            self.assertFalse(any(event.kind == "split_step" for event in part.events))

    def test_measured_dropout_asks_without_confirming_a_hit(self):
        doc, _truth = script(
            n=90, fps=FPS,
            near=PlayerPlan(court_xy=(3.05, 9.7), path=Path(0, 89, (4.8, 12.4))),
            far=PlayerPlan(court_xy=(3.05, 2.9), swing=Swing(46, 2.2, 160)),
        )
        preview = make_preview(doc, 4)
        _blank_ankles(preview, "near")
        _hide_samples(preview, "near", 20, 75)
        part = movement_analysis(load_tracks(preview), [_swing(Role.FAR, 46)])
        reasons = [window.reason for window in part.windows]
        self.assertTrue(any("unjudgeable" in reason for reason in reasons))
        self.assertFalse(any(event.kind == "split_step" for event in part.events))

    def test_no_court_drops_metre_findings(self):
        doc, truth = script(
            n=80, fps=FPS,
            near=PlayerPlan(court_xy=(3.05, 9.7), split_step=SplitStep(40)),
            far=PlayerPlan(court_xy=(3.05, 2.9), swing=Swing(46, 2.2, 160)),
        )
        doc = dict(doc)
        doc.pop("court_lines_normalized", None)
        doc.pop("court_fit", None)
        tracks = load_tracks(doc)
        self.assertIsNone(tracks.court)
        part = movement_analysis(tracks, [_swing(Role.FAR, 46, truth["players"]["far"]["swing"]["peak_frame"]), _swing(Role.NEAR, 60)])
        self.assertIn("no_court", part.warnings)
        dropped = {"footwork.recovery", "footwork.first_step", "footwork.step_pattern", "footwork.coverage", "tactics.base_discipline", "tactics.initiative"}
        self.assertFalse(dropped & {f.id for f in part.findings})
        self.assertTrue({"footwork.split_step", "tactics.tempo"} & {f.id for f in part.findings})

    def test_report_assembles(self):
        doc, truth = script(
            n=80, fps=FPS,
            near=PlayerPlan(court_xy=(3.05, 9.7), lunge=Lunge(24, 1.8, 0.45), split_step=SplitStep(48)),
            far=PlayerPlan(court_xy=(3.05, 2.9), swing=Swing(54, 2.2, 160)),
        )
        events = [_lunge_from(truth, Role.NEAR), _swing(Role.FAR, 54, truth["players"]["far"]["swing"]["peak_frame"])]
        tracks = load_tracks(doc)
        part = movement_analysis(tracks, events)
        report = assemble(tracks, events, (part,), METRIC_CATALOG, "synth", 1)
        self.assertGreaterEqual(len(report.findings), 4)
        self.assertTrue(report.series["near"]["court_x"])

    def test_speed_budget(self):
        doc, _ = script(n=900, fps=FPS)
        tracks = load_tracks(doc)
        events = [_swing(Role.FAR, frame) for frame in range(30, 860, 40)]
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            movement_analysis(tracks, events)
            times.append(time.perf_counter() - t0)
        self.assertLess(min(times), 0.040, times)
        self.assertLess(max(times), 0.100, times)


class _Rank:
    def __init__(self, severity, value):
        self.rank = {"strength": 0, "info": 1, "improve": 2, "priority": 3}[severity.value]
        self.value = value


def _late_step_doc():
    docs = []
    for _ in range(3):
        doc, _truth = script(
            n=90, fps=FPS,
            near=PlayerPlan(court_xy=(3.05, 9.7), path=Path(32, 44, (1.2, 12.4))),
            far=PlayerPlan(court_xy=(3.05, 2.9), swing=Swing(20, 2.2, 160)),
        )
        docs.append(doc)
    return _stitch(docs)


def _swing_anchored_recovery():
    docs = []
    events = []
    for i in range(3):
        doc, truth = script(
            n=120, fps=FPS, noise=Noise(jitter_px=0.4, seed=9),
            near=PlayerPlan(court_xy=(3.05, 9.7), lunge=Lunge(40, 1.8, 1.80)),
            far=PlayerPlan(court_xy=(3.05, 2.9)),
        )
        docs.append(doc)
        bottom = truth["players"]["near"]["lunge"]["bottom_frame"] + i * 120
        events.append(_swing(Role.NEAR, bottom, peak=bottom - 3))
    return movement_analysis(load_tracks(_stitch(docs)), events)


def _mixed_returns():
    fast, fast_truth = script(
        n=120, fps=FPS, noise=Noise(jitter_px=0.4, seed=8),
        near=PlayerPlan(court_xy=(3.05, 9.7), lunge=Lunge(40, 1.8, 0.40)),
        far=PlayerPlan(court_xy=(3.05, 2.9)),
    )
    slow, slow_truth = script(
        n=120, fps=FPS, noise=Noise(jitter_px=0.4, seed=9),
        near=PlayerPlan(court_xy=(3.05, 9.7), lunge=Lunge(40, 1.8, 1.80)),
        far=PlayerPlan(court_xy=(3.05, 2.9)),
    )
    bottom = slow_truth["players"]["near"]["lunge"]["bottom_frame"] + 120
    events = [
        _lunge_from(fast_truth, Role.NEAR),
        _swing(Role.NEAR, bottom, peak=bottom - 3),
    ]
    return movement_analysis(load_tracks(_stitch([fast, slow])), events)


def _movement_text(findings):
    chunks = []
    for finding in findings:
        if finding.category.value not in ("footwork", "tactics"):
            continue
        chunks.extend((finding.headline, finding.explain))
        chunks.extend(metric.label for metric in finding.metrics)
        chunks.extend(moment.label for moment in finding.moments)
    return "\n".join(chunks)


def _recovery_severity(recovery_s):
    docs = []
    truths = []
    for _ in range(3):
        doc, truth = script(
            n=120, fps=FPS, noise=Noise(jitter_px=0.4, seed=8),
            near=PlayerPlan(court_xy=(3.05, 9.7), lunge=Lunge(40, 1.8, recovery_s)),
            far=PlayerPlan(court_xy=(3.05, 2.9), swing=Swing(18, 2.2, 160)),
        )
        docs.append(doc)
        truths.append(truth)
    events = []
    for i, truth in enumerate(truths):
        off = i * 120
        events.append(_lunge_from(truth, Role.NEAR, off))
        events.append(_swing(Role.FAR, 18 + off, truth["players"]["far"]["swing"]["peak_frame"] + off))
    part = movement_analysis(load_tracks(_stitch(docs)), events)
    finding = _finding(part, "footwork.recovery")
    if finding is None:
        raise AssertionError("recovery finding missing")
    return _Rank(finding.severity, _metric(finding, "recovery_s").value)


if __name__ == "__main__":
    unittest.main()
