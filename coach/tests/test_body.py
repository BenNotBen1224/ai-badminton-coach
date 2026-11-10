"""Technique and contact findings against synthetic truth."""
from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from coach.analyze import assemble
from coach.analyze_body import (
    HIP_MOVE_FLOOR, NOT_A_HIT, PRACTICE, WRIST_MOVE_FLOOR, _arm, _hand_speed, _knee_bend,
    _lunge_depth, _reach, _ready_mask, _signal_reps, _stance_width, _stay_lower, _trunk_lean,
    _value, body_analysis,
)
from coach.catalog import METRIC_CATALOG
from coach.catalog_body import BODY_SPECS
from coach.events import detect_events
from coach.schema import Category, FINDING_CATALOG, Partial, Role, Severity, SEVERITY_RANK, Tier, swing_event
from coach.signals_body import (
    elbow_deg, hip_height_ratio, knee_flexion_deg, stance_width_ratio, standing_hip_px,
    trunk_lean_deg, wrist_height_ratio,
)
from coach.synth import Lunge, Noise, PlayerPlan, Posture, Swing, make_preview, script
from coach.tracks import JOINT_INDEX, load_tracks


def _played():
    return script(
        n=80,
        fps=30,
        near=PlayerPlan(
            posture=Posture(knee_flexion_deg=25, stance_width_m=0.40, trunk_lean_deg=8),
            swing=Swing(30, 2.05, 170),
            lunge=Lunge(50, 0.80, 0.4),
        ),
        far=PlayerPlan(court_xy=(3.05, 2.9), swing=Swing(45, 1.90, 160)),
    )


def _run(doc):
    tracks = load_tracks(doc)
    events = detect_events(tracks)
    body = body_analysis(tracks, events)
    report = assemble(tracks, events, (body, Partial()), METRIC_CATALOG)
    return tracks, events, body, report


def _metric(body, role, name):
    return next(metric for metric in body.metrics[role.value] if metric.name == name)


def _sharpened_knees(windows):
    """Preview document whose listed windows were replaced with measured rows."""
    doc, _ = script(
        n=160,
        fps=30,
        near=PlayerPlan(posture=Posture(knee_flexion_deg=0), swing=Swing(40, 2.05, 170)),
    )
    preview = make_preview(doc, 4)
    dense = {frame["frame_index"]: frame for frame in doc["frames"]}
    for frame in preview["frames"]:
        frame["players"] = copy.deepcopy(dense[frame["frame_index"]]["players"])
        for player in frame["players"]:
            player.pop("src", None)
    preview["windows_refined"] = [
        {"start": start, "end": end, "s": 0.1} for start, end in windows
    ]
    events = [
        swing_event(Role.FAR, 20, start=16, end=20, peak_frame=17, peak_speed=1.2, overhead=True),
        swing_event(Role.NEAR, 40, start=36, end=40, peak_frame=37, peak_speed=1.2, overhead=True),
        swing_event(Role.FAR, 60, start=56, end=60, peak_frame=57, peak_speed=1.2, overhead=True),
        swing_event(Role.NEAR, 80, start=76, end=80, peak_frame=77, peak_speed=1.2, overhead=True),
        swing_event(Role.FAR, 100, start=96, end=100, peak_frame=97, peak_speed=1.2, overhead=True),
        swing_event(Role.NEAR, 120, start=116, end=120, peak_frame=117, peak_speed=1.2, overhead=True),
        swing_event(Role.FAR, 140, start=136, end=140, peak_frame=137, peak_speed=1.2, overhead=True),
    ]
    return load_tracks(preview), events


def _finding(body, finding_id):
    return next(item for item in body.findings if item.id == finding_id and item.player == Role.NEAR)


class BodyTests(unittest.TestCase):
    def test_catalog_covers_technique_and_contact(self):
        owned = {fid for fid, (category, _) in FINDING_CATALOG.items() if category in (Category.TECHNIQUE, Category.CONTACT)}
        self.assertEqual({spec.finding_id for spec in BODY_SPECS}, owned)

    def test_metrics_are_within_five_percent_of_truth(self):
        doc, truth = _played()
        tracks, _events, body, report = _run(doc)
        near = truth["players"]["near"]
        knee = _metric(body, Role.NEAR, "ready_knee_flexion")
        lean = _metric(body, Role.NEAR, "ready_trunk_lean")
        elbow = _metric(body, Role.NEAR, "contact_elbow")
        reach = _metric(body, Role.NEAR, "contact_reach")
        self.assertLess(abs(knee.value - 25.0) / 25.0, 0.05)
        self.assertLess(abs(lean.value - 8.0) / 8.0, 0.05)
        self.assertLess(abs(elbow.value - 170.0) / 170.0, 0.05)
        expected_reach = near["swing"]["achieved_contact_height_m"] / near["stature_m"]
        self.assertLess(abs(reach.value - expected_reach) / expected_reach, 0.05)
        self.assertEqual(report.n_frames, 80)
        for named in body.series.values():
            for values in named.values():
                self.assertEqual(len(values), tracks.n)
                self.assertFalse(np.isinf(np.asarray(values, dtype=float)).any())

    def test_static_knees_do_not_establish_ready_stance_or_severity(self):
        for flexion in (25, 12, 0):
            doc, _ = script(n=90, cuts=(30, 60), near=PlayerPlan(posture=Posture(knee_flexion_deg=flexion)))
            _tracks, _events, body, _built = _run(doc)
            finding = next(item for item in body.findings if item.id == "technique.knee_bend" and item.player == Role.NEAR)
            self.assertIsNone(finding.metrics[0].value)
            self.assertEqual(finding.metrics[0].n_obs, 0)
            self.assertEqual(finding.severity, Severity.INFO)
            self.assertEqual(finding.moments, ())

    def test_every_explanation_calls_the_range_a_practice_guide(self):
        doc, _ = _played()
        _tracks, _events, body, _built = _run(doc)
        self.assertGreaterEqual(len(body.findings), 8)
        for finding in body.findings:
            self.assertIn(PRACTICE, finding.explain)
            self.assertNotIn("center of mass", finding.explain.lower())
            self.assertNotIn("centre of mass", finding.explain.lower())
            self.assertLessEqual(len(finding.headline), 90)
            self.assertLessEqual(len(finding.explain), 400)
        stay = [item for item in body.findings if item.id == "technique.stay_lower"]
        self.assertTrue(stay)
        self.assertTrue(all("projected hip height" in item.explain for item in stay))

    def test_visible_moments_carry_marks(self):
        doc, _ = _played()
        _tracks, _events, body, _built = _run(doc)
        near = [item for item in body.findings if item.player == Role.NEAR]
        self.assertEqual(len(near), 8)
        for finding in near:
            self.assertTrue(finding.moments)
            for moment in finding.moments:
                self.assertTrue(moment.marks)
                for mark in moment.marks:
                    self.assertLessEqual(len(mark.label), 40)
                    self.assertTrue(mark.points)
        knee = next(item for item in near if item.id == "technique.knee_bend")
        self.assertEqual(knee.severity, Severity.INFO)
        self.assertIsNone(knee.best_moment)
        self.assertEqual(knee.fix_tags, ())

    def test_noise_and_a_deep_crouch_do_not_crash(self):
        noisy, _ = script(
            n=80,
            near=PlayerPlan(swing=Swing(30, 2.05, 170), lunge=Lunge(50, 0.80, 0.4)),
            noise=Noise(jitter_px=2.0, dropout=0.1, seed=2),
        )
        tracks, events, _body, _built = _run(noisy)
        self.assertLessEqual(sum(event.kind == "swing" for event in events), 6)
        crouched, _ = script(n=30, near=PlayerPlan(posture=Posture(knee_flexion_deg=80), swing=Swing(20, 1.6, 150)))
        _tracks, events, body, _built = _run(crouched)
        self.assertEqual([event for event in events if event.kind == "swing"], [])
        self.assertIn("few_swings", body.warnings)

    def test_preview_findings_stay_uncertain(self):
        doc, _ = script(n=40, near=PlayerPlan(swing=Swing(20, 2.05, 170)))
        refined, events, body, _built = _run(doc)
        self.assertEqual(body.windows, ())
        preview = load_tracks(make_preview(doc, 4))
        self.assertEqual(preview.tier.value, "preview")
        found = detect_events(preview)
        sampled = body_analysis(preview, found)
        for finding in sampled.findings:
            self.assertEqual(finding.severity, Severity.INFO)
            for metric in finding.metrics:
                self.assertLessEqual(metric.confidence, 0.55)
        self.assertGreater(refined.n, 0)

    def test_regular_sample_with_movement_still_asks_for_a_closer_look(self):
        self.assertEqual(WRIST_MOVE_FLOOR, 0.20)
        self.assertEqual(HIP_MOVE_FLOOR, 0.06)
        doc, _ = script(n=48, fps=30, near=PlayerPlan())
        preview = make_preview(doc, 8)
        for frame in preview["frames"]:
            frame["players"] = [row for row in frame["players"] if row["role"] == "near"]
        tracks = load_tracks(preview)
        player = tracks.players[Role.NEAR]
        frames = list(range(0, player.n, 8))
        player.measured[:] = False
        player.measured[frames] = True
        self.assertEqual(float(np.median(np.diff(frames))), 8.0)
        stature = 200.0
        player.stature_px[:] = stature
        for name in ("left_hip", "right_hip", "left_ankle", "right_ankle", "right_wrist", "left_wrist"):
            index = JOINT_INDEX[name]
            origin = player.xy[0, index].copy()
            for frame in frames:
                player.xy[frame, index] = origin
                player.visible[frame, index] = True
        shift = np.array([0.0, 0.25 * stature])
        for name in ("right_wrist", "left_wrist"):
            index = JOINT_INDEX[name]
            moved = player.xy[0, index] + shift
            for frame in (16, 24, 32, 40):
                player.xy[frame, index] = moved
        small = np.array([0.10 * stature, 0.0])
        for name in ("right_wrist", "left_wrist"):
            index = JOINT_INDEX[name]
            for frame in (32, 40):
                player.xy[frame, index] = player.xy[24, index] + small
        events = [swing_event(
            Role.NEAR, 4, start=2, end=6, peak_frame=4, peak_speed=4.0,
            overhead=False, right_hand=True, confidence=0.8,
        )]
        body = body_analysis(tracks, events)
        asked = [window for window in body.windows if window.reason == "unresolved body move"]
        self.assertEqual([(window.start_frame, window.end_frame) for window in asked], [(8, 16)])

    def test_between_rally_standing_is_not_a_ready_sample(self):
        doc, _ = script(n=150, near=PlayerPlan(swing=Swing(120, 2.05, 170)))
        tracks = load_tracks(doc)
        player = tracks.players[Role.NEAR]
        player.xy[:61, 11, 1] -= 28
        player.xy[:61, 12, 1] -= 28
        events = detect_events(tracks)
        mask = _ready_mask(player, tracks, events)
        self.assertFalse(bool(mask[:50].any()))
        self.assertFalse(bool(mask.any()))
        body = body_analysis(tracks, events)
        stay = next(item for item in body.findings if item.id == "technique.stay_lower" and item.player == Role.NEAR)
        self.assertEqual(stay.moments, ())
        self.assertIsNone(stay.metrics[0].value)

    def test_a_one_player_pickup_is_not_ready_during_an_exchange(self):
        doc, _ = script(
            n=180,
            near=PlayerPlan(swing=Swing(130, 2.05, 170)),
            far=PlayerPlan(court_xy=(3.05, 2.9), swing=Swing(140, 1.90, 160)),
        )
        tracks = load_tracks(doc)
        events = list(detect_events(tracks))
        events.append(swing_event(
            Role.NEAR, 12, start=10, end=14, peak_frame=11, peak_speed=4.0,
            overhead=False, right_hand=True, confidence=0.8,
        ))
        mask = _ready_mask(tracks.players[Role.NEAR], tracks, events)
        self.assertFalse(bool(mask[:40].any()))
        self.assertFalse(bool(mask[100:125].any()))
        self.assertTrue(bool(mask[130:141].any()))

    def test_contact_findings_do_not_claim_a_confirmed_hit(self):
        doc, _ = _played()
        _tracks, _events, body, _built = _run(doc)
        contact = [item for item in body.findings if item.id.startswith("contact.")]
        self.assertEqual(len(contact), 6)
        for finding in contact:
            self.assertIn(NOT_A_HIT, finding.explain)
            self.assertNotIn("meets the shuttle", finding.explain.lower())
            self.assertNotIn("meets the shuttle", finding.headline.lower())
            self.assertNotIn("through the shot", finding.explain.lower())
            self.assertNotIn("through the shot", finding.headline.lower())
            self.assertEqual(finding.severity, Severity.INFO)
            self.assertLess(finding.metrics[0].confidence, 0.6)

    def test_unresolved_recovery_is_a_warning_not_a_measured_lunge(self):
        from coach.tests.test_events import _drop, _hide_knees, _standing, _widen

        tracks = _standing(100)
        _drop(tracks, range(30, 100), 0.16)
        _widen(tracks, range(30, 100), 0.18)
        _hide_knees(tracks, range(30, 100))
        events = detect_events(tracks)
        self.assertFalse(any(event.kind == "lunge" for event in events))
        body = body_analysis(tracks, events)
        self.assertIn("unresolved_recovery", body.warnings)
        report = assemble(tracks, events, (body, Partial()), METRIC_CATALOG)
        self.assertIn("unresolved_recovery", report.warnings)
        self.assertEqual(report.windows, ())
        self.assertFalse(any(event.kind == "lunge" for event in report.events))

        played, _truth = _played()
        _clean_tracks, clean_events, clean_body, _clean_report = _run(played)
        self.assertNotIn("unresolved_recovery", clean_body.warnings)
        self.assertTrue(any(event.kind == "lunge" and "end_frame" in event.data for event in clean_events))

    def test_lunge_mark_uses_this_frame_stature(self):
        doc, _ = script(n=80, near=PlayerPlan(lunge=Lunge(40, 0.80, 0.4)))
        tracks, _events, body, _built = _run(doc)
        finding = next(item for item in body.findings if item.id == "technique.lunge_depth" and item.player == Role.NEAR)
        mark = next(item for item in finding.moments[0].marks if item.label == "standing hip line")
        frame = mark.frame
        player = tracks.players[Role.NEAR]
        ankles = [player.xy[frame, index, 1] for index in (15, 16) if player.visible[frame, index]]
        expected = max(ankles) - standing_hip_px(player, frame)
        self.assertLess(abs(mark.points[0][1] - expected), 1e-6)

    def test_a_fully_sharpened_window_can_leave_the_preview_cap(self):
        tracks, events = _sharpened_knees(((0, 159),))
        self.assertEqual(tracks.tier, Tier.PREVIEW)
        self.assertEqual(tracks.refined_windows, ((0, 159),))
        body = body_analysis(tracks, events)
        knee = _finding(body, "technique.knee_bend")
        self.assertGreaterEqual(knee.metrics[0].n_obs, 3)
        self.assertLess(knee.metrics[0].value, 20)
        self.assertGreaterEqual(knee.metrics[0].confidence, 0.6)
        self.assertEqual(knee.severity, Severity.INFO)
        self.assertEqual(knee.fix_tags, ())
        self.assertEqual(knee.tier, Tier.REFINED)
        speed = _finding(body, "contact.hand_speed")
        self.assertEqual(speed.severity, Severity.INFO)
        self.assertLessEqual(speed.metrics[0].confidence, 0.55)
        self.assertIn(NOT_A_HIT, speed.explain)
        self.assertEqual(speed.tier, Tier.REFINED)
        report = assemble(tracks, events, (body, Partial()), METRIC_CATALOG)
        self.assertEqual(report.tier, Tier.PREVIEW)
        self.assertIn("preview_estimate", report.warnings)

    def test_mixed_sharpened_evidence_stays_capped(self):
        tracks, events = _sharpened_knees(((0, 75),))
        body = body_analysis(tracks, events)
        knee = _finding(body, "technique.knee_bend")
        self.assertGreaterEqual(knee.metrics[0].n_obs, 3)
        self.assertEqual(knee.tier, Tier.PREVIEW)
        self.assertLessEqual(knee.metrics[0].confidence, 0.55)
        self.assertEqual(knee.severity, Severity.INFO)

    def test_one_sharpened_frame_does_not_cover_the_bout(self):
        tracks, events = _sharpened_knees(((0, 0),))
        body = body_analysis(tracks, events)
        knee = _finding(body, "technique.knee_bend")
        self.assertEqual(knee.tier, Tier.PREVIEW)
        self.assertLessEqual(knee.metrics[0].confidence, 0.55)
        self.assertEqual(knee.severity, Severity.INFO)

    def test_a_missing_row_inside_a_window_is_not_refined_evidence(self):
        tracks, events = _sharpened_knees(((0, 159),))
        tracks.players[Role.NEAR].present[37] = False
        body = body_analysis(tracks, events)
        speed = _finding(body, "contact.hand_speed")
        self.assertEqual(speed.tier, Tier.PREVIEW)
        self.assertLessEqual(speed.metrics[0].confidence, 0.55)
        self.assertEqual(speed.severity, Severity.INFO)

    def test_no_observation_is_absent_not_a_guide_midpoint(self):
        doc, _ = script(n=40, near=PlayerPlan())
        tracks = load_tracks(doc)
        player = tracks.players[Role.NEAR]
        hand = "right_wrist"
        wrist = wrist_height_ratio(player, hand)
        elbow = elbow_deg(player, hand)
        built = (
            _stay_lower(player, tracks, []),
            _knee_bend(player, tracks, []),
            _lunge_depth(player, tracks, []),
            _stance_width(player, tracks, []),
            _trunk_lean(player, tracks, []),
            _reach(player, tracks, [], wrist, hand),
            _arm(player, tracks, [], elbow, hand),
            _hand_speed(player, tracks, [], hand),
        )
        self.assertEqual({item.id for item in built}, {spec.finding_id for spec in BODY_SPECS})
        for finding in built:
            spec = next(item for item in BODY_SPECS if item.finding_id == finding.id)
            metric = finding.metrics[0]
            midpoint = (spec.target[0] + spec.target[1]) / 2
            self.assertIsNone(metric.value)
            self.assertNotEqual(metric.value, midpoint)
            self.assertEqual(metric.n_obs, 0)
            self.assertEqual(metric.confidence, 0.0)
            self.assertIsNone(metric.best)
            self.assertEqual(metric.target, spec.target)
            self.assertEqual(finding.severity, Severity.INFO)
            self.assertEqual(finding.score, 0.0)
            self.assertEqual(finding.moments, ())
            self.assertIn(PRACTICE, finding.explain)
            self.assertTrue(finding.headline)
            self.assertNotIn(str(midpoint), finding.explain)

    def test_no_exchange_does_not_get_static_ready_stance_fallback(self):
        root = Path(__file__).resolve().parent / "data" / "ready-stance"
        signals = {
            "technique.stay_lower": hip_height_ratio,
            "technique.knee_bend": knee_flexion_deg,
            "technique.stance_width": stance_width_ratio,
            "technique.trunk_lean": lambda player: np.abs(trunk_lean_deg(player)),
        }
        expected = {
            "technique.stay_lower": 0.5509120690586571,
            "technique.knee_bend": 19.09108151709347,
            "technique.stance_width": 0.19999370023036178,
            "technique.trunk_lean": 5.369906196605314,
        }
        seen = {}
        for rev in ("r0", "r1"):
            pose = root / rev / "pose.json"
            before = hashlib.sha256(pose.read_bytes()).hexdigest()
            tracks = load_tracks(json.loads(pose.read_text(encoding="utf-8")))
            events = detect_events(tracks)
            player = tracks.players[Role.NEAR]
            mask = _ready_mask(player, tracks, events)
            readings = {}
            for finding_id, signal in signals.items():
                spec = next(item for item in BODY_SPECS if item.finding_id == finding_id)
                reps = _signal_reps(player, tracks, mask, signal(player))
                readings[finding_id] = (_value(spec, reps), len(reps))
            seen[rev] = {
                "readings": readings,
                "swings": sum(event.kind == "swing" and event.player == Role.NEAR for event in events),
            }
            self.assertEqual(hashlib.sha256(pose.read_bytes()).hexdigest(), before)
        for finding_id, value in expected.items():
            spec = next(item for item in BODY_SPECS if item.finding_id == finding_id)
            midpoint = (spec.target[0] + spec.target[1]) / 2
            got, count = seen["r0"]["readings"][finding_id]
            self.assertEqual(count, 0)
            self.assertIsNone(got)
            self.assertNotEqual(got, midpoint)
            absent, absent_count = seen["r1"]["readings"][finding_id]
            self.assertIsNone(absent)
            self.assertEqual(absent_count, 0)
            self.assertNotEqual(absent, midpoint)
        self.assertEqual(seen["r0"]["swings"], 0)
        self.assertEqual(seen["r1"]["swings"], 0)  # Legacy input has no declared racket hand.

    def test_unavailable_ready_stance_finding_is_a_note_without_a_number(self):
        root = Path(__file__).resolve().parent / "data" / "ready-stance"
        pose = root / "r1" / "pose.json"
        before = hashlib.sha256(pose.read_bytes()).hexdigest()
        tracks = load_tracks(json.loads(pose.read_text(encoding="utf-8")))
        body = body_analysis(tracks, detect_events(tracks))
        ready = (
            "technique.stay_lower", "technique.knee_bend",
            "technique.stance_width", "technique.trunk_lean",
        )
        for finding_id in ready:
            finding = next(item for item in body.findings if item.id == finding_id and item.player == Role.NEAR)
            self.assertIsNone(finding.metrics[0].value)
            self.assertEqual(finding.metrics[0].n_obs, 0)
            self.assertEqual(finding.metrics[0].confidence, 0.0)
            self.assertIsNone(finding.metrics[0].best)
            self.assertEqual(finding.metrics[0].target, next(item.target for item in BODY_SPECS if item.finding_id == finding_id))
            self.assertEqual(finding.severity, Severity.INFO)
            self.assertEqual(finding.score, 0.0)
            self.assertEqual(finding.moments, ())
            self.assertIn(PRACTICE, finding.explain)
        self.assertFalse(any(item.id.startswith("contact.") for item in body.findings))
        self.assertEqual(hashlib.sha256(pose.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
