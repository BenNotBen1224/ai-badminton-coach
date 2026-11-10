from __future__ import annotations

import base64
import json
import math
import os
import time
import unittest

import numpy as np

from coach.bundle import BundleContractError, PreparedBundle, build_bundle, prepare_bundle
from coach.schema import (
    SCHEMA_VERSION,
    COCO_JOINTS,
    Basis,
    Category,
    Finding,
    Metric,
    Moment,
    Report,
    Role,
    Severity,
    Tier,
)
from coach.synth import make_preview, repeat_frames, script
from coach.tracks import load_tracks

LOCK = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "work", "cpu.lock"))


def _metric():
    return Metric(
        "hip_height_ratio",
        "Hip height",
        0.53,
        "ratio",
        Basis.PROJECTED_2D,
        0.8,
        (0.42, 0.48),
    )


def _finding(moments, finding_id="technique.stay_lower", severity=Severity.IMPROVE):
    tags = ("lower_stance",) if severity in (Severity.IMPROVE, Severity.PRIORITY) else ()
    return Finding(
        finding_id,
        Category.TECHNIQUE,
        Role.NEAR,
        severity,
        "Hips stay high while you wait",
        "Your hips sit above the practice guide range while you wait.",
        (_metric(),),
        moments,
        Tier.REFINED,
        fix_tags=tags,
    )


def _report(n, findings, players=(Role.NEAR, Role.FAR), fps=30.0):
    return Report(
        schema_version=SCHEMA_VERSION,
        source={"width": 1280, "height": 720, "frame_count": n, "fps": fps},
        tier=Tier.REFINED,
        n_frames=n,
        fps=fps,
        players=players,
        metrics={},
        series={},
        events=(),
        findings=tuple(findings),
        timestamps=tuple(i / fps for i in range(n)),
    )


def _decode(packed):
    kind = {"int16": "<i2", "uint8": "|u1"}[packed["dtype"]]
    raw = np.frombuffer(base64.b64decode(packed["b64"]), dtype=np.dtype(kind))
    return raw.reshape(packed["shape"]).copy()


def _pixels(packed):
    stored = _decode(packed)
    if "scale" not in packed:
        return stored
    values = stored.astype(np.float64) * packed["scale"]
    if "unknown" in packed:
        values = np.where(stored == packed["unknown"], np.nan, values)
    return values


class BundleTests(unittest.TestCase):
    def test_preview_round_trip_matches_tracks(self):
        doc, _truth = script(40, 30.0)
        preview = make_preview(doc, 4)
        repeated = repeat_frames(preview, 40)
        self.assertEqual(repeated, [39])
        issue = Moment(Role.NEAR, 4, 12, 20, "Hips high", "issue")
        best = Moment(Role.NEAR, 24, 30, 36, "Lower hips", "best")
        finding = _finding((issue, best))
        other = _finding((), "technique.knee_bend", Severity.STRENGTH)
        report = _report(40, [finding, other])
        bundle = build_bundle(
            preview, report, video_url="clip.mp4",
            exercises={("technique.stay_lower", "near"): ["wall_sit", "sit_low"]},
        )
        tracks = load_tracks(preview)
        for role in ("near", "far"):
            decoded = _pixels(bundle["poses"][role]["xy"])
            measured = _decode(bundle["poses"][role]["measured"])
            want = tracks.players[Role(role)].measured
            self.assertTrue(np.array_equal(measured.astype(bool), want))
            for frame in preview["frames"]:
                player = next(row for row in frame["players"] if row["role"] == role)
                idx = frame["frame_index"]
                for j, joint in enumerate(player["joints"]):
                    if joint is None or not np.isfinite(joint[0]) or not np.isfinite(joint[1]):
                        continue
                    self.assertLessEqual(abs(decoded[idx, j, 0] - joint[0]), 0.1)
                    self.assertLessEqual(abs(decoded[idx, j, 1] - joint[1]), 0.1)
        self.assertEqual(bool(measured[0]), True)
        self.assertEqual(bool(_decode(bundle["poses"]["near"]["measured"])[39]), False)
        got = np.asarray(bundle["court"]["to_court"], dtype=np.float64)
        self.assertTrue(np.allclose(got, tracks.court.to_court, atol=1e-5, rtol=0.0))
        self.assertEqual(bundle["court"]["width_m"], 6.10)
        self.assertEqual(bundle["court"]["length_m"], 13.40)

    def test_required_keys_and_exercise_ids(self):
        doc, _truth = script(40, 30.0)
        issue = Moment(Role.NEAR, 4, 12, 20, "Hips high", "issue")
        best = Moment(Role.NEAR, 24, 30, 36, "Lower hips", "best")
        finding = _finding((issue, best))
        other = _finding((), "technique.knee_bend", Severity.STRENGTH)
        report = _report(40, [finding, other])
        bundle = build_bundle(
            doc, report, video_url="clip.mp4",
            exercises={("technique.stay_lower", "near"): ["wall_sit", "sit_low"]},
        )
        for key in (
            "n_frames", "fps", "findings", "poses", "tier", "players", "source",
            "style", "timestamps", "series", "events", "windows", "warnings",
            "video_url", "court", "schema_version", "joints",
        ):
            self.assertIn(key, bundle)
        self.assertEqual(bundle["n_frames"], 40)
        self.assertEqual(bundle["fps"], 30.0)
        self.assertEqual(bundle["video_url"], "clip.mp4")
        self.assertEqual(bundle["schema_version"], SCHEMA_VERSION)
        self.assertEqual(bundle["tier"], "refined")
        self.assertEqual(bundle["players"], ["near", "far"])
        self.assertEqual(bundle["joints"], list(COCO_JOINTS))
        self.assertEqual(bundle["style"]["roles"]["near"], "#FF00AA")
        self.assertEqual(bundle["style"]["confidence_gate"], 0.5)
        self.assertEqual(len(bundle["style"]["bones"]), 16)
        self.assertEqual(bundle["findings"][0]["exercise_ids"], ["wall_sit", "sit_low"])
        self.assertEqual(bundle["findings"][1]["exercise_ids"], [])
        self.assertEqual(bundle["findings"][1]["timelines"], [])
        timelines = bundle["findings"][0]["timelines"]
        self.assertEqual(len(timelines), 2)
        self.assertEqual(timelines[0]["segments"][-1]["show"], "best")
        self.assertEqual(timelines[0]["segments"][-1]["start_frame"], 30)
        self.assertEqual(timelines[0]["segments"][-1]["kind"], "freeze")

    def test_exercise_ids_follow_player(self):
        doc, _truth = script(40, 30.0)
        near_moment = Moment(Role.NEAR, 4, 12, 20, "Hips high", "issue")
        far_moment = Moment(Role.FAR, 4, 12, 20, "Hips high", "issue")
        near = _finding((near_moment,))
        far = Finding(
            "technique.stay_lower",
            Category.TECHNIQUE,
            Role.FAR,
            Severity.IMPROVE,
            "Hips stay high while you wait",
            "Your hips sit above the practice guide range while you wait.",
            (_metric(),),
            (far_moment,),
            Tier.REFINED,
            fix_tags=("lower_stance",),
        )
        report = _report(40, [near, far])
        bundle = build_bundle(
            doc, report, video_url="clip.mp4",
            exercises={
                ("technique.stay_lower", "near"): ["wall_sit"],
                ("technique.stay_lower", "far"): ["goblet_squat"],
            },
        )
        self.assertEqual(bundle["findings"][0]["player"], "near")
        self.assertEqual(bundle["findings"][0]["exercise_ids"], ["wall_sit"])
        self.assertEqual(bundle["findings"][1]["player"], "far")
        self.assertEqual(bundle["findings"][1]["exercise_ids"], ["goblet_squat"])
        by_string = build_bundle(
            doc, report, video_url="clip.mp4",
            exercises={"far:technique.stay_lower": ["sit_low_hold"]},
        )
        self.assertEqual(by_string["findings"][0]["exercise_ids"], [])
        self.assertEqual(by_string["findings"][1]["exercise_ids"], ["sit_low_hold"])
        bare = build_bundle(
            doc, report, video_url="clip.mp4",
            exercises={"technique.stay_lower": ["wall_sit"]},
        )
        self.assertEqual(bare["findings"][0]["exercise_ids"], [])
        self.assertEqual(bare["findings"][1]["exercise_ids"], [])

    def test_tenths_round_trip(self):
        joints = [[0.0, 0.0, 0.9] for _ in range(17)]
        joints[0] = [10.04, 20.06, 0.9]
        doc = {
            "source": {"width": 1280, "height": 720, "frame_count": 1, "fps": 30.0},
            "frames": [{
                "frame_index": 0,
                "players": [{
                    "role": "near",
                    "box_confidence": 0.9,
                    "bbox_xyxy": [1.04, 2.06, 3.04, 4.06],
                    "joints": joints,
                }],
            }],
        }
        moment = Moment(Role.NEAR, 0, 0, 0, "One frame", "issue")
        report = _report(1, [_finding((moment,))], players=(Role.NEAR, Role.FAR))
        bundle = build_bundle(doc, report, video_url="one.mp4")
        xy = _pixels(bundle["poses"]["near"]["xy"])
        self.assertAlmostEqual(xy[0, 0, 0], 10.0, delta=1e-9)
        self.assertAlmostEqual(xy[0, 0, 1], 20.1, delta=1e-9)
        box = _pixels(bundle["poses"]["near"]["bbox"])
        self.assertAlmostEqual(box[0, 0], 1.0, delta=1e-9)
        self.assertAlmostEqual(box[0, 1], 2.1, delta=1e-9)
        self.assertEqual(int(_decode(bundle["poses"]["near"]["measured"])[0]), 1)
        far = _decode(bundle["poses"]["far"]["xy"])
        self.assertTrue(np.all(far == bundle["poses"]["far"]["xy"]["unknown"]))
        self.assertEqual(int(_decode(bundle["poses"]["far"]["measured"])[0]), 0)

    def test_confidence_stays_on_the_correct_side_of_the_gate(self):
        joints = [[1.0, 2.0, 0.9] for _ in range(17)]
        joints[0][2] = 0.499
        joints[1][2] = -0.2
        joints[2][2] = 0.5
        doc = {
            "source": {"width": 1280, "height": 720, "frame_count": 1, "fps": 30.0},
            "frames": [{
                "frame_index": 0,
                "players": [{
                    "role": "near",
                    "box_confidence": 0.9,
                    "bbox_xyxy": [1.0, 2.0, 3.0, 4.0],
                    "joints": joints,
                }],
            }],
        }
        moment = Moment(Role.NEAR, 0, 0, 0, "One frame", "issue")
        report = _report(1, [_finding((moment,))], players=(Role.NEAR,))
        bundle = build_bundle(doc, report, video_url="conf.mp4")
        stored = _decode(bundle["poses"]["near"]["conf"])
        self.assertEqual(int(stored[0, 0]), 49)
        self.assertLess(int(stored[0, 0]) * bundle["poses"]["near"]["conf"]["scale"], 0.5)
        self.assertEqual(int(stored[0, 1]), 0)
        self.assertEqual(int(stored[0, 2]), 50)
        self.assertGreaterEqual(int(stored[0, 2]) * bundle["poses"]["near"]["conf"]["scale"], 0.5)

    def test_xy_outside_int16_tenths_raises(self):
        joints = [[0.0, 0.0, 0.9] for _ in range(17)]
        joints[0][0] = 4000.0
        doc = {
            "source": {"width": 1280, "height": 720, "frame_count": 1, "fps": 30.0},
            "frames": [{
                "frame_index": 0,
                "players": [{
                    "role": "near",
                    "box_confidence": 0.9,
                    "bbox_xyxy": [1.0, 2.0, 3.0, 4.0],
                    "joints": joints,
                }],
            }],
        }
        moment = Moment(Role.NEAR, 0, 0, 0, "One frame", "issue")
        report = _report(1, [_finding((moment,))], players=(Role.NEAR,))
        with self.assertRaises(ValueError):
            build_bundle(doc, report, video_url="wide.mp4")

    def test_frame_count_must_match_the_report(self):
        doc = {
            "source": {"width": 1280, "height": 720, "frame_count": 41, "fps": 30.0},
            "frames": [],
        }
        moment = Moment(Role.NEAR, 0, 0, 0, "One frame", "issue")
        report = _report(40, [_finding((moment,))], players=(Role.NEAR,))
        with self.assertRaises(ValueError):
            build_bundle(doc, report, video_url="count.mp4")

    def test_timeline_index_follows_moments(self):
        doc = {
            "source": {"width": 1280, "height": 720, "frame_count": 40, "fps": 30.0},
            "frames": [],
        }
        issue = Moment(Role.NEAR, 4, 12, 20, "Hips high", "issue")
        best = Moment(Role.NEAR, 24, 30, 36, "Lower hips", "best")
        report = _report(40, [_finding((issue, best))], players=(Role.NEAR,))
        bundle = build_bundle(doc, report, video_url="moments.mp4")
        timelines = bundle["findings"][0]["timelines"]
        self.assertEqual(len(timelines), 2)
        frames = [
            (seg["start_frame"], seg["end_frame"])
            for seg in timelines[1]["segments"]
        ]
        self.assertEqual(frames[0][0], 24)
        self.assertTrue(all(24 <= start and end <= 36 for start, end in frames))
        freezes = [seg for seg in timelines[1]["segments"] if seg["kind"] == "freeze"]
        self.assertEqual(len(freezes), 1)
        self.assertEqual(freezes[0]["start_frame"], 30)
        self.assertEqual(freezes[0]["end_frame"], 30)
        self.assertEqual(freezes[0]["show"], "best")
        self.assertEqual([seg["show"] for seg in timelines[1]["segments"]], ["best"] * len(timelines[1]["segments"]))

    def test_strength_best_moment_is_seekable(self):
        doc = {
            "source": {"width": 1280, "height": 720, "frame_count": 40, "fps": 30.0},
            "frames": [],
        }
        best = Moment(Role.NEAR, 24, 30, 36, "Lower hips", "best")
        finding = _finding((best,), "technique.knee_bend", Severity.STRENGTH)
        report = _report(40, [finding], players=(Role.NEAR,))
        bundle = build_bundle(doc, report, video_url="strength.mp4")
        timelines = bundle["findings"][0]["timelines"]
        self.assertEqual(len(timelines), 1)
        freezes = [seg for seg in timelines[0]["segments"] if seg["kind"] == "freeze"]
        self.assertEqual(len(freezes), 1)
        self.assertEqual(freezes[0]["start_frame"], 30)
        self.assertEqual(freezes[0]["end_frame"], 30)
        self.assertEqual(freezes[0]["show"], "best")
        self.assertEqual(freezes[0]["speed"], 0.0)

    def test_budget(self):
        n = 1080
        frames = []
        for i in range(n):
            near = [[100.0 + j + i * 0.01, 200.0 + j, 0.9] for j in range(17)]
            far = [[400.0 + j + i * 0.01, 180.0 + j, 0.8] for j in range(17)]
            frames.append({
                "frame_index": i,
                "players": [
                    {"role": "near", "box_confidence": 0.9, "bbox_xyxy": [10.0, 20.0, 80.0, 200.0], "joints": near},
                    {"role": "far", "box_confidence": 0.8, "bbox_xyxy": [30.0, 40.0, 70.0, 120.0], "joints": far},
                ],
            })
        doc = {
            "source": {"width": 1280, "height": 720, "frame_count": n, "fps": 30.0},
            "frames": frames,
        }
        moment = Moment(Role.NEAR, 10, 40, 70, "Hips high", "issue")
        report = _report(n, [_finding((moment,))])
        owned = False
        handle = None
        try:
            deadline = time.monotonic() + 30 * 60
            while True:
                try:
                    handle = open(LOCK, "x", encoding="utf-8")
                    break
                except FileExistsError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(LOCK)
                    time.sleep(30)
            handle.write(f"C2 {os.getpid()} bundle budget")
            handle.close()
            handle = None
            owned = True
            build_bundle(doc, report, video_url="mem.mp4")
            times = []
            bundle = None
            for _ in range(5):
                started = time.perf_counter()
                bundle = build_bundle(doc, report, video_url="mem.mp4")
                times.append((time.perf_counter() - started) * 1000.0)
            size = len(json.dumps(bundle).encode("utf-8"))
            print("budget_ms", " ".join(f"{item:.3f}" for item in times), "bytes", size)
            self.assertEqual(len(times), 5)
            self.assertLess(min(times), 40.0)
            self.assertLess(max(times), 80.0)
            self.assertLess(size, 3_000_000)
        finally:
            if handle is not None:
                handle.close()
            if owned:
                os.remove(LOCK)


def _saved_metric(**overrides):
    row = {
        "name": "hip_height_ratio",
        "label": "Hip height",
        "value": 1.2,
        "unit": "ratio",
        "basis": "projected_2d",
        "confidence": 0,
        "target": [0.42, 0.48],
        "series": "hip_height_ratio",
        "n_obs": 0,
        "best": None,
        "note": "keep",
    }
    row.update(overrides)
    return row


def _one_finding(version, metric, **finding):
    body = {
        "id": "technique.stay_lower",
        "severity": "info",
        "score": 0,
        "moments": [],
        "metrics": [metric],
    }
    body.update(finding)
    return {"schema_version": version, "findings": [body]}


class PrepareBundleTests(unittest.TestCase):
    def test_version2_placeholder_clears_value_and_leaves_the_input(self):
        finding_metric = _saved_metric()
        top_metric = _saved_metric()
        bundle = _one_finding(2, finding_metric)
        bundle["metrics"] = {"near": [top_metric]}
        before = json.dumps(bundle)
        prepared = prepare_bundle(bundle)
        got = prepared.bundle["findings"][0]["metrics"][0]
        self.assertIsInstance(prepared, PreparedBundle)
        self.assertIsNone(got["value"])
        self.assertIsNone(prepared.bundle["metrics"]["near"][0]["value"])
        self.assertEqual(prepared.bundle["schema_version"], 2)
        self.assertFalse(prepared.send_saved_bytes)
        self.assertEqual(got["name"], "hip_height_ratio")
        self.assertEqual(got["label"], "Hip height")
        self.assertEqual(got["unit"], "ratio")
        self.assertEqual(got["basis"], "projected_2d")
        self.assertEqual(got["confidence"], 0)
        self.assertEqual(got["target"], [0.42, 0.48])
        self.assertEqual(got["series"], "hip_height_ratio")
        self.assertEqual(got["n_obs"], 0)
        self.assertIsNone(got["best"])
        self.assertEqual(got["note"], "keep")
        self.assertEqual(json.dumps(bundle), before)
        self.assertEqual(finding_metric["value"], 1.2)

    def test_version2_already_null_sends_saved_bytes(self):
        row = _saved_metric(value=None)
        listed = _saved_metric(value=None)
        bundle = _one_finding(2, row)
        bundle["metrics"] = [listed]
        prepared = prepare_bundle(bundle)
        self.assertIsNone(prepared.bundle["findings"][0]["metrics"][0]["value"])
        self.assertIsNone(prepared.bundle["metrics"][0]["value"])
        self.assertTrue(prepared.send_saved_bytes)
        self.assertEqual(prepared.bundle["schema_version"], 2)

    def test_version2_contradictory_confidence_raises(self):
        bundle = _one_finding(2, _saved_metric(confidence=0.2))
        with self.assertRaises(BundleContractError):
            prepare_bundle(bundle)
        self.assertEqual(bundle["findings"][0]["metrics"][0]["value"], 1.2)

    def test_version2_non_finite_observed_value_raises(self):
        bundle = _one_finding(2, _saved_metric(value=float("nan"), n_obs=4, confidence=0.8, best=0.4))
        with self.assertRaises(BundleContractError):
            prepare_bundle(bundle)
        self.assertTrue(math.isnan(bundle["findings"][0]["metrics"][0]["value"]))

    def test_version2_observed_confidence_zero_keeps_the_value(self):
        prepared = prepare_bundle(_one_finding(2, _saved_metric(n_obs=4, confidence=0, best=0.4)))
        got = prepared.bundle["findings"][0]["metrics"][0]
        self.assertEqual(got["value"], 1.2)
        self.assertEqual(got["confidence"], 0)
        self.assertEqual(got["n_obs"], 4)
        self.assertTrue(prepared.send_saved_bytes)
        self.assertEqual(prepared.bundle["schema_version"], 2)

    def test_version3_null_observation_is_accepted(self):
        prepared = prepare_bundle(_one_finding(3, _saved_metric(value=None)))
        self.assertIsNone(prepared.bundle["findings"][0]["metrics"][0]["value"])
        self.assertTrue(prepared.send_saved_bytes)
        self.assertEqual(prepared.bundle["schema_version"], 3)

    def test_version3_numeric_zero_observation_raises(self):
        bundle = _one_finding(3, _saved_metric())
        with self.assertRaises(BundleContractError):
            prepare_bundle(bundle)
        self.assertEqual(bundle["findings"][0]["metrics"][0]["value"], 1.2)

    def test_missing_schema_version_raises(self):
        bundle = _one_finding(2, _saved_metric(value=0.5, n_obs=4, confidence=0.8, best=0.4))
        del bundle["schema_version"]
        with self.assertRaises(BundleContractError):
            prepare_bundle(bundle)
        bundle["schema_version"] = 1
        with self.assertRaises(BundleContractError):
            prepare_bundle(bundle)

    def test_missing_value_key_is_rejected(self):
        for version in (2, 3):
            row = _saved_metric(n_obs=0, confidence=0, best=None)
            del row["value"]
            bundle = _one_finding(version, row)
            before = json.dumps(bundle)
            with self.assertRaises(BundleContractError) as caught:
                prepare_bundle(bundle)
            self.assertIn("missing value", str(caught.exception))
            self.assertEqual(json.dumps(bundle), before)

    def test_bad_moments_on_an_all_missing_finding_are_rejected(self):
        row = _saved_metric(value=None, n_obs=0, confidence=0, best=None)
        for version in (2, 3):
            for moments in (0, False, "", "no", {"start_frame": 1}):
                bundle = _one_finding(version, row, moments=moments)
                before = json.dumps(bundle)
                with self.assertRaises(BundleContractError) as caught:
                    prepare_bundle(bundle)
                self.assertIn("moments must be a list", str(caught.exception))
                self.assertEqual(json.dumps(bundle), before)
            for moments in (None, "OMIT"):
                finding = {
                    "id": "technique.stay_lower",
                    "severity": "info",
                    "score": 0,
                    "metrics": [dict(row)],
                }
                if moments != "OMIT":
                    finding["moments"] = moments
                bundle = {"schema_version": version, "findings": [finding]}
                before = json.dumps(bundle)
                with self.assertRaises(BundleContractError) as caught:
                    prepare_bundle(bundle)
                self.assertIn("moments must be a list", str(caught.exception))
                self.assertEqual(json.dumps(bundle), before)
            empty = _one_finding(version, dict(row), moments=[])
            loaded = prepare_bundle(empty).bundle["findings"][0]
            self.assertIsNone(loaded["metrics"][0]["value"])
            self.assertEqual(loaded["moments"], [])

    def test_null_with_observations_is_rejected(self):
        for version in (2, 3):
            bundle = _one_finding(version, _saved_metric(value=None, n_obs=3, confidence=0.8, best=None))
            before = json.dumps(bundle)
            with self.assertRaises(BundleContractError):
                prepare_bundle(bundle)
            self.assertEqual(json.dumps(bundle), before)

    def test_null_with_confidence_or_best_is_rejected(self):
        for version in (2, 3):
            for row in (
                _saved_metric(value=None, n_obs=0, confidence=0.4, best=None),
                _saved_metric(value=None, n_obs=0, confidence=0, best=0.2),
            ):
                bundle = _one_finding(version, row)
                before = json.dumps(bundle)
                with self.assertRaises(BundleContractError):
                    prepare_bundle(bundle)
                self.assertEqual(json.dumps(bundle), before)

    def test_float_observation_count_is_rejected(self):
        bundle = _one_finding(2, _saved_metric(value=0.45, n_obs=0.0, confidence=0, best=None))
        before = json.dumps(bundle)
        with self.assertRaises(BundleContractError):
            prepare_bundle(bundle)
        self.assertEqual(json.dumps(bundle), before)

    def test_all_missing_finding_that_overclaims_is_rejected(self):
        moment = [{"player": "near", "start_frame": 1, "key_frame": 2, "end_frame": 3, "label": "Hip"}]
        cases = (
            {"severity": "strength"},
            {"moments": moment},
            {"score": 0.0005},
        )
        for version in (2, 3):
            for patch in cases:
                row = _saved_metric(value=1.2 if version == 2 else None)
                bundle = _one_finding(version, row, **patch)
                before = json.dumps(bundle)
                with self.assertRaises(BundleContractError) as caught:
                    prepare_bundle(bundle)
                self.assertEqual(json.dumps(bundle), before)
                text = str(caught.exception)
                self.assertTrue(
                    "must be info" in text or "must not have a moment" in text or "score must be 0" in text,
                    text,
                )

    def test_measured_confidence_zero_and_mixed_findings_stay(self):
        measured = _one_finding(
            2,
            _saved_metric(value=0.2, n_obs=4, confidence=0, best=0.1),
            severity="improve",
            score=0.4,
            moments=[{"player": "near", "start_frame": 1, "key_frame": 2, "end_frame": 3, "label": "Hip"}],
        )
        loaded = prepare_bundle(measured).bundle["findings"][0]
        self.assertEqual(loaded["metrics"][0]["value"], 0.2)
        self.assertEqual(loaded["metrics"][0]["confidence"], 0)
        self.assertEqual(loaded["severity"], "improve")
        self.assertTrue(loaded["moments"])

        absent = _saved_metric(name="contact_reach", label="Reach", value=1.2, n_obs=0, confidence=0, best=None)
        seen = _saved_metric(value=0.5, n_obs=4, confidence=0, best=0.4)
        mixed = {
            "schema_version": 2,
            "findings": [{
                "id": "technique.stay_lower",
                "severity": "strength",
                "score": 0,
                "moments": [],
                "metrics": [absent, seen],
            }],
        }
        before = json.dumps(mixed)
        prepared = prepare_bundle(mixed)
        self.assertEqual(json.dumps(mixed), before)
        got = prepared.bundle["findings"][0]
        self.assertIsNone(got["metrics"][0]["value"])
        self.assertEqual(got["metrics"][1]["value"], 0.5)
        self.assertEqual(got["metrics"][1]["confidence"], 0)
        self.assertEqual(got["severity"], "strength")
        self.assertFalse(prepared.send_saved_bytes)

        current = json.loads(json.dumps(prepared.bundle))
        current["schema_version"] = 3
        current["findings"][0]["metrics"][0]["value"] = None
        again = prepare_bundle(current).bundle["findings"][0]
        self.assertIsNone(again["metrics"][0]["value"])
        self.assertEqual(again["severity"], "strength")

    def test_second_prepare_does_not_change_the_cleared_value(self):
        prepared = prepare_bundle(_one_finding(2, _saved_metric()))
        self.assertIsNone(prepared.bundle["findings"][0]["metrics"][0]["value"])
        self.assertFalse(prepared.send_saved_bytes)
        again = prepare_bundle(prepared.bundle)
        self.assertIsNone(again.bundle["findings"][0]["metrics"][0]["value"])
        self.assertIsNone(prepared.bundle["findings"][0]["metrics"][0]["value"])
        self.assertTrue(again.send_saved_bytes)
        self.assertEqual(again.bundle["schema_version"], 2)


if __name__ == "__main__":
    unittest.main()
