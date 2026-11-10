"""Harness gates. These tests do not read turbo/sealed."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

from turbo.accept import (
    LIMIT_NS,
    Refusal,
    acquire_cpu_lock,
    call_timed,
    cpu_lock_held,
    execution_binding,
    headline,
    new_lock_token,
    parse_args,
    quality_gate,
    reference_path,
    refuse_final,
    release_cpu_lock,
    schedule,
    time_gate,
    report_contract_problems,
    validate_artifacts,
    validate_pose,
)
from turbo.fresh import evidence, fresh_copy
from turbo.judge_phase import kept
from turbo.replay import ReplayRunner
from turbo.score import exclude_indexes, score_pair

FFMPEG = Path(__file__).resolve().parents[2] / "tools" / "ffmpeg.exe"
PROJECT = Path(__file__).resolve().parents[2]


def pose(n, tier="preview"):
    frames = []
    for i in range(n):
        frames.append({
            "frame_index": i,
            "players": [
                {"role": "near", "joints": [[1.0, 2.0, 0.9]] * 17},
                {"role": "far", "joints": [[8.0, 9.0, 0.8]] * 17},
            ],
        })
    return {
        "schema_version": 3,
        "tier": tier,
        "source": {"width": 1920, "height": 1080, "fps": 30.0, "frame_count": n},
        "frames": frames,
    }


def person(role, x, y=100, h=200):
    joints = [[x, y] for _ in range(17)]
    return {
        "role": role,
        "bbox_xyxy": [x, y, x + 40, y + h],
        "joints": joints,
        "sapiens": joints,
        "posewarper": joints,
    }


def prediction(frames):
    return {
        "schema_version": 3,
        "tier": "preview",
        "source": {"width": 1920, "height": 1080, "fps": 30.0, "frame_count": len(frames)},
        "frames": [{"frame_index": i, "players": players} for i, players in enumerate(frames)],
    }


def reference(frames):
    return {"frames": [{"frame_index": i, "people": people} for i, people in enumerate(frames)]}


def player(role, x, y=100):
    return {"role": role, "joints": [[x, y, 1.0] for _ in range(17)]}


class HarnessTests(unittest.TestCase):
    def test_kept_phase_offset(self):
        for i in range(12):
            self.assertEqual(kept(i, 3, 0), i % 3 == 0)
        self.assertTrue(kept(1, 3, 1))
        self.assertTrue(kept(4, 3, 1))
        self.assertFalse(kept(5, 3, 1))
        self.assertTrue(kept(2, 3, 2))

    def test_schedule_rotates_the_start_clip(self):
        self.assertEqual(schedule(["01", "02", "03"], 2), ["01", "02", "03", "02", "03", "01"])

    def test_dev_reference_path_is_not_sealed(self):
        path = reference_path("dev", "01")
        self.assertNotIn("sealed", path.parts)
        self.assertEqual(path.name, "validation-01.json")
        self.assertIn("sealed", reference_path("final", "01").parts)

    def test_scoring_modules_do_not_name_sealed(self):
        for name in ("score.py", "fresh.py", "replay.py", "runner.py"):
            text = (PROJECT / "turbo" / name).read_text(encoding="utf-8")
            self.assertNotIn("sealed", text)

    def test_time_gate_is_strict_and_ignores_a_fast_looking_runner_timing(self):
        fast = {"clip": "01", "gating": True, "claim": "S-A2", "wall_ns": LIMIT_NS - 1}
        slow = {"clip": "01", "gating": True, "claim": "S-A2", "wall_ns": LIMIT_NS}
        self.assertFalse(time_gate([slow], required_runs=1)["passed"])
        self.assertTrue(time_gate([fast], required_runs=1)["passed"])

        class Slow:
            def run(self, video, work):
                time.sleep(1.02)
                return {"doc": {}, "timings": {"total_s": 0.001}}

        timed = call_timed(Slow(), "video.mp4", tempfile.mkdtemp())
        self.assertEqual(timed["runner_timings"]["total_s"], 0.001)
        self.assertGreaterEqual(timed["wall_ns"], LIMIT_NS)
        gate = time_gate([{"clip": "01", "gating": True, "claim": "S-A2", "wall_ns": timed["wall_ns"]}], required_runs=1)
        self.assertFalse(gate["passed"])
        self.assertIn("not under", gate["failures"][0])

    def test_replay_clocks_cannot_fill_the_s_a2_counter(self):
        clips = ("01", "02", "03", "04", "05", "08", "09", "10")
        replay = [
            {"clip": clip, "gating": True, "claim": "replay", "is_replay": True, "wall_ns": LIMIT_NS - 1}
            for clip in clips for _ in range(5)
        ]
        gate = time_gate(replay)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["replay_clocks_eligible"])
        self.assertEqual(gate["ineligible_runs"], 40)
        self.assertTrue(any("no S-A2 eligible" in item for item in gate["failures"]))
        self.assertFalse(any("need 5" in item for item in gate["failures"]))
        mixed = replay + [
            {"clip": "01", "gating": True, "claim": "S-A2", "wall_ns": LIMIT_NS - 1}
            for _ in range(4)
        ]
        mixed_gate = time_gate(mixed)
        self.assertFalse(mixed_gate["passed"])
        self.assertTrue(any(item.startswith("01: 4 timed runs") for item in mixed_gate["failures"]))

    def test_dropped_frames_fail_validation(self):
        problems = validate_pose(pose(2), 3, 30.0, (1920, 1080), 2)
        self.assertTrue(any("2 frames saved, expected 3" in item for item in problems))
        self.assertEqual(validate_pose(pose(2), 2, 30.0, (1920, 1080), 2), [])

    def test_bad_src_and_short_joint_are_rejected(self):
        doc = pose(1)
        doc["frames"][0]["players"][0]["src"] = "copied"
        self.assertTrue(any("not interpolated" in item for item in validate_pose(doc, 1, 30.0, (1920, 1080), 1)))
        doc = pose(1)
        doc["frames"][0]["players"][0]["joints"][0] = [1.0, 2.0]
        self.assertTrue(any("malformed joint" in item for item in validate_pose(doc, 1, 30.0, (1920, 1080), 1)))

    def _e2e(self, work, n, *, bundle=None, doc=None, write=True):
        doc = pose(n) if doc is None else doc
        report = {
            "schema_version": 2,
            "source": {"file": "x.mp4"},
            "tier": "preview",
            "n_frames": n,
            "fps": 30.0,
            "players": ["near"],
            "metrics": {"near": []},
            "series": {},
            "events": [],
            "findings": [],
        }
        if bundle is None:
            from coach.bundle import build_bundle
            from coach.schema import report_from_dict
            bundle = build_bundle(doc, report_from_dict(report), video_url="clip.mp4")
        written = dict(report)
        written["schema_version"] = 3
        pose_path = work / "pose.json"
        report_path = work / "report.json"
        bundle_path = work / "bundle.json"
        if write:
            pose_path.write_text(json.dumps(doc), encoding="utf-8")
            report_path.write_text(json.dumps(written), encoding="utf-8")
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
        result = {
            "doc": doc,
            "pose_path": str(pose_path),
            "report_path": str(report_path),
            "bundle_path": str(bundle_path),
        }
        created = [str(pose_path), str(report_path), str(bundle_path)] if write else []
        return result, created

    def test_report_with_the_wrong_frame_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            result, created = self._e2e(work, 2)
            problems = validate_artifacts(result, 3, work, "S-A2", created=created, fps=30.0)
            self.assertTrue(any("n_frames 2 != 3" in item for item in problems))
            self.assertTrue(any("pose frames 2 != 3" in item for item in problems))
            outside = Path(tempfile.mkdtemp()) / "report.json"
            outside.write_text((work / "report.json").read_text(encoding="utf-8"), encoding="utf-8")
            leaked = {"doc": pose(2), "report_path": str(outside), "bundle_path": str(work / "bundle.json")}
            self.assertTrue(any("outside" in item for item in validate_artifacts(leaked, 2, work, "S-A")))

    def test_s_a2_requires_files_written_inside_the_call_and_an_agreeing_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            result, created = self._e2e(work, 2)
            self.assertEqual(validate_artifacts(result, 2, work, "S-A2", created=created, fps=30.0), [])
            no_pose = dict(result)
            no_pose.pop("pose_path")
            no_pose_created = [item for item in created if not str(item).endswith("pose.json")]
            missing = validate_artifacts(no_pose, 2, work, "S-A2", created=no_pose_created, fps=30.0)
            self.assertTrue(any("pose_path" in item for item in missing))
            other = work / "other.json"
            other.write_text((work / "pose.json").read_text(encoding="utf-8"), encoding="utf-8")
            renamed = dict(result)
            renamed["pose_path"] = str(other)
            wrong_name = validate_artifacts(renamed, 2, work, "S-A2", created=created + [str(other)], fps=30.0)
            self.assertTrue(any("work_dir/pose.json" in item for item in wrong_name))
            thin = {"n_frames": 2, "findings": []}
            thin_result, thin_created = self._e2e(work, 2, bundle=thin)
            thin_problems = validate_artifacts(thin_result, 2, work, "S-A2", created=thin_created, fps=30.0)
            self.assertTrue(any("bundle missing" in item for item in thin_problems))
            self.assertTrue(any("poses" in item for item in thin_problems))
            stale = validate_artifacts(result, 2, work, "S-A2", created=[], fps=30.0)
            self.assertTrue(any("not written inside the call" in item for item in stale))
            changed = dict(result)
            changed["doc"] = pose(2, tier="refined")
            mismatch = validate_artifacts(changed, 2, work, "S-A2", created=created, fps=30.0)
            self.assertTrue(any("does not match" in item for item in mismatch))

    def test_matching_ids_with_stale_text_or_geometry_fail(self):
        import base64
        import array
        from coach.schema import (
            Basis, Category, Finding, Metric, Moment, Report, Role, Severity, Tier,
        )
        from coach.bundle import build_bundle

        moment = Moment(Role.NEAR, 0, 0, 1, "Hips high")
        finding = Finding(
            "technique.stay_lower", Category.TECHNIQUE, Role.NEAR, Severity.IMPROVE,
            "Hips stay high while you wait", "Your hips sit above the practice guide range.",
            (Metric("hip_height_ratio", "Hip height", 0.53, "ratio", Basis.PROJECTED_2D, 0.8),),
            (moment,), Tier.PREVIEW, fix_tags=("lower_stance",),
        )
        doc = pose(2)
        report = Report(
            3, {"file": "x.mp4"}, Tier.PREVIEW, 2, 30.0, (Role.NEAR,),
            {"near": ()}, {}, (), (finding,), timestamps=(0.0, 1.0 / 30.0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            bundle = build_bundle(doc, report, video_url="clip.mp4", exercises={"technique.stay_lower": ["wall_sit"]})
            result, created = self._e2e(work, 2, bundle=bundle, doc=doc)
            report_path = work / "report.json"
            saved = report.to_dict()
            saved["schema_version"] = 3
            report_path.write_text(json.dumps(saved), encoding="utf-8")
            self.assertEqual(validate_artifacts(result, 2, work, "S-A2", created=created, fps=30.0), [])
            stale = json.loads((work / "bundle.json").read_text(encoding="utf-8"))
            self.assertEqual(stale["findings"][0]["id"], report.to_dict()["findings"][0]["id"])
            stale["findings"][0]["headline"] = "This text was not in the report"
            (work / "bundle.json").write_text(json.dumps(stale), encoding="utf-8")
            text_problems = validate_artifacts(result, 2, work, "S-A2", created=created, fps=30.0)
            self.assertTrue(any("text does not match" in item for item in text_problems))
            packed = bundle["poses"]["near"]["xy"]
            values = array.array("h")
            values.frombytes(base64.b64decode(packed["b64"]))
            values[0] = 0 if values[0] != 0 else 1
            packed["b64"] = base64.b64encode(values.tobytes()).decode("ascii")
            (work / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
            geometry = validate_artifacts(result, 2, work, "S-A2", created=created, fps=30.0)
            self.assertTrue(any("does not match the document" in item for item in geometry))

    def test_report_availability_and_separate_versions(self):
        def metric(value, n_obs, confidence, best):
            return {
                "name": "hip_height_ratio",
                "label": "Hip height",
                "value": value,
                "unit": "ratio",
                "basis": "projected_2d",
                "confidence": confidence,
                "n_obs": n_obs,
                "best": best,
                "target": [0.4, 0.6],
            }

        def finding(rows, severity="info", score=0, moments=None, player="near"):
            return {
                "id": "technique.stay_lower",
                "player": player,
                "severity": severity,
                "score": score,
                "moments": [] if moments is None else moments,
                "metrics": rows,
                "headline": "No measured hip height",
                "explain": "The practice guide stays available.",
            }

        def report(rows, findings, version=3):
            return {"schema_version": version, "metrics": rows, "findings": findings}

        absent = metric(None, 0, 0, None)
        seen = metric(0.53, 2, 0, None)
        self.assertEqual(report_contract_problems(report({"near": [absent]}, [finding([absent])])), [])
        self.assertEqual(
            report_contract_problems(report(
                {"near": [absent], "far": [seen]},
                [finding([absent], player="near"), finding([seen], "improve", 0.4, [{"kind": "issue"}], "far")],
            )),
            [],
        )
        self.assertTrue(any("schema_version 2" in item for item in report_contract_problems(report({}, [], 2))))
        self.assertTrue(any("n_obs 0" in item for item in report_contract_problems(report({"near": [metric(None, 1, 0, None)]}, []))))
        self.assertTrue(any("confidence 0" in item for item in report_contract_problems(report({"near": [metric(None, 0, 0.2, None)]}, []))))
        self.assertTrue(any("best null" in item for item in report_contract_problems(report({"near": [metric(None, 0, 0, 0.5)]}, []))))
        self.assertTrue(any("n_obs > 0" in item for item in report_contract_problems(report({"near": [metric(0.5, 0, 0, None)]}, []))))
        self.assertTrue(any("finite number" in item for item in report_contract_problems(report({"near": [metric(float("nan"), 1, 1, None)]}, []))))
        self.assertTrue(any("finite number" in item for item in report_contract_problems(report({"near": [metric(float("inf"), 1, 1, None)]}, []))))
        self.assertTrue(any("finite number" in item for item in report_contract_problems(report({"near": [metric(True, 1, 1, None)]}, []))))
        missing = metric(0.5, 1, 1, None)
        missing.pop("n_obs")
        self.assertTrue(any("needs value" in item for item in report_contract_problems(report({"near": [missing]}, []))))
        self.assertTrue(any("must stay info" in item for item in report_contract_problems(report({"near": []}, [finding([absent], "improve")]))))
        self.assertTrue(any("score must be 0" in item for item in report_contract_problems(report({"near": []}, [finding([absent], score=1)]))))
        self.assertTrue(any("no moments" in item for item in report_contract_problems(report({"near": []}, [finding([absent], moments=[{"kind": "issue"}])]))))
        self.assertEqual(report_contract_problems(report({"far": [seen]}, [finding([seen], "improve", 1, [{"kind": "issue"}], "far")])), [])

        doc = pose(1)
        doc["schema_version"] = 2
        self.assertTrue(any("pose schema_version 2 != 3" in item for item in validate_pose(doc, 1, 30.0, (1920, 1080), 1)))

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            result, created = self._e2e(work, 2)
            saved = json.loads((work / "report.json").read_text(encoding="utf-8"))
            packed = json.loads((work / "bundle.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["schema_version"], 3)
            packed["schema_version"] = 1
            (work / "bundle.json").write_text(json.dumps(packed), encoding="utf-8")
            problems = validate_artifacts(result, 2, work, "S-A2", created=created, fps=30.0)
            self.assertNotEqual(packed["schema_version"], saved["schema_version"])
            self.assertFalse(any("schema_version does not match" in item for item in problems))
            self.assertEqual(problems, [])

    def test_final_refuses_replay_lock_and_busy_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            absent = Path(tmp) / "missing.lock"
            refuse_final(type("Fast", (), {"IS_REPLAY": False}), type("Inst", (), {"IS_REPLAY": False})(), absent, 4.9)
            with self.assertRaises(Refusal):
                refuse_final(ReplayRunner, type("Inst", (), {"IS_REPLAY": False})(), absent, 0.1)
            present = Path(tmp) / "cpu.lock"
            present.write_text("held", encoding="utf-8")
            with self.assertRaises(Refusal):
                refuse_final(type("Fast", (), {"IS_REPLAY": False}), type("Inst", (), {"IS_REPLAY": False})(), present, 0.1)
            with self.assertRaises(Refusal):
                refuse_final(type("Fast", (), {"IS_REPLAY": False}), type("Inst", (), {"IS_REPLAY": False})(), absent, 5.1)
            token = "H 1 owned accept-final"
            present.write_text(token, encoding="utf-8")
            refuse_final(type("Fast", (), {"IS_REPLAY": False}), type("Inst", (), {"IS_REPLAY": False})(), present, 4.9, owned_token=token)
            with self.assertRaises(Refusal):
                refuse_final(ReplayRunner, type("Inst", (), {"IS_REPLAY": False})(), present, 0.1, owned_token=token)
            with self.assertRaises(Refusal):
                refuse_final(type("Fast", (), {"IS_REPLAY": False}), type("Inst", (), {"IS_REPLAY": False})(), present, 5.1, owned_token=token)
            with self.assertRaises(Refusal):
                refuse_final(type("Fast", (), {"IS_REPLAY": False}), type("Inst", (), {"IS_REPLAY": False})(), present, 0.1, owned_token="H 1 other accept-final")
            self.assertEqual(present.read_text(encoding="utf-8"), token)

    def test_cpu_lock_owner_checked_hold(self):
        import turbo.accept as accept_mod

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cpu.lock"
            other = "S 77 census"
            calls = []
            real_acquire = accept_mod.acquire_cpu_lock

            def spy(lock_path, token):
                calls.append(token)
                return real_acquire(lock_path, token)

            accept_mod.acquire_cpu_lock = spy
            try:
                with self.assertRaises(Refusal):
                    with cpu_lock_held(path, 5.1):
                        self.fail("busy CPU must not enter the locked body")
            finally:
                accept_mod.acquire_cpu_lock = real_acquire
            self.assertEqual(calls, [])
            self.assertFalse(path.exists())

            def steal():
                path.write_text(other, encoding="utf-8")

            with self.assertRaises(Refusal):
                with cpu_lock_held(path, 1.0, before_acquire=steal):
                    self.fail("a lock taken after preflight must not be entered")
            self.assertEqual(path.read_text(encoding="utf-8"), other)
            with self.assertRaises(Refusal):
                acquire_cpu_lock(path, new_lock_token())
            self.assertEqual(path.read_text(encoding="utf-8"), other)
            path.unlink()

            class Runner:
                IS_REPLAY = False

                def __init__(self, models_dir):
                    self.models_dir = models_dir
                    self.lock_at_init = path.read_text(encoding="utf-8")

                def warm_up(self, video):
                    self.lock_at_warm = path.read_text(encoding="utf-8")

                def score(self):
                    return path.read_text(encoding="utf-8")

            with cpu_lock_held(path, 5.0) as token:
                runner = Runner("models")
                refuse_final(type(runner), runner, path, 5.0, owned_token=token)
                runner.warm_up("clip")
                self.assertEqual(runner.score(), token)
                self.assertEqual(runner.lock_at_init, token)
                self.assertEqual(runner.lock_at_warm, token)
                self.assertIn("accept-final", token)
            self.assertFalse(path.exists())

            with self.assertRaises(RuntimeError):
                with cpu_lock_held(path, 0.0) as token:
                    self.assertEqual(path.read_text(encoding="utf-8"), token)
                    raise RuntimeError("scoring failed")
            self.assertFalse(path.exists())

            stale = f"H {os.getpid()} stale accept-final"
            path.write_text(stale, encoding="utf-8")
            release_cpu_lock(path, new_lock_token())
            self.assertEqual(path.read_text(encoding="utf-8"), stale)
            path.unlink()
            with cpu_lock_held(path, 0.2) as token:
                path.unlink()
                path.write_text("C6 8 app", encoding="utf-8")
            self.assertEqual(path.read_text(encoding="utf-8"), "C6 8 app")

    def test_lock_cleanup_keeps_unknown_owner_and_io_errors(self):
        import builtins
        from unittest.mock import patch

        real_open = builtins.open
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cpu.lock"
            token = "H 1 owned accept-final"

            def open_with(mode_hook):
                def wrapped(file, mode="r", *args, **kwargs):
                    handle = real_open(file, mode, *args, **kwargs)
                    if mode == "x":
                        mode_hook(handle)
                    return handle
                return wrapped

            def fail_partial(handle):
                real_write = handle.write

                def bad_write(data):
                    real_write(data[:1])
                    raise OSError("write failed")

                handle.write = bad_write

            with patch("builtins.open", open_with(fail_partial)):
                with self.assertRaises(OSError) as caught:
                    acquire_cpu_lock(path, token)
            self.assertEqual(str(caught.exception), "write failed")
            self.assertEqual(path.read_text(encoding="utf-8"), token[:1])
            path.unlink()

            def fail_foreign(handle):
                real_write = handle.write

                def bad_write(data):
                    real_write("S 9 other")
                    raise OSError("write failed")

                handle.write = bad_write

            with patch("builtins.open", open_with(fail_foreign)):
                with self.assertRaises(OSError) as caught:
                    acquire_cpu_lock(path, token)
            self.assertEqual(str(caught.exception), "write failed")
            self.assertEqual(path.read_text(encoding="utf-8"), "S 9 other")
            path.unlink()

            def fail_flush(handle):
                def bad_flush():
                    raise OSError("flush failed")

                handle.flush = bad_flush

            with patch("builtins.open", open_with(fail_flush)):
                with self.assertRaises(OSError) as caught:
                    acquire_cpu_lock(path, token)
            self.assertEqual(str(caught.exception), "flush failed")
            self.assertTrue(path.exists())
            self.assertNotEqual(path.read_text(encoding="utf-8"), token)
            path.unlink()

            def fail_before_bytes(handle):
                def bad_write(data):
                    raise OSError("write failed")

                handle.write = bad_write

            def unreadable(self, *args, **kwargs):
                raise PermissionError("denied")

            with patch("builtins.open", open_with(fail_before_bytes)):
                with patch.object(Path, "read_text", unreadable):
                    with self.assertRaises(OSError) as caught:
                        acquire_cpu_lock(path, token)
            self.assertEqual(str(caught.exception), "write failed")
            self.assertNotIsInstance(caught.exception, PermissionError)
            self.assertTrue(path.exists())
            path.unlink()

            path.write_text(token, encoding="utf-8")
            with patch.object(Path, "read_text", unreadable):
                with self.assertRaises(PermissionError):
                    release_cpu_lock(path, token)
            self.assertEqual(path.read_text(encoding="utf-8"), token)

            real_unlink = Path.unlink

            def deny_unlink(self, *args, **kwargs):
                if self == path:
                    raise PermissionError("cannot delete")
                return real_unlink(self, *args, **kwargs)

            with patch.object(Path, "unlink", deny_unlink):
                with self.assertRaises(PermissionError):
                    release_cpu_lock(path, token)
            self.assertEqual(path.read_text(encoding="utf-8"), token)

            release_cpu_lock(Path(tmp) / "gone.lock", token)
            other = "C6 8 app"
            path.write_text(other, encoding="utf-8")
            release_cpu_lock(path, token)
            self.assertEqual(path.read_text(encoding="utf-8"), other)

    def test_final_lock_covers_construction_and_releases(self):
        import turbo.accept as accept_mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = root / "cpu.lock"
            video = root / "clip.mp4"
            video.write_bytes(b"not-a-video")
            out = root / "out"
            seen = {}

            class Runner:
                IS_REPLAY = False

                def __init__(self, models_dir):
                    seen["init_lock"] = lock.read_text(encoding="utf-8")
                    seen["models"] = models_dir

                def warm_up(self, video_path):
                    seen["warm_lock"] = lock.read_text(encoding="utf-8")
                    raise RuntimeError("stop before runs")

            clips = {
                "envelope": {},
                "clips": {
                    clip: {"file": str(video), "probe": {"frame_count": 1, "fps": 30.0, "width": 1, "height": 1}}
                    for clip in ("01", "02", "03", "04", "05", "07", "08", "09", "10")
                },
            }
            saved = (
                accept_mod.LOCK,
                accept_mod.background_cpu,
                accept_mod.load_class,
                accept_mod.load_clip_table,
                accept_mod.warmup_video,
            )
            try:
                accept_mod.LOCK = lock
                accept_mod.background_cpu = lambda seconds: {"cpu_percent": 0.4, "sample_seconds": seconds}
                accept_mod.load_class = lambda spec: ("coach", Runner)
                accept_mod.load_clip_table = lambda: clips
                accept_mod.warmup_video = lambda out_dir, table: video
                args = parse_args(["final", "--runner", "coach=coach.app.pipeline:E2ERunner", "--models-dir", "models", "--out", str(out)])
                with self.assertRaises(RuntimeError):
                    accept_mod.run_acceptance(args)
            finally:
                (
                    accept_mod.LOCK,
                    accept_mod.background_cpu,
                    accept_mod.load_class,
                    accept_mod.load_clip_table,
                    accept_mod.warmup_video,
                ) = saved
            self.assertFalse(lock.exists())
            self.assertTrue(seen["init_lock"].startswith("H "))
            self.assertEqual(seen["warm_lock"], seen["init_lock"])
            self.assertEqual(seen["models"], "models")

            other = "S 77 census"
            constructed = []

            class Idle:
                IS_REPLAY = False

                def __init__(self, models_dir):
                    constructed.append(models_dir)

            def sample_and_steal(seconds):
                lock.write_text(other, encoding="utf-8")
                return {"cpu_percent": 1.0, "sample_seconds": seconds}

            sampled = []
            try:
                accept_mod.LOCK = lock
                accept_mod.load_class = lambda spec: ("coach", Idle)
                accept_mod.background_cpu = lambda seconds: sampled.append(seconds)
                lock.write_text("C6 3 live", encoding="utf-8")
                args = parse_args(["final", "--runner", "coach=x:Y", "--models-dir", "m", "--out", str(root / "blocked")])
                with self.assertRaises(Refusal):
                    accept_mod.run_acceptance(args)
                self.assertEqual(sampled, [])
                self.assertEqual(lock.read_text(encoding="utf-8"), "C6 3 live")
                self.assertEqual(constructed, [])

                lock.unlink()
                accept_mod.background_cpu = sample_and_steal
                with self.assertRaises(Refusal) as caught:
                    accept_mod.run_acceptance(args)
                self.assertIn("cpu.lock exists", str(caught.exception))
                self.assertEqual(lock.read_text(encoding="utf-8"), other)
                self.assertEqual(constructed, [])

                lock.unlink()
                accept_mod.background_cpu = lambda seconds: {"cpu_percent": 9.0, "sample_seconds": seconds}
                with self.assertRaises(Refusal):
                    accept_mod.run_acceptance(args)
                self.assertFalse(lock.exists())
                self.assertEqual(constructed, [])

                def stop():
                    raise SystemExit("stop")

                accept_mod.load_clip_table = stop
                accept_mod.load_class = lambda spec: ("replay", type("R", (), {"IS_REPLAY": True}))
                dev = parse_args(["dev", "--runner", "replay=turbo.replay:ReplayRunner", "--models-dir", "m", "--out", str(root / "dev"), "--runs", "1"])
                with self.assertRaises(SystemExit):
                    accept_mod.run_acceptance(dev)
                self.assertFalse(lock.exists())
            finally:
                (
                    accept_mod.LOCK,
                    accept_mod.background_cpu,
                    accept_mod.load_class,
                    accept_mod.load_clip_table,
                    accept_mod.warmup_video,
                ) = saved

    def test_final_cli_has_no_limit_or_run_flag(self):
        with self.assertRaises(SystemExit):
            parse_args(["final", "--runner", "a=b:C", "--models-dir", "m", "--out", "o", "--runs", "1"])
        with self.assertRaises(SystemExit):
            parse_args(["final", "--runner", "a=b:C", "--models-dir", "m", "--out", "o", "--limit", "9"])

    def test_headline_is_only_s_a2(self):
        self.assertEqual(headline("S-A", True, True, True)["verdict"], "NOT_MET")
        self.assertEqual(headline("S-A2", True, True, True)["verdict"], "MET")
        self.assertEqual(headline("S-A2", False, True, True)["verdict"], "NOT_MET")

    def test_fresh_copy_changes_hash_and_keeps_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.mp4"
            proc = subprocess.run(
                [str(FFMPEG), "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=30:duration=0.3",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-400:])
            copied = fresh_copy(src, Path(tmp) / "out")
            proof = evidence(copied)
            self.assertNotEqual(proof["source_sha256"], proof["copy_sha256"])
            self.assertTrue(proof["frames_match"])
            self.assertGreaterEqual(len(proof["checked_frames"]), 5)
            self.assertNotEqual(copied.name, src.name)

    def test_swapped_roles_score_zero_while_best_row_scores_high(self):
        ref = reference([[person("near", 100), person("far", 800)]])
        pred = prediction([[player("near", 800), player("far", 100)]])
        scored = score_pair(pred, ref, exclude=set(), fps=30)
        self.assertEqual(scored["all"]["pck10"]["hits"], 0)
        self.assertEqual(scored["all"]["pck10"]["total"], 34)
        self.assertGreaterEqual(scored["best_row"]["pck10_exact"], 0.99)

    def test_repeated_frame_is_excluded(self):
        ref = reference([[person("near", 0, h=100)], [person("near", 0, h=100)]])
        pred = prediction([[player("near", 0)], [player("near", 500)]])
        kept_score = score_pair(pred, ref, exclude={1}, fps=30)
        raw = score_pair(pred, ref, exclude=set(), fps=30)
        self.assertEqual(kept_score["all"]["pck10"]["hits"], 17)
        self.assertEqual(kept_score["all"]["pck10"]["total"], 17)
        self.assertLess(raw["all"]["pck10"]["hits"], raw["all"]["pck10"]["total"])

    def test_fast_subset_is_the_top_quarter(self):
        people = []
        players = []
        xs = [0, 0, 300, 300]
        for x in xs:
            people.append([person("near", x, h=100)])
            players.append([player("near", x)])
        scored = score_pair(prediction(players), reference(people), exclude=set(), fps=30)
        self.assertEqual(scored["eligible_frames"], 4)
        self.assertEqual(scored["fast_frames"], [1])
        tied = score_pair(prediction([[player("near", 0)]] * 4), reference([[person("near", 0, h=100)]] * 4),
                          exclude=set(), fps=30)
        self.assertEqual(tied["fast_frames"], [0])
        masked = score_pair(prediction(players), reference(people), exclude={2}, fps=30)
        self.assertNotIn(2, masked["fast_frames"])

    def test_exclude_index_records_repeats_and_phase0_copies(self):
        rec = exclude_indexes(["a", "a", "b", "a"])
        self.assertEqual(rec["repeat_frames"], [1])
        self.assertIn(0, rec["phase0_frames"])
        self.assertIn(3, rec["phase0_frames"])
        self.assertEqual(rec["phase0_pixel_duplicates"], [1])
        self.assertIn(1, rec["exclude"])
        self.assertIn(3, rec["exclude"])

    def _both_roles(self, doc):
        scored = dict(doc)
        scored["by_role"] = {"near": doc, "far": doc}
        return scored

    def test_empty_quality_evidence_is_not_a_pass(self):
        empty = {"all": {"pck10": {"hits": 0, "total": 0}, "found": {"hits": 0, "total": 0}},
                 "fast": {"pck10": {"hits": 0, "total": 0}}}
        gate = quality_gate(empty, empty, "preview")
        self.assertFalse(gate["passed"])
        self.assertTrue(gate["insufficient"])

    def test_quality_margin_uses_exact_counts(self):
        def pack(hits, total):
            return {"hits": hits, "total": total, "rate": hits / total}
        base = self._both_roles({
            "all": {"pck10": pack(80, 100), "pck05": pack(70, 100), "found": pack(90, 100)},
            "fast": {"pck10": pack(50, 100)},
        })
        ok = self._both_roles({
            "all": {"pck10": pack(77, 100), "found": pack(88, 100)},
            "fast": {"pck10": pack(44, 100)},
        })
        low = self._both_roles({
            "all": {"pck10": pack(76, 100), "found": pack(88, 100)},
            "fast": {"pck10": pack(44, 100)},
        })
        self.assertTrue(quality_gate(ok, base, "preview")["passed"])
        self.assertFalse(quality_gate(low, base, "preview")["passed"])
        refined_ok = self._both_roles({"all": {"pck05": pack(700, 1000)}})
        refined_base = self._both_roles({"all": {"pck05": pack(702, 1000)}})
        refined_far = self._both_roles({"all": {"pck05": pack(699, 1000)}})
        self.assertTrue(quality_gate(refined_ok, refined_base, "refined")["passed"])
        self.assertFalse(quality_gate(refined_far, refined_base, "refined")["passed"])

    def test_near_pass_cannot_hide_a_far_loss(self):
        def pack(hits, total):
            return {"hits": hits, "total": total, "rate": hits / total}

        def role(pck, found, fast):
            return {"all": {"pck10": pack(*pck), "found": pack(*found)}, "fast": {"pck10": pack(*fast)}}

        base_role = role((80, 100), (90, 100), (50, 100))
        near = role((100, 100), (100, 100), (100, 100))
        far = role((60, 100), (60, 100), (30, 100))
        score = {
            "all": {"pck10": pack(160, 200), "found": pack(160, 200)},
            "fast": {"pck10": pack(130, 200)},
            "by_role": {"near": near, "far": far},
        }
        base = {
            "all": {"pck10": pack(160, 200), "found": pack(180, 200)},
            "fast": {"pck10": pack(100, 200)},
            "by_role": {"near": base_role, "far": base_role},
        }
        gate = quality_gate(score, base, "preview")
        self.assertFalse(gate["passed"])
        self.assertTrue(any(item.startswith("far ") for item in gate["failures"]))
        self.assertFalse(any(item.startswith("near ") for item in gate["failures"]))
        self.assertEqual(score["all"]["pck10"]["hits"], 160)

    def test_binding_refuses_mixed_models_and_a_changed_asset(self):
        asset_name = "h-binding-probe.onnx"

        class Holder:
            def __init__(self, models_dir, svc_dir=None):
                self.models_dir = Path(models_dir)
                self.model = Path(models_dir) / asset_name
                self.svc = None if svc_dir is None else type("Svc", (), {"models_dir": Path(svc_dir)})()

        with tempfile.TemporaryDirectory() as tmp:
            given = Path(tmp) / "given"
            other = Path(tmp) / "other"
            given.mkdir()
            other.mkdir()
            with self.assertRaises(Refusal):
                execution_binding(Holder(other), given)
            with self.assertRaises(Refusal):
                execution_binding(Holder(given, other), given)
            asset = given / asset_name
            asset.write_bytes(b"model-v1")
            runner = Holder(given)
            first = execution_binding(runner, given)
            self.assertTrue(any(asset_name in key for key in first))
            asset.write_bytes(b"model-v2")
            second = execution_binding(runner, given)
            self.assertNotEqual(first, second)

    def test_nested_helper_and_config_mutation_are_bound(self):
        decoy_name = "decoy-not-loaded.onnx"

        class Holder:
            def __init__(self, models_dir):
                self.models_dir = Path(models_dir)
                from turbo.tests.nested_helper import NestedModel
                self._pose = NestedModel(Path(models_dir) / "nested-helper.onnx",
                                         Path(models_dir) / "nested-config.json")

        with tempfile.TemporaryDirectory() as tmp:
            given = Path(tmp) / "given"
            given.mkdir()
            (given / decoy_name).write_bytes(b"not-a-runtime-input")
            model = given / "nested-helper.onnx"
            config = given / "nested-config.json"
            model.write_bytes(b"weights")
            config.write_text('{"limit": 1}', encoding="utf-8")
            runner = Holder(given)
            first = execution_binding(runner, given)
            self.assertTrue(any(key.endswith("nested_helper.py") for key in first))
            self.assertTrue(any(key.endswith("nested-config.json") for key in first))
            self.assertTrue(any(key.endswith("coach/style.json") for key in first))
            self.assertFalse(any(decoy_name in key for key in first))
            config.write_text('{"limit": 2}', encoding="utf-8")
            second = execution_binding(runner, given)
            self.assertNotEqual(first[str(config.resolve())], second[str(config.resolve())])
            self.assertEqual(first[str(model.resolve())], second[str(model.resolve())])

    def test_compiled_fallback_model_stays_bound_and_video_paths_do_not(self):
        from turbo.service import TurboService

        with tempfile.TemporaryDirectory() as tmp:
            given = Path(tmp) / "given"
            given.mkdir()
            video = Path(tmp) / "clip.mp4"
            output = Path(tmp) / "pose.json"
            decoy = Path(tmp) / "not-compiled.onnx"
            video.write_bytes(b"not-a-model")
            output.write_bytes(b"{}")
            decoy.write_bytes(b"decoy")
            svc = TurboService(given)
            fallback = Path(svc._onnx("rtmpose-t-ft3.onnx")).resolve()
            self.assertTrue(fallback.is_file())
            self.assertNotEqual(fallback.parent, given.resolve())
            self.assertTrue(str(fallback).replace("\\", "/").endswith(
                "fast/weights/rtmpose-t-ft3.onnx"))
            st = fallback.stat()
            mod = types.ModuleType("ovcache")
            mod._source_hashes = {(str(fallback), st.st_size, st.st_mtime_ns): "digest"}
            previous = sys.modules.get("ovcache")
            sys.modules["ovcache"] = mod
            try:
                class Holder:
                    def __init__(self):
                        self.models_dir = given
                        self.svc = svc
                        self.video = str(video)
                        self.out_path = output
                        self.extra_model = decoy

                bound = execution_binding(Holder(), given)
            finally:
                if previous is None:
                    sys.modules.pop("ovcache", None)
                else:
                    sys.modules["ovcache"] = previous
            key = str(fallback.relative_to(PROJECT)).replace("\\", "/")
            digest = hashlib.sha256(fallback.read_bytes()).hexdigest()
            self.assertEqual(bound.get(key), digest)
            joined = "\n".join(bound)
            self.assertNotIn("clip.mp4", joined)
            self.assertNotIn("pose.json", joined)
            self.assertNotIn("not-compiled.onnx", joined)

    def test_unavailable_pose_reference_is_not_a_score(self):
        import turbo.accept as accept

        called = []
        original = accept.read_reference

        def boom(mode, clip):
            called.append((mode, clip))
            raise AssertionError("sealed reference was opened")

        accept.read_reference = boom
        try:
            aggregate = {
                "clips": {
                    "11": {"pose_quality_reference": "unavailable", "probe": {"fps": 30.0}},
                    "01": {"probe": {"fps": 30.0}},
                },
                "runs": [
                    {"clip": "11", "valid": True, "output": "should-not-be-read.json"},
                    {"clip": "11", "valid": False, "output": "should-not-be-read.json"},
                ],
            }
            accept._score("final", aggregate)
            with self.assertRaises(AssertionError):
                accept._score("dev", {"clips": aggregate["clips"], "runs": [
                    {"clip": "01", "valid": True, "output": "also-not-a-score.json"},
                ]})
        finally:
            accept.read_reference = original
        self.assertEqual(called, [("dev", "01")])
        valid, invalid = aggregate["runs"]
        for run in (valid, invalid):
            self.assertFalse(run["quality"]["passed"])
            self.assertTrue(run["quality"]["unavailable"])
            self.assertIsNone(run["score"])
            self.assertEqual(run["quality"]["failures"], [])
            self.assertIn("pose-quality reference unavailable", run["quality"]["insufficient"])
        self.assertEqual(valid["quality"]["insufficient"], ["pose-quality reference unavailable"])
        self.assertEqual(invalid["quality"]["insufficient"][0], "invalid output")

    def test_measured_flags_follow_raw_joints_not_packed_copies(self):
        from turbo.accept import _expected_pose, _q_conf, _q_xy

        def row(x=10.0, y=20.0, conf=0.80):
            return [[x, y, conf] for _ in range(17)]

        def document(rows, cuts=None):
            frames = []
            for index, players in enumerate(rows):
                frame = {"frame_index": index, "players": []}
                for role, joints in players.items():
                    if joints is None:
                        continue
                    frame["players"].append({
                        "role": role,
                        "joints": joints,
                        "bbox_xyxy": [1.0, 2.0, 30.0, 40.0],
                    })
                frames.append(frame)
            return {"frames": frames, "cuts": cuts or []}

        conf_a = row(conf=0.6200)
        conf_b = row(conf=0.6207)
        self.assertEqual(_q_conf(0.6200), _q_conf(0.6207))
        slots, measured = _expected_pose(document([{"far": conf_a}, {"far": conf_b}]), 2, ["far"])
        self.assertEqual(slots["far"]["conf"][:17], slots["far"]["conf"][17:])
        self.assertEqual(measured["far"], [1, 1])

        moved = row(x=10.04)
        self.assertEqual(_q_xy(10.01), _q_xy(10.04))
        slots, measured = _expected_pose(document([{"near": row(x=10.01)}, {"near": moved}]), 2, ["near"])
        self.assertEqual(slots["near"]["xy"][:34], slots["near"]["xy"][34:68])
        self.assertEqual(measured["near"], [1, 1])

        copied = row()
        _, measured = _expected_pose(document([
            {"near": copied, "far": row(x=8.0)},
            {"near": [list(item) for item in copied], "far": row(x=8.0)},
        ]), 2, ["near", "far"])
        self.assertEqual(measured["near"], [1, 0])
        self.assertEqual(measured["far"], [1, 0])

        present_zero = row()
        present_zero[3] = [10.0, 20.0, 0.0]
        missing = row()
        missing[3] = [10.0, 20.0, None]
        self.assertEqual(_q_conf(0.0), _q_conf(None))
        _, measured = _expected_pose(document([{"far": present_zero}, {"far": missing}]), 2, ["far"])
        self.assertEqual(measured["far"], [1, 1])

        same = row(x=4.0)
        _, measured = _expected_pose(document([
            {"near": same},
            {},
            {"near": [list(item) for item in same]},
        ]), 3, ["near"])
        self.assertEqual(measured["near"], [1, 0, 1])

        _, measured = _expected_pose(document([
            {"far": same},
            {"far": [list(item) for item in same]},
        ], cuts=[1]), 2, ["far"])
        self.assertEqual(measured["far"], [1, 1])


if __name__ == "__main__":
    unittest.main()
