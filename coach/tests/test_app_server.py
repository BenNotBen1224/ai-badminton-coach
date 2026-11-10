"""The local app over real HTTP: queue order, revisions, request checks, files and the provider routes.

The pose backend is a ReplayBackend that logs every call and can hold one kind of call at a gate, so the tests
can look at the app while an analysis is in flight. Chat, plan and review-video providers are test fakes, except
in the tests that drive unit C4's real coach and plan through the routes.
"""
import copy
import hashlib
import http.client
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Optional
from unittest import mock
from urllib.parse import quote

from coach.app import server
from coach.app.backend import PROJECT, NoCourt, ReplayBackend, sha256_file
from coach.app.pipeline import MESSAGES
from coach.app.server import WARM_FAILED, WEB, App, CoachChat, PlanMaker, Server
from coach.app.store import Store
from coach.chat.plan import EASED, NO_PLAN
from coach.schema import WARNINGS, Role, Turn
from coach.synth import Noise, PlayerPlan, Swing, script
from coach.tests.fixture_report import full_report
from coach.tests.test_app_pipeline import FPS, N, STAGES, write_video
from coach.tests.test_app_store import (REPLAY, UNSURE, UNTAKEN, legacy_folder, metric_rows, replay_bytes, replay_report,
                                        snapshot)

# Contact frames 20 (near) and 65 (far) of the 90-frame rally give these key-moment windows at stride 3.
WINDOWS = [[15, 18], [60, 63]]
LOGGER = logging.getLogger("coach.app")
_level = LOGGER.level


def setUpModule():
    LOGGER.setLevel(logging.CRITICAL)


def tearDownModule():
    LOGGER.setLevel(_level)


class Recording(ReplayBackend):
    """Logs (kind, session id, window) per call. A cleared gate holds that kind of call until it is opened.
    With keep_open, a call holds the video open the way a real analyser reading it does."""

    def __init__(self, docs, preview_stride: Optional[int] = 3, errors=None):
        super().__init__(docs, preview_stride)
        self.errors = dict(errors or {})
        self.log: list[tuple[str, str, Optional[tuple[int, int]]]] = []
        self.cv = threading.Condition()
        self.gates = {kind: threading.Event() for kind in ("preview", "refine", "refined")}
        self.release()
        self.active = 0
        self.most_active = 0
        self.keep_open = False

    def hold(self, kind: str) -> None:
        self.gates[kind].clear()

    def open(self, kind: str) -> None:
        self.gates[kind].set()

    def release(self) -> None:
        for gate in self.gates.values():
            gate.set()

    def wait_calls(self, count: int, timeout: float = 30.0) -> list:
        with self.cv:
            if not self.cv.wait_for(lambda: len(self.log) >= count, timeout):
                raise AssertionError(f"expected {count} backend calls, got {self.log}")
            return list(self.log)

    def _run(self, kind: str, video: Path, window, work):
        handle = open(video, "rb") if self.keep_open else None
        with self.cv:
            self.log.append((kind, Path(video).parent.name, window))
            self.active += 1
            self.most_active = max(self.most_active, self.active)
            self.cv.notify_all()
        try:
            if not self.gates[kind].wait(30):
                raise RuntimeError(f"the test never opened the {kind} gate")
            error = self.errors.get(sha256_file(Path(video))) if self.errors else None
            if error is not None:
                raise error
            return work()
        finally:
            if handle is not None:
                handle.close()
            with self.cv:
                self.active -= 1

    def preview(self, video, out_json=None):
        return self._run("preview", video, None, partial(super().preview, video, out_json))

    def refine(self, video, doc, windows, out_json=None):
        window = tuple(windows[0]) if len(windows) == 1 else tuple(map(tuple, windows))
        return self._run("refine", video, window, partial(super().refine, video, doc, windows, out_json))

    def refined(self, video, out_json=None):
        return self._run("refined", video, None, partial(super().refined, video, out_json))


@dataclass
class Reply:
    status: int
    headers: Any
    body: bytes

    def json(self):
        return json.loads(self.body)


class Running:
    """A store, an App and a Server on a free port, with plain HTTP helpers."""

    def __init__(self, data_dir: Path, backend=None, *, make_backend=None, samples_dir: Path, **app_options):
        self.backend = backend
        self.data_dir = Path(data_dir)
        self.store = Store(self.data_dir)
        self.app = App(self.store, make_backend or (lambda: backend), samples_dir=samples_dir, **app_options)
        self.server = Server(self.app, 0)
        self.port = self.server.server_address[1]
        self.host = f"127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()
        self.app.start()
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.backend is not None:
            self.backend.release()
        self.app.stop()
        self.server.shutdown()
        self.server.server_close()
        self.store.close()

    def folder(self, session_id: str) -> Path:
        return self.data_dir / "sessions" / session_id

    def call(self, method: str, path: str, body: Optional[bytes] = None, headers: Optional[dict] = None) -> Reply:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            return Reply(response.status, response.headers, response.read())
        finally:
            conn.close()

    def raw(self, method: str, path: str, headers=(), body: Optional[bytes] = None, host: bool = True) -> Reply:
        """Exactly these headers: http.client adds no Host, Accept-Encoding or Content-Length of its own."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        try:
            conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
            if host:
                conn.putheader("Host", self.host)
            for key, value in headers:
                conn.putheader(key, value)
            conn.endheaders(body)
            response = conn.getresponse()
            return Reply(response.status, response.headers, response.read())
        finally:
            conn.close()

    def get(self, path: str) -> tuple[int, Any]:
        reply = self.call("GET", path)
        return reply.status, reply.json()

    def post(self, path: str, payload: Any) -> tuple[int, Any]:
        reply = self.call("POST", path, json.dumps(payload).encode("utf-8"), {"Content-Type": "application/json"})
        return reply.status, reply.json()

    def upload(self, data: bytes, name: str = "rally.mp4") -> str:
        reply = self.call("POST", "/api/upload", data, {"X-Filename": quote(name)})
        if reply.status != 201:
            raise AssertionError(f"upload answered {reply.status}: {reply.body!r}")
        return reply.json()["id"]

    def status(self, session_id: str) -> dict:
        code, body = self.get(f"/api/session/{session_id}")
        if code != 200:
            raise AssertionError(f"status answered {code}: {body}")
        return body

    def wait_for(self, session_id: str, done, timeout: float = 30.0) -> dict:
        """Long-polls the status until done(status) is true."""
        deadline = time.monotonic() + timeout
        status = self.status(session_id)
        while not done(status):
            if time.monotonic() > deadline:
                raise AssertionError(f"timed out; the last status was {status}")
            code, status = self.get(f"/api/session/{session_id}?since={status['stamp']}&wait=2")
            if code != 200:
                raise AssertionError(f"status answered {code}: {status}")
        return status

    def ready(self, session_id: str, revision: int = 0) -> dict:
        return self.wait_for(session_id, lambda s: s["status"] in ("ready", "error") and (
            s["status"] == "error" or s["revision"] == revision))

    def cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-m", "coach.app", *args, "--data-dir", str(self.data_dir)],
                              cwd=PROJECT, capture_output=True, text=True, timeout=180)


FAKE_STOP = "I will stop the coaching here."


class FakeChat:
    stop_texts = (FAKE_STOP,)

    def __init__(self):
        self.calls = []

    def __call__(self, key, report, history, text, context, earlier):
        self.calls.append({"key": key, "session": report.video_id, "history": list(history), "text": text,
                           "context": dict(context), "earlier": [r.video_id for r in earlier]})
        said = FAKE_STOP if text == "my knee hurts" else f"you asked: {text}"
        return {"text": said}, {**context, "turns": len(history) + 2}


class StoppingChat(FakeChat):
    """Answers the way the coach does around a stop for pain: the pain message gets the stop, and once the chat is
    stopped every answer is the stop. Any other question gets a drill."""

    def __call__(self, key, report, history, text, context, earlier):
        super().__call__(key, report, history, text, context, earlier)
        if text == "my knee hurts" or context.get("pain_stop"):
            return {"text": FAKE_STOP}, {**context, "pain_stop": True}
        return {"text": f"a drill for: {text}"}, context


class HeldChat:
    """A chat whose first answer to `held` waits until the test opens the gate, the way a slow reply is still being
    made while another tab records a stop. Everything else is the wrapped chat's."""

    def __init__(self, chat, held: str):
        self.chat, self.held = chat, held
        self.entered, self.gate = threading.Event(), threading.Event()

    def __getattr__(self, name):
        return getattr(self.chat, name)

    def __call__(self, key, report, history, text, context, earlier):
        if text == self.held and not self.entered.is_set():
            self.entered.set()
            if not self.gate.wait(30):
                raise RuntimeError("the test never opened the chat gate")
        return self.chat(key, report, history, text, context, earlier)


class FakePlans:
    def __init__(self):
        self.calls = []

    def __call__(self, report, player, weeks, level, equipment, avoid=()):
        self.calls.append((report.video_id, player.value, weeks, level, equipment, avoid))
        return {"player": player.value, "weeks": weeks, "level": level, "equipment": list(equipment or ())}


class HeldPlans(FakePlans):
    """Each call waits while the gate is closed, the way a slow planner is still working while another tab records
    a stop. At level 1 it finds nothing to plan."""

    def __init__(self):
        super().__init__()
        self.entered, self.gate = threading.Event(), threading.Event()
        self.gate.set()

    def __call__(self, report, player, weeks, level, equipment, avoid=()):
        self.entered.set()
        if not self.gate.wait(30):
            raise RuntimeError("the test never opened the plan gate")
        plan = super().__call__(report, player, weeks, level, equipment, avoid)
        return {"weeks": [], "message": NO_PLAN} if level == 1 else plan


class FakeExporter:
    """Writes the part file, then fails once for each session id in `fail`."""

    def __init__(self):
        self.calls = []
        self.gate = threading.Event()
        self.gate.set()
        self.fail: set[str] = set()

    def __call__(self, video, pose_doc, report, part: Path):
        self.calls.append((report.video_id, report.revision, part.name))
        if not self.gate.wait(30):
            raise RuntimeError("the test never opened the export gate")
        part.write_bytes(b"review:" + report.video_id.encode())
        if report.video_id in self.fail:
            self.fail.discard(report.video_id)
            raise RuntimeError("the encoder failed")


def until(check, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while True:
        value = check()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.02)


class ServerCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.dir = Path(cls.tmp.name)
        # Different sizes give different bytes, so each video has its own SHA-256.
        cls.rally = write_video(cls.dir / "rally.mp4", N, FPS).read_bytes()
        cls.no_players = write_video(cls.dir / "no-players.mp4", 60, FPS, size=(48, 32)).read_bytes()
        cls.no_court = write_video(cls.dir / "no-court.mp4", 30, FPS, size=(32, 32)).read_bytes()
        cls.crash = write_video(cls.dir / "crash.mp4", 30, FPS, size=(40, 24)).read_bytes()
        cls.too_long = write_video(cls.dir / "too-long.mp4", 182, 1.0, size=(16, 16)).read_bytes()
        cls.long = write_video(cls.dir / "long.mp4", 40, 1.0, size=(24, 24)).read_bytes()
        two_swings, _ = script(n=N, fps=FPS, near=PlayerPlan(court_xy=(3.05, 10.5), swing=Swing(20, 2.6, 160.0)),
                               far=PlayerPlan(court_xy=(3.05, 2.9), swing=Swing(65, 2.6, 160.0)))
        one_player, _ = script(n=60, fps=FPS, noise=Noise(missing_role="far", missing_frames=tuple(range(60))))
        digest = lambda name: sha256_file(cls.dir / name)
        cls.docs = {digest("rally.mp4"): two_swings, digest("no-players.mp4"): one_player}
        cls.errors = {digest("no-court.mp4"): NoCourt("no court found in this clip"),
                      digest("crash.mp4"): RuntimeError("the decoder crashed")}
        cls.samples = cls.dir / "samples"
        cls.samples.mkdir()
        shutil.copyfile(cls.dir / "rally.mp4", cls.samples / "01-test.mp4")
        cls.no_samples = cls.dir / "no-samples"
        cls.no_samples.mkdir()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def backend(self, preview_stride: Optional[int] = 3) -> Recording:
        return Recording(self.docs, preview_stride, self.errors)

    def data_dir(self) -> Path:
        return Path(tempfile.mkdtemp(dir=self.dir))

    def start(self, backend=None, data_dir: Optional[Path] = None, samples_dir: Optional[Path] = None,
              **options) -> Running:
        running = Running(data_dir or self.data_dir(), backend, samples_dir=samples_dir or self.no_samples, **options)
        self.addCleanup(running.close)
        return running


class QueueTest(ServerCase):
    def test_a_clip_is_previewed_then_its_key_moments_sharpened_into_a_new_revision(self):
        backend = self.backend()
        running = self.start(backend)
        backend.hold("refine")
        sid = running.upload(self.rally)
        self.assertEqual(backend.wait_calls(2), [("preview", sid, None), ("refine", sid, (15, 18))])

        held = running.status(sid)
        self.assertEqual((held["status"], held["detail"], held["revision"], held["tier"]),
                         ("refining", "Sharpening key moments", 0, "preview"))
        self.assertEqual((held["windows"], held["sharpened"]), ({"done": 0, "total": 2}, []))
        self.assertEqual(held["bundle_url"], f"/api/session/{sid}/bundle?revision=0")
        first_files = {name: (running.folder(sid) / "r0" / name).read_bytes()
                       for name in ("pose.json", "report.json", "bundle.json")}

        backend.open("refine")
        done = running.ready(sid, revision=1)
        self.assertEqual(backend.wait_calls(3), [("preview", sid, None), ("refine", sid, (15, 18)),
                                                 ("refine", sid, (60, 63))])
        self.assertEqual((done["status"], done["detail"], done["tier"], done["windows"], done["error"]),
                         ("ready", "", "preview", None, None))
        self.assertEqual(done["sharpened"], WINDOWS)
        self.assertEqual((done["n_frames"], done["fps"], done["duration_s"], done["long_clip"]), (N, FPS, 3.0, False))
        self.assertEqual(done["bundle_url"], f"/api/session/{sid}/bundle?revision=1")
        for name, data in first_files.items():
            self.assertEqual((running.folder(sid) / "r0" / name).read_bytes(), data, name)

        pose = json.loads((running.folder(sid) / "r1" / "pose.json").read_text("utf-8"))
        self.assertEqual(pose["tier"], "preview")
        self.assertEqual([[w["start"], w["end"]] for w in pose["windows_refined"]], WINDOWS)
        report = json.loads((running.folder(sid) / "r1" / "report.json").read_text("utf-8"))
        self.assertEqual((report["video_id"], report["revision"], report["tier"]), (sid, 1, "preview"))
        self.assertEqual([p["role"] for p in done["summary"]["players"]], ["near", "far"])

        for revision in (0, 1):
            reply = running.call("GET", f"/api/session/{sid}/bundle?revision={revision}")
            self.assertEqual(reply.status, 200)
            self.assertEqual(reply.body, (running.folder(sid) / f"r{revision}" / "bundle.json").read_bytes())
        self.assertEqual(running.call("GET", f"/api/session/{sid}/bundle").body,
                         (running.folder(sid) / "r1" / "bundle.json").read_bytes())
        self.assertEqual(running.call("GET", f"/api/session/{sid}/bundle?revision=2").status, 404)
        self.assertEqual(running.call("GET", f"/api/session/{sid}/bundle?revision=one").status, 400)
        bundle = json.loads(running.call("GET", f"/api/session/{sid}/bundle?revision=1").body)
        self.assertEqual((bundle["video_url"], bundle["n_frames"], bundle["schema_version"]), (f"/media/{sid}", N, 3))

        first, latest = done["timings"]["first"], done["timings"]["latest"]
        self.assertEqual(set(first), STAGES | {"backend", "upload_s", "queue_s", "warm_wait_s"})
        self.assertEqual(first["backend"], "replay")
        self.assertGreaterEqual(first["upload_s"], 0.0)
        self.assertGreaterEqual(first["queue_s"], 0.0)
        self.assertEqual(set(latest), (STAGES - {"probe"}) | {"backend", "kind", "windows"})
        self.assertEqual((latest["backend"], latest["kind"], latest["windows"]), ("replay", "windows", 2))
        self.assertEqual(len(latest["pose_stages"]["calls"]), 2)

        code, listing = running.get("/api/sessions")
        self.assertEqual(code, 200)
        self.assertEqual([(s["id"], s["status"], s["tier"], s["backend"], s["analysis_s"])
                          for s in listing["sessions"]], [(sid, "ready", "preview", "replay", first["total"])])
        media = running.call("GET", f"/media/{sid}")
        self.assertEqual((media.status, media.headers["Content-Type"], media.body), (200, "video/mp4", self.rally))
        self.assertEqual(backend.most_active, 1)

    def test_sharpening_pauses_between_windows_while_a_new_clip_waits(self):
        backend = self.backend()
        running = self.start(backend)
        backend.hold("refine")
        first = running.upload(self.rally, "first.mp4")
        backend.wait_calls(2)
        backend.hold("preview")
        second = running.upload(self.rally, "second.mp4")
        backend.open("refine")
        backend.wait_calls(3)

        paused = running.status(first)
        self.assertEqual((paused["status"], paused["detail"], paused["windows"], paused["revision"]),
                         ("refining", "Paused to analyse a new clip first", {"done": 1, "total": 2}, 0))
        self.assertEqual(paused["sharpened"], [])
        self.assertEqual(running.status(second)["status"], "preview")

        backend.open("preview")
        self.assertEqual(running.ready(first, revision=1)["sharpened"], WINDOWS)
        self.assertEqual(running.ready(second, revision=1)["sharpened"], WINDOWS)
        self.assertEqual(backend.wait_calls(6), [
            ("preview", first, None), ("refine", first, (15, 18)), ("preview", second, None),
            ("refine", first, (60, 63)), ("refine", second, (15, 18)), ("refine", second, (60, 63)),
        ])
        self.assertEqual(backend.most_active, 1)

    def test_key_moments_covering_most_of_a_short_clip_are_measured_in_one_pass(self):
        # The rally's two key moments cover 8 of its 90 frames, so lowering the share makes them "most of it".
        cases = [(36.0, [("refined", None)], "all", "refined"),
                 (2.0, [("refine", (15, 18)), ("refine", (60, 63))], "windows", "preview")]
        for tested_s, after_preview, kind, tier in cases:
            with self.subTest(tested_s=tested_s), mock.patch.multiple(server, WHOLE_CLIP_SHARE=0.05,
                                                                       TESTED_SECONDS=tested_s):
                backend = self.backend()
                running = self.start(backend)
                backend.hold(after_preview[0][0])
                sid = running.upload(self.rally)
                backend.wait_calls(2)
                held = running.status(sid)
                done_moments = {"done": 0, "total": 0 if kind == "all" else 2}
                self.assertEqual((held["status"], held["windows"]), ("refining", done_moments))
                backend.release()
                done = running.ready(sid, revision=1)
                self.assertEqual(backend.wait_calls(1 + len(after_preview)),
                                 [("preview", sid, None)] + [(k, sid, w) for k, w in after_preview])
                self.assertEqual((done["tier"], done["timings"]["latest"]["kind"]), (tier, kind))

    def test_sharpen_all_quotes_a_time_only_for_the_measured_analyser_and_clip_size(self):
        backend = self.backend()
        running = self.start(backend, auto_refine=False)
        sid = running.upload(self.rally)
        self.assertIsNone(running.ready(sid)["sharpen_all_s"])
        # The rally is 90 frames, 3.0 s long.
        cases = [("turbo", 36.0, 1080, 11), ("turbo", 3.0, 90, 11), ("turbo", 2.9, 1080, None),
                 ("turbo", 36.0, 89, None), ("fast", 36.0, 1080, None)]
        for name, tested_s, frames, seconds in cases:
            with self.subTest(name=name, tested_s=tested_s, frames=frames), mock.patch.object(backend, "name", name), \
                    mock.patch.multiple(server, TESTED_SECONDS=tested_s, SHARPEN_ALL_FRAMES=frames):
                self.assertEqual(running.status(sid)["sharpen_all_s"], seconds)

    def test_clips_uploaded_together_are_analysed_one_at_a_time(self):
        backend = self.backend()
        running = self.start(backend)
        ids: list[str] = []
        threads = [threading.Thread(target=lambda i=i: ids.append(running.upload(self.rally, f"clip{i}.mp4")))
                   for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        self.assertEqual(len(set(ids)), 4)
        for sid in ids:
            self.assertEqual(running.ready(sid, revision=1)["sharpened"], WINDOWS)
        calls = backend.wait_calls(12)
        self.assertEqual(sorted(kind for kind, _, _ in calls), ["preview"] * 4 + ["refine"] * 8)
        self.assertEqual(backend.most_active, 1)

    def refine(self, running: Running, sid: str, start: Optional[int] = None, end: Optional[int] = None, **body):
        if start is not None:
            body["window"] = {"start_frame": start, "end_frame": end}
        return running.post("/api/refine", {"session": sid, **body})

    def test_the_refine_route_sharpens_one_moment_or_every_frame_once(self):
        backend = self.backend()
        running = self.start(backend, auto_refine=False)
        sid = running.upload(self.rally)
        self.assertEqual(running.ready(sid)["sharpened"], [])

        self.assertEqual(self.refine(running, sid, 15, 18), (202, {"queued": True}))
        self.assertEqual(running.ready(sid, revision=1)["sharpened"], [[15, 18]])
        self.assertEqual(self.refine(running, sid, 15, 18),
                         (200, {"queued": False, "message": "this moment is already sharpened"}))

        bad = [{}, {"window": {"start_frame": 1, "end_frame": 2}, "all": True}, {"all": False}, {"all": "yes"},
               {"window": [15, 18]}, {"window": {"start_frame": "15", "end_frame": 18}},
               {"window": {"start_frame": True, "end_frame": 18}}, {"window": {"start_frame": -1, "end_frame": 18}},
               {"window": {"start_frame": 15, "end_frame": N}}, {"window": {"start_frame": 20, "end_frame": 19}},
               {"window": {"start_frame": 15}}]
        for body in bad:
            self.assertEqual(running.post("/api/refine", {"session": sid, **body})[0], 400, body)
        for session in ("0123456789abcdef", "../" + sid, sid.upper(), 7, None):
            self.assertEqual(running.post("/api/refine", {"session": session, "all": True})[0], 404, session)

        backend.hold("refine")
        self.assertEqual(self.refine(running, sid, 60, 63)[0], 202)
        before = len(backend.wait_calls(3))
        # The same window again while it runs is queued, then found already sharpened and skipped.
        self.assertEqual(self.refine(running, sid, 60, 63)[0], 202)
        self.assertEqual(self.refine(running, sid, 30, 33)[0], 202)
        self.assertEqual(self.refine(running, sid, 30, 33),
                         (200, {"queued": False, "message": "this is already waiting to be sharpened"}))
        self.assertEqual(self.refine(running, sid, all=True)[0], 202)
        self.assertEqual(self.refine(running, sid, all=True),
                         (200, {"queued": False, "message": "this is already waiting to be sharpened"}))
        backend.open("refine")
        done = running.ready(sid, revision=4)
        self.assertEqual(backend.wait_calls(before + 2)[before - 1:], [
            ("refine", sid, (60, 63)), ("refine", sid, (30, 33)), ("refined", sid, None)])
        self.assertEqual(len(backend.log), before + 2)
        pose = json.loads((running.folder(sid) / "r3" / "pose.json").read_text("utf-8"))
        self.assertEqual([[w["start"], w["end"]] for w in pose["windows_refined"]], [[15, 18], [60, 63], [30, 33]])
        self.assertEqual((done["tier"], done["sharpened"], done["timings"]["latest"]["kind"]), ("refined", [], "all"))
        for body in ({"window": {"start_frame": 40, "end_frame": 43}}, {"all": True}):
            self.assertEqual(running.post("/api/refine", {"session": sid, **body}),
                             (409, {"error": "every frame of this clip is already measured"}))

        other = running.upload(self.rally, "other.mp4")
        running.ready(other)
        calls = len(backend.log)
        backend.hold("refined")
        self.assertEqual(self.refine(running, other, all=True)[0], 202)
        backend.wait_calls(calls + 1)
        # A moment asked for while every frame is being measured is queued, then has nothing left to do.
        self.assertEqual(self.refine(running, other, 15, 18), (202, {"queued": True}))
        backend.open("refined")
        self.assertEqual(running.ready(other, revision=1)["tier"], "refined")
        self.assertEqual(backend.log[calls:], [("refined", other, None)])

    def test_an_analyser_that_cannot_sharpen_one_moment_offers_the_whole_clip(self):
        data = self.data_dir()
        running = self.start(self.backend(), data_dir=data, auto_refine=False)
        sid = running.upload(self.rally)
        running.ready(sid)
        running.close()

        backend = self.backend(preview_stride=None)
        running = self.start(backend, data_dir=data)
        self.assertFalse(running.get("/api/health")[1]["backend"]["sharpens_windows"])
        again = running.status(sid)
        self.assertEqual((again["status"], again["revision"], again["tier"], again["sharpened"]),
                         ("ready", 0, "preview", []))
        self.assertEqual(self.refine(running, sid, 15, 18),
                         (409, {"error": "this analyser cannot sharpen one moment; sharpen the whole clip instead"}))
        self.assertEqual(self.refine(running, sid, all=True), (202, {"queued": True}))
        self.assertEqual(running.ready(sid, revision=1)["tier"], "refined")
        self.assertEqual(backend.log, [("refined", sid, None)])

    def test_a_clip_that_cannot_be_analysed_says_why_and_is_not_retried(self):
        backend = self.backend()
        running = self.start(backend)
        cases = [(b"hello", "hello.mp4", "not_a_video"), (self.too_long, "long.mp4", "too_long"),
                 (self.no_players, "solo.mp4", "no_players"), (self.no_court, "close-up.mp4", "no_court"),
                 (self.crash, "crash.mp4", "failed")]
        ids = [running.upload(data, name) for data, name, _ in cases]
        for sid, (_, _, code) in zip(ids, cases):
            status = running.ready(sid)
            self.assertEqual((status["status"], status["error"]), ("error", {"code": code, "message": MESSAGES[code]}))
            self.assertEqual((status["revision"], status["bundle_url"], status["summary"]), (None, None, None))
            self.assertEqual(running.call("GET", f"/api/session/{sid}/bundle").status, 409)
        # Only clips that passed the header check reach pose, each exactly once, and nothing is sharpened.
        self.assertEqual(backend.log, [("preview", sid, None) for sid in ids[2:]])
        listing = {s["id"]: s for s in running.get("/api/sessions")[1]["sessions"]}
        for sid, (_, _, code) in zip(ids, cases):
            self.assertEqual((listing[sid]["status"], listing[sid]["error"], listing[sid]["analysis_s"]),
                             ("error", MESSAGES[code], None))

    def test_a_clip_being_analysed_shows_its_status_and_video_but_nothing_that_needs_the_report(self):
        backend = self.backend()
        running = self.start(backend)
        backend.hold("preview")
        sid = running.upload(self.long, "long.mp4")
        backend.wait_calls(1)
        held = running.status(sid)
        self.assertEqual((held["status"], held["detail"], held["duration_s"], held["long_clip"], held["media_url"]),
                         ("preview", "Analysing the clip", 40.0, True, f"/media/{sid}"))
        self.assertEqual({key: held[key] for key in ("revision", "version", "bundle_url", "tier", "n_frames", "fps",
                                                       "summary", "error", "windows")},
                         dict.fromkeys(("revision", "version", "bundle_url", "tier", "n_frames", "fps", "summary",
                                        "error", "windows")))
        for path, body in (("/api/refine", {"all": True}), ("/api/chat", {"text": "how?"}), ("/api/export", {}),
                           ("/api/plan", {"player": "near"})):
            self.assertEqual(running.post(path, {"session": sid, **body}),
                             (409, {"error": "this clip has not been analysed yet"}), path)
        for path in (f"/api/export/{sid}", f"/api/session/{sid}/bundle", f"/download/{sid}"):
            self.assertEqual(running.call("GET", path).status, 409, path)
        self.assertEqual(running.call("GET", f"/media/{sid}").body, self.long)
        self.assertEqual(running.post("/api/labels", {"session": sid, "near": "Ana", "far": "Bea"}),
                         (200, {"labels": {"near": "Ana", "far": "Bea"}}))

    def test_deleting_a_clip_mid_analysis_leaves_no_files_and_the_queue_moves_on(self):
        backend = self.backend()
        running = self.start(backend)
        backend.hold("preview")
        gone = running.upload(self.rally, "gone.mp4")
        backend.wait_calls(1)
        deleted = running.cli("delete", gone)
        self.assertEqual((deleted.returncode, deleted.stdout.strip()), (0, "Deleted."), deleted.stderr)
        self.assertEqual(running.call("GET", f"/api/session/{gone}").status, 404)
        self.assertFalse(running.folder(gone).exists())

        backend.open("preview")
        kept = running.upload(self.rally, "kept.mp4")
        self.assertEqual(running.ready(kept, revision=1)["sharpened"], WINDOWS)
        self.assertFalse(running.folder(gone).exists())
        self.assertEqual([s["id"] for s in running.get("/api/sessions")[1]["sessions"]], [kept])
        again = running.cli("delete", gone)
        self.assertEqual((again.returncode, again.stdout.strip()), (1, "There is no session with that id."))
        self.assertEqual(running.cli("delete", "../" + kept).returncode, 2)
        self.assertTrue(running.folder(kept).exists())

    def test_a_clip_deleted_while_the_analyser_reads_it_is_removed_once_the_analyser_lets_go(self):
        backend = self.backend()
        backend.keep_open = True
        running = self.start(backend)
        for kind, calls in (("preview", 1), ("refine", 3)):
            backend.hold(kind)
            sid = running.upload(self.rally, f"{kind}.mp4")
            backend.wait_calls(calls)
            deleted = running.cli("delete", sid)
            if os.name == "nt":
                # Windows keeps a file that is open somewhere else, so the delete can only remove the rest.
                self.assertEqual(deleted.returncode, 3, deleted.stdout)
                self.assertIn("The session is deleted, but some of its files are still in use", deleted.stderr)
                self.assertTrue(running.folder(sid).exists())
            else:
                self.assertEqual(deleted.returncode, 0, deleted.stderr)
            self.assertEqual(running.call("GET", f"/api/session/{sid}").status, 404, kind)
            backend.open(kind)
            until(lambda: not running.folder(sid).exists())
        self.assertEqual(running.get("/api/sessions")[1]["sessions"], [])

    def test_clips_wait_for_the_next_start_when_the_analyser_cannot_start(self):
        data = self.data_dir()

        def broken():
            raise RuntimeError("the model weights are missing")

        running = self.start(make_backend=broken, data_dir=data)
        self.assertEqual(running.app.wait_until_warm(30)["state"], "failed")
        health = running.get("/api/health")[1]
        self.assertEqual((health["warm"]["state"], health["backend"]), ("failed", None))
        sid = running.upload(self.rally)
        waiting = running.wait_for(sid, lambda s: s["detail"] == WARM_FAILED)
        self.assertEqual((waiting["status"], waiting["revision"], waiting["error"]), ("queued", None, None))
        self.assertEqual(running.post("/api/refine", {"session": sid, "all": True})[0], 409)
        running.close()

        backend = self.backend()
        running = self.start(backend, data_dir=data, auto_refine=False)
        self.assertEqual((running.ready(sid)["status"], backend.log), ("ready", [("preview", sid, None)]))

    def test_sharpening_says_to_wait_or_restart_while_the_analyser_is_not_running(self):
        data = self.data_dir()
        running = self.start(self.backend(), data_dir=data, auto_refine=False)
        sid = running.upload(self.rally)
        running.ready(sid)
        running.close()

        let_start = threading.Event()

        def slow():
            if not let_start.wait(30):
                raise RuntimeError("the test never let the analyser start")
            return self.backend()

        running = self.start(make_backend=slow, data_dir=data)
        self.assertEqual(running.post("/api/refine", {"session": sid, "all": True}),
                         (503, {"error": "the analyser is still starting"}))
        let_start.set()
        self.assertEqual(running.app.wait_until_warm(30)["state"], "ready")
        running.close()

        def broken():
            raise RuntimeError("the model weights are missing")

        running = self.start(make_backend=broken, data_dir=data)
        self.assertEqual(running.app.wait_until_warm(30)["state"], "failed")
        self.assertEqual(running.post("/api/refine", {"session": sid, "all": True}),
                         (503, {"error": "the analyser could not start; restart the app"}))

    def test_status_long_polls_until_the_session_changes(self):
        running = self.start(self.backend(), auto_refine=False)
        sid = running.upload(self.rally)
        stamp = running.ready(sid)["stamp"]
        started = time.monotonic()
        code, same = running.get(f"/api/session/{sid}?since={stamp}&wait=0.3")
        self.assertEqual((code, same["stamp"]), (200, stamp))
        self.assertGreaterEqual(time.monotonic() - started, 0.25)

        woken: dict = {}
        poll = threading.Thread(target=lambda: woken.update(running.get(f"/api/session/{sid}?since={stamp}&wait=15")[1]))
        started = time.monotonic()
        poll.start()
        time.sleep(0.2)
        self.assertEqual(running.post("/api/refine", {"session": sid, "all": True})[0], 202)
        poll.join(20)
        self.assertLess(time.monotonic() - started, 10)
        self.assertGreater(woken["stamp"], stamp)

        stamp = running.ready(sid, revision=1)["stamp"]
        for query in ("wait=abc", "since=-1", "since=abc", "since=1e3", f"since={stamp}&wait=1,5"):
            self.assertEqual(running.call("GET", f"/api/session/{sid}?{query}").status, 400, query)
        for wait in ("nan", "-5"):
            started = time.monotonic()
            self.assertEqual(running.get(f"/api/session/{sid}?since={stamp}&wait={wait}")[1]["stamp"], stamp)
            self.assertLess(time.monotonic() - started, 1.0, wait)


class RouteTest(ServerCase):
    def test_requests_are_checked_before_anything_is_stored(self):
        running = self.start(self.backend())
        upload = lambda headers, body=None: running.raw("POST", "/api/upload", headers, body).status
        self.assertEqual(upload([("X-Filename", "a.mp4"), ("Content-Length", "0")]), 400)
        self.assertEqual(upload([("X-Filename", "notes.txt"), ("Content-Length", "5")], b"hello"), 415)
        self.assertEqual(upload([("Content-Length", "5")], b"hello"), 400)
        self.assertEqual(upload([("X-Filename", "%00%01"), ("Content-Length", "5")], b"hello"), 400)
        self.assertEqual(upload([("X-Filename", "a.mp4"), ("Content-Length", str(3 * 2 ** 30))]), 413)
        self.assertEqual(upload([("X-Filename", "a.mp4"), ("Content-Length", "1" * 19)]), 400)
        self.assertEqual(upload([("X-Filename", "a.mp4"), ("Transfer-Encoding", "chunked")]), 411)
        self.assertEqual(running.raw("POST", "/api/labels", [("Content-Type", "application/json")]).status, 411)
        self.assertEqual(running.get("/api/sessions"), (200, {"sessions": []}))
        self.assertEqual(list((running.data_dir / "sessions").glob("*")), [])

        self.assertEqual(running.call("GET", "/api/nothing").status, 404)
        self.assertEqual(running.call("GET", "/API/health").status, 404)
        self.assertEqual(running.call("DELETE", "/api/sessions").status, 405)
        self.assertEqual(running.call("GET", "/api/upload").status, 405)
        self.assertEqual(running.call("POST", "/api/health", b"{}").status, 405)
        self.assertEqual(running.call("GET", "/api/sessions?" + "&".join(f"k{i}=1" for i in range(21))).status, 400)
        self.assertEqual(running.call("GET", "/api/sessions?" + "&".join(f"k{i}=1" for i in range(20))).status, 200)

        post = lambda path, body: running.call("POST", path, body, {"Content-Type": "application/json"})
        for body in (b"{bad", b"[]", b"null", b'"text"', b"\xff", b"[" * 60000):
            self.assertEqual(post("/api/labels", body).status, 400, body[:8])
        self.assertEqual(post("/api/labels", b" " * (64 * 1024 + 1)).status, 413)
        self.assertEqual(post("/api/chat", b" " * (10 * 1024 + 1)).status, 413)
        self.assertEqual(post("/api/labels", b'{"session": "0123456789abcdef"}').status, 404)

    def test_only_this_page_at_this_address_can_use_the_api(self):
        running = self.start(self.backend())
        health = lambda headers=(), host=True: running.raw("GET", "/api/health", headers, host=host).status
        self.assertEqual(health(host=False), 403)
        self.assertEqual(health([("Host", "evil.example")], host=False), 403)
        self.assertEqual(health([("Host", f"127.0.0.1:{running.port + 1}")], host=False), 403)
        self.assertEqual(health([("Host", f"localhost:{running.port}")], host=False), 200)
        self.assertEqual(health([("Origin", "http://evil.example")]), 200)
        body = [("Content-Type", "application/json"), ("Content-Length", "2")]
        for origin in ("http://evil.example", f"http://127.0.0.1:{running.port + 1}", "null"):
            self.assertEqual(running.raw("POST", "/api/labels", [("Origin", origin)] + body, b"{}").status, 403, origin)
        for origin in (f"http://localhost:{running.port}", f"http://127.0.0.1:{running.port}"):
            # Past the origin check, an empty request names no session.
            self.assertEqual(running.raw("POST", "/api/labels", [("Origin", origin)] + body, b"{}").status, 404, origin)
        page = running.call("GET", "/static/" + next(WEB.glob("*.css")).name)
        self.assertEqual((page.status, page.headers["Content-Type"], page.headers["X-Content-Type-Options"],
                          page.headers["Cache-Control"]), (200, "text/css; charset=utf-8", "nosniff", "no-store"))

    def test_videos_are_served_whole_or_as_one_byte_range(self):
        running = self.start(self.backend(), auto_refine=False)
        sid = running.upload(self.rally)
        size = len(self.rally)
        whole = running.raw("GET", f"/media/{sid}")
        self.assertEqual((whole.status, whole.headers["Accept-Ranges"], whole.headers["Content-Type"], whole.body),
                         (200, "bytes", "video/mp4", self.rally))
        for header, start, end in (("bytes=0-9", 0, 9), ("bytes=10-", 10, size - 1), ("bytes=-5", size - 5, size - 1),
                                   (f"bytes=5-{size + 100}", 5, size - 1), (f"bytes=-{size + 9}", 0, size - 1),
                                   (f"bytes={size - 1}-{size - 1}", size - 1, size - 1)):
            reply = running.raw("GET", f"/media/{sid}", [("Range", header)])
            self.assertEqual((reply.status, reply.headers["Content-Range"], reply.body),
                             (206, f"bytes {start}-{end}/{size}", self.rally[start:end + 1]), header)
        for header in ("bytes=-0", f"bytes={size}-", "bytes=9-3", "bytes=abc", "items=0-9", "bytes=0-1,4-5", "bytes=-"):
            reply = running.raw("GET", f"/media/{sid}", [("Range", header)])
            self.assertEqual((reply.status, reply.headers["Content-Range"]), (416, f"bytes */{size}"), header)

    def test_paths_outside_the_served_folders_are_not_found(self):
        running = self.start(self.backend(), auto_refine=False)
        sid = running.upload(self.rally)
        running.ready(sid)
        for path in ("/static/..%2Fapp%2Fserver.py", "/static/..\\app\\server.py", "/static/../app/server.py",
                     "/static/%2e%2e%2fapp.html", "/static/.hidden.css", "/static/server.py", "/static/",
                     "/gifs/..%2F..%2Fschema.py", "/gifs/../library.json", "/gifs/", f"/media/{sid}/../../coach.db",
                     f"/media/{sid.upper()}", "/media/..%2Fcoach.db", f"/download/{sid}/..", "/download/coach.db",
                     f"/api/session/{sid}/bundle/..", "/api/plan/..%2F..%2Fcoach.db", "/static/app.html%00.css"):
            self.assertEqual(running.call("GET", path).status, 404, path)
        self.assertEqual(running.call("GET", f"/api/session/{sid}/bundle?revision=..%2F0").status, 400)

    def test_the_session_page_gets_every_file_it_names(self):
        running = self.start(self.backend(), auto_refine=False)
        page = running.call("GET", "/").body.decode("utf-8")
        named = re.findall(r'(?:href|src)="(/static/[^"]+)"', page)
        self.assertEqual(sorted(named), ["/static/app.css", "/static/app.js", "/static/viewer.css"])
        for url in named:
            self.assertEqual(running.call("GET", url).status, 200, url)
        sheet = running.call("GET", "/static/viewer.css")
        self.assertEqual((sheet.headers["Content-Type"], sheet.body), ("text/css; charset=utf-8",
                                                                        (WEB / "viewer.css").read_bytes()))

    def test_exercise_cards_point_at_gifs_that_are_served(self):
        running = self.start(self.backend())
        code, body = running.get("/api/exercises")
        self.assertEqual(code, 200)
        urls = [card["gif_url"] for card in body["exercises"] if card["gif_url"]]
        self.assertTrue(urls)
        for url in urls:
            reply = running.call("GET", url)
            self.assertEqual((reply.status, reply.headers["Content-Type"], reply.body[:4]), (200, "image/gif", b"GIF8"),
                             url)

    def test_exercise_cards_carry_the_easier_dose_the_coach_gives_after_too_hard(self):
        def doses(body):
            return {card["id"]: (card["sets_reps"], card["easier_dose"]) for card in body["exercises"]}

        planned = doses(self.start(self.backend(), auto_refine=False, plans=PlanMaker()).get("/api/exercises")[1])
        self.assertEqual(planned["sit_low_hold"], ("3 x 30 seconds", "2 x 15 seconds"))
        self.assertEqual(planned["goblet_squat"], ("3 x 8", "2 x 4"))
        self.assertEqual(planned["side_plank"], ("3 x 20 seconds each side", "2 x 10 seconds each side"))
        unplanned = doses(self.start(self.backend(), auto_refine=False).get("/api/exercises")[1])
        self.assertEqual(unplanned["sit_low_hold"], ("3 x 30 seconds", None))

    def test_health_gives_the_report_words_for_every_warning_code(self):
        running = self.start(self.backend(), auto_refine=False)
        words = running.get("/api/health")[1]["warnings"]
        self.assertEqual(words, dict(WARNINGS))
        self.assertTrue({"preview_estimate", "unresolved_recovery", "few_swings", "repeated_frames"} <= set(words))

    def test_a_bundled_sample_is_analysed_by_name_only(self):
        running = self.start(self.backend(), samples_dir=self.samples, auto_refine=False)
        self.assertEqual(running.get("/api/health")[1]["samples"], ["01-test"])
        code, made = running.post("/api/sample", {"name": "01-test"})
        self.assertEqual(code, 201)
        done = running.ready(made["id"])
        self.assertEqual((done["video_name"], done["timings"]["first"]["upload_s"]), ("01-test.mp4", None))
        self.assertEqual((running.folder(made["id"]) / "video.mp4").read_bytes(), self.rally)
        for name in ("missing", "../01-test", "01-test.mp4", 5, {}, None):
            self.assertEqual(running.post("/api/sample", {"name": name})[0], 404, name)

    def test_names_and_labels_come_back_exactly_as_typed(self):
        running = self.start(self.backend(), auto_refine=False)
        tricky = "<img src=x onerror=alert(1)> '; DROP TABLE sessions;--.mp4"
        sid = running.upload(self.rally, "C:\\Users\\me\\Videos\\" + tricky)
        self.assertEqual(running.status(sid)["video_name"], tricky)
        cut = running.status(running.upload(self.rally, "a" * 200 + ".MP4"))["video_name"]
        self.assertEqual((len(cut), cut[-5:]), (120, "….MP4"))
        labels = {"near": "<b>Ana</b>", "far": "Robert'); DROP TABLE plans;--"}
        self.assertEqual(running.post("/api/labels", {"session": sid, **labels}), (200, {"labels": labels}))
        self.assertEqual(running.status(sid)["labels"], labels)
        self.assertEqual(running.get("/api/sessions")[1]["sessions"][-1]["video_name"], tricky)

    def test_named_players_have_progress_across_their_sessions(self):
        running = self.start(self.backend(), auto_refine=False)
        first = running.upload(self.rally, "monday.mp4")
        running.ready(first)
        self.assertEqual(running.get("/api/players"), (200, {"players": []}))
        for query in ("player=Near%20player", "", "player=" + "x" * 41):
            self.assertEqual(running.call("GET", f"/api/progress?{query}").status, 400, query)
        second = running.upload(self.rally, "tuesday.mp4")
        running.ready(second)
        for sid in (first, second):
            self.assertEqual(running.post("/api/labels", {"session": sid, "near": "Ana", "far": "Bea"})[0], 200)
        self.assertEqual(running.get("/api/players"), (200, {"players": ["Ana", "Bea"]}))

        code, ana = running.get("/api/progress?player=Ana")
        self.assertEqual((code, ana["player"]), (200, "Ana"))
        self.assertTrue(ana["metrics"])
        for metric in ana["metrics"]:
            self.assertEqual([(p["session_id"], p["player"]) for p in metric["points"]],
                             [(first, "near"), (second, "near")], metric["name"])
            self.assertEqual(metric["trend"]["word"], "same", metric["name"])
        name = ana["metrics"][0]["name"]
        self.assertEqual([m["name"] for m in running.get(f"/api/progress?player=Ana&metric={name}")[1]["metrics"]],
                         [name])

        self.assertEqual(running.post("/api/labels", {"session": second, "near": "Bea", "far": "Ana"})[0], 200)
        for metric in running.get("/api/progress?player=Ana")[1]["metrics"]:
            self.assertEqual([p["player"] for p in metric["points"]], ["near", "far"], metric["name"])
        self.assertEqual(running.get("/api/progress?player=Cy"), (200, {"player": "Cy", "metrics": []}))

        for near, far in (("Ana", "ana"), ("", "Bea"), (5, "Bea"), ("x" * 41, "Bea"), (None, "Bea")):
            self.assertEqual(running.post("/api/labels", {"session": first, "near": near, "far": far})[0], 400, near)
        self.assertEqual(running.post("/api/labels", {"session": "0123456789abcdef", "near": "A", "far": "B"})[0], 404)


class ProviderTest(ServerCase):
    def analysed(self, running: Running, count: int) -> list[str]:
        ids = [running.upload(self.rally, f"clip{i}.mp4") for i in range(count)]
        for sid in ids:
            running.ready(sid)
        return ids

    @staticmethod
    def in_background(running: Running, path: str, payload: Any) -> tuple[threading.Thread, list]:
        """Posts from another thread, the way another tab does. Its (code, body) is put in the list when it ends."""
        answer: list = []
        thread = threading.Thread(target=lambda: answer.append(running.post(path, payload)), daemon=True)
        thread.start()
        return thread, answer

    def test_chat_gets_its_history_and_only_same_player_earlier_reports(self):
        chat = FakeChat()
        running = self.start(self.backend(), auto_refine=False, chat=chat)
        s1, s2, s3, s4, s5 = self.analysed(running, 5)
        for sid, near, far in ((s2, "Ana", "Bea"), (s3, "Bea", "Ana"), (s4, "Ana", "Cy")):
            self.assertEqual(running.post("/api/labels", {"session": sid, "near": near, "far": far})[0], 200)

        def ask(sid, text, player=None):
            context = None if player is None else {"player": player}
            code, body = running.post("/api/chat", {"session": sid, "text": text, "context": context})
            self.assertEqual(code, 200, body)
            return body, chat.calls[-1]

        body, call = ask(s4, "  how do I lunge lower?  ", "near")
        self.assertEqual(body, {"reply": {"text": "you asked: how do I lunge lower?"},
                                "context": {"player": "near", "turns": 2}})
        self.assertEqual((call["key"], call["session"], call["history"], call["earlier"]), ((s4, 0), s4, [], [s2]))
        body, call = ask(s4, "<script>alert(1)</script>", "near")
        self.assertEqual(body["reply"]["text"], "you asked: <script>alert(1)</script>")
        self.assertEqual(call["history"], [Turn("user", "how do I lunge lower?"),
                                           Turn("coach", "you asked: how do I lunge lower?")])
        self.assertEqual(ask(s4, "and the far player?", "far")[1]["earlier"], [])
        self.assertEqual(ask(s3, "has Ana improved?", "far")[1]["earlier"], [])
        self.assertEqual(ask(s5, "and me?")[1]["earlier"], [])
        self.assertEqual(ask(s1, "who is this?", "both")[1]["context"], {"player": "both"})

    def test_chat_questions_are_checked_before_the_coach_sees_them(self):
        chat = FakeChat()
        running = self.start(self.backend(), auto_refine=False, chat=chat)
        (sid,) = self.analysed(running, 1)
        cases = [({"text": "x" * 501}, 413), ({"text": "   "}, 400), ({}, 400), ({"text": 5}, 400),
                 ({"text": "how?", "context": []}, 400), ({"text": "how?", "context": "near"}, 400)]
        for body, code in cases:
            self.assertEqual(running.post("/api/chat", {"session": sid, **body})[0], code, body)
        self.assertEqual(running.post("/api/chat", {"session": sid, "text": "x" * 500})[0], 200)
        self.assertEqual(len(chat.calls), 1)

    def test_a_stop_for_pain_holds_for_the_whole_conversation(self):
        chat = FakeChat()
        running = self.start(self.backend(), auto_refine=False, chat=chat)
        sid, fresh = self.analysed(running, 2)

        def ask(session, text, context):
            code, body = running.post("/api/chat", {"session": session, "text": text, "context": context})
            self.assertEqual(code, 200, body)
            return chat.calls[-1]

        def stopped(session):
            return running.get(f"/api/session/{session}")[1]["chat_stopped"]

        self.assertEqual(ask(sid, "how do I lunge lower?", {"player": "near"})["context"], {"player": "near"})
        self.assertEqual(stopped(sid), False)
        self.assertEqual(ask(sid, "my knee hurts", {"player": "near"})["context"], {"player": "near"})
        self.assertEqual(stopped(sid), True)
        # Seven more exchanges push the stop out of the 12 turns the coach is shown.
        for i in range(7):
            call = ask(sid, f"question {i}", {"player": "near", "pain_stop": False})
            self.assertEqual(call["context"], {"player": "near", "pain_stop": True})
        self.assertNotIn(Turn("coach", FAKE_STOP), call["history"])
        self.assertEqual(len(call["history"]), 12)
        # A page reloaded in a new tab sends no context at all.
        self.assertEqual(ask(sid, "make me a practice plan", None)["context"], {"pain_stop": True})
        self.assertEqual(ask(sid, "yes", {"player": "far", "last_offer": "practice_plan"})["context"],
                         {"player": "far", "last_offer": "practice_plan", "pain_stop": True})
        said = running.store.turns(sid, 1000)
        self.assertEqual(len(said), 22)
        self.assertEqual(said[3], ("coach", FAKE_STOP))

        # Another clip's conversation is not stopped, and a flag the page kept is passed on unchanged.
        self.assertEqual(ask(fresh, "how do I lunge lower?", None)["context"], {})
        self.assertEqual(ask(fresh, "and then?", {"player": "near", "pain_stop": True})["context"],
                         {"player": "near", "pain_stop": True})
        self.assertEqual(ask(fresh, "and after that?", {"player": "near", "pain_stop": False})["context"],
                         {"player": "near", "pain_stop": False})
        # Only the coach's own stop is recorded, never a flag a page sends.
        self.assertEqual((stopped(sid), stopped(fresh)), (True, False))

    def test_after_a_stop_for_pain_no_new_plan_is_made_and_saved_plans_stay(self):
        chat, plans = FakeChat(), FakePlans()
        running = self.start(self.backend(), auto_refine=False, chat=chat, plans=plans)
        sid, fresh = self.analysed(running, 2)
        code, saved = running.post("/api/plan", {"session": sid, "player": "near"})
        self.assertEqual(code, 201)
        self.assertEqual(running.post("/api/chat", {"session": sid, "text": "my knee hurts", "context": None})[0], 200)

        before = snapshot(running.data_dir)
        for body in ({"player": "near"}, {"player": "far", "weeks": 3, "level": 1, "equipment": ["band"]}):
            self.assertEqual(running.post("/api/plan", {"session": sid, **body}), (409, {"error": server.PLAN_STOPPED}))
        self.assertEqual(snapshot(running.data_dir), before)
        self.assertEqual(len(plans.calls), 1)
        self.assertEqual(running.get(f"/api/plan/{saved['id']}")[1]["plan"], saved["plan"])
        self.assertEqual(running.get("/api/exercises")[0], 200)

        self.assertEqual(running.post("/api/plan", {"session": fresh, "player": "near"})[0], 201)
        self.assertEqual(plans.calls[1][0], fresh)

    def test_without_the_coach_chat_no_plan_is_made_even_for_a_stopped_conversation(self):
        plans = FakePlans()
        running = self.start(self.backend(), auto_refine=False, plans=plans)
        sid, fresh = self.analysed(running, 2)
        # The chat stopped this conversation for pain before it went missing.
        running.store.add_turns(sid, [("user", "my knee hurts"), ("coach", FAKE_STOP)])
        health = running.get("/api/health")[1]
        self.assertEqual((health["chat"], health["plan"]), (False, False))

        before = snapshot(running.data_dir)
        for session in (sid, fresh):
            self.assertEqual(running.post("/api/plan", {"session": session, "player": "near"}),
                             (503, {"error": "the practice plan is not installed yet"}))
        self.assertEqual(snapshot(running.data_dir), before)
        self.assertEqual(plans.calls, [])
        self.assertEqual(running.get("/api/exercises")[0], 200)

    def test_a_reply_being_made_when_another_tab_stops_the_chat_is_never_saved_or_shown(self):
        chat = HeldChat(StoppingChat(), "give me a drill")
        running = self.start(self.backend(), auto_refine=False, chat=chat)
        sid, fresh = self.analysed(running, 2)
        self.assertEqual(running.post("/api/chat", {"session": sid, "text": "how do I lunge lower?", "context": None}),
                         (200, {"reply": {"text": "a drill for: how do I lunge lower?"}, "context": {}}))

        # The first tab's question has passed the route's check for a stop and is being answered.
        asking, answer = self.in_background(running, "/api/chat",
                                            {"session": sid, "text": "give me a drill", "context": {"player": "near"}})
        try:
            self.assertTrue(chat.entered.wait(30))
            # Meanwhile another tab's pain message gets the stop, which is recorded first.
            self.assertEqual(running.post("/api/chat", {"session": sid, "text": "my knee hurts", "context": None}),
                             (200, {"reply": {"text": FAKE_STOP}, "context": {"pain_stop": True}}))
        finally:
            chat.gate.set()
            asking.join(30)

        self.assertEqual(answer, [(200, {"reply": {"text": FAKE_STOP}, "context": {"player": "near", "pain_stop": True}})])
        self.assertEqual(running.store.turns(sid, 1000), [
            ("user", "how do I lunge lower?"), ("coach", "a drill for: how do I lunge lower?"),
            ("user", "my knee hurts"), ("coach", FAKE_STOP),
            ("user", "give me a drill"), ("coach", FAKE_STOP)])
        # Another clip's conversation is not stopped.
        self.assertEqual(running.post("/api/chat", {"session": fresh, "text": "give me a drill", "context": None}),
                         (200, {"reply": {"text": "a drill for: give me a drill"}, "context": {}}))
        self.assertEqual([running.get(f"/api/session/{s}")[1]["chat_stopped"] for s in (sid, fresh)], [True, False])

    def test_a_plan_being_made_when_another_tab_stops_the_chat_is_refused_and_never_saved(self):
        plans = HeldPlans()
        running = self.start(self.backend(), auto_refine=False, chat=FakeChat(), plans=plans)
        made, nothing, fresh = self.analysed(running, 3)
        code, saved = running.post("/api/plan", {"session": made, "player": "near"})
        self.assertEqual(code, 201)

        # Level 1 is where this planner finds nothing: that answer is refused too, rather than told to the player.
        for sid, level in ((made, 3), (nothing, 1)):
            with self.subTest(level=level):
                plans.gate.clear()
                plans.entered.clear()
                # The first tab's plan has passed the route's check for a stop and is being made.
                asking, answer = self.in_background(running, "/api/plan",
                                                    {"session": sid, "player": "far", "level": level})
                try:
                    self.assertTrue(plans.entered.wait(30))
                    # Meanwhile another tab's pain message gets the stop, which is recorded first.
                    self.assertEqual(running.post("/api/chat", {"session": sid, "text": "my knee hurts",
                                                                "context": None})[0], 200)
                    before = snapshot(running.data_dir)
                finally:
                    plans.gate.set()
                    asking.join(30)
                self.assertEqual(answer, [(409, {"error": server.PLAN_STOPPED})])
                self.assertEqual(snapshot(running.data_dir), before)

        self.assertEqual(running.get(f"/api/plan/{saved['id']}")[1]["plan"], saved["plan"])
        self.assertEqual(running.store.turns(made, 1000), [("user", "my knee hurts"), ("coach", FAKE_STOP)])
        self.assertEqual(running.post("/api/plan", {"session": fresh, "player": "near"})[0], 201)

    def test_a_practice_plan_is_saved_and_can_be_opened_again(self):
        plans = FakePlans()
        running = self.start(self.backend(), auto_refine=False, chat=FakeChat(), plans=plans)
        (sid,) = self.analysed(running, 1)
        running.post("/api/labels", {"session": sid, "near": "Ana", "far": "Bea"})
        code, made = running.post("/api/plan", {"session": sid, "player": "far", "weeks": 3, "level": 1,
                                                "equipment": ["band", "mat", "band"]})
        self.assertEqual(code, 201)
        self.assertEqual((made["url"], made["plan"]), (f"/plan?id={made['id']}",
                                                       {"player": "far", "weeks": 3, "level": 1, "equipment": ["band", "mat"]}))
        self.assertEqual(plans.calls, [(sid, "far", 3, 1, ("band", "mat"), ())])
        self.assertEqual(running.post("/api/plan", {"session": sid, "player": "near"})[0], 201)
        self.assertEqual(running.post("/api/plan", {"session": sid, "player": "near", "equipment": []})[0], 201)
        self.assertEqual(running.post("/api/plan", {"session": sid, "player": "near",
                                                    "avoid": ["wall_sit", "band_row", "wall_sit"]})[0], 201)
        self.assertEqual(plans.calls[1:], [(sid, "near", 2, 3, None, ()), (sid, "near", 2, 3, (), ()),
                                           (sid, "near", 2, 3, None, ("wall_sit", "band_row"))])

        code, saved = running.get(f"/api/plan/{made['id']}")
        self.assertEqual((code, saved["session_id"], saved["player"], saved["label"], saved["video_name"],
                          saved["plan"]), (200, sid, "far", "Bea", "clip0.mp4", made["plan"]))
        for body in ({}, {"player": "both"}, {"player": "near", "weeks": 0}, {"player": "near", "weeks": 5},
                     {"player": "near", "weeks": "2"}, {"player": "near", "level": 4}, {"player": "near", "level": True},
                     {"player": "near", "equipment": "band"}, {"player": "near", "equipment": [{}]},
                     {"player": "near", "equipment": ["band"] * 10}, {"player": "near", "equipment": ["sofa"]},
                     {"player": "near", "avoid": "wall_sit"}, {"player": "near", "avoid": None},
                     {"player": "near", "avoid": [3]}, {"player": "near", "avoid": [""]},
                     {"player": "near", "avoid": ["x" * 81]}, {"player": "near", "avoid": ["x"] * 65}):
            self.assertEqual(running.post("/api/plan", {"session": sid, **body})[0], 400, body)
        self.assertEqual(running.call("GET", "/api/plan/0123456789abcdef").status, 404)
        self.assertEqual(len(plans.calls), 4)

    def test_the_real_coach_and_plan_answer_through_the_routes(self):
        chat, plans = CoachChat(), PlanMaker()
        built = []
        make = chat.coach
        chat.coach = lambda report, *rest: built.append((report.video_id, report.revision)) or make(report, *rest)
        running = self.start(self.backend(), auto_refine=False, chat=chat, plans=plans)
        (sid,) = self.analysed(running, 1)
        safety = json.loads((PROJECT / "coach" / "kb" / "safety.json").read_text("utf-8"))["text"]

        def ask(text, context):
            return running.post("/api/chat", {"session": sid, "text": text, "context": context})

        code, first = ask("What should I work on first?", {"player": "near"})
        self.assertEqual((code, first["reply"]["intent"], first["context"]["player"]), (200, "summary", "near"))
        roles = running.get(f"/api/session/{sid}/bundle?revision=0")[1]["players"]
        self.assertTrue(all(a["kind"] == "finding" and roles[a["arg"]] == "near" for a in first["reply"]["actions"]),
                        first["reply"]["actions"])
        code, made = running.post("/api/plan", {"session": sid, "player": "far", "level": 1, "equipment": []})
        self.assertEqual((code, made["id"], made["url"], made["message"], made["plan"]["weeks"]),
                         (200, None, None, NO_PLAN, []))
        code, hurt = ask("my knee has been painful since the lunges", first["context"])
        self.assertEqual((code, hurt["reply"]["text"], hurt["reply"]["intent"], hurt["reply"]["actions"],
                          hurt["reply"]["suggestions"]), (200, safety, "safety", [], []))
        for context in ({"player": "middle"}, {"level": 9}, {"finding_id": "technique.not_a_finding"}):
            self.assertEqual(ask("how?", context), (400, {"error": "the chat lost its place; reload the page"}))
        self.assertEqual(len(running.store.turns(sid)), 4)
        self.assertEqual(built, [(sid, 0)])

        def blocks(plan):
            return [b for week in plan["weeks"] for session in week["sessions"] for b in session["blocks"]]

        report = full_report()
        full = plans(report, Role.NEAR, 2, 3, None, ())
        hard = blocks(full)[0]
        eased = plans(report, Role.NEAR, 2, 3, None, (hard["exercise_id"],))
        kept = [b for b in blocks(eased) if b["exercise_id"] == hard["exercise_id"]]
        self.assertTrue(all(b["reason"].endswith(EASED) and b["dose"] != hard["dose"] for b in kept), kept)
        self.assertNotEqual(blocks(eased), blocks(full))

    def test_the_real_coach_keeps_a_stop_for_pain_through_a_long_chat_and_a_reload(self):
        chat, plans = CoachChat(), PlanMaker()
        running = self.start(self.backend(), auto_refine=False, chat=chat, plans=plans)
        sid, fresh = self.analysed(running, 2)
        stop = json.loads((PROJECT / "coach" / "kb" / "safety.json").read_text("utf-8"))["text"]
        files = [running.folder(sid) / "r0" / name for name in ("pose.json", "report.json", "bundle.json")]
        digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in files]
        sent = []
        offered = {}

        def ask(session, text, context):
            code, body = running.post("/api/chat", {"session": session, "text": text, "context": context})
            self.assertEqual(code, 200, body)
            sent.append((session, text))
            return body["reply"], body["context"]

        def blocked(text, context, intent):
            reply, context = ask(sid, text, context)
            self.assertEqual((reply["text"], reply["intent"], reply["actions"], reply["suggestions"],
                              context["last_offer"], context["pain_stop"]), (stop, intent, [], [], None, True), text)

        def answered(text, context):
            reply, context = ask(sid, text, context)
            offer = (context["last_offer"] or "").partition("@")[0].partition(":")[0]
            self.assertEqual((reply["text"] == stop, "exercise" in [a["kind"] for a in reply["actions"]],
                              offer in ("", "show_best", "show_moment"), context["pain_stop"]),
                             (False, False, True, True), (text, reply, context))
            for chip in reply["suggestions"]:
                offered.setdefault(chip, context)
            return context

        def push_the_stop_out(context):
            # Six answered exchanges fill the 12 turns the coach is shown, so the stop is no longer among them.
            for text in ("What did I do well?", "What can't you measure?", "Show me the moment") * 2:
                context = answered(text, context)
            self.assertNotIn(("coach", stop), running.store.turns(sid, server.HISTORY_TURNS))

        blocked("my knee has been painful since the lunges", {"player": "near"}, "safety")
        push_the_stop_out({"player": "near", "pain_stop": True})
        # A page reloaded in a new tab sends no context, and a page may send the flag as false.
        blocked("give me a drill", None, "exercise_request")
        push_the_stop_out({"player": "near"})
        blocked("make me a practice plan", {"player": "near", "pain_stop": False}, "practice_plan")
        push_the_stop_out({"player": "near"})
        blocked("yes", {"player": "near", "pain_stop": False, "last_offer": "practice_plan"}, "affirm")
        # Every chip the coach offered after the stop gets an answer, not the stop again.
        self.assertTrue(offered)
        for chip, context in list(offered.items()):
            answered(chip, context)

        before = snapshot(running.data_dir)
        self.assertEqual(running.post("/api/plan", {"session": sid, "player": "near"}),
                         (409, {"error": server.PLAN_STOPPED}))
        self.assertEqual(snapshot(running.data_dir), before)
        said = running.store.turns(sid, 1000)
        self.assertEqual(said[:2], [("user", "my knee has been painful since the lunges"), ("coach", stop)])
        self.assertEqual([text for speaker, text in said if speaker == "user"], [text for _, text in sent])
        self.assertEqual(len(said), 2 * len(sent))
        self.assertEqual([hashlib.sha256(path.read_bytes()).hexdigest() for path in files], digests)

        # No stop in the other clip, but descriptive observations do not prescribe drills.
        reply, context = ask(fresh, "give me a drill", {"player": "near"})
        self.assertEqual(("exercise" in [a["kind"] for a in reply["actions"]], context["pain_stop"]), (False, False),
                         reply)
        code, made = running.post("/api/plan", {"session": fresh, "player": "near"})
        self.assertEqual((code, made["id"], made["message"]), (200, None, NO_PLAN))
        # A page opened without its saved chat learns of the stop from the session's status.
        self.assertEqual([running.get(f"/api/session/{s}")[1]["chat_stopped"] for s in (sid, fresh)], [True, False])

    def test_the_real_coach_answers_a_question_caught_by_another_tab_s_stop_with_the_stop(self):
        chat = HeldChat(CoachChat(), "give me a drill")
        running = self.start(self.backend(), auto_refine=False, chat=chat, plans=PlanMaker())
        sid, fresh = self.analysed(running, 2)
        stop = json.loads((PROJECT / "coach" / "kb" / "safety.json").read_text("utf-8"))["text"]
        pain = "my knee has been painful since the lunges"

        # The first tab's question has passed the route's check for a stop and is being answered.
        asking, answer = self.in_background(running, "/api/chat",
                                            {"session": sid, "text": "give me a drill", "context": {"player": "near"}})
        try:
            self.assertTrue(chat.entered.wait(30))
            # Meanwhile another tab's pain message gets the stop, which is recorded first.
            code, hurt = running.post("/api/chat", {"session": sid, "text": pain, "context": {"player": "near"}})
            self.assertEqual((code, hurt["reply"]["text"], hurt["context"]["pain_stop"]), (200, stop, True))
        finally:
            chat.gate.set()
            asking.join(30)

        self.assertEqual(len(answer), 1, "the first tab got no answer")
        code, body = answer[0]
        reply, context = body["reply"], body["context"]
        self.assertEqual((code, reply["text"], reply["intent"], reply["actions"], reply["suggestions"],
                          context["last_offer"], context["pain_stop"]), (200, stop, "exercise_request", [], [], None, True))
        self.assertEqual(running.store.turns(sid, 1000),
                         [("user", pain), ("coach", stop), ("user", "give me a drill"), ("coach", stop)])
        # Another clip has no safety stop and no supported corrective drill.
        code, body = running.post("/api/chat", {"session": fresh, "text": "give me a drill", "context": {"player": "near"}})
        self.assertEqual((code, "exercise" in [a["kind"] for a in body["reply"]["actions"]], body["context"]["pain_stop"]),
                         (200, False, False), body)

    def test_missing_providers_say_so_and_are_never_faked(self):
        running = self.start(self.backend(), auto_refine=False)
        (sid,) = self.analysed(running, 1)
        health = running.get("/api/health")[1]
        self.assertEqual((health["chat"], health["plan"], health["export"]), (False, False, False))
        self.assertEqual(running.post("/api/chat", {"session": sid, "text": "how?"}),
                         (503, {"error": "the coach chat is not installed yet"}))
        self.assertEqual(running.post("/api/plan", {"session": sid, "player": "near"}),
                         (503, {"error": "the practice plan is not installed yet"}))
        self.assertEqual(running.post("/api/export", {"session": sid}),
                         (503, {"error": "the review video maker is not installed yet"}))
        self.assertEqual(running.store.turns(sid), [])

    def test_the_review_video_is_made_once_per_revision_and_can_be_downloaded(self):
        exporter = FakeExporter()
        running = self.start(self.backend(), auto_refine=False, exporter=exporter)
        sid, other = self.analysed(running, 2)
        self.assertEqual(running.get(f"/api/export/{sid}"), (200, {"status": "none", "message": ""}))
        self.assertEqual(running.call("GET", f"/download/{sid}").status, 404)

        exporter.gate.clear()
        self.assertEqual(running.post("/api/export", {"session": sid}), (202, {"status": "running", "message": "", "seconds": 0}))
        self.assertEqual(running.post("/api/export", {"session": sid})[1]["status"], "running")
        until(lambda: exporter.calls)
        self.assertEqual(running.get(f"/api/export/{sid}")[1]["status"], "running")
        until(lambda: running.get(f"/api/export/{sid}")[1]["seconds"] >= 1, timeout=10)
        exporter.gate.set()
        state = until(lambda: (lambda s: s if s["status"] == "ready" else None)(running.get(f"/api/export/{sid}")[1]))
        self.assertGreaterEqual(state.pop("seconds"), 1)
        self.assertEqual(state, {"status": "ready", "message": "", "url": f"/download/{sid}?revision=0"})
        self.assertEqual(exporter.calls, [(sid, 0, "review.part.mp4")])
        reply = running.call("GET", state["url"])
        self.assertEqual((reply.status, reply.headers["Content-Type"], reply.headers["Content-Disposition"], reply.body),
                         (200, "video/mp4", 'attachment; filename="review.mp4"', b"review:" + sid.encode()))
        self.assertEqual(running.raw("GET", state["url"], [("Range", "bytes=0-6")]).body, b"review:")
        self.assertEqual(sorted(p.name for p in (running.folder(sid) / "r0").iterdir()),
                         ["bundle.json", "pose.json", "report.json", "review.mp4"])
        for query, code in (("revision=5", 404), ("revision=x", 400), ("revision=-1", 400)):
            self.assertEqual(running.call("GET", f"/download/{sid}?{query}").status, code, query)

        exporter.fail.add(other)
        running.post("/api/export", {"session": other})
        failed = until(lambda: (lambda s: s if s["status"] == "error" else None)(running.get(f"/api/export/{other}")[1]))
        self.assertEqual(failed, {"status": "error", "message": "the review video could not be made"})
        self.assertEqual(sorted(p.name for p in (running.folder(other) / "r0").iterdir()),
                         ["bundle.json", "pose.json", "report.json"])
        self.assertEqual(running.call("GET", f"/download/{other}").status, 404)
        self.assertEqual(running.post("/api/export", {"session": other}), (202, {"status": "running", "message": "", "seconds": 0}))
        until(lambda: running.get(f"/api/export/{other}")[1]["status"] == "ready")
        self.assertEqual([call for call in exporter.calls if call[0] == other],
                         [(other, 0, "review.part.mp4")] * 2)


    def test_complete_export_downloads_the_matching_analysis_and_evidence(self):
        class WithAnalysis(FakeExporter):
            def __call__(self, video, doc, report, part):
                super().__call__(video, doc, report, part)
                for name, body in (("analysis.html", "<h1>matching analysis</h1>"),
                                   ("analysis.json", '{"same_findings":true}'),
                                   ("narration.txt", "matching narration")):
                    (part.parent / name).write_text(body, "utf-8")

        running = self.start(self.backend(), auto_refine=False, exporter=WithAnalysis())
        (sid,) = self.analysed(running, 1)
        self.assertEqual(running.call("GET", f"/download/{sid}/analysis.html").status, 404)
        running.post("/api/export", {"session": sid})
        state = until(lambda: (lambda s: s if s["status"] == "ready" else None)(
            running.get(f"/api/export/{sid}")[1]))
        self.assertEqual(state["analysis_url"], f"/download/{sid}/analysis.html?revision=0")
        self.assertEqual(state["evidence_url"], f"/download/{sid}/analysis.json?revision=0")
        for name in ("analysis.html", "analysis.json", "narration.txt"):
            reply = running.call("GET", f"/download/{sid}/{name}")
            self.assertEqual(reply.status, 200)
            self.assertEqual(reply.headers["Content-Disposition"], f'attachment; filename="{name}"')
            self.assertEqual(reply.body, (running.folder(sid) / "r0" / name).read_bytes())
        self.assertEqual(running.call("GET", f"/download/{sid}/analysis.html?revision=1").status, 404)


APP_DATA = Path(__file__).with_name("app_data")
# The bundle.json files the app saved beside the Replay reports in app_data, before unseen readings were left empty.
REPLAY_BUNDLE_SHA256 = {
    0: "d1a2bf6711a1547e99f2ad4c4bf4aace2fb5062ef63d7679d7f3c35398a6adb9",
    1: "8132dc4a6ad4ebd9c39f9641b88c7925210e0854dbc1767d14334dd856604168",
}
# The practice-guide midpoint those bundles hold where a reading was not taken.
MIDPOINTS = {"contact_reach": 1.2, "contact_elbow": 167.5, "swing_hand_speed": 12.0, "ready_hip_ratio": 0.45,
             "ready_knee_flexion": 35.0, "ready_stance_width": 0.525, "ready_trunk_lean": 6.0}
UNREADABLE = {"error": "this revision's saved data could not be read"}


def replay_bundle(revision: int) -> bytes:
    data = (APP_DATA / f"replay-{REPLAY}-r{revision}-bundle.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != REPLAY_BUNDLE_SHA256[revision]:
        raise AssertionError(f"coach/tests/app_data: the Replay r{revision} bundle changed")
    return data


def readings(bundle: dict, role: str, name: str) -> list:
    return [m["value"] for f in bundle["findings"] if f["player"] == role for m in f["metrics"] if m["name"] == name]


def as_missing(bundle: dict, untaken: set) -> dict:
    """A copy of `bundle` with the (player, metric) readings in `untaken` left empty."""
    copied = copy.deepcopy(bundle)
    for finding in copied["findings"]:
        for metric in finding["metrics"]:
            if (finding["player"], metric["name"]) in untaken:
                metric["value"] = None
    return copied


def differences(served: Any, expected: Any, limit: int = 10) -> list[str]:
    """Up to `limit` places where two JSON values differ. A short failure message: for two unequal bundles,
    unittest's own message is a difflib diff of both pretty-printed in full, which does not finish in useful time."""
    found: list[str] = []

    def walk(a: Any, b: Any, where: str) -> None:
        if len(found) >= limit:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(a.keys() | b.keys(), key=str):
                if key in a and key in b:
                    walk(a[key], b[key], f"{where}.{key}")
                else:
                    found.append(f"{where}.{key}: only {'expected' if key in b else 'served'}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                found.append(f"{where}: {len(a)} items served, {len(b)} expected")
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{where}[{i}]")
        elif a != b:
            found.append(f"{where}: served {a!r:.60}, expected {b!r:.60}")

    walk(served, expected, "bundle")
    return found[:limit]


class AvailabilityTest(ServerCase):
    """Readings the analysis did not take: what the bundle and progress routes show, and a start-up repair that fails."""

    def seed(self, running: Running, revision: int, bundle: bytes) -> Path:
        """Save the Replay report and `bundle` as REPLAY's newest revision, the way a finished analysis leaves them."""
        folder = running.folder(REPLAY) / f"r{revision}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "report.json").write_bytes(replay_bytes(revision))
        (folder / "bundle.json").write_bytes(bundle)
        running.store.commit_revision(REPLAY, replay_report(revision), {"total": 0.5})
        return folder / "bundle.json"

    def replay(self) -> Running:
        running = self.start(self.backend(), auto_refine=False)
        running.store.add_session(REPLAY, "replay.mp4", "ab" * 32)
        return running

    def assert_served(self, running: Running, path: str, revision: int) -> dict:
        saved = json.loads(replay_bundle(revision))
        reply = running.call("GET", path)
        self.assertEqual((reply.status, reply.headers["Content-Type"]), (200, "application/json; charset=utf-8"))
        body = reply.json()
        self.assertEqual(differences(body, as_missing(saved, UNTAKEN[revision])), [])
        self.assertEqual(body["schema_version"], 2)
        for role, name in UNTAKEN[revision]:
            self.assertEqual((readings(saved, role, name), readings(body, role, name)), ([MIDPOINTS[name]], [None]),
                             (role, name))
        return body

    def test_a_saved_bundle_shows_untaken_readings_as_missing_in_each_revision(self):
        running = self.replay()
        url = f"/api/session/{REPLAY}/bundle"
        paths = {0: self.seed(running, 0, replay_bundle(0))}
        running.store.set_labels(REPLAY, "Ana", "Ben")

        ana = running.get("/api/progress?player=Ana")[1]["metrics"]
        self.assertEqual([m["name"] for m in ana], [
            "half_coverage", "lunge_depth", "lunge_launch_m", "ready_hip_ratio", "ready_knee_flexion",
            "ready_stance_width", "ready_trunk_lean", "steps_before_lunge", "weaker_side_share"])
        hip = next(m for m in ana if m["name"] == "ready_hip_ratio")
        self.assertEqual([(p["session_id"], p["player"], p["value"], p["confidence"]) for p in hip["points"]],
                         [(REPLAY, "near", 0.5509120690586571, 0.55)])
        self.assertEqual(hip["trend"], {"word": "first", "text": "Nothing to compare it with yet"})
        body = self.assert_served(running, url, 0)
        self.assertEqual(readings(body, "near", "ready_hip_ratio"), [0.5509120690586571])

        paths[1] = self.seed(running, 1, replay_bundle(1))
        self.assertEqual(running.get("/api/progress?player=Ana&metric=ready_hip_ratio"),
                         (200, {"player": "Ana", "metrics": []}))
        self.assertEqual([m["name"] for m in running.get("/api/progress?player=Ana")[1]["metrics"]], [
            "exchange_gap_s", "first_step_s", "fixed_base_gap_m", "habitual_base_gap_m", "half_coverage",
            "lunge_depth", "lunge_launch_m", "net_ahead_share", "split_step_share", "steps_before_lunge",
            "swing_hand_speed", "weaker_side_share"])
        ben = running.get("/api/progress?player=Ben&metric=ready_hip_ratio")[1]["metrics"]
        self.assertEqual([(p["session_id"], p["player"], p["value"]) for p in ben[0]["points"]],
                         [(REPLAY, "far", 0.5195009686061087)])
        for served, revision in ((url, 1), (f"{url}?revision=1", 1), (f"{url}?revision=0", 0)):
            self.assert_served(running, served, revision)
        self.assertEqual(readings(self.assert_served(running, url, 1), "far", "ready_hip_ratio"), [0.5195009686061087])
        for revision, path in paths.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), REPLAY_BUNDLE_SHA256[revision])

    def test_a_bundle_whose_unseen_readings_are_already_empty_is_sent_as_saved(self):
        running = self.replay()
        path = self.seed(running, 0, replay_bundle(0))
        for version in (2, 3):
            cleared = as_missing(json.loads(replay_bundle(0)), UNTAKEN[0])
            cleared["schema_version"] = version
            saved = json.dumps(cleared, indent=1).encode("utf-8")
            path.write_bytes(saved)
            reply = running.call("GET", f"/api/session/{REPLAY}/bundle")
            self.assertEqual((reply.status, len(reply.body), hashlib.sha256(reply.body).hexdigest()),
                             (200, len(saved), hashlib.sha256(saved).hexdigest()), version)

    def test_a_bundle_that_breaks_the_reading_rule_or_is_not_json_is_an_error_not_a_guess(self):
        running = self.replay()
        path = self.seed(running, 0, replay_bundle(0))
        saved = json.loads(replay_bundle(0))
        current = as_missing(saved, UNTAKEN[0])
        current["schema_version"] = 3

        def near_reach(data: dict) -> dict:
            return next(m for f in data["findings"] if f["player"] == "near" for m in f["metrics"]
                        if m["name"] == "contact_reach")

        def changed(edit, data: dict = saved) -> bytes:
            data = copy.deepcopy(data)
            edit(data)
            return json.dumps(data).encode("utf-8")

        broken = {
            "unseen but sure": changed(lambda d: near_reach(d).update(confidence=0.3)),
            "version 3 with a midpoint": changed(lambda d: near_reach(d).update(value=1.2), current),
            "no version": changed(lambda d: d.pop("schema_version")),
            "version as text": changed(lambda d: d.update(schema_version="2")),
            "cut short": replay_bundle(0)[:4096],
            "not UTF-8": b"\xff" + replay_bundle(0)[1:],
        }
        for problem, payload in broken.items():
            with self.subTest(problem):
                path.write_bytes(payload)
                reply = running.call("GET", f"/api/session/{REPLAY}/bundle")
                self.assertEqual(reply.status, 500)
                self.assertEqual(reply.json(), UNREADABLE)
                self.assertEqual(path.read_bytes(), payload)

    def test_progress_compares_only_readings_the_analysis_was_sure_of(self):
        running = self.start(self.backend(), auto_refine=False)
        sessions: list[str] = []

        def analysed(hip: float, confidence: float) -> str:
            """A session where Ana, the near player, has this hip-height reading and a knee reading of confidence 0."""
            n = len(sessions) + 1
            report = replay_report(0)
            near = [replace(m, value=hip, confidence=confidence) if m.name == "ready_hip_ratio"
                    else replace(m, confidence=0.0) if m.name == "ready_knee_flexion" else m
                    for m in report.metrics["near"]]
            sid = f"a{n:015x}"
            running.store.add_session(sid, f"day{n}.mp4", f"{n:064x}")
            running.store.commit_revision(sid, replace(report, metrics={**report.metrics, "near": near}), {"total": 0.5})
            running.store.set_labels(sid, "Ana", "Ben")
            sessions.append(sid)
            return sid

        first, unsure, third = analysed(0.60, 0.55), analysed(0.45, 0.0), analysed(0.55, 0.55)
        self.assertEqual([(r["session_id"], r["value"], r["confidence"])
                          for r in running.store.progress("Ana", "ready_hip_ratio")],
                         [(first, 0.6, 0.55), (unsure, 0.45, 0.0), (third, 0.55, 0.55)])
        ana = running.get("/api/progress?player=Ana")[1]["metrics"]
        self.assertEqual([m["name"] for m in ana], [
            "half_coverage", "lunge_depth", "lunge_launch_m", "ready_hip_ratio", "ready_stance_width",
            "ready_trunk_lean", "steps_before_lunge", "weaker_side_share"])
        hip = next(m for m in ana if m["name"] == "ready_hip_ratio")
        self.assertEqual([(p["session_id"], p["value"]) for p in hip["points"]], [(first, 0.6), (third, 0.55)])
        self.assertEqual(hip["trend"], {"word": "lower", "text": "Reading is lower; this is not a technique score"})
        self.assertEqual(len(running.store.progress("Ana", "ready_knee_flexion")), 3)
        self.assertEqual(running.get("/api/progress?player=Ana&metric=ready_knee_flexion"),
                         (200, {"player": "Ana", "metrics": []}))

        fourth = analysed(0.58, 0.55)
        hip = running.get("/api/progress?player=Ana&metric=ready_hip_ratio")[1]["metrics"][0]
        self.assertEqual([(p["session_id"], p["value"]) for p in hip["points"]],
                         [(first, 0.6), (third, 0.55), (fourth, 0.58)])
        self.assertEqual(hip["trend"], {"word": "higher", "text": "Reading is higher; this is not a technique score"})

    def test_a_data_folder_that_cannot_be_repaired_does_not_start_and_says_how_to_go_on(self):
        data = self.data_dir()
        legacy_folder(data)
        lost = data / "sessions" / UNSURE / "r0" / "report.json"
        lost.unlink()
        legacy = snapshot(data)

        def cli(*args: str) -> subprocess.CompletedProcess:
            return subprocess.run([sys.executable, "-m", "coach.app", *args, "--data-dir", str(data)],
                                  cwd=PROJECT, capture_output=True, text=True, timeout=180)

        started = cli("--port", "0", "--no-browser", "--backend", "replay")
        self.assertEqual(started.returncode, 3, started.stderr)
        self.assertIn(
            f"The saved measures could not be rebuilt. Session {UNSURE}, revision 0: its report.json is missing. "
            f"Nothing was changed. Put back {lost}, or delete that session with: "
            f'python -m coach.app delete {UNSURE} --data-dir "{data}"', started.stderr.splitlines())
        self.assertNotIn("Badminton coach:", started.stdout)
        self.assertEqual(snapshot(data), legacy)

        deleted = cli("delete", UNSURE)
        self.assertEqual((deleted.returncode, deleted.stdout), (0, "Deleted.\n"), deleted.stderr)
        self.assertFalse((data / "sessions" / UNSURE).exists())
        store = Store(data)
        try:
            self.assertEqual(snapshot(data)[0], 2)
            self.assertIsNone(store.get(UNSURE))
            self.assertEqual(len(metric_rows(data, REPLAY)), 28)
            self.assertEqual(store.get(REPLAY).labels, {"near": "Ana", "far": "Ben"})
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
