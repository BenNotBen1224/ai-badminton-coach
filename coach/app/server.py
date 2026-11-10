"""The local web app: a stdlib HTTP server on 127.0.0.1, a one-at-a-time analysis queue and the session store.

Each session lives in <data>/sessions/<id>/: the uploaded video.<ext> and one folder per revision, r<N>/, with
pose.json, report.json and bundle.json (and review.mp4 once exported). A revision folder is complete before the
store commits it and is never written again, so a page that asks for revision N gets the report and the bundle
of revision N together. The status names its revision and the bundle URL carries it.

One worker thread runs every analysis. A new clip's first analysis goes ahead of queued sharpening, and a
running sharpening stops between windows while a new clip waits. The chat, plan and review-video providers are
passed in; a missing one answers 503 with a plain message instead of a made-up result.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, unquote, urlsplit

from coach.bundle import BundleContractError, prepare_bundle
from coach.catalog import METRIC_CATALOG
from coach.exercises.library import EQUIPMENT
from coach.exercises.library import ROOT as EXERCISES_ROOT
from coach.schema import WARNINGS, Role, Turn, report_from_dict, to_plain
from coach.severity import distance

from .backend import SAMPLES
from .pipeline import MESSAGES, TESTED_SECONDS, AnalysisError, analyze_video, library, probe, revise
from .store import DEFAULT_LABELS, SessionRow, Store, check_id, new_id

log = logging.getLogger("coach.app")

HOST = "127.0.0.1"
WEB = Path(__file__).resolve().parents[1] / "web"
GIFS = EXERCISES_ROOT / "gifs"
MAX_UPLOAD_BYTES = 2 * 1024 ** 3
MAX_JSON_BYTES = 64 * 1024
MAX_CHAT_BYTES = 10 * 1024
MAX_CHAT_CHARS = 500
MAX_AVOID = 64
MAX_ID_CHARS = 80
MAX_DRAIN_BYTES = 1 << 20
CHUNK = 1 << 20
MAX_WAIT_S = 20.0
REPORT_CACHE = 8
HISTORY_TURNS = 12
HISTORY_REPORTS = 5
PLAN_STOPPED = "The coach stopped for your safety in this chat, so it will not make a practice plan."
MAX_WINDOW_S = 15.0
# Measured on clip03 (36 s, key moments over 83% of the frames): one pass over every frame did about a sixth of the
# analyser work of 17 window calls and held a new upload for less time than the slowest of those calls.
WHOLE_CLIP_SHARE = 0.5
# "Sharpen all" with Turbo on clip03 (36 s, 1080 frames) took 10.5 s from the request until the sharpened result was
# saved (11.3 s when it started by itself; one run each on a shared computer). The page quotes it before the click
# only for that analyser and for clips no longer and no bigger than that one.
SHARPEN_ALL_S = 11
SHARPEN_ALL_FRAMES = 1080
# A change smaller than this many tolerance units of a metric counts as "about the same".
SAME_BAND = 0.25
VIDEO_TYPES = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"}
WEB_TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml",
    ".png": "image/png", ".json": "application/json; charset=utf-8",
}
PAGES = {"/": "app.html", "/plan": "plan.html", "/progress": "progress.html"}
FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")
DIGITS = re.compile(r"\d{1,18}")
RANGE = re.compile(r"bytes=(\d{0,18})-(\d{0,18})")
CSP = (
    "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; script-src 'self'; style-src 'self';"
    " connect-src 'self'; worker-src 'self' blob:; object-src 'none'; base-uri 'none'; frame-ancestors 'none';"
    " form-action 'none'"
)
STATUS_WORDS = {400: "the request was not understood", 414: "the address is too long", 501: "this server does not do that"}
SHARPENING_STOPPED = "Sharpening stopped, so this is still the quick analysis"
WARM_FAILED = "The analyser could not start. Restart the app to analyse this clip."
# Names nobody typed. Clips under these names may be different people, so they are not compared over time.
UNNAMED = frozenset(DEFAULT_LABELS.values())


class HttpError(Exception):
    def __init__(self, status: int, message: str, headers: tuple = ()):
        super().__init__(message)
        self.status = status
        self.message = message
        self.headers = headers


def bad_range(size: int) -> HttpError:
    return HttpError(416, "that part of the file does not exist", (("Content-Range", f"bytes */{size}"),))


@dataclass
class Job:
    """What the worker is doing for one session. Memory only; the store holds every finished result."""

    status: str
    detail: str
    windows_done: int = 0
    windows_total: int = 0
    notice: str = ""


@dataclass
class Refinement:
    """Sharpening of one session: inclusive windows still to do, or the whole clip with everything=True."""

    session_id: str
    windows: list[tuple[int, int]]
    everything: bool = False
    total: int = 0
    done: int = 0
    doc: Optional[dict] = None
    pose_s: float = 0.0
    calls: list[dict] = field(default_factory=list)


def whole_clip(windows: list[tuple[int, int]], n_frames: int, duration_s: Optional[float]) -> bool:
    """True when automatic sharpening should measure every frame in one pass instead of one call per key moment:
    the clip is no longer than the tested length and its key moments cover at least WHOLE_CLIP_SHARE of it.
    That pass cannot pause for a new upload, so longer clips keep the window calls."""
    covered = len({frame for first, last in windows for frame in range(first, last + 1)})
    return duration_s is not None and duration_s <= TESTED_SECONDS and covered >= WHOLE_CLIP_SHARE * n_frames


def sharpen_all_estimate(backend_name: Optional[str], n_frames: Optional[int],
                         duration_s: Optional[float]) -> Optional[int]:
    """The seconds the page may quote before "Sharpen all", or None when nothing was measured for this case."""
    if backend_name != "turbo" or not n_frames or duration_s is None:
        return None
    return SHARPEN_ALL_S if duration_s <= TESTED_SECONDS and n_frames <= SHARPEN_ALL_FRAMES else None


def clean_filename(raw: Optional[str]) -> str:
    """The display name from the X-Filename header (percent-encoded by the page), without any folder part."""
    if raw is None:
        raise HttpError(400, "send the file name in the X-Filename header")
    name = unquote(raw, errors="replace").replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    if not name:
        raise HttpError(400, "the file needs a name")
    if len(name) > 120:
        suffix = Path(name).suffix[:10]
        name = name[: 120 - len(suffix) - 1] + "\u2026" + suffix
    return name


def copy_with_hash(source: Path, target: Path) -> str:
    digest = hashlib.sha256()
    with open(source, "rb") as src, open(target, "xb") as out:
        for block in iter(lambda: src.read(CHUNK), b""):
            digest.update(block)
            out.write(block)
    return digest.hexdigest()


def clip_seconds(video: Path) -> Optional[float]:
    probed = probe(video)
    return None if isinstance(probed, AnalysisError) else probed[0] / probed[1]


def windows_done(doc: Optional[dict]) -> list[tuple[int, int]]:
    """The inclusive windows a pose document says were measured after its preview, in the order they were done."""
    return [(int(w["start"]), int(w["end"])) for w in (doc or {}).get("windows_refined") or ()]


def _optional(factory: Callable[[], Any], what: str) -> Any:
    """factory() or None when one of the coach's own modules is not installed yet. Every other error is raised."""
    try:
        return factory()
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("coach."):
            raise
        log.warning("%s is not available: %s is missing", what, exc.name)
        return None


class CoachChat:
    """Unit C4's coach behind the chat route. Building a Coach takes up to 0.1 s, so one is kept per revision."""

    def __init__(self):
        from coach.chat.engine import Coach, context_from_dict, context_to_dict
        from coach.chat.kb import load_kb

        self.coach, self.from_dict, self.to_dict = Coach, context_from_dict, context_to_dict
        self.kb = load_kb()
        # The coach's stop for pain, exactly as the chat table keeps it.
        self.stop_texts = (self.kb.safety["text"],)
        self._coaches: OrderedDict[tuple[str, int], Any] = OrderedDict()
        self._lock = threading.Lock()

    def _coach_for(self, key: tuple[str, int], report):
        with self._lock:
            coach = self._coaches.get(key)
            if coach is not None:
                self._coaches.move_to_end(key)
                return coach
        coach = self.coach(report, library(), self.kb)
        with self._lock:
            self._coaches[key] = coach
            while len(self._coaches) > REPORT_CACHE:
                self._coaches.popitem(last=False)
        return coach

    def __call__(self, key, report, history, text, context, history_reports):
        try:
            ctx = self.from_dict(context, history_reports=tuple(history_reports))
        except ValueError:
            raise HttpError(400, "the chat lost its place; reload the page") from None
        reply, ctx = self._coach_for(key, report).reply(history, text, ctx)
        return to_plain(reply), self.to_dict(ctx)


class PlanMaker:
    """Unit C4's practice plan behind the plan route."""

    def __init__(self):
        from coach.chat.plan import NO_PLAN, build_plan, easier_dose

        self.build_plan, self.no_plan, self.easier_dose = build_plan, NO_PLAN, easier_dose

    def __call__(self, report, player, weeks, level, equipment, avoid=()):
        try:
            plan = self.build_plan(report, library(), player=player, weeks=weeks, level=level,
                                   equipment=equipment, avoid=avoid)
        except ValueError:
            log.exception("the practice plan refused its options")
            raise HttpError(400, "a plan cannot be made with these choices") from None
        out = plan.to_dict()
        if not out["weeks"]:
            out["message"] = self.no_plan
        return out


class ReviewExporter:
    """Continuous review and matching report behind the export route."""

    def __init__(self):
        from coach.complete_review import export_complete

        self.export_review = export_complete

    def __call__(self, video, doc, report, out_mp4):
        import os
        import uuid
        from turbo.accept import acquire_cpu_lock, release_cpu_lock
        from .backend import PROJECT

        lock = PROJECT / "work" / "cpu.lock"
        token = f"app-review {os.getpid()} {uuid.uuid4().hex}"
        acquire_cpu_lock(lock, token)
        try:
            self.export_review(video, doc, report, out_mp4)
        finally:
            release_cpu_lock(lock, token)


def real_providers() -> dict[str, Any]:
    return {
        "chat": _optional(CoachChat, "the coach chat"),
        "plans": _optional(PlanMaker, "the practice plan"),
        "exporter": _optional(ReviewExporter, "the review video"),
    }


class App:
    """Sessions, the analysis worker and the providers behind the HTTP routes."""

    def __init__(self, store: Store, make_backend: Callable[[], Any], *, auto_refine: bool = True,
                 chat=None, plans=None, exporter=None, samples_dir: Path = SAMPLES):
        self.store = store
        self.make_backend = make_backend
        self.backend = None
        self.auto_refine = auto_refine
        self.chat = chat
        self.plans = plans
        self.exporter = exporter
        self.samples = {p.stem: p for p in sorted(Path(samples_dir).glob("*.mp4"))}
        self.warm: dict[str, Any] = {"state": "warming", "seconds": None}
        self._warm_done: Optional[float] = None
        self._cv = threading.Condition()
        self._previews: deque[str] = deque()
        self._refinements: deque[Refinement] = deque()
        self._jobs: dict[str, Job] = {}
        self._stamps: dict[str, int] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        self._reports: OrderedDict[tuple[str, int], Any] = OrderedDict()
        # Per session: (revision, inclusive windows that revision's pose document already measured).
        self._sharpened: dict[str, tuple[int, list[tuple[int, int]]]] = {}
        self._exports: dict[tuple[str, int], dict[str, str]] = {}
        self._export_lock = threading.Lock()
        self._stopping = False
        self._threads: list[threading.Thread] = []
        self._exercises: Optional[bytes] = None

    # lifecycle

    def start(self) -> None:
        for row in reversed(self.store.sessions()):
            if row.revision < 0 and row.error is None:
                video = self.video_path(row.id)
                if video is not None:
                    self._enqueue(row.id, clip_seconds(video), upload_s=None)
        for name, target in (("warm-up", self._warm_up), ("worker", self._work)):
            thread = threading.Thread(target=target, name=f"coach-{name}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        with self._cv:
            self._stopping = True
            self._cv.notify_all()
        for thread in self._threads:
            thread.join(timeout=30)

    def _warm_up(self) -> None:
        start = time.perf_counter()
        backend, state = None, {"state": "ready"}
        try:
            backend = self.make_backend()
            backend.warm_up()
        except Exception:
            log.exception("the analyser could not start")
            state = {"state": "failed", "message": "the analyser could not start"}
        with self._cv:
            self.backend = backend
            self.warm = {**state, "seconds": time.perf_counter() - start}
            self._warm_done = time.perf_counter()
            self._cv.notify_all()

    def wait_until_warm(self, timeout: Optional[float] = None) -> dict[str, Any]:
        with self._cv:
            self._cv.wait_for(lambda: self.warm["state"] != "warming" or self._stopping, timeout)
            return dict(self.warm)

    # queue

    def add(self, session_id: str, video_name: str, digest: str, video: Path, upload_s: Optional[float]) -> None:
        duration = clip_seconds(video)
        self.store.add_session(session_id, video_name, digest)
        self._enqueue(session_id, duration, upload_s)

    def _enqueue(self, session_id: str, duration: Optional[float], upload_s: Optional[float]) -> None:
        with self._cv:
            self._meta[session_id] = {"upload_s": upload_s, "enqueued": time.perf_counter(), "duration_s": duration}
            self._previews.append(session_id)
            self._set(session_id, "queued", "Waiting to start")

    def _set(self, session_id: str, status: str, detail: str, **extra) -> None:
        with self._cv:
            self._jobs[session_id] = Job(status, detail, **extra)
            self._stamps[session_id] = self._stamps.get(session_id, 0) + 1
            self._cv.notify_all()

    def _work(self) -> None:
        with self._cv:
            while self.warm["state"] == "warming" and not self._stopping:
                self._cv.wait()
        while True:
            with self._cv:
                while not (self._previews or self._refinements or self._stopping):
                    self._cv.wait()
                if self._stopping:
                    return
                if self._previews:
                    task, item = self._preview, self._previews.popleft()
                else:
                    task, item = self._refine, self._refinements.popleft()
            try:
                task(item)
            except Exception:
                if isinstance(item, str):
                    log.exception("analysing session %s failed", item)
                    self._fail_session(item)
                else:
                    log.exception("sharpening session %s failed", item.session_id)
                    self._settle(item.session_id, notice=SHARPENING_STOPPED)

    def _fail_session(self, session_id: str) -> None:
        with self._cv:
            try:
                if self.store.get(session_id) is not None:
                    self.store.fail(session_id, "failed", {})
            finally:
                self._set(session_id, "error", MESSAGES["failed"])

    def _preview(self, session_id: str) -> None:
        row = self.store.get(session_id)
        video = self.video_path(session_id)
        if row is None or row.revision >= 0 or video is None:
            return
        if self.backend is None:
            # Kept as queued, not failed: the next start of the app analyses it.
            self._set(session_id, "queued", WARM_FAILED)
            return
        meta = self._meta.get(session_id, {})
        started = time.perf_counter()
        enqueued = meta.get("enqueued", started)
        warm_wait = max(0.0, min(self._warm_done or enqueued, started) - enqueued)
        self._set(session_id, "preview", "Analysing the clip")
        session = analyze_video(video, self.backend, self._revision_dir(session_id, 0), session_id)
        timings = dict(session.timings)
        timings.update({
            "backend": self.backend.name,
            "upload_s": meta.get("upload_s"),
            "queue_s": started - enqueued - warm_wait,
            "warm_wait_s": warm_wait,
        })
        if self._gone(session_id):
            return
        # The store row and the job change together, so a status read never sees one without the other.
        with self._cv:
            if session.error is not None:
                self.store.fail(session_id, session.error.code, {"first": timings})
                self._set(session_id, "error", session.error.message)
                return
            self.store.commit_revision(session_id, session.report, {"first": timings})
            self._remember(session_id, 0, session.report)
            self._sharpened[session_id] = (0, windows_done(session.doc))
            windows = self._auto_windows(session.report, session.doc)
            if windows and whole_clip(windows, len(session.doc["frames"]), session.duration_s):
                self._refinements.append(Refinement(session_id, [], everything=True))
                self._set(session_id, "refining", "Waiting to measure every frame")
            elif windows:
                self._refinements.append(Refinement(session_id, windows, total=len(windows)))
                self._set(session_id, "refining", "Waiting to sharpen key moments", windows_total=len(windows))
            else:
                self._set(session_id, "ready", "Ready")

    def _auto_windows(self, report, doc) -> list[tuple[int, int]]:
        if not (self.auto_refine and getattr(self.backend, "sharpens_windows", False)):
            return []
        if report.tier.value != "preview":
            return []
        done = set(windows_done(doc))
        return [(w.start_frame, w.end_frame) for w in report.windows if (w.start_frame, w.end_frame) not in done]

    def _refine(self, job: Refinement) -> None:
        session_id = job.session_id
        row = self.store.get(session_id)
        video = self.video_path(session_id)
        if row is None or row.revision < 0 or video is None or row.tier == "refined":
            # A job queued before every frame was measured has nothing left to do.
            self._settle(session_id)
            return
        revision = row.revision + 1
        try:
            if job.doc is None and not job.everything:
                job.doc = self.pose(session_id, row.revision)
                done = set(windows_done(job.doc))
                job.windows = [pair for pair in job.windows if pair not in done]
                job.total = job.done + len(job.windows)
                if not job.windows:
                    self._settle(session_id)
                    return
            if job.everything:
                self._set(session_id, "refining", "Measuring every frame")
                out = self._revision_dir(session_id, revision)
                start = time.perf_counter()
                job.doc, stages = self.backend.refined(video, out / "pose.json")
                job.pose_s += time.perf_counter() - start
                job.calls.append(stages)
            if job.windows:
                self._set(session_id, "refining", "Sharpening key moments", windows_done=job.done,
                          windows_total=job.total)
            while job.windows:
                first, last = job.windows[0]
                out = self._revision_dir(session_id, revision) if len(job.windows) == 1 else None
                start = time.perf_counter()
                job.doc, stages = self.backend.refine(video, job.doc, [(first, last)],
                                                      None if out is None else out / "pose.json")
                job.pose_s += time.perf_counter() - start
                job.calls.append(stages)
                job.windows.pop(0)
                job.done += 1
                if job.windows and self._preview_waiting():
                    with self._cv:
                        self._refinements.appendleft(job)
                        self._set(session_id, "refining", "Paused to analyse a new clip first",
                                  windows_done=job.done, windows_total=job.total)
                    return
                self._set(session_id, "refining", "Sharpening key moments", windows_done=job.done,
                          windows_total=job.total)
        except Exception:
            if self._gone(session_id):
                return
            log.exception("sharpening session %s failed", session_id)
            self._settle(session_id, notice=SHARPENING_STOPPED)
            return
        stages = {"backend": self.backend.name, "calls": job.calls}
        out = self.store.session_dir(session_id) / f"r{revision}"
        session = revise(job.doc, stages, job.pose_s, session_id, revision, out)
        if self._gone(session_id):
            return
        if session.error is not None:
            self._settle(session_id, notice=SHARPENING_STOPPED)
            return
        latest = {**session.timings, "backend": self.backend.name, "kind": "all" if job.everything else "windows",
                  "windows": job.done}
        with self._cv:
            if self.store.get(session_id) is None:
                return
            self.store.commit_revision(session_id, session.report,
                                       {"first": row.timings.get("first"), "latest": latest})
            self._remember(session_id, revision, session.report)
            self._sharpened[session_id] = (revision, windows_done(job.doc))
            self._settle(session_id)

    def _settle(self, session_id: str, notice: str = "") -> None:
        """ready, or refining while more sharpening of this session waits in the queue."""
        with self._cv:
            pending = [job for job in self._refinements if job.session_id == session_id]
            if pending:
                self._set(session_id, "refining", "Waiting to sharpen", notice=notice,
                          windows_done=pending[0].done, windows_total=pending[0].total)
            else:
                self._set(session_id, "ready", "Ready", notice=notice)

    def _preview_waiting(self) -> bool:
        with self._cv:
            return bool(self._previews)

    def _gone(self, session_id: str) -> bool:
        """True when the session was deleted meanwhile. Removes what a delete could not while the analyser
        still had the video open."""
        if self.store.get(session_id) is not None:
            return False
        shutil.rmtree(self.store.session_dir(session_id), ignore_errors=True)
        return True

    def _revision_dir(self, session_id: str, revision: int) -> Path:
        """An empty folder for a revision the store has not committed. A crashed attempt's leftovers are removed.
        The session folder itself is never recreated, so a session deleted meanwhile stays deleted."""
        folder = self.store.session_dir(session_id) / f"r{revision}"
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()
        return folder

    def request_refine(self, row: SessionRow, window: Optional[tuple[int, int]]) -> dict[str, Any]:
        """Queue sharpening of one inclusive window, or of every frame when window is None."""
        backend = self.backend
        if backend is None:
            raise HttpError(503, "the analyser could not start; restart the app" if self.warm["state"] == "failed"
                            else "the analyser is still starting")
        if row.tier == "refined":
            raise HttpError(409, "every frame of this clip is already measured")
        if window is not None and not backend.sharpens_windows:
            raise HttpError(409, "this analyser cannot sharpen one moment; sharpen the whole clip instead")
        if window is not None and window in self.sharpened(row):
            return {"queued": False, "message": "this moment is already sharpened"}
        with self._cv:
            for job in self._refinements:
                if job.session_id == row.id and (job.everything or (window is not None and window in job.windows)):
                    return {"queued": False, "message": "this is already waiting to be sharpened"}
            job = Refinement(row.id, [] if window is None else [window], everything=window is None,
                             total=0 if window is None else 1)
            self._refinements.append(job)
            self._set(row.id, "refining", "Waiting to sharpen", windows_total=job.total)
        return {"queued": True}

    # reading

    def video_path(self, session_id: str) -> Optional[Path]:
        folder = self.store.session_dir(session_id)
        for suffix in VIDEO_TYPES:
            path = folder / f"video{suffix}"
            if path.is_file():
                return path
        return None

    def report(self, session_id: str, revision: int):
        key = (session_id, revision)
        with self._cv:
            if key in self._reports:
                self._reports.move_to_end(key)
                return self._reports[key]
        path = self.store.session_dir(session_id) / f"r{revision}" / "report.json"
        report = report_from_dict(json.loads(path.read_text("utf-8")))
        self._remember(session_id, revision, report)
        return report

    def _remember(self, session_id: str, revision: int, report) -> None:
        with self._cv:
            self._reports[(session_id, revision)] = report
            self._reports.move_to_end((session_id, revision))
            while len(self._reports) > REPORT_CACHE:
                self._reports.popitem(last=False)

    def pose(self, session_id: str, revision: int) -> dict:
        return json.loads((self.store.session_dir(session_id) / f"r{revision}" / "pose.json").read_text("utf-8"))

    def chat_stopped(self, session_id: str) -> bool:
        """Whether the coach has stopped this session's chat for pain. The stop lasts for the whole conversation,
        however long ago it was said and whatever context a page sends back. Only the coach chat writes it."""
        return self.chat is not None and self.store.coach_said(session_id, self.chat.stop_texts)

    @property
    def plans_ready(self) -> bool:
        """Whether a practice plan can be made. The coach chat must be there too, because it is what tells a
        recorded pain stop: without it no plan is made at all, rather than one for a stopped conversation."""
        return self.plans is not None and self.chat is not None

    def sharpened(self, row: SessionRow) -> list[tuple[int, int]]:
        """Windows the current revision's pose document already measured. None are listed for a refined clip,
        where every frame is measured."""
        if row.revision < 0 or row.tier != "preview":
            return []
        with self._cv:
            cached = self._sharpened.get(row.id)
        if cached is not None and cached[0] == row.revision:
            return cached[1]
        pairs = windows_done(self.pose(row.id, row.revision))
        with self._cv:
            current = self._sharpened.get(row.id)
            if current is None or current[0] < row.revision:
                self._sharpened[row.id] = (row.revision, pairs)
        return pairs

    def _state(self, row: SessionRow) -> tuple[str, Optional[Job]]:
        """The session's status word and job. The caller holds self._cv and read `row` under it."""
        job = self._jobs.get(row.id)
        if row.revision < 0:
            if row.error is not None:
                return "error", job
            return (job.status if job and job.status in ("queued", "preview") else "queued"), job
        if job is not None and job.status == "refining":
            return "refining", job
        return "ready", job

    def wait_for_change(self, session_id: str, since: Optional[int], wait_s: float) -> None:
        deadline = time.monotonic() + max(0.0, min(wait_s, MAX_WAIT_S))
        with self._cv:
            while since is not None and self._stamps.get(session_id, 0) == since and not self._stopping:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._cv.wait(left)

    def status(self, session_id: str) -> Optional[dict[str, Any]]:
        """One consistent snapshot of a session, or None when it does not exist."""
        with self._cv:
            row = self.store.get(session_id)
            if row is None:
                return None
            status, job = self._state(row)
            stamp = self._stamps.get(session_id, 0)
            meta = dict(self._meta.get(session_id, {}))
        error = None
        if status == "error":
            code = row.error or "failed"
            error = {"code": code, "message": MESSAGES.get(code, MESSAGES["failed"])}
        summary = None
        try:
            if row.revision >= 0:
                summary = summarize(self.report(row.id, row.revision), row.labels)
            sharpened = [list(pair) for pair in self.sharpened(row)]
        except FileNotFoundError:
            if self.store.get(session_id) is None:
                return None
            raise
        duration = row.duration_s or meta.get("duration_s")
        return {
            "id": row.id,
            "stamp": stamp,
            "video_name": row.video_name,
            "created": row.created,
            "status": status,
            "detail": job.detail if job and status in ("queued", "preview", "refining") else "",
            "notice": job.notice if job else "",
            "error": error,
            "revision": row.revision if row.revision >= 0 else None,
            "version": row.revision if row.revision >= 0 else None,
            "bundle_url": f"/api/session/{row.id}/bundle?revision={row.revision}" if row.revision >= 0 else None,
            "media_url": f"/media/{row.id}",
            "tier": row.tier,
            "n_frames": row.n_frames,
            "fps": row.fps,
            "duration_s": duration,
            "long_clip": bool(duration and duration > TESTED_SECONDS),
            "sharpen_all_s": sharpen_all_estimate(getattr(self.backend, "name", None), row.n_frames, duration),
            "labels": row.labels,
            "timings": row.timings,
            "windows": ({"done": job.windows_done, "total": job.windows_total}
                        if job and status == "refining" else None),
            "sharpened": sharpened,
            "summary": summary,
            # A page opened without this tab's saved chat learns here that the coach has stopped, so it offers no
            # training chips before its first question.
            "chat_stopped": self.chat_stopped(row.id),
        }

    def session_list(self) -> list[dict[str, Any]]:
        with self._cv:
            rows = [(row, self._state(row)[0]) for row in self.store.sessions()]
        out = []
        for row, status in rows:
            first = row.timings.get("first") or {}
            out.append({
                "id": row.id, "video_name": row.video_name, "created": row.created, "status": status,
                "to_work_on": row.to_work_on, "tier": row.tier, "duration_s": row.duration_s,
                "analysis_s": first.get("total") if status != "error" else None, "backend": first.get("backend"),
                "labels": row.labels,
                "error": MESSAGES.get(row.error or "", MESSAGES["failed"]) if status == "error" else None,
            })
        return out

    def exercises_payload(self) -> bytes:
        if self._exercises is None:
            # The dose the coach and the plan give once the player has called an exercise too hard.
            eased = getattr(self.plans, "easier_dose", None)
            rows = []
            for exercise in library().values():
                plain = to_plain(exercise)
                gif = EXERCISES_ROOT / exercise.gif
                ok = gif.parent.resolve() == GIFS.resolve() and gif.is_file()
                plain["gif_url"] = f"/gifs/{gif.name}" if ok else None
                plain["easier_dose"] = eased(exercise.sets_reps) if eased else None
                rows.append(plain)
            self._exercises = json.dumps({"exercises": rows}, separators=(",", ":")).encode("utf-8")
        return self._exercises

    def progress(self, label: str, metric: Optional[str]) -> dict[str, Any]:
        """Each measure's comparable readings for this player, oldest first. A reading the analysis was not at all
        sure of (confidence 0) stays saved but is not a point here, the same rule the coach uses to compare clips,
        and a measure with no comparable reading is left out."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for point in self.store.progress(label, metric):
            if point["confidence"] > 0:
                grouped.setdefault(point["name"], []).append(point)
        metrics = []
        for name, points in grouped.items():
            spec = METRIC_CATALOG.get(name)
            entry: dict[str, Any] = {
                "name": name, "label": spec.label if spec else name, "unit": points[-1]["unit"],
                "direction": spec.direction if spec else None, "target": list(spec.target) if spec else None,
                "points": [{key: p[key] for key in ("session_id", "created", "video_name", "player", "value",
                                                     "confidence")} for p in points],
            }
            entry["trend"] = trend(spec, [p["value"] for p in points])
            metrics.append(entry)
        return {"player": label, "metrics": metrics}

    # review video

    def export(self, row: SessionRow) -> dict[str, Any]:
        if self.exporter is None:
            raise HttpError(503, "the review video maker is not installed yet")
        key = (row.id, row.revision)
        with self._export_lock:
            state = self._exports.get(key)
            if state is None or state["status"] == "error":
                started = time.monotonic()
                state = self._exports[key] = {"status": "running", "message": "", "started": started}
                threading.Thread(target=self._export, args=(row, started), name="coach-export", daemon=True).start()
            return _export_reply(state)

    def _export(self, row: SessionRow, started: float) -> None:
        key = (row.id, row.revision)
        folder = self.store.session_dir(row.id) / f"r{row.revision}"
        part = folder / "review.part.mp4"
        try:
            self.exporter(self.video_path(row.id), self.pose(row.id, row.revision), self.report(row.id, row.revision), part)
            part.replace(folder / "review.mp4")
            state = {"status": "ready", "message": "", "seconds": round(time.monotonic() - started)}
        except Exception as exc:
            log.exception("review video of %s failed", row.id)
            part.unlink(missing_ok=True)
            message = "the review video could not be made"
            if isinstance(exc, FileNotFoundError) or str(exc).startswith(("Only ", "cpu.lock", "Local Chinese neural TTS")):
                message = str(exc)
            state = {"status": "error", "message": message}
        with self._export_lock:
            self._exports[key] = state

    def export_state(self, row: SessionRow, revision: int) -> dict[str, Any]:
        with self._export_lock:
            state = self._exports.get((row.id, revision))
            if state is not None:
                return _export_reply(state)
        # Made before this server started: the file is there, but how long it took is not known.
        if (self.store.session_dir(row.id) / f"r{revision}" / "review.mp4").is_file():
            return {"status": "ready", "message": ""}
        return {"status": "none", "message": ""}


def _export_reply(state: dict[str, Any]) -> dict[str, Any]:
    """The client's view of an export: wall-clock seconds so far while it runs, and in total once it is ready."""
    reply = {"status": state["status"], "message": state["message"]}
    if state["status"] == "running":
        reply["seconds"] = int(time.monotonic() - state["started"])
    elif "seconds" in state:
        reply["seconds"] = state["seconds"]
    return reply


def summarize(report, labels: dict[str, str]) -> dict[str, Any]:
    def brief(finding):
        return {"id": finding.id, "category": finding.category.value, "severity": finding.severity.value,
                "headline": finding.headline}

    players = []
    for role in report.players:
        players.append({
            "role": role.value,
            "label": labels.get(role.value, role.value),
            "to_work_on": [brief(f) for f in report.top_findings(3, role)],
            "strengths": [brief(f) for f in report.strengths(2, role)],
        })
    return {"players": players, "warnings": list(report.warnings)}


def trend(spec, values: list[float]) -> dict[str, str]:
    """Better, same or worse from the last two comparable readings, measured as distance from the practice target.
    Sessions without a comparable reading are skipped, so the words say "reading", not "session" or "last time"."""
    if len(values) < 2:
        return {"word": "first", "text": "Nothing to compare it with yet"}
    if spec.descriptive:
        change = values[-1] - values[-2]
        word = "same" if abs(change) <= spec.tolerance * SAME_BAND else "higher" if change > 0 else "lower"
        return {"word": word, "text": f"Reading is {word}; this is not a technique score"}
    if spec is None:
        return {"word": "same", "text": "No practice target for this measure"}
    before, after = distance(spec, values[-2]), distance(spec, values[-1])
    if before == 0 and after == 0:
        return {"word": "same", "text": "Inside the practice target in the last two readings"}
    if after < before - SAME_BAND:
        return {"word": "better", "text": "Closer to the practice target than the reading before"}
    if after > before + SAME_BAND:
        return {"word": "worse", "text": "Further from the practice target than the reading before"}
    return {"word": "same", "text": "About the same as the reading before"}


def _int_in(value: Any, low: int, high: int, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise HttpError(400, f"{what} must be a whole number from {low} to {high}")
    return value


def _role(value: Any) -> Role:
    if value not in ("near", "far"):
        raise HttpError(400, "player must be near or far")
    return Role(value)


ROUTES: list[tuple[str, re.Pattern, str]] = [
    (method, re.compile(pattern), name) for method, pattern, name in (
        ("GET", r"/|/plan|/progress", "page"),
        ("GET", r"/static/([^/]+)", "static"),
        ("GET", r"/gifs/([^/]+)", "gif"),
        ("GET", r"/media/([0-9a-f]{16})", "media"),
        ("GET", r"/download/([0-9a-f]{16})", "download"),
        ("GET", r"/download/([0-9a-f]{16})/(analysis\.html|analysis\.json|narration\.txt|frame-map\.json)", "download_analysis"),
        ("GET", r"/api/health", "health"),
        ("GET", r"/api/sessions", "sessions"),
        ("GET", r"/api/session/([0-9a-f]{16})", "session"),
        ("GET", r"/api/session/([0-9a-f]{16})/bundle", "bundle"),
        ("GET", r"/api/exercises", "exercises"),
        ("GET", r"/api/players", "players"),
        ("GET", r"/api/progress", "progress"),
        ("GET", r"/api/plan/([0-9a-f]{16})", "plan_get"),
        ("GET", r"/api/export/([0-9a-f]{16})", "export_get"),
        ("POST", r"/api/upload", "upload"),
        ("POST", r"/api/sample", "sample"),
        ("POST", r"/api/refine", "refine"),
        ("POST", r"/api/labels", "labels"),
        ("POST", r"/api/chat", "chat"),
        ("POST", r"/api/plan", "plan"),
        ("POST", r"/api/export", "export"),
    )
]


class Handler(BaseHTTPRequestHandler):
    server_version = "BadmintonCoach"
    sys_version = ""
    timeout = 60

    @property
    def app(self) -> App:
        return self.server.app

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def do_PATCH(self):
        self._dispatch("PATCH")

    def log_message(self, format, *args):
        log.debug("%s %s", self.address_string(), format % args)

    def send_error(self, code, message=None, explain=None):
        self.close_connection = True
        self._send(code, json.dumps({"error": STATUS_WORDS.get(code, "the request failed")}).encode(),
                   "application/json; charset=utf-8")

    # plumbing

    def _dispatch(self, method: str) -> None:
        self._started = False
        self._unread = 0
        url = urlsplit(self.path)
        try:
            declared = self.headers.get("Content-Length")
            if declared is not None and DIGITS.fullmatch(declared):
                self._unread = int(declared)
            self._check_origin(method)
            try:
                self.query = parse_qs(url.query, max_num_fields=20)
            except ValueError:
                raise HttpError(400, "the address has too many parts") from None
            other_method = False
            for verb, pattern, name in ROUTES:
                match = pattern.fullmatch(url.path)
                if match is None:
                    continue
                if verb != method:
                    other_method = True
                    continue
                getattr(self, name)(*match.groups())
                return
            raise HttpError(405 if other_method else 404, "this server does not do that" if other_method else "not found")
        except HttpError as err:
            self._fail(err.status, err.message, err.headers)
        except (ConnectionError, TimeoutError):
            self.close_connection = True
        except Exception:
            log.exception("%s %s failed", method, url.path)
            self._fail(500, "something went wrong on the server")

    def _check_origin(self, method: str) -> None:
        if self.headers.get("Host") not in self.server.hosts:
            raise HttpError(403, "this server only answers at its own address")
        origin = self.headers.get("Origin")
        if method != "GET" and origin is not None and origin not in self.server.origins:
            raise HttpError(403, "requests from other sites are refused")

    def _fail(self, status: int, message: str, headers: tuple = ()) -> None:
        self.close_connection = True
        if self._started:
            return
        if 0 < self._unread <= MAX_DRAIN_BYTES:
            try:
                self.rfile.read(self._unread)
            except OSError:
                return
        self._json_reply(status, {"error": message}, headers)

    def _send(self, status: int, body: bytes, content_type: str, extra: tuple = ()) -> None:
        self._started = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._common_headers(content_type)
        for key, value in extra:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _common_headers(self, content_type: str) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        if content_type.startswith("text/html"):
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Frame-Options", "DENY")

    def _json_reply(self, status: int, payload: Any, extra: tuple = ()) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8", extra)

    def _length(self, limit: int, too_large: str) -> int:
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            raise HttpError(411, "send the body with a Content-Length header")
        raw = self.headers.get("Content-Length")
        if raw is None:
            raise HttpError(411, "send the body with a Content-Length header")
        if not DIGITS.fullmatch(raw):
            raise HttpError(400, "the Content-Length header is not a number")
        length = int(raw)
        if length > limit:
            raise HttpError(413, too_large)
        return length

    def _read(self, length: int) -> bytes:
        data = self.rfile.read(length)
        self._unread = max(0, self._unread - len(data))
        if len(data) != length:
            raise HttpError(400, "the request body was cut short")
        return data

    def _json(self, limit: int = MAX_JSON_BYTES) -> dict[str, Any]:
        raw = self._read(self._length(limit, "this request is too large"))
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, RecursionError):
            raise HttpError(400, "the request body is not valid JSON") from None
        if not isinstance(data, dict):
            raise HttpError(400, "the request body must be a JSON object")
        return data

    def _query(self, key: str) -> Optional[str]:
        values = self.query.get(key)
        return values[0] if values else None

    def _query_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        value = self._query(key)
        if value is None:
            return default
        if not DIGITS.fullmatch(value):
            raise HttpError(400, f"{key} must be a whole number")
        return int(value)

    def _row(self, session_id: Any) -> SessionRow:
        try:
            row = self.app.store.get(check_id(session_id))
        except ValueError:
            row = None
        if row is None:
            raise HttpError(404, "unknown session")
        return row

    def _analysed(self, session_id: Any) -> SessionRow:
        row = self._row(session_id)
        if row.revision < 0:
            raise HttpError(409, "this clip has not been analysed yet")
        return row

    def _send_file(self, path: Path, content_type: str, extra: tuple = ()) -> None:
        """The file, or the one byte range the Range header asks for."""
        size = path.stat().st_size
        header = self.headers.get("Range")
        start, end, status = 0, size - 1, 200
        if header is not None:
            match = RANGE.fullmatch(header.strip())
            if match is None or not (match[1] or match[2]):
                raise bad_range(size)
            if match[1]:
                start = int(match[1])
                end = min(int(match[2]), size - 1) if match[2] else size - 1
            else:
                suffix = int(match[2])
                if suffix == 0:
                    raise bad_range(size)
                start = max(0, size - suffix)
            if start >= size or end < start:
                raise bad_range(size)
            status = 206
        length = end - start + 1 if size else 0
        self._started = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self._common_headers(content_type)
        for key, value in extra:
            self.send_header(key, value)
        self.end_headers()
        with open(path, "rb") as handle:
            handle.seek(start)
            left = length
            while left > 0:
                block = handle.read(min(CHUNK, left))
                if not block:
                    break
                self.wfile.write(block)
                left -= len(block)

    # pages and files

    def page(self):
        path = WEB / PAGES[urlsplit(self.path).path]
        if not path.is_file():
            raise HttpError(404, "not found")
        self._send(200, path.read_bytes(), WEB_TYPES[".html"])

    def static(self, name: str):
        self._named_file(WEB, name, WEB_TYPES)

    def gif(self, name: str):
        self._named_file(GIFS, name, {".gif": "image/gif"})

    def _named_file(self, folder: Path, name: str, types: dict[str, str]) -> None:
        """Only a plain file name directly inside folder; the raw path is never decoded or joined as a path."""
        suffix = Path(name).suffix.lower()
        if not FILE_NAME.fullmatch(name) or suffix not in types:
            raise HttpError(404, "not found")
        path = folder / name
        if path.parent.resolve() != folder.resolve() or not path.is_file():
            raise HttpError(404, "not found")
        self._send(200, path.read_bytes(), types[suffix])

    def media(self, session_id: str):
        self._row(session_id)
        video = self.app.video_path(session_id)
        if video is None:
            raise HttpError(404, "not found")
        self._send_file(video, VIDEO_TYPES[video.suffix])

    def download(self, session_id: str):
        row = self._analysed(session_id)
        revision = self._query_int("revision", row.revision)
        path = self.app.store.session_dir(row.id) / f"r{revision}" / "review.mp4"
        if revision > row.revision or not path.is_file():
            raise HttpError(404, "the review video is not ready")
        self._send_file(path, "video/mp4", (("Content-Disposition", 'attachment; filename="review.mp4"'),))

    def download_analysis(self, session_id: str, name: str):
        row = self._analysed(session_id)
        revision = self._query_int("revision", row.revision)
        folder = self.app.store.session_dir(row.id) / f"r{revision}"
        path = folder / name
        if revision > row.revision or not (folder / "review.mp4").is_file() or not path.is_file():
            raise HttpError(404, "the matching analysis is not ready")
        content_type = "text/plain; charset=utf-8" if name.endswith(".txt") else WEB_TYPES[path.suffix]
        self._send_file(path, content_type, (("Content-Disposition", f'attachment; filename="{name}"'),))

    # api

    def health(self):
        app = self.app
        backend = app.backend
        self._json_reply(200, {
            "warm": app.warm,
            "backend": None if backend is None else {"name": backend.name, "label": backend.label,
                                                      "sharpens_windows": bool(backend.sharpens_windows)},
            "auto_refine": app.auto_refine,
            "chat": app.chat is not None, "plan": app.plans_ready, "export": app.exporter is not None,
            "samples": sorted(app.samples),
            "tested_seconds": TESTED_SECONDS,
            "warnings": dict(WARNINGS),
        })

    def sessions(self):
        self._json_reply(200, {"sessions": self.app.session_list()})

    def session(self, session_id: str):
        self._row(session_id)
        wait = self._query("wait")
        try:
            wait_s = float(wait) if wait is not None else 0.0
        except ValueError:
            raise HttpError(400, "wait must be a number of seconds") from None
        self.app.wait_for_change(session_id, self._query_int("since"), wait_s if wait_s == wait_s else 0.0)
        status = self.app.status(session_id)
        if status is None:
            raise HttpError(404, "unknown session")
        self._json_reply(200, status)

    def bundle(self, session_id: str):
        """The saved bundle.json, byte for byte, unless it is an older bundle that put a practice-guide midpoint
        where nothing was seen. Then the reply carries that reading as missing. The file itself is never rewritten.
        A file that is not JSON, or that breaks the rule for unseen readings, is an error, not a guess."""
        row = self._analysed(session_id)
        revision = self._query_int("revision", row.revision)
        path = self.app.store.session_dir(row.id) / f"r{revision}" / "bundle.json"
        if revision > row.revision or not path.is_file():
            raise HttpError(404, "this revision is not ready")
        saved = path.read_bytes()
        try:
            prepared = prepare_bundle(json.loads(saved))
        except (json.JSONDecodeError, UnicodeDecodeError, BundleContractError):
            log.exception("bundle of %s revision %s could not be read", row.id, revision)
            raise HttpError(500, "this revision's saved data could not be read") from None
        if prepared.send_saved_bytes:
            self._send(200, saved, "application/json; charset=utf-8")
        else:
            self._json_reply(200, prepared.bundle)

    def exercises(self):
        self._send(200, self.app.exercises_payload(), "application/json; charset=utf-8")

    def players(self):
        self._json_reply(200, {"players": [name for name in self.app.store.labels() if name not in UNNAMED]})

    def progress(self):
        label = self._query("player")
        if not label or len(label) > 40:
            raise HttpError(400, "say which player")
        if label in UNNAMED:
            raise HttpError(400, "give this player a name on a session page first, so different people are not mixed")
        self._json_reply(200, self.app.progress(label, self._query("metric")))

    def plan_get(self, plan_id: str):
        saved = self.app.store.plan(plan_id)
        if saved is None:
            raise HttpError(404, "unknown plan")
        row = self.app.store.get(saved["session_id"])
        saved["label"] = row.labels.get(saved["player"]) if row else saved["player"]
        saved["video_name"] = row.video_name if row else ""
        self._json_reply(200, saved)

    def export_get(self, session_id: str):
        row = self._analysed(session_id)
        revision = self._query_int("revision", row.revision)
        state = self.app.export_state(row, revision)
        if state["status"] == "ready":
            state["url"] = f"/download/{row.id}?revision={revision}"
            folder = self.app.store.session_dir(row.id) / f"r{revision}"
            if (folder / "analysis.html").is_file():
                state["analysis_url"] = f"/download/{row.id}/analysis.html?revision={revision}"
                state["evidence_url"] = f"/download/{row.id}/analysis.json?revision={revision}"
        self._json_reply(200, state)

    def upload(self):
        length = self._length(MAX_UPLOAD_BYTES, "this video is larger than 2 GB")
        name = clean_filename(self.headers.get("X-Filename"))
        suffix = Path(name).suffix.lower()
        if suffix not in VIDEO_TYPES:
            raise HttpError(415, "send an .mp4, .mov or .webm video")
        if length == 0:
            raise HttpError(400, "this file is empty")
        session_id = new_id()
        folder = self.app.store.session_dir(session_id)
        folder.mkdir(parents=True)
        started = time.perf_counter()
        try:
            digest = hashlib.sha256()
            with open(folder / f"video{suffix}", "xb") as out:
                left = length
                while left:
                    block = self._read_some(min(CHUNK, left))
                    digest.update(block)
                    out.write(block)
                    left -= len(block)
            self.app.add(session_id, name, digest.hexdigest(), folder / f"video{suffix}",
                         upload_s=time.perf_counter() - started)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        self._json_reply(201, {"id": session_id})

    def _read_some(self, size: int) -> bytes:
        block = self.rfile.read(size)
        if not block:
            raise HttpError(400, "the upload stopped before the whole file arrived")
        self._unread = max(0, self._unread - len(block))
        return block

    def sample(self):
        name = self._json().get("name")
        source = self.app.samples.get(name) if isinstance(name, str) else None
        if source is None:
            raise HttpError(404, "there is no sample with that name")
        session_id = new_id()
        folder = self.app.store.session_dir(session_id)
        folder.mkdir(parents=True)
        try:
            target = folder / f"video{source.suffix.lower()}"
            self.app.add(session_id, source.name, copy_with_hash(source, target), target, upload_s=None)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        self._json_reply(201, {"id": session_id})

    def refine(self):
        body = self._json()
        row = self._analysed(body.get("session"))
        window, everything = body.get("window"), body.get("all") is True
        if (window is None) == (not everything):
            raise HttpError(400, "ask for one window or for all")
        pair = None
        if window is not None:
            if not isinstance(window, dict):
                raise HttpError(400, "window needs start_frame and end_frame")
            first = _int_in(window.get("start_frame"), 0, row.n_frames - 1, "start_frame")
            last = _int_in(window.get("end_frame"), first, row.n_frames - 1, "end_frame")
            if (last - first + 1) / row.fps > MAX_WINDOW_S:
                raise HttpError(400, f"sharpen at most {MAX_WINDOW_S:.0f} seconds at a time")
            pair = (first, last)
        result = self.app.request_refine(row, pair)
        self._json_reply(202 if result["queued"] else 200, result)

    def labels(self):
        body = self._json()
        row = self._row(body.get("session"))
        try:
            labels = self.app.store.set_labels(row.id, body.get("near"), body.get("far"))
        except ValueError as exc:
            raise HttpError(400, str(exc)) from None
        self._json_reply(200, {"labels": labels})

    def chat(self):
        body = self._json(MAX_CHAT_BYTES)
        row = self._analysed(body.get("session"))
        text, context = body.get("text"), body.get("context")
        context = {} if context is None else context
        if not isinstance(text, str) or not text.strip():
            raise HttpError(400, "type a question first")
        text = text.strip()
        if len(text) > MAX_CHAT_CHARS:
            raise HttpError(413, f"keep a question under {MAX_CHAT_CHARS} characters")
        if not isinstance(context, dict):
            raise HttpError(400, "context must be an object")
        if self.app.chat is None:
            raise HttpError(503, "the coach chat is not installed yet")
        store = self.app.store
        report = self.app.report(row.id, row.revision)
        role = context.get("player") if context.get("player") in ("near", "far") else "near"
        label = row.labels[role]
        earlier = []
        for other_id, other_role in ([] if label in UNNAMED else store.earlier(row.id, label, HISTORY_REPORTS)):
            other = store.get(other_id)
            # The chat contract has no role per earlier report, so only sessions where this name played the
            # same side are passed.
            if other is not None and other_role == role:
                earlier.append(self.app.report(other.id, other.revision))
        stopped = False
        while True:
            history = [Turn(speaker, said) for speaker, said in store.turns(row.id, HISTORY_TURNS)]
            stopped = stopped or self.app.chat_stopped(row.id)
            asked = {**context, "pain_stop": True} if stopped else context
            reply, answered = self.app.chat((row.id, row.revision), report, history, text, asked, tuple(earlier))
            # Another tab may record a stop while this reply is made. Then it is not saved, and the coach answers
            # again knowing the stop. That answer is always saved, so this happens at most once.
            if store.add_turns(row.id, [("user", text), ("coach", reply["text"])],
                               unless_coach_said=() if stopped else self.app.chat.stop_texts):
                break
            stopped = True
        self._json_reply(200, {"reply": reply, "context": answered})

    def plan(self):
        body = self._json()
        row = self._analysed(body.get("session"))
        if not self.app.plans_ready:
            raise HttpError(503, "the practice plan is not installed yet")
        if self.app.chat_stopped(row.id):
            raise HttpError(409, PLAN_STOPPED)
        player = _role(body.get("player"))
        weeks = _int_in(body.get("weeks", 2), 1, 4, "weeks")
        level = _int_in(body.get("level", 3), 1, 3, "level")
        equipment = body.get("equipment")
        if equipment is not None:
            if not isinstance(equipment, list) or len(equipment) > len(EQUIPMENT) or not all(
                    isinstance(item, str) and item in EQUIPMENT for item in equipment):
                raise HttpError(400, "equipment must be a list of: " + ", ".join(sorted(EQUIPMENT)))
            equipment = tuple(dict.fromkeys(equipment))
        avoid = body.get("avoid", [])
        if not isinstance(avoid, list) or len(avoid) > MAX_AVOID or not all(
                isinstance(item, str) and 0 < len(item) <= MAX_ID_CHARS for item in avoid):
            raise HttpError(400, "avoid must be a list of exercise names")
        avoid = tuple(dict.fromkeys(avoid))
        report = self.app.report(row.id, row.revision)
        if player not in report.players:
            raise HttpError(400, f"the {player.value} player was not found in this clip")
        plan = self.app.plans(report, player, weeks, level, equipment, avoid)
        # Another tab may record a stop while this plan is made. Then the plan is refused and not saved, and so is
        # an answer that there is nothing to plan.
        if not plan.get("weeks"):
            if self.app.chat_stopped(row.id):
                raise HttpError(409, PLAN_STOPPED)
            self._json_reply(200, {"id": None, "plan": plan, "url": None,
                                   "message": plan.get("message") or "There is nothing to plan yet."})
            return
        plan_id = self.app.store.add_plan(row.id, player.value, plan, unless_coach_said=self.app.chat.stop_texts)
        if plan_id is None:
            raise HttpError(409, PLAN_STOPPED)
        self._json_reply(201, {"id": plan_id, "plan": plan, "url": f"/plan?id={plan_id}"})

    def export(self):
        row = self._analysed(self._json().get("session"))
        self._json_reply(202, self.app.export(row))


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, app: App, port: int):
        super().__init__((HOST, port), Handler)
        self.app = app
        port = self.server_address[1]
        self.hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        self.origins = {f"http://{host}" for host in self.hosts}
        self.url = f"http://127.0.0.1:{port}/"
