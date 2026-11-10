"""One clip in, one coaching session out: pose, analysis, bundle and files, timed as one call.

The app runs analyze_video for every upload, and E2ERunner times the same call for the end-to-end
speed claim (S-A2). revise() builds a later revision from a sharpened pose document. Every revision goes
into its own folder and is never overwritten, so a reader never sees a report and a bundle from
different revisions.
"""
from __future__ import annotations

import json
import logging
import math
import os
import struct
import sys
import tempfile
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2

from coach.analyze import analyze
from coach.bundle import build_bundle
from coach.exercises.library import LIBRARY_PATH, for_tag, load_library
from coach.schema import Report
from coach.player_input import player_inputs

from .backend import DEFAULT_MODELS, PROJECT, FastBackend, NoCourt, TurboBackend, Unreadable

log = logging.getLogger("coach.app")

MAX_SECONDS = 180.0
TESTED_SECONDS = 36.0
# The bundle stores pixel positions as int16 tenths, so a frame side past 3276 px cannot be packed.
MAX_SIDE_PX = 3276
MIN_ROLE_SHARE = 0.2
EXERCISES_PER_FINDING = 3
MOOV_MAX_BYTES = 64 << 20
INT32_MAX = 2 ** 31 - 1
MESSAGES = {
    "not_a_video": "this file is not a video I can read",
    "too_long": "this clip is longer than 3 minutes",
    "too_large": "this video is more than 3276 pixels across; save it at 1080p and try again",
    "no_court": "I could not find the court in this video; film with a still camera behind one baseline "
                "so the whole court is in view",
    "no_players": "I could not find two players on court",
    "failed": "something went wrong while analysing this clip",
}
JSON_OPTS = {"separators": (",", ":"), "allow_nan": False}


@dataclass(frozen=True)
class AnalysisError:
    code: str
    message: str

    @classmethod
    def of(cls, code: str) -> "AnalysisError":
        return cls(code, MESSAGES[code])


@dataclass
class Session:
    """One revision of one clip. `error` is set instead of raising when the clip cannot be analysed."""

    video_id: str
    revision: int
    timings: dict[str, Any] = field(default_factory=dict)
    doc: Optional[dict] = None
    report: Optional[Report] = None
    bundle: Optional[dict] = None
    pose_path: Optional[Path] = None
    report_path: Optional[Path] = None
    bundle_path: Optional[Path] = None
    duration_s: Optional[float] = None
    error: Optional[AnalysisError] = None


def _boxes(data, start: int = 0):
    """(kind, body) for each MP4 box in data[start:]. ValueError or struct.error when a size does not fit."""
    pos = start
    while pos < len(data):
        size, kind = struct.unpack_from(">I4s", data, pos)
        head = 8
        if size == 1:
            (size,), head = struct.unpack_from(">Q", data, pos + 8), 16
        elif size == 0:
            size = len(data) - pos
        if size < head or pos + size > len(data):
            raise ValueError("box out of range")
        yield kind, data[pos + head:pos + size]
        pos += size


def _only(body, kind: bytes):
    found = [child for name, child in _boxes(body) if name == kind]
    if len(found) != 1:
        raise ValueError(f"{len(found)} {kind!r} boxes")
    return found[0]


def _top_boxes(handle):
    """(kind, body offset, body size) of each top-level box, reading only the box headers."""
    total = os.fstat(handle.fileno()).st_size
    pos = 0
    while pos < total:
        handle.seek(pos)
        head = handle.read(16)
        size, kind = struct.unpack_from(">I4s", head)
        hlen = 8
        if size == 1:
            (size,), hlen = struct.unpack_from(">Q", head, 8), 16
        elif size == 0:
            size = total - pos
        if size < hlen or pos + size > total:
            raise ValueError("box out of range")
        yield kind, pos + hlen, size - hlen
        pos += size


def _plain_h264(video_path: Path) -> Optional[tuple[int, float, int]]:
    """Frame count, fps and longest side read straight from the MP4 boxes, without opening a decoder.

    Only for the layout where these are exactly OpenCV's numbers: an unfragmented file that starts with ftyp and
    holds one track, H.264, with one constant frame duration. Anything else returns None and OpenCV opens it, so
    phone clips with sound or a varying frame rate take the slower path.
    """
    try:
        with open(video_path, "rb") as handle:
            top = _top_boxes(handle)
            if next(top)[0] != b"ftyp":
                return None
            moov = next(((at, size) for kind, at, size in top if kind == b"moov"), None)
            if moov is None or moov[1] > MOOV_MAX_BYTES:
                return None
            handle.seek(moov[0])
            body = memoryview(handle.read(moov[1]))
        if any(kind == b"mvex" for kind, _ in _boxes(body)):
            return None
        mdia = _only(_only(body, b"trak"), b"mdia")
        if bytes(_only(mdia, b"hdlr")[8:12]) != b"vide":
            return None
        mdhd = _only(mdia, b"mdhd")
        (timescale,) = struct.unpack_from(">I", mdhd, 20 if mdhd[0] == 1 else 12)
        stbl = _only(_only(mdia, b"minf"), b"stbl")
        stsd, stts = _only(stbl, b"stsd"), _only(stbl, b"stts")
        entries = list(_boxes(stsd, 8))
        if struct.unpack_from(">I", stsd, 4)[0] != 1 or len(entries) != 1 or entries[0][0] not in (b"avc1", b"avc3"):
            return None
        width, height = struct.unpack_from(">HH", entries[0][1], 24)
        if len(stts) != 16 or struct.unpack_from(">I", stts, 4)[0] != 1:
            return None
        frames, frame_ticks = struct.unpack_from(">II", stts, 8)
    except (OSError, ValueError, IndexError, struct.error, StopIteration):
        return None
    if frames <= 0 or not 0 < timescale <= INT32_MAX or not 0 < frame_ticks <= INT32_MAX:
        return None
    return frames, timescale / frame_ticks, max(width, height)


def _opencv_header(video_path: Path) -> Optional[tuple[int, float, float]]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return None
        return (int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), float(capture.get(cv2.CAP_PROP_FPS)),
                max(capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()


def probe(video_path: Path) -> tuple[int, float] | AnalysisError:
    """Frame count and fps from the container header, without decoding a frame."""
    header = _plain_h264(video_path) or _opencv_header(video_path)
    if header is None:
        return AnalysisError.of("not_a_video")
    frames, fps, side = header
    if frames <= 0 or not math.isfinite(fps) or fps <= 0:
        return AnalysisError.of("not_a_video")
    if frames / fps > MAX_SECONDS + 1.0 / fps:
        return AnalysisError.of("too_long")
    if side > MAX_SIDE_PX:
        return AnalysisError.of("too_large")
    return frames, fps


def players_found(doc: Mapping[str, Any]) -> bool:
    """Both the near and the far player appear in at least MIN_ROLE_SHARE of the frames."""
    frame_count = int((doc.get("source") or {}).get("frame_count") or 0)
    seen = {"near": 0, "far": 0}
    for frame in doc.get("frames") or ():
        roles = {player.get("role") for player in frame.get("players") or ()}
        for role in seen:
            seen[role] += role in roles
    return frame_count > 0 and all(count >= MIN_ROLE_SHARE * frame_count for count in seen.values())


@lru_cache(maxsize=1)
def library():
    return load_library()


def finding_exercises(lib, fix_tags) -> list[str]:
    """Up to EXERCISES_PER_FINDING ids, taking turns across the fix tags so each tag gets its best exercise."""
    options = [for_tag(lib, tag, limit=EXERCISES_PER_FINDING) for tag in fix_tags]
    picks: list[str] = []
    for rank in range(EXERCISES_PER_FINDING):
        for exercises in options:
            if rank < len(exercises) and exercises[rank].id not in picks and len(picks) < EXERCISES_PER_FINDING:
                picks.append(exercises[rank].id)
    return picks


def exercise_ids(report: Report) -> dict[tuple[str, str], list[str]]:
    """Exercise ids for each finding, keyed by (finding id, player) as build_bundle takes them."""
    lib = library()
    return {(f.id, f.player.value): finding_exercises(lib, f.fix_tags) for f in report.findings}


def _report_and_bundle(session: Session, out_dir: Path) -> None:
    timings = session.timings
    session.report = analyze(session.doc, session.video_id, session.revision, timings)
    start = time.perf_counter()
    session.bundle = build_bundle(session.doc, session.report, video_url=f"/media/{session.video_id}",
                                  exercises=exercise_ids(session.report))
    timings["bundle"] = time.perf_counter() - start
    start = time.perf_counter()
    session.report_path = out_dir / "report.json"
    session.bundle_path = out_dir / "bundle.json"
    session.report_path.write_text(json.dumps(session.report.to_dict(), **JSON_OPTS), "utf-8")
    session.bundle_path.write_text(json.dumps(session.bundle, **JSON_OPTS), "utf-8")
    timings["write"] = time.perf_counter() - start


def _guarded(session: Session, started: float, work) -> Session:
    try:
        work()
    except NoCourt:
        session.error = AnalysisError.of("no_court")
    except Unreadable:
        session.error = AnalysisError.of("not_a_video")
    except Exception:
        log.exception("analysis of %s revision %s failed", session.video_id, session.revision)
        session.error = AnalysisError.of("failed")
    session.timings["total"] = time.perf_counter() - started
    return session


def analyze_video(video_path: Path, backend, session_dir: Path, video_id: str, *, players=None) -> Session:
    """Revision 0 of a clip: probe, pose, analysis, bundle and the three files in session_dir."""
    started = time.perf_counter()
    session = Session(video_id=video_id, revision=0)
    timings = session.timings
    session_dir = Path(session_dir)

    def work():
        probed = probe(video_path)
        timings["probe"] = time.perf_counter() - started
        if isinstance(probed, AnalysisError):
            session.error = probed
            return
        frames, fps = probed
        session.duration_s = frames / fps
        session_dir.mkdir(parents=True, exist_ok=True)
        session.pose_path = session_dir / "pose.json"
        start = time.perf_counter()
        session.doc, timings["pose_stages"] = backend.preview(video_path, session.pose_path)
        declared = dict(session.doc["source"].get("players", {}))
        declared.update(player_inputs(video_path) if players is None else players)
        session.doc["source"]["players"] = declared
        session.pose_path.write_text(json.dumps(session.doc, **JSON_OPTS), "utf-8")
        timings["pose"] = time.perf_counter() - start
        if not players_found(session.doc):
            session.error = AnalysisError.of("no_players")
            return
        _report_and_bundle(session, session_dir)

    return _guarded(session, started, work)


def revise(doc: dict, pose_stages: Mapping[str, Any], pose_s: float, video_id: str, revision: int,
           out_dir: Path) -> Session:
    """A later revision from a sharpened pose document that the backend already wrote to out_dir/pose.json."""
    started = time.perf_counter()
    session = Session(video_id=video_id, revision=revision, doc=doc, pose_path=Path(out_dir) / "pose.json")
    session.timings.update({"pose": pose_s, "pose_stages": dict(pose_stages)})
    return _guarded(session, started - pose_s, lambda: _report_and_bundle(session, Path(out_dir)))


class E2ERunner:
    """The runner protocol of turbo/runner.py around analyze_video: the time the user waits for a result."""

    IS_REPLAY = False

    def __init__(self, models_dir: Path = DEFAULT_MODELS, backend=None):
        self.models_dir = Path(models_dir)
        if backend is None:
            backend = TurboBackend(self.models_dir)
        elif isinstance(backend, FastBackend) and self.models_dir.resolve() != DEFAULT_MODELS.resolve():
            raise ValueError("the fast backend always uses fast\\weights")
        self.backend = backend
        self.svc = getattr(backend, "svc", None)
        if backend.name == "replay":
            self.IS_REPLAY = True
        library()

    def warm_up(self, video_path: Path) -> None:
        self.backend.warm_up(video_path)
        with tempfile.TemporaryDirectory(prefix="coach-warm-") as tmp:
            session = analyze_video(Path(video_path), self.backend, Path(tmp), "warmup")
        if session.error is not None:
            raise RuntimeError(session.error.message)

    def run(self, video_path: Path, work_dir: Path) -> dict[str, Any]:
        session = analyze_video(Path(video_path), self.backend, Path(work_dir), Path(video_path).stem)
        if session.error is not None:
            raise RuntimeError(session.error.message)
        return {
            "doc": session.doc,
            "pose_path": str(session.pose_path),
            "report_path": str(session.report_path),
            "bundle_path": str(session.bundle_path),
            "timings": session.timings,
        }

    def binding_manifest(self) -> dict[str, list[str]]:
        """Sources this process has loaded from coach\\ and turbo\\, the frozen CLI's sources when the fast backend
        runs it, and the data files the coaching reads, for the harness to hash."""
        roots = (PROJECT / "coach", PROJECT / "turbo")
        loaded = set()
        for module in list(sys.modules.values()):
            file = getattr(module, "__file__", None)
            if file and file.endswith(".py"):
                path = Path(file).resolve()
                if any(path.is_relative_to(root) for root in roots) and "tests" not in path.parts:
                    loaded.add(path)
        cli = getattr(self.backend, "cli", None)
        if cli is not None:
            loaded.update(Path(cli).resolve().parent.glob("*.py"))
        assets = [LIBRARY_PATH, PROJECT / "coach" / "style.json"]
        return {"sources": sorted(map(str, loaded)), "assets": [str(path.resolve()) for path in assets]}
