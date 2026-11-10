"""Pose backends. Each one turns a video into a pose document, the schema-3 JSON that coach.tracks reads.

Every call returns (doc, timings). timings always has "backend" plus the seconds the backend measured itself.

  TurboBackend   the resident TurboService in turbo/service.py: a sparse preview first, then sharpened
                 key-moment windows or the full refined tier.
  FastBackend    the frozen dense CLI fast/pipeline.py, started as a new process for each clip. It measures
                 every frame, so its documents are already refined and have no windows left to sharpen.
  ReplayBackend  stored pose documents, for tests and an explicit `--backend replay` only. The app never
                 falls back to it when a live analysis fails.

Windows are inclusive (start_frame, end_frame) pairs, the same as coach.schema.Window.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT = Path(__file__).resolve().parents[2]
FAST_CLI = PROJECT / "fast" / "pipeline.py"
DEFAULT_MODELS = PROJECT / "fast" / "weights"
SAMPLES = PROJECT / "sample-videos"
STORED_DENSE = {
    "01-singles-rally": PROJECT / "fast/runs/step4-rebalanced-dev/outputs/01-r1.json",
    "02-singles-dive": PROJECT / "fast/runs/step4-rebalanced-dev/outputs/02-r1.json",
    "03-womens-singles-rally": PROJECT / "fast/runs/step4-rebalanced-dev/outputs/03-r1.json",
    "04-china-open-ws-a": PROJECT / "fast/runs/step4-rebalanced-dev/outputs/04-r1.json",
    "05-china-open-ws-b": PROJECT / "fast/runs/step4-rebalanced-dev/outputs/05-r1.json",
    "08-china-open-ws-f": PROJECT / "fast/runs/step4-rebalanced-holdout08/outputs/08-r1.json",
}


class NoCourt(Exception):
    """The court lines were not found."""


class Unreadable(Exception):
    """The video could not be decoded."""


class NotApplicable(Exception):
    """The backend cannot do this, for example sharpening a document that already measured every frame."""


_NO_COURT = re.compile(r"(?:RuntimeError: )?no court found\b")
_UNREADABLE = re.compile(r"(?:RuntimeError: )?(?:cannot read |could not read frame |decoded \d+ of \d+ frames)")


def classify(message: str) -> Optional[Exception]:
    """NoCourt or Unreadable for the frozen pose code's own errors, else None. Reads the last line only."""
    lines = [line for line in message.strip().splitlines() if line.strip()]
    last = lines[-1].strip() if lines else ""
    if _NO_COURT.match(last):
        return NoCourt(last)
    if _UNREADABLE.match(last):
        return Unreadable(last)
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _last_line(text: str) -> str:
    lines = [line for line in (text or "").strip().splitlines() if line.strip()]
    return lines[-1].strip() if lines else ""


def _write(doc: Mapping[str, Any], out_json: Optional[Path]) -> None:
    if out_json is not None:
        Path(out_json).write_text(json.dumps(doc, separators=(",", ":")), "utf-8")


class TurboBackend:
    name = "turbo"
    label = "Turbo (resident preview, then sharpened)"
    sharpens_windows = True

    def __init__(self, models_dir: Path = DEFAULT_MODELS, decode_method: Optional[str] = None):
        """decode_method None keeps TurboService's own default, the same setting turbo.service.PoseRunner uses."""
        from turbo.service import TurboService

        self.models_dir = Path(models_dir)
        options = {} if decode_method is None else {"decode_method": decode_method}
        self.svc = TurboService(self.models_dir, **options)

    def warm_up(self, video: Optional[Path] = None) -> float:
        start = time.perf_counter()
        self.svc.warm_up(None if video is None else str(video))
        return time.perf_counter() - start

    def preview(self, video: Path, out_json: Optional[Path] = None):
        return self._call(video, out_json, tier="preview")

    def refined(self, video: Path, out_json: Optional[Path] = None):
        return self._call(video, out_json, tier="refined")

    def refine(self, video: Path, doc: Mapping[str, Any], windows: Sequence[tuple[int, int]],
               out_json: Optional[Path] = None):
        pairs = [(int(start), int(end)) for start, end in windows]
        return self._call(video, out_json, tier="window", windows=pairs, base_doc=doc)

    def _call(self, video, out_json, **kwargs):
        try:
            doc, timings = self.svc.analyze(str(video), None if out_json is None else str(out_json), **kwargs)
        except RuntimeError as exc:
            mapped = classify(str(exc))
            if mapped is not None:
                raise mapped from exc
            raise
        return doc, {"backend": self.name, **timings}


class FastBackend:
    name = "fast"
    label = "Fast (every frame, a new process per clip)"
    sharpens_windows = False

    def __init__(self, python: str = sys.executable, timeout_s: float = 900.0, cli: Path = FAST_CLI):
        self.python = python
        self.timeout_s = timeout_s
        self.cli = Path(cli)
        self.models_dir = DEFAULT_MODELS

    def warm_up(self, video: Optional[Path] = None) -> float:
        """Nothing stays loaded between clips. Given a video, one untimed run warms the disk cache."""
        if video is None:
            return 0.0
        start = time.perf_counter()
        self.preview(video)
        return time.perf_counter() - start

    def preview(self, video: Path, out_json: Optional[Path] = None):
        if out_json is not None:
            return self._run(video, Path(out_json))
        with tempfile.TemporaryDirectory(prefix="coach-fast-") as tmp:
            return self._run(video, Path(tmp) / "pose.json")

    refined = preview

    def refine(self, video, doc, windows, out_json=None):
        raise NotApplicable("this analysis already measured every frame")

    def _run(self, video: Path, out: Path):
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [self.python, str(self.cli), str(video), str(out)], cwd=PROJECT, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"fast pose took longer than {self.timeout_s:.0f} s") from exc
        process_s = time.perf_counter() - start
        if proc.returncode != 0:
            mapped = classify(proc.stderr)
            if mapped is not None:
                raise mapped
            raise RuntimeError("fast pose failed: " + _last_line(proc.stderr))
        stats = json.loads(_last_line(proc.stdout))
        read_start = time.perf_counter()
        doc = json.loads(out.read_text("utf-8"))
        timings = {"backend": self.name, "process_s": process_s, "read_s": time.perf_counter() - read_start}
        for key in ("total_s", "load_s", "court_s", "decode_done_s", "follow_s", "cpu_s", "frames", "crops"):
            if key in stats:
                timings[key] = stats[key]
        return doc, timings


class ReplayBackend:
    """Stored pose documents instead of live pose. Tests and `--backend replay` only; timings say "replay".

    `docs` maps a video's SHA-256 to its stored dense document (a dict or a JSON path). With `preview_stride`,
    preview returns coach.synth.make_preview of that document, refine copies the stored rows back inside each
    window (the base tier stays, windows_refined grows by one {"start", "end", "s"} entry per window, as
    TurboService does), and refined returns the stored dense document.
    """

    name = "replay"
    label = "Replay of a stored pose file, not a live analysis"

    def __init__(self, docs: Mapping[str, Any], preview_stride: Optional[int] = None):
        self.docs = dict(docs)
        self.preview_stride = preview_stride
        self.sharpens_windows = preview_stride is not None
        self._parsed: dict[str, dict] = {}
        self._hashes: dict[tuple[str, int, int], str] = {}

    @classmethod
    def for_samples(cls, preview_stride: Optional[int] = None) -> "ReplayBackend":
        docs = {}
        for stem, doc_path in STORED_DENSE.items():
            video = SAMPLES / f"{stem}.mp4"
            if video.is_file() and doc_path.is_file():
                docs[sha256_file(video)] = doc_path
        return cls(docs, preview_stride)

    def warm_up(self, video: Optional[Path] = None) -> float:
        return 0.0

    def preview(self, video: Path, out_json: Optional[Path] = None):
        start = time.perf_counter()
        dense = self._dense(video)
        if self.preview_stride is None:
            doc = dense
        else:
            from coach.synth import make_preview

            doc = make_preview(dense, self.preview_stride)
            doc["windows_refined"] = []
        _write(doc, out_json)
        return doc, {"backend": self.name, "replay_s": time.perf_counter() - start}

    def refined(self, video: Path, out_json: Optional[Path] = None):
        start = time.perf_counter()
        doc = self._dense(video)
        _write(doc, out_json)
        return doc, {"backend": self.name, "replay_s": time.perf_counter() - start}

    def refine(self, video: Path, doc: Mapping[str, Any], windows: Sequence[tuple[int, int]],
               out_json: Optional[Path] = None):
        if not self.sharpens_windows:
            raise NotApplicable("this analysis already measured every frame")
        start = time.perf_counter()
        dense = {frame["frame_index"]: frame for frame in self._dense(video)["frames"]}
        out = copy.deepcopy(dict(doc))
        added = []
        for first, last in windows:
            first, last = int(first), int(last)
            if last < first:
                raise ValueError("window end is inclusive and must be >= start")
            window_start = time.perf_counter()
            for frame in out["frames"]:
                index = frame["frame_index"]
                if first <= index <= last and index in dense:
                    players = copy.deepcopy(dense[index]["players"])
                    for player in players:
                        player.pop("src", None)
                    frame["players"] = players
            added.append({"start": first, "end": last, "s": time.perf_counter() - window_start})
        out["windows_refined"] = list(doc.get("windows_refined") or []) + added
        _write(out, out_json)
        return out, {"backend": self.name, "replay_s": time.perf_counter() - start}

    def _dense(self, video: Path) -> dict:
        stat = Path(video).stat()
        key = (str(Path(video).resolve()), stat.st_size, stat.st_mtime_ns)
        digest = self._hashes.get(key) or self._hashes.setdefault(key, sha256_file(Path(video)))
        if digest not in self.docs:
            raise RuntimeError("replay has no stored pose for this video")
        if digest not in self._parsed:
            stored = self.docs[digest]
            self._parsed[digest] = stored if isinstance(stored, dict) else json.loads(Path(stored).read_text("utf-8"))
        return copy.deepcopy(self._parsed[digest])
