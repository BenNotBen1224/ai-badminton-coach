"""Replay stored dense pose documents. Identity is the first picture plus frame count, not the file hash."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

BUILTIN_VIDEOS = {
    "01": PROJECT / "sample-videos" / "01-singles-rally.mp4",
    "02": PROJECT / "sample-videos" / "02-singles-dive.mp4",
    "03": PROJECT / "sample-videos" / "03-womens-singles-rally.mp4",
    "04": PROJECT / "sample-videos" / "04-china-open-ws-a.mp4",
    "05": PROJECT / "sample-videos" / "05-china-open-ws-b.mp4",
    "08": PROJECT / "sample-videos" / "08-china-open-ws-f.mp4",
}


def dense_path(clip):
    if clip in {"01", "02", "03", "04", "05"}:
        return PROJECT / "fast" / "runs" / "step4-rebalanced-dev" / "outputs" / f"{clip}-r1.json"
    if clip == "08":
        return PROJECT / "fast" / "runs" / "step4-rebalanced-holdout08" / "outputs" / "08-r1.json"
    return HERE / "refs" / f"dense-{clip}.json"


def video_catalog():
    found = dict(BUILTIN_VIDEOS)
    path = HERE / "clips.json"
    if path.is_file():
        doc = json.loads(path.read_text(encoding="utf-8"))
        for clip, info in doc.get("clips", {}).items():
            found[clip] = PROJECT / info["file"]
    return found


def signature(path):
    """First decoded picture only. A fresh copy changes the file hash, not this picture."""
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read a frame from {path}")
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


class ReplayRunner:
    IS_REPLAY = True

    def __init__(self, models_dir):
        self.models_dir = Path(models_dir)
        self._identity = None

    def warm_up(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        ok = cap.read()[0]
        cap.release()
        if not ok:
            raise RuntimeError(f"warm-up video did not decode: {video_path}")

    def _index(self):
        if self._identity is None:
            index = {}
            for clip, video in video_catalog().items():
                if not Path(video).is_file():
                    continue
                index.setdefault(signature(video), []).append(clip)
            self._identity = index
        return self._identity

    def identify(self, video_path):
        hits = self._index().get(signature(video_path), [])
        if len(hits) != 1:
            raise RuntimeError(f"cannot identify {video_path}: matches {hits}")
        return hits[0]

    def run(self, video_path, work_dir):
        clip = self.identify(video_path)
        path = dense_path(clip)
        if not path.is_file():
            raise RuntimeError(f"no stored dense document for clip {clip}: {path}")
        doc = json.loads(path.read_text(encoding="utf-8"))
        if "tier" not in doc:
            doc["tier"] = "refined"
        return {"doc": doc, "timings": {"source": "replay", "clip": clip}}
