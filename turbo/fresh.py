"""Byte-unique copies that still decode to the same pictures.

The returned path is the only public result. The proof is not optional: a copy
that does not decode to the same frames raises, and the caller cannot skip it.
"""
from __future__ import annotations

import hashlib
import secrets
import shutil
from pathlib import Path

import cv2
import numpy as np

_EVIDENCE = {}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _free_atom(payload):
    size = 8 + len(payload)
    return size.to_bytes(4, "big") + b"free" + payload


def _decoded(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    hashes = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hashes.append(hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest())
    cap.release()
    if not hashes:
        raise RuntimeError(f"no frames decoded from {path}")
    n = len(hashes)
    indexes = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    return n, {i: hashes[i] for i in indexes}


def _prove(source, copy):
    source_hash = sha256(source)
    copy_hash = sha256(copy)
    if source_hash == copy_hash:
        raise RuntimeError("fresh copy SHA256 matches the source")
    source_n, source_frames = _decoded(source)
    copy_n, copy_frames = _decoded(copy)
    if source_n != copy_n:
        raise RuntimeError(f"fresh copy decoded {copy_n} frames, source decoded {source_n}")
    if source_frames != copy_frames:
        raise RuntimeError("fresh copy frame checksums do not match the source")
    if len(source_frames) < min(5, source_n):
        raise RuntimeError("fresh copy proof did not check the required frames")
    return {
        "source_sha256": source_hash,
        "copy_sha256": copy_hash,
        "frame_count": source_n,
        "checked_frames": source_frames,
        "sha256_differs": True,
        "frames_match": True,
    }


def fresh_copy(video, dest_dir):
    video = Path(video)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{secrets.token_hex(16)}.mp4"
    shutil.copyfile(video, out)
    payload = secrets.token_bytes(32)
    with open(out, "ab") as f:
        f.write(_free_atom(payload))
    proof = _prove(video, out)
    proof["free_atom_bytes"] = 8 + len(payload)
    _EVIDENCE[str(out.resolve())] = proof
    return out


def evidence(path):
    return _EVIDENCE[str(Path(path).resolve())]
