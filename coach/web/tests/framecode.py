"""60-frame 640x360 mp4 whose top-left blocks encode the frame index.

12 blocks, 6 columns by 2 rows, origin (32, 32), size 40, gap 8.
Bits 0-5 and bits 6-11 are the same index, least significant bit first.
White is 1, black is 0. Both copies must survive compression.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
FFMPEG = PROJECT / "tools" / "ffmpeg.exe"
OUT = PROJECT / "work" / "C2" / "fixture" / "framecode.mp4"

WIDTH = 640
HEIGHT = 360
FRAMES = 60
FPS = 30
ORIGIN = 32
SIZE = 40
GAP = 8
COLS = 6


def paint(index: int) -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), (32, 64, 32), np.uint8)
    for slot in range(12):
        bit = (index >> (slot % 6)) & 1
        col = slot % COLS
        row = slot // COLS
        x = ORIGIN + col * (SIZE + GAP)
        y = ORIGIN + row * (SIZE + GAP)
        image[y : y + SIZE, x : x + SIZE] = 255 if bit else 0
    return image


def encode(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(FFMPEG),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "8",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "1",
        "-keyint_min",
        "1",
        "-bf",
        "0",
        "-movflags",
        "+faststart",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for index in range(FRAMES):
        proc.stdin.write(paint(index).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed for {path}")


def read_back(path: Path) -> None:
    cap = cv2.VideoCapture(str(path))
    seen = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        bits = []
        for slot in range(12):
            col = slot % COLS
            row = slot // COLS
            x = ORIGIN + col * (SIZE + GAP) + SIZE // 2
            y = ORIGIN + row * (SIZE + GAP) + SIZE // 2
            bits.append(1 if int(frame[y, x, 2]) > 128 else 0)
        low = sum(bit << i for i, bit in enumerate(bits[:6]))
        high = sum(bit << i for i, bit in enumerate(bits[6:]))
        if low != high or low != seen:
            raise SystemExit(f"framecode mismatch at {seen}: decoded {low}/{high}")
        seen += 1
    cap.release()
    if seen != FRAMES:
        raise SystemExit(f"framecode wrote {seen} frames, expected {FRAMES}")


def main() -> None:
    encode(OUT)
    read_back(OUT)
    print(f"framecode {FRAMES} {OUT}")


if __name__ == "__main__":
    main()
