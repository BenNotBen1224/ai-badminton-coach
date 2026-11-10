import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "footage" / "china-open-2025-ws-final.webm"
FFMPEG = ROOT / "tools" / "ffmpeg.exe"
PYTHON = Path(sys.executable)
RUNTIME = ROOT / "models" / "badminton-pipeline-repro" / "scripts" / "tracknet_runtime"
WEIGHTS = ROOT / "models" / "TrackNetV3" / "ckpts" / "TrackNet_best.pt"
OUT = HERE / "runs"
SPANS = {"space": (56.8, 65.8), "time": (125.6, 135.0), "options": (271.8, 286.0)}


def cut(name, start, end):
    clip = OUT / f"{name}.mp4"
    if not clip.exists():
        subprocess.run([str(FFMPEG), "-v", "error", "-n", "-ss", f"{start}", "-i", str(SOURCE), "-t", f"{end - start}",
                        "-vf", "fps=30,scale=1280:720", "-an", "-c:v", "libx264", "-crf", "12", "-preset", "fast",
                        str(clip)], check=True)
    return clip


def track(clip):
    csv = OUT / f"{clip.stem}_ball.csv"
    if csv.exists():
        return csv
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    started = time.time()
    subprocess.run([str(PYTHON), str(RUNTIME / "predict.py"), "--video_file", str(clip), "--tracknet_file", str(WEIGHTS),
                    "--save_dir", str(OUT), "--large_video", "--device", "cpu", "--batch_size", "8"],
                   check=True, env=env, cwd=RUNTIME)
    print(clip.stem, "tracked in", round(time.time() - started, 1), "s", flush=True)
    return csv


def main():
    OUT.mkdir(exist_ok=True)
    lease = ROOT / "work" / "cpu.lock"
    marker = {"owner": "shuttle-tracknet", "pid": os.getpid(), "id": str(uuid.uuid4())}
    with lease.open("x") as handle:
        json.dump(marker, handle)
    try:
        names = sys.argv[1:] or list(SPANS)
        for name in names:
            start, end = SPANS[name]
            track(cut(name, start, end))
        (OUT / "spans.json").write_text(json.dumps({"source": str(SOURCE), "fps": 30, "size": [1280, 720], "spans": SPANS}, indent=1), encoding="utf-8")
    finally:
        if json.loads(lease.read_text(encoding="utf-8")) != marker:
            raise RuntimeError("CPU lease ownership changed")
        lease.unlink()


if __name__ == "__main__":
    main()

