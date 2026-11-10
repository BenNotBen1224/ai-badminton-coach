"""Time the fast pipeline the way a user runs it: a fresh process per run, wall clock included.

usage: python timing.py OUT.csv [runs=5] [clip ...]   (run with comparison\\.venv; clips default to all)
"""
import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent / "sample-videos"
TIMEOUT = 60  # a run this slow has hung; record it and move on


def run(cmd):
    """stdout of cmd. On timeout, kill the whole tree: a venv python.exe is a launcher with the real python as its child."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = p.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
        p.communicate()
        raise
    if p.returncode:
        raise subprocess.CalledProcessError(p.returncode, cmd, out, err)
    return out


def main():
    out = Path(sys.argv[1])
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    clips = sys.argv[3:] or sorted(v.name[:2] for v in VIDEOS.glob("0*.mp4"))
    fields = ["clip", "run", "wall", "internal", "load", "court", "crops", "cuts", "status"]
    with tempfile.TemporaryDirectory() as tmp, open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for clip in clips:
            video = next(VIDEOS.glob(f"{clip}-*.mp4"))
            for r in range(1, runs + 1):
                row = dict.fromkeys(fields, "")
                row.update(clip=clip, run=r)
                t = time.perf_counter()
                try:
                    res = run([sys.executable, str(HERE / "pipeline.py"), str(video), str(Path(tmp) / "run.json")])
                    info = json.loads(res.strip().splitlines()[-1])
                    row.update(internal=info["total_s"], load=info["load_s"], court=info["court_s"],
                               crops=info["crops"], cuts=len(info["cuts"]), status="ok")
                except subprocess.TimeoutExpired:
                    row["status"] = "timeout"
                except subprocess.CalledProcessError as e:
                    row["status"] = f"error {e.returncode}"
                row["wall"] = round(time.perf_counter() - t, 2)
                w.writerow(row)
                f.flush()
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
