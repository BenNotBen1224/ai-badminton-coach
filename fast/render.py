"""Draw a run on its video. Players with a role are drawn near = magenta, far = yellow; runs
without roles (the baseline) draw every person in a per-track colour. --vs puts OTHER on the
left and RUN on the right at half size.

usage: python render.py VIDEO RUN.json OUT.mp4 [--label TEXT] [--vs OTHER.json --vs-label TEXT]
       (run with the project .venv)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from badminton_coach.__main__ import require_ffmpeg  # noqa: E402
from badminton_coach.overlay import PALETTE, assign_track_slots, draw_pose  # noqa: E402

ROLE_COLOR = {"near": PALETTE[0], "far": PALETTE[1]}


def draw(image, players, slots):
    out = image.copy()
    roles = any(p.get("role") for p in players) or slots is None
    for p in players:
        if roles:
            if not p.get("role"):
                continue
            tag, color = p["role"], ROLE_COLOR[p["role"]]
        else:
            tag = p["track_id"]
            color = PALETTE[slots[tag] % len(PALETTE)] if tag is not None else (215, 215, 215)
        draw_pose(out, tuple(p["bbox_xyxy"]), p["joints"], tag, color)
    return out


def banner(image, text, scale):
    cv2.rectangle(image, (0, 0), (image.shape[1], int(60 * scale)), (20, 20, 20), -1)
    cv2.putText(image, text, (int(16 * scale), int(42 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                0.9 * scale, (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--label", default="")
    ap.add_argument("--vs")
    ap.add_argument("--vs-label", default="")
    a = ap.parse_args()
    run = json.loads(Path(a.run).read_text("utf-8"))["frames"]
    other = json.loads(Path(a.vs).read_text("utf-8"))["frames"] if a.vs else None
    slots_run = None if any(p.get("role") for f in run for p in f["players"]) else assign_track_slots(run)
    slots_other = assign_track_slots(other) if other else None
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (w, h // 2) if other else (w, h)
    ffmpeg, ffprobe = require_ffmpeg()
    enc = subprocess.Popen([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                            "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "pipe:0",
                            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "21", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    for i, f in enumerate(run):
        ok, image = cap.read()
        if not ok:
            raise SystemExit(f"video ended at frame {i}")
        right = draw(image, f["players"], slots_run)
        t = f"{i / fps:5.2f}s"
        if other:
            left = draw(image, other[i]["players"], slots_other)
            halves = []
            for img, text in ((left, a.vs_label), (right, a.label)):
                img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                banner(img, f"{text}  {t}", 0.6)
                halves.append(img)
            frame = cv2.hconcat(halves)
        else:
            frame = right
            banner(frame, f"{a.label}  {t}", 1.0)
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    if enc.wait():
        raise SystemExit("ffmpeg failed")
    n = subprocess.run([str(ffprobe), "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0", a.out], capture_output=True, text=True).stdout
    print(json.dumps({"out": a.out, "frames_written": len(run), "frames_in_file": int(n.strip())}))


if __name__ == "__main__":
    main()
