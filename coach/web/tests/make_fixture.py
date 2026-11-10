"""Hand-built fixture bundle. Numbers are scripted, not a measured coaching result."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from coach.bundle import build_bundle
from coach.schema import (
    SCHEMA_VERSION,
    Basis,
    Category,
    Finding,
    Mark,
    Metric,
    Moment,
    Report,
    Role,
    Severity,
    Tier,
    Window,
    swing_event,
)
from coach.synth import Path as Move, PlayerPlan, Posture, Swing, script

OUT = PROJECT / "work" / "C2" / "fixture"
FFMPEG = PROJECT / "tools" / "ffmpeg.exe"
N = 60
FPS = 30.0
WIDTH = 1280
HEIGHT = 720
COCO = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
BONES = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)


def joint(doc, frame, role, name):
    row = next(p for p in doc["frames"][frame]["players"] if p["role"] == role)
    x, y, _conf = row["joints"][COCO.index(name)]
    return (float(x), float(y))


def trail(doc, frame, role, name, count, earliest):
    points = []
    for index in range(max(earliest, frame - count + 1), frame + 1):
        points.append(joint(doc, index, role, name))
    return tuple(points)


def paint(doc, index):
    image = np.full((HEIGHT, WIDTH, 3), (36, 92, 48), np.uint8)
    corners = np.array(doc["court_lines_normalized"], np.float32)
    corners[:, 0] *= WIDTH
    corners[:, 1] *= HEIGHT
    pts = np.round(corners).astype(np.int32)
    cv2.polylines(image, [pts], True, (236, 236, 236), 2, cv2.LINE_AA)
    mid = ((pts[0] + pts[1]) * 0.5 + (pts[2] + pts[3]) * 0.5) * 0.5
    net_a = ((pts[1] + pts[2]) * 0.5).astype(int)
    net_b = ((pts[0] + pts[3]) * 0.5).astype(int)
    cv2.line(image, tuple(net_a), tuple(net_b), (236, 236, 236), 2, cv2.LINE_AA)
    cv2.circle(image, tuple(np.round(mid).astype(int)), 3, (236, 236, 236), -1, cv2.LINE_AA)
    for player in doc["frames"][index]["players"]:
        color = (210, 210, 210) if player["role"] == "near" else (170, 190, 210)
        joints = player["joints"]
        for a, b in BONES:
            ax, ay, ac = joints[a]
            bx, by, bc = joints[b]
            if ac < 0.5 or bc < 0.5:
                continue
            cv2.line(image, (int(ax), int(ay)), (int(bx), int(by)), color, 6, cv2.LINE_AA)
        for x, y, conf in joints:
            if conf < 0.5:
                continue
            cv2.circle(image, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
    return image


def encode(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(int(FPS)), "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-g", "1", "-keyint_min", "1", "-bf", "0",
        "-movflags", "+faststart", str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for index in range(N):
        proc.stdin.write(paint(doc, index).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed for {path}")


def read_count(path):
    cap = cv2.VideoCapture(str(path))
    seen = 0
    while True:
        ok, _frame = cap.read()
        if not ok:
            break
        seen += 1
    cap.release()
    if seen != N:
        raise SystemExit(f"{path.name} has {seen} frames, expected {N}")


def build(doc):
    near = Role.NEAR
    issue_marks = (
        Mark("dot", 18, (joint(doc, 18, "near", "left_hip"),), "Fixture hip", 0.53, "actual"),
        Mark("label", 18, (joint(doc, 18, "near", "nose"),), "Fixture wait", None, "neutral"),
        Mark("line", 18, (joint(doc, 18, "near", "left_hip"), joint(doc, 18, "near", "right_hip")), "Fixture hips", None, "actual"),
        Mark("bracket", 18, (joint(doc, 18, "near", "left_ankle"), joint(doc, 18, "near", "right_ankle")), "Fixture width", None, "actual"),
        Mark("band", 18, (joint(doc, 18, "near", "left_shoulder"), joint(doc, 18, "near", "right_hip")), "Fixture zone", None, "target"),
        Mark("angle", 18, (
            joint(doc, 18, "near", "right_shoulder"),
            joint(doc, 18, "near", "right_elbow"),
            joint(doc, 18, "near", "right_wrist"),
        ), "Fixture angle", None, "actual"),
        Mark("trail", 18, trail(doc, 18, "near", "right_wrist", 8, 6), "Fixture path", None, "actual"),
    )
    best_marks = (
        Mark("dot", 48, (joint(doc, 48, "near", "left_hip"),), "Fixture best", 0.47, "actual"),
        Mark("label", 48, (joint(doc, 48, "near", "nose"),), "Fixture low", None, "neutral"),
    )
    stay = Finding(
        "technique.stay_lower", Category.TECHNIQUE, near, Severity.PRIORITY,
        "Fixture hips sit high in this clip",
        "Fixture text from a scripted pose. Not a measured coaching result.",
        (Metric(
            "hip_height_ratio", "Hip height", 0.53, "ratio", Basis.PROJECTED_2D, 0.82,
            (0.42, 0.48), "hip_height_ratio", 14, 0.47,
        ),),
        (
            Moment(near, 6, 18, 30, "Fixture issue", "issue", issue_marks),
            Moment(near, 36, 48, 56, "Fixture best", "best", best_marks),
        ),
        Tier.PREVIEW,
        ("left_hip", "right_hip", "left_knee", "right_knee"),
        ("hip_height_ratio",),
        ("lower_stance",),
        0.9,
    )
    foot = Finding(
        "footwork.recovery", Category.FOOTWORK, near, Severity.IMPROVE,
        "Fixture return to base",
        "Fixture text from a scripted pose. Not a measured coaching result.",
        (Metric(
            "recovery_time_s", "Return time", 1.12, "s", Basis.TIMING, 0.62,
            (0.6, 0.9), "base_distance", 8, 0.78,
        ),),
        (Moment(near, 8, 22, 32, "Fixture feet", "issue", (
            Mark("dot", 22, (joint(doc, 22, "near", "left_ankle"),), "Fixture foot", None, "actual"),
            Mark("line", 22, (joint(doc, 22, "near", "left_ankle"), joint(doc, 22, "near", "right_ankle")), "Fixture step", None, "neutral"),
        )),),
        Tier.PREVIEW,
        ("left_ankle", "right_ankle"),
        ("base_distance",),
        ("base_recovery",),
        0.5,
    )
    tactics = Finding(
        "tactics.base_discipline", Category.TACTICS, near, Severity.IMPROVE,
        "Fixture distance from base",
        "Fixture text from a scripted pose. Not a measured coaching result.",
        (Metric(
            "base_distance_m", "Distance from base", 0.9, "m", Basis.COURT_PLANE, 0.4,
            (0.3, 0.8), "base_distance", 6, None,
        ),),
        (Moment(near, 10, 26, 40, "Fixture spot", "issue", (
            Mark("dot", 26, (joint(doc, 26, "near", "left_ankle"),), "Fixture spot", None, "actual"),
            Mark("line", 26, (joint(doc, 26, "near", "left_ankle"), joint(doc, 26, "near", "right_ankle")), "Fixture base", None, "target"),
        )),),
        Tier.PREVIEW,
        ("left_ankle", "right_ankle"),
        ("court_x",),
        ("base_recovery",),
        0.4,
    )
    series = {
        "near": {
            "hip_height_ratio": tuple(round(0.5 + 0.03 * np.sin(i / 6.0), 3) for i in range(N)),
            "base_distance": tuple(round(0.7 + 0.15 * np.sin(i / 9.0), 3) for i in range(N)),
            "court_x": tuple(round(1.6 + 0.02 * i, 3) for i in range(N)),
            "court_y": tuple(round(10.2 - 0.01 * i, 3) for i in range(N)),
        },
        "far": {
            "hip_height_ratio": tuple(0.45 for _ in range(N)),
        },
    }
    report = Report(
        SCHEMA_VERSION,
        {"file": "clip.mp4", "width": WIDTH, "height": HEIGHT, "fps": FPS, "fixture": True},
        Tier.PREVIEW,
        N,
        FPS,
        (near, Role.FAR),
        {"near": stay.metrics + foot.metrics + tactics.metrics},
        series,
        (swing_event(near, 15, start=12, end=18, peak_frame=14, peak_speed=4.2, overhead=True),),
        (stay, foot, tactics),
        (Window(0, 12, "fixture"),),
        "fixture",
        0,
        ("preview_estimate", "few_swings"),
        tuple(i / FPS for i in range(N)),
    )
    exercises = {
        ("technique.stay_lower", "near"): ["wall_sit"],
        ("footwork.recovery", "near"): ["ladder_drill"],
        ("tactics.base_discipline", "near"): ["base_return"],
    }
    bundle = build_bundle(doc, report, video_url="clip.mp4", exercises=exercises)
    return bundle


def main():
    doc, _truth = script(
        n=N,
        fps=FPS,
        width=WIDTH,
        height=HEIGHT,
        near=PlayerPlan(
            court_xy=(1.6, 10.2),
            posture=Posture(hip_height_ratio=0.85, stance_width_m=0.42),
            path=Move(8, 28, (2.3, 9.6)),
            swing=Swing(16, 1.15, 150),
        ),
        far=PlayerPlan(court_xy=(4.6, 3.2), posture=Posture(hip_height_ratio=0.9, stance_width_m=0.34)),
    )
    bundle = build(doc)
    OUT.mkdir(parents=True, exist_ok=True)
    clip = OUT / "clip.mp4"
    encode(clip, doc)
    read_count(clip)
    (OUT / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    sync = dict(bundle)
    sync["video_url"] = "framecode.mp4"
    (OUT / "sync.json").write_text(json.dumps(sync), encoding="utf-8")
    segs = bundle["findings"][0]["timelines"][0]["segments"]
    slow = min((s for s in segs if s["kind"] == "play"), key=lambda s: s["speed"])
    freeze = next(s for s in segs if s["kind"] == "freeze" and s["show"] == "issue")
    compare = next(s for s in segs if s["show"] == "best")
    boxes = {}
    for frame in (18, 22, 26, 48):
        row = next(p for p in doc["frames"][frame]["players"] if p["role"] == "near")
        boxes[frame] = [round(v, 1) for v in row["bbox_xyxy"]]
    print(json.dumps({
        "slow": slow["speed"],
        "freeze": freeze["start_frame"],
        "compare": compare["start_frame"],
        "boxes": boxes,
        "exercises": bundle["findings"][0]["exercise_ids"],
    }))


if __name__ == "__main__":
    main()
