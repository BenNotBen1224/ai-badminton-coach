"""Copy of fast/judge.py with a start offset.

Frame i is kept when (i - start) % step == 0. --start 0 is the stored phase-0 judge.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

DET = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip"
POSE = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip")
MAX_OUT_M = 2.5


def kept(i, step=3, start=0):
    return (i - start) % step == 0


def by_court(court, boxes):
    best = {}
    for b in boxes:
        x, y = court.feet_metres(b)
        out = float(np.hypot(max(0, -x, x - 6.1), max(0, -y, y - 13.4)))
        if out > MAX_OUT_M:
            continue
        key = (out, -(b[2] - b[0]) * (b[3] - b[1]))
        role = "far" if y < 6.7 else "near"
        if role not in best or key < best[role][0]:
            best[role] = (key, b)
    return {role: b for role, (_, b) in best.items()}


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def main(video, roles_json, out, step=3, start=0):
    from rtmlib import RTMPose, YOLOX

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "comparison"))
    import run as cmp  # noqa: E402

    t0 = time.perf_counter()
    det = YOLOX(DET, model_input_size=(640, 640), score_thr=0.3, backend="onnxruntime", device="cpu")
    pose = RTMPose(POSE, model_input_size=(288, 384), backend="onnxruntime", device="cpu")
    court = guide = None
    if roles_json == "court":
        from court import court_from_video
        from detect import Court
        cap = cv2.VideoCapture(str(video))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        court = Court(court_from_video(video)[0], w, h)
    else:
        guide = json.loads(Path(roles_json).read_text("utf-8"))["frames"]
    cap = cv2.VideoCapture(str(video))
    frames, unmatched = [], 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        people = []
        if kept(i, step, start):
            boxes = [list(map(float, b)) for b in det(frame)]
            if court:
                chosen = by_court(court, boxes)
                unmatched += 2 - len(chosen)
            else:
                want = {p["role"]: p["bbox_xyxy"] for p in guide[i]["players"] if p.get("role")}
                chosen = {}
                for role, rb in want.items():
                    overlap, box = max(((iou(rb, b), b) for b in boxes), default=(0, None))
                    if overlap >= 0.3:
                        chosen[role] = box
                    else:
                        unmatched += 1
            if chosen:
                xy, sc = pose(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), bboxes=list(chosen.values()))
                for (role, b), pts, s in zip(chosen.items(), xy, sc):
                    joints = [None if r is None else r[:2] for r in cmp.joint_record(np.column_stack((pts, s)), .25)]
                    people.append({"role": role, "bbox_xyxy": b, "joints": joints, "sapiens": joints, "posewarper": joints})
        frames.append({"frame_index": i, "people": people})
        i += 1
    cap.release()
    Path(out).write_text(json.dumps({"judge": "YOLOX-x + RTMPose-x 384x288", "step": step,
                                     "frames": frames}), "utf-8")
    print(json.dumps({"frames": i, "judged": sum(bool(f["people"]) for f in frames),
                      "unmatched_roles": unmatched, "seconds": round(time.perf_counter() - t0, 1),
                      "start": start, "step": step}))


def cli(argv=None):
    ap = argparse.ArgumentParser(description="Phase-offset judge. Frame i is kept when (i - start) % step == 0.")
    ap.add_argument("video")
    ap.add_argument("roles_json")
    ap.add_argument("out")
    ap.add_argument("step_pos", nargs="?", type=int)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args(argv)
    step = args.step_pos if args.step_pos is not None else args.step
    main(args.video, args.roles_json, args.out, step, args.start)


if __name__ == "__main__":
    cli()
