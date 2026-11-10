"""An independent, much heavier pose 'judge' for clips that have no hand-checked reference.

YOLOX-x finds people (a different detector family from both pipelines), RTMPose-x 384x288 (the
largest RTMPose body model) draws each player. The player boxes are chosen by overlap with the
role boxes of ROLES.json, or, with ROLES = court, independently of any pipeline: per half of the
auto-found court, the person whose feet are nearest to inside the court lines (ties: the bigger box;
nobody further than MAX_OUT_M outside). The judge's own detector box and skeleton are what get written.
Output uses the reference/consensus.json schema, so evaluate.py --ref can score against it.

usage: python judge.py VIDEO ROLES.json|court OUT.json [step=3]      (run with comparison/.venv)
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "comparison"))
import run as cmp  # noqa: E402

DET = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip"
POSE = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip")


MAX_OUT_M = 2.5


def by_court(court, boxes):
    """{role: box}: per side, the box whose feet are least far outside the court lines."""
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


def main(video, roles_json, out, step=3):
    from rtmlib import YOLOX, RTMPose
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
        if i % step == 0:
            boxes = [list(map(float, b)) for b in det(frame)]
            if court:
                chosen = by_court(court, boxes)
                unmatched += 2 - len(chosen)
            else:
                want = {p["role"]: p["bbox_xyxy"] for p in guide[i]["players"] if p.get("role")}
                chosen = {}
                for role, rb in want.items():
                    o, b = max(((iou(rb, b), b) for b in boxes), default=(0, None))
                    if o >= 0.3:
                        chosen[role] = b
                    else:
                        unmatched += 1
            if chosen:
                xy, sc = pose(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), bboxes=list(chosen.values()))
                for (role, b), pts, s in zip(chosen.items(), xy, sc):
                    j = [None if r is None else r[:2] for r in cmp.joint_record(np.column_stack((pts, s)), .25)]
                    people.append({"role": role, "bbox_xyxy": b, "joints": j, "sapiens": j, "posewarper": j})
        frames.append({"frame_index": i, "people": people})
        i += 1
    Path(out).write_text(json.dumps({"judge": "YOLOX-x + RTMPose-x 384x288", "step": step,
                                     "frames": frames}), "utf-8")
    print(json.dumps({"frames": i, "judged": sum(bool(f["people"]) for f in frames),
                      "unmatched_roles": unmatched, "seconds": round(time.perf_counter() - t0, 1)}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 3)
