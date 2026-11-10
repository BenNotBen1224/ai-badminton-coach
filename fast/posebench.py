"""Run a pose model on the shared acquisition boxes (isolates the pose model).

usage: python posebench.py MODEL.onnx OUT.json
"""
import json
import sys
import time
from pathlib import Path

import cv2

from pose import Pose, warp

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "sample-videos" / "01-singles-rally.mp4"
ACQ = ROOT / "comparison" / "results" / "crop" / "acquisition.json"


def joints(k, threshold=0.25):
    return [[round(float(x), 2), round(float(y), 2), round(float(c), 4)] if c >= threshold else None
            for x, y, c in k]


def main(model, out):
    boxes = json.loads(ACQ.read_text("utf-8"))["frames"]
    t0 = time.perf_counter()
    pose = Pose(Path(model))
    t_load = time.perf_counter() - t0
    cap = cv2.VideoCapture(str(VIDEO))
    t1 = time.perf_counter()
    t_wait = 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for j, p in enumerate(boxes[i]["players"]):
            crop, c, s = warp(frame, p["bbox_xyxy"])
            pose.submit(crop, c, s, (i, j))
        i += 1
    tw = time.perf_counter()
    res = pose.wait()
    t_wait = time.perf_counter() - tw
    t_run = time.perf_counter() - t1
    frames = []
    for f in boxes:
        players = [{"track_id": p["track_id"], "bbox_xyxy": p["bbox_xyxy"], "box_confidence": p["box_confidence"],
                    "joints": joints(res[(f["frame_index"], j)])} for j, p in enumerate(f["players"])]
        frames.append({"frame_index": f["frame_index"], "players": players})
    Path(out).write_text(json.dumps({"model": str(model), "frames": frames}), "utf-8")
    n = sum(len(f["players"]) for f in boxes)
    print(json.dumps({"model": Path(model).name, "crops": n, "load_s": round(t_load, 2),
                      "decode+pose_s": round(t_run, 2), "tail_wait_s": round(t_wait, 2),
                      "ms_per_crop": round(t_run / n * 1000, 2)}))


if __name__ == "__main__":
    main(*sys.argv[1:3])
