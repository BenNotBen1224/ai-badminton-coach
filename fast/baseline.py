"""The clip-01 baseline (YOLO26s-pose@1280 + ByteTrack + far-court retry, then RTMPose-m on every
box) on any clip. Reuses comparison/run.py, so nothing is re-implemented. The far-court retry box and
the 'far player' pixel row were tuned by hand on the clip-01 camera; on another camera they are
re-derived from the auto-found court in the same proportions (on clip 01 this gives back the originals).

usage: python baseline.py VIDEO OUT.json      (run with comparison/.venv)
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "comparison"))
import run as cmp  # noqa: E402

# clip-01 retry box and far row, in units of the far-baseline-to-net height H and the net width W
Y0_H, Y1_H, X0_W, X1_W, FAR_H = -1.434, 1.192, -0.372, 0.336, 0.329


def fit_far_retry(video):
    from court import court_from_video
    from detect import COURT_M, Court
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    court = Court(court_from_video(video)[0], w, h)
    to_image = cv2.getPerspectiveTransform(COURT_M, court.lines)
    fl, fr, nl, nr = cv2.perspectiveTransform(np.float32([[[0, 0], [6.1, 0], [0, 6.7], [6.1, 6.7]]]), to_image)[0]
    far, net = (fl[1] + fr[1]) / 2, (nl[1] + nr[1]) / 2
    H, W = net - far, nr[0] - nl[0]
    cmp.FAR_REGION = (max(0, round(nl[0] + X0_W * W)), max(0, round(far + Y0_H * H)),
                      min(w, round(nr[0] + X1_W * W)), min(h, round(net + Y1_H * H)))
    cmp.FAR_BOTTOM = round(net + FAR_H * H)
    return {"far_region": cmp.FAR_REGION, "far_bottom": cmp.FAR_BOTTOM}


def main(video, out):
    from rtmlib import RTMPose
    t0 = time.perf_counter()
    retry = fit_far_retry(video)
    yolo = cmp.Yolo("crop")
    weight = cmp.checkpoint("rtmpose")
    pose = RTMPose(str(weight), model_input_size=(192, 256), backend="onnxruntime", device="cpu")
    pose.session = cmp.ort_session(weight)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i = len(frames)
        people = yolo(frame, i)
        players = []
        if people:
            xy, sc = pose(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), bboxes=[p["bbox_xyxy"] for p in people])
            players = [{"track_id": p["track_id"], "bbox_xyxy": p["bbox_xyxy"],
                        "box_confidence": p["box_confidence"],
                        "joints": cmp.joint_record(np.column_stack((pts, s)), .25)}
                       for p, pts, s in zip(people, xy, sc)]
        frames.append({"frame_index": i, "timestamp_seconds": round(i / fps, 4), "players": players})
    Path(out).write_text(json.dumps({"model": {"label": "baseline: YOLO26s-pose/ByteTrack + RTMPose-m"},
                                     "far_retry": retry, "frames": frames}), "utf-8")
    print(json.dumps({"frames": len(frames), **retry, "seconds": round(time.perf_counter() - t0, 1)}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
