"""Dev-only weak-joint candidate. Production defaults stay frozen during screening."""
import argparse
import json
import os
import sys

sys.modules.setdefault("openvino.tools.ovc", None)
import pipeline
from risk_features import crop_features, risk_value


def route_tail(self, crops, meta, boxes, tracks, threshold, risk=None, hold=8, joints=None, motion=0):
    flags = [not t.get("large_left", 0) for t in tracks]
    kps = self.pose_crops(crops, meta, flags)
    mean_weak = [flag and float(k[:, 2].mean()) < self.small_confidence
                 for flag, k in zip(flags, kps)]
    redo = []
    for i, (flag, k) in enumerate(zip(flags, kps)):
        if not flag:
            continue
        weak = k[:, 2] < threshold
        if joints is not None:
            allowed = weak.copy()
            allowed[:] = False
            allowed[joints] = True
            weak &= allowed
        tail = bool(weak.any())
        if tail and motion and "kp" in tracks[i]:
            delta = k[weak, :2] - tracks[i]["kp"][weak, :2]
            tail = bool((delta ** 2).sum(1).max() > (motion * (boxes[i][3] - boxes[i][1])) ** 2)
        if mean_weak[i] or (tail and (risk is None or risk_value(risk, crop_features(k, *meta[i])) > 0)):
            redo.append(i)
    for flag, track in zip(flags, tracks):
        if not flag:
            track["large_left"] -= 1
    if redo:
        strong = self.pose_crops([crops[i] for i in redo], [meta[i] for i in redo], [False] * len(redo))
        self.calls["fallback"] += len(redo)
        for i, k in zip(redo, strong):
            kps[i] = k
            if mean_weak[i]:
                tracks[i]["large_left"] = hold
    return kps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--tail", type=float, required=True)
    ap.add_argument("--risk", type=argparse.FileType("r"))
    ap.add_argument("--hold", type=int, default=8)
    ap.add_argument("--tail-joints", choices=["all", "arms", "wrists"], default="all")
    ap.add_argument("--start-small", action="store_true")
    ap.add_argument("--motion", type=float, default=0)
    ap.add_argument("--mean", type=float, default=1.05)
    a = ap.parse_args()
    risk = json.load(a.risk)["tree"] if a.risk else None
    joints = {"all": None, "arms": [7, 8, 9, 10], "wrists": [9, 10]}[a.tail_joints]
    pipeline.Segment.route_poses = lambda self, *args: route_tail(
        self, *args, threshold=a.tail, risk=risk, hold=a.hold, joints=joints, motion=a.motion)
    options = {"start_det": pipeline.WEIGHTS / "yolo26n-384x640.onnx"} if a.start_small else {}
    print(json.dumps(pipeline.run(a.video, a.out, small_confidence=a.mean, **options)), flush=True)
    os._exit(0)
