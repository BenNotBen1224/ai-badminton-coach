"""Teacher labels for distilling RTMPose-s on badminton players.

For every sampled player box from a pipeline run, two large pose models (neither is
the judge's RTMPose-x body7) each look at the frame and its mirror image. The four
answers are averaged into one label per joint, with a spread that says how much the
teachers agree. A patch 1.5x larger than the student's crop is saved so training can
jitter the crop without re-reading the video.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from rtmlib import RTMPose

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fast"))
from pose import INPUT_H, INPUT_W, crop_transform  # noqa: E402

TEACHERS = {
    "rtmpose-x-halpe26": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                         "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip",
    "rtmw-x-wholebody": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
                        "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
}
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
CONTEXT = 1.5
PATCH_W, PATCH_H = int(INPUT_W * CONTEXT), int(INPUT_H * CONTEXT)


def patch_transform(box):
    """Frame -> patch affine: same center as the student crop, CONTEXT x its extent."""
    _, center, scale = crop_transform(box)
    s = PATCH_W / (scale[0] * CONTEXT)
    return np.array([[s, 0, PATCH_W / 2 - s * center[0]], [0, s, PATCH_H / 2 - s * center[1]]], np.float32)


def teach(models, frame, box):
    """4 x (17, 3) predictions in frame coordinates: each teacher, plain and mirrored."""
    w = frame.shape[1]
    mirror = cv2.flip(frame, 1)
    mbox = [w - 1 - box[2], box[1], w - 1 - box[0], box[3]]
    out = []
    for model in models:
        k, s = model(frame, bboxes=[box])
        out.append(np.concatenate([k[0, :17], s[0, :17, None]], -1))
        k, s = model(mirror, bboxes=[mbox])
        m = np.concatenate([k[0, :17], s[0, :17, None]], -1)[FLIP]
        m[:, 0] = w - 1 - m[:, 0]
        out.append(m)
    return np.stack(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="output .npz")
    ap.add_argument("runs", nargs="+", help="pipeline run JSONs (video path from train/clips)")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    models = [RTMPose(url, model_input_size=(288, 384), backend="onnxruntime", device="cpu")
              for url in TEACHERS.values()]
    patches, preds, meta = [], [], []
    t0 = time.perf_counter()
    for run_path in args.runs:
        run = json.loads(Path(run_path).read_text())
        video = Path(run_path).parents[1] / "clips" / run["source"]["file"]
        boxes = {f["frame_index"]: [(p["role"], p["bbox_xyxy"]) for p in f["players"] if p.get("role")]
                 for f in run["frames"] if f["frame_index"] % args.every == 0}
        cap = cv2.VideoCapture(str(video))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for role, box in boxes.get(idx, []):
                m = patch_transform(box)
                patches.append(cv2.warpAffine(frame, m, (PATCH_W, PATCH_H), flags=cv2.INTER_LINEAR))
                p = teach(models, frame, box)
                p[..., :2] = p[..., :2] @ m[:, :2].T + m[:, 2]
                preds.append(p)
                meta.append((video.stem, idx, role, box[3] - box[1]))
                if args.limit and len(meta) >= args.limit:
                    break
            if args.limit and len(meta) >= args.limit:
                break
            idx += 1
        print(f"{video.name}: {len(meta)} instances, {time.perf_counter() - t0:.0f}s", flush=True)
        if args.limit and len(meta) >= args.limit:
            break
    np.savez(args.out, patches=np.stack(patches), preds=np.stack(preds),
             clip=np.array([m[0] for m in meta]), frame=np.array([m[1] for m in meta]),
             role=np.array([m[2] for m in meta]), box_h=np.array([m[3] for m in meta], np.float32),
             teachers=np.array(list(TEACHERS)))
    print(f"saved {len(meta)} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
