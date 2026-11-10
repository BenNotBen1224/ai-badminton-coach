"""Post-training INT8 quantization of RTMPose with NNCF.

Calibration crops come from clips 02 and 03; clip 01 stays held out for scoring.

usage: python quantize.py [MODEL.onnx] [--ignore-head]
"""
import sys
from pathlib import Path

import cv2
import nncf
import numpy as np
import openvino as ov

from court import court_from_video
from detect import Court, Detector
from pose import INPUT_H, INPUT_W, MEAN, STD, warp

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights"
CLIPS = [HERE.parent / "sample-videos" / n for n in ("02-singles-dive.mp4", "03-womens-singles-rally.mp4")]


def calibration_crops(step=5):
    cache = HERE / "runs" / f"_calib{step}.npy"
    if cache.exists():
        return list(np.load(cache))
    det = Detector(WEIGHTS / "yolo26s-448x768.onnx")
    out = []
    for clip in CLIPS:
        cap = cv2.VideoCapture(str(clip))
        court = Court(court_from_video(clip)[0], int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                out += [warp(frame, b)[0] for b, _ in det(frame, court.roi, min_conf=0.25) if court.contains_feet(b)]
            i += 1
    np.save(cache, np.stack(out))
    return out


def to_input(crop):
    x = (crop[:, :, ::-1].astype(np.float32) - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype(np.float32)


# Quantizing either of these alone moves joints by 169 px / 23 px (qsens.py); every other layer < 2.1 px.
KEEP_FLOAT = ["Conv_51/WithoutBiases", "Conv_110/WithoutBiases"]

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src = Path(args[0]) if args else WEIGHTS / "rtmpose-s.onnx"
    crops = calibration_crops()
    print("calibration crops", len(crops))
    model = ov.Core().read_model(str(src))
    model.reshape([1, 3, INPUT_H, INPUT_W])
    preset = nncf.QuantizationPreset.MIXED if "--mixed" in sys.argv else nncf.QuantizationPreset.PERFORMANCE
    q = nncf.quantize(model, nncf.Dataset(crops, to_input), subset_size=len(crops), preset=preset,
                      ignored_scope=nncf.IgnoredScope(names=KEEP_FLOAT))
    dst = WEIGHTS / f"{src.stem}-int8{'-mixed' if '--mixed' in sys.argv else ''}.xml"
    ov.save_model(q, str(dst), compress_to_fp16=False)
    print("saved", dst)
