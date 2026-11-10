"""Batched RTMPose (SimCC) on OpenVINO, matching rtmlib pre/post-processing.

Normalization and BGR->RGB run inside the compiled graph, so callers pass the
uint8 BGR crop produced by cv2.warpAffine.
"""
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
from openvino.preprocess import ColorFormat, PrePostProcessor

from ovcache import compile_model

INPUT_W, INPUT_H = 192, 256
_WEIGHTS = Path(__file__).resolve().parents[2] / "fast" / "weights"
DEFAULT_POSE = _WEIGHTS / "rtmpose-s-ft1.onnx"
DEFAULT_SMALL_POSE = _WEIGHTS / "rtmpose-t-ft3.onnx"
DEFAULT_SMALL_CONFIDENCE = 1.0
DEFAULT_SMALL_MIN_JOINT_CONFIDENCE = 0.65
MEAN = [123.675, 116.28, 103.53]
STD = [58.395, 57.12, 57.375]


def crop_transform(box, padding=1.25):
    """Box xyxy -> (2x3 warp matrix, center, aspect-fixed scale), as rtmlib."""
    x0, y0, x1, y1 = box
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
    w, h = (x1 - x0) * padding, (y1 - y0) * padding
    aspect = INPUT_W / INPUT_H
    scale = np.array([w, w / aspect] if w > h * aspect else [h * aspect, h], np.float32)
    s = INPUT_W / scale[0]
    warp = np.array([[s, 0, INPUT_W / 2 - s * center[0]],
                     [0, s, INPUT_H / 2 - s * center[1]]], np.float32)
    return warp, center, scale


def warp(frame, box):
    m, center, scale = crop_transform(box)
    return cv2.warpAffine(frame, m, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR), center, scale


def decode(simcc_x, simcc_y, center, scale):
    x = simcc_x.argmax(-1).astype(np.float32)
    y = simcc_y.argmax(-1).astype(np.float32)
    conf = 0.5 * (simcc_x.max(-1) + simcc_y.max(-1))
    xy = np.stack([x, y], -1) / 2.0
    xy = xy / np.array([INPUT_W, INPUT_H], np.float32) * scale + center - scale / 2
    return np.concatenate([xy, conf[:, None]], -1)


class Pose:
    def __init__(self, onnx: Path, hint="THROUGHPUT", streams=None, threads=None, core=None, dynamic=False):
        core = core or ov.Core()

        def build(model):
            model.reshape([-1 if dynamic else 1, 3, INPUT_H, INPUT_W])
            ppp = PrePostProcessor(model)
            ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC")) \
                .set_color_format(ColorFormat.BGR)
            ppp.input().model().set_layout(ov.Layout("NCHW"))
            ppp.input().preprocess().convert_element_type(ov.Type.f32) \
                .convert_color(ColorFormat.RGB).mean(MEAN).scale(STD)
            return ppp.build()

        config = {"PERFORMANCE_HINT": hint}
        if streams:
            config["NUM_STREAMS"] = str(streams)
        if threads:
            config["INFERENCE_NUM_THREADS"] = str(threads)
        key = ("pose", dynamic, INPUT_H, INPUT_W, MEAN, STD)
        self.compiled = compile_model(core, onnx, key, build, config)
        self.queue = ov.AsyncInferQueue(self.compiled)
        self.queue.set_callback(self._done)
        self.results = {}

    def _done(self, request, key):
        center, scale, tag = key
        sx = request.get_output_tensor(0).data[0]
        sy = request.get_output_tensor(1).data[0]
        self.results[tag] = decode(sx, sy, center, scale)

    def submit(self, crop, center, scale, tag):
        self.queue.start_async({0: crop[None]}, (center, scale, tag))

    def wait(self):
        self.queue.wait_all()
        out, self.results = self.results, {}
        return out
