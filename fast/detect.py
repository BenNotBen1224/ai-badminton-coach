"""YOLO26 (NMS-free) person detector on a court-derived region, OpenVINO."""
import numpy as np
import openvino as ov
from openvino.preprocess import ColorFormat, PrePostProcessor

from ovcache import compile_model, infer

import cv2


def point_in_polygon(pt, poly):
    x, y = pt
    inside = False
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


COURT_M = np.float32([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]])  # doubles court lines, net at 6.7
# where the two players' feet can be, in court metres: the region hand-tuned on the first camera
# (2.2 m behind the far baseline, widening to ~2.4 m outside the far sidelines), so it moves with the court
PLAY_AREA_M = np.float32([[-2.45, -2.15], [8.55, -2.15], [6.1, 13.4], [0, 13.4]])
HEADROOM_M = 4.15


class Court:
    """Court lines (far-left, far-right, near-right, near-left; 0..1) and the play area around them, in pixels."""

    def __init__(self, court_normalized, width, height, play_area=PLAY_AREA_M):
        self.lines = np.float32(np.float32(court_normalized) * [width, height])
        self.width, self.height = width, height
        self.to_court = cv2.getPerspectiveTransform(self.lines, COURT_M)
        to_image = cv2.getPerspectiveTransform(COURT_M, self.lines)
        self.poly = [tuple(p) for p in cv2.perspectiveTransform(np.float32(play_area)[None], to_image)[0].tolist()]
        (fl, fr, nr, nl) = self.poly
        # the detector looks this far above the far edge: a jump with racket raised, plus box slack
        per_metre = (fr[0] - fl[0]) / float(play_area[1][0] - play_area[0][0])
        player_h = per_metre * HEADROOM_M
        xs = [p[0] for p in self.poly]
        margin = 0.08 * (max(xs) - min(xs))
        self.roi = (max(0, int(min(xs) - margin)), max(0, int(min(fl[1], fr[1]) - player_h)),
                    min(width, int(max(xs) + margin)), height)

    def contains_feet(self, box, slack=0.0):
        foot = ((box[0] + box[2]) / 2, box[3] - slack * (box[3] - box[1]))
        return point_in_polygon(foot, self.poly)

    def feet_metres(self, box):
        foot = np.float32([[[(box[0] + box[2]) / 2, box[3]]]])
        return cv2.perspectiveTransform(foot, self.to_court)[0, 0]

    def side(self, box):
        return "far" if self.feet_metres(box)[1] < 6.7 else "near"


class Detector:
    def __init__(self, onnx, hint="THROUGHPUT", core=None, streams=None):
        core = core or ov.Core()

        def build(model):
            ppp = PrePostProcessor(model)
            ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC")) \
                .set_color_format(ColorFormat.BGR)
            ppp.input().model().set_layout(ov.Layout("NCHW"))
            ppp.input().preprocess().convert_element_type(ov.Type.f32) \
                .convert_color(ColorFormat.RGB).scale(255.0)
            return ppp.build()

        config = {"PERFORMANCE_HINT": hint}
        if streams:
            config["NUM_STREAMS"] = str(streams)
        self.compiled = compile_model(core, onnx, ("det", 255.0), build, config)
        _, self.h, self.w, _ = self.compiled.inputs[0].shape
        self.request = self.compiled.create_infer_request()

    def letterbox(self, image):
        ih, iw = image.shape[:2]
        r = min(self.w / iw, self.h / ih)
        nw, nh = round(iw * r), round(ih * r)
        canvas = np.full((self.h, self.w, 3), 114, np.uint8)
        dx, dy = (self.w - nw) // 2, (self.h - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, r, dx, dy

    def __call__(self, frame, roi, min_conf=0.1, request=None):
        x0, y0, x1, y1 = roi
        img, r, dx, dy = self.letterbox(frame[y0:y1, x0:x1])
        out = infer(request or self.request, img[None])[0][0]  # (300, 6): x1 y1 x2 y2 score cls
        keep = (out[:, 4] >= min_conf) & (out[:, 5] == 0)
        det = out[keep]
        boxes = (det[:, :4] - [dx, dy, dx, dy]) / r + [x0, y0, x0, y0]
        return [(b.tolist(), float(s)) for b, s in zip(boxes, det[:, 4])]
