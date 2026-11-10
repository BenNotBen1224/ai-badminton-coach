from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import cv2
import numpy as np

from badminton_coach.__main__ import (
    COCO17_NAMES, joint_record, point_in_polygon, probe_video, require_ffmpeg,
    validate_output,
)
from badminton_coach.overlay import annotate_frame, assign_track_slots

HERE = ROOT / "comparison"
RESULTS = HERE / "results"
WEIGHTS = HERE / "weights"
SOURCE = ROOT / "sample-videos" / "01-singles-rally.mp4"
ORIGINAL = ROOT / "outputs" / "01-singles-rally-20261026T011249Z"
POLYGON = [[.312, .371], [.677, .371], [.838, .894], [.148, .894]]
SDK = "https://download.openmmlab.com/mmpose/v1/projects"
METHODS = {
    "original": {
        "label": "Original YOLO26s-Pose + ByteTrack", "license": "AGPL-3.0",
        "checkpoint": "https://github.com/ultralytics/assets/releases",
        "input_size": 1280, "tracker": "ByteTrack",
        "notes": "Preserved original video and JSON; person cutoff 0.25, joint cutoff 0.25, tracker acquisition 0.1.",
    },
    "crop": {
        "acquisition_revision": 2,
        "label": "YOLO26s + targeted far-court crop retries", "license": "AGPL-3.0",
        "checkpoint": "local original yolo26s-pose.pt", "input_size": [1280, 640],
        "tracker": "ByteTrack for full-frame detections; added crop IDs unknown",
        "notes": "Retain full-frame detections >=0.25 unchanged; retry far region pixel[384,238,1498,713] when far detection (box bottom<600px) is missing or confidence<0.5. Replace only sub0.25 overlapping detections when crop confidence is higher. Common top-down acquisition also retains 0.1-0.25 boxes; this display uses >=0.25.",
    },
    "rtmpose": {
        "acquisition_revision": 2,
        "label": "Tracked crops + RTMPose-m ONNX", "license": "Apache-2.0 (MMPose/RTMLib)",
        "checkpoint": SDK + "/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
        "input_size": [192, 256], "tracker": "Common YOLO26s/ByteTrack + crop recovery",
        "notes": "Fresh RTMPose inference on every current-frame box acquired at >=0.1, before baseline display cutoff. Common acquisition time recorded separately. COCO17, BGR converted to RGB per checkpoint pipeline.json, affine padding 1.25.",
    },
    "rtmo": {
        "label": "RTMO-s one-stage ONNX", "license": "Apache-2.0 (MMPose/RTMLib)",
        "checkpoint": SDK + "/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip",
        "input_size": [640, 640], "tracker": "None; ID unknown",
        "notes": "Native detector and pose outputs; confidence 0.25, NMS 0.45, top-left letterbox. No substituted YOLO boxes.",
    },
    "vitpose": {
        "acquisition_revision": 2,
        "label": "Tracked crops + ViTPose++-small ONNX", "license": "Apache-2.0 upstream ViTPose; third-party ONNX export has no explicit license in its Hugging Face model-card metadata",
        "checkpoint": "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco/vitpose-s-coco.onnx",
        "input_size": [192, 256], "tracker": "Common YOLO26s/ByteTrack + crop recovery",
        "notes": "Actual small COCO17 export linked in RTMLib model zoo; fresh pose on boxes >=0.1. RGB per exporter, mean/std normalization, UDP/DARK decoding. Common acquisition time separate.",
    },
    "mediapipe": {
        "label": "MediaPipe Pose Landmarker Full", "license": "Apache-2.0",
        "checkpoint": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "input_size": "native task detector 224 and landmark 256", "tracker": "VIDEO mode internal tracking; exported IDs unknown",
        "native_layout": "MediaPipe33",
        "notes": "Up to 4 poses, detection/presence 0.25, tracking 0.5. Native 33 points retained; explicit COCO17 display mapping [0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28]. RGB input; confidence=min(visibility,presence).",
    },
    "movenet": {
        "label": "MoveNet MultiPose Lightning", "license": "Apache-2.0 (TensorFlow model)",
        "checkpoint": "https://www.kaggle.com/models/google/movenet/tensorFlow2/multipose-lightning/1",
        "input_size": [448, 256], "tracker": "None; ID unknown",
        "notes": "TensorFlow SavedModel on CPU, RGB int32 top-left padded to multiples of 32; up to 6 people. COCO17; instance and joint cutoff 0.25.",
    },
    "botsort": {
        "label": "YOLO26s-Pose + BoT-SORT", "license": "AGPL-3.0",
        "checkpoint": "local original yolo26s-pose.pt", "input_size": 1280,
        "tracker": "BoT-SORT, appearance ReID DISABLED",
        "notes": "Ultralytics botsort.yaml defaults; with_reid=false, sparseOptFlow GMC; acquisition0.1, display0.25, joint0.25. Not an appearance-ReID result.",
    },
    "posewarper": {
        "acquisition_revision": 2,
        "label": "Temporal PoseWarper", "license": "Apache-2.0 (reference repository and MMPose)",
        "checkpoint": "https://download.openmmlab.com/mmpose/top_down/posewarper/hrnet_w48_posetrack18_384x288_posewarper_stage2-4abf88db_20211130.pth",
        "input_size": [288, 384], "tracker": "Temporal learned feature warping",
        "native_layout": "PoseTrack17",
        "notes": "Real HRNetW48 + learned temporal neck, current,-2,-1,+1,+2 frames weighted .3,.1,.25,.25,.1, clamped only at source boundaries. CPU torchvision deform_conv2d, all2010 state entries strict. COCO display mapping [0,null,null,null,null,5,6,7,8,9,10,11,12,13,14,15,16]; missing face joints null, native pose preserved. Common acquired boxes>=0.1. RGB ImageNet normalization, current-box affine used on all five actual images; not pose smoothing.",
    },
    "sapiens": {
        "acquisition_revision": 2,
        "label": "Sapiens 0.3B Goliath308", "license": "CC-BY-NC-4.0",
        "checkpoint": "https://huggingface.co/facebook/sapiens-pose-0.3b-torchscript/resolve/main/sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2",
        "input_size": [768, 1024], "tracker": "Common YOLO26s/ByteTrack + crop recovery",
        "native_layout": "Goliath308",
        "notes": "Public official Goliath head, NOT COCO-trained. COCO17 display indices [0,1,2,3,4,5,6,7,8,62,41,9,10,11,12,13,14]. Native308 retained. FP32 TorchScript checkpoint; unchanged graph executed by verified OpenVINO CPU, with TorchScript CPU fallback. COCO17 checkpoint was gated (HTTP401), not bypassed. RGB, UDP affine at trained3:4 aspect, padding1.25, official UDP decoder; no demo hardcoded CUDA.",
    },
    "onnx": {
        "label": "Original YOLO26s exported to ONNX Runtime CPU", "license": "AGPL-3.0",
        "checkpoint": "Export of local original yolo26s-pose.pt", "input_size": 1280,
        "tracker": "ByteTrack",
        "notes": "Ultralytics portable ONNX export, dynamic=False, simplify=False, opset17, batch1; same detector and display thresholds as baseline. Fixed1280x1280 letterbox, whereas PyTorch uses rectangular padding for this clip; backend/input-padding comparison, not guaranteed bit-identical poses.",
    },
}
for _name, _config in METHODS.items():
    _config["device"] = "cpu"
    _config["execution_location"] = "local"
    _config.setdefault("native_layout", "COCO17")
    _config["display_layout"] = "COCO17"
    _config["joint_display_threshold"] = .25
    _config["person_acquisition_threshold"] = (
        .1 if _name in ("original", "crop", "rtmpose", "vitpose", "sapiens", "posewarper", "botsort", "onnx")
        else .25
    )
    _config["person_display_threshold"] = (
        .1 if _name in ("rtmpose", "vitpose", "sapiens", "posewarper") else .25
    )
    if _name in ("original", "crop", "botsort", "onnx"):
        _config["yolo_iou_parameter"] = .5
        _config["yolo_iou_note"] = "Passed to Ultralytics; YOLO26 end-to-end model path is not conventional NMS."
    if _name == "rtmo":
        _config["nms_iou_threshold"] = .45
    if _name == "mediapipe":
        _config["tracking_confidence_threshold"] = .5
        _config["person_display_threshold"] = None
        _config["person_display_threshold_note"] = "No detector score is exposed; visible-landmark-derived bbox court filter, visibility/presence>=0.25."


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def download(url, name):
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    target = WEIGHTS / name
    if not target.exists():
        started = time.perf_counter()
        print("Downloading", url, flush=True)
        temp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as response, temp.open("wb") as out:
            shutil.copyfileobj(response, out)
        temp.replace(target)
        save(WEIGHTS / f"{name}.download.json", {
            "url": url, "bytes": target.stat().st_size, "sha256": sha(target),
            "download_seconds": round(time.perf_counter() - started, 3),
        })
    return target


def checkpoint(method):
    url = METHODS[method]["checkpoint"]
    path = download(url, url.split("/")[-1])
    if path.suffix == ".zip":
        extracted = WEIGHTS / f"{method}.onnx"
        if not extracted.exists():
            with zipfile.ZipFile(path) as archive:
                files = [name for name in archive.namelist() if name.endswith(".onnx")]
                if len(files) != 1:
                    raise ValueError(f"Expected one ONNX model, got {files}")
                with archive.open(files[0]) as source, extracted.open("wb") as output:
                    shutil.copyfileobj(source, output)
        path = extracted
    return path


def inside(box):
    return point_in_polygon(
        ((box[0] + box[2]) / 2, box[3]),
        [(x * 1920, y * 1080) for x, y in POLYGON],
    )


def player(box, pose, confidence, track_id=None):
    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or not inside(box):
        return None
    pose = np.asarray(pose, dtype=float)
    return {
        "track_id": int(track_id) if track_id is not None else None,
        "bbox_xyxy": np.round(box, 2).tolist(),
        "box_confidence": None if confidence is None else round(float(confidence), 5),
        "joints": joint_record(pose, .25),
        "raw_joints": pose.tolist(),
    }


def result_players(result, threshold=.1, offset=(0, 0)):
    if result.boxes is None or result.keypoints is None:
        return []
    ids = result.boxes.id
    ids = ids.int().cpu().tolist() if ids is not None else [None] * len(result.boxes)
    output = []
    for box, conf, pose, track_id in zip(
        result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
        result.keypoints.data.cpu().numpy(), ids, strict=True,
    ):
        if conf < threshold:
            continue
        box = box + np.array([*offset, *offset])
        pose = pose.copy()
        pose[:, :2] += offset
        record = player(box, pose, conf, track_id)
        if record:
            output.append(record)
    return output


def overlap(first, second):
    a, b = np.asarray(first), np.asarray(second)
    wh = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = np.prod(wh)
    return intersection / max(1, np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection)


FAR_BOTTOM = 600  # a box whose bottom is above this pixel row is the far player (clip-01 camera)
FAR_REGION = (384, 238, 1498, 713)  # where the far player can be (clip-01 camera)


def needs_far_retry(records):
    far = [p for p in records if p["bbox_xyxy"][3] < FAR_BOTTOM]
    return not far or max(p["box_confidence"] for p in far) < .5


class Yolo:
    def __init__(self, method):
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4)
        self.method = method
        weight = ROOT / "yolo26s-pose.pt"
        if method == "onnx":
            weight = WEIGHTS / "yolo26s-pose.onnx"
            if not weight.exists():
                copied = WEIGHTS / "yolo26s-pose.pt"
                shutil.copy2(ROOT / copied.name, copied)
                YOLO(str(copied)).export(
                    format="onnx", imgsz=1280, opset=17, simplify=False,
                    dynamic=False, device="cpu", batch=1,
                )
        self.model = YOLO(str(weight), task="pose")
        self.weight = weight
        self.retry = YOLO(str(ROOT / "yolo26s-pose.pt")) if method == "crop" else None

    def __call__(self, frame, index):
        result = self.model.track(
            frame, persist=True, tracker="botsort.yaml" if self.method == "botsort" else "bytetrack.yaml",
            device="cpu", imgsz=1280, conf=.1, iou=.5, classes=[0], verbose=False,
        )[0]
        self.tracker_config = vars(self.model.predictor.trackers[0].args)
        records = result_players(result)
        if self.retry is not None:
            if needs_far_retry(records):
                x1, y1, x2, y2 = FAR_REGION
                extra = result_players(self.retry.predict(
                    frame[y1:y2, x1:x2], imgsz=640, conf=.1, iou=.5,
                    device="cpu", verbose=False,
                )[0], offset=(x1, y1))
                for candidate in extra:
                    matches = [p for p in records if overlap(p["bbox_xyxy"], candidate["bbox_xyxy"]) > .3]
                    if matches:
                        best = max(matches, key=lambda p: p["box_confidence"])
                        if best["box_confidence"] < .25 and candidate["box_confidence"] > best["box_confidence"]:
                            candidate["track_id"] = best["track_id"]
                            records.remove(best)
                            records.append(candidate)
                    else:
                        records.append(candidate)
            return records
        return [p for p in records if p["box_confidence"] >= .25]


def ort_session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


class TopDown:
    def __init__(self, method):
        from rtmlib import RTMPose, ViTPose
        self.boxes = load(RESULTS / "crop" / "acquisition.json")["frames"]
        cls = RTMPose if method == "rtmpose" else ViTPose
        self.weight = checkpoint(method)
        self.model = cls(str(self.weight), model_input_size=(192, 256), backend="onnxruntime", device="cpu")
        self.model.session = ort_session(self.weight)

    def __call__(self, frame, index):
        people = self.boxes[index]["players"]
        if not people:
            return []
        xy, scores = self.model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                bboxes=[p["bbox_xyxy"] for p in people])
        if xy.shape != (len(people), 17, 2) or scores.shape != (len(people), 17):
            raise ValueError(f"Unexpected top-down output: {xy.shape}, {scores.shape}")
        return [player(p["bbox_xyxy"], np.column_stack((pts, score)),
                       p["box_confidence"], p["track_id"])
                for p, pts, score in zip(people, xy, scores, strict=True)]


class Rtmo:
    def __init__(self):
        from rtmlib import RTMO
        self.weight = checkpoint("rtmo")
        self.model = RTMO(str(self.weight), model_input_size=(640, 640),
                          score_thr=.25, nms_thr=.45, backend="onnxruntime", device="cpu")
        self.model.session = ort_session(self.weight)

    def __call__(self, frame, index):
        from rtmlib.tools.object_detection.post_processings import multiclass_nms
        image, ratio = self.model.preprocess(frame)
        detections, poses = self.model.inference(image)
        boxes, confidence = detections[0, :, :4] / ratio, detections[0, :, 4]
        _, keep = multiclass_nms(boxes, confidence[:, None], nms_thr=.45, score_thr=.25)
        if keep is None:
            return []
        result = []
        for i in keep:
            pose = poses[0, i].copy()
            pose[:, :2] /= ratio
            record = player(boxes[i], pose, confidence[i])
            if record:
                result.append(record)
        return result


class MediaPipe:
    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        self.weight = checkpoint("mediapipe")
        self.model = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self.weight)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=4,
                min_pose_detection_confidence=.25, min_pose_presence_confidence=.25,
                min_tracking_confidence=.5,
            )
        )

    def __call__(self, frame, index):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.model.detect_for_video(image, round(index * 1000 / 30))
        records = []
        mapping = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        for landmarks in result.pose_landmarks:
            raw = np.asarray([[p.x * 1920, p.y * 1080, min(p.visibility, p.presence)] for p in landmarks])
            valid = raw[raw[:, 2] >= .25]
            if not len(valid):
                continue
            box = [*valid[:, :2].min(axis=0), *valid[:, :2].max(axis=0)]
            record = player(box, raw[mapping], None)
            if record:
                record["native_joints_33"] = raw.tolist()
                record["bbox_kind"] = "visible-landmark extent, not detector confidence"
                records.append(record)
        return records


class MoveNet:
    def __init__(self):
        import tensorflow as tf
        self.tf = tf
        self.weight = WEIGHTS / "movenet-multipose"
        self.model = tf.saved_model.load(str(self.weight)).signatures["serving_default"]

    def __call__(self, frame, index):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (448, 252))
        image = np.pad(image, ((0, 4), (0, 0), (0, 0)))
        outputs = self.model(self.tf.convert_to_tensor(image[None], dtype=self.tf.int32))
        raw = next(iter(outputs.values())).numpy()
        if raw.shape != (1, 6, 56):
            raise ValueError(f"MoveNet expected [1,6,56], got {raw.shape}")
        result = []
        for row in raw[0]:
            if row[55] < .25:
                continue
            pose = row[:51].reshape(17, 3)[:, [1, 0, 2]].copy()
            pose[:, 0] *= 1920
            pose[:, 1] *= 256 * 1080 / 252
            box = row[51:55][[1, 0, 3, 2]] * [1920, 256 * 1080 / 252, 1920, 256 * 1080 / 252]
            record = player(box, pose, row[55])
            if record:
                result.append(record)
        return result


class Sapiens:
    def __init__(self):
        import torch
        self.torch = torch
        torch.set_num_threads(8)
        self.weight = download(METHODS["sapiens"]["checkpoint"], "sapiens_0.3b_goliath_torchscript.pt2")
        if (WEIGHTS / "sapiens-port.json").exists():
            import openvino as ov
            self.model = ov.Core().compile_model(str(WEIGHTS / "sapiens.xml"), "CPU", {
                "INFERENCE_NUM_THREADS": 8, "NUM_STREAMS": 1, "INFERENCE_PRECISION_HINT": "f32",
            })
            self.backend = "OpenVINO FP32 CPU (verified unchanged TorchScript graph)"
        else:
            self.model = torch.jit.load(str(self.weight), map_location="cpu").float().eval()
            self.backend = "TorchScript FP32 CPU"
        self.boxes = load(RESULTS / "crop" / "acquisition.json")["frames"]
        sys.path.insert(0, str(WEIGHTS))
        import sapiens_pose_utils
        self.utils = sapiens_pose_utils

    def __call__(self, frame, index):
        records = []
        mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
        for p in self.boxes[index]["players"]:
            box = np.asarray(p["bbox_xyxy"], dtype=np.float32)
            center = (box[:2] + box[2:]) / 2
            scale = (box[2:] - box[:2]) * 1.25
            if scale[0] / scale[1] > .75:
                scale[1] = scale[0] / .75
            else:
                scale[0] = scale[1] * .75
            warp = self.utils.get_udp_warp_matrix(center, scale, 0, (768, 1024))
            image = cv2.warpAffine(frame, warp, (768, 1024), flags=cv2.INTER_LINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image = (image - [123.5, 116.5, 103.5]) / [58.5, 57., 57.5]
            image = image.transpose(2, 0, 1).astype(np.float32)[None]
            if self.backend.startswith("OpenVINO"):
                heatmaps = self.model(image)[self.model.output(0)]
            else:
                tensor = self.torch.from_numpy(image)
                with self.torch.inference_mode():
                    heatmaps = self.model(tensor).cpu().numpy()
            if heatmaps.shape != (1, 308, 256, 192):
                raise ValueError(f"Expected Sapiens Goliath [1,308,256,192], got {heatmaps.shape}")
            xy, scores = self.utils.udp_decode(heatmaps[0], (768, 1024), (192, 256))
            xy = xy[0] / [768, 1024] * scale + center - .5 * scale
            raw = np.column_stack((xy, scores[0]))
            record = player(box, raw[mapping], p["box_confidence"], p["track_id"])
            record["native_joints_308"] = raw.tolist()
            records.append(record)
        return records


def header(method):
    return {
        "schema_version": 2,
        "source": {"file": SOURCE.name, "width": 1920, "height": 1080,
                   "fps": {"numerator": 30, "denominator": 1}, "frame_count": 780,
                   "sha256": sha(SOURCE)},
        "model": {**METHODS[method], "device": "cpu", "display_layout": "COCO17",
                  "keypoint_names": COCO17_NAMES, "joint_threshold": .25},
        "court_region_normalized": POLYGON,
    }


def render(method, data):
    ffmpeg, ffprobe = require_ffmpeg()
    folder = RESULTS / method
    target = folder / "annotated.mp4"
    temporary = folder / "rendering.mp4"
    slots = assign_track_slots(data["frames"])
    capture = cv2.VideoCapture(str(SOURCE))
    command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x1080", "-r", "30",
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "4", "-preset",
               "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
    with (folder / "encode.log").open("wb") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for record in data["frames"]:
                ok, image = capture.read()
                if not ok:
                    raise ValueError("Source ended before pose records")
                annotated = annotate_frame(image, record["players"], slots)
                index = record["frame_index"]
                cv2.rectangle(annotated, (0, 0), (1920, 68), (20, 20, 20), -1)
                cv2.putText(annotated, f"{METHODS[method]['label']} | {index / 30:.2f}s",
                            (18, 44), cv2.FONT_HERSHEY_SIMPLEX, .85, (255, 255, 255), 2, cv2.LINE_AA)
                process.stdin.write(annotated.tobytes())
                if index in (240, 450, 660):
                    cv2.imwrite(str(folder / f"frame-{index // 30}s.jpg"), annotated)
            if capture.read()[0]:
                raise ValueError("Source has excess frames")
        finally:
            capture.release()
            process.stdin.close()
            code = process.wait()
        if code:
            raise RuntimeError(f"FFmpeg failed: see {folder / 'encode.log'}")
    validate_output(temporary, ffprobe, 1920, 1080, 780)
    temporary.replace(target)


def validate_data(data):
    frames = data["frames"]
    if [f["frame_index"] for f in frames] != list(range(780)):
        raise ValueError("Expected all 780 ordered frame records")
    for frame in frames:
        for p in frame["players"]:
            if p is None or len(p["joints"]) != 17:
                raise ValueError("Invalid COCO17 player")
            raw = p.get("raw_joints")
            if raw is not None and (np.asarray(raw).shape != (17, 3) or not np.isfinite(raw).all()):
                raise ValueError("Invalid original-frame joints")


def update_manifest(method, **values):
    path = HERE / "manifest.json"
    manifest = load(path) if path.exists() else {
        "source": header("original")["source"],
        "caveats": "No labeled ground truth. Counts and confidence are not accuracy. Timings are single-run elapsed times, not general benchmarks. Setup/download, shared detector acquisition, pose inference and render are distinct.",
        "methods": {key: {"status": "pending", **config} for key, config in METHODS.items()},
    }
    for key, config in METHODS.items():
        manifest["methods"][key].update(config)
    manifest["methods"]["original"].pop("elapsed_original_inference_and_render_seconds", None)
    if values.get("status") in ("running", "complete"):
        manifest["methods"][method].pop("error", None)
    manifest["methods"][method].update(values)
    save(path, manifest)


def original():
    folder = RESULTS / "original"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("annotated.mp4", "keypoints.json", "summary.json"):
        if not (folder / name).exists():
            shutil.copy2(ORIGINAL / name, folder / name)
    data = load(folder / "keypoints.json")
    validate_data(data)
    _, probe = require_ffmpeg()
    validate_output(folder / "annotated.mp4", probe, 1920, 1080, 780)
    if sha(folder / "annotated.mp4") != sha(ORIGINAL / "annotated.mp4"):
        raise ValueError("Original video copy differs")
    capture = cv2.VideoCapture(str(folder / "annotated.mp4"))
    for second in (8, 15, 22):
        capture.set(cv2.CAP_PROP_POS_FRAMES, second * 30)
        ok, image = capture.read()
        if not ok:
            raise ValueError("Original hard-frame extraction failed")
        cv2.imwrite(str(folder / f"frame-{second}s.jpg"), image)
    capture.release()
    update_manifest("original", status="complete", frame_count=780, duration_seconds=26,
                    preserved_video_sha256=sha(folder / "annotated.mp4"),
                    elapsed_original_detection_and_drawing_excludes_final_encoding_seconds=load(folder / "summary.json")["elapsed_seconds"])


def run(method):
    folder = RESULTS / method
    folder.mkdir(parents=True, exist_ok=True)
    data_file = folder / "keypoints.json"
    top_down = method in ("rtmpose", "vitpose", "sapiens", "posewarper")
    acquisition_file = RESULTS / "crop" / "acquisition.json"
    acquisition_sha = sha(acquisition_file) if top_down and acquisition_file.exists() else None
    if method in ("crop", "rtmpose", "vitpose", "sapiens", "posewarper") and data_file.exists():
        existing = load(data_file)
        if (existing["model"].get("acquisition_revision") != METHODS[method]["acquisition_revision"]
                or (top_down and existing.get("shared_acquisition_sha256") != acquisition_sha)):
            archive = HERE / "attempts" / f"{method}-{time.time_ns()}"
            archive.parent.mkdir(parents=True, exist_ok=True)
            folder.rename(archive)
            folder.mkdir()
    if method == "original":
        original()
        return
    if method in ("rtmpose", "vitpose", "sapiens", "posewarper") and not (RESULTS / "crop" / "acquisition.json").exists():
        raise RuntimeError("Run crop first to acquire current-frame boxes >=0.1")
    update_manifest(method, status="running")
    gallery()
    try:
        if data_file.exists():
            data = load(data_file)
            validate_data(data)
        else:
            started = time.perf_counter()
            if method in ("crop", "botsort", "onnx", "sapiens", "posewarper"):
                import torch
                torch.set_num_threads(8 if method in ("sapiens", "posewarper") else 4)
            if method in ("crop", "botsort", "onnx"):
                model = Yolo(method)
            elif method in ("rtmpose", "vitpose"):
                model = TopDown(method)
            elif method == "rtmo":
                model = Rtmo()
            elif method == "mediapipe":
                model = MediaPipe()
            elif method == "movenet":
                model = MoveNet()
            elif method == "sapiens":
                model = Sapiens()
            elif method == "posewarper":
                from comparison.posewarper import PoseWarper
                model = PoseWarper()
            else:
                raise RuntimeError(f"{method} adapter awaiting verified portable checkpoint/API")
            setup_seconds = time.perf_counter() - started
            capture = cv2.VideoCapture(str(SOURCE))
            data = header(method)
            if top_down:
                data["shared_acquisition_sha256"] = acquisition_sha
            frames = []
            journal = folder / "frames.jsonl"
            resumable = method in ("rtmpose", "vitpose", "rtmo", "movenet", "sapiens", "posewarper")
            if resumable and journal.exists():
                lines = journal.read_text(encoding="utf-8").splitlines()
                for line_index, line in enumerate(lines):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        if line_index != len(lines) - 1:
                            raise
                        print(f"Discarding incomplete final journal line {line_index}", flush=True)
                        break
                    if record["frame_index"] != len(frames):
                        raise ValueError("Cannot resume out-of-order journal")
                    frames.append(record)
                if frames:
                    print(f"Resuming {method} at frame {len(frames)}", flush=True)
                    capture.set(cv2.CAP_PROP_POS_FRAMES, len(frames))
                    journal.write_text("".join(json.dumps(f, allow_nan=False) + "\n" for f in frames), encoding="utf-8")
            progress_file = folder / "progress.json"
            previous_seconds = load(progress_file).get("inference_seconds", 0) if frames and progress_file.exists() else 0
            started = time.perf_counter()
            try:
                with journal.open("a" if frames else "w", encoding="utf-8") as stream:
                    for index in range(len(frames), 780):
                        ok, image = capture.read()
                        if not ok:
                            raise ValueError(f"Source ended at {index}")
                        people = model(image, index)
                        if method == "posewarper":
                            for p in people:
                                p.pop("raw_joints", None)
                        record = {"frame_index": index, "timestamp_seconds": index / 30,
                                  "players": people}
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        stream.flush()
                        frames.append(record)
                        if resumable or index % 60 == 0:
                            save(progress_file, {"frames": index + 1,
                                                "inference_seconds": previous_seconds + time.perf_counter() - started})
                        if index % 60 == 0:
                            print(f"{method}: {index}/780 elapsed {time.perf_counter() - started:.1f}s", flush=True)
                    if capture.read()[0]:
                        raise ValueError("Source has more than 780 frames")
            finally:
                capture.release()
            inference_seconds = previous_seconds + time.perf_counter() - started
            data["frames"] = frames
            if method == "crop":
                save(folder / "acquisition.json", data)
                data = deepcopy(data)
                for frame in data["frames"]:
                    frame["players"] = [p for p in frame["players"] if p["box_confidence"] >= .25]
            validate_data(data)
            save(data_file, data)
            update_manifest(method, setup_and_download_seconds=round(setup_seconds, 3),
                            inference_seconds=round(inference_seconds, 3),
                            checkpoint_path=str(model.weight.relative_to(ROOT)),
                            checkpoint_sha256=sha(model.weight) if model.weight.is_file() else None,
                            environment={"python": sys.version, "executable": sys.executable,
                                         "numpy": np.__version__, "opencv": cv2.__version__},
                            cpu_inference_threads=(
                                "native library default; CPU-only" if method in ("mediapipe", "onnx")
                                else 8 if method in ("sapiens", "posewarper") else 4
                            ),
                            packages={package: importlib.metadata.version(package)
                                      for package in ("numpy", "opencv-python", "torch", "ultralytics",
                                                      "onnxruntime", "rtmlib", "mediapipe")
                                      if importlib.util.find_spec(package.replace("-", "_")) is not None})
            if hasattr(model, "backend"):
                update_manifest(method, backend=model.backend)
            if hasattr(model, "tracker_config"):
                update_manifest(method, actual_tracker_config=model.tracker_config)
            if hasattr(model, "close"):
                model.close()
            del model
        started = time.perf_counter()
        rendered = not (folder / "annotated.mp4").exists()
        if rendered:
            render(method, data)
        _, ffprobe = require_ffmpeg()
        validate_output(folder / "annotated.mp4", ffprobe, 1920, 1080, 780)
        update_manifest(method, status="complete", frame_count=780, duration_seconds=26,
                        **{("render_and_verify_seconds" if rendered else "last_cached_video_verify_seconds"):
                           round(time.perf_counter() - started, 3)},
                        video_sha256=sha(folder / "annotated.mp4"),
                        people_per_frame=dict(Counter(len(f["players"]) for f in data["frames"])),
                        hard_frame_players={str(t): len(data["frames"][t * 30]["players"]) for t in (8, 15, 22)})
        print(f"COMPLETE {method}: {folder}", flush=True)
    except Exception as error:
        (folder / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        update_manifest(method, status="failed", error=str(error),
                        failure_log=f"results/{method}/failure.txt")
        raise


def gallery():
    manifest = load(HERE / "manifest.json")
    attribution_source = ROOT / "sample-videos" / "source-attribution.json"
    attribution_copy = HERE / "sample-videos" / "source-attribution.json"
    attribution_copy.parent.mkdir(parents=True, exist_ok=True)
    if not attribution_copy.exists() or sha(attribution_copy) != sha(attribution_source):
        shutil.copy2(attribution_source, attribution_copy)
    attribution_data = load(attribution_copy)
    credit = attribution_data["attribution"]
    source_clip = next(clip for clip in attribution_data["clips"] if clip["file"] == SOURCE.name)
    changes = "These derivative videos add AI skeletons and model labels to the silent, trimmed test clip."
    manifest["footage_attribution"] = {
        "creator": credit["creator"], "source_title": credit["source_title"],
        "source_url": source_clip["source_url"], "license": "CC-BY-4.0",
        "license_url": credit["cc_by_url_linked_by_youtube"],
        "derivative_changes": changes, "no_endorsement": credit["endorsement"],
        "original_attribution_copy": "sample-videos/source-attribution.json",
        "attribution_sha256": sha(attribution_copy),
    }
    for key, config in METHODS.items():
        manifest["methods"][key].update(config)
    save(HERE / "manifest.json", manifest)
    cards = []
    for method, config in manifest["methods"].items():
        label = html.escape(config["label"])
        notes = html.escape(config["notes"])
        status = html.escape(config["status"])
        details = html.escape(json.dumps(config, indent=2))
        if config["status"] == "complete":
            version = config.get("video_sha256", config.get("preserved_video_sha256", ""))[:20]
            suffix = f"?v={version}"
            media = f'<div class="playback-error" role="alert" hidden></div><video controls preload="metadata" playsinline src="results/{method}/annotated.mp4{suffix}" poster="results/{method}/frame-8s.jpg{suffix}"></video>'
            images = "".join(f'<a href="results/{method}/frame-{t}s.jpg{suffix}"><img loading="lazy" src="results/{method}/frame-{t}s.jpg{suffix}" alt="{label} at {t}s"><span>{t}s</span></a>' for t in (8, 15, 22))
            media += f'<div class="frames">{images}</div><a href="results/{method}/keypoints.json">Per-frame data</a>'
            if "people_per_frame" in config:
                empty = int(config["people_per_frame"].get("0", 0))
                media += f'<p>Frames with any on-court output: <strong>{780 - empty}/780</strong>. Missing detections remain missing. This is output availability, not pose accuracy.</p>'
        else:
            progress = ("Inference running locally; saving per-frame checkpoints. The video appears only after all780 frames are rendered and verified."
                        if config["status"] == "running" else "Queued; not yet executed.")
            media = f'<div class="blocked">{html.escape(config.get("error", progress))}</div>'
        cards.append(f'<article><h2>{label}</h2><b class="{status}">{status.upper()}</b>{media}<p>{notes}</p><details><summary>Model, license, configuration and measured time</summary><pre>{details}</pre></details></article>')
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Badminton pose comparison</title><style>
*{box-sizing:border-box}body{margin:0;background:#10141e;color:#e8eef7;font:16px system-ui}header,footer{padding:28px;max-width:1200px}footer{border-top:1px solid #35455c}
h1{font-size:32px;margin:0 0 12px}p{line-height:1.5;color:#b8c5d8}nav{position:sticky;top:0;z-index:2;padding:14px 28px;background:#202c40;display:flex;flex-wrap:wrap;gap:12px;align-items:center}
button,a{color:#8edfff}button{background:#10243c;border:1px solid #6388ab;padding:8px 18px;border-radius:6px;cursor:pointer}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:20px;padding:24px}article{background:#1b2332;border:1px solid #35455c;border-radius:12px;padding:18px}
h2{font-size:20px;margin:0 0 12px}b{display:block;margin:8px 0}.complete{color:#88f2b0}.blocked,.failed{color:#ffd092}.blocked{padding:28px;background:#322b21}
video{width:100%;background:black}.playback-error{background:#4f2029;color:#ffd4d8;padding:20px;border:1px solid #ff6c8b;line-height:1.5}.playback-error button{display:block;margin-top:12px}.frames{display:flex;gap:8px;margin:10px 0}.frames a{width:33%;position:relative}.frames img{width:100%}.frames span{position:absolute;bottom:5px;left:5px;background:#000a;padding:2px 5px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}summary{cursor:pointer;margin-top:12px}input{flex:1;min-width:180px}@media(max-width:550px){main{grid-template-columns:1fr;padding:10px}}
</style><header><h1>Same rally. Real models. Every frame.</h1>
<p>26 seconds · 780 frames · 1920 × 1080 · local Windows CPU. Magenta/yellow identify tracker labels, not known players. Gray means unknown identity. Missing or low-confidence joints stay missing.</p>
<p>No ground-truth accuracy ranking. Model confidence scales differ. Runtime is one elapsed run, with setup/download and rendering separate. Top-down models share fresh detector boxes acquired before the original display cutoff.</p>
<a href="manifest.json">Machine-readable manifest</a> · <a href="setup.json">Environment and design</a></header>
<nav><button id="play">Play all</button><button id="pause">Pause all</button><button data-time="8">8s</button><button data-time="15">15s</button><button data-time="22">22s</button><input id="seek" aria-label="Shared video time" type="range" min="0" max="25.9667" step="0.033333" value="0"><output id="time">0.00s</output><label><input type="checkbox" id="sync" checked> Sync on seek</label></nav><main>"""
    page += "".join(cards)
    page += (
        f'</main><footer><h2>Footage credit</h2><p>'
        f'<strong>{html.escape(credit["creator"])}</strong> &mdash; '
        f'<a href="{html.escape(source_clip["source_url"], quote=True)}">{html.escape(credit["source_title"])}</a>. '
        f'Licensed under <a href="{html.escape(credit["cc_by_url_linked_by_youtube"], quote=True)}">CC BY 4.0</a>.</p>'
        f'<p>{html.escape(changes)} {html.escape(credit["endorsement"])}</p>'
        f'<a href="sample-videos/source-attribution.json">Original source attribution, unchanged</a></footer>'
    )
    page += """<script>
const videos=[...document.querySelectorAll('video')], seek=document.querySelector('#seek'), time=document.querySelector('#time');
let changing=false;
function jump(t,source){changing=true;for(const v of videos)if(v!==source&&v.readyState>0)v.currentTime=t;seek.value=t;time.value=Number(t).toFixed(2)+'s';setTimeout(()=>changing=false,100)}
seek.addEventListener('input',()=>jump(Number(seek.value)));
document.querySelectorAll('[data-time]').forEach(b=>b.onclick=()=>jump(Number(b.dataset.time)));
document.querySelector('#play').onclick=()=>{const t=Number(seek.value);jump(t);videos.forEach(v=>v.play().catch(e=>{time.value='Playback needs individual play: '+e.message}))};
document.querySelector('#pause').onclick=()=>videos.forEach(v=>v.pause());
videos.forEach(v=>{v.addEventListener('error',()=>{
const box=v.closest('article').querySelector('.playback-error');v.hidden=true;box.hidden=false;
const text=document.createElement('div');text.textContent='Video playback failed. The static images below are not playback. '+(v.error?.message||'Browser could not decode this MP4.');
const retry=document.createElement('button');retry.textContent='Reload this video';retry.onclick=()=>{const url=new URL(v.src,location.href);url.searchParams.set('reload',Date.now());v.src=url.href;v.load()};
box.replaceChildren(text,retry)});
v.addEventListener('loadeddata',()=>{v.hidden=false;v.closest('article').querySelector('.playback-error').hidden=true});
v.addEventListener('seeking',()=>{if(!changing&&document.querySelector('#sync').checked)jump(v.currentTime,v)});
v.addEventListener('timeupdate',()=>{if(!changing&&!v.paused){seek.value=v.currentTime;time.value=v.currentTime.toFixed(2)+'s'}})});
</script></html>"""
    (HERE / "index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=[*METHODS, "gallery"])
    args = parser.parse_args()
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    try:
        if args.method != "gallery":
            run(args.method)
    finally:
        if (HERE / "manifest.json").exists():
            gallery()
