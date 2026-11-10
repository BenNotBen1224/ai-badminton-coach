"""Real five-frame PoseWarper CPU inference; no MMCV binary or video upload."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from .posewarper_vendor.model import HRNet, PoseWarperNeck

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CHECKPOINT_NAME = "hrnet_w48_posetrack18_384x288_posewarper_stage2-4abf88db_20211130.pth"
CHECKPOINT_URL = "https://download.openmmlab.com/mmpose/top_down/posewarper/" + CHECKPOINT_NAME
CHECKPOINT_BYTES = 285763417
CHECKPOINT_SHA256 = "4abf88db25e90e0e3f3c4113f3269173e438beebaf8dee7ee3a56089bc796194"
FRAME_OFFSETS = (0, -2, -1, 1, 2)
FRAME_WEIGHTS = (0.3, 0.1, 0.25, 0.25, 0.1)
NATIVE_NAMES = (
    "nose", "head_bottom", "head_top", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
COCO_NATIVE_INDICES = (0, None, None, None, None, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)


class TemporalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        extra = {}
        for stage, modules, channels in (
            (1, 1, (64,)), (2, 1, (48, 96)),
            (3, 4, (48, 96, 192)), (4, 3, (48, 96, 192, 384)),
        ):
            extra[f"stage{stage}"] = dict(
                num_modules=modules, num_branches=len(channels),
                block="BOTTLENECK" if stage == 1 else "BASIC",
                num_blocks=(4,) * len(channels), num_channels=channels)
        self.backbone = HRNet(extra, frozen_stages=4)
        self.neck = PoseWarperNeck(
            in_channels=48, out_channels=17, inner_channels=128,
            deform_groups=17, dilations=(3, 6, 12, 18, 24),
            trans_conv_kernel=1, res_blocks_cfg=dict(block="BASIC", num_blocks=20),
            offsets_kernel=3, deform_conv_kernel=3, freeze_trans_layer=True)
        self.keypoint_head = nn.Identity()

    def forward(self, crops):
        if len(crops) != 5:
            raise ValueError("PoseWarper requires current,-2,-1,+1,+2 crops")
        features = [self.backbone(crop) for crop in crops]
        return self.keypoint_head(self.neck(features, FRAME_WEIGHTS))


def affine_for_box(box):
    box = np.asarray(box, dtype=np.float32)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError(f"Invalid acquisition bbox: {box}")
    center = (box[:2] + box[2:]) * .5
    width, height = box[2:] - box[:2]
    width = max(width, height * .75)
    extent = np.array([width, width / .75], dtype=np.float32) * 1.25
    # The original non-UDP TopDownAffine uses width for both source axes.
    source = np.array([
        center, center + [0, -.5 * extent[0]],
        center + [-.5 * extent[0], -.5 * extent[0]],
    ], dtype=np.float32)
    dest = np.array([[144, 192], [144, 48], [0, 48]], dtype=np.float32)
    return cv2.getAffineTransform(source, dest), center, extent


def decode(heatmaps, center, extent):
    if heatmaps.shape != (17, 96, 72) or not np.isfinite(heatmaps).all():
        raise ValueError(f"Invalid PoseWarper heatmaps: {heatmaps.shape}")
    flat = heatmaps.reshape(17, -1)
    maxima = flat.argmax(axis=1)
    confidence = flat[np.arange(17), maxima]
    coordinates = np.column_stack((maxima % 72, maxima // 72)).astype(np.float32)
    coordinates[confidence <= 0] = -1
    for k, (x, y) in enumerate(coordinates.astype(int)):
        if 1 < x < 71 and 1 < y < 95:
            gradient = np.array([
                heatmaps[k, y, x + 1] - heatmaps[k, y, x - 1],
                heatmaps[k, y + 1, x] - heatmaps[k, y - 1, x],
            ])
            coordinates[k] += np.sign(gradient) * .25
    coordinates = coordinates * extent / [72, 96] + center - extent * .5
    return np.column_stack((coordinates, confidence))


class PoseWarper:
    """Callable comparison adapter using common >=0.1 current-frame boxes.

    Neighbor frames use the *current box's same affine crop*, as in the official
    video dataset. Neighbor detections and neighboring pose outputs are not used.
    Only genuine PoseTrack nose/body joints map to COCO; eyes/ears remain null.
    """

    def __init__(self, weight: Path | None = None, *,
                 acquisition: Path | None = None, source: Path | None = None):
        self.weight = Path(weight) if weight else HERE / "weights" / CHECKPOINT_NAME
        if not self.weight.exists():
            from .posewarper_vendor.download import download
            download(self.weight)
        if self.weight.stat().st_size != CHECKPOINT_BYTES:
            raise ValueError("Unexpected size for the official PoseWarper checkpoint")
        with self.weight.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != CHECKPOINT_SHA256:
            raise ValueError("PoseWarper checkpoint SHA256 does not match the official release")
        acquisition = Path(acquisition) if acquisition else HERE / "results" / "crop" / "acquisition.json"
        self.boxes = json.loads(acquisition.read_text(encoding="utf-8"))["frames"]
        self.source = Path(source) if source else ROOT / "sample-videos" / "01-singles-rally.mp4"
        self.capture = cv2.VideoCapture(str(self.source))
        if not self.capture.isOpened():
            raise ValueError(f"Cannot open source video: {self.source}")
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if len(self.boxes) != self.frame_count:
            self.capture.release()
            raise ValueError("Acquisition frame count does not match the complete source video")
        self._cache = OrderedDict()
        self.last_temporal_indices = None
        self._next_decode = 0
        self.model = TemporalNetwork().cpu().eval()
        # The official metadata contains a NumPy scalar; never enable pickle code execution.
        numpy_core = np._core if hasattr(np, "_core") else np.core
        with torch.serialization.safe_globals([
            (numpy_core.multiarray.scalar, "numpy.core.multiarray.scalar"),
            np.dtype, type(np.dtype("float64")), type(np.dtype("float32")),
        ]):
            checkpoint = torch.load(self.weight, map_location="cpu", weights_only=True)
        state = checkpoint["state_dict"]
        # Strict loading includes all learned neck offsets and deformable kernels.
        self.load_result = self.model.load_state_dict(state, strict=True)
        del checkpoint, state
        self.model.requires_grad_(False)
        self.mean = torch.tensor([.485, .456, .406])[:, None, None]
        self.std = torch.tensor([.229, .224, .225])[:, None, None]

    def close(self):
        self.capture.release()
        self._cache.clear()

    def _frame(self, index):
        if index in self._cache:
            return self._cache[index]
        if self._next_decode != index:
            if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise ValueError(f"Cannot seek temporal frame {index}")
        ok, image = self.capture.read()
        if not ok:
            raise ValueError(f"Cannot decode temporal frame {index}")
        self._next_decode = index + 1
        self._cache[index] = image
        return image

    def _temporal_frames(self, frame, index):
        indices = [min(max(index + offset, 0), self.frame_count - 1) for offset in FRAME_OFFSETS]
        self.last_temporal_indices = indices
        # Decode chronological neighbors sequentially, retaining a bounded window.
        self._cache[index] = frame
        for position in sorted(set(indices)):
            self._frame(position)
        images = [self._cache[position] for position in indices]
        for old in list(self._cache):
            if old < index - 2 or old > index + 2:
                del self._cache[old]
        return images

    @torch.inference_mode()
    def infer_box(self, frames, person):
        from comparison.run import player

        matrix, center, extent = affine_for_box(person["bbox_xyxy"])
        crops = []
        for image in frames:
            crop = cv2.warpAffine(image, matrix, (288, 384), flags=cv2.INTER_LINEAR)
            rgb = np.ascontiguousarray(crop[:, :, ::-1].transpose(2, 0, 1))
            tensor = torch.from_numpy(rgb).float().div_(255)
            crops.append(((tensor - self.mean) / self.std).unsqueeze(0))
        heatmaps = self.model(crops)[0].cpu().numpy()
        native = decode(heatmaps, center, extent)
        pose = np.zeros((17, 3), dtype=np.float32)
        for coco, native_index in enumerate(COCO_NATIVE_INDICES):
            if native_index is not None:
                pose[coco] = native[native_index]
        record = player(person["bbox_xyxy"], pose, person["box_confidence"], person.get("track_id"))
        if record is None:
            return None
        for coco, native_index in enumerate(COCO_NATIVE_INDICES):
            if native_index is None:
                record["joints"][coco] = None
        record["unavailable_coco_indices"] = [1, 2, 3, 4]
        record["native_layout"] = "PoseTrack17 (head_bottom/head_top are not COCO eyes)"
        record["native_joint_names"] = list(NATIVE_NAMES)
        record["native_joints"] = native.tolist()
        record["temporal_offsets"] = list(FRAME_OFFSETS)
        record["temporal_weights"] = list(FRAME_WEIGHTS)
        record["temporal_frame_indices"] = list(self.last_temporal_indices)
        return record

    def __call__(self, frame, index: int):
        if not isinstance(index, int) or not 0 <= index < self.frame_count:
            raise ValueError(f"Invalid source frame index: {index}")
        if frame.shape != (1080, 1920, 3) or frame.dtype != np.uint8:
            raise ValueError("Expected source-resolution 1920x1080 BGR uint8 frame")
        people = [
            person for person in self.boxes[index]["players"]
            if person["box_confidence"] >= .1
        ]
        if not people:
            return []
        frames = self._temporal_frames(frame, index)
        return [record for person in people if (record := self.infer_box(frames, person)) is not None]
