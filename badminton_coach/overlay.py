from __future__ import annotations

from itertools import count
from typing import Any

import cv2
import numpy as np


COCO17_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)
FACE_JOINTS = frozenset(range(5))
# BGR, chosen to stand out against a green court, blue boards, and white or red kits.
PALETTE = (
    (170, 0, 255),    # magenta
    (0, 225, 255),    # yellow
    (255, 230, 0),    # cyan
    (0, 120, 255),    # orange
    (255, 90, 150),   # violet
)
UNTRACKED_COLOR = (215, 215, 215)
OUTLINE_COLOR = (24, 24, 24)
JOINT_COLOR = (255, 255, 255)
MIN_LIMB_PIXELS = 4
LIMB_PER_BOX_HEIGHT = 42
OUTLINE_EXTRA_PIXELS = 3


def assign_track_slots(frames: list[dict[str, Any]]) -> dict[int, int]:
    """Give every tracker ID one color slot. IDs seen in the same frame never share one."""
    frames_seen: dict[int, int] = {}
    seen_together: dict[int, set[int]] = {}
    for frame in frames:
        ids = [player["track_id"] for player in frame["players"] if player["track_id"] is not None]
        for track_id in ids:
            frames_seen[track_id] = frames_seen.get(track_id, 0) + 1
            seen_together.setdefault(track_id, set()).update(ids)
    slots: dict[int, int] = {}
    for track_id in sorted(frames_seen, key=lambda value: (-frames_seen[value], value)):
        taken = {slots[other] for other in seen_together[track_id] if other in slots}
        slots[track_id] = next(slot for slot in count() if slot not in taken)
    return slots


def draw_pose(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    joints: list[list[float] | None],
    track_id: int | None,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = (round(value) for value in bbox)
    limb = max(MIN_LIMB_PIXELS, round((y2 - y1) / LIMB_PER_BOX_HEIGHT))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    label = f"ID {track_id}" if track_id is not None else "ID unknown"
    label_origin = (x1, max(28, y1 - 10))
    for thickness, label_color in ((5, OUTLINE_COLOR), (2, color)):
        cv2.putText(
            frame, label, label_origin,
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, thickness, cv2.LINE_AA,
        )

    points = [
        None if joint is None else (round(joint[0]), round(joint[1]))
        for joint in joints
    ]
    segments = [
        (points[first], points[second], first in FACE_JOINTS and second in FACE_JOINTS)
        for first, second in COCO17_EDGES
        if points[first] is not None and points[second] is not None
    ]
    for outline_extra, segment_color in ((OUTLINE_EXTRA_PIXELS, OUTLINE_COLOR), (0, color)):
        for start, end, is_face in segments:
            width = (max(2, limb // 2) if is_face else limb) + outline_extra
            cv2.line(frame, start, end, segment_color, width, cv2.LINE_AA)
    for index, point in enumerate(points):
        if point is None:
            continue
        radius = max(2, limb // 2) if index in FACE_JOINTS else limb // 2 + 2
        cv2.circle(frame, point, radius + 2, OUTLINE_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, point, radius, JOINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, point, max(1, radius - 2), color, -1, cv2.LINE_AA)


def annotate_frame(
    frame: np.ndarray, players: list[dict[str, Any]], slots: dict[int, int],
) -> np.ndarray:
    annotation = frame.copy()
    for player in players:
        track_id = player["track_id"]
        color = (
            UNTRACKED_COLOR if track_id is None
            else PALETTE[slots[track_id] % len(PALETTE)]
        )
        draw_pose(
            annotation, tuple(player["bbox_xyxy"]), player["joints"], track_id, color,
        )
    return annotation
