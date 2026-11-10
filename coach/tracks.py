"""Arrays built from a pose JSON document.

The types below are the contract between the foundation unit and both analysis
units. The foundation unit implements load_tracks and CourtGeometry. Field names
and meanings do not change without the parent session.

Conventions
  pixels      x right, y down, in the source video frame
  court       metres. x runs across the court from the left sideline (0) to the
              right sideline (width_m). y runs from the far baseline (0) to the
              near baseline (length_m). The net is at y = length_m / 2. Corners
              are ordered far-left, far-right, near-right, near-left.
  NaN         unknown. Analysis code must treat NaN as unknown, never as zero
  t           seconds for every frame, from the document. Use it instead of
              frame / fps, because videos can repeat frames
  measured    present, not interpolated and not an exact repeat of the previous
              frame. A missing previous frame is not a repeat, and the first frame
              of a camera shot is always a fresh measurement. Only measured rows
              carry new information. Speeds must be
              computed with coach.signals, which differences between measured
              samples using real time
  visible     per joint: the row is present, x and y are finite, and confidence
              is finite and at or above JOINT_CONF_GATE. Interpolated and repeated
              rows keep the confidence they carry, so they can be visible while
              measured is False. Never measure or draw through a joint that is not
              visible
  stature_px  the player's standing height in pixels at that depth. It does not
              shrink when the player squats or lunges, so sizes stay comparable
  px_per_m    vertical pixels per metre on the floor at the player's feet
  shots       continuous camera shots (first_frame, last_frame), inclusive.
              doc["cuts"] holds the first frame of every shot after the first.
              Differences and velocities must never span a shot boundary.
  warnings    codes from coach.schema.WARNINGS that the loader raised
  refined_windows
              inclusive (start, end) pairs from doc["windows_refined"]. Missing
              metadata or an empty list means no window was sharpened. A present
              value must be a list of windows with integer start and end, and
              0 <= start <= end < n. Any other shape raises. This does not change
              tier, measured, or warnings.

stature_px
  Per role, on measured frames with a visible head and at least one visible ankle.
  Raw height is the vertical pixel span from the nose (else the eyes, else the ears)
  down to the ankles. Foot depth is the image y of the feet. A frame is upright when
  every visible knee is at least 150 degrees and the trunk is within 20 degrees of
  vertical, so a crouch or a lunge cannot shrink its own yardstick. Upright samples
  are grouped by foot depth. Each bin keeps the median height, and stature_px is the
  piecewise-linear curve through those bins. A player who stays at one depth gets the
  median upright height. This is a nose-to-ankle yardstick, not a tape-measured height.

court
  The outline is the outer doubles court: 6.10 m wide and 13.40 m long. The evidence
  is the overlay on clips 01 and 04 (obj2/work/F/court-01.jpg and court-04.jpg): the
  outer quad sits on the doubles sidelines, and the singles sidelines, service lines,
  centre line and net drawn at doubles scale sit on the paint. fast/court.py searches
  for that same outer doubles quad.
  Feet inside the court plus 2 m are only an off-camera rejection check. If fewer
  than a quarter of a shot's feet, and at least 10 feet, land inside that margin,
  foot_court and px_per_m are NaN for that shot. Landing inside does not prove the
  homography still matches the paint after a cut.
  Clips 04 and 05 were checked on a frame after each listed cut
  (court-04-after-212.jpg, court-04-after-367.jpg, court-04-after-665.jpg,
  court-05-after-41.jpg, court-05-after-369.jpg, court-05-after-883.jpg). The same
  doubles lines still sit on the paint, so those shots keep court positions. That
  claim is the overlays, not feet landing inside.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from .schema import COCO_JOINTS, Role, Tier, WARNINGS
from .player_input import Hand, parse_hand

JOINT_INDEX = {name: i for i, name in enumerate(COCO_JOINTS)}

# Render draws a joint whenever it is not None. The pipeline writes None below 0.25.
# On clips 01 to 05 and 08 the stored SimCC scores have a median above 1.1. A gate of
# 0.5 drops 0.3% to 1.7% of joints and at most 1.5% of ankles. A gate of 0.65 would
# drop up to 4.3% of joints and does not make the feet more reliable. Keep 0.5.
JOINT_CONF_GATE = 0.5

COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
MIN_COURT_FIT = 0.85
REPEAT_ROW_FRACTION = 0.05
PARTIAL_PRESENT = 0.80
HIGH_FPS = 40.0
KNEE_UPRIGHT_DEG = 150.0
LEAN_UPRIGHT_DEG = 20.0
OFF_COURT_M = 2.0
FOREIGN_SHOT_INSIDE = 0.25
FOREIGN_SHOT_MIN_FEET = 10


@dataclass(eq=False)
class CourtGeometry:
    corners_px: np.ndarray
    to_court: np.ndarray
    width_m: float
    length_m: float

    @property
    def net_y(self) -> float:
        return self.length_m / 2

    def project(self, xy_px: np.ndarray) -> np.ndarray:
        return _homography_map(xy_px, self.to_court)

    def unproject(self, xy_m: np.ndarray) -> np.ndarray:
        inverse = getattr(self, "from_court", None)
        if inverse is None:
            inverse = np.linalg.inv(self.to_court)
            self.from_court = inverse
        return _homography_map(xy_m, inverse)


@dataclass(eq=False)
class PlayerTrack:
    role: Role
    n: int
    present: np.ndarray
    measured: np.ndarray
    visible: np.ndarray
    xy: np.ndarray
    conf: np.ndarray
    bbox: np.ndarray
    foot_px: np.ndarray
    foot_court: np.ndarray
    stature_px: np.ndarray
    px_per_m: np.ndarray
    hand: Hand = "unknown"

    def joint(self, name: str) -> np.ndarray:
        return self.xy[:, JOINT_INDEX[name], :]

    def joint_visible(self, name: str) -> np.ndarray:
        return self.visible[:, JOINT_INDEX[name]]


@dataclass(eq=False)
class Tracks:
    tier: Tier
    n: int
    fps: float
    t: np.ndarray
    width: int
    height: int
    players: dict[Role, PlayerTrack]
    court: Optional[CourtGeometry]
    shots: tuple[tuple[int, int], ...]
    source: dict[str, Any]
    warnings: tuple[str, ...] = ()
    refined_windows: tuple[tuple[int, int], ...] = ()


def load_tracks(doc: Mapping[str, Any]) -> Tracks:
    source = dict(doc.get("source") or {})
    n = _int_field(source.get("frame_count"), "frame_count")
    fps = _finite(source.get("fps"), "fps")
    width = _int_field(source.get("width"), "width")
    height = _int_field(source.get("height"), "height")
    if n <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("source frame_count, fps, width and height must be positive")
    tier = _tier(doc)
    t = _timestamps(doc.get("frames") or [], n, fps)
    shots = _shots(doc.get("cuts") or [], n)
    slots = {role: _empty_slot(n) for role in ("near", "far")}
    _fill_slots(slots, doc.get("frames") or [], n)
    built: dict[Role, PlayerTrack] = {}
    repeat_rows = 0
    present_rows = 0
    for name, slot in slots.items():
        track = _finish_slot(name, slot, n, shots)
        track.hand = parse_hand(source.get("players", {}).get(name, {}).get("hand", "unknown"))
        built[Role(name)] = track
        present_rows += int(track.present.sum())
        repeat_rows += int((track.present & ~track.measured & ~slot["interpolated"]).sum())
    court = _court_geometry(doc, width, height)
    warnings: list[str] = []
    if court is None:
        warnings.append("no_court")
        for track in built.values():
            track.foot_court[:] = np.nan
            track.px_per_m[:] = np.nan
    else:
        for track in built.values():
            _place_on_court(track, court, shots)
    for track in built.values():
        _set_stature(track)
    if present_rows and repeat_rows / present_rows > REPEAT_ROW_FRACTION:
        warnings.append("repeated_frames")
    if _partial(built, shots):
        warnings.append("partial_track")
    if fps > HIGH_FPS:
        warnings.append("high_fps")
    unknown = [code for code in warnings if code not in WARNINGS]
    if unknown:
        raise ValueError(f"unknown warning codes: {unknown}")
    return Tracks(
        tier, n, fps, t, width, height, built, court, shots, source, tuple(warnings),
        _refined_windows(doc, n),
    )


def _int_field(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer, float, np.floating)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(number)


def _finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


def _refined_windows(doc, n):
    if "windows_refined" not in doc:
        return ()
    raw = doc["windows_refined"]
    if isinstance(raw, (list, tuple)) and not raw:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"windows_refined must be a list, got {raw!r}")
    kept = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or "start" not in item or "end" not in item:
            raise ValueError(f"windows_refined[{index}] must have integer start and end, got {item!r}")
        start = _window_int(item["start"], f"windows_refined[{index}].start")
        end = _window_int(item["end"], f"windows_refined[{index}].end")
        if not (0 <= start <= end < n):
            raise ValueError(f"windows_refined[{index}] needs 0 <= start <= end < {n}, got ({start}, {end})")
        kept.append((start, end))
    return tuple(kept)


def _window_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _tier(doc):
    if "tier" not in doc:
        return Tier.REFINED
    value = doc["tier"]
    if value == "preview":
        return Tier.PREVIEW
    if value == "refined":
        return Tier.REFINED
    raise ValueError(f"tier must be preview or refined, got {value!r}")


def _timestamps(frames, n, fps):
    t = np.arange(n, dtype=np.float64) / fps
    for pos, frame in enumerate(frames):
        idx = _frame_index(frame, pos, n)
        if idx is None or "timestamp_seconds" not in frame or frame["timestamp_seconds"] is None:
            continue
        t[idx] = _finite(frame["timestamp_seconds"], "timestamp_seconds")
    if np.any(t < 0):
        raise ValueError("timestamp_seconds must not be negative")
    if n >= 2 and np.any(t[1:] < t[:-1]):
        raise ValueError("timestamps must not go backwards")
    return t


def _frame_index(frame, pos, n):
    idx = frame.get("frame_index", pos)
    if isinstance(idx, bool) or not isinstance(idx, (int, np.integer, float, np.floating)):
        raise ValueError(f"frame_index must be an integer, got {idx!r}")
    number = float(idx)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"frame_index must be an integer, got {idx!r}")
    idx = int(number)
    if idx < 0 or idx >= n:
        return None
    return idx


def _shots(cuts, n):
    points = []
    for cut in cuts:
        if isinstance(cut, bool) or not isinstance(cut, (int, np.integer, float, np.floating)):
            raise ValueError(f"cut must be a frame index, got {cut!r}")
        number = float(cut)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"cut must be a frame index, got {cut!r}")
        cut = int(number)
        if 0 < cut < n:
            points.append(cut)
    bounds = [0, *sorted(set(points)), n]
    return tuple((a, b - 1) for a, b in zip(bounds, bounds[1:]) if b > a)


def _empty_slot(n):
    return {
        "present": np.zeros(n, dtype=bool),
        "interpolated": np.zeros(n, dtype=bool),
        "xy": np.full((n, 17, 2), np.nan, dtype=np.float64),
        "conf": np.full((n, 17), np.nan, dtype=np.float64),
        "bbox": np.full((n, 4), np.nan, dtype=np.float64),
    }


def _fill_slots(slots, frames, n):
    for pos, frame in enumerate(frames):
        idx = _frame_index(frame, pos, n)
        if idx is None:
            continue
        best = {}
        for player in frame.get("players") or []:
            role = player.get("role")
            if role not in slots:
                continue
            score = player.get("box_confidence")
            score = 0.0 if score is None else score
            prev = best.get(role)
            if prev is None or score > prev[0]:
                best[role] = (score, player)
        for role, (_, player) in best.items():
            slot = slots[role]
            slot["present"][idx] = True
            if player.get("src") == "interpolated":
                slot["interpolated"][idx] = True
            box = player.get("bbox_xyxy")
            if box is not None:
                slot["bbox"][idx] = box
            joints = player.get("joints")
            if joints is None or len(joints) != 17:
                raise ValueError("each player row needs 17 joints")
            if any(joint is None for joint in joints):
                xy = slot["xy"]
                conf = slot["conf"]
                for j, joint in enumerate(joints):
                    if joint is None:
                        continue
                    xy[idx, j, 0] = joint[0]
                    xy[idx, j, 1] = joint[1]
                    conf[idx, j] = joint[2]
            else:
                block = np.asarray(joints, dtype=np.float64)
                if block.shape != (17, 3):
                    raise ValueError("each player row needs 17 joints")
                slot["xy"][idx] = block[:, :2]
                slot["conf"][idx] = block[:, 2]
    for slot in slots.values():
        if np.isinf(slot["xy"]).any() or np.isinf(slot["conf"]).any() or np.isinf(slot["bbox"]).any():
            raise ValueError("infinite value in the pose document")


def _finish_slot(name, slot, n, shots):
    present = slot["present"]
    xy = slot["xy"]
    conf = slot["conf"]
    finite_xy = np.isfinite(xy).all(axis=2)
    visible = present[:, None] & finite_xy & np.isfinite(conf) & (conf >= JOINT_CONF_GATE)
    identical = _identical_to_previous(xy, conf, present, shots)
    measured = present & ~slot["interpolated"] & ~identical
    return PlayerTrack(
        Role(name), n, present, measured, visible, xy, conf, slot["bbox"],
        _foot_px(xy, visible),
        np.full((n, 2), np.nan), np.full(n, np.nan), np.full(n, np.nan),
    )


def _identical_to_previous(xy, conf, present, shots):
    n = present.shape[0]
    repeat = np.zeros(n, dtype=bool)
    if n < 2:
        return repeat
    rows = np.concatenate([xy, conf[:, :, None]], axis=2)
    filled = np.where(np.isnan(rows), 0.0, rows)
    missing = np.isnan(rows)
    same = np.all(filled[1:] == filled[:-1], axis=(1, 2)) & np.all(missing[1:] == missing[:-1], axis=(1, 2))
    repeat[1:] = present[1:] & present[:-1] & same
    for start, _end in shots:
        if start > 0:
            repeat[start] = False
    return repeat


def _foot_px(xy, visible):
    foot = np.full((xy.shape[0], 2), np.nan)
    left = visible[:, 15]
    right = visible[:, 16]
    both = left & right
    foot[both] = (xy[both, 15] + xy[both, 16]) * 0.5
    only_left = left & ~right
    only_right = right & ~left
    foot[only_left] = xy[only_left, 15]
    foot[only_right] = xy[only_right, 16]
    return foot


def _court_geometry(doc, width, height):
    lines = doc.get("court_lines_normalized")
    if not lines or len(lines) != 4:
        return None
    fit = doc.get("court_fit") or {}
    if "fit" in fit and fit["fit"] is not None and float(fit["fit"]) < MIN_COURT_FIT:
        return None
    corners = np.asarray(lines, dtype=np.float64)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return None
    corners_px = corners * np.array([width, height], dtype=np.float64)
    if corners_px[2:, 1].mean() <= corners_px[:2, 1].mean() + 1.0:
        return None
    contour = np.ascontiguousarray(corners_px, np.float32)
    if cv2.contourArea(contour) < 0.002 * width * height or not cv2.isContourConvex(contour):
        return None
    dst = np.array(
        [[0.0, 0.0], [COURT_WIDTH_M, 0.0], [COURT_WIDTH_M, COURT_LENGTH_M], [0.0, COURT_LENGTH_M]],
        np.float32,
    )
    try:
        to_court = cv2.getPerspectiveTransform(contour, dst)
        from_court = cv2.getPerspectiveTransform(dst, contour)
    except cv2.error:
        return None
    if not np.isfinite(to_court).all():
        return None
    court = CourtGeometry(corners_px, to_court.astype(np.float64), COURT_WIDTH_M, COURT_LENGTH_M)
    court.from_court = from_court.astype(np.float64)
    return court


def _homography_map(xy, matrix):
    pts = np.asarray(xy, dtype=np.float64)
    single = pts.ndim == 1
    if single:
        pts = pts.reshape(1, 2)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (2,) or (n, 2)")
    if np.isinf(pts).any():
        raise ValueError("infinite point")
    out = np.full(pts.shape, np.nan, dtype=np.float64)
    good = np.isfinite(pts).all(axis=1)
    if not good.any():
        return out[0] if single else out
    raw = np.ascontiguousarray(pts[good], np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(raw, np.ascontiguousarray(matrix, np.float32))
    out[good] = mapped.reshape(-1, 2)
    return out[0] if single else out


def _place_on_court(track, court, shots):
    good = np.isfinite(track.foot_px).all(axis=1)
    if good.any():
        track.foot_court[good] = court.project(track.foot_px[good])
        track.px_per_m[good] = _px_per_m(court, track.foot_px[good])
    _blank_foreign_shots(track, court, shots)


def _px_per_m(court, xy):
    step = np.array([0.0, 0.5])
    dist = np.linalg.norm(court.project(xy + step) - court.project(xy - step), axis=1)
    out = np.full(xy.shape[0], np.nan)
    ok = np.isfinite(dist) & (dist > 1e-9)
    out[ok] = 1.0 / dist[ok]
    return out


def _blank_foreign_shots(track, court, shots):
    # Rejection only. Feet inside the margin do not prove the homography survived a cut.
    width, length = court.width_m, court.length_m
    for start, end in shots:
        pts = track.foot_court[start:end + 1]
        ok = np.isfinite(pts).all(axis=1)
        if int(ok.sum()) < FOREIGN_SHOT_MIN_FEET:
            continue
        kept = pts[ok]
        inside = (
            (kept[:, 0] >= -OFF_COURT_M) & (kept[:, 0] <= width + OFF_COURT_M)
            & (kept[:, 1] >= -OFF_COURT_M) & (kept[:, 1] <= length + OFF_COURT_M)
        )
        if float(inside.mean()) < FOREIGN_SHOT_INSIDE:
            track.foot_court[start:end + 1] = np.nan
            track.px_per_m[start:end + 1] = np.nan


def _partial(players, shots):
    for track in players.values():
        for start, end in shots:
            present = track.present[start:end + 1]
            if present.any() and float(present.mean()) < PARTIAL_PRESENT:
                return True
    return False


def _set_stature(track):
    head = _head_xy(track.xy, track.visible)
    foot_y = track.foot_px[:, 1]
    raw = foot_y - head[:, 1]
    upright = _upright_mask(track)
    sample = track.measured & upright & np.isfinite(raw) & (raw > 1.0) & np.isfinite(foot_y)
    on_model = np.isfinite(track.foot_court).all(axis=1)
    if not on_model.any():
        track.stature_px[:] = _stature_curve(foot_y, raw, sample, foot_y)
        return
    track.stature_px[:] = _stature_curve(foot_y, raw, sample & on_model, foot_y)
    foreign = np.isfinite(foot_y) & ~on_model
    if foreign.any():
        pred = _stature_curve(foot_y, raw, sample & foreign, foot_y)
        track.stature_px[foreign] = pred[foreign]


def _head_xy(xy, visible):
    head = np.full((xy.shape[0], 2), np.nan)
    nose = visible[:, 0]
    head[nose] = xy[nose, 0]
    missing = ~np.isfinite(head[:, 0])
    for indices in ((1, 2), (3, 4)):
        alt = _mean_joints(xy, visible, indices)
        take = missing & np.isfinite(alt[:, 0])
        head[take] = alt[take]
        missing = ~np.isfinite(head[:, 0])
    return head


def _mean_joints(xy, visible, indices):
    acc = np.zeros((xy.shape[0], 2), dtype=np.float64)
    count = np.zeros(xy.shape[0], dtype=np.float64)
    for joint in indices:
        seen = visible[:, joint]
        acc[seen] += xy[seen, joint]
        count[seen] += 1.0
    out = np.full((xy.shape[0], 2), np.nan)
    ok = count > 0
    out[ok] = acc[ok] / count[ok, None]
    return out


def _joint_angle(xy, visible, a, b, c):
    chain = visible[:, a] & visible[:, b] & visible[:, c]
    ang = np.full(xy.shape[0], np.nan)
    if not chain.any():
        return ang
    ba = xy[chain, a] - xy[chain, b]
    bc = xy[chain, c] - xy[chain, b]
    denom = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    cos = np.ones(int(chain.sum()), dtype=np.float64)
    ok = denom > 1e-6
    cos[ok] = np.sum(ba[ok] * bc[ok], axis=1) / denom[ok]
    ang[chain] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return ang


def _upright_mask(track):
    left = _joint_angle(track.xy, track.visible, 11, 13, 15)
    right = _joint_angle(track.xy, track.visible, 12, 14, 16)
    bent = (np.isfinite(left) & (left < KNEE_UPRIGHT_DEG)) | (np.isfinite(right) & (right < KNEE_UPRIGHT_DEG))
    has_knee = np.isfinite(left) | np.isfinite(right)
    shoulders = _mean_joints(track.xy, track.visible, (5, 6))
    hips = _mean_joints(track.xy, track.visible, (11, 12))
    vec = shoulders - hips
    norm = np.linalg.norm(vec, axis=1)
    lean = np.full(track.n, np.nan)
    ok = (norm > 1e-6) & np.isfinite(vec).all(axis=1)
    # Image y points down, so a vertical trunk has a negative y component.
    lean[ok] = np.degrees(np.arccos(np.clip(-vec[ok, 1] / norm[ok], -1.0, 1.0)))
    leaned = np.isfinite(lean) & (lean > LEAN_UPRIGHT_DEG)
    return has_knee & ~bent & ~leaned


def _stature_curve(foot_y, raw_h, sample, query_y):
    out = np.full(query_y.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(query_y)
    y = foot_y[sample]
    h = raw_h[sample]
    if y.size == 0:
        return out
    if y.size < 4 or float(y.max() - y.min()) < 12.0:
        out[finite] = float(np.median(h))
        return out
    order = np.argsort(y)
    y = y[order]
    h = h[order]
    n_bins = int(np.clip(y.size // 6, 4, 10))
    edges = np.linspace(y[0], y[-1], n_bins + 1)
    knots_y = []
    knots_h = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        chosen = (y >= lo) & (y <= hi if hi == edges[-1] else y < hi)
        if int(chosen.sum()) < 2:
            continue
        knots_y.append(float(np.median(y[chosen])))
        knots_h.append(float(np.median(h[chosen])))
    if len(knots_y) < 2:
        out[finite] = float(np.median(h))
        return out
    out[finite] = np.interp(query_y[finite], knots_y, knots_h)
    out[~(out > 0)] = np.nan
    return out
