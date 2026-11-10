from __future__ import annotations

import base64
import copy
import json
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .schema import (
    COCO_JOINTS,
    LEGACY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    Basis,
    Metric,
    normalize_metric_record,
    to_plain,
)
from .timeline import build_timeline

_STYLE_PATH = Path(__file__).with_name("style.json")
with _STYLE_PATH.open(encoding="utf-8") as _style_file:
    STYLE = json.load(_style_file)

_COURT_DST = np.array(
    [[0.0, 0.0], [6.10, 0.0], [6.10, 13.40], [0.0, 13.40]],
    dtype=np.float32,
)
_XY_UNKNOWN = -32768


def build_bundle(doc, report, *, video_url, exercises=None) -> dict:
    exercises = exercises or {}
    doc_source = doc.get("source") or {}
    if "frame_count" in doc_source:
        counted = int(doc_source["frame_count"])
        if counted != report.n_frames:
            raise ValueError(
                f"doc frame_count {counted} does not match report.n_frames {report.n_frames}"
            )
    n = report.n_frames
    if report.timestamps:
        stamps = list(report.timestamps)
    else:
        stamps = [i / report.fps for i in range(report.n_frames)]
    roles = [role.value if hasattr(role, "value") else str(role) for role in report.players]
    poses = _poses(doc.get("frames") or (), n, roles, _shots(doc.get("cuts") or (), n))
    source = dict(report.source)
    for key in ("width", "height"):
        if key not in source or source[key] is None:
            if key in doc_source:
                source[key] = doc_source[key]
    findings = []
    for finding in report.findings:
        plain = to_plain(finding)
        best = finding.best_moment
        plain["timelines"] = [
            _timeline_for(moment, best, report.fps, report.n_frames, stamps)
            for moment in finding.moments
        ]
        plain["exercise_ids"] = _exercise_ids(exercises, finding)
        findings.append(plain)
    tier = report.tier.value if hasattr(report.tier, "value") else report.tier
    return {
        "schema_version": report.schema_version,
        "n_frames": report.n_frames,
        "fps": report.fps,
        "tier": tier,
        "players": roles,
        "source": source,
        "style": copy.deepcopy(STYLE),
        "timestamps": stamps,
        "findings": findings,
        "poses": poses,
        "series": _plain_series(report.series),
        "events": to_plain(report.events),
        "windows": to_plain(report.windows),
        "warnings": to_plain(report.warnings),
        "video_url": video_url,
        "court": _court(doc),
        "joints": list(COCO_JOINTS),
    }


def _plain_series(series):
    return {
        role: {name: list(values) for name, values in named.items()}
        for role, named in series.items()
    }


def _exercise_ids(exercises, finding):
    player = finding.player.value if hasattr(finding.player, "value") else str(finding.player)
    for key in (
        (finding.id, player),
        (finding.id, finding.player),
        f"{player}:{finding.id}",
    ):
        if key in exercises:
            return list(exercises[key])
    return []


def _timeline_for(moment, best, fps, n_frames, stamps):
    if moment.kind == "best":
        built = build_timeline(
            moment, fps, n_frames, timestamps=stamps, compare=None, show="best",
        )
    else:
        built = build_timeline(
            moment, fps, n_frames, timestamps=stamps, compare=best, show="issue",
        )
    return built.to_dict()


def _poses(frames, n, roles, shots):
    role_set = set(roles)
    slots = {
        role: {
            "xy": np.full((n, 17, 2), np.nan, np.float64),
            "conf": np.full((n, 17), np.nan, np.float64),
            "bbox": np.full((n, 4), np.nan, np.float64),
            "present": np.zeros(n, dtype=bool),
            "interpolated": np.zeros(n, dtype=bool),
        }
        for role in roles
    }
    pending = {role: [] for role in roles}
    pending_at = {role: [] for role in roles}
    for pos, frame in enumerate(frames):
        idx = _index(frame, pos, n)
        if idx is None:
            continue
        best = {}
        for player in frame.get("players") or ():
            role = player.get("role")
            if role not in role_set:
                continue
            score = player.get("box_confidence")
            if score is None:
                score = 0.0
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
            pending[role].append(player.get("joints"))
            pending_at[role].append(idx)
    for role in roles:
        _flush_joints(slots[role], pending[role], pending_at[role])
    packed = {}
    for role in roles:
        slot = slots[role]
        repeat = _repeat(slot["xy"], slot["conf"], slot["present"], shots)
        measured = slot["present"] & ~slot["interpolated"] & ~repeat
        packed[role] = {
            "xy": _encoded(_quantize_xy(slot["xy"]), "int16", scale=0.1, unknown=_XY_UNKNOWN),
            "conf": _encoded(_quantize_conf(slot["conf"]), "uint8", scale=0.01, unknown=0),
            "measured": _encoded(measured.astype(np.uint8), "uint8"),
            "bbox": _encoded(_quantize_xy(slot["bbox"]), "int16", scale=0.1, unknown=_XY_UNKNOWN),
        }
    return packed


def _flush_joints(slot, rows, indices):
    if not rows:
        return
    stacked = _stacked(rows)
    if stacked is not None:
        ix = np.asarray(indices, dtype=np.intp)
        # One setitem with a repeated index does not keep the last row.
        if len(np.unique(ix)) == len(ix):
            slot["xy"][ix] = stacked[:, :, :2]
            slot["conf"][ix] = stacked[:, :, 2]
            return
        for i, idx in enumerate(ix):
            slot["xy"][idx] = stacked[i, :, :2]
            slot["conf"][idx] = stacked[i, :, 2]
        return
    for idx, joints in zip(indices, rows):
        _write_holes(slot, idx, joints)


def _stacked(rows):
    try:
        stacked = np.asarray(rows, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if stacked.shape != (len(rows), 17, 3):
        return None
    return stacked


def _write_holes(slot, idx, joints):
    if joints is None or len(joints) != 17:
        raise ValueError("each player row needs 17 joints")
    xy = slot["xy"]
    conf = slot["conf"]
    for j, joint in enumerate(joints):
        if joint is None:
            continue
        xy[idx, j, 0] = joint[0]
        xy[idx, j, 1] = joint[1]
        conf[idx, j] = joint[2]


def _index(frame, pos, n):
    idx = frame.get("frame_index", pos)
    if isinstance(idx, bool) or not isinstance(idx, (int, np.integer)):
        return None
    idx = int(idx)
    if idx < 0 or idx >= n:
        return None
    return idx


def _shots(cuts, n):
    points = []
    for cut in cuts:
        if isinstance(cut, bool) or not isinstance(cut, (int, np.integer, float, np.floating)):
            continue
        number = float(cut)
        if not math.isfinite(number) or not number.is_integer():
            continue
        cut = int(number)
        if 0 < cut < n:
            points.append(cut)
    bounds = [0, *sorted(set(points)), n]
    return tuple((a, b - 1) for a, b in zip(bounds, bounds[1:]) if b > a)


def _repeat(xy, conf, present, shots):
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


def _quantize_xy(values):
    out = np.full(values.shape, _XY_UNKNOWN, dtype=np.int16)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    rounded = np.rint(values[finite] * 10.0)
    if np.any((rounded < -32767) | (rounded > 32767)):
        raise ValueError("rounded tenths fall outside -32767..32767")
    out[finite] = rounded.astype(np.int16)
    return out


def _quantize_conf(values):
    out = np.zeros(values.shape, dtype=np.uint8)
    usable = np.isfinite(values) & (values >= 0.0)
    if not usable.any():
        return out
    chosen = values[usable]
    scaled = np.rint(chosen * 100.0)
    under_gate = chosen < 0.5
    scaled[under_gate] = np.minimum(scaled[under_gate], 49.0)
    scaled[~under_gate] = np.clip(scaled[~under_gate], 50.0, 255.0)
    out[usable] = scaled.astype(np.uint8)
    return out


def _encoded(arr, dtype, scale=None, unknown=None):
    if dtype == "uint8":
        raw = np.ascontiguousarray(arr, dtype=np.uint8)
    else:
        raw = np.ascontiguousarray(arr, dtype="<i2")
    packed = {
        "b64": base64.b64encode(raw.tobytes()).decode("ascii"),
        "dtype": dtype,
        "shape": [int(v) for v in arr.shape],
        "endian": "little",
    }
    if scale is not None:
        packed["scale"] = scale
    if unknown is not None:
        packed["unknown"] = unknown
    return packed


def _court(doc):
    lines = doc.get("court_lines_normalized")
    if not lines or len(lines) != 4:
        return None
    fit = doc.get("court_fit") or {}
    if "fit" in fit and fit["fit"] is not None and float(fit["fit"]) < 0.85:
        return None
    source = doc.get("source") or {}
    width = source.get("width")
    height = source.get("height")
    if not width or not height:
        return None
    corners = np.asarray(lines, dtype=np.float64)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return None
    corners_px = corners * np.array([float(width), float(height)], dtype=np.float64)
    if corners_px[2:, 1].mean() <= corners_px[:2, 1].mean():
        return None
    contour = np.ascontiguousarray(corners_px, np.float32)
    try:
        matrix = cv2.getPerspectiveTransform(contour, _COURT_DST)
    except cv2.error:
        return None
    if matrix is None or not np.isfinite(matrix).all():
        return None
    return {
        "width_m": 6.10,
        "length_m": 13.40,
        "to_court": matrix.astype(np.float64).tolist(),
    }


class BundleContractError(ValueError):
    """A bundle metric contradicts the observation rule. Do not guess a value."""


@dataclass(frozen=True)
class PreparedBundle:
    bundle: dict
    send_saved_bytes: bool


def prepare_bundle(data: Mapping) -> PreparedBundle:
    if not isinstance(data, Mapping):
        raise BundleContractError("bundle must be a mapping")
    version = _bundle_version(data)
    bundle = dict(data)
    cleared = False
    if "findings" in data:
        findings, changed = _prepare_findings(data["findings"], version)
        bundle["findings"] = findings
        cleared = changed
    if "metrics" in data:
        metrics, changed = _prepare_metric_group(data["metrics"], version)
        bundle["metrics"] = metrics
        cleared = cleared or changed
    return PreparedBundle(bundle, not cleared)


def _bundle_version(data: Mapping) -> int:
    if "schema_version" not in data:
        raise BundleContractError("bundle is missing schema_version")
    version = data["schema_version"]
    if version not in (LEGACY_SCHEMA_VERSION, SCHEMA_VERSION):
        raise BundleContractError(
            f"schema_version must be {LEGACY_SCHEMA_VERSION} or {SCHEMA_VERSION}, got {version!r}"
        )
    return version


def _prepare_findings(findings, version: int) -> tuple[list, bool]:
    if not isinstance(findings, (list, tuple)):
        raise BundleContractError("findings must be a list")
    copied = []
    cleared = False
    for finding in findings:
        item, changed = _prepare_finding(finding, version)
        copied.append(item)
        cleared = cleared or changed
    return copied, cleared


def _prepare_finding(finding, version: int) -> tuple[dict, bool]:
    if not isinstance(finding, Mapping):
        raise BundleContractError("finding must be a mapping")
    copied = dict(finding)
    if "metrics" not in finding:
        return copied, False
    metrics, cleared = _prepare_metric_list(finding["metrics"], version)
    copied["metrics"] = metrics
    _require_unavailable_finding(copied)
    return copied, cleared


def _require_unavailable_finding(finding: Mapping) -> None:
    metrics = finding.get("metrics")
    if not isinstance(metrics, (list, tuple)) or not metrics:
        return
    if any(metric.get("value") is not None for metric in metrics):
        return
    name = finding.get("id", "?")
    if finding.get("severity") != "info":
        raise BundleContractError(f"finding {name} is unavailable, so it must be info")
    score = finding.get("score", 0)
    if isinstance(score, bool) or not isinstance(score, numbers.Real) or float(score) != 0.0:
        raise BundleContractError(f"finding {name} is unavailable, so its score must be 0")
    moments = finding["moments"] if "moments" in finding else None
    if isinstance(moments, (str, bytes)) or not isinstance(moments, (list, tuple)):
        raise BundleContractError(f"finding {name} is unavailable, so moments must be a list")
    if moments:
        raise BundleContractError(f"finding {name} is unavailable, so it must not have a moment")


def _prepare_metric_group(metrics, version: int) -> tuple[dict | list, bool]:
    if isinstance(metrics, Mapping):
        copied = {}
        cleared = False
        for key, rows in metrics.items():
            prepared, changed = _prepare_metric_list(rows, version)
            copied[key] = prepared
            cleared = cleared or changed
        return copied, cleared
    return _prepare_metric_list(metrics, version)


def _prepare_metric_list(rows, version: int) -> tuple[list, bool]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, (list, tuple)):
        raise BundleContractError("metrics must be a list")
    copied = []
    cleared = False
    for row in rows:
        prepared, changed = _prepare_metric(row, version)
        copied.append(prepared)
        cleared = cleared or changed
    return copied, cleared


def _prepare_metric(record, version: int) -> tuple[dict, bool]:
    if not isinstance(record, Mapping):
        raise BundleContractError("metric must be a mapping")
    try:
        normalized = normalize_metric_record(record, version)
    except ValueError as exc:
        raise BundleContractError(str(exc)) from exc
    if "value" not in normalized:
        raise BundleContractError(f"metric {normalized.get('name', '?')} is missing value")
    try:
        _require_metric(normalized)
    except ValueError as exc:
        raise BundleContractError(str(exc)) from exc
    except KeyError as exc:
        raise BundleContractError(f"metric record is missing {exc}") from exc
    except TypeError as exc:
        raise BundleContractError(str(exc)) from exc
    cleared = (
        version == LEGACY_SCHEMA_VERSION
        and record.get("value") is not None
        and normalized.get("value") is None
    )
    return normalized, cleared


def _require_metric(record: Mapping) -> None:
    target = record.get("target")
    Metric(
        record["name"],
        record["label"],
        record["value"],
        record["unit"],
        Basis(record["basis"]),
        record["confidence"],
        tuple(target) if target else None,
        record.get("series"),
        record.get("n_obs", 1),
        record.get("best"),
    )
