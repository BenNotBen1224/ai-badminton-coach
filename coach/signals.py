"""Signal helpers shared by both analysis units. Implemented by the foundation unit.

Speeds are computed between measured samples using real time, never by differencing
neighbouring frames. A source that repeats frames makes frame differences alternate
between zero and double the true step, which fakes speed peaks.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def velocity(
    values: np.ndarray,
    t: np.ndarray,
    valid: np.ndarray,
    shots: Sequence[tuple[int, int]] = (),
    max_gap_s: float = 0.25,
) -> np.ndarray:
    """Velocity of values[n, ...] in units per second, same shape as values.

    For each frame, take the nearest valid sample before it and after it inside the same
    camera shot (the frame itself counts when valid) and divide their difference by the
    time between them. NaN when either side is missing, or the two samples are more
    than max_gap_s apart.
    """
    values = np.asarray(values, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.shape[0] != t.shape[0] or valid.shape[0] != t.shape[0]:
        raise ValueError("values, t and valid must share the frame axis")
    if t.ndim != 1:
        raise ValueError("t must be one series of timestamps")
    if not np.isfinite(t).all():
        raise ValueError("t must be finite")
    if np.isinf(values).any():
        raise ValueError("infinite value in the series")
    out = np.full(values.shape, np.nan, dtype=np.float64)
    n = t.shape[0]
    spans = tuple(shots) if len(shots) else ((0, n - 1),)
    for start, end in spans:
        _fill_shot(out, values, t, valid, int(start), int(end), float(max_gap_s))
    return out


def speed(
    values: np.ndarray,
    t: np.ndarray,
    valid: np.ndarray,
    shots: Sequence[tuple[int, int]] = (),
    max_gap_s: float = 0.25,
) -> np.ndarray:
    """Length of velocity() over the last axis, so values[n, 2] gives speed[n]."""
    vel = velocity(values, t, valid, shots, max_gap_s)
    if vel.ndim == 1:
        return np.abs(vel)
    return np.linalg.norm(vel, axis=-1)


def _fill_shot(out, values, t, valid, start, end, max_gap_s):
    n = t.shape[0]
    if n == 0 or end < start:
        return
    start = max(0, start)
    end = min(n - 1, end)
    idx = np.flatnonzero(valid[start:end + 1]) + start
    if idx.size < 2:
        return
    frames = np.arange(start, end + 1)
    pos = np.searchsorted(idx, frames, side="left")
    left = np.full(frames.shape, -1, dtype=int)
    right = np.full(frames.shape, -1, dtype=int)
    on = valid[frames]
    # A gap takes the valid samples that bracket it.
    span = ~on & (pos > 0) & (pos < idx.size)
    left[span] = idx[pos[span] - 1]
    right[span] = idx[pos[span]]
    # A valid frame counts as one side. Prefer the sample before it so the step ends
    # on the frame. If that step is too long, the sample after it is the other side.
    has_prev = on & (pos > 0)
    has_next = on & (pos + 1 < idx.size)
    left[has_prev] = idx[pos[has_prev] - 1]
    right[has_prev] = frames[has_prev]
    forward_only = on & ~has_prev & has_next
    left[forward_only] = frames[forward_only]
    right[forward_only] = idx[pos[forward_only] + 1]
    _apply_pair(out, values, t, frames, left, right, max_gap_s)
    too_far = has_prev & has_next & (right >= 0)
    if np.any(too_far):
        dt = t[right[too_far]] - t[left[too_far]]
        switch = np.zeros(frames.shape, dtype=bool)
        switch[np.flatnonzero(too_far)[dt > max_gap_s]] = True
        if np.any(switch):
            left[switch] = frames[switch]
            right[switch] = idx[pos[switch] + 1]
            _apply_pair(out, values, t, frames, left, right, max_gap_s, only=switch)


def _apply_pair(out, values, t, frames, left, right, max_gap_s, only=None):
    ok = (left >= 0) & (right > left)
    if only is not None:
        ok &= only
    if not np.any(ok):
        return
    dt = t[right[ok]] - t[left[ok]]
    keep = (dt > 0) & (dt <= max_gap_s)
    if not np.any(keep):
        return
    picked = np.flatnonzero(ok)[keep]
    delta = values[right[picked]] - values[left[picked]]
    extra = values.ndim - 1
    denom = dt[keep].reshape((-1, *([1] * extra))) if extra else dt[keep]
    out[frames[picked]] = delta / denom
