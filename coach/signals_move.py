"""Array helpers for footwork and tactics.

Speeds always come from coach.signals, which differences measured samples in real
time. A step is a local peak in that smoothed speed, never a frame-to-frame spike
from a repeated video frame.
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np

from .schema import Role
from .signals import speed as signal_speed
from .tracks import COURT_LENGTH_M, COURT_WIDTH_M

# 3.0 m behind the net sits between the short service line (1.98 m) and the
# middle of a half (3.35 m). That is the fixed ready base.
BASE_BEHIND_NET_M = 3.0
# Doubles outline is 6.10 m. Singles sidelines sit 0.46 m in from each side.
SINGLES_ALLEY_M = 0.46
SINGLES_WIDTH_M = COURT_WIDTH_M - 2.0 * SINGLES_ALLEY_M
ZONE_M = 0.50
SLOW_M_S = 1.35
STEP_MIN_M_S = 1.70
STEP_GAP_S = 0.18
SMALL_STEP_MAX_M_S = 3.2
SPLIT_HOP_MIN = 0.015
SPLIT_DIP_MIN = 0.045
SPLIT_DIP_RUN = 4
# Association window only. Not a timing to aim for. A hop can land just after contact.
SPLIT_BEFORE_S = 0.50
SPLIT_AFTER_S = 0.20
RALLY_GAP_S = 2.5
RETURN_LOOK_S = 0.12
RETURN_CLOSE_M = 0.03
FINE_PX_PER_M = 28.0


def ankle_y(track) -> np.ndarray:
    """Mean image y of the visible ankles. NaN when neither ankle is visible."""
    y = track.xy[:, :, 1]
    left = track.visible[:, 15]
    right = track.visible[:, 16]
    out = np.full(track.n, np.nan)
    both = left & right
    out[both] = (y[both, 15] + y[both, 16]) * 0.5
    out[left & ~right] = y[left & ~right, 15]
    out[right & ~left] = y[right & ~left, 16]
    return out


def hip_y(track) -> np.ndarray:
    y = track.xy[:, :, 1]
    left = track.visible[:, 11]
    right = track.visible[:, 12]
    out = np.full(track.n, np.nan)
    both = left & right
    out[both] = (y[both, 11] + y[both, 12]) * 0.5
    out[left & ~right] = y[left & ~right, 11]
    out[right & ~left] = y[right & ~left, 12]
    return out


def _stature(track) -> np.ndarray:
    st = np.asarray(track.stature_px, dtype=np.float64)
    return np.where(np.isfinite(st) & (st > 1.0), st, np.nan)


def smooth_in_shots(values: np.ndarray, shots: Sequence[tuple[int, int]]) -> np.ndarray:
    """3-frame median inside each camera shot. NaN stays NaN when every sample is NaN."""
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan)
    for start, end in shots:
        out[start:end + 1] = _nanmedian3(values[start:end + 1])
    return out


def _nanmedian3(values: np.ndarray) -> np.ndarray:
    prev = np.empty_like(values)
    nxt = np.empty_like(values)
    prev[0] = np.nan
    prev[1:] = values[:-1]
    nxt[-1] = np.nan
    nxt[:-1] = values[1:]
    stacked = np.vstack([prev, values, nxt])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(stacked, axis=0)


def foot_speed(foot: np.ndarray, t: np.ndarray, measured: np.ndarray, shots) -> tuple[np.ndarray, np.ndarray]:
    """Raw and smoothed court speed. Raw is NaN on repeated and missing frames."""
    valid = measured & np.isfinite(foot).all(axis=1)
    raw = signal_speed(foot, t, valid, shots)
    smoothed = smooth_in_shots(raw, shots)
    # A repeated frame has no raw speed. Do not let the 3-frame median fill it in.
    smoothed = np.where(np.isfinite(raw), smoothed, np.nan)
    return raw, smoothed


def hop_and_dip(track, court, shots, window: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Hop and hip-dip ratios, both divided by stature_px.

    Hop is ankle lift above the standing floor. With a court, the floor is the
    rolling median court position unprojected back to pixels, so running toward
    the net does not look like a hop. Without a court, a slow image baseline is
    the fallback and the caller should trust it less.
    """
    ankles = ankle_y(track)
    hips = hip_y(track)
    stature = _stature(track)
    span = ankles - hips
    hop = _hop(track, court, ankles, stature, window)
    dip = np.full(track.n, np.nan)
    for start, end in shots:
        piece = span[start:end + 1]
        ok = np.isfinite(piece)
        if int(ok.sum()) < 5:
            continue
        standing = float(np.nanpercentile(piece, 70))
        dip[start:end + 1] = (standing - piece) / stature[start:end + 1]
    return hop, dip


def _hop(track, court, ankles, stature, window: int) -> np.ndarray:
    n = track.n
    if court is not None and np.isfinite(track.foot_court).any():
        floor = _rolling_median_xy(track.foot_court, window)
        floor_px = np.full((n, 2), np.nan)
        ok = np.isfinite(floor).all(axis=1)
        if ok.any():
            floor_px[ok] = court.unproject(floor[ok])
        hop = (floor_px[:, 1] - ankles) / stature
        return np.where(np.isfinite(hop), hop, np.nan)
    hop = np.full(n, np.nan)
    half = max(3, window)
    for i in range(n):
        piece = ankles[max(0, i - half):i + half + 1]
        piece = piece[np.isfinite(piece)]
        if piece.size < 5 or not np.isfinite(ankles[i]) or not np.isfinite(stature[i]):
            continue
        hop[i] = (float(np.percentile(piece, 80)) - ankles[i]) / stature[i]
    return hop


def _rolling_median_xy(points: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    n = points.shape[0]
    padded = np.full((n + 2 * half, points.shape[1]), np.nan)
    padded[half:half + n] = points
    stacked = np.stack([padded[i:i + n] for i in range(window)], axis=0)
    finite = np.isfinite(stacked).all(axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmedian(stacked, axis=0)
    out[finite.sum(axis=0) < 3] = np.nan
    return out


def fixed_base(role: Role, court) -> np.ndarray:
    y = court.net_y + BASE_BEHIND_NET_M if role == Role.NEAR else court.net_y - BASE_BEHIND_NET_M
    return np.array([court.width_m / 2.0, y], dtype=np.float64)


def habitual_base(foot: np.ndarray, t: np.ndarray, contacts: Sequence[int]) -> Optional[np.ndarray]:
    """Median feet in the last second before each opponent contact, then the median of those."""
    spots = []
    for frame in contacts:
        if frame <= 0:
            continue
        t0 = t[frame] - 1.0
        sl = (t >= t0) & (t < t[frame] - 0.04) & np.isfinite(foot).all(axis=1)
        if int(sl.sum()) < 3:
            continue
        spots.append(np.median(foot[sl], axis=0))
    if len(spots) < 3:
        return None
    return np.median(np.vstack(spots), axis=0)


def waiting_base(foot: np.ndarray, speed: np.ndarray, role: Role, court) -> np.ndarray:
    """Fallback when there are not enough opponent contacts: median of slow frames in the own half."""
    own = _own_half(foot[:, 1], role, court.net_y)
    slow = own & np.isfinite(foot).all(axis=1) & np.isfinite(speed) & (speed <= SLOW_M_S)
    if int(slow.sum()) < 5:
        slow = own & np.isfinite(foot).all(axis=1)
    if not slow.any():
        return fixed_base(role, court)
    return np.median(foot[slow], axis=0)


def _own_half(y: np.ndarray, role: Role, net_y: float) -> np.ndarray:
    return y > net_y if role == Role.NEAR else y < net_y


def distances(foot: np.ndarray, base: np.ndarray) -> np.ndarray:
    out = np.full(foot.shape[0], np.nan)
    ok = np.isfinite(foot).all(axis=1) & np.isfinite(base).all()
    out[ok] = np.linalg.norm(foot[ok] - base, axis=1)
    return out


def split_landings(hop: np.ndarray, dip: np.ndarray, shots) -> np.ndarray:
    """Frames where a brief hop is followed by a short hip dip.

    The landing frame is the dip. A lunge dips for much longer, so a dip that
    stays down is rejected. Found from vertical signals only.
    """
    n = hop.shape[0]
    hits = []
    high = np.isfinite(dip) & (dip >= SPLIT_DIP_MIN)
    for start, end in shots:
        for i in range(max(start + 2, 2), end - 3):
            if not high[i]:
                continue
            if np.isfinite(dip[i - 1]) and dip[i] < dip[i - 1]:
                continue
            if np.isfinite(dip[i + 1]) and dip[i] < dip[i + 1]:
                continue
            window = dip[i - 2:i + 7]
            if int(np.nansum(window > 0.04)) > SPLIT_DIP_RUN:
                continue
            prev = hop[max(start, i - 3):i]
            if not np.isfinite(prev).any() or float(np.nanmax(prev)) < SPLIT_HOP_MIN:
                continue
            hits.append(i)
    return np.asarray(hits, dtype=int)


def pair_splits(landings: np.ndarray, contacts: Sequence[int], t: np.ndarray) -> list[tuple[int, int]]:
    """Pair each hop landing with the nearest opponent contact in the association window.

    The window is only for matching. It is not a good-timing band.
    """
    pairs = []
    for landing in landings:
        for contact in contacts:
            dt = float(t[int(landing)] - t[int(contact)])
            if -SPLIT_BEFORE_S <= dt <= SPLIT_AFTER_S:
                pairs.append((abs(dt), int(landing), int(contact)))
    pairs.sort()
    used_landing: set[int] = set()
    used_contact: set[int] = set()
    matched = []
    for _, landing, contact in pairs:
        if landing in used_landing or contact in used_contact:
            continue
        used_landing.add(landing)
        used_contact.add(contact)
        matched.append((landing, contact))
    return matched


def hop_onset(hop: np.ndarray, landing: int, start: int) -> int:
    """First frame of the hop, kept separate from the landing (the hip dip)."""
    i = int(landing)
    while i > start and np.isfinite(hop[i - 1]) and hop[i - 1] >= SPLIT_HOP_MIN:
        i -= 1
    return i


def step_peaks(raw: np.ndarray, smoothed: np.ndarray, t: np.ndarray, shots, min_speed: float = STEP_MIN_M_S) -> np.ndarray:
    """Local peaks of smoothed foot speed.

    The raw speed at the peak must be finite, so a repeated frame cannot become
    a step. Peaks closer than STEP_GAP_S are one step; the higher one is kept.
    """
    picked = []
    for start, end in shots:
        for i in range(start + 1, end):
            value = smoothed[i]
            if not np.isfinite(value) or value < min_speed or not np.isfinite(raw[i]):
                continue
            left, right = smoothed[i - 1], smoothed[i + 1]
            if np.isfinite(left) and value < left:
                continue
            if np.isfinite(right) and value < right:
                continue
            if (not np.isfinite(left) or value <= left) and (not np.isfinite(right) or value <= right):
                continue
            if picked and t[i] - t[picked[-1]] < STEP_GAP_S:
                if value >= smoothed[picked[-1]]:
                    picked[-1] = i
                continue
            picked.append(i)
    return np.asarray(picked, dtype=int)


def first_return(dist: np.ndarray, raw_speed: np.ndarray, t: np.ndarray, start: int, stop: int, measured: np.ndarray) -> Optional[int]:
    """First measured frame after `start` that is inside the zone and has settled.

    Settled means a finite speed at or under SLOW_M_S, and no nearer measured
    position within RETURN_LOOK_S. A repeated frame or a dropout is not a plant.
    """
    if stop <= start:
        return None
    measured = np.asarray(measured, dtype=bool)
    dt = float(np.median(np.diff(t[start:stop + 1]))) if stop > start else 1.0 / 30.0
    look = max(1, int(round(RETURN_LOOK_S / max(dt, 1e-3))))
    for i in range(start + 1, stop + 1):
        if i >= measured.shape[0] or not measured[i]:
            continue
        if not np.isfinite(dist[i]) or dist[i] > ZONE_M:
            continue
        speed = raw_speed[i]
        if not np.isfinite(speed) or speed > SLOW_M_S:
            continue
        ahead_idx = np.arange(i, min(stop, i + look) + 1)
        ahead_idx = ahead_idx[measured[ahead_idx]]
        ahead = dist[ahead_idx]
        ahead = ahead[np.isfinite(ahead)]
        if ahead.size and dist[i] > float(ahead.min()) + RETURN_CLOSE_M:
            continue
        return i
    return None


def measured_gap(measured: np.ndarray, t: np.ndarray, start: int, end: int, limit_s: float = 0.25) -> bool:
    """True when measured frames do not cover the window within limit_s.

    A hole between samples counts, and so does a missing start or end.
    """
    start = max(0, int(start))
    end = min(len(t) - 1, int(end))
    if end < start:
        return True
    idx = np.flatnonzero(np.asarray(measured[start:end + 1], dtype=bool)) + start
    if idx.size < 2:
        return True
    if float(t[idx[0]] - t[start]) > limit_s or float(t[end] - t[idx[-1]]) > limit_s:
        return True
    return bool(np.any(np.diff(t[idx]) > limit_s))


def fine_motion_reliable(track) -> bool:
    """Far-player feet are only a few dozen pixels, so small hops are often noise."""
    ppm = np.nanmedian(track.px_per_m)
    stature = np.nanmedian(track.stature_px)
    return bool(np.isfinite(ppm) and ppm >= FINE_PX_PER_M and np.isfinite(stature) and stature >= 70.0)


def coverage_cells(foot: np.ndarray, role: Role, court) -> np.ndarray:
    """Cell id 0..8 on the player's singles half, -1 when unknown or on the other half.

    x is measured on the doubles rectangle. The 0.46 m alleys are folded into the
    side columns so the grid is the singles court, not the doubles lines.
    """
    n = foot.shape[0]
    out = np.full(n, -1, dtype=int)
    ok = np.isfinite(foot).all(axis=1)
    if not ok.any():
        return out
    x = np.clip(foot[:, 0], SINGLES_ALLEY_M, COURT_WIDTH_M - SINGLES_ALLEY_M)
    col = np.zeros(n, dtype=int)
    col[ok] = np.minimum(2, ((x[ok] - SINGLES_ALLEY_M) / (SINGLES_WIDTH_M / 3.0)).astype(int))
    if role == Role.NEAR:
        own = ok & (foot[:, 1] > court.net_y)
        span = max(court.length_m - court.net_y, 1e-6)
        row = np.zeros(n, dtype=int)
        row[ok] = np.minimum(2, ((np.clip(foot[ok, 1], court.net_y, court.length_m) - court.net_y) / span * 3.0).astype(int))
    else:
        own = ok & (foot[:, 1] < court.net_y)
        span = max(court.net_y, 1e-6)
        row = np.zeros(n, dtype=int)
        row[ok] = np.minimum(2, (np.clip(foot[ok, 1], 0.0, court.net_y) / span * 3.0).astype(int))
    out[own] = row[own] * 3 + col[own]
    return out


def cell_centre(cell: int, role: Role, court) -> np.ndarray:
    row, col = divmod(int(cell), 3)
    x = SINGLES_ALLEY_M + (col + 0.5) * SINGLES_WIDTH_M / 3.0
    if role == Role.NEAR:
        span = court.length_m - court.net_y
        y = court.net_y + (row + 0.5) * span / 3.0
    else:
        y = (row + 0.5) * court.net_y / 3.0
    return np.array([x, y], dtype=np.float64)


def rally_bounds(swing_frames: Sequence[int], t: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive groups of swing frames. A gap above 2.5 s starts a new rally."""
    frames = sorted(int(f) for f in swing_frames)
    if not frames:
        return []
    groups = [[frames[0]]]
    for frame in frames[1:]:
        if t[frame] - t[groups[-1][-1]] > RALLY_GAP_S:
            groups.append([frame])
        else:
            groups[-1].append(frame)
    return [(g[0], g[-1]) for g in groups]


def median(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def confidence(n_obs: int, *, pose: float = 1.0, fine: bool = False, fine_ok: bool = True,
               preview: bool = False) -> float:
    """Trust drops when few reps, a weak pose, a far player, or a preview document back the number.

    A player who stands still repeats the same pose. Those frames are not new
    speed samples, but they are not a weak measurement of where they stood.
    """
    value = 0.88
    if n_obs < 3:
        value *= 0.55
    elif n_obs < 6:
        value *= 0.86
    value *= 0.60 + 0.40 * float(np.clip(pose, 0.0, 1.0))
    if fine and not fine_ok:
        value = min(value, 0.42)
    if preview:
        value *= 0.82
    return float(np.clip(value, 0.05, 0.95))
