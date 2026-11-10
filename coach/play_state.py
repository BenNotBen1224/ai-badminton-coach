"""Activity support from alternating declared-arm swings or observed lunge cycles.

This is not shuttle tracking or confirmed rally segmentation. Quiet time before
the first observed movement and time after the last supported movement are unknown.
"""
from __future__ import annotations

import numpy as np

MAX_EXCHANGE_GAP_S = 3.0


def exchange_mask(tracks, events):
    mask = np.zeros(tracks.n, dtype=bool)
    for start, end in tracks.shots:
        swings = sorted(
            (event for event in events
             if event.kind == "swing" and start <= event.frame <= end),
            key=lambda event: event.frame,
        )
        for left, right in zip(swings, swings[1:]):
            if left.player == right.player:
                continue
            if float(tracks.t[right.frame] - tracks.t[left.frame]) > MAX_EXCHANGE_GAP_S:
                continue
            mask[left.frame:right.frame + 1] = True
        for event in events:
            if event.kind != "lunge" or not event.data.get("knee_stride"):
                continue
            first, last = event.frame, int(event.data["end_frame"])
            if start <= first <= last <= end:
                mask[first:last + 1] = True
    return mask


def exchange_events(tracks, events):
    mask = exchange_mask(tracks, events)
    kept = []
    for event in events:
        if event.kind == "lunge":
            start, end = int(event.frame), int(event.data["end_frame"])
        elif event.kind == "swing":
            start, end = int(event.data["start"]), int(event.data["end"])
        else:
            start = end = int(event.frame)
        if event.kind == "swing":
            supported = mask[event.frame]
        else:
            supported = bool(mask[start:end + 1].all())
        if supported and any(a <= start <= end <= b for a, b in tracks.shots):
            kept.append(event)
    return tuple(kept)
