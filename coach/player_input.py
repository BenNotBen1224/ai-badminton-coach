"""Explicit player metadata; pose motion is not evidence of racket handedness."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

Hand = Literal["left", "right", "unknown"]


def parse_hand(value) -> Hand:
    if value not in ("left", "right", "unknown"):
        raise ValueError("Player hand must be left, right or unknown")
    return value


def player_inputs(video, attribution_file=None, overrides=None):
    video = Path(video)
    if attribution_file is not None:
        paths = [Path(attribution_file)]
    else:
        paths = sorted(video.parent.glob("*attribution*.json"))
    players = {}
    for path in paths:
        metadata = json.loads(path.read_text("utf-8"))
        if attribution_file is None and not any(c.get("file") == video.name for c in metadata.get("clips", [])):
            continue
        declared = dict(metadata.get("players", {}))
        for clip in metadata.get("clips", []):
            if clip.get("file") == video.name:
                declared.update(clip.get("players", {}))
        for role, player in declared.items():
            if role not in ("near", "far"):
                raise ValueError(f"Unknown player role in attribution: {role}")
            players[role] = {**player, "hand": parse_hand(player.get("hand", "unknown"))}
    for role, hand in (overrides or {}).items():
        if role not in ("near", "far"):
            raise ValueError(f"Unknown player role: {role}")
        players[role] = {"hand": parse_hand(hand), "evidence": "Explicit CLI/API input"}
    return players
