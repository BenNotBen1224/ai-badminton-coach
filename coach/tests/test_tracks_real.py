"""Real pose documents for clips 01 to 05 and 08.

Skips with a clear message when a frozen JSON file is missing. Clip 04's
repeat count is bound to the hash of that frozen file, not to a percentage.
"""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from coach.schema import Role
from coach.tracks import JOINT_CONF_GATE, load_tracks

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "fast" / "runs" / "step4-rebalanced-dev" / "outputs"
HOLD = ROOT / "fast" / "runs" / "step4-rebalanced-holdout08" / "outputs"
CLIPS = {f"{i:02d}": DEV / f"{i:02d}-r1.json" for i in range(1, 6)}
CLIPS["08"] = HOLD / "08-r1.json"
CLIP04_SHA256 = "31f554fcb9fea80ba2749bf1f6f2cfccba617fbc1129505e9ab3cb13abe27bab"
CLIP04_REPEATS = (369, 192, 177)


def _load(name):
    path = CLIPS[name]
    if not path.is_file():
        raise unittest.SkipTest(f"missing pose JSON for clip {name}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _same_pose(left, right):
    if left is None or right is None or len(left) != 17 or len(right) != 17:
        return False
    for a, b in zip(left, right):
        if a is None or b is None:
            if a is not b:
                return False
            continue
        for x, y in zip(a, b):
            x, y = float(x), float(y)
            if x != y and not (x != x and y != y):
                return False
    return True


def _source_unmeasured(doc):
    """Present rows that are interpolated or an exact repeat of the previous frame."""
    roles = ("near", "far")
    cuts = {int(c) for c in doc.get("cuts") or []}
    chosen_frames = []
    for frame in doc["frames"]:
        chosen = {}
        for player in frame.get("players") or []:
            role = player.get("role")
            if role not in roles:
                continue
            score = player.get("box_confidence")
            score = 0.0 if score is None else float(score)
            prev = chosen.get(role)
            if prev is None or score > prev[0]:
                chosen[role] = (score, player)
        chosen_frames.append(chosen)
    unmeasured = set()
    for i, chosen in enumerate(chosen_frames):
        for role in roles:
            if role not in chosen:
                continue
            player = chosen[role][1]
            if player.get("src") == "interpolated":
                unmeasured.add((i, role))
                continue
            if i == 0 or i in cuts:
                continue
            prev = chosen_frames[i - 1].get(role)
            if prev is not None and _same_pose(player.get("joints"), prev[1].get("joints")):
                unmeasured.add((i, role))
    return unmeasured


class RealTrackTests(unittest.TestCase):
    def test_clips_match_the_loader_contract(self):
        for name in ("01", "02", "03", "04", "05", "08"):
            with self.subTest(clip=name):
                doc = _load(name)
                tracks = load_tracks(doc)
                self.assertEqual(tracks.n, int(doc["source"]["frame_count"]))
                self.assertIsNotNone(tracks.court)
                near = tracks.players[Role.NEAR]
                far = tracks.players[Role.FAR]
                self.assertGreaterEqual(float(near.present.mean()), 0.90)
                self.assertGreaterEqual(float(far.present.mean()), 0.90)
                near_h = float(np.nanmean(near.stature_px))
                far_h = float(np.nanmean(far.stature_px))
                self.assertGreater(near_h, 0.0)
                self.assertGreater(far_h, 0.0)
                self.assertGreater(near_h, far_h)
                for track in (near, far):
                    foot = track.foot_court
                    ok = np.isfinite(foot).all(axis=1)
                    self.assertGreater(int(ok.sum()), 0)
                    kept = foot[ok]
                    inside = (
                        (kept[:, 0] >= -2.0) & (kept[:, 0] <= tracks.court.width_m + 2.0)
                        & (kept[:, 1] >= -2.0) & (kept[:, 1] <= tracks.court.length_m + 2.0)
                    )
                    self.assertGreaterEqual(float(inside.mean()), 0.80)
                    finite_xy = np.isfinite(track.xy).all(axis=2)
                    expect = (
                        track.present[:, None] & finite_xy
                        & np.isfinite(track.conf) & (track.conf >= JOINT_CONF_GATE)
                    )
                    self.assertTrue(np.array_equal(track.visible, expect))
                if name == "04":
                    digest = hashlib.sha256(CLIPS["04"].read_bytes()).hexdigest()
                    self.assertEqual(digest, CLIP04_SHA256)
                    source = _source_unmeasured(doc)
                    loader = set()
                    for role_name, track in (("near", near), ("far", far)):
                        for i in range(tracks.n):
                            if track.present[i] and not track.measured[i]:
                                loader.add((i, role_name))
                    self.assertEqual(loader, source)
                    total, near_n, far_n = CLIP04_REPEATS
                    self.assertEqual(len(source), total)
                    self.assertEqual(sum(role == "near" for _, role in source), near_n)
                    self.assertEqual(sum(role == "far" for _, role in source), far_n)
                    self.assertIn("repeated_frames", tracks.warnings)


if __name__ == "__main__":
    unittest.main()
