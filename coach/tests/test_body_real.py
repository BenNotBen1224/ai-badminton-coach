"""Real-clip stability. Skips when a pose file is missing.

Clip 01 hand labels are a bounded model-assisted reference, not a full-clip
score. Clip 05 stays held out. The skipped test records that limit.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from coach.analyze import assemble
from coach.analyze_body import body_analysis
from coach.catalog import METRIC_CATALOG
from coach.events import detect_events
from coach.schema import Partial
from coach.tracks import load_tracks

PROJECT = Path(__file__).resolve().parents[2]
DENSE = PROJECT / "fast" / "runs" / "step4-rebalanced-dev" / "outputs"
BASELINE = PROJECT / "fast" / "runs"
CLIPS = ("01", "02", "03", "04", "05")


def _load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _hip_y(player):
    joints = player.get("joints") or []
    ys = [joint[1] for joint in (joints[11:13] if len(joints) >= 13 else []) if joint and len(joint) >= 2]
    return None if not ys else sum(ys) / len(ys)


def _with_confidence(joints):
    """Judge files store x,y only. 0.90 lets the contract visibility gate run."""
    out = []
    for joint in joints:
        if joint is None or len(joint) < 2:
            out.append(None)
        elif len(joint) >= 3:
            out.append(joint)
        else:
            out.append([joint[0], joint[1], 0.90])
    return out


def _assign_roles(players):
    if players and all(player.get("role") in ("near", "far") for player in players):
        return players
    scored = [(y, player) for player in players if (y := _hip_y(player)) is not None]
    if len(scored) < 2:
        return []
    scored.sort(key=lambda item: item[0])
    far_y, far = scored[0]
    near_y, near = scored[-1]
    if near_y - far_y < 40:
        return []
    return [{**far, "role": "far"}, {**near, "role": "near"}]


def _pose_document(path: Path):
    """Read a pose file. Baseline and judge files are not the Tracks contract, so adapt them here."""
    raw = _load(path)
    if isinstance(raw.get("source"), dict) and raw["source"].get("frame_count"):
        return raw
    clip = path.stem.split("-")[-1]
    dense = _load(DENSE / f"{clip}-r1.json")
    source = dense["source"]
    frames = []
    for frame in raw.get("frames") or []:
        people = frame.get("players")
        if people is None:
            people = frame.get("people") or []
            people = [{**person, "joints": _with_confidence(person.get("joints") or [])} for person in people]
        frames.append({
            "frame_index": frame.get("frame_index"),
            "timestamp_seconds": frame.get("timestamp_seconds"),
            "players": _assign_roles(people),
        })
    return {"source": source, "frames": frames, "tier": "refined"}


def _partial(path: Path):
    tracks = load_tracks(_pose_document(path))
    events = detect_events(tracks)
    body = body_analysis(tracks, events)
    report = assemble(tracks, events, (body, Partial()), METRIC_CATALOG)
    return tracks, events, body, report


class RealBodyTests(unittest.TestCase):
    def test_three_pose_sources_agree_on_clip_01(self):
        paths = {
            "dense": DENSE / "01-r1.json",
            "baseline": BASELINE / "baseline-01.json",
            "judge": BASELINE / "judge-01.json",
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            self.skipTest("missing pose files: " + ", ".join(missing))
        built = {name: _partial(path) for name, path in paths.items()}
        dense_swings = [event for event in built["dense"][1] if event.kind == "swing"]
        for name in ("baseline", "judge"):
            other = [event for event in built[name][1] if event.kind == "swing"]
            if len(dense_swings) >= 2:
                ratio = (len(other) + 1) / (len(dense_swings) + 1)
                self.assertGreater(ratio, 0.25)
                self.assertLess(ratio, 4.0)
            self.assertEqual(
                {finding.id for finding in built["dense"][2].findings},
                {finding.id for finding in built[name][2].findings},
            )
            self.assertGreater(built[name][3].n_frames, 0)

    def test_dense_clips_validate_when_present(self):
        present = [DENSE / f"{clip}-r1.json" for clip in CLIPS if (DENSE / f"{clip}-r1.json").is_file()]
        if not present:
            self.skipTest("dense pose outputs are missing")
        for path in present:
            _tracks, events, body, report = _partial(path)
            self.assertGreater(report.n_frames, 0)
            self.assertTrue(body.findings)
            self.assertTrue(all(event.frame < report.n_frames for event in events))

    def test_hand_labels_are_not_scored_as_full_clip_truth(self):
        self.skipTest(
            "clip 05 is held out; clip 01 contacts are four model-assisted "
            "references inside frames 482-603, not full-clip precision"
        )


if __name__ == "__main__":
    unittest.main()
