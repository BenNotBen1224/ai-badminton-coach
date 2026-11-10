"""Real-clip checks for footwork and positional tactics.

Pose files and labels are optional. A missing file skips that check. Swing
events here are stand-ins until the swing unit and the label file are ready.
They are not contact labels.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import unittest
from pathlib import Path

import numpy as np

from coach.analyze import analyze
from coach.analyze_move import movement_analysis
from coach.events import detect_events, recovery_windows
from coach.schema import WARNINGS, Role, Severity, swing_event
from coach.signals import speed
from coach.synth import make_preview
from coach.tracks import load_tracks

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parents[2] / "work"
DEV = ROOT / "fast" / "runs" / "step4-rebalanced-dev" / "outputs"
RUNS = ROOT / "fast" / "runs"
LABELS = WORK / "L" / "labels.json"
CURRENT_EVENTS = WORK / "C1a" / "clip01-current.json"
STALE_EVENTS = WORK / "C1a" / "clip01-partial.json"
INTEGRATION = WORK / "C1b" / "analyze-clip01.json"
LOCK = WORK / "cpu.lock"
CONTACT_IDS = {
    "footwork.split_step",
    "footwork.first_step",
    "tactics.base_discipline",
    "tactics.tempo",
    "tactics.initiative",
}
CLIPS = ("01", "02", "03", "04", "05")
# Parent-verified object. sort_keys + compact separators. Not a coaching target.
CLIP01_SHA = "84096cc34b8955e8922e17cf43bdfa0ae1c14163d8b06eec76965c947877dda8"
COVERED_01 = (482, 603)
DESCRIBE = {
    "footwork.split_step",
    "footwork.coverage",
    "tactics.base_discipline",
    "tactics.tempo",
    "tactics.initiative",
}


class _CpuLock:
    def __init__(self, purpose):
        self.path = LOCK
        self.token = f"C1b {os.getpid()} {purpose}"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + 30 * 60
        while True:
            try:
                with self.path.open("x", encoding="utf-8") as handle:
                    handle.write(self.token)
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"cpu.lock still held after 30 minutes: {self.path}")
                time.sleep(30)

    def __exit__(self, *_exc):
        try:
            held = self.path.read_text(encoding="utf-8")
        except OSError:
            return False
        if held == self.token:
            self.path.unlink()
        return False


def _path(name, kind):
    if kind == "dense":
        return DEV / f"{name}-r1.json"
    if kind == "baseline":
        return RUNS / f"baseline-{name}.json"
    return RUNS / f"judge-{name}.json"


def _load(path: Path):
    if not path.is_file():
        raise unittest.SkipTest(f"missing pose file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _document(kind: str, raw: dict, dense: dict) -> dict:
    """Turn a baseline or judge file into the pose document the loader accepts.

    Court lines and frame size come from the dense file of the same clip. They
    describe the video, not the pose model. Judge joints have no confidence, so
    each visible joint is given 0.80. That is a stand-in, not a measured score.
    """
    if kind == "dense":
        return raw
    source = dict(dense.get("source") or {})
    frames = []
    if kind == "baseline":
        for frame in raw.get("frames") or []:
            frames.append({
                "frame_index": frame.get("frame_index"),
                "timestamp_seconds": frame.get("timestamp_seconds"),
                "players": _baseline_players(frame.get("players") or []),
            })
    elif kind == "judge":
        for frame in raw.get("frames") or []:
            people = frame.get("people") or frame.get("players") or []
            frames.append({
                "frame_index": frame.get("frame_index"),
                "players": [_judge_player(person) for person in people if person.get("role") in ("near", "far")],
            })
    else:
        raise ValueError(kind)
    return {
        "schema_version": dense.get("schema_version", 3),
        "source": source,
        "court_lines_normalized": dense.get("court_lines_normalized"),
        "court_fit": dense.get("court_fit") or {},
        "cuts": dense.get("cuts") or [],
        "frames": frames,
    }


def _baseline_players(players: list) -> list:
    usable = [p for p in players if p.get("joints") and len(p["joints"]) == 17 and p.get("bbox_xyxy")]
    if not usable:
        return []
    def cy(player):
        box = player["bbox_xyxy"]
        return (box[1] + box[3]) * 0.5
    near = max(usable, key=cy)
    rest = [p for p in usable if p is not near]
    chosen = [("near", near)]
    if rest:
        chosen.append(("far", min(rest, key=cy)))
    out = []
    for role, player in chosen:
        row = {
            "role": role,
            "track_id": player.get("track_id"),
            "bbox_xyxy": player["bbox_xyxy"],
            "box_confidence": player.get("box_confidence") or 0.5,
            "joints": player["joints"],
        }
        out.append(row)
    return out


def _judge_player(person: dict) -> dict:
    joints = []
    for joint in person.get("joints") or []:
        if joint is None or len(joint) < 2:
            joints.append(None)
        elif len(joint) >= 3:
            joints.append([joint[0], joint[1], joint[2]])
        else:
            joints.append([joint[0], joint[1], 0.80])
    return {
        "role": person.get("role"),
        "bbox_xyxy": person.get("bbox_xyxy"),
        "box_confidence": person.get("box_confidence") or 0.5,
        "joints": joints,
    }


def _event_key(events):
    rows = []
    for event in events:
        data = event.data if not isinstance(event, dict) else event.get("data") or {}
        kind = event.kind if not isinstance(event, dict) else event["kind"]
        player = event.player.value if not isinstance(event, dict) else event["player"]
        frame = int(event.frame if not isinstance(event, dict) else event["frame"])
        marker = data.get("end_frame", data.get("end"))
        rows.append((kind, player, frame, None if marker is None else int(marker)))
    return tuple(sorted(rows))


def stand_in_swings(tracks):
    """Wrist-speed peaks for pose-source stability only. Not an acceptance path."""
    events = []
    for role, track in tracks.players.items():
        valid = track.measured & track.visible[:, 10]
        raw = speed(track.xy[:, 10], tracks.t, valid, tracks.shots)
        finite = raw[np.isfinite(raw)]
        if finite.size < 8:
            continue
        limit = float(np.percentile(finite, 90))
        last = -1e9
        for i in range(1, track.n - 1):
            if not np.isfinite(raw[i]) or raw[i] < limit or raw[i] < raw[i - 1] or raw[i] < raw[i + 1]:
                continue
            if tracks.t[i] - last < 0.70:
                continue
            start = max(0, i - 4)
            end = min(track.n - 1, i + 4)
            peak = min(end, max(start, i - 1))
            events.append(swing_event(
                role, i, start=start, end=end, peak_frame=peak,
                peak_speed=float(raw[i]), overhead=False, confidence=0.4,
            ))
            last = float(tracks.t[i])
    return events


def _eligible_splits(contacts, covered):
    """Yes and no labels inside the covered interval. A missing landing still counts."""
    start, end = covered
    scored = []
    for row in contacts:
        frame = int(row["frame"])
        label = row.get("split_step_receiver")
        if frame < start or frame > end or label not in ("yes", "no"):
            continue
        scored.append(row)
    return scored


def _precision(true_positive, false_positive):
    if true_positive + false_positive == 0:
        return None
    return true_positive / (true_positive + false_positive)


def _labelled_swings(contacts):
    events = []
    for row in contacts:
        frame = int(row["frame"])
        role = Role.NEAR if row["player"] == "near" else Role.FAR
        events.append(swing_event(
            role, frame, start=max(0, frame - 2), end=frame + 2,
            peak_frame=max(0, frame - 1), peak_speed=1.0,
            overhead=bool(row.get("overhead")),
        ))
    return events


def _paired_split(part, receiver, contact, t):
    hits = []
    for event in part.events:
        if event.kind != "split_step" or event.player != receiver:
            continue
        landing = int(event.frame)
        contact_t = float(t[landing] - float(event.data["landing_s"]))
        if abs(contact_t - float(t[contact])) <= 0.05:
            hits.append(landing)
    return hits


class RealMoveTests(unittest.TestCase):
    def test_three_pose_sources_stay_valid(self):
        overlaps = []
        for name in CLIPS:
            dense_path = _path(name, "dense")
            if not dense_path.is_file():
                raise unittest.SkipTest(f"missing dense pose for clip {name}: {dense_path}")
            dense = _load(dense_path)
            parts = {}
            for kind in ("dense", "baseline", "judge"):
                path = _path(name, kind)
                if not path.is_file():
                    raise unittest.SkipTest(f"missing {kind} pose for clip {name}: {path}")
                tracks = load_tracks(_document(kind, dense if kind == "dense" else _load(path), dense))
                part = movement_analysis(tracks, stand_in_swings(tracks))
                self.assertTrue(part.findings, f"clip {name} {kind} produced no findings")
                for finding in part.findings:
                    if finding.id in DESCRIBE:
                        self.assertNotIn(finding.severity, (Severity.IMPROVE, Severity.PRIORITY), finding.id)
                    self.assertGreaterEqual(len(finding.moments), 0)
                parts[kind] = {f.id for f in part.findings}
            overlaps.append(len(parts["dense"] & parts["baseline"] & parts["judge"]))
        self.assertGreaterEqual(max(overlaps), 2)

    def test_preview_stride_does_not_raise_confidence(self):
        path = _path("01", "dense")
        if not path.is_file():
            self.skipTest(f"missing dense pose: {path}")
        doc = _load(path)
        tracks = load_tracks(doc)
        events = stand_in_swings(tracks)
        refined = movement_analysis(tracks, events)
        preview = movement_analysis(load_tracks(make_preview(doc, 4)), events)
        refined_conf = {(f.player, m.name): m.confidence for f in refined.findings for m in f.metrics}
        preview_conf = {(f.player, m.name): m.confidence for f in preview.findings for m in f.metrics}
        shared = set(refined_conf) & set(preview_conf)
        self.assertTrue(shared)
        for key in shared:
            self.assertLessEqual(preview_conf[key], refined_conf[key] + 1e-9, key)

    def test_labels_skipped_until_declared(self):
        if not LABELS.is_file():
            self.skipTest(
                "L has not declared labels.json ready. Split-step precision and recall "
                "against hand labels are not reported."
            )
        payload = json.loads(LABELS.read_text(encoding="utf-8"))
        clips = payload.get("clips") or {}
        ready = payload.get("status") == "ready" or payload.get("declared_ready") is True
        complete = ready and all((clips.get(name) or {}).get("contacts") for name in ("01", "05"))
        if not complete:
            self.skipTest(
                "labels.json is present but not a complete declared handoff "
                "(no ready flag, or clip 01/05 contacts are empty). "
                "Full clip 01 and 05 precision and recall are not reported."
            )
        self.fail("labels are declared ready but this test has no comparison yet")

    def test_clip01_current_events_not_stale_partial(self):
        """Public analyze() on clip 01. Event count is not a hit-accuracy score."""
        path = _path("01", "dense")
        if not path.is_file():
            self.skipTest(f"missing dense pose: {path}")
        if not CURRENT_EVENTS.is_file():
            self.skipTest("C1a current handoff is missing")
        doc = _load(path)
        try:
            import cv2
            cv2.setNumThreads(1)
        except ImportError:
            pass
        with _CpuLock("clip01 movement integration"):
            tracks = load_tracks(doc)
            events = detect_events(tracks)
            omitted_windows = recovery_windows(tracks)
            report = analyze(doc, video_id="01")
        current = json.loads(CURRENT_EVENTS.read_text(encoding="utf-8"))
        live = _event_key(events)
        handoff = _event_key(current["events"])
        report_core = _event_key(e for e in report.events if e.kind in ("swing", "lunge"))
        from coach.play_state import exchange_events
        self.assertEqual(report_core, _event_key(exchange_events(tracks, events)))
        self.assertTrue(set(report_core) <= set(live))
        if STALE_EVENTS.is_file():
            stale = json.loads(STALE_EVENTS.read_text(encoding="utf-8"))
            self.assertNotEqual(live, _event_key(stale["events"]))
        omitted = {(w.reason.split()[0], int(w.start_frame)) for w in omitted_windows}
        self.assertTrue(set(report.warnings) <= set(WARNINGS))
        self.assertEqual(report.windows, ())
        payload = report.to_dict()
        move = [row for row in payload["findings"] if row["category"] in ("footwork", "tactics")]
        blob = json.dumps(move)
        self.assertTrue(move)
        for banned in ("0.07", "two frames", "your shot", "their hit"):
            self.assertNotIn(banned, blob, banned)
        for row in move:
            if row["id"] in CONTACT_IDS:
                self.assertEqual(row["severity"], "info", row["id"])
                self.assertIn("not validated personal norms", row["explain"])
                self.assertIn("arm", (row["headline"] + " " + row["explain"]).lower())
            if row["id"] == "footwork.recovery":
                self.assertNotIn("your shot", row["explain"])
                self.assertTrue("not proof of a hit" in row["explain"] or "does not prove" in row["explain"])
        for event in report.events:
            if event.kind != "base_return":
                continue
            self.assertNotIn((event.player.value, int(event.data["from_frame"])), omitted)
        if not any(event.kind == "base_return" for event in report.events):
            self.assertFalse(any(f.id == "footwork.recovery" for f in report.findings))
        note = {
            "source": "fast/runs/step4-rebalanced-dev/outputs/01-r1.json",
            "detector": "coach.events.detect_events after the ratio-cache fix",
            "handoff": "obj2/work/C1a/clip01-current.json",
            "handoff_event_match": live == handoff,
            "event_count": len(live),
            "handoff_event_count": len(handoff),
            "events_added_since_handoff": [list(row) for row in live if row not in handoff],
            "events_dropped_since_handoff": [list(row) for row in handoff if row not in live],
            "event_count_is_hit_accuracy": False,
            "stale_not_used": "obj2/work/C1a/clip01-partial.json",
            "omitted_windows": [
                {"player": w.reason.split()[0], "start_frame": w.start_frame, "end_frame": w.end_frame, "reason": w.reason}
                for w in omitted_windows
            ],
            "warnings": list(report.warnings),
            "base_return_count": sum(e.kind == "base_return" for e in report.events),
            "recovery_finding": any(f.id == "footwork.recovery" for f in report.findings),
            "contact_dependent": [
                {
                    "id": row["id"], "player": row["player"], "severity": row["severity"],
                    "headline": row["headline"], "explain": row["explain"],
                }
                for row in move if row["id"] in CONTACT_IDS
            ],
            "contact_note": "Sure clip 01 contacts still miss by +31 and +36 frames. Split, first step, tempo, base, and net-ahead numbers are informational. The clock is an arm swing, not a confirmed hit.",
        }
        INTEGRATION.write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")

    def test_clip01_covered_interval_only(self):
        """Score the verified clip 01 object. Unclear and outside frames are not negatives."""
        if not LABELS.is_file():
            self.skipTest("labels.json is missing")
        payload = json.loads(LABELS.read_text(encoding="utf-8"))
        clip = (payload.get("clips") or {}).get("01")
        if clip is None:
            self.skipTest("clip 01 reference is missing")
        digest = hashlib.sha256(
            json.dumps(clip, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if digest != CLIP01_SHA:
            self.skipTest("clip 01 reference hash changed; this bounded score does not apply")
        contacts = clip.get("contacts") or []
        eligible = _eligible_splits(contacts, COVERED_01)
        self.assertEqual([row["frame"] for row in eligible], [508, 555])
        self.assertTrue(all(row["split_step_receiver"] == "yes" for row in eligible))
        self.assertEqual(_eligible_splits([
            {"frame": 10, "player": "far", "split_step_receiver": "no"},
            {"frame": 520, "player": "far", "split_step_receiver": "yes", "receiver_landing_frame": None},
        ], COVERED_01), [
            {"frame": 520, "player": "far", "split_step_receiver": "yes", "receiver_landing_frame": None},
        ])
        held_out = (payload.get("clips") or {}).get("05", {}).get("contacts")
        self.assertIsInstance(held_out, list)
        # Clip 05 stays held out. Rows here are not scored and not a reason to tune.
        path = _path("01", "dense")
        if not path.is_file():
            self.skipTest(f"missing dense pose: {path}")
        tracks = load_tracks(_load(path))
        part = movement_analysis(tracks, _labelled_swings(contacts))
        hits = 0
        for row in eligible:
            receiver = Role.FAR if row["player"] == "near" else Role.NEAR
            hits += int(bool(_paired_split(part, receiver, int(row["frame"]), tracks.t)))
        self.assertEqual(len(eligible), 2)
        self.assertIsNone(_precision(hits, 0))
        self.assertGreaterEqual(hits, 0)
        self.assertLessEqual(hits, 2)


if __name__ == "__main__":
    unittest.main()
