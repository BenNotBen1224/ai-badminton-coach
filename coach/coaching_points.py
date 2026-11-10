"""Rank coaching concepts only after their local visual evidence is established."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .play_state import exchange_mask
from .schema import Role
from .signals_body import elbow_deg, knee_angle_deg, stance_width_ratio
from .tracks import JOINT_INDEX, load_tracks

PRIORITIES = Path(__file__).parent / "coaching_corpus" / "coaching_priorities.json"
FINDING_CONCEPT = {
    "footwork.step_pattern": "footwork",
    "technique.knee_bend": "knee_bend",
    "footwork.recovery": "recovery",
    "contact.arm_extension": "arm_extension",
    "technique.lunge_depth": "lunge",
    "technique.stay_lower": "ready_position",
    "technique.stance_width": "stance_width",
}
LEGS = tuple(f"{side}_{joint}" for side in ("left", "right") for joint in ("hip", "knee", "ankle"))


def priorities():
    return json.loads(PRIORITIES.read_text("utf-8"))


def corpus_description():
    provenance = dict(priorities()["provenance"])
    counts = json.loads(PRIORITIES.with_name("provenance_counts.json").read_text("utf-8"))
    if counts["source"]["sha256"] != provenance["learned_sha256"]:
        raise ValueError("Corpus metadata counts and priorities refer to different learned inputs")
    provenance.pop("coaching_channels")
    return {**provenance, **counts}


def rank_candidates(candidates, ranking=None, count=3):
    ranking = priorities() if ranking is None else ranking
    selected = []
    concepts = {row["key"]: row for row in ranking["concepts"]}
    for key in ranking["visible_order"]:
        pool = [row for row in candidates if row["concept"] == key]
        if not pool:
            continue
        point = max(pool, key=lambda row: (row["confidence"], row.get("evidence_strength", 0)))
        selected.append({**point, "priority": concepts[key]})
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError(f"Only {len(selected)} supported coaching concepts; need {count}. No invented replacement")
    return selected


def _visible(player, frame, names):
    return bool(player.measured[frame] and all(player.joint_visible(name)[frame] for name in names))


def _point(tracks, player, frame, concept, finding_id, metric, value, unit, window, focus):
    if not np.isfinite(value) or not _visible(player, frame, focus):
        return None
    shot = next((a, b) for a, b in tracks.shots if a <= frame <= b)
    start, end = window
    if not shot[0] <= start < frame < end <= shot[1]:
        return None
    confidence = min(1.0, *(float(player.conf[frame, JOINT_INDEX[name]]) for name in focus))
    return {
        "id": concept, "concept": concept, "finding_id": finding_id, "player": player.role.value,
        "source_frame": frame, "source_time_s": float(tracks.t[frame]), "source_start_frame": start,
        "source_end_frame": end, "shot": list(shot), "metric": metric, "value": float(value),
        "unit": unit, "n": 1, "confidence": confidence, "focus_joints": list(focus),
        "bbox": player.bbox[frame].tolist(),
        "keypoints": {name: player.joint(name)[frame].tolist() for name in focus},
    }


def evidence_candidates(doc, report):
    tracks = load_tracks(doc)
    active = exchange_mask(tracks, report.events)
    candidates = []
    for event in report.events:
        if event.kind != "lunge" or not event.data.get("knee_stride"):
            continue
        player = tracks.players[event.player]
        side = "right" if event.data["lead_right"] else "left"
        bottom, plant, end = (int(event.data[key]) for key in ("bottom_frame", "plant_frame", "end_frame"))
        start = event.frame
        shot_start, shot_end = next((a, b) for a, b in tracks.shots if a <= bottom <= b)
        context_start = max(shot_start, start - round(0.5 * tracks.fps))
        following = sorted((swing for swing in report.events
                            if swing.kind == "swing" and swing.player == event.player
                            and end < swing.frame <= min(shot_end, end + round(1.6 * tracks.fps))),
                           key=lambda swing: swing.frame)
        context_end = min(shot_end, end + round(0.35 * tracks.fps))
        next_move_end = context_end
        if following:
            next_move_end = min(shot_end, int(following[0].data["end"]) + round(0.25 * tracks.fps))
        evidence = {
            "kind": "knee_stride_cycle", "start_frame": start, "plant_frame": plant,
            "bottom_frame": bottom, "return_frame": end, "lead_side": side,
            "knee_return_s": float(tracks.t[end] - tracks.t[bottom]),
            "next_move_end_frame": next_move_end,
            "criterion": "Unilateral sustained knee flexion >=75deg, separated ankles >=0.40 stature, travelling lead foot, observed return <=20deg.",
        }
        if player.hand != "unknown" and side == player.hand:
            point = _point(tracks, player, plant, "footwork", "footwork.step_pattern",
                           "plant_stride_ratio", stance_width_ratio(player)[plant], "heights",
                           (context_start, context_end), LEGS)
            if point:
                point.update({"cycle": evidence, "hand": player.hand, "evidence_strength": event.data["lead_knee_flexion_deg"]})
                candidates.append(point)
        point = _point(tracks, player, bottom, "knee_bend", "technique.knee_bend",
                       "lead_knee_flexion_deg", 180 - knee_angle_deg(player, side)[bottom], "deg",
                       (context_start, context_end), LEGS)
        if point:
            point.update({"cycle": evidence, "hand": player.hand, "evidence_strength": event.data["lead_knee_flexion_deg"]})
            if player.hand != "unknown":
                balance_side = "left" if player.hand == "right" else "right"
                names = tuple(f"{balance_side}_{joint}" for joint in ("shoulder", "elbow", "wrist"))
                if _visible(player, bottom, names):
                    angle = float(elbow_deg(player, f"{balance_side}_wrist")[bottom])
                    point["balance_arm"] = {
                        "concept": "non_racket_arm", "side": balance_side, "source_frame": bottom,
                        "elbow_deg": angle, "n": 1,
                        "keypoints": {name: player.joint(name)[bottom].tolist() for name in names},
                        "confidence": min(1.0, *(float(player.conf[bottom, JOINT_INDEX[name]]) for name in names)),
                    }
            candidates.append(point)
        if following:
            foot = player.foot_px
            visible_path = player.measured[end:next_move_end + 1] & np.isfinite(foot[end:next_move_end + 1]).all(axis=1)
            movement = foot[end:next_move_end + 1][visible_path]
            if len(movement) >= 3 and np.linalg.norm(movement[-1] - movement[0]) > player.stature_px[end] * 0.15:
                point = _point(tracks, player, end, "recovery", "footwork.recovery",
                               "knee_return_s", tracks.t[end] - tracks.t[bottom], "s",
                               (context_start, next_move_end), LEGS)
                if point:
                    point.update({"cycle": evidence, "hand": player.hand, "evidence_strength": event.data["lead_knee_flexion_deg"]})
                    candidates.append(point)
    # Lower-priority stationary concepts remain available, but cannot outrank supported movement.
    for finding in report.findings:
        concept = FINDING_CONCEPT.get(finding.id)
        if concept not in ("stance_width", "ready_position", "arm_extension"):
            continue
        for moment in finding.moments:
            frame = moment.key_frame
            player = tracks.players[finding.player]
            focus = finding.focus_joints
            metric = finding.metrics[0]
            if metric.value is None or not active[frame] or not moment.marks:
                continue
            actual = next((mark.value for mark in moment.marks if mark.value is not None and mark.style == "actual"), None)
            if actual is None:
                continue
            shot_start, shot_end = next((a, b) for a, b in tracks.shots if a <= frame <= b)
            point = _point(tracks, player, frame, concept, finding.id, metric.name, actual, metric.unit,
                           (max(shot_start, moment.start_frame), min(shot_end, moment.end_frame)), focus)
            if point:
                candidates.append(point)
    return candidates
