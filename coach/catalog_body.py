"""Metric specs for the technique and contact findings.

Targets are practice guides for broadcast video of elite players, not validated
personal norms. The parent calibrates them. Each comment is the reason for the range.
"""
from __future__ import annotations

from dataclasses import replace

from .schema import Basis
from .severity import MetricSpec

BODY_SPECS: tuple[MetricSpec, ...] = (
    # A fully straight projected hip sits just above half of height; a ready wait sits lower.
    MetricSpec("ready_hip_ratio", "Projected hip height", "heights", Basis.PROJECTED_2D,
               "lower_better", (0.40, 0.50), 0.04, "technique.stay_lower"),
    # A soft ready bend is about 20 to 50 degrees. Straighter knees are a tall wait.
    MetricSpec("ready_knee_flexion", "Ready knee bend", "deg", Basis.PROJECTED_2D,
               "higher_better", (20.0, 50.0), 8.0, "technique.knee_bend"),
    # A useful lunge drops projected hip height by about 15 to 45 percent of the standing hip.
    MetricSpec("lunge_depth", "Lunge hip drop", "fraction", Basis.PROJECTED_2D,
               "higher_better", (0.15, 0.45), 0.05, "technique.lunge_depth"),
    # Shoulder-width feet. The camera stretches width, so the tolerance is wide.
    MetricSpec("ready_stance_width", "Ready foot width", "heights", Basis.PROJECTED_2D,
               "in_range", (0.25, 0.80), 0.15, "technique.stance_width"),
    # Sideways lean past about 12 degrees is a tilted wait. This angle is foreshortened.
    MetricSpec("ready_trunk_lean", "Sideways trunk lean", "deg", Basis.PROJECTED_2D,
               "lower_better", (0.0, 12.0), 6.0, "technique.trunk_lean"),
    # Overhead contact around standing reach or higher. Only overhead shots count.
    MetricSpec("contact_reach", "Overhead reach", "heights", Basis.PROJECTED_2D,
               "higher_better", (1.00, 1.40), 0.08, "contact.reach_height"),
    # A long arm is nearly straight. The elbow angle is a projected estimate.
    MetricSpec("contact_elbow", "Hitting elbow", "deg", Basis.PROJECTED_2D,
               "higher_better", (155.0, 180.0), 12.0, "contact.arm_extension"),
    # A clear swing moves the racket hand several heights per second.
    MetricSpec("swing_hand_speed", "Racket-hand speed", "heights/s", Basis.PROJECTED_2D,
               "higher_better", (4.0, 20.0), 1.5, "contact.hand_speed"),
)

BODY_SPECS = tuple(replace(spec, descriptive=True) for spec in BODY_SPECS)
