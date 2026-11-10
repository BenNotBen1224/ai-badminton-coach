"""Metric specs for the footwork and tactics findings. Owned by the footwork and tactics unit."""
from __future__ import annotations

from .schema import Basis
from .severity import MetricSpec

# Targets are coaching heuristics, not measured norms. The pose clips are elite
# players, so none of these ranges is validated against club players.
# Split-step timing, a late first step, court coverage, base distance, rally
# gaps and net share are observations. Their clocks use an arm swing, not a
# confirmed hit. The analysis keeps their confidence under 0.6, so they cannot
# become an improve or priority finding. The numbers below are practice guides,
# not validated personal norms.
MOVE_SPECS: tuple[MetricSpec, ...] = (
    # A missing hop is not a fault: some shots need no obvious hop.
    MetricSpec("split_step_share", "Share of marked swings with a small hop nearby", "share", Basis.COUNT, "higher_better", (0.0, 1.0), 1.0, "footwork.split_step"),
    # Not a timing to aim for. Onset and landing are reported as observations.
    MetricSpec("split_step_timing_s", "Observed hop landing relative to a marked arm swing", "s", Basis.TIMING, "in_range", (-1.0, 0.40), 0.50, "footwork.split_step"),
    # Getting back to the spot you wait from, not to a fixed centre.
    MetricSpec("recovery_s", "Time to get back to the waiting spot", "s", Basis.TIMING, "lower_better", (0.0, 0.90), 0.40, "footwork.recovery"),
    # The clock is an arm swing, not a confirmed hit, so this stays informational.
    MetricSpec("first_step_s", "Delay before the first step", "s", Basis.TIMING, "lower_better", (0.0, 0.35), 0.20, "footwork.first_step"),
    # Extra small steps before a lunge are a late push, visible without the shuttle.
    MetricSpec("steps_before_lunge", "Small steps before a lunge", "count", Basis.COUNT, "lower_better", (0.0, 4.0), 2.0, "footwork.step_pattern"),
    # Distance from the player's own waiting spot, not from the court centre.
    MetricSpec("lunge_launch_m", "Distance from the waiting spot at the lunge", "m", Basis.COURT_PLANE, "lower_better", (0.0, 1.30), 0.55, "footwork.step_pattern"),
    # How much of the singles half was used. An empty corner is not a fault by itself.
    MetricSpec("half_coverage", "Share of the own half used", "share", Basis.COURT_PLANE, "higher_better", (0.0, 1.0), 1.0, "footwork.coverage"),
    # Which side was quieter. Not a tactics score.
    MetricSpec("weaker_side_share", "Share of time on the quieter side", "share", Basis.COUNT, "higher_better", (0.0, 1.0), 1.0, "footwork.coverage"),
    # Where they stood versus where they usually wait. Not a fault without the shuttle.
    MetricSpec("habitual_base_gap_m", "Distance from the waiting spot at their arm swing", "m", Basis.COURT_PLANE, "lower_better", (0.0, 6.0), 2.0, "tactics.base_discipline"),
    # Centre spot 3.0 m behind the net, reference only.
    MetricSpec("fixed_base_gap_m", "Distance from the centre spot at their arm swing", "m", Basis.COURT_PLANE, "lower_better", (0.0, 6.0), 2.0, "tactics.base_discipline"),
    # Time between marked arm swings. A gap over 2.5 s is a new rally, not a slow shot.
    MetricSpec("exchange_gap_s", "Time between marked arm swings inside a rally", "s", Basis.TIMING, "in_range", (0.0, 2.5), 1.0, "tactics.tempo"),
    # Nearer the net is a position, not an attack. The shuttle is not tracked.
    MetricSpec("net_ahead_share", "Share of their arm swings with you nearer the net", "share", Basis.COUNT, "higher_better", (0.0, 1.0), 1.0, "tactics.initiative"),
)
