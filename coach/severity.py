"""One rule for severity, so every finding means the same thing.

The analysis units do not pick severities by hand. They report a metric, and the
catalog says what range is good for it. severity_for turns the distance outside
that range into a severity. assemble rejects a finding whose severity or score
disagrees with this rule.

Distance d is how far the value sits outside the good range, in tolerance units.
  inside the range            STRENGTH when reliable, INFO otherwise
  0 < d <= INFO_MAX           INFO
  INFO_MAX < d <= IMPROVE_MAX IMPROVE
  d > IMPROVE_MAX             PRIORITY
A metric is reliable when its confidence is at least MIN_CONFIDENCE and it rests on
at least MIN_OBS observations. An unreliable metric never rises above INFO.
distance, severity_for and score_for stay numeric. check_findings does not call
them for a metric with no value. A finding whose catalog metrics are all missing
must be INFO, score 0, and have no moments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .schema import FINDING_CATALOG, SEVERITY_RANK, Basis, Finding, Severity, _snake

DIRECTIONS = ("lower_better", "higher_better", "in_range")
INFO_MAX = 1.0
IMPROVE_MAX = 2.0
MIN_CONFIDENCE = 0.6
MIN_OBS = 3
SCORE_TOLERANCE = 1e-3


@dataclass(frozen=True)
class MetricSpec:
    """What a metric means and where its good range is.

    target is (low, high). For lower_better only high matters (good when value <= high).
    For higher_better only low matters (good when value >= low). tolerance is the size of
    one distance unit, in the metric's own unit.
    """

    name: str
    label: str
    unit: str
    basis: Basis
    direction: str
    target: tuple[float, float]
    tolerance: float
    finding_id: str
    descriptive: bool = False

    def __post_init__(self) -> None:
        _snake(self.name, "metric spec name")
        if not (self.label and self.unit):
            raise ValueError(f"metric spec {self.name} needs a label and a unit")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"metric spec {self.name} direction must be one of {DIRECTIONS}")
        if len(self.target) != 2 or self.target[0] > self.target[1]:
            raise ValueError(f"metric spec {self.name} target must be (low, high)")
        if not self.tolerance > 0:
            raise ValueError(f"metric spec {self.name} tolerance must be > 0")
        if self.finding_id not in FINDING_CATALOG:
            raise ValueError(f"metric spec {self.name} points at unknown finding {self.finding_id!r}")


def distance(spec: MetricSpec, value: float) -> float:
    low, high = spec.target
    if spec.direction == "lower_better":
        outside = value - high
    elif spec.direction == "higher_better":
        outside = low - value
    else:
        outside = max(low - value, value - high)
    return max(0.0, outside) / spec.tolerance


def severity_for(spec: MetricSpec, value: float, confidence: float, n_obs: int) -> Severity:
    if spec.descriptive:
        return Severity.INFO
    d = distance(spec, value)
    reliable = confidence >= MIN_CONFIDENCE and n_obs >= MIN_OBS
    if d == 0.0:
        return Severity.STRENGTH if reliable else Severity.INFO
    if not reliable or d <= INFO_MAX:
        return Severity.INFO
    return Severity.IMPROVE if d <= IMPROVE_MAX else Severity.PRIORITY


def score_for(spec: MetricSpec, value: float, confidence: float) -> float:
    if spec.descriptive:
        return 0.0
    return distance(spec, value) * confidence


def check_findings(findings: Iterable[Finding], catalog: Mapping[str, MetricSpec]) -> None:
    """Raise unless each finding's severity and score follow from its catalog metrics.

    Observed metrics use the numeric rule. A finding whose catalog metrics all have
    no value must be info, score exactly 0, and have no moments.
    """
    for finding in findings:
        rated = [(catalog[m.name], m) for m in finding.metrics if m.name in catalog and catalog[m.name].finding_id == finding.id]
        if not rated:
            raise ValueError(f"finding {finding.id} has no metric that the catalog ties to it")
        observed = [(spec, metric) for spec, metric in rated if metric.value is not None]
        if not observed:
            if finding.severity != Severity.INFO:
                raise ValueError(f"finding {finding.id} is unavailable, so it must be info")
            if finding.score != 0:
                raise ValueError(f"finding {finding.id} is unavailable, so its score must be 0")
            if finding.moments:
                raise ValueError(f"finding {finding.id} is unavailable, so it must not have a moment")
            continue
        severity = max((severity_for(s, m.value, m.confidence, m.n_obs) for s, m in observed), key=SEVERITY_RANK.__getitem__)
        score = max(score_for(s, m.value, m.confidence) for s, m in observed)
        if finding.severity != severity:
            raise ValueError(f"finding {finding.id} says {finding.severity.value} but the catalog rule gives {severity.value}")
        if abs(finding.score - score) > SCORE_TOLERANCE:
            raise ValueError(f"finding {finding.id} score {finding.score:.4f} does not match the catalog rule {score:.4f}")
