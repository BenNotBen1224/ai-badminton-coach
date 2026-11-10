"""Every metric a finding can rest on, with its good range. Merged from both analysis units."""
from __future__ import annotations

from typing import Mapping, Sequence

from .catalog_body import BODY_SPECS
from .catalog_move import MOVE_SPECS
from .severity import MetricSpec


def build_catalog(*groups: Sequence[MetricSpec]) -> Mapping[str, MetricSpec]:
    catalog: dict[str, MetricSpec] = {}
    for group in groups:
        for spec in group:
            if spec.name in catalog:
                raise ValueError(f"metric {spec.name} is specified twice")
            catalog[spec.name] = spec
    return catalog


METRIC_CATALOG = build_catalog(BODY_SPECS, MOVE_SPECS)
