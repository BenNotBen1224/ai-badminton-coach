"""Resident runner contract. The harness times run(); returned timings are not a gate.

S-A2 run() must persist work_dir/pose.json, the report, and the bundle dict
before it returns, and return pose_path with the other paths. pose.json is the
raw document, equal to the returned doc. The bundle must also decode back to
that document. Declare binding_manifest() with
real source and asset paths; the harness hashes those and the files the warmed
object actually holds.
A file that appears or changes during run() is a mixed input and is refused.
"""
from __future__ import annotations

import importlib
from pathlib import Path


class Runner:
    IS_REPLAY = False

    def __init__(self, models_dir):
        self.models_dir = Path(models_dir)

    def warm_up(self, video_path):
        raise NotImplementedError

    def run(self, video_path, work_dir):
        raise NotImplementedError


def load_class(spec):
    """spec is 'label=package.module:Class' or 'package.module:Class'."""
    label, sep, target = spec.partition("=")
    if not sep:
        target = label
        label = target.rsplit(":", 1)[-1]
    module_name, class_sep, class_name = target.partition(":")
    if not class_sep or not module_name or not class_name:
        raise ValueError(f"runner spec needs [label=]package.module:Class, got {spec}")
    cls = getattr(importlib.import_module(module_name), class_name)
    return label, cls
