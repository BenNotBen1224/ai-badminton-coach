"""Imported only by a runner instance, not by the harness module."""
from pathlib import Path


class NestedModel:
    def __init__(self, model_path, config_path):
        self.model = Path(model_path)
        self.config = Path(config_path)
