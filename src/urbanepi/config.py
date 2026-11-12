"""Configuration loading."""
from pathlib import Path

import yaml


def load_config(path: str | Path = "configs/base.yaml") -> dict:
    """Read a YAML configuration file."""
    with open(path, encoding="utf8") as handle:
        return yaml.safe_load(handle)
