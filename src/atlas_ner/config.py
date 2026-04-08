"""Configuration helpers for AtlasNER."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {path}, got {type(data)!r}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(
    base_path: str | Path,
    experiment_path: str | Path | None = None,
    extra_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    config = load_yaml(base_path)
    if experiment_path is not None:
        config = deep_merge(config, load_yaml(experiment_path))
    for path in extra_paths or []:
        config = deep_merge(config, load_yaml(path))
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=False, sort_keys=False)
