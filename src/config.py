"""Load yaml configs and resolve the contest data path."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path, apply_local: bool = True) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    include = cfg.pop("include", None)
    if include:
        parent = load_config(include, apply_local=False)
        _deep_update(parent, cfg)
        cfg = parent
    if apply_local:
        local_path = ROOT / "configs" / "local.yaml"
        if local_path.is_file():
            with local_path.open(encoding="utf-8") as f:
                extra = yaml.safe_load(f) or {}
            extra.pop("include", None)
            _deep_update(cfg, extra)
    return cfg


def _deep_update(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def resolve_data_path(cfg: dict[str, Any], cli_path: str | None = None) -> Path:
    raw = cli_path or os.environ.get("JINGGE_DATA") or (cfg.get("data") or {}).get("path")
    if not raw:
        raise FileNotFoundError(
            "未指定数据路径。使用 --data、环境变量 JINGGE_DATA，或在 configs/default.yaml 填写 data.path"
        )
    path = Path(raw)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


# Older scripts import these names.
load_config = load_config
resolve_data_path = resolve_data_path
