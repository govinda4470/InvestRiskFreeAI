"""Application configuration loader (config.yaml + overrides)."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "config.yaml")


@lru_cache(maxsize=1)
def load_config(path: str = CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # absolute paths
    cfg.setdefault("data", {})["repo_root"] = REPO_ROOT
    for key in ("cache_dir", "bundled_dir"):
        if key in cfg["data"]:
            p = cfg["data"][key]
            if not os.path.isabs(p):
                cfg["data"][key] = os.path.join(REPO_ROOT, p)
    return cfg


def get(path: str, default: Any = None) -> Any:
    """Get a config value by dotted path, e.g. get('brain.min_confidence')."""
    cfg = load_config()
    node: Any = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
