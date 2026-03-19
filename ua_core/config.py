import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONFIG_PATH = "ua_config.json"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    config_file = Path(config_path or DEFAULT_CONFIG_PATH)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_file}. "
            "Create it from ua_config.json defaults."
        )

    with config_file.open("r", encoding="utf-8") as f:
        user_cfg = json.load(f)

    # Keep a stable, complete schema even if user config is partial.
    with Path(DEFAULT_CONFIG_PATH).open("r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    return _deep_merge(base_cfg, user_cfg)
