from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_env_file(path: Path) -> bool:
    """Load simple KEY=VALUE pairs without executing the file or overriding the shell."""
    if not path.exists():
        return False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key and not os.environ.get(key):
            os.environ[key] = value
    return True


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = _expand_env(yaml.safe_load(handle) or {})
    base = path.resolve().parent
    for key in ("database", "output_dir"):
        value = Path(config["radar"][key])
        if not value.is_absolute():
            config["radar"][key] = str(base / value)
    if "audio" in config and config["audio"].get("cache_dir"):
        value = Path(config["audio"]["cache_dir"])
        if not value.is_absolute():
            config["audio"]["cache_dir"] = str(base / value)
    if "deep_reading" in config and config["deep_reading"].get("cache_dir"):
        value = Path(config["deep_reading"]["cache_dir"])
        if not value.is_absolute():
            config["deep_reading"]["cache_dir"] = str(base / value)
    return config


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value
