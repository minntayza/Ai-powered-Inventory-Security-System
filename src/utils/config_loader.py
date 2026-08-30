"""Validated YAML configuration loading for the application."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Raised when application configuration is missing or malformed."""


def load_environment(path: str | Path | None = None) -> bool:
    """Load project-local environment values without overriding the process."""
    env_path = resolve_project_path(path or ".env")
    return load_dotenv(dotenv_path=env_path, override=False)


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML mapping and return an empty mapping for an empty file."""
    resolved = resolve_project_path(path)
    if not resolved.exists():
        raise ConfigError(f"Configuration file not found: {resolved}")
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration root must be a mapping: {resolved}")
    return value


def require_keys(config: Dict[str, Any], keys: Iterable[str], label: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ConfigError(f"Missing {label} setting(s): {', '.join(missing)}")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_app_config(config_dir: str | Path = "configs") -> Dict[str, Any]:
    """Load all application configuration files into one mapping."""
    load_environment()
    directory = resolve_project_path(config_dir)
    config = {
        "camera": load_yaml(directory / "camera_config.yaml"),
        "models": load_yaml(directory / "model_config.yaml"),
        "alerts": load_yaml(directory / "alert_config.yaml"),
        "thresholds": load_yaml(directory / "thresholds.yaml"),
    }
    require_keys(config["camera"], ("source", "width", "height", "fps"), "camera")
    require_keys(config["models"], ("yolo", "vlm"), "model")
    require_keys(config["alerts"], ("siren", "telegram", "storage"), "alert")
    _validate_inventory_policy(config["models"])
    return config


def _validate_inventory_policy(models: Dict[str, Any]) -> None:
    """Validate the security policy independently of a particular YOLO model."""
    policy = models.get("inventory_policy", {})
    protected = policy.get("protected", [])
    contextual = policy.get("contextual", [])
    if not isinstance(protected, list) or not isinstance(contextual, list):
        raise ConfigError("inventory_policy protected/contextual values must be lists")
    protected_set = {str(item) for item in protected}
    contextual_set = {str(item) for item in contextual}
    overlap = protected_set & contextual_set
    if overlap:
        raise ConfigError(
            "Inventory policy classes cannot overlap: " + ", ".join(sorted(overlap))
        )
    if "person" in protected_set | contextual_set:
        raise ConfigError("person is a security subject and cannot be inventory")
    targets = models.get("yolo", {}).get("target_classes")
    if targets is not None:
        unknown = (protected_set | contextual_set) - {str(item) for item in targets}
        if unknown:
            raise ConfigError(
                "Inventory policy classes missing from yolo.target_classes: "
                + ", ".join(sorted(unknown))
            )
