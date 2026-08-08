"""Central application logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

from .config_loader import resolve_project_path


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    log_dir = resolve_project_path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("inventory_security")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(Path(log_dir) / "application.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger

