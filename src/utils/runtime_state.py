"""Thread-safe state shared by the monitoring engine and dashboard."""

from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Any, Dict


class RuntimeState:
    def __init__(self) -> None:
        self._lock = RLock()
        self._state: Dict[str, Any] = {
            "running": False,
            "camera_healthy": False,
            "model_status": "not_started",
            "device_info": None,
            "frame": None,
            "perception": None,
            "security": {"state": "NORMAL"},
            "monitoring": {
                "mode": "DISARMED",
                "baseline_ready": False,
                "baseline_counts": {},
                "current_counts": {},
                "missing_items": {},
                "extra_items": {},
            },
            "source": {"type": "live", "label": "Live camera"},
            "performance": {},
            "last_error": None,
        }

    def update(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            result = dict(self._state)
            result["frame"] = (
                None if self._state["frame"] is None else self._state["frame"].copy()
            )
            for key in ("perception", "security", "monitoring", "source", "performance"):
                result[key] = deepcopy(self._state[key])
            return result
