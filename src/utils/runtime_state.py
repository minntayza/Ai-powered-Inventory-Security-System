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

    def snapshot(
        self, *, include_frame: bool = True, include_perception: bool = True
    ) -> Dict[str, Any]:
        """Return an isolated state copy, optionally excluding expensive payloads."""
        with self._lock:
            result = {}
            for key, value in self._state.items():
                if key == "frame":
                    if include_frame:
                        result[key] = None if value is None else value.copy()
                elif key == "perception":
                    if include_perception:
                        result[key] = deepcopy(value)
                else:
                    result[key] = deepcopy(value)
            return result

    def get(self, key: str, default: Any = None) -> Any:
        """Return one isolated value without copying unrelated video/state data."""
        with self._lock:
            value = self._state.get(key, default)
            if key == "frame" and value is not None:
                return value.copy()
            return deepcopy(value)

    def perception_snapshot(self) -> Any:
        """Copy perception metadata without also copying the stored video frame."""
        with self._lock:
            return deepcopy(self._state["perception"])
