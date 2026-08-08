"""Small in-memory rolling performance monitor."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import RLock
from typing import Deque, Dict


class PerformanceMonitor:
    def __init__(self, sample_count: int = 60) -> None:
        self.sample_count = max(2, int(sample_count))
        self._values: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.sample_count)
        )
        self._capture_times: Deque[float] = deque(maxlen=self.sample_count)
        self._processed_times: Deque[float] = deque(maxlen=self.sample_count)
        self._skipped_frames = 0
        self._lock = RLock()

    def record(self, name: str, value: float) -> None:
        with self._lock:
            self._values[name].append(float(value))

    def capture_tick(self, now: float | None = None) -> None:
        with self._lock:
            self._capture_times.append(time.perf_counter() if now is None else float(now))

    def processed_tick(self, now: float | None = None) -> None:
        with self._lock:
            self._processed_times.append(time.perf_counter() if now is None else float(now))

    def skip(self, count: int = 1) -> None:
        with self._lock:
            self._skipped_frames += max(0, int(count))

    @staticmethod
    def _fps(values: Deque[float]) -> float:
        if len(values) < 2:
            return 0.0
        duration = values[-1] - values[0]
        return round((len(values) - 1) / duration, 2) if duration > 0 else 0.0

    def snapshot(self) -> Dict:
        with self._lock:
            result = {
                "capture_fps": self._fps(self._capture_times),
                "processed_fps": self._fps(self._processed_times),
                "skipped_frames": self._skipped_frames,
            }
            for name, values in self._values.items():
                if values:
                    result[name] = round(values[-1], 2)
                    result[f"{name}_avg"] = round(sum(values) / len(values), 2)
                    result[f"{name}_max"] = round(max(values), 2)
            return result

    def reset(self) -> None:
        with self._lock:
            self._values.clear()
            self._capture_times.clear()
            self._processed_times.clear()
            self._skipped_frames = 0
