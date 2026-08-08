"""Temporal inventory counting and persistent drop detection."""

from __future__ import annotations

import time
from collections import Counter, deque
from copy import deepcopy
from typing import Deque, Dict, List, Mapping, Optional, Tuple


CountSignature = Tuple[Tuple[str, int], ...]


class InventoryCounter:
    """Convert noisy frame detections into stable per-class inventory counts."""

    def __init__(
        self,
        window_size: int = 15,
        confirmation_frames: int = 5,
        warmup_frames: int = 15,
    ) -> None:
        if min(window_size, confirmation_frames, warmup_frames) < 1:
            raise ValueError("Inventory counter sizes must be positive")
        if warmup_frames > window_size:
            raise ValueError("warmup_frames cannot exceed window_size")
        self.window_size = window_size
        self.confirmation_frames = confirmation_frames
        self.warmup_frames = warmup_frames
        self._window: Deque[CountSignature] = deque(maxlen=window_size)
        self._stable_counts: Dict[str, int] = {}
        self._raw_counts: Dict[str, int] = {}
        self._candidate: Optional[CountSignature] = None
        self._candidate_frames = 0
        self._initialized = False
        self._last_change: Optional[Dict] = None
        self.history: List[Dict] = []

    @staticmethod
    def _signature(counts: Mapping[str, int]) -> CountSignature:
        return tuple(sorted((name, int(count)) for name, count in counts.items() if count))

    @staticmethod
    def _counts(signature: CountSignature) -> Dict[str, int]:
        return dict(signature)

    def update(self, detections: Dict, timestamp: Optional[float] = None) -> Dict:
        timestamp = time.time() if timestamp is None else timestamp
        labels = [item.get("label", "unknown") for item in detections.get("inventory", [])]
        self._raw_counts = dict(Counter(labels))
        signature = self._signature(self._raw_counts)
        self._window.append(signature)
        mode = Counter(self._window).most_common(1)[0][0]
        self._last_change = None

        if not self._initialized:
            if len(self._window) >= self.warmup_frames:
                self._stable_counts = self._counts(mode)
                self._initialized = True
        elif mode == self._signature(self._stable_counts):
            self._candidate = None
            self._candidate_frames = 0
        elif mode == self._candidate:
            self._candidate_frames += 1
            if self._candidate_frames >= self.confirmation_frames:
                previous = deepcopy(self._stable_counts)
                current = self._counts(mode)
                labels_union = set(previous) | set(current)
                removed = {
                    label: previous.get(label, 0) - current.get(label, 0)
                    for label in labels_union
                    if previous.get(label, 0) > current.get(label, 0)
                }
                added = {
                    label: current.get(label, 0) - previous.get(label, 0)
                    for label in labels_union
                    if current.get(label, 0) > previous.get(label, 0)
                }
                self._stable_counts = current
                self._last_change = {
                    "timestamp": timestamp,
                    "previous_counts": previous,
                    "current_counts": deepcopy(current),
                    "removed_items": removed,
                    "added_items": added,
                }
                self._candidate = None
                self._candidate_frames = 0
        else:
            self._candidate = mode
            self._candidate_frames = 1

        snapshot = self.snapshot()
        self.history.append({"timestamp": timestamp, **snapshot})
        if len(self.history) > 500:
            self.history = self.history[-500:]
        return snapshot

    def snapshot(self) -> Dict:
        return {
            "raw_counts": deepcopy(self._raw_counts),
            "stable_counts": deepcopy(self._stable_counts),
            "total_raw_count": sum(self._raw_counts.values()),
            "total_stable_count": sum(self._stable_counts.values()),
            "initialized": self._initialized,
            "change": deepcopy(self._last_change),
            "drop": (
                deepcopy(self._last_change)
                if self._last_change and self._last_change["removed_items"]
                else None
            ),
        }

    def update_count(self, detections: Dict) -> int:
        """Backward-compatible total stable count update."""
        return self.update(detections)["total_stable_count"]

    def detect_drop(self) -> bool:
        return self.snapshot()["drop"] is not None

    def get_count(self) -> int:
        return sum(self._stable_counts.values())

    def get_history(self) -> List[Dict]:
        return deepcopy(self.history)

    def reset(self) -> None:
        self._window.clear()
        self._stable_counts = {}
        self._raw_counts = {}
        self._candidate = None
        self._candidate_frames = 0
        self._initialized = False
        self._last_change = None
        self.history = []
