"""Inventory counting and drop detection."""

import time
from typing import Dict, List


class InventoryCounter:
    """Track inventory count and detect drops."""

    def __init__(self):
        self.current_count = 0
        self.previous_count = 0
        self.history = []

    def update_count(self, detections: Dict) -> int:
        self.previous_count = self.current_count
        self.current_count = len(detections.get("inventory", []))

        self.history.append({
            "timestamp": time.time(),
            "count": self.current_count
        })

        if len(self.history) > 100:
            self.history = self.history[-100:]

        return self.current_count

    def detect_drop(self) -> bool:
        return self.current_count < self.previous_count

    def get_count(self) -> int:
        return self.current_count

    def get_history(self) -> List[Dict]:
        return self.history

    def reset(self):
        self.current_count = 0
        self.previous_count = 0
        self.history = []
