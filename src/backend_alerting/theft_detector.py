"""Deterministic theft decision state machine."""

from __future__ import annotations

import time
import uuid
from copy import deepcopy
from typing import Dict, List, Optional

from .grace_buffer import GraceBuffer


class TheftDetector:
    STATES = {"NORMAL", "PENDING", "CANCELLED", "CONFIRMED", "COOLDOWN"}

    def __init__(
        self,
        grace_period_seconds: float = 3.0,
        cooldown_seconds: float = 30.0,
        shelf_region: Optional[List[float]] = None,
    ):
        self.state = "NORMAL"
        self.buffer = GraceBuffer(grace_period_seconds)
        self.cooldown_seconds = cooldown_seconds
        self.cooldown_until = 0.0
        self.pending: Optional[Dict] = None
        self.last_event: Optional[Dict] = None
        self.shelf_region = shelf_region

    def evaluate(self, perception: Dict, now: Optional[float] = None) -> Dict:
        now = time.time() if now is None else now
        persons = self._relevant_persons(perception)
        inventory = perception.get("inventory", {})
        change = inventory.get("change")
        drop = inventory.get("drop")

        if self.state == "COOLDOWN":
            if now >= self.cooldown_until:
                self.state = "NORMAL"
                self.last_event = None
            return self.snapshot()

        if self.state == "CANCELLED":
            self.state = "NORMAL"

        if self.state == "NORMAL" and drop:
            if self._all_persons_authorized(persons):
                event = self._make_event("authorized_removal", "cancelled", drop, persons, now)
                self.last_event = event
                self.state = "CANCELLED"
                return self.snapshot(new_event=event)
            self.pending = deepcopy(drop)
            self.buffer.start(now)
            self.buffer.observe(persons)
            self.state = "PENDING"
            return self.snapshot()

        if self.state == "PENDING":
            self.buffer.observe(persons)
            if self._inventory_recovered(change):
                event = self._make_event(
                    "inventory_recovered", "cancelled", self.pending or {}, persons, now
                )
                self._cancel(event)
                return self.snapshot(new_event=event)
            if self.buffer.all_tracks_authorized:
                event = self._make_event(
                    "authorized_removal", "cancelled", self.pending or {}, persons, now
                )
                self._cancel(event)
                return self.snapshot(new_event=event)
            if self.buffer.expired(now):
                if not self.buffer.track_ids:
                    event = self._make_event(
                        "unattributed_inventory_change", "cancelled",
                        self.pending or {}, persons, now
                    )
                    self._cancel(event)
                    return self.snapshot(new_event=event)
                event = self._make_event(
                    "suspected_theft", "confirmed", self.pending or {}, persons, now
                )
                self.last_event = event
                self.state = "CONFIRMED"
                self.pending = None
                self.buffer.reset()
                return self.snapshot(new_event=event)

        if self.state == "CONFIRMED":
            self.state = "COOLDOWN"
            self.cooldown_until = now + self.cooldown_seconds

        return self.snapshot()

    @staticmethod
    def _all_persons_authorized(persons: List[Dict]) -> bool:
        return bool(persons) and all(
            person.get("authorization_state") == "authorized" for person in persons
        )

    def _relevant_persons(self, perception: Dict) -> List[Dict]:
        persons = perception.get("persons", [])
        if not self.shelf_region:
            return persons
        width, height = perception.get("frame_size", [0, 0])
        if not width or not height or len(self.shelf_region) != 4:
            return persons
        rx1, ry1, rx2, ry2 = self.shelf_region
        region = (rx1 * width, ry1 * height, rx2 * width, ry2 * height)
        relevant = []
        for person in persons:
            x1, y1, x2, y2 = person.get("bbox", [0, 0, 0, 0])
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            if region[0] <= center_x <= region[2] and region[1] <= center_y <= region[3]:
                relevant.append(person)
        return relevant

    @staticmethod
    def _inventory_recovered(change: Optional[Dict]) -> bool:
        return bool(change and change.get("added_items"))

    def _cancel(self, event: Dict) -> None:
        self.last_event = event
        self.state = "CANCELLED"
        self.pending = None
        self.buffer.reset()

    def _make_event(
        self, event_type: str, status: str, change: Dict, persons: List[Dict], now: float
    ) -> Dict:
        track_states = dict(self.buffer.track_states)
        for person in persons:
            track_id = person.get("track_id", person.get("id"))
            if track_id is not None:
                track_states[int(track_id)] = person.get(
                    "authorization_state", "not_visible"
                )
        states = sorted(set(track_states.values()) | set(self.buffer.observed_states))
        track_ids = sorted(track_states)
        return {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "status": status,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            "removed_items": deepcopy(change.get("removed_items", {})),
            "previous_counts": deepcopy(change.get("previous_counts", {})),
            "current_counts": deepcopy(change.get("current_counts", {})),
            "person_track_ids": track_ids,
            "authorization_states": states,
            "person_states": {str(key): value for key, value in track_states.items()},
            "snapshot_path": None,
            "telegram_status": "pending" if status == "confirmed" else "disabled",
            "acknowledged": False,
        }

    def acknowledge(self) -> None:
        if self.last_event:
            self.last_event["acknowledged"] = True

    def snapshot(self, new_event: Optional[Dict] = None) -> Dict:
        return {
            "state": self.state,
            "pending": deepcopy(self.pending),
            "last_event": deepcopy(self.last_event),
            "new_event": deepcopy(new_event),
            "cooldown_until": self.cooldown_until or None,
        }
