"""Time-based observation buffer used by the security state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class GraceBuffer:
    duration_seconds: float = 3.0
    started_at: float | None = None
    observed_states: Set[str] = field(default_factory=set)
    track_ids: Set[int] = field(default_factory=set)
    track_states: Dict[int, str] = field(default_factory=dict)

    def start(self, now: float) -> None:
        self.started_at = now
        self.observed_states.clear()
        self.track_ids.clear()
        self.track_states.clear()

    def observe(self, persons: List[dict]) -> None:
        for person in persons:
            self.observed_states.add(person.get("authorization_state", "not_visible"))
            track_id = person.get("track_id", person.get("id"))
            if track_id is not None:
                normalized_id = int(track_id)
                self.track_ids.add(normalized_id)
                self.track_states[normalized_id] = person.get(
                    "authorization_state", "not_visible"
                )

    def expired(self, now: float) -> bool:
        return self.started_at is not None and now - self.started_at >= self.duration_seconds

    @property
    def authorized_seen(self) -> bool:
        return "authorized" in self.observed_states

    @property
    def all_tracks_authorized(self) -> bool:
        return bool(self.track_states) and all(
            state == "authorized" for state in self.track_states.values()
        )

    def reset(self) -> None:
        self.started_at = None
        self.observed_states.clear()
        self.track_ids.clear()
        self.track_states.clear()
