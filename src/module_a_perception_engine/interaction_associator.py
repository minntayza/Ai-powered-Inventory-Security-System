"""Deterministic item-to-person proximity evidence for inventory changes."""

from __future__ import annotations

import math
import time
from copy import deepcopy
from typing import Dict, List, Optional


class InteractionAssociator:
    """Retain short item histories and rank people near disappearing items."""

    def __init__(
        self,
        lookback_seconds: float = 2.0,
        person_bbox_expansion: float = 0.15,
        minimum_score: float = 0.35,
        ambiguity_margin: float = 0.10,
    ) -> None:
        self.lookback_seconds = float(lookback_seconds)
        self.person_bbox_expansion = float(person_bbox_expansion)
        self.minimum_score = float(minimum_score)
        self.ambiguity_margin = float(ambiguity_margin)
        self._tracks: Dict[int, Dict] = {}
        self._previous_items: List[Dict] = []
        self._next_fallback_id = -1

    def observe(
        self,
        items: List[Dict],
        persons: List[Dict],
        frame_size: List[int],
        drop: Optional[Dict] = None,
        now: Optional[float] = None,
    ) -> List[Dict]:
        now = time.time() if now is None else float(now)
        self._assign_fallback_ids(items)
        current_ids = set()
        for item in items:
            track_id = int(item["track_id"])
            current_ids.add(track_id)
            track = self._tracks.setdefault(
                track_id,
                {"label": item.get("label", "unknown"), "candidates": {}},
            )
            track.update({"bbox": list(item.get("bbox", [])), "last_seen": now})
            for person in persons:
                score = self._score(item.get("bbox", []), person.get("bbox", []), frame_size)
                if score <= 0:
                    continue
                person_id = person.get("track_id", person.get("id"))
                if person_id is None:
                    continue
                candidate = {
                    "track_id": int(person_id),
                    "name": person.get("name", "Unknown"),
                    "authorization_state": person.get("authorization_state", "not_visible"),
                    "association_score": round(score, 3),
                }
                previous = track["candidates"].get(int(person_id))
                if previous is None or score > previous["association_score"]:
                    track["candidates"][int(person_id)] = candidate

        associations = self._associations_for_drop(drop, current_ids, now)
        self._previous_items = [deepcopy(item) for item in items]
        self._tracks = {
            track_id: value
            for track_id, value in self._tracks.items()
            if now - value.get("last_seen", now) <= self.lookback_seconds
        }
        return associations

    def _assign_fallback_ids(self, items: List[Dict]) -> None:
        claimed = set()
        for item in items:
            if item.get("track_id") is not None:
                claimed.add(int(item["track_id"]))
                continue
            best_id, best_iou = None, 0.0
            for previous in self._previous_items:
                previous_id = previous.get("track_id")
                if previous_id is None or int(previous_id) in claimed:
                    continue
                if previous.get("label") != item.get("label"):
                    continue
                overlap = self._iou(previous.get("bbox", []), item.get("bbox", []))
                if overlap > best_iou:
                    best_id, best_iou = int(previous_id), overlap
            if best_id is None or best_iou < 0.3:
                best_id = self._next_fallback_id
                self._next_fallback_id -= 1
            item["track_id"] = best_id
            claimed.add(best_id)

    def _associations_for_drop(
        self, drop: Optional[Dict], current_ids: set[int], now: float
    ) -> List[Dict]:
        if not drop:
            return []
        associations = []
        for label, quantity in drop.get("removed_items", {}).items():
            vanished = [
                (track_id, value)
                for track_id, value in self._tracks.items()
                if value.get("label") == label
                and track_id not in current_ids
                and now - value.get("last_seen", now) <= self.lookback_seconds
            ]
            vanished.sort(key=lambda pair: pair[1].get("last_seen", 0), reverse=True)
            for track_id, track in vanished[: int(quantity)]:
                candidates = sorted(
                    track.get("candidates", {}).values(),
                    key=lambda value: value["association_score"],
                    reverse=True,
                )
                qualified = [
                    candidate for candidate in candidates
                    if candidate["association_score"] >= self.minimum_score
                ]
                ambiguous = (
                    len(qualified) > 1
                    and qualified[0]["association_score"] - qualified[1]["association_score"]
                    < self.ambiguity_margin
                )
                associations.append(
                    {
                        "item_track_id": track_id,
                        "label": label,
                        "primary_actor": qualified[0] if qualified and not ambiguous else None,
                        "actor_candidates": qualified,
                        "ambiguous": ambiguous,
                    }
                )
        return associations

    def _score(self, item_bbox: List[int], person_bbox: List[int], frame_size: List[int]) -> float:
        if len(item_bbox) != 4 or len(person_bbox) != 4 or len(frame_size) != 2:
            return 0.0
        ix1, iy1, ix2, iy2 = map(float, item_bbox)
        px1, py1, px2, py2 = map(float, person_bbox)
        expand_x = (px2 - px1) * self.person_bbox_expansion
        expand_y = (py2 - py1) * self.person_bbox_expansion
        expanded = [px1 - expand_x, py1 - expand_y, px2 + expand_x, py2 + expand_y]
        item_center = ((ix1 + ix2) / 2, (iy1 + iy2) / 2)
        inside = expanded[0] <= item_center[0] <= expanded[2] and expanded[1] <= item_center[1] <= expanded[3]
        overlap = self._intersection_over_item(item_bbox, expanded)
        if inside:
            return min(1.0, 0.7 + 0.3 * overlap)
        person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
        distance = math.dist(item_center, person_center)
        diagonal = max(1.0, math.hypot(float(frame_size[0]), float(frame_size[1])))
        return max(0.0, 0.5 * (1.0 - distance / (0.25 * diagonal)))

    @staticmethod
    def _intersection_over_item(first: List[float], second: List[float]) -> float:
        if len(first) != 4 or len(second) != 4:
            return 0.0
        x1, y1 = max(first[0], second[0]), max(first[1], second[1])
        x2, y2 = min(first[2], second[2]), min(first[3], second[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area = max(1.0, (first[2] - first[0]) * (first[3] - first[1]))
        return intersection / area

    @staticmethod
    def _iou(first: List[int], second: List[int]) -> float:
        if len(first) != 4 or len(second) != 4:
            return 0.0
        x1, y1 = max(first[0], second[0]), max(first[1], second[1])
        x2, y2 = min(first[2], second[2]), min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    def reset(self) -> None:
        self._tracks.clear()
        self._previous_items.clear()
        self._next_fallback_id = -1
