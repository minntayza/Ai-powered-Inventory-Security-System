from unittest import TestCase

import numpy as np

from src.module_a_perception_engine.inventory_counter import InventoryCounter
from src.module_a_perception_engine.person_tracker import PersonTracker


class FakeYolo:
    def detect(self, _frame):
        return {
            "persons": [],
            "inventory": [
                {"label": "chair", "bbox": [10, 10, 30, 30], "confidence": 0.9},
                {"label": "bottle", "bbox": [70, 70, 90, 90], "confidence": 0.9},
            ],
        }


class FakeFace:
    def detect_faces(self, _frame):
        return []


class PersonTrackerPolicyTests(TestCase):
    def test_context_is_visible_but_outside_protected_item_is_not_counted(self):
        tracker = PersonTracker(
            yolo_detector=FakeYolo(),
            face_detector=FakeFace(),
            inventory_counter=InventoryCounter(
                window_size=1, warmup_frames=1, confirmation_frames=1,
                require_manual_baseline=True,
            ),
            contextual_counter=InventoryCounter(
                window_size=1, warmup_frames=1, confirmation_frames=1
            ),
            protected_classes={"bottle"},
            contextual_classes={"chair"},
            shelf_region=[0, 0, 0.5, 0.5],
        )
        result = tracker.process_frame(np.zeros((100, 100, 3), dtype=np.uint8))
        tracker.shutdown()
        self.assertEqual(result["inventory"]["protected_counts"], {})
        self.assertEqual(result["inventory"]["contextual_counts"], {"chair": 1})
        bottle = next(item for item in result["inventory"]["items"] if item["label"] == "bottle")
        self.assertFalse(bottle["in_zone"])

    def test_one_face_cannot_authorize_multiple_overlapping_people(self):
        tracker = PersonTracker(
            yolo_detector=FakeYolo(),
            face_detector=FakeFace(),
            inventory_counter=InventoryCounter(window_size=1, warmup_frames=1, confirmation_frames=1),
            contextual_counter=InventoryCounter(window_size=1, warmup_frames=1, confirmation_frames=1),
        )
        people = [
            {"track_id": 1, "bbox": [0, 0, 80, 100], "confidence": 0.9},
            {"track_id": 2, "bbox": [20, 0, 100, 100], "confidence": 0.9},
        ]
        faces = [{
            "bbox": [40, 20, 20, 20],
            "name": "Alice",
            "authorized": True,
            "authorization_state": "authorized",
            "confidence": 0.95,
        }]
        try:
            correlated = tracker._correlate_persons_faces(people, faces)
        finally:
            tracker.shutdown()

        self.assertEqual(
            sum(person["authorization_state"] == "authorized" for person in correlated),
            1,
        )
