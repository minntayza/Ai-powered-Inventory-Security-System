from unittest import TestCase

from src.module_a_perception_engine.interaction_associator import InteractionAssociator


class InteractionAssociatorTests(TestCase):
    def test_disappearing_item_is_associated_with_nearby_person(self):
        associator = InteractionAssociator(lookback_seconds=2, minimum_score=0.35)
        item = {"label": "laptop", "bbox": [40, 40, 60, 60], "track_id": 9}
        person = {
            "bbox": [20, 10, 80, 100], "track_id": 3, "name": "Unknown",
            "authorization_state": "unknown",
        }
        associator.observe([item], [person], [100, 100], now=1)
        result = associator.observe(
            [], [person], [100, 100],
            drop={"removed_items": {"laptop": 1}}, now=2,
        )
        self.assertEqual(result[0]["primary_actor"]["track_id"], 3)

    def test_close_candidates_are_marked_ambiguous(self):
        associator = InteractionAssociator(ambiguity_margin=0.1)
        item = {"label": "book", "bbox": [45, 45, 55, 55], "track_id": 2}
        people = [
            {"bbox": [20, 10, 60, 90], "track_id": 1, "authorization_state": "unknown"},
            {"bbox": [40, 10, 80, 90], "track_id": 2, "authorization_state": "unknown"},
        ]
        associator.observe([item], people, [100, 100], now=1)
        result = associator.observe(
            [], people, [100, 100], drop={"removed_items": {"book": 1}}, now=2
        )
        self.assertTrue(result[0]["ambiguous"])
        self.assertIsNone(result[0]["primary_actor"])
