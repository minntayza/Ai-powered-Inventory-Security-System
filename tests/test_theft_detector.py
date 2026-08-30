from unittest import TestCase

from src.backend_alerting.theft_detector import TheftDetector


DROP = {
    "removed_items": {"bottle": 1},
    "previous_counts": {"bottle": 2},
    "current_counts": {"bottle": 1},
}


def perception(persons=None, drop=None, change=None):
    return {
        "persons": persons or [],
        "inventory": {"drop": drop, "change": change},
    }


class TheftDetectorTests(TestCase):
    def setUp(self):
        self.detector = TheftDetector(grace_period_seconds=3, cooldown_seconds=30)
        self.unknown = [{"track_id": 4, "authorization_state": "unknown"}]
        self.authorized = [{"track_id": 2, "authorization_state": "authorized"}]

    def test_unknown_removal_confirms_after_grace_period(self):
        first = self.detector.evaluate(perception(self.unknown, DROP, DROP), now=0)
        self.assertEqual(first["state"], "PENDING")
        result = self.detector.evaluate(perception(self.unknown), now=3.1)
        self.assertEqual(result["state"], "CONFIRMED")
        self.assertEqual(result["new_event"]["event_type"], "suspected_theft")

    def test_authorized_removal_is_cancelled_immediately(self):
        result = self.detector.evaluate(perception(self.authorized, DROP, DROP), now=0)
        self.assertEqual(result["state"], "CANCELLED")
        self.assertEqual(result["new_event"]["event_type"], "authorized_removal")

    def test_late_authorization_cancels_pending_event(self):
        self.detector.evaluate(perception(self.unknown, DROP, DROP), now=0)
        same_person_authorized = [{"track_id": 4, "authorization_state": "authorized"}]
        result = self.detector.evaluate(perception(same_person_authorized), now=1)
        self.assertEqual(result["state"], "CANCELLED")

    def test_authorized_bystander_does_not_clear_unknown_person(self):
        people = self.authorized + self.unknown
        result = self.detector.evaluate(perception(people, DROP, DROP), now=0)
        self.assertEqual(result["state"], "PENDING")
        result = self.detector.evaluate(perception(people), now=3.1)
        self.assertEqual(result["state"], "CONFIRMED")

    def test_shelf_region_ignores_person_outside_region(self):
        detector = TheftDetector(
            grace_period_seconds=0,
            shelf_region=[0.5, 0.0, 1.0, 1.0],
        )
        outside = [{
            "track_id": 4,
            "authorization_state": "unknown",
            "bbox": [0, 0, 20, 100],
        }]
        scene = perception(outside, DROP, DROP)
        scene["frame_size"] = [100, 100]
        detector.evaluate(scene, now=0)
        followup = perception(outside)
        followup["frame_size"] = [100, 100]
        result = detector.evaluate(followup, now=0.1)
        self.assertEqual(result["new_event"]["event_type"], "unattributed_inventory_change")

    def test_inventory_recovery_cancels_pending_event(self):
        self.detector.evaluate(perception(self.unknown, DROP, DROP), now=0)
        recovery = {"added_items": {"bottle": 1}}
        result = self.detector.evaluate(perception(self.unknown, change=recovery), now=1)
        self.assertEqual(result["new_event"]["event_type"], "inventory_recovered")

    def test_adding_a_different_item_does_not_cancel_pending_theft(self):
        self.detector.evaluate(perception(self.unknown, DROP, DROP), now=0)
        unrelated_addition = {
            "added_items": {"laptop": 1},
            "current_counts": {"bottle": 1, "laptop": 1},
        }

        pending = self.detector.evaluate(
            perception(self.unknown, change=unrelated_addition), now=1
        )
        confirmed = self.detector.evaluate(perception(self.unknown), now=3.1)

        self.assertEqual(pending["state"], "PENDING")
        self.assertEqual(confirmed["new_event"]["event_type"], "suspected_theft")

    def test_drop_without_person_is_not_confirmed_as_theft(self):
        self.detector.evaluate(perception(drop=DROP, change=DROP), now=0)
        result = self.detector.evaluate(perception(), now=3.1)
        self.assertEqual(result["state"], "CANCELLED")
        self.assertEqual(result["new_event"]["event_type"], "unattributed_inventory_change")

    def test_retained_drop_association_attributes_person_who_left_frame(self):
        associated_drop = {
            **DROP,
            "associations": [{
                "item_track_id": 9,
                "label": "bottle",
                "primary_actor": {
                    "track_id": 4,
                    "name": "Unknown",
                    "authorization_state": "unknown",
                    "association_score": 0.8,
                },
                "actor_candidates": [{
                    "track_id": 4,
                    "name": "Unknown",
                    "authorization_state": "unknown",
                    "association_score": 0.8,
                }],
            }],
        }
        first = self.detector.evaluate(
            perception(drop=associated_drop, change=associated_drop), now=0
        )
        self.assertEqual(first["state"], "PENDING")

        result = self.detector.evaluate(perception(), now=3.1)

        self.assertEqual(result["state"], "CONFIRMED")
        event = result["new_event"]
        self.assertEqual(event["event_type"], "suspected_theft")
        self.assertEqual(event["person_track_ids"], [4])
        self.assertEqual(event["primary_actor"]["track_id"], 4)

    def test_current_identity_refreshes_retained_primary_actor(self):
        associated_drop = {
            **DROP,
            "associations": [{
                "primary_actor": {
                    "track_id": 2,
                    "name": "Unknown",
                    "authorization_state": "not_visible",
                    "association_score": 0.9,
                },
                "actor_candidates": [],
            }],
        }
        current = [{
            "track_id": 2,
            "name": "Alice",
            "authorization_state": "authorized",
        }]

        result = self.detector.evaluate(
            perception(current, associated_drop, associated_drop), now=0
        )

        event = result["new_event"]
        self.assertEqual(event["event_type"], "authorized_removal")
        self.assertEqual(event["primary_actor"]["name"], "Alice")
        self.assertEqual(event["primary_actor"]["authorization_state"], "authorized")

    def test_confirmed_event_enters_cooldown(self):
        self.detector.evaluate(perception(self.unknown, DROP, DROP), now=0)
        self.detector.evaluate(perception(self.unknown), now=3.1)
        result = self.detector.evaluate(perception(self.unknown), now=3.2)
        self.assertEqual(result["state"], "COOLDOWN")
