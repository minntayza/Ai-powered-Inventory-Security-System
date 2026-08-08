from unittest import TestCase

from src.module_a_perception_engine.inventory_counter import InventoryCounter


def detections(**counts):
    items = []
    for label, count in counts.items():
        items.extend({"label": label} for _ in range(count))
    return {"inventory": items}


class InventoryCounterTests(TestCase):
    def setUp(self):
        self.counter = InventoryCounter(window_size=3, warmup_frames=3, confirmation_frames=2)

    def establish(self):
        for _ in range(3):
            result = self.counter.update(detections(bottle=2), timestamp=0)
        self.assertTrue(result["initialized"])
        self.assertEqual(result["stable_counts"], {"bottle": 2})

    def test_single_missed_detection_does_not_change_stable_count(self):
        self.establish()
        result = self.counter.update(detections(bottle=1), timestamp=1)
        self.assertEqual(result["stable_counts"], {"bottle": 2})
        self.assertIsNone(result["drop"])

    def test_persistent_drop_emits_removed_items_once(self):
        self.establish()
        self.counter.update(detections(bottle=1), timestamp=1)
        self.counter.update(detections(bottle=1), timestamp=2)
        result = self.counter.update(detections(bottle=1), timestamp=3)
        self.assertEqual(result["stable_counts"], {"bottle": 1})
        self.assertEqual(result["drop"]["removed_items"], {"bottle": 1})
        following = self.counter.update(detections(bottle=1), timestamp=4)
        self.assertIsNone(following["drop"])

    def test_tracks_classes_independently(self):
        for _ in range(3):
            result = self.counter.update(detections(bottle=2, backpack=1))
        self.assertEqual(result["stable_counts"], {"bottle": 2, "backpack": 1})


class InventoryCounterValidationTests(TestCase):
    def test_warmup_cannot_exceed_window(self):
        with self.assertRaises(ValueError):
            InventoryCounter(window_size=3, warmup_frames=4)

