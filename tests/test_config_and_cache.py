from unittest import TestCase

import numpy as np

from src.module_b_vlm_layer.prompt_cache import PromptCache
from src.utils.config_loader import ConfigError, _validate_inventory_policy, load_app_config


class ConfigurationTests(TestCase):
    def test_repository_configuration_loads(self):
        config = load_app_config()
        self.assertEqual(config["camera"]["source"], 0)
        self.assertIn("yolo", config["models"])

    def test_inventory_policy_rejects_overlap(self):
        with self.assertRaises(ConfigError):
            _validate_inventory_policy(
                {
                    "yolo": {"target_classes": ["person", "chair"]},
                    "inventory_policy": {
                        "protected": ["chair"],
                        "contextual": ["chair"],
                    },
                }
            )

    def test_person_cannot_be_inventory(self):
        with self.assertRaises(ConfigError):
            _validate_inventory_policy(
                {
                    "yolo": {"target_classes": ["person"]},
                    "inventory_policy": {"protected": ["person"], "contextual": []},
                }
            )


class PromptCacheTests(TestCase):
    def test_cache_key_changes_with_question_or_image(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        first = PromptCache.key(image, "What is here?")
        second = PromptCache.key(image, "Who is here?")
        image[0, 0] = 1
        third = PromptCache.key(image, "What is here?")
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)

    def test_cache_evicts_oldest_entry(self):
        cache = PromptCache(max_entries=1)
        cache.put("one", "first")
        cache.put("two", "second")
        self.assertIsNone(cache.get("one"))
        self.assertEqual(cache.get("two"), "second")
