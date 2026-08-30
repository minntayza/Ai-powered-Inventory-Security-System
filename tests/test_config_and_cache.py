import os
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from src.module_b_vlm_layer.prompt_cache import PromptCache
from src.utils.config_loader import (
    ConfigError,
    _validate_inventory_policy,
    load_app_config,
    load_environment,
)


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

    def test_project_environment_file_is_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("INVENTORY_TEST_VALUE=from-file\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertTrue(load_environment(env_path))
                self.assertEqual(os.environ["INVENTORY_TEST_VALUE"], "from-file")

    def test_process_environment_takes_precedence_over_file(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("INVENTORY_TEST_VALUE=from-file\n", encoding="utf-8")
            with patch.dict(
                os.environ, {"INVENTORY_TEST_VALUE": "from-process"}, clear=True
            ):
                load_environment(env_path)
                self.assertEqual(os.environ["INVENTORY_TEST_VALUE"], "from-process")


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
