from unittest import TestCase

from packaging.version import Version
from transformers import __version__ as transformers_version

from src.module_b_vlm_layer.vlm_engine import FlorenceEngine


class FlorenceCompatibilityTests(TestCase):
    def test_transformers_stays_on_supported_major_version(self):
        version = Version(transformers_version)
        self.assertGreaterEqual(version, Version("4.41.0"))
        self.assertLess(version, Version("5.0.0"))

    def test_action_question_uses_detailed_caption_task(self):
        self.assertEqual(
            FlorenceEngine._select_task("What is the person doing?"),
            "<MORE_DETAILED_CAPTION>",
        )

    def test_text_question_uses_ocr_task(self):
        self.assertEqual(
            FlorenceEngine._select_task("What does the sign say?"), "<OCR>"
        )
