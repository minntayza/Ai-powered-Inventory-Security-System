from unittest import TestCase

import numpy as np
from packaging.version import Version
from transformers import __version__ as transformers_version

from src.module_b_vlm_layer.vlm_engine import FlorenceEngine
from src.module_b_vlm_layer.vqa_pipeline import VQAPipeline


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


class VqaPipelineTests(TestCase):
    def test_identical_question_and_image_reuse_cached_answer(self):
        class Engine:
            def __init__(self):
                self.calls = 0

            def answer(self, _image, _question):
                self.calls += 1
                return "One laptop is visible"

        engine = Engine()
        pipeline = VQAPipeline(engine)
        image = np.zeros((10, 10, 3), dtype=np.uint8)

        first = pipeline.ask(image, "What is visible?")
        second = pipeline.ask(image, "What is visible?")

        self.assertEqual(
            {
                "answers": [first["answer"], second["answer"]],
                "cached": [first["cached"], second["cached"]],
                "model_calls": engine.calls,
            },
            {
                "answers": ["One laptop is visible", "One laptop is visible"],
                "cached": [False, True],
                "model_calls": 1,
            },
        )
