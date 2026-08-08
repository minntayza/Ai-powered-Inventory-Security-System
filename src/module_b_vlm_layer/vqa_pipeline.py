"""Validated, cached VQA facade for the dashboard."""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from .prompt_cache import PromptCache
from .vlm_engine import FlorenceEngine


class VQAPipeline:
    def __init__(self, engine: FlorenceEngine, cache: Optional[PromptCache] = None) -> None:
        self.engine = engine
        self.cache = cache or PromptCache()

    def ask(self, image: np.ndarray, question: str, source_id: str = "live") -> Dict:
        if image is None or image.size == 0:
            return {"ok": False, "answer": "", "error": "No image is available", "source_id": source_id}
        if not question or not question.strip():
            return {"ok": False, "answer": "", "error": "Question cannot be empty", "source_id": source_id}
        key = self.cache.key(image, question)
        cached = self.cache.get(key)
        if cached is not None:
            return self._response(cached, source_id, cached=True)
        try:
            answer = self.engine.answer(image, question)
            self.cache.put(key, answer)
            return self._response(answer, source_id, cached=False)
        except Exception as exc:
            return {"ok": False, "answer": "", "error": str(exc), "source_id": source_id}

    @staticmethod
    def _response(answer: str, source_id: str, cached: bool) -> Dict:
        return {
            "ok": True,
            "answer": answer,
            "error": None,
            "source_id": source_id,
            "cached": cached,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

