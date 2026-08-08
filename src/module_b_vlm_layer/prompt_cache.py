"""Small in-memory cache for repeated image/question requests."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Optional

import numpy as np


class PromptCache:
    def __init__(self, max_entries: int = 32) -> None:
        self.max_entries = max_entries
        self._values: OrderedDict[str, str] = OrderedDict()

    @staticmethod
    def key(image: np.ndarray, question: str) -> str:
        digest = hashlib.sha256()
        digest.update(image.tobytes())
        digest.update(str(image.shape).encode())
        digest.update(question.strip().lower().encode("utf-8"))
        return digest.hexdigest()

    def get(self, key: str) -> Optional[str]:
        value = self._values.get(key)
        if value is not None:
            self._values.move_to_end(key)
        return value

    def put(self, key: str, value: str) -> None:
        self._values[key] = value
        self._values.move_to_end(key)
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)

