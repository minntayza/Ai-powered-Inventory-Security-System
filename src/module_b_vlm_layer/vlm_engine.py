"""Lazy local Florence-2 inference engine."""

from __future__ import annotations

import threading
from json import dumps
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from src.utils.gpu_manager import resolve_torch_device


class FlorenceEngine:
    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-base-ft",
        device: str = "auto",
        max_new_tokens: int = 128,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device_setting = device
        self.max_new_tokens = max_new_tokens
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.processor = None
        self.device: Optional[str] = None
        self._lock = threading.RLock()
        self._cpu_fallback_attempted = False

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self) -> None:
        with self._lock:
            if self.loaded:
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            self.device = resolve_torch_device(self.device_setting)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=dtype,
            ).to(self.device)
            self.model.eval()

    def answer(self, image_bgr: np.ndarray, question: str) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        self.load()
        import torch

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        task = self._select_task(question)
        try:
            with self._lock, torch.inference_mode():
                inputs = self.processor(text=task, images=image, return_tensors="pt")
                inputs = {
                    key: value.to(self.device, dtype=self.model.dtype)
                    if key == "pixel_values"
                    else value.to(self.device)
                    for key, value in inputs.items()
                }
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=3,
                )
                generated_text = self.processor.batch_decode(
                    generated, skip_special_tokens=False
                )[0]
                parsed = self.processor.post_process_generation(
                    generated_text,
                    task=task,
                    image_size=(image.width, image.height),
                )
            answer = parsed.get(task, parsed) if isinstance(parsed, dict) else parsed
            if isinstance(answer, (dict, list)):
                return dumps(answer, ensure_ascii=False)
            return str(answer).strip()
        except RuntimeError:
            if self.device == "cpu" or self._cpu_fallback_attempted:
                raise
            self._cpu_fallback_attempted = True
            self.unload()
            self.device_setting = "cpu"
            return self.answer(image_bgr, question)

    @staticmethod
    def _select_task(question: str) -> str:
        normalized = question.lower()
        text_keywords = ("read", "text", "written", "write", "say", "word")
        if any(keyword in normalized for keyword in text_keywords):
            return "<OCR>"
        # Florence-2 does not implement general chat prompting. A detailed visual
        # caption is the reliable supported task for action/scene questions.
        return "<MORE_DETAILED_CAPTION>"

    def unload(self) -> None:
        with self._lock:
            self.model = None
            self.processor = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if (
                    self.device == "mps"
                    and hasattr(torch, "mps")
                    and hasattr(torch.mps, "empty_cache")
                ):
                    torch.mps.empty_cache()
            except ImportError:
                pass
