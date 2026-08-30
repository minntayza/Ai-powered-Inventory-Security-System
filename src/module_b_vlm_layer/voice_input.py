"""Local speech-to-text input for visual questions."""

from __future__ import annotations

from typing import Callable, Dict, Optional

from src.utils.audio_control import AudioCoordinator


class VoiceQuestionTranscriber:
    """Turn a dashboard audio recording into a text question."""

    def transcribe(
        self,
        audio_bytes: bytes,
        recognizer: Optional[Callable[[bytes], object]] = None,
        coordinator: Optional[AudioCoordinator] = None,
    ) -> Dict[str, object]:
        if not audio_bytes:
            return {
                "ok": False,
                "question": "",
                "error": "Voice recording is empty",
            }
        if coordinator is not None and not coordinator.try_begin_microphone():
            return {
                "ok": False,
                "question": "",
                "error": "Voice input is unavailable while audio is playing",
            }
        try:
            if recognizer is None:
                recognizer = getattr(self, "_recognizer", None)
                if recognizer is None:
                    import transformers

                    recognizer = transformers.pipeline(
                        "automatic-speech-recognition",
                        model="openai/whisper-tiny.en",
                        device=-1,
                    )
                    self._recognizer = recognizer
            result = recognizer(audio_bytes)
        except Exception as exc:
            return {"ok": False, "question": "", "error": str(exc)}
        finally:
            if coordinator is not None:
                coordinator.end_microphone()
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        question = text.strip()
        if not question:
            return {
                "ok": False,
                "question": "",
                "error": "No speech was recognized",
            }
        return {"ok": True, "question": question, "error": None}
