"""Non-blocking local text-to-speech coordinated with the siren."""

from __future__ import annotations

import threading
import logging

from src.utils.audio_control import AudioCoordinator


class TTSEngine:
    def __init__(self, coordinator: AudioCoordinator, enabled: bool = True) -> None:
        self.coordinator = coordinator
        self.enabled = enabled

    def health(self) -> dict:
        """Return whether the most recent speech playback completed."""
        error = getattr(self, "last_error", None)
        return {"healthy": error is None, "last_error": error}

    def speak(self, text: str) -> bool:
        if not self.enabled or not text.strip() or not self.coordinator.try_begin_tts():
            return False
        self.last_error = None

        def worker() -> None:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                self.last_error = str(exc)
                logging.getLogger("inventory_security.tts").error(
                    "Text-to-speech failed: %s", exc
                )
            finally:
                self.coordinator.end_tts()

        threading.Thread(target=worker, name="text-to-speech", daemon=True).start()
        return True
