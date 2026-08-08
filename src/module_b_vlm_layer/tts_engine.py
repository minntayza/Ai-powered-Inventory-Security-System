"""Non-blocking local text-to-speech coordinated with the siren."""

from __future__ import annotations

import threading

from src.utils.audio_control import AudioCoordinator


class TTSEngine:
    def __init__(self, coordinator: AudioCoordinator, enabled: bool = True) -> None:
        self.coordinator = coordinator
        self.enabled = enabled

    def speak(self, text: str) -> bool:
        if not self.enabled or not text.strip() or not self.coordinator.try_begin_tts():
            return False

        def worker() -> None:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            finally:
                self.coordinator.end_tts()

        threading.Thread(target=worker, name="text-to-speech", daemon=True).start()
        return True

