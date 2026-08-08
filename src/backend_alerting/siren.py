"""Idempotent local siren playback."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from src.utils.audio_control import AudioCoordinator
from src.utils.config_loader import resolve_project_path


class Siren:
    def __init__(self, audio_path: str, enabled: bool = True, coordinator: AudioCoordinator | None = None):
        self.audio_path = resolve_project_path(audio_path)
        self.enabled = enabled
        self.coordinator = coordinator or AudioCoordinator()
        self._active = False
        self._lock = threading.Lock()
        self._logger = logging.getLogger("inventory_security.siren")

    def start(self) -> bool:
        with self._lock:
            if self._active:
                return True
            if not self.enabled:
                return False
            if not self.audio_path.exists():
                self._logger.warning("Siren audio file does not exist: %s", self.audio_path)
                return False
            try:
                import pygame
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(str(self.audio_path))
                pygame.mixer.music.play(loops=-1)
                self._active = True
                self.coordinator.begin_siren()
                return True
            except Exception as exc:
                self._logger.error("Could not start siren: %s", exc)
                return False

    def stop(self) -> None:
        with self._lock:
            if self._active:
                try:
                    import pygame
                    pygame.mixer.music.stop()
                except Exception as exc:
                    self._logger.warning("Could not stop siren cleanly: %s", exc)
            self._active = False
            self.coordinator.end_siren()

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._active

