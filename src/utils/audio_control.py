"""Coordinate siren and text-to-speech playback."""

from __future__ import annotations

from threading import RLock


class AudioCoordinator:
    def __init__(self) -> None:
        self._lock = RLock()
        self._siren_active = False
        self._tts_active = False

    def begin_siren(self) -> None:
        with self._lock:
            self._siren_active = True

    def end_siren(self) -> None:
        with self._lock:
            self._siren_active = False

    def try_begin_tts(self) -> bool:
        with self._lock:
            if self._siren_active or self._tts_active:
                return False
            self._tts_active = True
            return True

    def end_tts(self) -> None:
        with self._lock:
            self._tts_active = False

    @property
    def siren_active(self) -> bool:
        with self._lock:
            return self._siren_active

