"""Coordinate siren and text-to-speech playback."""

from __future__ import annotations

from threading import RLock


class AudioCoordinator:
    def __init__(self) -> None:
        self._lock = RLock()
        self._siren_active = False
        self._tts_active = False
        self._microphone_active = False

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

    def try_begin_microphone(self) -> bool:
        """Acquire exclusive microphone capture when playback is idle."""
        with self._lock:
            if self._siren_active or self._tts_active or self._microphone_active:
                return False
            self._microphone_active = True
            return True

    def end_microphone(self) -> None:
        """Release exclusive microphone capture."""
        with self._lock:
            self._microphone_active = False

    @property
    def siren_active(self) -> bool:
        with self._lock:
            return self._siren_active

    @property
    def microphone_available(self) -> bool:
        """Return whether capturing operator speech is currently safe."""
        with self._lock:
            return not self._siren_active and not self._tts_active and not self._microphone_active
