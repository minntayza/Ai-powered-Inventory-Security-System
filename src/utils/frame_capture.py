"""Background OpenCV capture that always exposes the newest frame."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Union
from pathlib import Path

import cv2
import numpy as np


class FrameCapture:
    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 15,
        reconnect_delay_seconds: float = 1.0,
        max_reconnect_delay_seconds: float = 10.0,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.reconnect_delay = reconnect_delay_seconds
        self.max_reconnect_delay = max_reconnect_delay_seconds
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._healthy = False
        self._frame_id = 0
        self._logger = logging.getLogger("inventory_security.camera")

    def start(self) -> "FrameCapture":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="frame-capture", daemon=True)
        self._thread.start()
        return self

    def _open(self) -> bool:
        self._release()
        capture = cv2.VideoCapture(self.source)
        # Keep live sources responsive when inference briefly falls behind.
        # Backends that support this retain the newest frame instead of
        # making the UI work through an old camera-frame backlog.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        if not capture.isOpened():
            capture.release()
            return False
        self._capture = capture
        return True

    def _run(self) -> None:
        delay = self.reconnect_delay
        while not self._stop.is_set():
            if self._capture is None and not self._open():
                self._set_health(False)
                self._stop.wait(delay)
                delay = min(delay * 2, self.max_reconnect_delay)
                continue
            ok, frame = self._capture.read()
            if not ok or frame is None:
                self._set_health(False)
                self._release()
                self._stop.wait(delay)
                delay = min(delay * 2, self.max_reconnect_delay)
                continue
            delay = self.reconnect_delay
            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self._healthy = True
        self._release()
        self._set_health(False)

    def read(self) -> tuple[bool, Optional[np.ndarray], int]:
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            return self._healthy and frame is not None, frame, self._frame_id

    def _set_health(self, healthy: bool) -> None:
        with self._lock:
            self._healthy = healthy

    @property
    def healthy(self) -> bool:
        with self._lock:
            return self._healthy

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._release()

    def _release(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()

    def __enter__(self) -> "FrameCapture":
        return self.start()

    def __exit__(self, *_args: object) -> None:
        self.stop()


class VideoFileCapture:
    """Read a prerecorded file at a bounded real-time rate."""

    def __init__(self, path: str | Path, max_fps: float = 10, loop: bool = False) -> None:
        self.path = str(path)
        self.max_fps = max(1.0, float(max_fps))
        self.loop = bool(loop)
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._healthy = False
        self._frame_id = 0

    def start(self) -> "VideoFileCapture":
        if self._thread and self._thread.is_alive():
            return self
        capture = cv2.VideoCapture(self.path)
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Could not open replay video: {self.path}")
        source_fps = capture.get(cv2.CAP_PROP_FPS) or self.max_fps
        self._period = 1.0 / min(self.max_fps, max(1.0, source_fps))
        self._capture = capture
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="video-replay", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set() and self._capture is not None:
            started = time.perf_counter()
            ok, frame = self._capture.read()
            if not ok or frame is None:
                if self.loop:
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                with self._lock:
                    self._healthy = False
                break
            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self._healthy = True
            self._stop.wait(max(0.0, self._period - (time.perf_counter() - started)))
        self._release()

    def read(self) -> tuple[bool, Optional[np.ndarray], int]:
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            return self._healthy and frame is not None, frame, self._frame_id

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._release()

    def _release(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()
