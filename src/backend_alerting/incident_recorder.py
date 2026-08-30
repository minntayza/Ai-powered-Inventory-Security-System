"""Bounded pre/post-event video evidence recording."""

from __future__ import annotations

import logging
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.config_loader import resolve_project_path


EncodedFrame = Tuple[float, bytes]


class IncidentRecorder:
    def __init__(
        self,
        directory: str,
        fps: float = 10,
        pre_event_seconds: float = 10,
        post_event_seconds: float = 10,
        jpeg_quality: int = 75,
        retention_days: int = 30,
        enabled: bool = True,
    ) -> None:
        self.directory = resolve_project_path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fps = max(1.0, float(fps))
        self.pre_event_seconds = max(0.0, float(pre_event_seconds))
        self.post_event_seconds = max(0.0, float(post_event_seconds))
        self.jpeg_quality = max(20, min(95, int(jpeg_quality)))
        self.retention_days = max(1, int(retention_days))
        self.enabled = bool(enabled)
        self._buffer: Deque[EncodedFrame] = deque()
        self._buffer_bytes = 0
        self._active: Dict[str, Dict] = {}
        self._futures: Dict[Future, str] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="incident-video")
        self._logger = logging.getLogger("inventory_security.video")
        self.cleanup()

    def observe(self, frame: np.ndarray, now: Optional[float] = None) -> None:
        if not self.enabled or frame is None:
            return
        now = time.time() if now is None else float(now)
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            return
        item = (now, encoded.tobytes())
        self._buffer.append(item)
        self._buffer_bytes += len(item[1])
        cutoff = now - self.pre_event_seconds
        while self._buffer and self._buffer[0][0] < cutoff:
            _timestamp, payload = self._buffer.popleft()
            self._buffer_bytes -= len(payload)
        completed = []
        for event_id, job in self._active.items():
            job["frames"].append(item)
            if now >= job["end_at"]:
                completed.append(event_id)
        for event_id in completed:
            job = self._active.pop(event_id)
            future = self._executor.submit(self._write_video, event_id, job["frames"])
            self._futures[future] = event_id

    def start_event(self, event_id: str, now: Optional[float] = None) -> None:
        if not self.enabled or event_id in self._active:
            return
        now = time.time() if now is None else float(now)
        self._active[event_id] = {
            "end_at": now + self.post_event_seconds,
            "frames": list(self._buffer),
        }
        if self.post_event_seconds == 0:
            job = self._active.pop(event_id)
            future = self._executor.submit(self._write_video, event_id, job["frames"])
            self._futures[future] = event_id

    def poll_completed(self) -> List[Dict]:
        results = []
        for future, event_id in list(self._futures.items()):
            if not future.done():
                continue
            del self._futures[future]
            try:
                results.append({"event_id": event_id, "path": future.result(), "error": None})
            except Exception as exc:
                self._logger.exception("Could not write incident video %s", event_id)
                results.append({"event_id": event_id, "path": None, "error": str(exc)})
        return results

    def _write_video(self, event_id: str, frames: List[EncodedFrame]) -> str:
        if not frames:
            raise ValueError("No frames available for incident video")
        first = cv2.imdecode(np.frombuffer(frames[0][1], dtype=np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            raise ValueError("Could not decode buffered incident frame")
        height, width = first.shape[:2]
        mp4_path = self.directory / f"{event_id}.mp4"
        writer = cv2.VideoWriter(
            str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
        )
        output_path = mp4_path
        if not writer.isOpened():
            writer.release()
            output_path = self.directory / f"{event_id}.avi"
            writer = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (width, height)
            )
        if not writer.isOpened():
            raise IOError("No supported OpenCV video encoder is available")
        try:
            for _timestamp, payload in frames:
                image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                if image.shape[1] != width or image.shape[0] != height:
                    image = cv2.resize(image, (width, height))
                writer.write(image)
        finally:
            writer.release()
        return str(output_path)

    def buffer_megabytes(self) -> float:
        return round(self._buffer_bytes / 1_048_576, 2)

    def reset_buffer(self) -> None:
        self._buffer.clear()
        self._buffer_bytes = 0

    def cleanup(self) -> None:
        cutoff = time.time() - self.retention_days * 86400
        for pattern in ("*.mp4", "*.avi"):
            for path in self.directory.glob(pattern):
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                except OSError as exc:
                    self._logger.warning("Could not remove expired incident video %s: %s", path, exc)

    def shutdown(self) -> List[Dict]:
        for event_id, job in list(self._active.items()):
            future = self._executor.submit(self._write_video, event_id, job["frames"])
            self._futures[future] = event_id
        self._active.clear()
        self._executor.shutdown(wait=True, cancel_futures=False)
        return self.poll_completed()
