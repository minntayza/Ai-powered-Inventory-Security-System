"""Application controller that orchestrates capture, perception, decisions, and alerts."""

from __future__ import annotations

import logging
import threading
import time
from copy import deepcopy
from typing import Dict, Optional

from src.utils.audio_control import AudioCoordinator
from src.utils.config_loader import load_app_config, resolve_project_path
from src.utils.frame_capture import FrameCapture
from src.utils.gpu_manager import device_info
from src.utils.logger import configure_logging
from src.utils.runtime_state import RuntimeState

from .audit_logger import AuditLogger
from .siren import Siren
from .telegram_bot import TelegramAlerter
from .theft_detector import TheftDetector


class SystemController:
    """Own long-lived application resources and publish dashboard-safe snapshots."""

    def __init__(self, config_dir: str = "configs", tracker=None, capture=None) -> None:
        self.config = load_app_config(config_dir)
        self.logger = configure_logging()
        self.state = RuntimeState()
        self.audio = AudioCoordinator()
        self.tracker = tracker
        self.capture = capture or FrameCapture(**self.config["camera"])
        security = self.config["thresholds"].get("security", {})
        self.theft_detector = TheftDetector(
            grace_period_seconds=float(security.get("grace_period_seconds", 3)),
            cooldown_seconds=float(security.get("alert_cooldown_seconds", 30)),
            shelf_region=security.get("shelf_region"),
        )
        storage = self.config["alerts"]["storage"]
        self.audit = AuditLogger(storage["database_path"], storage["snapshot_directory"])
        siren = self.config["alerts"]["siren"]
        self.siren = Siren(
            audio_path=siren["audio_path"],
            enabled=bool(siren.get("enabled", True)),
            coordinator=self.audio,
        )
        telegram = self.config["alerts"]["telegram"]
        self.telegram = TelegramAlerter(
            enabled=bool(telegram.get("enabled", False)),
            timeout_seconds=float(telegram.get("timeout_seconds", 10)),
            max_attempts=int(telegram.get("max_attempts", 2)),
        )
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def _build_tracker(self):
        from src.module_a_perception_engine.face_detector import FaceDetector
        from src.module_a_perception_engine.inventory_counter import InventoryCounter
        from src.module_a_perception_engine.person_tracker import PersonTracker
        from src.module_a_perception_engine.yolo_detector import YOLODetector

        yolo_config = self.config["models"]["yolo"]
        targets = yolo_config.get("target_classes", ["person", "bottle", "backpack"])
        detector = YOLODetector(
            model_path=str(resolve_project_path(yolo_config.get("model_path", "yolov8n.pt")))
            if "/" in yolo_config.get("model_path", "") or "\\" in yolo_config.get("model_path", "")
            else yolo_config.get("model_path", "yolov8n.pt"),
            confidence=float(yolo_config.get("confidence_threshold", 0.5)),
            inventory_classes={name for name in targets if name != "person"},
            iou_threshold=float(yolo_config.get("iou_threshold", 0.45)),
            image_size=int(yolo_config.get("image_size", 640)),
            device=yolo_config.get("device", "auto"),
            tracking=bool(yolo_config.get("tracking", True)),
        )
        face_config = self.config["thresholds"].get("face_recognition", {})
        face = FaceDetector(
            db_path=str(resolve_project_path(face_config.get("db_path", "assets/known_faces"))),
            model_name=face_config.get("model_name", "VGG-Face"),
            detector_backend=face_config.get("detector_backend", "opencv"),
            recognition_threshold=float(face_config.get("recognition_threshold", 0.4)),
            anti_spoofing=bool(face_config.get("anti_spoofing", True)),
        )
        inventory_config = self.config["thresholds"].get("inventory", {})
        counter = InventoryCounter(
            window_size=int(inventory_config.get("window_size", 15)),
            confirmation_frames=int(inventory_config.get("confirmation_frames", 5)),
            warmup_frames=int(inventory_config.get("warmup_frames", 15)),
        )
        tracking = self.config["thresholds"].get("person_tracking", {})
        return PersonTracker(
            grace_period=float(tracking.get("authorization_cache_seconds", 3)),
            face_interval_frames=int(tracking.get("face_interval_frames", 5)),
            track_expiry_seconds=float(tracking.get("track_expiry_seconds", 5)),
            yolo_detector=detector,
            face_detector=face,
            inventory_counter=counter,
        )

    def start(self) -> "SystemController":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self.capture.start()
        self._thread = threading.Thread(target=self._run, name="security-engine", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        self.state.update(running=True, model_status="initializing", last_error=None)
        last_frame_id = -1
        try:
            if self.tracker is None:
                self.tracker = self._build_tracker_with_camera_preview()
                if self.tracker is None:
                    return
            selected_device = device_info(
                self.config["models"]["yolo"].get("device", "auto")
            )
            yolo = getattr(self.tracker, "yolo_detector", None)
            selected_device["selected"] = getattr(
                yolo, "device", selected_device["selected"]
            )
            self.state.update(model_status="ready", device_info=selected_device)
            while not self._stop.is_set():
                healthy, frame, frame_id = self.capture.read()
                self.state.update(camera_healthy=healthy)
                if not healthy or frame is None or frame_id == last_frame_id:
                    self._stop.wait(0.02)
                    continue
                last_frame_id = frame_id
                perception = self.tracker.process_frame(frame)
                perception["frame_id"] = frame_id
                perception["camera_healthy"] = healthy
                annotated = self.tracker.draw_tracking(frame, perception)
                decision = self.theft_detector.evaluate(perception)
                event = decision.get("new_event")
                if event:
                    confirmed = event["status"] == "confirmed"
                    record = self.audit.log_event(event, annotated if confirmed else None)
                    self.theft_detector.last_event = deepcopy(record)
                    decision["last_event"] = deepcopy(record)
                    decision["new_event"] = deepcopy(record)
                    if confirmed:
                        self.siren.start()
                        threading.Thread(
                            target=self._notify,
                            args=(deepcopy(record),),
                            name=f"telegram-{record['event_id'][:8]}",
                            daemon=True,
                        ).start()
                active_device = self.state.snapshot().get("device_info") or {}
                yolo = getattr(self.tracker, "yolo_detector", None)
                actual_device = getattr(yolo, "device", active_device.get("selected"))
                if active_device.get("selected") != actual_device:
                    active_device["selected"] = actual_device
                self.state.update(
                    frame=annotated,
                    perception=perception,
                    security=decision,
                    device_info=active_device,
                    last_error=None,
                )
        except Exception as exc:
            self.logger.exception("Monitoring engine stopped: %s", exc)
            self.state.update(last_error=str(exc))
        finally:
            self.state.update(running=False, camera_healthy=False, model_status="stopped")

    def _build_tracker_with_camera_preview(self):
        """Initialize heavy models while continuing to publish raw webcam frames."""
        completed = threading.Event()
        result: Dict = {}

        def initialize() -> None:
            try:
                result["tracker"] = self._build_tracker()
            except Exception as exc:
                result["error"] = exc
            finally:
                completed.set()

        threading.Thread(
            target=initialize, name="model-initialization", daemon=True
        ).start()
        last_preview_id = -1
        while not completed.is_set() and not self._stop.is_set():
            healthy, frame, frame_id = self.capture.read()
            values = {"camera_healthy": healthy, "model_status": "initializing"}
            if healthy and frame is not None and frame_id != last_preview_id:
                values["frame"] = frame
                last_preview_id = frame_id
            self.state.update(**values)
            self._stop.wait(0.03)
        if self._stop.is_set():
            return None
        if "error" in result:
            raise result["error"]
        return result["tracker"]

    def _notify(self, event: Dict) -> None:
        result = self.telegram.send_event(event)
        self.audit.update_event(event["event_id"], telegram_status=result["status"])
        self.audit.log_alert_attempt(
            event["event_id"], "telegram", result["status"], result.get("detail", "")
        )
        snapshot = self.state.snapshot()
        security = snapshot["security"]
        current = security.get("last_event")
        if current and current.get("event_id") == event["event_id"]:
            current["telegram_status"] = result["status"]
            security["last_event"] = current
            self.state.update(security=security)

    def acknowledge(self, event_id: Optional[str] = None) -> None:
        security = self.state.snapshot()["security"]
        event = security.get("last_event")
        target = event_id or (event or {}).get("event_id")
        if target:
            self.audit.update_event(target, acknowledged=True)
            self.audit.log_operator_action(target, "acknowledge_and_stop_siren")
            if event and event.get("event_id") == target:
                event["acknowledged"] = True
                security["last_event"] = event
                self.state.update(security=security)
        self.theft_detector.acknowledge()
        self.siren.stop()

    def recent_events(self, limit: int = 50):
        return self.audit.recent_events(limit)

    def snapshot(self) -> Dict:
        return self.state.snapshot()

    def stop(self) -> None:
        self._stop.set()
        self.capture.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        if self.tracker is not None and hasattr(self.tracker, "shutdown"):
            self.tracker.shutdown()
        self.siren.stop()
        self.state.update(running=False, camera_healthy=False)


def main() -> None:
    controller = SystemController().start()
    print("Inventory security engine running. Press Ctrl+C to stop.")
    try:
        while controller.snapshot()["running"]:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
