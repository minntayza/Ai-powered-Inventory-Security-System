"""Application controller that orchestrates capture, perception, decisions, and alerts."""

from __future__ import annotations

import logging
import json
import threading
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

from src.utils.audio_control import AudioCoordinator
from src.utils.config_loader import load_app_config, resolve_project_path
from src.utils.frame_capture import FrameCapture, VideoFileCapture
from src.utils.gpu_manager import device_info
from src.utils.logger import configure_logging
from src.utils.performance_monitor import PerformanceMonitor
from src.utils.runtime_state import RuntimeState

from .audit_logger import AuditLogger
from .incident_recorder import IncidentRecorder
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
        self._engine_lock = threading.RLock()
        self._capture_lock = threading.RLock()
        self._monitoring_mode = "DISARMED"
        self._source_type = "live"
        self._source_label = "Live camera"
        self._source_generation = 0
        self.performance = PerformanceMonitor()
        security = self.config["thresholds"].get("security", {})
        storage = self.config["alerts"]["storage"]
        database_path = resolve_project_path(storage["database_path"])
        self._settings_path = database_path.parent / "runtime_settings.json"
        settings = self._load_runtime_settings()
        self._shelf_region = settings.get("shelf_region", security.get("shelf_region"))
        self.theft_detector = TheftDetector(
            grace_period_seconds=float(security.get("grace_period_seconds", 3)),
            cooldown_seconds=float(security.get("alert_cooldown_seconds", 30)),
            shelf_region=self._shelf_region,
        )
        self.audit = AuditLogger(storage["database_path"], storage["snapshot_directory"])
        video = storage.get("incident_video", {})
        self.incident_recorder = IncidentRecorder(
            directory=video.get("directory", "data/logs/incidents"),
            fps=float(self.config["camera"].get("fps", 10)),
            pre_event_seconds=float(video.get("pre_event_seconds", 10)),
            post_event_seconds=float(video.get("post_event_seconds", 10)),
            jpeg_quality=int(video.get("jpeg_buffer_quality", 75)),
            retention_days=int(storage.get("retention_days", 30)),
            enabled=bool(video.get("enabled", False)),
        )
        self.audit.cleanup_media(
            int(storage.get("retention_days", 30)),
            replay_directory="data/logs/replay_uploads",
        )
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
        self._face_enrollment = None
        self.state.update(
            monitoring=self._monitoring_snapshot(),
            source={"type": self._source_type, "label": self._source_label},
        )

    def _build_tracker(self):
        from src.module_a_perception_engine.face_detector import FaceDetector
        from src.module_a_perception_engine.inventory_counter import InventoryCounter
        from src.module_a_perception_engine.interaction_associator import InteractionAssociator
        from src.module_a_perception_engine.person_tracker import PersonTracker
        from src.module_a_perception_engine.yolo_detector import YOLODetector

        yolo_config = self.config["models"]["yolo"]
        targets = yolo_config.get("target_classes", ["person", "bottle", "backpack"])
        policy = self.config["models"].get("inventory_policy", {})
        protected = set(policy.get("protected") or [name for name in targets if name != "person"])
        contextual = set(policy.get("contextual") or [])
        detector = YOLODetector(
            model_path=str(resolve_project_path(yolo_config.get("model_path", "yolov8n.pt")))
            if "/" in yolo_config.get("model_path", "") or "\\" in yolo_config.get("model_path", "")
            else yolo_config.get("model_path", "yolov8n.pt"),
            confidence=float(yolo_config.get("confidence_threshold", 0.5)),
            inventory_classes=protected | contextual,
            protected_classes=protected,
            contextual_classes=contextual,
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
            require_manual_baseline=True,
        )
        contextual_counter = InventoryCounter(
            window_size=int(inventory_config.get("window_size", 15)),
            confirmation_frames=int(inventory_config.get("confirmation_frames", 5)),
            warmup_frames=int(inventory_config.get("warmup_frames", 15)),
        )
        tracking = self.config["thresholds"].get("person_tracking", {})
        association = self.config["thresholds"].get("association", {})
        return PersonTracker(
            grace_period=float(tracking.get("authorization_cache_seconds", 3)),
            face_interval_frames=int(tracking.get("face_interval_frames", 5)),
            track_expiry_seconds=float(tracking.get("track_expiry_seconds", 5)),
            yolo_detector=detector,
            face_detector=face,
            inventory_counter=counter,
            contextual_counter=contextual_counter,
            protected_classes=protected,
            contextual_classes=contextual,
            shelf_region=self._shelf_region,
            interaction_associator=InteractionAssociator(
                lookback_seconds=float(association.get("lookback_seconds", 2.0)),
                person_bbox_expansion=float(association.get("person_bbox_expansion", 0.15)),
                minimum_score=float(association.get("minimum_score", 0.35)),
                ambiguity_margin=float(association.get("ambiguity_margin", 0.10)),
            ),
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
        source_generation = self._source_generation
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
                with self._capture_lock:
                    healthy, frame, frame_id = self.capture.read()
                    current_generation = self._source_generation
                if current_generation != source_generation:
                    source_generation = current_generation
                    last_frame_id = -1
                self.state.update(camera_healthy=healthy)
                if not healthy or frame is None or frame_id == last_frame_id:
                    self._stop.wait(0.02)
                    continue
                previous_frame_id = last_frame_id
                last_frame_id = frame_id
                self.performance.capture_tick()
                if previous_frame_id >= 0 and frame_id > previous_frame_id + 1:
                    self.performance.skip(frame_id - previous_frame_id - 1)
                processing_started = time.perf_counter()
                with self._engine_lock:
                    perception = self.tracker.process_frame(frame)
                    perception["frame_id"] = frame_id
                    perception["camera_healthy"] = healthy
                    annotated = self.tracker.draw_tracking(frame, perception)
                    self.incident_recorder.observe(annotated)
                    if self._monitoring_mode == "ARMED":
                        decision = self.theft_detector.evaluate(perception)
                    else:
                        decision = self.theft_detector.snapshot()
                    monitoring = self._monitoring_snapshot(perception)
                processing_ms = (time.perf_counter() - processing_started) * 1000
                self.performance.record("processing_ms", processing_ms)
                self.performance.processed_tick()
                yolo = getattr(self.tracker, "yolo_detector", None)
                face = getattr(self.tracker, "face_detector", None)
                if yolo is not None:
                    self.performance.record("yolo_ms", getattr(yolo, "last_inference_ms", 0.0))
                if face is not None and getattr(face, "last_inference_ms", 0.0):
                    self.performance.record("face_ms", face.last_inference_ms)
                event = decision.get("new_event")
                if event:
                    event["source_type"] = self._source_type
                    if self._source_type == "replay":
                        event["telegram_status"] = "disabled"
                    confirmed = event["status"] == "confirmed"
                    record = self.audit.log_event(event, annotated if confirmed else None)
                    self.theft_detector.last_event = deepcopy(record)
                    decision["last_event"] = deepcopy(record)
                    decision["new_event"] = deepcopy(record)
                    if confirmed:
                        self.incident_recorder.start_event(record["event_id"])
                        if self._source_type == "live":
                            self.siren.start()
                            threading.Thread(
                                target=self._notify,
                                args=(deepcopy(record),),
                                name=f"telegram-{record['event_id'][:8]}",
                                daemon=True,
                            ).start()
                self._collect_completed_videos()
                active_device = self.state.snapshot().get("device_info") or {}
                yolo = getattr(self.tracker, "yolo_detector", None)
                actual_device = getattr(yolo, "device", active_device.get("selected"))
                if active_device.get("selected") != actual_device:
                    active_device["selected"] = actual_device
                self.state.update(
                    frame=annotated,
                    perception=perception,
                    security=decision,
                    monitoring=monitoring,
                    performance=self._performance_snapshot(),
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
            with self._capture_lock:
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

    def _collect_completed_videos(self) -> None:
        for result in self.incident_recorder.poll_completed():
            if result.get("path"):
                self.audit.update_event(result["event_id"], video_path=result["path"])
                snapshot = self.state.snapshot()
                security = snapshot.get("security", {})
                current = security.get("last_event")
                if current and current.get("event_id") == result["event_id"]:
                    current["video_path"] = result["path"]
                    security["last_event"] = current
                    self.state.update(security=security)
            elif result.get("error"):
                self.audit.log_alert_attempt(
                    result["event_id"], "incident_video", "failed", result["error"]
                )

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

    def _enrollment_service(self):
        if self._face_enrollment is None:
            from src.module_a_perception_engine.face_enrollment import FaceEnrollmentService

            face = self.config["thresholds"].get("face_recognition", {})
            tracking = self.config["thresholds"].get("person_tracking", {})
            self._face_enrollment = FaceEnrollmentService(
                db_path=str(resolve_project_path(face.get("db_path", "assets/known_faces"))),
                detector_backend=face.get("detector_backend", "opencv"),
                min_face_size=int(tracking.get("min_face_size", 50)),
            )
        return self._face_enrollment

    def known_faces(self) -> Dict[str, int]:
        return self._enrollment_service().identities()

    def enroll_face(self, name: str, uploads) -> Dict:
        result = self._enrollment_service().enroll(name, uploads)
        if result.get("ok"):
            tracker = self.tracker
            if tracker is not None:
                tracker.face_detector.refresh_database()
            self.audit.log_operator_action(None, f"enroll_face:{result['name']}")
        return result

    def remove_face(self, name: str) -> Dict:
        result = self._enrollment_service().remove(name)
        if result.get("ok"):
            tracker = self.tracker
            if tracker is not None:
                tracker.face_detector.refresh_database()
            self.audit.log_operator_action(None, f"remove_face:{name}")
        return result

    def set_baseline_and_arm(self) -> Dict:
        with self._engine_lock:
            counter = self._inventory_counter()
            if counter is None:
                return {"ok": False, "message": "Models are still initializing"}
            if not counter.set_baseline():
                return {
                    "ok": False,
                    "message": "Wait for stable protected inventory inside the zone",
                }
            self.theft_detector.reset()
            self._monitoring_mode = "ARMED"
            monitoring = self._monitoring_snapshot()
            self.state.update(monitoring=monitoring, security=self.theft_detector.snapshot())
            self.audit.log_operator_action(None, "set_baseline_and_arm")
            return {"ok": True, "message": "Baseline set; monitoring armed"}

    def pause_monitoring(self) -> Dict:
        with self._engine_lock:
            if self._monitoring_mode != "ARMED":
                return {"ok": False, "message": "Monitoring is not armed"}
            self._monitoring_mode = "PAUSED"
            self.theft_detector.reset()
            self.state.update(monitoring=self._monitoring_snapshot(), security=self.theft_detector.snapshot())
            self.audit.log_operator_action(None, "pause_monitoring")
            return {"ok": True, "message": "Monitoring paused"}

    def resume_monitoring(self) -> Dict:
        with self._engine_lock:
            counter = self._inventory_counter()
            if self._monitoring_mode != "PAUSED" or counter is None:
                return {"ok": False, "message": "Monitoring is not paused"}
            if not counter.resume():
                return {
                    "ok": False,
                    "message": "Inventory differs from the baseline; reset the baseline before arming",
                }
            self.theft_detector.reset()
            self._monitoring_mode = "ARMED"
            self.state.update(monitoring=self._monitoring_snapshot(), security=self.theft_detector.snapshot())
            self.audit.log_operator_action(None, "resume_monitoring")
            return {"ok": True, "message": "Monitoring resumed"}

    def reset_baseline(self) -> Dict:
        with self._engine_lock:
            counter = self._inventory_counter()
            if counter is not None:
                counter.reset_baseline(clear_observations=True)
            self.theft_detector.reset()
            self._monitoring_mode = "DISARMED"
            self.state.update(monitoring=self._monitoring_snapshot(), security=self.theft_detector.snapshot())
            self.audit.log_operator_action(None, "reset_baseline")
            return {"ok": True, "message": "Baseline cleared; monitoring disarmed"}

    def set_shelf_region(self, region) -> Dict:
        normalized = self._validate_region(region)
        with self._engine_lock:
            self._shelf_region = normalized
            if self.tracker is not None and hasattr(self.tracker, "set_shelf_region"):
                self.tracker.set_shelf_region(normalized)
            self.theft_detector.set_shelf_region(normalized)
            self._save_runtime_settings()
            self.reset_baseline()
            self.audit.log_operator_action(None, "set_shelf_region")
        return {"ok": True, "message": "Monitored zone updated; set a new baseline"}

    def switch_to_replay(self, filename: str, payload: bytes, loop: bool = False) -> Dict:
        suffix = Path(filename).suffix.lower()
        if suffix not in {".mp4", ".avi", ".mov"}:
            return {"ok": False, "message": "Replay must be an MP4, AVI, or MOV file"}
        if not payload:
            return {"ok": False, "message": "Replay video is empty"}
        replay_dir = resolve_project_path("data/logs/replay_uploads")
        replay_dir.mkdir(parents=True, exist_ok=True)
        path = replay_dir / f"{uuid.uuid4().hex}{suffix}"
        path.write_bytes(payload)
        try:
            capture = VideoFileCapture(
                path, max_fps=float(self.config["camera"].get("fps", 10)), loop=loop
            )
            self._switch_capture(capture, "replay", Path(filename).name)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return {"ok": True, "message": f"Replay started: {Path(filename).name}"}

    def switch_to_live(self) -> Dict:
        capture = FrameCapture(**self.config["camera"])
        self._switch_capture(capture, "live", "Live camera")
        return {"ok": True, "message": "Live camera restored"}

    def _switch_capture(self, capture, source_type: str, label: str) -> None:
        capture.start()
        with self._capture_lock, self._engine_lock:
            previous = self.capture
            self.capture = capture
            previous.stop()
            self._source_type = source_type
            self._source_label = label
            self._source_generation += 1
            if self.tracker is not None and hasattr(self.tracker, "reset_tracking"):
                self.tracker.reset_tracking()
            self.incident_recorder.reset_buffer()
            self.performance.reset()
            self.theft_detector.reset()
            self.siren.stop()
            self._monitoring_mode = "DISARMED"
            self.state.update(
                source={"type": source_type, "label": label},
                monitoring=self._monitoring_snapshot(),
                security=self.theft_detector.snapshot(),
            )
            self.audit.log_operator_action(None, f"switch_source:{source_type}")

    def record_vlm_latency(self, latency_ms: float) -> None:
        if latency_ms:
            self.performance.record("vlm_ms", float(latency_ms))
            self.state.update(performance=self._performance_snapshot())

    def _performance_snapshot(self) -> Dict:
        result = self.performance.snapshot()
        result["video_buffer_mb"] = self.incident_recorder.buffer_megabytes()
        device = self.state.snapshot().get("device_info") or {}
        result["active_device"] = device.get("selected", "initializing")
        return result

    def _inventory_counter(self):
        return getattr(self.tracker, "inventory_counter", None) if self.tracker else None

    def _monitoring_snapshot(self, perception: Optional[Dict] = None) -> Dict:
        inventory = (perception or {}).get("inventory", {})
        if not inventory:
            counter = self._inventory_counter()
            inventory = counter.snapshot() if counter is not None else {}
        return {
            "mode": self._monitoring_mode,
            "baseline_ready": bool(inventory.get("baseline_ready")),
            "baseline_counts": deepcopy(inventory.get("baseline_counts", {})),
            "current_counts": deepcopy(inventory.get("stable_counts", {})),
            "missing_items": deepcopy(inventory.get("missing_items", {})),
            "extra_items": deepcopy(inventory.get("extra_items", {})),
            "shelf_region": deepcopy(self._shelf_region),
        }

    @staticmethod
    def _validate_region(region):
        if region is None:
            return None
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            raise ValueError("Shelf region must contain x1, y1, x2, y2")
        values = [float(value) for value in region]
        x1, y1, x2, y2 = values
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("Shelf region coordinates must be between 0 and 1")
        if x2 - x1 < 0.05 or y2 - y1 < 0.05:
            raise ValueError("Shelf region must cover at least 5% of frame width and height")
        return values

    def _load_runtime_settings(self) -> Dict:
        try:
            if self._settings_path.is_file():
                value = json.loads(self._settings_path.read_text(encoding="utf-8"))
                return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.warning("Could not load runtime settings: %s", exc)
        return {}

    def _save_runtime_settings(self) -> None:
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._settings_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"shelf_region": self._shelf_region}, indent=2), encoding="utf-8"
        )
        temporary.replace(self._settings_path)

    def snapshot(self) -> Dict:
        return self.state.snapshot()

    def stop(self) -> None:
        self._stop.set()
        with self._capture_lock:
            self.capture.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        if self.tracker is not None and hasattr(self.tracker, "shutdown"):
            self.tracker.shutdown()
        for result in self.incident_recorder.shutdown():
            if result.get("path"):
                self.audit.update_event(result["event_id"], video_path=result["path"])
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
