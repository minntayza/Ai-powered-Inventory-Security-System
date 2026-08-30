import tempfile
import threading
import time
from pathlib import Path
from unittest import TestCase, mock

import numpy as np
import yaml
import cv2

from src.backend_alerting.event_loop import SystemController
from src.module_a_perception_engine.interaction_associator import InteractionAssociator
from src.module_a_perception_engine.inventory_counter import InventoryCounter
from src.module_a_perception_engine.person_tracker import PersonTracker


class FakeCapture:
    def __init__(self):
        self.frame_id = 0
        self.stopped = False

    def start(self):
        return self

    def read(self):
        if self.stopped:
            return False, None, self.frame_id
        self.frame_id += 1
        return True, np.zeros((20, 20, 3), dtype=np.uint8), self.frame_id

    def stop(self):
        self.stopped = True


class ControlledCapture:
    """Publish one deterministic frame at a time to the controller thread."""

    def __init__(self):
        self.frame = None
        self.frame_id = 0
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        return self

    def publish(self, marker):
        with self.lock:
            self.frame_id += 1
            self.frame = np.full((100, 100, 3), marker, dtype=np.uint8)
            return self.frame_id

    def read(self):
        with self.lock:
            if self.stopped or self.frame is None:
                return False, None, self.frame_id
            return True, self.frame.copy(), self.frame_id

    def stop(self):
        with self.lock:
            self.stopped = True


class RemovedInventoryDetector:
    """Return an item/person scene selected by the frame's pixel marker."""

    def detect(self, frame):
        marker = int(frame[0, 0, 0])
        item = {
            "label": "bottle",
            "bbox": [40, 40, 60, 60],
            "confidence": 0.99,
            "track_id": 9,
        }
        person = {
            "bbox": [20, 10, 80, 95],
            "confidence": 0.99,
            "track_id": 7,
        }
        return {
            "persons": [person] if marker == 1 else [],
            "inventory": [item] if marker in (1, 2) else [],
        }


class NoopFaceDetector:
    def detect_faces(self, _frame):
        return []


class UnknownPersonTracker(PersonTracker):
    """Use the real tracking/counting pipeline with deterministic identity data."""

    def _completed_face_results(self):
        return [{
            "bbox": [35, 15, 30, 30],
            "name": "Unknown",
            "authorized": False,
            "authorization_state": "unknown",
            "confidence": 0.0,
        }]


class FakeTracker:
    class Counter:
        def __init__(self):
            self.armed = False

        def snapshot(self):
            return {
                "baseline_ready": True,
                "baseline_counts": {"bottle": 2} if self.armed else {},
                "stable_counts": {"bottle": 2},
                "missing_items": {},
                "extra_items": {},
            }

        def set_baseline(self):
            self.armed = True
            return True

        def reset_baseline(self, clear_observations=True):
            self.armed = False

        def disarm(self):
            self.armed = False

        def resume(self):
            self.armed = True
            return True

    def __init__(self):
        self.inventory_counter = self.Counter()

    def process_frame(self, _frame):
        drop = {
            "removed_items": {"bottle": 1},
            "previous_counts": {"bottle": 2},
            "current_counts": {"bottle": 1},
        }
        return {
            "timestamp": "2026-08-08T12:00:00",
            "persons": [{
                "track_id": 1,
                "id": 1,
                "bbox": [0, 0, 10, 20],
                "name": "Unknown",
                "authorized": False,
                "authorization_state": "unknown",
                "face_confidence": 0.0,
            }],
            "inventory": {
                "stable_counts": {"bottle": 1},
                "total_stable_count": 1,
                "drop": drop,
                "change": drop,
                "items": [],
            },
            "frame_size": [20, 20],
        }

    def draw_tracking(self, frame, _output):
        return frame.copy()

class FlakyTracker(FakeTracker):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def process_frame(self, frame):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary inference failure")
        return super().process_frame(frame)


class ResettableTracker(FakeTracker):
    def __init__(self):
        super().__init__()
        self.reset_count = 0

    def reset_tracking(self):
        self.reset_count += 1


class DegradedFaceTracker(FakeTracker):
    class Face:
        last_inference_ms = 0.0

        @staticmethod
        def health():
            return {"healthy": False, "last_error": "face backend failed"}

    def __init__(self):
        super().__init__()
        self.face_detector = self.Face()

class AuthorizedThenUnknownTracker(FakeTracker):
    """Simulate authorization expiring while the same baseline stays missing."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def process_frame(self, frame):
        output = super().process_frame(frame)
        self.calls += 1
        if self.calls == 1:
            output["persons"][0].update({
                "name": "Alice",
                "authorized": True,
                "authorization_state": "authorized",
            })
        return output


class PreviewTracker(FakeTracker):
    def draw_tracking(self, frame, _output):
        annotated = frame.copy()
        annotated[0, 0, 0] = 99
        return annotated

class SystemControllerTests(TestCase):
    def test_snapshot_exposes_degraded_face_recognition_health(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config"
            config.mkdir()
            documents = {
                "camera_config.yaml": {"source": 0, "width": 20, "height": 20, "fps": 1},
                "model_config.yaml": {"yolo": {}, "vlm": {}},
                "thresholds.yaml": {"security": {}},
                "alert_config.yaml": {
                    "siren": {"enabled": False, "audio_path": str(root / "missing.mp3")},
                    "telegram": {"enabled": False},
                    "storage": {
                        "database_path": str(root / "events.db"),
                        "snapshot_directory": str(root / "snapshots"),
                    },
                },
            }
            for name, content in documents.items():
                with (config / name).open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(content, handle)
            controller = SystemController(
                config_dir=str(config),
                tracker=DegradedFaceTracker(),
                capture=FakeCapture(),
            ).start()
            deadline = time.time() + 2
            snapshot = controller.snapshot()
            while time.time() < deadline and "component_health" not in snapshot:
                time.sleep(0.01)
                snapshot = controller.snapshot()
            controller.stop()

        self.assertEqual(
            snapshot["component_health"]["face"],
            {"healthy": False, "last_error": "face backend failed"},
        )

    def test_replay_incident_suppresses_external_alerts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config"
            config.mkdir()
            documents = {
                "camera_config.yaml": {"source": 0, "width": 20, "height": 20, "fps": 5},
                "model_config.yaml": {"yolo": {}, "vlm": {}},
                "thresholds.yaml": {
                    "security": {
                        "grace_period_seconds": 0,
                        "alert_cooldown_seconds": 30,
                    }
                },
                "alert_config.yaml": {
                    "siren": {"enabled": False, "audio_path": str(root / "missing.mp3")},
                    "telegram": {"enabled": True},
                    "storage": {
                        "database_path": str(root / "events.db"),
                        "snapshot_directory": str(root / "snapshots"),
                    },
                },
            }
            for name, content in documents.items():
                with (config / name).open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(content, handle)
            video_path = root / "demo.avi"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (20, 20)
            )
            for _ in range(10):
                writer.write(np.zeros((20, 20, 3), dtype=np.uint8))
            writer.release()
            controller = SystemController(
                config_dir=str(config),
                tracker=ResettableTracker(),
                capture=FakeCapture(),
            )
            siren = mock.Mock()
            telegram = mock.Mock()
            controller.siren = siren
            controller.telegram = telegram
            controller.switch_to_replay(
                "demo.avi", video_path.read_bytes(), loop=True
            )
            replay_path = Path(controller.capture.path)
            controller.set_baseline_and_arm()
            controller.start()
            deadline = time.time() + 5
            while time.time() < deadline and not controller.recent_events(1):
                time.sleep(0.01)
            events = controller.recent_events(1)
            controller.stop()
            replay_path.unlink(missing_ok=True)

        self.assertEqual(
            {
                "source_type": events[0]["source_type"],
                "telegram_status": events[0]["telegram_status"],
                "siren_started": siren.start.called,
                "telegram_called": telegram.send_event.called,
            },
            {
                "source_type": "replay",
                "telegram_status": "disabled",
                "siren_started": False,
                "telegram_called": False,
            },
        )

    def test_replay_switch_resets_tracking_and_disarms_monitoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config"
            config.mkdir()
            documents = {
                "camera_config.yaml": {"source": 0, "width": 20, "height": 20, "fps": 5},
                "model_config.yaml": {"yolo": {}, "vlm": {}},
                "thresholds.yaml": {"security": {}},
                "alert_config.yaml": {
                    "siren": {"enabled": False, "audio_path": str(root / "missing.mp3")},
                    "telegram": {"enabled": False},
                    "storage": {
                        "database_path": str(root / "events.db"),
                        "snapshot_directory": str(root / "snapshots"),
                    },
                },
            }
            for name, content in documents.items():
                with (config / name).open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(content, handle)
            video_path = root / "demo.avi"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (20, 20)
            )
            writer.write(np.zeros((20, 20, 3), dtype=np.uint8))
            writer.release()
            tracker = ResettableTracker()
            controller = SystemController(
                config_dir=str(config), tracker=tracker, capture=FakeCapture()
            )
            controller.set_baseline_and_arm()
            controller.performance.record("yolo_ms", 42)
            controller.performance.skip(3)

            result = controller.switch_to_replay("demo.avi", video_path.read_bytes())
            snapshot = controller.snapshot()
            performance = controller.performance.snapshot()
            replay_path = Path(controller.capture.path)
            controller.stop()
            replay_path.unlink(missing_ok=True)

            self.assertEqual(
                {
                    "ok": result["ok"],
                    "source": snapshot["source"]["type"],
                    "mode": snapshot["monitoring"]["mode"],
                    "tracker_resets": tracker.reset_count,
                    "performance": performance,
                },
                {
                    "ok": True,
                    "source": "replay",
                    "mode": "DISARMED",
                    "tracker_resets": 1,
                    "performance": {
                        "capture_fps": 0.0,
                        "processed_fps": 0.0,
                        "skipped_frames": 0,
                    },
                },
            )

    def test_controller_continues_after_one_frame_processing_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config"
            config.mkdir()
            documents = {
                "camera_config.yaml": {"source": 0, "width": 20, "height": 20, "fps": 1},
                "model_config.yaml": {"yolo": {}, "vlm": {}},
                "thresholds.yaml": {"security": {}},
                "alert_config.yaml": {
                    "siren": {"enabled": False, "audio_path": str(root / "missing.mp3")},
                    "telegram": {"enabled": False},
                    "storage": {
                        "database_path": str(root / "events.db"),
                        "snapshot_directory": str(root / "snapshots"),
                    },
                },
            }
            for name, content in documents.items():
                with (config / name).open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(content, handle)
            tracker = FlakyTracker()
            controller = SystemController(
                config_dir=str(config), tracker=tracker, capture=FakeCapture()
            ).start()
            deadline = time.time() + 2
            while time.time() < deadline and tracker.calls < 2:
                time.sleep(0.01)

            running_after_recovery = controller.snapshot()["running"]
            controller.stop()

            self.assertEqual(
                {"calls": tracker.calls >= 2, "running": running_after_recovery},
                {"calls": True, "running": True},
            )

    def test_zone_validation_rejects_tiny_or_reversed_regions(self):
        with self.assertRaises(ValueError):
            SystemController._validate_region([0.5, 0.5, 0.4, 0.6])
        with self.assertRaises(ValueError):
            SystemController._validate_region([0.1, 0.1, 0.12, 0.9])
        self.assertEqual(
            SystemController._validate_region([0.1, 0.2, 0.9, 0.8]),
            [0.1, 0.2, 0.9, 0.8],
        )

    def test_confirmed_event_is_automatically_summarized_in_sqlite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config"
            config.mkdir()
            documents = {
                "camera_config.yaml": {"source": 0, "width": 20, "height": 20, "fps": 1},
                "model_config.yaml": {"yolo": {}, "vlm": {}},
                "thresholds.yaml": {
                    "security": {"grace_period_seconds": 0, "alert_cooldown_seconds": 30}
                },
                "alert_config.yaml": {
                    "siren": {"enabled": False, "audio_path": str(root / "missing.mp3")},
                    "telegram": {"enabled": False},
                    "storage": {
                        "database_path": str(root / "events.db"),
                        "snapshot_directory": str(root / "snapshots"),
                    },
                },
            }
            for name, content in documents.items():
                with (config / name).open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(content, handle)
            def summarize(_frame, event):
                return {
                    "ok": True,
                    "answer": f"Automatic report for {event['event_id']}",
                    "latency_ms": 4.0,
                }

            controller = SystemController(
                config_dir=str(config),
                tracker=FakeTracker(),
                capture=FakeCapture(),
            )
            self.assertIs(controller.set_incident_summarizer(summarize), controller)
            controller.start()
            self.assertTrue(controller.set_baseline_and_arm()["ok"])
            # A cold PyTorch import can take several seconds on CPU-only CI.
            deadline = time.time() + 10
            events = []
            while time.time() < deadline:
                events = controller.recent_events(1)
                if events and events[0].get("summary_status") == "completed":
                    break
                time.sleep(0.01)
            live_event = controller.snapshot()["security"]["last_event"]
            controller.stop()
            events = controller.recent_events(1)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["event_type"], "suspected_theft")
            self.assertTrue(Path(events[0]["snapshot_path"]).is_file())
            self.assertEqual(events[0]["summary_status"], "completed")
            self.assertEqual(
                events[0]["ai_summary"],
                f"Automatic report for {events[0]['event_id']}",
            )
            self.assertEqual(live_event["summary_status"], "completed")
            self.assertEqual(live_event["ai_summary"], events[0]["ai_summary"])

    def test_controller_starts_disarmed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config"
            config.mkdir()
            documents = {
                "camera_config.yaml": {"source": 0, "width": 20, "height": 20, "fps": 1},
                "model_config.yaml": {"yolo": {}, "vlm": {}},
                "thresholds.yaml": {"security": {}},
                "alert_config.yaml": {
                    "siren": {"enabled": False, "audio_path": str(root / "missing.mp3")},
                    "telegram": {"enabled": False},
                    "storage": {
                        "database_path": str(root / "events.db"),
                        "snapshot_directory": str(root / "snapshots"),
                    },
                },
            }
            for name, content in documents.items():
                with (config / name).open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(content, handle)
            controller = SystemController(
                config_dir=str(config), tracker=FakeTracker(), capture=FakeCapture()
            )
            self.assertEqual(controller.snapshot()["monitoring"]["mode"], "DISARMED")
            controller.stop()

    @staticmethod
    def _wait_for(controller, predicate, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            snapshot = controller.snapshot()
            if predicate(snapshot):
                return snapshot
            time.sleep(0.01)
        raise AssertionError("controller did not reach the expected state")

    @staticmethod
    def _write_config(root, security=None):
        config = root / "config"
        config.mkdir()
        documents = {
            "camera_config.yaml": {
                "source": 0,
                "width": 100,
                "height": 100,
                "fps": 10,
            },
            "model_config.yaml": {"yolo": {}, "vlm": {}},
            "thresholds.yaml": {"security": security or {}},
            "alert_config.yaml": {
                "siren": {
                    "enabled": False,
                    "audio_path": str(root / "missing.mp3"),
                },
                "telegram": {"enabled": False},
                "storage": {
                    "database_path": str(root / "events.db"),
                    "snapshot_directory": str(root / "snapshots"),
                },
            },
        }
        for name, content in documents.items():
            with (config / name).open("w", encoding="utf-8") as handle:
                yaml.safe_dump(content, handle)
        return config

    def test_removed_inventory_keeps_departed_unknown_person_as_primary_actor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_config(
                root,
                {"grace_period_seconds": 0, "alert_cooldown_seconds": 30},
            )
            capture = ControlledCapture()
            tracker = UnknownPersonTracker(
                yolo_detector=RemovedInventoryDetector(),
                face_detector=NoopFaceDetector(),
                inventory_counter=InventoryCounter(
                    window_size=3,
                    confirmation_frames=2,
                    warmup_frames=1,
                    require_manual_baseline=True,
                ),
                contextual_counter=InventoryCounter(
                    window_size=1,
                    confirmation_frames=1,
                    warmup_frames=1,
                ),
                protected_classes={"bottle"},
                interaction_associator=InteractionAssociator(
                    lookback_seconds=10,
                    minimum_score=0.35,
                ),
            )
            controller = SystemController(
                config_dir=str(config), tracker=tracker, capture=capture
            ).start()
            try:
                first_frame = capture.publish(1)
                self._wait_for(
                    controller,
                    lambda state: (state.get("perception") or {}).get("frame_id")
                    == first_frame,
                )
                self.assertTrue(controller.set_baseline_and_arm()["ok"])

                # The unknown actor handles the protected bottle, then exits.
                handled_frame = capture.publish(1)
                self._wait_for(
                    controller,
                    lambda state: (state.get("perception") or {}).get("frame_id")
                    == handled_frame,
                )
                for marker in (3, 3, 3):
                    frame_id = capture.publish(marker)
                    snapshot = self._wait_for(
                        controller,
                        lambda state, expected=frame_id: (
                            state.get("perception") or {}
                        ).get("frame_id")
                        == expected,
                    )

                # Stable counting confirms the drop only after the actor is gone.
                self.assertEqual(snapshot["perception"]["persons"], [])
                self.assertEqual(snapshot["security"]["state"], "PENDING")
                final_frame = capture.publish(3)
                self._wait_for(
                    controller,
                    lambda state: (
                        (state.get("perception") or {}).get("frame_id") == final_frame
                        and state.get("security", {}).get("state") == "CONFIRMED"
                    ),
                )
                self.assertEqual(
                    controller.snapshot()["monitoring"]["mode"], "PAUSED"
                )
            finally:
                controller.stop()

            events = controller.recent_events(1)
            self.assertEqual(len(events), 1)
            event = events[0]
            self.assertEqual(event["event_type"], "suspected_theft")
            self.assertEqual(event["removed_items"], {"bottle": 1})
            self.assertEqual(event["primary_actor"]["track_id"], 7)
            self.assertEqual(event["primary_actor"]["name"], "Unknown")

    def test_completed_removal_pauses_before_identity_expiry_can_create_theft(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_config(
                root,
                {"grace_period_seconds": 0, "alert_cooldown_seconds": 0},
            )
            controller = SystemController(
                config_dir=str(config),
                tracker=AuthorizedThenUnknownTracker(),
                capture=FakeCapture(),
            )
            self.assertTrue(controller.set_baseline_and_arm()["ok"])
            controller.start()
            try:
                self._wait_for(controller, lambda _state: bool(controller.recent_events(1)))
                time.sleep(0.1)
                snapshot = controller.snapshot()
                events = controller.recent_events(10)
            finally:
                controller.stop()

            self.assertEqual([event["event_type"] for event in events], [
                "authorized_removal"
            ])
            self.assertEqual(snapshot["monitoring"]["mode"], "PAUSED")
            self.assertEqual(
                snapshot["monitoring"]["pause_reason"], "event_recorded"
            )

    def test_camera_preview_reads_newest_raw_frame_without_waiting_for_tracker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_config(root)
            capture = ControlledCapture()
            controller = SystemController(
                config_dir=str(config), tracker=PreviewTracker(), capture=capture
            )
            try:
                expected_frame_id = capture.publish(8)
                controller.state.update(perception={"persons": [], "inventory": {}})
                preview = controller.camera_preview()
            finally:
                controller.stop()

            self.assertTrue(preview["camera_healthy"])
            self.assertTrue(preview["raw_preview"])
            self.assertTrue(preview["has_detection_overlay"])
            self.assertEqual(preview["frame_id"], expected_frame_id)
            self.assertEqual(int(preview["frame"][0, 0, 0]), 99)
            self.assertEqual(int(preview["frame"][0, 1, 0]), 8)

    def test_confirmed_event_flows_to_sqlite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config"
            config.mkdir()
            documents = {
                "camera_config.yaml": {"source": 0, "width": 20, "height": 20, "fps": 1},
                "model_config.yaml": {"yolo": {}, "vlm": {}},
                "thresholds.yaml": {
                    "security": {"grace_period_seconds": 0, "alert_cooldown_seconds": 30}
                },
                "alert_config.yaml": {
                    "siren": {"enabled": False, "audio_path": str(root / "missing.mp3")},
                    "telegram": {"enabled": False},
                    "storage": {
                        "database_path": str(root / "events.db"),
                        "snapshot_directory": str(root / "snapshots"),
                    },
                },
            }
            for name, content in documents.items():
                with (config / name).open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(content, handle)
            controller = SystemController(
                config_dir=str(config), tracker=FakeTracker(), capture=FakeCapture()
            ).start()
            self.assertTrue(controller.set_baseline_and_arm()["ok"])
            # A cold PyTorch import can take several seconds on CPU-only CI.
            deadline = time.time() + 10
            while time.time() < deadline and not controller.recent_events(1):
                time.sleep(0.01)
            controller.stop()
            events = controller.recent_events(1)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["event_type"], "suspected_theft")
            self.assertTrue(Path(events[0]["snapshot_path"]).is_file())
