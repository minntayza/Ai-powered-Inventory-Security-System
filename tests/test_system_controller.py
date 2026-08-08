import tempfile
import time
from pathlib import Path
from unittest import TestCase

import numpy as np
import yaml

from src.backend_alerting.event_loop import SystemController


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


class FakeTracker:
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


class SystemControllerTests(TestCase):
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
            deadline = time.time() + 2
            while time.time() < deadline and not controller.recent_events(1):
                time.sleep(0.01)
            controller.stop()
            events = controller.recent_events(1)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["event_type"], "suspected_theft")
            self.assertTrue(Path(events[0]["snapshot_path"]).is_file())

