import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np

from src.backend_alerting.audit_logger import AuditLogger


class AuditLoggerTests(TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.logger = AuditLogger(str(root / "events.db"), str(root / "snapshots"))
        self.event = {
            "event_id": "event-1",
            "event_type": "suspected_theft",
            "status": "confirmed",
            "timestamp": "2026-08-08T12:00:00",
            "removed_items": {"bottle": 1},
            "previous_counts": {"bottle": 2},
            "current_counts": {"bottle": 1},
            "person_track_ids": [1],
            "authorization_states": ["unknown"],
            "snapshot_path": None,
            "telegram_status": "pending",
            "acknowledged": False,
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_event_and_snapshot_are_persistent(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        record = self.logger.log_event(self.event, frame)
        self.assertTrue(Path(record["snapshot_path"]).is_file())
        stored = self.logger.recent_events(1)[0]
        self.assertEqual(stored["removed_items"], {"bottle": 1})

    def test_updates_delivery_and_acknowledgement(self):
        self.logger.log_event(self.event)
        self.logger.update_event("event-1", telegram_status="sent", acknowledged=True)
        stored = self.logger.recent_events(1)[0]
        self.assertEqual(stored["telegram_status"], "sent")
        self.assertTrue(stored["acknowledged"])

