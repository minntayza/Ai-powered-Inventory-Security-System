import tempfile
import sqlite3
import os
import time
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
            "baseline_counts": {"bottle": 2},
            "primary_actor": {"track_id": 1, "name": "Unknown"},
            "actor_candidates": [{"track_id": 1, "association_score": 0.8}],
            "decision_reason": "test",
            "zone_region": [0, 0, 1, 1],
            "source_type": "replay",
            "video_path": None,
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_event_and_snapshot_are_persistent(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        record = self.logger.log_event(self.event, frame)
        self.assertTrue(Path(record["snapshot_path"]).is_file())
        stored = self.logger.recent_events(1)[0]
        self.assertEqual(stored["removed_items"], {"bottle": 1})
        self.assertEqual(stored["primary_actor"]["track_id"], 1)
        self.assertEqual(stored["source_type"], "replay")

    def test_ai_summary_update_is_visible_in_recent_events(self):
        self.logger.log_event(self.event)

        self.logger.update_ai_summary(
            "event-1",
            status="completed",
            summary="One bottle was removed by an unknown person.",
        )

        stored = self.logger.recent_events(1)[0]
        self.assertEqual(stored["summary_status"], "completed")
        self.assertEqual(
            stored["ai_summary"],
            "One bottle was removed by an unknown person.",
        )
        self.assertIsNone(stored["summary_error"])

    def test_updates_delivery_and_acknowledgement(self):
        self.logger.log_event(self.event)
        self.logger.update_event("event-1", telegram_status="sent", acknowledged=True)
        stored = self.logger.recent_events(1)[0]
        self.assertEqual(stored["telegram_status"], "sent")
        self.assertTrue(stored["acknowledged"])

    def test_legacy_database_is_migrated_without_losing_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "legacy.db"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    """
                    CREATE TABLE events (
                        event_id TEXT PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        removed_items TEXT NOT NULL,
                        previous_counts TEXT NOT NULL,
                        current_counts TEXT NOT NULL,
                        person_track_ids TEXT NOT NULL,
                        authorization_states TEXT NOT NULL,
                        snapshot_path TEXT,
                        telegram_status TEXT NOT NULL,
                        acknowledged INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "legacy-1",
                        "suspected_theft",
                        "confirmed",
                        "2026-08-01T10:00:00",
                        '{"bottle": 1}',
                        '{"bottle": 2}',
                        '{"bottle": 1}',
                        "[1]",
                        '["unknown"]',
                        None,
                        "sent",
                        0,
                    ),
                )

            logger = AuditLogger(str(database), str(root / "snapshots"))
            stored = logger.recent_events(1)[0]

        self.assertEqual(
            {
                "event_id": stored["event_id"],
                "baseline_counts": stored["baseline_counts"],
                "source_type": stored["source_type"],
            },
            {
                "event_id": "legacy-1",
                "baseline_counts": {},
                "source_type": "live",
            },
        )

    def test_cleanup_removes_expired_media_but_retains_event(self):
        root = Path(self.temp.name)
        snapshot = root / "old.jpg"
        video = root / "old.mp4"
        snapshot.write_bytes(b"image")
        video.write_bytes(b"video")
        old_time = time.time() - 3 * 86400
        os.utime(snapshot, (old_time, old_time))
        os.utime(video, (old_time, old_time))
        event = dict(self.event, snapshot_path=str(snapshot), video_path=str(video))
        self.logger.log_event(event)

        self.logger.cleanup_media(retention_days=1)
        stored = self.logger.recent_events(1)[0]

        self.assertEqual(
            {
                "snapshot_exists": snapshot.exists(),
                "video_exists": video.exists(),
                "event_id": stored["event_id"],
                "snapshot_path": stored["snapshot_path"],
                "video_path": stored["video_path"],
            },
            {
                "snapshot_exists": False,
                "video_exists": False,
                "event_id": "event-1",
                "snapshot_path": None,
                "video_path": None,
            },
        )
