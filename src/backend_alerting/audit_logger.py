"""SQLite audit logging and incident snapshot persistence."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.utils.config_loader import resolve_project_path


class AuditLogger:
    def __init__(self, database_path: str, snapshot_directory: str) -> None:
        self.database_path = resolve_project_path(database_path)
        self.snapshot_directory = resolve_project_path(snapshot_directory)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
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
                );
                CREATE TABLE IF NOT EXISTS alert_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    attempted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    channel TEXT NOT NULL,
                    status TEXT NOT NULL,
                    detail TEXT,
                    FOREIGN KEY(event_id) REFERENCES events(event_id)
                );
                CREATE TABLE IF NOT EXISTS inventory_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    counts TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES events(event_id)
                );
                CREATE TABLE IF NOT EXISTS persons (
                    event_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    authorization_state TEXT NOT NULL,
                    PRIMARY KEY(event_id, track_id),
                    FOREIGN KEY(event_id) REFERENCES events(event_id)
                );
                CREATE TABLE IF NOT EXISTS operator_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT,
                    action TEXT NOT NULL,
                    performed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            existing = {
                row["name"] for row in connection.execute("PRAGMA table_info(events)").fetchall()
            }
            migrations = {
                "baseline_counts": "TEXT NOT NULL DEFAULT '{}'",
                "primary_actor": "TEXT",
                "actor_candidates": "TEXT NOT NULL DEFAULT '[]'",
                "decision_reason": "TEXT NOT NULL DEFAULT ''",
                "zone_region": "TEXT",
                "source_type": "TEXT NOT NULL DEFAULT 'live'",
                "video_path": "TEXT",
                "ai_summary": "TEXT",
                "summary_status": "TEXT NOT NULL DEFAULT 'not_requested'",
                "summary_error": "TEXT",
            }
            for column, definition in migrations.items():
                if column not in existing:
                    connection.execute(f"ALTER TABLE events ADD COLUMN {column} {definition}")

    def save_snapshot(self, event_id: str, frame: np.ndarray) -> str:
        path = self.snapshot_directory / f"{event_id}.jpg"
        if not cv2.imwrite(str(path), frame):
            raise IOError(f"Could not save incident snapshot: {path}")
        return str(path)

    def log_event(self, event: Dict, frame: Optional[np.ndarray] = None) -> Dict:
        record = deepcopy(event)
        if frame is not None and not record.get("snapshot_path"):
            record["snapshot_path"] = self.save_snapshot(record["event_id"], frame)
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO events (
                    event_id, event_type, status, timestamp, removed_items,
                    previous_counts, current_counts, person_track_ids,
                    authorization_states, snapshot_path, telegram_status,
                    acknowledged, baseline_counts, primary_actor, actor_candidates,
                    decision_reason, zone_region, source_type, video_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["event_id"], record["event_type"], record["status"],
                    record["timestamp"], json.dumps(record.get("removed_items", {})),
                    json.dumps(record.get("previous_counts", {})),
                    json.dumps(record.get("current_counts", {})),
                    json.dumps(record.get("person_track_ids", [])),
                    json.dumps(record.get("authorization_states", [])),
                    record.get("snapshot_path"), record.get("telegram_status", "disabled"),
                    int(record.get("acknowledged", False)),
                    json.dumps(record.get("baseline_counts", {})),
                    json.dumps(record.get("primary_actor")) if record.get("primary_actor") else None,
                    json.dumps(record.get("actor_candidates", [])),
                    record.get("decision_reason", ""),
                    json.dumps(record.get("zone_region")) if record.get("zone_region") else None,
                    record.get("source_type", "live"),
                    record.get("video_path"),
                ),
            )
            connection.execute(
                "INSERT INTO inventory_snapshots(event_id, timestamp, counts) VALUES (?, ?, ?)",
                (
                    record["event_id"], record["timestamp"],
                    json.dumps(record.get("current_counts", {})),
                ),
            )
            person_states = record.get("person_states", {})
            states = record.get("authorization_states", []) or ["not_visible"]
            for index, track_id in enumerate(record.get("person_track_ids", [])):
                state = person_states.get(str(track_id), states[min(index, len(states) - 1)])
                connection.execute(
                    "INSERT OR REPLACE INTO persons(event_id, track_id, authorization_state) VALUES (?, ?, ?)",
                    (record["event_id"], track_id, state),
                )
        return record

    def update_event(self, event_id: str, **values: object) -> None:
        allowed = {
            "telegram_status",
            "acknowledged",
            "snapshot_path",
            "video_path",
            "status",
            "ai_summary",
            "summary_status",
            "summary_error",
        }
        updates = {key: value for key, value in values.items() if key in allowed}
        if not updates:
            return
        columns = ", ".join(f"{key} = ?" for key in updates)
        parameters = [int(value) if key == "acknowledged" else value for key, value in updates.items()]
        with self._lock, self._connection() as connection:
            connection.execute(
                f"UPDATE events SET {columns} WHERE event_id = ?",
                [*parameters, event_id],
            )

    def update_ai_summary(
        self,
        event_id: str,
        *,
        status: str,
        summary: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Persist the lifecycle and result of an automatic incident summary."""
        self.update_event(
            event_id,
            summary_status=status,
            ai_summary=summary,
            summary_error=error,
        )

    def log_alert_attempt(self, event_id: str, channel: str, status: str, detail: str = "") -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO alert_attempts(event_id, channel, status, detail) VALUES (?, ?, ?, ?)",
                (event_id, channel, status, detail[:1000]),
            )

    def log_operator_action(self, event_id: str, action: str) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO operator_actions(event_id, action) VALUES (?, ?)",
                (event_id, action),
            )

    def recent_events(self, limit: int = 50) -> List[Dict]:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        json_fields = {
            "removed_items", "previous_counts", "current_counts",
            "person_track_ids", "authorization_states", "baseline_counts",
            "actor_candidates",
        }
        for row in rows:
            item = dict(row)
            for field in json_fields:
                item[field] = json.loads(item[field])
            item["primary_actor"] = (
                json.loads(item["primary_actor"]) if item.get("primary_actor") else None
            )
            item["zone_region"] = (
                json.loads(item["zone_region"]) if item.get("zone_region") else None
            )
            item["acknowledged"] = bool(item["acknowledged"])
            result.append(item)
        return result

    def cleanup_media(self, retention_days: int, replay_directory: Optional[str] = None) -> None:
        """Remove expired runtime media while retaining the event audit rows."""
        cutoff = time.time() - max(1, int(retention_days)) * 86400
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT event_id, snapshot_path, video_path FROM events"
            ).fetchall()
            for row in rows:
                updates = {}
                for column in ("snapshot_path", "video_path"):
                    value = row[column]
                    if not value:
                        continue
                    path = Path(value)
                    expired = path.exists() and path.stat().st_mtime < cutoff
                    if expired:
                        try:
                            path.unlink()
                        except OSError:
                            continue
                    if expired or not path.exists():
                        updates[column] = None
                if updates:
                    columns = ", ".join(f"{key} = ?" for key in updates)
                    connection.execute(
                        f"UPDATE events SET {columns} WHERE event_id = ?",
                        [*updates.values(), row["event_id"]],
                    )
        if replay_directory:
            directory = resolve_project_path(replay_directory)
            if directory.is_dir():
                for path in directory.iterdir():
                    try:
                        if path.is_file() and path.stat().st_mtime < cutoff:
                            path.unlink()
                    except OSError:
                        continue
