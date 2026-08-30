"""Telegram photo alert client with bounded retries."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import requests


class TelegramAlerter:
    def __init__(
        self,
        enabled: bool = False,
        timeout_seconds: float = 10,
        max_attempts: int = 2,
        notify_event_types: Optional[Iterable[str]] = None,
    ):
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, max_attempts)
        self.notify_event_types = set(
            notify_event_types or ("authorized_removal", "suspected_theft")
        )
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

    def should_notify(self, event: Dict) -> bool:
        return self.enabled and event.get("event_type") in self.notify_event_types

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def send_event(self, event: Dict) -> Dict:
        if not self.enabled:
            return {"status": "disabled", "detail": "Telegram is disabled"}
        if not self.configured:
            return {"status": "failed", "detail": "Telegram credentials are missing"}
        removed = ", ".join(
            f"{name}: {count}" for name, count in event.get("removed_items", {}).items()
        ) or "unknown"
        event_type = event.get("event_type")
        title = (
            "Authorized inventory removal"
            if event_type == "authorized_removal"
            else "Suspected theft"
        )
        actor = event.get("primary_actor") or {}
        actor_name = actor.get("name", "Unknown")
        caption = (
            f"{title}\nTime: {event['timestamp']}\nRemoved: {removed}\n"
            f"Actor: {actor_name}\nEvent: {event['event_id']}"
        )
        url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
        snapshot = event.get("snapshot_path")
        if not snapshot or not Path(snapshot).is_file():
            return {"status": "failed", "detail": "Incident snapshot is missing"}
        last_error = ""
        for attempt in range(self.max_attempts):
            try:
                with Path(snapshot).open("rb") as photo:
                    response = requests.post(
                        url,
                        data={"chat_id": self.chat_id, "caption": caption},
                        files={"photo": photo},
                        timeout=self.timeout_seconds,
                    )
                response.raise_for_status()
                payload = response.json()
                if payload.get("ok"):
                    return {"status": "sent", "detail": "Telegram alert sent"}
                last_error = str(payload)
            except Exception as exc:
                last_error = str(exc)
            if attempt + 1 < self.max_attempts:
                time.sleep(min(2 ** attempt, 2))
        return {"status": "failed", "detail": last_error}
