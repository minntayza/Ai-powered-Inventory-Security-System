"""Telegram photo alert client with bounded retries."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict

import requests


class TelegramAlerter:
    def __init__(self, enabled: bool = False, timeout_seconds: float = 10, max_attempts: int = 2):
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, max_attempts)
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

    def send_event(self, event: Dict) -> Dict:
        if not self.enabled:
            return {"status": "disabled", "detail": "Telegram is disabled"}
        if not self.token or not self.chat_id:
            return {"status": "failed", "detail": "Telegram credentials are missing"}
        removed = ", ".join(
            f"{name}: {count}" for name, count in event.get("removed_items", {}).items()
        ) or "unknown"
        caption = (
            f"Suspected theft\nTime: {event['timestamp']}\n"
            f"Removed: {removed}\nEvent: {event['event_id']}"
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

