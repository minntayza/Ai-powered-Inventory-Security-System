import tempfile
from pathlib import Path
from unittest import TestCase, mock

from src.backend_alerting.telegram_bot import TelegramAlerter


class TelegramAlerterTests(TestCase):
    @mock.patch("src.backend_alerting.telegram_bot.time.sleep")
    @mock.patch("src.backend_alerting.telegram_bot.requests.post")
    def test_transient_failure_is_retried(self, post, _sleep):
        response = mock.Mock()
        response.json.return_value = {"ok": True}
        post.side_effect = [RuntimeError("network down"), response]
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "event.jpg"
            snapshot.write_bytes(b"jpeg")
            alerter = TelegramAlerter(enabled=True, max_attempts=2)
            alerter.token = "token"
            alerter.chat_id = "chat"

            result = alerter.send_event(
                {
                    "event_id": "event-1",
                    "timestamp": "2026-08-29T10:00:00",
                    "removed_items": {"laptop": 1},
                    "snapshot_path": str(snapshot),
                }
            )

        self.assertEqual(
            {"result": result, "attempts": post.call_count},
            {
                "result": {"status": "sent", "detail": "Telegram alert sent"},
                "attempts": 2,
            },
        )
