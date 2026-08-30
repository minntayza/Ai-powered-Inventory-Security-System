import tempfile
from pathlib import Path
from unittest import TestCase, mock

from src.backend_alerting.telegram_bot import TelegramAlerter


class TelegramAlerterTests(TestCase):
    @mock.patch.dict(
        "os.environ",
        {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""},
    )
    def test_authorized_and_suspected_events_are_selected(self):
        alerter = TelegramAlerter(
            enabled=True,
            notify_event_types=["authorized_removal", "suspected_theft"],
        )

        self.assertTrue(alerter.should_notify({"event_type": "authorized_removal"}))
        self.assertTrue(alerter.should_notify({"event_type": "suspected_theft"}))
        self.assertFalse(alerter.should_notify({"event_type": "inventory_recovered"}))
        self.assertFalse(alerter.configured)

    @mock.patch.dict(
        "os.environ",
        {"TELEGRAM_BOT_TOKEN": "test-token", "TELEGRAM_CHAT_ID": "test-chat"},
    )
    @mock.patch("src.backend_alerting.telegram_bot.requests.post")
    def test_authorized_removal_message_has_actor_and_inventory(self, post):
        response = mock.Mock()
        response.json.return_value = {"ok": True}
        post.return_value = response
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "event.jpg"
            snapshot.write_bytes(b"image")
            alerter = TelegramAlerter(enabled=True, max_attempts=1)
            self.assertTrue(alerter.configured)

            result = alerter.send_event(
                {
                    "event_id": "event-1",
                    "event_type": "authorized_removal",
                    "timestamp": "2026-08-10T20:30:00",
                    "removed_items": {"bottle": 1},
                    "primary_actor": {"name": "Alice"},
                    "snapshot_path": str(snapshot),
                }
            )

        self.assertEqual(result["status"], "sent")
        caption = post.call_args.kwargs["data"]["caption"]
        self.assertIn("Authorized inventory removal", caption)
        self.assertIn("bottle: 1", caption)
        self.assertIn("Actor: Alice", caption)

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
