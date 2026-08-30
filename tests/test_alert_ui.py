from unittest import TestCase, mock

from src.module_c_ui_dashboard.alert_ui import render_alert


class AlertUiTests(TestCase):
    @mock.patch("src.module_c_ui_dashboard.alert_ui.st")
    def test_acknowledgement_stops_active_alert(self, streamlit):
        streamlit.button.return_value = True
        controller = mock.Mock()
        snapshot = {
            "security": {
                "last_event": {
                    "event_id": "event-1",
                    "timestamp": "2026-08-29T10:00:00",
                    "status": "confirmed",
                    "acknowledged": False,
                }
            }
        }

        render_alert(snapshot, controller)

        self.assertEqual(
            {
                "acknowledged": controller.acknowledge.call_args.args[0],
                "rerun": streamlit.rerun.called,
            },
            {"acknowledged": "event-1", "rerun": True},
        )

    @mock.patch("src.module_c_ui_dashboard.alert_ui.st")
    def test_confirmed_alert_applies_page_wide_red_theme(self, streamlit):
        streamlit.button.return_value = False
        snapshot = {
            "security": {
                "last_event": {
                    "event_id": "event-1",
                    "timestamp": "2026-08-29T10:00:00",
                    "status": "confirmed",
                    "acknowledged": False,
                    "removed_items": {"laptop": 1},
                }
            }
        }

        render_alert(snapshot, controller=object())

        rendered_markdown = "\n".join(
            call.args[0] for call in streamlit.markdown.call_args_list
        )
        self.assertIn("background-color", rendered_markdown)

    @mock.patch("src.module_c_ui_dashboard.alert_ui.st")
    def test_confirmed_alert_shows_telegram_delivery_status(self, streamlit):
        streamlit.button.return_value = False
        snapshot = {
            "security": {
                "last_event": {
                    "event_id": "event-1",
                    "timestamp": "2026-08-29T10:00:00",
                    "status": "confirmed",
                    "acknowledged": False,
                    "telegram_status": "sent",
                }
            }
        }

        render_alert(snapshot, controller=object())

        rendered_markdown = "\n".join(
            call.args[0] for call in streamlit.markdown.call_args_list
        )
        self.assertIn("Telegram: sent", rendered_markdown)

    @mock.patch("src.module_c_ui_dashboard.alert_ui.st")
    def test_confirmed_alert_shows_persisted_ai_incident_report(self, streamlit):
        streamlit.button.return_value = False
        snapshot = {
            "security": {
                "last_event": {
                    "event_id": "event-1",
                    "timestamp": "2026-08-30T10:00:00",
                    "status": "confirmed",
                    "acknowledged": False,
                    "summary_status": "completed",
                    "ai_summary": "One bottle was removed by an unknown person.",
                }
            }
        }

        render_alert(snapshot, controller=object())

        streamlit.info.assert_called_once_with(
            "AI incident report: One bottle was removed by an unknown person."
        )
