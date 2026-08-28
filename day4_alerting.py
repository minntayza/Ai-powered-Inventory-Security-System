import csv
import os
import requests
from datetime import datetime

class AlertingAndLoggingManager:
    def __init__(self, bot_token: str, chat_id: str, csv_filename: str = "audit_log.csv"):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.csv_filename = csv_filename
        self._init_csv()

    def _init_csv(self):
        """Creates CSV log file with header structure if it doesn't exist."""
        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Event Type", "Inventory Count", 
                    "Person Detected", "ArUco Detected", "Details", "Image Path"
                ])

    def log_event_to_csv(self, event_type: str, inventory: int, person_present: bool, aruco_present: bool, details: str, img_path: str = "N/A"):
        """Logs security events to the local CSV audit trail."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.csv_filename, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, event_type, inventory, person_present, aruco_present, details, img_path])

    def send_telegram_breach_alert(self, text_message: str, image_path: str = None) -> bool:
        """Sends markdown text alert and evidence photo payload via Telegram Bot API."""
        base_url = f"https://api.telegram.org/bot{self.bot_token}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            # 1. Send Text Notification
            text_payload = {
                "chat_id": self.chat_id,
                "text": f"🚨 **WAREHOUSE SECURITY BREACH** 🚨\n\n**Details**: {text_message}\n**Timestamp**: `{timestamp}`",
                "parse_mode": "Markdown"
            }
            requests.post(f"{base_url}/sendMessage", json=text_payload, timeout=5)

            # 2. Send Evidence Photo Payload if present
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as photo_file:
                    files = {"photo": photo_file}
                    data = {
                        "chat_id": self.chat_id,
                        "caption": f"📸 **Evidence Capture** - {timestamp}"
                    }
                    requests.post(f"{base_url}/sendPhoto", data=data, files=files, timeout=10)
            return True
        except Exception as e:
            print(f"Error dispatching Telegram payload: {e}")
            return False