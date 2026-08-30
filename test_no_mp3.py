import os
import time
from native_theft_engine import CoreTheftEngineNoPygame
from day4_alerting import AlertingAndLoggingManager

# Configuration is loaded at runtime so credentials never enter source control.
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

def run_test():
    if not BOT_TOKEN or not CHAT_ID:
        raise RuntimeError(
            "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID before running this script."
        )

    engine = CoreTheftEngineNoPygame(siren_sound_path="siren.mp3")
    manager = AlertingAndLoggingManager(bot_token=BOT_TOKEN, chat_id=CHAT_ID)

    print("\n--- 1. Setting Baseline (10 items) ---")
    engine.evaluate_frame_state(current_inventory=10, person_detected=False, aruco_badge_detected=False)

    print("\n--- 2. Simulating Theft Breach Event (8 items, Unauthorized Person) ---")
    res = engine.evaluate_frame_state(current_inventory=8, person_detected=True, aruco_badge_detected=False)
    print("Engine Result:", res)

    if res["is_theft"]:
        # Log event to local CSV audit trail
        manager.log_event_to_csv("THEFT_ALERT", 8, True, False, res["message"])
        print("[SUCCESS] Event logged to audit_log.csv.")

        # Transmit to Telegram directly
        print("📱 [TELEGRAM STATUS] Sending alert to Telegram...")
        manager.send_telegram_breach_alert(res["message"])
        print("✅ [TELEGRAM STATUS] Command executed!")

    # Wait 5 seconds to ensure the MP3 audio has time to play before the script closes
    print("\nWaiting for audio playback to complete...")
    time.sleep(5)

if __name__ == "__main__":
    run_test()
