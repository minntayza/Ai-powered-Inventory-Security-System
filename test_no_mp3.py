import os
import time
from native_theft_engine import CoreTheftEngineNoPygame
from day4_alerting import AlertingAndLoggingManager

# Configuration
BOT_TOKEN = "8877453484:AAFkBgSau29l6Hek5bESXnWh3f5hZwVswes"  
CHAT_ID = "2029905108"      

def run_test():
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