import os
import sys
import time
import threading
import ctypes
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class CoreTheftEngineNoPygame:
    def __init__(self, siren_sound_path: str = "freesound_community-alert-33762.mp3", cooldown_sec: float = 5.0):
        # By default, it looks for siren.mp3 in the exact same folder as this script
        self.siren_sound_path = siren_sound_path
        self.cooldown_sec = cooldown_sec
        self.last_inventory_count = -1
        self.last_trigger_time = 0.0

    def trigger_siren(self):
        """Plays native MP3 using Windows MCI without requiring pygame."""
        def _play_sound():
            abs_path = os.path.abspath(self.siren_sound_path)
            print(f"\n[AUDIO CHECK] Looking for sound file at: {abs_path}")
            
            if os.path.exists(abs_path) and sys.platform == "win32":
                print("🔊 [AUDIO STATUS] MP3 File found! Playing siren...")
                try:
                    # Native Windows command to play MP3
                    winmm = ctypes.windll.winmm
                    winmm.mciSendStringW("close siren", None, 0, 0)
                    winmm.mciSendStringW(f'open "{abs_path}" type mpegvideo alias siren', None, 0, 0)
                    winmm.mciSendStringW("play siren", None, 0, 0)
                except Exception as e:
                    print(f"❌ [AUDIO ERROR] MP3 playback failed: {e}")
            else:
                print("⚠️ [AUDIO WARNING] MP3 file NOT found. Playing system warning beeps instead!")
                if sys.platform == "win32":
                    import winsound
                    for _ in range(3):
                        winsound.Beep(1500, 250)
                        time.sleep(0.1)

        threading.Thread(target=_play_sound, daemon=True).start()

    def evaluate_frame_state(self, current_inventory: int, person_detected: bool, aruco_badge_detected: bool) -> dict:
        """Evaluates theft logic: Inventory drop + Unauthorized person present."""
        now = time.time()
        is_theft = False
        message = "NORMAL"

        if self.last_inventory_count == -1:
            self.last_inventory_count = current_inventory
            return {"is_theft": False, "message": "BASELINE_SET"}

        inventory_dropped = current_inventory < self.last_inventory_count
        drop_amount = self.last_inventory_count - current_inventory if inventory_dropped else 0
        is_unknown_person = person_detected and not aruco_badge_detected

        if inventory_dropped and is_unknown_person:
            if (now - self.last_trigger_time) > self.cooldown_sec:
                self.last_trigger_time = now
                is_theft = True
                message = f"THEFT DETECTED! {drop_amount} item(s) removed by unauthorized person."
                self.trigger_siren()

        self.last_inventory_count = current_inventory

        return {
            "is_theft": is_theft,
            "message": message,
            "current_inventory": current_inventory
        }