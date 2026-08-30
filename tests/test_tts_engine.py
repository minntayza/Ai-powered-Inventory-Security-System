import sys
import time
from types import SimpleNamespace
from unittest import TestCase, mock

from src.module_b_vlm_layer.tts_engine import TTSEngine
from src.utils.audio_control import AudioCoordinator


class TtsEngineTests(TestCase):
    def test_new_engine_reports_healthy_state(self):
        engine = TTSEngine(AudioCoordinator())

        self.assertEqual(engine.health(), {"healthy": True, "last_error": None})

    def test_playback_failure_releases_audio_coordinator(self):
        engine = mock.Mock()
        engine.runAndWait.side_effect = RuntimeError("speaker unavailable")
        coordinator = AudioCoordinator()
        pyttsx3 = SimpleNamespace(init=mock.Mock(return_value=engine))
        tts = TTSEngine(coordinator)

        with mock.patch.dict(sys.modules, {"pyttsx3": pyttsx3}):
            started = tts.speak("Alert summary")
            deadline = time.time() + 1
            while time.time() < deadline and not coordinator.microphone_available:
                time.sleep(0.01)

        self.assertEqual(
            {
                "started": started,
                "microphone_available": coordinator.microphone_available,
                "health": tts.health(),
            },
            {
                "started": True,
                "microphone_available": True,
                "health": {
                    "healthy": False,
                    "last_error": "speaker unavailable",
                },
            },
        )
