import sys
from types import SimpleNamespace
from unittest import TestCase, mock

from src.module_b_vlm_layer.voice_input import VoiceQuestionTranscriber
from src.utils.audio_control import AudioCoordinator


class VoiceQuestionTranscriberTests(TestCase):
    def test_empty_recording_is_rejected(self):
        transcriber = VoiceQuestionTranscriber()

        result = transcriber.transcribe(b"")

        self.assertEqual(
            result,
            {"ok": False, "question": "", "error": "Voice recording is empty"},
        )

    def test_recording_is_transcribed_into_a_question(self):
        transcriber = VoiceQuestionTranscriber()

        result = transcriber.transcribe(
            b"wav-data", recognizer=lambda _audio: {"text": "  What is missing?  "}
        )

        self.assertEqual(
            result,
            {"ok": True, "question": "What is missing?", "error": None},
        )

    def test_default_recognizer_runs_locally_and_is_reused(self):
        pipeline = mock.Mock()
        pipeline.return_value.return_value = {"text": "Who moved the laptop?"}
        transcriber = VoiceQuestionTranscriber()

        with mock.patch.dict(
            sys.modules, {"transformers": SimpleNamespace(pipeline=pipeline)}
        ):
            first = transcriber.transcribe(b"first-wav")
            second = transcriber.transcribe(b"second-wav")

        self.assertEqual(
            {
                "questions": [first["question"], second["question"]],
                "loads": pipeline.call_count,
            },
            {
                "questions": ["Who moved the laptop?", "Who moved the laptop?"],
                "loads": 1,
            },
        )

    def test_recognition_failure_is_returned_to_the_caller(self):
        transcriber = VoiceQuestionTranscriber()

        def unavailable(_audio):
            raise RuntimeError("speech model unavailable")

        result = transcriber.transcribe(b"wav-data", recognizer=unavailable)

        self.assertEqual(
            result,
            {
                "ok": False,
                "question": "",
                "error": "speech model unavailable",
            },
        )

    def test_voice_capture_is_rejected_while_siren_is_active(self):
        coordinator = AudioCoordinator()
        coordinator.begin_siren()
        recognizer = mock.Mock()

        result = VoiceQuestionTranscriber().transcribe(
            b"wav-data", recognizer=recognizer, coordinator=coordinator
        )

        self.assertEqual(
            {"result": result, "recognizer_called": recognizer.called},
            {
                "result": {
                    "ok": False,
                    "question": "",
                    "error": "Voice input is unavailable while audio is playing",
                },
                "recognizer_called": False,
            },
        )

    def test_completed_transcription_releases_microphone(self):
        coordinator = AudioCoordinator()

        VoiceQuestionTranscriber().transcribe(
            b"wav-data",
            recognizer=lambda _audio: {"text": "Describe the shelf"},
            coordinator=coordinator,
        )

        self.assertTrue(coordinator.microphone_available)

    def test_silent_recording_is_rejected(self):
        result = VoiceQuestionTranscriber().transcribe(
            b"wav-data", recognizer=lambda _audio: {"text": "   "}
        )

        self.assertEqual(
            result,
            {
                "ok": False,
                "question": "",
                "error": "No speech was recognized",
            },
        )
