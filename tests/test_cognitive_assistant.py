from unittest import TestCase

from src.module_b_vlm_layer.voice_input import VoiceQuestionTranscriber
from src.module_c_ui_dashboard.cognitive_assistant import (
    select_assistant_question,
    transcribe_voice_recording,
)
from src.utils.audio_control import AudioCoordinator


class Recording:
    def getvalue(self):
        return b"wav-data"


class CognitiveAssistantTests(TestCase):
    def test_recorded_question_is_transcribed_for_the_assistant(self):
        result = transcribe_voice_recording(
            Recording(),
            VoiceQuestionTranscriber(),
            AudioCoordinator(),
            recognizer=lambda _audio: {"text": "What is on the shelf?"},
        )

        self.assertEqual(result["question"], "What is on the shelf?")

    def test_voice_question_is_used_when_text_is_empty(self):
        question = select_assistant_question(
            typed_question=None,
            voice_result={"ok": True, "question": "Who moved the laptop?"},
            action=None,
        )

        self.assertEqual(question, "Who moved the laptop?")
