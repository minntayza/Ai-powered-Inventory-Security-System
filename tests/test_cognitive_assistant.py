from unittest import TestCase

from src.module_c_ui_dashboard.cognitive_assistant import select_assistant_question


class CognitiveAssistantTests(TestCase):
    def test_typed_question_is_used(self):
        question = select_assistant_question(
            typed_question="Who moved the laptop?",
            action=None,
        )

        self.assertEqual(question, "Who moved the laptop?")

    def test_action_supplies_default_question(self):
        question = select_assistant_question(typed_question=None, action="describe")

        self.assertEqual(question, "Describe the scene in detail")
