from unittest import TestCase

from src.utils.audio_control import AudioCoordinator


class AudioCoordinatorTests(TestCase):
    def test_microphone_is_blocked_while_siren_is_active(self):
        coordinator = AudioCoordinator()

        before = coordinator.microphone_available
        coordinator.begin_siren()
        during = coordinator.microphone_available
        coordinator.end_siren()
        after = coordinator.microphone_available

        self.assertEqual([before, during, after], [True, False, True])

    def test_microphone_cannot_start_while_siren_is_active(self):
        coordinator = AudioCoordinator()
        coordinator.begin_siren()

        self.assertFalse(coordinator.try_begin_microphone())

    def test_ending_microphone_capture_releases_it(self):
        coordinator = AudioCoordinator()
        coordinator.try_begin_microphone()

        coordinator.end_microphone()

        self.assertTrue(coordinator.microphone_available)
