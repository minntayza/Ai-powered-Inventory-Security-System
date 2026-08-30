import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

from src.backend_alerting.siren import Siren
from src.utils.audio_control import AudioCoordinator


class SirenTests(TestCase):
    def test_start_and_stop_coordinate_microphone_availability(self):
        mixer = SimpleNamespace(
            get_init=mock.Mock(return_value=False),
            init=mock.Mock(),
            music=SimpleNamespace(
                load=mock.Mock(), play=mock.Mock(), stop=mock.Mock()
            ),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules, {"pygame": SimpleNamespace(mixer=mixer)}
        ):
            audio_path = Path(directory) / "siren.mp3"
            audio_path.write_bytes(b"audio")
            coordinator = AudioCoordinator()
            siren = Siren(str(audio_path), coordinator=coordinator)

            started = siren.start()
            during = coordinator.microphone_available
            siren.stop()
            after = coordinator.microphone_available

        self.assertEqual(
            {"started": started, "during": during, "after": after},
            {"started": True, "during": False, "after": True},
        )
