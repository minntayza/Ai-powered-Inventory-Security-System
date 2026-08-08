import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np

from src.backend_alerting.incident_recorder import IncidentRecorder


class IncidentRecorderTests(TestCase):
    def test_zero_post_event_recording_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = IncidentRecorder(
                directory, fps=2, pre_event_seconds=1, post_event_seconds=0,
                enabled=True,
            )
            frame = np.zeros((20, 30, 3), dtype=np.uint8)
            recorder.observe(frame, now=1)
            recorder.start_event("event-1", now=1)
            results = recorder.shutdown()
            self.assertEqual(results[0]["event_id"], "event-1")
            self.assertTrue(Path(results[0]["path"]).is_file())

    def test_buffer_is_trimmed_by_time(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = IncidentRecorder(directory, pre_event_seconds=1, enabled=True)
            frame = np.zeros((10, 10, 3), dtype=np.uint8)
            recorder.observe(frame, now=1)
            recorder.observe(frame, now=3)
            self.assertEqual(len(recorder._buffer), 1)
            recorder.shutdown()
