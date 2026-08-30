import tempfile
from pathlib import Path
from unittest import TestCase, mock

import numpy as np

from src.backend_alerting.incident_recorder import IncidentRecorder


class IncidentRecorderTests(TestCase):
    @mock.patch("src.backend_alerting.incident_recorder.cv2.VideoWriter")
    def test_mp4_failure_falls_back_to_mjpeg_avi(self, video_writer):
        mp4_writer = mock.Mock()
        mp4_writer.isOpened.return_value = False
        avi_writer = mock.Mock()
        avi_writer.isOpened.return_value = True
        video_writer.side_effect = [mp4_writer, avi_writer]

        with tempfile.TemporaryDirectory() as directory:
            recorder = IncidentRecorder(
                directory,
                fps=2,
                pre_event_seconds=1,
                post_event_seconds=0,
                enabled=True,
            )
            recorder.observe(np.zeros((20, 30, 3), dtype=np.uint8), now=1)
            recorder.start_event("event-fallback", now=1)
            result = recorder.shutdown()[0]

        self.assertEqual(
            {
                "suffix": Path(result["path"]).suffix,
                "error": result["error"],
                "mp4_released": mp4_writer.release.called,
                "avi_frames": avi_writer.write.call_count,
            },
            {
                "suffix": ".avi",
                "error": None,
                "mp4_released": True,
                "avi_frames": 1,
            },
        )

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
