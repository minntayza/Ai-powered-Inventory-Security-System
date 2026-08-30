import time
from unittest import TestCase, mock

import numpy as np

from src.utils.frame_capture import FrameCapture, VideoFileCapture


class FrameCaptureTests(TestCase):
    @mock.patch("src.utils.frame_capture.cv2.VideoCapture")
    def test_unavailable_camera_is_reopened_until_a_frame_arrives(self, video_capture):
        unavailable = mock.Mock()
        unavailable.isOpened.return_value = False
        recovered = mock.Mock()
        recovered.isOpened.return_value = True
        recovered.read.return_value = (
            True,
            np.zeros((10, 10, 3), dtype=np.uint8),
        )
        video_capture.side_effect = [unavailable, recovered]
        capture = FrameCapture(
            source="camera-url",
            reconnect_delay_seconds=0.01,
            max_reconnect_delay_seconds=0.01,
        ).start()

        deadline = time.time() + 1
        healthy, frame, _frame_id = capture.read()
        while time.time() < deadline and not healthy:
            time.sleep(0.01)
            healthy, frame, _frame_id = capture.read()
        capture.stop()

        self.assertEqual(
            {
                "healthy": healthy,
                "has_frame": frame is not None,
                "open_attempts": video_capture.call_count,
            },
            {"healthy": True, "has_frame": True, "open_attempts": 2},
        )

    @mock.patch("src.utils.frame_capture.cv2.VideoCapture")
    def test_stopped_replay_does_not_report_the_last_frame_as_healthy(
        self, video_capture
    ):
        backend = mock.Mock()
        backend.isOpened.return_value = True
        backend.get.return_value = 30
        backend.read.return_value = (
            True,
            np.zeros((10, 10, 3), dtype=np.uint8),
        )
        video_capture.return_value = backend
        capture = VideoFileCapture("demo.mp4", max_fps=30).start()

        deadline = time.time() + 1
        healthy = False
        while time.time() < deadline and not healthy:
            time.sleep(0.01)
            healthy, _frame, _frame_id = capture.read()
        capture.stop()
        stopped_healthy, _frame, _frame_id = capture.read()

        self.assertTrue(healthy)
        self.assertFalse(stopped_healthy)

    @mock.patch("src.utils.frame_capture.cv2.VideoCapture")
    def test_looping_broken_replay_is_rate_limited(self, video_capture):
        backend = mock.Mock()
        backend.isOpened.return_value = True
        backend.get.return_value = 20
        backend.read.return_value = (False, None)
        video_capture.return_value = backend
        capture = VideoFileCapture("broken.mp4", max_fps=20, loop=True).start()

        time.sleep(0.12)
        capture.stop()

        self.assertLessEqual(backend.read.call_count, 4)
