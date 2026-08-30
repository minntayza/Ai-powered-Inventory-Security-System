import time
from unittest import TestCase, mock

import numpy as np

from src.utils.frame_capture import FrameCapture


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
