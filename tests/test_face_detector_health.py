import tempfile
from unittest import TestCase, mock

import numpy as np

from src.module_a_perception_engine.face_detector import FaceDetector


class FaceDetectorHealthTests(TestCase):
    def test_new_detector_reports_healthy_state(self):
        with tempfile.TemporaryDirectory() as directory:
            detector = FaceDetector(db_path=directory, anti_spoofing=False)

        self.assertEqual(detector.health(), {"healthy": True, "last_error": None})

    @mock.patch(
        "src.module_a_perception_engine.face_detector.DeepFace.extract_faces",
        side_effect=RuntimeError("face backend failed"),
    )
    def test_recognition_failure_is_visible_in_detector_health(self, _extract):
        with tempfile.TemporaryDirectory() as directory:
            detector = FaceDetector(db_path=directory, anti_spoofing=False)
            detector.detect_faces(np.zeros((20, 20, 3), dtype=np.uint8))

        self.assertEqual(
            detector.health(),
            {"healthy": False, "last_error": "face backend failed"},
        )

    @mock.patch(
        "src.module_a_perception_engine.face_detector.DeepFace.find",
        side_effect=RuntimeError("embedding lookup failed"),
    )
    def test_embedding_lookup_failure_is_visible_in_detector_health(self, _find):
        with tempfile.TemporaryDirectory() as directory:
            detector = FaceDetector(db_path=directory, anti_spoofing=False)
            detector.known_names = ["Alice Smith"]
            result = detector._recognize_face(
                np.zeros((20, 20, 3), dtype=np.uint8), 0, 0, 20, 20
            )

        self.assertEqual(result, ("Unknown", False, 0.0))
        self.assertEqual(
            detector.health(),
            {
                "healthy": False,
                "last_error": "Face recognition failed: embedding lookup failed",
            },
        )
