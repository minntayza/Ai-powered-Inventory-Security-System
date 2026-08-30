import tempfile
from pathlib import Path
from unittest import TestCase

import cv2
import numpy as np

from src.module_a_perception_engine.face_enrollment import FaceEnrollmentService


def encoded_image():
    ok, value = cv2.imencode(".jpg", np.zeros((100, 100, 3), dtype=np.uint8))
    assert ok
    return value.tobytes()


class FaceEnrollmentTests(TestCase):
    def test_enrollment_rejects_unsupported_image_type(self):
        with tempfile.TemporaryDirectory() as directory:
            service = FaceEnrollmentService(directory, validator=lambda _image: 1)

            result = service.enroll("Demo User", [("face.gif", b"gif-data")])

            self.assertEqual(
                {"ok": result["ok"], "identities": service.identities()},
                {"ok": False, "identities": {}},
            )

    def test_enrollment_rejects_unsafe_identity_name(self):
        with tempfile.TemporaryDirectory() as directory:
            service = FaceEnrollmentService(directory, validator=lambda _image: 1)

            result = service.enroll(
                "../outside", [("face.jpg", encoded_image())]
            )

            self.assertEqual(
                {"ok": result["ok"], "identities": service.identities()},
                {"ok": False, "identities": {}},
            )

    def test_valid_upload_is_enrolled_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            service = FaceEnrollmentService(directory, validator=lambda _image: 1)
            result = service.enroll("Demo User", [("face.jpg", encoded_image())])
            self.assertTrue(result["ok"])
            self.assertEqual(service.identities(), {"Demo User": 1})

    def test_invalid_second_image_does_not_partially_write(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = iter([1, 0])
            service = FaceEnrollmentService(directory, validator=lambda _image: next(calls))
            result = service.enroll(
                "Demo User", [("one.jpg", encoded_image()), ("two.jpg", encoded_image())]
            )
            self.assertFalse(result["ok"])
            self.assertEqual(service.identities(), {})

    def test_remove_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            service = FaceEnrollmentService(directory, validator=lambda _image: 1)
            self.assertFalse(service.remove("../")["ok"])
