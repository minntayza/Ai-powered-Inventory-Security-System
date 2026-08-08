import tempfile
from pathlib import Path
from unittest import TestCase

from src.module_a_perception_engine.face_database import FaceDatabase


class FaceDatabaseTests(TestCase):
    def test_empty_person_directories_are_not_enrolled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "empty-person").mkdir()
            database = FaceDatabase(str(root))
            self.assertEqual(database.get_known_names(), [])

    def test_person_with_supported_image_is_enrolled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            person = root / "alice"
            person.mkdir()
            (person / "front.jpg").touch()
            database = FaceDatabase(str(root))
            self.assertEqual(database.get_known_names(), ["alice"])
