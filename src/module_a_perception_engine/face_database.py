"""Face database management for known personnel."""

import os
import glob
from typing import List, Dict


class FaceDatabase:
    """Manage known face database directory."""

    def __init__(self, db_path: str = "assets/known_faces"):
        self.db_path = db_path
        self._ensure_db_structure()

    def _ensure_db_structure(self):
        os.makedirs(self.db_path, exist_ok=True)

    def get_known_names(self) -> List[str]:
        if not os.path.exists(self.db_path):
            return []

        names = []
        for item in os.listdir(self.db_path):
            item_path = os.path.join(self.db_path, item)
            if os.path.isdir(item_path):
                names.append(item)

        return names

    def get_person_images(self, name: str) -> List[str]:
        person_dir = os.path.join(self.db_path, name)
        if not os.path.exists(person_dir):
            return []

        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        images = []

        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(person_dir, ext)))
            images.extend(glob.glob(os.path.join(person_dir, ext.upper())))

        return images

    def get_all_images(self) -> Dict[str, List[str]]:
        result = {}
        for name in self.get_known_names():
            result[name] = self.get_person_images(name)
        return result

    def add_person_directory(self, name: str):
        person_dir = os.path.join(self.db_path, name)
        os.makedirs(person_dir, exist_ok=True)

    def print_database_info(self):
        print(f"Face Database: {self.db_path}")
        print(f"Known persons: {len(self.get_known_names())}")

        for name in self.get_known_names():
            images = self.get_person_images(name)
            print(f"  - {name}: {len(images)} images")


if __name__ == "__main__":
    db = FaceDatabase()
    db.print_database_info()
