"""Validated, atomic management of authorized face images."""

from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


Upload = Tuple[str, bytes]


class FaceEnrollmentService:
    NAME_PATTERN = re.compile(r"^[A-Za-z0-9 _-]{1,50}$")
    ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        db_path: str,
        detector_backend: str = "opencv",
        min_face_size: int = 50,
        max_file_bytes: int = 5 * 1024 * 1024,
        validator: Optional[Callable[[np.ndarray], int]] = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.detector_backend = detector_backend
        self.min_face_size = int(min_face_size)
        self.max_file_bytes = int(max_file_bytes)
        self.validator = validator or self._deepface_count

    def enroll(self, name: str, uploads: Iterable[Upload]) -> Dict:
        normalized = " ".join(name.strip().split())
        if not self.NAME_PATTERN.fullmatch(normalized):
            return {"ok": False, "message": "Name must use letters, numbers, spaces, _ or -"}
        files = list(uploads)
        if not 1 <= len(files) <= 5:
            return {"ok": False, "message": "Upload between 1 and 5 face images"}

        validated = []
        for filename, payload in files:
            suffix = Path(filename).suffix.lower()
            if suffix not in self.ALLOWED_SUFFIXES:
                return {"ok": False, "message": f"Unsupported image type: {filename}"}
            if not payload or len(payload) > self.max_file_bytes:
                return {"ok": False, "message": f"Image must be between 1 byte and 5 MB: {filename}"}
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return {"ok": False, "message": f"Could not decode image: {filename}"}
            try:
                count = self.validator(image)
            except Exception as exc:
                return {"ok": False, "message": f"Face validation failed for {filename}: {exc}"}
            if count != 1:
                return {"ok": False, "message": f"Exactly one clear face is required in {filename}"}
            validated.append((suffix, payload))

        person_dir = self.db_path / normalized
        person_dir.mkdir(parents=True, exist_ok=True)
        written = []
        try:
            for suffix, payload in validated:
                target = person_dir / f"{uuid.uuid4().hex}{suffix}"
                target.write_bytes(payload)
                written.append(target)
        except Exception:
            for path in written:
                path.unlink(missing_ok=True)
            if not any(person_dir.iterdir()):
                person_dir.rmdir()
            raise
        return {
            "ok": True,
            "message": f"Enrolled {len(written)} image(s) for {normalized}",
            "name": normalized,
            "count": len(written),
        }

    def remove(self, name: str) -> Dict:
        if not self.NAME_PATTERN.fullmatch(name):
            return {"ok": False, "message": "Invalid identity path"}
        root = self.db_path.resolve()
        target = (root / name).resolve()
        if not target.is_dir() or target.parent != root:
            return {"ok": False, "message": "Identity does not exist"}
        shutil.rmtree(target)
        return {"ok": True, "message": f"Removed authorized identity: {name}"}

    def identities(self) -> Dict[str, int]:
        result = {}
        for directory in sorted(self.db_path.iterdir()):
            if directory.is_dir():
                count = sum(
                    1 for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() in self.ALLOWED_SUFFIXES
                )
                if count:
                    result[directory.name] = count
        return result

    def _deepface_count(self, image: np.ndarray) -> int:
        from deepface import DeepFace

        faces = DeepFace.extract_faces(
            img_path=image,
            detector_backend=self.detector_backend,
            enforce_detection=True,
            anti_spoofing=False,
        )
        valid = 0
        for face in faces:
            area = face.get("facial_area", {})
            if int(area.get("w", 0)) >= self.min_face_size and int(area.get("h", 0)) >= self.min_face_size:
                valid += 1
        return valid
