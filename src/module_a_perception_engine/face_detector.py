"""DeepFace-based face detection and recognition."""

import os
import cv2
import numpy as np
import time
from deepface import DeepFace
from typing import List, Dict
from .face_database import FaceDatabase


class FaceDetector:
    """DeepFace face detector and recognizer."""

    def __init__(
        self,
        db_path: str = "assets/known_faces",
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        recognition_threshold: float = 0.4,
        anti_spoofing: bool = True
    ):
        self.db_path = db_path
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.recognition_threshold = recognition_threshold
        self.anti_spoofing = anti_spoofing

        self.face_db = FaceDatabase(db_path)
        self.known_names = self.face_db.get_known_names()
        self.last_inference_ms = 0.0

        print(f"Face Detector initialized: {len(self.known_names)} known persons")

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        started = time.perf_counter()
        results = []

        try:
            face_objs = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                anti_spoofing=self.anti_spoofing
            )

            for face_obj in face_objs:
                if face_obj["face"] is not None:
                    facial_area = face_obj["facial_area"]
                    x = facial_area["x"]
                    y = facial_area["y"]
                    w = facial_area["w"]
                    h = facial_area["h"]

                    is_real = face_obj.get("is_real", True)

                    if not is_real:
                        results.append({
                            "bbox": [x, y, w, h],
                            "name": "Spoof",
                            "authorized": False,
                            "authorization_state": "spoof",
                            "confidence": 0.0,
                            "face_detected": True,
                            "is_real": False
                        })
                        continue

                    name, authorized, confidence = self._recognize_face(
                        frame, x, y, w, h
                    )

                    results.append({
                        "bbox": [x, y, w, h],
                        "name": name,
                        "authorized": authorized,
                        "authorization_state": "authorized" if authorized else "unknown",
                        "confidence": confidence,
                        "face_detected": True,
                        "is_real": True
                    })

        except Exception as e:
            print(f"Face detection error: {e}")

        self.last_inference_ms = (time.perf_counter() - started) * 1000
        return results

    def refresh_database(self) -> None:
        """Reload enrolled folder names after authorized images are added."""
        self.known_names = self.face_db.get_known_names()

    def _recognize_face(
        self, frame: np.ndarray, x: int, y: int, w: int, h: int
    ) -> tuple:
        if not self.known_names:
            return "Unknown", False, 0.0

        try:
            face_img = frame[y:y + h, x:x + w]

            if face_img.size == 0:
                return "Unknown", False, 0.0

            dfs = DeepFace.find(
                img_path=face_img,
                db_path=self.db_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True
            )

            if dfs and len(dfs) > 0 and not dfs[0].empty:
                best_match = dfs[0].iloc[0]
                distance = best_match.get("distance", 1.0)

                if distance < self.recognition_threshold:
                    identity_path = best_match.get("identity", "")
                    name = os.path.basename(os.path.dirname(identity_path))
                    return name.capitalize(), True, 1 - distance

            return "Unknown", False, 0.0

        except Exception as e:
            print(f"Recognition error: {e}")
            return "Unknown", False, 0.0

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        frame = frame.copy()

        for det in detections:
            x, y, w, h = det["bbox"]

            if det["authorized"]:
                color = (0, 255, 0)
                label = f"{det['name']} (Auth)"
            else:
                color = (0, 0, 255)
                label = det["name"]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame


if __name__ == "__main__":
    detector = FaceDetector()

    cap = cv2.VideoCapture(0)
    print("Face Detector - Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_faces(frame)
        annotated = detector.draw_detections(frame, detections)

        cv2.imshow("Face Recognition", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
