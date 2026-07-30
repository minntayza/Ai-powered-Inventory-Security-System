"""Person tracking with face recognition correlation."""

import cv2
import numpy as np
import time
from typing import Dict, List
from .yolo_detector import YOLODetector
from .face_detector import FaceDetector
from .inventory_counter import InventoryCounter


class PersonTracker:
    """Track persons and correlate with face recognition."""

    def __init__(
        self,
        grace_period: float = 3.0,
        face_db_path: str = "assets/known_faces"
    ):
        self.yolo_detector = YOLODetector()
        self.face_detector = FaceDetector(db_path=face_db_path)
        self.inventory_counter = InventoryCounter()

        self.grace_period = grace_period
        self.person_history = {}
        self.next_person_id = 1

    def process_frame(self, frame: np.ndarray) -> Dict:
        yolo_results = self.yolo_detector.detect(frame)
        inventory_count = self.inventory_counter.update_count(yolo_results)

        face_results = self.face_detector.detect_faces(frame)

        persons = self._correlate_persons_faces(
            yolo_results["persons"], face_results
        )

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "persons": persons,
            "inventory": {
                "count": inventory_count,
                "items": yolo_results["inventory"]
            }
        }

    def _correlate_persons_faces(
        self,
        person_bboxes: List[Dict],
        face_results: List[Dict]
    ) -> List[Dict]:
        correlated = []

        for person in person_bboxes:
            px1, py1, px2, py2 = person["bbox"]

            matched_face = None
            best_overlap = 0

            for face in face_results:
                fx, fy, fw, fh = face["bbox"]

                overlap = self._calculate_overlap(
                    px1, py1, px2, py2,
                    fx, fy, fx + fw, fy + fh
                )

                if overlap > best_overlap:
                    best_overlap = overlap
                    matched_face = face

            person_data = {
                "id": self._get_or_create_person_id(px1, py1, px2, py2),
                "bbox": person["bbox"],
                "confidence": person["confidence"]
            }

            if matched_face and best_overlap > 0.3:
                person_data["name"] = matched_face["name"]
                person_data["authorized"] = matched_face["authorized"]
                person_data["face_confidence"] = matched_face["confidence"]
            else:
                person_data["name"] = "Unknown"
                person_data["authorized"] = False
                person_data["face_confidence"] = 0.0

            correlated.append(person_data)

        return correlated

    def _calculate_overlap(
        self, x1: int, y1: int, x2: int, y2: int,
        fx1: int, fy1: int, fx2: int, fy2: int
    ) -> float:
        x_left = max(x1, fx1)
        y_top = max(y1, fy1)
        x_right = min(x2, fx2)
        y_bottom = min(y2, fy2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        person_area = (x2 - x1) * (y2 - y1)

        if person_area == 0:
            return 0.0

        return intersection / person_area

    def _get_or_create_person_id(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> int:
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        for person_id, history in self.person_history.items():
            last_pos = history["last_position"]
            distance = ((center_x - last_pos[0]) ** 2 +
                       (center_y - last_pos[1]) ** 2) ** 0.5

            if distance < 100:
                history["last_position"] = (center_x, center_y)
                history["last_seen"] = time.time()
                return person_id

        new_id = self.next_person_id
        self.next_person_id += 1
        self.person_history[new_id] = {
            "last_position": (center_x, center_y),
            "last_seen": time.time()
        }

        return new_id

    def draw_tracking(self, frame: np.ndarray, output: Dict) -> np.ndarray:
        frame = frame.copy()

        for person in output["persons"]:
            x1, y1, x2, y2 = person["bbox"]

            if person["authorized"]:
                color = (0, 255, 0)
                label = f"{person['name']} (ID:{person['id']})"
            else:
                color = (0, 0, 255)
                label = f"Unknown (ID:{person['id']})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        inventory_count = output["inventory"]["count"]
        cv2.putText(frame, f"Inventory: {inventory_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return frame


if __name__ == "__main__":
    tracker = PersonTracker()

    cap = cv2.VideoCapture(0)
    print("Person Tracker - Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = tracker.process_frame(frame)
        annotated = tracker.draw_tracking(frame, output)

        cv2.imshow("Person Tracker", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
