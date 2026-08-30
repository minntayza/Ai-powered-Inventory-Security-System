"""Person tracking with face recognition correlation."""

import cv2
import numpy as np
import time
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from typing import Dict, Iterable, List, Optional
from .yolo_detector import YOLODetector
from .face_detector import FaceDetector
from .inventory_counter import InventoryCounter
from .interaction_associator import InteractionAssociator


class PersonTracker:
    """Track persons and correlate with face recognition."""

    def __init__(
        self,
        grace_period: float = 3.0,
        face_db_path: str = "assets/known_faces",
        face_interval_frames: int = 5,
        track_expiry_seconds: float = 5.0,
        yolo_detector=None,
        face_detector=None,
        inventory_counter=None,
        contextual_counter=None,
        protected_classes: Optional[Iterable[str]] = None,
        contextual_classes: Optional[Iterable[str]] = None,
        shelf_region: Optional[List[float]] = None,
        interaction_associator=None,
    ):
        self.yolo_detector = yolo_detector or YOLODetector()
        self.face_detector = face_detector or FaceDetector(db_path=face_db_path)
        self.inventory_counter = inventory_counter or InventoryCounter()
        self.contextual_counter = contextual_counter or InventoryCounter()
        self.protected_classes = set(protected_classes or ())
        self.contextual_classes = set(contextual_classes or ())
        self.shelf_region = shelf_region
        self.interaction_associator = interaction_associator or InteractionAssociator()

        self.grace_period = grace_period
        self.person_history = {}
        self.next_person_id = 1
        self.face_interval_frames = max(1, face_interval_frames)
        self.track_expiry_seconds = track_expiry_seconds
        self.frame_number = 0
        self._face_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="face-recognition"
        )
        self._face_future: Future | None = None

    def process_frame(self, frame: np.ndarray) -> Dict:
        yolo_results = self.yolo_detector.detect(frame)
        classified_items = []
        for item in yolo_results.get("inventory", []):
            enriched = dict(item)
            label = enriched.get("label", "unknown")
            if label in self.protected_classes:
                enriched["policy"] = "protected"
            elif label in self.contextual_classes:
                enriched["policy"] = "contextual"
            else:
                enriched.setdefault("policy", "ignored")
            enriched["in_zone"] = self._item_in_zone(enriched, frame)
            classified_items.append(enriched)

        protected = [
            item for item in classified_items
            if item["policy"] == "protected" and item["in_zone"]
        ]
        contextual = [
            item for item in classified_items if item["policy"] == "contextual"
        ]
        inventory = self.inventory_counter.update({"inventory": protected})
        context_inventory = self.contextual_counter.update({"inventory": contextual})

        self.frame_number += 1
        face_results = self._completed_face_results()
        if (
            self._face_future is None
            and (self.frame_number == 1 or self.frame_number % self.face_interval_frames == 0)
        ):
            self._face_future = self._face_executor.submit(
                self.face_detector.detect_faces, frame.copy()
            )

        persons = self._correlate_persons_faces(
            yolo_results["persons"], face_results
        )
        associations = self.interaction_associator.observe(
            protected,
            persons,
            [int(frame.shape[1]), int(frame.shape[0])],
            drop=inventory.get("drop"),
        )
        if inventory.get("drop") is not None:
            inventory["drop"]["associations"] = deepcopy(associations)
        if inventory.get("change") is not None:
            inventory["change"]["associations"] = deepcopy(associations)

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "persons": persons,
            "inventory": {
                **inventory,
                "count": inventory["total_stable_count"],
                "items": classified_items,
                "protected_counts": inventory["stable_counts"],
                "contextual_counts": context_inventory["stable_counts"],
                "associations": associations,
            },
            "frame_number": self.frame_number,
            "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
            "face_analysis_pending": self._face_future is not None,
        }

    def _item_in_zone(self, item: Dict, frame: np.ndarray) -> bool:
        if not self.shelf_region:
            return True
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = item.get("bbox", [0, 0, 0, 0])
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        rx1, ry1, rx2, ry2 = self.shelf_region
        return (
            rx1 * width <= center_x <= rx2 * width
            and ry1 * height <= center_y <= ry2 * height
        )

    def set_shelf_region(self, region: Optional[List[float]]) -> None:
        self.shelf_region = list(region) if region else None
        self.interaction_associator.reset()

    def _completed_face_results(self) -> List[Dict]:
        if self._face_future is None or not self._face_future.done():
            return []
        future, self._face_future = self._face_future, None
        try:
            return future.result()
        except Exception as exc:
            print(f"Background face detection error: {exc}")
            return []

    def shutdown(self) -> None:
        """Stop accepting face-analysis work during application shutdown."""
        if self._face_future is not None:
            self._face_future.cancel()
        self._face_executor.shutdown(wait=False, cancel_futures=True)

    def reset_tracking(self) -> None:
        """Clear source-specific histories while retaining loaded models."""
        self.person_history.clear()
        self.next_person_id = 1
        self.frame_number = 0
        self.inventory_counter.reset_baseline(clear_observations=True)
        self.contextual_counter.reset()
        self.interaction_associator.reset()
        predictor = getattr(self.yolo_detector.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []) or []:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

    def _correlate_persons_faces(
        self,
        person_bboxes: List[Dict],
        face_results: List[Dict]
    ) -> List[Dict]:
        correlated = []
        self._expire_tracks()
        assigned_ids = set()
        assigned_face_indexes = set()

        for person in person_bboxes:
            px1, py1, px2, py2 = person["bbox"]

            matched_face = None
            matched_face_index = None
            best_score = 0.0

            for face_index, face in enumerate(face_results):
                if face_index in assigned_face_indexes:
                    continue
                fx, fy, fw, fh = face["bbox"]

                face_center_x = fx + fw / 2
                face_center_y = fy + fh / 2
                inside = px1 <= face_center_x <= px2 and py1 <= face_center_y <= py2
                score = float(fw * fh) if inside else 0.0
                if score > best_score:
                    best_score = score
                    matched_face = face
                    matched_face_index = face_index

            model_track_id = person.get("track_id")
            if model_track_id is not None:
                person_id = int(model_track_id)
                self._remember_model_track(person_id, px1, py1, px2, py2)
            else:
                person_id = self._get_or_create_person_id(
                    px1, py1, px2, py2, excluded_ids=assigned_ids
                )
            assigned_ids.add(person_id)
            person_data = {
                "id": person_id,
                "track_id": person_id,
                "bbox": person["bbox"],
                "confidence": person["confidence"]
            }

            history = self.person_history[person_id]
            if matched_face:
                assigned_face_indexes.add(matched_face_index)
                person_data["name"] = matched_face["name"]
                person_data["authorized"] = matched_face["authorized"]
                person_data["face_confidence"] = matched_face["confidence"]
                person_data["authorization_state"] = matched_face.get(
                    "authorization_state",
                    "authorized" if matched_face["authorized"] else "unknown",
                )
                history["identity"] = person_data["name"]
                history["authorization_state"] = person_data["authorization_state"]
                history["face_confidence"] = person_data["face_confidence"]
                if person_data["authorized"]:
                    history["last_authorized"] = time.time()
            else:
                recently_authorized = (
                    history.get("last_authorized", 0) + self.grace_period >= time.time()
                )
                person_data["name"] = history.get("identity", "Unknown") if recently_authorized else "Unknown"
                person_data["authorized"] = recently_authorized
                person_data["face_confidence"] = history.get("face_confidence", 0.0) if recently_authorized else 0.0
                person_data["authorization_state"] = "authorized" if recently_authorized else "not_visible"

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
        self, x1: int, y1: int, x2: int, y2: int,
        excluded_ids=None,
    ) -> int:
        excluded_ids = excluded_ids or set()
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        for person_id, history in self.person_history.items():
            if person_id in excluded_ids:
                continue
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
            "last_seen": time.time(),
            "authorization_state": "not_visible",
            "identity": "Unknown",
            "face_confidence": 0.0,
            "last_authorized": 0.0,
        }

        return new_id

    def _remember_model_track(self, person_id: int, x1: int, y1: int, x2: int, y2: int) -> None:
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        history = self.person_history.setdefault(
            person_id,
            {
                "authorization_state": "not_visible",
                "identity": "Unknown",
                "face_confidence": 0.0,
                "last_authorized": 0.0,
            },
        )
        history["last_position"] = center
        history["last_seen"] = time.time()
        self.next_person_id = max(self.next_person_id, person_id + 1)

    def _expire_tracks(self) -> None:
        now = time.time()
        expired = [
            person_id
            for person_id, history in self.person_history.items()
            if now - history["last_seen"] > self.track_expiry_seconds
        ]
        for person_id in expired:
            del self.person_history[person_id]

    def draw_tracking(self, frame: np.ndarray, output: Dict) -> np.ndarray:
        frame = frame.copy()

        if self.shelf_region:
            height, width = frame.shape[:2]
            rx1, ry1, rx2, ry2 = self.shelf_region
            cv2.rectangle(
                frame,
                (int(rx1 * width), int(ry1 * height)),
                (int(rx2 * width), int(ry2 * height)),
                (255, 180, 0),
                2,
            )
            cv2.putText(
                frame, "MONITORED ZONE", (int(rx1 * width) + 5, int(ry1 * height) + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2,
            )

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
        for item in output["inventory"].get("items", []):
            x1, y1, x2, y2 = item["bbox"]
            if item.get("policy") == "protected" and item.get("in_zone"):
                color = (0, 255, 255)
            elif item.get("policy") == "contextual":
                color = (160, 160, 160)
            else:
                color = (90, 90, 90)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                frame, item.get("label", "item"), (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
            )
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
