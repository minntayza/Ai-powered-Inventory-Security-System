"""YOLOv8 object detector for inventory counting."""

from ultralytics import YOLO
import cv2
import numpy as np
from typing import Dict, List


class YOLODetector:
    """YOLOv8 object detector for persons, bottles, backpacks."""

    TARGET_CLASSES = {
        0: "person",
        24: "backpack",
        39: "bottle"
    }

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> Dict[str, List[Dict]]:
        results = self.model(frame, conf=self.confidence, verbose=False)

        persons = []
        inventory = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls_id == 0:
                    persons.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf
                    })
                elif cls_id in [24, 39]:
                    inventory.append({
                        "label": self.TARGET_CLASSES[cls_id],
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf
                    })

        return {"persons": persons, "inventory": inventory}

    def count_inventory(self, detections: Dict) -> int:
        return len(detections.get("inventory", []))

    def draw_detections(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        frame = frame.copy()

        for person in detections.get("persons", []):
            x1, y1, x2, y2 = person["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Person {person['confidence']:.2f}",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        for item in detections.get("inventory", []):
            x1, y1, x2, y2 = item["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{item['label']} {item['confidence']:.2f}",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame


if __name__ == "__main__":
    detector = YOLODetector()
    cap = cv2.VideoCapture(0)

    print("YOLO Detector - Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        annotated = detector.draw_detections(frame, detections)

        count = detector.count_inventory(detections)
        cv2.putText(annotated, f"Inventory: {count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("YOLO Detector", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
