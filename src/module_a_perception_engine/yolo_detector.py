"""Ultralytics YOLO detector for people and custom inventory classes."""

from ultralytics import YOLO
import cv2
import logging
import numpy as np
from typing import Dict, Iterable, List, Optional

from src.utils.gpu_manager import resolve_torch_device


class YOLODetector:
    """Detect people separately from the inventory classes of a YOLO model."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        inventory_classes: Optional[Iterable[str]] = None,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        device: Optional[str] = None,
        tracking: bool = True,
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = resolve_torch_device(device)
        self.tracking = tracking
        self._logger = logging.getLogger("inventory_security.yolo")
        # None means that every non-person class supported by the model is
        # returned as inventory. Pass an iterable of names to filter the list.
        self.inventory_classes = (
            set(inventory_classes) if inventory_classes is not None else None
        )

    def _class_name(self, class_id: int) -> str:
        """Resolve a class by name so custom models do not depend on COCO IDs."""
        names = self.model.names
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        return str(names[class_id])

    def detect(self, frame: np.ndarray) -> Dict[str, List[Dict]]:
        inference = self.model.track if self.tracking else self.model.predict
        options = {
            "conf": self.confidence,
            "iou": self.iou_threshold,
            "imgsz": self.image_size,
            "device": self.device,
            "verbose": False,
        }
        if self.tracking:
            options.update({"persist": True, "tracker": "bytetrack.yaml"})
        try:
            results = inference(frame, **options)
        except RuntimeError as exc:
            if self.device == "cpu":
                raise
            self._logger.warning(
                "%s inference failed; retrying YOLO on CPU: %s", self.device, exc
            )
            self.device = "cpu"
            options["device"] = "cpu"
            results = inference(frame, **options)

        persons = []
        inventory = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = self._class_name(cls_id)
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label == "person":
                    persons.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "track_id": int(box.id[0]) if box.id is not None else None,
                    })
                elif self.inventory_classes is None or label in self.inventory_classes:
                    inventory.append({
                        "label": label,
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
