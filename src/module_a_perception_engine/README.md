# Module A: Perception & Security Engine

**Owner:** CV Engineer

## Folder
```
src/module_a_perception_engine/
```

## Files to Create

| File | Purpose |
|---|---|
| `yolo_detector.py` | YOLOv8 model loading, inference, and inventory object detection (bottles, backpacks) |
| `aruco_detector.py` | OpenCV ArUco/QR marker detection to identify authorized personnel badges |
| `person_tracker.py` | Correlate person bounding boxes with ArUco marker presence (authorized vs unknown) |
| `inventory_counter.py` | Track and output dynamic item counts on shelves, detect count drops |

## Key Integration Points

- **Output to Backend (`src/backend_alerting/`):** Person ID, authorization status (badge detected: yes/no), current inventory count, bounding box coordinates
- **Input from `src/utils/frame_capture.py`:** Camera frames
- **Shared config:** `configs/thresholds.yaml` — YOLOv8 confidence thresholds, ArUco dictionary ID

## Day-by-Day Deliverables

| Day | Task |
|---|---|
| 1 | Setup YOLOv8 environment, generate test ArUco markers, gather sample images |
| 2 | Integrate ArUco detection inside person bounding boxes, output dynamic item counts |
| 3 | Link YOLO + ArUco output states to backend logic engine |
| 4 | Refine confidence thresholds, fix lighting variances for ArUco readability |
| 5 | Code freeze, remove debug statements, finalize `requirements.txt` |

## Notes

- Check for ArUco marker **inside** a person's bounding box to flag as "Authorized"
- No badge detected = "Unknown/Intruder"
- Theft = inventory count drops while an unknown person is nearby
