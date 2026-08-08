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
| `face_detector.py` | DeepFace face detection + recognition to identify authorized personnel |
| `face_database.py` | Known face database management (load, encode, match face embeddings) |
| `person_tracker.py` | Correlate person bounding boxes with face recognition results (authorized vs unknown) |
| `inventory_counter.py` | Track and output dynamic item counts on shelves, detect count drops |

## Key Integration Points

- **Output to Backend (`src/backend_alerting/`):** Person ID, authorization status (face recognized: yes/no), current inventory count, bounding box coordinates
- **Input from `src/utils/frame_capture.py`:** Camera frames
- **Shared config:** `configs/thresholds.yaml` — YOLOv8 confidence thresholds, face recognition similarity threshold

## Day-by-Day Deliverables

| Day | Task |
|---|---|
| 1 | Setup YOLOv8 + DeepFace environment, collect face photos for known personnel, gather sample images |
| 2 | Integrate DeepFace recognition inside person bounding boxes, output dynamic item counts |
| 3 | Link YOLO + face recognition output states to backend logic engine |
| 4 | Refine confidence thresholds, fine-tune face recognition accuracy under varying lighting |
| 5 | Code freeze, remove debug statements, finalize `requirements.txt` |

## Notes

- Check for recognized face **inside** a person's bounding box to flag as "Authorized"
- No face recognized = "Unknown/Intruder"
- Theft = inventory count drops while an unknown person is nearby
