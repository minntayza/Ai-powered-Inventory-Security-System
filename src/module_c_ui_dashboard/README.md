# Module C: Action & User Interface Layer (Streamlit Dashboard)

**Owner:** Frontend Developer

## Folder
```
src/module_c_ui_dashboard/
```

## Files to Create

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application entry point and page configuration |
| `operational_view.py` | Left panel: live video feed, bounding boxes overlay, real-time inventory count |
| `cognitive_assistant.py` | Right panel: chat box for VLM Q&A, activity feed with timestamped events |
| `alert_ui.py` | Red Alert state UI overlay triggered during theft events |
| `components.py` | Reusable Streamlit UI components (cards, badges, status indicators) |

## Key Integration Points

- **Input from Module A (`src/module_a_perception_engine/`):** Bounding boxes, person authorization status, inventory count
- **Input from Module B (`src/module_b_vlm_layer/`):** VLM text answers for chat display
- **Input from Backend (`src/backend_alerting/`):** Theft event state, alert status, audit log entries
- **Output to Module B:** User questions from chat box (text input)

## Day-by-Day Deliverables

| Day | Task |
|---|---|
| 1 | Initialize Streamlit repository, draft dual-panel wireframe (Left: Ops View, Right: Cognitive Assistant) |
| 2 | Build real-time inventory count widget, develop chat box for VLM interaction |
| 3 | Connect Streamlit chat to VLM pipeline, route VLM responses to UI and TTS audio |
| 4 | Implement Red Alert state UI, finalize timestamped activity feed |
| 5 | Finalize audit trail display, polish UI for exhibition live demo |

## Notes

- Dashboard must show live video with bounding box overlay
- Red Alert = full UI color change + visible theft notification
- Activity feed must be a running, timestamped log visible to the operator
