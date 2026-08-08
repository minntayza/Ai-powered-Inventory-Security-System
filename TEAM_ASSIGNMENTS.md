# Team Folder Assignments

This document maps each team member to their assigned project folders, expected deliverable files, and the day-by-day timeline from the execution plan.

---

## Folder-to-Member Map

| Folder | Assigned Member | Role |
|---|---|---|
| `src/module_a_perception_engine/` | **CV Engineer** | Computer Vision |
| `src/module_b_vlm_layer/` | **VLM & AI Integration Engineer** | VLM / AI |
| `src/module_c_ui_dashboard/` | **Frontend Developer** | Frontend |
| `src/backend_alerting/` | **Backend & Alerting Engineer** | Backend |
| `src/utils/` | **All Members** (shared) | Shared helpers |
| `hardware_setup/` | **Hardware & Network Specialist** | Hardware / Network |
| `assets/known_faces/` | **CV Engineer** + **Hardware Specialist** | Face database images |
| `assets/audio/` | **Backend & Alerting Engineer** | Siren MP3 |
| `assets/sample_images/` | **CV Engineer** + **VLM Engineer** | Test images |
| `configs/` | **Hardware Specialist** + **Backend Engineer** | YAML configs |
| `tests/` | **PM & QA Lead** | QA / Testing |
| `docs/api_contracts/` | **PM & QA Lead** | Inter-module schemas |
| `docs/presentations/` | **PM & QA Lead** | Pitch deck |
| `data/logs/` | **Backend & Alerting Engineer** | Runtime logs (gitignored) |

---

## 1. CV Engineer

**Primary Folder:** `src/module_a_perception_engine/`

**Expected Files:**
```
src/module_a_perception_engine/
├── __init__.py              (exists)
├── yolo_detector.py         # YOLOv8 inventory counting (bottles, backpacks)
├── face_detector.py         # DeepFace face detection + recognition for authorization
├── face_database.py         # Known face database management
├── person_tracker.py        # Person bounding box + face recognition correlation logic
├── inventory_counter.py     # Dynamic item count output
└── README.md                (exists)
```

**Also Contributes To:**
- `assets/sample_images/` — upload test images of target items
- `assets/known_faces/` — collect and organize face photos for authorized personnel
- `configs/` — provide YOLOv8 confidence thresholds, face recognition settings

**Day-by-Day:**
| Day | Deliverable |
|---|---|
| 1 | Setup YOLOv8 + DeepFace environment, collect face photos, gather sample images |
| 2 | Integrate DeepFace + YOLOv8, output dynamic item counts |
| 3 | Link YOLO/face recognition outputs to backend logic engine |
| 4 | Refine confidence thresholds, fine-tune face recognition accuracy |
| 5 | Code freeze, clean scripts, finalize `requirements.txt` |

---

## 2. VLM & AI Integration Engineer

**Primary Folder:** `src/module_b_vlm_layer/`

**Expected Files:**
```
src/module_b_vlm_layer/
├── __init__.py              (exists)
├── vlm_engine.py            # Florence-2 / Moondream2 model loading & inference
├── vqa_pipeline.py          # Visual Question Answering (image + text prompt -> answer)
├── tts_engine.py            # pyttsx3 Text-to-Speech integration
├── prompt_cache.py          # Prompt caching and frame-skipping optimization
└── README.md                (exists)
```

**Also Contributes To:**
- `assets/sample_images/` — upload static test images for VQA validation
- `configs/` — provide VLM model parameters (resolution, device, max tokens)

**Day-by-Day:**
| Day | Deliverable |
|---|---|
| 1 | Download Florence-2/Moondream2, verify basic prompt-response with static images |
| 2 | Build VLM + pyttsx3 integration, accept cropped frames + text prompts |
| 3 | Connect VLM pipeline to Streamlit chat UI and TTS audio engine |
| 4 | Optimize inference speed, implement prompt caching or frame skipping |
| 5 | Code freeze, clean scripts, finalize `requirements.txt` |

---

## 3. Frontend Developer

**Primary Folder:** `src/module_c_ui_dashboard/`

**Expected Files:**
```
src/module_c_ui_dashboard/
├── __init__.py              (exists)
├── app.py                   # Main Streamlit application entry point
├── operational_view.py      # Left panel: video feed, bounding boxes, inventory count
├── cognitive_assistant.py   # Right panel: chat box, VQA interaction, activity feed
├── alert_ui.py              # Red Alert state UI (theft event overlay)
├── components.py            # Reusable Streamlit UI components
└── README.md                (exists)
```

**Also Contributes To:**
- `docs/api_contracts/` — coordinate with PM on VLM response schema for chat box

**Day-by-Day:**
| Day | Deliverable |
|---|---|
| 1 | Initialize Streamlit repo, draft dual-panel wireframe layout |
| 2 | Build real-time inventory count widget + chat box for VLM interaction |
| 3 | Connect Streamlit chat to VLM pipeline, route responses to UI + TTS |
| 4 | Implement Red Alert state UI, finalize timestamped activity feed |
| 5 | Finalize audit trail display, polish UI for exhibition demo |

---

## 4. Backend & Alerting Engineer

**Primary Folder:** `src/backend_alerting/`

**Expected Files:**
```
src/backend_alerting/
├── __init__.py              (exists)
├── event_loop.py            # Core Python event loop framework
├── theft_detector.py        # Theft detection logic (inventory drop + unknown person)
├── telegram_bot.py          # Telegram Bot API alert payload (photo + text)
├── siren.py                 # MP3 siren trigger
├── audit_logger.py          # CSV / database audit trail logging
├── grace_buffer.py          # Multi-frame buffer (3-sec grace period)
└── README.md                (exists)
```

**Also Contributes To:**
- `assets/audio/` — provide siren MP3 file
- `configs/` — provide Telegram bot token, admin chat ID, alert settings
- `data/logs/` — runtime event logs and audit trail output

**Day-by-Day:**
| Day | Deliverable |
|---|---|
| 1 | Create base event loop framework, register Telegram bot via BotFather |
| 2 | Code theft detection logic, connect MP3 siren trigger |
| 3 | Link YOLO/face recognition states to event engine, ensure correct state tracking |
| 4 | Test Telegram payload (photo + alert sent during breach) |
| 5 | Finalize audit trail (CSV/DB), ensure all events are timestamped and logged |

---

## 5. Hardware & Network Specialist

**Primary Folders:** `hardware_setup/`, `assets/known_faces/`, `configs/`

**Expected Files:**
```
hardware_setup/
├── README.md                (exists)
├── cuda_setup_guide.md      # NVIDIA/CUDA driver installation steps for Legion 5
├── camera_setup_guide.md    # DroidCam/Iriun smartphone IP camera configuration
├── network_guide.md         # Local hotspot LAN setup and troubleshooting
└── face_database_specs.md   # Face photo collection guidelines, camera positioning

assets/known_faces/
└── (face photos organized by person name)

configs/
├── camera_config.yaml       # IP camera stream URL, resolution, FPS
├── model_config.yaml        # YOLOv8 and VLM model parameters
├── alert_config.yaml        # Telegram bot token, admin chat ID, siren settings
└── thresholds.yaml          # Detection confidence, face recognition threshold, grace period duration
```

**Day-by-Day:**
| Day | Deliverable |
|---|---|
| 1 | Configure CUDA/NVIDIA drivers, set up hotspot + DroidCam stream |
| 2 | Test thermal throttling, finalize camera placement angles |
| 3 | Assist with GPU memory allocation for concurrent YOLO + VLM |
| 4 | Stress-test local network, simulate connection drops |
| 5 | Set up physical exhibition desk, tape wires, calibrate camera, secure hotspot |

---

## 6. PM & QA Lead

**Primary Folders:** `tests/`, `docs/api_contracts/`, `docs/presentations/`

**Expected Files:**
```
tests/
├── test_module_a.py         # Unit tests for Perception Engine
├── test_module_b.py         # Unit tests for VLM Layer
├── test_module_c.py         # Unit tests for UI Dashboard
├── test_backend.py          # Unit tests for Backend & Alerting
├── test_integration.py      # End-to-end pipeline test
└── test_qa_scenarios.py     # Simulated incident tests (authorized vs unauthorized)

docs/api_contracts/
├── yolo_output_schema.md    # YOLO bounding box + face recognition -> Backend format
├── vlm_response_schema.md   # VLM answer -> Streamlit UI format
├── alert_payload_schema.md  # Alert data -> Telegram Bot format
└── event_log_schema.md      # Event log -> Database/CSV format

docs/presentations/
└── (pitch deck, slide files)
```

**Also Responsible For:**
- `docs/5-Day Agile Project Execution Plan.md` (already exists)
- Git repository management and merge coordination
- Defining testing criteria and acceptance standards

**Day-by-Day:**
| Day | Deliverable |
|---|---|
| 1 | Kickoff meeting, establish Git repo, define API contracts, draft testing criteria |
| 2 | End-of-day module reviews, identify integration blockers, draft docs + slides |
| 3 | Facilitate integration merge, run first end-to-end test |
| 4 | Lead formal QA sessions, run simulated incident scenarios |
| 5 | Run full-scale dry runs of exhibition pitch, finalize all deliverables |

---

## Shared Folder: `src/utils/`

All members may add shared helpers here:

| File | Who Creates It | Purpose |
|---|---|---|
| `frame_capture.py` | Hardware Specialist | Camera frame grab utility |
| `gpu_manager.py` | Hardware Specialist + CV Engineer | GPU memory allocation helper |
| `config_loader.py` | Backend Engineer | YAML config loading utility |
| `audio_control.py` | VLM Engineer | Microphone muting during siren/TTS playback |
| `logger.py` | Backend Engineer | Centralized logging helper |

---

## Quick Reference: Who Works Where

```
CV Engineer       → src/module_a_perception_engine/   + assets/sample_images/ + assets/known_faces/
VLM Engineer      → src/module_b_vlm_layer/           + assets/sample_images/
Frontend Dev      → src/module_c_ui_dashboard/
Backend Engineer  → src/backend_alerting/             + assets/audio/ + data/logs/
Hardware Spec.    → hardware_setup/ + configs/        + assets/known_faces/ (camera positioning)
PM & QA Lead      → tests/ + docs/api_contracts/      + docs/presentations/
Everyone          → src/utils/
```
