# Hardware & Network Setup

**Owner:** Hardware & Network Specialist

## Folder
```
hardware_setup/
```

## Setup guides

| File | Purpose |
|---|---|
| `cuda_setup_guide.md` | NVIDIA/PyTorch installation, verification, fallback, and thermal rehearsal |
| `camera_setup_guide.md` | USB and vendor-neutral smartphone/IP camera configuration |
| `network_guide.md` | Local smartphone hotspot LAN setup, IP addressing, troubleshooting connection drops |
| `face_database_specs.md` | Face photo collection guidelines, camera positioning, lighting requirements for enrollment |

Use `docs/QA_VALIDATION_RESULTS.md` to record the final automated run and all tests performed on the physical exhibition equipment.

## Also Responsible For

| Folder | Task |
|---|---|
| `assets/known_faces/` | Assist with collecting and organizing face photos for authorized personnel |
| `configs/` | Create and maintain all YAML config files: `camera_config.yaml`, `model_config.yaml`, `alert_config.yaml`, `thresholds.yaml` |

## Day-by-Day Deliverables

| Day | Task |
|---|---|
| 1 | Configure CUDA/NVIDIA drivers, set up smartphone hotspot + DroidCam stream |
| 2 | Run thermal throttling tests under AI load, finalize camera angles for demo |
| 3 | Assist with GPU memory allocation for concurrent YOLO + VLM (no OOM errors) |
| 4 | Stress-test local network, simulate connection drops, measure recovery times |
| 5 | Set up physical exhibition desk, tape wires, calibrate camera focus, secure hotspot |

## Notes

- Keep a USB webcam on standby as backup if hotspot drops
- Camera angle must capture faces clearly at typical standing distance
- Ensure demo lighting is adequate for reliable face recognition
