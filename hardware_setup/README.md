# Hardware & Network Setup

**Owner:** Hardware & Network Specialist

## Folder
```
hardware_setup/
```

## Files to Create

| File | Purpose |
|---|---|
| `cuda_setup_guide.md` | Step-by-step NVIDIA/CUDA driver installation and verification for Lenovo Legion 5 |
| `camera_setup_guide.md` | DroidCam/Iriun smartphone IP camera configuration, stream URL, resolution settings |
| `network_guide.md` | Local smartphone hotspot LAN setup, IP addressing, troubleshooting connection drops |
| `aruco_badge_specs.md` | ArUco marker dictionary ID, badge print dimensions, chest placement instructions |

## Also Responsible For

| Folder | Task |
|---|---|
| `assets/aruco_markers/` | Generate and save printable ArUco badge PNG images |
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
- ArUco badges must be clearly visible to camera when worn on chest
- Camera angle must ensure ArUco markers are always readable under demo lighting
