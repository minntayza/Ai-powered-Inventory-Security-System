# Utility & Shared Helpers

**Owner:** All Members (shared across modules)

## Folder
```
src/utils/
```

## Files to Create

| File | Who Creates It | Purpose |
|---|---|---|
| `frame_capture.py` | Hardware Specialist | Grab frames from IP camera stream (DroidCam/Iriun) or USB webcam fallback |
| `gpu_manager.py` | Hardware Specialist + CV Engineer | GPU memory allocation helper, prevent OOM when running YOLO + VLM concurrently |
| `config_loader.py` | Backend Engineer | Load and validate YAML config files from `configs/` directory |
| `audio_control.py` | VLM Engineer | Software microphone muting while siren or TTS is playing (prevent feedback loops) |
| `logger.py` | Backend Engineer | Centralized logging helper with consistent format and file output |

## Key Integration Points

- `frame_capture.py` is used by **Module A** and **Module B** as the primary frame source
- `gpu_manager.py` is critical for **Day 3 integration** when all models run concurrently
- `audio_control.py` must be called by **Module B** (TTS playback) and **Backend** (siren playback)
- `config_loader.py` is used by all modules to read settings from `configs/`
- `logger.py` replaces `print()` statements across all modules

## Notes

- These files are shared — coordinate before modifying
- Keep functions stateless and well-documented so all members can use them
- Import from `src.utils` (e.g., `from src.utils.frame_capture import grab_frame`)
