# QA Validation Results

This document separates automated verification from physical-hardware checks. A hardware item is not considered passed until it is run on the exhibition equipment.

## Automated verification

Run from the repository root:

```bash
.venv/bin/python -m pytest -q
```

Covered behaviors include:

- camera read failure, source reopen, and recovery;
- frame processing recovery after a perception exception;
- face-recognition and TTS degraded-health reporting;
- authorized/unknown decision behavior and replay-mode alert suppression;
- baseline reset when sources change or monitoring is disarmed;
- incident image/video persistence, AVI fallback, audit migration, and retention cleanup;
- Telegram retry status and siren/microphone coordination;
- voice transcription validation, microphone locking, and assistant question precedence;
- VQA caching and full-response preservation;
- automatic background incident summaries, SQLite report persistence, and live report display;
- enrollment validation and dashboard live-frame rendering;
- CUDA memory-pressure reporting and safe accelerator-cache release.

The final command result and date should be recorded here after the release run:

| Check | Result | Evidence |
|---|---|---|
| Full pytest suite | **Pass — 87 tests** | `.venv/bin/python -m pytest -q` on 2026-08-30 |
| Environment verifier | **Pass — CPU selected** | `.venv/bin/python scripts/verify_environment.py` on 2026-08-29 |

The environment run reported that OpenCV and pygame both bundle SDL classes on
macOS. It did not fail verification, but the live siren/camera combination should
be included in the Mac rehearsal because duplicate SDL libraries can cause GUI or
audio instability. CUDA and MPS were unavailable in this specific verification
run; that does not validate the exhibition laptop's NVIDIA configuration.

## Manual exhibition-hardware validation

| Scenario | Status | Acceptance criterion |
|---|---|---|
| NVIDIA acceleration | Not run | Dashboard reports CUDA and stable memory during a 20-minute run |
| USB camera | Not run | 960×540 live view remains responsive near 10 FPS |
| IP camera reconnect | Not run | A 10-second interruption shows degraded state and then recovers |
| Authorized removal | Not run | Authorized identity is shown and no theft alert is confirmed |
| Unknown removal | Not run | Red alert appears with evidence and configured local/Telegram status |
| Face grace period | Not run | Brief face loss does not immediately create a false incident |
| Voice question | Not run | Recorded speech becomes text and produces a local VQA response |
| Siren/microphone arbitration | Not run | Voice capture is blocked while the siren is active |
| Offline fallback | Not run | USB camera and local audit/dashboard operate without internet |
| Retention/privacy | Not run | Evidence cleanup works and enrolled faces are stored only as intended |

For each manual run, add the date, machine, camera source, operator, observed result, and links/paths to logs or captured evidence. Do not put Telegram credentials or biometric images in this document.
