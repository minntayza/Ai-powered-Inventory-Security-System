# Backend & Alerting Engine

**Owner:** Backend & Alerting Engineer

## Folder
```
src/backend_alerting/
```

## Files to Create

| File | Purpose |
|---|---|
| `event_loop.py` | Core Python event loop framework that orchestrates all module states |
| `theft_detector.py` | Theft detection logic: flag event when inventory drops + unknown person nearby |
| `telegram_bot.py` | Telegram Bot API integration: send photo + text alert to admin phone |
| `siren.py` | MP3 siren audio trigger using local speakers |
| `audit_logger.py` | CSV / database audit trail: log every event with timestamps and images |
| `grace_buffer.py` | Multi-frame buffer (3-second grace period) to prevent false positive theft alerts |

## Key Integration Points

- **Input from Module A (`src/module_a_perception_engine/`):** Person authorization status, inventory count, bounding box data
- **Output to Module C (`src/module_c_ui_dashboard/`):** Theft event state (for Red Alert UI), audit log entries (for activity feed)
- **Output to `assets/audio/`:** Play siren MP3 file
- **Output to `data/logs/`:** Write runtime event logs and CSV audit trail
- **Shared config:** `configs/alert_config.yaml` — Telegram bot token, admin chat ID, siren settings

## Day-by-Day Deliverables

| Day | Task |
|---|---|
| 1 | Create base Python event loop framework, register Telegram bot via BotFather, test text delivery |
| 2 | Code automated theft detection logic, connect MP3 siren trigger |
| 3 | Link YOLO + face recognition states to event engine, ensure correct state change tracking |
| 4 | Test Telegram payload: verify photo + alert sent to admin during breach |
| 5 | Finalize comprehensive audit trail (CSV/DB), ensure all events are timestamped and logged |

## Notes

- Theft condition: inventory count drops AND no recognized face detected on nearby person
- Grace buffer: 3-second window to re-detect face before confirming theft (prevents false alarms)
- Telegram payload must include a photo of the incident frame + text alert
- Siren must coordinate with `src/utils/audio_control.py` to mute mic during playback
