# Local Network Guide

A network is needed only for an IP camera or Telegram delivery. Detection, recognition, audit logging, the dashboard, siren, VQA, voice transcription, and TTS run locally after model weights are available.

## Exhibition LAN

1. Create a private phone hotspot or use a trusted local router.
2. Connect the camera phone and laptop to the same network.
3. Disable automatic switching to unrelated Wi-Fi networks during the demo.
4. Note the camera phone's current LAN address and stream URL; hotspot addresses can change after reconnecting.
5. Verify the stream from the laptop before starting Streamlit.

Avoid exposing the camera stream to a public network. Use a strong hotspot password and disconnect it after the demonstration.

## Telegram

Telegram needs outbound internet access plus both environment variables in `.env.example`. It must also be enabled in `configs/alert_config.yaml`. Never commit the bot token or chat ID.

Uploaded-video replay is simulation mode: the controller forcibly suppresses Telegram and siren delivery even if configuration enables them.

## Diagnosing a dropped stream

- Confirm both devices still show the same network name.
- Recheck the phone's LAN address and update `source` if it changed.
- Confirm the phone camera app is awake and still serving video.
- Move the hotspot/router closer and disconnect unnecessary clients.
- Wait for the application's reconnect backoff (1–10 seconds) before restarting it.
- If recovery is unreliable, switch to the USB camera fallback.

## Offline demonstration plan

Before arriving, download required model weights and run each model once. Keep Telegram optional, use a USB camera, and demonstrate the complete local path: frame → detection/decision → SQLite evidence → dashboard alert. This remains functional without internet access.

## Stress-test record

During rehearsal, record in `docs/QA_VALIDATION_RESULTS.md`:

- network type and approximate distance;
- 20-minute stream stability;
- recovery after a 10-second disconnect;
- whether the stream URL changed;
- Telegram sent/retry/disabled state;
- USB fallback time.
