# Camera Setup Guide

The application accepts either a local webcam index or an OpenCV-compatible network stream URL through `configs/camera_config.yaml`.

## USB or built-in camera

Use the default source for the first available camera:

```yaml
source: 0
width: 960
height: 540
fps: 10
```

If the wrong camera opens, try `source: 1` and then `source: 2`. Close other programs that may have exclusive camera access.

## Smartphone/IP camera

1. Install and start the chosen camera application on the phone and its required desktop companion, if any.
2. Connect the phone and laptop to the same trusted local network.
3. Obtain the stream URL exposed by that application.
4. Confirm that the URL opens from the laptop using the vendor's viewer or a local media player.
5. Replace `source: 0` with the quoted URL:

```yaml
source: "http://PHONE_ADDRESS:PORT/STREAM_PATH"
```

The exact port and path are camera-app specific; copy them from the app rather than guessing.

## Placement and calibration

- Mount the camera so the monitored shelf and approaching person's upper body are visible.
- Avoid strong backlight, reflective glare, and moving backgrounds.
- Keep faces at least 50 pixels wide; this is the configured `min_face_size`.
- Use the dashboard's **Monitored zone** tool to limit decisions to the shelf or desk.
- Arrange the inventory, wait for counts to stabilize, then select **Set baseline and arm**.
- Pause/disarm before moving the camera or intentionally rearranging items.

## Recovery behavior

The capture service retries failed reads, reopens the source, and backs off from 1 to 10 seconds. During recovery the dashboard can report a degraded camera state. If the network stream remains unavailable, switch `source` back to a USB camera index and restart the dashboard.

## Pre-demo checklist

- Live frames update without another camera application running.
- The configured resolution is close to 960×540 and throughput is stable near 10 FPS.
- A face is large and well lit enough for enrollment/recognition.
- The monitored zone includes protected items but excludes irrelevant movement.
- Unplugging or interrupting the camera produces a degraded state and recovery after reconnection.
- A USB webcam is connected or ready as the offline fallback.
