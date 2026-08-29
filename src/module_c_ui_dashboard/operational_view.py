"""Operational camera and perception view."""

from __future__ import annotations

import cv2
import streamlit as st

from .components import authorization_table, inventory_metrics, status_badge


def render_operational_view(snapshot: dict) -> None:
    if snapshot.get("source", {}).get("type") == "replay":
        st.warning("⚠️ SIMULATION MODE — siren and Telegram notifications are disabled.")
    status_columns = st.columns(4)
    with status_columns[0]:
        status_badge("Engine", bool(snapshot.get("running")))
    with status_columns[1]:
        status_badge("Camera", bool(snapshot.get("camera_healthy")))
    with status_columns[2]:
        model_ready = snapshot.get("model_status") == "ready"
        status_badge(
            "Models" if model_ready else "Models loading",
            model_ready,
        )
    with status_columns[3]:
        sec_state = snapshot.get("security", {}).get("state", "UNKNOWN")
        sec_healthy = sec_state not in ("INCIDENT", "UNKNOWN")
        status_badge(f"Security: {sec_state}", sec_healthy)

    monitoring = snapshot.get("monitoring", {})
    mode = monitoring.get("mode", "DISARMED")
    if mode == "DISARMED":
        st.warning("🔒 **Monitoring is DISARMED.** Arrange protected items in camera view, then click **Set baseline and arm**.")
    elif mode == "PAUSED":
        st.info("⏸️ **Monitoring is PAUSED.** Perception continues but alert triggers are suppressed.")
    else:
        st.success("🛡️ **Monitoring is ARMED.** Inventory counts are actively benchmarked against baseline.")

    device = snapshot.get("device_info") or {}
    if device:
        if str(device.get("selected", "cpu")).startswith("cuda"):
            st.caption(
                f"⚡ **Inference Accelerator:** {device['selected']} — {device.get('gpu_name')} "
                f"({device.get('gpu_memory_gb')} GB)"
            )
        elif device.get("selected") == "mps":
            st.caption("⚡ **Inference Accelerator:** Apple Silicon GPU (MPS)")
        else:
            st.caption("💻 **Inference Accelerator:** CPU Mode (No active CUDA GPU)")

    frame = snapshot.get("frame")
    if frame is not None:
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    else:
        st.warning("📷 Waiting for live camera feed…")

    perception = snapshot.get("perception") or {}
    inventory = perception.get("inventory", {})

    st.markdown("### 📦 Protected Inventory")
    inventory_metrics(inventory.get("protected_counts", inventory.get("stable_counts", {})))
    context = inventory.get("contextual_counts", {})
    if context:
        with st.expander("Context Objects (Non-triggering background objects)"):
            inventory_metrics(context)
    baseline = monitoring.get("baseline_counts", {})
    if baseline:
        items_str = ", ".join(f"**{k.replace('_', ' ').title()}**: {v}" for k, v in baseline.items())
        st.caption(f"🎯 **Armed Baseline:** {items_str}")

    st.markdown("### 👥 Tracked Personnel & Authorization")
    authorization_table(perception.get("persons", []))

    performance = snapshot.get("performance", {})
    if performance:
        with st.expander("⚡ System Performance & Device Health"):
            st.caption(f"Active Device Target: **{performance.get('active_device', 'initializing')}**")
            metrics = st.columns(4)
            metrics[0].metric("Capture FPS", performance.get("capture_fps", 0))
            metrics[1].metric("Processed FPS", performance.get("processed_fps", 0))
            metrics[2].metric("YOLO Latency", f"{performance.get('yolo_ms', 0):.1f} ms")
            metrics[3].metric("Face Latency", f"{performance.get('face_ms', 0):.1f} ms")
            st.caption(
                f"Processing Avg/Max: **{performance.get('processing_ms_avg', 0):.1f} / "
                f"{performance.get('processing_ms_max', 0):.1f} ms** | "
                f"Skipped Frames: **{performance.get('skipped_frames', 0)}** | "
                f"Buffer: **{performance.get('video_buffer_mb', 0):.2f} MB** | "
                f"Last VLM Latency: **{performance.get('vlm_ms', 0):.1f} ms**"
            )
    if snapshot.get("last_error"):
        st.error(snapshot["last_error"])

