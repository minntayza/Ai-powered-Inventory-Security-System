"""Operational camera and perception view."""

from __future__ import annotations

import cv2
import streamlit as st

from .components import authorization_table, inventory_metrics, status_badge


def render_operational_view(snapshot: dict) -> None:
    if snapshot.get("source", {}).get("type") == "replay":
        st.warning("SIMULATION MODE - siren and Telegram notifications are disabled.")
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
        st.markdown(f"**Security:** {snapshot.get('security', {}).get('state', 'UNKNOWN')}")

    monitoring = snapshot.get("monitoring", {})
    mode = monitoring.get("mode", "DISARMED")
    if mode == "DISARMED":
        st.warning("Monitoring is disarmed. Arrange protected items, then set the baseline.")
    elif mode == "PAUSED":
        st.info("Monitoring is paused; perception continues but alerts are suppressed.")
    else:
        st.success("Monitoring is armed with a fixed protected-inventory baseline.")

    device = snapshot.get("device_info") or {}
    if device:
        if str(device.get("selected", "cpu")).startswith("cuda"):
            st.caption(
                f"Inference: {device['selected']} — {device.get('gpu_name')} "
                f"({device.get('gpu_memory_gb')} GB)"
            )
        elif device.get("selected") == "mps":
            st.caption("Inference: mps — Apple Silicon GPU (Metal Performance Shaders)")
        else:
            st.caption("Inference: CPU fallback — no supported GPU accelerator is available")

    frame = snapshot.get("frame")
    if frame is not None:
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    else:
        st.warning("Waiting for the first camera frame…")

    perception = snapshot.get("perception") or {}
    inventory = perception.get("inventory", {})
    st.subheader("Protected inventory")
    inventory_metrics(inventory.get("protected_counts", inventory.get("stable_counts", {})))
    context = inventory.get("contextual_counts", {})
    if context:
        with st.expander("Context objects (do not trigger alerts)"):
            inventory_metrics(context)
    baseline = monitoring.get("baseline_counts", {})
    if baseline:
        st.caption(f"Armed baseline: {baseline}")
    st.subheader("People")
    authorization_table(perception.get("persons", []))
    performance = snapshot.get("performance", {})
    if performance:
        with st.expander("Performance and device health"):
            st.caption(f"Active device: {performance.get('active_device', 'initializing')}")
            metrics = st.columns(4)
            metrics[0].metric("Capture FPS", performance.get("capture_fps", 0))
            metrics[1].metric("Processed FPS", performance.get("processed_fps", 0))
            metrics[2].metric("YOLO", f"{performance.get('yolo_ms', 0):.1f} ms")
            metrics[3].metric("Face", f"{performance.get('face_ms', 0):.1f} ms")
            st.caption(
                f"Processing avg/max: {performance.get('processing_ms_avg', 0):.1f}/"
                f"{performance.get('processing_ms_max', 0):.1f} ms | "
                f"Skipped frames: {performance.get('skipped_frames', 0)} | "
                f"Video buffer: {performance.get('video_buffer_mb', 0):.2f} MB | "
                f"Last VLM: {performance.get('vlm_ms', 0):.1f} ms"
            )
    if snapshot.get("last_error"):
        st.error(snapshot["last_error"])
