"""Operational camera and perception view."""

from __future__ import annotations

import base64

import cv2
import streamlit as st

from .components import authorization_table, inventory_metrics, status_badge


def encode_frame_data_uri(frame) -> str:
    encoded, jpeg = cv2.imencode(".jpg", frame)
    if not encoded:
        raise ValueError("Could not encode the live camera frame")
    payload = base64.b64encode(jpeg).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def format_device_status(device: dict) -> str:
    """Format accelerator and memory health for the operator."""
    selected = str(device.get("selected", "cpu"))
    if selected.startswith("cuda"):
        memory = device.get("memory") or {}
        if memory:
            status = (
                f"Inference: {selected} — {device.get('gpu_name')} "
                f"({memory.get('reserved_gb', 0)}/{memory.get('total_gb', 0)} GB reserved)"
            )
            if memory.get("pressure") == "high":
                status += " — HIGH MEMORY PRESSURE"
            return status
        return (
            f"Inference: {selected} — {device.get('gpu_name')} "
            f"({device.get('gpu_memory_gb')} GB)"
        )
    if selected == "mps":
        return "Inference: mps — Apple Silicon GPU (Metal Performance Shaders)"
    return "Inference: CPU fallback — no supported GPU accelerator is available"


def component_health_messages(component_health: dict) -> list[str]:
    """Return operator-facing messages for degraded perception components."""
    messages = []
    face = component_health.get("face") or {}
    if face and not face.get("healthy", True):
        messages.append(
            f"Face recognition degraded: {face.get('last_error') or 'unknown error'}"
        )
    return messages


def inventory_status_rows(current_counts: dict, baseline_counts: dict) -> list[dict]:
    """Build explicit operator-facing inventory statuses against the armed baseline."""
    rows = []
    for label in sorted(set(current_counts) | set(baseline_counts)):
        current = int(current_counts.get(label, 0))
        baseline = int(baseline_counts.get(label, 0))
        difference = current - baseline
        if difference < 0:
            status = f"Missing ({abs(difference)})"
        elif difference > 0:
            status = f"Discrepancy (+{difference})"
        else:
            status = "Present"
        rows.append(
            {
                "Item": label.replace("_", " ").title(),
                "Current": current,
                "Baseline": baseline,
                "Difference": difference,
                "Status": status,
            }
        )
    return rows


def render_inventory_status(current_counts: dict, baseline_counts: dict) -> None:
    """Render the current, baseline, and discrepancy state of every protected item."""
    st.dataframe(
        inventory_status_rows(current_counts, baseline_counts),
        hide_index=True,
        use_container_width=True,
    )


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
        st.caption(format_device_status(device))
    for message in component_health_messages(snapshot.get("component_health") or {}):
        st.error(message)

    frame = snapshot.get("frame")
    if frame is not None:
        frame_uri = encode_frame_data_uri(frame)
        st.markdown(
            f'<img src="{frame_uri}" alt="Live camera frame" '
            'style="display:block;width:100%;height:auto">',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Waiting for the first camera frame…")

    perception = snapshot.get("perception") or {}
    inventory = perception.get("inventory", {})
    st.subheader("Protected inventory")
    protected_counts = inventory.get(
        "protected_counts", inventory.get("stable_counts", {})
    )
    inventory_metrics(protected_counts)
    context = inventory.get("contextual_counts", {})
    if context:
        with st.expander("Context objects (do not trigger alerts)"):
            inventory_metrics(context)
    baseline = monitoring.get("baseline_counts", {})
    if baseline:
        render_inventory_status(protected_counts, baseline)
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
