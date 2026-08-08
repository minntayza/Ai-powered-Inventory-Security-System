"""Operational camera and perception view."""

from __future__ import annotations

import cv2
import streamlit as st

from .components import authorization_table, inventory_metrics, status_badge


def render_operational_view(snapshot: dict) -> None:
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
    st.subheader("Stable inventory")
    inventory_metrics(inventory.get("stable_counts", {}))
    st.subheader("People")
    authorization_table(perception.get("persons", []))
    if snapshot.get("last_error"):
        st.error(snapshot["last_error"])
