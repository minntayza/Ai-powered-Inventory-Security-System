"""Streamlit entry point for the inventory security dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

# The `streamlit` console launcher places this file's directory on sys.path but
# not the repository root. Add the root before importing the local `src` package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.backend_alerting.event_loop import SystemController
from src.module_b_vlm_layer.tts_engine import TTSEngine
from src.module_b_vlm_layer.vlm_engine import FlorenceEngine
from src.module_b_vlm_layer.vqa_pipeline import VQAPipeline

from src.module_c_ui_dashboard.alert_ui import render_alert
from src.module_c_ui_dashboard.cognitive_assistant import render_assistant
from src.module_c_ui_dashboard.operational_view import render_operational_view


st.set_page_config(page_title="Inventory Security", page_icon="🛡️", layout="wide")


@st.cache_resource
def get_controller() -> SystemController:
    return SystemController().start()


@st.cache_resource
def get_vqa(_controller: SystemController) -> tuple[VQAPipeline, TTSEngine]:
    config = _controller.config["models"]["vlm"]
    engine = FlorenceEngine(
        model_name=config.get("model_name", "microsoft/Florence-2-base-ft"),
        device=config.get("device", "auto"),
        max_new_tokens=int(config.get("max_new_tokens", 128)),
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )
    return VQAPipeline(engine), TTSEngine(_controller.audio)


controller = get_controller()
vqa, tts = get_vqa(controller)

st.title("AI-Powered Inventory Security System")

with st.sidebar:
    st.header("Controls")
    monitoring = controller.snapshot().get("monitoring", {})
    mode = monitoring.get("mode", "DISARMED")
    st.caption(f"Monitoring: {mode}")
    if mode == "DISARMED":
        if st.button(
            "Set baseline and arm",
            use_container_width=True,
        ):
            result = controller.set_baseline_and_arm()
            (st.success if result["ok"] else st.warning)(result["message"])
            st.rerun()
        if not monitoring.get("baseline_ready", False):
            st.caption(
                "Waiting for a stable protected item inside the monitored zone. "
                "You may click the button to re-check readiness."
            )
    elif mode == "ARMED":
        if st.button("Pause monitoring", use_container_width=True):
            result = controller.pause_monitoring()
            (st.success if result["ok"] else st.warning)(result["message"])
            st.rerun()
    else:
        if st.button("Resume monitoring", use_container_width=True):
            result = controller.resume_monitoring()
            (st.success if result["ok"] else st.warning)(result["message"])
            st.rerun()
    if st.button("Reset baseline", use_container_width=True):
        result = controller.reset_baseline()
        st.success(result["message"])
        st.rerun()

    with st.expander("Monitored zone"):
        active_region = monitoring.get("shelf_region") or [0.0, 0.0, 1.0, 1.0]
        x1 = st.slider("Left", 0.0, 0.95, float(active_region[0]), 0.01)
        y1 = st.slider("Top", 0.0, 0.95, float(active_region[1]), 0.01)
        x2 = st.slider("Right", 0.05, 1.0, float(active_region[2]), 0.01)
        y2 = st.slider("Bottom", 0.05, 1.0, float(active_region[3]), 0.01)
        zone_columns = st.columns(2)
        if zone_columns[0].button("Save zone", use_container_width=True):
            try:
                result = controller.set_shelf_region([x1, y1, x2, y2])
                st.success(result["message"])
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if zone_columns[1].button("Full frame", use_container_width=True):
            result = controller.set_shelf_region(None)
            st.success(result["message"])
            st.rerun()

    with st.expander("Camera / replay source"):
        source = controller.snapshot().get("source", {})
        st.caption(f"Active: {source.get('label', 'Live camera')}")
        replay_file = st.file_uploader(
            "Demo recording", type=["mp4", "avi", "mov"], key="replay-video"
        )
        replay_loop = st.checkbox("Loop replay", value=False)
        if st.button(
            "Start safe replay",
            disabled=replay_file is None,
            use_container_width=True,
        ):
            try:
                result = controller.switch_to_replay(
                    replay_file.name, replay_file.getvalue(), loop=replay_loop
                )
                (st.success if result["ok"] else st.error)(result["message"])
                if result["ok"]:
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not start replay: {exc}")
        if source.get("type") == "replay" and st.button(
            "Return to live camera", use_container_width=True
        ):
            result = controller.switch_to_live()
            st.success(result["message"])
            st.rerun()

    st.toggle("Speak VLM answers", key="speak_answers")
    if st.button("Stop siren"):
        controller.siren.stop()
    if st.button("Refresh authorized faces"):
        tracker = controller.tracker
        if tracker is not None:
            tracker.face_detector.refresh_database()
            st.success("Face folders reloaded")
    with st.expander("Authorized face enrollment"):
        face_name = st.text_input("Person name", key="face-name")
        face_uploads = st.file_uploader(
            "1-5 clear face images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="face-images",
        )
        if st.button("Validate and enroll", use_container_width=True):
            uploads = [(item.name, item.getvalue()) for item in face_uploads]
            result = controller.enroll_face(face_name, uploads)
            (st.success if result["ok"] else st.error)(result["message"])
        identities = controller.known_faces()
        if identities:
            st.caption(
                "Enrolled: " + ", ".join(
                    f"{name} ({count})" for name, count in identities.items()
                )
            )
            remove_name = st.selectbox("Remove identity", list(identities), key="remove-face")
            confirm_remove = st.checkbox(
                f"Confirm permanent removal of {remove_name}", key="confirm-remove-face"
            )
            if st.button(
                "Remove selected identity",
                disabled=not confirm_remove,
                use_container_width=True,
            ):
                result = controller.remove_face(remove_name)
                (st.success if result["ok"] else st.error)(result["message"])
                if result["ok"]:
                    st.rerun()
    st.caption("Telegram: " + ("enabled" if controller.telegram.enabled else "disabled"))

left, right = st.columns([3, 2])
with left:
    @st.fragment(run_every=0.5)
    def live_operational_panel() -> None:
        current = controller.snapshot()
        render_alert(current, controller)
        render_operational_view(current)

    live_operational_panel()
with right:
    render_assistant(controller, vqa, tts)
    st.subheader("Recent activity")
    events = controller.recent_events(20)
    if events:
        st.dataframe(events, hide_index=True, use_container_width=True)
    else:
        st.caption("No events recorded")

if st.button("Refresh dashboard", use_container_width=True):
    st.rerun()
