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
    st.toggle("Speak VLM answers", key="speak_answers")
    if st.button("Stop siren"):
        controller.siren.stop()
    if st.button("Refresh authorized faces"):
        tracker = controller.tracker
        if tracker is not None:
            tracker.face_detector.refresh_database()
            st.success("Face folders reloaded")
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
