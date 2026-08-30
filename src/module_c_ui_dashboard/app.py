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
from src.module_b_vlm_layer.voice_input import VoiceQuestionTranscriber

from src.module_c_ui_dashboard.alert_ui import render_alert
from src.module_c_ui_dashboard.cognitive_assistant import render_assistant
from src.module_c_ui_dashboard.operational_view import (
    render_live_frame,
    render_operational_details,
    render_operational_status,
)


st.set_page_config(page_title="Inventory Security", page_icon="🛡️", layout="wide")



@st.cache_resource
def get_controller() -> SystemController:
    return SystemController().start()


@st.cache_resource
def get_vqa(
    _controller: SystemController,
) -> tuple[VQAPipeline, TTSEngine, VoiceQuestionTranscriber]:
    config = _controller.config["models"]["vlm"]
    engine = FlorenceEngine(
        model_name=config.get("model_name", "microsoft/Florence-2-base-ft"),
        device=config.get("device", "auto"),
        max_new_tokens=int(config.get("max_new_tokens", 128)),
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )
    vqa = VQAPipeline(engine)
    _controller.set_incident_summarizer(vqa.summarize_incident)
    return vqa, TTSEngine(_controller.audio), VoiceQuestionTranscriber()


controller = get_controller()
vqa, tts, voice = get_vqa(controller)

# Header Banner
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.2rem; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 0.8rem;">
        <div>
            <div style="font-size: 2rem; font-weight: 800; background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🛡️ AI-Powered Inventory Security System
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 2px;">
                YOLO Perception • DeepFace Authorization • Deterministic Theft Engine • Florence-2 VLM
            </div>
        </div>
        <div>
            <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); color: #34d399; font-size: 0.75rem; font-weight: 700; padding: 6px 14px; border-radius: 20px; letter-spacing: 0.5px;">
                ● SYSTEM ACTIVE
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Controls")
    sidebar_snapshot = controller.snapshot(
        include_frame=False, include_perception=False
    )
    monitoring = sidebar_snapshot.get("monitoring", {})
    mode = monitoring.get("mode", "DISARMED")
    st.caption(f"Status: **{mode}**")
    current_counts = monitoring.get("current_counts", {})
    st.caption(f"Stable protected count: {current_counts or 'not ready'}")
    if mode == "DISARMED":
        st.info(
            "1. Put the protected items inside the monitored zone. "
            "2. Stop handling them and step away. 3. Wait for the count to settle."
        )
        if st.button(
            "🛡️ Set baseline and arm",
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
        st.success(
            "Monitoring is armed. The next stable removal creates one event, "
            "then monitoring pauses automatically."
        )
        if st.button("⏸️ Pause monitoring", use_container_width=True):
            result = controller.pause_monitoring()
            (st.success if result["ok"] else st.warning)(result["message"])
            st.rerun()
    elif monitoring.get("pause_reason") == "event_recorded":
        st.warning(
            "A removal was recorded and this baseline is paused. This prevents "
            "the same missing item from later becoming a second, contradictory alert."
        )
        if st.button("Item returned — resume original baseline", use_container_width=True):
            result = controller.resume_monitoring()
            (st.success if result["ok"] else st.warning)(result["message"])
            st.rerun()
        if st.button("Use current count as the new baseline", use_container_width=True):
            result = controller.set_baseline_and_arm()
            (st.success if result["ok"] else st.warning)(result["message"])
            st.rerun()
    else:
        st.info("Monitoring was paused manually; perception is still running.")
        if st.button("▶️ Resume monitoring", use_container_width=True):
            result = controller.resume_monitoring()
            (st.success if result["ok"] else st.warning)(result["message"])
            st.rerun()
    if st.button("🔄 Clear baseline and start over", use_container_width=True):
        result = controller.reset_baseline()
        st.success(result["message"])
        st.rerun()

    with st.expander("🎯 Monitored Zone Selector"):
        active_region = monitoring.get("shelf_region") or [0.0, 0.0, 1.0, 1.0]
        x1 = st.slider("Left Boundary", 0.0, 0.95, float(active_region[0]), 0.01)
        y1 = st.slider("Top Boundary", 0.0, 0.95, float(active_region[1]), 0.01)
        x2 = st.slider("Right Boundary", 0.05, 1.0, float(active_region[2]), 0.01)
        y2 = st.slider("Bottom Boundary", 0.05, 1.0, float(active_region[3]), 0.01)
        zone_columns = st.columns(2)
        if zone_columns[0].button("💾 Save zone", use_container_width=True):
            try:
                result = controller.set_shelf_region([x1, y1, x2, y2])
                st.success(result["message"])
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if zone_columns[1].button("🖼️ Full frame", use_container_width=True):
            result = controller.set_shelf_region(None)
            st.success(result["message"])
            st.rerun()

    with st.expander("📹 Camera / Replay Source"):
        source = sidebar_snapshot.get("source", {})
        st.caption(f"Active Source: **{source.get('label', 'Live camera')}**")
        replay_file = st.file_uploader(
            "Demo recording", type=["mp4", "avi", "mov"], key="replay-video"
        )
        replay_loop = st.checkbox("Loop replay", value=False)
        if st.button(
            "▶️ Start safe replay",
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
            "📷 Return to live camera", use_container_width=True
        ):
            result = controller.switch_to_live()
            st.success(result["message"])
            st.rerun()

    st.toggle("🔊 Speak VLM answers", key="speak_answers")
    if st.button("🔇 Stop siren", use_container_width=True):
        controller.siren.stop()
    if st.button("🔄 Refresh face DB", use_container_width=True):
        tracker = controller.tracker
        if tracker is not None:
            tracker.face_detector.refresh_database()
            st.success("Face folders reloaded")
    with st.expander("👤 Authorized Face Enrollment"):
        face_name = st.text_input("Person Name", key="face-name")
        face_uploads = st.file_uploader(
            "1-5 clear face images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="face-images",
        )
        if st.button("✅ Validate and enroll", use_container_width=True):
            uploads = [(item.name, item.getvalue()) for item in face_uploads]
            result = controller.enroll_face(face_name, uploads)
            (st.success if result["ok"] else st.error)(result["message"])
        identities = controller.known_faces()
        if identities:
            st.caption(
                "Enrolled Identities: " + ", ".join(
                    f"**{name}** ({count})" for name, count in identities.items()
                )
            )
            remove_name = st.selectbox("Remove identity", list(identities), key="remove-face")
            confirm_remove = st.checkbox(
                f"Confirm removal of {remove_name}", key="confirm-remove-face"
            )
            if st.button(
                "🗑️ Remove selected identity",
                disabled=not confirm_remove,
                use_container_width=True,
            ):
                result = controller.remove_face(remove_name)
                (st.success if result["ok"] else st.error)(result["message"])
                if result["ok"]:
                    st.rerun()
    if controller.telegram.enabled and controller.telegram.configured:
        st.caption("Telegram: ready for authorized removal and suspected theft")
    elif controller.telegram.enabled:
        st.warning(
            "Telegram routing is enabled, but TELEGRAM_BOT_TOKEN and "
            "TELEGRAM_CHAT_ID are not available to this process."
        )
    else:
        st.caption("Telegram: disabled")

left, right = st.columns([3, 2])
with left:
    @st.fragment(run_every=0.5)
    def operational_status_panel() -> None:
        current = controller.snapshot(
            include_frame=False, include_perception=False
        )
        render_alert(current, controller)
        render_operational_status(current)

    operational_status_panel()

    camera_fps = max(1.0, float(controller.config["camera"].get("fps", 10)))

    @st.fragment(run_every=max(0.05, 1.0 / camera_fps))
    def live_camera_panel() -> None:
        render_live_frame(controller.camera_preview())

    live_camera_panel()

    @st.fragment(run_every=0.5)
    def operational_details_panel() -> None:
        render_operational_details(controller.snapshot(include_frame=False))

    operational_details_panel()
with right:
    render_assistant(controller, vqa, tts, voice)

    @st.fragment(run_every=0.5)
    def recent_activity_panel() -> None:
        st.markdown("### 📋 Recent Activity Logs")
        events = controller.recent_events(20)
        if events:
            st.dataframe(events, hide_index=True, use_container_width=True)
        else:
            st.caption("No security events recorded yet")

    recent_activity_panel()

if st.button("🔄 Refresh Dashboard", use_container_width=True):
    st.rerun()
