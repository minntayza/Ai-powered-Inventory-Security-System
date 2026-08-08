"""Florence-2 incident investigation interface."""

from __future__ import annotations

from pathlib import Path

import cv2
import streamlit as st


def render_assistant(controller, vqa, tts) -> None:
    st.subheader("Visual assistant")
    events = [event for event in controller.recent_events(20) if event.get("snapshot_path")]
    source_options = {"Live camera": None}
    source_options.update(
        {
            f"{event['timestamp']} - {event['event_type']}": event
            for event in events
        }
    )
    selected_label = st.selectbox("Image source", list(source_options))
    selected = source_options[selected_label]
    if selected and selected.get("video_path"):
        st.video(selected["video_path"])

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    actions = st.columns(3)
    action = None
    if actions[0].button("Describe scene", use_container_width=True):
        action = "describe"
    if actions[1].button("Read text", use_container_width=True):
        action = "ocr"
    if actions[2].button(
        "Summarize incident", disabled=selected is None, use_container_width=True
    ):
        action = "summary"
    typed_question = st.chat_input("Ask what is visible in the selected image")
    question = typed_question or {
        "describe": "Describe the scene in detail",
        "ocr": "Read all visible text",
        "summary": "Summarize this incident",
    }.get(action)
    if not question:
        return
    st.session_state.chat_messages.append({"role": "user", "content": question})

    if selected is None:
        image = controller.snapshot().get("frame")
        source_id = "live"
    else:
        path = Path(selected["snapshot_path"])
        image = cv2.imread(str(path)) if path.is_file() else None
        source_id = selected["event_id"]
    with st.spinner("Florence-2 is examining the image..."):
        if action == "summary" and selected is not None:
            response = vqa.summarize_incident(image, selected)
        else:
            response = vqa.ask(image, question, source_id=source_id)
    answer = response["answer"] if response["ok"] else f"VLM unavailable: {response['error']}"
    controller.record_vlm_latency(response.get("latency_ms", 0.0))
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    if response["ok"] and st.session_state.get("speak_answers", False):
        tts.speak(answer)
    st.rerun()
