"""Security alert and acknowledgement UI."""

from __future__ import annotations

import streamlit as st


def apply_alert_theme() -> None:
    """Apply a visible page-wide theme while an incident awaits acknowledgement."""
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #2b0508;
            background-image: linear-gradient(180deg, #4a080d 0%, #180305 100%);
        }
        [data-testid="stSidebar"] {
            background-color: #30070a;
        }
        [data-testid="stHeader"] {
            background-color: rgba(74, 8, 13, 0.96);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_alert(snapshot: dict, controller) -> None:
    security = snapshot.get("security", {})
    event = security.get("last_event")
    if not event or event.get("status") != "confirmed" or event.get("acknowledged"):
        return
    apply_alert_theme()
    removed = ", ".join(
        f"{name}: {count}" for name, count in event.get("removed_items", {}).items()
    ) or "unknown item"
    st.error(f"🚨 SUSPECTED THEFT — Removed: {removed}")
    st.caption(f"Event {event['event_id']} at {event['timestamp']}")
    st.caption(f"Telegram delivery: {event.get('telegram_status', 'pending')}")
    actor = event.get("primary_actor")
    if actor:
        st.caption(
            f"Likely actor: {actor.get('name', 'Unknown')} "
            f"(track {actor.get('track_id')}, score {actor.get('association_score', 0):.2f})"
        )
    summary_status = event.get("summary_status")
    if summary_status == "completed" and event.get("ai_summary"):
        st.info(f"AI incident report: {event['ai_summary']}")
    elif summary_status == "pending":
        st.caption("AI incident report: generating...")
    elif summary_status == "failed":
        st.warning(
            "AI incident report unavailable: "
            f"{event.get('summary_error') or 'unknown error'}"
        )
    if event.get("video_path"):
        st.video(event["video_path"])
    if st.button("Acknowledge alert and stop siren", type="primary", key="ack-alert"):
        controller.acknowledge(event["event_id"])
        st.rerun()
