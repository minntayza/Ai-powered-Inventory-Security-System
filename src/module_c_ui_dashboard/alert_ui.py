"""Security alert and acknowledgement UI."""

from __future__ import annotations

import streamlit as st


def render_alert(snapshot: dict, controller) -> None:
    security = snapshot.get("security", {})
    event = security.get("last_event")
    if not event or event.get("status") != "confirmed" or event.get("acknowledged"):
        return
    removed = ", ".join(
        f"{name}: {count}" for name, count in event.get("removed_items", {}).items()
    ) or "unknown item"
    actor = event.get("primary_actor")
    actor_str = ""
    if actor:
        actor_str = (
            f"• Suspect: <strong>{actor.get('name', 'Unknown')}</strong> "
            f"(Track #{actor.get('track_id')}, confidence {actor.get('association_score', 0):.2f})"
        )

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.25), rgba(185, 28, 28, 0.35));
            border: 2px solid #ef4444;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
        ">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span style="font-size: 24px;">🚨</span>
                <span style="font-size: 18px; font-weight: 800; color: #fca5a5; letter-spacing: 0.5px;">SUSPECTED THEFT ALERT</span>
            </div>
            <div style="font-size: 15px; color: #ffffff; margin-bottom: 6px;">
                <strong>Missing Inventory:</strong> <span style="color: #fef08a; font-weight: 700; font-size: 16px;">{removed}</span>
            </div>
            <div style="font-size: 12px; color: #fca5a5;">
                Event ID: <strong>{event['event_id']}</strong> • {event['timestamp']} {actor_str}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if event.get("video_path"):
        st.video(event["video_path"])
    if st.button("🚨 Acknowledge Alert & Stop Siren", type="primary", key="ack-alert", use_container_width=True):
        controller.acknowledge(event["event_id"])
        st.rerun()

