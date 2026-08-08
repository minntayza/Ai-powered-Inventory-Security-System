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
    st.error(f"🚨 SUSPECTED THEFT — Removed: {removed}")
    st.caption(f"Event {event['event_id']} at {event['timestamp']}")
    actor = event.get("primary_actor")
    if actor:
        st.caption(
            f"Likely actor: {actor.get('name', 'Unknown')} "
            f"(track {actor.get('track_id')}, score {actor.get('association_score', 0):.2f})"
        )
    if event.get("video_path"):
        st.video(event["video_path"])
    if st.button("Acknowledge alert and stop siren", type="primary", key="ack-alert"):
        controller.acknowledge(event["event_id"])
        st.rerun()
