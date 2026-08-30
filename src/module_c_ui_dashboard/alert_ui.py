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
    if security.get("state") == "PENDING" and security.get("pending"):
        removed = _removed_summary(security["pending"])
        st.warning(f"Verifying protected-item removal — {removed}")

    event = security.get("last_event")
    if not event or event.get("acknowledged"):
        return
    removed = _removed_summary(event)
    event_type = event.get("event_type")
    if event.get("status") == "confirmed":
        apply_alert_theme()
    elif event_type == "authorized_removal":
        st.success(f"Authorized removal recognized — {removed}")
    elif event_type == "inventory_recovered":
        st.info(f"Inventory returned during verification — {removed}")
    elif event_type == "unattributed_inventory_change":
        st.warning(
            "Inventory removal recognized, but no nearby person was observed — "
            f"{removed}"
        )
    else:
        return

    if event_type != "inventory_recovered":
        st.caption(
            "This removal was recorded once and monitoring is now paused. "
            "Choose how to continue in the sidebar."
        )

    actor = event.get("primary_actor")
    if event.get("status") == "confirmed":
        actor_str = ""
        if actor:
            actor_str = (
                f"• Suspect: <strong>{actor.get('name', 'Unknown')}</strong> "
                f"(Track #{actor.get('track_id')}, confidence "
                f"{actor.get('association_score', 0):.2f})"
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
                    • Telegram: {event.get('telegram_status', 'pending')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"Event {event['event_id']} at {event['timestamp']}")
        if actor:
            st.caption(
                f"Likely actor: {actor.get('name', 'Unknown')} "
                f"(track {actor.get('track_id')}, score "
                f"{actor.get('association_score', 0):.2f})"
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
    if event.get("status") == "confirmed" and event.get("video_path"):
        st.video(event["video_path"])
    if event.get("status") == "confirmed" and st.button(
        "🚨 Acknowledge Alert & Stop Siren",
        type="primary",
        key="ack-alert",
        use_container_width=True,
    ):
        controller.acknowledge(event["event_id"])
        st.rerun()


def _removed_summary(event: dict) -> str:
    return ", ".join(
        f"{name.replace('_', ' ').title()} ×{count}"
        for name, count in sorted(event.get("removed_items", {}).items())
    ) or "unknown protected item"
