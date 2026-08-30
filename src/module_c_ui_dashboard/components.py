"""Reusable Streamlit dashboard components."""

from __future__ import annotations

from typing import Dict

import streamlit as st


ITEM_ICONS = {
    "laptop": "💻",
    "cell_phone": "📱",
    "cell phone": "📱",
    "backpack": "🎒",
    "handbag": "👜",
    "book": "📚",
    "bottle": "🍾",
    "cup": "☕",
    "chair": "🪑",
    "tv": "📺",
    "clock": "⏰",
    "scissors": "✂️",
    "mouse": "🖱️",
    "keyboard": "⌨️",
    "remote": "🎮",
    "person": "👤",
}


def status_badge(label: str, healthy: bool) -> None:
    icon = "🟢" if healthy else "🔴"
    bg_color = "#d1fae5" if healthy else "#fee2e2"
    text_color = "#065f46" if healthy else "#991b1b"
    border_color = "#34d399" if healthy else "#f87171"

    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            border: 1px solid {border_color};
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 13px;
            font-weight: 700;
            color: {text_color} !important;
            display: flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 4px;
        ">
            <span>{icon}</span>
            <span style="color: {text_color} !important;">{label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )



def inventory_metrics(counts: Dict[str, int]) -> None:
    if not counts:
        st.info("Waiting for a stable inventory baseline…")
        return
    columns = st.columns(min(len(counts), 4))
    for index, (label, count) in enumerate(sorted(counts.items())):
        icon = ITEM_ICONS.get(label.lower(), "📦")
        display_name = f"{icon} {label.replace('_', ' ').title()}"
        columns[index % len(columns)].metric(display_name, count)


def authorization_table(persons: list[Dict]) -> None:
    if not persons:
        st.caption("No people detected in camera view")
        return
    rows = [
        {
            "Track": f"#{person.get('track_id', person.get('id'))}",
            "Identity": person.get("name", "Unknown"),
            "State": person.get("authorization_state", "not_visible"),
            "Confidence": round(float(person.get("face_confidence", 0)), 3),
        }
        for person in persons
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)
