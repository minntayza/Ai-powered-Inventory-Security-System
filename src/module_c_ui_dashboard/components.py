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
    bg_color = "#064e3b" if healthy else "#4c1d24"
    text_color = "#ecfdf5" if healthy else "#fff1f2"
    border_color = "#34d399" if healthy else "#f87171"

    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            border: 1px solid {border_color};
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            box-sizing: border-box;
            width: 100%;
            min-height: 48px;
            padding: 9px 12px;
            font-size: 13px;
            font-weight: 700;
            color: {text_color} !important;
            display: flex;
            align-items: center;
            gap: 9px;
            margin-bottom: 4px;
            overflow: hidden;
            white-space: nowrap;
        ">
            <span aria-hidden="true" style="
                flex: 0 0 auto;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: {border_color};
                box-shadow: 0 0 0 4px {border_color}22;
            "></span>
            <span style="
                min-width: 0;
                overflow: hidden;
                color: {text_color} !important;
                text-overflow: ellipsis;
                white-space: nowrap;
            ">{label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )



def inventory_metrics(
    counts: Dict[str, int],
    empty_message: str = "No protected inventory is currently detected.",
) -> None:
    if not counts:
        st.info(empty_message)
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
    state_labels = {
        "authorized": "Authorized",
        "unknown": "Unknown / not authorized",
        "not_visible": "Face not verified",
    }
    rows = [
        {
            "Track": f"#{person.get('track_id', person.get('id'))}",
            "Identity": person.get("name", "Unknown"),
            "State": state_labels.get(
                person.get("authorization_state", "not_visible"),
                person.get("authorization_state", "not_visible"),
            ),
            "Confidence": round(float(person.get("face_confidence", 0)), 3),
        }
        for person in persons
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)
