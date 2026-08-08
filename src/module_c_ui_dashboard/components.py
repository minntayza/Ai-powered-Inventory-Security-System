"""Reusable Streamlit dashboard components."""

from __future__ import annotations

from typing import Dict

import streamlit as st


def status_badge(label: str, healthy: bool) -> None:
    icon = "🟢" if healthy else "🔴"
    st.markdown(f"{icon} **{label}**")


def inventory_metrics(counts: Dict[str, int]) -> None:
    if not counts:
        st.info("Waiting for a stable inventory baseline…")
        return
    columns = st.columns(min(len(counts), 4))
    for index, (label, count) in enumerate(sorted(counts.items())):
        columns[index % len(columns)].metric(label.replace("_", " ").title(), count)


def authorization_table(persons: list[Dict]) -> None:
    if not persons:
        st.caption("No people detected")
        return
    rows = [
        {
            "Track": person.get("track_id", person.get("id")),
            "Identity": person.get("name", "Unknown"),
            "State": person.get("authorization_state", "not_visible"),
            "Confidence": round(float(person.get("face_confidence", 0)), 3),
        }
        for person in persons
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)

