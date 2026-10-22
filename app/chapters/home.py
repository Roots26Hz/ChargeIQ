"""Chapter 1 — the problem ChargeIQ sets out to answer."""

import streamlit as st

from content import home as text
from core import classification as clf
from core.artifacts import load_metadata
from ui import components as ui
from ui import navigation

ui.chapter_header(
    "ChargeIQ · Chapter 1",
    "The problem",
    "Three independent machine-learning tracks, each answering a different question about electric vehicles.",
)

ui.team_todo(
    "Why these three questions matter — the motivation behind ChargeIQ, in the team's own words."
)

# (track, method, the track's first chapter, its objective in its notebook's words, icon)
TRACKS = (
    ("The vehicle", "Classification", "vehicle_data", text.VEHICLE_OBJECTIVE, ":material/directions_car:"),
    ("The battery", "Regression", "battery_data", text.BATTERY_OBJECTIVE, ":material/battery_charging_full:"),
    ("The driver", "Clustering", "charging_data", text.DRIVER_OBJECTIVE, ":material/ev_station:"),
)

for name, method, first_chapter, objective, icon in TRACKS:
    with st.container(border=True):
        st.subheader(f"{icon} {name}", anchor=False)
        ready = navigation.chapter(first_chapter).ready
        st.badge(method, color="primary" if ready else "gray")
        ui.team_quote(objective)
        if ready:
            ui.chapter_link(first_chapter, f"Explore the {name.split()[-1]} track")
        else:
            st.caption("This track's chapters are still being built.")

with st.expander("How this app was built", icon=":material/info:"):
    meta = load_metadata(clf.TRACK)
    st.markdown(
        "**The models.** "
        + meta["provenance"]["method"]
        + "\n\n**The words.** Indented quotations are verbatim excerpts of the team's notebooks, each "
        "cited by notebook and cell; an automated test checks every one against its cell. Boxes marked "
        "**TODO(team)** hold text the team has still to write.\n\n"
        "**The code.** The app's code was scaffolded with AI assistance (Claude Code). The analysis and "
        "its interpretation are the team's own, taken from the notebooks."
    )

ui.next_chapter("vehicle_data", "Start with the vehicle track")
