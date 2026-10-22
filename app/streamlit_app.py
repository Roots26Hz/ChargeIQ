"""ChargeIQ — an interactive walk through the 23CSE301 Machine Learning capstone.

Run from the repository root, so .streamlit/config.toml is picked up exactly as it
is on Streamlit Community Cloud:

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:  # make `core` and `ui` importable from every chapter
    sys.path.insert(0, str(APP_DIR))

import streamlit as st  # noqa: E402

from ui import components, navigation  # noqa: E402

st.set_page_config(
    page_title="ChargeIQ",
    page_icon=":material/bolt:",
    layout="centered",
    initial_sidebar_state="expanded",
)
components.inject_css()

page = st.navigation(navigation.build(), position="sidebar")

with st.sidebar:
    st.caption("23CSE301 · Machine Learning Capstone · AY 2026–27")

page.run()
