"""Reusable presentation pieces, so every chapter looks and reads the same way."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable, Sequence

import streamlit as st

from . import navigation

STYLESHEET = Path(__file__).with_name("style.css")


def inject_css() -> None:
    """Load the stylesheet. Streamlit sends style-only HTML to an event container, so it takes no space."""
    st.html(STYLESHEET)


def chapter_header(kicker: str, title: str, lede: str | None = None) -> None:
    st.markdown(f'<p class="ciq-kicker">{html.escape(kicker)}</p>', unsafe_allow_html=True)
    st.title(title, anchor=False)
    if lede:
        st.markdown(f'<p class="ciq-lede">{html.escape(lede)}</p>', unsafe_allow_html=True)


def stat_row(stats: Sequence[tuple[str, str, str | None]]) -> None:
    """A row of headline numbers: (label, value, help text)."""
    for column, (label, value, help_text) in zip(st.columns(len(stats)), stats):
        column.metric(label, value, help=help_text)


def source_note(notebook: str, cells: Iterable[int] = (), note: str | None = None) -> None:
    """Cite where a number or statement comes from, so every claim is traceable."""
    cells = list(cells)
    parts = [f"`notebooks/{notebook}`"]
    if cells:
        parts.append(("cell " if len(cells) == 1 else "cells ") + ", ".join(str(c) for c in cells))
    st.caption("Source: " + " · ".join(parts) + (f" — {note}" if note else ""))


def team_todo(text: str) -> None:
    """A visible marker for interpretive copy the team must write in its own words.

    The course guidelines allow AI for code scaffolding but not for analysis or
    interpretation, so these stay on screen until the team replaces them.
    """
    st.markdown(f'<div class="ciq-todo"><b>TODO(team)</b> — {html.escape(text)}</div>',
                unsafe_allow_html=True)


def placeholder(milestone: str, summary: str) -> None:
    st.markdown(
        f'<div class="ciq-placeholder"><p class="ciq-kicker">Coming in {html.escape(milestone)}</p>'
        f"<p>{html.escape(summary)}</p></div>",
        unsafe_allow_html=True,
    )


def next_chapter(key: str, label: str | None = None) -> None:
    """Link to the next chapter — only if that chapter is currently shown in the sidebar."""
    if not navigation.is_visible(key):
        return
    target = navigation.chapter(key)
    st.divider()
    st.page_link(target.path, label=label or f"Next: {target.title}", icon=":material/arrow_forward:")
