"""The story's chapters, in reading order, and how they appear in the sidebar."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

CHAPTERS_DIR = Path(__file__).resolve().parents[1] / "chapters"

# Visitors only ever see finished chapters. Set this to True locally to review
# the whole story arc, unfinished chapters included; keep it False when deploying
# (app/tests/test_deploy.py fails otherwise).
SHOW_UNFINISHED = False


@dataclass(frozen=True)
class Chapter:
    key: str          # file name in app/chapters/, without .py
    section: str      # sidebar group
    title: str
    icon: str
    ready: bool       # True once the chapter's content is complete

    @property
    def path(self) -> Path:
        return CHAPTERS_DIR / f"{self.key}.py"

    @property
    def url_path(self) -> str:
        return self.key.replace("_", "-")


DEFAULT_KEY = "home"

VEHICLE = "Vehicle · classification"
BATTERY = "Battery · regression"
DRIVER = "Driver · clustering"

CHAPTERS: tuple[Chapter, ...] = (
    Chapter("home", "ChargeIQ", "The problem", ":material/bolt:", ready=True),
    Chapter("vehicle_data", VEHICLE, "Meet the data", ":material/table_chart:", ready=True),
    Chapter("vehicle_model", VEHICLE, "How the model thinks", ":material/account_tree:", ready=True),
    Chapter("vehicle_predict", VEHICLE, "Make a prediction", ":material/directions_car:", ready=True),
    Chapter("battery_data", BATTERY, "Meet the data", ":material/table_chart:", ready=True),
    Chapter("battery_model", BATTERY, "How the model learns", ":material/insights:", ready=True),
    Chapter("battery_predict", BATTERY, "Estimate battery health", ":material/battery_charging_full:", ready=True),
    Chapter("charging_data", DRIVER, "Meet the data", ":material/table_chart:", ready=True),
    Chapter("charging_model", DRIVER, "How the model groups", ":material/hub:", ready=True),
    Chapter("charging_predict", DRIVER, "Place a charging session", ":material/ev_station:", ready=True),
    Chapter("conclusion", "Wrap-up", "Conclusion", ":material/flag:", ready=False),
)


def chapter(key: str) -> Chapter:
    for ch in CHAPTERS:
        if ch.key == key:
            return ch
    raise KeyError(f"no chapter called {key!r}")


def visible(show_unfinished: bool | None = None) -> tuple[Chapter, ...]:
    """Chapters shown in the sidebar. The default chapter is always shown."""
    show = SHOW_UNFINISHED if show_unfinished is None else show_unfinished
    return tuple(ch for ch in CHAPTERS if show or ch.ready or ch.key == DEFAULT_KEY)


def is_visible(key: str) -> bool:
    return any(ch.key == key for ch in visible())


def build() -> dict[str, list[st.Page]]:
    """Sidebar sections for st.navigation, in reading order."""
    sections: dict[str, list[st.Page]] = {}
    for ch in visible():
        page = st.Page(ch.path, title=ch.title, icon=ch.icon, url_path=ch.url_path,
                       default=ch.key == DEFAULT_KEY)
        sections.setdefault(ch.section, []).append(page)
    return sections
