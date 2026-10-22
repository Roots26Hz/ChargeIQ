"""Battery track, chapter 5 — the NASA battery data the regression model learned from."""

from ui import components as ui

ui.chapter_header(
    "Battery track · Chapter 5",
    "Meet the data",
    "The NASA battery dataset: how each charge–discharge cycle became a row, and how State of Health "
    "was defined.",
)

ui.placeholder(
    "Milestone 7",
    "The cells and cycles, the features extracted from each cycle, the State-of-Health target, and "
    "why the data was split by battery rather than by row — in the team's own words.",
)

ui.next_chapter("battery_model")
