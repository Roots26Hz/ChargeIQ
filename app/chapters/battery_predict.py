"""Battery track, chapter 7 — estimate a cell's State of Health."""

from ui import components as ui

ui.chapter_header(
    "Battery track · Chapter 7",
    "Estimate battery health",
    "Pick a real cycle from a battery the model never saw, then adjust its measurements.",
)

ui.placeholder(
    "Milestone 7",
    "Real held-out battery cycles as presets, adjustable measurements within the training ranges, and"
    " the predicted State of Health shown beside the measured value.",
)

ui.next_chapter("charging_data")
