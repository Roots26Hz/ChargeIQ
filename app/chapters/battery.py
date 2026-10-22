"""Battery track — the regression model's estimate of a cell's State of Health."""

from ui import components as ui

ui.chapter_header(
    "Battery track",
    "Battery health",
    "The regression model estimates a lithium-ion cell's State of Health from its charge and discharge behaviour.",
)

ui.placeholder(
    "Milestone 6",
    "Real held-out battery cycles as presets, adjustable measurements, and the predicted State of Health "
    "shown beside the measured value.",
)

ui.next_chapter("charging")
