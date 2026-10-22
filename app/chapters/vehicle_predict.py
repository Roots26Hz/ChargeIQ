"""Vehicle track, chapter 4 — try the model, then read its result."""

from ui import components as ui

ui.chapter_header(
    "Vehicle track · Chapter 4",
    "Make a prediction",
    "Describe a vehicle and the trained model predicts whether it is a battery-electric or a plug-in hybrid.",
)

ui.placeholder(
    "Milestone 4",
    "The input form, the predicted type, the exact decision path the input followed, and how many "
    "training vehicles reached the same rule.",
)

ui.next_chapter("battery_data")
