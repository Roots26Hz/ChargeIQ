"""Battery track, chapter 6 — how the regression model was chosen and how well it does."""

from ui import components as ui

ui.chapter_header(
    "Battery track · Chapter 6",
    "How the model learns",
    "Ten regression algorithms compared on unseen batteries, and the gradient-boosting model that was"
    " selected.",
)

ui.placeholder(
    "Milestone 7",
    "The ten-model comparison, cross-validation by battery, tuning, and the predicted-versus-measured"
    " and residual plots, with cell references.",
)

ui.next_chapter("battery_predict")
