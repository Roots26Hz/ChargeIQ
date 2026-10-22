"""Vehicle track, chapter 2 — the dataset the classification model learned from."""

from ui import components as ui

ui.chapter_header(
    "Vehicle track · Chapter 2",
    "Meet the data",
    "The Washington State Electric Vehicle Population dataset, which the classification model was trained on.",
)

ui.placeholder(
    "Milestone 3",
    "Dataset size and source, the target and its class balance, the features the model receives, "
    "and the columns deliberately excluded — all taken from the exported metadata.",
)

ui.next_chapter("vehicle_model")
