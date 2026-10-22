"""Chapter 1 — each track's question, in the words of its own notebook."""

from . import Quote

VEHICLE_OBJECTIVE = Quote(
    "classification.ipynb", 0,
    "The objective of this classification task is to predict the Electric Vehicle Type of a vehicle "
    "based on its available vehicle, technical, manufacturer, and geographic characteristics.",
)

BATTERY_OBJECTIVE = Quote(
    "regression.ipynb", 0,
    "**Target:** predict **State-of-Health (SOH, %)** of a battery cell — not raw Capacity directly.",
)

DRIVER_OBJECTIVE = Quote(
    "clustering.ipynb", 1,
    "The goal is to identify groups of charging sessions with similar connection times, "
    "charging durations, and energy-delivery patterns.",
)

QUOTES = (VEHICLE_OBJECTIVE, BATTERY_OBJECTIVE, DRIVER_OBJECTIVE)
FACTS = ()
