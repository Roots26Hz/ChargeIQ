"""Battery track, chapter 5 — the NASA battery data the regression model learned from."""

import pandas as pd
import streamlit as st

from content import battery as text
from core import regression as reg
from core.artifacts import load_metadata
from ui import components as ui

meta = load_metadata(reg.TRACK)
data = meta["dataset"]
results = meta["notebook_results"]
NB = text.NB

ui.chapter_header(
    "Battery track · Chapter 5",
    "Meet the data",
    "The NASA battery dataset: how each charge–discharge cycle became a row, and how State of Health "
    "was defined.",
)
ui.team_quote(text.SOURCE)

ui.stat_row([
    ("Batteries", f"{data['batteries']} of {data['raw_batteries']}", "Batteries kept after cleaning, of those in the data"),
    ("Discharge cycles", f"{data['rows']:,}", f"Rows after cleaning, from {data['discharge_cycles']:,} discharge cycles"),
    ("Model inputs", f"{len(meta['feature_order'])}", "Plus one engineered feature, cycle_index²"),
    ("Target", "SOH %", "State of Health"),
])

# ---------------------------------------------------------------- the target

st.header("What the model predicts", anchor=False)
ui.team_quote(text.TARGET)
soh = data["soh_pct"]
st.markdown(
    f"`NOMINAL_CAPACITY` is **{meta['nominal_capacity_ah']:.1f} Ah** (cell 2). Across the {data['rows']:,} cycles, "
    f"SOH ranges from **{soh['min']:.1f}%** to **{soh['max']:.1f}%**, with a median of **{soh['50%']:.1f}%**."
)
ui.source_note(NB, [2, 21])
ui.figure("regression/eda/03_capacity_fade_and_scatter.png",
          "Left: Capacity over the cycles of every battery, before cleaning. "
          "Right: Capacity against early voltage-sag rate.", NB, 15)
ui.team_quote(text.FADE)

# ---------------------------------------------------------------- the inputs

st.header("From raw curves to model inputs", anchor=False)
types = data["raw_row_types"]
st.markdown(
    f"`metadata.csv` has {data['raw_rows']:,} rows — {types['discharge']:,} discharge, {types['charge']:,} charge "
    f"and {types['impedance']:,} impedance tests — and each charge or discharge row points to its own raw "
    "voltage, current and temperature curve. Each measurement lives on a different kind of row:"
)
ui.team_quote(text.MISSINGNESS)
st.markdown("So the notebook builds one row per discharge cycle, and takes care not to restate the target:")
ui.team_quote(text.FULL_CURVE)
ui.team_quote(text.EARLY_WINDOW)
ui.team_quote(text.PRECEDING_CHARGE)

def _number(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:.4g}"


ranges = meta["numeric_ranges"]
st.dataframe(
    pd.DataFrame([
        {"Input": field,
         "What it measures": text.INPUT_DESCRIPTIONS[field] + (" ¹" if field in text.DESCRIBED_IN_README else ""),
         "In the training rows": f"{_number(ranges[field]['min'])} to {_number(ranges[field]['max'])}"}
        for field in meta["feature_order"]
    ]),
    hide_index=True, width="stretch",
    column_config={"What it measures": st.column_config.TextColumn(width="large")},
)
ui.source_note(NB, [7, 9, 25], f"¹ described in `{text.README}`")

st.subheader("A look at the inputs", anchor=False)
ui.figure("regression/eda/01_raw_feature_distributions.png", "Distributions of the five curve features", NB, 13)
ui.team_quote(text.SAG_FEATURE)
ui.figure("regression/eda/02_correlation_heatmap.png", "Correlations between the inputs and Capacity", NB, 14)
ui.team_quote(text.COLLINEARITY)

# ---------------------------------------------------------------- cleaning

st.header("Cleaning", anchor=False)
excluded = meta["batteries"]["excluded_corrupted"]
st.markdown(f"Four batteries are removed before modelling: {', '.join(f'`{b}`' for b in excluded)}.")
ui.team_quote(text.CORRUPTED)
ui.team_quote(text.CLEANING)
low, high = data["capacity_outlier_bounds"]
st.markdown(
    f"**{data['capacity_outliers_capped']}** Capacity readings fell outside the IQR bounds "
    f"[{low:.3f}, {high:.3f}] Ah and were capped to them."
)
ui.source_note(NB, [results["cleaning"]["cell"]])

# ---------------------------------------------------------------- the split

st.header("Why the split is by battery", anchor=False)
ui.team_quote(text.PANEL_DATA)
experiment = results["split_experiment"]
st.dataframe(
    pd.DataFrame([
        {"Split": "(a) by row, with battery_id as a feature", "Test R²": experiment["row_level_with_battery_id_r2"]},
        {"Split": "(b) by battery, without battery_id", "Test R²": experiment["battery_grouped_r2"]},
    ]),
    hide_index=True, width="stretch",
    column_config={"Test R²": st.column_config.NumberColumn(format="%.4f")},
)
ui.source_note(NB, [experiment["cell"]])
ui.team_quote(text.EASIER_QUESTION)
ui.team_quote(text.THE_FIX)

split = results["split"]
ui.stat_row([
    ("Training", f"{split['train_rows']:,} cycles", f"{split['train_batteries']} batteries"),
    ("Test", f"{split['test_rows']:,} cycles", f"{split['test_batteries']} batteries the model never sees in training"),
])
st.caption("Test batteries: " + ", ".join(meta["batteries"]["test"]))
ui.source_note(NB, [split["cell"]])

ui.next_chapter("battery_model")
