"""Vehicle track, chapter 2 — the dataset the classification model learned from."""

import pandas as pd
import streamlit as st

from content import vehicle as text
from core import classification as clf
from core.artifacts import load_metadata
from ui import components as ui

meta = load_metadata(clf.TRACK)
data = meta["dataset"]
NB = text.NB

ui.chapter_header(
    "Vehicle track · Chapter 2",
    "Meet the data",
    "The Washington State Electric Vehicle Population dataset, which the classification model was trained on.",
)
st.caption("Source, as documented in `data/README.md`: data.wa.gov — Electric Vehicle Population Data.")

ui.stat_row([
    ("Registered vehicles", f"{data['rows']:,}", "Rows in the dataset"),
    ("Columns", f"{data['columns']}", "Columns in the dataset, including the target"),
    ("Model inputs", f"{len(meta['feature_order'])}", "Columns the model receives, incl. the derived Vehicle Age"),
    ("Classes", f"{len(meta['classes'])}", "BEV and PHEV"),
])

# ---------------------------------------------------------------- the target

st.header("What the model predicts", anchor=False)
st.markdown(f"The target is **{meta['target']}**, which takes one of two values.")

counts = data["class_counts"]
balance = pd.DataFrame({
    "Vehicle type": [clf.short_label(c) for c in counts],
    "Registered vehicles": list(counts.values()),
})
st.bar_chart(balance, x="Vehicle type", y="Registered vehicles", horizontal=True, height=160)
st.caption(" · ".join(f"**{c}**: {v:,} ({v / data['rows']:.2%})" for c, v in counts.items()))
ui.team_quote(text.CLASS_IMBALANCE)

# ---------------------------------------------------------------- the inputs

st.header("What the model sees", anchor=False)
st.markdown(
    f"Each vehicle reaches the model as {len(meta['feature_order'])} columns. "
    "The values below are those found in the model's training rows."
)


def _number(value: float) -> str:
    return f"{int(value)}" if float(value).is_integer() else f"{value:g}"


rows = []
for column in meta["feature_order"]:
    if column in meta["categorical_features"]:
        kind, values = "Category", f"{len(meta['category_domains'][column]):,} distinct values"
    else:
        r = meta["numeric_ranges"][column]
        kind, values = "Number", f"{_number(r['min'])} to {_number(r['max'])}"
    if column in meta["derived_features"]:
        formula = meta["derived_features"][column]
        kind = "Number (derived)"
        values += " — " + formula.replace("VEHICLE_AGE_REFERENCE_YEAR", str(meta["vehicle_age_reference_year"]))
    rows.append({"Column": column, "Kind": kind, "In the training rows": values})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
ui.source_note(NB, [22, 23, 30], "derived from the exported model's metadata")

st.subheader("Left out on purpose", anchor=False)
excluded = meta["excluded_columns"]
st.markdown("**Identifiers** — " + ", ".join(f"`{c}`" for c in excluded["identifiers"]))
ui.team_quote(text.IDENTIFIERS)
st.markdown(
    "**Listed as a target proxy** — "
    + ", ".join(f"`{c}`" for c in excluded["target_proxy"])
    + f". The notebook removes it in cell {text.TARGET_PROXY_DROP.cell}, where the code lists it under "
    f"`{text.TARGET_PROXY_DROP.evidence}`."
)

# ---------------------------------------------------------------- two features

st.header("Two features that differ by type", anchor=False)
ui.figure("classification/eda/05_electric_range_vs_target.png",
          "Electric Range by vehicle type", NB, 15)
ui.figure("classification/eda/06_model_year_vs_target.png",
          "Model Year by vehicle type", NB, 15)
ui.team_quote(text.FEATURE_TARGET_OBSERVATIONS)

# ---------------------------------------------------------------- preparation

st.header("How the data was prepared", anchor=False)

st.subheader("Missing values and outliers", anchor=False)
ui.team_quote(text.CLEANING)
ui.team_quote(text.OUTLIERS)

st.subheader("A derived feature: Vehicle Age", anchor=False)
st.markdown(
    f"`Vehicle Age = {meta['vehicle_age_reference_year']} − Model Year`, computed in the notebook "
    f"and recomputed the same way by this app."
)
ui.team_quote(text.VEHICLE_AGE)

st.subheader("Training and test rows", anchor=False)
ui.team_quote(text.SPLIT)
ui.stat_row([
    ("Training pool", text.TRAINING_POOL_ROWS.value, "80% of the rows"),
    ("Model trained on", text.MODEL_TRAINING_ROWS.value,
     f"Rows from the training pool used to fit the final model, cell {text.MODEL_TRAINING_ROWS.cell}"),
    ("Test rows", f"{data['test_rows']:,}", "20% of the rows, held out for evaluation"),
])
ui.source_note(NB, [26, 30])

ui.next_chapter("vehicle_model")
