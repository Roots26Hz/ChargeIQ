"""Driver track, chapter 8 — the charging-session data the clustering model learned from."""

import pandas as pd
import streamlit as st

from content import driver as text
from core import clustering as clu
from core.artifacts import load_metadata
from ui import components as ui

meta = load_metadata(clu.TRACK)
results = meta["notebook_results"]
FEATURES = clu.input_fields(meta)
NB = text.NB

ui.chapter_header(
    "Driver track · Chapter 8",
    "Meet the data",
    "Charging sessions described by when they start, how long they last and how much energy they deliver. "
    "The data is synthetic.",
)

st.info(
    f"**These charging sessions are synthetic.** The notebook loads them from a file named "
    f"`{text.SOURCE_FILE.value}` (cell {text.SOURCE_FILE.cell}).",
    icon=":material/science:",
)
ui.team_note(
    "Where this data comes from",
    "The charging sessions are taken from the “Electric Vehicle Charging Dataset” by Nastaran "
    "Gholizadeh (Mendeley Data, 2024, doi:10.17632/5zrtmp7gwd.1, CC BY 4.0). We obtained it through "
    "its Kaggle mirror. The data is synthetic: it was produced with conditional tabular GANs (CTGAN) "
    "and kernel density estimation, using real sessions logged at a Caltech EV parking lot as the "
    "starting point. The Caltech recordings span 185 days, and the synthetic version stretches that "
    "to 29,600 days and 1,965,239 sessions. A row describes one session through three measurements: "
    "when the car plugged in (hour of day, 0–24), how long it charged (hours) and how much energy it "
    "took (kWh). The fourth column, dayIndicator, only numbers the days, so we leave it out of the "
    "clustering. Since these sessions were generated and not observed, any groups we find reflect the "
    "patterns the generator picked up from one parking lot, not how people charge in general.",
)

source = results["source"]
ui.stat_row([
    ("Charging sessions", f"{source['rows']:,}", f"Rows in {source['file']}"),
    ("Sessions clustered", f"{meta['dataset']['sample_rows']:,}", "A reproducible random sample"),
    ("Features used", f"{len(FEATURES)}", f"Of the file's {source['columns']} columns"),
])
ui.source_note(NB, [source["cell"], results["sample"]["cell"]])

# ---------------------------------------------------------------- the features

st.header("The three features", anchor=False)
ui.team_quote(text.FEATURES)
stats = results["full_dataset_statistics"]["features"]
st.dataframe(
    pd.DataFrame([
        {"Feature": f, "Meaning": f"{text.LABELS[f]} ({text.UNITS[f]})",
         "Mean": text.show(f, stats[f]["mean"]), "Median": text.show(f, stats[f]["50%"]),
         "Min": text.show(f, stats[f]["min"]), "Max": text.show(f, stats[f]["max"])}
        for f in FEATURES
    ]),
    hide_index=True, width="stretch",
)
ui.source_note(NB, [results["full_dataset_statistics"]["cell"]], "all 1,965,239 sessions")

# ---------------------------------------------------------------- cleaning

st.header("Cleaning", anchor=False)
st.markdown(
    f"The file has no missing values and **{text.DUPLICATES.value}** duplicate rows; "
    f"**{text.ROWS_RETAINED.value}** sessions are kept for clustering."
)
ui.source_note(NB, [text.DUPLICATES.cell, text.ROWS_RETAINED.cell])
ui.team_quote(text.PARTIAL_ROWS)
ui.team_quote(text.OUTLIERS)

# ---------------------------------------------------------------- the sample

st.header(f"A sample of {meta['dataset']['sample_rows']:,} sessions", anchor=False)
ui.team_quote(text.SAMPLE)
ranges = meta["sample_ranges"]
st.dataframe(
    pd.DataFrame([
        {"Feature": f, "Median, all sessions": text.show(f, stats[f]["50%"]),
         "Median, sample": text.show(f, ranges[f]["p50"])}
        for f in FEATURES
    ]),
    hide_index=True, width="stretch",
)
st.caption(f"The sample is the one the notebook saved in `{meta['provenance']['sample']}`; "
           "this app's model was rebuilt from it.")
ui.team_quote(text.SAMPLE_SCOPE)

# ---------------------------------------------------------------- a first look

st.header("A first look", anchor=False)
ui.figure("clustering/eda/01_feature_distributions.png",
          "Distributions of the three features, from a random 10,000 sessions", NB, 11)
ui.figure("clustering/eda/02_correlation_heatmap.png",
          "Correlations between the three features, same 10,000 sessions", NB, 11)
ui.team_quote(text.EDA)

ui.next_chapter("charging_model")
