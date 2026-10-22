"""Driver track, chapter 10 — place a charging session in a group."""

import pandas as pd
import streamlit as st

from content import driver as text
from core import clustering as clu
from core.artifacts import load_metadata, load_model
from ui import components as ui

meta = load_metadata(clu.TRACK)
model = load_model(clu.TRACK)
FEATURES = clu.input_fields(meta)
GROUPS = clu.clusters(meta)
PICKER = "cp_preset"
STEPS = {"connectionTime_decimal": 0.25, "chargingDuration": 0.05, "kWhDelivered": 0.1}


def key(field: str) -> str:
    return f"cp_{field}"


# ---------------------------------------------------------------- state


def _apply_preset() -> None:
    group = st.session_state.get(PICKER)
    if group is None:
        return
    for field, value in clu.profile(group, meta).items():
        st.session_state[key(field)] = value


def _edited() -> None:
    st.session_state[PICKER] = None


if key(FEATURES[0]) not in st.session_state:  # first visit: start from group 0's mean session
    st.session_state[PICKER] = GROUPS[0]
    _apply_preset()

# ---------------------------------------------------------------- page

ui.chapter_header(
    "Driver track · Chapter 10",
    "Place a charging session",
    "Describe a charging session and the K-Means model assigns it to a group.",
)
st.caption("The sessions the model learned from are synthetic (Chapter 8).")

st.pills("Start from a group's mean session", GROUPS, format_func=lambda g: f"Group {g}", selection_mode="single",
         key=PICKER, on_change=_apply_preset)

with st.container(border=True):
    for field in FEATURES:
        low, high = clu.input_range(field, meta)
        st.slider(f"{text.LABELS[field]} ({text.UNITS[field]})", min_value=low, max_value=high, step=STEPS[field],
                  format="%.2f", key=key(field), on_change=_edited)
    session = {f: st.session_state[key(f)] for f in FEATURES}
    st.caption("This session: " + " · ".join(f"{text.LABELS[f].lower()} {text.show(f, session[f])}" for f in FEATURES))

try:
    result = clu.assign(session, model, meta)
except clu.ValidationError as error:
    st.error("These inputs can't be used:\n\n" + "\n".join(f"- **{f}** {m}" for f, m in error.errors.items()))
    st.stop()

# ---------------------------------------------------------------- the group

st.header("The model's answer", anchor=False)
st.markdown(
    f'<div class="ciq-result"><p class="ciq-kicker">K-Means group</p>'
    f'<p class="ciq-result-label">Group {result.cluster}</p>'
    f"<p>{result.cluster_size:,} of the {meta['dataset']['sample_rows']:,} sample sessions are in this group</p></div>",
    unsafe_allow_html=True,
)

st.subheader("How close it is to each group", anchor=False)
st.bar_chart(pd.DataFrame({"Group": [f"Group {g}" for g in result.distances],
                           "Distance to centre": list(result.distances.values())}),
             x="Group", y="Distance to centre", horizontal=True, height=180)
st.caption(f"Distance from this session to each group's centre, in the model's transformed, standardised space. "
           f"K-Means assigns the nearest; the next nearest is group {result.runner_up}.")

st.subheader("This session beside its group", anchor=False)
st.dataframe(
    pd.DataFrame([{"Feature": f"{text.LABELS[f]} ({text.UNITS[f]})", "This session": text.show(f, session[f]),
                   f"Group {result.cluster} mean": text.show(f, result.profile[f])} for f in FEATURES]),
    hide_index=True, width="stretch",
)
st.caption(f"Group means as the notebook reports them (cell {meta['notebook_results']['profiles']['cell']}).")

# ---------------------------------------------------------------- the map

st.subheader("Where it lands on the map", anchor=False)
points = clu.map_points(meta).rename(columns={"pc1": "PC1", "pc2": "PC2"})
points["Group"] = "Group " + points["cluster"].astype(str)
points["Size"] = 12
pc1, pc2 = clu.pca_point(session, model, meta)
here = pd.DataFrame([{"PC1": pc1, "PC2": pc2, "Group": "This session", "Size": 220}])
st.scatter_chart(pd.concat([points[["PC1", "PC2", "Group", "Size"]], here], ignore_index=True),
                 x="PC1", y="PC2", color="Group", size="Size", height=420)
variance = meta["notebook_results"]["pca"]["variance_explained_pct"]
st.caption(f"{len(points):,} sample sessions on the notebook's two-component PCA plane (cell "
           f"{meta['notebook_results']['pca']['cell']}, {variance:.2f}% of the variance), coloured by K-Means group, "
           "with this session placed by the same projection.")

ui.next_chapter("conclusion")
