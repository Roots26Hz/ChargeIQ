"""Driver track, chapter 9 — how the sessions were grouped."""

import pandas as pd
import streamlit as st

from content import driver as text
from core import clustering as clu
from core.artifacts import load_metadata, load_model
from ui import components as ui

meta = load_metadata(clu.TRACK)
model = load_model(clu.TRACK)
results = meta["notebook_results"]
FEATURES = clu.input_fields(meta)
SAMPLE_ROWS = meta["dataset"]["sample_rows"]
NB = text.NB
METRICS = {"silhouette": "Silhouette", "davies_bouldin": "Davies–Bouldin", "calinski_harabasz": "Calinski–Harabasz"}


def group_table(algorithm: str) -> pd.DataFrame:
    sizes, profiles = meta[algorithm]["sizes"], meta[algorithm]["profiles_mean"]
    return pd.DataFrame([
        {"Group": int(g), "Sessions": sizes[g], "Share": f"{sizes[g] / SAMPLE_ROWS:.1%}",
         **{text.LABELS[f]: text.show(f, profiles[g][f]) for f in FEATURES}}
        for g in sorted(sizes, key=int)
    ])


ui.chapter_header(
    "Driver track · Chapter 9",
    "How the model groups",
    "Two clustering algorithms, how the number of groups was chosen, and what each group looks like.",
)

prepare_tab, groups_tab, compare_tab, see_tab = st.tabs(
    ["Preparing the data", "The three groups", "Two algorithms compared", "Seeing the groups"]
)

# ---------------------------------------------------------------- preparing the data

with prepare_tab:
    steps = model.named_steps
    st.markdown(
        f"1. Missing values are filled with each feature's **median** (`SimpleImputer`).\n"
        f"2. A **Yeo–Johnson** power transform reduces skew (`PowerTransformer`, "
        f"`standardize={steps['power'].get_params()['standardize']}`).\n"
        f"3. Every feature is **standardised** (`StandardScaler`).\n"
        f"4. **K-Means** with K = {steps['kmeans'].n_clusters} groups the sessions."
    )
    ui.source_note(NB, [results["sample"]["cell"], results["kmeans_sizes"]["cell"]], "read from the exported model")
    ui.team_quote(text.PREPROCESSING)

    st.subheader("Choosing the number of groups", anchor=False)
    elbow = meta["kmeans"]["elbow_inertia"]
    st.line_chart(pd.DataFrame({"K": [int(k) for k in elbow], "Inertia": list(elbow.values())}), x="K", y="Inertia",
                  height=260)
    st.caption("K-Means inertia for K = 2 to 10, recomputed by the export from the notebook's cell 13; the notebook "
               "itself shows these values only as the plot below.")
    ui.team_quote(text.ELBOW)
    ui.team_quote(text.K_CHOICE)
    with st.expander("The notebook's elbow plot", icon=":material/show_chart:"):
        ui.figure("clustering/models/01_kmeans_elbow_curve.png", "K-Means elbow curve", NB, 13)

# ---------------------------------------------------------------- the three groups

with groups_tab:
    st.markdown(f"K-Means divides the {SAMPLE_ROWS:,} sessions into three groups. Each row is the group's "
                "**mean session**:")
    st.dataframe(group_table("kmeans"), hide_index=True, width="stretch")
    ui.source_note(NB, [results["kmeans_sizes"]["cell"], results["profiles"]["cell"]])
    ui.team_quote(text.PROFILES)
    ui.team_quote(text.ARBITRARY_LABELS)
    ui.team_note(
        "Three kinds of session",
        "Group 0 (35% of sessions) is a short evening top-up: sessions start in the late afternoon or "
        "evening (median 17:03, middle half 15:21–20:30), charge for about 1.7 hours and take about 5.7 kWh. "
        "Group 1 (46%), the largest, starts at a similar time of day (median 16:34) but charges far longer, "
        "a median of 5.6 hours, and takes the most energy, about 13.6 kWh. Group 2 (19%) is the night group: "
        "92% of its sessions start between midnight and 06:00 (median 01:57), and they are the shortest and "
        "smallest, about 1.1 hours and 4.2 kWh. Groups 0 and 1 overlap in start time, so it is duration and "
        "energy, not time of day, that separates them. Group 1 also has the lowest energy per hour of "
        "charging (a median of 2.3 kWh per hour, against 3.4 and 3.9), which fits longer, slower charges. As "
        "the data is synthetic and has no ground-truth labels, these describe patterns in the sample, not "
        "established types of driver.",
    )
    with st.expander("Group centres are not group means", icon=":material/info:"):
        st.markdown("K-Means places each centre in the transformed, standardised space. Transformed back to original "
                    "units, the centres sit near — but not exactly at — each group's mean session:")
        centres = meta["kmeans"]["centres_original_units"]
        st.dataframe(pd.DataFrame([{"Group": int(g), **{text.LABELS[f]: text.show(f, c[f]) for f in FEATURES}}
                                   for g, c in sorted(centres.items(), key=lambda p: int(p[0]))]),
                     hide_index=True, width="stretch")
        st.caption("Read from the exported model.")

# ---------------------------------------------------------------- two algorithms

with compare_tab:
    algorithms = {"K-Means": "kmeans", "Agglomerative (Ward + kNN connectivity)": "agglomerative"}
    scores = pd.DataFrame([
        {"Algorithm": name, **{METRICS[m]: meta[key]["metrics"][m] for m in METRICS}}
        for name, key in algorithms.items()
    ])
    for label, better_high in (("Silhouette", True), ("Davies–Bouldin", False), ("Calinski–Harabasz", True)):
        scores[f"{label} rank"] = scores[label].rank(ascending=not better_high, method="min").astype(int)
    st.dataframe(scores, hide_index=True, width="stretch",
                 column_config={"Silhouette": st.column_config.NumberColumn(format="%.4f"),
                                "Davies–Bouldin": st.column_config.NumberColumn(format="%.4f"),
                                "Calinski–Harabasz": st.column_config.NumberColumn(format="%.1f")})
    ui.source_note(NB, [results["metrics"]["cell"], 33],
                   f"Silhouette on a shared 5,000-session subset; the other two on all {SAMPLE_ROWS:,}")
    ui.team_quote(text.SILHOUETTE)
    ui.team_quote(text.DAVIES_BOULDIN)
    ui.team_quote(text.CALINSKI_HARABASZ)
    ui.team_quote(text.INTERNAL_METRICS)
    ui.team_quote(text.PREFERRED)

    st.subheader("Agglomerative clustering", anchor=False)
    ui.team_quote(text.WARD)
    ui.team_quote(text.CONNECTIVITY)
    st.dataframe(group_table("agglomerative"), hide_index=True, width="stretch")
    ui.source_note(NB, [results["agglomerative_sizes"]["cell"], results["profiles"]["cell"]],
                   "its group numbers are independent of K-Means'")
    st.markdown("Only K-Means can place a **new** session: scikit-learn's `AgglomerativeClustering` groups the "
                "sessions it is fitted on and has no way to assign another. Chapter 10 therefore uses K-Means.")

    with st.expander("Dendrograms", icon=":material/account_tree:"):
        ui.figure("clustering/models/02_linkage_comparison_dendrograms.png",
                  "Single, complete and average linkage, on up to 500 sessions", NB, 17)
        ui.team_quote(text.LINKAGES)
        ui.figure("clustering/models/03_ward_dendrogram.png", "Ward linkage, on up to 500 sessions", NB, 19)
        ui.team_quote(text.DENDROGRAM)

# ---------------------------------------------------------------- seeing the groups

with see_tab:
    variance = results["pca"]["variance_explained_pct"]
    st.markdown(f"The first two principal components keep **{variance:.2f}%** of the variance of the three "
                "preprocessed features.")
    ui.source_note(NB, [results["pca"]["cell"]])
    ui.figure("clustering/models/04_kmeans_pca.png", "K-Means groups on the PCA plane", NB, 25)
    ui.team_quote(text.PCA)
    ui.figure("clustering/models/05_agglomerative_pca.png", "Agglomerative groups on the same plane", NB, 27)
    ui.team_quote(text.PCA_COMPARISON)
    ui.figure("clustering/models/06_kmeans_tsne.png", "K-Means groups in a t-SNE view of up to 1,000 sessions", NB, 29)
    ui.team_quote(text.TSNE)

st.header("What these groups are", anchor=False)
ui.team_quote(text.NO_GROUND_TRUTH)

ui.next_chapter("charging_predict")
