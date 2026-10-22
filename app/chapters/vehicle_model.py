"""Vehicle track, chapter 3 — how the selected model makes its decisions."""

import pandas as pd
import streamlit as st

from content import vehicle as text
from core import classification as clf
from core.artifacts import load_metadata, load_model
from ui import components as ui

meta = load_metadata(clf.TRACK)
model = load_model(clf.TRACK)
results = meta["notebook_results"]
final = meta["final_model"]
NB = text.NB

METRICS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
           "weighted_f1": "Weighted F1", "roc_auc": "ROC-AUC"}
SCORE = st.column_config.NumberColumn(format="%.4f")

ui.chapter_header(
    "Vehicle track · Chapter 3",
    "How the model thinks",
    "Preprocessing, the ten algorithms compared, why a decision tree was selected, and the rules it learned.",
)

pipeline_tab, comparison_tab, validation_tab, rules_tab = st.tabs(
    ["Pipeline", "Ten models compared", "Validation", "The rules it learned"]
)

# ---------------------------------------------------------------- pipeline

with pipeline_tab:
    preprocess = model.named_steps["preprocess"]
    numeric, categorical = preprocess.transformers_[0][2], preprocess.transformers_[1][2]
    params = ", ".join(f"{k}={v}" for k, v in final["params"].items())
    st.markdown(
        f"1. **{len(meta['feature_order'])} input columns** arrive in a fixed order.\n"
        f"2. **{len(numeric)} numeric columns** are standardised with `StandardScaler`: "
        + ", ".join(numeric) + ".\n"
        f"3. **{len(categorical)} categorical columns** are one-hot encoded with `OneHotEncoder`: "
        + ", ".join(categorical) + ".\n"
        f"4. Together they become **{meta['dataset']['encoded_features']:,} numeric columns**.\n"
        f"5. A `{final['estimator']}({params})` makes the prediction, fitted on "
        f"**{meta['dataset']['train_rows']:,}** training rows."
    )
    ui.source_note(NB, [30, 46], "read from the exported model")
    ui.team_quote(text.TRAIN_ONLY_FITTING)

# ---------------------------------------------------------------- comparison

with comparison_tab:
    ui.team_quote(text.FAIR_COMPARISON)
    comparison = pd.DataFrame(results["comparison"]["rows"]).rename(columns={"model": "Model", **METRICS})
    comparison.insert(1, "Selected", comparison["Model"] == final["name"])
    st.dataframe(comparison, hide_index=True, width="stretch",
                 column_config={**{label: SCORE for label in METRICS.values()},
                                "Selected": st.column_config.CheckboxColumn(disabled=True)})
    ui.source_note(NB, [results["comparison"]["cell"]], "test-set scores, in the notebook's order")

    best = results["best_model"]
    ui.stat_row([(f"Best by {METRICS[m]}", best[m], None) for m in ("weighted_f1", "accuracy", "roc_auc")])
    ui.source_note(NB, [best["cell"]])
    ui.team_quote(text.SELECTION)
    ui.team_quote(text.COMPARISON_CAUTION)

# ---------------------------------------------------------------- validation

with validation_tab:
    st.subheader("Choosing the depth", anchor=False)
    ui.team_quote(text.TREE_TUNING)
    sweep = results["depth_sweep"]
    st.dataframe(
        pd.DataFrame({"max_depth": [int(d) for d in sweep["values"]],
                      "Validation weighted F1": list(sweep["values"].values())}),
        hide_index=True, width="stretch",
        column_config={"Validation weighted F1": st.column_config.NumberColumn(format="%.6f")},
    )
    st.markdown(f"Selected depth: **{sweep['best_depth']}**.")
    ui.source_note(NB, [sweep["cell"]])

    st.subheader("On the test set", anchor=False)
    ui.team_quote(text.FINAL_TREE)
    tested = results["test_metrics"]
    ui.stat_row([(label, f"{tested['values'][key]:.4f}", None) for key, label in METRICS.items()])
    ui.source_note(NB, [tested["cell"]])

    labels = [clf.short_label(c) for c in final["confusion_matrix"]["labels"]]
    matrix = final["confusion_matrix"]["matrix"]
    st.dataframe(
        pd.DataFrame(matrix, index=[f"Actual {c}" for c in labels], columns=[f"Predicted {c}" for c in labels]),
        width="stretch",
    )
    total = sum(map(sum, matrix))
    wrong = total - sum(matrix[i][i] for i in range(len(matrix)))
    st.markdown(f"{wrong:,} of {total:,} test vehicles were misclassified.")
    ui.source_note(NB, [tested["cell"]], "confusion matrix")

    st.subheader("Five-fold cross-validation", anchor=False)
    cv = results["cross_validation"]
    st.dataframe(
        pd.DataFrame(cv["rows"]).rename(columns={"model": "Model", "mean_weighted_f1": "Mean weighted F1",
                                                 "std": "Standard deviation"}),
        hide_index=True, width="stretch",
        column_config={"Mean weighted F1": SCORE, "Standard deviation": SCORE},
    )
    ui.source_note(NB, [cv["cell"]])
    ui.team_quote(text.CROSS_VALIDATION)

# ---------------------------------------------------------------- the fitted tree

with rules_tab:
    root = clf.tree_rules(model)
    consulted = clf.consulted_features(model, meta)
    ui.stat_row([
        ("Depth", f"{final['depth']}", None),
        ("Leaves", f"{final['leaves']}", "Final groups a vehicle can land in"),
        ("Inputs used", f"{len(consulted)} of {len(meta['feature_order'])}", "Inputs that appear in any rule"),
    ])
    st.caption("Everything on this tab is read directly from the exported model.")

    st.subheader("The first question", anchor=False)

    def mix(node: clf.TreeNode) -> str:
        return ", ".join(f"{n:,} {clf.short_label(c)}" for c, n in node.class_counts.items())

    st.markdown(
        f"Every vehicle is first tested against **{root.left_condition}**.\n\n"
        f"- **{root.left_condition}**: {root.left.samples:,} of the {root.samples:,} training vehicles "
        f"({mix(root.left)}).\n"
        f"- **{root.right_condition}**: {root.right.samples:,} training vehicles ({mix(root.right)})."
    )
    ui.team_todo(
        "What the first rule, Electric Range ≤ 0.5, means for this dataset and for the model — "
        "an interpretation in the team's own words."
    )

    st.subheader(f"All {final['leaves']} rules", anchor=False)
    st.dataframe(
        pd.DataFrame([
            {"Rule": i, "If": " AND ".join(conditions), "Predicts": clf.short_label(leaf.prediction),
             "Training vehicles": leaf.samples}
            for i, (conditions, leaf) in enumerate(root.rules(), start=1)
        ]),
        hide_index=True, width="stretch",
        column_config={"If": st.column_config.TextColumn(width="large")},
    )

    def outline(node: clf.TreeNode, depth: int = 0) -> list[str]:
        pad = "    " * depth
        if node.is_leaf:
            return [f"{pad}→ {clf.short_label(node.prediction)}   ({node.samples:,} vehicles: {mix(node)})"]
        return ([f"{pad}if {node.left_condition}:"] + outline(node.left, depth + 1)
                + [f"{pad}else ({node.right_condition}):"] + outline(node.right, depth + 1))

    with st.expander("The whole tree as indented rules", icon=":material/account_tree:"):
        st.code("\n".join(outline(root)), language=None)

    st.subheader("Which inputs matter", anchor=False)
    unused = [c for c in meta["feature_order"] if c not in consulted]
    st.markdown(
        "Only these inputs appear in any rule: " + ", ".join(f"**{c}**" for c in consulted) + ".\n\n"
        "The others never change this model's prediction: " + ", ".join(unused) + "."
    )
    importances = pd.DataFrame(clf.feature_importances(model), columns=["Feature", "Importance"])
    st.bar_chart(importances, x="Feature", y="Importance", horizontal=True, sort="-Importance")
    st.caption(
        "Impurity-based importances of the fitted tree. They describe the model as a whole, "
        "not any single prediction."
    )

# ---------------------------------------------------------------- the team's caution

st.header("The team's caution", anchor=False)
ui.team_quote(text.LEAKAGE_CAUTION)
ui.team_quote(text.RANGE_AND_MODEL_CAUTION)

ui.next_chapter("vehicle_predict")
