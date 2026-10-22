"""Battery track, chapter 6 — how the regression model was chosen and how well it does."""

import math

import pandas as pd
import streamlit as st

from content import battery as text
from core import regression as reg
from core.artifacts import load_metadata, load_model
from ui import components as ui

meta = load_metadata(reg.TRACK)
model = load_model(reg.TRACK)
results = meta["notebook_results"]
final = meta["final_model"]
NB = text.NB

SCORE = st.column_config.NumberColumn(format="%.4f")

ui.chapter_header(
    "Battery track · Chapter 6",
    "How the model learns",
    "Ten regression algorithms compared on unseen batteries, and the gradient-boosting model that was selected.",
)

pipeline_tab, comparison_tab, validation_tab, drivers_tab = st.tabs(
    ["Pipeline", "Ten models compared", "Validation", "What drives it"]
)

# ---------------------------------------------------------------- pipeline

with pipeline_tab:
    params = ", ".join(f"{k}={v}" for k, v in final["params"].items())
    st.markdown(
        f"1. **{len(meta['feature_order'])} inputs** arrive in a fixed order.\n"
        f"2. All of them are standardised with `StandardScaler`.\n"
        f"3. One feature is added: **cycle_index²**, the square of the standardised cycle index.\n"
        f"4. A `{final['estimator']}({params})` makes the prediction, fitted on "
        f"**{meta['dataset']['train_rows']:,}** cycles from {len(meta['batteries']['train'])} batteries."
    )
    ui.source_note(NB, [28, 30, results["tuning"]["cell"]], "read from the exported model")
    ui.team_quote(text.SCALING)
    ui.team_quote(text.CYCLE_SQUARED)
    ui.team_quote(text.GRADIENT_BOOSTING)

# ---------------------------------------------------------------- comparison

with comparison_tab:
    ui.team_quote(text.SAME_TEST_SET)
    comparison = pd.DataFrame(results["comparison"]["rows"]).rename(
        columns={"model": "Model", "rank": "Rank", "r2": "R²", "rmse": "RMSE", "mae": "MAE"})
    comparison.insert(2, "Selected", comparison["Model"] == final["notebook_name"])
    st.dataframe(comparison[["Rank", "Model", "Selected", "R²", "RMSE", "MAE"]], hide_index=True, width="stretch",
                 column_config={"R²": SCORE, "RMSE": SCORE, "MAE": SCORE,
                                "Selected": st.column_config.CheckboxColumn(disabled=True)})
    ui.source_note(NB, [results["comparison"]["cell"]], "test-set scores on the 6 unseen batteries, ranked by R²")

    st.subheader("Tuning", anchor=False)
    tuning = results["original_vs_tuned"]
    st.dataframe(
        pd.DataFrame([{"Model": name, "Before tuning (test R²)": row["original_test_r2"],
                       "After tuning (test R²)": row["tuned_test_r2"], "Best CV R² while tuning": row["best_cv_r2"]}
                      for name, row in tuning["models"].items()]),
        hide_index=True, width="stretch",
        column_config={c: SCORE for c in ("Before tuning (test R²)", "After tuning (test R²)",
                                          "Best CV R² while tuning")},
    )
    best = results["tuning"]["best_params"]
    st.markdown("Parameters selected for Gradient Boosting: " + ", ".join(f"`{k}={v}`" for k, v in best.items()) + ".")
    ui.source_note(NB, [tuning["cell"], results["tuning"]["best_params_cell"]])
    ui.team_quote(text.GB_TUNING)
    ui.team_quote(text.TUNING_SUMMARY)

# ---------------------------------------------------------------- validation

with validation_tab:
    st.subheader("On the 6 unseen batteries", anchor=False)
    tested = results["test_metrics"]["values"]
    ui.stat_row([("R²", f"{tested['r2']:.4f}", None), ("RMSE", f"{tested['rmse']:.4f}", "SOH percentage points"),
                 ("MAE", f"{tested['mae']:.4f}", "SOH percentage points")])
    ui.source_note(NB, [results["test_metrics"]["cell"]])

    ui.figure("regression/models/02_predicted_vs_actual.png", "Predicted against actual SOH on the test batteries",
              NB, 81)
    ui.figure("regression/models/03_residual_plot.png", "Residuals against predicted SOH", NB, 82)
    ui.team_quote(text.DIAGNOSTICS)

    st.subheader("Battery by battery", anchor=False)
    rows = []
    for battery in reg.held_out_batteries(meta):
        cycles = reg.held_out_cycles(battery, meta)
        errors = [p - c.measured_soh
                  for p, c in zip(reg.predict_many([c.inputs for c in cycles], model, meta), cycles)]
        rows.append({
            "Battery": battery,
            "Ambient °C": ", ".join(str(t) for t in meta["batteries"]["ambient_temperatures"][battery]),
            "Cycles": len(cycles),
            "MAE": sum(abs(e) for e in errors) / len(errors),
            "RMSE": math.sqrt(sum(e * e for e in errors) / len(errors)),
            "Cycles outside training range": sum(bool(reg.outside_training_range(c.inputs, meta)) for c in cycles),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"MAE": st.column_config.NumberColumn(format="%.2f"),
                                "RMSE": st.column_config.NumberColumn(format="%.2f"),
                                "Cycles outside training range": st.column_config.NumberColumn(
                                    help="Cycles with at least one input outside the training range")})
    st.caption("Computed by the app from the exported model and the held-out cycles. Over all "
               f"{sum(r['Cycles'] for r in rows)} cycles together they give the notebook's R², RMSE and MAE above.")

    st.subheader("Cross-validation by battery", anchor=False)
    ui.team_quote(text.GROUPKFOLD)
    cv = results["cross_validation"]
    st.dataframe(
        pd.DataFrame([{"Model": r["model"], **{f"Fold {i}": f for i, f in enumerate(r["folds"], start=1)},
                       "Mean R²": r["mean_r2"], "Std": r["std_r2"]} for r in cv["rows"]]),
        hide_index=True, width="stretch",
        column_config={c: SCORE for c in [f"Fold {i}" for i in range(1, 6)] + ["Mean R²", "Std"]},
    )
    ui.source_note(NB, [cv["cell"]], "R² on each fold of the training batteries")
    ui.team_quote(text.GROUPKFOLD_WHY)

# ---------------------------------------------------------------- what drives it

with drivers_tab:
    importances = pd.DataFrame(reg.feature_importances(model, meta), columns=["Feature", "Importance"])
    st.bar_chart(importances, x="Feature", y="Importance", horizontal=True, sort="-Importance")
    st.caption(
        "Impurity-based importances of the selected Gradient Boosting model, read from the exported model. "
        "They describe the model as a whole, not any single prediction."
    )
    top = importances.iloc[0]
    st.markdown(f"**{top['Feature']}** alone accounts for **{top['Importance']:.0%}** of the total importance.")
    ui.team_note(
        "What the model leans on",
        "The model depends heavily on one feature. charge_time_to_cv, the time the preceding charge takes to "
        "reach 4.15 V, accounts for 66.6% of total importance. It follows SOH closely: the correlation is r "
        "= +0.83 across all training batteries and a median of r = +0.67 within individual batteries, "
        "positive in 18 of the 24. Because the relationship holds inside single batteries too, the feature "
        "is tracking each cell’s own ageing and not just identifying which battery a row came from. The "
        "physics fits this. A cell that stores less charge, or has higher internal resistance, hits the "
        "voltage threshold sooner, so a shorter time to CV signals a more degraded cell. The feature also "
        "sorts the batteries by test temperature. The 4 °C group reaches CV in a median of 340 s and "
        "averages 47% SOH, while the warmer groups take roughly 2,000–3,000 s and average 78–87% SOH. Some "
        "of the feature’s weight therefore comes from temperature group, not ageing alone, which is what "
        "cell 36 found for the linear model. By comparison, cycle count contributes very little: cycle_index "
        "and its square together account for only 1.7%.",
    )
    with st.expander("The notebook's importance plot is for a different model", icon=":material/info:"):
        tree = next(r for r in results["comparison"]["rows"] if "Decision Tree" in r["model"])
        st.markdown(f"The notebook plots importances for its **Decision Tree** regressor, ranked {tree['rank']} "
                    f"of {len(results['comparison']['rows'])}, not for the selected Gradient Boosting model above.")
        ui.figure("regression/models/01_decision_tree_feature_importance.png",
                  "Decision Tree feature importance (not the selected model)", NB, 54)

st.header("What these results cover", anchor=False)
ui.team_quote(text.SCOPE)

ui.next_chapter("battery_predict")
