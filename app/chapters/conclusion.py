"""Chapter 11 — what the project set out to do, what it found, and its limits."""

import pandas as pd
import streamlit as st

from content import battery, driver, home, vehicle
from core import classification as clf
from core import clustering as clu
from core import regression as reg
from core.artifacts import load_metadata
from ui import components as ui

vehicle_meta = load_metadata(clf.TRACK)
battery_meta = load_metadata(reg.TRACK)
driver_meta = load_metadata(clu.TRACK)

ui.chapter_header(
    "Chapter 11 · Wrap-up",
    "Conclusion",
    "The questions, the methods, the selected models and their measured performance, side by side.",
)

# ---------------------------------------------------------------- side by side

vehicle_test = vehicle_meta["final_model"]["test_metrics"]
battery_test = battery_meta["notebook_results"]["test_metrics"]["values"]
kmeans, agglomerative = driver_meta["kmeans"]["metrics"], driver_meta["agglomerative"]["metrics"]
kmeans_first = (kmeans["silhouette"] > agglomerative["silhouette"]
                and kmeans["davies_bouldin"] < agglomerative["davies_bouldin"]
                and kmeans["calinski_harabasz"] > agglomerative["calinski_harabasz"])

st.dataframe(
    pd.DataFrame([
        {"Track": "Vehicle · classification",
         "Data": f"{vehicle_meta['dataset']['rows']:,} registered vehicles",
         "Selected model": f"{vehicle_meta['final_model']['name']} (max_depth "
                           f"{vehicle_meta['final_model']['params']['max_depth']})",
         "Result": f"Weighted F1 {vehicle_test['weighted_f1']:.4f} on {vehicle_meta['dataset']['test_rows']:,} "
                   "test vehicles"},
        {"Track": "Battery · regression",
         "Data": f"{battery_meta['dataset']['rows']:,} cycles from {battery_meta['dataset']['batteries']} batteries",
         "Selected model": battery_meta["final_model"]["name"],
         "Result": f"R² {battery_test['r2']:.4f}, RMSE {battery_test['rmse']:.2f} SOH points on "
                   f"{len(battery_meta['batteries']['test'])} unseen batteries"},
        {"Track": "Driver · clustering",
         "Data": f"{driver_meta['dataset']['sample_rows']:,} synthetic charging sessions",
         "Selected model": f"K-Means (K = {driver_meta['kmeans']['params']['n_clusters']})",
         "Result": f"Silhouette {kmeans['silhouette']:.4f}"
                   + ("; first of the two algorithms on all three metrics" if kmeans_first else "")},
    ]),
    hide_index=True, width="stretch",
    column_config={"Result": st.column_config.TextColumn(width="large")},
)
st.caption("Read from the exported models' metadata; each figure is the one the notebook reports.")

# ---------------------------------------------------------------- per track, in the team's words


def track(title: str, objective, chosen: list, limits: list, try_it: str) -> None:
    st.header(title, anchor=False)
    ui.team_quote(objective)
    st.markdown("**How the model was chosen**")
    for quote in chosen:
        ui.team_quote(quote)
    st.markdown("**Its limits**")
    for quote in limits:
        ui.team_quote(quote)
    ui.chapter_link(try_it, "Try this model again", icon=":material/replay:")


track(":material/directions_car: The vehicle", home.VEHICLE_OBJECTIVE,
      [vehicle.SELECTION], [vehicle.LEAKAGE_CAUTION], "vehicle_predict")
track(":material/battery_charging_full: The battery", home.BATTERY_OBJECTIVE,
      [battery.SAME_TEST_SET], [battery.SCOPE], "battery_predict")
track(":material/ev_station: The driver", home.DRIVER_OBJECTIVE,
      [driver.PREFERRED], [driver.NO_GROUND_TRUTH], "charging_predict")

ui.next_chapter("home", "Back to the start")
