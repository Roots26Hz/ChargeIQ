"""Battery track (regression.ipynb) — the team's own words and reported numbers.

Each quote was extracted from its cell programmatically, so it is verbatim down to the
notebook's own characters; app/tests/test_content.py re-checks every one.
"""

from . import Quote

NB = "regression.ipynb"

# ---- Chapter 5: Meet the data

SOURCE = Quote(
    NB, 0,
    "Source: NASA Battery Dataset (Kaggle, `patrickfleith/nasa-battery-dataset`, `cleaned_dataset`) — "
    "`metadata.csv` plus the ~7,000 per-cycle raw CSVs it references via its `filename` column.",
)

MISSINGNESS = Quote(
    NB, 5,
    "`Capacity` only on `discharge` rows, `Re`/`Rct` only on `impedance` rows.",
)

TARGET = Quote(
    NB, 20,
    "**TARGET:** the regression target is **State-of-Health (SOH, %)**, not raw Capacity. `SOH_pct = 100 "
    "× Capacity / NOMINAL_CAPACITY`, defined for every row.",
)

FULL_CURVE = Quote(
    NB, 8,
    "That's expected, since Capacity **is defined as** the discharge current integrated over time — but "
    "it means a feature built from the **full** discharge curve (total duration, full-curve integral) is "
    "just the target restated, not a real predictor.",
)

EARLY_WINDOW = Quote(
    NB, 8,
    "**From each discharge cycle's own curve** — only the **first ~300 seconds**: voltage-sag rate and "
    "temperature-rise rate under load. This reflects internal resistance, not how long the discharge ran.",
)

PRECEDING_CHARGE = Quote(
    NB, 8,
    "**From the *preceding* charge cycle** (nearest-prior per battery, same join pattern as `Re`/`Rct`): "
    "full charge duration, max temperature, and time to reach the CC→CV voltage threshold. This is a "
    "separate physical process, so using its full curve doesn't restate this discharge's capacity.",
)

FADE = Quote(
    NB, 16,
    "The fade curves reproduce the same pattern seen before: a curved decline per battery, plus the same "
    "four short, anomalous runs.",
)

SAG_FEATURE = Quote(
    NB, 16,
    "The new `early_voltage_sag_rate` feature shows a visible negative relationship with Capacity — a "
    "faster voltage sag under load early in discharge tracks a more degraded (higher-impedance) cell, as "
    "hoped.",
)

CORRUPTED = Quote(
    NB, 16,
    "**B0049, B0050, B0051, B0052** again stand out in the per-battery summary — 25 cycles each vs. "
    "28–197 for every other battery, and B0052 alone is missing 84% of its Capacity readings. That gap "
    "in cycle count and missingness, on its own, is reason enough to treat these four as unreliable "
    "rather than as genuinely short-lived cells.",
)

CLEANING = Quote(
    NB, 19,
    "corrupted-log batteries dropped outright (too little reliable signal to salvage), remaining gaps in "
    "impedance/curve-derived features filled from each battery's **own** nearest reading, and Capacity "
    "outliers **capped** rather than dropped so `cycle_index` sequences stay intact for every remaining "
    "battery.",
)

COLLINEARITY = Quote(
    NB, 36,
    "`ambient_temperature` correlates 0.94 with `charge_max_temp`, 0.82 with `charge_time_to_cv`, and "
    "‑0.78 with `Rct` — because NASA's test protocol fixes a matching charge/discharge regime per "
    "ambient-temperature group, so these four features are largely standing in for **which temperature "
    "group a battery belongs to**, not four independent physical signals.",
)

PANEL_DATA = Quote(
    NB, 22,
    "**This is panel data** — repeated cycles from the same ~30 batteries, not independent rows — and "
    "`battery_id` is a candidate feature. If the *same* battery's cycles are scattered across both train "
    "and test (a random row-level split), the model can partly fit battery-specific quirks via that "
    "battery's one-hot column instead of learning the general relationship between the physical features "
    "and SOH.",
)

EASIER_QUESTION = Quote(
    NB, 24,
    "Either way, (a) is answering an easier question than the one we actually care about (\"how well does "
    "this work on a battery the model has never seen?\").",
)

THE_FIX = Quote(
    NB, 24,
    "**The fix:** split by **battery**, not by row — every cycle from a given battery goes entirely into "
    "train *or* test, never both — and **drop `battery_id` as a model feature**, since a battery-level "
    "split makes it structurally unable to generalise to unseen batteries anyway (its one-hot column is "
    "all zeros for every test battery).",
)

# ---- Chapter 6: How the model learns

SCALING = Quote(
    NB, 26,
    "`StandardScaler` is fit on the training split only.",
)

CYCLE_SQUARED = Quote(
    NB, 29,
    "`cycle_index_squared` — Capacity/SOH fades in a curve, not a line, so squaring the (already-scaled) "
    "cycle index lets the linear-family models below capture that curvature.",
)

GRADIENT_BOOSTING = Quote(
    NB, 60,
    "Gradient Boosting builds a sequence of weak prediction models, usually decision trees, where each "
    "new tree attempts to reduce the errors made by the previous trees.",
)

SAME_TEST_SET = Quote(
    NB, 72,
    "All ten models were evaluated using the same held-out test set containing batteries that were not "
    "present in the training set.",
)

GB_TUNING = Quote(
    NB, 66,
    "Hyperparameter tuning improved the Gradient Boosting model's test R² from 0.8053 to 0.8153, "
    "resulting in an improvement of 0.01.",
)

TUNING_SUMMARY = Quote(
    NB, 67,
    "These results show that hyperparameter tuning can improve one model while not improving another, "
    "depending on the selected hyperparameters and data split.",
)

GROUPKFOLD = Quote(
    NB, 77,
    "GroupKFold is used so that cycles from the same battery do not appear in both the training and "
    "validation folds.",
)

GROUPKFOLD_WHY = Quote(
    NB, 83,
    "Keeping battery groups separate during validation helps reduce information leakage and provides a "
    "more realistic estimate of generalisation to unseen batteries.",
)

DIAGNOSTICS = Quote(
    NB, 83,
    "The predicted-versus-actual plot shows how closely the model predictions follow the actual SOH "
    "values. The residual plot helps identify systematic errors or changes in error spread.",
)

SCOPE = Quote(
    NB, 83,
    "The results apply to battery-wise generalisation, where test batteries were kept separate from the "
    "training batteries.",
)

# ---- Descriptions the input table takes from data/README.md (checked by test_content.py)

README = "data/README.md"
README_AMBIENT = "Test-protocol temperature group (°C)"
README_RESISTANCE = "Electrolyte and charge-transfer resistance"

# What each model input measures, described from the notebook's code (cells 7 and 9) and,
# for the first three, from data/README.md.
INPUT_DESCRIPTIONS = {
    "ambient_temperature": README_AMBIENT,
    "Re": "Electrolyte resistance, from the most recent impedance test before this cycle",
    "Rct": "Charge-transfer resistance, from the same impedance test",
    "cycle_index": "This battery's discharge-cycle number, counting from 1",
    "early_voltage_sag_rate": "Voltage drop per second (V/s) over the first ~300 s of this discharge",
    "early_temp_rise_rate": "Temperature rise per second over the same window",
    "charge_duration": "Length of the preceding charge cycle (s)",
    "charge_max_temp": "Highest measured temperature during the preceding charge",
    "charge_time_to_cv": "Time until the preceding charge first reached 4.15 V, the CC→CV threshold (s)",
}
DESCRIBED_IN_README = ("ambient_temperature", "Re", "Rct")

QUOTES = (
    SOURCE, MISSINGNESS, TARGET, FULL_CURVE, EARLY_WINDOW,
    PRECEDING_CHARGE, FADE, SAG_FEATURE, CORRUPTED, CLEANING,
    COLLINEARITY, PANEL_DATA, EASIER_QUESTION, THE_FIX, SCALING,
    CYCLE_SQUARED, GRADIENT_BOOSTING, SAME_TEST_SET, GB_TUNING, TUNING_SUMMARY,
    GROUPKFOLD, GROUPKFOLD_WHY, DIAGNOSTICS, SCOPE,
)
FACTS = ()