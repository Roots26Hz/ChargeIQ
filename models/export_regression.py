#!/usr/bin/env python3
"""Export the final Regression model trained in notebooks/regression.ipynb.

The notebook never saved its trained model, so this script reproduces it by
replaying the notebook's own preprocessing and training cells verbatim:

    cell  2  configuration: dataset location, NOMINAL_CAPACITY = 2.0
    cell  4  load metadata.csv, coerce Capacity / Re / Rct to numbers
    cell  7  one row per discharge cycle; cycle_index; Re/Rct from the preceding impedance test
    cell  9  per-cycle curve features (cell 10 reads each cycle's raw CSV)
    cell 13  raw_feats, the five curve features
    cell 18  drop the corrupted-log batteries and duplicates; fill gaps; cap Capacity outliers
    cell 21  target SOH_pct = 100 * Capacity / NOMINAL_CAPACITY
    cell 25  the nine model features
    cell 27  battery-grouped 80:20 split (GroupShuffleSplit)
    cell 28  StandardScaler, fitted on train only
    cell 30  append (scaled cycle_index)^2
    cell 61  GridSearchCV over Gradient Boosting, then refit with the best parameters
    cell 78  GroupKFold cross-validation of the selected model

Nothing is retuned or changed. Before writing anything, the replay is checked
against fingerprints taken from the notebook's stored outputs:

    1. cell 18/21  2,694 rows across 30 batteries
    2. cell 27     2,345 training rows / 24 batteries, 349 test rows / 6 batteries
    3. cell 62     best parameters and best CV R^2, to full stored precision
    4. cell 61     test R^2, RMSE and MAE, to 4 decimal places
    5. cell 78     the five GroupKFold fold R^2 values, to 4 decimal places

If any fingerprint disagrees the script exits non-zero and writes nothing.

The saved artifact is a single scikit-learn Pipeline. Cell 30's squared
cycle_index column becomes a ColumnTransformer step (passthrough + np.square), so
the model loads with no custom code; the script checks that the pipeline's
predictions are bit-for-bit identical to the notebook's scaler + hstack + model.

The raw data is not in the repository (575 MB, see data/README.md). It is read from
data/Regression_Dataset, or from the folder in CHARGEIQ_REGRESSION_DATA, exactly as
cell 2 does.

Usage (from the repository root):
    python models/export_regression.py           # verify, then write
    python models/export_regression.py --check   # verify only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "regression.ipynb"
DATASET_DIR = Path(os.environ.get("CHARGEIQ_REGRESSION_DATA", REPO / "data" / "Regression_Dataset"))
DATA_PATH = DATASET_DIR / "metadata.csv"
RAW_DIR = DATASET_DIR / "data"
OUT_MODEL = REPO / "models" / "regression_model.joblib"
OUT_META = REPO / "models" / "regression_metadata.json"

RANDOM_STATE = 42
NOMINAL_CAPACITY = 2.0  # cell 2, in Ah
CORRUPTED_BATTERIES = ["B0049", "B0050", "B0051", "B0052"]  # cell 18
RAW_FEATS = ["early_voltage_sag_rate", "early_temp_rise_rate", "charge_duration", "charge_max_temp",
             "charge_time_to_cv"]  # cell 13
FEATURE_COLS = ["ambient_temperature", "Re", "Rct", "cycle_index"] + RAW_FEATS  # cell 25
ENGINEERED = "cycle_index_squared"  # cell 30
GB_PARAM_GRID = {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [2, 3, 4]}
FINAL_MODEL_NAME = "8. Gradient Boosting Regression"


class FingerprintMismatch(RuntimeError):
    """The replay does not reproduce the notebook's stored results."""


# ---------------------------------------------------------------- notebook outputs


def _notebook_cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _find_cell(cells: list[dict], marker: str) -> tuple[int, str]:
    """Locate a code cell by a phrase in its source; return (index, its text output)."""
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code" and marker in "".join(cell["source"]):
            parts = []
            for out in cell.get("outputs", []):
                text = out.get("text") or (out.get("data") or {}).get("text/plain", "")
                parts.append("".join(text))
            return i, "\n".join(parts)
    raise FingerprintMismatch(f"no notebook cell contains {marker!r} — has the notebook changed?")


def _number(pattern: str, text: str, cell: int, what: str) -> float:
    match = re.search(pattern, text, re.M)
    if not match:
        raise FingerprintMismatch(f"could not read {what} from notebook cell {cell}")
    return float(match.group(1))


def read_notebook_results() -> dict:
    """Parse the values this script must reproduce, plus the notebook's reported results."""
    cells = _notebook_cells()

    i18, out18 = _find_cell(cells, "CORRUPTED_BATTERIES = ")
    i21, out21 = _find_cell(cells, 'df_clean["SOH_pct"] = 100')
    cleaning = {
        "rows": int(_number(r"duplicates:\s*(\d+)", out18, i18, "the cleaned row count")),
        "capacity_outliers_capped": int(_number(r"outliers outside .*?:\s*(\d+)", out18, i18, "the outlier count")),
        "batteries": int(_number(r"across (\d+) batteries", out21, i21, "the battery count")),
    }

    i27, out27 = _find_cell(cells, "Battery overlap between train/test:")
    split = {
        "train_rows": int(_number(r"^Train:\s*(\d+) rows", out27, i27, "training rows")),
        "train_batteries": int(_number(r"^Train:.*?/\s*(\d+) batteries", out27, i27, "training batteries")),
        "test_rows": int(_number(r"^Test:\s*(\d+) rows", out27, i27, "test rows")),
        "test_batteries": int(_number(r"^Test:.*?/\s*(\d+) batteries", out27, i27, "test batteries")),
    }

    i61, out61 = _find_cell(cells, "gb_grid = GridSearchCV(")
    i62, out62 = _find_cell(cells, 'print("Best CV score:", gb_grid.best_score_)')
    best_params = json.loads(re.search(r"Best parameters:\s*(\{.*\})", out62).group(1).replace("'", '"'))
    test_metrics = {
        key: _number(rf"{re.escape(FINAL_MODEL_NAME)}\s*\|.*?{label}=([\d.]+)", out61, i61, label)
        for key, label in (("r2", "R2"), ("rmse", "RMSE"), ("mae", "MAE"))
    }
    tuning = {"cell": i61, "best_params_cell": i62, "best_params": best_params,
              "best_cv_r2": _number(r"Best CV score:\s*([\d.]+)", out62, i62, "the best CV score")}

    i73, out73 = _find_cell(cells, "Rebuild the final comparison table after all ten algorithms")
    comparison = [
        {"model": name.strip(), "rank": int(rank), "r2": float(r2), "rmse": float(rmse), "mae": float(mae)}
        for name, rank, r2, rmse, mae in re.findall(
            r"^(\d+\.\s.+?)\s+(\d+)\s+([\d.-]+)\s+([\d.]+)\s+([\d.]+)\s*$", out73, re.M)
    ]

    i76, out76 = _find_cell(cells, "Consolidated original-vs-tuned comparison")
    tuned = {}
    for name, original, tuned_r2, best_cv in re.findall(
            r"^(\S.*Regression)\s*\n\s*Original test R²\s*:\s*([\d.-]+)\s*\n\s*Tuned test R²\s*:\s*([\d.-]+)"
            r"\s*\n.*?\n\s*Best CV R² during tuning:\s*([\d.-]+)", out76, re.M):
        tuned[name.strip()] = {"original_test_r2": float(original), "tuned_test_r2": float(tuned_r2),
                               "best_cv_r2": float(best_cv)}

    i78, out78 = _find_cell(cells, "group_kfold = GroupKFold(n_splits=5)")
    cross_validation = []
    for index, name in re.findall(r"^(\d+)\s+(\d+\.\s.+?Regression)\s", out78, re.M):
        first = re.search(rf"^{index}\s+{re.escape(name)}\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)", out78, re.M)
        rest = re.findall(rf"^{index}\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s*$", out78, re.M)
        if not first or len(rest) != 1:
            raise FingerprintMismatch(f"could not read the cross-validation row for {name!r} (cell {i78})")
        f4, f5, mean, std = map(float, rest[0])
        cross_validation.append({"model": name.strip(), "folds": [*map(float, first.groups()), f4, f5],
                                 "mean_r2": mean, "std_r2": std})

    i23, out23 = _find_cell(cells, "Row-level split")
    split_experiment = {
        "row_level_with_battery_id_r2": _number(r"^\(a\).*?R2 = ([\d.-]+)", out23, i23, "experiment (a)"),
        "battery_grouped_r2": _number(r"^\(b\).*?R2 = ([\d.-]+)", out23, i23, "experiment (b)"),
    }

    if len(comparison) != 10 or len(cross_validation) != 2 or FINAL_MODEL_NAME not in {
            r["model"] for r in comparison}:
        raise FingerprintMismatch("notebook outputs did not parse into the expected shapes")

    return {
        "cleaning": {"cell": i18, "target_cell": i21, **cleaning},
        "split": {"cell": i27, **split},
        "split_experiment": {"cell": i23, **split_experiment},
        "tuning": tuning,
        "test_metrics": {"cell": i61, "model": FINAL_MODEL_NAME, "values": test_metrics},
        "comparison": {"cell": i73, "rows": comparison},
        "original_vs_tuned": {"cell": i76, "models": tuned},
        "cross_validation": {"cell": i78, "rows": cross_validation},
    }


# ---------------------------------------------------------------- replay


def extract_discharge_early_features(path, early_seconds=300):
    """Cell 9 — verbatim."""
    d = pd.read_csv(path, usecols=["Voltage_measured", "Temperature_measured", "Time"])
    v0, t0 = d["Voltage_measured"].iloc[0], d["Temperature_measured"].iloc[0]
    window = min(early_seconds, 0.2 * d["Time"].max())
    early = d[d["Time"] <= window]
    if len(early) < 2:
        early = d.iloc[:max(2, int(0.1 * len(d)))]
    v_end, t_end, temp_end = (early["Voltage_measured"].iloc[-1], early["Time"].iloc[-1],
                              early["Temperature_measured"].iloc[-1])
    return {
        "early_voltage_sag_rate": (v0 - v_end) / t_end if t_end > 0 else np.nan,
        "early_temp_rise_rate": (temp_end - t0) / t_end if t_end > 0 else np.nan,
    }


def extract_charge_features(path, cv_threshold=4.15):
    """Cell 9 — verbatim."""
    d = pd.read_csv(path, usecols=["Voltage_measured", "Temperature_measured", "Time"])
    duration, max_temp = d["Time"].max(), d["Temperature_measured"].max()
    above = d[d["Voltage_measured"] >= cv_threshold]
    time_to_cv = above["Time"].iloc[0] if len(above) else duration
    return {"charge_duration": duration, "charge_max_temp": max_temp, "charge_time_to_cv": time_to_cv}


def prepare_data() -> dict:
    """Cells 4, 7, 10, 18, 21, 25, 27 — verbatim logic."""
    if not DATA_PATH.is_file() or not RAW_DIR.is_dir():
        raise FileNotFoundError(
            f"Regression data not found in {DATASET_DIR}. Download it as described in data/README.md, "
            "or set CHARGEIQ_REGRESSION_DATA to the folder that holds metadata.csv and data/."
        )

    df_raw = pd.read_csv(DATA_PATH)  # cell 4
    for col in ["Capacity", "Re", "Rct"]:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    # Cell 7. Its parsed_time column is never used downstream, so it is not recreated.
    discharge = df_raw[df_raw["type"] == "discharge"].sort_values(["battery_id", "test_id"]).copy()
    impedance = df_raw[df_raw["type"] == "impedance"].sort_values(["battery_id", "test_id"]).copy()
    charge = df_raw[df_raw["type"] == "charge"].sort_values(["battery_id", "test_id"]).copy()
    discharge["cycle_index"] = discharge.groupby("battery_id").cumcount() + 1
    disc_slim = discharge[["battery_id", "test_id", "ambient_temperature", "cycle_index", "Capacity", "filename"]]
    imp_slim = impedance[["battery_id", "test_id", "Re", "Rct"]]
    analysis_df = pd.merge_asof(
        disc_slim.sort_values("test_id"), imp_slim.sort_values("test_id"),
        on="test_id", by="battery_id", direction="backward",
    )

    disc_feats = []  # cell 10
    for _, row in discharge.iterrows():
        try:
            feats = extract_discharge_early_features(f"{RAW_DIR}/{row['filename']}")
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
            feats = {"early_voltage_sag_rate": np.nan, "early_temp_rise_rate": np.nan}
        feats["filename"] = row["filename"]
        disc_feats.append(feats)
    disc_feats_df = pd.DataFrame(disc_feats)

    chg_feats = []
    for _, row in charge.iterrows():
        try:
            feats = extract_charge_features(f"{RAW_DIR}/{row['filename']}")
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
            feats = {"charge_duration": np.nan, "charge_max_temp": np.nan, "charge_time_to_cv": np.nan}
        feats["battery_id"], feats["test_id"] = row["battery_id"], row["test_id"]
        chg_feats.append(feats)
    chg_feats_df = pd.DataFrame(chg_feats).sort_values("test_id")

    analysis_df = analysis_df.merge(disc_feats_df, on="filename", how="left")
    analysis_df = pd.merge_asof(
        analysis_df.sort_values("test_id"), chg_feats_df,
        on="test_id", by="battery_id", direction="backward",
    )

    df_clean = analysis_df[~analysis_df["battery_id"].isin(CORRUPTED_BATTERIES)].copy()  # cell 18
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    fill_cols = ["Re", "Rct"] + RAW_FEATS
    for col in fill_cols:
        df_clean[col] = df_clean.groupby("battery_id")[col].transform(lambda s: s.bfill().ffill())
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    q1, q3 = df_clean["Capacity"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = int(((df_clean["Capacity"] < lower) | (df_clean["Capacity"] > upper)).sum())
    df_clean["Capacity"] = df_clean["Capacity"].clip(lower, upper)

    df_clean["SOH_pct"] = 100 * df_clean["Capacity"] / NOMINAL_CAPACITY  # cell 21

    X = df_clean[FEATURE_COLS].copy()  # cells 25, 27
    y = df_clean["SOH_pct"].copy()
    groups = df_clean["battery_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))

    return {
        "df_raw": df_raw, "df_clean": df_clean, "capacity_bounds": (float(lower), float(upper)),
        "capacity_outliers": n_outliers,
        "X_train": X.iloc[train_idx], "X_test": X.iloc[test_idx],
        "y_train": y.iloc[train_idx], "y_test": y.iloc[test_idx],
        "groups_train": groups.iloc[train_idx], "groups_test": groups.iloc[test_idx],
    }


def append_cycle_index_squared(X_scaled: np.ndarray) -> np.ndarray:
    """Cell 30 — verbatim."""
    cyc_pos = FEATURE_COLS.index("cycle_index")
    return np.hstack([X_scaled, (X_scaled[:, [cyc_pos]]) ** 2])


def build_pipeline(scaler: StandardScaler, X_train_s: np.ndarray, model) -> Pipeline:
    """The notebook's scaler, cell-30 step and model as one Pipeline that needs no custom code to load."""
    cyc_pos = FEATURE_COLS.index("cycle_index")
    engineer = ColumnTransformer([
        ("features", "passthrough", list(range(len(FEATURE_COLS)))),
        (ENGINEERED, FunctionTransformer(np.square), [cyc_pos]),
    ]).fit(X_train_s)  # stateless; fitting only records the input width
    return Pipeline([("scale", scaler), ("engineer", engineer), ("model", model)])


def replay() -> dict:
    data = prepare_data()
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

    scaler = StandardScaler().fit(X_train)  # cell 28
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    X_train_fe, X_test_fe = append_cycle_index_squared(X_train_s), append_cycle_index_squared(X_test_s)

    gb_grid = GridSearchCV(  # cell 61
        estimator=GradientBoostingRegressor(random_state=RANDOM_STATE),
        param_grid=GB_PARAM_GRID, scoring="r2", cv=5, n_jobs=-1,
    )
    gb_grid.fit(X_train_fe, y_train)
    model = GradientBoostingRegressor(**gb_grid.best_params_, random_state=RANDOM_STATE)
    model.fit(X_train_fe, y_train)
    preds = model.predict(X_test_fe)

    folds = cross_val_score(  # cell 78
        clone(model), X_train_fe, y_train, cv=GroupKFold(n_splits=5), groups=data["groups_train"],
        scoring="r2", n_jobs=-1,
    )

    pipeline = build_pipeline(scaler, X_train_s, model)
    pipeline_preds = pipeline.predict(X_test)
    return {
        **data, "scaler": scaler, "model": model, "pipeline": pipeline, "predictions": preds,
        "pipeline_matches_notebook": bool(np.array_equal(pipeline_preds, preds)),
        "best_params": gb_grid.best_params_, "best_cv_r2": float(gb_grid.best_score_),
        "cv_folds": [float(f) for f in folds],
        "metrics": {
            "r2": float(r2_score(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "mae": float(mean_absolute_error(y_test, preds)),
        },
    }


def check_fingerprints(result: dict, notebook: dict) -> list[str]:
    problems = []

    def compare(what, got, expected):
        if got != expected:
            problems.append(f"{what}: replay {got} vs notebook {expected}")

    compare("cleaned rows", len(result["df_clean"]), notebook["cleaning"]["rows"])
    compare("batteries", result["df_clean"]["battery_id"].nunique(), notebook["cleaning"]["batteries"])
    compare("capacity outliers capped", result["capacity_outliers"], notebook["cleaning"]["capacity_outliers_capped"])

    split = notebook["split"]
    compare("training rows", len(result["X_train"]), split["train_rows"])
    compare("training batteries", result["groups_train"].nunique(), split["train_batteries"])
    compare("test rows", len(result["X_test"]), split["test_rows"])
    compare("test batteries", result["groups_test"].nunique(), split["test_batteries"])
    compare("batteries in both splits", set(result["groups_train"]) & set(result["groups_test"]), set())

    compare("best parameters", result["best_params"], notebook["tuning"]["best_params"])
    compare("best CV R²", result["best_cv_r2"], notebook["tuning"]["best_cv_r2"])  # stored at full precision

    for key, expected in notebook["test_metrics"]["values"].items():
        compare(f"test {key}", round(result["metrics"][key], 4), expected)

    gb_cv = next(r for r in notebook["cross_validation"]["rows"] if r["model"] == FINAL_MODEL_NAME)
    compare("GroupKFold folds", [round(f, 4) for f in result["cv_folds"]], gb_cv["folds"])

    if not result["pipeline_matches_notebook"]:
        problems.append("the exported pipeline does not reproduce the notebook's predictions bit for bit")
    return problems


# ---------------------------------------------------------------- metadata


def _plain(value):
    """numpy scalars -> plain Python, so the metadata is valid JSON."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_metadata(result: dict, notebook: dict) -> dict:
    X_train, X_test = result["X_train"], result["X_test"]
    df_raw, df_clean = result["df_raw"], result["df_clean"]
    model = result["model"]

    held_out = X_test.assign(battery_id=result["groups_test"].values, soh_pct=result["y_test"].values)
    held_out = held_out.sort_values(["battery_id", "cycle_index"])
    held_out_cycles = [
        {key: _plain(value) for key, value in row.items()}
        for row in held_out[["battery_id"] + FEATURE_COLS + ["soh_pct"]].to_dict("records")
    ]

    battery_temperatures = (df_clean.groupby("battery_id")["ambient_temperature"]
                            .agg(lambda s: sorted({_plain(v) for v in s})).to_dict())
    importances = sorted(zip(FEATURE_COLS + [ENGINEERED], model.feature_importances_), key=lambda p: -p[1])

    return {
        "track": "regression",
        "provenance": {
            "notebook": "notebooks/regression.ipynb",
            "notebook_sha256": _sha256(NOTEBOOK),
            "dataset": "data/Regression_Dataset (NASA battery dataset; local only, see data/README.md)",
            "dataset_metadata_sha256": _sha256(DATA_PATH),
            "method": (
                "The notebook did not save its trained model. This artifact was produced by "
                "replaying the notebook's preprocessing and training cells verbatim, and was "
                "written only after the replay reproduced the notebook's stored results exactly."
            ),
            "replayed_cells": [2, 4, 7, 9, 10, 13, 18, 21, 25, 27, 28, 30, 61, 78],
            "fingerprints_verified": [
                f"cells {notebook['cleaning']['cell']}, {notebook['cleaning']['target_cell']}: "
                "cleaned rows, batteries, capped outliers",
                f"cell {notebook['split']['cell']}: battery-grouped train/test split",
                f"cell {notebook['tuning']['best_params_cell']}: best parameters and best CV R², full precision",
                f"cell {notebook['test_metrics']['cell']}: test R², RMSE, MAE",
                f"cell {notebook['cross_validation']['cell']}: GroupKFold fold R² values",
                "pipeline predictions identical to the notebook's scaler + hstack + model",
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "target": "SOH_pct",
        "target_definition": "100 * Capacity / NOMINAL_CAPACITY",
        "nominal_capacity_ah": NOMINAL_CAPACITY,
        "feature_order": FEATURE_COLS,
        "dtypes": {col: str(dtype) for col, dtype in X_train.dtypes.items()},
        "engineered_features": {ENGINEERED: "(standardised cycle_index) ** 2, appended after scaling"},
        "numeric_ranges": {
            col: {"min": _plain(X_train[col].min()), "max": _plain(X_train[col].max()),
                  "median": _plain(X_train[col].median())}
            for col in FEATURE_COLS
        },
        "batteries": {
            "train": sorted(result["groups_train"].unique().tolist()),
            "test": sorted(result["groups_test"].unique().tolist()),
            "excluded_corrupted": CORRUPTED_BATTERIES,
            "ambient_temperatures": battery_temperatures,
        },
        "dataset": {
            "raw_rows": int(df_raw.shape[0]),
            "raw_row_types": {str(k): int(v) for k, v in df_raw["type"].value_counts().items()},
            "raw_batteries": int(df_raw["battery_id"].nunique()),
            "discharge_cycles": int((df_raw["type"] == "discharge").sum()),
            "rows": int(len(df_clean)),
            "batteries": int(df_clean["battery_id"].nunique()),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "capacity_outlier_bounds": [round(b, 6) for b in result["capacity_bounds"]],
            "capacity_outliers_capped": result["capacity_outliers"],
            "soh_pct": {k: float(v) for k, v in df_clean["SOH_pct"].describe().items()},
        },
        "final_model": {
            "name": "Gradient Boosting",
            "notebook_name": FINAL_MODEL_NAME,
            "estimator": "GradientBoostingRegressor",
            "params": {**{k: _plain(v) for k, v in result["best_params"].items()}, "random_state": RANDOM_STATE},
            "test_metrics": {k: round(v, 4) for k, v in result["metrics"].items()},
            "cv_folds_r2": [round(f, 4) for f in result["cv_folds"]],
            "feature_importances": [{"feature": f, "importance": float(v)} for f, v in importances],
        },
        "held_out_cycles": held_out_cycles,
        "notebook_results": notebook,
    }


# ---------------------------------------------------------------- writing


def _atomic_write(path: Path, write) -> None:
    """Write via a temp file in the same directory, so a failure never leaves a partial artifact."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    umask = os.umask(0)
    os.umask(umask)
    try:
        write(Path(tmp))
        os.chmod(tmp, 0o666 & ~umask)  # mkstemp creates 0600; give the artifact normal permissions
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify fingerprints, write nothing")
    args = parser.parse_args(argv)

    print(f"Reading expected results from {NOTEBOOK.relative_to(REPO)} ...")
    notebook = read_notebook_results()
    shown = DATASET_DIR.relative_to(REPO) if DATASET_DIR.is_relative_to(REPO) else DATASET_DIR
    print(f"Replaying feature extraction and training on {shown} (reads every cycle file) ...")
    result = replay()

    problems = check_fingerprints(result, notebook)
    if problems:
        print("\nFINGERPRINT MISMATCH — nothing was written:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    m = result["metrics"]
    print("\nAll fingerprints match the notebook:")
    print(f"  rows           {len(result['df_clean'])} across {result['df_clean']['battery_id'].nunique()} batteries")
    print(f"  split          train {len(result['X_train'])} / test {len(result['X_test'])} rows, no shared battery")
    print(f"  tuning         {result['best_params']}  best CV R² {result['best_cv_r2']}")
    print(f"  test metrics   R² {m['r2']:.4f}  RMSE {m['rmse']:.4f}  MAE {m['mae']:.4f}")
    print(f"  GroupKFold     {', '.join(f'{f:.4f}' for f in result['cv_folds'])}")
    print("  pipeline       predictions identical to the notebook's")

    if args.check:
        print("\n--check: verification only, nothing written.")
        return 0

    metadata = build_metadata(result, notebook)
    _atomic_write(OUT_MODEL, lambda p: joblib.dump(result["pipeline"], p, compress=3))
    _atomic_write(OUT_META, lambda p: p.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"))

    print(f"\nWrote {OUT_MODEL.relative_to(REPO)}  ({OUT_MODEL.stat().st_size / 1024:.1f} KB)")
    print(f"Wrote {OUT_META.relative_to(REPO)}  ({OUT_META.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
