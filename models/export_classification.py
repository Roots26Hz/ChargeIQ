#!/usr/bin/env python3
"""Export the final Classification model trained in notebooks/classification.ipynb.

The notebook never saved its trained model, so this script reproduces it by
replaying the notebook's own preprocessing and training cells verbatim:

    cell  3  load the dataset
    cell 19  median / mode imputation, drop duplicates
    cell 22  drop identifiers and the CAFV target proxy; target = Electric Vehicle Type
    cell 24  engineer Vehicle Age = 2026 - Model Year
    cell 26  stratified 80:20 train/test split
    cell 27  stratified 60,000-row training subset
    cell 30  ColumnTransformer(StandardScaler, OneHotEncoder), fitted on train only
    cell 45  max_depth validation sweep
    cell 46  final DecisionTreeClassifier(max_depth=best_depth)

Nothing is retuned or changed. Before writing anything, the replay is checked
against three fingerprints taken from the notebook itself:

    1. the cell-45 validation sweep, to 6 decimal places
    2. the five cell-46 test metrics, to 4 decimal places
    3. the cell-46 confusion matrix

If any fingerprint disagrees the script exits non-zero and writes nothing.

Usage (from the repository root):
    python models/export_classification.py           # verify, then write
    python models/export_classification.py --check   # verify only
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
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "classification.ipynb"
DATASET = REPO / "notebooks" / "Electric_Vehicle_Population_Data.csv"  # the file cell 3 reads
OUT_MODEL = REPO / "models" / "classification_model.joblib"
OUT_META = REPO / "models" / "classification_metadata.json"

RANDOM_STATE = 42
VEHICLE_AGE_REFERENCE_YEAR = 2026  # hard-coded in cell 24; inference must use the same year
TARGET = "Electric Vehicle Type"
POSITIVE_CLASS = "Plug-in Hybrid Electric Vehicle (PHEV)"  # cell 31
IDENTIFIER_COLUMNS = ["VIN (1-10)", "DOL Vehicle ID"]
TARGET_PROXY_COLUMNS = ["Clean Alternative Fuel Vehicle (CAFV) Eligibility"]
MAX_DEPTH_VALUES = [3, 5, 7, 10, 15, 20]

# Read from the confusion-matrix image stored in the output of the cell that trains
# tree_final (the only fingerprint that exists as an image rather than text).
EXPECTED_CONFUSION_MATRIX = [[47509, 5], [0, 11325]]

LOCATION_FIELDS = [
    "State",
    "Postal Code",
    "Legislative District",
    "2020 GEOID",
    "Vehicle Location",
    "Electric Utility",
]


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


def read_notebook_results() -> dict:
    """Parse the values this script must reproduce, plus the notebook's reported results."""
    cells = _notebook_cells()

    i45, out45 = _find_cell(cells, "max_depth_values = [3, 5, 7, 10, 15, 20]")
    sweep = {int(d): float(f) for d, f in re.findall(r"^\s*\d+\s+(\d+)\s+([\d.]+)\s*$", out45, re.M)}

    i46, out46 = _find_cell(cells, "Train final Decision Tree using all 60K")
    labels = {"Accuracy": "accuracy", "Precision": "precision", "Recall": "recall",
              "Weighted F1": "weighted_f1", "ROC-AUC": "roc_auc"}
    test_metrics = {}
    for label, key in labels.items():
        m = re.search(rf"^{re.escape(label)}\s*:\s*([\d.]+)\s*$", out46, re.M)
        if not m:
            raise FingerprintMismatch(f"could not read {label!r} from notebook cell {i46}")
        test_metrics[key] = float(m.group(1))
    best_depth = int(re.search(r"Best max_depth:\s*(\d+)", out46).group(1))

    i81, out81 = _find_cell(cells, "Consolidated Review 2 comparison of all 10 classifiers")
    row = re.compile(r"^\s*\d+\s+(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", re.M)
    comparison = [
        {"model": name.strip(), "accuracy": float(a), "precision": float(p), "recall": float(r),
         "weighted_f1": float(f), "roc_auc": float(auc)}
        for name, a, p, r, f, auc in row.findall(out81)
    ]

    i84, out84 = _find_cell(cells, "Select the two best models using the final comparison table")
    cv = [
        {"model": name.strip(), "mean_weighted_f1": float(mean), "std": float(std)}
        for name, mean, std in re.findall(r"^\s*\d+\s+(.+?)\s+([\d.]+)\s+([\d.]+)\s*$", out84, re.M)
    ]

    i86, out86 = _find_cell(cells, "Best model by weighted F1:")
    best = {key: re.search(rf"Best model by {phrase}:\s*(.+)", out86).group(1).strip()
            for key, phrase in [("weighted_f1", "weighted F1"), ("accuracy", "accuracy"), ("roc_auc", "ROC-AUC")]}

    if len(sweep) != len(MAX_DEPTH_VALUES) or len(comparison) != 10 or len(cv) != 2:
        raise FingerprintMismatch("notebook outputs did not parse into the expected shapes")

    return {
        "depth_sweep": {"cell": i45, "values": sweep, "best_depth": best_depth},
        "test_metrics": {"cell": i46, "values": test_metrics},
        "comparison": {"cell": i81, "rows": comparison},
        "cross_validation": {"cell": i84, "rows": cv},
        "best_model": {"cell": i86, **best},
    }


# ---------------------------------------------------------------- replay


def prepare_data(dataset: Path = DATASET) -> dict:
    """Cells 3, 19, 22, 24, 26, 27 — verbatim logic."""
    df = pd.read_csv(dataset)

    df_clean = df.copy()
    for column in df_clean.select_dtypes(include=np.number).columns:
        if df_clean[column].isnull().sum() > 0:
            df_clean[column] = df_clean[column].fillna(df_clean[column].median())
    for column in df_clean.select_dtypes(include="object").columns:
        if df_clean[column].isnull().sum() > 0:
            df_clean[column] = df_clean[column].fillna(df_clean[column].mode()[0])
    df_clean = df_clean.drop_duplicates()

    X = df_clean.drop(columns=[TARGET] + IDENTIFIER_COLUMNS + TARGET_PROXY_COLUMNS)
    y = df_clean[TARGET]
    X["Vehicle Age"] = VEHICLE_AGE_REFERENCE_YEAR - X["Model Year"]

    X_train_1, X_test, y_train_1, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, _, y_train, _ = train_test_split(
        X_train_1, y_train_1, train_size=60000, random_state=RANDOM_STATE, stratify=y_train_1
    )
    return {"df": df, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Cell 30 — verbatim configuration (unfitted)."""
    numerical = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical = X_train.select_dtypes(include="object").columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def depth_sweep(X_train_processed, y_train) -> dict[int, float]:
    """Cell 45 — validation split carved from the 60k training data."""
    X_dt_train, X_dt_val, y_dt_train, y_dt_val = train_test_split(
        X_train_processed, y_train, test_size=0.20, random_state=RANDOM_STATE, stratify=y_train
    )
    scores = {}
    for depth in MAX_DEPTH_VALUES:
        model = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        model.fit(X_dt_train, y_dt_train)
        scores[depth] = f1_score(y_dt_val, model.predict(X_dt_val), average="weighted")
    return scores


def replay() -> dict:
    data = prepare_data()
    preprocessor = build_preprocessor(data["X_train"])
    X_train_processed = preprocessor.fit_transform(data["X_train"])
    X_test_processed = preprocessor.transform(data["X_test"])

    sweep = depth_sweep(X_train_processed, data["y_train"])
    # Cell 45's own selection rule: idxmax over the sweep table, which keeps the first
    # (shallowest) depth when several tie.
    sweep_table = pd.DataFrame({"max_depth": list(sweep), "Weighted F1": list(sweep.values())})
    best_depth = int(sweep_table.loc[sweep_table["Weighted F1"].idxmax(), "max_depth"])

    tree = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE)
    tree.fit(X_train_processed, data["y_train"])

    y_pred = tree.predict(X_test_processed)
    y_test = data["y_test"]
    y_test_bin = (y_test == POSITIVE_CLASS).astype(int)
    positive_col = list(tree.classes_).index(POSITIVE_CLASS)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_test_bin, tree.predict_proba(X_test_processed)[:, positive_col]),
    }
    cm = confusion_matrix(y_test, y_pred, labels=tree.classes_).tolist()
    return {**data, "preprocessor": preprocessor, "tree": tree, "sweep": sweep,
            "best_depth": best_depth, "metrics": metrics, "confusion_matrix": cm}


def check_fingerprints(result: dict, notebook: dict) -> list[str]:
    problems = []

    expected_sweep = notebook["depth_sweep"]["values"]
    for depth, expected in expected_sweep.items():
        got = round(result["sweep"][depth], 6)
        if got != expected:
            problems.append(f"depth sweep d={depth}: replay {got:.6f} vs notebook {expected:.6f}")
    if result["best_depth"] != notebook["depth_sweep"]["best_depth"]:
        problems.append(f"best depth: replay {result['best_depth']} vs notebook "
                        f"{notebook['depth_sweep']['best_depth']}")

    for key, expected in notebook["test_metrics"]["values"].items():
        got = round(result["metrics"][key], 4)
        if got != expected:
            problems.append(f"test {key}: replay {got} vs notebook {expected}")

    if result["confusion_matrix"] != EXPECTED_CONFUSION_MATRIX:
        problems.append(f"confusion matrix: replay {result['confusion_matrix']} "
                        f"vs notebook {EXPECTED_CONFUSION_MATRIX}")
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
    X_train = result["X_train"]
    preprocessor = result["preprocessor"]
    tree = result["tree"]
    numerical = list(preprocessor.transformers_[0][2])
    categorical = list(preprocessor.transformers_[1][2])
    encoder = preprocessor.named_transformers_["cat"]

    category_domains = {col: [str(c) for c in cats] for col, cats in zip(categorical, encoder.categories_)}

    make_to_models = {
        make: sorted(models.unique().tolist())
        for make, models in X_train.groupby("Make")["Model"]
    }
    location_defaults: dict[str, dict[str, dict]] = {}
    for (county, city), group in X_train.groupby(["County", "City"]):
        combo = group[LOCATION_FIELDS].value_counts().index[0]  # most common combination
        location_defaults.setdefault(county, {})[city] = {
            field: _plain(v) for field, v in zip(LOCATION_FIELDS, combo)
        }

    numeric_ranges = {
        col: {"min": _plain(X_train[col].min()), "max": _plain(X_train[col].max()),
              "median": _plain(X_train[col].median())}
        for col in numerical
    }

    names = preprocessor.get_feature_names_out()
    importances = [
        {"encoded_feature": str(names[i]), "importance": float(tree.feature_importances_[i])}
        for i in np.argsort(-tree.feature_importances_) if tree.feature_importances_[i] > 0
    ]

    df = result["df"]
    class_counts = df[TARGET].value_counts()

    return {
        "track": "classification",
        "provenance": {
            "notebook": "notebooks/classification.ipynb",
            "notebook_sha256": _sha256(NOTEBOOK),
            "dataset": "notebooks/Electric_Vehicle_Population_Data.csv",
            "dataset_sha256": _sha256(DATASET),
            "method": (
                "The notebook did not save its trained model. This artifact was produced by "
                "replaying the notebook's preprocessing and training cells verbatim, and was "
                "written only after the replay reproduced the notebook's stored results exactly."
            ),
            "replayed_cells": [3, 19, 22, 24, 26, 27, 30, 45, 46],
            "fingerprints_verified": [
                f"cell {notebook['depth_sweep']['cell']}: max_depth sweep, 6 decimal places",
                f"cell {notebook['test_metrics']['cell']}: accuracy, precision, recall, weighted F1, ROC-AUC",
                f"cell {notebook['test_metrics']['cell']}: confusion matrix {EXPECTED_CONFUSION_MATRIX}",
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "target": TARGET,
        "classes": [str(c) for c in tree.classes_],
        "positive_class": POSITIVE_CLASS,
        "feature_order": list(X_train.columns),
        "dtypes": {col: str(dtype) for col, dtype in X_train.dtypes.items()},
        "numerical_features": numerical,
        "categorical_features": categorical,
        "derived_features": {"Vehicle Age": "VEHICLE_AGE_REFERENCE_YEAR - Model Year"},
        "vehicle_age_reference_year": VEHICLE_AGE_REFERENCE_YEAR,
        "excluded_columns": {"identifiers": IDENTIFIER_COLUMNS, "target_proxy": TARGET_PROXY_COLUMNS},
        "category_domains": category_domains,
        "make_to_models": make_to_models,
        "location_defaults": location_defaults,
        "numeric_ranges": numeric_ranges,
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "train_rows": int(len(X_train)),
            "test_rows": int(len(result["X_test"])),
            "encoded_features": int(len(names)),
        },
        "final_model": {
            "name": "Decision Tree",
            "estimator": "DecisionTreeClassifier",
            "params": {"max_depth": int(result["best_depth"]), "random_state": RANDOM_STATE},
            "depth": int(tree.get_depth()),
            "leaves": int(tree.get_n_leaves()),
            "nodes": int(tree.tree_.node_count),
            "test_metrics": {k: round(float(v), 4) for k, v in result["metrics"].items()},
            "confusion_matrix": {"labels": [str(c) for c in tree.classes_],
                                 "matrix": result["confusion_matrix"]},
            "nonzero_feature_importances": importances,
        },
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
    print(f"Replaying training on {DATASET.relative_to(REPO)} ...")
    result = replay()

    problems = check_fingerprints(result, notebook)
    if problems:
        print("\nFINGERPRINT MISMATCH — nothing was written:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    m = result["metrics"]
    print("\nAll fingerprints match the notebook:")
    print(f"  depth sweep    {', '.join(f'd{d}={v:.6f}' for d, v in result['sweep'].items())}")
    print(f"  test metrics   acc {m['accuracy']:.4f}  prec {m['precision']:.4f}  rec {m['recall']:.4f}"
          f"  wF1 {m['weighted_f1']:.4f}  AUC {m['roc_auc']:.4f}")
    print(f"  confusion      {result['confusion_matrix']}")

    if args.check:
        print("\n--check: verification only, nothing written.")
        return 0

    pipeline = Pipeline([("preprocess", result["preprocessor"]), ("model", result["tree"])])
    metadata = build_metadata(result, notebook)

    _atomic_write(OUT_MODEL, lambda p: joblib.dump(pipeline, p, compress=3))
    _atomic_write(OUT_META, lambda p: p.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"))

    print(f"\nWrote {OUT_MODEL.relative_to(REPO)}  ({OUT_MODEL.stat().st_size / 1024:.1f} KB)")
    print(f"Wrote {OUT_META.relative_to(REPO)}  ({OUT_META.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
