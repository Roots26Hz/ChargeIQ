"""Inference for the Classification track: predict a vehicle's Electric Vehicle Type.

The model is the notebook's final Decision Tree inside a Pipeline that also holds
the fitted ColumnTransformer, so every prediction goes through exactly the
preprocessing used in training. Nothing here is ever fitted on user input.

Because the model is a single decision tree, the explanation returned with each
prediction is exact: it is the sequence of split conditions this particular input
satisfied on its way to a leaf, translated back to original units. It is a local
explanation of one prediction, not a global feature-importance summary.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

import pandas as pd

from .artifacts import load_metadata, load_model

TRACK = "classification"


class ValidationError(ValueError):
    """One or more inputs are missing, malformed, outside the training domain, or inconsistent."""

    def __init__(self, errors: Mapping[str, str]):
        self.errors = dict(errors)
        super().__init__("; ".join(f"{field}: {message}" for field, message in self.errors.items()))


@dataclass(frozen=True)
class PathStep:
    """One split on the way to the leaf, phrased as the condition this input satisfied."""

    feature: str        # original input column
    condition: str      # e.g. "Electric Range ≤ 0.5" or "Make is not BMW"
    input_value: Any    # this input's value for that column
    node_samples: int   # training rows that reached the split


@dataclass(frozen=True)
class Prediction:
    label: str                          # full class name, e.g. "Battery Electric Vehicle (BEV)"
    short_label: str                    # "BEV" or "PHEV"
    leaf_id: int
    leaf_samples: int                   # training rows that ended in the same leaf
    leaf_class_counts: dict[str, int]   # how those training rows were labelled
    path: tuple[PathStep, ...]

    @property
    def leaf_class_shares(self) -> dict[str, float]:
        """Fraction of training rows in the leaf per class.

        For a decision tree this is exactly what predict_proba returns. It describes
        the leaf's training data; it is not a calibrated probability.
        """
        return {label: count / self.leaf_samples for label, count in self.leaf_class_counts.items()}


# ---------------------------------------------------------------- input domains


def short_label(label: str) -> str:
    match = re.search(r"\(([^)]+)\)\s*$", label)
    return match.group(1) if match else label


def input_fields(meta: Mapping | None = None) -> list[str]:
    """Columns a caller supplies, in model order. Derived columns are excluded."""
    meta = meta or load_metadata(TRACK)
    return [c for c in meta["feature_order"] if c not in meta["derived_features"]]


def models_for_make(make: str, meta: Mapping | None = None) -> list[str]:
    meta = meta or load_metadata(TRACK)
    return list(meta["make_to_models"].get(make, []))


def cities_for_county(county: str, meta: Mapping | None = None) -> list[str]:
    meta = meta or load_metadata(TRACK)
    return sorted(meta["location_defaults"].get(county, {}))


def location_defaults(county: str, city: str, meta: Mapping | None = None) -> dict[str, Any]:
    """Most common location codes for this county and city in the training data."""
    meta = meta or load_metadata(TRACK)
    try:
        return dict(meta["location_defaults"][county][city])
    except KeyError:
        raise ValidationError({"City": f"{city!r} is not a city of {county!r} in the training data"}) from None


# ---------------------------------------------------------------- validation


def _number(value: Any) -> float | None:
    if isinstance(value, bool):  # bool is an int subclass; never a valid measurement
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def validate(inputs: Mapping[str, Any], meta: Mapping | None = None) -> dict[str, Any]:
    """Check every input and return a clean copy, or raise ValidationError listing all problems."""
    meta = meta or load_metadata(TRACK)
    fields = input_fields(meta)
    categorical = set(meta["categorical_features"])
    errors: dict[str, str] = {}
    clean: dict[str, Any] = {}

    for key in inputs:
        if key in meta["derived_features"]:
            errors[key] = f"is derived from {meta['derived_features'][key].split(' - ')[-1]}; do not supply it"
        elif key not in fields:
            errors[key] = "is not a model input"  # also rejects the target and excluded columns

    for field in fields:
        value = inputs.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            errors[field] = "is required"
            continue

        if field in categorical:
            if not isinstance(value, str):
                errors[field] = "must be text"
            elif value not in meta["category_domains"][field]:
                errors[field] = f"{value!r} does not appear in the training data"
            else:
                clean[field] = value
            continue

        number = _number(value)
        bounds = meta["numeric_ranges"][field]
        if number is None:
            errors[field] = "must be a finite number"
        elif meta["dtypes"][field].startswith("int") and not number.is_integer():
            errors[field] = "must be a whole number"
        elif not bounds["min"] <= number <= bounds["max"]:
            errors[field] = f"must be between {bounds['min']:g} and {bounds['max']:g} (the training range)"
        else:
            clean[field] = int(number) if meta["dtypes"][field].startswith("int") else number

    if "Make" in clean and "Model" in clean and clean["Model"] not in meta["make_to_models"].get(clean["Make"], []):
        errors["Model"] = f"{clean['Model']!r} is not a {clean['Make']} model in the training data"
    if "County" in clean and "City" in clean and clean["City"] not in meta["location_defaults"].get(clean["County"], {}):
        errors["City"] = f"{clean['City']!r} is not a city of {clean['County']!r} in the training data"

    if errors:
        raise ValidationError(errors)
    return clean


def build_row(inputs: Mapping[str, Any], meta: Mapping | None = None) -> pd.DataFrame:
    """One-row DataFrame in the exact column order and dtypes the pipeline was fitted on."""
    meta = meta or load_metadata(TRACK)
    values = validate(inputs, meta)
    values["Vehicle Age"] = meta["vehicle_age_reference_year"] - values["Model Year"]
    order = meta["feature_order"]
    return pd.DataFrame([[values[c] for c in order]], columns=order).astype(meta["dtypes"])


# ---------------------------------------------------------------- prediction


@lru_cache(maxsize=None)
def _encoded_columns(model) -> tuple[tuple[str, str | None, int | None], ...]:
    """Map each encoded column to (original column, category or None, numeric index or None)."""
    preprocess = model.named_steps["preprocess"]
    numeric = list(preprocess.transformers_[0][2])
    categorical = list(preprocess.transformers_[1][2])
    encoder = preprocess.named_transformers_["cat"]

    columns: list[tuple[str, str | None, int | None]] = [(c, None, j) for j, c in enumerate(numeric)]
    for column, categories in zip(categorical, encoder.categories_):
        columns.extend((column, str(category), None) for category in categories)

    if len(columns) != len(preprocess.get_feature_names_out()):
        raise RuntimeError("encoded column map does not match the fitted preprocessor")
    return tuple(columns)


def _fmt(number: float) -> str:
    return f"{number:,.2f}".rstrip("0").rstrip(".")


def decision_path(row: pd.DataFrame, model=None) -> tuple[tuple[PathStep, ...], int]:
    """Return the split conditions this row satisfied, and the leaf it reached."""
    model = model if model is not None else load_model(TRACK)
    preprocess = model.named_steps["preprocess"]
    tree = model.named_steps["model"]
    scaler = preprocess.named_transformers_["num"]
    columns = _encoded_columns(model)

    encoded = preprocess.transform(row)
    nodes = tree.decision_path(encoded).indices
    leaf = int(tree.apply(encoded)[0])
    structure = tree.tree_

    steps = []
    for node in nodes:
        if node == leaf:
            break
        feature_index = structure.feature[node]
        threshold = structure.threshold[node]
        went_left = encoded[0, feature_index] <= threshold  # the tree's own comparison, in encoded space
        column, category, numeric_index = columns[feature_index]

        if category is None:
            raw = threshold * scaler.scale_[numeric_index] + scaler.mean_[numeric_index]
            condition = f"{column} {'≤' if went_left else '>'} {_fmt(raw)}"
        else:
            condition = f"{column} {'is not' if went_left else 'is'} {category}"

        value = row.iloc[0][column]
        value = value.item() if hasattr(value, "item") else value  # numpy scalar -> plain Python
        steps.append(PathStep(column, condition, value, int(structure.n_node_samples[node])))
    return tuple(steps), leaf


def predict(inputs: Mapping[str, Any], model=None, meta: Mapping | None = None) -> Prediction:
    """Validate the inputs, predict the vehicle type, and explain the path taken."""
    meta = meta or load_metadata(TRACK)
    model = model if model is not None else load_model(TRACK)
    row = build_row(inputs, meta)

    label = str(model.predict(row)[0])
    path, leaf = decision_path(row, model)

    tree = model.named_steps["model"]
    leaf_samples = int(tree.tree_.n_node_samples[leaf])
    value = tree.tree_.value[leaf][0]
    fractions = value / value.sum()  # recent scikit-learn stores fractions, older stores counts
    counts = {str(c): int(round(f * leaf_samples)) for c, f in zip(tree.classes_, fractions)}

    if label != str(tree.classes_[fractions.argmax()]):
        raise RuntimeError("decision path and pipeline prediction disagree")  # should be impossible

    return Prediction(label, short_label(label), leaf, leaf_samples, counts, path)
