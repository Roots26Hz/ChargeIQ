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
    """Threshold for display: no thousands separator, so years read as years (2020.5, not 2,020.5)."""
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _split(model, node: int, left: bool) -> tuple[str, str]:
    """(original column, readable condition) for one branch of a split node.

    The single source of wording for every rule shown in the app, so the full tree
    and an individual decision path can never describe the same split differently.
    """
    preprocess = model.named_steps["preprocess"]
    structure = model.named_steps["model"].tree_
    column, category, numeric_index = _encoded_columns(model)[structure.feature[node]]
    if category is None:  # scaled numeric: convert the threshold back to original units
        scaler = preprocess.named_transformers_["num"]
        raw = structure.threshold[node] * scaler.scale_[numeric_index] + scaler.mean_[numeric_index]
        return column, f"{column} {'≤' if left else '>'} {_fmt(raw)}"
    return column, f"{column} {'is not' if left else 'is'} {category}"  # one-hot: threshold is 0.5


def _class_counts(model, node: int) -> dict[str, int]:
    """How the training rows that reached a node were labelled."""
    tree = model.named_steps["model"]
    samples = int(tree.tree_.n_node_samples[node])
    value = tree.tree_.value[node][0]
    fractions = value / value.sum()  # recent scikit-learn stores fractions, older stores counts
    return {str(c): int(round(f * samples)) for c, f in zip(tree.classes_, fractions)}


def decision_path(row: pd.DataFrame, model=None) -> tuple[tuple[PathStep, ...], int]:
    """Return the split conditions this row satisfied, and the leaf it reached."""
    model = model if model is not None else load_model(TRACK)
    tree = model.named_steps["model"]
    encoded = model.named_steps["preprocess"].transform(row)
    nodes = tree.decision_path(encoded).indices
    leaf = int(tree.apply(encoded)[0])
    structure = tree.tree_

    steps = []
    for node in nodes:
        if node == leaf:
            break
        went_left = encoded[0, structure.feature[node]] <= structure.threshold[node]  # the tree's own test
        column, condition = _split(model, node, went_left)
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

    leaf_samples = int(model.named_steps["model"].tree_.n_node_samples[leaf])
    counts = _class_counts(model, leaf)
    if label != max(counts, key=counts.get):
        raise RuntimeError("decision path and pipeline prediction disagree")  # should be impossible

    return Prediction(label, short_label(label), leaf, leaf_samples, counts, path)


# ---------------------------------------------------------------- the whole tree


@dataclass(frozen=True)
class TreeNode:
    """One node of the fitted tree, in original units. Leaves have no children."""

    node_id: int
    samples: int                        # training rows that reached this node
    class_counts: dict[str, int]
    left_condition: str | None = None   # condition for taking the left branch
    right_condition: str | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None

    @property
    def prediction(self) -> str:
        return max(self.class_counts, key=self.class_counts.get)

    def leaves(self) -> list["TreeNode"]:
        return [self] if self.is_leaf else self.left.leaves() + self.right.leaves()

    def rules(self, conditions: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], "TreeNode"]]:
        """Every leaf with the full chain of conditions that leads to it, left to right."""
        if self.is_leaf:
            return [(conditions, self)]
        return (self.left.rules(conditions + (self.left_condition,))
                + self.right.rules(conditions + (self.right_condition,)))


def tree_rules(model=None) -> TreeNode:
    """The complete fitted decision tree as nested, human-readable rules."""
    model = model if model is not None else load_model(TRACK)
    structure = model.named_steps["model"].tree_

    def build(node: int) -> TreeNode:
        samples, counts = int(structure.n_node_samples[node]), _class_counts(model, node)
        left, right = structure.children_left[node], structure.children_right[node]
        if left == -1:
            return TreeNode(node, samples, counts)
        return TreeNode(node, samples, counts,
                        _split(model, node, True)[1], _split(model, node, False)[1],
                        build(int(left)), build(int(right)))

    return build(0)


def readable_feature(encoded_index: int, model=None) -> str:
    """'Electric Range' for a numeric column, 'Make = BMW' for a one-hot column."""
    model = model if model is not None else load_model(TRACK)
    column, category, _ = _encoded_columns(model)[encoded_index]
    return column if category is None else f"{column} = {category}"


def consulted_features(model=None, meta: Mapping | None = None) -> list[str]:
    """Original input columns that appear in at least one split, in model column order.

    Any other input can never change this model's prediction.
    """
    model = model if model is not None else load_model(TRACK)
    meta = meta or load_metadata(TRACK)
    structure = model.named_steps["model"].tree_
    columns = _encoded_columns(model)
    used = {columns[f][0] for f in structure.feature if f >= 0}
    return [c for c in meta["feature_order"] if c in used]


def feature_importances(model=None) -> list[tuple[str, float]]:
    """Non-zero impurity-based importances, largest first.

    These are global — they summarise the whole tree, not any single prediction.
    """
    model = model if model is not None else load_model(TRACK)
    importances = model.named_steps["model"].feature_importances_
    ranked = sorted(((readable_feature(i, model), float(v)) for i, v in enumerate(importances) if v > 0),
                    key=lambda pair: -pair[1])
    return ranked
