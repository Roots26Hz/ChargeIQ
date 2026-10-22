"""Inference for the Regression track: estimate a battery cell's State of Health (SOH, %).

The model is the notebook's final Gradient Boosting regressor inside a Pipeline that
also holds the fitted StandardScaler and the notebook's engineered feature, the
squared standardised cycle_index (cell 30), so every prediction goes through exactly
the preprocessing used in training. Nothing here is ever fitted on user input.

The metadata ships the 349 held-out test cycles — six batteries the model never saw
in training — with their measured SOH, so predictions can be compared with reality.
Some of those cycles lie outside the training range of a feature; inputs are
accepted anywhere within the range the data shows, and outside_training_range()
reports when the model is being asked to extrapolate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .artifacts import load_metadata, load_model

TRACK = "regression"


class ValidationError(ValueError):
    """One or more inputs are missing, malformed, or outside the range the data shows."""

    def __init__(self, errors: Mapping[str, str]):
        self.errors = dict(errors)
        super().__init__("; ".join(f"{field}: {message}" for field, message in self.errors.items()))


@dataclass(frozen=True)
class HeldOutCycle:
    """One discharge cycle of a battery the model never saw in training."""

    battery_id: str
    cycle_index: int
    inputs: dict[str, Any]      # the nine model inputs, in model order
    measured_soh: float         # SOH_pct computed from the measured capacity


# ---------------------------------------------------------------- input domains


def input_fields(meta: Mapping | None = None) -> list[str]:
    """The nine model inputs, in model order. The engineered cycle_index² is derived, never supplied."""
    meta = meta or load_metadata(TRACK)
    return list(meta["feature_order"])


def held_out_cycles(battery_id: str | None = None, meta: Mapping | None = None) -> list[HeldOutCycle]:
    """Held-out cycles in battery and cycle order, optionally for one battery."""
    meta = meta or load_metadata(TRACK)
    cycles = [
        HeldOutCycle(row["battery_id"], int(row["cycle_index"]),
                     {f: row[f] for f in meta["feature_order"]}, float(row["soh_pct"]))
        for row in meta["held_out_cycles"]
    ]
    return [c for c in cycles if battery_id is None or c.battery_id == battery_id]


def held_out_batteries(meta: Mapping | None = None) -> list[str]:
    """The batteries held out from training, as split by the notebook."""
    meta = meta or load_metadata(TRACK)
    return list(meta["batteries"]["test"])


def training_range(field: str, meta: Mapping | None = None) -> tuple[float, float]:
    meta = meta or load_metadata(TRACK)
    r = meta["numeric_ranges"][field]
    return r["min"], r["max"]


_INPUT_RANGES: dict[int, dict[str, tuple[float, float]]] = {}


def input_range(field: str, meta: Mapping | None = None) -> tuple[float, float]:
    """The range the data shows for a field: the training rows together with the held-out cycles."""
    meta = meta or load_metadata(TRACK)
    ranges = _INPUT_RANGES.get(id(meta))
    if ranges is None:  # computed once per metadata object; load_metadata returns a shared one
        ranges = {}
        for f in input_fields(meta):
            low, high = training_range(f, meta)
            values = [row[f] for row in meta["held_out_cycles"]]
            ranges[f] = (min(low, *values), max(high, *values))
        _INPUT_RANGES[id(meta)] = ranges
    return ranges[field]


def outside_training_range(inputs: Mapping[str, Any], meta: Mapping | None = None) -> list[str]:
    """Inputs the model never saw values like during training, in model order."""
    meta = meta or load_metadata(TRACK)
    clean = validate(inputs, meta)
    out = []
    for field in input_fields(meta):
        low, high = training_range(field, meta)
        if not low <= clean[field] <= high:
            out.append(field)
    return out


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
    errors: dict[str, str] = {}
    clean: dict[str, Any] = {}

    for key in inputs:
        if key in meta["engineered_features"]:
            errors[key] = "is computed from cycle_index by the model; do not supply it"
        elif key not in fields:
            errors[key] = "is not a model input"  # also rejects the target and battery_id

    for field in fields:
        number = _number(inputs.get(field))
        low, high = input_range(field, meta)
        if inputs.get(field) is None:
            errors[field] = "is required"
        elif number is None:
            errors[field] = "must be a finite number"
        elif meta["dtypes"][field].startswith("int") and not number.is_integer():
            errors[field] = "must be a whole number"
        elif not low <= number <= high:
            errors[field] = f"must be between {low:g} and {high:g} (the range in the data)"
        else:
            clean[field] = int(number) if meta["dtypes"][field].startswith("int") else number

    if errors:
        raise ValidationError(errors)
    return clean


def build_frame(rows: list[Mapping[str, Any]], meta: Mapping | None = None) -> pd.DataFrame:
    """Validated rows as a DataFrame in the exact column order and dtypes the pipeline was fitted on."""
    meta = meta or load_metadata(TRACK)
    order = input_fields(meta)
    clean = [validate(row, meta) for row in rows]
    return pd.DataFrame([[row[c] for c in order] for row in clean], columns=order).astype(meta["dtypes"])


# ---------------------------------------------------------------- prediction


def predict_many(rows: list[Mapping[str, Any]], model=None, meta: Mapping | None = None) -> np.ndarray:
    """Predicted SOH (%) for each row, in order."""
    meta = meta or load_metadata(TRACK)
    model = model if model is not None else load_model(TRACK)
    return model.predict(build_frame(rows, meta))


def predict(inputs: Mapping[str, Any], model=None, meta: Mapping | None = None) -> float:
    """Predicted SOH (%) for one cycle."""
    return float(predict_many([inputs], model, meta)[0])


def feature_importances(model=None, meta: Mapping | None = None) -> list[tuple[str, float]]:
    """Impurity-based importances of the fitted model, largest first, including cycle_index².

    These are global — they summarise the whole ensemble, not any single prediction.
    """
    meta = meta or load_metadata(TRACK)
    model = model if model is not None else load_model(TRACK)
    names = input_fields(meta) + list(meta["engineered_features"])
    values = model.named_steps["model"].feature_importances_
    return sorted(((n, float(v)) for n, v in zip(names, values)), key=lambda pair: -pair[1])
