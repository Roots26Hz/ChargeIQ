"""Inference for the Clustering track: place a charging session in one of the K-Means groups.

The model is the notebook's K-Means inside a Pipeline that also holds the fitted
median imputer, Yeo-Johnson transform and StandardScaler, so every session goes
through exactly the preprocessing used in fitting. Nothing here is ever fitted on
user input.

Only K-Means is used for new sessions. Agglomerative clustering has no way to place
a session it was not fitted on, so its results appear in the app as the notebook
reported them, never as an assignment.

Cluster numbers are arbitrary labels (the notebook's cell 30 says so too): group 2
is not "more" than group 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .artifacts import load_metadata, load_model

TRACK = "clustering"


class ValidationError(ValueError):
    """One or more inputs are missing, malformed, or outside the range the dataset shows."""

    def __init__(self, errors: Mapping[str, str]):
        self.errors = dict(errors)
        super().__init__("; ".join(f"{field}: {message}" for field, message in self.errors.items()))


@dataclass(frozen=True)
class Assignment:
    cluster: int                     # the K-Means group this session belongs to
    distances: dict[int, float]      # distance to each group's centre, in the model's standardised space
    cluster_size: int                # sample sessions K-Means put in the same group
    profile: dict[str, float]        # that group's mean session, in original units

    @property
    def runner_up(self) -> int:
        """The next-closest group."""
        return sorted(self.distances, key=self.distances.get)[1]


# ---------------------------------------------------------------- input domains


def input_fields(meta: Mapping | None = None) -> list[str]:
    meta = meta or load_metadata(TRACK)
    return list(meta["feature_order"])


def input_range(field: str, meta: Mapping | None = None) -> tuple[float, float]:
    """From 0 up to the largest value in the full 1,965,239-session dataset (cell 7). None can be negative."""
    meta = meta or load_metadata(TRACK)
    return 0.0, float(meta["notebook_results"]["full_dataset_statistics"]["features"][field]["max"])


def clusters(meta: Mapping | None = None) -> list[int]:
    meta = meta or load_metadata(TRACK)
    return sorted(int(c) for c in meta["kmeans"]["sizes"])


def profile(cluster: int, meta: Mapping | None = None) -> dict[str, float]:
    """A group's mean session in original units, as cell 31 reports it."""
    meta = meta or load_metadata(TRACK)
    return dict(meta["kmeans"]["profiles_mean"][str(cluster)])


def centre(cluster: int, meta: Mapping | None = None) -> dict[str, float]:
    """A group's K-Means centre, transformed back to original units. Not the same as its mean session."""
    meta = meta or load_metadata(TRACK)
    return dict(meta["kmeans"]["centres_original_units"][str(cluster)])


# ---------------------------------------------------------------- validation


def _number(value: Any) -> float | None:
    if isinstance(value, bool):  # bool is an int subclass; never a valid measurement
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def validate(inputs: Mapping[str, Any], meta: Mapping | None = None) -> dict[str, float]:
    """Check every input and return a clean copy, or raise ValidationError listing all problems."""
    meta = meta or load_metadata(TRACK)
    fields = input_fields(meta)
    errors: dict[str, str] = {}
    clean: dict[str, float] = {}

    for key in inputs:
        if key not in fields:
            errors[key] = "is not a model input"  # also rejects dayIndicator and the label columns

    for field in fields:
        value = inputs.get(field)
        number = _number(value)
        low, high = input_range(field, meta)
        if value is None:
            errors[field] = "is required"
        elif number is None:
            errors[field] = "must be a finite number"
        elif not low <= number <= high:
            errors[field] = f"must be between {low:g} and {high:g} (the range in the dataset)"
        else:
            clean[field] = number

    if errors:
        raise ValidationError(errors)
    return clean


def build_frame(rows: list[Mapping[str, Any]], meta: Mapping | None = None) -> pd.DataFrame:
    meta = meta or load_metadata(TRACK)
    order = input_fields(meta)
    clean = [validate(row, meta) for row in rows]
    return pd.DataFrame([[row[c] for c in order] for row in clean], columns=order, dtype=float)


# ---------------------------------------------------------------- assignment


def assign_many(rows: list[Mapping[str, Any]], model=None, meta: Mapping | None = None) -> np.ndarray:
    """The K-Means group of each session, in order."""
    meta = meta or load_metadata(TRACK)
    model = model if model is not None else load_model(TRACK)
    return model.predict(build_frame(rows, meta))


def assign(inputs: Mapping[str, Any], model=None, meta: Mapping | None = None) -> Assignment:
    """Place one session in a group, with its distance to every group's centre."""
    meta = meta or load_metadata(TRACK)
    model = model if model is not None else load_model(TRACK)
    frame = build_frame([inputs], meta)
    cluster = int(model.predict(frame)[0])
    distances = model.transform(frame)[0]  # KMeans.transform: distance to each centre
    if int(np.argmin(distances)) != cluster:
        raise RuntimeError("nearest centre and K-Means assignment disagree")  # should be impossible
    return Assignment(
        cluster=cluster,
        distances={c: float(d) for c, d in enumerate(distances)},
        cluster_size=int(meta["kmeans"]["sizes"][str(cluster)]),
        profile=profile(cluster, meta),
    )


# ---------------------------------------------------------------- the PCA map


def pca_point(inputs: Mapping[str, Any], model=None, meta: Mapping | None = None) -> tuple[float, float]:
    """Where a session lands on the notebook's two-component PCA plane (cell 25)."""
    meta = meta or load_metadata(TRACK)
    model = model if model is not None else load_model(TRACK)
    scaled = model[:-1].transform(build_frame([inputs], meta))[0]  # every step except K-Means
    components, mean = np.array(meta["pca"]["components"]), np.array(meta["pca"]["mean"])
    pc1, pc2 = components @ (scaled - mean)
    return float(pc1), float(pc2)


def map_points(meta: Mapping | None = None) -> pd.DataFrame:
    """A fixed random subset of the fitted sample on the PCA plane, with each session's K-Means group."""
    meta = meta or load_metadata(TRACK)
    return pd.DataFrame(meta["pca"]["map_points"])
