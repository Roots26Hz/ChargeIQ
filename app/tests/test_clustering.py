"""The exported Clustering model: what it is, what it reproduces, and how sessions are placed."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.cluster import KMeans

from conftest import REPO_ROOT, clu

SAMPLE = REPO_ROOT / "notebooks" / "EV_clustering_sample_with_labels.csv"


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(SAMPLE)


def _session(connection=18.0, duration=2.0, energy=6.0) -> dict:
    return {"connectionTime_decimal": connection, "chargingDuration": duration, "kWhDelivered": energy}


# ---------------------------------------------------------------- the artifact


def test_artifact_is_the_notebooks_four_steps(clu_model, clu_meta):
    assert list(clu_model.named_steps) == ["imputer", "power", "scaler", "kmeans"]
    kmeans = clu_model.named_steps["kmeans"]
    assert isinstance(kmeans, KMeans)
    params = kmeans.get_params()
    assert {k: params[k] for k in clu_meta["kmeans"]["params"]} == clu_meta["kmeans"]["params"]
    assert clu_model.named_steps["power"].get_params()["standardize"] is False


def test_recorded_scikit_learn_version_is_installed(clu_meta):
    assert clu_meta["environment"]["scikit-learn"] == sklearn.__version__


def test_shipped_model_reassigns_every_sample_session_to_its_stored_label(clu_model, sample):
    """The strongest check there is: all 100,000 sessions, one by one."""
    labels = clu.assign_many(sample[clu.input_fields()].to_dict("records"), clu_model)
    assert np.array_equal(labels, sample["KMeans_Cluster"].to_numpy())


def test_metadata_matches_the_notebooks_reported_results(clu_meta):
    reported = clu_meta["notebook_results"]
    assert clu_meta["kmeans"]["sizes"] == {str(k): v for k, v in reported["kmeans_sizes"]["sizes"].items()}
    assert clu_meta["agglomerative"]["sizes"] == {
        str(k): v for k, v in reported["agglomerative_sizes"]["sizes"].items()}
    for algorithm, key in (("kmeans", "K-Means"),
                           ("agglomerative", "Agglomerative Clustering (Ward + kNN connectivity)")):
        for metric, value in clu_meta[algorithm]["metrics"].items():
            if metric != "rows":
                assert math.isclose(value, reported["metrics"]["full_precision"][key][metric], rel_tol=1e-6)
        assert {c: [round(v, 3) for v in p.values()] for c, p in clu_meta[algorithm]["profiles_mean"].items()} == {
            str(c): p for c, p in reported["profiles"][key].items()}


def test_dataset_sizes_are_the_notebooks(clu_meta):
    assert clu_meta["dataset"]["source_rows"] == 1_965_239
    assert clu_meta["dataset"]["sample_rows"] == 100_000 == sum(clu_meta["kmeans"]["sizes"].values())
    assert clu_meta["provenance"]["source_file"] == "SYNTHETIC_EV_DATA.csv"


# ---------------------------------------------------------------- assignment


def test_assign_reports_the_nearest_centre(clu_meta):
    result = clu.assign(_session())
    assert result.cluster == min(result.distances, key=result.distances.get)
    assert result.runner_up != result.cluster
    assert result.cluster_size == clu_meta["kmeans"]["sizes"][str(result.cluster)]
    assert result.profile == clu.profile(result.cluster)


@pytest.mark.parametrize("cluster", [0, 1, 2])
def test_each_centre_is_assigned_to_its_own_group(cluster):
    assert clu.assign(clu.centre(cluster)).cluster == cluster


def test_every_group_is_reachable_and_numbered_like_the_notebook(clu_meta):
    assert clu.clusters() == [0, 1, 2]
    assert len(clu_meta["kmeans"]["centres_original_units"]) == 3


# ---------------------------------------------------------------- validation


def test_validation_reports_every_problem_at_once():
    bad = {"connectionTime_decimal": 25, "chargingDuration": -1, "kWhDelivered": float("nan"), "dayIndicator": 3}
    with pytest.raises(clu.ValidationError) as error:
        clu.validate(bad)
    assert set(error.value.errors) == {"connectionTime_decimal", "chargingDuration", "kWhDelivered", "dayIndicator"}


@pytest.mark.parametrize("value", [None, "abc", True, float("inf")])
def test_malformed_values_are_rejected(value):
    with pytest.raises(clu.ValidationError) as error:
        clu.validate({**_session(), "kWhDelivered": value})
    assert "kWhDelivered" in error.value.errors


def test_the_dataset_extremes_are_accepted():
    clu.validate(_session(connection=24.0, duration=0.0, energy=0.0))
    clu.validate({f: clu.input_range(f)[1] for f in clu.input_fields()})


# ---------------------------------------------------------------- PCA map


def test_map_points_and_pca_point_share_one_projection(clu_replay):
    """pca_point() must put a session exactly where the notebook's PCA did."""
    X_original, X_pca = clu_replay["X_original"], clu_replay["X_pca"]
    for i in (0, 1234, 99_999):
        got = clu.pca_point(X_original.iloc[i].to_dict())
        assert np.allclose(got, X_pca[i], atol=1e-9)


def test_map_points_are_a_labelled_subset(clu_meta):
    points = clu.map_points()
    assert len(points) == 2000 and set(points["cluster"]) == {0, 1, 2}


# ---------------------------------------------------------------- replay and figures


def test_replay_reproduces_every_fingerprint(clu_replay, clu_export_module):
    notebook = clu_export_module.read_notebook_results()
    assert clu_export_module.check_fingerprints(clu_replay, notebook) == []


def test_shipped_metadata_is_what_the_export_would_write_today(clu_replay, clu_export_module, clu_meta):
    rebuilt = clu_export_module.build_metadata(clu_replay, clu_export_module.read_notebook_results())
    assert json.loads(json.dumps(rebuilt)) == clu_meta


def test_exported_figures_are_the_notebooks_own_images(clu_export_module):
    for name, png in clu_export_module.notebook_figures().items():
        assert (clu_export_module.FIGURES_DIR / name).read_bytes() == png, name
