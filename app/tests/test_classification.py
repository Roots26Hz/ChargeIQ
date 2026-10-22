"""Tests for the Classification track's exported model and inference module."""

from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
import pytest
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from conftest import REPO_ROOT, vehicle
from core import artifacts
from core import classification as clf

NOTEBOOK_CONFUSION_MATRIX = [[47509, 5], [0, 11325]]


# ---------------------------------------------------------------- the artifact


def test_artifact_files_exist():
    assert artifacts.model_path(clf.TRACK).is_file()
    assert artifacts.metadata_path(clf.TRACK).is_file()


def test_artifact_is_the_notebooks_final_model(model, meta):
    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["preprocess", "model"]
    tree = model.named_steps["model"]
    assert isinstance(tree, DecisionTreeClassifier)
    assert tree.max_depth == 7 and tree.random_state == 42
    assert meta["final_model"]["name"] == "Decision Tree"


def test_recorded_scikit_learn_version_matches_environment(meta):
    assert meta["environment"]["scikit-learn"] == sklearn.__version__


def test_artifact_was_exported_from_the_current_notebook_and_dataset(meta):
    """Fails if the notebook or dataset changed after export — re-run the export script."""
    for key in ("notebook", "dataset"):
        path = REPO_ROOT / meta["provenance"][key]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == meta["provenance"][f"{key}_sha256"], f"{path.name} changed since export"


def test_metadata_carries_the_notebooks_reported_results(meta):
    results = meta["notebook_results"]
    assert len(results["comparison"]["rows"]) == 10
    assert results["best_model"]["weighted_f1"] == "Decision Tree"
    assert [r["model"] for r in results["cross_validation"]["rows"]] == ["Decision Tree", "Bagging"]


# ---------------------------------------------------------------- reproduces the notebook


def test_reloaded_artifact_reproduces_notebook_confusion_matrix(model, replay):
    predictions = model.predict(replay["X_test"])
    cm = confusion_matrix(replay["y_test"], predictions, labels=model.classes_).tolist()
    assert cm == NOTEBOOK_CONFUSION_MATRIX


def test_artifact_predicts_identically_to_a_fresh_replay(model, replay):
    fresh = replay["tree"].predict(replay["preprocessor"].transform(replay["X_test"]))
    assert (model.predict(replay["X_test"]) == fresh).all()


def test_export_refuses_when_a_fingerprint_disagrees(export_module, replay):
    notebook = export_module.read_notebook_results()
    assert export_module.check_fingerprints(replay, notebook) == []

    tampered = copy.deepcopy(notebook)
    tampered["test_metrics"]["values"]["accuracy"] = 0.5
    assert export_module.check_fingerprints(replay, tampered)


# ---------------------------------------------------------------- input schema


def test_feature_order_matches_the_fitted_pipeline(model, meta):
    assert list(model.feature_names_in_) == meta["feature_order"]


def test_build_row_uses_training_column_order_and_dtypes(bev_inputs, meta):
    row = clf.build_row(bev_inputs)
    assert list(row.columns) == meta["feature_order"]
    assert {c: str(t) for c, t in row.dtypes.items()} == meta["dtypes"]
    assert len(row) == 1


def test_vehicle_age_uses_the_notebooks_2026_reference_year(bev_inputs):
    row = clf.build_row(bev_inputs)
    assert row.loc[0, "Vehicle Age"] == 2026 - 2020


def test_dropdown_domains_come_from_the_fitted_encoder(model, meta):
    encoder = model.named_steps["preprocess"].named_transformers_["cat"]
    for column, categories in zip(meta["categorical_features"], encoder.categories_):
        assert meta["category_domains"][column] == [str(c) for c in categories]


# ---------------------------------------------------------------- predictions


def test_known_bev_example(bev_inputs, model):
    result = clf.predict(bev_inputs)
    assert result.short_label == "BEV"
    assert result.label in model.classes_


def test_known_phev_example(phev_inputs, model):
    result = clf.predict(phev_inputs)
    assert result.short_label == "PHEV"
    assert result.label in model.classes_


def test_range_of_zero_follows_the_root_rule():
    result = clf.predict(vehicle("TESLA", "MODEL 3", 2023, 0))
    assert result.short_label == "BEV"
    assert result.path[0].condition == "Electric Range ≤ 0.5"


def test_leaf_shares_equal_predict_proba_and_are_zero_or_one(bev_inputs, phev_inputs, model):
    for inputs in (bev_inputs, phev_inputs):
        result = clf.predict(inputs)
        proba = model.predict_proba(clf.build_row(inputs))[0]
        shares = [result.leaf_class_shares[c] for c in model.classes_]
        assert np.allclose(shares, proba)
        assert set(np.round(proba, 12)) <= {0.0, 1.0}  # every leaf of this tree is pure
        assert sum(result.leaf_class_counts.values()) == result.leaf_samples


def test_inference_module_agrees_with_the_pipeline_on_real_test_rows(model, meta, replay):
    """Feed real held-out rows through validate → build_row → predict, as the GUI will."""
    X_test = replay["X_test"]
    domains = meta["category_domains"]
    usable = X_test[
        np.logical_and.reduce([X_test[c].isin(domains[c]) for c in meta["categorical_features"]])
    ]
    usable = usable[[
        r["Model"] in meta["make_to_models"].get(r["Make"], [])
        and r["City"] in meta["location_defaults"].get(r["County"], {})
        for _, r in usable.iterrows()
    ]]
    sample = usable.sample(n=300, random_state=0)

    for (_, row) in sample.iterrows():
        inputs = {c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in clf.input_fields()}
        assert clf.predict(inputs).label == model.predict(row.to_frame().T.astype(meta["dtypes"]))[0]


def test_decision_path_conditions_hold_for_the_input(bev_inputs, phev_inputs):
    for inputs in (bev_inputs, phev_inputs):
        for step in clf.predict(inputs).path:
            text = step.condition
            if " is not " in text:
                assert step.input_value != text.split(" is not ", 1)[1]
            elif " is " in text:
                assert step.input_value == text.split(" is ", 1)[1]
            else:
                op = "≤" if "≤" in text else ">"
                threshold = float(text.rsplit(" ", 1)[1].replace(",", ""))
                assert (step.input_value <= threshold) if op == "≤" else (step.input_value > threshold)


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "field, value, fragment",
    [
        ("Make", "NOT A MAKE", "does not appear in the training data"),
        ("Model Year", 1990, "between"),
        ("Model Year", 2020.5, "whole number"),
        ("Electric Range", -1, "between"),
        ("Electric Range", "fast", "finite number"),
        ("Electric Range", float("nan"), "finite number"),
        ("Electric Range", math.inf, "finite number"),
        ("Electric Range", True, "finite number"),
        ("County", 7, "must be text"),
    ],
)
def test_invalid_values_are_rejected_with_a_clear_message(bev_inputs, field, value, fragment):
    inputs = {**bev_inputs, field: value}
    with pytest.raises(clf.ValidationError) as excinfo:
        clf.predict(inputs)
    assert field in excinfo.value.errors
    assert fragment in excinfo.value.errors[field]


def test_missing_fields_are_all_reported(bev_inputs):
    inputs = {k: v for k, v in bev_inputs.items() if k not in ("Make", "Electric Range")}
    with pytest.raises(clf.ValidationError) as excinfo:
        clf.validate(inputs)
    assert excinfo.value.errors == {"Make": "is required", "Electric Range": "is required"}


def test_inconsistent_make_and_model_is_rejected(bev_inputs):
    with pytest.raises(clf.ValidationError) as excinfo:
        clf.validate({**bev_inputs, "Model": "VOLT"})
    assert "Model" in excinfo.value.errors


def test_inconsistent_county_and_city_is_rejected(bev_inputs):
    with pytest.raises(clf.ValidationError) as excinfo:
        clf.validate({**bev_inputs, "City": "Boise"})
    assert "City" in excinfo.value.errors


@pytest.mark.parametrize("column", ["Electric Vehicle Type",
                                    "Clean Alternative Fuel Vehicle (CAFV) Eligibility",
                                    "VIN (1-10)"])
def test_target_and_excluded_columns_are_rejected(bev_inputs, column):
    with pytest.raises(clf.ValidationError) as excinfo:
        clf.validate({**bev_inputs, column: "anything"})
    assert excinfo.value.errors[column] == "is not a model input"


def test_derived_vehicle_age_cannot_be_supplied(bev_inputs):
    with pytest.raises(clf.ValidationError) as excinfo:
        clf.validate({**bev_inputs, "Vehicle Age": 6})
    assert "Vehicle Age" in excinfo.value.errors


def test_unknown_location_lookup_is_rejected():
    with pytest.raises(clf.ValidationError):
        clf.location_defaults("King", "Boise")


# ---------------------------------------------------------------- artifact failures


def test_missing_artifact_gives_an_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts, "MODELS_DIR", tmp_path)
    artifacts.load_metadata.cache_clear()
    artifacts.load_model.cache_clear()
    try:
        with pytest.raises(artifacts.ArtifactError, match="export_classification.py"):
            artifacts.load_model(clf.TRACK)
    finally:
        artifacts.load_metadata.cache_clear()
        artifacts.load_model.cache_clear()


def test_scikit_learn_version_mismatch_is_refused(monkeypatch):
    class OtherSklearn:
        __version__ = "0.0.0"

    monkeypatch.setattr(artifacts, "sklearn", OtherSklearn)
    artifacts.load_model.cache_clear()
    try:
        with pytest.raises(artifacts.ArtifactError, match="scikit-learn"):
            artifacts.load_model(clf.TRACK)
    finally:
        artifacts.load_model.cache_clear()


# ---------------------------------------------------------------- the whole tree (Chapter 3)


def test_tree_rules_mirror_the_fitted_tree(model, meta):
    root = clf.tree_rules()
    leaves = root.leaves()
    assert root.samples == meta["dataset"]["train_rows"]
    assert len(leaves) == meta["final_model"]["leaves"]
    assert all(sum(leaf.class_counts.values()) == leaf.samples for leaf in leaves)
    assert sum(leaf.samples for leaf in leaves) == root.samples


def test_root_rule_is_the_electric_range_split():
    root = clf.tree_rules()
    assert (root.left_condition, root.right_condition) == ("Electric Range ≤ 0.5", "Electric Range > 0.5")


def test_every_leaf_is_pure():
    assert all(min(leaf.class_counts.values()) == 0 for leaf in clf.tree_rules().leaves())


@pytest.mark.parametrize("inputs_fixture", ["bev_inputs", "phev_inputs"])
def test_rule_tree_and_decision_path_describe_the_same_model(request, inputs_fixture):
    """Following a prediction's path through the rendered rules must reach its leaf."""
    prediction = clf.predict(request.getfixturevalue(inputs_fixture))
    node = clf.tree_rules()
    for step in prediction.path:
        assert step.condition in (node.left_condition, node.right_condition)
        node = node.left if step.condition == node.left_condition else node.right
    assert node.is_leaf
    assert node.node_id == prediction.leaf_id
    assert node.prediction == prediction.label


def test_thresholds_are_shown_without_thousands_separators():
    def conditions(node):
        if node.is_leaf:
            return []
        return [node.left_condition, node.right_condition] + conditions(node.left) + conditions(node.right)
    assert all("," not in c for c in conditions(clf.tree_rules()))
    assert "Model Year ≤ 2020.5" in conditions(clf.tree_rules())


def test_consulted_features_are_the_six_split_columns():
    assert clf.consulted_features() == [
        "County", "Model Year", "Make", "Model", "Electric Range", "Vehicle Location",
    ]


def test_feature_importances_are_readable_ranked_and_complete(model):
    ranked = clf.feature_importances()
    assert ranked[0][0] == "Electric Range"
    assert [v for _, v in ranked] == sorted((v for _, v in ranked), reverse=True)
    assert math.isclose(sum(v for _, v in ranked), 1.0)
    assert len(ranked) == int((model.named_steps["model"].feature_importances_ > 0).sum())
    assert ("Make = BMW" in dict(ranked))


@pytest.mark.parametrize("inputs_fixture", ["bev_inputs", "phev_inputs"])
def test_each_leaf_rule_is_the_full_path_to_that_leaf(request, inputs_fixture):
    prediction = clf.predict(request.getfixturevalue(inputs_fixture))
    rules = {leaf.node_id: conditions for conditions, leaf in clf.tree_rules().rules()}
    assert len(rules) == len(clf.tree_rules().leaves())
    assert rules[prediction.leaf_id] == tuple(step.condition for step in prediction.path)
