"""The exported Regression model: what it is, what it reproduces, and how inputs are checked."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from conftest import reg


def _median_inputs(meta) -> dict:
    return {f: meta["numeric_ranges"][f]["median"] for f in reg.input_fields(meta)}


# ---------------------------------------------------------------- the artifact


def test_artifact_is_the_notebooks_three_steps(reg_model, reg_meta):
    assert list(reg_model.named_steps) == ["scale", "engineer", "model"]
    gb = reg_model.named_steps["model"]
    assert isinstance(gb, GradientBoostingRegressor)
    params = gb.get_params()
    assert {k: params[k] for k in reg_meta["final_model"]["params"]} == reg_meta["final_model"]["params"]


def test_recorded_scikit_learn_version_is_installed(reg_meta):
    assert reg_meta["environment"]["scikit-learn"] == sklearn.__version__


def test_scaler_was_fitted_on_the_nine_inputs_in_order(reg_model, reg_meta):
    assert list(reg_model.named_steps["scale"].feature_names_in_) == reg_meta["feature_order"]


def test_engineered_step_is_the_notebooks_squared_cycle_index(reg_model, reg_meta):
    """Cell 30: np.hstack([X_s, X_s[:, [cycle_index]] ** 2]) — reproduced exactly."""
    X = reg.build_frame([c.inputs for c in reg.held_out_cycles()], reg_meta)
    scaled = reg_model.named_steps["scale"].transform(X)
    position = reg_meta["feature_order"].index("cycle_index")
    expected = np.hstack([scaled, scaled[:, [position]] ** 2])
    assert np.array_equal(reg_model.named_steps["engineer"].transform(scaled), expected)


# ---------------------------------------------------------------- held-out batteries


def test_held_out_set_is_six_batteries_the_model_never_saw(reg_meta):
    cycles = reg.held_out_cycles()
    batteries = {c.battery_id for c in cycles}
    assert len(cycles) == reg_meta["dataset"]["test_rows"] == reg_meta["notebook_results"]["split"]["test_rows"]
    assert batteries == set(reg.held_out_batteries()) and len(batteries) == 6
    assert not batteries & set(reg_meta["batteries"]["train"])


def test_held_out_cycles_reproduce_the_notebooks_test_metrics(reg_meta):
    cycles = reg.held_out_cycles()
    measured = [c.measured_soh for c in cycles]
    predicted = reg.predict_many([c.inputs for c in cycles])
    got = {
        "r2": round(r2_score(measured, predicted), 4),
        "rmse": round(math.sqrt(mean_squared_error(measured, predicted)), 4),
        "mae": round(mean_absolute_error(measured, predicted), 4),
    }
    assert got == reg_meta["notebook_results"]["test_metrics"]["values"] == {"r2": 0.8153, "rmse": 8.0449,
                                                                              "mae": 5.6427}


def test_cycles_of_one_battery_are_in_cycle_order():
    for battery in reg.held_out_batteries():
        indices = [c.cycle_index for c in reg.held_out_cycles(battery)]
        assert indices == sorted(indices) and len(indices) == len(set(indices))


def test_every_held_out_cycle_is_a_valid_input(reg_meta):
    for cycle in reg.held_out_cycles():
        reg.validate(cycle.inputs, reg_meta)


def test_extrapolation_is_reported_not_rejected(reg_meta):
    """Battery B0029's impedance lies below anything in training; the model must say so, not refuse."""
    first = reg.held_out_cycles("B0029")[0]
    assert {"Re", "Rct"} <= set(reg.outside_training_range(first.inputs))
    assert reg.outside_training_range(_median_inputs(reg_meta)) == []


def test_input_range_spans_training_and_held_out_values(reg_meta):
    for field in reg.input_fields(reg_meta):
        low, high = reg.input_range(field, reg_meta)
        t_low, t_high = reg.training_range(field, reg_meta)
        assert low <= t_low and high >= t_high
        assert all(low <= c.inputs[field] <= high for c in reg.held_out_cycles())


# ---------------------------------------------------------------- prediction


def test_predict_is_the_pipeline_prediction(reg_model, reg_meta):
    inputs = _median_inputs(reg_meta)
    expected = reg_model.predict(reg.build_frame([inputs], reg_meta))[0]
    assert isinstance(reg.predict(inputs), float)
    assert reg.predict(inputs) == expected


def test_cycle_index_changes_the_prediction(reg_meta):
    early, late = _median_inputs(reg_meta), _median_inputs(reg_meta)
    early["cycle_index"], late["cycle_index"] = 1, 150
    assert reg.predict(early) != reg.predict(late)


def test_feature_importances_cover_all_ten_columns(reg_meta):
    ranked = reg.feature_importances()
    names = [n for n, _ in ranked]
    assert sorted(names) == sorted(reg_meta["feature_order"] + ["cycle_index_squared"])
    assert [v for _, v in ranked] == sorted((v for _, v in ranked), reverse=True)
    assert math.isclose(sum(v for _, v in ranked), 1.0)
    assert ranked[0][0] == reg_meta["final_model"]["feature_importances"][0]["feature"]


# ---------------------------------------------------------------- validation


def test_validation_reports_every_problem_at_once(reg_meta):
    bad = _median_inputs(reg_meta)
    bad.update({"Re": "abc", "Rct": float("nan"), "cycle_index": 3.5, "ambient_temperature": True,
                "charge_duration": 10 ** 9})
    del bad["charge_max_temp"]
    with pytest.raises(reg.ValidationError) as error:
        reg.validate(bad, reg_meta)
    assert set(error.value.errors) == {"Re", "Rct", "cycle_index", "ambient_temperature", "charge_duration",
                                       "charge_max_temp"}


@pytest.mark.parametrize("extra", ["cycle_index_squared", "battery_id", "SOH_pct", "Capacity"])
def test_non_inputs_are_rejected(reg_meta, extra):
    inputs = {**_median_inputs(reg_meta), extra: 1}
    with pytest.raises(reg.ValidationError) as error:
        reg.validate(inputs, reg_meta)
    assert extra in error.value.errors


def test_whole_number_inputs_are_returned_as_integers(reg_meta):
    clean = reg.validate({**_median_inputs(reg_meta), "cycle_index": 12.0}, reg_meta)
    assert clean["cycle_index"] == 12 and isinstance(clean["cycle_index"], int)


# ---------------------------------------------------------------- the notebook's reported results


def test_notebook_results_were_parsed_completely(reg_meta):
    results = reg_meta["notebook_results"]
    assert len(results["comparison"]["rows"]) == 10
    best = min(results["comparison"]["rows"], key=lambda r: r["rank"])
    assert best["model"] == reg_meta["final_model"]["notebook_name"]
    gb_cv = next(r for r in results["cross_validation"]["rows"] if r["model"] == best["model"])
    assert gb_cv["folds"] == reg_meta["final_model"]["cv_folds_r2"]


# ---------------------------------------------------------------- replay (needs the local raw data)


def test_replay_reproduces_every_fingerprint(reg_replay, reg_export_module):
    notebook = reg_export_module.read_notebook_results()
    assert reg_export_module.check_fingerprints(reg_replay, notebook) == []


def test_shipped_model_is_the_replayed_model(reg_replay, reg_model):
    X_test = reg_replay["X_test"]
    assert np.array_equal(reg_model.predict(X_test), reg_replay["predictions"])


def test_shipped_metadata_is_what_the_export_would_write_today(reg_replay, reg_export_module, reg_meta):
    rebuilt = reg_export_module.build_metadata(reg_replay, reg_export_module.read_notebook_results())
    assert json.loads(json.dumps(rebuilt)) == reg_meta
