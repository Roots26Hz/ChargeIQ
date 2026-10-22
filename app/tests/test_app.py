"""Smoke tests: the Streamlit app starts, and every chapter renders without error."""

from __future__ import annotations

import re
import subprocess

import pytest
from streamlit.testing.v1 import AppTest

from conftest import APP_DIR, REPO_ROOT
from content import home, vehicle
from core import classification as clf
from ui import navigation

ENTRYPOINT = str(APP_DIR / "streamlit_app.py")


@pytest.fixture
def app():
    at = AppTest.from_file(ENTRYPOINT, default_timeout=60)
    at.run()
    return at


def test_app_starts_without_error(app):
    assert not app.exception


def test_home_is_the_default_chapter(app):
    assert [t.value for t in app.title] == ["The problem"]


@pytest.mark.parametrize("chapter", navigation.CHAPTERS, ids=lambda ch: ch.key)
def test_every_chapter_renders(app, chapter):
    app.switch_page(f"chapters/{chapter.key}.py").run()
    assert not app.exception
    assert [t.value for t in app.title] == [chapter.title]


def test_every_chapter_file_exists():
    for chapter in navigation.CHAPTERS:
        assert chapter.path.is_file(), chapter.path


def test_hiding_unfinished_chapters_always_keeps_the_home_page():
    keys = [ch.key for ch in navigation.visible(show_unfinished=False)]
    assert navigation.DEFAULT_KEY in keys
    assert all(navigation.chapter(k).ready or k == navigation.DEFAULT_KEY for k in keys)


def test_streamlit_config_is_at_the_repository_root():
    """Community Cloud only reads .streamlit/config.toml from the repo root."""
    assert (REPO_ROOT / ".streamlit" / "config.toml").is_file()


def test_app_requirements_pin_the_versions_the_model_was_exported_with(meta):
    pins = dict(
        re.match(r"^([A-Za-z0-9_.-]+)==(\S+)$", line).groups()
        for line in (APP_DIR / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )
    for package in ("scikit-learn", "numpy", "pandas", "joblib"):
        assert pins.get(package) == meta["environment"][package], package
    assert "streamlit" in pins


# ---------------------------------------------------------------- finished chapters' content


def _render(app, key: str):
    app.switch_page(f"chapters/{key}.py").run()
    assert not app.exception
    return app


def _page_text(app) -> str:
    parts = [m.value for m in app.markdown] + [c.value for c in app.caption]
    parts += [f"{m.label} {m.value}" for m in app.metric] + [c.value for c in app.code]
    return "\n".join(parts)


@pytest.mark.parametrize("chapter", [ch for ch in navigation.CHAPTERS if ch.ready], ids=lambda ch: ch.key)
def test_finished_chapters_have_no_placeholder(app, chapter):
    assert "ciq-placeholder" not in _page_text(_render(app, chapter.key))


@pytest.mark.parametrize("chapter", [ch for ch in navigation.CHAPTERS if not ch.ready], ids=lambda ch: ch.key)
def test_unfinished_chapters_say_so(app, chapter):
    assert "ciq-placeholder" in _page_text(_render(app, chapter.key))


def test_home_quotes_each_tracks_objective(app):
    text = _page_text(app)
    for quote in home.QUOTES:
        assert quote.text in text
        assert f"cell {quote.cell}" in text


def test_data_chapter_shows_the_dataset_numbers(app, meta):
    text = _page_text(_render(app, "vehicle_data"))
    assert f"{meta['dataset']['rows']:,}" in text
    for count in meta["dataset"]["class_counts"].values():
        assert f"{count:,}" in text
    assert "235,354" in text and "60,000" in text and f"{meta['dataset']['test_rows']:,}" in text


def test_model_chapter_shows_the_real_tree_and_results(app, meta):
    root = clf.tree_rules()
    text = _page_text(_render(app, "vehicle_model"))
    assert root.left_condition in text
    assert f"{root.left.samples:,} of the {root.samples:,} training vehicles" in text
    assert "5 of 58,839 test vehicles were misclassified" in text
    assert "6 of 13" in text
    for leaf in root.leaves():  # the indented outline lists every leaf
        assert f"({leaf.samples:,} vehicles:" in text


FIGURE_CALL = re.compile(r'ui\.figure\(\s*"([^"]+)"')


def test_every_figure_shown_is_committed_to_the_repository():
    """The deployed app serves figures from git, so each one must be tracked, not just present locally."""
    tracked = set(subprocess.run(["git", "ls-files", "assets/figures"], cwd=REPO_ROOT,
                                 capture_output=True, text=True, check=True).stdout.split())
    shown = [m for path in (APP_DIR / "chapters").glob("*.py") for m in FIGURE_CALL.findall(path.read_text())]
    assert shown, "expected at least one figure"
    for figure in shown:
        assert f"assets/figures/{figure}" in tracked, figure


def test_every_track_has_a_data_model_and_predict_chapter():
    keys = {ch.key for ch in navigation.CHAPTERS}
    for track in ("vehicle", "battery", "charging"):
        assert {f"{track}_data", f"{track}_model", f"{track}_predict"} <= keys


# ---------------------------------------------------------------- the prediction chapter


def _predict_page(app):
    return _render(app, "vehicle_predict")


def _shown_label(app) -> str:
    card = next(m.value for m in app.markdown if "ciq-result-label" in m.value)
    return re.search(r'ciq-result-label">([^<]+)<', card).group(1)


def _form_inputs(app) -> dict:
    return {field: app.session_state[f"vp_{field}"] for field in clf.input_fields()}


def test_prediction_page_starts_from_the_first_example(app):
    page = _predict_page(app)
    first = vehicle.EXAMPLES[0]
    assert page.session_state["vp_Make"] == first.make and page.session_state["vp_Model"] == first.model
    assert _shown_label(page) == first.predicted


@pytest.mark.parametrize("example_key", ["volt", "i3_97", "i3_81", "ranger"])
def test_choosing_an_example_fills_the_form_and_shows_the_models_answer(app, example_key):
    example = next(e for e in vehicle.EXAMPLES if e.key == example_key)
    page = _predict_page(app)
    page.button_group(key="vp_example").set_value(example_key).run()
    assert not page.exception
    inputs = _form_inputs(page)
    assert (inputs["Make"], inputs["Model"], inputs["Model Year"], inputs["Electric Range"]) == (
        example.make, example.model, example.model_year, example.electric_range)
    assert _shown_label(page) == example.predicted == clf.predict(inputs).short_label


def test_the_shown_result_is_exactly_the_inference_modules(app):
    page = _predict_page(app)
    page.button_group(key="vp_example").set_value("i3_97").run()
    expected = clf.predict(_form_inputs(page))
    text = _page_text(page)
    for step in expected.path:
        assert step.condition in text
    assert f"{expected.leaf_samples:,}" in text


def test_the_misclassified_example_is_flagged(app):
    page = _predict_page(app)
    page.button_group(key="vp_example").set_value("ranger").run()
    assert any("of the 5 test vehicles it misclassifies" in w.value for w in page.warning)


def test_changing_make_resets_model_and_clears_the_example(app):
    page = _predict_page(app)
    page.selectbox(key="vp_Make").set_value("BMW").run()
    assert not page.exception
    assert page.session_state["vp_Model"] in clf.models_for_make("BMW")
    assert page.session_state["vp_example"] is None
    assert not page.info and not page.warning  # no claim about the data once the inputs are custom


def test_changing_county_refills_city_and_location_codes(app):
    page = _predict_page(app)
    page.selectbox(key="vp_County").set_value("Spokane").run()
    assert not page.exception
    city = page.session_state["vp_City"]
    assert city in clf.cities_for_county("Spokane")
    for field, value in clf.location_defaults("Spokane", city).items():
        assert page.session_state[f"vp_{field}"] == value


def test_electric_range_alone_can_flip_the_prediction(app):
    page = _predict_page(app)
    page.button_group(key="vp_example").set_value("i3_97").run()
    assert _shown_label(page) == "PHEV"
    page.number_input(key="vp_Electric Range").set_value(81).run()
    assert _shown_label(page) == "BEV"


def test_decision_path_shows_whole_numbers_without_a_decimal_point(app):
    page = _predict_page(app)
    page.button_group(key="vp_example").set_value("ranger").run()
    assert "this vehicle: `58` ·" in _page_text(page)
