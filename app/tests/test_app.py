"""Smoke tests: the Streamlit app starts, and every chapter renders without error."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess

import pyarrow as pa
import pytest
from streamlit.testing.v1 import AppTest

from conftest import APP_DIR, REPO_ROOT
from content import home, vehicle
from core import classification as clf
from core import clustering as clu
from core import regression as reg
from ui import navigation

ENTRYPOINT = str(APP_DIR / "streamlit_app.py")


@pytest.fixture
def app():
    at = AppTest.from_file(ENTRYPOINT, default_timeout=60)
    at.run()
    return at


@pytest.fixture
def full_app(monkeypatch):
    """The app with unfinished chapters shown, as a developer sees it with SHOW_UNFINISHED = True."""
    monkeypatch.setattr(navigation, "SHOW_UNFINISHED", True)
    at = AppTest.from_file(ENTRYPOINT, default_timeout=60)
    at.run()
    return at


def test_app_starts_without_error(app):
    assert not app.exception


def test_home_is_the_default_chapter(app):
    assert [t.value for t in app.title] == ["The problem"]


@pytest.mark.parametrize("chapter", navigation.CHAPTERS, ids=lambda ch: ch.key)
def test_every_chapter_renders(full_app, chapter):
    full_app.switch_page(f"chapters/{chapter.key}.py").run()
    assert not full_app.exception
    assert [t.value for t in full_app.title] == [chapter.title]


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
def test_unfinished_chapters_say_so(full_app, chapter):
    assert "ciq-placeholder" in _page_text(_render(full_app, chapter.key))


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


@pytest.mark.parametrize("key, title", [("home", "Why these three questions?"),
                                        ("vehicle_model", "What a range of 0 really means"),
                                        ("battery_model", "What the model leans on")])
def test_team_notes_render_with_their_titles(app, key, title):
    notes = [m.value for m in _render(app, key).markdown if 'class="ciq-note"' in m.value]
    assert len(notes) == 1
    assert f'<p class="ciq-note-title">{title}</p>' in notes[0]


# ---------------------------------------------------------------- figures


FIGURE_WITH_CELL = re.compile(r'ui\.figure\(\s*"([^"]+)",.*?,\s*NB,\s*(\d+)\)', re.S)


def _notebook_images(notebook: str) -> dict[int, set[str]]:
    cells = json.loads((REPO_ROOT / "notebooks" / notebook).read_text(encoding="utf-8"))["cells"]
    images: dict[int, set[str]] = {}
    for i, cell in enumerate(cells):
        for out in cell.get("outputs", []):
            png = (out.get("data") or {}).get("image/png")
            if png:
                images.setdefault(i, set()).add(hashlib.sha256(base64.b64decode("".join(png))).hexdigest())
    return images


def test_every_figure_is_the_image_its_cited_cell_produced():
    """Compares the committed file (what the deployed app serves) with the notebook's stored output."""
    checked = 0
    for chapter in (APP_DIR / "chapters").glob("*.py"):
        for figure, cell in FIGURE_WITH_CELL.findall(chapter.read_text()):
            notebook = figure.split("/")[0] + ".ipynb"  # assets/figures/<track>/... comes from <track>.ipynb
            committed = subprocess.run(["git", "show", f"HEAD:assets/figures/{figure}"], cwd=REPO_ROOT,
                                       capture_output=True, check=True).stdout
            digest = hashlib.sha256(committed).hexdigest()
            assert digest in _notebook_images(notebook).get(int(cell), set()), f"{figure} is not cell {cell}'s output"
            checked += 1
    assert checked >= 8


# ---------------------------------------------------------------- battery chapters


def test_battery_data_chapter_shows_the_dataset_numbers(app, reg_meta):
    text = _page_text(_render(app, "battery_data"))
    data = reg_meta["dataset"]
    assert f"{data['batteries']} of {data['raw_batteries']}" in text
    assert f"{data['rows']:,}" in text and f"**{data['capacity_outliers_capped']}**" in text
    assert all(b in text for b in reg_meta["batteries"]["test"])


def test_battery_model_chapter_scores_every_unseen_battery(app, reg_meta):
    page = _render(app, "battery_model")
    per_battery = next(d.value for d in page.dataframe if "Battery" in d.value.columns)
    assert list(per_battery["Battery"]) == reg_meta["batteries"]["test"]
    assert per_battery["Cycles"].sum() == reg_meta["dataset"]["test_rows"]
    top = reg_meta["final_model"]["feature_importances"][0]
    assert f"**{top['feature']}** alone accounts for **{top['importance']:.0%}**" in _page_text(page)


def _battery_page(app):
    return _render(app, "battery_predict")


def _metric(page, label: str) -> str:
    return next(m.value for m in page.metric if m.label == label)


def test_battery_page_starts_on_the_longest_lived_unseen_battery(app):
    page = _battery_page(app)
    longest = max(reg.held_out_batteries(), key=lambda b: len(reg.held_out_cycles(b)))
    assert page.session_state["bp_battery"] == longest
    first = reg.held_out_cycles(longest)[0]
    assert _metric(page, "Measured SOH") == f"{first.measured_soh:.1f}%"
    assert _metric(page, "Predicted SOH") == f"{reg.predict(first.inputs):.1f}%"


def test_choosing_a_battery_and_cycle_shows_that_cycle(app):
    page = _battery_page(app)
    page.selectbox(key="bp_battery").set_value("B0029").run()
    cycles = reg.held_out_cycles("B0029")
    page.select_slider(key="bp_cycle").set_value(cycles[10].cycle_index).run()
    assert not page.exception
    assert _metric(page, "Measured SOH") == f"{cycles[10].measured_soh:.1f}%"
    assert _metric(page, "Predicted SOH") == f"{reg.predict(cycles[10].inputs):.1f}%"
    assert page.session_state["bp_Rct"] == cycles[10].inputs["Rct"]  # sliders follow the chosen cycle
    assert any("extrapolation" in i.value for i in page.info)       # B0029's impedance is out of range


def test_adjusting_an_input_updates_the_estimate_and_reset_restores_it(app):
    page = _battery_page(app)
    battery, cycle_index = page.session_state["bp_battery"], page.session_state["bp_cycle"]
    measured_inputs = next(c for c in reg.held_out_cycles(battery) if c.cycle_index == cycle_index).inputs
    page.slider(key="bp_cycle_index").set_value(150).run()
    expected = reg.predict({**measured_inputs, "cycle_index": 150})
    assert _metric(page, "Predicted SOH for these inputs") == f"{expected:.1f}%"
    page.button[0].click().run()
    assert page.session_state["bp_cycle_index"] == measured_inputs["cycle_index"]
    assert _metric(page, "Predicted SOH for these inputs") == f"{reg.predict(measured_inputs):.1f}%"


# ---------------------------------------------------------------- driver chapters


def test_charging_data_chapter_says_the_data_is_synthetic_and_shows_its_size(app, clu_meta):
    page = _render(app, "charging_data")
    assert any("synthetic" in i.value and "SYNTHETIC_EV_DATA.csv" in i.value for i in page.info)
    text = _page_text(page)
    assert f"{clu_meta['dataset']['source_rows']:,}" in text and f"{clu_meta['dataset']['sample_rows']:,}" in text


def test_charging_model_chapter_tables_match_the_metadata(app, clu_meta):
    page = _render(app, "charging_model")
    tables = [d.value for d in page.dataframe]
    groups = next(t for t in tables if "Sessions" in t.columns)
    assert list(groups["Sessions"]) == [clu_meta["kmeans"]["sizes"][g] for g in ("0", "1", "2")]
    scores = next(t for t in tables if "Silhouette rank" in t.columns)
    assert list(scores["Silhouette rank"]) == [1, 2]  # K-Means ranks first on all three, as in cell 33
    assert list(scores["Davies–Bouldin rank"]) == [1, 2] and list(scores["Calinski–Harabasz rank"]) == [1, 2]


def _charging_page(app):
    return _render(app, "charging_predict")


def _session(page) -> dict:
    return {f: page.session_state[f"cp_{f}"] for f in clu.input_fields()}


def _shown_group(page) -> int:
    card = next(m.value for m in page.markdown if "ciq-result-label" in m.value)
    return int(re.search(r'ciq-result-label">Group (\d+)<', card).group(1))


@pytest.mark.parametrize("group", [0, 1, 2])
def test_each_groups_mean_session_is_placed_in_that_group(app, group):
    page = _charging_page(app)
    page.button_group(key="cp_preset").set_value(group).run()
    assert not page.exception
    assert _session(page) == clu.profile(group)
    assert _shown_group(page) == group == clu.assign(_session(page)).cluster


def test_moving_a_slider_reassigns_the_session_and_clears_the_preset(app):
    page = _charging_page(app)
    page.slider(key="cp_connectionTime_decimal").set_value(3.0).run()
    assert page.session_state["cp_preset"] is None
    assert _shown_group(page) == clu.assign(_session(page)).cluster


def _chart_frames(page) -> list:
    """The data behind every Vega-Lite chart on the page (AppTest exposes charts only as raw protos)."""
    def walk(node):
        for child in getattr(node, "children", {}).values():
            yield child
            yield from walk(child)

    frames = []
    for element in walk(page.main):
        for dataset in getattr(getattr(element, "proto", None), "datasets", []):
            frames.append(pa.ipc.open_stream(dataset.data.data).read_pandas())
    return frames


def test_the_map_marks_the_session_exactly_where_pca_point_puts_it(app):
    page = _charging_page(app)
    page.slider(key="cp_kWhDelivered").set_value(30.0).run()
    session = _session(page)
    for frame in _chart_frames(page):
        here = frame[frame.astype(str).eq("This session").any(axis=1)]
        if len(here):
            numbers = here.select_dtypes("number").iloc[0].tolist()
            assert all(any(abs(n - v) < 1e-9 for n in numbers) for v in clu.pca_point(session))
            return
    pytest.fail("no chart marks 'This session'")
