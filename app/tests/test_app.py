"""Smoke tests: the Streamlit app starts, and every chapter renders without error."""

from __future__ import annotations

import re

import pytest
from streamlit.testing.v1 import AppTest

from conftest import APP_DIR, REPO_ROOT
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
