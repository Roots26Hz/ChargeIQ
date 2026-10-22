"""Deployment readiness: what a visitor to the public app sees, and what Streamlit Community Cloud needs.

These run with the rest of the suite. If one fails, the app is not ready to deploy.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys

import pytest
from streamlit.testing.v1 import AppTest

from conftest import APP_DIR, REPO_ROOT
from core import artifacts
from ui import navigation

ENTRYPOINT = APP_DIR / "streamlit_app.py"
LOCAL_PACKAGES = {"core", "ui", "content"}
# import name -> distribution name in app/requirements.txt
DISTRIBUTIONS = {"streamlit": "streamlit", "sklearn": "scikit-learn", "numpy": "numpy",
                 "pandas": "pandas", "scipy": "scipy", "joblib": "joblib"}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout


def _app_sources() -> list:
    """Every Python file the deployed app runs: everything in app/ except the tests."""
    return [p for p in APP_DIR.rglob("*.py") if "tests" not in p.relative_to(APP_DIR).parts]


def _requirements() -> dict[str, str]:
    return dict(
        re.match(r"^([A-Za-z0-9_.-]+)==(\S+)$", line).groups()
        for line in (APP_DIR / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )


def _walk(node):
    for child in getattr(node, "children", {}).values():
        yield child
        yield from _walk(child)


# ---------------------------------------------------------------- what visitors see


def test_unfinished_chapters_are_hidden_from_visitors():
    assert navigation.SHOW_UNFINISHED is False, "set SHOW_UNFINISHED = False in app/ui/navigation.py"
    assert all(ch.ready for ch in navigation.visible())


@pytest.mark.parametrize("chapter", navigation.visible(), ids=lambda ch: ch.key)
def test_visible_chapters_are_complete_and_link_only_to_visible_chapters(chapter):
    at = AppTest.from_file(str(ENTRYPOINT), default_timeout=60)
    at.run()
    if chapter.key != navigation.DEFAULT_KEY:
        at.switch_page(f"chapters/{chapter.key}.py").run()
    assert not at.exception
    elements = list(_walk(at.main))
    markdown = "\n".join(e.value for e in elements if type(e).__name__ == "Markdown")
    assert "ciq-todo" not in markdown, "a TODO(team) box is still on a visible chapter"
    assert "ciq-placeholder" not in markdown, "a 'coming soon' placeholder is on a visible chapter"
    visible_urls = {ch.url_path for ch in navigation.visible()}
    for link in (e for e in elements if type(e).__name__ == "PageLink"):
        assert link.page in visible_urls, f"{link.label!r} points to hidden chapter {link.page!r}"


def test_a_missing_model_shows_an_explanation_not_a_traceback(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts, "MODELS_DIR", tmp_path)  # an empty models/ directory
    artifacts.load_metadata.cache_clear()
    artifacts.load_model.cache_clear()
    try:
        at = AppTest.from_file(str(ENTRYPOINT), default_timeout=60)
        at.run()
        at.switch_page("chapters/vehicle_model.py").run()
        assert not at.exception
        assert any("can't load its model" in e.value for e in at.error)
        assert any("Missing metadata file" in c.value for c in at.code)
    finally:
        artifacts.load_metadata.cache_clear()  # later tests load the real files again
        artifacts.load_model.cache_clear()


# ---------------------------------------------------------------- what Community Cloud needs


def test_every_third_party_import_is_pinned_in_the_app_requirements():
    pins = _requirements()
    for path in _app_sources():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top in sys.stdlib_module_names or top in LOCAL_PACKAGES:
                    continue
                assert top in DISTRIBUTIONS, f"{path.name} imports {top!r}, which is not in app/requirements.txt"
                assert DISTRIBUTIONS[top] in pins, f"{DISTRIBUTIONS[top]} is not pinned"


def test_the_app_never_reads_a_dataset():
    """The datasets are large or local-only; everything the app shows comes from models/ and assets/."""
    for path in _app_sources():
        source = path.read_text(encoding="utf-8")
        for pattern in (".csv", "read_csv", "read_parquet", '/ "data"', '"data/'):  # citing data/README.md is fine
            assert pattern not in source, f"{path.relative_to(REPO_ROOT)} mentions {pattern!r}"


def test_the_deployed_model_files_are_committed_and_unmodified():
    tracked = set(_git("ls-files", "models").split())
    for track in ("classification",):
        for path in (artifacts.model_path(track), artifacts.metadata_path(track)):
            relative = str(path.relative_to(REPO_ROOT))
            assert relative in tracked, f"{relative} is not committed"
            assert _git("status", "--porcelain", "--", relative) == "", f"{relative} differs from the commit"


def test_no_secrets_are_committed():
    tracked = _git("ls-files").split()
    for name in tracked:
        assert not name.endswith(("secrets.toml", ".env", ".pem", ".key")), name


def test_the_deployment_guide_matches_the_repository():
    guide = (APP_DIR / "DEPLOYMENT.md").read_text(encoding="utf-8")
    entrypoint = str(ENTRYPOINT.relative_to(REPO_ROOT))
    assert f"`{entrypoint}`" in guide
    assert "3.13" in guide
    assert "public" in guide.lower()
    python = re.search(r"Python (\d+\.\d+)", (APP_DIR / "requirements.txt").read_text()).group(1)
    assert python == "3.13", "requirements.txt and the guide disagree on the Python version"
