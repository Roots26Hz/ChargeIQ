"""Shared fixtures. Run from the repository root:  python -m pytest app/tests"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))  # so `import core` works exactly as it does under Streamlit

from core import classification as clf  # noqa: E402
from core.artifacts import load_metadata, load_model  # noqa: E402


@pytest.fixture(scope="session")
def meta():
    return load_metadata(clf.TRACK)


@pytest.fixture(scope="session")
def model():
    return load_model(clf.TRACK)


@pytest.fixture(scope="session")
def export_module():
    """The export script, imported as a module so tests can reuse its replay functions."""
    path = REPO_ROOT / "models" / "export_classification.py"
    spec = importlib.util.spec_from_file_location("export_classification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def replay(export_module):
    """One replay of the notebook's training data, shared by every test that needs it."""
    return export_module.replay()


def vehicle(make: str, model: str, year: int, electric_range: float,
            county: str = "King", city: str = "Seattle") -> dict:
    """A complete, valid input with location codes filled from the training defaults."""
    return {
        "Make": make,
        "Model": model,
        "Model Year": year,
        "Electric Range": electric_range,
        "County": county,
        "City": city,
        **clf.location_defaults(county, city),
    }


@pytest.fixture
def bev_inputs():
    return vehicle("TESLA", "MODEL 3", 2020, 266)


@pytest.fixture
def phev_inputs():
    return vehicle("CHEVROLET", "VOLT", 2017, 53)
