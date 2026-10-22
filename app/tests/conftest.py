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
from core import clustering as clu  # noqa: E402
from core import regression as reg  # noqa: E402
from core.artifacts import load_metadata, load_model  # noqa: E402


@pytest.fixture(scope="session")
def meta():
    return load_metadata(clf.TRACK)


@pytest.fixture(scope="session")
def model():
    return load_model(clf.TRACK)


def _import_export_script(name: str):
    """An export script, imported as a module so tests can reuse its replay functions."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "models" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def export_module():
    return _import_export_script("export_classification")


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


# ---------------------------------------------------------------- regression


@pytest.fixture(scope="session")
def reg_meta():
    return load_metadata(reg.TRACK)


@pytest.fixture(scope="session")
def reg_model():
    return load_model(reg.TRACK)


@pytest.fixture(scope="session")
def reg_export_module():
    return _import_export_script("export_regression")


@pytest.fixture(scope="session")
def reg_replay(reg_export_module):
    """One replay of the regression notebook. Skipped where the raw NASA data is absent (it is not in git)."""
    if not reg_export_module.DATA_PATH.is_file():
        pytest.skip(f"regression data not found in {reg_export_module.DATASET_DIR} (local only)")
    return reg_export_module.replay()


# ---------------------------------------------------------------- clustering


@pytest.fixture(scope="session")
def clu_meta():
    return load_metadata(clu.TRACK)


@pytest.fixture(scope="session")
def clu_model():
    return load_model(clu.TRACK)


@pytest.fixture(scope="session")
def clu_export_module():
    return _import_export_script("export_clustering")


@pytest.fixture(scope="session")
def clu_replay(clu_export_module):
    """One replay of the clustering notebook. Its sample is committed, so this always runs."""
    return clu_export_module.replay()
