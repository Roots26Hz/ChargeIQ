"""Load the exported model artifacts from models/.

Each track ships two files written by its export script:

    models/<track>_model.joblib     the fitted scikit-learn object
    models/<track>_metadata.json    schema, input domains and provenance

Both are loaded once per process and cached, so Streamlit reruns never reload
them. Paths are resolved relative to this file, never to the working directory,
so the app behaves the same locally and on a hosting platform.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import sklearn

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"


class ArtifactError(RuntimeError):
    """A model artifact is missing, unreadable, or incompatible with this environment."""


def model_path(track: str) -> Path:
    return MODELS_DIR / f"{track}_model.joblib"


def metadata_path(track: str) -> Path:
    return MODELS_DIR / f"{track}_metadata.json"


def _export_hint(track: str) -> str:
    return f"Re-create it from the repository root with:  python models/export_{track}.py"


def _shown(path: Path) -> str:
    """Repo-relative path for messages; absolute if the file lives outside the repo."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@lru_cache(maxsize=None)
def load_metadata(track: str) -> dict[str, Any]:
    """Return the track's metadata. Treat the result as read-only — it is shared."""
    path = metadata_path(track)
    if not path.is_file():
        raise ArtifactError(f"Missing metadata file {_shown(path)}. {_export_hint(track)}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{_shown(path)} is not valid JSON ({exc}). {_export_hint(track)}") from exc


@lru_cache(maxsize=None)
def load_model(track: str) -> Any:
    """Return the track's fitted model, refusing to load under a different scikit-learn.

    A pickled estimator is only guaranteed to behave identically under the
    scikit-learn version that wrote it, so a mismatch is an error, not a warning.
    """
    path = model_path(track)
    if not path.is_file():
        raise ArtifactError(f"Missing model file {_shown(path)}. {_export_hint(track)}")

    recorded = load_metadata(track).get("environment", {}).get("scikit-learn")
    if recorded != sklearn.__version__:
        raise ArtifactError(
            f"The {track} model was exported with scikit-learn {recorded}, but this environment "
            f"has {sklearn.__version__}. Install scikit-learn=={recorded} (see app/requirements.txt)."
        )
    try:
        return joblib.load(path)
    except Exception as exc:  # unpickling can fail in many library-specific ways
        raise ArtifactError(f"Could not load {_shown(path)}: {exc}. {_export_hint(track)}") from exc
