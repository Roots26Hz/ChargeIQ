"""Every quote the app shows must appear verbatim in the notebook cell it cites.

This is what keeps the app's interpretive copy in the team's own words: a
paraphrase, a typo fix, or a quote attributed to the wrong cell fails here.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

import pytest

from conftest import REPO_ROOT
from content import Fact, Quote, home, vehicle

MODULES = (home, vehicle)
ALL_QUOTES = [q for m in MODULES for q in m.QUOTES]
ALL_FACTS = [f for m in MODULES for f in m.FACTS]


@lru_cache(maxsize=None)
def _cells(notebook: str) -> list[dict]:
    return json.loads((REPO_ROOT / "notebooks" / notebook).read_text(encoding="utf-8"))["cells"]


def _normalise(text: str) -> str:
    """Reflow whitespace and drop the '¶' anchors that HTML exports leave behind — nothing else."""
    return re.sub(r"\s+", " ", text.replace("¶", "")).strip()


def _outputs(cell: dict) -> str:
    parts = []
    for out in cell.get("outputs", []):
        parts.append("".join(out.get("text") or (out.get("data") or {}).get("text/plain", "")))
    return "\n".join(parts)


@pytest.mark.parametrize("quote", ALL_QUOTES, ids=lambda q: f"{q.notebook}#{q.cell}:{q.text[:30]}")
def test_quote_appears_verbatim_in_its_markdown_cell(quote: Quote):
    cell = _cells(quote.notebook)[quote.cell]
    assert cell["cell_type"] == "markdown", "quotes must come from the team's markdown, not code"
    assert _normalise(quote.text) in _normalise("".join(cell["source"]))


@pytest.mark.parametrize("fact", ALL_FACTS, ids=lambda f: f"{f.notebook}#{f.cell}:{f.value[:30]}")
def test_fact_is_backed_by_its_cell(fact: Fact):
    cell = _cells(fact.notebook)[fact.cell]
    assert fact.evidence in "".join(cell["source"]) + _outputs(cell)


def test_training_pool_fact_matches_its_evidence():
    shown = int(vehicle.TRAINING_POOL_ROWS.value.replace(",", ""))
    assert re.search(rf"\({shown}, 13\)", vehicle.TRAINING_POOL_ROWS.evidence)


def test_every_module_lists_its_quotes():
    for module in MODULES:
        declared = {v for v in vars(module).values() if isinstance(v, Quote)}
        assert declared == set(module.QUOTES), module.__name__


def test_model_training_rows_fact_matches_the_exported_model(meta):
    assert int(vehicle.MODEL_TRAINING_ROWS.value.replace(",", "")) == meta["dataset"]["train_rows"]
