"""Words and numbers the app takes from the team's notebooks, each with its source cell,
and the example inputs it offers, each checked against the data.

The course guidelines allow AI for code scaffolding but not for analysis or
interpretation, so every interpretive statement in the app is a verbatim quote
from the team's own notebook markdown. app/tests/test_content.py checks that
each quote really appears in the cell it cites, and that each fact appears in
its cell's source or stored output — so nothing can be paraphrased or drift. It also
replays the notebook's data split and checks every Example against it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Quote:
    notebook: str   # file name in notebooks/
    cell: int       # zero-based cell index
    text: str       # verbatim excerpt of a markdown cell; whitespace may be reflowed


@dataclass(frozen=True)
class Fact:
    notebook: str
    cell: int
    value: str      # how the app displays it
    evidence: str   # text that must appear in that cell's source or stored output


@dataclass(frozen=True)
class Example:
    """A ready-made input for a prediction chapter.

    It is a combination of values that occurs in the data, not an individual record:
    the test suite checks that `rows` rows in the named split share these values and
    are all labelled `label_in_data`, and that the model predicts `predicted`.
    """

    key: str
    title: str          # shown on the picker
    make: str
    model: str
    model_year: int
    electric_range: float
    split: str          # "training" or "test"
    rows: int           # rows in that split with this make, model, year and range
    label_in_data: str  # short label of those rows
    predicted: str      # short label the model predicts for this example
    county: str = "King"
    city: str = "Seattle"
