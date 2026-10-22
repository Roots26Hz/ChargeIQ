"""Words and numbers the app takes from the team's notebooks, each with its source cell.

The course guidelines allow AI for code scaffolding but not for analysis or
interpretation, so every interpretive statement in the app is a verbatim quote
from the team's own notebook markdown. app/tests/test_content.py checks that
each quote really appears in the cell it cites, and that each fact appears in
its cell's source or stored output — so nothing can be paraphrased or drift.
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
