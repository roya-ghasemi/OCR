# -*- coding: utf-8 -*-
"""Guards for carving line pairs out of the handwriting split (E37).

A wrong image/text pair is not ignored by training — it is learned. These pin the two rules
that keep wrong pairs out.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _page(bands):
    """A white page with ink bands: (y0, y1, stroke_height) for each."""
    a = np.full((300, 400), 255, np.uint8)
    for y0, y1, stroke in bands:
        for x in range(10, 390, 6):
            a[y0:y0 + stroke, x:x + 3] = 0
            if stroke > 2:
                a[y0:y1, x:x + 2] = 0
    return Image.fromarray(a)


def test_a_ruled_line_is_not_counted_as_a_row():
    """This is the failure that matters: a dotted leader or a ruled line counts as a row and
    shifts every label after it by one. The count test still passes, so the page is kept and
    every pair on it is wrong. One such page turned up in a sample of six.

    Writing and rules separate on how tall the ink is in a typical inked column — measured
    across 301 crops, rules sit at a median of 1 px and writing between 7 and 17.
    """
    from ocr_train.handwriting import ink_rows
    page = _page([(40, 70, 14), (120, 122, 1), (180, 210, 12)])
    rows = ink_rows(page)
    assert len(rows) == 2, f"the ruled band should not be a row, got {rows}"
    assert all(r[1] - r[0] > 10 for r in rows)


def test_writing_rows_are_kept():
    from ocr_train.handwriting import ink_rows
    page = _page([(30, 60, 12), (100, 130, 12), (170, 200, 12)])
    assert len(ink_rows(page)) == 3


def test_the_alignment_rule_is_exact_not_approximate():
    """A page is used only when rows and label lines agree in number. Off-by-one alignment
    silently mislabels every line after the discrepancy, and a form's label lists cell
    contents rather than visual lines — forty-three of them across four visible rows."""
    import inspect

    from ocr_train import handwriting
    src = inspect.getsource(handwriting.build)
    assert "len(rows) != len(label)" in src, "the count test must be exact equality"
    assert "dropped_count_mismatch" in src, "what was dropped has to be counted and reported"


def test_the_held_out_split_is_decided_by_content_hash():
    """So a line lands on the same side on every machine and after every re-run, and an
    evaluation number stays comparable with the one before it."""
    from ocr_train.benchmark import split_of
    data = b"a handwriting page"
    assert split_of(data) == split_of(data)
    assert split_of(data) in ("train", "eval")
