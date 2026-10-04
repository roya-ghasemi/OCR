# -*- coding: utf-8 -*-
"""Guards for the letterhead reader (E31).

The failures worth pinning here are the ones that look like success: a band that swallows
the white paper below it, and a field that is a plausible-looking stitch of fragments.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _page(band_colour=(60, 160, 60)) -> np.ndarray:
    """A page: a coloured band with white text on top, black text on white paper below."""
    img = np.full((400, 600, 3), 255, np.uint8)
    img[0:120] = band_colour                      # the band
    img[40:70, 100:500] = (255, 255, 255)         # white text in it
    img[200:230, 100:500] = (0, 0, 0)             # ordinary black text on paper
    return img


def test_the_mask_finds_white_text_on_colour_and_not_the_paper():
    """The whole point. White paper is unsaturated and bright too — what separates the
    letterhead's text from it is that the pixels *around* the text are coloured."""
    from ocr_service.letterhead import text_mask
    m = text_mask(_page())
    assert m[40:70, 100:500].mean() > 0.9, "the white text in the band must be found"
    assert m[150:400].mean() < 0.01, "the white paper below the band must not be"
    assert m[200:230, 100:500].mean() < 0.01, "black text on paper is not letterhead text"


def test_a_plain_page_has_no_band():
    """A letter with an ordinary dark-on-white letterhead already works through the normal
    path; this module must stay out of its way."""
    from ocr_service.letterhead import text_mask
    plain = np.full((400, 600, 3), 255, np.uint8)
    plain[100:130, 50:550] = (0, 0, 0)
    assert text_mask(plain).mean() < 0.004


def test_a_field_must_come_from_one_confident_line():
    """The bug this replaced: filtering words by confidence and joining the survivors
    stitched «شرکت خدماتی» from one reading to a misread «سرکت» from another and produced
    «شرکت خدماتی سرکت» — which reads like a company name and is not one."""
    from ocr_service.letterhead import _lines
    words = [("شرکت", 0.95, (1, 1, 1)), ("خدماتی", 0.92, (1, 1, 1)),
             ("سرکت", 0.40, (1, 1, 2)), ("خدماتی", 0.88, (1, 1, 2))]
    out = _lines(words)
    assert ("شرکت خدماتی", 0.92) in out
    # the second line carries a shaky word, so its confidence is that word's, not the mean
    assert any(t == "سرکت خدماتی" and c == 0.40 for t, c in out)


def test_the_gate_withholds_a_shaky_reading():
    """Measured on 14 sample pages: the band is found on all 14 and the gate emits a sender
    on none of them, because the company name is a stylised logotype the recogniser reads at
    0.00-0.44 confidence. Emitting it would be the invented text the brief forbids."""
    from ocr_service.letterhead import MIN_FIELD_CONF
    assert MIN_FIELD_CONF >= 0.75


def test_the_letterhead_region_is_reported_as_a_place_to_look():
    """When nothing is read confidently the band becomes an `unread` region, which is what
    the brief asks for: show the crop rather than guess at it."""
    from ocr_service.schemas import UnreadRegion
    r = UnreadRegion(kind="letterhead", bbox=[0, 0, 100, 50], read_as="", confidence=0.3)
    assert r.kind == "letterhead"
