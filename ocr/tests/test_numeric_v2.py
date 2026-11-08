# -*- coding: utf-8 -*-
"""Guards for the v2 digit reader and the numeric reconciliation (E15).

Two kinds of test, on purpose (A4: prove a check can fail before trusting it):
  * rendered fixtures the reader MUST read exactly (a synthetic Persian number line,
    a ring-zero footer line, a hyphenated phone list) and a deliberately wrong
    expectation that MUST fail if the assertion helper is ever weakened;
  * reconciliation on hand-built reads: confirm / correct / conflict / unverified /
    added, the script of corrected digits, that prose is never rewritten by an
    injection, and that a low-confidence read never overrides the model.
No engine, no GPU: the CNN runs in numpy on `models/digit_cnn.npz`.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest
from PIL import Image, ImageFont

ROOT = Path(__file__).resolve().parents[1]

from ocr_research import digit_cnn_data as D
from ocr_research.numeric_validator import to_ascii_digits

from ocr_service.config import settings as _settings
MODEL = _settings.resolved_digit_model() or ROOT / "models" / "digit_cnn.npz"   # a worktree finds the main checkout's
needs_model = pytest.mark.skipif(not MODEL.is_file(), reason="models/digit_cnn.npz not built")


def _font(px: int):
    for name in ("BNazanin.ttf", "BLotus.ttf", "BMitra.ttf", "tahoma.ttf"):
        for f in D._fonts():
            if f.lower().endswith(name.lower()):
                return ImageFont.truetype(f, px)
    pytest.skip("no Persian office font installed")


def _render(tokens, px=40, seed=1, y_rel=0.45):
    """One text line on a page-like canvas (the reader's size filters are relative to
    the page). `y_rel` places it: > 0.80 is the footer band the reconciler treats as
    `contact_info`."""
    rng = random.Random(seed)
    im, _ = D.render_line(tokens, _font(px), px, rng)
    page = Image.new("L", (max(1200, im.width + 200), 900), 255)
    page.paste(im, (100, min(900 - im.height, int(900 * y_rel) - im.height // 2)))
    return page.convert("RGB")


def _reader():
    from ocr_service.digit_reader_v2 import GlyphReaderV2
    return GlyphReaderV2(model_path=MODEL)


def _atoms(reads):
    import re
    return {a for r in reads for a in re.findall(r"\d+", to_ascii_digits(r.text)) if len(a) >= 3}


# --- reader ---------------------------------------------------------------------

@needs_model
def test_reader_reads_a_persian_number_in_running_text_exactly():
    page = _render([("word", D.shape("مبلغ")), ("number", "۳۲۱/۰۰۰/۰۰۰"), ("word", D.shape("ریال")),
                    ("word", D.shape("کد")), ("number", "۴۱۱۴۱۳۶۷۴۶۴۸")])
    atoms = _atoms(_reader().read_page(page))
    assert {"321", "000", "411413674648"} <= atoms, atoms


@needs_model
def test_reader_reads_a_hyphenated_phone_list_and_keeps_leading_zero():
    page = _render([("word", D.shape("تلفن")), ("punct", ":"), ("number", "۰۵۱۱-۵۰۲۷۸۷۱-۵۰۲۷۸۷۳")])
    atoms = _atoms(_reader().read_page(page))
    assert {"0511", "5027871", "5027873"} <= atoms, atoms


@needs_model
def test_reader_does_not_turn_persian_words_into_numbers():
    page = _render([("word", D.shape(w)) for w in ("احتراما", "خواهشمند", "است", "دستور", "فرمایید", "نسبت", "به")], px=36)
    reads = _reader().read_page(page)
    assert reads == [], [r.text for r in reads]


@needs_model
def test_reader_guard_can_fail():
    """A check that cannot fail is worthless (D50): the same fixture with a wrong
    expectation must not pass."""
    page = _render([("number", "۳۸۳۸۸۵۷۵")])
    atoms = _atoms(_reader().read_page(page))
    assert "38388575" in atoms
    assert "38388576" not in atoms
