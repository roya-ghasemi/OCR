# -*- coding: utf-8 -*-
"""Tesseract runner: TSV parsing, batched reads, and the digit-reader confidence path."""
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

from ocr_service.config import Settings
from ocr_service.tesseract import Tesseract, parse_tsv
from ocr_service.transcribe import Line, Transcriber, Word

HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"


def test_parse_tsv_keeps_words_and_splits_pages():
    tsv = "\n".join([
        HEADER,
        "1\t1\t0\t0\t0\t0\t0\t0\t100\t40\t-1\t",
        "5\t1\t1\t1\t1\t1\t10\t5\t30\t20\t91.5\tسلام",
        "5\t1\t1\t1\t1\t2\t50\t5\t30\t20\t-1\t",          # no confidence: not a word
        "5\t3\t1\t1\t1\t1\t2\t3\t4\t5\t40\tabc",
    ])
    pages = parse_tsv(tsv)
    assert set(pages) == {1, 3}
    assert pages[1] == [{"text": "سلام", "conf": 91.5, "box": (10, 5, 40, 25), "key": ("1", "1", "1")}]
    assert pages[3][0]["box"] == (2, 3, 6, 8)


def test_parse_tsv_of_nothing_is_empty():
    assert parse_tsv("") == {}


def _tess() -> Tesseract:
    cmd, data = Settings().resolved_tesseract()
    t = Tesseract(cmd, data)
    if not cmd or "eng" not in t.probe()[1]:
        pytest.skip("Tesseract with English data not installed")
    return t


def _line(text: str) -> Image.Image:
    im = Image.new("L", (360, 60), 255)
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(im).text((10, 10), text, fill=0, font=font)
    return im


def test_a_batch_reads_each_image_as_reading_it_alone():
    t = _tess()
    imgs = [_line("Invoice 2024"), Image.new("L", (200, 60), 255), _line("Total amount")]
    alone = [t.read(im, "7", "eng") for im in imgs]
    batch = t.read_many(imgs, "7", "eng")
    assert len(batch) == 3 and batch == alone


def test_a_failing_tesseract_reads_nothing_instead_of_raising():
    t = _tess()
    assert t.read(_line("x"), "7", "no_such_language") == []


def test_a_number_from_the_digit_reader_reports_its_own_probability():
    tr = Transcriber(None, None, None, spellfix=False)
    w = Word("۱۲۵,۰۰۰", 100.0, (0, 0, 50, 20), digit_prob=0.83)
    _, numbers, _ = tr._assemble([Line([Word("مبلغ", 90.0, (60, 0, 100, 20)), w], (0, 0, 100, 20), "psm4")], 1.0)
    assert len(numbers) == 1
    assert numbers[0].source == "glyph" and numbers[0].confidence == 0.83


def _row(*words):
    ws = [Word(t, c, (x0, 0, x1, 20)) for t, c, x0, x1 in words]
    return [[Line(ws, (min(w.box[0] for w in ws), 0, max(w.box[2] for w in ws), 20), "psm4")]]


def test_latin_inside_a_persian_line_is_kept_as_printed():
    rows = _row(("حروف", 90.0, 200, 260), ("2-", 0.0, 100, 140), ("کوچک", 92.0, 0, 80))
    Transcriber._keep_latin(rows, [Word("a-z", 93.0, (98, 0, 142, 20))])
    assert [w.text for w in rows[0][0].words] == ["حروف", "a-z", "کوچک"]


def test_a_persian_word_is_never_replaced():
    rows = _row(("۱۲مریم", 40.0, 100, 160))
    Transcriber._keep_latin(rows, [Word("VY", 85.0, (100, 0, 160, 20))])
    assert rows[0][0].words[0].text == "۱۲مریم"


def test_persian_digits_read_as_letters_stay_persian():
    rows = _row(("۸", 20.0, 100, 112))
    Transcriber._keep_latin(rows, [Word("A", 90.0, (100, 0, 112, 20))])          # one letter: a digit shape
    assert rows[0][0].words[0].text == "۸"
    rows = _row(("۰/.۲۳", 20.0, 100, 160))
    Transcriber._keep_latin(rows, [Word("INY", 90.0, (100, 0, 160, 20))], [(100, 0, 160, 20)])
    assert rows[0][0].words[0].text == "۰/.۲۳"                                  # the digit reader saw a number
