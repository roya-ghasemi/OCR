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
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ocr_service import digit_cnn_data as D  # noqa: E402
from ocr_service.numeric_reconcile import Policy, reconcile, summarize  # noqa: E402
from ocr_service.numeric_validator import to_ascii_digits  # noqa: E402

MODEL = ROOT / "models" / "digit_cnn.npz"
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
    return GlyphReaderV2()


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


# --- reconciliation ------------------------------------------------------------------

class R:                                     # a NumberRead stand-in
    def __init__(self, text, y, conf=0.8, mean=0.9, script="persian", x=(100, 600)):
        self.text, self.box, self.confidence, self.mean_prob, self.script = text, (x[0], y, x[1], y + 40), conf, mean, script


def test_exact_agreement_is_high_and_text_untouched():
    fields = {"body_text": "مبلغ ۳۲۱/۰۰۰/۰۰۰ ریال", "contact_info": None}
    out, recs = reconcile(fields, [R("۳۲۱/۰۰۰/۰۰۰", 500)], 1000)
    assert out["body_text"] == fields["body_text"]
    assert [r.confidence for r in recs] == ["high"]


def test_confident_read_corrects_the_models_digits_in_place_and_in_script():
    fields = {"body_text": "کد اقتصادی ۴۱۱۴۱۳۶۷۴۶۴۹ است", "contact_info": None}
    out, recs = reconcile(fields, [R("۴۱۱۴۱۳۶۷۴۶۴۸", 500)], 1000)
    assert out["body_text"] == "کد اقتصادی ۴۱۱۴۱۳۶۷۴۶۴۸ است"       # Persian digits kept, value fixed
    assert recs[0].confidence == "corrected" and recs[0].resolved == "۴۱۱۴۱۳۶۷۴۶۴۸"
    assert recs[0].value == "۴۱۱۴۱۳۶۷۴۶۴۹"                          # what the model wrote is preserved


def test_low_confidence_read_never_overrides():
    fields = {"body_text": "کد ۴۱۱۴۱۳۶۷۴۶۴۹", "contact_info": None}
    out, recs = reconcile(fields, [R("۴۱۱۴۱۳۶۷۴۶۴۸", 500, conf=0.1, mean=0.4)], 1000)
    assert out["body_text"] == fields["body_text"]
    assert recs[0].confidence == "low" and "۴۱۱۴۱۳۶۷۴۶۴۸" in recs[0].candidates


def test_short_ambiguous_atom_is_not_corrected_when_two_reads_tie():
    # 1402 vs 1403 on a page that carries both: never guess
    fields = {"body_text": "مورخ ۱۴۰۱", "contact_info": None}
    out, recs = reconcile(fields, [R("۱۴۰۲", 500), R("۱۴۰۳", 560)], 1000)
    assert out["body_text"] == fields["body_text"]
    assert recs[0].confidence == "low"


def test_multipart_number_is_aligned_as_a_unit():
    # 587 -> 588 fails the per-atom threshold alone (2 of 3 digits) but the whole
    # amount aligns, so each part is corrected by position
    fields = {"body_text": "مبلغ ۲/۵۸۷/۸۰۹/۵۸۶ ریال", "contact_info": None}
    out, recs = reconcile(fields, [R("۲/۵۸۸/۸۰۸/۵۸۸", 500)], 1000)
    assert out["body_text"] == "مبلغ ۲/۵۸۸/۸۰۸/۵۸۸ ریال"
    assert recs[0].confidence == "corrected"


def test_nothing_aligned_is_unverified_and_kept():
    fields = {"body_text": "شماره ۹۸۷۶۵۴۳۲۱", "contact_info": None}
    out, recs = reconcile(fields, [R("۱۲۳", 500)], 1000)
    assert out["body_text"] == fields["body_text"] and recs[0].confidence == "unverified"


def test_footer_number_the_model_omitted_is_added_to_contact_info_only():
    fields = {"body_text": "با سلام", "contact_info": "آدرس: مشهد"}
    reads = [R("۳۸۳۸۸۵۷۵-۳۸۳۸۸۴۸۱", 900), R("۱۴۰۳/۰۹/۲۰", 300)]        # footer + a body date
    out, recs = reconcile(fields, reads, 1000)
    added = [r for r in recs if r.confidence == "added"]
    assert {r.field for r in added if r.field == "contact_info"} == {"contact_info"}
    assert "۳۸۳۸۸۵۷۵" in out["contact_info"] and "۳۸۳۸۸۴۸۱" in out["contact_info"]
    assert out["body_text"] == "با سلام"                                # prose is never rewritten by an injection
    body_added = [r for r in added if r.field == "body_text"]
    # reported as a record — and as the WHOLE number, separators intact, not the bare
    # leading group (a date or an amount must never reach the caller in pieces)
    assert body_added and body_added[0].value_ascii == "1403/09/20"


def test_injection_can_be_disabled_and_needs_confidence():
    fields = {"body_text": None, "contact_info": "x"}
    out, recs = reconcile(fields, [R("۳۸۳۸۸۵۷۵", 900, conf=0.2, mean=0.5)], 1000)
    assert recs == [] and out["contact_info"] == "x"
    out, recs = reconcile(fields, [R("۳۸۳۸۸۵۷۵", 900)], 1000, Policy(inject=False, inject_body=False))
    assert recs == []


def test_summary_counts_every_confidence_class():
    fields = {"body_text": "الف ۱۱۱۱۱۱۱ ب ۲۲۲۲۲۲۲", "contact_info": None}
    _, recs = reconcile(fields, [R("۱۱۱۱۱۱۱", 500), R("۲۲۲۲۲۲۳", 560), R("۳۸۳۸۸۵۷۵", 950)], 1000)
    s = summarize(recs)
    assert s["n_numeric"] == 3 and s["high"] == 1 and s["corrected"] == 1 and s["added"] == 1


# --- end-to-end wiring (no engine, no GPU) -------------------------------------

@needs_model
def test_pipeline_glyph2_path_corrects_fields_and_reports_new_confidences(tmp_path):
    """The whole `OcrPipeline.run` path with a stub backend: the page is read, the
    model's digits are corrected in the returned `fields`, and the response carries
    the `corrected`/`added` classes. Guards the wiring, not the reader."""
    import asyncio
    import io

    from ocr_service.backends import Extraction
    from ocr_service.config import PRIMARY_DEFAULT, SECONDARY_DEFAULT, Settings
    from ocr_service.pipeline import OcrPipeline

    page = _render([("word", D.shape("تلفن")), ("punct", ":"), ("number", "۰۵۱۱-۵۰۲۷۸۷۱-۵۰۲۷۸۷۳")],
                   px=44, y_rel=0.90)                      # the footer band
    buf = io.BytesIO(); page.save(buf, "JPEG", quality=92)

    class StubBackend:
        name = "stub"

        async def start(self): pass

        async def stop(self): pass

        async def health(self): return {"model": "stub", "ok": True}

        async def extract(self, data, mime, doc_id):
            # the model read one phone number with two digits wrong
            return Extraction(ok=True, model="stub", latency_s=0.0, attempts=1, grammar_used=True,
                              fields={"sender": None, "receiver": None, "subject": None,
                                      "body_text": None, "contact_info": "تلفن: ۰۵۱۱-۵۰۲۷۸۷۳"})

    cfg = Settings(primary=PRIMARY_DEFAULT, secondary=SECONDARY_DEFAULT, preprocess_incoming=False)
    p = OcrPipeline(cfg)
    if p.reconciler is None:
        pytest.skip("glyph2 reader unavailable")
    p.primary = StubBackend()
    res = asyncio.run(p.run(buf.getvalue(), "image/jpeg", "t1"))

    assert res.service["numeric_reader"] == "glyph-cnn-v2"
    classes = {n.confidence for n in res.numeric_fields}
    assert classes & {"high", "corrected", "added"}, [n.model_dump() for n in res.numeric_fields]
    # the number the model omitted entirely is reported and appended to contact_info
    assert "۵۰۲۷۸۷۱" in (res.fields.contact_info or "")


# --- D61: a declared setting must reach the wire ---------------------------------

def test_cache_prompt_is_actually_sent_to_the_engine():
    """Regression guard for D61: `Settings.cache_prompt` was stamped into provenance
    for weeks while llama.cpp ran with its own default (true), because nothing put it
    in the request. Provenance that reports a value the request does not carry is the
    D48 / false-green class of defect, so assert on the REQUEST, not the config."""
    import asyncio

    from ocr_pipeline.extraction import LetterExtractor

    seen = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop after capturing the request")

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    for flag in (False, True):
        seen.clear()
        ex = LetterExtractor(FakeClient(), "m", extra_body={"cache_prompt": flag})
        asyncio.run(ex.extract("data:image/jpeg;base64,AA==", doc_id="t"))
        assert seen.get("extra_body", {}).get("cache_prompt") is flag, seen.get("extra_body")
        assert "grammar" in seen["extra_body"]          # and it did not displace the GBNF grammar


def test_unverified_numbers_flag_the_document_for_review():
    """E15: flagging only `low` left 1 of 51 dev documents flagged while 34 carried a
    wrong number. `unverified` is the least reliable class measured, so it must flag."""
    import asyncio
    import io

    from ocr_service.backends import Extraction
    from ocr_service.config import PRIMARY_DEFAULT, SECONDARY_DEFAULT, Settings
    from ocr_service.pipeline import OcrPipeline

    page = _render([("word", D.shape("مبلغ")), ("number", "۳۲۱/۰۰۰/۰۰۰")], px=44)
    buf = io.BytesIO(); page.save(buf, "JPEG", quality=92)

    class Stub:
        name = "stub"

        async def start(self): pass

        async def stop(self): pass

        async def health(self): return {}

        async def extract(self, data, mime, doc_id):
            # a number that is nowhere on the page -> unverified
            return Extraction(ok=True, model="stub", latency_s=0.0, attempts=1,
                              fields={"sender": None, "receiver": None, "subject": None,
                                      "body_text": "شماره ۹۸۷۶۵۴۳۲۱۰", "contact_info": None})

    p = OcrPipeline(Settings(primary=PRIMARY_DEFAULT, secondary=SECONDARY_DEFAULT, preprocess_incoming=False))
    if p.reconciler is None:
        pytest.skip("glyph2 reader unavailable")
    p.primary = Stub()
    res = asyncio.run(p.run(buf.getvalue(), "image/jpeg", "t2"))
    assert any(n.confidence == "unverified" for n in res.numeric_fields)
    assert res.needs_review is True


# --- grouped amounts must survive intact ------------------------------------------

def test_grouped_amount_is_injected_whole_not_chunk_by_chunk():
    """The groups of a large sum are 3 digits each. Judged separately they fall under
    the body floor and the whole amount disappears; judged as one number it survives
    with its separators exactly as printed."""
    fields = {"body_text": "مبلغ قرارداد به شرح زیر است", "contact_info": None}
    out, recs = reconcile(fields, [R("۳۲۱/۰۰۰/۰۰۰", 500)], 1000)
    added = [r for r in recs if r.confidence == "added"]
    assert len(added) == 1
    assert added[0].value == "۳۲۱/۰۰۰/۰۰۰" and added[0].value_ascii == "321/000/000"
    assert out["body_text"] == fields["body_text"]          # prose untouched


def test_trailing_zeros_of_an_amount_are_never_dropped():
    """`۳۲۱/۰۰۰/۰۰۰` read as `۳۲۱/۰۰۰/۰۰` is an amount divided by ten. The reader's
    run extension exists for this; this guards the reconciliation half."""
    fields = {"body_text": "مبلغ ۳۲۱/۰۰۰/۰۰ ریال", "contact_info": None}
    out, recs = reconcile(fields, [R("۳۲۱/۰۰۰/۰۰۰", 500)], 1000)
    assert "۳۲۱/۰۰۰/۰۰۰" in out["body_text"]
    assert recs[0].resolved == "۳۲۱/۰۰۰/۰۰۰"


def test_group_order_may_be_flipped_by_the_page_and_the_models_order_wins():
    """A multi-group number is laid out right-to-left on the page, so the reader sees
    `۰۳/۰۸/۱۴۰۳` for the date `۱۴۰۳/۰۸/۰۳`. The digits come from the reader, the
    ordering from the model — the caller must not receive a reversed date."""
    fields = {"body_text": "مورخ ۱۴۰۳/۰۸/۰۴", "contact_info": None}
    out, recs = reconcile(fields, [R("۰۳/۰۸/۱۴۰۳", 500)], 1000)
    assert recs[0].resolved == "۱۴۰۳/۰۸/۰۳", recs[0].resolved


def test_a_grouped_number_the_model_already_has_is_not_added_twice():
    fields = {"body_text": None, "contact_info": "تلفن ۳۸۳۸۸۵۷۵-۳۸۳۸۸۴۸۱"}
    out, recs = reconcile(fields, [R("۳۸۳۸۸۵۷۵-۳۸۳۸۸۴۸۱", 900)], 1000)
    assert sum(r.confidence == "added" for r in recs) == 0
    assert out["contact_info"].count("۳۸۳۸۸۵۷۵") == 1
