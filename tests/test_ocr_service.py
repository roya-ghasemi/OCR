# -*- coding: utf-8 -*-
"""Guards for ocr_service — no engine, no Tesseract binary required."""
from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ocr_service.numeric_validator import (  # noqa: E402
    NumericValidator, TesseractReader, Word, classify, digits_only, find_numeric_spans, summarize, to_ascii_digits)
from ocr_service.preprocess import compress, estimate_skew, prepare, rotate, _to_cv  # noqa: E402
from ocr_service.schemas import OcrResponse  # noqa: E402


# --- span finding / classification -------------------------------------------

def test_spans_cover_all_three_digit_scripts_and_mixed_lines():
    fields = {"body_text": "مبلغ ۳۲۱/۰۰۰/۰۰۰ ریال و کد ملی ۰۹۲۴۴۲۴۱۱۷ و شماره 1403/4/۵۲۴ و ٠٩١٢٣٤٥٦٧٨٩",
              "sender": "شرکت ۱۲", "contact_info": None}
    spans = find_numeric_spans(fields, min_digits=3)
    got = {(s.field, s.value_ascii) for s in spans}
    assert ("body_text", "321/000/000") in got
    assert ("body_text", "0924424117") in got
    assert ("body_text", "1403/4/524") in got            # Latin and Persian digits in ONE number
    assert ("body_text", "09123456789") in got           # Arabic-Indic
    assert not any(s.field == "sender" for s in spans)   # "۱۲" is below min_digits


def test_classify():
    assert classify("۰۹۲۴۴۲۴۱۱۷") == "national_id"
    assert classify("09123456789") == "mobile"
    assert classify("1403/07/26") == "date"
    assert classify("IR650100004060031207656012") == "iban"
    assert classify("321/000/000") == "amount"


def test_digit_normalisation():
    assert to_ascii_digits("۱۴۰۳/٠٧/26") == "1403/07/26"
    assert digits_only("IR۶۵۰۱") == "6501"


# --- validator decisions, with a fake reader ---------------------------------

class FakeReader(TesseractReader):
    """Deterministic stand-in: a fixed page layout and a fixed crop answer."""
    def __init__(self, words, crop_text):
        self._words, self._crop = words, crop_text
        self._version = "fake-5.0"
        self._pt = object(); self._cmd = None; self.langs = "fas+eng"
    def available(self): return True
    def page_words(self, im): return list(self._words)
    def read_crop(self, im, box, pad=0.35): return self._crop


def _img():
    return Image.new("RGB", (1000, 1400), "white")


def test_agreement_is_high_confidence():
    r = FakeReader([Word("۳۲۱/۰۰۰/۰۰۰", (100, 100, 400, 140), 90.0)], "۳۲۱/۰۰۰/۰۰۰")
    out = NumericValidator(r).validate(_img(), {"body_text": "مبلغ ۳۲۱/۰۰۰/۰۰۰ ریال"})
    assert len(out) == 1 and out[0].confidence == "high" and out[0].candidates == ["۳۲۱/۰۰۰/۰۰۰"]


def test_disagreement_is_low_with_both_candidates():
    # The D53 failure shape: VLM emits 221, the page says 321.
    r = FakeReader([Word("۳۲۱/۰۰۰/۰۰۰", (100, 100, 400, 140), 90.0)], "۳۲۱/۰۰۰/۰۰۰")
    out = NumericValidator(r).validate(_img(), {"body_text": "مبلغ ۲۲۱/۰۰۰/۰۰۰ ریال"})
    assert out[0].confidence == "low"
    assert out[0].candidates == ["۲۲۱/۰۰۰/۰۰۰", "321/000/000"]
    assert out[0].bbox == (100, 100, 400, 140)
    s = summarize(out); assert s["conflict_rate"] == 1.0


def test_no_region_found_is_unverified_not_high():
    # A check that cannot fail is worse than none: a page with no digits must not
    # confirm anything.
    r = FakeReader([], "")
    out = NumericValidator(r).validate(_img(), {"body_text": "کد ملی ۰۹۲۴۴۲۴۱۱۷"})
    assert out[0].confidence == "unverified"


def test_tesseract_missing_marks_everything_unverified():
    class Missing(TesseractReader):
        def __init__(self): self._pt = None; self._version = None; self._cmd = None; self.langs = "fas"
    out = NumericValidator(Missing()).validate(_img(), {"body_text": "کد ملی ۰۹۲۴۴۲۴۱۱۷"})
    assert out and all(o.confidence == "unverified" for o in out)


def test_split_number_boxes_are_merged():
    words = [Word("۱۴۰۳", (100, 100, 180, 130), 80), Word("/", (182, 100, 190, 130), 50),
             Word("۴", (192, 100, 210, 130), 80), Word("/", (212, 100, 220, 130), 50),
             Word("۵۲۴", (222, 100, 280, 130), 80)]
    r = FakeReader(words, "۱۴۰۳/۴/۵۲۴")
    out = NumericValidator(r).validate(_img(), {"sender": "شماره: ۱۴۰۳/۴/۵۲۴"})
    assert out[0].confidence == "high" and out[0].bbox[0] == 100 and out[0].bbox[2] >= 280


# --- preprocessing --------------------------------------------------------------

def _page(skew=0.0):
    im = Image.new("RGB", (1600, 2200), (120, 110, 100))          # desk
    page = Image.new("RGB", (1200, 1700), "white")
    d = ImageDraw.Draw(page)
    for y in range(150, 1500, 60):
        d.line([(120, y), (1080, y)], fill="black", width=6)        # text lines
    if skew:
        page = page.rotate(skew, expand=True, fillcolor=(120, 110, 100))
    im.paste(page, ((1600 - page.width) // 2, (2200 - page.height) // 2))
    return im


def test_prepare_crops_deskews_and_fits_under_cap():
    buf = io.BytesIO(); _page(skew=4.0).save(buf, "JPEG", quality=95)
    out, st = prepare(buf.getvalue(), max_bytes=300_000)
    assert st.out_bytes <= 300_000
    assert st.cropped and st.crop_box is not None
    assert st.deskewed and abs(abs(st.skew_deg) - 4.0) < 1.5
    assert Image.open(io.BytesIO(out)).format == "JPEG"


def test_compress_always_fits_even_for_huge_input():
    im = Image.effect_noise((4000, 4000), 80).convert("RGB")       # incompressible
    data, q, size = compress(im, max_bytes=120_000, fmt="JPEG")
    assert len(data) <= 120_000 and q >= 40


def test_estimate_skew_sign():
    # Crop first, as the pipeline does: on the raw frame the desk edges vote too.
    from ocr_service.preprocess import find_page
    bgr = _to_cv(_page(skew=3.0))
    x0, y0, x1, y1 = find_page(bgr)
    bgr = bgr[y0:y1, x0:x1]
    assert abs(abs(estimate_skew(bgr)) - 3.0) < 1.5
    assert abs(estimate_skew(rotate(bgr, estimate_skew(bgr)))) < 0.8


# --- contract -------------------------------------------------------------------

def test_response_contract_is_text_first_with_line_and_number_provenance():
    r = OcrResponse(doc_id="x", text="با سلام\nمبلغ ۳۲۱/۰۰۰/۰۰۰ ریال",
                    lines=[{"text": "با سلام", "bbox": [1, 2, 3, 4], "confidence": 0.9, "source": "psm4", "paragraph": 0}],
                    numbers=[{"value": "۳۲۱/۰۰۰/۰۰۰", "value_ascii": "321/000/000", "bbox": [5, 6, 7, 8],
                              "source": "glyph", "confidence": 0.8}],
                    is_letter=False, fields={}, needs_review=False, skew_deg=0.0, image_size=[10, 10],
                    timing={}, service={})
    assert r.text.startswith("با سلام") and r.numbers[0].source == "glyph"
    # a non-letter page carries no invented structure: every field is null
    assert all(v is None for v in r.fields.model_dump().values())


def test_the_prepared_page_reports_the_scale_it_was_sent_at():
    """The review crops depend on this and their failure is silent.

    Every bbox is measured in the prepared page, but `prepared_image` is capped at 1600 px
    so the response stays sendable. On the brief's letter that is a 0.534 factor: without it
    the four «نیاز به بازبینی» crops were cut from the wrong pixels and the one at x=1843
    fell past the capped edge and was never drawn at all.
    """
    from PIL import Image

    from ocr_service.pipeline import PREPARED_MAX_SIDE, _data_url

    big = Image.new("RGB", (2192, 2996), "white")
    url, scale = _data_url(big)
    assert url and url.startswith("data:image/jpeg;base64,")
    assert scale is not None and 0 < scale < 1, "a page past the cap must report a scale"
    # the factor has to be the one that actually maps a bbox onto the sent pixels, so
    # decode what was sent and check the width it claims
    import base64
    import io

    from PIL import Image as _I
    sent = _I.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert max(sent.size) <= PREPARED_MAX_SIDE
    # One scalar cannot describe both dimensions exactly — the thumbnail's sides are
    # integers, so the aspect ratio shifts by a fraction of a pixel. Within a pixel is the
    # honest contract, and a pixel does not matter to a crop.
    assert abs(big.width * scale - sent.width) <= 1
    assert abs(big.height * scale - sent.height) <= 1

    small = Image.new("RGB", (400, 300), "white")
    url, scale = _data_url(small)
    assert scale == 1.0, "an uncapped page is sent at full size and must say so"

    assert _data_url(None) == (None, None)
