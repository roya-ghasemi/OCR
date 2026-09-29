# -*- coding: utf-8 -*-
"""Full-text transcription (ocr_service/transcribe.py) and rule-cut letter fields
(ocr_service/letter_fields.py). Pure tests need nothing installed; the end-to-end
test needs Tesseract with Persian data and skips without it."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "ocr_eval")]

from ocr_service.letter_fields import FIELDS, extract  # noqa: E402
from ocr_service.transcribe import Line, Transcriber, Word, clean_ink  # noqa: E402


@dataclass
class L:                       # the shape letter_fields reads: .text, .bbox, .confidence
    text: str
    bbox: tuple
    confidence: float = 0.9


def _page(texts, h=3000):
    step = h // (len(texts) + 1)
    return [L(t, (100, (i + 1) * step, 2000, (i + 1) * step + 50)) for i, t in enumerate(texts)]


# --- letter fields: verbatim or null, never generated -------------------------------

def test_letter_fields_are_verbatim_runs_of_lines():
    lines = _page(["شماره: ۱۲۳", "جناب آقای دکتر سلطانی", "معاون محترم پژوهش", "موضوع: درخواست لغو ضمانت‌نامه",
                   "با سلام؛", "احتراماً عطف به نامه شماره ۷۰۹", "با تشکر", "آدرس: مشهد - بلوار کوثر",
                   "تلفن: ۳۸۳۸۸۵۷۵"])
    f, is_letter = extract(lines, 3000)
    assert is_letter
    assert f["receiver"] == "جناب آقای دکتر سلطانی\nمعاون محترم پژوهش"
    assert f["subject"] == "درخواست لغو ضمانت‌نامه"
    assert f["body_text"] == "با سلام؛\nاحتراماً عطف به نامه شماره ۷۰۹\nبا تشکر"
    assert f["contact_info"] == "آدرس: مشهد - بلوار کوثر\nتلفن: ۳۸۳۸۸۵۷۵"
    every = "\n".join(l.text for l in lines)
    assert all(v is None or all(part in every for part in v.split("\n")) for v in f.values())


def test_subject_is_null_unless_a_subject_label_is_printed():
    f, _ = extract(_page(["ریاست محترم بانک سامان", "با سلام؛", "احتراما خواهشمند است ..."]), 3000)
    assert f["subject"] is None             # the GT's summarised subjects are not invented


def test_a_page_that_is_not_a_letter_gets_no_fields():
    f, is_letter = extract(_page(["بهار دل‌انگیز از راه رسید", "کودکان با شادی بازی می‌کردند"]), 3000)
    assert not is_letter and all(v is None for v in f.values())


def test_a_stray_mark_before_the_salutation_does_not_hide_it():
    f, _ = extract(_page(["ریاست محترم بانک سامان شعبه هاشمیه", "۰ با سلام؛", "احتراما خواهشمند است"]), 3000)
    assert f["receiver"] == "ریاست محترم بانک سامان شعبه هاشمیه"
    assert f["body_text"].startswith("۰ با سلام")


def test_an_ocr_damaged_opening_is_still_found_but_the_closing_formula_is_not_an_opening():
    f, _ = extract(_page(["ریاست محترم بانک ملت", "اشه احترام بر مذاکرات قبلی پیوست", "با احترام"]), 3000)
    assert f["body_text"].startswith("اشه احترام بر مذاکرات")
    f, is_letter = extract(_page(["گزارش کار ماهانه", "با احترام"]), 3000)
    assert f["body_text"] is None


def test_low_confidence_or_label_lines_never_join_the_receiver():
    lines = _page(["پیوست: ندارد", "سا ۹۴ او سم", "مدیریت محترم شرکت آب", "موضوع: گواهی اشتغال", "با سلام"])
    lines[1].confidence = 0.2
    f, _ = extract(lines, 3000)
    assert f["receiver"] == "مدیریت محترم شرکت آب"


# --- line selection and clean-up ------------------------------------------------------

def _w(t, c, x0, x1, y0=100, y1=140):
    return Word(t, c, (x0, y0, x1, y1))


def test_best_reading_per_line_wins_and_same_row_reads_right_to_left():
    weak = Line([_w("بسا", 40, 500, 600), _w("توچسه", 45, 300, 480)], (300, 100, 600, 140), "psm4")
    strong = Line([_w("با", 95, 520, 600), _w("توجه", 92, 300, 480)], (300, 100, 600, 140), "line-fas")
    left = Line([_w("شماره:", 90, 20, 200)], (20, 100, 200, 140), "psm3")
    rows = Transcriber._select([weak, strong, left])
    assert [[L.text for L in r] for r in rows] == [["با توجه", "شماره:"]]
    assert Transcriber._join_row(rows[0]).text == "با توجه شماره:"


def test_two_clean_halves_of_a_tilted_line_beat_one_merged_reading():
    # the user's page: psm 6 read the whole tilted line merged with the next one and
    # lost «نسیم خنک صبحگاهی»; psm 3/4 read it as two clean halves
    merged = Line([_w("بهار", 90, 1800, 1900), _w("دل‌انگیز", 90, 1600, 1780), _w("از", 90, 1500, 1580),
                   _w("راه", 90, 1400, 1480), _w("سیم", 70, 600, 700), _w("بحگاهی", 60, 300, 500),
                   _w("ک", 40, 250, 280)], (220, 166, 2196, 273), "psm6")
    right = Line([_w("بهار", 92, 1800, 1900), _w("دل‌انگیز", 91, 1600, 1780), _w("از", 95, 1500, 1580),
                  _w("راه", 93, 1400, 1480)], (1238, 189, 1904, 260), "psm4")
    left = Line([_w("نسیم", 90, 600, 720), _w("خنک", 88, 500, 590), _w("صبحگاهی", 86, 300, 490),
                 _w("از", 92, 250, 290)], (222, 159, 1221, 227), "psm4")
    rows = Transcriber._select([merged, right, left])
    assert Transcriber._join_row(rows[0]).text == "بهار دل‌انگیز از راه نسیم خنک صبحگاهی از"


def test_a_junk_piece_is_dropped_before_its_row_is_joined():
    t = Transcriber(None, None, None)
    real = Line([_w("شادی", 90, 1500, 1600), _w("و", 90, 1450, 1480), _w("هیجان", 88, 1300, 1430)],
                (1300, 100, 1600, 140), "psm4")
    speck = Line([_w("اس", 35, 2150, 2170), _w("ی", 20, 2180, 2190)], (2150, 100, 2190, 140), "psm6")
    lines = t._finish_rows([[speck, real]])
    assert [L.text for L in lines] == ["شادی و هیجان"]


def test_invented_harakat_are_stripped_but_a_real_single_mark_stays():
    assert Transcriber._clean_word("بِمُدیرِیَت", latin=False) == "بمدیریت"
    assert Transcriber._clean_word("احتراماً", latin=False) == "احتراماً"
    assert Transcriber._clean_word("بهارِ", latin=False) == "بهارِ"


def test_a_lone_far_character_at_a_line_end_is_trimmed_but_a_real_short_word_is_kept():
    line = Line([_w("۰", 60, 1900, 1910), _w("با", 90, 1700, 1760), _w("و", 90, 1640, 1660), _w("سلام", 90, 1500, 1630)],
                (1500, 100, 1910, 140), "psm4")
    Transcriber._trim_edges(line)
    assert line.text == "با و سلام"


def test_table_cells_read_right_to_left_even_when_tesseract_emits_numbers_left_to_right():
    # row «۲۳ | ۱۳ | ۹ مشهد»: Tesseract gave the two number cells in visual LTR order
    row = Line([_w("۱۳", 90, 1000, 1060), _w("۲۳", 90, 1500, 1560), _w("۹", 90, 400, 430), _w("مشهد", 90, 250, 390)],
               (250, 100, 1560, 140), "psm4")
    Transcriber._order_segments(row)
    assert row.text == "۲۳ ۱۳ ۹ مشهد"
    prose = Line([_w("با", 90, 560, 600), _w("توجه", 90, 470, 550)], (470, 100, 600, 140), "psm4")
    Transcriber._order_segments(prose)
    assert prose.text == "با توجه"                  # normal word spacing: order untouched


def test_rules_and_specks_go_glyphs_and_their_dots_stay():
    ink = np.zeros((1000, 1000), bool)
    ink[500:503, 50:950] = True                     # a table rule
    ink[300:330, 400:420] = True                    # a letter body (shorter than the H/25 vertical-rule kernel)
    ink[334:338, 405:409] = True                    # its dot
    ink[800:802, 100:102] = True                    # a speck far from anything
    out = clean_ink(ink)
    assert not out[501, 500] and out[315, 410] and out[336, 407] and not out[801, 101]


# --- end to end on a rendered Persian page ---------------------------------------------

def _font(px):
    for f in (r"C:\Windows\Fonts\tahoma.ttf", r"C:\Windows\Fonts\arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(f).is_file():
            return ImageFont.truetype(f, px)
    pytest.skip("no font with Persian glyphs")


def _render(lines, px=44):
    import arabic_reshaper
    from bidi.algorithm import get_display
    font = _font(px)
    W, H = 1800, 240 + len(lines) * int(px * 2.2)
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    for i, t in enumerate(lines):
        s = get_display(arabic_reshaper.reshape(t))
        w = d.textbbox((0, 0), s, font=font)[2]
        d.text((W - 120 - w, 120 + i * int(px * 2.2)), s, font=font, fill="black")
    return im


def test_every_line_of_a_multi_paragraph_page_comes_back_in_order():
    from ocr_service.config import Settings
    cfg = Settings()
    cmd, data = cfg.resolved_tesseract()
    t = Transcriber(cmd, data, cfg.resolved_digit_model())
    if not t.health()["tesseract"]["available"]:
        pytest.skip("Tesseract with Persian data not installed")
    gt = ["جلسه بررسی طرح توسعه خدمات هوشمند", "روز دوشنبه ساعت ده صبح برگزار می‌شود",
          "خواهشمند است مستندات را ارسال کنید", "همچنین نسخه نهایی گزارش تهیه شود",
          "پاسخ این پرسش به عوامل زیادی بستگی دارد"]
    tr = t.transcribe(_render(gt))
    from fulltext_score import semiglobal
    from normalize import normalize
    text = normalize(tr.text)
    for line in gt:                                  # nothing dropped (the 1,400-char cap bug)
        g = normalize(line)
        assert semiglobal(g, text) / len(g) < 0.2, (line, tr.text)
    firsts = [text.find(normalize(l).split()[0]) for l in gt]
    assert firsts == sorted(firsts), tr.text          # reading order kept


# --- numbers: read what is printed, flag what cannot be right ---------------------------

def test_a_grouped_amount_is_judged_by_its_groups_and_a_date_is_left_alone():
    from ocr_service.transcribe import amount_grouping
    assert amount_grouping("۱۲۵,۰۰۰,۰۰۰") == "ok"
    assert amount_grouping("۷/۷۷۵/۰۰۰/۰۰۰") == "ok"
    assert amount_grouping("۳۲۱/۰۰۰/۰۰") == "broken"          # a digit went missing
    assert amount_grouping("۱۲.۹۷۸۰۲۱۹.۵۷۱") == "broken"      # two groups ran together
    assert amount_grouping("۱۶۵.۰۰۰۰۰۰") == "broken"          # a separator was missed
    assert amount_grouping("۱۴۰۳/۰۹/۲۰") == "none"            # a date
    assert amount_grouping("۱۶۰/۱۶۰۰/۱۴۰۳") == "none"         # a reference number
    assert amount_grouping("۰۵۱۱-۵۰۲۷۸۷۱") == "none"          # a phone number


def test_one_number_read_in_two_pieces_is_rejoined_without_inventing_a_separator():
    from types import SimpleNamespace
    t = Transcriber(None, None, None, number_min_prob=0.6)
    def read(text, x0, x1):          # what _fix_numbers reads off a digit-reader result
        return SimpleNamespace(text=text, box=(x0, 12, x1, 48), n_digits=len(text),
                               mean_prob=0.9, line_h=36.0)
    line = Line([_w("۱۴۰۳۰۹۰۶", 90, 100, 300, 10, 50)], (100, 10, 300, 50), "psm4")
    t._fix_numbers([line], [read("۱۴۰۳", 100, 200), read("۰۹۰۶", 205, 300)], 1.0)
    assert line.words[0].text == "۱۴۰۳۰۹۰۶"        # joined, and «۱۴۰۳/۰۹/۰/۶» never happens

    far = Line([_w("۱۴۰۳۰۹۰۶", 90, 100, 400, 10, 50)], (100, 10, 400, 50), "psm4")
    t._fix_numbers([far], [read("۱۴۰۳", 100, 200), read("۰۹۰۶", 300, 400)], 1.0)
    assert far.words[0].text == "۱۴۰۳۰۹۰۶"         # too far apart to be one number: untouched


def test_a_table_row_is_split_into_cells_and_prose_is_not():
    row = [_w("مبلغ صورت وضعیت", 90, 1500, 1900), _w("۱۲۵,۰۰۰,۰۰۰", 90, 400, 700)]
    assert [c["text"] for c in Transcriber._cells(row, 40, 1.0)] == ["مبلغ صورت وضعیت", "۱۲۵,۰۰۰,۰۰۰"]
    assert Transcriber._cells([_w("با", 90, 560, 600), _w("توجه", 90, 470, 550)], 40, 1.0) == []


def test_a_stretched_connector_is_shortened_and_a_dash_is_left_alone():
    from ocr_service.transcribe import squeeze_kashida
    ink = np.zeros((60, 200), bool)
    ink[10:50, 10:40] = True                    # a letter
    ink[37:50, 40:100] = True                   # joined to the next by a long kashida
    ink[10:50, 100:130] = True                  # the next letter
    ink[20:33, 150:190] = True                  # a dash: as flat, but standing alone
    im = Image.fromarray(np.where(ink, 0, 255).astype(np.uint8))
    out, xmap = squeeze_kashida(im)
    assert out.width < im.width and len(xmap) == out.width
    kept = set(xmap.tolist())
    assert all(x in kept for x in range(150, 190))            # the dash stands alone: untouched
    assert len([x for x in range(40, 100) if x in kept]) <= 15   # 60 columns -> a connector
    assert all(x in kept for x in list(range(10, 40)) + list(range(100, 130)))   # letters whole


def test_a_low_resolution_page_says_so_instead_of_guessing_the_letters():
    """Persian dots live in 1-2 px. Below ~20 px text height they are not in the image at
    all, so «صبحگاهی» reads «صبخگاهی» and no post-processing can know which is right."""
    from ocr_service.pipeline import REVIEW_MIN_GLYPH_PX
    from ocr_service.schemas import OcrResponse

    def reasons(glyph_px):
        out = []
        if glyph_px and glyph_px < REVIEW_MIN_GLYPH_PX:
            out.append(f"image resolution too low: text is {glyph_px:.0f} px tall, "
                       f"under the {REVIEW_MIN_GLYPH_PX:.0f} px needed to resolve Persian dots — "
                       f"rescan at 300 DPI for accurate text")
        return out

    assert reasons(12.0) and "12 px" in reasons(12.0)[0]     # the user's prose photo
    assert reasons(7.0)                                      # exampel_paper.png
    assert not reasons(31.0)                                 # a corpus scan
    assert not reasons(0.0)                                  # not measured: no claim
    assert "glyph_px" in OcrResponse.model_fields


def test_a_page_that_is_not_a_letter_still_comes_back_whole_and_in_order():
    """The standing requirement: not every image is an administrative letter. A page of
    prose must return every line, top to bottom, in paragraphs — with the letter fields
    null rather than invented, and nothing dropped because the page has no salutation."""
    from ocr_service.config import Settings
    from ocr_service.pipeline import OcrPipeline
    page = ROOT / "ocr_eval" / "samples" / "prose_page.jpg"
    if not page.is_file():
        pytest.skip("sample page not present")
    pipe = OcrPipeline(Settings())
    if not pipe.health()["tesseract"]["available"]:
        pytest.skip("Tesseract with Persian data not installed")
    r = pipe.run_sync(page.read_bytes(), page.name)

    assert not r.is_letter
    assert all(getattr(r.fields, f) is None for f in FIELDS)     # nothing invented
    assert len(r.lines) >= 20 and len(r.text) > 1200             # and nothing dropped
    assert len({L.paragraph for L in r.lines}) >= 4              # its four paragraphs

    tops = [L.bbox[1] for L in r.lines]
    assert tops == sorted(tops), "lines must come back top to bottom"
    for L in r.lines:                                            # each line reads right to left
        assert L.text and L.text in r.text
    assert any("resolution too low" in x for x in r.review_reasons)   # 12 px page: says so


def test_a_low_resolution_capture_is_upsampled_and_a_good_scan_is_left_alone():
    """Adaptive upscaling keys on the text height **as supplied**. Keying it on the
    working height instead pulled the 24-29 px corpus scans up too and cost dev CER
    14.7% → 15.1% (E21)."""
    from ocr_service.transcribe import working_scale
    photo = Image.new("RGB", (1125, 1500))            # the user's 12 px page
    assert working_scale(photo, 12.0) == pytest.approx(30.0 / 12.0)
    assert working_scale(Image.new("RGB", (472, 669)), 7.0) == 4.0        # capped at 4x
    scan = Image.new("RGB", (2424, 3232))
    assert working_scale(scan, 31.0) == 1.0           # resolves its own dots: untouched
    assert working_scale(scan, 24.0) == 1.0           # and so does this one
