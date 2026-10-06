# -*- coding: utf-8 -*-
"""Guards for the legibility gate (E33).

The failure it exists to stop is the only one a reader cannot detect for themselves: a page
of handwriting answered with «ee aa ay», printed as though it were a transcript.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_the_readings_that_came_off_the_real_pages_are_rejected():
    """Verbatim from the images that prompted this: a page of nastaliq, two of handwriting."""
    from ocr_service.legible import judge
    for junk in ("ff", "ee aa ay", "فو اف  as SY fe  میک",
                 "at | r a} oe _ \" - | ss ) 4 . & ' A . Cy ' at- al"):
        assert not judge(junk).legible, junk


def test_real_persian_transcripts_pass():
    """Verbatim from pages that were read correctly. A gate that withholds these is worse
    than no gate — it loses work that was already right."""
    from ocr_service.legible import judge
    for good in ("مجلس شورای اسلامی قرار گرفت و پیرو این سئوال، نشست مشترکی با حضور",
                 "غربالگری و توجه ویژه به پویش «هر خانه بماند» از مهمترین راهکارهای",
                 "جناب آقای دکتر میکائیل باقری\nسرپرست محترم اداره کل آموزش و پرورش"):
        assert judge(good).legible, good


def test_persian_carrying_latin_and_digits_still_passes():
    """An address block is Persian with an e-mail and a phone number in it. Those are Latin
    and digits, and the page is still a page."""
    from ocr_service.legible import judge
    v = judge("تلفن: ۳۸۳۸۸۵۷۵ فکس: ۳۸۳۸۸۲۹۷\nایمیل: info@sabzgostar.ir\nآدرس: مشهد")
    assert v.legible and v.fa_fraction < 1.0


def test_a_genuinely_latin_document_is_not_this_modules_business():
    """It may only withhold a Persian page that was not read. Real Latin words are long and
    many; junk Latin is «ff», «oe», «ss»."""
    from ocr_service.legible import judge
    assert judge("Dear Sir, please find attached the quarterly report for review.").legible


def test_a_short_page_with_no_persian_at_all_is_an_unread_one():
    """«ff» for a whole page of calligraphy. Too short for the fraction to mean anything, not
    too short to judge: no Persian and no Latin word either is what noise looks like."""
    from ocr_service.legible import judge
    assert not judge("ff").legible
    assert judge("سلام").legible, "a short Persian caption must survive"
    assert judge("").legible, "an empty transcript is already honest about itself"


def test_the_gate_can_only_withhold_never_alter():
    """It returns a verdict, not text. Nothing here can change what was read."""
    import inspect

    from ocr_service import legible
    src = inspect.getsource(legible.judge)
    assert "return Verdict" in src
    assert not any(bad in src for bad in ("replace(", "sub(", "strip()"))
