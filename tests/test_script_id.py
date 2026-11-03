# -*- coding: utf-8 -*-
"""Guards for Arabic detection (E34).

The gate only decides whether to spend a second pass; confidence decides which transcript
wins. So a false positive costs a second, and a false negative costs an Arabic page read as
damaged Persian. These pin the measured separation.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ARABIC = ("الله وَالرَّاسِخُونَ فِي الْعِلْمِ آل عمران: ٧، ولم يقل تفسيره، لأنّه أراد مايؤول "
          "من المتشابه إلى المحكم. وهذا عزيز من وجهين أحدهما أن تجمع صفة غير الحيوان")
PERSIAN = ("غربالگری و توجه ویژه به پویش «هر خانه بماند» از مهمترین راهکارهای پیشگیری است "
           "و مجلس شورای اسلامی قرار گرفت و پیرو این سئوال، نشست مشترکی با حضور وزیر")


def test_a_vocalised_arabic_page_is_detected():
    from ocr_service.script_id import looks_arabic
    assert looks_arabic(ARABIC)


def test_persian_prose_is_not():
    """Measured: every Persian page scored exactly zero harakat per hundred characters."""
    from ocr_service.script_id import looks_arabic
    assert not looks_arabic(PERSIAN)


def test_a_short_transcript_is_never_sent_for_a_second_read():
    """Below forty characters there is not enough to tell, and a second pass on every short
    page would cost more than the case is worth."""
    from ocr_service.script_id import looks_arabic
    assert not looks_arabic("سلام")
    assert not looks_arabic("")
    assert not looks_arabic("تلفن: ۳۸۳۸۸۵۷۵")


def test_the_persian_only_letters_decide_an_unvocalised_page():
    """Unvocalised Arabic scores zero harakat too, so پ چ ژ گ carry the rest — but only on a
    page long enough to have contained one."""
    from ocr_service.script_id import MIN_CHARS_FOR_LETTER_TEST, looks_arabic
    bare = "".join(c for c in ARABIC if not ("ً" <= c <= "ْ" or c == "ٰ"))
    bare = (bare + " ") * 4
    assert len([c for c in bare if not c.isspace()]) >= MIN_CHARS_FOR_LETTER_TEST
    assert looks_arabic(bare)
    assert not looks_arabic((PERSIAN + " ") * 4)


def test_arabic_letterforms_survive_an_arabic_read():
    """ي and ك are the correct letters in Arabic, not misspellings of ی and ک. Folding them
    produced «الولمه» for «الْعِلْمِ»."""
    from ocr_service.transcribe import Transcriber
    fa = Transcriber.__new__(Transcriber)
    fa.arabic_script = False
    ar = Transcriber.__new__(Transcriber)
    ar.arabic_script = True
    assert fa._clean_word("يك", latin=False) == "یک"
    assert ar._clean_word("يك", latin=False) == "يك"
    # and harakat are stripped for Persian, kept for Arabic
    assert ar._clean_word("وَالرَّاسِخُونَ", latin=False) == "وَالرَّاسِخُونَ"
