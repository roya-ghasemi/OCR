# -*- coding: utf-8 -*-
"""Guards for field validation (E31).

A verdict that is wrong in the lenient direction is worse than no verdict: it tells a
reviewer a damaged value is fine. These pin the cases that actually came off the brief's
letter.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_a_jalali_month_has_the_right_number_of_days():
    """The asymmetry is the whole value of the check. Months 1-6 have 31 days, 7-11 have 30,
    and month 12 has 29 except in a leap year. «۱۴۰۲/۱۲/۳۰» is a real-looking date that does
    not exist."""
    from ocr_service.validate import check_date
    assert check_date("۱۴۰۲/۱۲/۲۷").valid
    assert not check_date("۱۴۰۲/۱۲/۳۰").valid          # 1402 is not a leap year
    assert check_date("۱۴۰۳/۱۲/۳۰").valid              # 1403 is
    assert not check_date("۱۴۰۲/۰۷/۳۱").valid          # month 7 has 30 days
    assert check_date("۱۴۰۲/۰۶/۳۱").valid              # month 6 has 31


def test_a_digit_misread_into_the_month_is_caught():
    """What a four-digits-slash-two-digits check cannot see."""
    from ocr_service.validate import check_date
    f = check_date("۱۴۰۲/۱۳/۲۷")
    assert not f.valid and "13" in f.reason


def test_a_date_is_never_repaired():
    """The value comes back exactly as read. Correcting month 13 to 12 would be inventing
    the page's content, which is the one thing this service does not do."""
    from ocr_service.validate import check_date
    f = check_date("۱۴۰۲/۱۳/۲۷")
    assert f.value == "۱۴۰۲/۱۳/۲۷"
    assert f.normalized is None


def test_the_truncated_address_off_the_brief_letter():
    """«info@sabzgostar-govareshk ۱۲” ;» — the domain has no dot because the end of the line
    was not read. It must not pass."""
    from ocr_service.validate import check_email
    f = check_email("info@sabzgostar-govareshk")
    assert not f.valid and "dot" in f.reason
    assert check_email("info@sabzgostar.ir").valid


def test_a_list_of_telephone_numbers_is_not_one_long_number():
    """A letterhead prints «۳۸۳۸۸۵۷۵ - ۳۸۳۸۸۴۸۱ - ۳۸۳۸۸۲۶۱»: three numbers, not one of
    twenty-four digits. Taking the run whole reported a single broken number and buried the
    two that were fine."""
    from ocr_service.validate import split_numbers
    assert split_numbers("۳۸۳۸۸۵۷۵ - ۳۸۳۸۸۴۸۱ - ۳۸۳۸۸۲۶۱") == ["۳۸۳۸۸۵۷۵", "۳۸۳۸۸۴۸۱", "۳۸۳۸۸۲۶۱"]
    # ...but an area code and its subscriber number are one number, not two
    assert split_numbers("۰۵۱-۳۸۳۸۸۵۷۵") == ["۰۵۱-۳۸۳۸۸۵۷۵"]


def test_a_subscriber_number_with_one_extra_digit_is_rejected():
    """The commonest way a misread number still looks like a number. The recogniser produced
    «۳۸۳۸۸۲۶۱۰» on the brief's letter — «۳۸۳۸۸۲۶۱» with a spurious trailing zero — and
    nothing else in the service would have questioned it."""
    from ocr_service.validate import check_phone
    assert check_phone("۳۸۳۸۸۲۶۱").valid
    assert not check_phone("۳۸۳۸۸۲۶۱۰").valid
    assert check_phone("۰۹۱۲۳۴۵۶۷۸۹").valid           # a mobile is 11


def test_only_labelled_runs_are_judged_as_telephone_numbers():
    """Without the label every long number on the page — a registration number, an amount, a
    postal code — would be judged a telephone number and most would be reported broken:
    noise standing exactly where a real warning should be."""
    from ocr_service.validate import scan
    found = scan("قرارداد به شماره ۲۹۰۶۳ و مبلغ ۱۲۳۴۵۶۷ ریال")
    assert not [f for f in found if f.kind == "phone"]
