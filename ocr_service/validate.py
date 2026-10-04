# -*- coding: utf-8 -*-
r"""Check the shape of the values a letter carries: e-mail addresses, telephone numbers, dates.

This **never repairs anything**. A malformed value is reported as malformed and left exactly
as it was read, because the only way to turn «info@sabzgostar-govareshk ۱۲” ;» into an address
is to invent the missing part, and an invented e-mail address is worse than a visibly broken
one — it looks correct. The brief's rule governs here as everywhere: «درست و ناقص» بهتر از
«کامل و غلط».

What a verdict is for: the reader of a transcript cannot tell a correctly-read address from a
damaged one, because both are just text. A verdict says which, so a human checks the two that
are doubtful instead of re-reading all nine.

Dates are Jalali and the month lengths are not uniform — 31 days in months 1-6, 30 in 7-11,
and 29 in month 12 except in a leap year, when it has 30. That asymmetry is the whole value of
checking: «۱۴۰۲/۱۲/۳۰» is a real-looking date that does not exist, and a digit misread into
month position («۱۴۰۲/۱۳/۲۷») is invisible to anything that only checks for four digits, a
slash and two more.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

FA_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

# A local part, an @, a host with at least one dot and a plausible TLD. Deliberately stricter
# than the RFC: this is not validating what an address *may* be, it is asking whether what
# came out of a recogniser looks like one.
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+", re.A)
_EMAIL_OK = re.compile(r"^[A-Za-z0-9._%+\-]+@(?:[A-Za-z0-9\-]+\.)+[A-Za-z]{2,}$", re.A)

# Iranian numbers as they are printed on a letterhead: a mobile (۰۹xx xxx xxxx), a landline
# with or without its area code, or a bare subscriber number next to a «تلفن»/«فکس» label.
_PHONE_RUN = re.compile(r"[۰-۹0-9٠-٩][۰-۹0-9٠-٩\s\-()]{5,20}[۰-۹0-9٠-٩]")
_PHONE_LABEL = re.compile(r"(تلفن|تلفکس|فکس|نمابر|موبایل|همراه|تماس)")

# ۱۴۰۲/۱۲/۲۷ and the usual separators.
_DATE = re.compile(r"([۰-۹0-9٠-٩]{2,4})\s*[/\-.]\s*([۰-۹0-9٠-٩]{1,2})\s*[/\-.]\s*([۰-۹0-9٠-٩]{1,2})")

JALALI_MIN_YEAR, JALALI_MAX_YEAR = 1300, 1499


def _ascii(s: str) -> str:
    return s.translate(FA_DIGITS)


@dataclass
class Finding:
    kind: str                       # "email" | "phone" | "date"
    value: str                      # exactly as it was read, never normalised
    valid: bool
    reason: str = ""                # why not, when it is not
    normalized: str | None = None   # a comparable form, for a *valid* value only

    def as_dict(self) -> dict:
        return {"kind": self.kind, "value": self.value, "valid": self.valid,
                "reason": self.reason, "normalized": self.normalized}


def is_jalali_leap(year: int) -> bool:
    """The 33-year cycle. Years leaving remainder 1, 5, 9, 13, 17, 22, 26 or 30 are leap.

    This is the arithmetic ("33-year") rule, which is what calendars in administrative use
    follow and is exact for the range this service sees. The astronomical rule can differ for
    a handful of distant years, and a date this says is valid when it is not costs nothing —
    a date it wrongly rejects would send someone to check a correct value, which does.
    """
    return (year % 33) in (1, 5, 9, 13, 17, 22, 26, 30)


def jalali_month_days(year: int, month: int) -> int:
    if month <= 6:
        return 31
    if month <= 11:
        return 30
    return 30 if is_jalali_leap(year) else 29


def check_date(raw: str) -> Finding:
    m = _DATE.search(raw)
    if not m:
        return Finding("date", raw, False, "not a date")
    y, mo, d = (int(_ascii(g)) for g in m.groups())
    if y < 100:                                  # a two-digit year, «۰۲/۱۲/۲۷»
        y += 1400 if y < 50 else 1300
    if not (JALALI_MIN_YEAR <= y <= JALALI_MAX_YEAR):
        return Finding("date", raw, False, f"year {y} is outside {JALALI_MIN_YEAR}-{JALALI_MAX_YEAR}")
    if not 1 <= mo <= 12:
        return Finding("date", raw, False, f"month {mo} does not exist")
    last = jalali_month_days(y, mo)
    if not 1 <= d <= last:
        return Finding("date", raw, False, f"day {d} does not exist in month {mo} of {y} "
                                           f"({last} days)")
    return Finding("date", raw, True, normalized=f"{y:04d}/{mo:02d}/{d:02d}")


def check_email(raw: str) -> Finding:
    v = raw.strip()
    if _EMAIL_OK.match(v):
        return Finding("email", raw, True, normalized=v.lower())
    if "@" not in v:
        return Finding("email", raw, False, "no @")
    host = v.split("@", 1)[1]
    if "." not in host:
        return Finding("email", raw, False, "no dot in the domain — the end is probably missing")
    return Finding("email", raw, False, "characters that cannot appear in an address")


def check_phone(raw: str) -> Finding:
    digits = re.sub(r"\D", "", _ascii(raw))
    if not digits:
        return Finding("phone", raw, False, "no digits")
    n = len(digits)
    if digits.startswith("09") and n == 11:
        return Finding("phone", raw, True, normalized=digits)
    # 8 is a bare subscriber number, 10 and 11 carry an area or mobile prefix. **9 is not a
    # length an Iranian number has**, and saying so is the point: a subscriber number that
    # has picked up one extra digit is the commonest way a misread number still looks like a
    # number. On the brief's letter the recogniser produced «۳۸۳۸۸۲۶۱۰» — «۳۸۳۸۸۲۶۱» with a
    # spurious trailing zero — and nothing else in the service would have questioned it.
    if n in (8, 10, 11):
        return Finding("phone", raw, True, normalized=digits)
    if n == 9:
        return Finding("phone", raw, False,
                       "9 digits — one too many for a subscriber number, one too few with "
                       "an area code")
    return Finding("phone", raw, False,
                   f"{n} digits — an Iranian number has 8 without its area code, 10 or 11 with it")


def scan(text: str) -> list[Finding]:
    """Every e-mail, telephone number and date in a block of transcript, with a verdict.

    Phone runs are only taken from lines that carry a telephone label. Without that, every
    long number on the page — a registration number, an amount, a postal code — would be
    judged as a telephone number and most would be reported broken, which is noise standing
    exactly where a real warning should be.
    """
    out: list[Finding] = []
    for m in _EMAIL.finditer(text):
        out.append(check_email(m.group(0)))
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for m in _DATE.finditer(line):
            out.append(check_date(m.group(0)))
        if _PHONE_LABEL.search(line):
            after = line[_PHONE_LABEL.search(line).end():]
            for m in _PHONE_RUN.finditer(after):
                run = m.group(0).strip()
                if _DATE.search(run):
                    continue                      # a date next to a label is not a number
                out.extend(check_phone(p) for p in split_numbers(run))
    return out


def split_numbers(run: str) -> list[str]:
    """One run of digits and separators, cut into the numbers it actually contains.

    A letterhead prints «تلفن: ۳۸۳۸۸۵۷۵ - ۳۸۳۸۸۴۸۱ - ۳۸۳۸۸۲۶۱», which is three eight-digit
    numbers, not one of twenty-four. Taking the run whole reported it as a single broken
    number and buried the two that were fine.

    The ambiguity is real: a hyphen separates «۰۵۱-۳۸۳۸۸۵۷۵» (one number, area code and
    subscriber) as well as a list. They are told apart by what joining would produce — an
    area code is short, begins with a zero, and the pair reaches a canonical ten or eleven
    digits. Anything else is a list.
    """
    groups = [g for g in re.findall(r"[۰-۹0-9٠-٩]+", run) if g]
    if not groups:
        return [run]
    if len(groups) == 2:
        a, b = (_ascii(g) for g in groups)
        if a.startswith("0") and len(a) <= 4 and 10 <= len(a) + len(b) <= 11:
            return [run]                          # area code + subscriber: one number
    return groups if len(groups) > 1 else [run]
