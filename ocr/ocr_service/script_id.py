# -*- coding: utf-8 -*-
r"""Tell an Arabic page from a Persian one, cheaply, from a transcript already in hand.

Persian and Arabic share an alphabet and a different model reads each. Handed an Arabic page,
the Persian model produces «ل والوایخون ف الیلم» where the Arabic model produces
«الله وَالوَاسِخُونَ في الْولمه» — against a true «الله وَالرَّاسِخُونَ فِي الْعِلْمِ». The second line of
that page comes back exact. The gap is not subtle and it is worth a second pass.

A second pass is not worth paying for on every page, though: it would add a second or two to
every Persian letter the service will ever read, to catch a case that is rare. So this is a
**gate, not a decision**. It decides only whether to spend the pass; which transcript to keep
is then settled by comparing their confidence, which is measurement rather than inference.

Two signals, measured across six pages:

    page                 harakat/100 chars    پ چ ژ گ /100 chars
    the Arabic page            1.23                 0.53
    ------------------------------------------------------------
    benchmark line 18          0.00                 1.89
    benchmark line 20          0.00                 3.92
    the reference letter         0.00                 2.09
    photographed prose         0.00                 1.25
    low-resolution photo       0.00                 2.95

Harakat separates perfectly — every Persian page scores exactly zero — and it does so even
though `_clean_word` has already stripped most of them, which is itself evidence the Arabic
page is being damaged. But unvocalised Arabic would score zero too, so the absence of
Persian's own four letters carries the rest. That signal is weaker (0.53 against 1.25 is less
than a factor of three) and meaningless on a short page, where a Persian line may legitimately
contain no پ, چ, ژ or گ at all — so it only speaks for a page long enough to have said one.
"""
from __future__ import annotations

import re

# پ چ ژ گ exist in Persian and not in Arabic.
_PERSIAN_ONLY = re.compile(r"[پچژگ]")
# Fatha, damma, kasra, tanwin, shadda, sukun, dagger alef.
_HARAKAT = re.compile(r"[ً-ْٰ]")

# Above this many harakat per 100 characters the page is vocalised, which Persian prose is
# not. Measured 1.23 against a flat 0.00.
HARAKAT_PER_100 = 0.30
# Below this many Persian-only letters per 100 characters, on a page long enough for the
# count to mean something, the text is not Persian.
PERSIAN_ONLY_PER_100 = 0.80
MIN_CHARS_FOR_LETTER_TEST = 200


def looks_arabic(text: str) -> bool:
    """Is this transcript worth re-reading with the Arabic model?"""
    chars = [c for c in (text or "") if not c.isspace()]
    n = len(chars)
    if n < 40:
        return False                      # too little to tell, and too little to be worth it
    s = "".join(chars)
    if len(_HARAKAT.findall(s)) / n * 100 >= HARAKAT_PER_100:
        return True
    if n >= MIN_CHARS_FOR_LETTER_TEST:
        return len(_PERSIAN_ONLY.findall(s)) / n * 100 < PERSIAN_ONLY_PER_100
    return False
