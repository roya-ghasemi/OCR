# -*- coding: utf-8 -*-
r"""Decide whether a page's transcript is text at all, or noise shaped like text.

Handed handwriting or nastaliq calligraphy, the recogniser does not fail — it answers. On
the images that prompted this it answered «ff», «ee aa ay», «at r a} oe _ " - ss ) 4 . & ' A»
and «فو اف as SY fe میک». Every one of those reached `text` as though it were a transcript.

That is the worst failure this service can have, and the brief says so in the first line:
«درست و ناقص» بهتر از «کامل و غلط». A reader who is told nothing could be read goes and looks
at the image. A reader handed «ee aa ay» does not know anything is wrong.

**Confidence does not detect it.** That was measured, not assumed — the junk page scoring
highest was 0.79 while a correct one scored 0.56. Tesseract is confident about nonsense,
because a stroke of nastaliq genuinely does resemble the letters it names.

What separates them is what *script* came out. On nine pages:

    page                     chars   Persian fraction   mean confidence
    nastaliq 17                  6         0.00              0.49
    handwriting 16               2         0.00              0.79
    handwriting 19              27         0.00              0.64
    nastaliq 15                 13         0.54              0.53
    ------------------------------------------------------------------
    benchmark line 18           53         1.00              0.88
    benchmark line 20           51         1.00              0.56
    Arabic page 13             568         0.94              0.67
    the brief's letter         716         0.94              0.86
    photographed prose         559         0.95              0.81
    low-resolution photo      1355         0.96              0.81

0.54 against 0.94, with nothing in between. A Persian page that was read produces Persian
characters; one that was not produces whatever the shapes happened to resemble, which is
mostly Latin and punctuation because those are the simplest forms in the alphabet.

The gate is deliberately one-sided. It can only *withhold* a transcript, never alter one, and
it withholds only when the evidence is this stark — the margin above is wide enough that
moving the threshold anywhere between 0.6 and 0.9 changes nothing on these pages.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Arabic script, which Persian is written in. The range covers both.
_FA = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LATIN_WORD = re.compile(r"[A-Za-z]{4,}")

# Below this fraction of Arabic-script characters, a Persian page was not read. Measured
# separation is 0.54 against 0.94; this sits in the gap with room on both sides.
MIN_FA_FRACTION = 0.60
# A page with less ink than this is not enough evidence either way — a caption, a stamp, a
# single word. Withholding those would lose real text for no gain.
MIN_CHARS = 8
# A genuinely Latin document — an English letter, a form in both scripts — must pass. Junk
# Latin is «ff», «oe», «ss», «at»: one and two characters. Real words are longer, and there
# are many of them.
MIN_LATIN_WORDS = 4


@dataclass
class Verdict:
    legible: bool
    fa_fraction: float
    chars: int
    latin_words: int
    reason: str = ""

    def as_dict(self) -> dict:
        return {"legible": self.legible, "fa_fraction": round(self.fa_fraction, 3),
                "chars": self.chars, "latin_words": self.latin_words, "reason": self.reason}


def judge(text: str) -> Verdict:
    """Is this a transcript of a Persian page, or the shapes it happened to resemble?"""
    letters = [c for c in (text or "") if not c.isspace()]
    n = len(letters)
    latin_words = len(_LATIN_WORD.findall(text or ""))
    if n == 0:
        return Verdict(True, 0.0, 0, 0)        # an empty transcript is already honest
    fa = sum(1 for c in letters if _FA.match(c)) / n
    if n < MIN_CHARS:
        # Too little for the fraction to mean anything, but not too little to judge. A whole
        # page of calligraphy answered «ff», and another «ee aa ay»: no Persian at all and no
        # Latin word either, which is what noise looks like at this length. A short caption
        # that really was read has Persian in it, so one character of it is enough to pass.
        if fa == 0 and latin_words == 0:
            return Verdict(False, fa, n, latin_words,
                           f"{n} characters came out and none of them are Persian — "
                           f"that is not a short page, it is an unread one")
        return Verdict(True, fa, n, latin_words)
    if fa >= MIN_FA_FRACTION:
        return Verdict(True, fa, n, latin_words)
    if latin_words >= MIN_LATIN_WORDS:
        # Latin, but real Latin: words, not fragments. Not this module's business.
        return Verdict(True, fa, n, latin_words)
    return Verdict(False, fa, n, latin_words,
                   f"only {fa:.0%} of what was read is Persian script — the page was not "
                   f"read, and what came out is the shapes the strokes resembled")
