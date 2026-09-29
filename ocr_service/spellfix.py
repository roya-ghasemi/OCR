# -*- coding: utf-8 -*-
r"""Repair dot confusions in Persian words, against Tesseract's own Persian lexicon.

On a low-resolution capture the marks that separate ب/پ/ت/ث/ن/ی, ج/چ/ح/خ, س/ش and the
rest are 1-2 px and land below the sampling grid, so «صبحگاهی» comes back «صبخگاهی» and
«هوشمند» comes back «هوسمند» (D76). The letters themselves are read correctly; only the
dots are wrong. That is a much narrower problem than spelling, and it can be repaired
without guessing:

* Two words are **dot variants** when they are identical once every letter is replaced by
  its dot group's representative — its *skeleton*. «صبخگاهی» and «صبحگاهی» share one;
  «شادی» and «هادی» do not, because ش and ه are different shapes, not the same shape with
  different dots.
* A word is only touched when it is **not itself a word**, the reader was **unsure** of
  it, and **exactly one** lexicon word shares its skeleton. One candidate means there is
  nothing to choose between, so nothing is guessed.

This is why the earlier attempt failed (D77): a 462-word domain lexicon left 72% of a
prose page outside it, so "not in the lexicon" did not mean "not a word", and edit
distance let «شادی» become «هادی» and «۱۴۰۴» become «۱۴۰۳». Both are impossible here —
the first is not a skeleton match, and digits are never touched at all.

The lexicon is `data/fas_words.txt`, unpacked from the `fas.traineddata` this service
already runs on (`combine_tessdata -u`, then `dawg2wordlist`), so it is exactly the
vocabulary the recogniser was trained against, needs no new dependency and works offline.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

WORDS = Path(__file__).resolve().parent / "data" / "fas_words.txt"

# Letters that share a skeleton and differ only in the marks a low-resolution scan loses
# or invents: dots (بپتثنی, جچحخ, سش …) and the madda/hamza (اآأإ, وؤ, هة). A lost mark
# moves a letter within its group and nowhere else.
_GROUPS = ("بپتثنی", "جچحخ", "دذ", "رزژ", "سش", "صض", "طظ", "عغ", "فق", "کگ", "هة", "وؤ", "اآأإ")
_REP = {c: g[0] for g in _GROUPS for c in g}

_PERSIAN = re.compile(r"^[ء-غف-يپچژکگی‌]+$")
_ZWNJ = "‌"


def skeleton(w: str) -> str:
    """The word with every letter folded onto its mark-group's representative."""
    return "".join(_REP.get(c, c) for c in w)


class SpellFix:
    """Dot-confusion repair. `available()` is False when the lexicon is missing, and the
    service then leaves every word exactly as it was read."""

    def __init__(self, path: Path = WORDS, min_len: int = 3, max_conf: float = 85.0,
                 max_edits: int = 1):
        self.min_len = min_len
        self.max_conf = max_conf          # a word the reader was sure of is left alone
        self.max_edits = max_edits        # how many dots may be wrong at once
        self.words: set[str] = set()
        self.by_skeleton: dict[str, list[str]] = {}
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                w = line.strip()
                if w and _PERSIAN.match(w):
                    self.words.add(w)
                    self.by_skeleton.setdefault(skeleton(w), []).append(w)
        except OSError as exc:
            log.warning("spell fix disabled, no lexicon at %s (%s)", path, exc)

    def available(self) -> bool:
        return bool(self.words)

    def fix(self, word: str, conf: float) -> str:
        """The word as printed, or the single lexicon word it is a dot variant of."""
        if conf >= self.max_conf or len(word) < self.min_len or not _PERSIAN.match(word):
            return word
        if word in self.words:
            return word
        cand = self.by_skeleton.get(skeleton(word))
        if not cand or len(cand) != 1 or cand[0] == word:
            return word
        fixed = cand[0]
        # A repair may move dots; it may never change a letter's shape or the length.
        if len(fixed) != len(word):
            return word
        # And only so many at once: «بمار» (a misreading of «بهار») shares a skeleton with
        # «نماز» across TWO moved dots, and turning it into «نماز» would be an invention.
        if sum(a != b for a, b in zip(word, fixed)) > self.max_edits:
            return word
        return fixed
