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
# The fixed vocabulary of administrative letters, which the general list is missing.
ADMIN = Path(__file__).resolve().parent / "data" / "fas_admin.txt"

# Letters that share a skeleton and differ only in the marks a low-resolution scan loses
# or invents: dots (بپتثنی, جچحخ, سش …) and the madda/hamza (اآأإ, وؤ, هة). A lost mark
# moves a letter within its group and nowhere else.
_GROUPS = ("بپتثنی", "جچحخ", "دذ", "رزژ", "سش", "صض", "طظ", "عغ", "فق", "کگ", "هة", "وؤ", "اآأإ")
_REP = {c: g[0] for g in _GROUPS for c in g}

_PERSIAN = re.compile(r"^[ء-غف-يپچژکگی‌]+$")
_ZWNJ = "‌"

# Affixes Persian orthography legitimately separates with a ZWNJ. A word broken here was
# written that way, not stretched by a kashida.
_SUFFIXES = frozenset(("ها", "های", "هایی", "هایم", "هایت", "هایش", "هایمان", "هایتان",
                       "هایشان", "تر", "تری", "ترین", "ام", "ات", "اش", "مان", "تان",
                       "شان", "ای", "اند", "ایم", "اید", "گر", "گری", "گان"))
_PREFIXES = frozenset(("می", "نمی", "بی", "به"))


def skeleton(w: str) -> str:
    """The word with every letter folded onto its mark-group's representative."""
    return "".join(_REP.get(c, c) for c in w)


class SpellFix:
    """Dot-confusion repair. `available()` is False when the lexicon is missing, and the
    service then leaves every word exactly as it was read."""

    def __init__(self, path: Path = WORDS, min_len: int = 3, max_conf: float = 85.0,
                 max_edits: int = 1, admin: Path = ADMIN):
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
        # The administrative words are added the same way, so they protect themselves (a
        # word in the list is returned untouched) and can also be the answer for a word
        # that is one moved dot from them.
        try:
            for line in admin.read_text(encoding="utf-8").splitlines():
                w = line.strip()
                if w and not w.startswith("#") and _PERSIAN.match(w) and w not in self.words:
                    self.words.add(w)
                    self.by_skeleton.setdefault(skeleton(w), []).append(w)
        except OSError as exc:
            log.warning("administrative word list not loaded from %s (%s)", admin, exc)

    def available(self) -> bool:
        return bool(self.words)

    def unsplit(self, word: str) -> str | None:
        """A word a kashida was read as two pieces of, put back together.

        Justified Persian is stretched with kashida, and the recogniser reads a long
        connector as a gap: «استخراج» comes back «اس‌تخراج», «ابری» as «اب ری». Closing the
        gap is only allowed when the split form is **not** a word and the closed form is —
        so «می‌رساند», whose ZWNJ is real, is never touched.
        """
        if _ZWNJ not in word or word in self.words:
            return None
        # A zero-width non-joiner that separates a real Persian affix is orthography, not
        # damage, and the lexicon is no help: it holds «نیازمندیهای» without the mark, so
        # «نیازمندی‌های» looked like a word to repair. Affixes are therefore exempt.
        parts = word.split(_ZWNJ)
        if parts[-1] in _SUFFIXES or parts[0] in _PREFIXES:
            return None
        joined = word.replace(_ZWNJ, "")
        return joined if joined in self.words else None

    def undouble(self, word: str) -> str | None:
        """A word with one letter too many, where exactly one lexicon word is left by
        removing a single character.

        The other half of the same kashida damage: a stretched connector grows a tooth and
        «بسیاری» comes back «بسپیاری», «زمان» as «زمصان». As everywhere else here, one
        candidate or nothing — two candidates mean there is something to choose between,
        and choosing would be guessing.
        """
        if word in self.words or len(word) < self.min_len + 1:
            return None
        cand = {word[:i] + word[i + 1:] for i in range(len(word))}
        hit = [w for w in cand if w in self.words]
        return hit[0] if len(hit) == 1 else None

    def fix(self, word: str, conf: float) -> str:
        """The word as printed, or the single lexicon word it is a dot variant of."""
        if conf >= self.max_conf or len(word) < self.min_len or not _PERSIAN.match(word):
            return word
        if word in self.words:
            return word
        # Narrow to the words this one could be, then require the answer to be unique.
        # Both steps matter: «نبروی» shares a skeleton with نیروی, بیروت, بیرون and
        # پیروی, but only «نیروی» is one moved dot away, so there is still nothing to
        # choose between. Filtering after the uniqueness test would have declined it.
        near = [w for w in self.by_skeleton.get(skeleton(word), ())
                if w != word and len(w) == len(word)
                and sum(a != b for a, b in zip(word, w)) <= self.max_edits]
        # «بمار» (a misreading of «بهار») is two moved dots from «نماز» and none from
        # anything else, so it is left exactly as it was read.
        if len(near) == 1:
            return near[0]
        # Only now the length-changing repairs, and the order is the whole point. A moved
        # dot keeps the word's length and its skeleton, so it is by far the better-evidenced
        # guess; deleting a letter is a last resort. Run the other way round, deletion won
        # three words it had no business touching — «آداره» became «داره», «نسیت» «نیت» and
        # «منایع» «مایع», where the dot rule would have returned «اداره», «نسبت», «منابع».
        for repair in (self.unsplit, self.undouble):
            out = repair(word)
            if out:
                return out
        return word
