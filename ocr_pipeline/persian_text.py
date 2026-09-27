"""Persian text normalization for the API output contract.

THE TRAP THIS MODULE EXISTS TO AVOID
------------------------------------
"Convert ASCII digits to Persian-Indic digits" is correct for prose and
CATASTROPHIC applied blindly. These documents contain:

    IR650100060412075501234567     an IBAN
    Sabzgostar2006@gmail.com       an email
    0513-824691                    a phone number
    https://example.ir/x2          a URL

Rewriting those digits to ۰-۹ produces a string that is no longer a valid IBAN,
no longer a routable email, and no longer machine-comparable. Measured on this
corpus, 64% of returned digits are ASCII and a large share sit inside exactly
these identifier contexts.

So digit folding here is CONTEXT-AWARE: identifier spans are masked out before
any digit rewriting and restored afterwards. That is what "dynamically" has to
mean to be safe.

DIRECTION IS A POLICY DECISION, NOT A DEFAULT
---------------------------------------------
Two different jobs want opposite directions, and conflating them silently
corrupts data:

  * API output contract  -> PERSIAN_INDIC. The page is written in ۰-۹; a faithful
    transcription preserves that.
  * Machine comparison, evaluation, DB keys, checksum validation -> ASCII. You
    cannot run an IBAN mod-97 or a national-ID checksum over ۰-۹.

Both are provided. `for_output()` and `for_machine()` are the two intended entry
points; call the right one deliberately.

LETTERFORM FOLDING IS NOT OPTIONAL
----------------------------------
Measured on live output: 8-14% of field values come back with Arabic ي (U+064A)
and ك (U+0643) instead of Persian ی (U+06CC) and ک (U+06A9). These are visually
near-identical and compare unequal. Any consumer doing a string match against a
database of Persian names will silently miss those rows.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Iterable

NORMALIZER_VERSION = "2.0.0"


class DigitPolicy(str, Enum):
    """What to do with digits outside identifier spans."""

    PERSIAN_INDIC = "persian_indic"   # ۰-۹  - faithful to the page; API output
    ASCII = "ascii"                   # 0-9  - machine-comparable; validation
    PRESERVE = "preserve"             # leave exactly as the model emitted them


# -- character inventories ---------------------------------------------------
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"        # U+06F0..U+06F9
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"         # U+0660..U+0669
ASCII_DIGITS = "0123456789"

# Bidi controls: LRM RLM LRE RLE PDF LRO RLO + the four isolates.
BIDI_CONTROLS = "‎‏‪‫‬‭‮⁦⁧⁨⁩"
ZWNJ = "‌"
ZWJ = "‍"
TATWEEL = "ـ"
# U+064B..U+0652 plus superscript alef and the standalone hamza marks.
HARAKAT = "".join(chr(c) for c in range(0x064B, 0x0653)) + "ٰٕٔٓ"

# Arabic -> Persian letterforms. Only unambiguous mappings; anything requiring a
# judgement call gets its own toggle below.
ARABIC_TO_PERSIAN = {
    "ي": "ی",   # ARABIC YEH        -> FARSI YEH
    "ى": "ی",   # ALEF MAKSURA      -> FARSI YEH
    "ے": "ی",   # YEH BARREE        -> FARSI YEH
    "ك": "ک",   # ARABIC KAF        -> KEHEH
    "ڪ": "ک",   # SWASH KAF         -> KEHEH
}
# Presentation-form Arabic (U+FB50..U+FEFF) sometimes survives OCR of scanned
# text. NFKC folds these to their canonical letters; we then map to Persian.
ALEF_HAMZA = {"أ": "ا", "إ": "ا", "ٲ": "ا",
              "ٳ": "ا", "ٱ": "ا"}
ALEF_MADDA = {"آ": "ا"}
TEH_MARBUTA = {"ة": "ه"}

# Latin punctuation the model emits where the page has Persian punctuation.
PUNCT_TO_PERSIAN = {",": "،", ";": "؛", "?": "؟"}
# Arabic decimal / thousands separators -> ASCII, so numbers stay parseable.
SEPARATOR_TO_ASCII = {"٫": ".", "٬": ",", "⁄": "/"}

NBSP = " "
_SPACES = " \t               　​"
_WS_RUN = re.compile("[" + re.escape(_SPACES) + "]+")
_NL_RUN = re.compile(r"[ \t]*\n[ \t]*")

# -- identifier spans that must never have their digits rewritten -------------
# Order matters: IBAN and email before the bare phone/number patterns, so the
# greedier generic patterns cannot bite off part of a structured identifier.
_PROTECTED_PATTERNS = [
    ("iban",   re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{10,30}\b")),
    ("email",  re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("url",    re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)),
    # Latin-prefixed reference codes: INV-2024-0001, EMP-1234, REF/9/12
    ("code",   re.compile(r"\b[A-Za-z]{2,6}[-_/][A-Za-z0-9][A-Za-z0-9\-_/]*\b")),
    # An ASCII run containing a Latin letter is an identifier, not prose.
    ("alnum",  re.compile(r"\b(?=[A-Za-z0-9\-_/]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9\-_/]{2,}\b")),
]
_PLACEHOLDER = ""   # private-use area: cannot occur in real OCR output


@dataclass(frozen=True)
class NormalizationPolicy:
    """Every rule is individually toggleable so one can be ablated in isolation."""

    unicode_form: str = "NFC"          # "NFC" | "NFKC" | ""
    strip_bidi: bool = True
    strip_zwj: bool = True
    arabic_to_persian: bool = True     # the non-negotiable one
    fold_alef_hamza: bool = False      # أ إ ٱ -> ا  : loses orthographic detail
    fold_alef_madda: bool = False      # آ -> ا      : loses a phoneme
    fold_teh_marbuta: bool = True      # ة -> ه
    strip_harakat: bool = True
    strip_tatweel: bool = True
    normalize_zwnj: bool = True        # collapse runs, drop it around spaces
    digits: DigitPolicy = DigitPolicy.PERSIAN_INDIC
    protect_identifiers: bool = True   # NEVER disable in production
    fold_punctuation: bool = False     # off: punctuation is part of the read
    collapse_whitespace: bool = True

    def as_dict(self) -> dict:
        d = asdict(self)
        d["digits"] = self.digits.value
        d["normalizer_version"] = NORMALIZER_VERSION
        return d


# The API output contract: canonical Persian letterforms, Persian-Indic digits in
# prose, identifiers left untouched.
OUTPUT_POLICY = NormalizationPolicy(digits=DigitPolicy.PERSIAN_INDIC)

# For checksum validation, DB keys, and any string comparison.
MACHINE_POLICY = NormalizationPolicy(
    digits=DigitPolicy.ASCII,
    fold_alef_hamza=True,
    fold_teh_marbuta=True,
    protect_identifiers=False,   # we WANT identifier digits in ASCII here
)


def _mask_identifiers(text: str) -> tuple[str, list[str]]:
    """Replace identifier spans with placeholders so digit folding skips them."""
    saved: list[str] = []

    def _sub(m: re.Match) -> str:
        saved.append(m.group(0))
        return f"{_PLACEHOLDER}{len(saved) - 1}{_PLACEHOLDER}"

    for _name, pat in _PROTECTED_PATTERNS:
        text = pat.sub(_sub, text)
    return text, saved


def _unmask_identifiers(text: str, saved: list[str]) -> str:
    def _sub(m: re.Match) -> str:
        return saved[int(m.group(1))]

    return re.sub(f"{_PLACEHOLDER}(\\d+){_PLACEHOLDER}", _sub, text)


def _translate(text: str, table: dict[str, str]) -> str:
    return text.translate(str.maketrans(table))


def normalize(text: str | None, policy: NormalizationPolicy = OUTPUT_POLICY) -> str | None:
    """Normalize one field value. `None` stays `None` - a null field is data.

    Order is deliberate: Unicode form first (so composed and decomposed input
    behave alike), identifier masking before any digit work, whitespace last.
    """
    if text is None:
        return None
    s = text

    if policy.unicode_form:
        s = unicodedata.normalize(policy.unicode_form, s)
    if policy.strip_bidi:
        s = s.translate({ord(c): None for c in BIDI_CONTROLS})
    if policy.strip_zwj:
        s = s.replace(ZWJ, "")
    if policy.strip_tatweel:
        s = s.replace(TATWEEL, "")
    if policy.strip_harakat:
        s = s.translate({ord(c): None for c in HARAKAT})

    if policy.arabic_to_persian:
        s = _translate(s, ARABIC_TO_PERSIAN)
    if policy.fold_alef_hamza:
        s = _translate(s, ALEF_HAMZA)
    if policy.fold_alef_madda:
        s = _translate(s, ALEF_MADDA)
    if policy.fold_teh_marbuta:
        s = _translate(s, TEH_MARBUTA)
    if policy.fold_punctuation:
        s = _translate(s, PUNCT_TO_PERSIAN)
    s = _translate(s, SEPARATOR_TO_ASCII)

    # -- digits, with identifiers held out of harm's way ---------------------
    if policy.digits is not DigitPolicy.PRESERVE:
        saved: list[str] = []
        if policy.protect_identifiers:
            s, saved = _mask_identifiers(s)

        if policy.digits is DigitPolicy.ASCII:
            s = _translate(s, dict(zip(PERSIAN_DIGITS, ASCII_DIGITS)))
            s = _translate(s, dict(zip(ARABIC_DIGITS, ASCII_DIGITS)))
        else:  # PERSIAN_INDIC
            s = _translate(s, dict(zip(ASCII_DIGITS, PERSIAN_DIGITS)))
            s = _translate(s, dict(zip(ARABIC_DIGITS, PERSIAN_DIGITS)))

        if policy.protect_identifiers:
            s = _unmask_identifiers(s, saved)

    if policy.normalize_zwnj:
        # ZWNJ is a word-part boundary. Collapse runs, and drop it where a real
        # space already separates the parts - a doubled boundary breaks matching.
        s = re.sub(f"{ZWNJ}+", ZWNJ, s)
        s = re.sub(f"[ ]*{ZWNJ}[ ]*", lambda m: ZWNJ if m.group(0) == ZWNJ else " ", s)

    if policy.collapse_whitespace:
        s = s.replace(NBSP, " ")
        s = _NL_RUN.sub("\n", s)
        s = _WS_RUN.sub(" ", s)
        s = "\n".join(line.strip() for line in s.split("\n")).strip()

    return s


def for_output(text: str | None) -> str | None:
    """Canonical Persian for the API response. Identifiers preserved verbatim."""
    return normalize(text, OUTPUT_POLICY)


def for_machine(text: str | None) -> str | None:
    """ASCII digits and folded letterforms, for checksums and comparison."""
    return normalize(text, MACHINE_POLICY)


def normalize_extraction(
    data: dict[str, str | None],
    policy: NormalizationPolicy = OUTPUT_POLICY,
) -> dict[str, str | None]:
    """Normalize every string value in an extraction, leaving nulls as nulls."""
    return {
        k: (normalize(v, policy) if isinstance(v, str) else v)
        for k, v in data.items()
    }


# -- conformance auditing ----------------------------------------------------

def audit(text: str | None) -> dict[str, int]:
    """Character census of RAW text.

    Run this BEFORE normalization. Normalizing first erases the very evidence
    this is meant to surface, which is how a letterform defect stays invisible
    behind a healthy-looking error rate.
    """
    s = text or ""
    return {
        "chars": len(s),
        "arabic_yeh_U064A": s.count("ي"),
        "arabic_kaf_U0643": s.count("ك"),
        "alef_maksura_U0649": s.count("ى"),
        "teh_marbuta_U0629": s.count("ة"),
        "zwnj_U200C": s.count(ZWNJ),
        "tatweel_U0640": s.count(TATWEEL),
        "harakat": sum(1 for c in s if c in HARAKAT),
        "bidi_controls": sum(1 for c in s if c in BIDI_CONTROLS),
        "persian_digits": sum(s.count(c) for c in PERSIAN_DIGITS),
        "arabic_digits": sum(s.count(c) for c in ARABIC_DIGITS),
        "ascii_digits": sum(s.count(c) for c in ASCII_DIGITS),
    }


def letterform_conformance(values: Iterable[str | None]) -> dict:
    """Percentage of values carrying non-Persian letterforms, for monitoring."""
    stats = [audit(v) for v in values]
    n = len(stats)
    bad = sum(
        1 for a in stats
        if a["arabic_yeh_U064A"] or a["arabic_kaf_U0643"] or a["alef_maksura_U0649"]
    )
    return {
        "n_values": n,
        "n_non_conforming": bad,
        "pct_non_conforming": round(100.0 * bad / n, 2) if n else 0.0,
        "total_arabic_yeh": sum(a["arabic_yeh_U064A"] for a in stats),
        "total_arabic_kaf": sum(a["arabic_kaf_U0643"] for a in stats),
    }
