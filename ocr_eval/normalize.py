"""Persian text normalization for evaluation (Phase 1 4.4).

Applied IDENTICALLY to prediction and ground truth, never to one side only.

Two distinct jobs must not be confused:

  * This module measures **reading**. It folds away differences a human reader
    would not notice (Arabic vs Persian letterforms, digit systems, harakat), so
    CER reflects whether the model read the page.
  * Letterform *conformance* is a separate product question -- does the API emit
    canonical Persian? -- and is measured by `letterform_report()` on the RAW
    prediction, before any normalization. Folding U+064A into U+06CC here would
    otherwise hide defect D8 completely.

Every rule is individually toggleable so a single rule can be ablated. Every
headline metric is published twice: raw (RULES_RAW) and normalized (RULES_DEFAULT).

Non-ASCII control characters are written as escapes on purpose: they are
invisible in an editor and silently corrupt on copy/paste otherwise.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, replace
from typing import Iterable

NORMALIZER_VERSION = "1.0.0"

# -- Character inventories ----------------------------------------------------
# LRM RLM LRE RLE PDF LRO RLO  +  isolates LRI RLI FSI PDI
BIDI_CONTROLS = (
    "‎‏‪‫‬‭‮"
    "⁦⁧⁨⁩"
)
ZWNJ = "‌"
ZWJ = "‍"
TATWEEL = "ـ"
# U+064B..U+0652 harakat, plus superscript alef and the hamza marks that behave
# as diacritics rather than letters.
HARAKAT = "".join(chr(c) for c in range(0x064B, 0x0653)) + "ٰٕٔٓ"
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ASCII_DIGITS = "0123456789"

# Arabic -> Persian letterforms. Only characters whose Persian form is
# unambiguous; anything needing a judgement call gets its own rule below.
ARABIC_TO_PERSIAN = {
    "ي": "ی",   # ARABIC YEH        -> FARSI YEH
    "ى": "ی",   # ALEF MAKSURA      -> FARSI YEH
    "ے": "ی",   # YEH BARREE        -> FARSI YEH
    "ك": "ک",   # ARABIC KAF        -> KEHEH
    "ڪ": "ک",   # SWASH KAF         -> KEHEH
}

# Alef variants. POLICY (ablatable via `fold_alef`):
#   hamza carriers and wasla  ->  bare alef
#   alef madda (U+0622) -> bare alef ONLY when `fold_alef_madda` is on. In
#   Persian orthography madda is a distinct sound; folding it loses real
#   information, so it is OFF by default and reported separately.
ALEF_HAMZA = {
    "أ": "ا",   # ALEF WITH HAMZA ABOVE
    "إ": "ا",   # ALEF WITH HAMZA BELOW
    "ٲ": "ا",   # ALEF WITH WAVY HAMZA ABOVE
    "ٳ": "ا",   # ALEF WITH WAVY HAMZA BELOW
    "ٱ": "ا",   # ALEF WASLA
}
ALEF_MADDA = {"آ": "ا"}

# TEH MARBUTA -> HEH. POLICY: fold. Persian does not use teh marbuta; where it
# appears it spells a word Persian writes with heh. Ablatable.
TEH_MARBUTA = {"ة": "ه"}

# Latin punctuation -> Persian. POLICY: fold TO the Persian form, because the
# documents are Persian and the Latin glyph is the OCR error.
PUNCT_TO_PERSIAN = {",": "،", ";": "؛", "?": "؟"}
# Arabic decimal / thousands separators and fraction slash -> ASCII.
SEPARATOR_TO_ASCII = {"٫": ".", "٬": ",", "⁄": "/"}

NBSP = " "
# Every horizontal space Unicode knows about, so a run collapses to one space.
_SPACES = " \t               　​"
_WS_RUN = re.compile("[" + _SPACES + "]+")
_NL_RUN = re.compile(r"\s*\n\s*")


@dataclass(frozen=True)
class Rules:
    """One flag per normalization rule. `asdict()` goes into every results file."""

    nfc: bool = True                  # Unicode NFC, before anything else
    strip_bidi: bool = True           # U+200E/200F/202A-202E/2066-2069
    strip_zwj: bool = True            # U+200D carries no meaning in this corpus
    arabic_to_persian: bool = True    # yeh/kaf folding (see letterform_report)
    fold_alef: bool = True            # hamza carriers -> bare alef
    fold_alef_madda: bool = False     # U+0622 -> U+0627 (information loss)
    fold_teh_marbuta: bool = True     # U+0629 -> U+0647
    strip_zwnj: bool = True           # U+200C (both settings are reported)
    strip_harakat: bool = True        # U+064B-U+0652 and friends
    strip_tatweel: bool = True        # U+0640
    digits_to_ascii: bool = True      # Persian- and Arabic-Indic -> 0-9
    fold_punctuation: bool = True     # , ; ? -> Persian; U+066B/C -> . ,
    collapse_whitespace: bool = True  # runs -> one space, NBSP -> space, trim
    lowercase_latin: bool = False     # off: casing is part of the transcription


RULES_DEFAULT = Rules()
RULES_RAW = Rules(**{k: False for k in Rules().__dataclass_fields__})
RULES_KEEP_ZWNJ = replace(RULES_DEFAULT, strip_zwnj=False)


def _translate(text: str, table: dict) -> str:
    return text.translate(str.maketrans(table))


def normalize(text: str | None, rules: Rules = RULES_DEFAULT) -> str:
    """Normalize one string.

    `None` becomes `""`. Callers that must distinguish null from empty have to
    check before calling -- that distinction is what null-agreement scores.
    """
    if text is None:
        return ""
    s = text

    if rules.nfc:
        s = unicodedata.normalize("NFC", s)
    if rules.strip_bidi:
        s = s.translate({ord(c): None for c in BIDI_CONTROLS})
    if rules.strip_zwj:
        s = s.replace(ZWJ, "")
    if rules.strip_tatweel:
        s = s.replace(TATWEEL, "")
    if rules.strip_harakat:
        s = s.translate({ord(c): None for c in HARAKAT})
    if rules.arabic_to_persian:
        s = _translate(s, ARABIC_TO_PERSIAN)
    if rules.fold_alef:
        s = _translate(s, ALEF_HAMZA)
    if rules.fold_alef_madda:
        s = _translate(s, ALEF_MADDA)
    if rules.fold_teh_marbuta:
        s = _translate(s, TEH_MARBUTA)
    if rules.digits_to_ascii:
        s = _translate(s, dict(zip(PERSIAN_DIGITS, ASCII_DIGITS)))
        s = _translate(s, dict(zip(ARABIC_DIGITS, ASCII_DIGITS)))
    if rules.fold_punctuation:
        s = _translate(s, PUNCT_TO_PERSIAN)
        s = _translate(s, SEPARATOR_TO_ASCII)
    if rules.strip_zwnj:
        # ZWNJ is a word-part boundary. Replace it with a space rather than
        # deleting it, so token counts stay stable -- WER depends on that.
        s = s.replace(ZWNJ, " ")
    if rules.lowercase_latin:
        s = "".join(c.lower() if c.isascii() else c for c in s)
    if rules.collapse_whitespace:
        s = s.replace(NBSP, " ")
        s = _NL_RUN.sub("\n", s)
        s = _WS_RUN.sub(" ", s)
        s = "\n".join(ln.strip() for ln in s.split("\n")).strip()
    return s


# -- Conformance / audit counters (run on RAW text, never on normalized) ------
_COUNTERS = {
    "arabic_yeh_U064A": "ي",
    "arabic_kaf_U0643": "ك",
    "alef_maksura_U0649": "ى",
    "teh_marbuta_U0629": "ة",
    "zwnj_U200C": ZWNJ,
    "zwj_U200D": ZWJ,
    "tatweel_U0640": TATWEEL,
}


def audit(text: str | None) -> dict:
    """Per-string character census.

    Feeds the encoding audit (2.2) and the standalone letterform-conformance
    metric (1.4, 'Critical').
    """
    s = text or ""
    out = {name: s.count(ch) for name, ch in _COUNTERS.items()}
    out["persian_digits_U06F0_9"] = sum(s.count(c) for c in PERSIAN_DIGITS)
    out["arabic_digits_U0660_9"] = sum(s.count(c) for c in ARABIC_DIGITS)
    out["ascii_digits"] = sum(s.count(c) for c in ASCII_DIGITS)
    out["harakat"] = sum(1 for c in s if c in HARAKAT)
    out["bidi_controls"] = sum(1 for c in s if c in BIDI_CONTROLS)
    out["chars"] = len(s)
    return out


def letterform_report(texts: Iterable[str | None]) -> dict:
    """1.4: percentage of API responses containing Arabic yeh/kaf.

    Published on its own so the normalizer cannot hide defect D8.
    """
    texts = list(texts)
    n = len(texts)
    stats = [audit(t) for t in texts]
    non_conforming = sum(
        1
        for a in stats
        if a["arabic_yeh_U064A"] or a["arabic_kaf_U0643"] or a["alef_maksura_U0649"]
    )
    return {
        "n_responses": n,
        "n_with_arabic_letterforms": non_conforming,
        "pct_with_arabic_letterforms": round(100.0 * non_conforming / n, 2) if n else 0.0,
        "total_arabic_yeh": sum(a["arabic_yeh_U064A"] for a in stats),
        "total_arabic_kaf": sum(a["arabic_kaf_U0643"] for a in stats),
        "total_alef_maksura": sum(a["alef_maksura_U0649"] for a in stats),
    }


def rules_dict(rules: Rules = RULES_DEFAULT) -> dict:
    d = asdict(rules)
    d["normalizer_version"] = NORMALIZER_VERSION
    return d
