"""Unit tests for ocr_eval/normalize.py (Phase 1 4.4 requires per-rule tests).

Run: venv312\\Scripts\\python.exe -m pytest ocr_eval/test_normalize.py -q
"""
import sys
import unicodedata
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import normalize as N  # noqa: E402


# -- individual rules ---------------------------------------------------------

def only(**flags):
    """A Rules object with everything off except the named flags."""
    return N.Rules(**{**{k: False for k in N.Rules().__dataclass_fields__}, **flags})


def test_nfc_runs_first():
    decomposed = "A" + "́"  # A + combining acute
    assert N.normalize(decomposed, only(nfc=True)) == unicodedata.normalize("NFC", decomposed)


def test_strip_bidi_removes_all_seven_controls_and_four_isolates():
    marks = "‎‏‪‫‬‭‮⁦⁧⁨⁩"
    assert N.normalize("a" + marks + "b", only(strip_bidi=True)) == "ab"


def test_strip_bidi_off_preserves():
    assert N.normalize("a‏b", only()) == "a‏b"


def test_arabic_yeh_and_kaf_fold_to_persian():
    # "aadaare kol fanaavari" as the model actually emits it (defect D8)
    src = "اداره كل فناورى"
    out = N.normalize(src, only(arabic_to_persian=True))
    assert "ك" not in out and "ى" not in out
    assert "ک" in out and "ی" in out


def test_alef_hamza_folds_but_madda_does_not_by_default():
    src = "أإٱآ"  # hamza-above, hamza-below, wasla, madda
    out = N.normalize(src, N.RULES_DEFAULT)
    assert out == "اااآ", "madda must survive the default rules"


def test_alef_madda_folds_when_explicitly_enabled():
    assert N.normalize("آ", only(fold_alef_madda=True)) == "ا"


def test_teh_marbuta_folds_to_heh():
    assert N.normalize("ة", only(fold_teh_marbuta=True)) == "ه"


def test_zwnj_becomes_a_space_not_a_deletion():
    # Deleting ZWNJ would silently merge two tokens and corrupt WER.
    src = "می‌رود"
    assert N.normalize(src, only(strip_zwnj=True)) == "می رود"


def test_zwnj_preserved_under_keep_zwnj_rules():
    src = "می‌رود"
    assert "‌" in N.normalize(src, N.RULES_KEEP_ZWNJ)


def test_harakat_and_tatweel_stripped():
    src = "احتراماً" + "ــ"
    out = N.normalize(src, only(strip_harakat=True, strip_tatweel=True))
    assert "ً" not in out and "ـ" not in out
    assert out == "احتراما"


@pytest.mark.parametrize("digits", [N.PERSIAN_DIGITS, N.ARABIC_DIGITS])
def test_both_indic_digit_systems_map_to_ascii(digits):
    assert N.normalize(digits, only(digits_to_ascii=True)) == "0123456789"


def test_ascii_digits_are_left_alone():
    assert N.normalize("0123456789", only(digits_to_ascii=True)) == "0123456789"


def test_latin_punctuation_folds_to_persian():
    assert N.normalize("a,b;c?", only(fold_punctuation=True)) == "a؋b؛c؟".replace("؋", "،")


def test_arabic_separators_fold_to_ascii():
    assert N.normalize("1٫5", only(fold_punctuation=True)) == "1.5"
    assert N.normalize("1٬000", only(fold_punctuation=True)) == "1,000"


def test_whitespace_collapse_handles_nbsp_and_runs_and_trims():
    src = "  a  b \t c  \n\n  d  "
    assert N.normalize(src, only(collapse_whitespace=True)) == "a b c\nd"


def test_lowercase_latin_is_off_by_default():
    assert N.normalize("INV-2024", N.RULES_DEFAULT) == "INV-2024"
    assert N.normalize("INV-2024", only(lowercase_latin=True)) == "inv-2024"


# -- contracts ----------------------------------------------------------------

def test_none_becomes_empty_string():
    assert N.normalize(None) == ""


def test_raw_rules_are_the_identity_function():
    src = "  كـً۱‌  ,‮ "
    assert N.normalize(src, N.RULES_RAW) == src


def test_normalize_is_idempotent():
    src = "اداره كلـ ۱۴۰۲  أمر"
    once = N.normalize(src)
    assert N.normalize(once) == once


def test_every_rule_flag_has_a_test():
    """Guard: a new rule added without a test fails here rather than silently
    shipping untested."""
    tested = {
        "nfc", "strip_bidi", "strip_zwj", "arabic_to_persian", "fold_alef",
        "fold_alef_madda", "fold_teh_marbuta", "strip_zwnj", "strip_harakat",
        "strip_tatweel", "digits_to_ascii", "fold_punctuation",
        "collapse_whitespace", "lowercase_latin",
    }
    assert set(N.Rules().__dataclass_fields__) == tested


# -- the letterform metric must NOT be hidden by normalization ----------------

def test_letterform_report_sees_arabic_forms_in_raw_text():
    rep = N.letterform_report(["اداره كل", "سلام"])
    assert rep["n_responses"] == 2
    assert rep["n_with_arabic_letterforms"] == 1
    assert rep["pct_with_arabic_letterforms"] == 50.0
    assert rep["total_arabic_kaf"] == 1


def test_letterform_report_is_clean_after_normalization():
    """Proves the point of 4.4 'Critical': normalization erases the evidence, so
    conformance must be measured before it."""
    src = "اداره كل"
    assert N.letterform_report([N.normalize(src)])["n_with_arabic_letterforms"] == 0


def test_audit_counts_digit_systems_separately():
    a = N.audit("۱۲ ١٢ 12")
    assert a["persian_digits_U06F0_9"] == 2
    assert a["arabic_digits_U0660_9"] == 2
    assert a["ascii_digits"] == 2


def test_rules_dict_carries_the_version():
    assert N.rules_dict()["normalizer_version"] == N.NORMALIZER_VERSION
