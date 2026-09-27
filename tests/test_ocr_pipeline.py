"""Tests for the hardening modules.

Run: venv312\\Scripts\\python.exe -m pytest tests/test_ocr_pipeline.py -q

The cases are built from real observed failures, not invented ones. Where a test
encodes a measured fact (the 1.1-vs-1.2 threshold, the GBNF parser quirks), the
docstring says so, because those are the assertions that will look arbitrary to
whoever touches this next.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_pipeline import (  # noqa: E402
    DEFAULT_SPEC, PROVEN, ACCURACY_PRESERVING, RETRY, DigitPolicy,
    NormalizationPolicy, Verdict, detect_degeneracy, for_machine, for_output,
    normalize, repair_json, validate_extraction, validate_iban,
    validate_national_id,
)
from ocr_pipeline.grammar import _rule_name  # noqa: E402
from ocr_pipeline.persian_text import MACHINE_POLICY, letterform_conformance  # noqa: E402
from ocr_pipeline.validation import (  # noqa: E402
    GuardrailConfig, extract_identifiers, per_field_confidence,
    validate_iranian_mobile,
)

FA8, FA0 = "۸", "۰"


# ===========================================================================
# 1. sampling / degeneracy
# ===========================================================================

def test_llamacpp_samplers_go_in_extra_body_not_top_level():
    """Passing these as top-level kwargs is silently dropped by the OpenAI SDK,
    which is the usual reason a repetition fix appears to do nothing."""
    kw = PROVEN.openai_kwargs()
    assert "repeat_penalty" not in kw
    assert kw["extra_body"]["repeat_penalty"] == 1.2


def test_proven_profile_matches_the_measured_threshold():
    """1.05 and 1.1 were measured to leave the loop intact; 1.2 removes it.
    The effect is a threshold, so lowering this silently reverts the fix."""
    assert PROVEN.repeat_penalty == 1.2


def test_neutral_top_p_is_not_transmitted():
    assert "top_p" not in PROVEN.openai_kwargs()
    assert RETRY.openai_kwargs()["top_p"] == 0.9


def test_dry_profile_sends_sequence_breakers():
    """Without breakers the sampler treats JSON's own punctuation as repetition
    and fights the output format."""
    extra = ACCURACY_PRESERVING.openai_kwargs()["extra_body"]
    assert extra["dry_multiplier"] == 0.8
    assert "\n" in extra["dry_sequence_breakers"]


def test_retry_profile_is_not_greedy():
    """A retry at temperature 0.0 reproduces the same loop byte for byte."""
    assert RETRY.temperature > 0.0


def test_detects_character_repetition_loop():
    r = detect_degeneracy('{"sender": "شرکت' + FA8 * 900)
    assert r.is_degenerate and r.kind == "char_loop"
    assert r.codepoint == "U+06F8"


def test_detects_phrase_repetition_loop():
    r = detect_degeneracy("متن " + "۱-۵۰۳۷" * 40)
    assert r.is_degenerate and r.kind == "phrase_loop"


def test_legitimate_iranian_identifier_is_not_degenerate():
    """An IBAN is 26 characters of digits and must not trip the detector."""
    assert not detect_degeneracy("IR650100060412075501234567").is_degenerate
    assert not detect_degeneracy("شماره ملی: ۱۲۳۴۵۶۷۸۹۰").is_degenerate


def test_repeated_administrative_formula_is_not_degenerate():
    """These letters repeat formulae by nature; flagging that would be a false
    positive on nearly every document."""
    text = "احتراماً به استحضار می‌رساند. " * 3 + "خواهشمند است اقدام فرمایید."
    assert not detect_degeneracy(text).is_degenerate


# ===========================================================================
# 2. normalization
# ===========================================================================

def test_arabic_letterforms_fold_to_persian():
    out = for_output("اداره كل فناورى اطلاعات")
    assert "ك" not in out and "ى" not in out
    assert "ک" in out and "ی" in out


def test_output_policy_converts_prose_digits_to_persian_indic():
    assert for_output("تعداد 18 نفر") == "تعداد ۱۸ نفر"


def test_iban_digits_are_never_rewritten():
    """The whole reason this module is context-aware: folding these digits
    destroys the IBAN."""
    src = "شماره شبا IR650100060412075501234567 است"
    assert "IR650100060412075501234567" in for_output(src)


def test_email_digits_are_never_rewritten():
    src = "ایمیل: Sabzgostar2006@gmail.com"
    assert "Sabzgostar2006@gmail.com" in for_output(src)


def test_url_digits_are_never_rewritten():
    assert "https://example.ir/a2b3" in for_output("سایت https://example.ir/a2b3 را ببینید")


def test_latin_reference_code_digits_are_never_rewritten():
    assert "INV-2024-0001" in for_output("شماره INV-2024-0001")


def test_machine_policy_gives_ascii_digits_for_checksums():
    assert for_machine("کد ملی ۰۰۱۲۳۴۵۶۷۸") == "کد ملی 0012345678"


def test_the_two_policies_move_digits_in_opposite_directions():
    """Conflating them silently corrupts data, so the difference is pinned."""
    src = "تلفن ۰۵۱۳۸۲۴۶۹۱"
    assert "۰۵۱۳۸۲۴۶۹۱" in for_output(src)
    assert "05138246 91".replace(" ", "") in for_machine(src).replace(" ", "")


def test_null_stays_null_and_is_not_coerced_to_empty_string():
    """A null field is data - "this document has no subject" - and collapsing it
    to "" makes omission unmeasurable."""
    assert normalize(None) is None


def test_harakat_and_tatweel_are_stripped():
    assert for_output("احتراماً" + "ــ") == "احتراما"


def test_arabic_indic_digits_also_normalize():
    assert for_machine("رقم ٥٧") == "رقم 57"


def test_normalization_is_idempotent():
    src = "اداره كلـ ۱۴۰۲ IR650100060412075501234567"
    once = for_output(src)
    assert for_output(once) == once


def test_preserve_policy_leaves_digits_alone():
    p = NormalizationPolicy(digits=DigitPolicy.PRESERVE)
    assert "18" in normalize("تعداد 18 نفر", p)


def test_letterform_conformance_measured_on_raw_text():
    rep = letterform_conformance(["اداره كل", "سلام"])
    assert rep["n_non_conforming"] == 1
    assert rep["pct_non_conforming"] == 50.0


def test_conformance_is_invisible_after_normalization():
    """Why conformance must be measured BEFORE normalizing: the fold erases the
    evidence, so CER-style metrics are blind to this defect."""
    assert letterform_conformance([for_output("اداره كل")])["n_non_conforming"] == 0


# ===========================================================================
# 3. grammar
# ===========================================================================

def test_rule_names_use_dashes_not_underscores():
    """Measured GBNF parser quirk: an underscore in a rule name fails to parse
    with the same opaque error as a genuine syntax error."""
    assert _rule_name("body_text") == "v-body-text"
    # Only RULE NAMES are constrained. The JSON key literals `body_text` and
    # `contact_info` legitimately contain underscores inside quoted strings, so
    # a blanket "no underscore anywhere" assertion is wrong.
    for line in DEFAULT_SPEC.build().splitlines():
        rule = line.split("::=")[0].strip()
        assert "_" not in rule, f"underscore in rule name: {rule!r}"


def test_grammar_has_no_backslash_inside_a_character_class():
    """Second parser quirk: a literal backslash in any class is rejected. The
    hex form \\x5C is how the backslash gets excluded instead."""
    for line in DEFAULT_SPEC.build().splitlines():
        for cls in __import__("re").findall(r"\[[^\]]*\]", line):
            assert "\\\\" not in cls, f"literal backslash in class: {cls}"


def test_grammar_excludes_backslash_so_output_is_always_valid_json():
    g = DEFAULT_SPEC.build()
    assert "\\x5C" in g, "backslash must be excluded or the model can emit a bad escape"


def test_every_contract_field_is_required_and_bounded():
    g = DEFAULT_SPEC.build()
    for name in ("sender", "receiver", "subject", "body_text", "contact_info"):
        assert f'\\"{name}\\":' in g
    assert g.count("{0,") == 5, "each field must be length-bounded"


def test_header_fields_are_bounded_far_below_body_text():
    """The bound is what makes collapse unrepresentable: a whole letter cannot
    fit in a 180-character sender."""
    caps = DEFAULT_SPEC.caps
    assert caps["sender"] < caps["body_text"] / 5


def test_token_budget_covers_the_caps():
    """A grammar the budget cannot satisfy truncates mid-object and defeats
    itself - measured on a page where the caps overran max_tokens."""
    assert DEFAULT_SPEC.recommended_max_tokens() > sum(DEFAULT_SPEC.caps.values())


# ===========================================================================
# 4. JSON repair
# ===========================================================================

def test_recovers_complete_fields_from_a_truncated_response():
    raw = ('{"sender": "شرکت خدماتی سبز گستر", "receiver": "بانک سامان", '
           '"subject": "ضمانتنامه", "body_text": "احتراما ' + FA8 * 900)
    obj, rep = repair_json(raw)
    assert obj["sender"] == "شرکت خدماتی سبز گستر"
    assert obj["subject"] == "ضمانتنامه"
    assert "body_text" not in obj, "a half-read field must be dropped, not guessed"
    assert rep.dropped_partial_field


def test_repair_never_invents_a_value():
    obj, _ = repair_json('{"sender": "الف", "subject": "نیم')
    assert set(obj) == {"sender"}


def test_repair_returns_nothing_when_no_field_completed():
    """The D31 collapse shape: nothing was ever closed, so nothing is salvageable
    and returning a guess would be worse than failing."""
    assert repair_json('{"sender": "شرکت' + FA0 * 500)[0] is None


def test_repair_leaves_valid_json_untouched():
    src = {"sender": "الف", "receiver": None, "subject": "ب",
           "body_text": "ج", "contact_info": None}
    assert repair_json(json.dumps(src, ensure_ascii=False))[0] == src


def test_brackets_inside_values_do_not_confuse_depth_counting():
    raw = '{"sender": "الف {نمونه} [ب]", "subject": "ج"}'
    assert repair_json(raw)[0]["sender"] == "الف {نمونه} [ب]"


# ===========================================================================
# 5. critical-field validation
# ===========================================================================

@pytest.mark.parametrize("nid", ["0499370899", "0790419904", "0084575948"])
def test_valid_national_ids_pass(nid):
    assert validate_national_id(nid)


@pytest.mark.parametrize("nid", ["0499370898", "1234567890", "123", ""])
def test_invalid_national_ids_fail(nid):
    assert not validate_national_id(nid)


def test_repdigit_national_ids_are_rejected():
    """They satisfy the checksum arithmetically but are never issued - and they
    are exactly what a model emits when it invents a number."""
    for d in "0123456789":
        assert not validate_national_id(d * 10)


def test_national_id_accepts_persian_indic_digits_via_machine_normalization():
    ids = extract_identifiers("کد ملی ۰۴۹۹۳۷۰۸۹۹ است")
    assert "0499370899" in ids["national_id"]


def test_valid_iranian_iban_passes():
    assert validate_iban("IR062960000000100324200001")


def test_iban_with_one_digit_changed_fails():
    assert not validate_iban("IR062960000000100324200002")


def test_iban_rejects_wrong_length_and_country():
    assert not validate_iban("IR06296000000010032420000")
    assert not validate_iban("DE89370400440532013000")


def test_iranian_mobile_formats():
    for good in ["09123456789", "+989123456789", "00989123456789", "9123456789"]:
        assert validate_iranian_mobile(good), good
    for bad in ["08123456789", "0912345678", "12345678901"]:
        assert not validate_iranian_mobile(bad), bad


# ===========================================================================
# 6. anti-hallucination guardrails
# ===========================================================================

GOOD = {
    "sender": "شرکت خدماتی سبز گستر",
    "receiver": "ریاست محترم بانک سامان",
    "subject": "درخواست صدور ضمانتنامه",
    "body_text": "با سلام؛ احتراماً به استحضار می‌رساند خواهشمند است اقدام فرمایید.",
    "contact_info": "مشهد، بلوار کوثر، پلاک ۷",
}


def test_clean_extraction_is_accepted():
    assert validate_extraction(GOOD).verdict is Verdict.ACCEPT


def test_assistant_boilerplate_is_rejected():
    """The confirmed real fabrication: fluent Persian in a chat-assistant
    register, returned as a clean 200 for a bank-guarantee letter."""
    bad = dict(GOOD, body_text=(
        "با توجه به اطلاعاتی که در سامانه اطلاعاتی دریافت کرده‌ام، "
        "می‌توانم به شما اطلاعاتی ارائه دهم که در زیر آورده شده است:"
    ))
    r = validate_extraction(bad)
    assert r.verdict is Verdict.REJECT
    assert any(f.code == "assistant_boilerplate" for f in r.findings)


def test_english_assistant_boilerplate_is_also_caught():
    r = validate_extraction(dict(GOOD, body_text="As an AI, I cannot read this."))
    assert r.verdict is Verdict.REJECT


def test_all_null_extraction_is_rejected():
    r = validate_extraction({k: None for k in GOOD})
    assert r.verdict is Verdict.REJECT
    assert any(f.code == "empty_extraction" for f in r.findings)


def test_field_collapse_is_flagged():
    """The D31 shape: the whole letter written into `sender`.

    The filler text is varied deliberately. An earlier version of this test
    repeated one phrase 60 times, which the degeneracy detector correctly
    flagged as a repetition loop - masking the collapse finding under a REJECT.
    """
    long_letter = " ".join(
        f"بند شماره {i} از قرارداد منعقده در خصوص تامین نیروی انسانی"
        for i in range(1, 16)
    )
    r = validate_extraction(dict(GOOD, sender="شرکت سبز گستر " + long_letter))
    assert r.verdict is Verdict.REVIEW
    assert any(f.code == "field_collapse" for f in r.findings)


def test_numbered_list_is_not_mistaken_for_a_repetition_loop():
    """False-positive guard: real letters enumerate people and line items. Only
    VERBATIM repetition is degeneracy; a varying enumeration is not."""
    listing = "\n".join(f"{i}- نام و نام خانوادگی شماره {i}" for i in range(1, 20))
    assert not detect_degeneracy(listing).is_degenerate
    assert validate_extraction(dict(GOOD, body_text=listing)).verdict is not Verdict.REJECT


def test_cross_field_duplication_is_flagged():
    dup = "درخواست صدور ضمانتنامه بانکی برای شرکت سبز گستر"
    r = validate_extraction(dict(GOOD, subject=dup, body_text=dup))
    assert any(f.code == "cross_field_duplication" for f in r.findings)


def test_missing_document_anchor_is_flagged():
    r = validate_extraction({
        "sender": "چیزی", "receiver": "کسی", "subject": "مطلبی",
        "body_text": "یک متن طولانی بدون هیچ عبارت اداری استاندارد در آن.",
        "contact_info": None,
    })
    assert any(f.code == "no_document_anchor" for f in r.findings)


def test_degenerate_run_inside_a_parsed_field_is_rejected():
    """Worse than a 422: it parses, validates, and ships corrupted text."""
    r = validate_extraction(dict(GOOD, body_text="مبلغ " + FA0 * 400))
    assert r.verdict is Verdict.REJECT
    assert any(f.code == "degenerate_run" for f in r.findings)


def test_bad_national_id_routes_to_review():
    r = validate_extraction(dict(GOOD, body_text="با کد ملی 1234567890 معرفی می‌گردد"))
    assert r.needs_review
    assert any(f.code == "identifier_checksum_failed" for f in r.findings)


def test_valid_identifier_does_not_trigger_review():
    r = validate_extraction(dict(GOOD, body_text=(
        "احتراماً آقای الف با کد ملی 0499370899 معرفی می‌گردد. "
        "شماره شبا IR062960000000100324200001 است."
    )))
    assert r.verdict is Verdict.ACCEPT


def test_identifier_failure_can_be_escalated_to_critical():
    cfg = GuardrailConfig(identifier_failure_is_critical=True)
    r = validate_extraction(dict(GOOD, body_text="کد ملی 1234567890"), config=cfg)
    assert r.verdict is Verdict.REJECT


def test_placeholder_value_is_flagged():
    r = validate_extraction(dict(GOOD, sender="string"))
    assert any(f.code == "placeholder_value" for f in r.findings)


def test_confidence_falls_as_findings_accumulate():
    clean = validate_extraction(GOOD).confidence
    dirty = validate_extraction(dict(GOOD, sender="string",
                                     body_text="کد ملی 1234567890")).confidence
    assert clean == 1.0 and dirty < clean


def test_per_field_confidence_penalises_the_named_field():
    data = dict(GOOD, sender="شرکت " + "متن نامه " * 60)
    r = validate_extraction(data)
    scores = per_field_confidence(data, r)
    assert scores["sender"] < scores["receiver"]


def test_field_holding_a_failed_identifier_is_floored():
    data = dict(GOOD, body_text="کد ملی 1234567890 معرفی می‌گردد")
    scores = per_field_confidence(data, validate_extraction(data))
    assert scores["body_text"] <= 0.3


# ===========================================================================
# 7. degeneracy false-positive guards
#
# Each case below rejected a VALID document during end-to-end testing before the
# furniture exclusion landed. They are regression tests, not hypotheticals.
# ===========================================================================

def test_table_separator_is_not_degeneracy():
    """A financial table in a real letter was rejected over its rule line."""
    table = "| مبلغ |\n|" + "-" * 30 + "|\n| ۱۳,۸۹۷,۰۰۴ | جمع کل |"
    assert not detect_degeneracy(table).is_degenerate


def test_dotted_fill_line_is_not_degeneracy():
    """Form fill lines are ubiquitous in Iranian administrative stationery."""
    assert not detect_degeneracy("نام: " + "." * 30 + " تاریخ: " + "." * 20).is_degenerate


def test_signature_underline_is_not_degeneracy():
    assert not detect_degeneracy("امضا " + "_" * 40).is_degenerate


def test_repeated_illegibility_marker_gets_its_own_kind():
    """Distinct condition from a decoding loop: the page could not be read, so
    the remedy is a rescan, not a sampler change."""
    r = detect_degeneracy("[ناخوانا] " * 8)
    assert r.is_degenerate and r.kind == "illegible_loop"


def test_grammar_bounded_output_can_still_contain_a_loop():
    """Measured end-to-end: with the grammar active, output parsed with
    finish=stop and still carried `۸۸/۸۸` repeated 800 characters inside
    body_text. The grammar bounds the FIELD; it does not stop the loop. This is
    why the sampler and this detector are both still required."""
    bounded = '{"sender": "شرکت الف", "body_text": "' + "۸۸/۸۸" * 100 + '"}'
    import json as _json
    parsed = _json.loads(bounded)
    assert detect_degeneracy(parsed["body_text"]).is_degenerate
