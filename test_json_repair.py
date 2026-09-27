"""Tests for json_repair -- built from the actual failure shapes in debug_raw/.

Run: venv312\\Scripts\\python.exe -m pytest test_json_repair.py -q
"""
import json

import pytest

from json_repair import DEGENERATE_RUN, repair, strip_degenerate_tail

FA8 = "۸"   # Persian-Indic 8, the character most often looped
FA0 = "۰"


def test_strips_single_character_degenerate_tail():
    s = '{"sender": "شرکت الف' + FA8 * 900
    out, fired = strip_degenerate_tail(s)
    assert fired
    assert FA8 * DEGENERATE_RUN not in out


def test_strips_phrase_level_degenerate_tail():
    # The observed phrase loop: a short digit group repeated to the cap.
    s = '{"sender": "الف", "subject": "ب' + ("۱-۵۰۳۷" * 40)
    out, fired = strip_degenerate_tail(s)
    assert fired
    assert len(out) < len(s)


def test_does_not_strip_a_legitimate_identifier():
    """A real IBAN or national ID must survive untouched."""
    s = '{"contact_info": "IR650100060412075501234567"}'
    out, fired = strip_degenerate_tail(s)
    assert not fired and out == s


def test_recovers_complete_fields_from_a_truncated_response():
    """The dominant real shape: three good fields, then a loop inside the fourth."""
    raw = ('{"sender": "شرکت خدماتی سبز گستر", '
           '"receiver": "بانک سامان", '
           '"subject": "ضمانتنامه", '
           '"body_text": "احتراما ' + FA8 * 900)
    obj, rep = repair(raw)
    assert obj is not None
    assert obj["sender"] == "شرکت خدماتی سبز گستر"
    assert obj["receiver"] == "بانک سامان"
    assert obj["subject"] == "ضمانتنامه"
    assert "body_text" not in obj, "a half-read field must be dropped, not guessed"
    assert rep["dropped_partial_field"] is True
    assert rep["closed_brackets"] >= 1


def test_never_invents_a_value():
    raw = '{"sender": "الف", "subject": "نیم'
    obj, rep = repair(raw)
    assert obj is not None
    assert set(obj) == {"sender"}
    assert "نیم" not in json.dumps(obj, ensure_ascii=False)


def test_returns_none_when_not_even_one_field_completed():
    raw = '{"sender": "شرکت' + FA0 * 500
    obj, rep = repair(raw)
    assert obj is None


def test_brackets_inside_string_values_do_not_confuse_the_depth_count():
    raw = '{"sender": "الف {نمونه} [ب]", "subject": "ج"}'
    obj, _ = repair(raw)
    assert obj == {"sender": "الف {نمونه} [ب]", "subject": "ج"}


def test_escaped_quote_inside_a_value_is_handled():
    raw = '{"sender": "شرکت \\"الف\\"", "subject": "ب'
    obj, rep = repair(raw)
    assert obj is not None
    assert obj["sender"] == 'شرکت "الف"'


def test_valid_json_passes_through_unchanged():
    src = {"sender": "الف", "receiver": None, "subject": "ب",
           "body_text": "ج", "contact_info": None}
    obj, _ = repair(json.dumps(src, ensure_ascii=False))
    assert obj == src


def test_report_lists_the_recovered_fields():
    raw = '{"sender": "الف", "subject": "ب", "body_text": "ج' + FA8 * 400
    obj, rep = repair(raw)
    assert rep["recovered_fields"] == ["sender", "subject"]
    assert rep["degenerate_tail_removed"] is True


def test_empty_input_is_not_attempted():
    obj, rep = repair("")
    assert obj is None and rep["attempted"] is False


@pytest.mark.parametrize("digit", ["۰", "۲", "۵", "۸", "۹"])
def test_every_observed_looping_digit_is_handled(digit):
    raw = '{"sender": "الف", "subject": "ب' + digit * 800
    obj, _ = repair(raw)
    assert obj is not None and obj["sender"] == "الف"
