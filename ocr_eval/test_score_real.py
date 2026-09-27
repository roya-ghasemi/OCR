# -*- coding: utf-8 -*-
"""Guards for score_real.py — each check has a fixture that must make it fail."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_real import classify_failure  # noqa: E402

RAW = {"sender": "x", "receiver": None, "subject": None, "body_text": None, "contact_info": None}


def test_arm_a_row_shape_is_usable_only_on_200_with_content():
    assert classify_failure({"http_status": 200, "raw": RAW}) == (True, None)
    assert classify_failure({"http_status": 422, "raw": None}) == (False, "http_422")
    assert classify_failure({"http_status": 200, "raw": {k: None for k in RAW}}) == (False, "all_null_200")


def test_arm_b_row_shape_is_recognised_not_misread_as_http_none():
    # Before the fix an ocr_pipeline row scored as failure "http_None" and arm B
    # could not be scored at all. That is a silent zero, the worst kind.
    assert classify_failure({"ok": True, "raw": RAW}) == (True, None)
    assert classify_failure({"ok": False, "raw": {}}) == (False, "pipeline_rejected")


def test_unknown_row_shape_is_a_failure_not_a_pass():
    assert classify_failure({"raw": RAW}) == (False, "unknown_row_shape")


def test_transport_error_wins_regardless_of_shape():
    assert classify_failure({"ok": True, "raw": RAW, "error": "boom"}) == (False, "transport")
    assert classify_failure({"http_status": 200, "raw": RAW, "error": "boom"}) == (False, "transport")
