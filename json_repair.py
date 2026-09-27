"""Salvage a truncated JSON object (Phase 2a fix (ii)).

Every observed 422 on the real corpus is the same shape: the model entered a
repetition loop, decoding stopped at the token cap, and the response ends inside
an unterminated string. The prefix before the loop is usually good -- often three
or four fields are complete -- so discarding the whole response wastes a mostly
correct extraction.

This module closes the open string and the open bracket nesting, and drops the
trailing repetition, producing the best valid object the prefix supports.

It is deliberately conservative:
  * only ever runs AFTER a strict `json.loads` has failed, so a valid response is
    never touched;
  * never invents a value -- a field that was cut off is dropped, not guessed;
  * strips a detected degenerate run rather than keeping corrupted characters;
  * reports what it did, so a repaired extraction can be flagged low-confidence
    instead of being passed off as a clean read.
"""
from __future__ import annotations

import json
import re
from typing import Any

# A run this long is degeneracy, not content. Persian identifiers (national ID,
# IBAN, phone) top out around 26 characters; 20 identical characters in a row is
# not a real value.
DEGENERATE_RUN = 20
_DEGENERATE_TAIL = re.compile(r"(.)\1{%d,}$" % (DEGENERATE_RUN - 1))
# A short group repeated to the end -- the phrase-level form of the same loop.
_PHRASE_TAIL = re.compile(r"(.{2,40}?)\1{3,}$")


def strip_degenerate_tail(s: str) -> tuple[str, bool]:
    """Remove a trailing repetition loop. Returns (text, whether it fired)."""
    out = _DEGENERATE_TAIL.sub("", s)
    fired = out != s
    out2 = _PHRASE_TAIL.sub("", out)
    return out2, fired or out2 != out


def repair(raw: str) -> tuple[dict[str, Any] | None, dict]:
    """Try to recover an object from truncated JSON.

    Returns `(obj, report)`. `obj` is None when nothing usable could be salvaged.
    `report` records what was done so the caller can mark the result.
    """
    report = {
        "attempted": True,
        "degenerate_tail_removed": False,
        "closed_string": False,
        "closed_brackets": 0,
        "dropped_partial_field": False,
        "recovered_fields": [],
    }
    if not raw:
        return None, {**report, "attempted": False}

    s = raw.strip()

    # 1. Drop the repetition loop at the end, if there is one.
    s, fired = strip_degenerate_tail(s)
    report["degenerate_tail_removed"] = fired

    # 2. Walk the text tracking string state and bracket depth, so we know what
    #    is still open. Counting brackets naively would miscount any that appear
    #    inside string values -- and Persian letters routinely sit beside them.
    stack: list[str] = []
    in_string = False
    escaped = False
    last_safe = None          # index just after the last completed key/value pair
    for i, ch in enumerate(s):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
        elif ch == "," and len(stack) == 1:
            # A comma at depth 1 ends a complete top-level field.
            last_safe = i

    # 3. If we ended inside a string, that field was cut off mid-value. Rewind to
    #    the last complete field rather than closing the quote on a partial value
    #    -- a half-read field is worse than a missing one, because it looks real.
    if in_string:
        report["closed_string"] = True
        if last_safe is not None:
            s = s[:last_safe]
            report["dropped_partial_field"] = True
        else:
            return None, report   # not even one complete field survived

    # 4. Close whatever nesting is still open.
    s = s.rstrip().rstrip(",")
    while stack:
        s += stack.pop()
        report["closed_brackets"] += 1

    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None, report

    if not isinstance(obj, dict):
        return None, report

    report["recovered_fields"] = sorted(k for k, v in obj.items() if v)
    return obj, report
