"""Decoding parameters that eliminate repetition degeneracy.

BACKEND NOTE
------------
This targets llama.cpp `llama-server` reached over its OpenAI-compatible HTTP
API. llama.cpp-specific samplers are NOT top-level OpenAI parameters; they must
travel in `extra_body`, which the OpenAI SDK forwards verbatim. Passing them as
keyword arguments silently drops them, which is the usual reason a "fix" appears
to do nothing.

WHAT THE FAILURE ACTUALLY IS
----------------------------
Two distinct degeneracies produce the same 422:

  * character loop   - one Persian-Indic digit emitted 900+ times mid-string
  * phrase loop      - a short digit group ("۱-۵۰۳۷") repeated to the token cap

Both end `finish_reason=length` with an unterminated JSON string. Neither is
token-budget exhaustion: raising `max_tokens` was measured twice and moved
nothing, because every extra token goes into a longer loop.

SAMPLER CHOICE
--------------
`repeat_penalty` is a blunt instrument: it penalises *every* previously seen
token, including the fixed administrative formulas these letters are built from
(`احتراماً`, `خواهشمند است`, `به استحضار می‌رساند`). It fixes the loop at the cost
of suppressing legitimate repetition.

The DRY sampler is the correct tool. It penalises only *verbatim n-gram*
continuation, and only past `allowed_length` tokens of repeat, so a formula that
appears twice is untouched while a 900-character run is crushed.

EVIDENCE LEVELS - these differ, and the difference matters:

  * `repeat_penalty=1.2`  measured across the full corpus: real dev usable-output
    14.81% -> 96.30% -> 100% with repair and retry; held-out test 97.22%.
    1.05 and 1.1 do nothing. The effect is a THRESHOLD, not a gradient.
  * DRY                   measured on a single known-degenerate page: clean
    termination, longest character run 2. Not yet A/B'd at corpus scale.

WHICH PROFILE TO USE
--------------------
The right choice depends on whether the GBNF grammar is active, and the two
answers are opposite:

  * grammar OFF -> PROVEN (repeat_penalty 1.2)
  * grammar ON  -> BELT_AND_BRACES (DRY + repeat_penalty 1.1)

A grammar forbids EOS until the JSON object closes, so a model that wants to stop
mid-field pads with repetition instead. `repeat_penalty` alone does not stop that;
DRY does. Measured on 10 real letters, grammar + repeat_penalty 1.2 was degenerate
on 7 of 10, while grammar + DRY + repeat 1.1 was degenerate on 0 of 10 and
extracted more fields. Full table on BELT_AND_BRACES below.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any

# A run this long is degeneracy, not transcription. Iranian identifiers top out
# around 26 characters (IBAN); 20 identical alphanumerics in a row is never real.
DEGENERATE_CHAR_RUN = 20

# Document furniture, NOT degeneracy: table rules, dotted fill lines, underlines,
# box drawing. Scanned administrative letters are full of these. An earlier
# version of this detector rejected a valid financial table because it contained
# a 25-dash separator, and a form because it had a dotted fill line. Excluded
# outright rather than given a higher threshold - their length carries no signal.
_FURNITURE = ".-_=*~+#|/\\ \t·•–—─━═"
_FURNITURE_CLASS = re.escape(_FURNITURE)

_CHAR_RUN_RE = re.compile(r"([^%s\s])\1{%d,}" % (_FURNITURE_CLASS, DEGENERATE_CHAR_RUN - 1))

# The phrase-level form: a short group repeated to the end of the output. The
# lookahead requires at least one word character in the group, so a run of
# "| --- | --- |" table scaffolding does not qualify as a loop.
_PHRASE_RUN_RE = re.compile(r"((?=[^\n]*\w)[^\n]{2,40}?)\1{3,}")

# The model repeating its own illegibility marker is a DIFFERENT condition from a
# decoding loop: the page could not be read, sampling did not degenerate. It gets
# its own kind because the remedy differs - rescan or upscale, not retune.
_ILLEGIBLE_RE = re.compile(r"(\[\s*ناخوان[اa]?\s*\]\s*){3,}")


@dataclass(frozen=True)
class SamplingProfile:
    """One decoding configuration.

    `openai_kwargs()` splits the fields into the keyword arguments the OpenAI SDK
    accepts natively and the `extra_body` payload llama.cpp reads.
    """

    name: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048

    # llama.cpp samplers -> extra_body
    repeat_penalty: float = 1.0      # 1.0 = off
    repeat_last_n: int = 256         # window the penalty looks back over
    presence_penalty: float = 0.0    # measured ineffective against this failure
    frequency_penalty: float = 0.0
    dry_multiplier: float = 0.0      # 0.0 = off
    dry_base: float = 1.75
    dry_allowed_length: int = 4      # n-grams shorter than this are never penalised
    dry_penalty_last_n: int = -1     # -1 = whole context
    dry_sequence_breakers: tuple[str, ...] = ("\n", ":", "\"", "،", "؛", "*")

    def openai_kwargs(self) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if self.repeat_penalty != 1.0:
            extra["repeat_penalty"] = self.repeat_penalty
            extra["repeat_last_n"] = self.repeat_last_n
        if self.dry_multiplier:
            extra.update(
                dry_multiplier=self.dry_multiplier,
                dry_base=self.dry_base,
                dry_allowed_length=self.dry_allowed_length,
                dry_penalty_last_n=self.dry_penalty_last_n,
                # Breakers reset the n-gram window. Without them, the structural
                # punctuation of JSON itself looks like repetition and the
                # sampler fights the output format.
                dry_sequence_breakers=list(self.dry_sequence_breakers),
            )
        if self.frequency_penalty:
            extra["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty:
            extra["presence_penalty"] = self.presence_penalty

        kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # Only send top_p when it is doing something. Sending the neutral value
        # makes logged configs ambiguous about what was actually transmitted.
        if self.top_p != 1.0:
            kwargs["top_p"] = self.top_p
        if extra:
            kwargs["extra_body"] = extra
        return kwargs

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dry_sequence_breakers"] = list(self.dry_sequence_breakers)
        return d


# The measured, corpus-validated default. Use this in production today.
PROVEN = SamplingProfile(
    name="proven.repeat_penalty_1.2",
    temperature=0.0,
    repeat_penalty=1.2,
    repeat_last_n=256,
)

# Reasoned to preserve legitimate formula repetition. Validate before adopting.
ACCURACY_PRESERVING = SamplingProfile(
    name="dry.multiplier_0.8",
    temperature=0.0,
    dry_multiplier=0.8,
    dry_base=1.75,
    dry_allowed_length=4,
)

# RECOMMENDED WHEN THE GBNF GRAMMAR IS ACTIVE.
#
# A grammar forbids EOS until the object closes. When the model wants to stop
# mid-`body_text` it therefore cannot, and pads with repetition instead. Measured
# on 10 real letters:
#
#   config                              parsed   avg fields   degenerate
#   grammar + repeat_penalty 1.2        10/10       3.8         7/10
#   grammar + DRY                       10/10       3.8         1/10
#   grammar + DRY + repeat_penalty 1.1  10/10       4.0         0/10   <-- this
#   no grammar + repeat_penalty 1.2      9/10       2.6         0/10
#
# `repeat_penalty=1.2`, which is the right choice WITHOUT a grammar, is the worst
# choice WITH one: it does nothing about grammar-induced padding. DRY does,
# because that padding is verbatim n-gram repetition, which is exactly what DRY
# targets. The mild 1.1 penalty mops up what DRY leaves.
BELT_AND_BRACES = SamplingProfile(
    name="dry+repeat_1.1",
    temperature=0.0,
    dry_multiplier=0.8,
    repeat_penalty=1.1,
    repeat_last_n=128,
)

# Retry profile. A retry at temperature 0.0 with identical settings reproduces
# the same loop deterministically - it must perturb sampling to be worth its
# latency. Kept just off greedy so the output stays close to the primary read.
RETRY = SamplingProfile(
    name="retry.perturbed",
    temperature=0.2,
    top_p=0.9,
    repeat_penalty=1.15,
    dry_multiplier=0.8,
)

PROFILES = {p.name: p for p in (PROVEN, ACCURACY_PRESERVING, BELT_AND_BRACES, RETRY)}


# --------------------------------------------------------------------------
# runtime degeneracy detection
# --------------------------------------------------------------------------

@dataclass
class DegeneracyReport:
    is_degenerate: bool
    kind: str | None = None          # char_loop | phrase_loop
    repeated: str | None = None
    run_length: int = 0
    codepoint: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_degeneracy(text: str | None) -> DegeneracyReport:
    """Detect a repetition loop in raw model output.

    Sampler settings reduce the failure rate; they do not guarantee zero. This is
    the second line of defence - it catches a loop that survived decoding so the
    response can be rejected or retried instead of shipped. Run it on output that
    PARSED as well as output that did not: a loop inside a well-formed JSON string
    is more dangerous than a 422, because it passes validation.
    """
    if not text:
        return DegeneracyReport(False)

    m = _ILLEGIBLE_RE.search(text)
    if m:
        return DegeneracyReport(True, "illegible_loop", "[ناخوانا]", len(m.group(0)))

    m = _CHAR_RUN_RE.search(text)
    if m:
        ch = m.group(1)
        return DegeneracyReport(
            True, "char_loop", ch, len(m.group(0)), f"U+{ord(ch):04X}"
        )

    # Only inspect the tail: a legitimate document can repeat a short string
    # early on, but a loop runs to the end of the output.
    tail = text[-800:]
    m = _PHRASE_RUN_RE.search(tail)
    if m and len(m.group(0)) >= 60:
        return DegeneracyReport(True, "phrase_loop", m.group(1), len(m.group(0)))

    return DegeneracyReport(False)


def strip_degenerate_tail(text: str) -> tuple[str, bool]:
    """Remove a trailing repetition loop, preserving everything before it."""
    if not text:
        return text, False
    out = _CHAR_RUN_RE.sub("", text)
    out2 = _PHRASE_RUN_RE.sub(lambda m: m.group(1), out)
    return out2, out2 != text
