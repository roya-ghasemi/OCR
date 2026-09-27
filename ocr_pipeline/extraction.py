"""Extraction orchestrator: grammar-constrained decoding, repair, retry, guardrails.

DEFENCE IN DEPTH, IN THIS ORDER
-------------------------------
  1. GBNF grammar        - makes malformed output and field collapse
                           structurally unrepresentable
  2. repetition sampler  - kills the degeneracy the grammar cannot see
  3. strict JSON parse   - the happy path
  4. JSON repair         - salvages the complete fields from a truncated response
  5. bounded retry       - perturbed sampling; a greedy retry reproduces the
                           same failure byte for byte
  6. normalization       - canonical Persian letterforms, context-aware digits
  7. guardrails          - checksums and structural anomaly detection

Layers 1 and 2 reduce how often the later ones fire; they do not make them
redundant. Measured: with the grammar, six pages that previously needed repair
and a retry parsed on the first attempt - including the two that defeated every
earlier fix.

WHAT THE GRAMMAR DOES NOT DO
----------------------------
It guarantees SHAPE, not CORRECTNESS. Forcing all five fields makes the model
fill all five; it does not make it put the right text in each. Observed:
`contact_info` receiving body text rather than the footer block. That is what the
prompt's per-field descriptions and the duplication guardrail are for.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .grammar import DEFAULT_SPEC, FIELD_ORDER, GrammarSpec, SYSTEM_PROMPT, USER_TURN
from .persian_text import (
    NormalizationPolicy, OUTPUT_POLICY, normalize_extraction, audit,
)
from .sampling import (
    BELT_AND_BRACES, PROVEN, RETRY, SamplingProfile, detect_degeneracy,
    strip_degenerate_tail,
)
from .validation import (
    DEFAULT_GUARDRAILS, GuardrailConfig, ValidationResult, Verdict,
    per_field_confidence, validate_extraction,
)

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", re.IGNORECASE)
_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def clean_json_text(raw: str) -> str:
    """Strip markdown fences and any prose the model wrapped around the object.

    Still needed with a grammar in place: a grammar can be rejected at request
    time (a parse error, an engine without support) and the pipeline falls back
    to unconstrained decoding, which fences its output about a third of the time.
    """
    s = (raw or "").strip()
    m = _FENCE_RE.match(s)
    if m:
        s = m.group(1).strip()
    else:
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s).strip()
    if not s.startswith("{"):
        m = _OBJECT_RE.search(s)
        if m:
            s = m.group(0)
    return s


# ---------------------------------------------------------------------------
# JSON repair
# ---------------------------------------------------------------------------

@dataclass
class RepairReport:
    attempted: bool = False
    succeeded: bool = False
    degenerate_tail_removed: bool = False
    dropped_partial_field: bool = False
    recovered_fields: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "degenerate_tail_removed": self.degenerate_tail_removed,
            "dropped_partial_field": self.dropped_partial_field,
            "recovered_fields": list(self.recovered_fields),
        }


def repair_json(raw: str) -> tuple[dict[str, Any] | None, RepairReport]:
    """Recover an object from truncated JSON.

    Conservative by design:
      * runs only AFTER a strict parse has failed, so valid output is untouched
      * NEVER invents a value - a field cut off mid-write is dropped, because a
        half-read value looks real and is worse than a missing one
      * strips a detected repetition loop rather than keeping corrupted text
    """
    report = RepairReport(attempted=True)
    if not raw:
        return None, RepairReport(attempted=False)

    s = raw.strip()
    s, fired = strip_degenerate_tail(s)
    report.degenerate_tail_removed = fired

    # Walk the text tracking string state and bracket depth. Counting brackets
    # naively miscounts any that appear inside a value, and Persian business
    # letters are full of parenthesised and bracketed text.
    stack: list[str] = []
    in_string = escaped = False
    last_field_end: int | None = None
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
            last_field_end = i

    if in_string:
        # Truncated mid-value. Rewind to the last COMPLETE field rather than
        # closing the quote on a partial one.
        if last_field_end is None:
            return None, report
        s = s[:last_field_end]
        report.dropped_partial_field = True

    s = s.rstrip().rstrip(",")
    while stack:
        s += stack.pop()

    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None, report
    if not isinstance(obj, dict):
        return None, report

    report.succeeded = True
    report.recovered_fields = sorted(k for k, v in obj.items() if v)
    return obj, report


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    ok: bool
    data: dict[str, str | None] = field(default_factory=dict)
    validation: ValidationResult | None = None
    field_confidence: dict[str, float] = field(default_factory=dict)
    attempts: int = 0
    repaired: bool = False
    partial: bool = False
    grammar_used: bool = False
    latency_s: float = 0.0
    finish_reason: str | None = None
    error: str | None = None
    raw_len: int = 0
    letterform_audit: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "extraction": self.data,
            "meta": {
                "attempts": self.attempts,
                "repaired": self.repaired,
                "partial": self.partial,
                "grammar_used": self.grammar_used,
                "latency_s": round(self.latency_s, 3),
                "finish_reason": self.finish_reason,
                "raw_len": self.raw_len,
            },
            "quality": (self.validation.as_dict() if self.validation else None),
            "field_confidence": self.field_confidence,
        }


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

class LetterExtractor:
    """Drop-in extraction pipeline.

    `client` is an `openai.AsyncOpenAI` (or anything with the same
    `chat.completions.create`) pointed at llama.cpp's OpenAI-compatible server.
    """

    def __init__(
        self,
        client,
        model: str,
        *,
        grammar_spec: GrammarSpec | None = DEFAULT_SPEC,
        # Defaults to the grammar-appropriate profile because grammar_spec
        # defaults to on. With a grammar, repeat_penalty alone leaves 7 of 10
        # documents degenerate; DRY plus a mild penalty leaves 0. If you pass
        # `grammar_spec=None`, pass `sampling=PROVEN` with it.
        sampling: SamplingProfile = BELT_AND_BRACES,
        retry_sampling: SamplingProfile = RETRY,
        max_retries: int = 1,
        normalization: NormalizationPolicy = OUTPUT_POLICY,
        guardrails: GuardrailConfig = DEFAULT_GUARDRAILS,
        system_prompt: str = SYSTEM_PROMPT,
        user_turn: str = USER_TURN,
        # Engine-level request options merged into `extra_body` on every call.
        # `cache_prompt` lives here: llama.cpp defaults it to true, and a warm
        # KV cache makes greedy decoding session-dependent in the long tail
        # (D52). Declaring it in config is not enough — it has to be SENT.
        extra_body: dict[str, Any] | None = None,
        on_raw_failure: Callable[[str, str], None] | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.grammar_spec = grammar_spec
        self.sampling = sampling
        self.retry_sampling = retry_sampling
        self.max_retries = max_retries
        self.normalization = normalization
        self.guardrails = guardrails
        self.system_prompt = system_prompt
        self.user_turn = user_turn
        self.extra_body = dict(extra_body or {})
        self.on_raw_failure = on_raw_failure
        # Set once a grammar is rejected by the engine, so we degrade to
        # prompt-plus-repair for the rest of the process instead of paying a
        # guaranteed 400 on every subsequent request.
        self._grammar_disabled = False

    # -- request construction ------------------------------------------------

    def _messages(self, image_data_url: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.user_turn},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ]

    def _kwargs(self, profile: SamplingProfile, use_grammar: bool) -> dict[str, Any]:
        kwargs = profile.openai_kwargs()
        if self.extra_body:
            kwargs["extra_body"] = {**dict(kwargs.get("extra_body") or {}), **self.extra_body}
        if use_grammar and self.grammar_spec and not self._grammar_disabled:
            extra = dict(kwargs.get("extra_body") or {})
            # llama.cpp-native GBNF. NOT `response_format`, which this engine
            # accepts and then silently ignores on multimodal requests.
            extra["grammar"] = self.grammar_spec.build()
            kwargs["extra_body"] = extra
            # A grammar whose caps exceed the token budget truncates mid-object
            # and defeats itself. Raise the budget to fit the caps.
            kwargs["max_tokens"] = max(
                kwargs.get("max_tokens", 0),
                self.grammar_spec.recommended_max_tokens(),
            )
        return kwargs

    async def _call(self, image_data_url: str, profile: SamplingProfile, use_grammar: bool):
        return await self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(image_data_url),
            **self._kwargs(profile, use_grammar),
        )

    # -- main entry point ----------------------------------------------------

    async def extract(self, image_data_url: str, doc_id: str = "") -> ExtractionResult:
        started = time.perf_counter()
        attempts = 0
        raw = ""
        finish_reason: str | None = None
        data: dict[str, Any] | None = None
        repaired = partial = False
        grammar_used = bool(self.grammar_spec) and not self._grammar_disabled
        last_error: str | None = None

        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            profile = self.sampling if attempt == 0 else self.retry_sampling
            use_grammar = bool(self.grammar_spec) and not self._grammar_disabled

            try:
                response = await self._call(image_data_url, profile, use_grammar)
            except Exception as exc:
                msg = str(exc)
                # A grammar the engine will not compile is a permanent condition,
                # not a transient one: disable it and retry unconstrained rather
                # than failing the request.
                if "grammar" in msg.lower() and use_grammar:
                    logger.error(
                        "Engine rejected the GBNF grammar for '%s'; falling back to "
                        "prompt + repair for the rest of this process: %s",
                        doc_id, msg[:200],
                    )
                    self._grammar_disabled = True
                    grammar_used = False
                    continue
                last_error = f"{type(exc).__name__}: {msg}"
                logger.exception("Inference error for '%s' (attempt %d)", doc_id, attempts)
                continue

            choice = response.choices[0]
            raw = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)
            cleaned = clean_json_text(raw)

            try:
                data = json.loads(cleaned)
                if isinstance(data, dict):
                    break
                data = None
            except json.JSONDecodeError as exc:
                last_error = str(exc)
                if self.on_raw_failure:
                    self.on_raw_failure(doc_id, raw)
                candidate, report = repair_json(cleaned)
                if candidate is not None:
                    data, repaired = candidate, True
                    partial = report.dropped_partial_field
                    logger.warning(
                        "Recovered '%s' by JSON repair: fields=%s partial=%s",
                        doc_id, report.recovered_fields, partial,
                    )
                    break

            if attempt < self.max_retries:
                logger.warning(
                    "Unusable output for '%s' on attempt %d; retrying with %s",
                    doc_id, attempts, self.retry_sampling.name,
                )

        latency = time.perf_counter() - started

        if data is None:
            return ExtractionResult(
                ok=False, attempts=attempts, latency_s=latency,
                finish_reason=finish_reason, grammar_used=grammar_used,
                raw_len=len(raw),
                error=f"Model produced no parseable object. Last error: {last_error}",
            )

        # Normalize to the five contract fields. An extra key the model invented
        # is dropped; a missing one becomes an explicit null.
        contract = {k: data.get(k) for k in FIELD_ORDER}
        contract = {
            k: (v if isinstance(v, str) and v.strip() else None)
            for k, v in contract.items()
        }

        raw_audit = audit(" ".join(v for v in contract.values() if v))
        normalized = normalize_extraction(contract, self.normalization)

        validation = validate_extraction(normalized, raw_output=raw, config=self.guardrails)
        confidences = per_field_confidence(normalized, validation)

        return ExtractionResult(
            ok=validation.verdict is not Verdict.REJECT,
            data=normalized,
            validation=validation,
            field_confidence=confidences,
            attempts=attempts,
            repaired=repaired,
            partial=partial or repaired,
            grammar_used=grammar_used,
            latency_s=latency,
            finish_reason=finish_reason,
            raw_len=len(raw),
            letterform_audit=raw_audit,
        )
