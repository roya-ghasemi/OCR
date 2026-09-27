"""Anti-hallucination guardrails and critical-field validation.

THE FAILURE THIS DEFENDS AGAINST
--------------------------------
The dangerous case is not a 422. It is HTTP 200 carrying fluent, well-formed,
confident Persian that has nothing to do with the page. Observed on a real
document: the model returned generic assistant boilerplate - "based on the
information I received in the system, I can provide you information as follows" -
with an invented sender, for a page that is a bank-guarantee letter carrying an
IBAN and a national ID. It parsed. It validated. It read naturally. Nothing
downstream could tell.

Two independent detection strategies, because either alone is defeatable:

  1. STRUCTURAL - does this look like an Iranian administrative letter at all?
     Real ones carry near-obligatory formulae and a recognisable field shape.
     A fabrication reproduces the register of a chat assistant, not of a
     ministry letter.
  2. ARITHMETIC - do the identifiers check out? A hallucinated national ID or
     IBAN fails its checksum with probability ~10/11 and ~96/97 respectively.
     This is the strongest signal available and it needs no model.

CALIBRATION, STATED HONESTLY
----------------------------
The boilerplate phrase list is seeded from ONE confirmed fabrication plus the
generic register of instruction-tuned assistants. It is not a corpus-derived
model. Treat `REVIEW` as "a human looks at this", never as ground truth, and
grow the phrase list from your own confirmed cases.

Thresholds are dataclass fields, not literals, so they can be tuned per tenant
without editing logic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Iterable

from .persian_text import for_machine
from .sampling import detect_degeneracy


class Verdict(str, Enum):
    ACCEPT = "accept"                    # ship it
    REVIEW = "review"                    # ship it, flagged, route to a human
    REJECT = "reject"                    # do not ship; return an error


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"                        # contributes to REVIEW
    CRITICAL = "critical"                # forces REJECT


@dataclass
class Finding:
    code: str
    severity: Severity
    message: str
    field_name: str | None = None
    evidence: str | None = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


# ---------------------------------------------------------------------------
# critical-field checksums
# ---------------------------------------------------------------------------

_NATIONAL_ID_RE = re.compile(r"(?<!\d)(\d{10})(?!\d)")
_IBAN_RE = re.compile(r"\bIR[ ]?(\d[ ]?){24}\b", re.IGNORECASE)
_POSTAL_RE = re.compile(r"(?<!\d)(\d{10})(?!\d)")
_MOBILE_RE = re.compile(r"(?:(?:\+98|0098|98)|0)?9\d{9}\b")
_LANDLINE_RE = re.compile(r"\b0\d{2,3}[- ]?\d{7,8}\b")


def validate_national_id(value: str) -> bool:
    """Iranian کد ملی check digit (10 digits, weights 10..2, mod 11).

    Also rejects the repdigit IDs (0000000000, 1111111111, ...). They satisfy the
    checksum arithmetically but are never issued, and they are exactly what a
    model produces when it invents a number.
    """
    digits = re.sub(r"\D", "", value or "")
    if len(digits) != 10:
        return False
    if digits == digits[0] * 10:
        return False
    total = sum(int(digits[i]) * (10 - i) for i in range(9))
    remainder = total % 11
    check = int(digits[9])
    return check == remainder if remainder < 2 else check == 11 - remainder


def validate_iban(value: str) -> bool:
    """ISO 13616 mod-97. Iranian IBANs are IR + 24 digits (26 chars)."""
    s = re.sub(r"[\s-]", "", (value or "")).upper()
    if not re.fullmatch(r"IR\d{24}", s):
        return False
    rearranged = s[4:] + s[:4]
    numeric = "".join(
        str(ord(c) - 55) if c.isalpha() else c for c in rearranged
    )
    return int(numeric) % 97 == 1


def validate_iranian_mobile(value: str) -> bool:
    """Accepts 09xxxxxxxxx, +989xxxxxxxxx and 00989xxxxxxxxx.

    Order matters: strip the international access code (00) BEFORE the country
    code, or `00989...` keeps its leading zeros through the country-code check
    and is rejected as the wrong length.
    """
    d = re.sub(r"\D", "", value or "")
    if d.startswith("00"):
        d = d[2:]
    if d.startswith("98"):
        d = d[2:]
    d = d.lstrip("0")
    return len(d) == 10 and d.startswith("9")


def extract_identifiers(text: str | None) -> dict[str, list[str]]:
    """Pull candidate identifiers from a field, in ASCII digits.

    `for_machine` is used deliberately: a national ID written in Persian-Indic
    digits cannot be checksummed, and running the check over ۰-۹ silently passes
    everything. This is the one place the ASCII normalization direction is
    mandatory rather than a preference.
    """
    if not text:
        return {"national_id": [], "iban": [], "mobile": [], "landline": []}
    ascii_text = for_machine(text) or ""
    return {
        "national_id": _NATIONAL_ID_RE.findall(ascii_text),
        "iban": [m.group(0) for m in _IBAN_RE.finditer(ascii_text)],
        "mobile": _MOBILE_RE.findall(ascii_text),
        "landline": _LANDLINE_RE.findall(ascii_text),
    }


# ---------------------------------------------------------------------------
# structural anomaly detection
# ---------------------------------------------------------------------------

# Formulae that appear in essentially every Iranian administrative letter. A
# document containing none of them is either not such a letter, or was not read.
DOCUMENT_ANCHORS = (
    "احترام", "با سلام", "موضوع", "خواهشمند", "استحضار", "بدینوسیله",
    "بدين وسيله", "پیوست", "به پیوست", "گواهی", "جناب", "سرکار",
    "ریاست", "مدیریت", "شرکت", "سازمان", "اداره", "تقدیم", "مقتضی",
)

# First-person assistant register. A letter is written by an organisation TO a
# recipient; it does not offer to provide information, and it does not say "I".
# Seeded from one confirmed fabrication plus the generic instruction-tuned voice.
BOILERPLATE_MARKERS = (
    "می‌توانم به شما",
    "میتوانم به شما",
    "اطلاعاتی ارائه دهم",
    "در زیر آورده شده است",
    "چگونه می‌توانم کمک",
    "من یک مدل",
    "به عنوان یک هوش مصنوعی",
    "متاسفانه نمی‌توانم",
    "لطفا سوال خود را",
    "as an ai", "i cannot", "i'm sorry", "how can i help",
    "here is the information", "based on the information",
)

# Values that are placeholder text rather than transcription.
PLACEHOLDER_VALUES = (
    "string", "null", "none", "n/a", "na", "unknown", "نامشخص", "ندارد",
    "sender", "receiver", "subject", "body_text", "contact_info",
    "<نام سازمان یا شخص فرستنده>", "example", "sample", "test",
)


@dataclass
class GuardrailConfig:
    """Every threshold is a knob. None of these are laws of nature."""

    # A `sender` longer than this is the collapse defect, not a company name.
    max_sender_chars: int = 220
    max_receiver_chars: int = 220
    max_subject_chars: int = 300
    # Below this, an extraction is too thin to be a letter.
    min_total_chars: int = 40
    # Cross-field duplication ratio above which two fields are "the same text".
    duplication_similarity: float = 0.85
    min_duplication_chars: int = 25
    # A letter with no anchor phrase anywhere is structurally anomalous.
    require_document_anchor: bool = True
    # Checksum failures: warn, or reject outright?
    identifier_failure_is_critical: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_GUARDRAILS = GuardrailConfig()


def _similarity(a: str, b: str) -> float:
    """Cheap containment-based similarity; no external dependency."""
    if not a or not b:
        return 0.0
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if short in long_:
        return 1.0
    # token overlap fallback
    sa, sb = set(short.split()), set(long_.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class ValidationResult:
    verdict: Verdict
    findings: list[Finding] = field(default_factory=list)
    identifiers: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    @property
    def needs_review(self) -> bool:
        return self.verdict in (Verdict.REVIEW, Verdict.REJECT)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 3),
            "needs_review": self.needs_review,
            "findings": [f.as_dict() for f in self.findings],
            "identifiers": self.identifiers,
        }


def validate_extraction(
    data: dict[str, str | None],
    raw_output: str | None = None,
    config: GuardrailConfig = DEFAULT_GUARDRAILS,
) -> ValidationResult:
    """Run every guardrail over one extraction.

    `data` is the parsed (and normalized) extraction. `raw_output` is the model's
    unparsed text - pass it when available so degeneracy that the parser hid can
    still be caught.
    """
    findings: list[Finding] = []
    values = {k: (v or "") for k, v in data.items()}
    all_text = " ".join(values.values()).strip()

    # -- 1. empty / near-empty ------------------------------------------------
    if not all_text:
        findings.append(Finding(
            "empty_extraction", Severity.CRITICAL,
            "Every field is empty. This is a failed read, not a document without "
            "content.",
        ))
    elif len(all_text) < config.min_total_chars:
        findings.append(Finding(
            "extraction_too_short", Severity.WARN,
            f"Only {len(all_text)} characters extracted across all fields.",
        ))

    # -- 2. assistant boilerplate --------------------------------------------
    low = all_text.lower()
    for marker in BOILERPLATE_MARKERS:
        if marker.lower() in low:
            findings.append(Finding(
                "assistant_boilerplate", Severity.CRITICAL,
                "Output contains conversational assistant language, which an "
                "administrative letter never contains. Strong fabrication signal.",
                evidence=marker,
            ))
            break

    # -- 3. placeholder / schema echo ----------------------------------------
    for name, val in values.items():
        if val.strip().lower() in PLACEHOLDER_VALUES:
            findings.append(Finding(
                "placeholder_value", Severity.WARN,
                "Field contains a placeholder or an echo of the schema rather "
                "than transcribed text.",
                field_name=name, evidence=val[:60],
            ))

    # -- 4. field-schema collapse --------------------------------------------
    caps = {
        "sender": config.max_sender_chars,
        "receiver": config.max_receiver_chars,
        "subject": config.max_subject_chars,
    }
    for name, cap in caps.items():
        val = values.get(name, "")
        if len(val) > cap:
            findings.append(Finding(
                "field_collapse", Severity.WARN,
                f"`{name}` holds {len(val)} characters (cap {cap}). The model has "
                "written document body into a header field.",
                field_name=name, evidence=val[:80],
            ))

    # -- 5. cross-field duplication ------------------------------------------
    names = [n for n, v in values.items() if len(v) >= config.min_duplication_chars]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if _similarity(values[a], values[b]) >= config.duplication_similarity:
                findings.append(Finding(
                    "cross_field_duplication", Severity.WARN,
                    f"`{a}` and `{b}` carry substantially the same text. One of "
                    "them is misassigned.",
                    field_name=f"{a}+{b}",
                ))

    # -- 6. document anchors --------------------------------------------------
    if config.require_document_anchor and all_text:
        if not any(a in all_text for a in DOCUMENT_ANCHORS):
            findings.append(Finding(
                "no_document_anchor", Severity.WARN,
                "No Persian administrative formula found anywhere in the "
                "extraction. Either this is not such a letter, or it was not read.",
            ))

    # -- 7. degeneracy that survived parsing ---------------------------------
    for name, val in values.items():
        rep = detect_degeneracy(val)
        if rep.is_degenerate:
            findings.append(Finding(
                "degenerate_run", Severity.CRITICAL,
                f"Repetition loop inside a field that parsed successfully "
                f"({rep.kind}, run length {rep.run_length}). Corrupted content "
                "that would otherwise have shipped as a clean 200.",
                field_name=name, evidence=rep.codepoint or rep.repeated,
            ))
            break
    if raw_output:
        rep = detect_degeneracy(raw_output)
        if rep.is_degenerate and not any(f.code == "degenerate_run" for f in findings):
            findings.append(Finding(
                "degenerate_raw_output", Severity.WARN,
                f"Repetition loop in the raw model output ({rep.kind}). The "
                "parsed fields may be truncated.",
                evidence=rep.codepoint or rep.repeated,
            ))

    # -- 8. identifier checksums ---------------------------------------------
    ids = extract_identifiers(all_text)
    id_report: dict[str, Any] = {}
    sev = Severity.CRITICAL if config.identifier_failure_is_critical else Severity.WARN

    checked = [
        ("national_id", ids["national_id"], validate_national_id),
        ("iban", ids["iban"], validate_iban),
        ("mobile", ids["mobile"], validate_iranian_mobile),
    ]
    for kind, candidates, checker in checked:
        results = [{"value": c, "valid": checker(c)} for c in dict.fromkeys(candidates)]
        id_report[kind] = results
        for r in results:
            if not r["valid"]:
                findings.append(Finding(
                    "identifier_checksum_failed", sev,
                    f"A candidate {kind} failed its checksum. Either the digits "
                    "were misread or the value was invented - both require a "
                    "human to compare against the page.",
                    evidence=r["value"],
                ))
    id_report["landline"] = [{"value": v, "valid": True} for v in dict.fromkeys(ids["landline"])]

    # -- verdict --------------------------------------------------------------
    critical = [f for f in findings if f.severity is Severity.CRITICAL]
    warnings = [f for f in findings if f.severity is Severity.WARN]

    if critical:
        verdict = Verdict.REJECT
    elif warnings:
        verdict = Verdict.REVIEW
    else:
        verdict = Verdict.ACCEPT

    # A blunt, legible score: each warning costs 0.15, each critical 0.5. It is a
    # routing aid, NOT a probability - do not present it to users as one.
    confidence = max(0.0, 1.0 - 0.15 * len(warnings) - 0.5 * len(critical))

    return ValidationResult(verdict, findings, id_report, confidence)


def per_field_confidence(
    data: dict[str, str | None],
    result: ValidationResult,
) -> dict[str, float]:
    """Crude per-field confidence for routing individual fields to review.

    Starts at 1.0, subtracts for findings naming that field, and floors any field
    holding an identifier that failed its checksum. Heuristic by construction:
    the model exposes no token logprobs through this path, so there is no
    calibrated signal to use instead.
    """
    scores = {k: 1.0 for k in data}
    for f in result.findings:
        if not f.field_name:
            continue
        for name in f.field_name.split("+"):
            if name in scores:
                scores[name] -= 0.5 if f.severity is Severity.CRITICAL else 0.25
    bad_ids = {
        r["value"]
        for kind, rows in result.identifiers.items()
        for r in rows if not r["valid"]
    }
    for name, val in data.items():
        if val and any(b in (for_machine(val) or "") for b in bad_ids):
            scores[name] = min(scores.get(name, 1.0), 0.3)
    return {k: round(max(0.0, v), 2) for k, v in scores.items()}
