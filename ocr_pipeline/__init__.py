"""Production hardening for the Persian/bilingual OCR pipeline.

    from ocr_pipeline.api import router, init_pipeline

Modules:
    sampling      decoding parameters that eliminate repetition degeneracy
    persian_text  letterform + context-aware digit normalization
    grammar       GBNF generation; the structural fix for field collapse
    extraction    orchestrator: grammar -> parse -> repair -> retry -> guardrails
    validation    checksums and hallucination detection
    api           FastAPI router
"""
from .grammar import DEFAULT_SPEC, FIELD_ORDER, GrammarSpec, build_grammar
from .persian_text import (
    DigitPolicy, MACHINE_POLICY, NormalizationPolicy, OUTPUT_POLICY,
    for_machine, for_output, normalize, normalize_extraction,
)
from .sampling import (
    ACCURACY_PRESERVING, BELT_AND_BRACES, PROVEN, RETRY, SamplingProfile,
    detect_degeneracy,
)
from .validation import (
    GuardrailConfig, Severity, ValidationResult, Verdict,
    validate_extraction, validate_iban, validate_national_id,
)
from .extraction import ExtractionResult, LetterExtractor, repair_json

__all__ = [
    "DEFAULT_SPEC", "FIELD_ORDER", "GrammarSpec", "build_grammar",
    "DigitPolicy", "MACHINE_POLICY", "NormalizationPolicy", "OUTPUT_POLICY",
    "for_machine", "for_output", "normalize", "normalize_extraction",
    "ACCURACY_PRESERVING", "BELT_AND_BRACES", "PROVEN", "RETRY",
    "SamplingProfile", "detect_degeneracy",
    "GuardrailConfig", "Severity", "ValidationResult", "Verdict",
    "validate_extraction", "validate_iban", "validate_national_id",
    "ExtractionResult", "LetterExtractor", "repair_json",
]
