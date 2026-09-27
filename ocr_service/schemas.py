# -*- coding: utf-8 -*-
"""Response contract for POST /ocr (Section 5.1)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LetterFields(BaseModel):
    sender: str | None = None
    receiver: str | None = None
    subject: str | None = None
    body_text: str | None = None
    contact_info: str | None = None


class NumericFieldOut(BaseModel):
    field: str
    kind: str
    value: str = Field(description="as the primary model printed it (or as read from the page, when `added`)")
    value_ascii: str
    confidence: Literal["high", "corrected", "low", "unverified", "added"] = Field(
        description="high: page read agrees; corrected: page read replaced the model's digits (see `resolved`); "
                    "low: aligned but not confident enough to override; unverified: nothing on the page aligns; "
                    "added: read from the page, absent from the model output")
    tesseract_value: str | None = Field(default=None, description="the classical reader's value (name kept for compatibility)")
    classical_value: str | None = None
    candidates: list[str] = Field(default_factory=list, description="all readings; >1 means a conflict")
    bbox: list[int] | None = None
    similarity: float | None = None
    note: str | None = None
    resolved: str | None = Field(default=None, description="value a consumer should use (see numeric_validator.resolve)")


class ModelRun(BaseModel):
    model: str
    ok: bool
    latency_s: float
    attempts: int = 0
    finish_reason: str | None = None
    grammar_used: bool = False
    error: str | None = None


class CrossCheck(BaseModel):
    enabled: bool
    secondary: ModelRun | None = None
    field_agreement: dict[str, float] = Field(default_factory=dict, description="per-field similarity primary vs secondary")
    numeric_mismatches: list[dict] = Field(default_factory=list)
    needs_review: bool = False


class Timing(BaseModel):
    preprocess_s: float = 0.0
    primary_s: float = 0.0
    secondary_s: float = 0.0
    numeric_validation_s: float = 0.0
    total_s: float = 0.0


class OcrResponse(BaseModel):
    doc_id: str
    fields: LetterFields
    numeric_fields: list[NumericFieldOut]
    numeric_summary: dict
    primary: ModelRun
    cross_check: CrossCheck
    preprocess: dict | None = None
    timing: Timing
    needs_review: bool
    service: dict


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    result: OcrResponse | None = None
    error: str | None = None


class Health(BaseModel):
    status: Literal["ok", "degraded", "down"]
    backend: str
    models: list[dict]
    tesseract: dict
    queue: dict
    service: dict
