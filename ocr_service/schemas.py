# -*- coding: utf-8 -*-
"""Response contract for POST /ocr."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LetterFields(BaseModel):
    """Verbatim runs of transcribed lines, or null - never generated (letter_fields.py)."""
    sender: str | None = None
    receiver: str | None = None
    subject: str | None = Field(default=None, description="only the text after a printed «موضوع:» label")
    body_text: str | None = None
    contact_info: str | None = None


class Cell(BaseModel):
    text: str
    bbox: list[int]


class LineOut(BaseModel):
    text: str
    bbox: list[int] = Field(description="x0, y0, x1, y1 in the (deskewed) input image")
    confidence: float = Field(description="mean word confidence, 0..1")
    source: str = Field(description="which reading won: psm3 | psm4 | psm6 | line-fas | line-eng")
    paragraph: int
    cells: list[Cell] = Field(
        default_factory=list,
        description="the line's pieces when a wide gap splits it (a table row), in reading order; "
                    "empty on a line of ordinary prose")


class NumberOut(BaseModel):
    value: str = Field(description="as it appears in `text`")
    value_ascii: str
    bbox: list[int]
    source: Literal["glyph", "tesseract"] = Field(
        description="glyph: digits from the CNN digit reader; tesseract: the reader did not confirm this token")
    confidence: float = Field(description="0..1 (digit-reader mean probability, or Tesseract word confidence)")
    grouping: Literal["ok", "broken", "none"] = Field(
        default="none",
        description="thousands separators: ok = whole groups of three; broken = a group of the wrong "
                    "length, so a digit is missing or invented — check this amount; none = not an amount")


class OcrResponse(BaseModel):
    doc_id: str
    text: str = Field(description="every transcribed line in reading order; paragraphs separated by a blank line")
    lines: list[LineOut]
    numbers: list[NumberOut]
    is_letter: bool = Field(description="a salutation or addressee line was found; `fields` are null otherwise")
    fields: LetterFields
    needs_review: bool
    review_reasons: list[str] = Field(default_factory=list)
    skew_deg: float = Field(description="rotation applied before reading (degrees, counter-clockwise)")
    image_size: list[int]
    timing: dict
    service: dict


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    result: OcrResponse | None = None
    error: str | None = None


class Health(BaseModel):
    status: Literal["ok", "degraded", "down"]
    tesseract: dict
    digit_reader: dict
    queue: dict
    service: dict
