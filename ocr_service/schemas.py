# -*- coding: utf-8 -*-
"""Response contract for POST /ocr."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LetterFields(BaseModel):
    """Verbatim runs of transcribed lines, or null - never generated (letter_fields.py)."""
    sender: str | None = None
    registration_number: str | None = Field(
        default=None,
        description="the company registration number printed in the letterhead, when it was "
                    "read confidently. Null is the normal case: the letterhead is usually a "
                    "stylised graphic and a guessed registration number is worse than none")
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


class UnreadRegion(BaseModel):
    """A place on the page the service declined to transcribe, and why.

    A handwritten form value, a mark that is not text, or the letterhead band whose white
    text on colour the recogniser cannot read. The region is reported with its box so it can
    be shown to a human, never guessed at: these are exactly the places where printing a
    reading would mean inventing one.
    """
    kind: Literal["handwritten", "illegible", "letterhead"]
    bbox: list[int]
    read_as: str = Field(description="what the recogniser offered, kept only so a reviewer "
                                     "can see why it was rejected; never part of `text`")
    confidence: float


class OcrResponse(BaseModel):
    doc_id: str
    text: str = Field(description="every transcribed line in reading order; paragraphs separated by a blank line")
    lines: list[LineOut]
    numbers: list[NumberOut]
    is_letter: bool = Field(description="a salutation or addressee line was found; `fields` are null otherwise")
    fields: LetterFields
    needs_review: bool
    review_reasons: list[str] = Field(default_factory=list)
    unread: list[UnreadRegion] = Field(
        default_factory=list,
        description="regions left out of `text` because nobody read them: handwritten form "
                    "values and marks that are not text. Each carries a box to display")
    skew_deg: float = Field(description="rotation applied before reading (degrees, counter-clockwise)")
    glyph_px: float = Field(
        default=0.0,
        description="text height in pixels in the image as supplied. Below ~20 the dots that "
                    "separate ب/ی/پ/ن and ح/خ/ج are not resolved and typos are unavoidable")
    image_size: list[int] = Field(
        description="size of the PREPARED page — the image every bbox in this response is "
                    "measured in, which is not the uploaded file when it was cropped")
    prepared_image: str | None = Field(
        default=None,
        description="the page as the recogniser saw it (cropped, straightened, stamp removed), "
                    "as a data URL. Returned only when the request asks for it; boxes line up "
                    "with this image, not with the upload")
    prepared_scale: float | None = Field(
        default=None,
        description="multiply a bbox by this to reach pixel coordinates in `prepared_image`, "
                    "which is capped in size so the response stays sendable. 1.0 means it was "
                    "sent at full size. Ignoring it cuts review crops from the wrong pixels")
    prep: dict = Field(default_factory=dict,
                       description="what the preparation stage did: crop, perspective warp, ink")
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
    spellfix: dict = Field(default_factory=dict,
                           description="dot-confusion repair: enabled, available, word-list size")
    queue: dict
    service: dict
