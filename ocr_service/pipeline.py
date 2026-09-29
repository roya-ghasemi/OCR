# -*- coding: utf-8 -*-
"""The request pipeline: image -> full transcript (transcribe.py) -> optional letter
fields cut from it by rules (letter_fields.py) -> response. CPU only, no model server.
"""
from __future__ import annotations

import asyncio
import io
import time
import uuid

from PIL import Image

from . import __version__
from .config import Settings, settings as default_settings
from .letter_fields import extract as extract_fields
from .schemas import LetterFields, LineOut, NumberOut, OcrResponse
from .transcribe import Transcriber

REVIEW_LINE_CONF = 0.6      # a line below this mean confidence is worth a human look
REVIEW_NUMBER_CONF = 0.7    # a number below this (or not confirmed by the digit reader) likewise


class OcrPipeline:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or default_settings
        cmd, data = self.cfg.resolved_tesseract()
        self.transcriber = Transcriber(cmd, data, self.cfg.resolved_digit_model(), layout_psms=self.cfg.psms(),
                                       deskew_min_deg=self.cfg.deskew_min_deg, min_line_conf=self.cfg.min_line_conf,
                                       workers=self.cfg.workers, number_min_prob=self.cfg.number_min_prob)
        self._health: dict | None = None

    def health(self) -> dict:
        self._health = self.transcriber.health()
        return self._health

    def run_sync(self, image_bytes: bytes, doc_id: str | None = None) -> OcrResponse:
        doc_id = doc_id or uuid.uuid4().hex
        im = Image.open(io.BytesIO(image_bytes))
        tr = self.transcriber.transcribe(im)
        t = time.perf_counter()
        fields, is_letter = (extract_fields(tr.lines, tr.image_size[1]) if self.cfg.letter_fields
                             else ({}, False))
        tr.timing["fields_s"] = round(time.perf_counter() - t, 3)

        reasons = []
        if not tr.lines:
            reasons.append("no text found")
        weak_lines = sum(1 for L in tr.lines if L.confidence < REVIEW_LINE_CONF)
        if weak_lines:
            reasons.append(f"{weak_lines} low-confidence line(s)")
        weak_nums = sum(1 for n in tr.numbers if n.source == "tesseract" or n.confidence < REVIEW_NUMBER_CONF)
        if weak_nums:
            reasons.append(f"{weak_nums} number(s) not confirmed by the digit reader")

        return OcrResponse(
            doc_id=doc_id, text=tr.text,
            lines=[LineOut(**L.__dict__) for L in tr.lines],
            numbers=[NumberOut(**n.__dict__) for n in tr.numbers],
            is_letter=is_letter, fields=LetterFields(**fields),
            needs_review=bool(reasons), review_reasons=reasons,
            skew_deg=tr.skew_deg, image_size=list(tr.image_size), timing=tr.timing,
            service={"version": __version__, "engine": "tesseract(fas,eng) + digit_cnn",
                     "tesseract": (self._health or {}).get("tesseract", {}).get("version")})

    async def run(self, image_bytes: bytes, doc_id: str | None = None) -> OcrResponse:
        # CPU-bound (Tesseract subprocesses + numpy): keep the event loop free
        return await asyncio.to_thread(self.run_sync, image_bytes, doc_id)
