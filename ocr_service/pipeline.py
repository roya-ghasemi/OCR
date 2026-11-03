# -*- coding: utf-8 -*-
"""The request pipeline: image -> full transcript (transcribe.py) -> optional letter
fields cut from it by rules (letter_fields.py) -> response. CPU only, no model server.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import threading
import time
import uuid
from pathlib import Path

from PIL import Image

from . import __version__
from . import legible as _legible
from . import letterhead as _letterhead
from . import script_id as _script
from . import validate as _validate
from .cache import ResultCache
from .cache import key as cache_key
from .config import Settings
from .config import settings as default_settings
from .letter_fields import extract as extract_fields
from .schemas import LetterFields, LineOut, NumberOut, OcrResponse, UnreadRegion, Validation
from .transcribe import Transcriber, Transcript
from .transcribe import is_line_crop as _is_line_crop

log = logging.getLogger(__name__)

PREPARED_MAX_SIDE = 1600

REVIEW_LINE_CONF = 0.6      # a line below this mean confidence is worth a human look
REVIEW_NUMBER_CONF = 0.7    # a number below this (or not confirmed by the digit reader) likewise
REVIEW_MIN_GLYPH_PX = 20.0  # below this the dots of ب/ی/پ/ن and ح/خ/ج are not in the image at all


def _data_url(im) -> tuple[str | None, float | None]:
    """The prepared page as a JPEG data URL, and the scale it was sent at.

    The image is capped at PREPARED_MAX_SIDE to keep the response small, but every bbox is
    measured in the uncapped prepared page. A client drawing boxes or cutting review crops
    from this image multiplies the coordinates by the returned scale.
    """
    if im is None:
        return None, None
    out = im.copy()
    out.thumbnail((PREPARED_MAX_SIDE, PREPARED_MAX_SIDE), Image.LANCZOS)
    scale = out.width / im.width if im.width else 1.0
    buf = io.BytesIO()
    out.convert("RGB").save(buf, "JPEG", quality=82, optimize=True)
    return ("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
            round(scale, 6))


def _mean_conf(tr) -> float:
    """Mean line confidence weighted by line length, so one short confident line does not
    outrank a page of good text."""
    lines = getattr(tr, "lines", None) or []
    total = sum(len(L.text) for L in lines)
    if not total:
        return 0.0
    return sum(L.confidence * len(L.text) for L in lines) / total


class OcrPipeline:
    HEALTH_TTL_S = 30.0

    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or default_settings
        self.transcriber = self._make_transcriber(self.cfg.lang_fa, spellfix=self.cfg.spellfix)
        self.cache = ResultCache(
            self.cfg.cache_dir or (Path(__file__).resolve().parent.parent / ".cache" / "ocr"),
            max_entries=self.cfg.cache_max_entries, enabled=self.cfg.cache)
        self._arabic: Transcriber | None = None
        self._arabic_lock = threading.Lock()
        self._health: dict | None = None
        self._health_at = 0.0
        # Each page already fans out to `workers` Tesseract processes; more pages than this
        # at once only make every request slower.
        self._pages = asyncio.Semaphore(max(1, self.cfg.max_concurrent_pages))

    def _make_transcriber(self, lang: str, spellfix: bool) -> Transcriber:
        cmd, data = self.cfg.resolved_tesseract()
        return Transcriber(cmd, data, self.cfg.resolved_digit_model(), layout_psms=self.cfg.psms(),
                           text_psms=self.cfg.text_only_psms(),
                           deskew_min_deg=self.cfg.deskew_min_deg, min_line_conf=self.cfg.min_line_conf,
                           workers=self.cfg.workers, number_min_prob=self.cfg.number_min_prob,
                           spellfix=spellfix, crop_page=self.cfg.crop_page,
                           ink_separate=self.cfg.ink_separate, lang_fa=lang)

    # -- health ------------------------------------------------------------------
    def health(self, max_age_s: float | None = None) -> dict:
        """Engine, digit model, word list and cache status. Probing the engine starts two
        Tesseract processes, so a result younger than `max_age_s` is reused."""
        if self._health is None or max_age_s is None or time.monotonic() - self._health_at > max_age_s:
            self._health = self.transcriber.health()
            self._health_at = time.monotonic()
        return {**self._health, "cache": self.cache.stats()}

    def ready(self) -> bool:
        return self.health(self.HEALTH_TTL_S)["tesseract"]["available"]

    # -- reading -----------------------------------------------------------------
    async def run(self, image_bytes: bytes, doc_id: str | None = None,
                  want_prepared: bool = False) -> OcrResponse:
        # CPU-bound (Tesseract subprocesses + numpy): keep the event loop free
        async with self._pages:
            return await asyncio.to_thread(self.run_sync, image_bytes, doc_id, want_prepared)

    def run_sync(self, image_bytes: bytes, doc_id: str | None = None,
                 want_prepared: bool = False) -> OcrResponse:
        doc_id = doc_id or uuid.uuid4().hex
        # A cached result has no prepared image (megabytes of base64 only the viewer uses),
        # so a request that asks for one is always read.
        ck = None
        if self.cache.enabled and not want_prepared:
            try:
                ck = cache_key(image_bytes, self.cfg)
            except OSError:                     # a model file unreadable: read without the cache
                log.exception("cache key failed")
            hit = self.cache.get(ck) if ck else None
            if hit is not None:
                hit["doc_id"] = doc_id
                hit.setdefault("timing", {})["cached"] = True
                return OcrResponse(**hit)

        tr = self._read(image_bytes)
        t = time.perf_counter()
        fields, is_letter = (extract_fields(tr.lines, tr.image_size[1]) if self.cfg.letter_fields
                             else ({}, False))
        tr.timing["fields_s"] = round(time.perf_counter() - t, 3)

        # Handwriting and nastaliq do not make the recogniser fail, they make it answer:
        # «ee aa ay» for a page of calligraphy. Such a transcript is withheld and what the
        # recogniser offered goes to `unread`.
        verdict = _legible.judge(tr.text)
        if not verdict.legible:
            for L in tr.lines:
                tr.unread.append({"kind": "illegible", "bbox": list(L.bbox),
                                  "read_as": L.text, "confidence": L.confidence})
            tr.lines, tr.text = [], ""

        reasons = self._quality_reasons(tr, verdict)
        self._apply_letterhead(tr, fields, is_letter)
        checks = _validate.scan(tr.text or "")
        reasons += self._content_reasons(tr, checks)

        prepared_url, prepared_scale = _data_url(tr.prepared) if want_prepared else (None, None)
        res = OcrResponse(
            doc_id=doc_id, text=tr.text,
            lines=[LineOut(**L.__dict__) for L in tr.lines],
            numbers=[NumberOut(**n.__dict__) for n in tr.numbers],
            is_letter=is_letter, fields=LetterFields(**fields),
            needs_review=bool(reasons), review_reasons=reasons,
            unread=[UnreadRegion(**r) for r in tr.unread],
            validations=[Validation(**f.as_dict()) for f in checks],
            skew_deg=tr.skew_deg, glyph_px=tr.glyph_px, image_size=list(tr.image_size),
            prepared_image=prepared_url, prepared_scale=prepared_scale,
            legibility=verdict.as_dict(),
            prep=tr.prep, timing=tr.timing,
            service={"version": __version__, "engine": "tesseract(fas,eng) + digit_cnn",
                     "tesseract": (self._health or {}).get("tesseract", {}).get("version")})
        if ck:
            self.cache.put(ck, res.model_dump(mode="json"))
        return res

    def _read(self, image_bytes: bytes) -> Transcript:
        im = Image.open(io.BytesIO(image_bytes))
        # A single cropped line skips the page machinery (D82).
        tr = (self.transcriber.transcribe_line_crop(im) if _is_line_crop(im)
              else self.transcriber.transcribe(im))
        # The Persian model damages Arabic (ي/ك folded, harakat stripped). When the
        # transcript looks Arabic the page is read again with the Arabic model and the
        # reading the recogniser was surer of is kept.
        if self.cfg.lang_ar and _script.looks_arabic(tr.text):
            alt = self._arabic_read(image_bytes)
            if alt is not None and _mean_conf(alt) > _mean_conf(tr):
                tr = alt
        return tr

    @staticmethod
    def _quality_reasons(tr: Transcript, verdict) -> list[str]:
        reasons = []
        if not verdict.legible:
            reasons.append(f"this page could not be read — {verdict.reason}. "
                           f"What the recogniser offered is in `unread`, not in `text`")
        elif not tr.lines:
            reasons.append("no text found")
        if tr.glyph_px and tr.glyph_px < REVIEW_MIN_GLYPH_PX:
            reasons.append(f"image resolution too low: text is {tr.glyph_px:.0f} px tall, "
                           f"under the {REVIEW_MIN_GLYPH_PX:.0f} px needed to resolve Persian dots — "
                           f"rescan at 300 DPI for accurate text")
        weak_lines = sum(1 for L in tr.lines if L.confidence < REVIEW_LINE_CONF)
        if weak_lines:
            reasons.append(f"{weak_lines} low-confidence line(s)")
        weak_nums = sum(1 for n in tr.numbers if n.source == "tesseract" or n.confidence < REVIEW_NUMBER_CONF)
        if weak_nums:
            reasons.append(f"{weak_nums} number(s) not confirmed by the digit reader")
        return reasons

    @staticmethod
    def _content_reasons(tr: Transcript, checks: list) -> list[str]:
        reasons = []
        hand = [r for r in tr.unread if r["kind"] == "handwritten"]
        if hand:
            reasons.append(f"{len(hand)} handwritten field(s) left blank — read them from the image")
        marks = [r for r in tr.unread if r["kind"] == "illegible"]
        if marks:
            reasons.append(f"{len(marks)} region(s) could not be read and were left out")
        if any(r["kind"] == "letterhead" for r in tr.unread):
            reasons.append("the letterhead could not be read — the sender is in the image")
        bad = [f for f in checks if not f.valid]
        if bad:
            kinds = sorted({f.kind for f in bad})
            reasons.append(f"{len(bad)} value(s) do not have the shape they should "
                           f"({', '.join(kinds)}) — check them against the image")
        broken = [n.value for n in tr.numbers if n.grouping == "broken"]
        if broken:
            reasons.append(f"amount(s) with an incomplete group of three: {', '.join(broken)}")
        return reasons

    def _apply_letterhead(self, tr: Transcript, fields: dict, is_letter: bool) -> None:
        """The letterhead is read from its own crop and only ever adds fields or an unread
        region, never text: inverting the band inside the page cost the rest of the page
        2.1 pp of CER (E26)."""
        head = self._letterhead(tr)
        if not head.get("band"):
            return
        if is_letter and head.get("sender") and not fields.get("sender"):
            fields["sender"] = head["sender"]
        if is_letter and head.get("registration_number"):
            fields["registration_number"] = head["registration_number"]
        if not head.get("sender") and not head.get("registration_number"):
            tr.unread.append({"kind": "letterhead", "bbox": head["band"],
                              "read_as": " ".join(head.get("lines") or [])[:120],
                              "confidence": head.get("confidence", 0.0)})

    def _arabic_read(self, image_bytes: bytes) -> Transcript | None:
        """The same page through the Arabic model, or None if it cannot be had."""
        try:
            with self._arabic_lock:
                if self._arabic is None:
                    # spellfix off: the Persian word list has no opinion about Arabic
                    self._arabic = self._make_transcriber(self.cfg.lang_ar, spellfix=False)
            return self._arabic.transcribe(Image.open(io.BytesIO(image_bytes)))
        except Exception:
            log.exception("arabic re-read failed")
            return None

    def _letterhead(self, tr: Transcript) -> dict:
        """Read the letterhead off the prepared page, never raising into the request.

        The prepared page is used because every other box in the response is measured in it;
        an upload-space box would be wrong wherever the page was cropped or deskewed.
        """
        if tr.prepared is None:
            return {}
        try:
            import cv2
            import numpy as np
            cmd, data = self.cfg.resolved_tesseract()
            if not cmd:
                return {}
            rgb = np.asarray(tr.prepared.convert("RGB"))
            return _letterhead.read(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                    cmd, data, self.cfg.lang_fa)
        except Exception:                       # a letterhead is a bonus, never a failure
            log.exception("letterhead read failed")
            return {}
