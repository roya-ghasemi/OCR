# -*- coding: utf-8 -*-
"""The request pipeline: image -> full transcript (transcribe.py) -> optional letter
fields cut from it by rules (letter_fields.py) -> response. CPU only, no model server.
"""
from __future__ import annotations

import asyncio
import base64
import io
import time
import uuid

from PIL import Image

import logging
from pathlib import Path

from . import __version__
from .config import Settings, settings as default_settings
from .letter_fields import extract as extract_fields
from . import letterhead as _letterhead
from . import legible as _legible
from . import script_id as _script
from . import validate as _validate
from .cache import ResultCache, key as cache_key

log = logging.getLogger(__name__)
from .schemas import (LetterFields, LineOut, NumberOut, OcrResponse, UnreadRegion,
                      Validation)
from .transcribe import Transcriber, is_line_crop as _is_line_crop

PREPARED_MAX_SIDE = 1600


def _data_url(im) -> tuple[str | None, float | None]:
    """The prepared page as a JPEG data URL, **and the scale it was sent at**.

    The cap keeps a response sendable: a 2192×2996 page is 1171×1600 by the time it is
    base64. Every bbox in the response is measured in the *uncapped* prepared page, so a
    client drawing a crop straight from this image with those coordinates cuts the wrong
    pixels — and silently drops any region that falls past the capped edge. That is what
    happened to the review crops on the brief's letter: four cut from the wrong place, one
    beyond the right-hand edge and not drawn at all.

    Returning the factor is what makes the two agree. The boxes stay in one canonical space
    (the prepared page) and anything working in sent-image pixels multiplies by this.
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
    """Mean line confidence, weighted by how much text each line carries.

    Unweighted, a page of one correct word outscores a page of forty good lines and one bad
    one, which is the opposite of what is being asked.
    """
    lines = getattr(tr, "lines", None) or []
    total = sum(len(L.text) for L in lines)
    if not total:
        return 0.0
    return sum(L.confidence * len(L.text) for L in lines) / total


REVIEW_LINE_CONF = 0.6      # a line below this mean confidence is worth a human look
REVIEW_NUMBER_CONF = 0.7    # a number below this (or not confirmed by the digit reader) likewise
REVIEW_MIN_GLYPH_PX = 20.0  # below this the dots of ب/ی/پ/ن and ح/خ/ج are not in the image at all


class OcrPipeline:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or default_settings
        cmd, data = self.cfg.resolved_tesseract()
        self.transcriber = Transcriber(cmd, data, self.cfg.resolved_digit_model(), layout_psms=self.cfg.psms(),
                                       deskew_min_deg=self.cfg.deskew_min_deg, min_line_conf=self.cfg.min_line_conf,
                                       workers=self.cfg.workers, number_min_prob=self.cfg.number_min_prob,
                                       spellfix=self.cfg.spellfix, crop_page=self.cfg.crop_page,
                                       ink_separate=self.cfg.ink_separate,
                                       lang_fa=self.cfg.lang_fa)
        self.cache = ResultCache(
            self.cfg.cache_dir or (Path(__file__).resolve().parent.parent / ".cache" / "ocr"),
            max_entries=self.cfg.cache_max_entries, enabled=self.cfg.cache)
        self._arabic = None
        self._health: dict | None = None

    def health(self) -> dict:
        self._health = self.transcriber.health()
        self._health["cache"] = self.cache.stats()
        return self._health

    def run_sync(self, image_bytes: bytes, doc_id: str | None = None,
                 want_prepared: bool = False) -> OcrResponse:
        doc_id = doc_id or uuid.uuid4().hex
        # A cached transcript carries no prepared image — it is megabytes of base64 that only
        # the viewer wants — so a request asking for one is read properly instead.
        ck = None
        if self.cache.enabled and not want_prepared:
            try:
                ck = cache_key(image_bytes, self.cfg)
            except Exception:
                ck = None
            if ck:
                hit = self.cache.get(ck)
                if hit is not None:
                    hit["doc_id"] = doc_id      # the result is shared; this request is not
                    hit.setdefault("timing", {})["cached"] = True
                    return OcrResponse(**hit)
        im = Image.open(io.BytesIO(image_bytes))
        # A cropped line goes down its own route: native size, one single-line read, none of
        # the page machinery. See `Transcriber.transcribe_line_crop` and D82.
        tr = (self.transcriber.transcribe_line_crop(im) if _is_line_crop(im)
              else self.transcriber.transcribe(im))

        # An Arabic page read by the Persian model comes back damaged: ي and ك folded to ی
        # and ک, harakat stripped, «الْعِلْمِ» as «الیلم». When the transcript looks Arabic,
        # read it again with the Arabic model and keep whichever the recogniser was surer
        # of. The gate decides only whether to spend the pass; the comparison decides the
        # answer, so a false positive costs a second and nothing else.
        if self.cfg.lang_ar and _script.looks_arabic(tr.text):
            alt = self._arabic_read(image_bytes)
            if alt is not None and _mean_conf(alt) > _mean_conf(tr):
                tr = alt
        t = time.perf_counter()
        fields, is_letter = (extract_fields(tr.lines, tr.image_size[1]) if self.cfg.letter_fields
                             else ({}, False))
        tr.timing["fields_s"] = round(time.perf_counter() - t, 3)

        # Is this a transcript, or the shapes the strokes resembled? Handwriting and
        # nastaliq do not make the recogniser fail, they make it answer — «ee aa ay» for a
        # page of calligraphy. Withholding that is the brief's first rule; printing it is
        # the one failure a reader cannot detect for themselves.
        verdict = _legible.judge(tr.text)
        if not verdict.legible:
            for L in tr.lines:
                tr.unread.append({"kind": "illegible", "bbox": list(L.bbox),
                                  "read_as": L.text, "confidence": L.confidence})
            tr.lines, tr.text = [], ""

        reasons = []
        if not verdict.legible:
            reasons.append(f"this page could not be read — {verdict.reason}. "
                           f"What the recogniser offered is in `unread`, not in `text`")
        elif not tr.lines:
            reasons.append("no text found")
        # The commonest cause of Persian typos is not the reader: below ~20 px the dots
        # that tell ب/ی/پ/ن and ح/خ/ج apart are not in the image, so no amount of
        # post-processing can recover them without guessing. Say so instead.
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
        # The letterhead, read on its own terms. It never touches `text` or `lines`:
        # inverting the band inside the page was measured in E26 and cost the rest of the
        # page 2.1 pp of CER, because the band's box also holds white paper carrying the
        # form labels. This reads its own crop and contributes only fields and a region.
        head = self._letterhead(tr)
        if head.get("band"):
            if is_letter and head.get("sender") and not fields.get("sender"):
                fields["sender"] = head["sender"]
            if is_letter and head.get("registration_number"):
                fields["registration_number"] = head["registration_number"]
            if not head.get("sender") and not head.get("registration_number"):
                tr.unread.append({"kind": "letterhead", "bbox": head["band"],
                                  "read_as": " ".join(head.get("lines") or [])[:120],
                                  "confidence": head.get("confidence", 0.0)})

        hand = [r for r in tr.unread if r["kind"] == "handwritten"]
        if hand:
            reasons.append(f"{len(hand)} handwritten field(s) left blank — read them from the image")
        marks = [r for r in tr.unread if r["kind"] == "illegible"]
        if marks:
            reasons.append(f"{len(marks)} region(s) could not be read and were left out")
        if any(r["kind"] == "letterhead" for r in tr.unread):
            reasons.append("the letterhead could not be read — the sender is in the image")
        checks = _validate.scan(tr.text or "")
        bad = [f for f in checks if not f.valid]
        if bad:
            kinds = sorted({f.kind for f in bad})
            reasons.append(f"{len(bad)} value(s) do not have the shape they should "
                           f"({', '.join(kinds)}) — check them against the image")
        broken = [n.value for n in tr.numbers if n.grouping == "broken"]
        if broken:
            reasons.append(f"amount(s) with an incomplete group of three: {', '.join(broken)}")

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

    def _arabic_read(self, image_bytes: bytes):
        """The same page through the Arabic model, or None if it cannot be had."""
        try:
            if self._arabic is None:
                cmd, data = self.cfg.resolved_tesseract()
                self._arabic = Transcriber(
                    cmd, data, self.cfg.resolved_digit_model(), layout_psms=self.cfg.psms(),
                    deskew_min_deg=self.cfg.deskew_min_deg, min_line_conf=self.cfg.min_line_conf,
                    workers=self.cfg.workers, number_min_prob=self.cfg.number_min_prob,
                    spellfix=False,          # the Persian lexicon has no opinion about Arabic
                    crop_page=self.cfg.crop_page, ink_separate=self.cfg.ink_separate,
                    lang_fa=self.cfg.lang_ar)
            return self._arabic.transcribe(Image.open(io.BytesIO(image_bytes)))
        except Exception:
            log.exception("arabic re-read failed")
            return None

    def _letterhead(self, tr) -> dict:
        """Read the letterhead off the prepared page, never raising into the request.

        The prepared page is used rather than the upload because its coordinates are the
        ones every other box in the response is measured in — it is RGB and the same size,
        and the colour the band is found by survives preparation (blue stamp removal does
        not touch green). An upload-space box would be silently wrong wherever the page was
        cropped or deskewed.
        """
        if getattr(tr, "prepared", None) is None:
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

    async def run(self, image_bytes: bytes, doc_id: str | None = None,
                  want_prepared: bool = False) -> OcrResponse:
        # CPU-bound (Tesseract subprocesses + numpy): keep the event loop free
        return await asyncio.to_thread(self.run_sync, image_bytes, doc_id, want_prepared)
