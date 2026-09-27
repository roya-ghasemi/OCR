# -*- coding: utf-8 -*-
"""The request pipeline: preprocess -> primary VLM -> [secondary VLM] -> numeric
validation -> normalised response. Backend-agnostic; everything model-specific is
behind `InferenceBackend`.

Output normalisation (D43): Arabic letterforms are folded at this boundary with the
letterform-only policy — digits are left exactly as emitted so the numeric layer
compares what the model actually wrote.
"""
from __future__ import annotations

import asyncio
import difflib
import io
import logging
import time
import uuid
from dataclasses import replace

from PIL import Image

from ocr_pipeline.persian_text import OUTPUT_POLICY, DigitPolicy, normalize_extraction

from . import __version__
from .backends import InferenceBackend, Extraction, make_backend
from .config import Settings, settings as default_settings
from .numeric_validator import NumericField, NumericValidator, TesseractReader, digits_only, find_numeric_spans, summarize
from .preprocess import prepare
from .schemas import (CrossCheck, LetterFields, ModelRun, NumericFieldOut, OcrResponse, Timing)

log = logging.getLogger(__name__)
FIELDS = ["sender", "receiver", "subject", "body_text", "contact_info"]
# Letterforms folded, whitespace collapsed, digits untouched.
LETTERFORM_ONLY = replace(OUTPUT_POLICY, digits=DigitPolicy.PRESERVE)


def _run(x: Extraction) -> ModelRun:
    return ModelRun(model=x.model, ok=x.ok, latency_s=round(x.latency_s, 3), attempts=x.attempts,
                    finish_reason=x.finish_reason, grammar_used=x.grammar_used, error=x.error)


def _sim(a: str | None, b: str | None) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 3)


def compute_cross_check(fields: dict, sec_fields: dict, numeric: list, *, min_digits: int = 3,
                        policy: str = "flag", secondary_run: ModelRun | None = None) -> CrossCheck:
    """Compare the primary's fields/numbers with the secondary's. Pure; used online
    (cpu_offload / concurrent) and offline by the benchmark (sequential)."""
    cross = CrossCheck(enabled=True, secondary=secondary_run)
    cross.field_agreement = {f: _sim(fields.get(f), sec_fields.get(f)) for f in FIELDS}
    sec_by_field: dict[str, set[str]] = {}
    for s in find_numeric_spans(sec_fields, min_digits):
        sec_by_field.setdefault(s.field, set()).add(digits_only(s.value))
    sec_any = set().union(*sec_by_field.values()) if sec_by_field else set()
    for n in numeric:
        d = digits_only(n.value)
        # Assumption: a number counts as agreed if the secondary printed the same
        # digits in ANY field — the two models route text differently (D44/D45).
        if d not in sec_any:
            cross.numeric_mismatches.append({"field": n.field, "kind": n.kind, "primary": n.value,
                                             "secondary_has": sorted(sec_by_field.get(n.field, set()))[:5]})
            if policy == "flag" and n.confidence == "high":
                n.confidence = "low"; n.note = ((n.note or "") + " secondary model disagrees").strip()
            elif policy == "prefer_secondary" and sec_by_field.get(n.field):
                n.candidates = list(dict.fromkeys(n.candidates + sorted(sec_by_field[n.field])))
                n.confidence = "low"
    cross.needs_review = bool(cross.numeric_mismatches) or any(v < 0.5 for v in cross.field_agreement.values())
    return cross


class OcrPipeline:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or default_settings
        self.primary: InferenceBackend = make_backend(self.cfg.primary, self.cfg, self.cfg.backend_url_primary)
        self.secondary: InferenceBackend | None = None
        # `sequential` is a batch strategy (benchmark.py swaps models); online the
        # secondary is either concurrent (two GPUs) or CPU-offloaded beside the primary.
        if self.cfg.enable_secondary_model and self.cfg.secondary_mode != "sequential":
            ngl = self.cfg.secondary_n_gpu_layers if self.cfg.secondary_mode == "cpu_offload" else None
            self.secondary = make_backend(self.cfg.secondary, self.cfg, self.cfg.backend_url_secondary)
            if hasattr(self.secondary, "n_gpu_layers"):
                self.secondary.n_gpu_layers = ngl
        cmd, data = self.cfg.resolved_tesseract()
        self.tesseract = TesseractReader(cmd, self.cfg.tesseract_langs, data)
        reader = self.tesseract
        self.reconciler = None
        if self.cfg.numeric_reader == "glyph2":
            try:
                from .digit_reader_v2 import GlyphReaderV2
                from .numeric_reconcile import Policy
                reader = GlyphReaderV2()
                self.reconciler = Policy(inject=self.cfg.numeric_inject_footer, inject_body=self.cfg.numeric_inject_body)
            except FileNotFoundError as exc:
                log.warning("glyph2 digit model missing (%s); falling back to tesseract", exc)
        elif self.cfg.numeric_reader in ("glyph", "both"):
            try:
                from .digit_reader import GlyphReader
                reader = GlyphReader()
            except FileNotFoundError as exc:
                log.warning("glyph digit model missing (%s); falling back to tesseract", exc)
        self.numeric_reader = reader
        self.validator = NumericValidator(reader, self.cfg.numeric_min_digits,
                                          self.cfg.numeric_match_threshold,
                                          extra_reader=self.tesseract if self.cfg.numeric_reader == "both" else None)

    def _reconcile(self, im: Image.Image, fields: dict) -> tuple[dict, list[NumericField]]:
        """glyph2 path: page reads -> atom-level reconciliation (numeric_reconcile)."""
        from .numeric_reconcile import reconcile
        reads = self.numeric_reader.read_page(im)
        new_fields, records = reconcile(fields, reads, im.height, self.reconciler, self.cfg.numeric_min_digits)
        out = []
        for r in records:
            out.append(NumericField(r.field, r.kind, r.value, r.value_ascii, r.confidence,
                                    tesseract_value=r.classical_value, candidates=r.candidates, bbox=r.bbox,
                                    similarity=r.similarity, note=r.note, resolved=r.resolved))
        return new_fields, out

    def start_engines(self) -> None:
        """Spawn engines synchronously — call BEFORE `asyncio.run` (see llamacpp.py)."""
        for b in (self.primary, self.secondary):
            if b is not None and hasattr(b, "start_engine"):
                b.start_engine()

    async def start(self) -> None:
        await self.primary.start()
        if self.secondary:
            await self.secondary.start()
        self.tesseract.available()

    async def stop(self) -> None:
        await self.primary.stop()
        if self.secondary:
            await self.secondary.stop()

    async def health(self) -> dict:
        models = [await self.primary.health()]
        if self.secondary:
            models.append(await self.secondary.health())
        return {"models": models,
                "tesseract": {"available": self.tesseract.available(), "version": self.tesseract.version,
                              "langs": self.cfg.tesseract_langs, "required": self.cfg.tesseract_required}}

    async def run(self, image_bytes: bytes, mime: str = "image/jpeg", doc_id: str | None = None) -> OcrResponse:
        doc_id = doc_id or uuid.uuid4().hex
        t_all = time.perf_counter()
        timing = Timing()
        prep_stats = None

        # 1. preprocess (server-side fallback for raw uploads) -----------------
        if self.cfg.preprocess_incoming:
            t0 = time.perf_counter()
            try:
                image_bytes, st = await asyncio.to_thread(
                    prepare, image_bytes, max_bytes=self.cfg.preprocess_max_bytes,
                    fmt=self.cfg.preprocess_format, max_edge=self.cfg.preprocess_max_edge)
                mime = "image/webp" if self.cfg.preprocess_format == "WEBP" else "image/jpeg"
                prep_stats = st.__dict__
            except Exception as exc:
                log.warning("preprocess failed for %s, using the raw image: %s", doc_id, exc)
                prep_stats = {"error": str(exc)}
            timing.preprocess_s = round(time.perf_counter() - t0, 3)

        # 2. primary reader (+ optional secondary, concurrently) -----------------
        t0 = time.perf_counter()
        if self.secondary:
            async def _sec():
                try:
                    return await asyncio.wait_for(self.secondary.extract(image_bytes, mime, doc_id),
                                                  self.cfg.secondary_timeout_s)
                except asyncio.TimeoutError:
                    return Extraction(ok=False, model=self.cfg.secondary.name,
                                      latency_s=self.cfg.secondary_timeout_s, error="secondary timed out")
            prim, sec = await asyncio.gather(self.primary.extract(image_bytes, mime, doc_id), _sec())
            timing.secondary_s = round(sec.latency_s, 3)
        else:
            prim, sec = await self.primary.extract(image_bytes, mime, doc_id), None
        timing.primary_s = round(prim.latency_s, 3)

        fields = normalize_extraction(prim.fields, LETTERFORM_ONLY)

        # 3. numbers: read the page classically, reconcile with the model's digits ---
        t0 = time.perf_counter()
        im = Image.open(io.BytesIO(image_bytes))
        if self.reconciler is not None:
            fields, numeric = await asyncio.to_thread(self._reconcile, im, fields)
        else:
            numeric = await asyncio.to_thread(self.validator.validate, im, fields)
        timing.numeric_validation_s = round(time.perf_counter() - t0, 3)

        # 4. cross-check with the secondary model ---------------------------------
        cross = CrossCheck(enabled=bool(self.secondary))
        if sec is not None and sec.ok:
            cross = compute_cross_check(fields, normalize_extraction(sec.fields, LETTERFORM_ONLY), numeric,
                                        min_digits=self.cfg.numeric_min_digits,
                                        policy=self.cfg.numeric_mismatch_policy, secondary_run=_run(sec))
        elif sec is not None:
            cross.secondary = _run(sec)          # ran and failed / timed out: recorded, nothing flagged

        summary = summarize(numeric)
        if self.reconciler is not None:
            summary["corrected"] = sum(n.confidence == "corrected" for n in numeric)
            summary["added"] = sum(n.confidence == "added" for n in numeric)
        # `low` alone is not a review signal on the glyph2 path: it fires on 1 of 51 dev
        # documents while 34 of 51 carry at least one wrong number (E15). `unverified`
        # — nothing on the page aligned with what the model wrote — is the weakest class
        # measured (45.5% right vs 78-81% for corrected/high), so it flags too.
        # NOTE this flag is a triage hint, NOT a correctness gate: at 71% atom recall a
        # document can be unflagged and still hold a wrong number. Callers that need a
        # per-number decision must read `numeric_fields[].confidence`.
        weak = summary["low"] + summary.get("unverified", 0)
        needs_review = (not prim.ok) or weak > 0 or cross.needs_review
        timing.total_s = round(time.perf_counter() - t_all, 3)
        return OcrResponse(
            doc_id=doc_id, fields=LetterFields(**fields),
            numeric_fields=[NumericFieldOut(**n.as_dict(), classical_value=n.tesseract_value) for n in numeric],
            numeric_summary=summary, primary=_run(prim), cross_check=cross,
            preprocess=prep_stats, timing=timing, needs_review=needs_review,
            service={"version": __version__, "backend": self.cfg.backend, "primary_model": self.cfg.primary.name,
                     "numeric_reader": getattr(self.numeric_reader, "version", None),
                     "tesseract": self.tesseract.version, "normalization": "letterforms folded, digits preserved"},
        )
