"""FastAPI router - drop-in replacement for the current `/ocr` endpoint.

Mount it:

    from ocr_pipeline.api import router, init_pipeline

    @asynccontextmanager
    async def lifespan(app):
        engine.start()
        init_pipeline(AsyncOpenAI(base_url=engine.base_url, api_key="local",
                                 timeout=600.0), model=MODEL_NAME)
        yield
        engine.stop()

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)

RESPONSE CONTRACT CHANGE - READ BEFORE DEPLOYING
------------------------------------------------
The old endpoint returned a bare 5-field object, so a caller could not tell a
clean read from a partial one from a fabrication. This returns the extraction
under `extraction`, plus `quality` and `field_confidence`. That is a BREAKING
change and it is the point: the previous shape made the dangerous failure
undetectable by construction.

`/ocr/legacy` serves the old bare shape for callers that cannot migrate yet. It
carries the same guardrails and returns 422 on a rejected extraction, so a
fabrication still cannot reach a caller as a silent 200.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from .extraction import ExtractionResult, LetterExtractor
from .grammar import FIELD_ORDER
from .validation import Verdict

logger = logging.getLogger(__name__)
router = APIRouter()

_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg", "PNG": "image/png", "BMP": "image/bmp",
    "TIFF": "image/tiff", "WEBP": "image/webp",
}

# Input validation. A 40 MB TIFF or a 60-megapixel phone photo will exhaust the
# context window and time out rather than fail fast, so it is rejected up front.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 25 * 1024 * 1024))
MAX_PIXELS = int(os.getenv("MAX_PIXELS", 40_000_000))
# Below this, text is not reliably legible and the read is not worth its latency.
MIN_EDGE_PX = int(os.getenv("MIN_EDGE_PX", 600))

# Full model output on a parse failure goes here, never to the shared log: these
# are real letters carrying names, national IDs and bank details.
RAW_DUMP_DIR = os.getenv("RAW_DUMP_DIR", "debug_raw")

_extractor: LetterExtractor | None = None


def init_pipeline(client, model: str, **kwargs) -> LetterExtractor:
    """Build the extractor once at startup."""
    global _extractor
    _extractor = LetterExtractor(
        client, model, on_raw_failure=_dump_raw_output, **kwargs
    )
    return _extractor


def _pipeline() -> LetterExtractor:
    if _extractor is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised.")
    return _extractor


def _dump_raw_output(doc_id: str, raw: str) -> None:
    """Persist complete model output for one failure. Never raises."""
    try:
        os.makedirs(RAW_DUMP_DIR, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", doc_id or "unknown")[:80]
        path = os.path.join(RAW_DUMP_DIR, f"{safe}.{int(time.time())}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"doc_id": doc_id, "raw_len": len(raw), "raw_output": raw},
                      fh, ensure_ascii=False, indent=2)
        logger.error("Full model output for '%s' written to %s", doc_id, path)
    except Exception:
        logger.exception("Could not write raw-output dump for '%s'", doc_id)


# ---------------------------------------------------------------------------
# response models
# ---------------------------------------------------------------------------

class Extraction(BaseModel):
    sender: Optional[str] = None
    receiver: Optional[str] = None
    subject: Optional[str] = None
    body_text: Optional[str] = None
    contact_info: Optional[str] = None


class OcrResponse(BaseModel):
    extraction: Extraction
    quality: dict = Field(
        ..., description="verdict, confidence, findings, identifier checksums"
    )
    field_confidence: dict = Field(
        ..., description="per-field score in [0,1] for routing to human review"
    )
    meta: dict


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------

def _validate_image(upload: UploadFile, data: bytes) -> str:
    if not data:
        raise HTTPException(status_code=400, detail={
            "error": "empty_file", "message": "Uploaded file is empty."})
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail={
            "error": "file_too_large",
            "message": f"File is {len(data)} bytes; the limit is {MAX_UPLOAD_BYTES}.",
        })
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        w, h = img.size
        n_frames = getattr(img, "n_frames", 1)
        img.verify()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail={
            "error": "unreadable_image",
            "message": (f"'{upload.filename}' is not a readable image. Accepted: "
                        f"{', '.join(sorted(_FORMAT_TO_MIME))}."),
        })
    except Exception as exc:
        raise HTTPException(status_code=400, detail={
            "error": "unreadable_image", "message": f"Cannot decode image: {exc}"})

    mime = _FORMAT_TO_MIME.get(fmt or "")
    if mime is None:
        raise HTTPException(status_code=400, detail={
            "error": "unsupported_format",
            "message": (f"Unsupported format '{fmt}'. Accepted: "
                        f"{', '.join(sorted(_FORMAT_TO_MIME))}."),
        })
    if w * h > MAX_PIXELS:
        raise HTTPException(status_code=413, detail={
            "error": "image_too_large",
            "message": f"{w}x{h} exceeds the {MAX_PIXELS}-pixel limit.",
        })
    if min(w, h) < MIN_EDGE_PX:
        raise HTTPException(status_code=422, detail={
            "error": "resolution_too_low",
            "message": (f"{w}x{h}: the short edge is under {MIN_EDGE_PX}px. Text is "
                        "not reliably legible at this resolution."),
        })
    if n_frames > 1:
        # Silently reading page 1 of a multi-page fax and returning it as "the
        # document" loses pages with no signal to the caller.
        raise HTTPException(status_code=422, detail={
            "error": "multi_page_unsupported",
            "message": f"{n_frames} pages found. Split and submit one page per request.",
        })
    return mime


def _pick(file: Optional[UploadFile], image: Optional[UploadFile]) -> UploadFile:
    up = file or image
    if up is None:
        raise HTTPException(status_code=400, detail={
            "error": "no_file",
            "message": "Send a multipart form field named 'file' (or 'image').",
        })
    return up


async def _run(upload: UploadFile) -> ExtractionResult:
    data = await upload.read()
    mime = _validate_image(upload, data)
    url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
    return await _pipeline().extract(url, doc_id=upload.filename or "upload")


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------

@router.post("/ocr", response_model=OcrResponse)
async def ocr(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
):
    """Extract the five fields, with quality signals attached.

    200 means a usable extraction was produced. It does NOT mean the text is
    correct - read `quality.verdict` and `field_confidence` before acting on any
    field. `verdict == "review"` means a guardrail fired and a human should look.
    """
    result = await _run(_pick(file, image))

    if not result.ok:
        v = result.validation
        raise HTTPException(status_code=422, detail={
            "error": "rejected_extraction" if v else "model_failure",
            "message": (result.error or
                        "The extraction failed structural validation and was "
                        "withheld rather than returned as a successful read."),
            "quality": v.as_dict() if v else None,
            "meta": {"attempts": result.attempts,
                     "finish_reason": result.finish_reason,
                     "grammar_used": result.grammar_used},
        })

    payload = result.as_dict()
    if result.validation and result.validation.verdict is Verdict.REVIEW:
        logger.warning(
            "Extraction flagged for review: %s",
            [f.code for f in result.validation.findings],
        )
    return OcrResponse(
        extraction=Extraction(**result.data),
        quality=payload["quality"],
        field_confidence=payload["field_confidence"],
        meta=payload["meta"],
    )


@router.post("/ocr/legacy")
async def ocr_legacy(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
):
    """Old bare 5-field shape, for callers that cannot migrate yet.

    Guardrails still apply: a rejected extraction returns 422 rather than
    reaching the caller as a silent 200. Quality signals are unavailable in this
    shape, which is the reason to migrate.
    """
    result = await _run(_pick(file, image))
    if not result.ok:
        raise HTTPException(status_code=422, detail={
            "error": "rejected_extraction",
            "message": result.error or "Extraction failed structural validation.",
        })
    return {k: result.data.get(k) for k in FIELD_ORDER}


@router.get("/ocr/health")
async def ocr_health():
    p = _pipeline()
    return {
        "status": "ready",
        "model": p.model,
        "grammar_active": bool(p.grammar_spec) and not p._grammar_disabled,
        "sampling_profile": p.sampling.name,
        "max_retries": p.max_retries,
    }
