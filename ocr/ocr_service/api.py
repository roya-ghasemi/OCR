# -*- coding: utf-8 -*-
r"""FastAPI service.

    GET  /                the upload page (static/index.html)
    POST /ocr            image -> full transcript (+ letter fields when it is a letter)
                         (or, with OCRS_ASYNC_MODE=1 or ?mode=async, a job_id)
    GET  /jobs/{job_id}  status / result of a queued job
    GET  /health         readiness: Tesseract, digit reader, word list, cache, queue
    GET  /healthz        liveness: the process answers (no Tesseract call)

Run (dev):  venv312\Scripts\python.exe -m uvicorn ocr_service.api:app --host 127.0.0.1 --port 8010
"""
from __future__ import annotations

import asyncio
import hmac
import io
import re
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from . import __version__
from .config import settings
from .schemas import Health, JobStatus, OcrResponse

log = logging.getLogger("ocr_service")
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")

ALLOWED = {"JPEG", "PNG", "WEBP", "TIFF", "BMP", "GIF"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .pipeline import OcrPipeline
    app.state.pipeline = OcrPipeline(settings)
    h = app.state.pipeline.health()
    if not h["tesseract"]["available"]:
        log.error("Tesseract with Persian data is NOT available: %s — /ocr will return 503", h["tesseract"])
    if not h["digit_reader"]["available"]:
        log.warning("digit reader unavailable (%s): numbers come from Tesseract only", h["digit_reader"]["error"])
    log.info("ready: tesseract %s, tessdata %s", h["tesseract"]["version"], h["tesseract"]["tessdata_dir"])
    yield


app = FastAPI(title="Persian OCR — full-text transcription", version=__version__, lifespan=lifespan)

STATIC = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """The upload page: pick an image, read the page back as it is laid out."""
    return FileResponse(STATIC / "index.html", media_type="text/html; charset=utf-8")


def require_key(x_api_key: Optional[str] = Header(None)) -> None:
    """With OCRS_API_KEY set, reading and job lookup need the key in X-API-Key."""
    if settings.api_key and not hmac.compare_digest(x_api_key or "", settings.api_key):
        raise HTTPException(401, "missing or wrong X-API-Key")


def _check_image(data: bytes) -> None:
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"image larger than {settings.max_upload_mb} MB")
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, f"not a readable image: {exc}") from exc
    if fmt not in ALLOWED:
        raise HTTPException(400, f"unsupported image format {fmt}")


@app.post("/ocr", response_model=None, dependencies=[Depends(require_key)])
async def ocr(file: Optional[UploadFile] = File(None), image: Optional[UploadFile] = File(None),
              mode: Literal["sync", "async", "auto"] = Query("auto"),
              doc_id: Optional[str] = Query(None),
              prepared: bool = Query(False, description="also return the prepared page, "
                                     "which is the image every bbox is measured in")):
    up = file or image
    if up is None:
        raise HTTPException(400, "send the image as multipart field 'file' (or 'image')")
    data = await up.read()
    _check_image(data)
    doc_id = doc_id or (up.filename or uuid.uuid4().hex)

    use_async = mode == "async" or (mode == "auto" and settings.async_mode)
    if use_async:
        from .tasks import enqueue
        job_id = uuid.uuid4().hex
        await asyncio.to_thread(enqueue, data, doc_id, job_id)
        return JSONResponse({"job_id": job_id, "status": "queued", "poll": f"/jobs/{job_id}"}, status_code=202)

    p = app.state.pipeline
    if not p.ready():
        raise HTTPException(503, "Tesseract with Persian (fas) data is not available on this server")
    res: OcrResponse = await p.run(data, doc_id, want_prepared=prepared)
    return res


@app.get("/jobs/{job_id}", response_model=JobStatus, dependencies=[Depends(require_key)])
async def job(job_id: str):
    from celery.result import AsyncResult

    from .tasks import QUEUED, celery_app
    r = AsyncResult(job_id, app=celery_app)
    state = await asyncio.to_thread(lambda: r.state)
    if state == "PENDING":                  # never enqueued here, or already expired
        raise HTTPException(404, "unknown or expired job")
    if state in (QUEUED, "RECEIVED", "RETRY"):
        return JobStatus(job_id=job_id, status="queued")
    if state == "STARTED":
        return JobStatus(job_id=job_id, status="running")
    if state == "SUCCESS":
        return JobStatus(job_id=job_id, status="done", result=OcrResponse(**r.result))
    log.error("job %s failed: %r", job_id, r.result)
    return JobStatus(job_id=job_id, status="failed", error="the page could not be processed")


@app.get("/healthz", include_in_schema=False)
async def healthz():
    return {"status": "alive"}


@app.get("/health", response_model=Health)
async def health():
    p = app.state.pipeline
    h = await asyncio.to_thread(p.health, p.HEALTH_TTL_S)
    queue: dict = {"async_mode": settings.async_mode,
                   "broker": re.sub(r"//[^@/]*@", "//***@", settings.redis_url)}   # no password
    if settings.async_mode:
        if settings.redis_url.startswith("filesystem://"):
            queue["reachable"] = True
            queue["detail"] = "filesystem transport (dev fallback)"
        else:
            try:
                import redis
                redis.from_url(settings.redis_url, socket_connect_timeout=1).ping()
                queue["reachable"] = True
            except Exception as exc:
                queue["reachable"] = False
                queue["detail"] = str(exc)[:120]
    tess_ok = h["tesseract"]["available"]
    queue_ok = queue.get("reachable", True)
    status = "ok" if tess_ok and h["digit_reader"]["available"] and queue_ok else ("degraded" if tess_ok else "down")
    return Health(status=status, tesseract=h["tesseract"], digit_reader=h["digit_reader"],
                  spellfix=h.get("spellfix", {}), cache=h.get("cache", {}), queue=queue,
                  service={"version": __version__, **settings.provenance()})
