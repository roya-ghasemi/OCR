# -*- coding: utf-8 -*-
r"""FastAPI service.

    POST /ocr            image -> full pipeline -> OcrResponse
                         (or, with OCRS_ASYNC_MODE=1 or ?mode=async, a job_id)
    GET  /jobs/{job_id}  status / result of a queued job
    GET  /health         models, tesseract, queue

Run (dev):  venv312\Scripts\python.exe -m uvicorn ocr_service.api:app --port 8010
"""
from __future__ import annotations

import base64
import io
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from . import __version__
from .config import settings
from .schemas import Health, JobStatus, OcrResponse

log = logging.getLogger("ocr_service")
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")

ALLOWED = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "TIFF": "image/tiff", "BMP": "image/bmp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .pipeline import OcrPipeline
    app.state.pipeline = OcrPipeline(settings)
    if not settings.async_mode:
        app.state.pipeline.start_engines()        # synchronous spawn (see llamacpp.py)
        await app.state.pipeline.start()          # queued mode: workers own the models
    yield
    await app.state.pipeline.stop()


app = FastAPI(title="Hybrid Persian OCR", version=__version__, lifespan=lifespan)


def _sniff(data: bytes) -> str:
    if not data:
        raise HTTPException(400, "empty upload")
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, f"not a readable image: {exc}")
    if fmt not in ALLOWED:
        raise HTTPException(400, f"unsupported image format {fmt}")
    return ALLOWED[fmt]


@app.post("/ocr", response_model=None)
async def ocr(file: Optional[UploadFile] = File(None), image: Optional[UploadFile] = File(None),
              mode: Literal["sync", "async", "auto"] = Query("auto"),
              doc_id: Optional[str] = Query(None)):
    up = file or image
    if up is None:
        raise HTTPException(400, "send the image as multipart field 'file' (or 'image')")
    data = await up.read()
    mime = _sniff(data)
    doc_id = doc_id or (up.filename or uuid.uuid4().hex)

    use_async = mode == "async" or (mode == "auto" and settings.async_mode)
    if use_async:
        from .tasks import ocr_task
        job = ocr_task.apply_async(kwargs={"image_b64": base64.b64encode(data).decode(), "mime": mime, "doc_id": doc_id})
        return JSONResponse({"job_id": job.id, "status": "queued", "poll": f"/jobs/{job.id}"}, status_code=202)

    p = app.state.pipeline
    if settings.tesseract_required and not p.tesseract.available():
        raise HTTPException(503, "numeric validation requires Tesseract and it is not available")
    res: OcrResponse = await p.run(data, mime, doc_id)
    if not res.primary.ok:
        # A well-formed but empty extraction is a failure, not a document without content (D4).
        raise HTTPException(422, {"error": "extraction_failed", "detail": res.primary.error, "doc_id": doc_id})
    return res


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def job(job_id: str):
    from celery.result import AsyncResult
    from .tasks import celery_app
    r = AsyncResult(job_id, app=celery_app)
    state = r.state
    if state in ("PENDING", "RECEIVED", "RETRY"):
        return JobStatus(job_id=job_id, status="queued")
    if state == "STARTED":
        return JobStatus(job_id=job_id, status="running")
    if state == "SUCCESS":
        return JobStatus(job_id=job_id, status="done", result=OcrResponse(**r.result))
    return JobStatus(job_id=job_id, status="failed", error=str(r.result))


@app.get("/health", response_model=Health)
async def health():
    p = app.state.pipeline
    h = await p.health() if not settings.async_mode else {"models": [{"ok": None, "model": settings.primary.name,
                                                                        "detail": "async mode: models live in workers"}],
                                                          "tesseract": {"available": p.tesseract.available(),
                                                                        "version": p.tesseract.version}}
    queue: dict = {"async_mode": settings.async_mode, "broker": settings.redis_url}
    if settings.redis_url.startswith("filesystem://"):
        queue["reachable"] = True; queue["detail"] = "filesystem transport (dev fallback)"
    else:
        try:
            import redis
            rc = redis.from_url(settings.redis_url, socket_connect_timeout=1)
            rc.ping(); queue["reachable"] = True
        except Exception as exc:
            queue["reachable"] = False; queue["detail"] = str(exc)[:120]
    model_ok = all(m.get("ok") for m in h["models"]) if not settings.async_mode else queue["reachable"]
    tess_ok = h["tesseract"]["available"]
    status = "ok" if model_ok and (tess_ok or not settings.tesseract_required) else ("degraded" if model_ok else "down")
    return Health(status=status, backend=settings.backend, models=h["models"], tesseract=h["tesseract"],
                  queue=queue, service={"version": __version__, **settings.provenance()})
