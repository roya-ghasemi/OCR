# -*- coding: utf-8 -*-
"""Celery app. Broker and result backend are the same Redis (`OCRS_REDIS_URL`).

One `OcrPipeline` per worker process, created lazily on first task.

Windows dev note: Celery's prefork pool does not work on Windows; run
    celery -A ocr_service.tasks worker --pool=solo -l info
Production (Linux, docker-compose): default prefork. Each page already runs up to
`OCRS_WORKERS` Tesseract processes in parallel, so `--concurrency` x `OCRS_WORKERS`
should not exceed the CPU cores.
"""
from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

from celery import Celery

from .config import settings

log = logging.getLogger(__name__)

# State written when a job is enqueued, so /jobs can tell a queued job from an unknown id
# (Celery reports both as PENDING).
QUEUED = "QUEUED"


def _broker_and_backend(url: str) -> tuple[str, str, dict]:
    """Redis in production. `filesystem://<dir>` is a single-host fallback (kombu's
    filesystem transport) for development machines without Redis."""
    if url.startswith("filesystem://"):
        base = url[len("filesystem://"):] or ".celery_fs"
        for d in ("in", "out", "processed", "results"):
            os.makedirs(os.path.join(base, d), exist_ok=True)
        opts = {"data_folder_in": os.path.join(base, "in"), "data_folder_out": os.path.join(base, "in"),
                "processed_folder": os.path.join(base, "processed"), "store_processed": False}
        return "filesystem://", "file://" + os.path.join(base, "results"), opts
    return url, url, {}


_broker, _backend, _transport_opts = _broker_and_backend(settings.redis_url)
celery_app = Celery("ocr_service", broker=_broker, backend=_backend)
celery_app.conf.update(
    broker_transport_options=_transport_opts,
    task_serializer="json", result_serializer="json", accept_content=["json"],
    result_expires=settings.job_ttl_s, task_track_started=True,
    worker_prefetch_multiplier=1,      # one image at a time per worker: CPU-bound
    task_acks_late=True,
    task_time_limit=int(settings.request_timeout_s) + 60,
)

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from .pipeline import OcrPipeline
        _pipeline = OcrPipeline(settings)
        _pipeline.health()
    return _pipeline


def spool(data: bytes, job_id: str) -> str | None:
    """Write an upload where the workers can read it, so the image does not travel through
    Redis. Returns None when no spool directory is configured."""
    if not settings.spool_dir:
        return None
    d = Path(settings.spool_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{job_id}.img"
    tmp = p.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(p)
    return str(p)


def enqueue(data: bytes, doc_id: str, job_id: str) -> None:
    celery_app.backend.store_result(job_id, None, QUEUED)
    path = spool(data, job_id)
    kwargs = {"image_path": path} if path else {"image_b64": base64.b64encode(data).decode()}
    ocr_task.apply_async(kwargs={**kwargs, "doc_id": doc_id}, task_id=job_id)


@celery_app.task(name="ocr_service.ocr", bind=True)
def ocr_task(self, image_b64: str | None = None, doc_id: str | None = None,
             image_path: str | None = None, mime: str | None = None) -> dict:
    # `mime` is accepted and ignored so jobs queued by an older API still run
    self.update_state(state="STARTED")
    if image_path:
        p = Path(image_path)
        data = p.read_bytes()
    else:
        p, data = None, base64.b64decode(image_b64 or "")
    try:
        return get_pipeline().run_sync(data, doc_id).model_dump(mode="json")
    finally:
        if p is not None:
            p.unlink(missing_ok=True)
