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

from celery import Celery

from .config import settings

log = logging.getLogger(__name__)

def _broker_and_backend(url: str) -> tuple[str, str, dict]:
    """Redis in production. `filesystem://<dir>` is a zero-dependency dev fallback
    (kombu's filesystem transport) for boxes with no Redis/Docker — same task flow,
    same job polling, no external service. Assumption: single host only."""
    if url.startswith("filesystem://"):
        import os
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


@celery_app.task(name="ocr_service.ocr", bind=True)
def ocr_task(self, image_b64: str, doc_id: str | None = None, mime: str | None = None) -> dict:
    # `mime` is accepted and ignored so jobs queued by an older API still run
    self.update_state(state="STARTED")
    return get_pipeline().run_sync(base64.b64decode(image_b64), doc_id).model_dump()
