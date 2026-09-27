# -*- coding: utf-8 -*-
"""The inference-backend interface. Business logic (pipeline.py) only ever sees this.

Switching engines is a config change (`OCRS_BACKEND=llamacpp|vllm|triton`); nothing
above this module imports a concrete backend.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Extraction:
    """What a backend returns for one image. `fields` is the five-slot contract."""
    ok: bool
    model: str
    fields: dict[str, str | None] = field(default_factory=dict)
    latency_s: float = 0.0
    attempts: int = 0
    finish_reason: str | None = None
    grammar_used: bool = False
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)   # backend-specific diagnostics


class InferenceBackend(abc.ABC):
    """One loaded vision-language model behind a uniform contract.

    Implementations must be safe to call concurrently from async code and must not
    hold per-request state — Celery workers instantiate one per process.
    """

    name: str = "abstract"

    @abc.abstractmethod
    async def start(self) -> None:
        """Make the model reachable (spawn a server, open a client). Idempotent."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Release what `start` acquired. Idempotent."""

    @abc.abstractmethod
    async def health(self) -> dict[str, Any]:
        """{'ok': bool, 'model': str, 'detail': ...} — never raises."""

    @abc.abstractmethod
    async def extract(self, image_bytes: bytes, mime: str, doc_id: str = "") -> Extraction:
        """Read one image into the five-field contract."""

    # Optional capability: models that can ground text. Default: none.
    async def locate(self, image_bytes: bytes, mime: str, query: str) -> list[tuple[int, int, int, int]]:
        """Bounding boxes (x0,y0,x1,y1) in image pixels for `query`, or [] if unsupported."""
        return []
