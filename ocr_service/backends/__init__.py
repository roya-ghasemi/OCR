# -*- coding: utf-8 -*-
"""Backend factory. `OCRS_BACKEND` picks the implementation; nothing else changes."""
from __future__ import annotations

from ..config import ModelSpec, Settings
from .base import Extraction, InferenceBackend

__all__ = ["Extraction", "InferenceBackend", "make_backend"]


def make_backend(spec: ModelSpec, settings: Settings, base_url: str | None = None) -> InferenceBackend:
    if settings.backend == "llamacpp":
        from .llamacpp import LlamaCppBackend
        return LlamaCppBackend(spec, settings, base_url)
    if settings.backend == "vllm":
        from .vllm import VllmBackend
        return VllmBackend(spec, settings, base_url)
    if settings.backend == "triton":
        from .triton import TritonBackend
        return TritonBackend(spec, settings, base_url)
    raise ValueError(f"unknown backend {settings.backend!r}")
