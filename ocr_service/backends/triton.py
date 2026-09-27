# -*- coding: utf-8 -*-
"""Triton Inference Server backend — interface complete, transport deliberately thin.

Triton has no single "vision chat" protocol: the model repository defines the
input/output tensors. Assumption: a TensorRT-LLM (or Python) backend model named
`ocr_letter` that accepts `image` (bytes) and `prompt` (string) and returns `text`
(the JSON string), which is the shape the vLLM/llama.cpp paths already produce.
Anything else is a change to `_infer()` only; the rest of the service is unaffected.

Requires `tritonclient[http]` — intentionally not in the dev requirements, imported
lazily so the package loads without it.
"""
from __future__ import annotations

import json
import time
from typing import Any

from ocr_pipeline.extraction import clean_json_text, repair_json
from ocr_pipeline.grammar import SYSTEM_PROMPT, USER_TURN

from ..config import ModelSpec, Settings
from .base import Extraction, InferenceBackend

FIELDS = ["sender", "receiver", "subject", "body_text", "contact_info"]


class TritonBackend(InferenceBackend):
    name = "triton"

    def __init__(self, spec: ModelSpec, settings: Settings, base_url: str | None = None):
        self.spec, self.settings = spec, settings
        self.url = base_url or settings.triton_url
        self.model_name = "ocr_letter"
        self._client = None

    async def start(self) -> None:
        if self._client:
            return
        try:
            import tritonclient.http as th  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install 'tritonclient[http]' to use OCRS_BACKEND=triton") from exc
        self._client = th.InferenceServerClient(url=self.url)
        self._th = th

    async def stop(self) -> None:
        self._client = None

    async def health(self) -> dict[str, Any]:
        try:
            if not self._client:
                await self.start()
            ok = bool(self._client.is_server_ready() and self._client.is_model_ready(self.model_name))
            return {"ok": ok, "model": self.spec.name, "url": self.url, "triton_model": self.model_name}
        except Exception as exc:
            return {"ok": False, "model": self.spec.name, "url": self.url, "detail": str(exc)}

    def _infer(self, image_bytes: bytes, prompt: str) -> str:
        import numpy as np
        th = self._th
        img = th.InferInput("image", [1], "BYTES"); img.set_data_from_numpy(np.array([image_bytes], dtype=object))
        pr = th.InferInput("prompt", [1], "BYTES"); pr.set_data_from_numpy(np.array([prompt.encode()], dtype=object))
        out = self._client.infer(self.model_name, [img, pr], outputs=[th.InferRequestedOutput("text")])
        return out.as_numpy("text")[0].decode("utf-8", "replace")

    async def extract(self, image_bytes: bytes, mime: str, doc_id: str = "") -> Extraction:
        if not self._client:
            await self.start()
        t0 = time.perf_counter()
        try:
            raw = self._infer(image_bytes, SYSTEM_PROMPT + "\n\n" + USER_TURN)
        except Exception as exc:
            return Extraction(ok=False, model=self.spec.name, latency_s=time.perf_counter() - t0,
                              error=f"{type(exc).__name__}: {exc}")
        try:
            data = json.loads(clean_json_text(raw))
        except json.JSONDecodeError:
            data, _ = repair_json(raw)
        data = data if isinstance(data, dict) else {}
        fields = {f: (data.get(f) if isinstance(data.get(f), str) and data[f].strip() else None) for f in FIELDS}
        return Extraction(ok=any(fields.values()), model=self.spec.name, fields=fields,
                          latency_s=time.perf_counter() - t0, attempts=1)
