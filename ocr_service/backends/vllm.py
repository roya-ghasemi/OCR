# -*- coding: utf-8 -*-
"""vLLM backend — the production engine path.

vLLM serves the same OpenAI-compatible chat API, so the request shape is identical
to llama.cpp's. Two differences are handled here:

  * constrained decoding: vLLM has no GBNF. The equivalent is `guided_json` with the
    five-field JSON schema (and per-field `maxLength` for the D44 caps), passed via
    `extra_body`. Assumption: vLLM ≥ 0.6 with the `outlines`/`xgrammar` guided
    decoding backend.
  * sampling: vLLM has no DRY sampler; `repetition_penalty=1.1` is the closest knob.

Run vLLM with the HF weights of the same model, e.g.
    vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001 --max-model-len 16384 \
        --limit-mm-per-prompt image=1
and set OCRS_BACKEND=vllm, OCRS_VLLM_URL=http://host:8001/v1.
"""
from __future__ import annotations

import base64
import json
import time
from typing import Any

import httpx
from openai import AsyncOpenAI

from ocr_pipeline.extraction import clean_json_text, repair_json
from ocr_pipeline.grammar import DEFAULT_SPEC, SYSTEM_PROMPT, USER_TURN

from ..config import ModelSpec, Settings
from .base import Extraction, InferenceBackend

FIELDS = ["sender", "receiver", "subject", "body_text", "contact_info"]


def _json_schema(caps: dict[str, int]) -> dict:
    return {
        "type": "object",
        "properties": {f: {"type": ["string", "null"], "maxLength": caps.get(f, 1400)} for f in FIELDS},
        "required": FIELDS, "additionalProperties": False,
    }


class VllmBackend(InferenceBackend):
    name = "vllm"

    def __init__(self, spec: ModelSpec, settings: Settings, base_url: str | None = None):
        self.spec, self.settings = spec, settings
        self.base_url = base_url or settings.vllm_url
        self._client: AsyncOpenAI | None = None
        self._model_id: str | None = None
        self._schema = _json_schema(getattr(DEFAULT_SPEC, "caps", {}))

    async def start(self) -> None:
        if self._client:
            return
        self._client = AsyncOpenAI(base_url=self.base_url, api_key="local", timeout=self.settings.request_timeout_s)
        self._model_id = (await self._client.models.list()).data[0].id

    async def stop(self) -> None:
        self._client = None

    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{self.base_url}/models")
            return {"ok": r.status_code == 200, "model": self.spec.name, "served_id": self._model_id, "url": self.base_url}
        except Exception as exc:
            return {"ok": False, "model": self.spec.name, "url": self.base_url, "detail": str(exc)}

    async def extract(self, image_bytes: bytes, mime: str, doc_id: str = "") -> Extraction:
        if not self._client:
            await self.start()
        url = f"data:{mime};base64," + base64.b64encode(image_bytes).decode()
        t0 = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                model=self._model_id,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                       {"type": "text", "text": USER_TURN}]}],
                temperature=self.settings.temperature, max_tokens=self.settings.max_tokens,
                extra_body={"guided_json": self._schema, "repetition_penalty": 1.1} if self.spec.use_grammar
                           else {"repetition_penalty": 1.2},
            )
        except Exception as exc:
            return Extraction(ok=False, model=self.spec.name, latency_s=time.perf_counter() - t0,
                              error=f"{type(exc).__name__}: {exc}")
        raw = resp.choices[0].message.content or ""
        try:
            data = json.loads(clean_json_text(raw))
        except json.JSONDecodeError:
            data, _ = repair_json(raw)
        data = data if isinstance(data, dict) else {}
        fields = {f: (data.get(f) if isinstance(data.get(f), str) and data[f].strip() else None) for f in FIELDS}
        return Extraction(ok=any(fields.values()), model=self.spec.name, fields=fields,
                          latency_s=time.perf_counter() - t0, attempts=1,
                          finish_reason=resp.choices[0].finish_reason, grammar_used=self.spec.use_grammar)
