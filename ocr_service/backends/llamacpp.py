# -*- coding: utf-8 -*-
"""llama.cpp backend: one `llama-server` process per model, spoken to over the
OpenAI-compatible API, decoding through `ocr_pipeline.LetterExtractor` (GBNF grammar
+ DRY sampler — the configuration that measured best on the real dev split, E12).

The server binary is the same one the legacy path uses (`engine.find_backend()`,
pinned build 2.28.2, D48). Model/mmproj/port come from `ModelSpec`, so two models
can be served side by side on different ports.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI

from ocr_pipeline.extraction import LetterExtractor
from ocr_pipeline.grammar import DEFAULT_SPEC
from ocr_pipeline.sampling import BELT_AND_BRACES, PROVEN

from ..config import ModelSpec, Settings
from .base import Extraction, InferenceBackend

log = logging.getLogger(__name__)
FIELDS = ["sender", "receiver", "subject", "body_text", "contact_info"]


def _port_open(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def free_vram_mb() -> int | None:
    """Free VRAM on GPU 0 via nvidia-smi, or None if it cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return int(out[0]) if out else None
    except Exception:
        return None


def _server_supports(exe: Path, env: dict, flag: str) -> bool:
    try:
        out = subprocess.run([str(exe), "--help"], capture_output=True, text=True,
                             env=env, cwd=str(exe.parent), timeout=30).stdout
        return flag in out
    except Exception:
        return False


class LlamaServerProcess:
    """Owns one llama-server child. Reuses an already-listening port (e.g. the legacy
    engine on :18234) instead of double-loading a model into VRAM."""

    def __init__(self, spec: ModelSpec, settings: Settings, host: str = "127.0.0.1",
                 n_gpu_layers: int | None = None):
        self.spec, self.settings, self.host = spec, settings, host
        self.n_gpu_layers = settings.engine_n_gpu_layers if n_gpu_layers is None else n_gpu_layers
        self.process: subprocess.Popen | None = None
        self.adopted = False
        self.log_path = Path(f"engine_{spec.port}.log")

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.spec.port}/v1"

    def start(self) -> None:
        if _port_open(self.host, self.spec.port):
            self.adopted = True
            log.info("llama-server already listening on :%s — adopting it", self.spec.port)
            return
        import engine  # legacy module: find_backend() knows where the pinned binary lives
        exe, vendor = engine.find_backend()
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join([str(exe.parent)] + ([str(vendor)] if vendor else []) + [env.get("PATH", "")])
        ngl = self.n_gpu_layers
        free_mb = free_vram_mb()
        # VRAM guard: a 7B Q4 with 16k context needs ~8.5 GB on the GPU. If that is
        # not free, offload to CPU instead of letting CUDA OOM take the process down.
        if ngl > 0 and free_mb is not None and free_mb < self.settings.engine_min_free_vram_mb:
            log.warning("only %d MB VRAM free (< %d needed): starting %s with -ngl 0 (CPU offload). "
                        "Expect minutes per image.", free_mb, self.settings.engine_min_free_vram_mb, self.spec.name)
            ngl = 0
        self.effective_n_gpu_layers = ngl
        cmd = [str(exe), "-m", str(self.spec.gguf), "--host", self.host, "--port", str(self.spec.port),
               "-ngl", str(ngl), "-c", str(self.settings.engine_context)]
        if ngl == 0:
            cmd += ["-t", str(max(4, (os.cpu_count() or 8) - 2))]
        if self.spec.mmproj:
            cmd += ["--mmproj", str(self.spec.mmproj),
                    "--image-min-tokens", str(self.settings.engine_image_min_tokens)]
            # Assumption: only pass --image-max-tokens when this build advertises it;
            # older builds reject unknown flags and never come up.
            if _server_supports(exe, env, "--image-max-tokens"):
                cmd += ["--image-max-tokens", str(self.settings.engine_image_max_tokens)]
            else:
                log.warning("this llama-server build has no --image-max-tokens; running at the default cap (D54)")
        log.info("starting llama-server for %s on :%s", self.spec.name, self.spec.port)
        fh = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.process = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                        cwd=str(exe.parent),
                                        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited early; see {self.log_path}")
            try:
                if httpx.get(f"{self.base_url}/models", timeout=2).status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1.0)
        raise TimeoutError(f"llama-server on :{self.spec.port} did not become ready")

    def stop(self) -> None:
        if self.process and not self.adopted:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None


class LlamaCppBackend(InferenceBackend):
    name = "llamacpp"

    def __init__(self, spec: ModelSpec, settings: Settings, base_url: str | None = None,
                 n_gpu_layers: int | None = None):
        self.spec, self.settings = spec, settings
        self.n_gpu_layers = n_gpu_layers          # None -> settings.engine_n_gpu_layers
        self._explicit_url = base_url
        self._proc: LlamaServerProcess | None = None
        self._client: AsyncOpenAI | None = None
        self._extractor: LetterExtractor | None = None
        self._model_id: str | None = None

    @property
    def base_url(self) -> str:
        return self._explicit_url or (self._proc.base_url if self._proc else f"http://127.0.0.1:{self.spec.port}/v1")

    def start_engine(self) -> None:
        """Spawn (or adopt) the llama-server. SYNCHRONOUS on purpose.

        Spawning the child from a worker thread while a Windows ProactorEventLoop
        is running aborted the loop's self-pipe (`ConnectionAbortedError`, WinError
        1236) and killed the first benchmark run. Call this before `asyncio.run`, or
        from the loop thread itself; never via `asyncio.to_thread`.
        """
        if self._proc or self._explicit_url or not self.settings.autostart_engines:
            return
        self._proc = LlamaServerProcess(self.spec, self.settings,
                                        n_gpu_layers=self.n_gpu_layers)
        self._proc.start()

    async def start(self) -> None:
        if self._extractor:
            return
        self.start_engine()                      # no-op if already started synchronously
        self._client = AsyncOpenAI(base_url=self.base_url, api_key="local",
                                   timeout=self.settings.request_timeout_s)
        models = await self._client.models.list()
        self._model_id = models.data[0].id
        # Grammar + DRY when the grammar is on; the "proven" no-grammar profile
        # (repeat_penalty=1.2) otherwise — D39 showed the two must move together.
        self._extractor = LetterExtractor(
            self._client, self._model_id,
            grammar_spec=DEFAULT_SPEC if self.spec.use_grammar else None,
            sampling=BELT_AND_BRACES if self.spec.use_grammar else PROVEN,
            # D61: config declared `cache_prompt` and provenance stamped it, but
            # nothing ever put it on the wire. llama.cpp defaults it to true.
            extra_body={"cache_prompt": self.settings.cache_prompt},
        )

    def stop_engine(self) -> None:
        if self._proc:
            self._proc.stop()
            self._proc = None

    async def stop(self) -> None:
        self.stop_engine()
        self._extractor = self._client = None

    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{self.base_url}/models")
            ok = r.status_code == 200
            return {"ok": ok, "model": self.spec.name, "served_id": self._model_id,
                    "url": self.base_url, "grammar": self.spec.use_grammar}
        except Exception as exc:
            return {"ok": False, "model": self.spec.name, "url": self.base_url, "detail": str(exc)}

    async def extract(self, image_bytes: bytes, mime: str, doc_id: str = "") -> Extraction:
        if not self._extractor:
            await self.start()
        url = f"data:{mime};base64," + base64.b64encode(image_bytes).decode()
        t0 = time.perf_counter()
        try:
            res = await self._extractor.extract(url, doc_id=doc_id)
        except Exception as exc:
            return Extraction(ok=False, model=self.spec.name, latency_s=time.perf_counter() - t0,
                              error=f"{type(exc).__name__}: {exc}")
        data = res.data if isinstance(res.data, dict) else {}
        return Extraction(
            ok=res.ok, model=self.spec.name,
            fields={f: (data.get(f) if isinstance(data.get(f), str) and data[f].strip() else None) for f in FIELDS},
            latency_s=time.perf_counter() - t0, attempts=res.attempts,
            finish_reason=res.finish_reason, grammar_used=res.grammar_used, error=res.error,
            raw={"quality": res.validation.as_dict() if res.validation else None,
                 "repaired": res.repaired, "partial": res.partial},
        )
