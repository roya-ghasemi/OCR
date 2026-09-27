# -*- coding: utf-8 -*-
"""Settings for the hybrid service. Everything is an environment variable with a
documented default; `Settings().provenance()` is what goes into every benchmark file.

Model paths are the ones given in the brief, verbatim.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

QWEN25_DIR = Path(r"D:\models\models\lmstudio-community\Qwen2.5-VL-7B-Instruct-GGUF")
COREOCR_DIR = Path(r"D:\models\models\mradermacher\coreOCR-7B-050325-preview-GGUF")


class ModelSpec(BaseSettings):
    """One GGUF vision model as served by llama-server (or, in production, by the
    backend chosen in `Settings.backend` — the same `name` is what vLLM/Triton get)."""
    name: str
    gguf: Path
    mmproj: Path | None = None
    port: int = 18234
    # Assumption: the grammar + DRY sampler profile measured on coreOCR (D39) is
    # used for Qwen2.5-VL as well until measured otherwise. Both are Qwen2-VL-family
    # models on the same engine build.
    use_grammar: bool = True


PRIMARY_DEFAULT = ModelSpec(
    name="Qwen2.5-VL-7B-Instruct-Q4_K_M",
    gguf=QWEN25_DIR / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
    mmproj=QWEN25_DIR / "mmproj-model-f16.gguf",
    port=18236,
)
SECONDARY_DEFAULT = ModelSpec(
    name="coreOCR-7B-050325-preview-Q4_K_S",
    gguf=COREOCR_DIR / "coreOCR-7B-050325-preview.Q4_K_S.gguf",
    mmproj=COREOCR_DIR / "coreOCR-7B-050325-preview.mmproj-f16.gguf",
    port=18234,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OCRS_", env_file=".env", extra="ignore")

    # --- models -------------------------------------------------------------
    primary: ModelSpec = PRIMARY_DEFAULT
    secondary: ModelSpec = SECONDARY_DEFAULT
    enable_secondary_model: bool = Field(
        default=False,
        description=("Run coreOCR alongside Qwen2.5-VL and cross-check numeric fields. "
                     "Assumption: OFF by default — the dev box has 12 GB VRAM and two 7B "
                     "Q4 models with 16k context do not fit together; enable on a box with "
                     "two GPUs or ≥24 GB."))
    numeric_mismatch_policy: Literal["flag", "prefer_primary", "prefer_secondary"] = "flag"

    # --- inference backend ---------------------------------------------------
    backend: Literal["llamacpp", "vllm", "triton"] = "llamacpp"
    backend_url_primary: str | None = None      # set to attach to an external server
    backend_url_secondary: str | None = None
    vllm_url: str = "http://127.0.0.1:8001/v1"
    triton_url: str = "127.0.0.1:8002"
    autostart_engines: bool = True               # llamacpp only: spawn llama-server if not listening
    engine_context: int = 16384
    engine_n_gpu_layers: int = 99
    engine_image_min_tokens: int = 1024
    # D54: the engine's default cap is ~4k image tokens (a 3016x2304 scan reaches the
    # model at 0.67 scale). 16384 lets it through at native resolution — measured on
    # the smoke run: 9.3k tokens, prompt eval ~100 tok/s, 90-120 s per page instead
    # of ~10 s (the vision encoder is quadratic in patches). Assumption: keep the
    # comparable 4k cap as default; the benchmark measures 16384 on a subset.
    engine_image_max_tokens: int = 4096
    engine_min_free_vram_mb: int = 9000           # below this an engine starts on CPU instead of OOM-ing
    # How the secondary (cross-check) model shares one 12 GB GPU with the primary:
    #   concurrent  both on GPU (needs ≥24 GB / two GPUs)
    #   cpu_offload secondary runs with -ngl 0 in system RAM alongside the GPU primary (slow, online)
    #   sequential  batch only: run primary over the batch, swap models, run secondary (benchmark)
    secondary_mode: Literal["concurrent", "cpu_offload", "sequential"] = "cpu_offload"
    secondary_n_gpu_layers: int = 0
    secondary_timeout_s: float = 900.0            # cpu_offload is slow; don't let it hold the response forever

    # --- decoding ------------------------------------------------------------
    temperature: float = 0.0
    cache_prompt: bool = False                    # D52: session-dependent bodies with prompt cache on
    max_tokens: int = 2048

    # --- numeric validation ------------------------------------------------------
    # Which classical reader handles numbers. `glyph2` (default, E15) = the CNN glyph
    # reader with a reject class (models/digit_cnn.npz) + atom-level reconciliation
    # that CORRECTS the VLM's digits and adds footer numbers it omitted;
    # `glyph` = the v1 HOG-SVM reader (validation only, D55/D56); `tesseract` =
    # fas+eng; `both` = v1 glyph decides, tesseract adds a candidate.
    numeric_reader: Literal["glyph2", "glyph", "tesseract", "both"] = "glyph2"
    numeric_inject_footer: bool = True           # glyph2: add confident footer numbers the VLM omitted to contact_info
    numeric_inject_body: bool = True             # glyph2: add confident body-band numbers as records (never into the prose)
    tesseract_cmd: str | None = None             # None -> auto-detect the UB-Mannheim install, else PATH
    tessdata_dir: str | None = None              # None -> <repo>/tessdata if it holds fas.traineddata, else the install's
    tesseract_langs: str = "fas+eng"
    tesseract_required: bool = False             # True -> /ocr returns 503 when tesseract is missing
    numeric_min_digits: int = 3
    numeric_match_threshold: float = 1.0         # exact digit-string match = high confidence

    # --- preprocessing --------------------------------------------------------
    preprocess_incoming: bool = True
    preprocess_max_bytes: int = 300_000
    preprocess_format: Literal["WEBP", "JPEG"] = "JPEG"   # Assumption: JPEG — llama.cpp's stb loader
                                                         # has no WebP decoder; WebP is for the mobile
                                                         # client → gateway hop and is transcoded here.
    preprocess_max_edge: int = 3200

    # --- service --------------------------------------------------------------
    redis_url: str = "redis://127.0.0.1:6379/0"
    async_mode: bool = False                      # POST /ocr returns a job_id instead of a result
    job_ttl_s: int = 86400
    request_timeout_s: float = 600.0
    log_level: str = "INFO"

    def resolved_tesseract(self) -> tuple[str | None, str | None]:
        """(tesseract.exe, tessdata dir) after auto-detection. Program Files is not
        writable without elevation, so the Persian data lives in <repo>/tessdata."""
        import os, shutil
        cmd = self.tesseract_cmd
        if not cmd:
            for c in (os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Tesseract-OCR", "tesseract.exe"),
                      os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
                      shutil.which("tesseract")):
                if c and os.path.isfile(c):
                    cmd = c; break
        data = self.tessdata_dir
        if not data:
            local = Path(__file__).resolve().parents[1] / "tessdata"
            if (local / "fas.traineddata").is_file():
                data = str(local)
        return cmd, data

    @staticmethod
    def _digit_cnn_sha() -> str | None:
        import hashlib, pathlib
        p = pathlib.Path(__file__).resolve().parents[1] / "models" / "digit_cnn.npz"
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.is_file() else None

    def provenance(self) -> dict:
        import platform
        return {
            "service_version": __import__("ocr_service").__version__,
            "backend": self.backend,
            "primary": {"name": self.primary.name, "gguf": str(self.primary.gguf),
                        "mmproj": str(self.primary.mmproj), "grammar": self.primary.use_grammar},
            "secondary": {"name": self.secondary.name, "enabled": self.enable_secondary_model,
                          "gguf": str(self.secondary.gguf)},
            "temperature": self.temperature, "cache_prompt": self.cache_prompt,
            "max_tokens": self.max_tokens,
            "engine": {"context": self.engine_context, "n_gpu_layers": self.engine_n_gpu_layers,
                       "image_min_tokens": self.engine_image_min_tokens,
                       "image_max_tokens": self.engine_image_max_tokens},
            "numeric": {"reader": self.numeric_reader, "inject_footer": self.numeric_inject_footer,
                        "inject_body": self.numeric_inject_body, "digit_cnn_sha256": self._digit_cnn_sha()},
            "tesseract": {"langs": self.tesseract_langs, "cmd": self.resolved_tesseract()[0],
                          "tessdata_dir": self.resolved_tesseract()[1]},
            "preprocess": {"enabled": self.preprocess_incoming, "max_bytes": self.preprocess_max_bytes,
                           "format": self.preprocess_format, "max_edge": self.preprocess_max_edge},
            "python": platform.python_version(),
        }


settings = Settings()
