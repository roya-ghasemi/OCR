# -*- coding: utf-8 -*-
"""Settings for the OCR service. Every setting is an environment variable
`OCRS_<NAME>` with a documented default; `Settings().provenance()` is stamped into
every benchmark file and `/health`.

There is no model to configure: transcription is Tesseract (fas, eng) plus the
in-repo CNN digit reader (`models/digit_cnn.npz`), CPU only.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parent


def _find_up(rel: str) -> Path | None:
    """First `<ancestor>/<rel>` that exists, starting at the repo root. A git
    worktree lives inside the main checkout, so git-ignored data (tessdata, the
    digit model) is found in the main checkout without being copied."""
    for d in [PACKAGE_DIR.parent, *PACKAGE_DIR.parent.parents]:
        p = d / rel
        if p.exists():
            return p
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OCRS_", env_file=".env", extra="ignore")

    # --- recognition ------------------------------------------------------------
    tesseract_cmd: str | None = None       # None -> UB-Mannheim install, then PATH
    tessdata_dir: str | None = None        # None -> nearest <repo>/tessdata holding fas.traineddata
    digit_model: str | None = None         # None -> nearest <repo>/models/digit_cnn.npz
    layout_psms: str = "3,4,6"             # Tesseract layout passes whose lines compete (E19)
    deskew_min_deg: float = 1.0            # rotate only pages tilted at least this much (E19)
    min_line_conf: float = 30.0            # a line below this mean confidence is dropped as noise
    number_min_prob: float = 0.6           # digit reader must be this sure to replace Tesseract's digits
    workers: int = 8                       # parallel Tesseract processes per page
    letter_fields: bool = True             # also cut letter fields out of the transcript (rules only)
    spellfix: bool = True                  # repair dot confusions against the Persian word list
    max_upload_mb: float = 25.0

    # --- service ------------------------------------------------------------------
    redis_url: str = "redis://127.0.0.1:6379/0"
    async_mode: bool = False               # POST /ocr returns a job_id instead of a result
    job_ttl_s: int = 86400
    request_timeout_s: float = 600.0
    log_level: str = "INFO"

    def resolved_tesseract(self) -> tuple[str | None, str | None]:
        cmd = self.tesseract_cmd
        if not cmd:
            for c in (os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Tesseract-OCR", "tesseract.exe"),
                      os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
                      shutil.which("tesseract")):
                if c and os.path.isfile(c):
                    cmd = c
                    break
        data = self.tessdata_dir
        if not data:
            p = _find_up("tessdata/fas.traineddata")
            data = str(p.parent) if p else None
        return cmd, data

    def resolved_digit_model(self) -> Path | None:
        if self.digit_model:
            return Path(self.digit_model)
        return _find_up("models/digit_cnn.npz")

    def psms(self) -> tuple[str, ...]:
        return tuple(p.strip() for p in self.layout_psms.split(",") if p.strip())

    def provenance(self) -> dict:
        import hashlib
        import platform

        def sha(p: Path | None) -> str | None:
            return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p and p.is_file() else None

        cmd, data = self.resolved_tesseract()
        dm = self.resolved_digit_model()
        # Accuracy depends on WHICH traineddata is loaded (E19 measured fas 99e42096...,
        # tessdata_best); a distro package ships a different, weaker file.
        return {
            "service_version": __import__("ocr_service").__version__,
            "engine": "tesseract(fas,eng) + digit_cnn",
            "tesseract": {"cmd": cmd, "tessdata_dir": data,
                          "fas_sha256": sha(Path(data) / "fas.traineddata") if data else None,
                          "eng_sha256": sha(Path(data) / "eng.traineddata") if data else None},
            "digit_model": {"path": str(dm) if dm else None, "sha256": sha(dm)},
            "layout_psms": self.layout_psms, "deskew_min_deg": self.deskew_min_deg,
            "min_line_conf": self.min_line_conf, "number_min_prob": self.number_min_prob,
            "letter_fields": self.letter_fields,
            "python": platform.python_version(),
        }


settings = Settings()
