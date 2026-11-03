# -*- coding: utf-8 -*-
"""Settings for the OCR service. Every setting is an environment variable
`OCRS_<NAME>` with a documented default; `Settings().provenance()` is stamped into
every benchmark file and `/health`.

There is no model to configure: transcription is Tesseract (fas, eng) plus the
in-repo CNN digit reader (`models/digit_cnn.npz`), CPU only.
"""
from __future__ import annotations

import hashlib
import os
import platform
import shutil
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from . import __version__

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent
# Every model the service loads lives under models/: the Tesseract traineddata in
# models/tesseract/, the digit reader in models/digit_cnn.npz.
TESSERACT_MODELS = "models/tesseract"


def _find_up(rel: str) -> Path | None:
    """First `<ancestor>/<rel>` that exists, starting at the repo root. A git
    worktree lives inside the main checkout, so git-ignored data (Tesseract models, the
    digit model) is found in the main checkout without being copied."""
    for d in [PACKAGE_DIR.parent, *PACKAGE_DIR.parent.parents]:
        p = d / rel
        if p.exists():
            return p
    return None


@lru_cache(maxsize=32)
def _sha_cached(path: str, mtime_ns: int, size: int) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def file_sha(p: Path | None) -> str | None:
    """Short sha256 of a model file, computed once per file version (path, mtime, size):
    provenance() runs on every request because the cache key depends on it."""
    if not p or not p.is_file():
        return None
    st = p.stat()
    return _sha_cached(str(p), st.st_mtime_ns, st.st_size)


def repo_relative(path: str | os.PathLike | None) -> str | None:
    """A path inside the repository as repository-relative, so records and responses do
    not carry one machine's directory layout; anything else unchanged."""
    if not path:
        return None
    try:
        return Path(path).resolve().relative_to(REPO_DIR).as_posix()
    except ValueError:
        return str(path)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OCRS_", env_file=(REPO_DIR / ".env", ".env"),
                                      env_file_encoding="utf-8", extra="ignore")

    # --- recognition ------------------------------------------------------------
    tesseract_cmd: str | None = None       # None -> UB-Mannheim install, then PATH
    tessdata_dir: str | None = None        # None -> <repo>/models/tesseract
    lang_fa: str = "fas_custom"            # the E30 fine-tune; "fas" is the stock model
    lang_ar: str = "ara"                   # re-read a page that looks Arabic; "" disables it
    digit_model: str | None = None         # None -> nearest <repo>/models/digit_cnn.npz
    layout_psms: str = "3,4,6"             # Tesseract layout passes whose lines compete (E19)
    text_psms: str = "11"                  # passes whose lines compete only where they hold no digit (E38)
    crop_page: bool = True                 # cut the sheet out of the photo before reading (E26)
    ink_separate: bool = True              # erase coloured stamps, invert white-on-colour bands (E26)
    deskew_min_deg: float = 1.0            # rotate only pages tilted at least this much (E19)
    min_line_conf: float = 30.0            # a line below this mean confidence is dropped as noise
    number_min_prob: float = 0.6           # digit reader must be this sure to replace Tesseract's digits
    workers: int = 8                       # parallel Tesseract processes per page
    max_concurrent_pages: int = 2          # pages read at once per API process (sync mode)
    letter_fields: bool = True             # also cut letter fields out of the transcript (rules only)
    spellfix: bool = True                  # repair dot confusions against the Persian word list
    max_upload_mb: float = 25.0

    # --- cache --------------------------------------------------------------------
    cache: bool = True                     # reuse a transcript for a page already read
    cache_dir: str | None = None           # None -> <repo>/.cache/ocr
    cache_max_entries: int = 2000

    # --- service ------------------------------------------------------------------
    api_key: str | None = None             # when set, /ocr and /jobs require header X-API-Key
    redis_url: str = "redis://127.0.0.1:6379/0"
    async_mode: bool = False               # POST /ocr returns a job_id instead of a result
    spool_dir: str | None = None           # uploads handed to workers as files, not through Redis
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
            # Look for the model that will actually be used, and only then for the stock one.
            # Searching for `fas.traineddata` alone was right while that was the default; now
            # that it is `fas_custom`, a checkout carrying the fine-tune but not the stock
            # model found no tessdata directory at all, fell back to the Tesseract install's,
            # and failed there because the fine-tune is not in it — a default that cannot
            # load, reported as a missing language rather than a missing directory.
            p = (_find_up(f"{TESSERACT_MODELS}/{self.lang_fa}.traineddata")
                 or _find_up(f"{TESSERACT_MODELS}/fas.traineddata"))
            data = str(p.parent) if p else None
        return cmd, data

    def resolved_digit_model(self) -> Path | None:
        if self.digit_model:
            return Path(self.digit_model)
        return _find_up("models/digit_cnn.npz")

    def psms(self) -> tuple[str, ...]:
        return tuple(p.strip() for p in self.layout_psms.split(",") if p.strip())

    def text_only_psms(self) -> tuple[str, ...]:
        return tuple(p.strip() for p in self.text_psms.split(",") if p.strip())

    def provenance(self) -> dict:
        sha = file_sha
        cmd, data = self.resolved_tesseract()
        dm = self.resolved_digit_model()
        # Accuracy depends on WHICH traineddata is loaded (E19 measured fas 99e42096...,
        # tessdata_best); a distro package ships a different, weaker file.
        return {
            "service_version": __version__,
            "engine": "tesseract(fas,eng) + digit_cnn",
            "tesseract": {"cmd": cmd, "tessdata_dir": repo_relative(data),
                          "lang_fa": self.lang_fa,
                          # The model actually doing the reading. With a fine-tune selected
                          # this is NOT fas.traineddata, and a results file that only stamped
                          # the stock hash would describe a run that never happened.
                          "lang_fa_sha256": sha(Path(data) / f"{self.lang_fa}.traineddata") if data else None,
                          "fas_sha256": sha(Path(data) / "fas.traineddata") if data else None,
                          "eng_sha256": sha(Path(data) / "eng.traineddata") if data else None},
            "digit_model": {"path": repo_relative(dm), "sha256": sha(dm)},
            "layout_psms": self.layout_psms, "text_psms": self.text_psms, "deskew_min_deg": self.deskew_min_deg,
            "min_line_conf": self.min_line_conf, "number_min_prob": self.number_min_prob,
            "letter_fields": self.letter_fields,
            "python": platform.python_version(),
        }


settings = Settings()
