"""
Single source of truth for every tunable knob in the OCR service.

Phase 0 deliverable: these values were previously scattered as literals across
main.py and engine.py. They are reproduced here EXACTLY as they were — this file
centralizes them, it does not change behaviour. Any experiment that alters a value
must be recorded in ocr_eval/experiments.md with the resulting metrics.

Every knob stays overridable by the environment variable of the same name, so the
eval harness can sweep one variable at a time without editing code.
"""
import os
from pathlib import Path

# ── Provenance: pinned so every results file can record what produced it ───────
ENGINE_BUILD = "llama.cpp llama-server version 1 (fe2adf0), Clang 19.1.1, win-x86_64-nvidia-cuda12-avx2-2.28.2"

# The backend version every number in this repo was measured on. Discovery
# PREFERS this build; plain "newest wins" would silently swap the engine while
# ENGINE_BUILD above still claims 2.28.2, making every results file lie about its
# own provenance. It matters concretely here: the three GBNF parser quirks in
# ocr_pipeline/grammar.py were characterised against 2.28.2 and are not known to
# hold on other builds. Set ENGINE_BUILD_PIN="" to opt into newest-available.
ENGINE_BUILD_PIN = os.getenv("ENGINE_BUILD_PIN", "2.28.2")
MODEL_ID = "coreOCR-7B-050325-preview.Q4_K_S.gguf"
MODEL_SIZE_BYTES = 4457769440
MMPROJ_ID = "coreOCR-7B-050325-preview.mmproj-f16.gguf"
MMPROJ_SHA256 = "34933952a9ae2f1ac7af7a908189b3fc22106d3999fd3fae43b520ac4d308a78"

# ── Backend / engine discovery ────────────────────────────────────────────────
# LM Studio's tree still holds the llama.cpp BACKENDS. It no longer holds the
# MODELS: on 2026-09-05 the coreOCR GGUFs were moved to D:\models\models because
# C: was down to 1.1% free. Backend discovery and model discovery are therefore
# separate concerns and must not be derived from one root — that coupling is
# exactly what broke when the files moved.
LMSTUDIO_ROOT = Path(os.getenv("LMSTUDIO_ROOT", Path.home() / ".lmstudio"))

MODEL_GLOB = os.getenv("MODEL_GLOB", "**/*coreOCR*.gguf")
MODEL_GGUF = os.getenv("MODEL_GGUF")          # explicit path wins over the glob
MMPROJ_GGUF = os.getenv("MMPROJ_GGUF")
LLAMA_SERVER = os.getenv("LLAMA_SERVER")

# Roots searched for the model when MODELS_DIR is unset, in order. Set
# MODEL_SEARCH_ROOTS (os.pathsep-separated) to add a location without editing
# code. Paths are built with pathlib and forward slashes so nothing here depends
# on the platform separator.
_ENV_ROOTS = [Path(p) for p in os.getenv("MODEL_SEARCH_ROOTS", "").split(os.pathsep) if p.strip()]
MODEL_SEARCH_ROOTS = _ENV_ROOTS or [
    Path("D:/models/models"),           # current location, moved off C: 2026-09-05
    LMSTUDIO_ROOT / "models",           # LM Studio default, previous location
]


def _resolve_models_dir() -> Path:
    """First candidate root that actually CONTAINS the model.

    Preferring a root that holds a match — rather than the first root that merely
    exists — makes discovery self-correcting when the files move again. Falling
    back to a directory that exists but is empty produces "no model matching ...
    under <wrong dir>", which sends the reader to the wrong place.
    """
    existing = [r for r in MODEL_SEARCH_ROOTS if r.is_dir()]
    for root in existing:
        if any("mmproj" not in p.name.lower() for p in root.glob(MODEL_GLOB)):
            return root
    return existing[0] if existing else MODEL_SEARCH_ROOTS[0]


MODELS_DIR = (Path(os.environ["MODELS_DIR"]) if os.getenv("MODELS_DIR")
              else _resolve_models_dir())

# ── Engine runtime ────────────────────────────────────────────────────────────
ENGINE_HOST = os.getenv("ENGINE_HOST", "127.0.0.1")
ENGINE_PORT = int(os.getenv("ENGINE_PORT", "18234"))
ENGINE_LOG = Path(os.getenv("ENGINE_LOG", "engine.log"))
ENGINE_STARTUP_TIMEOUT = float(os.getenv("ENGINE_STARTUP_TIMEOUT", "600"))
N_GPU_LAYERS = os.getenv("N_GPU_LAYERS", "99")
CONTEXT_SIZE = os.getenv("CONTEXT_SIZE", "16384")
IMAGE_MIN_TOKENS = os.getenv("IMAGE_MIN_TOKENS", "1024")   # <1024 -> repetition loops

# ── Client / decoding ─────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL")      # set = attach to an external server
API_KEY = os.getenv("API_KEY", "local")
MODEL_NAME = os.getenv("MODEL_NAME", "")      # filled in at startup when unset
CLIENT_TIMEOUT = float(os.getenv("CLIENT_TIMEOUT", "600.0"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
TOP_P = float(os.getenv("TOP_P", "1.0"))      # sent only when != 1.0

# Degeneracy control. Both neutral by default so the baseline is unchanged.
# Phase 2a varies these one at a time against defect D23, where the model emits
# one Persian-Indic digit thousands of times until decoding stops.
# Phase 2a, measured: 1.0 (off) and 1.1 leave the digit repetition loop intact;
# 1.2 removes it. Real dev usable-output 14.81% -> 96.30%, latency p50 22.2s ->
# 4.44s. See experiments.md "repeat_penalty_12". The effect is a threshold, not a
# gradient -- do not lower this to 1.1 expecting a partial benefit.
REPEAT_PENALTY = float(os.getenv("REPEAT_PENALTY", "1.2"))
FREQUENCY_PENALTY = float(os.getenv("FREQUENCY_PENALTY", "0.0"))  # 0.0 = off

# ── Image preprocessing ───────────────────────────────────────────────────────
# NOTE: there is currently NO preprocessing. The uploaded bytes are base64-encoded
# and sent as-is: no resize, no DPI normalization, no deskew, no contrast work.
# Recorded here so Phase 5 has an explicit place to add it.
RESIZE_MAX_EDGE = int(os.getenv("RESIZE_MAX_EDGE", "0"))   # 0 = disabled (current)

# ── Structured decoding (Phase 2a fix (i)) ────────────────────────────────────
# When on, the request carries an OpenAI-style `response_format` so llama.cpp
# constrains sampling to a grammar derived from the LetterExtraction schema.
# A repetition loop cannot then produce unterminated JSON: the grammar forbids
# it. Off by default so the baseline is unchanged and the A/B is clean.
#   "off"          - no constraint (current behaviour)
#   "json_object"  - valid JSON, but any shape
#   "json_schema"  - valid JSON matching the 5-field schema exactly
RESPONSE_FORMAT = os.getenv("RESPONSE_FORMAT", "off")

# ── JSON repair (Phase 2a fix (ii)) ───────────────────────────────────────────
# Salvages a truncated response by closing the open string, object and array
# nesting. Only ever applied AFTER a strict parse has failed, so it cannot
# change the result for a response that was already valid.
# Phase 2a, measured: +1.85 pp usable-output on real dev at no latency cost.
# Recovers a PARTIAL extraction -- the fields completed before truncation -- so a
# repaired response must be surfaced as partial, never as a clean read.
JSON_REPAIR = os.getenv("JSON_REPAIR", "1") == "1"

# ── Retry policy ──────────────────────────────────────────────────────────────
# NOTE: there is currently NO retry. A 422 or an all-null extraction is returned
# to the caller as-is. Phase 2 will change this.
# Phase 2a, measured: one retry took real dev from 98.15% to 100.00% usable at no
# cost on the happy path -- it only fires on output that failed to parse.
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))
# Sampling used on a retry only. temperature 0.0 is deterministic, so retrying a
# repetition loop with identical settings reproduces it exactly -- a retry must
# change something to be worth spending latency on.
RETRY_TEMPERATURE = float(os.getenv("RETRY_TEMPERATURE", "0.2"))
RETRY_REPEAT_PENALTY = float(os.getenv("RETRY_REPEAT_PENALTY", "1.15"))


def provenance() -> dict:
    """Stamp into every results file so a number can always be traced to a build."""
    return {
        "engine_build": ENGINE_BUILD,
        "engine_build_pin": ENGINE_BUILD_PIN,
        "model_id": MODEL_ID,
        # Where the weights were loaded from. The GGUFs moved from C: to D: on
        # 2026-09-05; without this field, two results files produced either side
        # of such a move are indistinguishable.
        "models_dir": str(MODELS_DIR),
        "mmproj_sha256": MMPROJ_SHA256,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repeat_penalty": REPEAT_PENALTY,
        "frequency_penalty": FREQUENCY_PENALTY,
        "max_tokens": MAX_TOKENS,
        "context_size": CONTEXT_SIZE,
        "image_min_tokens": IMAGE_MIN_TOKENS,
        "n_gpu_layers": N_GPU_LAYERS,
        "resize_max_edge": RESIZE_MAX_EDGE,
        "max_retries": MAX_RETRIES,
        "response_format": RESPONSE_FORMAT,
        "json_repair": JSON_REPAIR,
    }
