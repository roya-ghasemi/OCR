"""
Persian Business Letter OCR — self-contained local backend.

The app starts its own llama.cpp engine on the GGUF model files. LM Studio does
not need to be running; only its model files and bundled engine binary are used.
Set API_BASE_URL to attach to an already-running OpenAI-compatible server
(LM Studio, a remote llama-server, ...) instead of starting one.
"""
import base64
import io
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ValidationError
from openai import AsyncOpenAI

import config
import json_repair
from engine import LlamaEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Runtime config ────────────────────────────────────────────────────────────
# Every tunable comes from config.py, which is loaded once and hashed into every
# results file. config.py existed before this change but nothing imported it, so
# the values it documented were not the values the service ran (defect D25).
API_BASE_URL = config.API_BASE_URL
API_KEY = config.API_KEY

engine = LlamaEngine()
client: Optional[AsyncOpenAI] = None
MODEL_NAME = config.MODEL_NAME               # filled in at startup when unset

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, MODEL_NAME

    if API_BASE_URL:
        base_url = API_BASE_URL
        logger.info("Using external OpenAI-compatible backend at %s", base_url)
    else:
        engine.start()                        # blocks until the weights are loaded
        base_url = engine.base_url
        if not MODEL_NAME and engine.model_path:
            MODEL_NAME = str(engine.model_path)

    if not MODEL_NAME:
        MODEL_NAME = "local-model"

    client = AsyncOpenAI(base_url=base_url, api_key=API_KEY,
                         timeout=config.CLIENT_TIMEOUT)
    try:
        yield
    finally:
        engine.stop()


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Persian Business Letter OCR",
    description="Local OCR for Iranian official correspondence (نامه اداری).",
    version="4.0.0",
    lifespan=lifespan,
)

# Clients are inconsistent about the declared content type (Postman often sends
# application/octet-stream), so the real format is read from the bytes instead.
_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG":  "image/png",
    "BMP":  "image/bmp",
    "TIFF": "image/tiff",
    "WEBP": "image/webp",
}


# ── Pydantic output schema ─────────────────────────────────────────────────────
class LetterExtraction(BaseModel):
    sender:       Optional[str] = None   # فرستنده
    receiver:     Optional[str] = None   # گیرنده
    subject:      Optional[str] = None   # موضوع
    body_text:    Optional[str] = None   # متن بدنه
    contact_info: Optional[str] = None   # اطلاعات تماس


# ── System prompt ──────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
تو یک موتور OCR متخصص برای نامه‌های اداری رسمی ایرانی هستی.

وظیفه تو: یک شیء JSON خالص و valid دقیقاً با ساختار زیر برگردان.
هیچ markdown fence، هیچ توضیح، هیچ متن اضافه‌ای — فقط raw JSON.

{
  "sender":       "<نام سازمان یا شخص فرستنده — از سربرگ یا امضا>",
  "receiver":     "<نام سازمان یا شخص گیرنده>",
  "subject":      "<موضوع نامه، دقیقاً همان‌طور که نوشته شده>",
  "body_text":    "<متن کامل بدنه نامه، شامل سلام و امضا، verbatim>",
  "contact_info": "<تمام اطلاعات تماس: آدرس، تلفن، فکس، کد پستی، ایمیل>"
}

قوانین:
1. هر کلمه را دقیقاً همان‌طور که در تصویر است بنویس — ترجمه نکن.
2. اگر فیلدی در سند وجود ندارد، مقدار null برگردان.
3. متن ناخوانا را با [ناخوانا] مشخص کن.
4. هیچ اطلاعاتی را جعل نکن.
"""

_USER_TURN = "متن این نامه اداری را OCR کن و JSON بده."

# Strips ```json ... ``` or ``` ... ``` wrappers (with optional leading/trailing whitespace)
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", re.IGNORECASE)


def _clean_json(raw: str) -> str:
    """Remove markdown code-fences the model may wrap around its JSON output."""
    stripped = raw.strip()
    m = _FENCE_RE.match(stripped)
    if m:
        return m.group(1).strip()
    # Fallback: strip individual opening / closing fence lines
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$",          "", stripped)
    return stripped.strip()


# Where full model output goes when JSON parsing fails. Kept out of the main log
# because these are real administrative letters and the log is not a safe place
# for document content.
_RAW_DUMP_DIR = os.getenv("RAW_DUMP_DIR", "debug_raw")


def _dump_raw_output(filename: str, raw: str, cleaned: str, exc: Exception) -> str:
    """Write the complete model output for one failed parse. Returns the path.

    Never raises: a failure to write a debug file must not turn a 422 into a 500.
    """
    try:
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), _RAW_DUMP_DIR)
        os.makedirs(d, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", filename or "unknown")[:80]
        path = os.path.join(d, f"{safe}.{int(time.time())}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "filename": filename,
                    "error": str(exc),
                    "raw_len": len(raw or ""),
                    "cleaned_len": len(cleaned or ""),
                    "raw_output": raw,
                    "cleaned_output": cleaned,
                },
                fh, ensure_ascii=False, indent=2,
            )
        return path
    except Exception:
        logger.exception("Could not write raw-output dump for '%s'", filename)
        return "<dump failed>"


def _sampling_extras() -> dict:
    """Sampling knobs that are only sent when set away from their neutral value.

    llama.cpp accepts `repeat_penalty` / `frequency_penalty` through the OpenAI
    `extra_body`. They stay off by default so this change cannot alter the
    baseline; Phase 2a turns them on one at a time to test the D23 repetition
    loop, and `top_p` is sent only when it is not 1.0 so the recorded config
    always reflects what was actually transmitted.
    """
    extra: dict = {}
    if config.REPEAT_PENALTY and config.REPEAT_PENALTY != 1.0:
        extra["repeat_penalty"] = config.REPEAT_PENALTY
    if config.FREQUENCY_PENALTY:
        extra["frequency_penalty"] = config.FREQUENCY_PENALTY
    out: dict = {}
    if extra:
        out["extra_body"] = extra
    if config.TOP_P != 1.0:
        out["top_p"] = config.TOP_P
    return out


def _blank(v) -> bool:
    """A field counts as empty when it is null or whitespace-only."""
    return v is None or (isinstance(v, str) and not v.strip())


def _response_format() -> Optional[dict]:
    """Structured-decoding constraint, per config.RESPONSE_FORMAT.

    llama.cpp compiles the schema to a grammar and constrains sampling to it, so
    a repetition loop cannot emit unterminated JSON -- the grammar forbids
    leaving a string open at the token cap. Off by default; Phase 2a A/Bs it.
    """
    mode = (config.RESPONSE_FORMAT or "off").lower()
    if mode == "json_object":
        return {"type": "json_object"}
    if mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "LetterExtraction",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        f: {"type": ["string", "null"]}
                        for f in ("sender", "receiver", "subject",
                                  "body_text", "contact_info")
                    },
                    "required": ["sender", "receiver", "subject",
                                 "body_text", "contact_info"],
                    "additionalProperties": False,
                },
            },
        }
    return None


def _parses(raw: Optional[str]) -> bool:
    """Whether the model output is already valid JSON, before any repair."""
    if not raw:
        return False
    try:
        json.loads(_clean_json(raw))
        return True
    except json.JSONDecodeError:
        return False


def encode_image(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:{mime_type};base64,{encoded}"


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "Persian Business Letter OCR",
        "version": "4.0.0",
        "model":    MODEL_NAME,
        "backend":  "external server" if API_BASE_URL else "embedded llama.cpp",
    }


@app.get("/health")
async def health():
    if client is None:
        raise HTTPException(status_code=503, detail="Service still starting up.")
    if not API_BASE_URL and not engine.is_alive():
        raise HTTPException(
            status_code=503,
            detail=f"Engine process is not running. See {engine.log_path}.",
        )
    try:
        await client.models.list()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Backend not reachable: {exc}")

    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "backend": "external server" if API_BASE_URL else "embedded llama.cpp",
        "mmproj": str(engine.mmproj_path) if engine.mmproj_path else None,
    }


async def _run_ocr(upload: UploadFile) -> LetterExtraction:
    """Validate the image, call the engine, parse+validate the JSON."""
    image_bytes = await upload.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 1. Detect the real format from the bytes and validate it ─────────────────
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        image_format = pil_image.format
        pil_image.verify()
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{upload.filename}' is not a readable image. "
                f"Accepted formats: {', '.join(sorted(_FORMAT_TO_MIME))}."
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")

    mime_type = _FORMAT_TO_MIME.get(image_format or "")
    if mime_type is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported image format '{image_format}'. "
                f"Accepted: {', '.join(sorted(_FORMAT_TO_MIME))}."
            ),
        )

    logger.info(
        "Processing '%s' format=%s bytes=%d",
        upload.filename, image_format, len(image_bytes),
    )

    # 2. Call the local engine ─────────────────────────────────────────────────
    base64_image_url = encode_image(image_bytes, mime_type)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _USER_TURN},
                {"type": "image_url", "image_url": {"url": base64_image_url}},
            ],
        },
    ]

    async def _call(attempt: int):
        """One inference call. `attempt` 0 is the primary; 1+ are retries, which
        deliberately sample differently -- at temperature 0.0 a retry with the
        same settings reproduces the same repetition loop exactly."""
        kwargs = dict(
            model=MODEL_NAME,
            messages=messages,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            **_sampling_extras(),
        )
        if _response_format():
            kwargs["response_format"] = _response_format()
        if attempt:
            kwargs["temperature"] = config.RETRY_TEMPERATURE
            extra = dict(kwargs.get("extra_body") or {})
            extra["repeat_penalty"] = config.RETRY_REPEAT_PENALTY
            kwargs["extra_body"] = extra
        return await client.chat.completions.create(**kwargs)

    raw_output, response, attempts = None, None, 0
    last_exc = None
    for attempt in range(config.MAX_RETRIES + 1):
        attempts = attempt + 1
        try:
            response = await _call(attempt)
            raw_output = response.choices[0].message.content
        except Exception as exc:
            last_exc = exc
            logger.exception("Inference error for '%s' (attempt %d)",
                             upload.filename, attempts)
            continue
        # Retry only on output that will not parse. A response that parses is
        # accepted on the first attempt; retries exist for the D23 loop, and
        # each one costs a full inference, so they are strictly bounded.
        if _parses(raw_output):
            break
        if attempt < config.MAX_RETRIES:
            logger.warning("Unparseable output for '%s' on attempt %d; retrying "
                           "with temperature=%s repeat_penalty=%s",
                           upload.filename, attempts,
                           config.RETRY_TEMPERATURE, config.RETRY_REPEAT_PENALTY)

    if response is None:
        raise HTTPException(status_code=500, detail=f"Inference error: {last_exc}")

    logger.info("Raw output for '%s': len=%d finish_reason=%s attempts=%d",
                upload.filename, len(raw_output or ""),
                getattr(response.choices[0], "finish_reason", None), attempts)

    # 3. Clean and parse JSON ──────────────────────────────────────────────────
    cleaned = _clean_json(raw_output)

    repair_report = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Phase 2a: the FULL raw output is required to tell token-budget
        # exhaustion from a repetition loop from an escaping bug. Truncating to
        # 300 chars is what made D3 undiagnosable for three sessions. The body
        # goes to a per-request debug file rather than the log line, so real
        # letter content never lands in the shared application log (PII rule).
        dump = _dump_raw_output(upload.filename, raw_output, cleaned, exc)
        logger.error(
            "JSON parse failed for '%s': %s | raw_len=%d cleaned_len=%d "
            "finish_reason=%s attempts=%d | full output written to %s",
            upload.filename, exc, len(raw_output or ""), len(cleaned),
            getattr(response.choices[0], "finish_reason", None), attempts, dump,
        )

        data = None
        if config.JSON_REPAIR:
            # Only ever runs after a strict parse has failed, so it cannot change
            # a response that was already valid. It recovers the complete fields
            # that precede the repetition loop and DROPS the field the loop was
            # inside -- a half-read field looks real and is worse than a missing
            # one. Measured offline on the 74 baseline failure dumps: 67.6%
            # yielded a usable extraction, almost always sender/receiver/subject
            # without body_text.
            data, repair_report = json_repair.repair(cleaned)
            if data is not None:
                logger.warning(
                    "Recovered '%s' by JSON repair: fields=%s dropped_partial=%s",
                    upload.filename, repair_report["recovered_fields"],
                    repair_report["dropped_partial_field"],
                )

        if data is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Model returned unparseable JSON: {exc}. "
                    f"raw_len={len(raw_output or '')} "
                    f"finish_reason={getattr(response.choices[0], 'finish_reason', None)} "
                    f"attempts={attempts} debug_dump={dump}"
                ),
            )

    try:
        extraction = LetterExtraction.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Schema validation failed: {exc}",
        )

    # 4. Response validation ───────────────────────────────────────────────────
    # Every field Optional means an all-null object VALIDATES and ships as a 200.
    # Nothing downstream can distinguish that from a document that genuinely has
    # no content, which is what makes D4 the more dangerous reliability defect.
    # An extraction with nothing in it is a failure and must say so.
    if all(_blank(getattr(extraction, f)) for f in LetterExtraction.model_fields):
        logger.error("All-null extraction for '%s' — returning 422, not a silent 200.",
                     upload.filename)
        raise HTTPException(
            status_code=422,
            detail={
                "error": "empty_extraction",
                "message": ("The model returned a well-formed response with every "
                            "field empty. This is a failure, not a document without "
                            "content."),
                "attempts": attempts,
            },
        )

    if repair_report is not None:
        logger.info("OCR complete for '%s' (PARTIAL — recovered by JSON repair).",
                    upload.filename)
    else:
        logger.info("OCR complete for '%s'.", upload.filename)
    return extraction


def _pick_upload(file: Optional[UploadFile], image: Optional[UploadFile]) -> UploadFile:
    upload = file or image
    if upload is None:
        raise HTTPException(
            status_code=400,
            detail="No image uploaded. Send a multipart form field named 'file' (or 'image').",
        )
    return upload


@app.post("/ocr", response_model=LetterExtraction)
async def extract_text(
    file: Optional[UploadFile] = File(None, description="JPEG, PNG, BMP, TIFF, or WebP image"),
    image: Optional[UploadFile] = File(None, description="Alias for 'file'"),
):
    return await _run_ocr(_pick_upload(file, image))


# ── Deprecated compatibility route ─────────────────────────────────────────────
# `POST /ocr/corrected` existed only during the Phase 0 remediation. It ran the
# identical `_run_ocr()` path and then applied the Dehkhoda dictionary corrector.
# The Phase 0 ablation measured that corrector at -0.03 pp corpus CER (inside
# noise), so the stage was removed from the runtime pipeline entirely; see
# `dehkhoda/REMOVED.md`.
#
# This stub is kept ONLY so that any out-of-repo caller does not break with a 404.
# It returns the same response shape with `corrected` identical to `raw` and
# `changed` empty, and logs a warning so a hidden caller surfaces itself. There is
# no dictionary, lexicon or any other post-processing stage behind it.
# Remove once the logs show no traffic. Tracked as D11 in ocr_eval/error_register.md.
@app.post("/ocr/corrected", deprecated=True)
async def extract_text_corrected(
    file: Optional[UploadFile] = File(None, description="JPEG, PNG, BMP, TIFF, or WebP image"),
    image: Optional[UploadFile] = File(None, description="Alias for 'file'"),
):
    logger.warning(
        "DEPRECATED endpoint /ocr/corrected was called. It is now a passthrough to "
        "/ocr; no correction stage runs. Migrate the caller to POST /ocr."
    )
    extraction = await _run_ocr(_pick_upload(file, image))
    raw = extraction.model_dump()
    return {
        "raw": raw,
        "corrected": raw,
        "changed": {},
        "deprecated": True,
        "notice": (
            "The Dehkhoda correction stage was removed after an ablation measured it at "
            "-0.03 pp corpus CER. 'corrected' is byte-identical to 'raw'. Use POST /ocr."
        ),
    }
