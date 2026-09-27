"""
Persian Business Letter OCR — local Qwen2-VL backend
"""
import asyncio
import io
import json
import logging
import os
import re
import torch

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Runtime config (override via environment variables) ────────────────────────
MODEL_NAME  = os.getenv("MODEL_NAME",     "Qwen/Qwen2-VL-7B-Instruct")
QUANTIZE    = os.getenv("QUANTIZE",       "4bit").lower()     # 4bit | 8bit | none
USE_FLASH   = os.getenv("USE_FLASH_ATTN", "false").lower() == "true" # <-- FIXED: Disabled by default
MAX_PIXELS  = int(os.getenv("MAX_PIXELS",    str(1024 * 28 * 28)))
MIN_PIXELS  = int(os.getenv("MIN_PIXELS",    str(256  * 28 * 28)))
MAX_NEW_TOK = int(os.getenv("MAX_NEW_TOKENS", "2048"))

# ── Module-level singletons ────────────────────────────────────────────────────
_model:     object = None
_processor: object = None
_executor = ThreadPoolExecutor(max_workers=1)   # serialise GPU work


# ── Model loader ───────────────────────────────────────────────────────────────
def _load_model() -> None:
    global _model, _processor
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    attn = "flash_attention_2" if USE_FLASH else "eager"
    logger.info(
        "Loading %s  quantize=%s  attn=%s  max_pixels=%d",
        MODEL_NAME, QUANTIZE, attn, MAX_PIXELS,
    )

    kwargs: dict = dict(attn_implementation=attn, device_map="auto")

    if QUANTIZE == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True
        )
    elif QUANTIZE == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    _model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_NAME, **kwargs)
    _model.eval()

    _processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    logger.info("Model ready on %s.", next(_model.parameters()).device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield
    global _model, _processor
    del _model, _processor
    _model = _processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Persian Business Letter OCR",
    description="Local Qwen2-VL OCR for Iranian official correspondence (نامه اداری).",
    version="3.0.0",
    lifespan=lifespan,
)

ALLOWED_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/bmp", "image/tiff", "image/webp",
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
تو یک موتور OCR و استخراج اطلاعات متخصص برای نامه‌های اداری رسمی ایرانی هستی.

وظیفه تو: یک شیء JSON خالص و valid دقیقاً با ساختار زیر برگردان.
هیچ markdown fence، هیچ توضیح، هیچ متن اضافه‌ای — فقط raw JSON.

{
  "sender":       "<نام شخص یا سازمان فرستنده (امضاکننده در پایین نامه)>",
  "receiver":     "<نام شخص یا سازمان گیرنده (معمولاً بعد از کلمه 'به:' در بالای نامه)>",
  "subject":      "<موضوع نامه، دقیقاً کلمه‌به‌کلمه همان‌طور که در نامه نوشته شده>",
  "body_text":    "<متن کامل بدنه نامه، شامل سلام و امضا، verbatim>",
  "contact_info": "<تمام اطلاعات تماس: آدرس، تلفن، فکس، کد پستی، ایمیل>"
}

قوانین بسیار مهم:
1. در نامه‌های فارسی، گیرنده در بالا (به: فلانی) و فرستنده در پایین (امضا: فلانی) است. فرستنده و گیرنده را برعکس ننویس!
2. کلمات را دقیقاً همان‌طور که در تصویر می‌بینی استخراج کن (موضوع را خلاصه یا تغییر نده).
3. اگر فیلدی در سند وجود ندارد، مقدار null برگردان.
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


# ── Synchronous inference (runs inside ThreadPoolExecutor) ────────────────────
def _run_inference(pil_image: Image.Image) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text": _USER_TURN},
            ],
        },
    ]

    # Build the tokenised prompt text (with vision placeholders)
    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Encode text + pixel values; let accelerate place tensors on the right device
    inputs = _processor(
        text=[text],
        images=[pil_image],
        padding=True,
        return_tensors="pt",
    ).to(next(_model.parameters()).device)

    with torch.inference_mode():
        generated_ids = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOK,
            do_sample=False,
        )

    # Strip the prompt tokens from the output
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return _processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "Persian Business Letter OCR",
        "version": "3.0.0",
        "model":    MODEL_NAME,
        "quantize": QUANTIZE,
    }


@app.get("/health")
async def health():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    device = str(next(_model.parameters()).device)
    return {"status": "healthy", "model": MODEL_NAME, "device": device}


@app.post("/ocr", response_model=LetterExtraction)
async def extract_text(
    file: UploadFile = File(..., description="JPEG, PNG, BMP, TIFF, or WebP image"),
):
    # 1. Validate content type ─────────────────────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported type '{file.content_type}'. "
                f"Accepted: {', '.join(sorted(ALLOWED_TYPES))}"
            ),
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 2. Decode image ──────────────────────────────────────────────────────────
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")

    logger.info(
        "Processing '%s'  size=%s  bytes=%d",
        file.filename, pil_image.size, len(image_bytes),
    )

    # 3. Run VLM inference in the GPU thread ───────────────────────────────────
    try:
        loop = asyncio.get_running_loop()
        raw_output: str = await loop.run_in_executor(_executor, _run_inference, pil_image)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(
            status_code=500,
            detail=(
                "GPU out of memory. Options: "
                "lower MAX_PIXELS, set QUANTIZE=4bit, or use the 2B model."
            ),
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            raise HTTPException(status_code=500, detail=f"GPU OOM: {exc}")
        logger.exception("RuntimeError during inference for '%s'", file.filename)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")
    except Exception as exc:
        logger.exception("Unexpected inference error for '%s'", file.filename)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    logger.info("Raw output (first 300 chars): %s", raw_output[:300])

    # 4. Clean and parse JSON ──────────────────────────────────────────────────
    cleaned = _clean_json(raw_output)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error(
            "JSON parse failed for '%s': %s\nCleaned output: %s",
            file.filename, exc, cleaned[:500],
        )
        raise HTTPException(
            status_code=422,
            detail=f"Model returned unparseable JSON: {exc}. Raw: {raw_output[:300]}",
        )

    try:
        extraction = LetterExtraction.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Schema validation failed: {exc}",
        )

    logger.info("OCR complete for '%s'.", file.filename)
    return extraction
