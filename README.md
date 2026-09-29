# Persian OCR — full-text transcription service

Upload **any** image — an administrative letter, a page of prose, a table, a form —
and get back every printed line in reading order, with the numbers read by a
dedicated Persian digit reader. When the page is a letter, the usual letter fields
are also cut out of the transcript. Nothing is generated: every character in the
output was recognised from the pixels.

CPU only. No GPU, no model server, no LM Studio.

## راه‌اندازی سریع (فارسی)

```powershell
cd E:\ghasemi\ocr
.\venv312\Scripts\python.exe -m uvicorn ocr_service.api:app --host 127.0.0.1 --port 8000
```

وقتی پیام `Application startup complete.` آمد (حدود ۱ ثانیه)، در مرورگر
**http://127.0.0.1:8000/** را باز کنید، تصویر را بکشید و رها کنید. متن صفحه همان‌طور که
روی کاغذ چیده شده برمی‌گردد: پاراگراف‌های راست‌به‌چپ، جدول به‌صورت جدول واقعی، فیلدهای
نامه در بالا، و هر عدد با درستی گروه‌های سه‌رقمی‌اش.

برای کار با API، http://127.0.0.1:8000/docs را باز کنید و `POST /ocr` → **Try it out**.

از خط فرمان:

```powershell
curl.exe -F "file=@exampel_paper.png" http://127.0.0.1:8000/ocr
```

برای خاموش کردن: `Ctrl + C`.

---

## What you get

`POST /ocr` (multipart field `file`) returns:

```json
{
  "doc_id": "letter.jpg",
  "text": "جناب آقای دکتر سلطانی\nمعاون محترم پژوهش ...\nسلام علیکم؛\nاحتراما عطف به نامه ...\n\nآدرس: مشهد - ...",
  "lines":   [{"text": "جناب آقای دکتر سلطانی", "bbox": [612, 402, 1480, 455], "confidence": 0.91, "source": "psm4", "paragraph": 0, "cells": []}],
  "numbers": [{"value": "۱۴۰۳/۰۹/۲۰", "value_ascii": "1403/09/20", "bbox": [...], "source": "glyph", "confidence": 0.82, "grouping": "none"},
              {"value": "۱۲۵,۰۰۰,۰۰۰", "value_ascii": "125,000,000", "bbox": [...], "source": "glyph", "confidence": 0.84, "grouping": "ok"}],
  "is_letter": true,
  "fields": {"sender": null, "receiver": "جناب آقای دکتر سلطانی\nمعاون محترم پژوهش ...", "subject": null,
             "body_text": "سلام علیکم؛\n...", "contact_info": "آدرس: ...\nتلفن: ..."},
  "needs_review": true,
  "review_reasons": ["3 low-confidence line(s)", "5 number(s) not confirmed by the digit reader"],
  "skew_deg": 0.0, "image_size": [2304, 3016], "timing": {"total_s": 2.8}, "service": {...}
}
```

* `text` — the whole page, lines in reading order, paragraphs separated by a blank line.
  **This is the primary output.**
* `lines` — each line with its box, confidence, and which reading won.
* `numbers` — every number with `source`: `glyph` (digits confirmed by the CNN digit
  reader) or `tesseract` (not confirmed — check it).
* `fields` — only for letters (`is_letter`), and only **verbatim lines** of `text`: a
  field that is not printed is `null`. `subject` is only the text after a printed
  `موضوع:` label. On a page that is not a letter every field is `null`.
* `needs_review` / `review_reasons` — low-confidence lines or unconfirmed numbers.

* `numbers[].grouping` — `ok` when a grouped amount's groups are all three digits
  («۱۲۵,۰۰۰,۰۰۰»), **`broken`** when one is not («۳۲۱/۰۰۰/۰۰», «۱۶۵.۰۰۰۰۰۰»): a digit was
  lost or run together, so the amount must not be trusted. It is flagged, never
  repaired — filling the gap in would mean printing a digit nobody read. `none` for
  dates, reference, account and phone numbers.
* `lines[].cells` — a line's pieces when a wide gap splits it (a table row), in reading
  order; empty on ordinary prose. This is what the page at `/` rebuilds tables from.

Other endpoints: `GET /` (the upload page), `GET /health` (Tesseract, digit model, their
hashes), `GET /jobs/{id}` (queued mode), `GET /docs` (interactive API form).

## How it works

```
image → deskew (≥1°) → binarise + remove rules/specks
      → line candidates: Tesseract layout psm 3, 4, 6  +  own line finder, kashida-squeezed,
                         read with psm 7/13 (fas, eng)
      → best reading per physical line → table cells ordered right-to-left
      → CNN digit reader replaces number tokens it confirms → junk-line filter
      → thousands grouping checked → text + lines + numbers
      → letter fields cut out by rules (letters only)
```

Code: `ocr_service/transcribe.py` (recognition), `ocr_service/letter_fields.py`
(fields), `ocr_service/pipeline.py`, `ocr_service/api.py`,
`ocr_service/static/index.html` (the page at `/`). The reasoning behind every
step, with the measurement that justified it, is in the `transcribe.py` docstring and
experiments **E19** and **E20** in `ocr_eval/experiments.md`.

## Measured accuracy (E20)

Real scanned letters with hand-checked ground truth (`ocr_eval/ground_truth_real_fixed_v2.jsonl`).
Tuned on dev only. "Previous" is the Qwen2.5-VL service this replaces, on the same documents.

| | dev (51) previous | dev (51) **now** | test (33) previous | test (33) **now** |
|---|---:|---:|---:|---:|
| page text error (coverage CER) | 28.5% | **14.7%** | 24.7% | **18.1%** |
| letter body text error | 32.9% | **11.3%** | 26.8% | **17.4%** |
| contact/address line error | 21.0% | **16.1%** | 24.5% | **14.6%** |
| digit groups found | 70.8% | **83.6%** | 52.0% | **80.6%** |
| whole numbers exact | 54.8% | **75.8%** | 36.2% | **68.8%** |
| text recovered (length ratio) | 0.80 | **0.95** | 0.84 | **0.92** |
| seconds per page (p50) | ~8–10 (GPU) | **2.9 (CPU)** | ~8 (GPU) | **3.0 (CPU)** |

**Still not money-safe:** about 1 in 4 whole numbers has a wrong digit somewhere (was 1
in 3 before E20). Use `numbers[].grouping == "broken"` — which never fires on a correct
amount in the whole corpus — together with `numbers[].source == "glyph"` and
`needs_review`. The service never repairs a number it cannot read; it says so instead.

Known limits (open defects D67–D69 in `ocr_eval/error_register.md`):
stylised letterhead fonts on coloured banners are not read (`sender` is usually null);
handwriting is not read; text under a stamp or signature can be lost; photos with
strong perspective or very small text read worse; `subject` is null unless a
`موضوع:` line is printed.

## The page at `/`

http://127.0.0.1:8000/ — drop an image in and the page comes back laid out as it is
printed: the letter's fields on top, right-to-left justified paragraphs, and a real
HTML table wherever the page has one (consecutive lines that split into aligned cells —
`lines[].cells`). Beside it, every number with its source and its thousands-grouping
verdict, and a switch that lays the line and cell boxes over the original scan. There is
also a raw-text view and the full JSON, both copyable.

One file, `ocr_service/static/index.html` — plain HTML, CSS and JavaScript, no CDN and
no build step, so it works on a server with no internet access.

## Setup (once)

Needs Python 3.12 (`venv312`), Tesseract 5 with the Persian/English `tessdata_best`
models, and the digit model `models/digit_cnn.npz`.

```powershell
powershell -ExecutionPolicy Bypass -File deploy\install_tesseract.ps1
.\venv312\Scripts\python.exe -m pip install -r requirements-service.txt
```

`tessdata/` and `models/` are found automatically in the repo (or in the main checkout
when running from a git worktree). Check with `GET /health`: `"status": "ok"`.

## Configuration

Every setting is an environment variable `OCRS_<NAME>` (see `ocr_service/config.py`):
`TESSERACT_CMD`, `TESSDATA_DIR`, `DIGIT_MODEL`, `LAYOUT_PSMS` (default `3,4,6`),
`DESKEW_MIN_DEG` (1.0), `MIN_LINE_CONF` (30), `NUMBER_MIN_PROB` (0.6), `WORKERS` (8),
`LETTER_FIELDS` (1), `MAX_UPLOAD_MB` (25), `ASYNC_MODE`, `REDIS_URL`.

## Queued mode and deployment

```powershell
set OCRS_ASYNC_MODE=1
.\venv312\Scripts\python.exe -m celery -A ocr_service.tasks worker --pool=solo -l info
.\venv312\Scripts\python.exe -m uvicorn ocr_service.api:app --port 8000
```

`POST /ocr` then returns `202 {job_id}`; poll `GET /jobs/{job_id}`. Production:
`docker compose -f deploy/docker-compose.yml up --build` (nginx → API → Redis → workers, CPU only).

## Evaluation

```powershell
.\venv312\Scripts\python.exe ocr_eval\tools\bench_fulltext.py --split dev    # tune here
.\venv312\Scripts\python.exe ocr_eval\tools\bench_fulltext.py --split test   # final check only
.\venv312\Scripts\python.exe -m pytest -q
```

Writes `ocr_eval/benchmarks/bench_fulltext_<split>.json` (per-document transcript,
fields, scores, and the previous service's scores on the same documents).

## History

Until 2026-09-27 this repo served a vision-language model (coreOCR-7B, then
Qwen2.5-VL-7B) that returned five letter fields as JSON. It was removed in E19: it
truncated long pages at a 1,400-character grammar cap, invented senders/receivers for
pages that are not letters, and scrambled right-to-left word order. The reports of
that period (`FINAL_REPORT*.md`, `BRIEF_NEXT.md`, `ocr_eval/experiments.md` E0–E18)
describe that system; its code is in git history.
