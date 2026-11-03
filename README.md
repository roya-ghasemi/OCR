# Persian OCR — full-text transcription service

Upload any image — an administrative letter, a page of prose, a table, a form — and get
back every printed line in reading order, with numbers checked by a dedicated Persian digit
reader. When the page is a letter, the letter fields are also cut out of the transcript.
Every character in the output was recognised from the pixels; nothing is generated.

CPU only. Runtime models total about 24 MB (see [Models](#models)).

## راه‌اندازی سریع

```powershell
cd <project folder>
.\venv312\Scripts\python.exe -m uvicorn ocr_service.api:app --host 127.0.0.1 --port 8000
```

بعد از پیام `Application startup complete.` آدرس **http://127.0.0.1:8000/** را در مرورگر باز
کنید و تصویر را بکشید و رها کنید. متن صفحه با همان چیدمان کاغذ برمی‌گردد: پاراگراف‌های
راست‌به‌چپ، جدول به‌صورت جدول، فیلدهای نامه در بالا و هر عدد با منبع و درستی گروه‌های
سه‌رقمی‌اش. مستندات API در http://127.0.0.1:8000/docs است.

```powershell
curl.exe -F "file=@exampel_paper.png" http://127.0.0.1:8000/ocr
```

## What you get

`POST /ocr` (multipart field `file`) returns:

```json
{
  "doc_id": "letter.jpg",
  "text": "جناب آقای دکتر سلطانی\nمعاون محترم پژوهش ...\nسلام علیکم؛\n...",
  "lines":   [{"text": "جناب آقای دکتر سلطانی", "bbox": [612, 402, 1480, 455], "confidence": 0.91,
               "source": "psm4", "paragraph": 0, "cells": []}],
  "numbers": [{"value": "۱۲۵,۰۰۰,۰۰۰", "value_ascii": "125,000,000", "bbox": [...],
               "source": "glyph", "confidence": 0.84, "grouping": "ok"}],
  "is_letter": true,
  "fields": {"sender": null, "receiver": "...", "subject": null, "body_text": "...", "contact_info": "..."},
  "needs_review": true,
  "review_reasons": ["3 low-confidence line(s)", "5 number(s) not confirmed by the digit reader"],
  "unread": [], "validations": [],
  "skew_deg": 0.0, "image_size": [2304, 3016], "timing": {"total_s": 2.4}, "service": {...}
}
```

* `text` — the whole page, lines in reading order, paragraphs separated by a blank line.
* `lines` — each line with its box, confidence, and which reading won; `cells` when the
  line is a table row.
* `numbers` — `source` is `glyph` (confirmed by the CNN digit reader) or `tesseract`
  (not confirmed). `grouping` is `broken` when a thousands group is not three digits: a
  digit was lost, so the amount must not be trusted. It is flagged, never repaired.
* `fields` — letters only, and only verbatim lines of `text`; a field that is not printed
  is `null`. On a page that is not a letter every field is `null`.
* `unread` — regions that were not transcribed (handwriting after a form label, marks,
  an unreadable letterhead), with what the recogniser offered.
* `needs_review` / `review_reasons` — why a human should look.

Other endpoints: `GET /` (upload page), `GET /healthz` (liveness), `GET /health`
(readiness: engine, model hashes, cache, queue), `GET /jobs/{id}` (queued mode),
`GET /docs`.

## How it works

```
image → crop the sheet, remove blue stamps → deskew (≥1°) → binarise, remove rules/specks
      → candidates: Tesseract layout passes (psm 3, 4, 6; psm 11 for lines without digits)
                    +  own line finder, each line
                    read with psm 7/13 in Persian and English (batched per process)
      → best reading per physical line → CNN digit reader confirms number tokens
      → table cells, paragraphs, dot-confusion repair, Persian comma
      → text + lines + numbers → letter fields by rules (letters only)
```

Details and module map: [ARCHITECTURE.md](ARCHITECTURE.md). Every design decision and the
measurement behind it: `ocr_eval/experiments.md`.

## Screenshots and rendered pages

A screenshot is not paper: text can be any colour on any colour, and the steps built for
scans (cutting the sheet out of a photo, erasing blue stamps) destroy it. A digital image is
recognised by its flat pixels and is also read through a screen route that keeps text of
every colour, and the more confident of the two readings is returned (E39). `prep.screen`
in the response says which one won.

## Measured accuracy

Real scanned letters with hand-checked ground truth (`ocr_eval/ground_truth_real_fixed_v2.jsonl`),
51 dev / 33 test documents. Settings are tuned on dev only; test is the check.

| | dev 1.0 | dev **1.1** | test 1.0 | test **1.1** |
|---|---:|---:|---:|---:|
| page text error (coverage CER) | 12.86% | **12.62%** | 17.57% | **17.23%** |
| letter body text error | 18.97% | **18.44%** | 32.48% | **29.08%** |
| receiver error | 13.22% | **13.17%** | 15.05% | **14.75%** |
| digit groups found | 86.07% | 86.07% | 81.22% | 81.22% |
| whole numbers exact | 78.79% | 78.79% | 69.53% | 69.53% |
| text recovered (length ratio) | 0.927 | **0.934** | 0.864 | **0.870** |
| seconds per page (p50, 14-core CPU) | 4.0 | **3.0** | 4.1 | **3.1** |
| Tesseract processes per page | ~110 | **~35** | | |

1.1 changes: line crops are read in batches (one process per batch instead of one per
line, identical output), and a sparse-text pass (psm 11) competes in regions without
digits (E38). Benchmark files: `ocr_eval/benchmarks/bench_fulltext_{dev,test}.json`.

**Not money-safe on its own:** about 1 in 4 whole numbers still has a wrong digit. Use
`numbers[].grouping == "broken"`, `numbers[].source == "glyph"` and `needs_review`.

Known limits (open defects in `ocr_eval/error_register.md`): stylised letterheads on
coloured banners are not read (`sender` is usually null); handwriting is not read; text
under a stamp or signature can be lost; strong perspective reads worse; captures under
~20 px text height lose the dots that separate ب/ی/پ/ن and ح/خ/ج — the service upsamples
and flags them, but a 300 DPI scan is the real fix.

## Models

All of them are in `models/` (Tesseract files in `models/tesseract/`).

| file | size | what it does |
|---|---:|---|
| `models/tesseract/fas_custom.traineddata` | 3.3 MB | Persian recogniser, fine-tuned here (`ocr_train/`), default |
| `fas.traineddata` (tessdata_best) | 3.3 MB | stock Persian model, rollback with `OCRS_LANG_FA=fas` |
| `eng.traineddata` (tessdata_fast) | 4.1 MB | Latin runs inside Persian text |
| `ara.traineddata` (tessdata_best) | 12.6 MB | optional re-read of Arabic pages |
| `models/digit_cnn.npz` | 0.6 MB | Persian/Latin digit reader (numpy inference, no torch) |

There is no neural network server and no GPU. The 7B vision-language models used before
2026-09-27 were removed (see History).

## Setup

Python 3.12, Tesseract 5, and the models above.

```powershell
powershell -ExecutionPolicy Bypass -File deploy\install_tesseract.ps1
.\venv312\Scripts\python.exe -m pip install -r requirements.txt      # service + tests + eval
```

`requirements-service.txt` is the runtime only (what the Docker image installs);
`requirements-train.txt` adds training. `GET /health` must report `"status": "ok"`.

## Configuration

All settings are in `.env` at the repository root (copy `.env.example`, which lists every
setting with its default); an environment variable `OCRS_<NAME>` overrides the file.
`API_KEY` makes `/ocr` and `/jobs` require the header `X-API-Key`. Settings:
`LANG_FA` (`fas_custom`), `LANG_AR` (`ara`, empty disables), `TESSERACT_CMD`, `TESSDATA_DIR`,
`DIGIT_MODEL`, `LAYOUT_PSMS` (`3,4,6`), `TEXT_PSMS` (`11`), `WORKERS` (8 Tesseract processes per page),
`MAX_CONCURRENT_PAGES` (2), `DESKEW_MIN_DEG` (1.0), `MIN_LINE_CONF` (30),
`NUMBER_MIN_PROB` (0.6), `LETTER_FIELDS` (1), `SPELLFIX` (1), `MAX_UPLOAD_MB` (25),
`CACHE` (1), `CACHE_DIR`, `API_KEY`, `ASYNC_MODE` (0), `REDIS_URL`, `SPOOL_DIR`.

## Deployment

```bash
docker compose -f deploy/docker-compose.yml up -d --build
```

nginx → API (enqueues) → Redis → Celery workers, CPU only. The image contains only the
service package and the runtime models (`.dockerignore` is an allow-list); downloaded
models are pinned and sha256-checked; the container runs as a non-root user. Bare-metal
(systemd) and operations: [RUNBOOK.md](RUNBOOK.md).

## Development

```powershell
.\venv312\Scripts\python.exe -m ruff check .
.\venv312\Scripts\python.exe -m pytest
.\venv312\Scripts\python.exe ocr_eval\tools\bench_fulltext.py --split dev    # tune here
.\venv312\Scripts\python.exe ocr_eval\tools\bench_fulltext.py --split test   # final check only
```

CI (`.github/workflows/ci.yml`) runs lint, the tests with Tesseract installed, and builds
the image.

| folder | contents |
|---|---|
| `ocr_service/` | the service (the only package that is deployed) |
| `ocr_train/` | Tesseract fine-tuning pipeline |
| `ocr_research/` | digit-CNN training and earlier approaches kept for comparison |
| `ocr_eval/` | ground truth, benchmarks, experiment log, reports |
| `deploy/` | Dockerfile, compose, nginx, systemd unit, Windows installer |
| `docs/history/` | reports from the vision-language-model period |

## History

Until 2026-09-27 the service ran a vision-language model (coreOCR-7B, then Qwen2.5-VL-7B)
that returned five letter fields as JSON. It was replaced (E19) because it truncated long
pages, invented senders and receivers for pages that are not letters, and scrambled
right-to-left word order. Reports from that period are in `docs/history/` and experiments
E0–E18 of `ocr_eval/experiments.md`.
