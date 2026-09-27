# Hybrid Persian OCR service — VLM reader + Tesseract numeric validation

`ocr_service/` is a new package beside the evaluated legacy path (`main.py`, `/ocr` on
:8000). The legacy path is untouched so every number in `ocr_eval/` stays comparable.

```
image ─► preprocess (crop · deskew · ≤300 KB) ─► primary VLM (Qwen2.5-VL, GBNF grammar)
      ─► [secondary VLM (coreOCR) cross-check, optional] ─► Tesseract numeric validation
      ─► JSON: 5 fields · per-number confidence high|low|unverified · candidates on conflict
```

Read `ocr_eval/reports/diagnosis_2026-09-13.md` first: it is why this design exists.
The two things it measured that this service is built around:

* both VLMs read Latin digits 12/12 and **Persian-Indic digits 0/12** (D53) — numbers
  the VLM emits are generated, not read. Tesseract's `fas` model is a glyph classifier
  and is the independent reader. **Every number goes through it.**
* the VLMs read Persian *words* on clean pages well (novel names at CER 0.006), so the
  VLM stays the reader of the text.

---

## Local (dev) setup — Windows, one GPU

Prerequisites already on this machine: `venv312` (Python 3.12), the llama.cpp
`llama-server` build pinned by `engine.py` (2.28.2), the two GGUF models at the paths
in `ocr_service/config.py`.

**Tesseract** (5.4.0, UB-Mannheim build) is installed via `deploy/install_tesseract.ps1`;
Persian/English data lives in `<repo>/tessdata`, auto-detected by `config.resolved_tesseract()`.
Note the 2026-09-17 section below: on this corpus the default classical reader is the
in-repo glyph reader, with Tesseract as `OCRS_NUMERIC_READER=tesseract|both`. Verify
either with `GET /health` → `"tesseract": {"available": true, ...}`.

```bash
venv312\Scripts\python.exe -m pip install -r requirements-service.txt
```

Run the API (spawns `llama-server` for the primary model on :18236 automatically):

```bash
venv312\Scripts\python.exe -m uvicorn ocr_service.api:app --host 127.0.0.1 --port 8010
```

```bash
curl -F "file=@ocr_eval/dataset_ex/some_letter.JPG" http://127.0.0.1:8010/ocr
```

**VRAM note (12 GB):** one 7B Q4 model with 16k context uses ~8.5 GB. The legacy
coreOCR engine on :18234 and the Qwen2.5-VL engine on :18236 do not fit together —
stop one before starting the other. `OCRS_ENABLE_SECONDARY_MODEL=1` (both models,
cross-check) needs ≥24 GB or two GPUs.

### Queued mode (dev)

```bash
docker run -p 6379:6379 redis:7-alpine
set OCRS_ASYNC_MODE=1
venv312\Scripts\python.exe -m celery -A ocr_service.tasks worker --pool=solo -l info
venv312\Scripts\python.exe -m uvicorn ocr_service.api:app --port 8010
```
`POST /ocr` → `202 {job_id}`; `GET /jobs/{job_id}` → `queued | running | done | failed`.
(`--pool=solo` is required on Windows.)

### Client-side pre-processing (reference for the mobile team)

```bash
venv312\Scripts\python.exe -m ocr_service.preprocess photo.jpg out.jpg --max-kb 300
```
Prints crop box, skew angle, chosen JPEG quality, in/out bytes. The server runs the same
function on every upload (`OCRS_PREPROCESS_INCOMING=1`), so a client that already did it
costs nothing extra. Output is JPEG (llama.cpp cannot decode WebP); send WebP to the
gateway if bandwidth matters and let the service transcode.

---

## Production setup — docker-compose

```
nginx :8080 ──► api ×N (stateless, async) ──► redis ──► worker ×M (Celery, prefetch 1)
                                                            └──► llama-primary :18236 (GPU)
                                                            └──► llama-secondary :18234 (GPU, profile)
```

```bash
docker compose -f deploy/docker-compose.yml up --build --scale api=2 --scale worker=1
docker compose -f deploy/docker-compose.yml --profile secondary up   # with the cross-check model
```

Sizing rule: **workers = concurrent images the GPU tier can serve**, not more. The
API tier is stateless and cheap; the engine is the bottleneck (one 7B Q4 on one 12 GB
GPU ≈ 1–2 concurrent images at 5–15 s each). Measure with the benchmark before setting
`--scale worker`.

Switching engines is one variable: `OCRS_BACKEND=llamacpp | vllm | triton`
(`ocr_service/backends/`). vLLM uses `guided_json` in place of GBNF; Triton assumes an
`ocr_letter` model with `image`/`prompt` → `text` tensors (documented in `triton.py`).

Environment reference: every setting is `OCRS_<NAME>` — see `ocr_service/config.py`.

---

## Output contract (`POST /ocr`)

```json
{
  "doc_id": "…", 
  "fields": {"sender": "…", "receiver": "…", "subject": null, "body_text": "…", "contact_info": "…"},
  "numeric_fields": [
    {"field": "body_text", "kind": "amount", "value": "۲۲۱/۰۰۰/۰۰۰", "value_ascii": "221/000/000",
     "confidence": "low", "tesseract_value": "321/000/000",
     "candidates": ["۲۲۱/۰۰۰/۰۰۰", "321/000/000"], "bbox": [812, 1440, 1102, 1488], "similarity": 0.89,
     "note": "vlm/tesseract disagree"}
  ],
  "numeric_summary": {"n_numeric": 4, "high": 2, "low": 1, "unverified": 1, "conflict_rate": 0.25},
  "primary":  {"model": "Qwen2.5-VL-7B-Instruct-Q4_K_M", "ok": true, "latency_s": 9.4, "grammar_used": true},
  "cross_check": {"enabled": false},
  "preprocess": {"cropped": true, "skew_deg": -1.4, "deskewed": true, "quality": 82, "out_bytes": 287113},
  "timing": {"preprocess_s": 0.6, "primary_s": 9.4, "numeric_validation_s": 1.8, "total_s": 11.9},
  "needs_review": true,
  "service": {"version": "0.1.0", "backend": "llamacpp", "normalization": "letterforms folded, digits preserved"}
}
```

`confidence` (since 2026-09-22, reader `glyph2`): `high` = the page reader read the
same digits; **`corrected`** = the page reader's digits replaced the model's — `value`
is what the model wrote, `resolved` (and the field text) carry the corrected number
in the model's own digit script; `low` = aligned but the page read was not confident
enough to override (both in `candidates`); `unverified` = nothing on the page aligns
with it; **`added`** = read from the page but absent from the model output (footer
numbers are also appended to `contact_info`; body-band numbers are records only, the
prose is never rewritten). `classical_value` is the page reader's value
(`tesseract_value` carries the same for compatibility). `needs_review` is true on any `low` **or `unverified`** number, on primary failure, or
on a secondary-model mismatch.

> **`needs_review` is triage, not a correctness gate (D62).** Measured on the 51 dev
> documents: `high` is 81% right, `corrected` 78%, `added` 68%, `unverified` 46%. The
> flag catches 18 of the 34 documents that hold at least one wrong number — an
> unflagged document can still be wrong. Anything that must not ship a wrong number
> has to read `numeric_fields[].confidence` per number, not the document boolean.

---

## Benchmark

```bash
venv312\Scripts\python.exe -m ocr_service.make_eval_set                  # ocr_eval/eval_set.jsonl (84 docs, 444 numbers)
venv312\Scripts\python.exe -m ocr_service.benchmark --split dev --primary qwen25  --label qwen25_dev
venv312\Scripts\python.exe -m ocr_service.benchmark --split dev --primary coreocr --label coreocr_dev
venv312\Scripts\python.exe -m ocr_service.benchmark --compare qwen25_dev coreocr_dev
```
Reports usable rate, CER (body and all fields), numeric recall **before** (VLM only) /
**after** (Tesseract decision) / **oracle** (any candidate), conflict rate, latency
p50/p95 per stage → `ocr_eval/benchmarks/bench_<label>.json` and a comparison table.
The eval set is derived from the human ground truth only (`ground_truth_real_fixed_v2.jsonl`);
never score the `test` split while tuning.

Tests: `venv312\Scripts\python.exe -m pytest tests/test_ocr_service.py` (no engine or
Tesseract needed; uses a fake reader with known-bad fixtures).

---

## What changed on 2026-09-22 (E15, E16)

### `cache_prompt` never reached the engine (D61)
 `Settings.cache_prompt=False` was
added after E11 to address session-dependent bodies (D52), and `config.provenance()`
has been reporting `"cache_prompt": false` ever since — but nothing put it in the
request, so llama.cpp ran with its own default (`true`). `LetterExtractor` now takes
`extra_body` and the llama.cpp backend passes the flag; `tests/test_numeric_v2.py`
asserts on the request rather than the config, because a value that is reported but
not applied is the same class of defect as the Phase-7 false green.

### Large amounts are preserved whole (D63)

A grouped number — an amount like `۳۲۱/۰۰۰/۰۰۰`, a date, a phone list — is now read,
corrected and injected **as one number with its separators exactly as printed**.

Three faults were fixed, all of which lost money-relevant digits silently:

* the reader **dropped trailing zeros**. A dot-zero is a third the height of a digit,
  and lines were seeded using a page-wide height percentile that noisy scans drag
  down, so an amount's last `۰` was clustered onto a neighbouring line and lost —
  `۳۲۱/۰۰۰/۰۰۰` was delivered as `۳۲۱/۰۰۰/۰۰`. A number now absorbs an adjacent digit
  whichever line it landed on, gated on the number's own **pitch**.
* injection judged an amount's **3-digit groups separately** against a 4-digit floor,
  so an amount the model omitted was discarded group by group. Grouped numbers are
  now injected whole and qualify on their total digit count.
* the **atom metric could not see either bug** (`۳۲۱/۰۰۰/۰۰` still yields `321` and
  `000`). Every benchmark now also reports **"grouped numbers intact"**.

Multi-group numbers are laid out right-to-left on the page, so the reader legitimately
sees `۰۳/۰۸/۱۴۰۳` for the date `۱۴۰۳/۰۸/۰۳`. Alignment accepts either order: the digits
come from the reader, the ordering from the model.

Measured on dev: grouped numbers delivered intact **13.6% → 41.8%**; numeric atom
recall 70.7% → 76.7%, precision 73.1% → 75.0%.

### Numbers are read from the page and corrected (E15)

**The v1 glyph reader could confirm but not correct** (D55/D56): measured alone
against the human GT it found 48% of the page's numbers at **10% precision** — it
had no "not a digit" class, so Persian letter fragments became digits, and as an
overrider it was net negative. Meanwhile the VLM's numbers are wrong 74% of the time
but *near-misses in the right field* for half of them (D53: generated, not read).

**`ocr_service/digit_reader_v2.py` is the default (`OCRS_NUMERIC_READER=glyph2`).**
A small CNN with a reject class, trained on rendered Persian text lines (Dehkhoda
words shaped through Presentation Forms-B + numbers in all three digit scripts, 137
fonts, scan augmentation; `digit_cnn_data.py` → `digit_cnn_train.py` in the torch
venv → `models/digit_cnn.npz`, run in numpy at inference: no new dependency). Page
locator: tall-glyph line seeding, dot-/ring-zero handling, colon suppression, runs
split at weak glyphs. Alone on the dev pages: **79% atom recall, 72% precision**.

**`ocr_service/numeric_reconcile.py` aligns and corrects.** Every digit run the model
wrote is matched to a page read (short numbers need an unambiguous match; multi-part
numbers such as `۲/۵۸۷/۸۰۹/۵۸۶` are aligned as a unit); a confident read replaces the
model's digits in place; confident footer numbers the model omitted are added to
`contact_info`. Replayed on the stored Qwen2.5-VL dev output: numeric atom recall
**33.6% → 68.2%**, precision **38.2% → 73.1%**, 236 corrections of which 83% are
right (13% were right before). Live and test-split numbers: see `ocr_eval/experiments.md` E15.

```bash
venv312\Scripts\python.exe -m ocr_service.digit_cnn_data --lines-per-font 100   # ~1 min -> models/digit_glyphs.npz
venv\Scripts\python.exe -m ocr_service.digit_cnn_train --epochs 15 --all-fonts    # ~40 s GPU -> models/digit_cnn.npz
venv312\Scripts\python.exe ocr_eval	oolsench_digit_reader.py --split dev --reader ocr_service.digit_reader_v2:GlyphReaderV2 --source prepared --label x
venv312\Scripts\python.exe ocr_eval	ools
eplay_reconcile.py --from qwen25_dev_glyph --label x   # policy change in seconds, no GPU
venv312\Scripts\python.exe -m pytest tests	est_numeric_v2.py -q
```

## What changed on 2026-09-17 (and why the design moved)

**Tesseract cannot read the digits in this typeface.** With Tesseract 5.4 installed
and `fas`/`ara` data (best and standard), a perfectly clean crop of `۰۹۲۴۴۲۴۱۱۷`
from a real letter came back as `۴۴۱۱۷` (fas) or garbage (ara). It reads the Persian
*words* on the same line almost perfectly and the bold footer phone numbers, but not
the body/header digits. So the mandatory "Tesseract numeric layer" is wired, working,
and — on this corpus — mostly returns `low`/`unverified`.

**`ocr_service/digit_reader.py` replaces it as the default classical reader.** Persian
digits never join, so each digit is one connected component; an SVM on HOG+pixel
features, trained on synthetic renders of `۰-۹`, `0-9` and separators across the 137
Persian fonts installed on this machine (the whole B-family, Mj-family, Tahoma, Arial,
Dubai …), reads the same crop as `۰۹2۴۴2۴117` — every digit value right. Synthetic
holdout: 93.8% digit-value accuracy. It locates numbers structurally (runs of ≥3
narrow isolated components on a text line) and is selected with
`OCRS_NUMERIC_READER=glyph` (default) / `tesseract` / `both`.

```bash
venv312\Scripts\python.exe -m ocr_service.digit_reader --train      # ~3 min CPU -> models/digits_hog_svm.joblib
venv312\Scripts\python.exe -m ocr_service.digit_reader --read crop.png
```

Each numeric field now also carries `resolved`: the value a consumer should use.
On a conflict the classical reading wins only when it is plausible (national-ID
checksum passes, or same digit count ±1 and ≥60% agreement) — otherwise the VLM value
stands and the conflict stays flagged.

**Tesseract install:** `deploy/install_tesseract.ps1` (winget `UB-Mannheim.TesseractOCR`
+ `fas`/`eng` traineddata). Program Files is not writable without elevation, so the
language data lives in `<repo>/tessdata` and is passed via `TESSDATA_PREFIX`;
`config.resolved_tesseract()` auto-detects both.

**One 12 GB GPU, two 7B models:** `OCRS_SECONDARY_MODE=cpu_offload` (default when
the secondary is enabled) starts coreOCR with `-ngl 0` in system RAM next to the GPU
primary, with a per-request timeout; `sequential` is the batch strategy the benchmark
uses (`--secondary sequential`: run the primary over the set, stop it, start the
secondary, cross-check offline). A VRAM guard (`OCRS_ENGINE_MIN_FREE_VRAM_MB`, 9000)
drops any engine to CPU instead of letting CUDA OOM kill the process.

**Windows asyncio:** engines are spawned synchronously before the event loop
(`OcrPipeline.start_engines()`); spawning from `asyncio.to_thread` aborted the
proactor self-pipe and killed the first benchmark.

**Queue without Redis:** `OCRS_REDIS_URL=filesystem://.celery_fs` uses kombu's
filesystem transport (needs `pywin32`) — the same `202 → /jobs/{id}` flow, verified
live. Redis itself: `deploy/run_redis_local.ps1` (Docker → Memurai → portable
redis-server; Memurai's MSI needs an elevated prompt).

**Image resolution:** `--image-max-tokens 16384` (native resolution) costs ~100 s per
page against ~10 s at the 4k default, because prompt eval runs at ~100 tok/s on the
9.3k-token images. Default stays 4096; the benchmark measures both on a subset.
