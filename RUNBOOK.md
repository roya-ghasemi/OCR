# Runbook — Persian OCR transcription service

How to run it, check it, measure it, and act on a failure. For someone picking this
up cold. What the service is: `README.md`. Why it is built this way: E19 in
`ocr_eval/experiments.md`.

---

## 1. Start

```powershell
cd E:\ghasemi\ocr
.\venv312\Scripts\python.exe -m uvicorn ocr_service.api:app --host 127.0.0.1 --port 8000
```

Ready in ~1 s (nothing to load). `GET http://127.0.0.1:8000/health` must say
`"status": "ok"`:

* `tesseract.available: true` and `langs` containing `fas` and `eng`
* `digit_reader.available: true`
* `service.tesseract.fas_sha256 == "99e420969b5ddd2c"` and
  `service.digit_model.sha256 == "e0e286aaf5595898"` — the files every published
  number was measured with.

`degraded` = the digit reader is missing (numbers then come from Tesseract only).
`down` = Tesseract or its Persian data is missing; `/ocr` returns 503.

**Port already in use** (`WinError 10048`): another server holds 8000.
```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen | ForEach-Object { Get-Process -Id $_.OwningProcess }
```

## 2. Use

```powershell
curl.exe -F "file=@page.jpg" http://127.0.0.1:8000/ocr
```
or the upload form at http://127.0.0.1:8000/docs. Read `text`; check `review_reasons`.

| status | meaning |
|---|---|
| 200 | transcript (possibly empty `text` + `needs_review` when nothing legible was found) |
| 400 | empty upload, not an image, unsupported format |
| 413 | larger than `OCRS_MAX_UPLOAD_MB` (25) |
| 503 | Tesseract/Persian data unavailable |

## 3. Measure

```powershell
.\venv312\Scripts\python.exe ocr_eval\tools\bench_fulltext.py --split dev
```
~3 min for 51 documents, CPU only. Compare `transcript.coverage_cer`,
`atom_recall`, `number_exact_recall` with E19 (dev: 14.9%, 81.6%, 67.2%). Change one
thing at a time, tune on dev, log the run in `ocr_eval/experiments.md` with the
`provenance` block the tool writes. The test split (`--split test`) is for a final
check only — it has been scored seven times in total (E17, E18, E19 ×5); make a
fresh split before the next tuning cycle.

```powershell
.\venv312\Scripts\python.exe -m pytest -q
```
Includes the phase-0 guards: no language model and no dictionary correction in the
runtime path, frozen splits, pinned engine files.

## 4. When the output is wrong

| symptom | where to look |
|---|---|
| words missing at one end of a line | `Transcriber._select` — the readings compared per line (`tesseract_s` candidates); a merged reading must not beat clean halves |
| a whole line missing | line candidates in `Transcriber._candidates`; try `OCRS_LAYOUT_PSMS` variants on dev |
| junk tokens at line ends | `Transcriber._trim_edges`, `OCRS_MIN_LINE_CONF` |
| words of a table row in the wrong order | `Transcriber._order_segments` (gap > 1.2 line heights) |
| a number wrong | `numbers[]`: `source: tesseract` means the digit reader did not confirm it; reader: `ocr_service/digit_reader_v2.py` |
| letterhead / handwriting / text under a stamp missing | known limits D67–D68 in `ocr_eval/error_register.md` |
| a letter field null or wrong | `ocr_service/letter_fields.py` rules; fields are never generated, only cut from `text` |
| everything worse after a machine move | `/health` hashes: a distro `fas.traineddata` is a different, weaker model |
