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
* `spellfix.available: true` with ~13,892 words (`enabled: false` if `OCRS_SPELLFIX=0`)
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
or the page at **http://127.0.0.1:8000/** — drop an image in and the transcript comes
back laid out as it is printed (paragraphs right-to-left, a table where the page has a
table, the letter's fields on top, every number with its verdict). `GET /docs` is the
API's own form. Read `text`; check `review_reasons`.

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
`atom_recall`, `number_exact_recall` with E24 (dev: 14.6%, 83.6%, 75.8%). Change one
thing at a time, tune on dev, log the run in `ocr_eval/experiments.md` with the
`provenance` block the tool writes. The test split (`--split test`) is for a final
check only — it has been scored nine times in total (E17, E18, E19 ×5, E20, E24); make a
fresh split before the next tuning cycle.

```powershell
.\venv312\Scripts\python.exe ocr_eval\tools\score_sample.py     # general documents, not letters
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
| an amount has a group that is not three digits | it is already flagged: `numbers[].grouping == "broken"` and a `review_reasons` line. The service never repairs one — the missing digit would have to be invented. Check it against the page |
| a spurious `۰` at the end of a number | `GlyphReaderV2._finish` — the end-trim wants P(zero) ≥ 0.4 and the run's own spacing (D72) |
| a `/` or `:` inside a number that is not on the page | `Transcriber._fix_numbers` — two reads are joined only when they touch, and never with an invented separator (D73) |
| a stretched word gains a letter («بلــوار» → «بلسوار») | `squeeze_kashida()` in `transcribe.py`; it only cuts a flat stroke standing on the baseline and joined at both ends (D74) |
| letterhead / handwriting / text under a stamp missing | known limits D67–D68 in `ocr_eval/error_register.md` |
| Persian typos everywhere («صبخگاهی», «یرای», «توسمعه») | check `glyph_px` first. Under ~20 px the dots of ب/ی/پ/ن and ح/خ/ج are not in the image; the service upsamples to ~30 px and flags it, but the fix is a 300 DPI rescan (D76). A spell-corrector was measured for this and rejected — 3 fixes to 11 corruptions (D77) |
| a word was silently changed | `ocr_service/spellfix.py` — it may only move dots (a word is replaced by one with the same skeleton), and only where the reader was unsure. `OCRS_SPELLFIX=0` turns it off; `/health` shows whether the list loaded |
| a correct word was "corrected" | it is missing from Tesseract's Persian list, so it looked like a non-word (`کتبا` is the known case, D80). Add it to `ocr_service/data/fas_words.txt` |
| a non-letter page looks "empty" | it is not: `fields` are null by design on a non-letter (D65), the whole page is in `text` and `lines` |
| a letter field null or wrong | `ocr_service/letter_fields.py` rules; fields are never generated, only cut from `text` |
| everything worse after a machine move | `/health` hashes: a distro `fas.traineddata` is a different, weaker model |
