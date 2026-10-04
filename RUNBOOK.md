# Runbook — Persian OCR transcription service

How to run it, check it, measure it, and act on a failure. For someone picking this
up cold. What the service is: `README.md`. Why it is built this way: E19 in
`ocr_eval/experiments.md`.

---

## 1. Start

```powershell
cd D:\OCR_ghasemi\ocr
.\venv312\Scripts\python.exe -m uvicorn ocr_service.api:app --host 127.0.0.1 --port 8000
```

Ready in ~1 s (nothing to load). `GET http://127.0.0.1:8000/health` must say
`"status": "ok"`:

* `tesseract.available: true` and `langs` containing `fas`, `fas_custom` and `eng`
* `tesseract.lang_fa == "fas_custom"` — the model that will actually read the page
* `digit_reader.available: true`
* `spellfix.available: true` with ~13,892 words (`enabled: false` if `OCRS_SPELLFIX=0`)
* `service.tesseract.lang_fa_sha256 == "d0e5099d3ea227ba"` and
  `service.digit_model.sha256 == "e0e286aaf5595898"` — the files every published
  number was measured with. `fas_sha256` stays `99e420969b5ddd2c`: the stock model is
  still there as the way back, it is just not the one reading.

`degraded` = the digit reader is missing (numbers then come from Tesseract only).
`down` = Tesseract or its Persian data is missing; `/ocr` returns 503.

**`down` with `langs: []` after the project folder moved**: check
`tesseract.tessdata_dir` in `/health`. `OCRS_TESSDATA_DIR` is a *persistent user*
environment variable on this machine and still named the old path, so the service was
looking for `fas.traineddata` in a directory that no longer exists. Left unset the service
finds `<repo>/tessdata` itself, which is what every published number was measured with:

```powershell
[Environment]::SetEnvironmentVariable('OCRS_TESSDATA_DIR', $null, 'User')   # remove it for good
$env:OCRS_TESSDATA_DIR = ""                                                 # or just this session
```

An explicitly configured directory is honoured even when it is wrong — the service will not
quietly read a different model than it was told to. `/health` names the path it used.

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

## 2b. Which Persian model is reading

`OCRS_LANG_FA` selects it. The default is **`fas_custom`**, the E30 fine-tune, whose
`lang_fa_sha256` is `d0e5099d3ea227ba`. `fas` is the stock tessdata_best model and the way
back:

```powershell
$env:OCRS_LANG_FA = "fas"           # roll back to the stock model
```

Both files live in `tessdata/`, so switching is a setting and rolling back is changing a
string. `/health` reports `tesseract.lang_fa`, and `provenance()` stamps `lang_fa_sha256` —
the hash of the model that actually did the reading, which is not `fas_sha256`. A results
file that stamped only the stock hash would describe a run that never happened.

Why it is the default, measured on 84 documents neither model was tuned on:

| | `fas` | `fas_custom` | |
|---|---:|---:|---|
| dev 51 — CER / atoms / numbers | 13.66% / 83.74% / 77.13% | **12.86% / 86.07% / 78.79%** | all better |
| test 33 — CER / atoms / numbers | 18.30% / 79.04% / 67.03% | **17.57% / 81.22% / 69.53%** | all better |

Every field improves on the test split. Two things to know before trusting it blindly:

* **The single hand-typed letter page is worse** (WER 12.07% → 16.09%), while 84 documents of
  the same kind are better. The regression is four lines, one of which is a genuine fix
  («میکائبل» → «میکائیل»); the numbers on the line that got worse are ones the digit reader
  already refuses to confirm under either model. E30 in `ocr_eval/experiments.md` has the
  alignment.
* **`len_ratio` falls about two points** on both splits: it returns slightly less text. With
  CER down and atom recall up, what it stops returning was wrong — but that is an inference,
  not a measurement.

The fine-tune is built by `ocr_train/` (see its README). Rebuild and reinstall with
`python -m ocr_train segments` then `install`; `train/work/segments.jsonl` is the ledger of
every segment and its measured page error.

## 3. Measure

```powershell
.\venv312\Scripts\python.exe ocr_eval\tools\bench_fulltext.py --split dev
```
~3 min for 51 documents, CPU only. Compare `transcript.coverage_cer`,
`atom_recall`, `number_exact_recall` with E30 (dev: **12.86%, 86.07%, 78.79%**; the stock
model on the same corpus is 13.66%, 83.74%, 77.13%). Change one
thing at a time, tune on dev, log the run in `ocr_eval/experiments.md` with the
`provenance` block the tool writes. The test split (`--split test`) is for a final
check only — it has been scored ten times in total (E17, E18, E19 ×5, E20, E24, E25); make a
fresh split before the next tuning cycle.

```powershell
.\venv312\Scripts\python.exe ocr_eval\tools\bench_pages.py --label after --compare before
```
Whole-page **CER and WER** against a hand-typed transcript of everything printed on the
page, plus *unsupported lines* — lines the service printed that are not on the page — and
the page's structure against what it actually has. Three pages in `ocr_eval/pages/`, seconds
to run. This is the only instrument that sees word error, a page that is not a letter, or
invented text; the corpus benchmark above sees none of the three. E28: letter 10.1% / 12.1%,
prose photo 3.9% / 13.3%, low-resolution photo 7.4% / 22.2%, and **0 unsupported lines on
all three**.

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
| the desk, a shadow or the next sheet is being read as text | the page was not cut out of the frame: `page_prep.crop_page`. It only fires when under 90% of the frame is page (`FULL_FRAME_FRAC`), because cropping a scan was measured to make it worse |
| a stamp is sitting on top of the signature block | `ink.drop_blue_ink` erases blue ink but never a dark pixel, so text under the stamp survives. `OCRS_INK_SEPARATE=0` turns it off |
| a handwritten number or date is missing from `text` | by design (E28). It is in `unread[]` with its box and what the recogniser offered; the viewer shows it as a cut-out under «نیاز به بازبینی» |
| a real word was dropped as unreadable | `Transcriber._unread`. A run is only dropped when nothing in it is a known word, the reader was unsure, and every token is short — raise the length bound before the confidence one |
| boxes do not line up with the image | they are measured in the **prepared** page. Use the «تصویر بهبودیافته» toggle, or `POST /ocr?prepared=true` |
| a paragraph is reported as a table | `Transcriber._table_cuts`: a wide gap only counts as a column when the row above or below breaks in the same place. Justified Persian is stretched by widening its spaces |
| a correct word was changed | `ocr_service/spellfix.py`. It may move dots, close a ZWNJ, or delete one character, and only ever returns a word that is already in the list. Add missing administrative words to `ocr_service/data/fas_admin.txt`, not to the general list |
| a number wrong | `numbers[]`: `source: tesseract` means the digit reader did not confirm it; reader: `ocr_service/digit_reader_v2.py` |
| an amount has a group that is not three digits | it is already flagged: `numbers[].grouping == "broken"` and a `review_reasons` line. The service never repairs one — the missing digit would have to be invented. Check it against the page |
| a spurious `۰` at the end of a number | `GlyphReaderV2._finish` — the end-trim wants P(zero) ≥ 0.4 and the run's own spacing (D72) |
| a `/` or `:` inside a number that is not on the page | `Transcriber._fix_numbers` — two reads are joined only when they touch, and never with an invented separator (D73) |
| a stretched word gains a letter («بلــوار» → «بلسوار») | `squeeze_kashida()` in `transcribe.py`; it only cuts a flat stroke standing on the baseline and joined at both ends (D74) |
| letterhead / handwriting / text under a stamp missing | known limits D67–D68 in `ocr_eval/error_register.md` |
| Persian typos everywhere («صبخگاهی», «یرای», «توسمعه») | check `glyph_px` first. Under ~20 px the dots of ب/ی/پ/ن and ح/خ/ج are not in the image; the service upsamples to ~30 px and flags it, but the fix is a 300 DPI rescan (D76). A spell-corrector was measured for this and rejected — 3 fixes to 11 corruptions (D77) |
| a comma looks wrong («»», «۰» where «،» belongs) | `Transcriber._fix_marks` — it restores «،» only where a Persian letter is on its left; quotations, table cells and numbers are exempt (D81) |
| a word was silently changed | `ocr_service/spellfix.py` — it may only move dots (a word is replaced by one with the same skeleton), and only where the reader was unsure. `OCRS_SPELLFIX=0` turns it off; `/health` shows whether the list loaded |
| a correct word was "corrected" | it is missing from Tesseract's Persian list, so it looked like a non-word (`کتبا` is the known case, D80). Add it to `ocr_service/data/fas_words.txt` |
| a non-letter page looks "empty" | it is not: `fields` are null by design on a non-letter (D65), the whole page is in `text` and `lines` |
| a letter field null or wrong | `ocr_service/letter_fields.py` rules; fields are never generated, only cut from `text` |
| everything worse after a machine move | `/health` hashes: a distro `fas.traineddata` is a different, weaker model |
