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

## 2c. The transcript cache

A page already read is served from `.cache/ocr` instead of being read again — about 4 s of
CPU down to 10 ms. `/health` reports `cache` with its hit and miss counts.

```powershell
$env:OCRS_CACHE = "0"          # off
$env:OCRS_CACHE_DIR = "D:\scratch\ocr-cache"
$env:OCRS_CACHE_MAX_ENTRIES = "500"
```

**The key is not the image.** It folds in the model hashes, the service version and every
setting that changes the output (`cache.KEYED_SETTINGS`), because a transcript is a function
of the image *and* of what read it. Keyed on the image alone, the first thing that happens
after a model change is that the service serves a transcript the current engine would never
produce — carrying the new model's hash, because `provenance()` is stamped at response time.

So **changing a model or a setting needs no cache clearing**: the old entries stop being
looked up and age out, and rolling the change back finds its work still there. Deleting
`.cache/` is always safe.

A request asking for `prepared=true` is read properly rather than served from here: the
prepared image is megabytes of base64 that only the viewer wants, so it is not stored.

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

---

## 5. Deploying to a server

### 5.1 What to copy

Everything the service needs, and nothing it does not. Sizes are the deciding factor only
for the two that are large.

```
ocr_service/                 the package (~190 KB of .py)
  data/fas_words.txt         the spell-fixer's lexicon      168 KB
  data/fas_admin.txt         administrative vocabulary        2 KB
  static/index.html          the viewer                      24 KB
models/digit_cnn.npz         the CNN digit reader           621 KB
models/digit_cnn.json        its architecture — a provenance record, not loaded at
                             runtime; the reader builds from the .npz alone   3 KB
tessdata/fas_custom.traineddata   the E30 fine-tune, the default model   3.2 MB
requirements-service.txt     pinned dependencies
requirements.txt             points at the above
deploy/                      Dockerfile, docker-compose.yml, nginx.conf, ocr-service.service
RUNBOOK.md                   this file
```

Plus two stock Tesseract models that are **downloads, not repository files** — the Dockerfile
fetches them; a bare-metal install needs them beside `fas_custom.traineddata`:

```
fas.traineddata   tessdata_best   the rollback target, and nothing else reads Persian if the fine-tune is removed
eng.traineddata   tessdata_fast   Latin runs inside Persian text
```

**Leave behind.** None of this is read at runtime; it is measured, not guessed:

| | why |
|---|---|
| `models/digits_hog_svm.joblib` | **120 MB**, the v1 reader. `joblib` and `sklearn` are never imported during a request — v2 superseded it, and only its `_fonts` helper is still used |
| `models/digit_glyphs.npz` | 14 MB of digit training data |
| `ocr_service/fonts/` | 816 KB, for rendering synthetic training lines. No font file is opened during a request |
| `ocr_service/digit_cnn_train.py` | needs torch, which is not a service dependency |
| `ocr_train/`, `train/` | the fine-tuning pipeline and its working files |
| `ocr_eval/`, `tests/` | benchmarks, ground truth, the test suite |
| `Persian-OCR-Double-Benchmark/` | 2.2 GB dataset (and there is a second copy at the repo root) |
| `Dehkhoda-SQL-master/`, `dehkhoda/` | dictionary build inputs |
| `control/`, `debug_raw/`, `graft/`, `hf_cache/`, `install_log.txt` | scratch, index, logs |
| `venv/`, `venv312/` | build these on the target, do not copy them |

`ocr_service/numeric_reconcile.py` and `ocr_service/preprocess.py` are not imported by
anything either. They are 25 KB and shipping the package whole is simpler than pruning it
file by file; both are listed here so the list is honest, not because they cost anything.

### 5.2 System dependencies

* **Python 3.12** (3.12.10 is what every published number was measured on).
* **Tesseract 5** — `apt-get install tesseract-ocr`, or `deploy/install_tesseract.ps1` on
  Windows. The distro's own `fas` package is a **different, weaker model**; use the
  tessdata_best file instead, which is what the Dockerfile fetches.
* `libgl1` and `libglib2.0-0` for OpenCV on a slim image.
* **Redis**, only for `OCRS_ASYNC_MODE=1`. The default synchronous mode needs none.

```bash
python3.12 -m venv /srv/ocr/venv && /srv/ocr/venv/bin/pip install -r requirements-service.txt
```

### 5.3 Environment

| variable | default | when to set it |
|---|---|---|
| `OCRS_LANG_FA` | `fas_custom` | `fas` rolls back to the stock model |
| `OCRS_TESSDATA_DIR` | *(found)* | only when the models are outside the checkout. **It is the directory itself, not its parent** |
| `OCRS_TESSERACT_CMD` | *(found)* | when the binary is not on `PATH` or in the usual place |
| `OCRS_CACHE_DIR` | `<repo>/.cache/ocr` | point it outside the checkout so a redeploy keeps the cache |
| `OCRS_CACHE` | `1` | `0` disables it |
| `OCRS_WORKERS` | `8` | parallel Tesseract processes **per request** |
| `OCRS_MAX_UPLOAD_MB` | `25` | |
| `OCRS_ASYNC_MODE` | `0` | `1` returns a `job_id` and needs Redis and a Celery worker |
| `OCRS_REDIS_URL` | `redis://127.0.0.1:6379/0` | with async mode |
| `OCRS_SPELLFIX` | `1` | `0` turns off dot-confusion repair |

A stale `OCRS_TESSDATA_DIR` is the failure that looks like a broken install: `/health` says
`down` with `langs: []`, naming a directory that does not exist. See section 1.

### 5.4 Running it

```bash
/srv/ocr/venv/bin/uvicorn ocr_service.api:app --host 127.0.0.1 --port 8000 --workers 2 \
    --proxy-headers --forwarded-allow-ips='*'
```

**On worker count.** Each request already fans out to `OCRS_WORKERS` (8) parallel Tesseract
processes, so uvicorn workers multiply that: 4 × 8 is 32 processes contending for the same
cores and is *slower* than 2 × 8 on most hosts. Start at one uvicorn worker per 2–3 cores and
measure. Gunicorn adds nothing here — uvicorn's own worker manager is the same mechanism
without the extra layer.

Behind nginx, raise the proxy timeout: a large scan legitimately takes 30 s, and the default
60 s read timeout is closer than it looks when several arrive at once. `deploy/nginx.conf`
has a working configuration.

### 5.5 Managing it

**systemd** — `deploy/ocr-service.service` is a complete unit; edit `User`,
`WorkingDirectory` and the venv path.

```bash
sudo cp deploy/ocr-service.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now ocr-service
journalctl -u ocr-service -f
```

Its `ProtectSystem=strict` means **`ReadWritePaths` must include the cache directory**, or
every write fails and every page is read twice with nothing in the log to say why.

**Docker** — the image carries `fas_custom` from the repository and downloads the two stock
models:

```bash
docker compose -f deploy/docker-compose.yml up --build --scale api=2 --scale worker=2
```

The api and worker replicas share one cache volume. Entries are content-addressed and
written beside-then-replaced, so concurrent access is safe.

### 5.6 Verifying a deployment

```bash
curl -s localhost:8000/health
```

Four things decide whether it is really working, not just answering:

1. `status` is `ok` — `degraded` means the digit reader is missing, so numbers come from
   Tesseract alone and the measured number accuracy does not apply.
2. `tesseract.lang_fa` is `fas_custom` and `service.tesseract.lang_fa_sha256` is
   **`d0e5099d3ea227ba`**. A different hash is a different model and every published number
   stops describing it.
3. `tesseract.langs` contains `fas_custom`, `fas` and `eng`.
4. `spellfix.words` is about 13,808.

Then read a page end to end and check the transcript is not empty:

```bash
curl -s -F "file=@page.jpg" localhost:8000/ocr | head -c 400
```
