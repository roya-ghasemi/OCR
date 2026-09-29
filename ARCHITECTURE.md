# Architecture

## Request lifecycle

```
POST /ocr (multipart image)
  api.py          size / format check (400, 413) · sync, or queued via Celery (202 + /jobs/{id})
  pipeline.py     OcrPipeline.run → thread → run_sync
  transcribe.py   Transcriber.transcribe
     1 EXIF orientation · text_skew() · rotate only if |skew| >= 1°
     2 working_scale(): 2x for photos < 2000 px, up to 4x when glyphs < 24 px, cap 3500 px
     3 flatten(): grey / blurred background → binary; clean_ink(): rules removed pixel-wise
       with kernels sized from page AND text height, specks removed unless next to a glyph
     4 candidates: Tesseract layout passes (psm 3, 4, 6; fas) + text_lines() (glyph-sized
       components only) read per line with psm 7 → 13 fallback, in fas and in eng
     5 _select(): per physical line the reading with the most confident characters wins;
       rows top-down, cells right-to-left
     6 _fix_numbers(): CNN digit reader (digit_reader_v2, models/digit_cnn.npz) replaces a
       number token where it saw enough confident digits at the same place
     7 _order_segments() (table cells RTL) · _trim_edges() (lone far marks) · _line_ok()
     8 _assemble(): paragraphs from line pitch, Arabic ي/ك folded to Persian, numbers list
  letter_fields.py  rules over the lines: salutation, addressee above it, printed «موضوع:»,
                    contact block in the bottom third, organisation line above the addressee;
                    verbatim or null; not a letter → all null
  schemas.py      OcrResponse: text, lines, numbers, is_letter, fields, needs_review, ...
```

No language model, no dictionary, no network call. Guarded by
`ocr_eval/test_phase0_guards.py` (`test_no_language_model_in_runtime`,
`test_no_lexicon_stage_in_runtime`, `test_engine_identity_is_pinned`).

## Dependencies that decide accuracy

| what | where | pinned by |
|---|---|---|
| Tesseract 5.4 engine | `C:\Program Files\Tesseract-OCR` (dev), distro package (docker) | `/health` version |
| `fas.traineddata` (tessdata_best), `eng.traineddata` | `<repo>/tessdata` (dev), `/opt/tessdata` (docker) | sha256 in `/health`, guard test |
| CNN digit reader weights | `<repo>/models/digit_cnn.npz` | sha256 in `/health`, guard test |

A git worktree finds `tessdata/` and `models/` in the main checkout (`config._find_up`).

## Files

| file | role |
|---|---|
| `ocr_service/transcribe.py` | recognition (everything above step 8) |
| `ocr_service/letter_fields.py` | rule-based letter fields |
| `ocr_service/pipeline.py`, `api.py`, `tasks.py`, `schemas.py`, `config.py` | service |
| `ocr_service/digit_reader_v2.py`, `digit_cnn_data.py`, `digit_cnn_train.py` | digit reader and its training |
| `ocr_service/numeric_validator.py`, `numeric_reconcile.py`, `digit_reader.py`, `preprocess.py` | helpers from the earlier VLM path, kept with their tests |
| `ocr_eval/fulltext_score.py`, `ocr_eval/tools/bench_fulltext.py` | full-text benchmark (E19) |
| `deploy/` | Dockerfile, compose (nginx, api, worker, redis), Tesseract installer |
