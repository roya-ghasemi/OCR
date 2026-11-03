# Architecture

## Request lifecycle

```
POST /ocr (multipart image)
  api.py          size / format check (400, 413) · sync, or queued via Celery (202 + /jobs/{id})
  pipeline.py     OcrPipeline.run → at most OCRS_MAX_CONCURRENT_PAGES at once → run_sync
                  cache lookup (key = image + model hashes + settings)
  transcribe.py   Transcriber.transcribe
     1 page_prep.crop_page(): cut the sheet out of a photo · ink.separate(): erase blue stamps
     2 imaging.text_skew() · rotate only if |skew| >= 1°
     3 imaging.working_scale(): 2x for photos < 2000 px, up to 4x when glyphs < 20-24 px
     4 imaging.flatten() + clean_ink(): binary page, rules removed, specks removed
     5 candidates, read in one pool of OCRS_WORKERS Tesseract processes:
         - whole-page layout passes (OCRS_LAYOUT_PSMS 3,4,6; OCRS_TEXT_PSMS 11, whose
           lines compete only where they hold no digit)
         - imaging.text_lines() crops, kashida squeezed, read as single lines (psm 7, psm 13
           where 7 is unsure) in Persian and English; tesseract.py hands each batch of
           crops to one process through a list file
     6 _select(): per physical line the reading with the most confident characters wins
     7 _fix_numbers(): CNN digit reader (digit_reader_v2) replaces number tokens it confirms
     8 _finish_rows() · _assemble(): table cells, edge marks, unread regions, paragraphs,
       dot-confusion repair (spellfix.py), Persian comma, numbers list
  pipeline.py     Arabic re-read when the page looks Arabic · legibility verdict ·
                  letter fields (letter_fields.py) · letterhead (letterhead.py) ·
                  value shape checks (validate.py) · review reasons
  schemas.py      OcrResponse
```

No language model, no network call. `ocr_eval/test_phase0_guards.py` fails if one is
introduced into the runtime files.

## Models

| file | size | source | checked by |
|---|---:|---|---|
| `models/tesseract/fas_custom.traineddata` | 3.3 MB | fine-tune, this repository (`ocr_train/`) | sha256 in `/health`, Dockerfile |
| `fas.traineddata` | 3.3 MB | tessdata_best (pinned commit) | sha256 in `/health`, Dockerfile, guard test |
| `eng.traineddata` | 4.1 MB | tessdata_fast (pinned commit) | sha256 in `/health`, Dockerfile |
| `ara.traineddata` | 12.6 MB | tessdata_best (pinned commit), optional | Dockerfile |
| `tools/tesseract/` (program, Windows) | 239 MB | UB-Mannheim 5.4.0 build, copied by `deploy/install_tesseract.ps1` | version in `/health` |
| `models/digit_cnn.npz` | 0.6 MB | trained by `ocr_research/digit_cnn_train.py` | sha256 in `/health`, guard test |

## Packages

| package | role | shipped |
|---|---|---|
| `ocr_service/` | the service | yes |
| `ocr_research/` | digit-CNN training, v1 digit reader, helpers of the earlier VLM path | no |
| `ocr_train/` | Tesseract fine-tuning pipeline | no |
| `ocr_eval/` | ground truth, benchmarks, reports | no |
| `dehkhoda/` | dictionary-correction experiment, not adopted (`dehkhoda/REMOVED.md`) | no |

## Service modules

| file | role |
|---|---|
| `api.py`, `tasks.py`, `schemas.py`, `config.py` | HTTP, queue, response contract, settings |
| `pipeline.py` | request flow, cache, review reasons |
| `transcribe.py` | candidate reading, selection, assembly |
| `imaging.py` | binarisation, rule removal, skew, line finding, kashida |
| `tesseract.py` | Tesseract process runner (single image and batched) |
| `page_prep.py`, `ink.py` | sheet crop and perspective, stamp removal |
| `digit_reader_v2.py`, `glyph_features.py` | CNN digit reader and its glyph features |
| `spellfix.py`, `legible.py`, `script_id.py`, `validate.py` | dot repair, legibility, script detection, value shapes |
| `letter_fields.py`, `letterhead.py` | letter fields and letterhead |
| `cache.py` | on-disk transcript cache |
