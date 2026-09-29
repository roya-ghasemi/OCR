# What changed — OCR accuracy eval + Dehkhoda dictionary

Everything below was added for the accuracy-testing task (and the Dehkhoda dictionary
extension). Nothing in the OCR model or the original inference logic was altered by the
eval work itself; the only change to the running service is one **new, opt-in** endpoint.

Legend:  🟢 new code  ·  🟡 modified code  ·  ⚪ generated output (produced by running the code, safe to delete/regenerate)

---

## 2026-09-29 — E22: stop printing what is not text

| Status | File | What it is |
|---|---|---|
| 🟡 | `ocr_service/transcribe.py` | `_line_ok` drops a line of nothing but 1-2 character fragments below 60 confidence (handwriting, stamps, registration blocks); `_trim_edges` cuts a whole junk tail across the line's widest gap, not just one lone character |
| 🟡 | `ocr_service/static/index.html` | The fields card no longer repeats `body_text` — it is the document rendered below it |
| 🟡 | `tests/test_transcribe.py` | Fixture corrected: it had Tesseract seeing 7 digits where the reader returned 8, the exact ambiguity the new rule resolves |

`_1`: 17 → 13 lines, all three noise lines gone, «با سلام؛ سح» → «با سلام؛». Dev CER
14.65% → 14.69%, whole numbers unchanged at 75.8%, len_ratio 0.948 → 0.915 — 3% less
text out and no more error, so what left was noise (D78).

Tried and reverted in the same pass: believing Tesseract over the digit reader when the
two disagree on digit count. It fixes «۱۴۰۳۵» → «۱۴۰۳» on the invoice and costs **1.7
points of whole-number accuracy** across dev (75.8% → 74.1%).

## 2026-09-29 — E21: the typos are the capture, not the reader

| Status | File | What it is |
|---|---|---|
| 🟡 | `ocr_service/transcribe.py` | `working_scale` resamples a page whose text is under 20 px **as supplied** up to ~30 px (Lanczos, ≤4x). Keyed on the native height so the 24-42 px corpus scans are untouched — verified on all 90, scale unchanged on every one. `glyph_px` is measured and carried into the response |
| 🟡 | `ocr_service/pipeline.py`, `schemas.py` | `glyph_px` in the response, and a review reason naming the DPI to rescan at when it is under 20 |
| 🟡 | `ocr_service/static/index.html` | A red banner for a low-resolution page, and a note on a page that is not a letter saying how much was extracted (null letter fields were being read as "empty output") |
| 🟢 | `ocr_eval/samples/prose_page.jpg`, `prose_page.gt.txt` | The general-document fixture and its hand-typed transcript — the corpus benchmark only covers administrative letters |
| 🟢 | `ocr_eval/tools/score_sample.py` | Scores any sample with a `.gt.txt` beside it: whole-page CER, word recall, length |
| 🟡 | `tests/test_transcribe.py` | Three guards: a non-letter page comes back whole and in order; a low-resolution capture is upsampled and a good scan is not; the resolution warning fires on 12 px and 7 px and not on 31 px |
| 🔴 | — | **No spell-corrector.** Built to the `dehkhoda/REMOVED.md` policy and measured: 3 fixes, 11 corruptions, and no candidate at all for the reported word. `test_no_lexicon_stage_in_runtime` stands unmodified (D77) |

## 2026-09-29 — E20: amounts that can be trusted, kashida, and a page to read the result on

| Status | File | What it is |
|---|---|---|
| 🟢 | `ocr_service/static/index.html` | The upload page: drop an image, get the page back laid out as it is printed — right-to-left paragraphs, letter fields on top, a real `<table>` where the page has a table, every number with its source and its thousands-grouping verdict, and the line boxes over the original. Vanilla HTML/CSS/JS, no CDN, works offline |
| 🟡 | `ocr_service/api.py` | Serves that page at `/` and its assets at `/static` |
| 🟡 | `ocr_service/digit_reader_v2.py` | `_finish`: a short glyph is only re-judged a dot-zero when the classifier gives zero real weight, and the run's ends are trimmed **after** that, against the run's own digit spacing (D72) |
| 🟡 | `ocr_service/transcribe.py` | `amount_grouping()` (thousands-group validation, flags, never repairs — D75); `squeeze_kashida()` + column remap in `_read_line` (D74); `_fix_numbers` joins two reads of one number without inventing a separator (D73); `_cells()` puts a line's table cells in the response |
| 🟡 | `ocr_service/schemas.py`, `pipeline.py` | `NumberOut.grouping`, `LineOut.cells`, and a review reason naming any amount whose groups are not whole |
| 🟡 | `tests/test_transcribe.py` | Four guards: grouping verdicts, no invented separator, cells vs prose, kashida cut back while a dash is not |

## 2026-09-27 — E19: full-text transcription of any image; the language-model path is removed

| Status | File | What it is |
|---|---|---|
| 🟢 | `ocr_service/transcribe.py` | The recogniser: deskew, binarise, rule/speck removal, line candidates from Tesseract psm 3/4/6 + own line finder (psm 7→13, fas and eng), best reading per line, CNN digit reader replaces confirmed numbers, table cells right-to-left, junk filter. CPU only |
| 🟢 | `ocr_service/letter_fields.py` | Letter fields cut from the transcript by rules — verbatim lines or null, never generated; non-letters get none |
| 🟡 | `ocr_service/pipeline.py`, `api.py`, `schemas.py`, `config.py`, `tasks.py`, `__init__.py` | Service rewired: response is `text` + `lines` + `numbers` (+ `fields` for letters); settings for Tesseract, digit model, thresholds; `/health` reports the traineddata and digit-model hashes |
| 🟢 | `ocr_eval/fulltext_score.py`, `ocr_eval/tools/bench_fulltext.py` | Full-page scoring (coverage CER, digit atoms, whole numbers) and the benchmark that runs the service in-process against the previous service's stored predictions |
| 🟢 | `tests/test_transcribe.py` | Rules, line selection, rule removal, table order, and an end-to-end rendered page |
| 🟡 | `ocr_eval/test_phase0_guards.py` | Guards moved to the new runtime: no language model, no lexicon stage, engine files pinned |
| 🟡 | `tests/test_numeric_v2.py`, `tests/test_ocr_service.py` | VLM-path tests removed; contract test for the new response; digit model found from a worktree |
| 🟡 | `deploy/Dockerfile`, `deploy/docker-compose.yml`, `requirements*.txt` | CPU-only stack; tessdata_best in the image; no llama.cpp services, no `openai` |
| 🟡 | `README.md`, `RUNBOOK.md`, `ARCHITECTURE.md` | Rewritten for the new service; `README_SERVICE.md` folded into `README.md` |
| 🟡 | `ocr_eval/experiments.md` (E19), `ocr_eval/error_register.md` (D64–D69) | Records |
| ⚪ | `ocr_eval/benchmarks/bench_fulltext_{dev,test}.json` | Benchmark outputs |
| 🔴 | `main.py`, `main_qwen.py`, `engine.py`, `config.py`, `json_repair.py`, `test_json_repair.py`, `test_ocr.py`, `_test_client.py`, `_dl_model.py`, `download_model.py`, `*.bak` | The coreOCR/LM Studio serving path |
| 🔴 | `ocr_pipeline/`, `ocr_service/backends/`, `ocr_service/benchmark.py`, `ocr_service/make_eval_set.py`, `tests/test_ocr_pipeline.py` | The Qwen2.5-VL letter-JSON path (grammar, sampling, extraction) |
| 🔴 | `ocr_eval/run_real.py`, `run_eval.py`, `run_experiment.py`, `run_pipeline_ab.py`, `ci_gate.py`, `analyze_fields.py`, `tools/regen_one.py`, `experiments/{orientation,repeat_stability}/run*.py`, `experiments/diagnostics_0913/novel_headers.py` | Runners of the removed endpoints |

🔴 = removed (in git history). Model weights under `D:\models` were not touched.

---

## 2026-09-22 — E15: numbers read from the page and corrected (Phase 4)

| Status | File | What it is |
|---|---|---|
| 🟢 | `ocr_service/digit_cnn_data.py` | Synthetic glyph dataset from rendered text lines: Dehkhoda words (own Arabic shaper → Presentation Forms-B), numbers in Persian/Arabic-Indic/Latin digits, Latin/email tokens, scan augmentation, 137 fonts; every connected component labelled digit / separator / **reject** |
| 🟢 | `ocr_service/digit_cnn_train.py` | Trains the glyph CNN (torch venv, GPU), by-font holdout report, exports numpy weights |
| 🟢 | `ocr_service/digit_reader_v2.py` | The v2 page reader: numpy CNN inference + robust line clustering + run building (dot/ring zeros, colons, weak-glyph splitting). Default reader (`glyph2`) |
| 🟢 | `ocr_service/numeric_reconcile.py` | Atom-level alignment of the VLM's numbers with page reads: confirm / **correct in place** / conflict / unverified; **adds** omitted footer numbers to `contact_info` |
| 🟢 | `ocr_eval/tools/bench_digit_reader.py` | Reader-alone benchmark vs GT (atom recall / precision by length) |
| 🟢 | `ocr_eval/tools/replay_reconcile.py` | Replays reconciliation on stored VLM predictions — policy iteration without inference |
| 🟢 | `tests/test_numeric_v2.py` | 13 guards incl. a must-fail fixture |
| 🟡 | `ocr_service/pipeline.py`, `config.py`, `schemas.py`, `benchmark.py` | `glyph2` wiring; `confidence` gains `corrected` and `added`; atom metrics in the benchmark; reader + model hash in provenance |
| 🟡 | `ocr_pipeline/extraction.py`, `ocr_service/backends/llamacpp.py` | **D61**: `LetterExtractor(extra_body=...)` — `cache_prompt` was stamped into provenance but never sent to the engine; now it is, with a test that asserts on the request |
| 🟢 | `ocr_eval/experiments/repeat_stability/run_service.py` | E16: `cache_prompt` on/off across fresh processes on the shipped path (closes D52) |
| 🟡 | `ocr_eval/error_register.md` (D57–D60), `experiments.md` (E15), `gt_real_audit.md` | Records |
| ⚪ | `models/digit_glyphs.npz`, `models/digit_cnn.npz` (+`.json`), `ocr_eval/experiments/digits_0922/*` | Data, model, run outputs |

Untouched: `ocr_service/digit_reader.py` (v1, selectable as `glyph`), `numeric_validator.py`, the legacy `main.py` path, everything under Phase 7 (`api.py`, `tasks.py`, `deploy/`).

---

## Task A — OCR accuracy evaluation harness  (`ocr_eval/`)

| Status | File | What it is | Prompt step |
|---|---|---|---|
| 🟢 | `ocr_eval/generate_dataset.py` | Renders 36 synthetic Persian business letters (18 Persian-only + 18 bilingual) with correct Arabic shaping + bidi; writes exact ground truth | Step 2 |
| 🟢 | `ocr_eval/run_eval.py` | Calls the **real** `/ocr` endpoint on every image, Persian-aware normalization, CER/WER via `jiwer`, per-source + per-language-segment breakdown, RTL/LTR digit-reversal detector | Steps 3–4 |
| 🟢 | `ocr_eval/make_report.py` | Renders `results.json` → `REPORT.md` (worst cases, confusion, conclusion) | Step 5 |
| 🟢 | `ocr_eval/report.html` | Shareable visual report (published as an Artifact) | Step 5 (extra) |
| ⚪ | `ocr_eval/images/*.png` | The 36 test images | Step 2 output |
| ⚪ | `ocr_eval/ground_truth.jsonl` | filename → reference text + `source` tag | Step 2 output |
| ⚪ | `ocr_eval/predictions.jsonl` | Raw model output per image (re-scoreable) | Step 3 output |
| ⚪ | `ocr_eval/results.json` | All computed metrics | Step 3 output |
| ⚪ | `ocr_eval/REPORT.md` | The written report | Step 5 output |

**Result:** Persian-only CER **4.5%**, bilingual **25.6%**, Latin/digit segment **32.9%**, no-output **16.7%**.

Re-run:
```bash
venv312\Scripts\python.exe ocr_eval\run_eval.py          # inference + score (server must be up)
venv312\Scripts\python.exe ocr_eval\make_report.py       # rebuild REPORT.md
```

---

## Task B — Dehkhoda dictionary correction  (`dehkhoda/`)

| Status | File | What it is |
|---|---|---|
| 🟢 | `dehkhoda/build_db.py` | Decompresses all 33 `Dehkhoda-SQL-master/*.sql.gz`, parses the MySQL dumps (handles escaped literals, drops MySQL-only syntax), loads a queryable SQLite DB |
| 🟢 | `dehkhoda/corrector.py` | v1 post-corrector — **superseded** (any single edit; measurably hurt accuracy) |
| 🟢 | `dehkhoda/corrector_v2.py` | **v2, now the default.** Edit model restricted to real OCR errors: visually-confusable substitutions + doubled-letter deletion, no transpositions/insertions |
| 🟢 | `dehkhoda/ablation.py` | Measures every correction rule independently to pick the safe one |
| 🟢 | `dehkhoda/eval_correction.py` | Before/after CER/WER over the same predictions; classifies each change helpful/harmful/neutral |
| ⚪ | `dehkhoda/dehkhoda.db` | SQLite DB, **312,507** headwords (311 MB) |
| ⚪ | `dehkhoda/results_correction.json` | Before/after metrics |
| ⚪ | `dehkhoda/REPORT_correction.md` | The before/after report |

**Result:** v1 **lowered** accuracy (17.04% → 18.08%, 83 harmful changes). Root cause: it modelled *typing* errors (س→ا, ژ→ت, transpositions) rather than *visual* OCR errors. v2 fixes this — **17.04% → 17.01%, 2 helpful, 0 harmful**. Remaining ceiling: Dehkhoda is classical, so ~22% of modern words are out-of-vocabulary.

Re-run:
```bash
venv312\Scripts\python.exe dehkhoda\build_db.py          # rebuild the SQLite DB
venv312\Scripts\python.exe dehkhoda\eval_correction.py   # before/after report
venv312\Scripts\python.exe dehkhoda\ablation.py          # compare all correction rules
```

---

## Phase 1 — trustworthy, field-aware measurement  (`ocr_eval/`)

Separates *character reading* from *field placement*, which the flat-blob scorer conflated.
No model, prompt, decoding, or preprocessing change — the model output (`predictions.jsonl`)
is re-scored, so the number movement is purely the measurement method.

| Status | File | What it is |
|---|---|---|
| 🟢 | `ocr_eval/build_structured_gt.py` | Reconstructs per-field ground truth from the same content pools (no image re-render); validates every segment is a verbatim substring of the frozen GT |
| 🟢 | `ocr_eval/score_fields.py` | Field-aware scorer: named-field vs content-matched CER, assignment loss, placement accuracy, omission, hallucination, orphan capture, digit-ordering buckets, latency percentiles, bootstrap CI over the 18 distinct texts |
| ⚪ | `ocr_eval/ground_truth_fields.jsonl` | Structured GT (fields + fine segments + orphan bucket) |
| ⚪ | `ocr_eval/results_fields.json` | Decomposed metrics |
| ⚪ | `ocr_eval/REPORT_fields.md` | The Phase 1 report |

**Result:** reading CER **9.67%** (not the flat 17.04%); named-field CER 36.68% → the
**27-pt assignment loss is misfiling, not misreading**. contact_info 0/6 correct, receiver
populated on 19/30 (GT always null), 79.5% of schema-orphaned content misfiled. Reading is
good; the gains are structural.

Re-run:
```bash
venv312\Scripts\python.exe ocr_eval\build_structured_gt.py   # (re)build structured GT
venv312\Scripts\python.exe ocr_eval\score_fields.py --compare  # rescore + old-vs-new
```

## The one change to the running service

| Status | File | Change |
|---|---|---|
| 🟡 | `main.py` | Refactored the OCR core into a shared `_run_ocr()` helper and added a **new** endpoint `POST /ocr/corrected`. It returns the raw extraction **and** the Dehkhoda-corrected version side by side, plus which fields changed. Uses corrector **v2** (safe defaults). The dictionary loads lazily on first call. |

- `POST /ocr` — unchanged, raw OCR.
- `POST /ocr/corrected` — raw + dictionary-corrected in one response (opt-in). Uses the safe v2 rule set; kept separate from `/ocr` so raw output stays untouched and the effect of correction is always visible.

Not part of this task (from the earlier "run the app without LM Studio" work): `engine.py`, the embedded-engine rewrite of `main.py`, and the rewritten `README.md`.

---

## Phase 0 (revised brief) — contaminant removal and audit

| Status | File | What changed |
|---|---|---|
| 🟡 | `main.py` | **`POST /ocr/corrected` and the `_get_corrector()` loader removed.** No runtime path now loads a dictionary. `POST /ocr` is byte-for-byte unchanged in behaviour. Pre-removal copy kept at `main.py.pre-dehkhoda-removal.bak`. |
| 🟢 | `dehkhoda/phase0_ablation.py` | The A/B that justified the removal: 4 arms over the frozen predictions, per-split, with the D1 watchlist and a full change log |
| 🟢 | `dehkhoda/REMOVED.md` | Why it went, the ablation table, the exhaustive 49-change hand audit, and the policy for readmitting any lexicon stage |
| 🟢 | `ocr_eval/gt_audit.md` | 17 images read by eye; two generator defects (G1 bidi digit-group reversal, G2 dropped tanween) affecting 18 of 36 images |
| 🟢 | `ocr_eval/gt_corrections.md` | Intentionally empty — the bug is in the generator, not the transcription |
| 🟢 | `ocr_eval/test_phase0_guards.py` | 9 guards: split hash-lock, no-leakage, no lexicon in runtime, no prompt example values, no fabricated names, model pinned |
| 🟡 | `ARCHITECTURE.md` | Dehkhoda removed from the component map; added §8 pinned model identity + what the card claims about Persian, and §9 the generator defects |
| ⚪ | `dehkhoda/phase0_ablation.json`, `phase0_changes.tsv` | Ablation output |

**Result:** the Dehkhoda stage was never in the measured path, so no published number was
ever affected by it; it is removed anyway (2 token fixes for a 311 MB dependency). The
`Roya Ghasemi` string is neither a prompt leak nor a hallucination — it is printed on the
page. A previously unknown dataset defect makes every digit-order metric unquotable.

Re-run:
```bash
venv312\Scripts\python.exe dehkhoda\phase0_ablation.py
venv312\Scripts\python.exe -m pytest ocr_eval\test_phase0_guards.py -q
```
