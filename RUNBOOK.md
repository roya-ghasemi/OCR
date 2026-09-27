# Runbook — Persian/Bilingual OCR evaluation

How to run the evaluation, read the numbers, and act on a failure. Written for
someone picking this up cold.

---

## 1. What the service is, in one paragraph

`POST /ocr` takes a scanned Persian administrative letter and returns five fields:
`sender`, `receiver`, `subject`, `body_text`, `contact_info`. A local llama.cpp
process runs a quantized `coreOCR-7B` vision model. There is no preprocessing, no
dictionary correction, and no post-processing between the model and the response.
`POST /ocr/corrected` is a deprecated passthrough kept only so out-of-repo callers
do not 404; it returns the identical extraction.

---

## 2. Start the service

```bash
venv312/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Startup loads ~4.5 GB of weights and takes 30–120 s. Wait for health:

```bash
curl -s http://127.0.0.1:8000/health
```

`503` means the engine is still loading or has died — check `engine.log`.

---

## 3. Re-run the evaluation

Everything is driven from `ocr_eval/`. **Nothing globs a single image directory**:
synthetic (`ocr_eval/images`, 36) and real (`ocr_eval/dataset_ex`, 90) are separate
roots and are never pooled into one headline number.

```bash
venv312/Scripts/python.exe ocr_eval/tools/build_manifest.py
```
Rebuilds `manifest.json` (126 records: sha256, source, language mode, template,
dimensions, estimated DPI, GT presence). Safe to re-run; the hash should not move
unless images changed.

```bash
venv312/Scripts/python.exe ocr_eval/score_v2.py
```
Re-scores the stored synthetic predictions with the Phase 1 harness and writes
`results_v2.json` + `reports/phase_1.md`. No inference; seconds to run.

```bash
venv312/Scripts/python.exe ocr_eval/run_real.py
```
Runs all 90 real documents through `/ocr`. **~35 minutes** — failures are slower
than successes because a looping request runs to the token cap. Writes
`predictions_real.jsonl` incrementally, so a crash loses nothing.

```bash
venv312/Scripts/python.exe ocr_eval/analyze_real.py
```
GT-free metrics for the real corpus → `reports/phase_2.md`.

```bash
venv312/Scripts/python.exe -m pytest ocr_eval/ test_json_repair.py -q
```
67 tests. Run before and after any change.

### Do not regenerate `splits_v2.json`

`ocr_eval/tools/make_splits_v2.py` is a one-shot. The split is hash-locked in
`splits_v2.sha256` and pinned by `test_splits_v2_hash_is_locked`. Regenerating it
silently invalidates every comparison ever made against it. `splits.json` (v1) is
frozen too, and exists only for the legacy 36-image comparison.

---

## 4. How to read each metric

### Always read the headline as a pair

**Never quote CER-given-output on its own.** It is computed over the documents
that produced output at all, so on a corpus with a 16.67% usable rate it describes
the best sixth of the corpus. Three numbers travel together:

| metric | meaning |
|---|---|
| **usable-output rate** | fraction of documents that returned a parseable response with at least one non-empty field. An all-null 200 counts as a **failure** here. |
| **CER-given-output** | character error over the documents that produced output |
| **effective CER** | CER with no-output documents charged at 100% error — the number a stakeholder actually wants |

### The five error classes are never merged

| class | meaning | why separate |
|---|---|---|
| field-assignment | correct text, wrong field | a prompt/schema problem, not a reading problem |
| omission | GT has content, prediction is null | invisible inside CER |
| hallucination | prediction has content, GT is null | the dangerous one; no reference characters to divide by, so it cannot live inside CER |
| character error | CER over **correctly-assigned pairs only** | otherwise routing errors inflate it |
| duplication | one GT field echoed into 2+ predicted fields | |

### Assignment vs reading

`assignment_vs_reading` decomposes CER by comparing actual routing against the
best possible routing of the *same* predicted strings (an exact bitmask DP).
`assignment_cost_pp` is the character error caused purely by putting correct text
in the wrong field; the rest is genuine misreading. The oracle is a lower bound by
construction and asserts it.

### Raw vs normalized, and the letterform trap

Every headline is published twice: raw and normalized. The normalizer folds Arabic
`ي`/`ك` into Persian `ی`/`ک` because it measures **reading**. That means **CER is
blind to letterform conformance** — so `letterform_conformance` is reported
separately, computed on raw API output. If you only read CER you will never see
that 14% of real responses ship non-canonical Persian.

### Confidence intervals

Every headline carries a bootstrap 95% CI, computed two ways:

- **by_image** — resamples documents
- **by_template** — resamples templates, and is the honest one

The synthetic corpus is 36 images from **6 templates**; the real corpus is 90
documents from what appears to be **one letterhead**. Quoting an image-level CI on
either overstates precision. When the two disagree, use the template-level number.

### `unscoreable` is not `null`

A field the ground truth never covers is `unscoreable`, not a null. Ground truth
covers `receiver` on **0 of 36** synthetic images, so `receiver` has never been
scored — and it is exactly where misassigned content lands. Assignment and
duplication detection deliberately run over unscoreable slots anyway.

---

## 5. Failure modes and what to do

| symptom | meaning | first move |
|---|---|---|
| `422` + `Unterminated string` + `finish_reason=length` | the dominant real-corpus failure | read the dump in `debug_raw/`; check whether the open field is `sender` (D31) or `body_text` (D23) |
| truncation at **column 12** | D31 — the model wrote the whole letter into `sender` and never closed it. Unrecoverable by JSON repair. | needs structured decoding; no amount of budget helps |
| a long run of one Persian-Indic digit | D23 — repetition loop | repetition penalty or structured decoding. **Do not raise `MAX_TOKENS`** — see §6 |
| `422` + `empty_extraction` | the model returned well-formed JSON with every field empty | previously a silent 200 (D4); now an explicit error |
| `200` with fluent but wrong content | D27 — fabrication. Confirmed once, on the one rotated page. | compare against the pixels; do not trust fluency |
| `503` on `/health` | engine not up | `engine.log` |
| a warning naming `/ocr/corrected` | an out-of-repo caller still uses the deprecated route | migrate it, then delete the stub |

### Reading a `debug_raw` dump

Written on every parse failure, outside the application log because these are real
letters. Each holds the **complete** model output, `raw_len`, and the parse error.
The 300-character truncation this replaced is the single reason D23 went
undiagnosed across three sessions. When adding logging, never log document content
to the shared log.

---

## 6. Things that look like fixes and are not

- **Raising `MAX_TOKENS`.** The intuitive reading of "truncated output" is an
  exhausted budget. Every dumped failure is a repetition loop or a schema
  collapse; a larger budget buys a longer loop. Falsify before adopting.
- **Deskew / upscale / contrast.** Failed and successful documents are
  statistically indistinguishable on every image property measured. Failure is
  content-driven.
- **Retrying at `temperature=0.0`.** Deterministic decoding reproduces the same
  loop exactly. A retry must change sampling to be worth its latency.
- **A dictionary corrector.** Measured at −0.03 pp corpus CER. `dehkhoda/REMOVED.md`
  records the conditions for bringing one back; a guard test fails if one returns
  without them.
- **Judging progress by CER alone.** Per-field scoring raised every published CER
  by 14–23 pp on byte-identical predictions. See "1.9" in `reports/phase_1.md`
  before attributing any movement to a fix.

---

## 7. Running an experiment

```bash
venv312/Scripts/python.exe ocr_eval/run_experiment.py \
  --name repeat_penalty_11 --env REPEAT_PENALTY=1.1 --split dev \
  --hypothesis "the D23 loop is decoding degeneracy"
```

Restarts the service with the override, runs the dev split, and appends to
`experiments.md` (append-only). **One variable per run.** Score the test split at
most once per phase, and never to choose between options.

Available knobs (all in `config.py`, all env-overridable): `MAX_TOKENS`,
`TEMPERATURE`, `TOP_P`, `REPEAT_PENALTY`, `FREQUENCY_PENALTY`, `RESPONSE_FORMAT`
(`off` | `json_object` | `json_schema`), `JSON_REPAIR`, `MAX_RETRIES`,
`RETRY_TEMPERATURE`, `RETRY_REPEAT_PENALTY`, `RESIZE_MAX_EDGE`, `IMAGE_MIN_TOKENS`.

All default to current production behaviour, so the baseline cannot drift by
accident.

---

## 8. Adding documents to the evaluation set

1. Put images in `ocr_eval/dataset_ex/` (real) or `ocr_eval/images/` (synthetic).
   **Never mix the two directories.**
2. Add ground truth to a JSONL keyed by filename, with all five fields. Encode
   "this document genuinely has no value here" as an explicit `null`, and leave
   the key **absent** only if you truly cannot say — the two are scored
   differently and conflating them makes omission unmeasurable.
3. `python ocr_eval/tools/build_manifest.py`
4. Do **not** regenerate `splits_v2.json`. Assign new documents to a split by
   extending the file, keeping existing assignments byte-identical, and update
   `splits_v2.sha256` in the same commit.
5. Re-run the evaluation and record the new baseline in `experiments.md`.

Ground truth records **what is on the page**, never what the document meant to
say. A typo in the source document belongs in the ground truth. Corrections go in
`gt_corrections.md` with a reason, and ground truth is never edited to improve a
metric.

---

## 9. The error register

`ocr_eval/error_register.md` is the single source of truth for defects. It is
append-only. A row closes only with a **re-measured number**, never with "fixed".
Two symptoms with one root cause are one row; one symptom with two root causes is
two rows (D23 and D31 are exactly that case).

---

## 10a. Where the model lives

The GGUFs are at `D:\models\models\mradermacher\coreOCR-7B-050325-preview-GGUF\`
(moved off `C:` on 2026-09-05). The llama.cpp **backends** are still under
`~/.lmstudio/extensions/backends` — these are two independent roots and must stay
that way.

Discovery searches `MODEL_SEARCH_ROOTS` in order and takes the first root that
actually *contains* a match, so either location works. To point somewhere else:

    set MODELS_DIR=X:\path\to\models
    set MODEL_GGUF=X:\path\to\model.gguf        # or name the file outright

Check what will be used without starting anything:

    venv312\Scripts\python.exe -c "import engine; print(engine.find_backend()); print(engine.find_model())"

**The backend is pinned to 2.28.2** via `ENGINE_BUILD_PIN`. Newer builds are
installed on this machine; discovery ignores them on purpose, because
`config.ENGINE_BUILD` is stamped into every results file. If you deliberately
move to a newer engine, update `ENGINE_BUILD`, re-record the CI baseline, and
re-check the GBNF quirks in `ocr_pipeline/grammar.py` — they were characterised
against 2.28.2 only.

---

## 10. Known blockers

- **D35 (supersedes D15) — the 90 real documents still have no ground truth.**
  `ocr_eval/dataset_ex` contains 90 `.JPG` files and nothing else; all 90
  manifest records read `"gt_exists": false`. Every real accuracy number is
  blocked. Reliability, latency, letterform conformance, degeneracy and
  identifier checksums are measured and reported; CER is not, and no report
  should imply otherwise.

  To clear it: fill in `ocr_eval/gt_real_template.jsonl` (90 pre-filled rows)
  following `ocr_eval/GT_SPEC.md`, save as `ocr_eval/ground_truth_real.jsonl`,
  then

      venv312\Scripts\python.exe ocr_eval/score_real.py --validate-only
      venv312\Scripts\python.exe ocr_eval/score_real.py

  Partial delivery works — untranscribed rows are excluded and counted. 20
  documents is enough to answer whether the repetition penalty costs accuracy;
  50 gives a usable confidence interval.

- **D41 — never quote the usable-output rate on its own.** It means "at least
  one non-null field", and `sender` is non-null on every usable output. Publish
  it paired with mean fields returned (currently 98.9% and 2.58 of 5).

- **D43, D44 — two fixes are tested but not wired in.** `ocr_pipeline` folds
  Arabic letterforms and bounds field length; `main.py` does not import it.
  Mounting it changes the response contract — a deliberate decision, not a
  silent one.
- **Persian script support is unverified.** The local GGUF conversion ships no
  model card, and nothing on record confirms the upstream model claims Persian or
  Arabic script. This is a live hypothesis for the failure rate, and it is what
  the Phase 5 model-fitness comparison exists to settle.
