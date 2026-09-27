# Phase 0 — Hard gates reached

Date: 2026-09-01. Written before any scoring, per Section 4.

---

## GATE 1 — Endpoint parity (F2). ANSWERED; confirmation requested.

**Which endpoint is production?** `POST /ocr`. It is the only OCR endpoint that has
ever been documented as production.

Evidence:

| Source | Says |
|---|---|
| `README.md:36` | "Example request: `POST http://localhost:8000/ocr` with the letter image attached." |
| `README.md:213` | Integration table — URL `http://localhost:8000/ocr` |
| `README.md:241` | The only curl example: `POST /ocr` |
| `README.md:59-61` | Architecture diagram: client → `POST /ocr` |
| `ocr_eval/run_eval.py:41` | `OCR_URL = os.getenv("OCR_URL", "http://127.0.0.1:8000/ocr")` |
| `CHANGES.md:88` | `/ocr/corrected` described as a **new** endpoint added during remediation |
| `CHANGES.md:91` | "kept separate from `/ocr` so raw output stays untouched" |

**What did the two do differently?** `/ocr/corrected` ran the identical `_run_ocr()`
path and then applied the Dehkhoda corrector v2 to each field, returning `raw`,
`corrected` and a `changed` map side by side. `/ocr` never called the corrector.

**Should `/ocr/corrected` still exist?** No, and it no longer does
(`main.py:270` is now the only OCR route; pre-removal copy at
`main.py.pre-dehkhoda-removal.bak`). It was a diagnostic endpoint created during
remediation, never shipped to a consumer, and the ablation
(`dehkhoda/phase0_ablation.py`) showed the v2d rule set it used moved corpus CER by
**−0.03 pp** — inside noise.

**Consequence for published numbers:** none. Every published number came from `/ocr`,
which *is* production. The eval has always scored the shipped pipeline. The
Dehkhoda removal changed no production behaviour and **will not improve any eval
number** — its value was diagnostic only.

**Residual risk:** I cannot see outside this repo. If a deployed client (a Postman
collection, a downstream service, an integration not in this repo) was calling
`/ocr/corrected`, it now gets 404. Nothing in the repo suggests one exists.
→ *Asking the user to confirm before proceeding.*

---

## GATE 2 — There is no ground truth for the 90 real documents. PLAN-CHANGING.

The task brief states: "all 90 have hand-verified ground truth". **They do not.**

- `ocr_eval/dataset_ex/` contains exactly 90 files, **all `.JPG`, nothing else.**
  No subdirectories, no sidecar files, no manifest.
- Repo-wide search for any file referencing a `CamScanner` filename: **zero hits.**
- `ocr_eval/ground_truth.jsonl` and `ground_truth_fields.jsonl` cover the **36
  synthetic images only** (built by `build_structured_gt.py`).
- No transcription file of any format was added to the repo alongside the images.

**Good news on the HAZARD:** the real images were *not* dropped into
`ocr_eval/images/`. That folder still holds exactly the original 36 synthetic PNGs,
so `splits.json` and every stored synthetic baseline remain unambiguous and the
legacy comparison stays valid. The hazard described in §2 did not occur.

### What this blocks

Without reference transcriptions, the following cannot be executed as written:

| Section | Blocked because |
|---|---|
| §2.2 GT validation | Nothing to validate |
| §2.3 (scoring half) | Characterization done (see below); scoring not possible |
| Phase 1.10 "first real baseline" | No reference to score against |
| Phase 2 exit ("≥99% usable-output on **real** dev") | Usable-output rate is measurable without GT; CER is not |
| Phase 3 decoupling decision | Needs field-assignment error on real dev |
| Phase 4 all accuracy exits | All defined against a real baseline |
| Phase 5 model comparison | Needs a real-set metric to compare on |
| FINAL_REPORT headline sentence | Explicitly "measured on the 90 real documents" |

### What is NOT blocked and is being done anyway

Corpus characterization (§2.3), manifest, splits, the Phase 1 harness rebuild,
`normalize.py`, the re-score of stored synthetic predictions (§1.9), config
centralization, error register, ARCHITECTURE.md, and the reliability work in
Phase 2a/2b that keys on *failure modes* rather than *accuracy*.

---

## §2.3 — Real-scan corpus characterization (complete, 90/90 readable)

Raw per-image records: `ocr_eval/reports/real_corpus_characterization.json`.

| Property | Finding |
|---|---|
| Count | 90, all readable |
| Format | JPEG ×90 (CamScanner export) |
| Colour | RGB ×90; mean per-pixel channel spread p50 = 10.6 → genuinely colour (green letterhead), not greyscale |
| Dimensions | p10 2140×3072, **p50 2276×3264**, p90 2424×3264, min 1920×2304, max 3016×3264 |
| Megapixels | min 5.51, p50 7.38, max 8.06 |
| DPI tag | 72×72 on all 90 — **a placeholder, not the true DPI.** At 2276 px across an A4 short edge (210 mm) the effective scan density is **≈275 DPI**. No image falls below a 200 DPI-equivalent floor. |
| EXIF orientation | `1` (normal) on all 90 → **no EXIF-rotation preprocessing needed** |
| Pages | `n_frames` = 1 on all 90 → no multi-page TIFFs |
| Contrast | p5–p95 luminance spread: min 69, p50 171, p90 220. The low tail (<100) is a real degraded-scan subset worth flagging. |
| Sharpness | Laplacian variance min 140, p50 412, p90 922 — a **10× spread**. The bottom decile is materially softer. |
| Background | mean luminance p50 232 (CamScanner has already whitened the page) |
| Ink coverage | dark-pixel fraction p50 0.123 — consistent, no blank pages |
| File size | 519 KB – 1.13 MB |

**Content observed by eye (3 sampled images):** colour letterhead with logo, Persian-Indic
digits throughout (registration numbers, dates, phone, national ID), **handwritten**
date/number fields in the header, an overlapping **blue ink stamp**, a **handwritten
signature**, a footer contact block, and Latin text (email addresses). One image showed
Latin `Emil:` (a typo in the source document itself) — a reminder that GT must record
pixels, not intent.

**Homogeneity warning:** all three sampled letters are from the *same* sender
organisation on the *same* letterhead. If that holds across all 90, this corpus has
n=90 documents but effectively **n=1 template**, exactly the effective-n problem §1.8
raises for the synthetic set. Confidence intervals must be bootstrapped over templates
as well as images, and the coverage audit (Phase 5) must say so plainly.

---

# Phase 0 — completion record

Written after both gates were answered by the user:
GT for the 90 will be supplied separately; `/ocr/corrected` keeps a deprecation stub.

## Artifact inventory (F3) — what existed, what is stale

| artifact | state | verdict |
|---|---|---|
| `ARCHITECTURE.md` | pre-existing | **updated** — endpoint-parity table added, `_dump_raw_output` step added |
| `config.py` | pre-existing | **current** — verified against `main.py`/`engine.py`; centralises model pin, decoding, timeouts, resize and retry flags |
| `ocr_eval/gt_audit.md` | pre-existing | **current but synthetic-only** — documents defects G1/G2 (now registered as D17) |
| `ocr_eval/gt_corrections.md` | pre-existing | **current** — records that no GT was edited, correctly |
| `ocr_eval/test_phase0_guards.py` | pre-existing | **updated** — two guards asserted the wrong invariant (see below) |
| `ocr_eval/splits.json` (v1) | pre-existing | **retained, frozen** — legacy 36-image comparison only |
| `ocr_eval/predictions.jsonl` | pre-existing | **current** — 36 synthetic predictions, re-scored under the new harness |
| `ocr_eval/results.json`, `REPORT.md` | pre-existing | **superseded** by `results_v2.json` / `reports/phase_1.md`; kept for the 1.9 side-by-side |
| `dehkhoda/*` | pre-existing | **current, out of the runtime** |

### Guard tests that were asserting the wrong thing

`test_no_lexicon_stage_in_runtime` and `test_corrected_endpoint_is_gone` both failed
after the deprecation stub was added — correctly detecting the change, but for the
wrong reason. They matched raw source text, so a *comment* explaining why the Dehkhoda
stage was removed tripped the guard against that stage, while a corrector smuggled
into `/ocr` itself would still have passed if it avoided the trigger words.

Both now check the AST with comments, docstrings and string literals stripped, and
assert what actually matters: no corrector import, no `correct_text` call, and the
passthrough must route through the same `_run_ocr` and declare itself deprecated.

## Model pinning and script support

| item | value |
|---|---|
| Model | `coreOCR-7B-050325-preview.Q4_K_S.gguf` (mradermacher GGUF conversion) |
| Size | 4,457,769,440 bytes |
| Projector | `coreOCR-7B-050325-preview.mmproj-f16.gguf`, sha256 `34933952a9ae2f1a…` |
| Quantization | Q4_K_S (4-bit) |
| Runtime | llama.cpp `llama-server` fe2adf0, CUDA 12 / AVX2 build 2.28.2 |
| Decoding | temperature 0.0, max_tokens 2048, top_p 1.0 (not sent), no repetition penalty |
| Image tokens | `IMAGE_MIN_TOKENS=1024` (below this the model enters repetition loops) |

**Declared script support:** the local GGUF conversion ships no model card, and no
document in this repo records a Persian or Arabic-script claim from the upstream
`coreOCR-7B` card. **Persian/Arabic-script support is therefore unverified, not
confirmed.** This matters for two reasons and feeds directly into the Phase 5
model-fitness check:

1. A model not trained on Persian script would explain the D23 digit repetition loop
   and the high Latin/digit CER (D6) better than any prompt defect.
2. The `IMAGE_MIN_TOKENS=1024` workaround already in `config.py` — added because lower
   values caused repetition loops — is a symptom of the same fragility.

## Exit criteria

| criterion | status |
|---|---|
| ARCHITECTURE.md current | **met** |
| `manifest.json` hash-locked | **met** — 126 records, sha256 `30bcbe75be2f7d9b` |
| `splits_v2.json` hash-locked | **met** — sha256 `d26202f31029f6ee`, guard test pins it |
| GT validated and audited | **partial** — synthetic done (D17); the 90 real have no GT (D15) |
| Config centralised | **met** — `config.py`, hashed into every results file |
| Model pinned | **met** — but script support unverified, see above |
| Error register seeded | **met** — 24 rows |
| Endpoint parity answered | **met** — `/ocr` is production; stub retained |

## What got worse

- **`splits_v2.json` was generated twice.** The first attempt stratified by
  `source x language_mode x template`, which made every synthetic stratum a single
  group and emptied the Persian-only test split entirely. Caught before any score was
  computed against it and regenerated one level coarser. The published hash is the
  second version, and it is now final.
- **The synthetic corpus is weaker than believed**: 33 distinct images not 36 (D13),
  6 effective templates not 36 (D14), and 10–20x lower resolution than the real
  documents (D12).
