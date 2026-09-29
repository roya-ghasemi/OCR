# Experiment log (append-only)

One entry per run. Never edit a past entry; add a correcting entry instead.
Every entry records the config that produced it so a number can be traced to a build.

Provenance constant for all entries below unless stated otherwise:

```
engine   llama.cpp llama-server version 1 (fe2adf0), win-x86_64-nvidia-cuda12-avx2-2.28.2
model    coreOCR-7B-050325-preview.Q4_K_S.gguf   (4,457,769,440 bytes)
mmproj   coreOCR-7B-050325-preview.mmproj-f16.gguf
         sha256 34933952a9ae2f1ac7af7a908189b3fc22106d3999fd3fae43b520ac4d308a78
decode   temperature=0.0  max_tokens=2048  top_p=default  timeout=600s
engine   n_gpu_layers=99  context_size=16384  image_min_tokens=1024
pipeline no image preprocessing, no retry
dataset  36 images / 18 distinct texts / 2 layouts, seed 20260829
```

---

## E000 — Baseline (pre-Phase-1 harness, flat-blob scoring)

**Date:** 2026-08-27 · **Split:** whole set (no split existed yet) · **Status:** superseded by Phase 1 re-scoring

Flat-blob metrics — all 5 fields concatenated and compared to a flat GT. These
conflate character misreading with field misassignment and are **not** to be quoted
going forward; retained only as the "before" side of the Phase 1 comparison.

| Metric | Value |
|---|--:|
| Usable-output rate | 83.33% (30/36) |
| HTTP 422 | 3 (all bilingual, all the same letter content) |
| Empty 200 (all-null) | 3 (all Persian-only) |
| CER given output — ALL | 17.04% |
| CER given output — bilingual | 25.59% |
| CER given output — Persian-only | 4.49% |
| WER given output — ALL | 28.17% |
| Bilingual Persian-script segment CER | 16.17% |
| Bilingual Latin/digit segment CER | 32.85% |
| Mean latency | 3.15 s/img (failures INCLUDED; 422s took ~17.7 s each) |

Note: the WER figure quoted in the remediation brief (22.38%) does not match this
run's 28.17%. Unresolved; the Phase 1 harness supersedes both.

## E001 — Dehkhoda corrector v1 (any single edit)

**Date:** 2026-08-27 · **Split:** whole set · **Verdict:** REGRESSION, reverted as default

CER 17.04% → 18.08% (+1.04). 99 tokens changed: 6 helpful, 83 harmful, 10 neutral.
Root cause: modelled *typing* errors (arbitrary substitution across all 35 letters,
plus transposition) rather than *visual* OCR errors. Harmful substitutions included
س→ا ×14, ژ→ت ×11, د→م ×7, ح→ن ×7 — none of which an OCR produces.

## E002 — Dehkhoda corrector v2 rule ablation

**Date:** 2026-08-27 · **Split:** whole set · **Verdict:** v2d adopted as default

| Config | CER | Δ | helpful | harmful |
|---|--:|--:|--:|--:|
| no correction | 17.04% | — | — | — |
| v1 any single edit | 18.08% | +1.04% | 6 | 83 |
| v2a confusable-sub + doubled-del | 17.28% | +0.24% | 3 | 23 |
| v2b … + unambiguous only | 17.13% | +0.09% | 3 | 11 |
| v2c confusable-sub only | 17.15% | +0.11% | 1 | 11 |
| **v2d doubled-letter deletion only** | **17.01%** | **−0.03%** | **2** | **0** |

Dropping the 32.8% of Dehkhoda entries that are pure cross-references was also
tested: no change (17.01%). Ceiling is Dehkhoda's classical vocabulary — ~22% of
correct modern words are out-of-vocabulary.

## E003 — Phase 0 audit (no model change)

**Date:** 2026-08-29 · **Split:** created · **Verdict:** measurement not yet trustworthy

No inference run. Findings that invalidate or reframe E000:

- Schema has **no `date` field**; 18/36 GT letters contain a date line.
- All 3 HTTP 422s are the **same letter content** (n=1 defect, not 3), all containing
  the 21-digit IBAN `IR820170000000123456789`.
- All 3 empty-200s are on **clean, legible** images — model abstention, not input quality.
- **No hallucination**: 0 cases of a name in prediction but absent from GT.
- Dataset is **2 layouts / 18 distinct texts**, not the ~5–6 templates assumed.
- 3 GT letters contain a **duplicated `تاریخ:` label** (generator defect).
- Determinism confirmed: all 36 images + GT regenerate byte-identically.
- Frozen split written to `splits.json`: dev 24 / test 12, grouped by distinct text
  so no text spans both splits.

## E004 — Phase 1 field-aware re-scoring (no model change)

**Date:** 2026-09-01 · **Split:** whole set (measurement change, not tuning) · **Verdict:** measurement now trustworthy; headline corrected

No new inference. Re-scored the same `predictions.jsonl` with per-field ground truth
(`ground_truth_fields.jsonl`) via `score_fields.py`. Decomposes the flat CER into reading
vs placement.

Provenance correction to E000: dataset seed is **20260827** (from generate_dataset.py);
20260829 is the *split* seed. Earlier entries said "seed 20260829" for the dataset — wrong.

| Metric | Value | 95% CI over 18 texts |
|---|--:|---|
| Usable output | 30/36 = 83.3% | — |
| Character reading CER (placement-free) | **9.67%** | [5.9, 13.7] |
| Named-field CER (placement-strict) | **36.68%** | [21.5, 55.1] |
| Assignment loss (strict − free) | **27.01 pts** | — |
| old flat-blob CER (E000, for reference) | 17.04% | — |

Per-field named CER: sender 54.82, subject 49.89, body_text 28.50, contact_info **100.0**
(n=6), receiver populated on **19/30** despite GT always-null.
Omission 4.6% (10/216 segments). Hallucination 0/101 (invented receivers are misfiled page
text, not fabrication). Orphan capture 79.5% (31/39 schema-homeless segments captured and
misfiled). Digit ordering: 40 exact / 1 reversed / 3 permuted / 2 partial — reversal
hypothesis NOT supported. Latency ok-only p50 1.86 / p90 2.29 / p99 2.56 / mean 1.93s.

The 22.38% WER in the remediation brief is traced: it is `wer_before` in
`dehkhoda/results_correction.json`, produced by `eval_correction.py`'s `corpus_wer` over a
different subset than run_eval.py's 28.17%. Both are superseded — Phase 1 does not quote a
single flat WER; it reports the decomposition above.

Conclusion feeding Phase 2/3 (not acted on): reading is good; the gains are in output
structure — receiver-null, contact routing, subject-label stripping, sender disambiguation,
and a home for schema-orphaned content.

## E005 — Phase 0 (revised brief): Dehkhoda ablation, name-leak verdict, GT audit

**Date:** 2026-09-01 · **Split:** whole set for the ablation (post-processing A/B, not
tuning); dev/test reported separately · **Verdict:** Dehkhoda stage REMOVED; D1 and D2
both refuted; a new dataset defect found that invalidates all digit-order metrics

No inference run. The corrector is pure post-processing over the model's JSON, so the
A/B applies / does not apply it to the same frozen `predictions.jsonl` — stronger than
re-running inference twice because it removes inference variance entirely.
Script: `dehkhoda/phase0_ablation.py`. Raw: `dehkhoda/phase0_ablation.json`,
`dehkhoda/phase0_changes.tsv`.

### Dehkhoda ablation

| Arm | changed | CER all | Δ | CER bil | CER fa | WER | CER dev | CER test |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| off (what `/ocr` ships) | 0 | 17.04% | — | 25.59% | 4.49% | 22.38% | 17.15% | 16.83% |
| **v2d** (what `/ocr/corrected` shipped) | **2** | 17.01% | **−0.03** | 25.54% | 4.49% | 22.23% | 17.11% | 16.83% |
| v2b (+confusable subs, unambiguous) | 17 | 17.13% | +0.09 | 25.54% | 4.77% | 22.76% | 17.24% | 16.90% |
| v2a (+confusable subs, ambiguous OK) | 30 | 17.28% | +0.24 | 25.67% | 4.96% | 23.66% | 17.44% | 16.98% |

Usable-output rate 30/36 in every arm — a post-processor cannot change it.

Hand audit of **all 49 changes** (8 distinct kinds, so labelled exhaustively rather than
sampling 30): **2 improved, 0 unchanged-meaning, 47 damaged.** The shipped v2d arm is
2 improved / 0 damaged, but those 2 are the only thing it ever did.

**D1 REFUTED, three ways.** (a) The stage was never in the measured path — `run_eval.py`
posts to `/ocr`; the corrector only ran on `/ocr/corrected`. (b) Every named substitution
is present in the raw `/ocr` output, which no dictionary ever touched: `احتمالا` ×3,
`اعلان` ×3, `فرمانیه` ×4, `پیشایپیش` ×3, `پیشابیش` ×2, `پیشایش` ×1, `پیشایی` ×5, all with
0 occurrences in GT. (c) The shipped arm cannot reach any of them — the watchlist counts
are byte-identical with the stage on and off.

**Removed anyway**, on different grounds than the brief assumed: 2 token fixes for a
311 MB dependency and a 154,396-word classical lexicon, with every more-aggressive setting
damaging output (`پروژه → پروره` ×11, `آید → آبد` ×12, `رضاى → رضاب` on a person's name)
and one boolean flag away from being enabled. `main.py` no longer loads it; guard test in
`ocr_eval/test_phase0_guards.py`. Not deleted from disk: **this project is not a git repo**,
so "keep it in git history" is not available and deleting would destroy the evidence.

### D2 — `Roya Ghasemi` — REFUTED, it is neither a leak nor a hallucination

- **Not in the prompt.** `_SYSTEM_PROMPT` contains no few-shot examples and no names —
  only a JSON skeleton with `<angle-bracket>` Persian field descriptions.
- **It is in the generator, as intended content.** `generate_dataset.py:92,97` — a
  signature pool entry: `"مدیرعامل CEO\nRoya Ghasemi - رویا قاسمی فر"`.
- **It is in the pixels.** `synthetic_fa_004.png` opened and read: `رویا قاسمی فر` is
  printed under `مدیر عامل`.
- **It is in the ground truth of exactly the images that predict it.** Cross-checked all
  36: GT contains a name token in 15, predictions contain one in 11, and
  **pred-only cases = 0.** The model never emits a name absent from that image's GT.

The brief's "different templates" premise is also wrong — the dataset has 2 layouts, not
5–6 (E003). Locked by `test_no_prediction_contains_a_value_absent_from_its_ground_truth`.

### GT audit (0.8) — a new, larger defect than the ones being hunted

17 images read by eye (the full worst-15 plus 2 controls). Two systematic **generator**
defects, both making the system look better than it is:

- **G1** — `python-bidi` reverses hyphen-separated digit-group order. `2026-08-27` is
  *drawn* as `27-08-2026`. 18 lines / 12 images. Latin-containing runs (`INV-`, `EMP-`,
  `TRK-`, `version`) and slash dates are drawn correctly.
- **G2** — `arabic_reshaper` drops tanween `U+064B`; `احتراماً` is drawn as `احتراما`.
  6 chars / 6 images. So the fa renders that emit `احتراما` are **correct** and were
  scored wrong.

18 of 36 images carry a defect. Re-scored against pixel-faithful GT:

| | CER all | CER bil | bil FA-seg | bil Latin/digit-seg |
|---|--:|--:|--:|--:|
| GT as recorded (published) | 17.04% | 25.59% | 16.17% | 32.85% |
| GT as actually printed | 17.79% | 26.85% | 16.17% | **36.22%** |

**The model reverses numeric date groups in 7 of 7 cases where it emits a date**, and
scores correct on 5 of them only because G1 made the same mistake first. The dataset
therefore **cannot measure digit ordering**; no Latin/digit or reversal metric from it is
quotable until the generator is fixed (Phase 5). GT was NOT edited (operating rule 4) —
the bug is in the generator. `ocr_eval/gt_corrections.md` is intentionally empty.

Also confirmed by eye: **all 3 empty-200 images are clean and fully legible**, so Phase 2b
should not chase image quality.

### D10 — model fitness, now confirmed as an open risk

`prithivMLmods/coreOCR-7B-050325-preview` (base `Qwen/Qwen2-VL-7B-Instruct`), served as
`mradermacher/...-GGUF` Q4_K_S. Its card names **no** Persian, Farsi or Arabic script —
only "Multilingual OCR workflows" and "performance on low-resource or rare scripts may
vary". The base model's card claims Arabic but not Persian. Fine-tuned on
`olmOCR-mix-0225` + two Openpdf/Opendoc sets. Nastaliq is untested — the dataset renders
only Tahoma and Arial. Recommend considering bringing the Phase 5.6 model comparison
forward.

### Exit criteria not met

**0.9 reality-check set does not exist.** It requires 20–30 real scanned/photographed
Persian documents with hand-verified GT; the brief left `{{source}}` unfilled and no such
documents are in the repo. This cannot be synthesised — a synthetic stand-in would defeat
the entire purpose of the set. **Blocked on the user.**

---

## Phase 2a — reliability on the real corpus

### real_baseline — 2026-09-01

**Hypothesis:** none; this is the reference point.

**Changed:** `nothing (baseline)` · split `all` · n=90

| metric | value |
|---|---|
| usable-output rate | **16.67%** [10.00, 24.44] (15/90) |
| failures | `{'http_422': 75}` — 100% of failures |
| degenerate character runs (>=20) | 30 (33.3%), every repeated char a Persian-Indic digit |
| latency p50 / p95 | 22.2s / 22.3s (failures) · 6.35s / 8.3s (successes) |
| contact_info filled | 0 / 15 usable |

Raw: `ocr_eval/predictions_real.jsonl`, `ocr_eval/results_real_gtfree.json`

### max_tokens_4096 — 2026-09-02 (PARTIAL, n=43 of 54)

**Hypothesis (from the remediation brief):** the 422s are token-budget exhaustion,
so doubling `MAX_TOKENS` should raise the usable-output rate. Listed there as the
leading hypothesis, to be tested first.

**Changed:** `MAX_TOKENS: 2048 → 4096` (one variable) · split `dev` · n=43 of 54

| metric | baseline (2048) | this run (4096) |
|---|---|---|
| usable-output rate, dev | **14.81%** (8/54) | **16.28%** (7/43) |
| failures | 46 | 36 |
| model output length | capped at ~2048 tokens | median 2480 chars, **max 7278** |

**Verdict: hypothesis falsified.** The override demonstrably took effect — outputs
grew well past the old cap, to 7278 characters. The usable-output rate did not
move: 16.28% vs 14.81% is comfortably inside the baseline's 95% CI of
[5.56, 24.07]. Every additional token went into a longer repetition loop.

This matches what the dumps already showed: 30 of 30 analysed failures were
repetition loops or schema collapse, and **none** were output that was merely too
long. `finish_reason=length` marks where the loop was cut off, not why it failed.

**Do not raise `MAX_TOKENS` to address D3/D23.** It costs latency — failures ran
~44s instead of ~22s, because a bigger budget means a longer loop before the cut —
and buys nothing.

**Caveat, stated plainly:** the run covered 43 of 54 dev images before the client
hung following an overnight machine suspend (llama-server logged `cancel task`).
It was not restarted because the result is unambiguous at n=43 and a rerun costs
~40 minutes for a number that cannot change the conclusion. The partial record is
in `experiments/max_tokens_4096.partial.json`. **This is the one experiment in
this file that did not complete its planned sample.**

### repeat_penalty_12 — 2026-09-02T12:27:53+00:00

**Hypothesis:** D23/D31 are decoding degeneracy. A direct engine sweep on one failing image showed repeat_penalty 1.05 and 1.1 do nothing but 1.2 terminates cleanly (finish=stop, maxrun=2, parses). Test whether that generalises across the real dev split.

**Changed:** `{'REPEAT_PENALTY': '1.2'}` · split `dev` · n=54

| metric | value |
|---|---|
| usable-output rate | **96.30%** (52/54) |
| failures | {'http_422': 2} |
| degenerate character runs (>=20) | 1 (1.85%) |
| latency p50 / p95 | 4.441s / 14.037s |

Raw: `ocr_eval/experiments/repeat_penalty_12.json`

### response_format_json_schema — 2026-09-02 (ABANDONED)

**Hypothesis:** constraining sampling to a grammar derived from the 5-field schema
makes both failure modes impossible to express.

**Result: the lever does not exist on this engine.** llama.cpp ignores
`response_format` on **multimodal** requests. Verified by calling the engine
directly, bypassing the OpenAI SDK:

| request | schema sent? | output |
|---|---|---|
| text-only | yes | `{"a": "yes"}` — correctly constrained |
| image | no | len 1575, `finish=length`, 804-char digit loop |
| image | yes | len 1575, `finish=length`, 804-char digit loop — **byte-identical** |

`json_object` is ignored on both paths. Registered as **D32**. The `RESPONSE_FORMAT`
knob is kept in `config.py`, defaulting to `off`, for an engine that supports it.

---

### repeat_penalty_12 — 2026-09-02

**Hypothesis:** D23/D31 are decoding degeneracy. A direct engine sweep on one
failing image showed `repeat_penalty` 1.05 and 1.1 change nothing but **1.2**
terminates cleanly. Test whether that generalises across the real dev split.

**Changed:** `REPEAT_PENALTY: 1.0 → 1.2` (one variable) · split `dev` · n=54

| metric | baseline | repeat_penalty=1.2 | change |
|---|---|---|---|
| **usable-output rate** | **14.81%** [5.56, 25.93] | **96.30%** [90.74, 100.00] | **+81.5 pp** |
| failures | 46 (all `http_422`) | 2 (both `http_422`) | −44 |
| degenerate runs | 33.3% of corpus | **1.85%** | −31.5 pp |
| latency p50 | 22.20s | **4.44s** | **5× faster** |
| latency p95 | 22.30s | 14.04s | −8.3s |

The CIs do not overlap, by a wide margin.

**Why latency collapsed too:** a looping request ran to the token cap every time.
Removing the loop removes the wasted decoding, so the reliability fix is also the
largest latency win available — as `reports/phase_2.md` predicted.

**Threshold, from the direct sweep on one image:**

| setting | finish_reason | longest run | parses |
|---|---|---|---|
| baseline | length | 804 | no |
| `repeat_penalty=1.05` | length | 715 | no |
| `repeat_penalty=1.1` | length | 797 | no |
| `presence_penalty=0.5` | length | 738 | no |
| **`repeat_penalty=1.2`** | **stop** | **2** | **yes** |
| **`frequency_penalty=0.5`** | **stop** | **2** | **yes** |

The effect is a threshold, not a gradient: 1.1 is indistinguishable from no penalty.

**Not yet met:** the Phase 2 exit criterion is ≥99% on dev. 96.30% falls short by
two documents (`_18.JPG`, `_40.JPG`), and neither shows a degenerate run
(`maxrun=2`) — so those two are a different failure, not leftover D23.

**Unverified risk, stated plainly:** a repetition penalty suppresses legitimately
repeated tokens. Persian administrative letters repeat formulae (`احتراماً`,
`خواهشمند است`) and digits, so this may trade reliability for accuracy. **That
trade cannot be measured on the real corpus without ground truth (D15).** The
number above is a reliability result only, and must not be read as an accuracy
improvement.

### rp12_maxtok4096 — 2026-09-02T12:35:35+00:00

**Hypothesis:** One variable ON TOP OF the new repeat_penalty=1.2 baseline: MAX_TOKENS 2048 -> 4096. The 2 residual failures are D31 schema collapse on long documents that hit the cap legitimately, with no degenerate run (maxrun=2). max_tokens was the wrong fix for the loop; it may be the right fix for what remains once the loop is gone.

**Changed:** `{'REPEAT_PENALTY': '1.2', 'MAX_TOKENS': '4096'}` · split `dev` · n=54

| metric | value |
|---|---|
| usable-output rate | **96.30%** (52/54) |
| failures | {'http_422': 2} |
| degenerate character runs (>=20) | 1 (1.85%) |
| latency p50 / p95 | 4.416s / 13.928s |

Raw: `ocr_eval/experiments/rp12_maxtok4096.json`

### rp12_maxtok4096 — 2026-09-02

**Hypothesis:** one variable on top of the new `repeat_penalty=1.2` baseline —
`MAX_TOKENS` 2048 → 4096. The two residual failures are D31 schema collapse on
long documents that hit the cap legitimately (`maxrun=2`, no degenerate run).
`max_tokens` was the wrong fix for the loop; it might be the right fix for what
remains now the loop is gone.

**Changed:** `MAX_TOKENS: 2048 → 4096`, on top of `REPEAT_PENALTY=1.2` · dev · n=54

| metric | rp=1.2 (baseline for this run) | rp=1.2 + 4096 tokens |
|---|---|---|
| usable-output rate | 96.30% | **96.30%** — no change |
| failures | 2 (`_18`, `_40`) | 2 (`_18`, `_40`) — the same two |
| degenerate runs | 1.85% | 1.85% |
| latency p50 | 4.441s | 4.416s |

**Verdict: hypothesis falsified, second time for `MAX_TOKENS`.** Doubling the
budget does not help these documents even with degeneracy removed. The same two
documents fail, identically. `MAX_TOKENS` is not the lever for D31 either — the
model collapses the whole letter into `sender` and would keep writing regardless
of budget.

`MAX_TOKENS` stays at 2048. Raising it costs latency on the failure path and buys
nothing in either regime.

### rp12_jsonrepair — 2026-09-02T12:41:33+00:00

**Hypothesis:** One variable on top of the repeat_penalty=1.2 baseline: enable JSON repair. Measured offline at 67.6% recovery on the original failure dumps. The 2 residual failures are D31 collapse into sender; repair should recover at least the one whose sender field completed.

**Changed:** `{'REPEAT_PENALTY': '1.2', 'JSON_REPAIR': '1'}` · split `dev` · n=54

| metric | value |
|---|---|
| usable-output rate | **98.15%** (53/54) |
| failures | {'http_422': 1} |
| degenerate character runs (>=20) | 1 (1.85%) |
| latency p50 / p95 | 4.415s / 13.899s |

Raw: `ocr_eval/experiments/rp12_jsonrepair.json`

### rp12_jsonrepair — 2026-09-02

**Hypothesis:** one variable on top of `repeat_penalty=1.2` — enable JSON repair.
Measured offline at 67.6% recovery on the original failure dumps.

**Changed:** `JSON_REPAIR: off → on`, on top of `REPEAT_PENALTY=1.2` · dev · n=54

| metric | rp=1.2 | rp=1.2 + repair |
|---|---|---|
| **usable-output rate** | 96.30% | **98.15%** (53/54) |
| failures | 2 (`_18`, `_40`) | **1** (`_40`) |
| latency p50 | 4.441s | 4.415s — repair is free |

Recovered exactly the document predicted: `_18`, whose `sender` field completed
before truncation. `_40` never completed a single field, so there is nothing to
salvage — repair correctly returns nothing rather than inventing a value.

**Caveat on what "usable" means here.** The recovered document is a *partial*
extraction — `sender` only, with the rest dropped. It counts as usable output but
is not a complete read, and the API must mark it as such rather than passing it
off as a clean extraction. That is Phase 7 work (the `partial-extraction` error
class).

**Still short of the exit criterion.** 98.15% vs ≥99%. On n=54 that criterion is
effectively 54/54, so a single stubborn document fails the phase.

### rp12_repair_retry — 2026-09-02T12:47:45+00:00

**Hypothesis:** One variable on top of rp=1.2 + repair: allow one bounded retry. The retry deliberately changes sampling (temperature 0.2, repeat_penalty 1.15) because a deterministic retry would reproduce the same failure exactly. Target: recover _40, the last failing document.

**Changed:** `{'REPEAT_PENALTY': '1.2', 'JSON_REPAIR': '1', 'MAX_RETRIES': '1'}` · split `dev` · n=54

| metric | value |
|---|---|
| usable-output rate | **100.00%** (54/54) |
| failures | none |
| degenerate character runs (>=20) | 1 (1.85%) |
| latency p50 / p95 | 4.422s / 13.875s |

Raw: `ocr_eval/experiments/rp12_repair_retry.json`

### rp12_repair_retry — 2026-09-02  ← **winning configuration**

**Hypothesis:** one variable on top of rp=1.2 + repair — allow one bounded retry.
The retry deliberately changes sampling (temperature 0.2, `repeat_penalty` 1.15);
a deterministic retry at temperature 0.0 would reproduce the same failure exactly.

**Changed:** `MAX_RETRIES: 0 → 1` · dev · n=54

| metric | rp=1.2 + repair | + one retry |
|---|---|---|
| **usable-output rate** | 98.15% | **100.00%** (54/54) |
| failures | 1 (`_40`) | **0** |
| latency p50 | 4.415s | 4.422s — unchanged |
| latency p95 | 13.899s | 13.875s |

The retry costs nothing on the happy path: it only fires on output that failed to
parse, which is now rare. `_40` recovered on its second attempt, confirming the
failure is sampling-dependent rather than a hard property of the document.

### Cumulative effect on real dev (n=54)

| configuration | usable-output | latency p50 |
|---|---|---|
| baseline | 14.81% [5.56, 25.93] | 22.20s |
| `MAX_TOKENS=4096` | 16.28% (n=43, partial) | ~44s |
| `RESPONSE_FORMAT=json_schema` | no effect — engine ignores it on image requests (D32) |  |
| `REPEAT_PENALTY=1.2` | 96.30% [90.74, 100.00] | 4.44s |
| + `JSON_REPAIR=1` | 98.15% | 4.42s |
| + `MAX_RETRIES=1` | **100.00%** | **4.42s** |

**+85.2 pp usable-output and a 5× latency reduction, from three changes, none of
which was the fix the brief predicted.**

### TEST_SPLIT_rp12_repair_retry — 2026-09-02T12:52:46+00:00

**Hypothesis:** Phase 2 exit test. The winning dev configuration, scored ONCE on the held-out test split. Not used to choose between options.

**Changed:** `{'REPEAT_PENALTY': '1.2', 'JSON_REPAIR': '1', 'MAX_RETRIES': '1'}` · split `test` · n=36

| metric | value |
|---|---|
| usable-output rate | **97.22%** (35/36) |
| failures | {'http_422': 1} |
| degenerate character runs (>=20) | 0 (0.0%) |
| latency p50 / p95 | 4.332s / 9.05s |

Raw: `ocr_eval/experiments/TEST_SPLIT_rp12_repair_retry.json`

### TEST_SPLIT_rp12_repair_retry — 2026-09-02  ← **Phase 2 exit test, scored once**

The winning dev configuration on the held-out test split. Not used to choose
between options; run exactly once.

**Config:** `REPEAT_PENALTY=1.2` · `JSON_REPAIR=1` · `MAX_RETRIES=1` · test · n=36

| metric | dev (n=54) | **test (n=36)** |
|---|---|---|
| usable-output rate | 100.00% | **97.22%** [91.67, 100.00] |
| failures | 0 | **1** (`_48.JPG`, `http_422`) |
| degenerate runs | 1.85% | **0.00%** |
| latency p50 | 4.422s | 4.332s |
| latency p95 | 13.875s | 9.050s |

The dev-to-test gap (100% → 97.22%) is the expected optimism of a configuration
chosen on dev. The test number is the one to quote.

**Phase 2 exit criteria**

| criterion | target | measured | status |
|---|---|---|---|
| usable-output on real dev | ≥99% | **100.00%** | **met** |
| failures on test | ≤1 | **1** | **met** |
| no code path returns a silently empty success | 0 | all-null 200s now rejected at the endpoint | **met** |

**Phase 2 passes.** These three settings are now the defaults in `config.py`.

---

## E9 — decoding strategy: shipped inline pipeline vs `ocr_pipeline`

**Date** 2026-09-03 · **Corpus** all 90 real administrative letters · **GT-free**

**Variable** the decoding strategy, changed as one unit. Arm A is what `/ocr`
serves: free-form JSON prompt, `repeat_penalty=1.2`, JSON repair, one retry. Arm
B is `ocr_pipeline.LetterExtractor` defaults: GBNF grammar with bounded field
caps, DRY + `repeat_penalty=1.1`, structural guardrails. Grammar and sampler move
together on purpose — D39 established the correct sampler is *inverted* by
whether the grammar is on, so they are not independently meaningful.

**Held constant** engine build, model, mmproj, images, session, scoring code.

| metric | A | B | Δ |
|---|---:|---:|---|
| shipped, of 90 | 89 | 88 | −1 |
| **mean fields (of 5)** | **2.58** | **3.86** | **+1.28** |
| `receiver` | 58.4% | 97.7% | +39.3 pp |
| `subject` | 21.3% | 98.9% | +77.6 pp |
| `body_text` | 20.2% | 97.7% | +77.5 pp |
| `contact_info` | 58.4% | 4.5% | **−53.9 pp** |
| `sender` | 100.0% | 87.5% | −12.5 pp |
| schema collapse | 20.2% | 0.0% | −20.2 pp |
| Arabic letterforms | 34.8% | 0.0% | −34.8 pp |
| national-ID checksum pass | 18.5% | 19.6% | ~0 |
| wall clock | 582s | 832s | +43% |

**Conclusion.** Arm B extracts the letter; arm A mostly returns a `sender` stub.
Schema collapse and Arabic letterforms go to zero *by construction* — bounded
grammar caps and output normalization, not tuned thresholds.

**Negative result, and it is the important one.** Digit accuracy did not move:
national-ID checksum pass 18.5% → 19.6%, unchanged within noise. **The grammar
guarantees shape, not correctness.** Every claim made for structured decoding is
bounded by this. Registered as D46.

**Regression that blocks cutover.** `contact_info` 58.4% → 4.5% (D45). Arm A's
high rate there is partly a symptom — footer text dumped into `contact_info`
because the body was skipped — but a 4.5% fill rate on the field holding phone,
email and IBAN is not shippable. Hypothesis: `contact_info` is last in
`FIELD_ORDER` and the model reaches it with no budget after a correctly-filled
`body_text`. Predicts the loss concentrates on long documents. Not tested.

**Decision.** No cutover. `main.py` continues to serve `/ocr`. Next experiment:
raise the `contact_info` cap or move it earlier in the grammar, re-run E9.

**Artefacts** `predictions_real_pipeline.jsonl`, `results_ab_gtfree.json`,
`reports/phase_6_decoding_ab.md`, runner `ocr_eval/run_pipeline_ab.py`.

## E10 — orientation: the sideways corpus image vs a true 90° rotate

**Date** 2026-09-11 · **Corpus** 1 real document (`_11`) · **GT** human (`ground_truth_real_fixed_v2.jsonl`)

**Provenance** engine `llama.cpp llama-server version 1 (fe2adf0), cuda12-avx2-2.28.2`;
model `coreOCR-7B-050325-preview.Q4_K_S.gguf`; mmproj `34933952…`; arm A
`temperature=0.0 repeat_penalty=1.2 max_tokens=2048 image_min_tokens=1024 resize_max_edge=0
max_retries=1 json_repair=1 response_format=off`; arm B `dry+repeat_1.1`.

**Variable** the pixels. `_11` is the only landscape image in the corpus
(3016×2304; 89 of 90 are portrait) and the thumbnail shows the page lying on its
side. D51 established that the EXIF flag added on 2026-09-10 never reaches the
model. Here the rotation is applied to the pixels (`ImageOps.exif_transpose`,
2304×3016, no EXIF) and both arms are run on the result. The rotated file lives in
`experiments/orientation/`, not in the corpus.

| field | A sideways | A upright | B sideways | B upright | (similarity to GT, raw strings) |
|---|---:|---:|---:|---:|---|
| `sender` | 0.703 | 0.629 | 0.703 | **1.000** | |
| `receiver` | — | 0.786 | 0.351 | **1.000** | |
| `subject` | — | 0.300 | 0.273 | 0.126 | GT subject is annotator-supplied, not on the page |
| `body_text` | 0.022 | — | 0.030 | **0.333** | B upright: 468 chars, `finish=stop`, not capped |
| `contact_info` | — | — | — | — | absent in every arm; B upright folded it into `body_text` |

**Conclusion.** Upright, arm B reads `sender` and `receiver` exactly and produces a
body that is the letter (D45's mechanism visible: the footer is absorbed into
`body_text`). Sideways, the stable short fields were stable *hallucinations* —
`receiver` `جناب آقای محسن شاهی‌پور` and a `subject` about `خودروهای تولید شده`
appear nowhere on the page. Orientation matters, but for exactly one document.

**The finding that outranks orientation.** In the upright body every number on the
page is replaced by a counting sequence: `۳۲۱/۰۰۰/۰۰۰` → `۱۲,۳۴۵,۶۷۸/۰۰۰,۰۰۰`,
`۴۱۱۴۱۳۶۷۴۶۴۸` → `۱۲۳۵۶۷۹۰۱۰۲۳۴`, IBAN `IR۶۵۰۱۰۰۰۰…` → `۱۲۴۵۶۸۹۰۲۳۱۲۳۴`; the
`subject` carries `۱۲۳,۴۵۶,۷۸۹` and `۱۲۳۴۵۶۷۸۹۰۱۲۳۴`. The words are read; the digits
are *generated*. This is D42/D46 seen against ground truth for the first time, and
it says the digit problem is not tokenisation noise on a misread — the model emits
a placeholder pattern where a number should be.

**Artefacts** `experiments/orientation/results_E10.json`, `rot90.provenance.json`, `run.py`.

## E11 — repeat stability across processes (sizes D52)

**Date** 2026-09-11 · **Corpus** 10 real dev documents (`ci_gate.dev_sample(10)`) · **GT-free**

**Provenance** as E10, arm B only (`dry+repeat_1.1`, `temperature=0.0`).

**Variable** the process. Same bytes, same code, same engine; run once per fresh
Python process (`run1`, `run2`), compared with each other and with the 2026-09-03
corpus run as a third process.

| field | run1~run2 | run1~corpus | run2~corpus | docs < 0.9 |
|---|---:|---:|---:|---:|
| `sender` | 1.000 | 1.000 | 1.000 | 0/10 |
| `receiver` | 0.940 | 1.000 | 0.940 | 1/10 |
| `subject` | 0.933 | 1.000 | 0.918 | 3/10 |
| `body_text` | **0.697** | 0.999 | 0.668 | **7/10** |
| `contact_info` | 1.000 | 1.000 | 1.000 | 0/10 |

3 of 10 bodies sit at the 1400-char cap in every process (`_25`, `_42`, `_76`).

**Conclusion.** `run1` — the first process after service start — reproduced the
2026-09-03 corpus run to 0.999 on `body_text`. `run2`, started warm, diverged on
7 of 10 bodies **and ran ~35% faster** (4.0–10.7 s vs 6.9–14.4 s). Greedy decoding
is not session-independent on this engine, and the long tail is where it moves.
Hypothesis, untested: llama-server prompt/KV cache state carried across requests.
The test is one flag — `cache_prompt=false` — across two processes.

**Consequence for every accuracy number below:** `body_text` CER for an affected
document measures the session it was scored in. Short fields are unaffected.

**Artefacts** `experiments/repeat_stability/{run1,run2}.jsonl`, `results_E11.json`, `run.py`.

## E12 — first accuracy numbers on the real corpus, dev split only

**Date** 2026-09-11 · **Corpus** real **dev**, 51 of 54 documents transcribed
(`_9`, `_90`–`_94` untranscribed) · **GT** `ground_truth_real_fixed_v2.jsonl`,
sha in results file · **Test split not scored** — held back until a configuration
is chosen.

**Provenance** as E10. Predictions are the stored 2026-09-03 runs with `_11`
re-generated after D51 (`splices` in each `.meta.json`). Scorer:
`score_real.py --split dev`, `RULES_DEFAULT` normalisation (ZWNJ→space, digits→ASCII,
letterforms folded). `subject` is additionally sliced by `subject_source`
(printed 6 / annotator 45): against an annotator-supplied subject the model is being
scored on summarisation, not OCR, and those rows are reported separately.

### Headline pair, plus effective CER (bootstrap 95% CI, by image)

| | arm A — shipped `/ocr` | arm B — `ocr_pipeline` grammar |
|---|---|---|
| usable output | 98.0% [94.1, 100] | 98.0% [94.1, 100] |
| omission rate (field slots) | **49.4%** [45.9, 53.3] | **23.1%** [20.4, 27.1] |
| hallucination rate | 0.4% | 1.2% |
| effective CER | **108.9%** [92.6, 124.9] | **92.6%** [76.6, 113.4] |
| CER given output | 132.2% | 90.3% |
| oracle-routed CER (best possible field assignment of the same strings) | **68.6%** [62.2, 74.9] | **61.2%** [55.0, 68.4] |
| share of CER from misassignment | 37% | 34% |

CER is Levenshtein / reference length and is **uncapped**: a 20-character
`sender` holding a 2,000-character letter scores ~100× — that is D44 arriving in
the accuracy number, and it is why arm A's `sender` CER is 448%.

### Per field (dev, scored slots only; `omit` = GT present, prediction absent)

| field | A n | A median CER | A ≤30% | A omit | B n | B median CER | B ≤30% | B omit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sender` | 50 | 45% | 9/50 | 1 | 46 | **0%** | 32/46 | 5 |
| `receiver` | 28 | 29% | 15/28 | 20 | 47 | 23% | 24/47 | 1 |
| `subject` (printed only, n=6) | 4 | — | — | 2 | 6 | 44% (micro) | — | 0 |
| `body_text` | 8 | 75% | **0/8** | 43 | 48 | **71%** | **1/48** | 3 |
| `contact_info` | 31 | 25% | 28/31 | 20 | 2 | 61% | 1/2 | 49 |

### What the numbers say

1. **The letter body is not being read.** Median body CER is 71–75% in both arms;
   **1 of 56 scored bodies** across both arms is within 30% CER. 15 of arm B's 48
   bodies exceed 100% — cap-running text (D52) and generated digits (E10).
2. **Arm B is better and still not good.** It halves omissions and fixes routing
   (`sender` exact-match 16% → 54%), but its best-case oracle-routed CER is 61%.
   The A/B question of E9 is answered on dev: the grammar recovers *presence*, not
   *correctness* — D46 generalises from digits to the whole body.
3. **What reads well is what repeats.** `sender` (one company's letterhead on most
   pages) and `contact_info` (the same footer) are the only fields with a majority
   of slots under 30% CER. They measure memorisation of a template as much as OCR.
4. **D49 now has a number.** With perfect routing the pinned model would still
   produce ~61–69% character error on these pages. That is not a tuning gap.

**Decision.** No configuration change; nothing tuned. Test split remains unscored.
Next: `cache_prompt=false` (D52), then the model-fitness call (D49) — a different
model, not more knobs, unless a Track A item moves body CER by tens of points.

**Artefacts** `results_real.armA_dev.json`, `results_real.armB_dev.json`
(`split_filter: dev`; `results_real.json` untouched), `gt_real_audit.md`,
scorer changes in `score_real.py` (arm-B rows, `--split`, `--out`, `subject_source`
slices) guarded by `test_score_real.py`.

## E13 — diagnostics: vision path, digit glyphs, tokenizer, memorisation

**Date** 2026-09-13 · **Report** `ocr_eval/reports/diagnosis_2026-09-13.md` · engine as E10; probes call `llama-server` directly (`/v1/chat/completions`, `/tokenize`), `temperature=0 top_k=1 cache_prompt=false`, no JSON prompt, no grammar unless stated.

**E13a — digit cards** (24 synthetic cards, Tahoma, 40/80/160 px). Blank page → "This image is blank" (vision path intact). Latin digits **12/12 exact**; Persian-Indic **0/12** at every size (`221` for `۳۲۱۰۰۰۰۰۰`, dates for random IDs, `١١١١…` runs). Digits-only GBNF grammar: still `221`. Tokenizer: Persian/Arabic-Indic digits are two byte-fallback tokens each (`[219][176..185]`); Latin digits one token. → **D53**.

**E13b — novel headers** (6 letters with never-seen names/numbers + 2 controls with the corpus letterhead, 2300×3200, arm B). Novel `receiver` CER **0.006**; novel person names inside bodies read exactly; novel `sender` 0.44 (2 of 6 not extracted) vs control 0.05; body CER novel 1.68 (one cap runaway) vs control 0.20; `contact_info` omitted 8/8; amount and national-ID digits found **0/8**. → memorisation refuted; digits fail on clean input too.

**E13c — Qwen2.5-VL-7B-Instruct Q4_K_M, same cards, same engine build** (probe on :18235, production engine stopped and restarted after): Latin **12/12**, Persian **0/12**. Not a quantisation comparison — a family comparison; the defect is shared.

**Effective resolution** (from `usage.prompt_tokens`): real scans reach the model at ~4k image tokens ≈ 0.67 scale. → **D54**.

**CER normalisation verified**: `۳۲۱/۰۰۰/۰۰۰` vs `321/000/000` → CER 0.00 under `RULES_DEFAULT`; arm B dev body CER median 0.71 as scored, **0.59 with all digits deleted from both sides** — digits are 5.4% of body characters and ~12 CER points; the rest is words on real scans.

**Not run:** Q8/fp16 comparison of coreOCR (no Q8 on disk; `hf_cache` base model missing shard 1/5; no torch; 12 GB VRAM). Tesseract cross-check (not installed).

**Artefacts** `experiments/diagnostics_0913/`.

## E14 — hybrid service benchmark: Qwen2.5-VL vs coreOCR, Tesseract vs glyph reader, resolution

**Date** 2026-09-17 · **Corpus** real **dev**, 51 documents, 281 GT numbers (`ocr_eval/eval_set.jsonl`, derived from the human GT only) · **Test split untouched**

**Pipeline** `ocr_service` (new package): preprocess (crop/deskew/≤300 KB JPEG) → primary VLM via `ocr_pipeline.LetterExtractor` (GBNF + DRY, `temperature=0`, `cache_prompt=false`, image cap 4096 tokens) → letterform normalisation → numeric validation → response. Engine: same pinned `llama-server` 2.28.2, one process per model. Provenance in each `ocr_eval/benchmarks/bench_*.json`.

| metric | Qwen2.5-VL-7B Q4_K_M | coreOCR-7B Q4_K_S | Qwen2.5-VL, glyph reader |
|---|---:|---:|---:|
| usable | 100.0% | 94.1% | 100.0% |
| CER body (micro, uncapped) | **37.5%** | 110.6% | 37.5% |
| CER all fields | 36.6% | 106.3% | 36.6% |
| per-field CER sender / receiver / body / contact | 13.1 / 16.6 / 37.5 / 26.4% | — | same |
| numeric recall, VLM only ("before") | **26.0%** (73/281) | 2.5% | 26.0% |
| numeric recall after classical layer | 8.9% (Tesseract, prefer-classical) | 2.9% | 21.3% (plausibility rule) → **26.0%** (checksum-only rule, offline) |
| numeric recall, oracle (any candidate) | 30.2% | 5.7% | 28.1% |
| `high` precision | 4/4 | — | **15/15** |
| conflict rate (`low`) | 88.4% | 90.2% | 92.7% |
| latency p50 / p95 | 9.6 / 13.0 s | 11.6 / 21.4 s | 10.3 / 15.1 s |
| cross-check (sequential coreOCR): numeric mismatch / body agreement | 83% / 0.33 | — | — |

**Resolution** (10 dev docs, one variable): 4096 vs 16384 image tokens → body CER 25.0% vs 27.5%, numeric recall 30.5% vs 32.2%, p50 10.9 s vs 122 s. → D54 closed-refuted.

**Conclusions.**
1. **Qwen2.5-VL replaces coreOCR on the evidence**: body CER 37.5% vs 110.6% through the identical pipeline; numeric recall 26% vs 2.5%. This also moves the real-corpus body CER from 71% (E12, coreOCR, old path) to 37.5%.
2. **Numbers remain the open problem**: 74% of the primary's numbers are wrong. Tesseract cannot read this typeface's digits (D55); the glyph reader reads clean crops (93.8% synthetic holdout, the `_46` national ID exactly) but its page-level locator is noisy, so as an *overrider* it is net negative (D56). As a *confirmer* it is exact: `high` = correct, 15/15.
3. **The service is operational**: sync `/ocr`, async `202 → /jobs/{id}` verified live (filesystem broker; Redis via `deploy/run_redis_local.ps1`), Tesseract installed, VRAM guard + `cpu_offload`/`sequential` secondary modes, Windows-asyncio spawn bug fixed.

**Next, in order:** (a) glyph-reader locator: line-level segmentation tuned on the footer/letterhead runs, then re-run `--label qwen25_dev_glyph2` — the target is oracle recall ≫ 30%; (b) `cache_prompt=false` is now the default here — re-run E11 on this path to confirm body stability; (c) test split once, for the chosen configuration.

**Artefacts** `ocr_eval/benchmarks/bench_{qwen25_dev,qwen25_dev_glyph,coreocr_dev,qwen25_dev10_4k,qwen25_dev10_native}.json`, `compare_*.md`; runner `ocr_service/benchmark.py`; model `models/digits_hog_svm.joblib`.

## E15 — v2 digit reader (CNN + reject class) and atom-level numeric reconciliation

**Date** 2026-09-22 · **Corpus** real **dev**, 51 documents, **434 GT atoms** (maximal digit runs ≥ 3, all fields; D57) / 281 GT spans · **Test split untouched** · **No VLM inference in this entry**: every number below is the classical reader alone, or a replay of the reconciliation over the *stored* Qwen2.5-VL output of E14 (`bench_qwen25_dev_glyph.json`), so the VLM side is byte-identical to E14.

**Why.** E14 left numeric recall at 26% with the v1 HOG-SVM reader usable only as a confirmer (D55/D56). Measured first (`ocr_eval/tools/bench_digit_reader.py`, reader alone vs GT): v1 finds 47.9% of atoms at **10.2% precision** — it has no "not a digit" class, so every narrow Persian letter fragment becomes a digit. Also measured: the VLM is *exact* on 33.6% of atoms, emits a near-miss in the right field for 47.7%, and omits only 13.8% — so alignment, not reading, is the missing piece.

**Reader (`ocr_service/digit_reader_v2.py`).** Small CNN (3 conv + fc, numpy inference, no new dependency) over 32×32 binary glyphs + 3 line-context scalars; 25 classes = 10 Persian/Arabic-Indic values, 10 Latin values, 4 separators, reject. Training data (`digit_cnn_data.py`): 393k components from rendered text lines mixing Dehkhoda words (shaped through Presentation Forms-B), numbers in all three digit scripts, Latin/email tokens, scan augmentation, across the 137 installed Persian fonts. Two data corrections that mattered: 123 of the 137 fonts draw ASCII digits with Persian glyphs (labels now describe pixels); the corpus's لادن footer prints `۰` as a small ring (rendered as a variant). By-font holdout (18 unseen fonts): digit accuracy 89.4%, **office fonts 93.7%**, letter→digit 12.6% per glyph (`train_v2g_holdout.log`); shipped model trained on all fonts (`models/digit_cnn.npz`, sha `e0e286aaf5595898`).

**Reader alone, dev, atom recall / precision** (`experiments/digits_0922/reader_*.json`), one change per row:

| step | recall | precision | what changed |
|---|---:|---:|---|
| v1 HOG-SVM (E14 reader) | 47.9% | 10.2% | baseline |
| v2b CNN, first model | 41.7% | 48.7% | reject class; Latin labels still by code point |
| v2c | 62.4% | 65.1% | pixel-true script labels; all fonts; colon pairs; digit-mass membership |
| v2d–e | 65.2% | 68.2% | dot-zero at run edges; zeros re-judged against the run's digit height |
| v2f | 68.7% | 70.6% | ring-zero rendering |
| v2g–h | 63.8% | 61.0% | *regression*: percentile line reference broke colon detection |
| v2i–k | 71.4% | 67.5% | line clustering rewritten (tall seeds, small blobs attached; `lines[-60:]` window bug; band guard by member median) |
| v2l–n | 73.7% | 67.8% | value-pooled probabilities; run splitting at weak glyphs; edge zeros need p≥0.4; speck lines dropped |
| **v2o, prepared image** | **79.0%** | **72.1%** | over-tall clusters re-split by local adjacency; run on the deskewed image the VLM gets |

By length (v2o): 8-digit 148/157, 10-digit 11/13, 11-digit 7/9, 24-digit 3/4. Residual misses: D58 GT errors (~14), D59 handwriting (~10), the email's bold-serif Latin `2006` (~12), duplicates. False positives are mostly real page numbers outside any GT field (registration number `۲۷۸۷۴`, the correct national ID of `_11`). ~0.5–1 s per page on CPU.

**Reconciliation (`ocr_service/numeric_reconcile.py`), replayed on E14's stored VLM output** (`replay_reconcile.py`, `recon_recon_final.json`):

| | before (VLM as emitted) | after |
|---|---:|---:|
| span recall (legacy unit, D57) | 26.0% | **44.1%** |
| atom recall, field text | 33.6% | **68.2%** |
| atom precision, field text | 38.2% | **73.1%** |
| atom recall, `numeric_fields` view (resolved + added) | — | **70.7%** |
| `body_text` atoms (163) | 61 | 91 (103 as records) |
| `contact_info` atoms (270) | 85 | 205 |

Decisions: confirmed 160 · **corrected 236 (83.1% now right, 12.7% were right before)** · conflict 1 · unverified 45 · added 44 (footer 24 at 96% precision; body 20, of which most "wrong" ones are D58 `_43`). Without injection: 64.5%. Policy sweep on dev (`override_conf` 0.45→0.35, `override_mean` 0.70→0.60: +0.9 pp; other knobs flat) — **tuned on dev**, the test split will say whether it held.

**Shipped.** `numeric_reader: "glyph2"` is the default (`config.py`); `pipeline.py` runs the reader on the prepared page, patches corrected digits into the field text in the model's own script, appends omitted footer numbers to `contact_info` (never to the body prose), and reports every number with confidence `high | corrected | low | unverified | added` (`schemas.py`). Guards: `tests/test_numeric_v2.py` — 14 tests incl. a must-fail fixture and an end-to-end `OcrPipeline.run` with a stub backend (no engine); full suite 150 passing. Provenance now stamps the reader and the model hash.

**Not done here, blocked on the GPU.** Another project's `llama-server` (Gemma-4, port 8080) holds 6.6 GB of the 12 GB card; Qwen2.5-VL Q4_K_M needs ~8.5 GB and `engine_min_free_vram_mb=9000` would drop it to CPU. Still pending, in order: (1) the live dev run through `OcrPipeline` end to end — the replay fixes the VLM side to E14's stored output, so only this measures the shipped path; (2) the E11 body-stability re-run with `cache_prompt=false` to close D52; (3) the **test split, scored once**, for the chosen configuration. The reconciliation policy was tuned on dev, so the test split is the honest number and none is claimed here.

## E16 — `cache_prompt` and session stability on the shipped path (closes D52)

**Date** 2026-09-22 · **Corpus** the 10-document `ci_gate.dev_sample` · **GT-free** · **Variable** `cache_prompt`, and the process

**Why now.** D52 (E11, 2026-09-11) found `body_text` differing between processes on **7 of 10** documents at temperature 0, on the *old* path (coreOCR on :18234 via `ocr_pipeline.LetterExtractor`). The proposed test was one flag. E14 then recorded "`cache_prompt=false` is now the default here" — which **was never true**: the flag was declared in `Settings` and stamped into `config.provenance()` but nothing put it on the wire, and llama.cpp defaults it to `true` (**D61**, found while building this experiment, now fixed with a test that asserts on the request rather than the config). So this is the first time the flag has ever been sent.

**Design.** Four fresh processes: two with `cache_prompt=true`, two with `false`, same bytes, same code, shipped path (Qwen2.5-VL + GBNF + DRY through `ocr_service.OcrPipeline`). Runner `ocr_eval/experiments/repeat_stability/run_service.py`.

| arm | `sender` | `receiver` | `subject` | `body_text` | `contact_info` | docs < 0.9 |
|---|---:|---:|---:|---:|---:|---:|
| `cache_prompt=true` | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 | 0/10 |
| `cache_prompt=false` | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 | 0/10 |

Cross-arm: `cache_prompt=false` run 1 vs `true` run 1 — **10 of 10 documents byte-identical**; mean latency 8.89 s vs 8.92 s.

**Conclusion.** **D52 does not reproduce on the shipped path, and `cache_prompt` is not the mechanism.** Both arms are perfectly repeatable across processes and identical to each other, so the flag changes neither output nor latency here. The instability E11 measured was a property of the **coreOCR configuration**, not of llama-server's prompt cache: E11's affected documents ran to the 1400-char grammar cap (3 of 10 in every process) — degenerate long-tail text, the D23/D39 family — whereas under Qwen2.5-VL the longest body in this sample is 988 characters and none reaches the cap. Removing the cap-running model removed the instability.

**Consequences.** (1) Every `body_text` number measured on the shipped path is a property of the model, not of the session — the E12 caveat "any body-level number for an affected document measures the session" no longer applies to the current configuration. (2) `cache_prompt=false` is kept as the default anyway: it costs nothing measurable and keeps requests independent, which is the safer contract for a service. (3) D52 is closed **for this configuration**; it stands as measured for coreOCR, which is no longer the primary.

**Artefacts** `experiments/repeat_stability/{svc_cache_on_1,svc_cache_on_2,svc_cache_off_1,svc_cache_off_2}.jsonl` + `.meta.json`, `results_E16.json`, log `experiments/digits_0922/e16.log`.

## E17 — test split, scored once (Phase 4 exit measurement)

**Date** 2026-09-22 · **Corpus** real **test**, 33 documents, **268 GT atoms** / 161 GT spans · **Configuration frozen before the run**: Qwen2.5-VL-7B Q4_K_M, GBNF + DRY, `temperature=0`, `cache_prompt=false`, image cap 4096, `numeric_reader=glyph2`, reconciliation `Policy(override_conf=0.35, override_mean=0.60, align_span=0.6, inject_min_digits=7, inject_body_min_digits=4, …)` — every threshold chosen on **dev** (E15) and unchanged here. **Scored once. Nothing was tuned after seeing these numbers.**

| metric | dev (E15/live, n=51) | **test (n=33)** |
|---|---:|---:|
| usable output | 100.0% | **100.0%** |
| numeric **atom** recall — before | 33.6% | 23.9% |
| numeric **atom** recall — after | 71.2% | **53.4%** |
| numeric atom precision — before | 38.2% | 31.1% |
| numeric atom precision — after | 71.5% | **64.1%** |
| span recall (legacy unit) before → after | 26.0% → 46.3% | 20.8% → **32.1%** |
| body CER | 37.2% | **31.9%** |
| CER all fields | 36.3% | 32.3% |
| latency p50 / p95 | 8.27 / 11.83 s | **8.38 / 10.82 s** |

**The generalisation gap is real: 71.2% dev vs 53.4% test.** It is not the reader. Reader alone against GT (`reader_cnn_v2o_test.json`): **74.6%** atom recall / 67.8% precision on test vs 79.0% / 72.1% on dev — a 4.4 pp drop, i.e. the CNN transfers. Decomposing the 268 test atoms:

| | n | share |
|---|---:|---:|
| delivered in the response | 152 | 56.7% |
| **reader read it correctly, but nothing delivered it** | **56** | **20.9%** |
| other loss | 3 | 1.1% |
| reader did not read it | 57 | 21.3% |

So a fifth of the test corpus's numbers were *read* and then dropped. Cause: reconciliation only emits a number when a VLM atom aligns to it, or when injection fires — and the VLM's own recall is lower on test (23.9% vs 33.6%), so there are fewer anchors. Of those 56 lost atoms, **43 are 3-digit** and 42 sit in `body_text`: they are the groups of long amounts (`۸۱۱/۹۶۶/۳۱۹…` in `_12` and similar) that the model omitted entirely, and `inject_body_min_digits=4` deliberately excludes 3-digit body atoms because on dev that floor was what kept injection precision acceptable. The trade-off was tuned on a split whose documents happen to carry fewer such amounts.

**Headroom, to be tuned on dev only:** allow a 3-digit body atom to be injected when it is part of a multi-group amount the reader read as one run (`۸۱۱/۹۶۶/۳۱۹` is one read, not three independent 3-digit guesses). That is a change to `numeric_reconcile.Policy` + `align_span`, measurable with `replay_reconcile.py` without inference. **Note for whoever does it: the test split has now been looked at once. Re-scoring it after tuning is a second look and must be reported as such.**

**Per-class reliability holds on test, and `high` is the trustworthy signal** (D62). Test precision of each confidence class, resolved value fully present in the GT field: `high` **82.6%** (23), `corrected` 68.8% (93), `added` 55.6% (18), `unverified` **20.0%** (20), `low` 0% (3). Same ordering as dev, with `corrected`/`added` ~9-13 pp lower. The review flag (`low` or `unverified`) fires on 15 of 33 documents and catches 14 of the 26 that hold a wrong number — a 46% miss rate, matching dev's 47%. `unverified` at 20% right is the clearest evidence that a number with no page anchor should never be trusted.

**Splits are comparable in composition** (4 distinct senders each, same dominant letterhead 48/51 vs 29/33), so the gap is not a corpus-family difference; it is the interaction of VLM omission with a conservative injection floor.

**Artefacts** `ocr_eval/benchmarks/bench_qwen25_test_glyph2.json`, `bench_qwen25_dev_glyph2.json`, `experiments/digits_0922/reader_cnn_v2o_test.json`, logs `benchmarks_qwen25_{dev,test}_glyph2.log`.

## E18 — amount integrity: the truncation bug, and the test split re-scored (second look)

**Date** 2026-09-22 · **Corpus** dev 51 (tuning) then **test 33, second look** · **Trigger** user report: large amounts arriving with parts missing.

**Three faults, all on grouped numbers** (`۳۲۱/۰۰۰/۰۰۰`, dates, phone lists) — registered as **D63**:

1. **The reader dropped trailing zeros.** A dot-zero is ~0.3 of digit height; line clustering seeded lines from glyphs "tall" relative to a page-wide percentile, which noisy scans drag to 19 px on a page whose digits are 40 px. The last `۰` of an amount therefore seeded its own line and was lost: `۳۲۱/۰۰۰/۰۰۰` was delivered as `۳۲۱/۰۰۰/۰۰` — **an amount divided by ten, silently.** Fixed with a page-level **run extension**: a number absorbs an adjacent digit glyph whichever line cluster it landed in, gated on the run's own **pitch** (a real next digit continues the number's spacing; a nearby letter's dot does not) plus a local colon check. Verified on `_11`.
2. **Injection judged chunks, not numbers.** An amount the model omitted entirely was offered to `inject_body_min_digits=4` as separate 3-digit atoms and discarded group by group. Grouped numbers are now injected **as one unit with separators intact**, qualifying on total digit count; the per-chunk floor is effectively gone (`inject_body_min_digits=3`, and an atom is 3+ digits by definition — retuned on dev, better on both recall and precision).
3. **The metric could not see either bug.** `۳۲۱/۰۰۰/۰۰` still yields the atoms `321` and `000`, so E15/E17 scored a truncated amount as a partial success. All three benchmarks now also report **grouped numbers intact**.

Multi-group numbers are laid out right-to-left, so the reader legitimately sees `۰۳/۰۸/۱۴۰۳` for `۱۴۰۳/۰۸/۰۳`; alignment accepts either order — digits from the reader, ordering from the model.

**Test split, second look** (frozen policy, tuned on dev only; the split was scored once in E17 and this is a disclosed re-score):

| metric | E17 test | **E18 test** | dev (live) |
|---|---:|---:|---:|
| numeric atom recall | 53.4% | **63.8%** | 71.2% → 76.7% (replay) |
| numeric atom precision | 64.1% | **67.1%** | 74.5% |
| span recall | 32.1% | **42.1%** | 46.3% |
| **grouped numbers intact** (model alone → delivered) | 6.5% → 29.0% | 6.5% → **30.6%** | 10.0% → 38.2% |
| body CER | 31.9% | 32.0% | 37.2% |
| usable | 100% | 100% | 100% |
| latency p50 | 8.38 s | **8.24 s** | 8.27 s |

Class precision on test: `high` 83% (23), `corrected` 71% (90), `added` 67% (33, up from 56%), `unverified` 20% (20), `low` 50% (6).

**Read this honestly.** The truncation defect is fixed and guarded, and atom recall gained **+10.4 pp on held-out data**. But **whole-amount exactness on test moved only 29.0% → 30.6%**: roughly **two thirds of grouped numbers still carry at least one wrong digit somewhere in them**, which the atom metric partially hides because most of their groups are right. The remaining loss is not the truncation bug — it is ordinary digit error distributed across a long number, plus the bidi ordering ambiguity. **For financial use this is the number that matters, and it is not production-ready.**

**Note on method:** the test split has now been looked at twice. Its value as an unbiased estimate is weakened; a third tuning cycle should use a fresh split or nested validation.

**Artefacts** `ocr_eval/benchmarks/bench_qwen25_test_amounts.json`, `reader_cnn_v3_dev.json`, `recon_amounts_final.json`.

## E19 — full-text transcription replaces the VLM letter-JSON path (Tesseract ensemble + CNN digits)

**Date** 2026-09-27 · **Corpus** real dev 51 (all tuning), real test 33 (final check), plus the user's own samples (a general-prose page with hand-typed GT, and three letters from this corpus) · **Trigger** user report: long / multi-paragraph pages and pages with tables came back as one paragraph; a non-letter page got an invented sender and receiver; "an image can be anything — extract everything in it".

**Root causes of the report** (registered as D64–D66): the GBNF grammar capped `body_text` at 1,400 characters, so a long page could not be written out and the model stopped after a paragraph; the letter-only prompt forced five letter fields and the model filled them for a page that has none (`دفتر خدمات شهرداری میاندوآب` / `دکتر علی حسینی` — neither on the page); without the grammar, Qwen2.5-VL-7B transcribed every paragraph but emitted words of dense right-to-left lines out of order, on full pages, single lines and line fragments alike.

**Screening** (8 dev letters + the prose page; metric = coverage CER, see `ocr_eval/fulltext_score.py`: per GT field, the edit distance to the best-matching substring of the transcript, so extra text such as signature blocks is not charged but missing, misread or reordered text is):

| method | letters CER | prose page CER | letters digit atoms |
|---|---:|---:|---:|
| previous service (stored predictions) | 23.0% | — (one paragraph) | 69.1% |
| Tesseract fas, page psm 3 | 20.5% | 23.2% | 64.7% |
| + page binarisation / rule + speck removal, layout psm 4 | 17.6% | 23.2% | 58.8% |
| + CNN digit reader replaces confirmed number tokens | 17.5% | 22.9% | 76.5% |
| + VLM line proofreading of the Tesseract draft | 17.2% | 19.3% | 76.5% |
| own line finder, psm 7→13, fas/eng per line | 15.0% | 20.8% | 83.1% |
| **candidates from psm 3+4+6 and own lines, best per line** | **12.9%** | **11.7%** | **84.6%** |

VLM proofreading was dropped: +12 s/page on the GPU for −0.3 pp on letters, and it dropped number placeholders in 72 of 160 lines; VLM reading of regions Tesseract could not read (letterheads) produced invented text (`چاپ سازنده` for a logo, digits for handwriting) and was dropped too. **No language model remains in the runtime.**

**Dead ends measured on the way** (kept here so nobody repeats them): `-l fas+eng` corrupts Persian words with Latin fragments (read each line with `fas` and `eng` separately instead); `--psm 7` returns nothing for some clean lines that `--psm 13` reads perfectly; dropping large sparse components (to remove grids) deleted text a signature touched (remove rules pixel-wise with long kernels instead); a 1/15-page-width rule kernel deleted long Persian baselines (1/4); page-relative kernels deleted every «ا» on an image of a few large lines (kernels now also scale with text height, found by an end-to-end test on a rendered page); deskewing every page cost the digit reader more than it gained (only at ≥ 1°).

**Final, frozen before test** (`ocr_service/transcribe.py`, provenance below), dev 51 / test 33, previous service = stored per-document predictions of E17/E18 on the same documents:

| metric | dev previous | **dev E19** | test previous | **test E19** |
|---|---:|---:|---:|---:|
| coverage CER, all fields | 28.5% | **14.9%** | 24.7% | **18.9%** |
| … body_text | 32.9% | **11.7%** | 26.8% | **18.5%** |
| … receiver | 11.8% | **4.8%** | 8.0% | **5.2%** |
| … contact_info | 21.0% | **16.5%** | 24.5% | **15.3%** |
| … sender (letterhead) | **7.7%** | 53.3% | **12.4%** | 50.8% |
| digit atom recall | 70.8% | **81.3%** | 52.0% | **78.6%** |
| whole numbers exact | 54.8% | **66.9%** | 36.2% | **62.0%** |
| transcript / GT length | 0.80 | **0.95** | 0.84 | **0.92** |
| latency p50 / p95 | ~8.3 s GPU | **2.9 / 3.7 s CPU** | ~8.2 s GPU | **2.9 / 4.3 s CPU** |

Letter fields cut from the transcript by rules (`letter_fields.py`), plain per-field CER (null prediction = 100%):

| field | dev previous | dev E19 | test previous | test E19 |
|---|---:|---:|---:|---:|
| body_text | 37.2% | **21.5%** | **32.0%** | 32.3% |
| receiver | 16.6% | **14.1%** | **12.0%** | 18.5% |
| contact_info | 25.9% | **23.6%** | 26.0% | **24.4%** |
| subject | 98.6% | 95.5% | 84.2% | 80.3% |
| sender | **13.1%** | 99.7% | **20.8%** | 100% |
| fields filled where GT is null | 3 | 1 | 2 | **0** |

**Read this honestly.** The transcript is better than the previous service on every field but the letterhead, on both splits, at a third of the latency and without a GPU. The rule-cut `receiver` is **worse on test** (18.5% vs 12.0%) though better on dev. `subject` is not printed on most of these letters — the GT carries annotator-written summaries (`درخواست ارائه ضمانتنامه انجام تعهدات`) — so it stays null by design; the previous model's 80–99% there was invention. `sender` is a stylised font on a coloured banner that Tesseract cannot read (D67). **About one whole number in three still carries a wrong digit** (test 62.0% exact): not production-ready for money fields. `needs_review` fires on 50/51 dev and 33/33 test documents — not yet a useful triage signal; `numbers[].source` is.

**Worst test documents** (diagnosis only, nothing tuned on them): `_53` page rotated > 10° and curved, two-column name lists; `_89` very small text (page photographed from afar); `_6` handwritten values on a form; `_41` rotated and cropped. Worst dev: `_68` handwritten letter (previous 43.9%, now 64.9%), `_51`/`_55` perspective distortion.

**User samples** (HTTP, final code): the prose page — all four paragraphs, coverage CER 11.4%, 14/19 digit atoms, every letter field null (previously: one paragraph + invented sender/receiver); the table letter — every cell number right (`۳۲۱`, `۱۵۰۰`; previously `۲۳۱`, `۵۰۶`), rows right-to-left; the long letters — every paragraph, including the sentence with `۰۳/۰۸/۱۴۰۳ … ۰۳/۱۱/۱۴۰۳` the previous service dropped.

**Method note.** Tuned on dev only. The test split was scored three times in this experiment: at the freeze; after the kernel/glyph-height bug fix found by the rendered-page test (18.93% → 18.88%); after three general fixes motivated by the user's samples (adaptive upscaling for small text, right-to-left ordering of table cells, `سلام`-only salutation lines) — identical numbers, those fixes do not touch these letters. With E17 and E18 it has now been looked at five times; the next tuning cycle needs a fresh split.

**Provenance** `{"service_version": "1.0.0", "engine": "tesseract(fas,eng) + digit_cnn", "tesseract": "v5.4.0.20240606", "fas_sha256": "99e420969b5ddd2c", "eng_sha256": "7d4322bd2a774972", "digit_model_sha256": "e0e286aaf5595898", "layout_psms": "3,4,6", "deskew_min_deg": 1.0, "min_line_conf": 30.0, "number_min_prob": 0.6, "python": "3.12.10"}`

**Artefacts** `ocr_eval/benchmarks/bench_fulltext_dev.json`, `bench_fulltext_test.json` (per-document transcript, fields, scores, previous-service scores); `ocr_eval/fulltext_score.py`; `ocr_eval/tools/bench_fulltext.py`; `tests/test_transcribe.py`.

**Removed with this experiment** (git history keeps them): `main.py`, `engine.py`, `config.py`, `main_qwen.py`, `json_repair.py`, `ocr_pipeline/`, `ocr_service/backends/`, `ocr_service/benchmark.py`, `ocr_service/make_eval_set.py`, and the eval runners that drove the removed endpoints (`run_real.py`, `run_eval.py`, `run_experiment.py`, `run_pipeline_ab.py`, `ci_gate.py`, `analyze_fields.py`, `tools/regen_one.py`, `experiments/{orientation,repeat_stability,diagnostics_0913/novel_headers}`). The model weights under `D:\models` were not touched.

### E19b — follow-up after the user's Postman test (same day)

**Report:** on the prose page the first line ended «…رنگار شدند. سیم بحگاهی ک ۹» for «…رنگارنگ شدند. نسیم خنک صبحگاهی از میان»; the user suspected RTL maths in the junk filter, erased dots and tight crops.

**Measured diagnosis of that line:** the cleaned binary image of the line is complete (the clean-up removed 157 of 13,152 ink pixels, all shadow specks at the right page edge — no dot inside the text); `_trim_edges` only drops whole one-character tokens a line-height away from the rest and is direction-agnostic; every line image already has a 30 px white frame. The loss was in **selection**: the page is tilted 0.9° (below the 1° deskew threshold), psm 3/4 read the line as two clean halves (scores 22.4 + 19.8) and psm 6 as one piece merged with the next line (33.7). Greedy selection kept the single highest-scoring piece. Registered as **D70**.

**Changes:** (1) `_select` compares every chosen reading with all compatible combinations of the readings it displaced (exact search over ≤ 8 neighbours, bitmask) and swaps when the combination scores ≥ 5% more; (2) pieces of one row are junk-filtered separately and then joined right-to-left into one line (`_finish_rows`, `_join_row`) — joining before filtering glued a shadow speck «اس ی» onto a real line; (3) a word with ≥ 2 harakat has them removed — Tesseract invents them on bold text («بِمُدیرِیَت»), printed Persian carries at most one (**D71**); (4) the opening formula is found when OCR damaged it («اشه احترام بر مذاکرات»), never the closing «با احترام». Tried and rejected: line images built from whole connected components (to keep descenders of tilted lines) — prose page 10.6% → 14.8%, removed.

| | dev before | **dev after** | test before | **test after** |
|---|---:|---:|---:|---:|
| coverage CER | 14.9% | **14.9%** | 18.9% | **18.5%** |
| … body_text | 11.7% | **11.5%** | 18.5% | **17.8%** |
| digit atoms | 81.3% | **81.6%** | 78.6% | **79.5%** |
| whole numbers exact | 66.9% | **67.2%** | 62.0% | **62.4%** |
| field body_text | 21.5% | **21.3%** | 32.3% | **30.2%** |
| field contact_info | 23.6% | **23.8%** | 24.4% | **22.5%** |
| field receiver | 14.1% | **13.9%** | 18.5% | **17.8%** |
| prose page (hand GT) | 11.4% | **10.6%** | | |

The line now reads «…رنگارنگ شدند. نسیم خنک صبخگاهی از میان» (one dot error left, Tesseract's). The test split was scored twice more here (seven looks in total); nothing was tuned on it. Guards: `test_two_clean_halves_of_a_tilted_line_beat_one_merged_reading`, `test_a_junk_piece_is_dropped_before_its_row_is_joined`, `test_invented_harakat_are_stripped_but_a_real_single_mark_stays`, `test_an_ocr_damaged_opening_is_still_found_but_the_closing_formula_is_not_an_opening`.

---

## E20 — 2026-09-29 — amounts that can be trusted, kashida, and a page to read the result on

Asked for: financial numbers good enough to ship, stretched Persian words fixed, and a
visual UI instead of Postman. Architecture unchanged (Tesseract + CNN digit reader, CPU).

### Where the number errors actually were

224 whole-number failures across dev+test, classified by comparing each missed GT number
with its nearest reading in the transcript:

| share | class | example |
|---:|---|---|
| 46% | not read at all (letterhead, stamps, handwriting — D67/D68) | `10380436012` |
| 14% | digit substitution | `38388575` → `38288575` |
| **11%** | **spurious trailing zero** | `8228` → `82280` |
| **9%** | **invented separator** | `1403/09/06` → `1403/09/0/6` |
| 9% | digit dropped | `0943019631` → `943019631` |
| 6% | longer and different (mostly a `:0` suffix) | `…38388575` → `…38388575:0` |
| 5% | other spurious digit | `38388575` → `238388575` |

**31% of all number failures were our own doing**: the pixels were read correctly and the
code then corrupted the result. That is the part worth fixing first, and it needs no
better recognition at all.

Root causes, from the glyph data rather than from reading the code (D72, D73):

* the trailing `۰` of «۸۲۲۸۰» has **P(zero) = 0.05** — the classifier called it a `5` —
  and stands 4× the run's inter-digit gap away. `_finish` re-judged any short mid-line
  glyph as a dot-zero *after* trimming the run's ends, with no zero evidence required, so
  a comma or a letter's dot became a digit and could not then be dropped.
* `_fix_numbers` glued two reads of one number with a **guessed** separator
  (`":" if ":" in w.text else ("," if … else "/")`) — punctuation printed by rule in a
  service whose contract is that nothing is generated.

### Kashida

Word-level alignment over dev+test: 7,447 aligned word pairs, **90% exact**; 58 single
mid-word insertions, **41 of them «بلوار» → «بلسوار»** — one letterhead address, on 21 of
the 51 dev pages. Measured on the page: the elongation in «مشهـــد» is a **flat** run
(identical top and bottom for 45 columns), 17 px thick, standing **on the baseline** and
joined to a glyph at both ends. The three `-` dashes on the same line are just as flat but
float 17–24 px above the baseline and stand alone. A first detector that keyed only on
"flat and joined" squeezed dashes between phone numbers and cost 1.1 points of number
accuracy; adding the stroke-thickness and baseline tests removed the regression entirely.

The alternative offered — a table of common kashida misreadings — was refused: a
word-replacement table is the lexicon stage removed in phase 0.3, `test_no_lexicon_stage_in_runtime`
fails on it, and it prints words nobody read.

### Measured

| | dev (51) E19b | **dev E20** | test (33) E19b | **test E20** |
|---|---:|---:|---:|---:|
| coverage CER | 14.9% | **14.7%** | 18.5% | **18.1%** |
| … body_text | 11.5% | **11.3%** | 17.8% | **17.4%** |
| … contact_info | 16.6% | **16.1%** | 15.5% | **14.6%** |
| digit atoms | 81.6% | **83.6%** | 79.5% | **80.6%** |
| **whole numbers exact** | 67.2% | **75.8%** | 62.4% | **68.8%** |
| seconds per page (p50) | 2.9 | **2.9** | 3.0 | **3.0** |

Whole-number error rate 32.8% → **24.2%** on dev, 37.6% → **31.2%** on test. The test
split was scored once more here (eight looks in total); nothing was tuned on it.

### Thousands-group validation (D75)

`amount_grouping()` labels a number `ok` / `broken` / `none`. Validated against the corpus
ground truth: **27 correct amounts judged `ok`, 0 false alarms.** On the user's invoice it
flags «۱۶۵.۰۰۰۰۰۰» and «۱۲.۹۷۸۰۲۱۹.۵۷۱` and passes the four amounts that are right. It
**never repairs** — the missing digit would have to be invented — and it is also used to
choose between two readings of one token.

### The page at `/`

`ocr_service/static/index.html`, vanilla HTML/CSS/JS, no CDN, works offline. Drop an
image in: letter fields on top, right-to-left justified paragraphs, and a real `<table>`
wherever consecutive lines split into aligned cells (`lines[].cells`, new in the
response; columns clustered by cell x-overlap, rightmost first). Every number is listed
with its source and grouping verdict, and the line/cell boxes can be laid over the
original. Verified end to end in a browser on the user's invoice `_12`: 1 table rebuilt,
4 fields shown, 6 amounts judged — the two broken ones flagged.

Guards: `test_a_grouped_amount_is_judged_by_its_groups_and_a_date_is_left_alone`,
`test_one_number_read_in_two_pieces_is_rejoined_without_inventing_a_separator`,
`test_a_table_row_is_split_into_cells_and_prose_is_not`,
`test_a_stretched_connector_is_shortened_and_a_dash_is_left_alone`. 110 tests pass.

Still open: D67 (letterhead `sender` unread), D68 (handwriting/stamps/perspective),
D69 (`needs_review` not discriminative). Numbers are better but **not yet money-safe**:
about 1 in 4 whole numbers still has a wrong digit somewhere.

---

## E21 — 2026-09-29 — the typos are the capture, not the reader

**Trigger** user report: "obvious OCR typos caused by low image quality — «صبخگاهی»
instead of «صبحگاهی», «توسمعه» instead of «توسعه»", with an instruction to add a
post-processing spell-correction layer and to relax `test_no_lexicon_stage_in_runtime`
for it.

### The spell-corrector was built to policy, measured, and rejected (D77)

`dehkhoda/REMOVED.md` does not forbid correction outright; it sets four conditions. Built
to (a)-(c) — domain lexicon from the corpus (462 words seen in ≥3 documents),
frequency-weighted, edit distance 1, single unambiguous candidate — and run over the
user's prose page:

| | |
|---|---|
| real fixes | **3** (`گزارس`→`گزارش`, `فنسی`→`فنی`, `ایسن`→`این`) |
| corruptions | **11** — `شادی`→`هادی`, `گذشته`→`گشته` (both onto the signatory's name), `باید`→`باشد`, `میان`→`میدان`, `انسان`→`انسانی`, `بدون`→`بدین`, **`۱۴۰۴`→`۱۴۰۳`** |
| the reported word | **no candidate**: «صبحگاهی», «نسیم», «خنک» appear 0× in the corpus |

72% of the prose page's tokens are outside the domain lexicon, so "not in the lexicon"
does not mean "not a word" — which is the assumption the whole design rests on. This
independently reproduces the 2026-09-01 hand audit (2 improved, 47 damaged). Condition
(d) cannot be met by this design, so the guard stands unmodified and nothing shipped.

### What the typos actually are (D76)

Text height at native resolution:

| | text height | reads |
|---|--:|---|
| corpus scans (all 90) | 24-42 px | correctly |
| `prose_page.jpg` | **12 px** | «صبخگاهی», «توسمعه», «هوسمند» |
| `exampel_paper.png` | **7 px** | «استحعضصار», «یاسلام», «یرای» |

A Persian dot is 1-2 px at 12 px text height. The ink that separates ب/ی/پ/ن and ح/خ/ج
is not in the file, so no post-processor can recover it without guessing.

### Adaptive upscaling, and where it stops

Sweep on the prose page (Lanczos, single resample inside `flatten`), CER against a new
hand-typed transcript (`ocr_eval/samples/prose_page.gt.txt`, 325 words):

| working text height | CER | word recall | marker words |
|---:|---:|---:|---:|
| **24 px (E20)** | **8.8%** | 76.0% | 4/15 |
| 28 px | 10.3% | 71.1% | 9/15 |
| **30 px (shipped)** | **9.2%** | 75.7% | **9/15** |
| 32 px | 12.7% | 66.8% | 10/15 |
| 34 px | 11.5% | 72.9% | 9/15 |
| 36 px | 15.5% | 65.2% | 8/15 |

30 px holds CER flat and more than doubles the whole words read correctly, now including
**«صبحگاهی» and «توسعه»** — the two words reported. No target improves CER; the page is
marginal and resampling trades error classes.

Keyed on the text height **as supplied**, not the working height: the latter pulled the
24-29 px corpus scans up too and cost dev coverage CER 14.7% → 15.1%, whole numbers
75.8% → 75.2%, latency 2.91 → 3.04 s. With the native-height gate, **all 90 corpus images
keep their scale**, so the dev and test benchmarks are unchanged by construction.

### Also

`glyph_px` is now in the response and a review reason names the DPI to rescan at; the page
at `/` shows a red banner for it. A page that is not a letter now says so on the page and
states how much it extracted, because null letter fields were being read as "empty output"
— the full text was always in `text` (guard:
`test_a_page_that_is_not_a_letter_still_comes_back_whole_and_in_order`, which asserts all
23 lines, 4 paragraphs and top-to-bottom order on the prose page).

New: `ocr_eval/samples/` + `ocr_eval/tools/score_sample.py` — the corpus benchmark only
covers administrative letters, so the general-document case had no metric at all. 113 tests.

---

## E23 — 2026-09-29 — deleting lines is the wrong mechanism

**Trigger** user report: "the output is not good at all and it has declined". The only
change since the last good state was E22's junk filter.

E22 judged a line by its **mean token length** — drop when the mean is under three
characters and mean confidence under 60 — to remove handwriting, stamps and margin
marks. It does remove those. It also removes:

* **«با سلام (»** — the letter's opening formula. The stray bracket is a one-character
  token and pulls the mean to 2.33. `letter_fields` cuts `body_text` from that anchor.
* **a line that is a single token of two characters, unconditionally** — «۱۰» at
  confidence **1.00**: a table cell, a form value, a page number.

### Three filters, measured on dev (51)

| | coverage CER | atom recall | whole numbers | len_ratio |
|---|---:|---:|---:|---:|
| **no filter (E21)** | **14.65%** | **83.60%** | 75.76% | **0.948** |
| mean token length (E22) | 14.69% | 83.45% | 75.76% | 0.915 |
| corrected: exempt a confident long token or a ≥2-digit number | 14.72% | 83.60% | 75.76% | 0.937 |

The corrected filter keeps «با سلام (» and «۱۰», and drops the handwriting row on `_1` —
and still lands on the **worst CER of the three**. Every filter that removes noise removes
real text with it, and none of them improves a measured number.

The reason the damage was not caught when E22 shipped: **coverage CER never charges for
extra text**, so deleting text can only ever look free. E22's own note says as much and
argued the loss was noise; the 3.3% it removed was not all noise.

### What shipped instead

`_line_ok` is confidence only. Noise is **surfaced, not deleted**: `lines[].confidence`
already separates it — the handwritten registration block and the stamp on `_1` score
**0.44 and 0.45** against **0.75-0.89** for the letter's own lines — and the page at `/`
greys a line under 0.6 with a dotted underline that explains itself on hover.

A deleted line cannot be recovered by the person reading the page. A marked one can.
E22's UI fix (not printing the letter body twice) is kept.

---

## E24 — 2026-09-29 — dot confusions are repairable; the earlier attempt failed on its lexicon

**Trigger** the user, for the third time: the low-quality captures still carry spelling
mistakes, fix them after OCR and before the output.

D77 had measured a corrector at **3 fixes to 11 corruptions** and closed the question.
That verdict was about a particular design, and the design was wrong twice over:

* **the lexicon** was 462 words built from corpus OCR output, so 72% of a prose page fell
  outside it — "not in the lexicon" did not mean "not a word";
* **the edit** was edit distance, which lets any word reach any other: «شادی»→«هادی»,
  «گذشته»→«گشته», «۱۴۰۴»→«۱۴۰۳».

### What changed

**The lexicon.** Tesseract ships one. `combine_tessdata -u fas.traineddata` then
`dawg2wordlist` on `fas.lstm-word-dawg` yields **13,892 Persian words** — 156 KB, no new
dependency, offline, and exactly the vocabulary the recogniser was trained against. It
contains «صبحگاهی», «بنفشه», «هوشمند» and also «شادی», «گذشته», «باقری», so those are
words and are never candidates.

**The edit.** Not distance but **skeleton identity**. Fold every letter onto its
mark-group representative — بپتثنی, جچحخ, دذ, رزژ, سش, صض, طظ, عغ, فق, کگ, اآأإ, وؤ, هة —
and two words are candidates only if the folded forms are equal. That is exactly the
error a low-resolution scan makes: it loses and invents the marks, never the shapes. So
«صبخگاهی»→«صبحگاهی» is reachable and «شادی»→«هادی» is **not** — ش and ه are different
shapes. Digits have no group and are never touched.

Then five gates: the word must not itself be in the list; the reader must have been
unsure of it (conf < 85); exactly **one** lexicon word may share the skeleton; only
**one** mark may move; the length may not change.

That last gate is load-bearing. Without it «بمار» (a misreading of «بهار») becomes
«نماز» — same skeleton across two moved dots, and «بهار» is not a candidate at all
because ه is not a dot variant of م.

### Measured

| dev (51 letters) | off | **on** |
|---|---:|---:|
| coverage CER | 14.65% | **14.61%** |
| … receiver | 5.44% | **5.30%** |
| … subject | 36.56% | **36.45%** |
| … body_text | 11.33% | **11.28%** |
| digit atoms / whole numbers / len_ratio | — | **unchanged** |
| seconds per page (p50) | 2.95 | 2.96 |

| the user's prose page (12 px, 325 GT words) | off | **on** |
|---|---:|---:|
| CER | 9.24% | **9.00%** |
| word recall | 75.69% | **76.92%** |

**Every change audited, not sampled.** 27 word changes across the 51 dev letters, 22
distinct: **23 correct** — `آداره→اداره` ×4, `مدبریت→مدیریت` ×2, `آقدام→اقدام` ×2,
`جهن→جهت`, `منایع→منابع`, `الق→الف`, `قرائی→قرائت`, `مذیر/مدثر/مدبر/مدیز→مدیر`,
`پرستل→پرسنل`, `آرسال→ارسال`, `همچتین→همچنین`, `چتانچه→چنانچه`, `آمکان→امکان`,
`پانکی→بانکی`, `مناقصة→مناقصه` — **1 wrong**: `کتبا→کتیا`, because «کتباً» is correct
Persian but is absent from Tesseract's list, so it became a candidate. 3 ambiguous
(`خراید→جراید`, `یمک→نمک`, `تلد→بلد`). On the prose page: 5 changes, 4 correct, 1 wrong
(`خواهن→خواهی`, where the right word «خواهد» was unreachable — د is not a dot variant
of ن). On the 7 px page: 2 changes, both correct.

### The guard

`test_no_lexicon_stage_in_runtime` is **narrowed, not removed**: `spellfix` is admitted by
name, and `dehkhoda`, `corrector`, `lexicon`, `levenshtein`, `edit_distance`, `rapidfuzz`
and `difflib` stay blocked. `test_the_spell_fix_can_only_move_dots` pins the failures that
kept every previous corrector out — «شادی», «گذشته», «باقری», «۱۴۰۴», Latin, and any word
the reader was confident of.

**Residual:** a correct word missing from Tesseract's list can still be changed, which is
the one wrong change on dev. Adding the corpus's own administrative vocabulary would close
it, but that vocabulary cannot be measured on this corpus without leaking into the metric.

| test (33, held out, scored once) | before | **on** |
|---|---:|---:|
| coverage CER | 18.05% | **18.01%** |
| … receiver | 5.22% | **4.99%** |
| … subject | 31.86% | **31.70%** |
| … body_text | 17.38% | **17.35%** |
| digit atoms / whole numbers / len_ratio | — | **unchanged** |

Every field improved or held on data never tuned against; nothing regressed. Ninth look
at the test split (E17, E18, E19 ×5, E20, E24) — make a fresh split before the next cycle.
