# Phase 6 — decoding strategy A/B on the real corpus

Run 2026-09-03. All 90 real administrative letters, both arms, same engine, same
model, same images, same session. **GT-free: every number below is field
presence, shape, or a self-verifying checksum. No CER is claimed (D35).**

| | arm A | arm B |
|---|---|---|
| path | `main.py` inline — what `/ocr` serves today | `ocr_pipeline.LetterExtractor` defaults |
| decoding | free-form JSON prompt, `repeat_penalty=1.2` | GBNF grammar + DRY + `repeat_penalty=1.1` |
| recovery | JSON repair + 1 retry | grammar + JSON repair + 1 retry |
| guardrails | all-null rejection only | full structural validation |

One variable: the decoding strategy. Both knobs (grammar and sampler) move
together, deliberately — D39 established that they are not separable, because the
correct sampler is inverted by whether the grammar is on.

## Result

| metric | A: `main.py` | B: `ocr_pipeline` | |
|---|---:|---:|---|
| shipped, of 90 | **89** | 88 | −1 |
| **mean fields returned (of 5)** | **2.58** | **3.86** | **+1.28** |
| `sender` present | 100.0% | 87.5% | −12.5 pp |
| `receiver` present | 58.4% | **97.7%** | +39.3 pp |
| `subject` present | 21.3% | **98.9%** | +77.6 pp |
| `body_text` present | 20.2% | **97.7%** | +77.5 pp |
| `contact_info` present | 58.4% | **4.5%** | **−53.9 pp** |
| schema collapse (D44) | 20.2% | **0.0%** | −20.2 pp |
| Arabic letterforms (D43) | 34.8% | **0.0%** | −34.8 pp |
| national-ID checksum pass | 12/65 (18.5%) | 10/51 (19.6%) | ~unchanged |
| wall clock, 90 documents | 582s | 832s | +43% |

## What this says

**The letter body comes back.** `body_text` 20.2% → 97.7% and `subject` 21.3% →
98.9%. Arm A returns a `sender`-only stub on most documents (D41); arm B returns
an actual extraction. That is the whole point of the exercise.

**Two defects go to zero, by construction rather than by tuning.** Schema
collapse (D44) is 0.0% because bounded GBNF caps make a 2116-character `sender`
unrepresentable — the grammar cannot emit it. Arabic letterforms (D43) are 0.0%
because `persian_text.for_output()` folds them on the way out. Neither is a
threshold that could drift.

**Digit accuracy did not move.** National-ID checksum pass is 18.5% → 19.6%, i.e.
unchanged within noise on n=51–65. This is the honest negative result of the
experiment: **the grammar guarantees shape, not correctness.** Forcing five
fields fills five fields; it does not make the model read digits better. D42
stands, and only ground truth can size it properly.

## What got worse

**`contact_info` collapses, 58.4% → 4.5%.** This is a real regression and the one
result that should stop a straight cutover.

The likely mechanism, stated as a hypothesis rather than a finding: in arm A the
model skips the body and dumps footer text into `contact_info`, so a high fill
rate there is partly a *symptom* of the same failure that empties `body_text`. In
arm B the model writes the body where it belongs and then, at the end of a
grammar-constrained object with `body_text` capped at 1400 characters, reaches
`contact_info` with little budget left. That predicts the loss concentrates on
long documents — testable, not tested.

**`sender` 100% → 87.5%**, and one fewer document ships (89 → 88). Both small,
both real.

**43% slower.** 582s → 832s for 90 documents. Grammar-constrained decoding costs
throughput; the p50 stays single-digit seconds.

## Recommendation

Do not cut over as-is. Arm B is clearly the better extraction strategy — +1.28
fields per document, two defects eliminated structurally — but trading 58.4% →
4.5% on `contact_info` for it is not an acceptable swap when `contact_info` is
where the phone, email and IBAN live.

The sequence that makes sense: reproduce the `contact_info` loss against document
length, raise the field cap or reorder the grammar so `contact_info` is not last,
re-run this A/B, and only then decide about the serving path. That is one
experiment, not a research programme.

Until then both paths are measured and neither is silently promoted. `main.py`
still serves `/ocr`.

## Artefacts

- arm A predictions: `predictions_real.jsonl` (+ `.meta.json`)
- arm B predictions: `predictions_real_pipeline.jsonl` (+ `.meta.json`)
- pre-fix baseline: `predictions_real.prefix_rp10.jsonl`
- comparison: `results_ab_gtfree.json`
- runner: `ocr_eval/run_pipeline_ab.py`

When ground truth arrives, both arms re-score with no re-inference:

```bash
venv312\Scripts\python.exe ocr_eval/score_real.py --predictions ocr_eval/predictions_real.jsonl --label arm_A
venv312\Scripts\python.exe ocr_eval/score_real.py --predictions ocr_eval/predictions_real_pipeline.jsonl --label arm_B
```
