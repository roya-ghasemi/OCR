# Phase 1 report — field-aware measurement

**Date:** 2026-09-01 · **Method change only, no model/prompt/decoding/preprocessing change.**
Re-scores the existing `predictions.jsonl` (30 usable of 36) — the movement in numbers
is attributable to the measurement, not to the model.

Provenance: engine `fe2adf0`, model `coreOCR-7B-050325-preview.Q4_K_S` (4,457,769,440 B),
mmproj sha256 `3493…d308a78`, temp 0.0, ctx 16384, image-min-tokens 1024.
Dataset seed **20260827** (split seed 20260829). 36 images / 18 distinct texts / 2 layouts.

## The headline was measuring the wrong thing

The flat-blob 17.04% CER concatenated all five fields and compared to one flat string.
That silently averaged *character reading* with *field placement*. Decomposed:

| Metric | CER | 95% CI over texts | Reads as |
|---|--:|---|---|
| Character reading (placement-free) | **9.67%** | [5.9, 13.7] | how well characters are read, wherever they land |
| Named-field (placement-strict) | **36.68%** | [21.5, 55.1] | right text **and** right JSON key |
| **Assignment loss** (strict − free) | **27.01 pts** | — | error that is **misfiling, not misreading** |
| old flat-blob (for reference) | 17.04% | — | a coincidental average of the two above |

**The model reads Persian/bilingual text at ~10% CER. The dominant error is putting
correct text under the wrong key.** This also explains why the Dehkhoda dictionary (which
only touches *reading*) had almost no room to help: reading was never the bottleneck.

## Where the placement error lives (per-field named CER)

| Field | n | Named CER | Defect |
|---|--:|--:|---|
| sender | 30 | 54.82% | org vs. signatory ambiguity — the prompt itself says "from letterhead **or** signature" |
| subject | 30 | 49.89% | **17/30** keep the `موضوع:` label; prompt asks for the value only |
| body_text | 30 | 28.50% | largest field; absorbs misfiled orphan content |
| contact_info | 6 | **100.00%** | the email/phone value **never** lands here |
| receiver | 0 | — | GT is null on every letter, yet **populated on 19/30** (false-positive field) |

## Recall, precision, and the schema gap

- **Omission (schema content): 4.6%** — 10/216 fine segments dropped. The model captures
  almost everything on the page.
- **Hallucination (pure invention): 0/101 predicted values.** The invented-looking
  `receiver` values (e.g. `مهمان سازمانی سرویس اینترنت پرسرعت`) are **borrowed page text
  misfiled**, not fabrication — they trace back to the subject line. Phase 0's name-only
  check said "no hallucination"; this confirms it at the value level and reclassifies the
  behaviour as placement error.
- **Orphan capture: 79.5%** — 31/39 schema-homeless segments (date, IBAN, economic/contract
  codes, amounts, national ID, version) are read and stuffed **somewhere** because the
  five-field schema has no home for them. This manufactured misassignment is a **schema
  limitation, not a model failure**, and it is a large part of the bilingual gap.

## Digit / code ordering (bilingual RTL/LTR)

40 exact · 1 reversed · 3 permuted · 2 partial · 17 misread-or-omitted. The systematic
digit-reversal hypothesis is **not** supported (1 reversal). Long numeric IDs
(`0071234567`) are the real weak point — permuted/partial, not reversed.

## Latency

| | p50 | p90 | p99 | mean |
|---|--:|--:|--:|--:|
| ok only | 1.86s | 2.29s | 2.56s | 1.93s |
| all (incl. 422s) | 1.90s | 2.56s | 17.77s | 3.15s |

p99 ≈ max at n=18 texts; the three 422s (one defect, n=1) dominate the all-images tail.

## What this hands Phase 2/3 (do NOT act yet)

Reading is already good; the gains are in **output structure**, all fixable via prompt/schema:
1. `receiver: null` unless an explicit addressee is present (19 false-positives).
2. Route email/phone to `contact_info` (currently 0/6 correct).
3. Strip the `موضوع:` label from `subject` (17/30).
4. Disambiguate `sender` = letterhead org, not signatory.
5. Decide a home for schema-orphaned content (date/IBAN/codes) — add fields or an explicit
   `other`/`extra` bucket — before it can be scored instead of silently misfiled.

Measurement is now trustworthy: every number is decomposed, CI-bounded, and reproducible
from `predictions.jsonl` with no inference. Phase 1 exit criteria met.
