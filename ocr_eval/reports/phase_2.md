# Phase 2 report — reliability on the 90 real documents

Generated 2026-09-03T12:49:00+00:00. **90 real administrative letters, GT-free metrics only (D15)**

> Every number here is GT-free. Accuracy on the real corpus remains blocked by defect D15 (no ground truth). Nothing below is a CER.


## Headline: usable-output rate

- **89 of 90 documents produced usable output — 98.89% [96.67, 100.00]**
- Dev split only: 98.15% [94.44, 100.00]
- Phase 2 exit criterion is **≥99%**.

### Failure taxonomy

| failure kind | count |
|---|---|
| `http_422` | 1 |

An all-null 200 counts as a FAILURE here. The service reports it as success and nothing downstream notices -- that is defect D4.


## Degeneracy — the repetition loop (D23)

A run of the same character ≥20 long is degenerate output, not transcription. Legitimate Persian identifiers top out around 12 digits.

- Documents containing a degenerate run: **2 (2.22%)**
- Of those, **1 passed as usable output** — they returned 200 with corrupted content.
- Repeated characters: `{'U+06F0': 1, 'U+002D': 1}`

A degenerate run inside a USABLE response is worse than a 422: the document passes validation and ships corrupted text.


## Latency (1.7)

| set | n | mean | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| failures_included | 90 | 6.46s | 4.67s | 7.84s | 9.9s | 43.32s | 43.32s |
| failures_excluded | 89 | 6.14s | 4.67s | 7.51s | 9.83s | 43.32s | 43.32s |
| failures_only | 1 | 34.98s | 34.98s | 34.98s | 34.98s | 34.98s | 34.98s |

## Output contract — Arabic vs Persian letterforms (D8)

- Field values inspected: **230**
- Containing Arabic letterforms: **35 (15.22%)**
- Totals — yeh U+064A: 358 · kaf U+0643: 89 · alef maksura U+0649: 35

No ground truth is needed for this: it is a property of the API output alone.


## Digit systems in the output

| system | count | share |
|---|---|---|
| persian_indic | 654 | 21.85% |
| arabic_indic | 8 | 0.27% |
| ascii | 2331 | 77.88% |

The real corpus is written in Persian-Indic digits. Whether the model preserves them or silently transliterates to ASCII is an output-contract question that the synthetic corpus could not raise.


## Field population

| field | filled | % of usable responses |
|---|---|---|
| `sender` | 89 | 100.0% |
| `receiver` | 52 | 58.43% |
| `subject` | 19 | 21.35% |
| `body_text` | 18 | 20.22% |
| `contact_info` | 52 | 58.43% |

A field the model almost never fills is either genuinely absent from these letters or being dropped. Ground truth is required to tell those apart — this table says which fields to check first.


## Do failures correlate with image properties?

| property | mean, usable | mean, failed |
|---|---|---|
| width | 2289.8 | 2336.0 |
| height | 3204.3 | 3264.0 |
| bytes | 761539.6 | 897444.0 |
| estimated_dpi_a4 | 276.0 | 282.5 |

A large gap would point at a resolution or size threshold; a small one says failure is driven by content, not image quality.


## Phase 2a — the fixes and what each one bought

Everything above is the **baseline**. Below is the same corpus after the Phase 2a changes, one variable per experiment (`experiments.md`).

| configuration | usable-output, real dev | latency p50 |
|---|---|---|
| baseline | 14.81% [5.56, 25.93] | 22.20s |
| `MAX_TOKENS=4096` | 16.28% (n=43, partial) — **hypothesis falsified** | ~44s |
| `RESPONSE_FORMAT=json_schema` | no effect — the engine ignores it on image requests (**D32**) | — |
| `REPEAT_PENALTY=1.2` | **96.30%** [90.74, 100.00] | 4.44s |
| + `JSON_REPAIR=1` | 98.15% | 4.42s |
| + `MAX_RETRIES=1` | **100.00%** | 4.422s |

**The fix was not the one predicted.** The brief's leading hypothesis was token-budget exhaustion; it was falsified twice. The 422s were decoding degeneracy — the model looping on Persian-Indic digits (D23) or collapsing the whole letter into `sender` (D31). A repetition penalty of 1.2 removes it; 1.1 does nothing, so the effect is a threshold rather than a gradient.

**Reliability and latency were the same fix.** A looping request ran to the token cap every time, so removing the loop cut p50 latency 5×.


## Phase 2 exit criteria

| criterion | target | baseline | after fixes | status |
|---|---|---|---|---|
| usable-output on real dev | ≥99% | 98.15% | **100.00%** | **MET** |
| failures on real test | ≤1 | not scored | **1** of 36 (97.22% usable) | **MET** |
| no code path returns a silently empty success | 0 | 0 all-null 200s | all-null extractions now rejected at the endpoint with an explicit `empty_extraction` 422 | **MET** |

All-corpus baseline for reference: 98.89% usable over 90 documents.

**Phase 2 passes.** The three settings are now the defaults in `config.py`. The dev-to-test gap (100.00% → 97.22%) is the expected optimism of a configuration chosen on dev; the test number is the one to quote.


## What this says about the fix

- **Failure is content-driven, not image-quality-driven.** Successful and failed documents are indistinguishable on width, height, file size and estimated DPI (table above). Deskewing, upscaling and contrast work will not move this number.
- **Every failure is the same failure.** 100% are `http_422`, 100% raise `Unterminated string`, 100% report `finish_reason=length`, and every degenerate run is a Persian-Indic digit. This is one bug, not a taxonomy.
- **Failures are ~3.5× slower than successes** (22.2s vs 6.35s p50) because a looping request runs until the token cap. Fixing the loop is also the largest latency win available.
- **`contact_info` is never populated** (0 of 15) although every sampled letter carries a footer contact block — defect D29.
- **42% of digits come back as ASCII** on a corpus written in Persian-Indic digits. Transliteration or misreading cannot be separated without ground truth; either way it is an output-contract question for Phase 4.


## What got worse

Nothing regressed. This is the first measurement of the real corpus, and no model, prompt or config change has been made against it. The 16.67% usable-output rate is **not** a drop from the synthetic 83.33% — it is the first honest measurement on documents the service actually exists to read. The synthetic corpus was never representative: 10–20× lower resolution (D12), 6 effective templates (D14), and no long Persian-Indic digit runs, which is precisely the content that triggers the failure.

---

## Correction to how the headline above must be read (D41)

**98.89% is a true statement about JSON validity and a false impression of
extraction.** `usable` here means "at least one non-null field". `sender` is
non-null on 89 of 89 usable outputs, so it alone carries the entire number.

| | pre-fix (`repeat_penalty` 1.0) | shipped (`repeat_penalty` 1.2) |
|---|---:|---:|
| usable outputs | 15 / 90 | **89 / 90** |
| mean fields returned (of 5) | 3.80 | **2.58** |
| `sender` present | 100.0% | 100.0% |
| `receiver` present | 100.0% | 58.4% |
| `subject` present | 93.3% | **21.3%** |
| `body_text` present | 86.7% | **20.2%** |
| `contact_info` present | 0.0% | **58.4%** |
| documents with body extracted, of 90 | 13 (14.4%) | **18 (20.0%)** |

Read the last row rather than the first. The usable-output rate moved 16.7% →
98.9%, a 5.9× improvement. The number of documents from which the letter body
was actually recovered moved 13 → 18, a 1.4× improvement. Both are real; only one
of them is what a caller cares about.

**This is the accuracy cost D33 flagged as unverified, and it is now partly
visible without ground truth.** The repetition penalty stops the loop by making
the model stop early, and stopping early costs whole fields. JSON repair is not
the explanation: the service log records only 2 repair events across all 90
requests.

`contact_info` is the one unambiguous gain — 0% → 58.4%, which partially answers
**D29**.

**Phase 2's exit criterion is met as written and should be rewritten.** Every
future headline must be published as a pair: usable-output rate *and* mean fields
returned. A single number cannot distinguish a clean read from a `sender`-only
stub, and this corpus is now mostly `sender`-only stubs.
