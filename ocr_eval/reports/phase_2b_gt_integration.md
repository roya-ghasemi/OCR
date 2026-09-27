# Phase 2b — Dataset integration (§2.1) and GT validation (§2.2)

Run 2026-09-03. Scope: the 90 real administrative letters in `ocr_eval/dataset_ex`
and the 36 synthetic images in `ocr_eval/images`.

## Verdict

**§2.1 passes. §2.2 cannot run on the real corpus: the ground truth is not there.**

The handover said GT for the 90 real documents was available at
`ocr_eval/dataset_ex`. It is not. That directory contains exactly 90 `.JPG` files
and nothing else — no subdirectory, no `.json`/`.jsonl`/`.csv`/`.txt`/`.xlsx`, and
no archive anywhere under `E:\ghasemi`. Every image is byte-identical to the
2026-09-01 manifest lock, and all 90 manifest records still read
`"gt_exists": false`. Registered as **D35**.

Everything in this report that does not require real GT was completed.

## §2.1 — Dataset integration protocol

Verification was non-destructive: hashes recomputed and both corpus roots
re-enumerated, without rewriting `manifest.json` or `splits_v2.json`.

| check | result |
|---|---|
| `manifest.json` sha256 vs lock | `30bcbe75be2f7d9b…` **MATCH** |
| `splits_v2.json` sha256 vs lock | `d26202f31029f6ee…` **MATCH** |
| images re-hashed against manifest | **126/126 unchanged**, 0 changed, 0 missing |
| files on disk absent from manifest | 0 |
| real corpus | 90 records, `gt_exists` true on **0** |
| synthetic corpus | 36 records, `gt_exists` true on 36 |
| split assignment (real) | dev 54 / test 36 |

The corpus and the split are intact. No regeneration was performed and none was
needed — the split remains the one recorded on 2026-09-01.

## §2.2 — GT validation

### Real corpus

Not runnable. `ocr_eval/score_real.py --validate-only` exits 2 with
`ground_truth_real.jsonl does not exist`.

### Synthetic corpus — validated, and it has two structural limits

The 36 synthetic GT records were put through the full §2.2 protocol. They are
schema-clean, NFC-normalised, free of bidi controls and tatweel. Two findings
bound what any synthetic number can mean.

**D36 — the synthetic set cannot test the pipeline's headline failure mode.**

| property | documents affected |
|---|---|
| Persian-Indic digits (`۰-۹`) | **0 / 36** |
| Arabic-Indic digits (`٠-٩`) | 0 / 36 |
| ASCII digits | 3 / 36 |
| Arabic letterforms (`ي`, `ك`, `ة`) | **0 / 36** |
| harakat | 6 / 36 |
| not NFC / bidi / tatweel | 0 / 36 |

The real-corpus decoding loop (D23) is a loop *on Persian-Indic digits*. The
synthetic corpus contains none. Arabic→Persian letterform folding (D8) has no
synthetic instance either. So every figure in `results_v2.json` is silent on
exactly the two axes the real corpus fails on, and no synthetic experiment can
validate a fix for them.

**D37 — two of the five API fields are unscoreable on synthetic GT.**

| field | records with a value |
|---|---:|
| `sender` | 36 / 36 |
| `subject` | 36 / 36 |
| `body_text` | 36 / 36 |
| `contact_info` | **6 / 36** |
| `receiver` | **0 / 36** |

`receiver` carries no reference at all. This is the mechanism behind D22 — the
field where misassigned content actually lands has nothing to be scored against.
It also means D29 (`contact_info` null on 100% of usable real outputs) can be
neither confirmed nor refuted without real GT.

Additionally: 36 records contain only **18 distinct GT texts** (12 duplicate
groups), so the effective sample size is 18, not 36; and 18 of 36 carry orphan
segments — image text belonging to no field — which caps achievable recall.

## What was built to clear the blocker

| file | purpose |
|---|---|
| `ocr_eval/GT_SPEC.md` | the transcription contract: field definitions, `null` vs `""`, digit and letterform conventions, PII handling, partial-delivery rules |
| `ocr_eval/gt_real_template.jsonl` | 90 pre-filled rows — `filename`, `sha256`, `language_mode`, `split` populated, five fields `null` |
| `ocr_eval/score_real.py` | validate (§2.2) then score, in one command |

`score_real.py` refuses to run on a GT file that does not reconcile against the
hash-locked manifest, contains unknown filenames or duplicate rows, or is still a
blank template. It warns — but proceeds — on `""`-instead-of-`null`, missing rows,
placeholder text inside `fields`, non-NFC content, and duplicate GT bodies.
Verified against both rejection paths and, with a throwaway fixture, against the
full scoring path including dev/test slicing and bootstrap CIs.

**Partial GT is useful.** Rows left entirely `null` are treated as
not-yet-transcribed and excluded, with the count reported beside every headline.
20 documents is enough to answer the biggest open question — whether the
repetition penalty costs accuracy (D33, D39). 50 gives a usable interval.

## What got worse

Nothing regressed in the corpus: 126/126 images and both hash locks are
unchanged.

One artefact was found to be stale rather than broken — **D38**. The stored real
predictions (`predictions_real.jsonl`, 2026-09-01T14:19Z) carry
`repeat_penalty = 1.0`, `max_retries = 0`, `n_usable = 15/90`: they predate the
D33 and D30 fixes. Re-scoring them against GT, which is exactly what the
handover asked for, would have produced an accuracy baseline for a pipeline that
no longer exists. Caught before publication. The pre-fix file is preserved as
`predictions_real.prefix_rp10.jsonl` for A/B use and predictions were regenerated
under the current configuration.

## Hard gate

Reached: **ground truth materially absent — 100% of the real corpus, not the 10%
threshold in the brief.** Phases 4, 5 and 6 have accuracy-based exit criteria and
cannot be evaluated. Work that does not require GT continued; see
`phase_7_ci_gate.md`.
