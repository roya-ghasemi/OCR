# The Dehkhoda correction stage is removed from the pipeline

**Removed:** 2026-09-01, Phase 0.3. **Justified by:** experiment E005 in
`ocr_eval/experiments.md`; raw numbers in `phase0_ablation.json`, every individual
token change in `phase0_changes.tsv`.

## What was removed

`main.py` no longer contains the `_get_corrector()` lazy loader or the
`POST /ocr/corrected` endpoint. **No runtime code path now loads
`dehkhoda.db` or imports `corrector.py` / `corrector_v2.py`.** `POST /ocr` is
unchanged — it never called the corrector in the first place (see below).

The code in this directory is kept as the **experimental record** that justifies the
removal, not as runtime code. `ocr_eval/test_phase0_guards.py` fails the build if any
lexicon stage is wired back into the service.

## Why — and why it is not the reason the brief assumed

The remediation brief's hypothesis D1 was that the Dehkhoda stage was net-harmful and had
corrupted the published baseline, producing the semantic substitutions in `REPORT.md`
(`احتراماً → احتمالا`, `اعلام → اعلان`, `فرمایید → فرمانیه`, the `پیشاپیش` variants).

**That hypothesis is false, on three independent counts:**

1. **The stage was never in the measured path.** `ocr_eval/run_eval.py` posts to `/ocr`.
   The corrector only ever ran on `/ocr/corrected`, which the harness never called. The
   published 17.04% CER baseline was produced with no dictionary anywhere in the pipeline.

2. **The substitutions are present in the raw model output.** Counted directly in
   `predictions.jsonl` — which is unmodified `/ocr` output — before any correction:

   | Substitution | in raw `/ocr` output | in ground truth |
   |---|--:|--:|
   | `احتمالا` | 3 | 0 |
   | `اعلان` | 3 | 0 |
   | `فرمانیه` | 4 | 0 |
   | `پیشایپیش` | 3 | 0 |
   | `پیشابیش` | 2 | 0 |
   | `پیشایش` | 1 | 0 |
   | `پیشایی` | 5 | 0 |

   These are vision-model reading errors. `پیشاپیش` in particular is a repeated
   tooth-skeleton word (پ-ی-ش-ا-پ-ی-ش) that the model gets wrong in **12 of 12**
   occurrences — a textbook Arabic-script visual confusion, not a dictionary snap.

3. **The shipped configuration cannot produce any of them.** The default `CorrectorV2`
   arm (`v2d`) permits *only* doubled-letter deletion with a single unambiguous candidate.
   None of the watchlist substitutions is reachable by that edit. Measured: the watchlist
   counts are **byte-identical** with the stage on and off.

## The A/B that was actually run

The corrector is pure post-processing over the model's JSON — it never touches the image,
the prompt, or decoding. So the honest A/B applies and does not apply it to the **same
frozen `predictions.jsonl`**. That is stronger than re-running inference twice: it removes
inference variance entirely, so every delta is attributable to the corrector alone.

| Arm | tokens changed | CER all | Δ | CER bilingual | CER Persian-only | WER | CER dev | CER test |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **off** (what `/ocr` ships) | 0 | 17.04% | — | 25.59% | 4.49% | 22.38% | 17.15% | 16.83% |
| **v2d** (what `/ocr/corrected` shipped) | **2** | 17.01% | **−0.03** | 25.54% | 4.49% | 22.23% | 17.11% | 16.83% |
| v2b (+ confusable subs, unambiguous) | 17 | 17.13% | +0.09 | 25.54% | 4.77% | 22.76% | 17.24% | 16.90% |
| v2a (+ confusable subs, ambiguous OK) | 30 | 17.28% | +0.24 | 25.67% | 4.96% | 23.66% | 17.44% | 16.98% |

Usable-output rate is 30/36 in every arm — a post-processor cannot change it.

## Hand audit — all 49 changes, not a sample

Across all three arms there are only 49 changes of 8 distinct kinds, so every one was
labelled rather than sampling 30. Labels are verified against `ground_truth.jsonl`.

| Change | ×  | Arms | Label | Evidence |
|---|--:|---|---|---|
| `استحضاار → استحضار` | 2 | v2d, v2b, v2a | **improved** | GT contains `استحضار` ×12, `استحضاار` ×0 |
| `پیشابیش → پیشاپیش` | 1 | v2b, v2a | **improved** | GT contains `پیشاپیش` ×12 |
| `پروژه → پروره` | 11 | v2b, v2a | **damaged** | `پروژه` is correct and appears in GT ×13; `پروره` ×0 |
| `آید. → آبد.` | 12 | v2a | **damaged** | `آید` correct, in GT ×12; `آبد` ×0 |
| `پیشایی → پیشانی` | 1 | v2b, v2a | **damaged** | neither is GT (`پیشاپیش`); turns visible garble into a confident wrong word |
| `سهامه → شهامه` | 1 | v2b, v2a | **damaged** | GT is `سه‌ماهه`; neither form is in GT |
| `رضاى → رضاب` | 1 | v2b, v2a | **damaged** | `رضا` is a person's name (GT ×15); the stage edited a proper noun |
| `پروزه → بروزه` | 1 | v2a | **damaged** | misread of `پروژه`; changed to a different wrong word |

Totals: **2 improved, 0 unchanged-meaning, 47 damaged.** In the shipped `v2d` arm
specifically: **2 improved, 0 damaged** — but those 2 are the *only* thing it ever did.

## The actual grounds for removal

The stage as shipped is not harmful. It is **worthless, and structurally unsafe to keep**:

- It bought **2 token fixes across the entire corpus** for a 311 MB SQLite dependency and
  a 154,396-word classical lexicon. The −0.03 pt CER delta is far inside the noise of a
  36-image / 18-distinct-text set.
- Every configuration that does more than that **damages the output**, and the damage is
  exactly the failure mode the brief predicted — a literary lexicon snapping *correctly
  read* modern administrative vocabulary onto classical neighbours (`پروژه → پروره`,
  `دیجیتال → دیژیتال`) and editing proper nouns (`رضاى → رضاب`).
- It sits one boolean flag away from the harmful arms. `allow_substitution=True` is a
  one-character change, and nothing in the codebase prevented it.

So the brief's *conclusion* is right and its *reasoning* is wrong, and the distinction
matters: removing this stage does not fix the semantic substitutions. Those are model
reading errors and remain Phase 4 work.

## Policy going forward

No lexicon, dictionary, or spell-correction stage may be reintroduced unless it:

- **(a)** uses a *domain* lexicon built from the actual document corpus, not a literary
  dictionary;
- **(b)** is restricted to Persian-script running text and never applied to names, codes,
  numbers, dates or Latin tokens;
- **(c)** is frequency-weighted rather than pure edit distance; and
- **(d)** passes a pre-registered A/B on the dev split showing a CER improvement whose
  confidence interval excludes zero.

Enforced by `ocr_eval/test_phase0_guards.py::test_no_lexicon_stage_in_runtime`.

## Not deleted — needs your decision

**This project is not a git repository** (`git rev-parse` fails; there is no `.git`). The
brief's instruction to "keep it only in git history" cannot be carried out — deleting these
files would destroy the only copy of the evidence above.

Left in place, pending your call:

| Path | Size | Note |
|---|--:|---|
| `dehkhoda/dehkhoda.db` | **311 MB** | Regenerable from `Dehkhoda-SQL-master/` via `build_db.py`. Safe to delete; nothing at runtime opens it. |
| `Dehkhoda-SQL-master/` | ~90 MB | The upstream source dumps. |
| `dehkhoda/*.py`, `*.json`, `*.tsv`, `*.md` | small | The experimental record. Recommend keeping. |

Recommended: `git init` first so the history exists, then delete `dehkhoda.db`. I have not
deleted anything.
