# Ground-truth audit — Phase 0.8

Every image below was **opened and read by eye**, then compared character-by-character
against its `ground_truth.jsonl` entry. Verdict is `correct` (GT matches the pixels) or
`gt_error` (it does not), with the specific correction.

Scope: all 15 images in the current worst-CER table, plus 2 controls. The worst-15 is
`synthetic_fa_{002,005,010}` and `synthetic_bilingual_{023,029,035}` (the six no-output
failures, all scored 100% CER) followed by the nine worst *readings*.

---

## Headline

**Two systematic ground-truth defects were found. Both come from the renderer, not from
the transcription.** `generate_dataset.py` writes the ground truth from the *logical*
string it intended to draw, and separately draws the page through
`arabic_reshaper.reshape()` → `bidi.algorithm.get_display()`. Those two pipelines do not
agree, and the ground truth records the intent rather than the pixels.

| Defect | Affects | Direction |
|---|---|---|
| **G1** — hyphen/space-separated numeric groups are drawn in reversed group order | 18 GT lines across 12 of 36 images | GT is wrong; the page shows something else |
| **G2** — `arabic_reshaper` silently drops tanween `U+064B` | 6 characters across 6 images | GT is wrong; the page shows something else |

Together, **18 of 36 images carry at least one ground-truth defect.**

Counted strictly: a line is a defect only when an *alphanumeric token's own character
order* differs between the ground truth and the pixels. Lines where only a neutral
character relocates around a Latin run — `Sepehr Digital Data Co.` drawn with the period
on the other side, `کد رهگیری: TRK-99A45B` drawn with the colon on the other side, `25%`
drawn as `%25` — are **correct** bidi behaviour and are *not* counted. A human
transcribing those lines would still write them the way the ground truth has them.

Neither defect was known when `REPORT.md` was published, and both contaminate the exact
metric the report leads with (bilingual Latin/digit CER, 32.85%).

---

## G1 — the renderer reverses hyphen-separated digit groups

Reproduced directly against the rendering functions the generator uses:

```
logical (recorded as ground truth)   →  visual (actually drawn on the page)
2026-08-27                           →  27-08-2026     ← group order reversed
1405-06-05                           →  05-06-1405     ← group order reversed
+98 21 8899 1234                     →  1234 8899 21 98+  ← group order reversed
25%                                  →  %25            (neutral sign moved — benign, not counted)
2026/07/13                           →  2026/07/13     unchanged
INV-2026-0093                        →  INV-2026-0093  unchanged
EMP-0451                             →  EMP-0451       unchanged
TRK-99A45B                           →  TRK-99A45B     unchanged
version 3.4.1                        →  version 3.4.1  unchanged
IR820170000000123456789              →  IR8201700...   unchanged
```

The split is explainable: `python-bidi` mishandles the European-Separator rule, so a run
of digits joined by `-` is broken into separate number runs which are then ordered
right-to-left. A group containing Latin *letters* (`INV-`, `EMP-`, `TRK-`, `version`) is a
strong-L run and survives intact. Slashes are handled correctly.

### Why this matters more than its size suggests

The model applies a **group reversal of its own** to numeric dates. Where the renderer had
already reversed them, the two cancel and the model scores as correct. Where the renderer
did not, the model's reversal is exposed:

| Image | GT | Printed on page | Model emitted | Scored |
|---|---|---|---|---|
| bilingual_020 | `2026-08-27` | `27-08-2026` | `2026-08-27` | ✅ correct |
| bilingual_026 | `2026-08-27` | `27-08-2026` | `2026-08-27` | ✅ correct |
| bilingual_032 | `2026-08-27` | `27-08-2026` | `2026-08-27` | ✅ correct |
| bilingual_022 | `1405-06-05` | `05-06-1405` | `1405-06-05` | ✅ correct |
| bilingual_034 | `1405-06-05` | `05-06-1405` | `1405-06-05` | ✅ correct |
| bilingual_021 | `2026/07/13` | `2026/07/13` | `13/07/2026` | ❌ wrong |
| bilingual_033 | `2026/07/13` | `2026/07/13` | `13/07/2026` | ❌ wrong |

The model reversed the group order in **7 of 7** cases where it emitted a date at all. It
scored correct on 5 of them purely because the renderer had made the same mistake first.

**Consequence: the current dataset cannot measure digit ordering.** Any Phase 4 number for
"digit-order integrity" computed on this data is meaningless until the generator is fixed.
This also invalidates the D7 bucketing question — reversal is not "one root cause split
across two buckets", it is a deterministic model behaviour that the harness is
accidentally cancelling out.

The other 12 of 18 bilingual images emit **no date at all** — the schema has no `date`
field (see `ARCHITECTURE.md` §3), so there is nowhere to put it.

Re-scoring against a pixel-faithful ground truth moves the numbers the *wrong* way, which
is the proof that the model is emitting logical order rather than reading order:

| | CER all | CER bilingual | bil Persian-seg | bil Latin/digit-seg |
|---|--:|--:|--:|--:|
| GT as recorded (published) | 17.04% | 25.59% | 16.17% | 32.85% |
| GT as actually printed | 17.79% | 26.85% | 16.17% | 36.22% |

So the honest reading is: **the published 32.85% Latin/digit CER is optimistic, not
pessimistic.** Against what is really on the page it is 36.22%.

---

## G2 — the renderer drops tanween

`arabic_reshaper.reshape("احتراماً")` returns `U+FE8D U+FEA3 U+FE98 U+FEAE U+FE8D U+FEE3
U+FE8E` — the tanween `U+064B` is **gone**. The page shows `احتراما`; the ground truth
records `احتراماً`.

Affects 6 characters (all `U+064B`) across 6 images. Small in CER terms, but it means the
model is penalised for a correct reading:

| Image | GT | On the page | Model emitted | Verdict |
|---|---|---|---|---|
| fa_001 / fa_007 / fa_013 | `احتراماً` | `احتراما` | `احتراما` | model **correct**, scored wrong |
| bilingual_024 / bilingual_030 | `احتراماً` | `احتراما` | `احتمالا` | model genuinely wrong |
| bilingual_036 | `احتراماً` | `احتراما` | `احتمالاً` | model wrong **and** invents a tanween that is not on the page |

This directly settles part of D1: the `احتراماً → احتمالا` substitution is real, but it
occurs on **one letter content rendered 3 times, bilingual only** — the Persian-only
renders of the same word are read correctly. It is n=1, not a pattern.

---

## Per-image verdicts

### The six no-output failures (all scored 100% CER)

| # | Image | Verdict | Notes |
|---|---|---|---|
| 1 | `synthetic_fa_002.png` | **correct** | Opened and read. Clean, high-contrast, arial 28, +1.5° rotation. Every one of the 9 GT lines is legible and matches. The empty-200 is **not** an input-quality failure. |
| 2 | `synthetic_fa_005.png` | **correct** | Opened and read. tahoma 28, no rotation — the cleanest render in the set. GT matches exactly. Empty-200 is model abstention. |
| 3 | `synthetic_fa_010.png` | **correct** | Opened and read. arial 26, +1.5°. Signature `رویا قاسمی فر` clearly printed. GT matches. |
| 4 | `synthetic_bilingual_023.png` | **gt_error (G1)** | Opened and read. Page shows `تاریخ: 27-08-2026`; GT says `تاریخ: 2026-08-27`. Correction: GT should read `27-08-2026`. All other lines match, incl. the IBAN `IR820170000000123456789` and `TRK-99A45B`. |
| 5 | `synthetic_bilingual_029.png` | **gt_error (G1)** | Same content as 023 with a different sender. Same date defect, same correction. |
| 6 | `synthetic_bilingual_035.png` | **gt_error (G1)** | Same content as 023. Same date defect, same correction. |

**All three empty-200 images are clean and fully legible.** This is confirmed by eye, not
inferred. Failure mode B is model abstention, and Phase 2b should not spend time on image
quality.

**All three HTTP-422 images are the same letter content** (the `IBAN` + `TRK-` letter),
differing only in the sender line. n=1 defect observed three times.

### The nine worst readings

| # | Image | Verdict | Notes |
|---|---|---|---|
| 7 | `synthetic_bilingual_024.png` | **gt_error (G2)** | Opened and read. Page shows `احتراما` (no tanween) and `%25`; GT records `احتراماً` and `25%`. Date `2026/07/13` and `version 3.4.1` are drawn correctly and match GT. |
| 8 | `synthetic_bilingual_036.png` | **gt_error (G2)** | Same content as 024, same corrections. |
| 9 | `synthetic_bilingual_028.png` | **gt_error (G1)** | `1405-06-05` printed as `05-06-1405`. |
| 10 | `synthetic_bilingual_025.png` | **gt_error (G1)** | `1405-06-05` printed as `05-06-1405`. |
| 11 | `synthetic_bilingual_020.png` | **gt_error (G1)** | `2026-08-27` printed as `27-08-2026`. |
| 12 | `synthetic_bilingual_022.png` | **gt_error (G1)** | `1405-06-05` printed as `05-06-1405`. |
| 13 | `synthetic_bilingual_019.png` | **gt_error (G1)** | `1405-06-05` printed as `05-06-1405`. |
| 14 | `synthetic_bilingual_026.png` | **gt_error (G1)** | `2026-08-27` printed as `27-08-2026`. |
| 15 | `synthetic_bilingual_032.png` | **gt_error (G1)** | `2026-08-27` printed as `27-08-2026`. |

### Controls (not in the worst-15)

| # | Image | Verdict | Notes |
|---|---|---|---|
| 16 | `synthetic_fa_004.png` | **correct** | Opened and read. 9 lines, arial. GT matches the pixels exactly, including the signature `مدیر عامل / رویا قاسمی فر`. No numeric content, so neither G1 nor G2 applies. |
| 17 | `synthetic_bilingual_021.png` | **correct** | `2026/07/13` is drawn in the correct order and GT matches. Confirms G1 is specific to hyphen separators. |

---

## What fraction of the current error is ground-truth error?

| Question | Answer |
|---|--:|
| Images audited by eye | 17 of 36 |
| Images with a GT defect | 11 of 17 audited · **18 of 36** corpus-wide |
| GT lines affected by G1 | 18 of 306 (5.9%), across 12 images |
| GT characters affected by G2 | 6 of 9,657 (0.06%), across 6 images |
| Effect on **overall** CER | **−0.75 pt** — the published 17.04% would be 17.79% against pixel-faithful GT |
| Effect on **bilingual Latin/digit** CER | **−3.37 pt** — the published 32.85% would be 36.22% |
| Effect on **Persian-only** CER | **0.00 pt** — no numeric content; the 4.49% figure is unaffected |

**Direction matters: the ground-truth error is making the system look *better* than it is,
not worse.** Every "GT is wrong so our real score is better" intuition is backwards here.

Nothing in this audit was corrected in `ground_truth.jsonl`. Per operating rule 4, ground
truth is not edited to move a metric; the defect is in the **generator**, and the fix
belongs in Phase 5 (regenerate with a correct bidi implementation, or mark numeric runs
with explicit LRM/RLM so the drawn order is the recorded order). Until then, no digit-order
or Latin-segment metric from this dataset should be quoted.

## Corrections applied

None. See `gt_corrections.md` — it is intentionally empty at the end of Phase 0.
