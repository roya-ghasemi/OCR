# Persian / Bilingual OCR service — remediation status

Updated 2026-09-03. Scope: `E:\ghasemi\ocr`. Endpoint under evaluation: `POST /ocr`
on `main.py` — the shipped path, per the endpoint-parity resolution in Phase 0.

---

## The blocker, stated first

**Ground truth for the 90 real administrative letters does not exist.**

The handover said it was at `ocr_eval/dataset_ex`. That directory contains
exactly 90 `.JPG` files and nothing else — no subdirectory, no
`.json`/`.jsonl`/`.csv`/`.txt`, no archive anywhere under `E:\ghasemi`. All 90
images are byte-identical to the 2026-09-01 manifest lock and all 90 manifest
records read `"gt_exists": false`. Registered as **D35**.

Consequence: **no CER, WER or field-accuracy number exists for the real corpus,
and none is quoted anywhere in this repository.** Phases 4, 5 and 6 have
accuracy-based exit criteria and cannot be closed.

To clear it: fill `ocr_eval/gt_real_template.jsonl` (90 pre-filled rows) per
`ocr_eval/GT_SPEC.md`, save as `ocr_eval/ground_truth_real.jsonl`, then

```bash
venv312\Scripts\python.exe ocr_eval/score_real.py --validate-only
venv312\Scripts\python.exe ocr_eval/score_real.py
```

Partial delivery works. **20 documents answers the single most important open
question — whether the repetition penalty costs accuracy. 50 gives a usable
confidence interval.** No re-inference is needed: predictions for both decoding
arms are already stored.

---

## What was fixed, measured

| defect | before | after | evidence |
|---|---|---|---|
| D3/D23 decoding loop | 15/90 usable (16.7%) | **89/90 (98.9%)** | `repeat_penalty=1.2`; dev A/B 14.81% → 96.30%, non-overlapping CIs |
| D30 JSON repair | — | +1.85 pp, zero latency cost | 16 unit tests built from real failure shapes |
| D34 residual failures | 2 of 54 dev | 0 | repair + one bounded retry |
| latency p50 | 22.20s | 4.42s | same fix — a looping request ran to the token cap |
| D4 silent empty success | returned 200 | rejected at the endpoint | Phase 0 guard tests |
| D40 structured decoding | believed impossible (D32) | **works via `grammar`** | 10/10 parsed on real letters |

Reliability is genuinely fixed. Test split holds at 97.22% usable.

---

## What is still broken

| defect | measurement |
|---|---|
| **D41** usable-output rate is carried by one field | 89/90 usable, but **mean 2.58 of 5 fields**; `body_text` present on **20.2%**; no document returns all five |
| **D42** digit misreading | **12 of 65** national-ID-shaped strings pass mod-11 (18.5%); the one IBAN fails mod-97 |
| **D43** Arabic letterforms reach the caller | 34.8% of usable outputs, all five fields |
| **D44** schema collapse | 20.2% of usable outputs; longest `sender` observed **2116 characters** |
| **D45** `contact_info` under grammar | 58.4% → **4.5%** — blocks cutover to `ocr_pipeline` |
| **D46** structured decoding does not fix digits | checksum pass 18.5% → 19.6%, unchanged |
| **D36** synthetic corpus cannot test the failure mode | **zero** Persian-Indic digits and zero Arabic letterforms across all 36 GT records |
| **D37** two of five fields unscoreable on synthetic GT | `receiver` 0/36, `contact_info` 6/36 |

---

## What got worse

**The repetition penalty bought parseability with content.**

| | pre-fix `rp=1.0` | shipped `rp=1.2` |
|---|---:|---:|
| usable outputs | 15/90 | **89/90** |
| mean fields (of 5) | 3.80 | **2.58** |
| `body_text` present | 86.7% | **20.2%** |
| `subject` present | 93.3% | **21.3%** |
| documents with body extracted, of 90 | 13 | **18** |

The usable-output rate moved 5.9×. The number of documents from which the letter
body was actually recovered moved 1.4×. Both are true; only the second is what a
caller cares about. JSON repair is not the explanation — the service log records
2 repair events across 90 requests. This is the accuracy cost D33 flagged as
unverified, now partly visible without ground truth. `contact_info` 0% → 58.4% is
the one clean gain.

A false-green was also found and fixed **in this session's own tooling**: the CI
gate's Arabic-letterform check read a key `letterform_conformance` does not
return and reported 0% on a corpus that is 34.8%. A check that cannot fail is
worse than no check.

---

## Phase status

| phase | status |
|---|---|
| 0 — gates, inventory, endpoint parity | **complete** |
| 1 — measurement harness | **complete** |
| 2 — reliability | **complete**, exit met (97.22% test) — but read D41 before quoting it |
| 2b — dataset integration §2.1 / GT validation §2.2 | §2.1 **passes**; §2.2 **blocked** on real, complete on synthetic |
| 3 — field assignment | **complete GT-free**; the brief's premise is disproved — misassignment is 14.46% of bilingual CER, 1.43% Persian-only |
| 4 — accuracy | **blocked (D35)**; partial GT-free evidence via checksums (D42) |
| 5 — preprocessing / model fitness | **blocked (D35)** — exit criteria are accuracy-based |
| 6 — decoding strategy | **complete GT-free**; full-corpus A/B run, cutover declined on D45 |
| 7 — productionisation | regression gate **complete**; accuracy thresholds **blocked**; two fixes **not wired in** |

---

## Fitness

**Not fit for production. On the 90 real documents the service returns
well-formed JSON 98.9% of the time and the actual letter body only 20.2% of the
time, misreads four out of five checkable identifiers, and its accuracy has never
been measured because the ground truth for those 90 documents does not exist.**
