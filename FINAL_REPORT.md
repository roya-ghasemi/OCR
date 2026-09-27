# Persian / Bilingual OCR service — remediation status

Updated **2026-09-22**. Scope `E:\ghasemi\ocr`. Endpoint under evaluation: the
`ocr_service` pipeline (Qwen2.5-VL primary). The previous edition of this file
(2026-09-03, kept as `FINAL_REPORT.2026-09-03.md`) predates the ground truth, the
model bake-off and the numeric work, and its headline numbers are superseded.

---

## Run it

```bash
venv312\Scripts\python.exe -m uvicorn ocr_service.api:app --host 127.0.0.1 --port 8010
```

```bash
curl -F "file=@ocr_eval/dataset_ex/some_letter.JPG" http://127.0.0.1:8010/ocr
```

The engine is spawned by the service (~40 s to load, ~8.5 GB VRAM: nothing else may
hold the GPU). `GET /health` reports models, Tesseract and queue. Read
`numeric_fields[].confidence` per number — see the live run below for why.

---

## Where the project stands, stated first

Ground truth now exists (84 of 90 documents transcribed, `ground_truth_real_fixed_v2.jsonl`),
the model has been chosen on measured evidence, and the numeric defect that made the
service unusable has been fixed and measured on a held-out split.

**Test split (n=33, 268 GT numbers) — E17 frozen, then E18 after the amount fix:**

| | before the numeric layer | **after** |
|---|---:|---:|
| numeric atom recall | 23.9% | **63.8%** |
| numeric atom precision | 31.1% | **67.1%** |
| **whole grouped numbers intact** (amounts, dates, phone lists) | 6.5% | **30.6%** |
| body CER | — | **32.0%** |
| usable output | — | **100.0%** |
| latency p50 / p95 | — | **8.2 / 10.0 s** |

Updated 2026-09-22 after E18 (amount-integrity fix, D63). The test split has been
scored twice — once frozen (E17) and once after that fix (E18, disclosed second
look) — so it is no longer a fully unbiased estimate.

Dev (n=51) is higher — 71.2% recall — and the gap is understood, not hand-waved: see
"What is still broken".

---

## What was fixed, measured

| defect | before | after | evidence |
|---|---|---|---|
| D3/D23 decoding loop | 15/90 usable (16.7%) | 89/90, now **100%** on both splits | `repeat_penalty=1.2`, then grammar + DRY |
| D49 model fitness | coreOCR, body CER 110.6% | **Qwen2.5-VL, 31.9% test** | E14 bake-off, identical pipeline |
| **D53/D60 digits are generated, not read** | VLM 0/12 on Persian digit cards; v1 reader 48% recall at **10%** precision | **v2 CNN reader 74.6% test / 79.0% dev**, and end-to-end **23.9% → 53.4%** | E15, E17 |
| D52 session-dependent bodies | 7/10 documents unstable | **1.000 across processes, both `cache_prompt` arms** | E16 — refutes the prompt-cache hypothesis; it was coreOCR's cap-running text |
| D61 `cache_prompt` never sent | provenance said `false`, engine ran `true` | sent, with a test asserting on the **request** | E16 |
| D62 `needs_review` false green | flagged 1/51 while 34/51 held a wrong number | flags `low` **or `unverified`**; contract says it is triage, not a gate | E15 |
| D43 Arabic letterforms | 34.8% of outputs | folded at the API boundary | `ocr_service.pipeline` |

---

## What is still broken

| defect | measurement |
|---|---|
| **Whole amounts are usually not exact** | **30.6%** of grouped numbers (amounts, dates, phone lists) arrive exactly right on test. Atom recall is 63.8% because most *groups* of a long number are correct — but a payment amount is only useful if every digit is. 27 of 33 test documents contain at least one wrong number. |
| **D63 residual** | The truncation bug is fixed (trailing zeros no longer dropped; grouped numbers injected whole; per-chunk floor removed) and gained +10.4 pp of atom recall on test. What remains is ordinary digit error spread through long numbers, plus bidi group-order ambiguity — not a structural bug with a known fix. |
| **D62 no reliable per-document gate** | `high` is 82.6% precise on test, `corrected` 68.8%, `added` 55.6%, `unverified` **20.0%**. The review flag catches 14 of 26 error documents. A caller must read per-number `confidence`. |
| **D58 GT errors** | a footer number is mis-transcribed across ~11 documents; `_11`'s national ID is wrong. Not corrected (GT is never edited from a reader's output) — costs ~3 pp of measured recall. |
| **D59 handwriting** | some GT numbers exist only in pen. No reader in this stack can read them. |
| **`subject` is not OCR** | CER 83.6% test — 45 of 51 dev subjects were written by the annotator, so the model is being scored on summarisation (E12). |
| **D41 field coverage** | body CER 31.9% means roughly a third of the letter body's characters are still wrong. |

---

## Live service run — verified end to end, 2026-09-22

Not a benchmark harness: the actual FastAPI service, started as a caller would start
it, with one real letter posted to `POST /ocr`.

```bash
venv312\Scripts\python.exe -m uvicorn ocr_service.api:app --host 127.0.0.1 --port 8010
curl -F "file=@ocr_eval/dataset_ex/<letter>.JPG" http://127.0.0.1:8010/ocr
```

`GET /health` → `status: ok`; primary `Qwen2.5-VL-7B-Instruct-Q4_K_M` on
`127.0.0.1:18236`, grammar on; Tesseract 5.4.0 present; Redis unreachable and
correctly reported (`async_mode: false`, so the sync path is unaffected).

`POST /ocr` on the letter whose amount was being truncated: **HTTP 200 in 10.0 s**
(preprocess 0.28 s, model 9.26 s, numeric layer 0.46 s), 10 numbers returned —
2 `high`, 5 `corrected`, 2 `unverified`, 1 `added`, 0 `low`, `needs_review: true`.

**The D63 fix is visible in production.** The model emitted the contract amount as
`۳۲۱/۰۰۰/۰۰۱۳۲۱`; the service returned `resolved: ۳۲۱/۰۰۰/۰۰۰` — correct, with its
separators and all three trailing zeros. The شناسه ملی the model omitted entirely was
read off the page and `added` (`۱۴۰۰۲۰۰۴۱۰۵`). Two IBAN/reference numbers were
corrected from the model's invented digits to the printed ones.

**Two caller-facing weaknesses this single request also exposed, both consistent with
the measured numbers above and neither newly broken:**

* **The same amount was returned twice, once right and once wrong** — `۳۲۱/۰۰۰/۰۰۰`
  (`corrected`) in `subject`, and `۳۲۱,۰۰۰,۰۰۱,۳۲۱` (`unverified`) in `body_text`.
  The model routed one copy into a field where nothing on the page aligned with it, so
  reconciliation could not fix it and correctly refused to guess. A caller reading
  `body_text` alone gets the wrong figure. **Consuming `numeric_fields` and preferring
  `high`/`corrected` over `unverified` is not optional — it is how the service is
  meant to be read.**
* **Field routing is still the model's weak point** (D41/D44): the amount belongs in
  `body_text`, not `subject`. The numeric layer fixes digits, not placement.

Artefacts: `ocr_eval/experiments/digits_0922/live_service.log`.

---

## Fitness

**Not fit for unattended production. Fit for a review-assisted workflow.** The service
runs, is stable, and answers in ~8-10 s per page (live run verified 2026-09-22).

The service now returns well-formed JSON on 100% of documents, reads the letter body
at 31.9% character error, and gets **just over half** of the printed numbers exactly
right — up from a quarter. Every number carries a per-number confidence, and the one
class that is safe to trust automatically (`high`, 82.6% precise) covers only ~14% of
numbers. That is a usable assistant for an operator who checks the figures; it is not
a system that should post a payment amount unattended.

**Phase 7 (productionisation, scale-out) remains paused by direction.** The correctness
bar for unattended operation is not met, and the throughput question is unchanged: one
`llama-server` on one GPU at ~8.4 s per page.

**For a financial workflow the governing number is 30.6%, not 63.8%.** An amount is
right or it is wrong; two thirds of grouped numbers still carry at least one bad
digit. The per-number `confidence` field is what makes the service usable: `high` is
83% precise on test but covers only 13% of numbers.

**Next step, in order of value:** (1) raise whole-number exactness — the reader's
per-glyph accuracy on office fonts is 93.7%, which over a 10-digit number compounds
to ~52%, so the gain has to come from a stronger glyph model or a second reading pass,
not from more reconciliation logic; (2) resolve the bidi group-order ambiguity with a
layout rule rather than a guess; (3) only then consider Phase 7.
