# Brief: Closing the Open Defects on the Persian/Bilingual OCR Service

> For a coding agent working inside `E:\ghasemi\ocr`.
> This brief supersedes the earlier 8-phase remediation prompt. That prompt described
> a greenfield project; this one describes the project as it actually stands.

---

## 0. Read these before you touch anything

In this order. Do not plan, refactor, or run an experiment until you have:

1. `FINAL_REPORT.md` — current status, what was fixed, what regressed, fitness verdict.
2. `ocr_eval/error_register.md` — the single source of truth for every defect (D1–D46).
3. `ocr_eval/experiments.md` — append-only log of every run, E000 onward.
4. `ARCHITECTURE.md` — request lifecycle, real output schema, model pin.
5. `ocr_eval/GT_SPEC.md` — the ground-truth spec for the real corpus.

If anything in this brief contradicts those files, **the files win** — and say so in
your first report. This brief was written by a human from memory and may be stale.

---

## 1. Ground truth about the system (corrects the previous brief)

**Output schema — five fields, and `date` is not one of them:**

```python
class LetterExtraction(BaseModel):
    sender:       Optional[str] = None
    receiver:     Optional[str] = None
    subject:      Optional[str] = None
    body_text:    Optional[str] = None
    contact_info: Optional[str] = None
```

The previous brief listed `date` as a field and asked you to "fix date dropping."
There is nowhere for a date to go. If dates matter to the caller, that is a
**schema change request**, not a model bug — raise it, do not silently invent it.

**Model — pinned, and it is not a Persian model:**

| | |
|---|---|
| Weights | `coreOCR-7B-050325-preview.Q4_K_S.gguf` (Q4_K_S, 4,457,769,440 B) |
| Base | `Qwen/Qwen2-VL-7B-Instruct` |
| Runtime | llama.cpp `llama-server` 1 (fe2adf0), cuda12-avx2-2.28.2 |
| Persian on the model card | **named nowhere** (D10, open risk) |

**Serving path:** `main.py` runs its own inline pipeline. The `ocr_pipeline/` package
exists, is unit-tested, and **is not mounted**. This matters — see Track A.

**Config:** every knob is already centralised in `config.py` behind env vars
(`REPEAT_PENALTY=1.2`, `MAX_TOKENS=2048`, `TEMPERATURE=0.0`, `JSON_REPAIR=1`,
`MAX_RETRIES=1`, `IMAGE_MIN_TOKENS=1024`, `RESIZE_MAX_EDGE=0`, `RESPONSE_FORMAT=off`).
Do not re-do this. `config.provenance()` is what you record in `experiments.md`.

---

## 2. Current measured state — quote these, not the old baseline

On the 90 real documents in `ocr_eval/dataset_ex`:

| metric | value |
|---|---|
| usable output (well-formed JSON) | **98.9%** (89/90) |
| mean fields populated, of 5 | **2.58** |
| `body_text` present | **20.2%** |
| `subject` present | **21.3%** |
| latency p50 | 4.42 s (was 22.20 s) |
| national-ID strings passing mod-11 | **12 of 65 (18.5%)** |
| outputs containing Arabic letterforms | **34.8%** |
| outputs with schema collapse | **20.2%** |
| **CER / WER / field accuracy** | **does not exist — see Gate 0** |

The old brief's headline figures (83.33% usable, 17.04% CER, 3.15 s) come from E000
under flat-blob scoring that the harness rebuild already invalidated. Do not quote them.

**Do not use usable-output rate as a headline metric on its own.** It moved 5.9× while
the number of documents from which the letter body was actually recovered moved 1.4×.
Report it only paired with mean-fields-populated and `body_text` presence.

---

## 3. Closed — do not reopen without new evidence

| claim | verdict |
|---|---|
| `Roya Ghasemi` is a hallucination or prompt leak | **REFUTED** (E005). It is in the generator at `generate_dataset.py:92`, in the pixels of `synthetic_fa_004.png`, and in the GT of every image that predicts it. Pred-only cases = 0, locked by a test. |
| Bilingual field misassignment is a dominant error source | **DISPROVED.** 14.46% of bilingual CER, 1.43% Persian-only. |
| Structured decoding is impossible on this stack | **REFUTED** (D40). GBNF grammar works, 10/10 parsed. |
| Phases 0, 1, 2, 3, 6 | complete. `ARCHITECTURE.md`, `config.py`, `splits.json`, the per-field harness, and `ci_gate.py` all exist. |

If you find yourself writing `ARCHITECTURE.md` or splitting the dataset, stop — you
have misread the state of the repo.

---

## 4. Gate 0 — the blocker

**There is no ground truth for the 90 real documents.** `ocr_eval/dataset_ex` holds
exactly 90 `.JPG` files and nothing else; all 90 manifest records read
`"gt_exists": false`. Registered as **D35**.

Consequence: no CER, WER, or field-accuracy number exists for the real corpus, and
Phases 4 and 5 have accuracy-based exit criteria they cannot meet.

**Absolute rule: never generate ground truth from model output, from a second model,
or from your own reading of the images.** A GT built from predictions makes every
downstream number meaningless while looking green. If you are tempted, stop and report.

The unblocking path is human transcription: fill `ocr_eval/gt_real_template.jsonl`
(90 pre-filled rows) per `GT_SPEC.md`, save as `ocr_eval/ground_truth_real.jsonl`, then

```bash
venv312\Scripts\python.exe ocr_eval/score_real.py --validate-only
venv312\Scripts\python.exe ocr_eval/score_real.py
```

Partial delivery is useful: **20 documents** answers whether `repeat_penalty` costs
accuracy; **50** gives a usable confidence interval. Predictions for both decoding arms
are already stored — no re-inference needed.

**Your first task is to report on Gate 0**, not to work around it: confirm the state,
estimate the transcription effort per document, and state exactly which numbers unlock
at 20 / 50 / 90 documents.

---

## Track A — work that is possible now, without ground truth

Do these in order. One variable per run, logged to `ocr_eval/experiments.md` with
`config.provenance()`. GT-free metrics (checksums, letterform counts, field-length
distributions, presence rates) are legitimate evidence — use them.

### A1 — D43: Arabic letterforms reach the caller (wiring task)

34.8% of usable outputs contain Arabic yeh/kaf/teh-marbuta, in all five fields.
`ocr_pipeline/persian_text.py` already normalises these and is unit-tested; `main.py`
does not call it. Wire it in at the API boundary. Expected outcome: 34.8% → 0%.
**This is the highest value-per-risk item in the repo.** Verify with the CI gate — and
first confirm that gate actually fails when it should, see A4.

### A2 — D44: schema collapse (constrained decoding, with a known cost)

18 of 89 usable outputs put body-length text in a header field; longest observed
`sender` is 2116 characters. Bounded GBNF caps in `ocr_pipeline/grammar.py` make this
unrepresentable and are implemented.

**But cutover was declined on D45:** under grammar, `contact_info` presence falls
58.4% → 4.5%. Resolve that tension rather than picking a side blindly. Suggested
approach: keep the caps on `sender`/`receiver`/`subject`, relax or remove the
constraint on `contact_info`, and A/B the variants on the full 90. Report presence
rates per field for every arm.

### A3 — D42/D46: digit integrity

Only 12 of 65 national-ID-shaped strings pass mod-11; the single IBAN fails mod-97.
Structured decoding did not help (18.5% → 19.6%). Checksums are a GT-free oracle, so
this is measurable today. Investigate whether the cause is image tokenisation
(`IMAGE_MIN_TOKENS`), the quantisation, or Persian-Indic digit handling — the synthetic
corpus contains **zero** Persian-Indic digits (D36), so it cannot be tested there.

### A4 — Trust your own tooling

A false-green was already found in this repo's CI gate: the Arabic-letterform check
read a key the function does not return and reported 0% on a corpus that is 34.8%.
Before relying on any check, prove it fails on known-bad input. Add a deliberately
failing fixture for each gate.

### A5 — D36/D37: the synthetic corpus cannot test the real failure modes

Zero Persian-Indic digits, zero Arabic letterforms, only 2 layouts (not the 5–6 the old
brief assumed), `receiver` unscoreable in 0/36 and `contact_info` in 6/36. Extend
`generate_dataset.py` to cover the modes that actually fail in production. This is
worth doing *because* it needs no human transcription.

---

## Track B — blocked on Gate 0, do not attempt

Nothing here can be honestly closed without ground truth. Do not produce numbers for
these; do not soften the exit criteria so they pass.

- True accuracy baseline on the real corpus (CER/WER/field accuracy).
- Resolution tuning (`RESIZE_MAX_EDGE`) — the accuracy/latency knee needs accuracy.
- **The model-fitness decision (D10).** The pinned model does not claim Persian. If
  measured accuracy is bad, the answer may be a different model, not more tuning —
  and no amount of Track A work substitutes for that call.

---

## Deferred — scale work

The previous brief's Phase 7 (Celery/Kafka, Kubernetes, millions of Mobodo users) is
**out of scope until Track A closes**. Two reasons, both concrete:

1. A service that returns the letter body 20.2% of the time does not have a throughput
   problem, it has a correctness problem. Scaling it multiplies wrong answers.
2. The stated remedy does not match the bottleneck. Inference is a single local
   `llama-server` process holding a 4.4 GB quantised model on one GPU. Making the
   FastAPI layer stateless does not create GPU capacity. The real questions are
   per-GPU concurrent-request capacity, VRAM per slot, and cost per thousand pages.

**The one thing worth doing now:** measure single-node throughput — max concurrent
requests before p99 degrades, and requests/sec at that concurrency. That number is the
input to every capacity decision later, and it costs one afternoon.

---

## Operating rules

- **Read before write.** Section 0 is not optional.
- **One variable per experiment**, logged in `ocr_eval/experiments.md` with
  `config.provenance()`.
- **Never tune against the test split.** Dev is in `ocr_eval/splits.json`.
- **Never fabricate ground truth.** See Gate 0.
- **Refuting a premise is a valid deliverable.** Several claims in earlier briefs were
  wrong. If a task in *this* brief is based on a false premise, say so with evidence
  and stop — that is the correct outcome, not a failure.
- **No metric without its denominator.** "98.9% usable" without "2.58 of 5 fields" is
  a misleading report.
- This project is **not a git repo.** Do not delete files that hold evidence; disable
  behind a flag and leave a guard test.

---

## Start here

Deliver one short report covering, in order:

1. **Gate 0 status** — confirm or refute D35 from the filesystem. Effort estimate for
   transcription. Which numbers unlock at 20 / 50 / 90 documents.
2. **Any correction to this brief** — anything above that the repo contradicts.
3. **Your A1 plan** — the D43 normalisation wiring, and how you will prove the CI gate
   can actually fail before you trust it.

Then stop and wait for confirmation. Do not begin A2.
