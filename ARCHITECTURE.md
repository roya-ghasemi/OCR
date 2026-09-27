# Architecture — Persian/Bilingual OCR Service

Phase 0 deliverable. Describes the system **as it actually is**, not as intended.

---

## 1. Request lifecycle: image upload → JSON response

```
POST /ocr   (multipart/form-data, field "file" or "image")
  │
  ▼
[1] _pick_upload()                          main.py
      accepts either field name; 400 if neither present
  │
  ▼
[2] upload.read()                           main.py
      400 if zero bytes
  │
  ▼
[3] Image.open(...).format + .verify()       main.py
      format is read FROM THE BYTES, not the declared Content-Type
      400 if unreadable or not in {JPEG,PNG,BMP,TIFF,WEBP}
  │
  ▼
[4] encode_image()                          main.py
      base64 data: URI.  *** NO PREPROCESSING ***
      no resize, no DPI normalization, no deskew, no contrast/denoise
  │
  ▼
[5] AsyncOpenAI.chat.completions.create()    main.py -> engine on :18234
      system = _SYSTEM_PROMPT (Persian, demands raw JSON, 5 named keys)
      user   = _USER_TURN + image_url
      temperature=0.0  max_tokens=2048  timeout=600s
      *** NO RETRY ***
  │
  ▼
[6] _clean_json()                            main.py
      strips ```json fences if the model added them
  │
  ▼
[7] json.loads()                             main.py
      422 on JSONDecodeError  <-- failure mode A
      _dump_raw_output() writes the COMPLETE model output + finish_reason
      to debug_raw/<file>.<ts>.json  (never to the shared log: real letters)
  │
  ▼
[8] LetterExtraction.model_validate()        main.py
      422 on schema violation
      *** all five fields are Optional -> an all-null object VALIDATES *** <-- failure mode B
  │
  ▼
200 OK  {sender, receiver, subject, body_text, contact_info}
```

`POST /ocr` is the **production** extraction endpoint and the one the evaluation
harness scores. There is no post-processing stage of any kind between step [8] and
the response.

### Endpoint parity (resolved, Phase 0)

| endpoint | status | pipeline |
|---|---|---|
| `POST /ocr` | **production** | steps [1]–[8] above, unchanged |
| `POST /ocr/corrected` | **deprecated passthrough** | steps [1]–[8], then returns `{raw, corrected, changed}` with `corrected` byte-identical to `raw` and `changed` empty; logs a warning on every call |

`/ocr` has always been the production endpoint: it is the only one in the README's
integration table, architecture diagram and curl example, and `run_eval.py` has always
called it. **Every published number therefore describes the shipped pipeline.**

`/ocr/corrected` was added during the Phase 0 remediation as an opt-in diagnostic that
applied the Dehkhoda `CorrectorV2`, and removed in Phase 0.3 once the ablation measured
that corrector at −0.03 pp corpus CER — inside noise. Because deletion would 404 any
out-of-repo caller, it was reinstated as a deprecation passthrough with no correction
stage behind it. Remove it once the logs show no traffic. See `dehkhoda/REMOVED.md`
for the ablation and `ocr_eval/test_phase0_guards.py` for the guard that keeps a
dictionary stage out of the runtime path.

**Consequence:** the Dehkhoda removal changed no production behaviour and cannot
improve any eval number. Its value was diagnostic only.

## 2. Engine lifecycle

`engine.py` owns a `llama-server.exe` child process for the app's lifetime.

1. On startup, scan `~/.lmstudio/extensions/backends` → the build pinned by
   `ENGINE_BUILD_PIN` (default `2.28.2`); newest CUDA-12 build only as a warned
   fallback, because a silent engine swap invalidates every recorded provenance.
2. Search `MODEL_SEARCH_ROOTS` in order (`D:\models\models`, then
   `~/.lmstudio/models`) → the coreOCR GGUF + its `mmproj` projector. Model and
   backend roots are independent: the models moved to `D:` on 2026-09-05, the
   backends stayed on `C:`.
3. Spawn `llama-server` on `127.0.0.1:18234`, adding the backend dir and the CUDA
   vendor DLL dir to the child's `PATH`.
4. Block until `/v1/models` returns 200 (~5 s), then FastAPI accepts traffic.
5. On shutdown, terminate the child. If a server is already on the port, adopt it
   and leave it running.

LM Studio the *application* is never launched; only its model files and bundled
engine binary are used.

## 3. Actual output schema — differs from what was assumed

```python
class LetterExtraction(BaseModel):
    sender:       Optional[str] = None
    receiver:     Optional[str] = None
    subject:      Optional[str] = None
    body_text:    Optional[str] = None
    contact_info: Optional[str] = None
```

**There is no `date` field.** Any date in the document has nowhere to go and is
either folded into `body_text` or dropped. 18 of 36 evaluation letters render a
`تاریخ:` line, so "the date is dropped" is a *schema gap*, not a model failure.

Every field is `Optional`, so `{null,null,null,null,null}` passes validation and is
returned as `200 OK` — failure mode B above.

## 4. Component map

| Concern | Location | Notes |
|---|---|---|
| Endpoint handlers | `main.py` `extract_text` | thin wrapper over `_run_ocr` |
| Shared OCR core | `main.py` `_run_ocr` | validation → engine call → parse |
| Prompt template | `main.py` `_SYSTEM_PROMPT`, `_USER_TURN` | Persian; 4 numbered rules |
| Model client | `main.py` module-level `client` | `AsyncOpenAI`, built in `lifespan` |
| JSON parse/validate | `main.py` `_clean_json` + `model_validate` | |
| Retry logic | — | **does not exist** |
| Image preprocessing | — | **does not exist** |
| Engine supervision | `engine.py` `LlamaEngine` | discovery, spawn, readiness, teardown |
| Config | `config.py` | Phase 0: centralized, values unchanged |
| Dictionary correction | — | **removed in Phase 0.3**; see `dehkhoda/REMOVED.md` |
| Eval | `ocr_eval/{generate_dataset,run_eval,make_report}.py` | |

## 5. Evaluation data — real shape

| Property | Value |
|---|---|
| Images | 36 |
| **Distinct ground-truth texts** | **18** (each rendered 1–3×) |
| **Distinct structural layouts** | **2** (Persian-only 9-line, bilingual 12-line) |
| Content pools | 6 senders, 6 subjects, 6 bodies, 3 closings, 2–3 signatures |
| Varied per render | font (tahoma/arial), size (26/28/30), rotation (0/±1.5°), gaussian noise (σ 0 or 4) |
| Determinism | seed 20260829; **all 36 images + GT regenerate byte-identically** |

Effective sample size is far below 36. Two structural layouts means the harness is
measuring two documents, dressed 18 ways.

## 6. Known failure modes (baseline, 36 images)

| Mode | Count | Where |
|---|--:|---|
| A — HTTP 422, unterminated JSON | 3 | all bilingual, **all the same letter content** |
| B — HTTP 200 with all fields null | 3 | all Persian-only, images are clean and legible |

Mode A's three cases differ only in the sender line and all contain the 21-digit
IBAN `IR820170000000123456789`. This is **one defect observed three times**, n=1.

## 7. Reproduce

```bash
venv312\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
venv312\Scripts\python.exe ocr_eval\run_eval.py
venv312\Scripts\python.exe ocr_eval\make_report.py
```

---

## 8. Model identity — pinned (Phase 0.6)

| Property | Value |
|---|---|
| GGUF repo | `mradermacher/coreOCR-7B-050325-preview-GGUF` |
| Weights | `coreOCR-7B-050325-preview.Q4_K_S.gguf` (4,457,769,440 bytes) |
| Projector | `coreOCR-7B-050325-preview.mmproj-f16.gguf` |
| Projector sha256 | `34933952a9ae2f1ac7af7a908189b3fc22106d3999fd3fae43b520ac4d308a78` |
| Quantization | Q4_K_S (4-bit) |
| Upstream FP16 model | `prithivMLmods/coreOCR-7B-050325-preview` |
| Base model | `Qwen/Qwen2-VL-7B-Instruct` |
| Runtime | llama.cpp `llama-server` 1 (fe2adf0), win-x86_64-nvidia-cuda12-avx2-2.28.2 |
| Tokenizer | bundled in the GGUF (Qwen2 BPE) |

### What the model card claims about Persian — **it does not**

From the `coreOCR-7B-050325-preview` card: the intended uses list
`"Multilingual OCR workflows"`, and the limitations say only that
`"While multilingual, performance on low-resource or rare scripts may vary."`
**Persian, Farsi and Arabic script are named nowhere on the card.** Its three
fine-tuning datasets are `prithivMLmods/Openpdf-Analysis-Recognition`,
`allenai/olmOCR-mix-0225` and `prithivMLmods/Opendoc1-Analysis-Recognition`.

The base model's card is more specific — Qwen2-VL-7B-Instruct claims support for
`"texts in different languages inside images, including most European languages,
Japanese, Korean, Arabic, Vietnamese, etc."` **Arabic script is claimed by the base
model; Persian is not named by either card.**

Persian is written in an extended Arabic script (`گ چ پ ژ`, Persian `ی`/`ک`) with heavy
ZWNJ use, so base-model Arabic coverage is a partial but genuine floor. What is *not*
established anywhere is that the coreOCR fine-tune preserved it — a fine-tune on
predominantly English document corpora can degrade the base model's other scripts.

**This is D10, and it is now confirmed as an open risk rather than a hypothesis.** No
evidence exists that this model is a good choice for Persian, and the observed behaviours
support the concern: `پیشاپیش` is misread in 12 of 12 occurrences, and Nastaliq has never
been tested at all (the dataset renders only Tahoma and Arial, both Naskh-style).

Per the brief, the model-fitness benchmark is Phase 5.6. Bringing it forward is worth
considering: if coreOCR is materially behind on Persian script, Phases 2–4 optimise a
model that is about to be replaced.

---

## 9. Dataset generator defects found in Phase 0.8

`ocr_eval/generate_dataset.py` writes ground truth from the *logical* string it intends to
draw, and separately draws the page via `arabic_reshaper.reshape()` → `get_display()`.
The two disagree, so the ground truth records intent rather than pixels:

| Defect | Effect | Scale |
|---|---|---|
| **G1** | `python-bidi` reverses the group order of hyphen-separated digit runs. `2026-08-27` is drawn as `27-08-2026`; `+98 21 8899 1234` as `1234 8899 21 98+`. Runs containing Latin letters (`INV-`, `EMP-`, `TRK-`, `version`) and slash-separated dates are drawn correctly. | 18 lines / 12 images |
| **G2** | `arabic_reshaper` silently drops tanween `U+064B`. `احتراماً` is drawn as `احتراما`. | 6 chars / 6 images |

18 of 36 images carry at least one defect. Full analysis in `ocr_eval/gt_audit.md`.

**Consequence:** the model applies a group reversal of its own to numeric dates (7 of 7
cases where it emits one). Where G1 had already reversed them the two cancel and the model
scores correct; where it had not, the error shows. **The dataset therefore cannot measure
digit ordering at all**, and no Latin/digit-segment or reversal metric from it should be
quoted until the generator is fixed (Phase 5).
