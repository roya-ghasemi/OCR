# Root-cause diagnosis — numeric hallucination, body CER, "memorisation"

Date 2026-09-13. Scope `E:\ghasemi\ocr`. Every number below was measured in this
session against the live engine; scripts and raw outputs are in
`ocr_eval/experiments/diagnostics_0913/`. Nothing here is inferred from the model's
name or from the brief.

Three premises in the brief are contradicted by the repository and are corrected
first, because the diagnosis depends on them.

| premise in the brief | what the repo shows |
|---|---|
| "the fine-tuning dataset", "training samples", "thousands of samples repeating a handful of templates" | **This project has never fine-tuned anything.** There is no training code, no training data and no adapter in the tree. The weights are a third-party checkpoint (`prithivMLmods/coreOCR-7B-050325-preview`, quantised by `mradermacher`). The 90 letters and 36 synthetic pages are *evaluation* data the model has never seen. |
| "the model has memorised the header words because they appear frequently in the training samples" | Impossible on the above; and refuted by measurement (§3). |
| "CER > 71% may be falsely high because digit systems are not normalised" | Checked (§2.1): all three digit systems fold to ASCII before comparison; `۳۲۱/۰۰۰/۰۰۰` vs `321/000/000` scores CER 0.00. The 71% is real. |

---

## 0. Which backend is actually serving the model

Read from `engine.py`, `config.py`, `engine.log` and the live process table — not assumed.

- **Binary:** `C:\Users\PC\.lmstudio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.28.2\llama-server.exe` — the plain **llama.cpp `llama-server`** (build `fe2adf0`, CUDA 12). LM Studio only *installed* the binary; the LM Studio application is not involved in serving. Not koboldcpp, not Ollama, not text-generation-webui.
- **Launcher:** the project's own `engine.py::LlamaEngine.start()` (`engine.py:172-216`) spawns it as a child of the FastAPI process with exactly:
  ```
  llama-server.exe -m <model.gguf> --host 127.0.0.1 --port 18234 -ngl 99 -c 16384
                   --mmproj <mmproj-f16.gguf> --image-min-tokens 1024
  ```
  Vendor CUDA DLLs are put on `PATH` from `...\backends\vendor\win-llama-cuda12-vendor-v2`.
- **Weights:** `D:\models\models\mradermacher\coreOCR-7B-050325-preview-GGUF\coreOCR-7B-050325-preview.Q4_K_S.gguf` (4,457,769,440 B) + `coreOCR-7B-050325-preview.mmproj-f16.gguf` (sha256 `34933952…`). `engine.log` confirms: `loaded multimodal model ... mmproj-f16.gguf`, `n_slots = 4`, `kv_unified = 'true'`.
- **Client path:** `main.py` → `AsyncOpenAI` → `POST http://127.0.0.1:18234/v1/chat/completions` with the image as a base64 data-URI, no preprocessing. The `ocr_pipeline` package hits the same endpoint with a GBNF `grammar`.
- **Build pin:** `ENGINE_BUILD_PIN=2.28.2` (D48). A newer 2.33.0 backend exists on disk and is deliberately not used.

---

## 1. Problem 1 — numeric hallucination

### Root cause (measured)

**The model cannot read Persian-Indic digit glyphs. At all. At any resolution.** The
vision path is fine; Latin digits read perfectly; the failure is specific to the
`۰-۹` / `٠-٩` glyphs, and it is a property of the model family, not of this
pipeline.

E13a — synthetic digit cards, white page, Tahoma, engine called directly with a
plain "transcribe" prompt, `temperature=0, top_k=1`, no grammar, no JSON prompt:

| input | 40 px | 80 px | 160 px | total |
|---|---:|---:|---:|---:|
| Latin digits `0-9` (4 numbers) | 4/4 | 4/4 | 4/4 | **12/12 exact** |
| Persian-Indic `۰-۹` (same 4 numbers) | 0/4 | 0/4 | 0/4 | **0/12 exact** |

What it produced for `۳۲۱۰۰۰۰۰۰` at three sizes: `221`, `221`, `221`. For
`۶۴۰۷۴۰۰۳۱۵`: `01.07.2013`, `٢٠٢٣٧٧٣٢٦`, `٢٠٢٣ • ٧ • ٣٦` — it emits *dates*. For
`۰۹۲۴۴۲۴۱۱۷` at 160 px: `١١١١١١١١…` to the token cap.

E13c — the same cards through **Qwen2.5-VL-7B-Instruct Q4_K_M** (on disk, same
llama-server build): Latin **12/12**, Persian **0/12**. Closer misses (`٢١٥٢٤٠٧٣٧٣`
for `4152607373`) but no exact read. Both Qwen-family 7B checkpoints share the defect.

Angle by angle, as asked:

1. **Is the image reaching the vision encoder?** Yes. A blank white page returns
   *"This image is blank and contains no text to transcribe."* — the model is
   looking. Latin digits at 40 px read 12/12. The `mmproj` is loaded (`engine.log`).
   The pipeline is not broken.
2. **Resolution / crop.** Measured from `usage.prompt_tokens`: a 3016×2304 scan
   becomes **4,017 image tokens ≈ 2030×1551 effective (scale 0.67)**. llama.cpp is
   capping at ~4k tokens; Qwen2-VL supports up to 16k. This costs some legibility
   on body text (see §2) but it is **not** the digit cause: Persian digits fail at
   160 px on a clean white card, where Latin digits pass at 40 px. Raising
   `--image-max-tokens` is worth testing for words, not for digits.
3. **Quantisation.** Only `Q4_K_S` of coreOCR exists locally; no Q8/fp16 of it is on
   disk, and the `hf_cache` copy of the base `Qwen2-VL-7B-Instruct` is missing shard
   1 of 5 and cannot be loaded (also: no `torch` in `venv312`, 12 GB VRAM). A same-
   model quant comparison therefore was **not run**. Evidence against quantisation
   as the cause: a *second* Q4 model of a different generation shows the identical
   Latin-perfect / Persian-zero split. Q4 does not selectively destroy one digit
   alphabet. To close this properly: download `coreOCR-7B-050325-preview.Q8_0.gguf`
   (~8 GB) and re-run `digits.py` — 20 minutes, decisive.
4. **Sampling.** Already deterministic: `temperature=0.0` in `config.py`
   (`TEMPERATURE`), and `/ocr` sends it. The probes above additionally forced
   `top_k=1` and got the same wrong digits. Creativity is not the mechanism.
5. **Constrained decoding for numbers.** Implemented as a probe and **it does not
   help**: with a grammar restricted to `[0-9۰-۹٠-٩/,.-]+` and the question "what
   is the amount", the model answers `221` for `۳۲۱/۰۰۰/۰۰۰` — the same wrong
   digits, now guaranteed to be digits. A grammar constrains the alphabet, not the
   reading (this generalises D46). Not shipped; the working GBNF syntax on this build
   is `root ::= [0-9۰-۹٠-٩/,.-]+` with literal characters (the `\uXXXX` form is
   rejected by the 2.28.2 parser).
6. **Classical-OCR cross-check.** Tesseract is not installed on this machine and
   `pytesseract` is not in the venv; not implemented. It is the right shape of fix
   — see §5.

**Why it produces counting sequences specifically.** From the tokenizer
(`POST /tokenize`): Latin digits are single tokens; every Persian-Indic and
Arabic-Indic digit is **two byte-fallback tokens** — `۰` = `[219][176]`, `۱` =
`[219][177]`, … `۹` = `[219][185]`. A Persian number is a constant byte `219`
followed by a byte in `176…185`. A model that has not learned these glyphs is
emitting the most probable continuation of that byte range, and the most probable
walk through `176,177,178,…` is exactly `۰۱۲۳۴۵۶۷۸۹`. The "1234567890" pattern is
the tokenizer's byte order leaking through.

### Fix ranking for Problem 1

| # | change | expected effect | cost |
|---|---|---|---|
| 1 | **Read numbers with something that can.** Digit-region cross-check: locate digit runs in the model's output, crop the corresponding region (Qwen2-VL can return grounding boxes; or use layout heuristics), run Tesseract `fas`+`eng` or a small digit CNN on the crop, and **prefer the classical read**; flag disagreement. | fixes the digit field outright for machine-printed numbers | 1–2 days; needs Tesseract |
| 2 | **Ask for Latin digits.** Instruct (and, in the grammar, constrain) numbers to be emitted as `0-9`. The glyph problem remains, so this is not a fix on its own — but it removes the byte-walk failure mode and makes every number one token per digit, checksummable and grammar-friendly. Test on the dev 10. | small; makes #1 easier | 1 hour |
| 3 | **Raise `--image-max-tokens`** so the scan reaches the model at native resolution. | words, not digits | 1 hour + latency cost |
| 4 | **Q8 comparison** — download and re-run `digits.py`. | closes the quantisation question with a number | 20 min |
| 5 | Model change (see §4). | the only thing that fixes the reading | days |

---

## 2. Problem 2 — body CER 71%

### 2.1 CER computation, verified

`ocr_eval/normalize.py` `RULES_DEFAULT`: NFC → strip bidi/ZWJ/tatweel/harakat →
Arabic→Persian letterforms → **`digits_to_ascii=True`** (Persian-Indic *and*
Arabic-Indic → `0-9`) → ZWNJ→space → collapse whitespace. Then Levenshtein /
reference length (`harness.py:63`). Checked in-session:

| a | b | after normalisation | CER |
|---|---|---|---|
| `۳۲۱/۰۰۰/۰۰۰` | `321/000/000` | identical | 0.00 |
| `٣٢١` | `321` | identical | 0.00 |
| `۱۴۰۳` | `١٤٠٣` | identical | 0.00 |

The number is not a normalisation artefact. Two things *do* shape it and are stated
with it in E12: CER is uncapped (a 20-char field holding 2,000 chars scores 100×), and
7 of 10 bodies are session-dependent (D52).

**Digits are not the main driver of body CER.** Arm B, dev, 48 scored bodies:

| | median CER | mean |
|---|---:|---:|
| as scored | 0.71 | 1.20 |
| digits deleted from both sides | **0.59** | 1.04 |
| capped at 1.0 | 0.71 | 0.71 |

Digits are 5.4% of body characters and account for ~12 CER points. **The remaining
~59% is words** — on real scans. On clean synthetic pages in the same layout
(E13b control letters) body CER is **0.18–0.22**, and novel person names inside the
body are read correctly. So the word error is not "cannot read Persian"; it is a
combination of (a) scan quality/fonts vs the ~0.67 downscale, (b) skipped clauses —
in E13b the model dropped the entire clause containing the amount on 2 of 6 letters,
(c) the footer being absorbed into `body_text` (D45), and (d) cap-running
degenerate text (D52; 15 of 48 bodies over 100% CER).

### 2.2 Dataset diversity

There is no training set. The *evaluation* corpus: 90 letters, **87 carry the same
letterhead** (`شرکت خدماتی سبز گستر`), one scanner (CamScanner), one font family,
two footers. That is why `sender` and `contact_info` look good and mean little (E12).
The synthetic set has 2 layouts, Tahoma/Arial only, zero Persian-Indic digits (D36).

### 2.3 Was the model trained on Persian?

No. Model card (fetched 2026-09-13, `prithivMLmods/coreOCR-7B-050325-preview`):
fine-tuned from `Qwen/Qwen2-VL-7B-Instruct` on **`allenai/olmOCR-mix-0225`,
`prithivMLmods/Openpdf-Analysis-Recognition`, `prithivMLmods/Opendoc1-Analysis-Recognition`
— 274,209 samples**, all English document/PDF corpora. Persian, Farsi and Arabic
are named nowhere; the card says "performance on low-resource or rare scripts may
vary". Whatever Persian ability exists is inherited from the Qwen2-VL base and was
not reinforced by this fine-tune. **D49, now with its evidence.**

### 2.4 Quantised vs fp16

Not runnable here (§1.3). Bound it from the other side: the *word* reading on clean
synthetic pages is good (CER 0.2) at Q4, and the *digit* failure reproduces on a
second Q4 model. Quantisation is not where the 71% comes from.

---

## 3. Problem 3 — "memorisation instead of reading"

### Diagnostic (E13b), as designed in the brief

Six letters in the corpus layout with company names, receivers, person names and
numbers **that appear nowhere in the corpus**, plus two control letters carrying
the corpus letterhead. Rendered 2300×3200 Tahoma; read by arm B; scored with the
project CER.

| field | novel (n=6) mean CER | control (n=2) mean CER |
|---|---:|---:|
| `sender` (78 px letterhead) | 0.44 (3 exact, 1 partial, 2 missed) | 0.05 |
| `receiver` (novel phrases) | **0.006** | 0.00 |
| `body_text` | 1.68 (one cap-runaway at 5.16) | 0.20 |
| `contact_info` | omitted 6/6 | omitted 2/2 |
| amount / national-ID digits found | **0/6** | **0/2** |

**Verdict: memorisation is refuted.** Never-seen receiver lines read at CER 0.006;
never-seen person names inside the body (`بهرام کیانی مقدم`, `لیلا شریفی نسب`,
`فرزانه توکلی راد`) come back letter-perfect. The model reads Persian *words* when
they are clean. The corpus `sender` reads better than novel senders (0.05 vs 0.44)
— with n=2 vs 6 that is suggestive of template familiarity from the 87 identical
letterheads *in the prompt history*, not from training; and two of the six novel
senders were simply not extracted, which is D41/D45 field omission, not misreading.

The real Problem 3 is the mirror image of the brief's: the model reads words and
cannot read digits — on seen and unseen content alike.

### 3.2 / 3.3

Not applicable as stated (no training set). The augmentation, split-hygiene and
header-vs-body evaluation proposals are correct *for a fine-tuning programme*, and
§4 turns them into one. Header-vs-paragraph CER already exists: `per_field` in
`results_real.*.json` reports each field separately; E12 shows it.

---

## 4. If fine-tuning: a concrete plan

Fine-tuning is only justified if a Persian-capable base is not good enough out of
the box, so step 0 is a model bake-off, not training.

**Step 0 — bake-off (1–2 days, no training).** Same harness, same dev 51, three
candidates that claim Persian/Arabic script: Qwen2.5-VL-7B (on disk), a
Persian-trained VLM (e.g. one of the AryaBoo/Persian-OCR-VLM family or
`Qwen2-VL` fine-tunes that name Persian), and a classical pipeline
(Tesseract `fas` + layout) as the floor. Gate: body CER on dev, digits exact-rate,
per field.

**Step 1 — data, if training.** Synthetic-first, because it is the only way to get
digits and layouts in volume without transcribing:
- Generator (`generate_dataset.py`, extended): ≥10 letterhead templates, ≥6 Persian
  fonts (B Nazanin, B Titr, IRANSans, Vazir, Tahoma, Arial), random company /
  person names from large name lists, random amounts, dates, national IDs, IBANs,
  phone numbers in **Persian-Indic, Arabic-Indic and Latin digits, mixed within
  lines**; scan-simulation augmentation (blur, JPEG q 60–90, rotation ±3°, slight
  perspective, contrast/noise, CamScanner-style shadows). Target 20–50k pages.
- Real: the 84 transcribed letters stay **evaluation-only**. Never in training.
- Splits by (template, font) so no template+font pair crosses splits; names and
  numbers never repeat across splits.

**Step 2 — training.** LoRA on the chosen base (rank 16–64, vision tower frozen at
first, then unfrozen for the projector), 1–2 epochs, `max_pixels` at native scan
resolution. Loss on the five-field JSON target, with the number tokens weighted up.

**Step 3 — gates.** Dev body CER, digit exact-rate, and the E13a digit cards (a fixed
regression set), before/after; test split scored once.

---

## 5. Prioritised change list

| priority | problem | change | how it is measured |
|---|---|---|---|
| **P0** | 1 | Digit-region cross-check with a classical reader (Tesseract `fas`/`eng`) on crops, classical result preferred, disagreements flagged in the response. | digit exact-rate on E13a cards + amount/ID found-rate on E13b + checksum pass-rate on the real corpus (`analyze_fields.py`, D42). |
| **P0** | 2, 1 | Model bake-off (§4 step 0). This is the decision the last three weeks have been circling; D49 now has a number (oracle-routed CER 61–69%). | `score_real.py --split dev` per candidate; body CER + digit rate. |
| **P1** | 2 | `cache_prompt=false` in both client paths; re-run E11. If bodies become session-stable, D52 closes and every body number becomes trustworthy. | E11 run1~run2 similarity → 1.0. |
| **P1** | 2 | `--image-max-tokens 16384` (native resolution). | dev body CER, latency p50. |
| **P1** | 1 | Emit numbers as Latin digits (prompt + grammar alternative). | digit exact-rate; token count per number. |
| **P2** | 1 | Q8_0 download and `digits.py` re-run. | E13a table, Q4 vs Q8. |
| **P2** | 2 | Extend the generator (§4 step 1) — needed for any training and for a digit regression set now. | D36 closes (Persian-Indic digits present). |
| P3 | 3 | Nothing — the premise is refuted. Keep E13b as a regression test for reading novel content. | E13b receiver CER stays ≈0. |

## 6. Before/after benchmark method

Fixed, versioned, three layers, all already runnable:

1. **Digit cards** (`experiments/diagnostics_0913/digits.py`) — 24 cards, exact-match
   per script per size. Deterministic (`seed 13`). Runs in 90 s. This is the unit
   test for Problem 1: a fix that does not move Persian off 0/12 has not fixed it.
2. **Novel letters** (`novel_headers.py`) — 8 pages, per-field CER + amount/ID found.
   Unit test for Problem 3 and for clause-skipping.
3. **Real dev split** (`score_real.py --gt ground_truth_real_fixed_v2.jsonl --split dev
   --label <name>`) — 51 documents, usable rate, per-field CER/WER, omission and
   hallucination rates, oracle-routed CER, all with bootstrap CIs. Compare
   `results_real.<before>.json` vs `results_real.<after>.json`. Run E11 first so
   `body_text` is session-stable, or report body CER as a range over two processes.
4. **Test split** — scored once, at the end, for the configuration chosen on dev.

Every run: one variable, `config.provenance()` in the results, a row in
`experiments.md`.

---

## Artefacts

`experiments/diagnostics_0913/{digits.py, novel_headers.py, cards/, novel_letters/,
results_E13a_digits.json, results_E13b_novel_headers.json,
results_E13c_qwen25_digits.json}`; tokenizer probe output in the session log;
model card summary above.
