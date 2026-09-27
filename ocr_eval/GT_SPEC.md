# Ground-truth specification — the 90 real administrative letters

Status: **the file this document specifies does not yet exist.** `ocr_eval/dataset_ex`
contains exactly 90 `.JPG` files and nothing else; every one is byte-identical to
the manifest lock recorded on 2026-09-01, and all 90 manifest records still carry
`"gt_exists": false`. Until a file matching this spec is present, no accuracy
number can be computed for the real corpus.

## Where the file goes

    ocr_eval/ground_truth_real.jsonl

A pre-filled skeleton is already generated for you at
`ocr_eval/gt_real_template.jsonl` — 90 rows, one per document, with `filename`,
`sha256`, `language_mode` and `split` populated and the five fields set to
`null`. Fill in the `fields` object and rename the file. Do not reorder rows, do
not add rows, do not edit `filename` or `sha256`: the scorer verifies both
against the manifest and refuses to run on a mismatch.

## Row schema

```json
{
  "filename": "CamScanner ⁨7-1-25 10.26⁩_1.JPG",
  "sha256": "…",
  "source": "real",
  "language_mode": "bilingual",
  "split": "dev",
  "fields": {
    "sender":       "…",
    "receiver":     "…",
    "subject":      "…",
    "body_text":    "…",
    "contact_info": "…"
  },
  "illegible": ["contact_info"],
  "notes": ""
}
```

### `fields` — the five scored slots

Transcribe **what is printed in the pixels**, not what the document means and not
what it ought to say. If the scan prints a misspelling, the GT carries the
misspelling.

| field | what belongs in it |
|---|---|
| `sender` | the originating organisation/person, as printed in the letterhead or signature block |
| `receiver` | the addressee — the `به:` / `جناب آقای` / `سرکار خانم` line |
| `subject` | the `موضوع:` line only, without the label |
| `body_text` | the letter body: salutation through closing formula, excluding letterhead, subject line and signature block |
| `contact_info` | phone, fax, email, postal address, website, IBAN — as printed, joined by newlines |

**`null` vs `""` is a real distinction and the scorer treats them differently.**
Use `null` when the field is genuinely absent from the document. Use a string
when it is present. Never use `""`.

### `illegible`

List any field name whose text is present on the page but cannot be read
confidently at this scan quality. Those fields are excluded from CER and reported
separately, so a bad scan does not silently become a model error. Put the
field in `illegible` **and** transcribe your best reading in `fields`.

## Transcription conventions

These exist so that GT and prediction are compared on the same footing. The
scorer normalises both sides before scoring (`ocr_eval/normalize.py`), so a
deviation here costs less than an inconsistency — but be consistent.

1. **Digits: transcribe the glyph that is printed.** If the page shows `۱۴۰۳`,
   write `۱۴۰۳`. If it shows `1403`, write `1403`. Do not convert. This matters:
   the synthetic corpus contains **zero** Persian-Indic digits across all 36
   documents, so the real GT is the only place the pipeline's digit handling can
   be measured at all.
2. **Letterforms: transcribe the glyph that is printed.** If the page uses Arabic
   `ي`/`ك`, write those, not the Persian `ی`/`ک`. The normaliser folds them; an
   un-folded GT is what lets us measure how often the *model* fails to.
3. **Line breaks:** `\n` where the document breaks a line. Do not re-wrap.
4. **ZWNJ:** write `‌` where the printed word requires it (`می‌رساند`).
5. **Do not expand abbreviations, do not fix grammar, do not translate.**
6. Leave text you cannot read as a best-effort reading plus an `illegible` entry —
   do not write a placeholder like `[ناخوانا]` into `fields`.

## PII

These are real administrative letters containing names, national IDs, phone
numbers and bank details. This file therefore:

- must **not** be committed to any public remote — it is covered by `.gitignore`;
- is never logged in full by the pipeline;
- appears in reports only as redacted excerpts.

## Validating before you hand it over

    venv312\Scripts\python.exe ocr_eval\score_real.py --validate-only

This runs the §2.2 protocol: schema check, filename/sha256 reconciliation against
the manifest, `null`-vs-empty check, per-field coverage, encoding audit
(NFC, bidi controls, tatweel, harakat, digit systems, Arabic letterforms), and
duplicate detection. It writes `ocr_eval/gt_real_audit.md` and exits non-zero if
the file is unusable.

## Partial delivery is acceptable

A subset is worth more than nothing. Rows whose `fields` are all `null` are
treated as "not yet transcribed" and excluded from scoring, with the count
reported alongside every headline. **50 of the 90 is enough to produce a real
baseline with a usable confidence interval; 20 is enough to falsify or confirm
the biggest open question** — whether `repeat_penalty`/DRY costs accuracy.
