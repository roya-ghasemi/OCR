# GT validation — 90 real administrative letters (§2.2)

Generated 2026-09-11T08:44:54+00:00  
File: `ground_truth_real_fixed_v2.jsonl`  
Verdict: **USABLE**

## Warnings — scoring proceeds, but these bound the interpretation

- CamScanner ⁨7-1-25 10.26⁩_1.JPG: unrecognised key(s) ['full_text', 'subject_source']
- 2 group(s) of documents share identical GT text. Effective sample size is 82, not 84; confidence intervals are computed over groups as well as images.

## Counts

- rows: 90 of 90 expected
- transcribed: 84   not yet transcribed: 6
- distinct GT texts: 82 (duplicate groups: 2)

### Per-field coverage

| field | rows with a value |
|---|---:|
| `sender` | 84 |
| `receiver` | 79 |
| `subject` | 84 |
| `body_text` | 84 |
| `contact_info` | 84 |

### Encoding audit (documents affected)

| property | n |
|---|---:|
| ascii_digit | 73 |
| harakat | 15 |
| persian_indic_digit | 84 |

## Suspected transcription errors found by the v2 digit reader (2026-09-22, D58)

Logged, **not** edited — GT is never changed from a reader's output. For the transcriber to verify against the scans.

| documents | field | GT says | page prints (verified by eye at native resolution) |
|---|---|---|---|
| all لادن-letterhead rows (`_11 _21 _25 _30 _37 _38 _64 _65 _67 _7 _80`, …) | `contact_info` | phone `۵۰۲۷۸۷۳-۵۰۲۷۸۷۱-۵۰۲۵۳۰۵-۰۵۱۱` | phone `۰۵۱۱-۵۰۲۷۸۷۱-۵۰۲۷۸۷۳` · fax `۰۵۱۱-۵۰۲۵۳۰۵` (`۵۰۲۵۳۰۵` is the fax only) |
| `_11` | `body_text` | شناسه ملی `۱۰۳۸۰۴۳۶۰۱۲` | `۱۴۰۰۲۰۰۴۱۰۵` |
| `_43` | `body_text` | only the footnote | a form with printed national IDs, phones and an economic code |
| `_76` (and likely `_37`, `_38`) | `body_text` | cheque numbers / amounts | handwritten in blue pen — not printed (D59) |
