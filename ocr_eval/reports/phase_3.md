# Phase 3 — field assignment on the real corpus (GT-free)

Generated 2026-09-03T12:50:00+00:00  
Scope: 90 real administrative letters; GT-free structural analysis only  
Usable outputs: **89 / 90**

## Why this report is GT-free

Phase 3 asks how much of the error is misassignment. That question needs CER, CER needs ground truth, and the real corpus has none (**D35**). On synthetic GT it was already measured and largely disproved: misassignment is 14.46% of bilingual CER and 1.43% of Persian-only CER — real, but not the dominant term the brief assumed.

What follows measures the *structural signatures* of misassignment, which need no reference, plus identifier checksums — which are ground truth the documents carry themselves.

## Field fill

| field | filled | % of usable | min | median | max |
|---|---:|---:|---:|---:|---:|
| `sender` | 89 | 100.0% | 7 | 15 | 2116 |
| `receiver` | 52 | 58.4% | 11 | 38 | 720 |
| `subject` | 19 | 21.3% | 10 | 100 | 1118 |
| `body_text` | 18 | 20.2% | 9 | 375 | 1093 |
| `contact_info` | 52 | 58.4% | 82 | 117 | 312 |

## D31 — schema collapse

A header field longer than its cap is carrying body text. Caps: `sender` 180, `receiver` 180, `subject` 240.

**18 of 89 usable outputs (20.2%)** show collapse.

| field | over cap |
|---|---:|
| `subject` | 8 |
| `sender` | 5 |
| `receiver` | 5 |

## Cross-field duplication

Two fields returning near-identical text (similarity ≥ 0.90, both ≥ 12 characters). **0 of 89 (0.0%)**.

## Digit systems, per field (D36)

The synthetic corpus contains **zero** Persian-Indic digits, so this table is the only evidence of what the pipeline actually emits.

| field | Persian-Indic | Arabic-Indic | ASCII |
|---|---:|---:|---:|
| `sender` | 8 | 1 | 5 |
| `receiver` | 6 | 0 | 5 |
| `subject` | 4 | 0 | 9 |
| `body_text` | 9 | 0 | 14 |
| `contact_info` | 40 | 0 | 50 |

## Arabic letterforms in the API response (D8)

| field | documents containing `ي`/`ك`/`ة` |
|---|---:|
| `receiver` | 10 |
| `subject` | 7 |
| `body_text` | 7 |
| `contact_info` | 6 |
| `sender` | 5 |

## Identifier integrity — ground truth the documents carry themselves

IBAN (ISO 13616 mod-97), Iranian national ID (mod-11) and mobile numbers are self-verifying. A failed checksum is a **confirmed** misread with no reference transcription required.

| identifier | found | checksum-valid | pass rate |
|---|---:|---:|---:|
| iban | 1 | 0 | 0.0% |
| national_id | 65 | 12 | 18.5% |
| mobile | 20 | 20 | 100.0% |

Documents containing at least one identifier: **42**.

A low pass rate here is the strongest accuracy evidence available before ground truth arrives: these are digits the model demonstrably got wrong.

## PII

This report contains counts and rates only. No field value, name, national ID, IBAN or phone number from the corpus appears in it.
