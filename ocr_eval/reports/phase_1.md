# Phase 1 report — evaluation harness rebuilt

Harness `1.0.0` · normalizer `1.0.0` · config hash `12c95b35c7cb38ed` · generated 2026-09-01T13:40:05+00:00

manifest `30bcbe75be2f7d9b` · splits_v2 `d26202f31029f6ee` · git `not-a-git-repo`

> **Scope.** Scored from predictions.jsonl, which covers the 36 SYNTHETIC images only. The 90 real documents have no ground truth (defect D15) and are therefore absent from every number below.


## Metrics — normalized

### Headline pair (never publish CER-given-output alone)

| slice | n docs | n templates | usable-output rate | CER given output | effective CER | CER given output (CI over templates) |
|---|---|---|---|---|---|---|
| all | 36 | 6 | 83.33% [69.44, 94.44] | 31.16% [22.38, 42.90] | 46.89% [34.70, 60.07] | 31.16% [21.73, 39.01] |
| real | 0 | — | — | — | — | — |
| synthetic | 36 | 6 | 83.33% [69.44, 94.44] | 31.16% [22.38, 42.90] | 46.89% [34.70, 60.07] | 31.16% [21.73, 39.01] |
| synthetic|persian_only | 18 | 6 | 83.33% [66.67, 100.00] | 10.27% [4.06, 17.61] | 29.76% [14.11, 48.32] | 10.27% [3.53, 25.17] |
| synthetic|bilingual | 18 | 6 | 83.33% [66.67, 100.00] | 48.72% [39.75, 65.57] | 60.90% [47.45, 76.95] | 48.72% [38.51, 70.07] |
| synthetic|dev | 22 | 5 | 86.36% [72.73, 100.00] | 31.53% [18.90, 49.58] | 44.28% [27.47, 63.72] | 31.53% [18.24, 42.39] |

### Error classes — measured separately, never merged

| slice | field-assignment | omission | hallucination | duplication | null-agreement | CER (correctly-assigned only) |
|---|---|---|---|---|---|---|
| all | 5 / 2.78% [0.56, 5.00] | 32 / 28.07% [16.51, 40.35] | 0 / 0.00% [0.00, 0.00] | 5 / 2.78% [0.56, 5.00] | 0 / 0.00% [0.00, 0.00] | 31.16% [21.79, 42.19] |
| synthetic | 5 / 2.78% [0.56, 5.00] | 32 / 28.07% [16.51, 40.35] | 0 / 0.00% [0.00, 0.00] | 5 / 2.78% [0.56, 5.00] | 0 / 0.00% [0.00, 0.00] | 31.16% [21.79, 42.19] |
| synthetic|persian_only | 1 / 1.11% [0.00, 3.33] | 16 / 29.63% [12.96, 50.00] | 0 / 0.00% [0.00, 0.00] | 1 / 1.11% [0.00, 3.33] | 0 / 0.00% [0.00, 0.00] | 10.27% [4.05, 18.38] |
| synthetic|bilingual | 4 / 4.44% [1.11, 8.89] | 16 / 26.67% [12.90, 42.37] | 0 / 0.00% [0.00, 0.00] | 4 / 4.44% [1.11, 8.89] | 0 / 0.00% [0.00, 0.00] | 48.72% [40.03, 64.78] |
| synthetic|dev | 1 / 0.91% [0.00, 2.73] | 15 / 21.43% [8.57, 36.11] | 0 / 0.00% [0.00, 0.00] | 1 / 0.91% [0.00, 2.73] | 0 / 0.00% [0.00, 0.00] | 31.53% [18.56, 48.80] |

### Per-field table (1.1)


**synthetic**

| field | slots | unscoreable | scored | CER | WER | exact | null-agree | omissions | hallucinations | misassigned |
|---|---|---|---|---|---|---|---|---|---|---|
| `sender` | 36 | 0 | 26 | 49.28% | 55.71% | 38.46% | 0.00% | 10 | 0 | 0 |
| `receiver` | 36 | 36 | 0 | — | — | — | — | 0 | 0 | 0 |
| `subject` | 36 | 0 | 27 | 43.91% | 50.74% | 22.22% | 0.00% | 9 | 0 | 0 |
| `body_text` | 36 | 0 | 29 | 25.76% | 26.67% | 10.34% | 0.00% | 7 | 0 | 0 |
| `contact_info` | 36 | 30 | 0 | — | — | — | 0.00% | 6 | 0 | 0 |

**synthetic|bilingual**

| field | slots | unscoreable | scored | CER | WER | exact | null-agree | omissions | hallucinations | misassigned |
|---|---|---|---|---|---|---|---|---|---|---|
| `sender` | 18 | 0 | 15 | 68.81% | 76.77% | 0.00% | 0.00% | 3 | 0 | 0 |
| `receiver` | 18 | 18 | 0 | — | — | — | — | 0 | 0 | 0 |
| `subject` | 18 | 0 | 15 | 62.39% | 69.33% | 33.33% | 0.00% | 3 | 0 | 0 |
| `body_text` | 18 | 0 | 14 | 41.27% | 40.00% | 0.00% | 0.00% | 4 | 0 | 0 |
| `contact_info` | 18 | 12 | 0 | — | — | — | 0.00% | 6 | 0 | 0 |

**synthetic|persian_only**

| field | slots | unscoreable | scored | CER | WER | exact | null-agree | omissions | hallucinations | misassigned |
|---|---|---|---|---|---|---|---|---|---|---|
| `sender` | 18 | 0 | 11 | 0.84% | 4.88% | 90.91% | 0.00% | 7 | 0 | 0 |
| `receiver` | 18 | 18 | 0 | — | — | — | — | 0 | 0 | 0 |
| `subject` | 18 | 0 | 12 | 21.68% | 27.87% | 8.33% | 0.00% | 6 | 0 | 0 |
| `body_text` | 18 | 0 | 15 | 9.41% | 12.30% | 20.00% | 0.00% | 3 | 0 | 0 |
| `contact_info` | 18 | 18 | 0 | — | — | — | — | 0 | 0 | 0 |

## Metrics — raw

### Headline pair (never publish CER-given-output alone)

| slice | n docs | n templates | usable-output rate | CER given output | effective CER | CER given output (CI over templates) |
|---|---|---|---|---|---|---|
| all | 36 | 6 | 83.33% [69.44, 94.44] | 32.98% [24.16, 44.62] | 48.28% [36.08, 61.26] | 32.98% [23.70, 40.44] |
| real | 0 | — | — | — | — | — |
| synthetic | 36 | 6 | 83.33% [69.44, 94.44] | 32.98% [24.16, 44.62] | 48.28% [36.08, 61.26] | 32.98% [23.70, 40.44] |
| synthetic|persian_only | 18 | 6 | 83.33% [66.67, 100.00] | 11.89% [5.77, 19.32] | 31.01% [15.56, 49.22] | 11.89% [4.81, 26.10] |
| synthetic|bilingual | 18 | 6 | 83.33% [66.67, 100.00] | 50.71% [41.76, 67.52] | 62.41% [49.34, 78.22] | 50.71% [40.27, 71.65] |
| synthetic|dev | 22 | 5 | 86.36% [72.73, 100.00] | 33.05% [20.33, 51.10] | 45.52% [28.91, 64.99] | 33.05% [20.28, 43.26] |

### Error classes — measured separately, never merged

| slice | field-assignment | omission | hallucination | duplication | null-agreement | CER (correctly-assigned only) |
|---|---|---|---|---|---|---|
| all | 5 / 2.78% [0.56, 5.00] | 32 / 28.07% [16.51, 40.35] | 0 / 0.00% [0.00, 0.00] | 5 / 2.78% [0.56, 5.00] | 0 / 0.00% [0.00, 0.00] | 32.98% [23.56, 43.87] |
| synthetic | 5 / 2.78% [0.56, 5.00] | 32 / 28.07% [16.51, 40.35] | 0 / 0.00% [0.00, 0.00] | 5 / 2.78% [0.56, 5.00] | 0 / 0.00% [0.00, 0.00] | 32.98% [23.56, 43.87] |
| synthetic|persian_only | 1 / 1.11% [0.00, 3.33] | 16 / 29.63% [12.96, 50.00] | 0 / 0.00% [0.00, 0.00] | 1 / 1.11% [0.00, 3.33] | 0 / 0.00% [0.00, 0.00] | 11.89% [5.80, 19.99] |
| synthetic|bilingual | 4 / 4.44% [1.11, 8.89] | 16 / 26.67% [12.90, 42.37] | 0 / 0.00% [0.00, 0.00] | 4 / 4.44% [1.11, 8.89] | 0 / 0.00% [0.00, 0.00] | 50.71% [41.86, 66.50] |
| synthetic|dev | 1 / 0.91% [0.00, 2.73] | 15 / 21.43% [8.57, 36.11] | 0 / 0.00% [0.00, 0.00] | 1 / 0.91% [0.00, 2.73] | 0 / 0.00% [0.00, 0.00] | 33.05% [19.86, 50.16] |

### Per-field table (1.1)


**synthetic**

| field | slots | unscoreable | scored | CER | WER | exact | null-agree | omissions | hallucinations | misassigned |
|---|---|---|---|---|---|---|---|---|---|---|
| `sender` | 36 | 0 | 26 | 49.76% | 60.61% | 34.62% | 0.00% | 10 | 0 | 0 |
| `receiver` | 36 | 36 | 0 | — | — | — | — | 0 | 0 | 0 |
| `subject` | 36 | 0 | 27 | 44.53% | 52.31% | 18.52% | 0.00% | 9 | 0 | 0 |
| `body_text` | 36 | 0 | 29 | 28.02% | 33.10% | 0.00% | 0.00% | 7 | 0 | 0 |
| `contact_info` | 36 | 30 | 0 | — | — | — | 0.00% | 6 | 0 | 0 |

**synthetic|bilingual**

| field | slots | unscoreable | scored | CER | WER | exact | null-agree | omissions | hallucinations | misassigned |
|---|---|---|---|---|---|---|---|---|---|---|
| `sender` | 18 | 0 | 15 | 69.15% | 80.21% | 0.00% | 0.00% | 3 | 0 | 0 |
| `receiver` | 18 | 18 | 0 | — | — | — | — | 0 | 0 | 0 |
| `subject` | 18 | 0 | 15 | 63.06% | 72.22% | 26.67% | 0.00% | 3 | 0 | 0 |
| `body_text` | 18 | 0 | 14 | 43.92% | 47.38% | 0.00% | 0.00% | 4 | 0 | 0 |
| `contact_info` | 18 | 12 | 0 | — | — | — | 0.00% | 6 | 0 | 0 |

**synthetic|persian_only**

| field | slots | unscoreable | scored | CER | WER | exact | null-agree | omissions | hallucinations | misassigned |
|---|---|---|---|---|---|---|---|---|---|---|
| `sender` | 18 | 0 | 11 | 1.68% | 8.33% | 81.82% | 0.00% | 7 | 0 | 0 |
| `receiver` | 18 | 18 | 0 | — | — | — | — | 0 | 0 | 0 |
| `subject` | 18 | 0 | 12 | 22.22% | 27.59% | 8.33% | 0.00% | 6 | 0 | 0 |
| `body_text` | 18 | 0 | 15 | 11.28% | 17.69% | 0.00% | 0.00% | 3 | 0 | 0 |
| `contact_info` | 18 | 18 | 0 | — | — | — | — | 0 | 0 | 0 |

## Letterform conformance — reported on RAW output, outside the normalizer

- Field values inspected: **101**
- Containing Arabic letterforms: **27 (26.73%)**
- Totals — Arabic yeh U+064A: 49 · Arabic kaf U+0643: 2 · alef maksura U+0649: 5

This is defect **D8**. It is published separately precisely because the normalizer folds it away — every CER number above is blind to it.


## Reversal buckets (1.6)

Latin/digit token mismatches: **29**

| bucket | count | % |
|---|---|---|
| absent | 24 | 82.76% |
| genuine_misread | 5 | 17.24% |

## Latency (1.7)

| set | n | mean | p50 | p90 | p95 | p99 |
|---|---|---|---|---|---|---|
| failures_included | 36 | 3.155 | 1.9 | 2.56 | 17.74 | 17.77 |
| failures_excluded | 30 | 1.93 | 1.9 | 2.32 | 2.5 | 2.56 |

## Failure taxonomy

Failures: **6** — {'http_422': 3, 'all_null_200': 3}
