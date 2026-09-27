# Dictionary Correction — Before / After

The Dehkhoda dictionary (312,507 headwords loaded into a local SQLite DB) is used as a conservative post-corrector on the OCR output: only Persian-script tokens that are **out of vocabulary** are replaced. Latin letters, digits, codes and punctuation are left untouched.

> **This is corrector v2.** The first version allowed any single edit over the whole Persian alphabet and measurably *hurt* accuracy (17.04% → 18.08% CER, 83 harmful changes). Diagnosis: its substitutions (س→ا, ژ→ت, د→م, ح→ن) were *typing* errors, not OCR errors, and transposition — a keyboard artifact OCR never produces — caused 10 harmful changes and zero helpful ones. v2 restricts the edit model to what an OCR can actually get wrong. See `ablation.py` for the full comparison.

## Net effect on accuracy

| Source | CER before | CER after | Δ CER | WER before | WER after |
|---|--:|--:|--:|--:|--:|
| ALL | 17.04% | 17.01% | -0.03% | 22.38% | 22.23% |
| synthetic_bilingual | 25.59% | 25.54% | -0.04% | 32.44% | 32.18% |
| synthetic_fa | 4.49% | 4.49% | +0.00% | 8.04% | 8.04% |

Bilingual **Persian-script segment** CER: 16.17% → 16.12% (-0.06%). The Latin/digit segment is unchanged by design — a Persian dictionary can't correct it.

## Why — token change accounting

- Persian tokens seen: **993**  
- Out of vocabulary (not in Dehkhoda): **179**  
- Tokens the corrector changed: **2**  
  - **Helpful** (wrong → a word that's in the ground truth): **2**  
  - **Harmful** (a correct word → something else): **0**  
  - **Neutral** (wrong → another word, still not the reference): **0**

**Helpful corrections:** `استحضاار → استحضار`, `استحضاار → استحضار`

## Rule ablation

Each correction rule measured independently on the same predictions:

| Config | CER | Δ | helpful | harmful |
|---|--:|--:|--:|--:|
| no correction (baseline) | 17.04% | — | — | — |
| v1 any single edit | 18.08% | +1.04% | 6 | 83 |
| v2a confusable-sub + doubled-del | 17.28% | +0.24% | 3 | 23 |
| v2b … + unambiguous only | 17.13% | +0.09% | 3 | 11 |
| v2c confusable-sub only | 17.15% | +0.11% | 1 | 11 |
| **v2d doubled-letter deletion only** | **17.01%** | **−0.03%** | **2** | **0** |

Only **v2d** never damages a correct word, so it is the shipped default. Dropping the 32.8% of Dehkhoda entries that are pure cross-references was also tested and changed nothing (17.01%).

## Conclusion

Overall CER moved -0.03% — **essentially neutral**, recovering the full ~1.07-point regression the first design caused.

The remaining ceiling is structural: Dehkhoda is a **classical** lexicon, so ~22% of correctly-read modern words (سامانه، اینترنت، دیجیتال، پروژه، فناوری) are simply not in it. Any rule aggressive enough to fix a real error is also aggressive enough to snap one of those modern words onto a classical neighbour — پروژه → پروره is the clearest case, and it is not fixable from the dictionary alone. Meanwhile the OCR's actual weak spot, embedded Latin/digit codes, is entirely outside a Persian dictionary's reach.

To get a real gain from dictionary correction, the next steps are: (a) a **modern** Persian frequency lexicon layered on top of Dehkhoda so modern words stop reading as out-of-vocabulary, (b) gating correction on OCR token confidence rather than vocabulary membership alone, and (c) a checksum/format validator — not a dictionary — for the Latin/digit fields.
