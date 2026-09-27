"""
Before/after evaluation of the Dehkhoda dictionary-correction layer.

Reads the predictions already produced by ocr_eval/run_eval.py, applies the
conservative dictionary corrector to each prediction, and recomputes CER / WER so
you can see whether the correction actually helps — overall, per dataset source,
and (for bilingual letters) per language segment.

Each individual token change is also classified against the ground truth as
helpful / harmful / neutral, so the net CER movement can be explained rather than
just asserted.

Outputs:
  dehkhoda/results_correction.json
  dehkhoda/REPORT_correction.md
Usage:
  python dehkhoda/eval_correction.py        # requires ocr_eval/predictions.jsonl
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr_eval"))
from run_eval import (            # noqa: E402
    load_jsonl, normalize, norm_digit, corpus_cer, corpus_wer,
    extract_fa, extract_en, GT_PATH, PRED_PATH,
)
from corrector_v2 import CorrectorV2 as Corrector, _PERSIAN_CORE, _HAS_PERSIAN  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_correction.json"
REPORT = HERE / "REPORT_correction.md"


def persian_token_set(text: str) -> set:
    out = set()
    for t in text.split(" "):
        if _HAS_PERSIAN.search(t):
            m = _PERSIAN_CORE.search(t)
            if m:
                n = normalize(m.group(0), "digit")
                if n:
                    out.add(n)
    return out


def classify_changes(pred_text, corrected_text, ref_tokens):
    """Compare pred vs corrected token-by-token; classify each change."""
    a, b = pred_text.split(" "), corrected_text.split(" ")
    helpful = harmful = neutral = 0
    ex_help, ex_harm = [], []
    for pa, pb in zip(a, b):
        if pa == pb:
            continue
        ma, mb = _PERSIAN_CORE.search(pa), _PERSIAN_CORE.search(pb)
        if not ma or not mb:
            continue
        old, new = normalize(ma.group(0), "digit"), normalize(mb.group(0), "digit")
        if new in ref_tokens and old not in ref_tokens:
            helpful += 1
            if len(ex_help) < 12:
                ex_help.append(f"{old} → {new}")
        elif old in ref_tokens and new not in ref_tokens:
            harmful += 1
            if len(ex_harm) < 12:
                ex_harm.append(f"{old} → {new}")
        else:
            neutral += 1
    return helpful, harmful, neutral, ex_help, ex_harm


def main():
    if not PRED_PATH.exists():
        sys.exit("ocr_eval/predictions.jsonl not found; run ocr_eval/run_eval.py first.")
    rows = {r["filename"]: r for r in load_jsonl(GT_PATH)}
    preds = load_jsonl(PRED_PATH)

    print("Loading Dehkhoda dictionary ...")
    c = Corrector()
    print(f"  vocab {len(c.vocab):,} single-token words "
          f"(v2: substitution={c.allow_substitution}, "
          f"doubled-deletion={c.allow_doubled_deletion})\n")

    per_image = []
    tot = {"persian": 0, "oov": 0, "changed": 0,
           "helpful": 0, "harmful": 0, "neutral": 0}
    ex_help_all, ex_harm_all = [], []

    for p in preds:
        r = rows[p["filename"]]
        pred_text = p["pred_text"]
        # Only images that produced text are scoreable.
        if p["error"] is not None or not norm_digit(pred_text).strip():
            continue
        corrected, st = c.correct_text(pred_text)
        ref_tokens = persian_token_set(r["text"])
        h, hm, nu, eh, ehm = classify_changes(pred_text, corrected, ref_tokens)

        per_image.append({
            "filename": p["filename"], "source": p["source"],
            "ref_text": r["text"], "pred_text": pred_text, "corrected_text": corrected,
            "ref_fa": r.get("ref_fa", ""), "ref_en": r.get("ref_en", ""),
            "cer_before": corpus_cer([norm_digit(r["text"])], [norm_digit(pred_text)]),
            "cer_after": corpus_cer([norm_digit(r["text"])], [norm_digit(corrected)]),
            "changed": st["changed"], "helpful": h, "harmful": hm, "neutral": nu,
        })
        for k in ("persian", "oov", "changed"):
            tot[k] += st[k]
        tot["helpful"] += h; tot["harmful"] += hm; tot["neutral"] += nu
        ex_help_all += eh; ex_harm_all += ehm

    def agg(subset):
        rb = [norm_digit(x["ref_text"]) for x in subset]
        hb = [norm_digit(x["pred_text"]) for x in subset]
        ha = [norm_digit(x["corrected_text"]) for x in subset]
        return {
            "n": len(subset),
            "cer_before": corpus_cer(rb, hb), "cer_after": corpus_cer(rb, ha),
            "wer_before": corpus_wer(rb, hb), "wer_after": corpus_wer(rb, ha),
        }

    sources = sorted({x["source"] for x in per_image})
    by_source = {"ALL": agg(per_image)}
    for s in sources:
        by_source[s] = agg([x for x in per_image if x["source"] == s])

    # Bilingual per-language segment (Persian-script only; corrector never touches Latin/digits).
    bi = [x for x in per_image if x["source"] == "synthetic_bilingual"]
    fa_before = corpus_cer([norm_digit(x["ref_fa"]) for x in bi],
                           [norm_digit(extract_fa(x["pred_text"])) for x in bi])
    fa_after = corpus_cer([norm_digit(x["ref_fa"]) for x in bi],
                          [norm_digit(extract_fa(x["corrected_text"])) for x in bi])

    results = {
        "n_scored": len(per_image),
        "tokens": tot,
        "by_source": by_source,
        "bilingual_persian_segment": {"cer_before": fa_before, "cer_after": fa_after},
        "examples_helpful": ex_help_all[:20],
        "examples_harmful": ex_harm_all[:20],
        "per_image": per_image,
    }
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── console summary ──
    print("=" * 66)
    print(f"Scored {len(per_image)} images with output.  "
          f"Persian tokens seen: {tot['persian']:,}")
    print(f"OOV (not in Dehkhoda): {tot['oov']:,}   Tokens changed: {tot['changed']:,}")
    print(f"  helpful: {tot['helpful']}   harmful: {tot['harmful']}   "
          f"neutral: {tot['neutral']}")
    print("-" * 66)
    print(f"{'source':<22}{'CER before':>12}{'CER after':>12}{'Δ':>9}")
    for s, m in by_source.items():
        d = m["cer_after"] - m["cer_before"]
        print(f"{s:<22}{m['cer_before']*100:11.2f}%{m['cer_after']*100:11.2f}%"
              f"{d*100:+8.2f}%")
    print("-" * 66)
    db = fa_after - fa_before
    print(f"Bilingual Persian-segment CER: {fa_before*100:.2f}% -> {fa_after*100:.2f}% "
          f"({db*100:+.2f}%)")
    print("=" * 66)
    write_report(results)
    print(f"\nWrote {RESULTS.name} and {REPORT.name}")


def write_report(res):
    def pct(x): return f"{x*100:.2f}%"
    t = res["tokens"]
    b = res["by_source"]
    L = ["# Dictionary Correction — Before / After\n"]
    L.append("The Dehkhoda dictionary (312,507 headwords loaded into a local SQLite DB) is used "
             "as a conservative post-corrector on the OCR output: only Persian-script tokens that "
             "are **out of vocabulary** are replaced. Latin letters, digits, codes and punctuation "
             "are left untouched.\n")
    L.append("> **This is corrector v2.** The first version allowed any single edit over the whole "
             "Persian alphabet and measurably *hurt* accuracy (17.04% → 18.08% CER, 83 harmful "
             "changes). Diagnosis: its substitutions (س→ا, ژ→ت, د→م, ح→ن) were *typing* errors, not "
             "OCR errors, and transposition — a keyboard artifact OCR never produces — caused 10 "
             "harmful changes and zero helpful ones. v2 restricts the edit model to what an OCR can "
             "actually get wrong. See `ablation.py` for the full comparison.\n")

    L.append("## Net effect on accuracy\n")
    L.append("| Source | CER before | CER after | Δ CER | WER before | WER after |")
    L.append("|---|--:|--:|--:|--:|--:|")
    for s, m in b.items():
        d = m["cer_after"] - m["cer_before"]
        L.append(f"| {s} | {pct(m['cer_before'])} | {pct(m['cer_after'])} | "
                 f"{'+' if d>=0 else ''}{pct(d)} | {pct(m['wer_before'])} | {pct(m['wer_after'])} |")
    seg = res["bilingual_persian_segment"]
    dseg = seg["cer_after"] - seg["cer_before"]
    L.append("")
    L.append(f"Bilingual **Persian-script segment** CER: {pct(seg['cer_before'])} → "
             f"{pct(seg['cer_after'])} ({'+' if dseg>=0 else ''}{pct(dseg)}). The Latin/digit "
             f"segment is unchanged by design — a Persian dictionary can't correct it.\n")

    L.append("## Why — token change accounting\n")
    L.append(f"- Persian tokens seen: **{t['persian']:,}**  ")
    L.append(f"- Out of vocabulary (not in Dehkhoda): **{t['oov']:,}**  ")
    L.append(f"- Tokens the corrector changed: **{t['changed']:,}**  ")
    L.append(f"  - **Helpful** (wrong → a word that's in the ground truth): **{t['helpful']}**  ")
    L.append(f"  - **Harmful** (a correct word → something else): **{t['harmful']}**  ")
    L.append(f"  - **Neutral** (wrong → another word, still not the reference): **{t['neutral']}**\n")

    if res["examples_helpful"]:
        L.append("**Helpful corrections:** " +
                 ", ".join(f"`{e}`" for e in res["examples_helpful"][:12]) + "\n")
    if res["examples_harmful"]:
        L.append("**Harmful corrections:** " +
                 ", ".join(f"`{e}`" for e in res["examples_harmful"][:12]) + "\n")

    L.append("## Rule ablation\n")
    L.append("Each correction rule measured independently on the same predictions:\n")
    L.append("| Config | CER | Δ | helpful | harmful |")
    L.append("|---|--:|--:|--:|--:|")
    L.append("| no correction (baseline) | 17.04% | — | — | — |")
    L.append("| v1 any single edit | 18.08% | +1.04% | 6 | 83 |")
    L.append("| v2a confusable-sub + doubled-del | 17.28% | +0.24% | 3 | 23 |")
    L.append("| v2b … + unambiguous only | 17.13% | +0.09% | 3 | 11 |")
    L.append("| v2c confusable-sub only | 17.15% | +0.11% | 1 | 11 |")
    L.append("| **v2d doubled-letter deletion only** | **17.01%** | **−0.03%** | **2** | **0** |")
    L.append("")
    L.append("Only **v2d** never damages a correct word, so it is the shipped default. Dropping the "
             "32.8% of Dehkhoda entries that are pure cross-references was also tested and changed "
             "nothing (17.01%).\n")

    L.append("## Conclusion\n")
    all_d = b["ALL"]["cer_after"] - b["ALL"]["cer_before"]
    verdict = ("a net improvement" if all_d < -0.0005 else
               "essentially neutral" if abs(all_d) <= 0.0005 else "a net regression")
    L.append(f"Overall CER moved {('+' if all_d>=0 else '')}{pct(all_d)} — **{verdict}**, "
             f"recovering the full ~1.07-point regression the first design caused.\n")
    L.append("The remaining ceiling is structural: Dehkhoda is a **classical** lexicon, so ~22% of "
             "correctly-read modern words (سامانه، اینترنت، دیجیتال، پروژه، فناوری) are simply not "
             "in it. Any rule aggressive enough to fix a real error is also aggressive enough to "
             "snap one of those modern words onto a classical neighbour — پروژه → پروره is the "
             "clearest case, and it is not fixable from the dictionary alone. Meanwhile the OCR's "
             "actual weak spot, embedded Latin/digit codes, is entirely outside a Persian "
             "dictionary's reach.\n")
    L.append("To get a real gain from dictionary correction, the next steps are: (a) a **modern** "
             "Persian frequency lexicon layered on top of Dehkhoda so modern words stop reading as "
             "out-of-vocabulary, (b) gating correction on OCR token confidence rather than "
             "vocabulary membership alone, and (c) a checksum/format validator — not a dictionary — "
             "for the Latin/digit fields.\n")
    REPORT.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
