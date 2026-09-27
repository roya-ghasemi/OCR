"""
Ablation: which correction rules actually help?

Scores the saved OCR predictions (ocr_eval/predictions.jsonl) under several
corrector configurations and reports CER/WER before and after each, plus a
helpful/harmful token accounting. No inference is re-run.

Usage:  python dehkhoda/ablation.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr_eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import (  # noqa: E402
    load_jsonl, normalize, norm_digit, corpus_cer, corpus_wer, GT_PATH, PRED_PATH,
)
from corrector import Corrector as CorrectorV1  # noqa: E402
from corrector_v2 import CorrectorV2, _PERSIAN_CORE, _HAS_PERSIAN  # noqa: E402

OUT = Path(__file__).resolve().parent / "ablation.json"


def persian_tokens(text: str) -> set:
    s = set()
    for tok in text.split(" "):
        if _HAS_PERSIAN.search(tok):
            m = _PERSIAN_CORE.search(tok)
            if m:
                n = normalize(m.group(0), "digit")
                if n:
                    s.add(n)
    return s


def score(items, corrector):
    """items: list of dicts with ref_text / pred_text / source."""
    refs, before, after = [], [], []
    helpful = harmful = neutral = changed = 0
    ex_help, ex_harm = [], []
    per_source = {}

    for x in items:
        corrected, _ = corrector.correct_text(x["pred_text"])
        refs.append(norm_digit(x["ref_text"]))
        before.append(norm_digit(x["pred_text"]))
        after.append(norm_digit(corrected))
        per_source.setdefault(x["source"], {"r": [], "b": [], "a": []})
        per_source[x["source"]]["r"].append(refs[-1])
        per_source[x["source"]]["b"].append(before[-1])
        per_source[x["source"]]["a"].append(after[-1])

        ref_toks = persian_tokens(x["ref_text"])
        for pa, pb in zip(x["pred_text"].split(" "), corrected.split(" ")):
            if pa == pb:
                continue
            ma, mb = _PERSIAN_CORE.search(pa), _PERSIAN_CORE.search(pb)
            if not ma or not mb:
                continue
            changed += 1
            old, new = normalize(ma.group(0), "digit"), normalize(mb.group(0), "digit")
            if new in ref_toks and old not in ref_toks:
                helpful += 1
                if len(ex_help) < 10:
                    ex_help.append(f"{old}→{new}")
            elif old in ref_toks and new not in ref_toks:
                harmful += 1
                if len(ex_harm) < 10:
                    ex_harm.append(f"{old}→{new}")
            else:
                neutral += 1

    res = {
        "cer_before": corpus_cer(refs, before), "cer_after": corpus_cer(refs, after),
        "wer_before": corpus_wer(refs, before), "wer_after": corpus_wer(refs, after),
        "changed": changed, "helpful": helpful, "harmful": harmful, "neutral": neutral,
        "examples_helpful": ex_help, "examples_harmful": ex_harm,
        "by_source": {},
    }
    for s, d in per_source.items():
        res["by_source"][s] = {
            "cer_before": corpus_cer(d["r"], d["b"]),
            "cer_after": corpus_cer(d["r"], d["a"]),
        }
    return res


def main():
    gt = {r["filename"]: r for r in load_jsonl(GT_PATH)}
    items = []
    for p in load_jsonl(PRED_PATH):
        if p["error"] is not None or not norm_digit(p["pred_text"]).strip():
            continue
        items.append({
            "source": p["source"],
            "pred_text": p["pred_text"],
            "ref_text": gt[p["filename"]]["text"],
        })
    print(f"Scoring {len(items)} images with output.\n")

    configs = [
        ("v1  any-edit (original)", CorrectorV1()),
        ("v2a confusable-sub + doubled-del", CorrectorV2(unambiguous_only=False)),
        ("v2b  ... + unambiguous only", CorrectorV2(unambiguous_only=True)),
        ("v2c confusable-sub only", CorrectorV2(allow_doubled_deletion=False, unambiguous_only=True)),
        ("v2d doubled-del only", CorrectorV2(allow_substitution=False, unambiguous_only=True)),
    ]

    results = {}
    print(f"{'config':<36}{'CER before':>11}{'CER after':>11}{'Δ':>9}"
          f"{'help':>6}{'harm':>6}{'neut':>6}")
    print("-" * 85)
    for name, c in configs:
        r = score(items, c)
        results[name] = r
        d = r["cer_after"] - r["cer_before"]
        mark = "  <-- best" if False else ""
        print(f"{name:<36}{r['cer_before']*100:10.2f}%{r['cer_after']*100:10.2f}%"
              f"{d*100:+8.2f}%{r['helpful']:6}{r['harmful']:6}{r['neutral']:6}{mark}")

    best = min(results.items(), key=lambda kv: kv[1]["cer_after"])
    print("-" * 85)
    print(f"Best config: {best[0]}  CER {best[1]['cer_after']*100:.2f}% "
          f"(baseline {best[1]['cer_before']*100:.2f}%)")
    print()
    for name, r in results.items():
        if r["examples_harmful"]:
            print(f"{name}\n   harmful: {', '.join(r['examples_harmful'][:6])}")
        if r["examples_helpful"]:
            print(f"   helpful: {', '.join(r['examples_helpful'][:6])}")

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT.name}")


if __name__ == "__main__":
    main()
