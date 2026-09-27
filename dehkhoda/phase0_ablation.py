"""
Phase 0.3 — Dehkhoda ablation: the experiment that justifies removing the stage.

The corrector is pure post-processing over the model's JSON. It never touches the
image, the prompt, or decoding, so the honest A/B is to apply / not-apply it to the
SAME frozen predictions.jsonl. That is strictly stronger than re-running inference
twice: it removes inference variance entirely, so every reported delta is caused by
the corrector and by nothing else.

Arms
  off  no correction                                    (what /ocr ships today)
  v2d  doubled-letter deletion, unambiguous only        (what /ocr/corrected ships)
  v2b  + visually-confusable substitution, unambiguous
  v2a  + confusable substitution, ambiguous allowed     (most aggressive v2)

Reports, per arm and per split: CER (all / bilingual / fa), WER, usable-output rate,
the D1 semantic-substitution watchlist counts, total tokens changed, and every
individual change so a sample can be hand-audited.

Usage:  venv312\\Scripts\\python.exe dehkhoda\\phase0_ablation.py
Writes: dehkhoda/phase0_ablation.json
        dehkhoda/phase0_changes.tsv     (every change, for the hand audit)
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "ocr_eval"))
sys.path.insert(0, str(HERE))

from run_eval import (normalize, extract_fa, extract_en, corpus_cer,  # noqa: E402
                      corpus_wer)
from corrector_v2 import CorrectorV2  # noqa: E402

PRED = ROOT / "ocr_eval" / "predictions.jsonl"
GT = ROOT / "ocr_eval" / "ground_truth.jsonl"
SPLITS = ROOT / "ocr_eval" / "splits.json"
OUT_JSON = HERE / "phase0_ablation.json"
OUT_TSV = HERE / "phase0_changes.tsv"

AUDIT_SEED = 20260901
AUDIT_N = 30

# D1's named semantic substitutions. Key = the wrong form the brief attributes to
# Dehkhoda; value = the correct form it claims was overwritten.
WATCHLIST = {
    "احتمالا": "احتراماً",
    "اعلان": "اعلام",
    "فرمانیه": "فرمایید",
    "پیشایپیش": "پیشاپیش",
    "پیشابیش": "پیشاپیش",
    "پیشایش": "پیشاپیش",
    "پیشایی": "پیشاپیش",
}

ARMS = {
    "off": None,
    "v2d": dict(allow_substitution=False, allow_doubled_deletion=True,
                unambiguous_only=True),
    "v2b": dict(allow_substitution=True, allow_doubled_deletion=True,
                unambiguous_only=True),
    "v2a": dict(allow_substitution=True, allow_doubled_deletion=True,
                unambiguous_only=False),
}


def load():
    gt = {}
    for line in GT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            gt[r["filename"]] = r
    preds = []
    for line in PRED.read_text(encoding="utf-8").splitlines():
        if line.strip():
            preds.append(json.loads(line))
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    of_split = {f: "dev" for f in splits["dev"]}
    of_split.update({f: "test" for f in splits["test"]})
    return gt, preds, of_split


def score(rows, key):
    """rows: list of (gt_row, hypothesis_text). key selects the arm's text."""
    refs = [normalize(g["text"]) for g, _ in rows]
    hyps = [normalize(h) for _, h in rows]
    out = {
        "n": len(rows),
        "cer": corpus_cer(refs, hyps),
        "wer": corpus_wer(refs, hyps),
    }
    for src in ("synthetic_bilingual", "synthetic_fa"):
        sel = [(normalize(g["text"]), normalize(h))
               for g, h in rows if g["source"] == src]
        if sel:
            r, hy = zip(*sel)
            out[f"cer_{src}"] = corpus_cer(list(r), list(hy))
    bil = [(g, h) for g, h in rows if g["source"] == "synthetic_bilingual"]
    if bil:
        out["cer_bilingual_fa_segment"] = corpus_cer(
            [normalize(extract_fa(g["text"])) for g, _ in bil],
            [normalize(extract_fa(h)) for _, h in bil])
        out["cer_bilingual_en_segment"] = corpus_cer(
            [normalize(extract_en(g["text"])) for g, _ in bil],
            [normalize(extract_en(h)) for _, h in bil])
    return out


def watchlist_counts(texts):
    c = Counter()
    for t in texts:
        for bad in WATCHLIST:
            n = t.count(bad)
            if n:
                c[bad] += n
    return dict(c)


def diff_tokens(before: str, after: str):
    """Token-aligned changes. The corrector splits on " " and rejoins with " ", so
    it rewrites tokens in place and a positional zip is exact. A space-token can
    still span a newline, so narrow each hit to the single whitespace-delimited
    word that actually differs — otherwise the audit reads unintelligibly."""
    b, a = before.split(" "), after.split(" ")
    assert len(b) == len(a), "corrector changed the token count"
    out = []
    for x, y in zip(b, a):
        if x == y:
            continue
        xw, yw = x.split(), y.split()
        if len(xw) == len(yw):
            narrowed = [(p, q) for p, q in zip(xw, yw) if p != q]
            if len(narrowed) == 1:
                out.append(narrowed[0])
                continue
        out.append((x.replace("\n", "\\n"), y.replace("\n", "\\n")))
    return out


def main():
    gt, preds, of_split = load()

    usable = [p for p in preds if not p.get("error") and p.get("pred_text")]
    print(f"predictions: {len(preds)}  usable (non-error, non-empty): {len(usable)}")
    print(f"HTTP errors: {sum(1 for p in preds if p.get('error'))}   "
          f"empty-200:  {sum(1 for p in preds if not p.get('error') and not p.get('pred_text'))}")

    results = {"arms": {}, "watchlist_definition": WATCHLIST}
    all_changes = []

    for arm, cfg in ARMS.items():
        corr = CorrectorV2(**cfg) if cfg else None
        if corr:
            print(f"[{arm}] vocab={len(corr.vocab):,}")

        texts = {}
        changed_tokens = 0
        for p in usable:
            src = p["pred_text"]
            if corr is None:
                texts[p["filename"]] = src
            else:
                out, stats = corr.correct_text(src)
                texts[p["filename"]] = out
                changed_tokens += stats["changed"]
                for b, a in diff_tokens(src, out):
                    all_changes.append({"arm": arm, "filename": p["filename"],
                                        "before": b, "after": a})

        arm_res = {"tokens_changed": changed_tokens, "splits": {}}
        for split in ("all", "dev", "test"):
            rows = [(gt[f], t) for f, t in texts.items()
                    if split == "all" or of_split.get(f) == split]
            arm_res["splits"][split] = score(rows, arm)
        # usable-output rate is a property of the model, not the corrector; it is
        # reported per arm to make explicit that the corrector cannot change it.
        arm_res["usable_output_rate"] = len(usable) / len(preds)
        arm_res["watchlist_in_output"] = watchlist_counts(list(texts.values()))
        results["arms"][arm] = arm_res

    # watchlist in ground truth, for reference
    results["watchlist_in_ground_truth"] = watchlist_counts(
        [g["text"] for g in gt.values()])

    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    with OUT_TSV.open("w", encoding="utf-8") as fh:
        fh.write("arm\tfilename\tbefore\tafter\n")
        for c in all_changes:
            fh.write(f"{c['arm']}\t{c['filename']}\t{c['before']}\t{c['after']}\n")

    # ── console summary ───────────────────────────────────────────────────────
    base = results["arms"]["off"]["splits"]["all"]["cer"]
    print(f"\n{'arm':5} {'changed':>8} {'CER all':>9} {'Δ':>7} {'CER bil':>9} "
          f"{'CER fa':>8} {'WER':>8} {'CER dev':>9} {'CER test':>9}")
    for arm in ARMS:
        a = results["arms"][arm]
        s = a["splits"]["all"]
        print(f"{arm:5} {a['tokens_changed']:8} {s['cer']*100:8.2f}% "
              f"{(s['cer']-base)*100:+6.2f} {s['cer_synthetic_bilingual']*100:8.2f}% "
              f"{s['cer_synthetic_fa']*100:7.2f}% {s['wer']*100:7.2f}% "
              f"{a['splits']['dev']['cer']*100:8.2f}% "
              f"{a['splits']['test']['cer']*100:8.2f}%")

    print("\nD1 watchlist occurrences (in ground truth / per arm's output):")
    print(f"  ground truth: {results['watchlist_in_ground_truth'] or '{} (none)'}")
    for arm in ARMS:
        print(f"  {arm:5}: {results['arms'][arm]['watchlist_in_output']}")

    # ── hand-audit sample ─────────────────────────────────────────────────────
    shipped = [c for c in all_changes if c["arm"] == "v2d"]
    pool = shipped if len(shipped) >= AUDIT_N else all_changes
    rng = random.Random(AUDIT_SEED)
    sample = rng.sample(pool, min(AUDIT_N, len(pool)))
    print(f"\nHand-audit sample ({len(sample)} of {len(pool)} changes; "
          f"pool = {'v2d only' if pool is shipped else 'all arms — v2d had too few'}):")
    for i, c in enumerate(sample, 1):
        print(f"  {i:2}. [{c['arm']}] {c['before']} -> {c['after']}   ({c['filename']})")

    print(f"\nWrote {OUT_JSON}\nWrote {OUT_TSV}")


if __name__ == "__main__":
    main()
