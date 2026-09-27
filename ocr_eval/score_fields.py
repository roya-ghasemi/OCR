"""
Phase 1 deliverable — field-aware scoring that separates reading from placement.

Re-scores an EXISTING predictions.jsonl (no new inference) so that the change in
numbers is attributable to the measurement method alone, never to the model. Run:

    python ocr_eval/score_fields.py            # rescore predictions.jsonl
    python ocr_eval/score_fields.py --compare  # also print old flat-blob vs new

What it computes, and why each exists
-------------------------------------
usable output rate      : ok / empty(200-null) / error(422). Never hidden inside CER.
named-field CER         : GT.field  vs  pred[same key].  Placement-SENSITIVE reading.
content-matched CER     : each GT segment vs the WHOLE prediction (best local
                          alignment). Placement-INSENSITIVE reading — "were the
                          characters read, wherever they landed?"
assignment loss         : named CER - content CER. The share of strict error caused
                          purely by putting right text under the wrong key.
placement accuracy      : for each GT field, is its best home the same-named pred field?
omission                : GT segment absent from the whole prediction (recall miss).
hallucination           : predicted value absent from the whole GT (precision miss),
                          e.g. an invented `receiver`.
orphan capture          : share of schema-orphaned page content (date, IBAN, codes,
                          amounts) the model captured anyway — pressure the schema
                          creates, not a model defect.
digit/code ordering     : Latin/digit runs classified exact / reversed / permuted /
                          partial / misread.
latency                 : p50/p90/p99/mean, reported twice (all imgs / ok only).
bootstrap 95% CI        : resampled over the 18 DISTINCT TEXTS, not the 36 images —
                          resampling correlated renders would understate the interval.

All CER numbers use the same Persian-aware digit normalization as run_eval.py, imported
from it so the two harnesses can never drift.
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import jiwer

from run_eval import norm_digit, safe_cer, corpus_cer, load_jsonl  # shared normalization

HERE = Path(__file__).resolve().parent
GT_FIELDS = HERE / "ground_truth_fields.jsonl"
PRED_PATH = HERE / "predictions.jsonl"
RESULTS_PATH = HERE / "results_fields.json"
OLD_RESULTS = HERE / "results.json"

SCHEMA_FIELDS = ["sender", "receiver", "subject", "body_text", "contact_info"]
MATCH_TAU = 0.50          # a segment/value counts as "found" if best local CER <= this
BOOT_ITERS = 2000
BOOT_SEED = 20260901


# ── Approximate substring CER (free leading/trailing gaps in the haystack) ───────
def substring_cer(needle: str, hay: str) -> float:
    """Min edit distance of `needle` against ANY substring of `hay`, / len(needle).
    0.0 means needle appears verbatim somewhere in hay; measures reading error of a
    segment independent of where in the prediction it landed."""
    needle = norm_digit(needle)
    hay = norm_digit(hay)
    if not needle:
        return 0.0
    if not hay:
        return 1.0
    m, n = len(needle), len(hay)
    prev = [0] * (n + 1)                      # dp[0][j] = 0: start anywhere for free
    for i in range(1, m + 1):
        cur = [i] + [0] * n                   # dp[i][0] = i: must consume needle prefix
        ni = needle[i - 1]
        for j in range(1, n + 1):
            sub = prev[j - 1] + (ni != hay[j - 1])
            cur[j] = sub if sub < prev[j] + 1 else prev[j] + 1     # sub vs delete needle char
            ins = cur[j - 1] + 1                                    # skip hay char
            if ins < cur[j]:
                cur[j] = ins
        prev = cur
    return min(prev) / m                       # end anywhere for free


# ── Digit / code ordering, 5 buckets ─────────────────────────────────────────────
_RUN = re.compile(r"[A-Za-z0-9]{3,}")


def classify_run(unit: str, pred: str) -> str:
    u = norm_digit(unit)
    p = norm_digit(pred)
    if u in p:
        return "exact"
    if u != u[::-1] and u[::-1] in p:
        return "reversed"
    # permutation: some window of pred is an anagram of u
    from collections import Counter as C
    target = C(u)
    k = len(u)
    for s in range(0, len(p) - k + 1):
        if C(p[s:s + k]) == target:
            return "permuted"
    # partial: a contiguous >=3 chunk of u survives
    for L in range(len(u), 2, -1):
        if any(u[a:a + L] in p for a in range(0, len(u) - L + 1)):
            return "partial"
    return "misread"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true",
                    help="also print old flat-blob metrics side by side")
    args = ap.parse_args()

    gt = {r["filename"]: r for r in load_jsonl(GT_FIELDS)}
    preds = load_jsonl(PRED_PATH)

    per_image = []
    named_pairs = []                 # (ref, hyp) for corpus named-field CER
    content_seg_weight = []          # (cer, len) for weighted content CER
    ordering = Counter()
    ordering_examples = defaultdict(list)
    placement_correct = placement_total = 0
    omit_schema = omit_orphan = seg_schema = seg_orphan = 0
    orphan_captured = orphan_total = 0
    hallucinated = pred_values_total = 0
    spurious_receiver = 0

    for p in preds:
        g = gt[p["filename"]]
        raw = p.get("raw") or {}
        pred_fields = {k: (raw.get(k) or "").strip() for k in SCHEMA_FIELDS}
        pred_fields = {k: v for k, v in pred_fields.items() if v}
        full_pred = "\n".join(pred_fields.values())

        if p["error"] is not None:
            status = "error"
        elif not full_pred.strip():
            status = "empty"
        else:
            status = "ok"

        row = {"filename": p["filename"], "source": p["source"], "status": status}

        if status == "ok":
            # 1) named-field CER (placement-sensitive)
            img_named = []
            for k, ref in g["fields"].items():
                hyp = pred_fields.get(k, "")
                named_pairs.append((norm_digit(ref), norm_digit(hyp)))
                img_named.append((norm_digit(ref), norm_digit(hyp)))
            row["named_cer"] = corpus_cer([r for r, _ in img_named], [h for _, h in img_named])

            # 2) content-matched CER + omission over fine segments
            img_content = []
            for seg in g["segments"]:
                if not seg["text"].strip():
                    continue
                sc = substring_cer(seg["text"], full_pred)
                w = len(norm_digit(seg["text"]))
                content_seg_weight.append((sc, w))
                img_content.append((sc, w))
                orphan = seg["belongs"] == "__orphan__"
                if orphan:
                    seg_orphan += 1
                    orphan_total += 1
                    if sc <= MATCH_TAU:
                        orphan_captured += 1
                    else:
                        omit_orphan += 1
                else:
                    seg_schema += 1
                    if sc > MATCH_TAU:
                        omit_schema += 1
            tw = sum(w for _, w in img_content) or 1
            row["content_cer"] = sum(sc * w for sc, w in img_content) / tw

            # 3) placement accuracy: does each GT field's best home carry its name?
            for k, ref in g["fields"].items():
                best_k, best_sc = None, 1.1
                for pk, pv in pred_fields.items():
                    sc = substring_cer(ref, pv)
                    if sc < best_sc:
                        best_sc, best_k = sc, pk
                if best_sc <= MATCH_TAU:
                    placement_total += 1
                    if best_k == k:
                        placement_correct += 1

            # 4) hallucination: predicted value with no support anywhere in GT
            for pk, pv in pred_fields.items():
                pred_values_total += 1
                if substring_cer(pv, g["text"]) > MATCH_TAU:
                    hallucinated += 1
                    if pk == "receiver":
                        spurious_receiver += 1
            # receiver is null in every GT letter: any non-empty receiver is spurious
            if pred_fields.get("receiver") and substring_cer(pred_fields["receiver"], g["text"]) > MATCH_TAU:
                pass  # already counted above

            # 5) digit/code ordering on orphan+contact runs
            latin_source = " ".join(g["orphan"] + [g["fields"].get("contact_info", "")])
            for unit in _RUN.findall(norm_digit(latin_source)):
                b = classify_run(unit, full_pred)
                ordering[b] += 1
                if b in ("reversed", "permuted", "partial", "misread") and len(ordering_examples[b]) < 8:
                    ordering_examples[b].append(unit)

        per_image.append(row)

    ok = [r for r in per_image if r["status"] == "ok"]
    n = len(per_image)

    named_cer = corpus_cer([r for r, _ in named_pairs], [h for _, h in named_pairs])
    tw = sum(w for _, w in content_seg_weight) or 1
    content_cer = sum(sc * w for sc, w in content_seg_weight) / tw

    # ── latency ──
    def pct(vals, q):
        if not vals:
            return 0.0
        vals = sorted(vals)
        i = min(len(vals) - 1, int(round(q * (len(vals) - 1))))
        return vals[i]
    all_lat = [p["seconds"] for p in preds]
    ok_names = {r["filename"] for r in ok}
    ok_lat = [p["seconds"] for p in preds if p["filename"] in ok_names]
    latency = {
        "all": {"p50": pct(all_lat, .5), "p90": pct(all_lat, .9), "p99": pct(all_lat, .99),
                "mean": round(sum(all_lat) / len(all_lat), 2)},
        "ok_only": {"p50": pct(ok_lat, .5), "p90": pct(ok_lat, .9), "p99": pct(ok_lat, .99),
                    "mean": round(sum(ok_lat) / max(len(ok_lat), 1), 2)},
    }

    # ── bootstrap 95% CI over the 18 DISTINCT TEXTS ──
    groups = defaultdict(list)          # distinct GT text -> [named pairs of its ok images]
    cgroups = defaultdict(list)         # distinct GT text -> [content (cer,len) of its segs]
    for p in preds:
        if p["filename"] not in ok_names:
            continue
        g = gt[p["filename"]]
        key = g["text"]
        raw = p.get("raw") or {}
        pf = {k: (raw.get(k) or "").strip() for k in SCHEMA_FIELDS}
        pf = {k: v for k, v in pf.items() if v}
        full_pred = "\n".join(pf.values())
        for k, ref in g["fields"].items():
            groups[key].append((norm_digit(ref), norm_digit(pf.get(k, ""))))
        for seg in g["segments"]:
            if seg["text"].strip():
                cgroups[key].append((substring_cer(seg["text"], full_pred), len(norm_digit(seg["text"]))))

    keys = list(groups.keys())
    rng = random.Random(BOOT_SEED)

    def boot(metric):
        out = []
        for _ in range(BOOT_ITERS):
            sample = [rng.choice(keys) for _ in keys]
            if metric == "named":
                pairs = [pr for k in sample for pr in groups[k]]
                out.append(corpus_cer([r for r, _ in pairs], [h for _, h in pairs]))
            else:
                cw = [x for k in sample for x in cgroups[k]]
                w = sum(x[1] for x in cw) or 1
                out.append(sum(c * l for c, l in cw) / w)
        out.sort()
        return {"lo": out[int(.025 * len(out))], "hi": out[int(.975 * len(out))]}
    ci = {"named_cer": boot("named"), "content_cer": boot("content")}

    results = {
        "note": "field-aware re-scoring of an existing predictions.jsonl; no new inference",
        "n_images": n,
        "usable": {"ok": len(ok),
                   "empty_200_null": sum(1 for r in per_image if r["status"] == "empty"),
                   "error_422": sum(1 for r in per_image if r["status"] == "error"),
                   "usable_rate": round(len(ok) / n, 4)},
        "reading": {
            "named_field_cer": named_cer,
            "content_matched_cer": content_cer,
            "assignment_loss": named_cer - content_cer,
            "ci95_over_texts": ci,
        },
        "placement_accuracy": {
            "correct": placement_correct, "total": placement_total,
            "rate": round(placement_correct / max(placement_total, 1), 4),
        },
        "omission": {
            "schema_segments": seg_schema, "schema_omitted": omit_schema,
            "schema_omit_rate": round(omit_schema / max(seg_schema, 1), 4),
            "orphan_segments": seg_orphan, "orphan_omitted": omit_orphan,
        },
        "hallucination": {
            "pred_values": pred_values_total, "unsupported": hallucinated,
            "rate": round(hallucinated / max(pred_values_total, 1), 4),
            "spurious_receiver": spurious_receiver,
        },
        "orphan_capture": {
            "orphan_content_segments": orphan_total, "captured_anyway": orphan_captured,
            "capture_rate": round(orphan_captured / max(orphan_total, 1), 4),
            "comment": "content the 5-field schema cannot hold (date/IBAN/codes/amounts)",
        },
        "ordering": {"buckets": dict(ordering), "examples": {k: v for k, v in ordering_examples.items()}},
        "latency_seconds": latency,
        "per_image": per_image,
    }
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── print ──
    r = results
    print("=" * 70)
    print("PHASE 1 — field-aware re-scoring (no new inference)")
    print("=" * 70)
    u = r["usable"]
    print(f"Usable output : {u['ok']}/{n} = {u['usable_rate']*100:.1f}%   "
          f"(empty-200 {u['empty_200_null']}, error-422 {u['error_422']})")
    rd = r["reading"]
    print(f"\nReading CER (over images that produced output):")
    print(f"  named-field   (placement-SENSITIVE) : {rd['named_field_cer']*100:6.2f}%   "
          f"95% CI [{ci['named_cer']['lo']*100:.1f}, {ci['named_cer']['hi']*100:.1f}]")
    print(f"  content-match (placement-FREE)      : {rd['content_matched_cer']*100:6.2f}%   "
          f"95% CI [{ci['content_cer']['lo']*100:.1f}, {ci['content_cer']['hi']*100:.1f}]")
    print(f"  assignment loss (named - content)   : {rd['assignment_loss']*100:6.2f}%  "
          f"<- error that is misfiling, not misreading")
    pa = r["placement_accuracy"]
    print(f"\nPlacement accuracy : {pa['correct']}/{pa['total']} = {pa['rate']*100:.1f}% "
          f"of GT fields land under the correct key")
    om = r["omission"]
    print(f"Omission (schema) : {om['schema_omitted']}/{om['schema_segments']} segments "
          f"= {om['schema_omit_rate']*100:.1f}% dropped")
    ha = r["hallucination"]
    print(f"Hallucination     : {ha['unsupported']}/{ha['pred_values']} predicted values "
          f"unsupported by the page ({ha['rate']*100:.1f}%); spurious receiver: {ha['spurious_receiver']}")
    oc = r["orphan_capture"]
    print(f"Orphan capture    : {oc['captured_anyway']}/{oc['orphan_content_segments']} "
          f"schema-homeless segments captured anyway ({oc['capture_rate']*100:.1f}%)")
    print(f"Digit/code runs   : {dict(ordering)}")
    for b, ex in ordering_examples.items():
        if ex:
            print(f"    {b}: {', '.join(ex)}")
    la = r["latency_seconds"]
    print(f"Latency (ok)      : p50 {la['ok_only']['p50']}s  p90 {la['ok_only']['p90']}s  "
          f"p99 {la['ok_only']['p99']}s  mean {la['ok_only']['mean']}s")
    print(f"Latency (all)     : p50 {la['all']['p50']}s  p90 {la['all']['p90']}s  "
          f"p99 {la['all']['p99']}s  mean {la['all']['mean']}s  (422s inflate this)")
    print("  (n=18 distinct texts -> p99 ~ max; CIs are wide by construction)")

    if args.compare and OLD_RESULTS.exists():
        old = json.loads(OLD_RESULTS.read_text(encoding="utf-8"))
        old_all = old["by_source"]["ALL"]
        print("\n" + "-" * 70)
        print("OLD flat-blob scoring  vs  NEW field-aware scoring")
        print("-" * 70)
        print(f"  flat-blob CER (all 5 fields concatenated) : {old_all['cer']*100:6.2f}%")
        print(f"  new named-field CER   (same, decomposed)  : {rd['named_field_cer']*100:6.2f}%")
        print(f"  new content-match CER (reading only)      : {rd['content_matched_cer']*100:6.2f}%")
        print(f"  => of the flat {old_all['cer']*100:.1f}%, about "
              f"{max(0.0,(old_all['cer']-rd['content_matched_cer']))*100:.1f} pts is NOT reading error")

    print(f"\nWrote {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
