# -*- coding: utf-8 -*-
r"""Before/after benchmark on the labelled evaluation set (Section 5.2).

For every document the pipeline runs ONCE with the Tesseract layer on. From that
single run two answers are scored:

  before  the primary VLM's numbers as emitted (what /ocr returned without the layer)
  after   the layer's decision: Tesseract's reading when it produced one (confidence
          high or low), the VLM's otherwise. Assumption: downstream prefers the
          classical read on conflict — that is the rule this benchmark measures. An
          `oracle` column (any candidate correct) bounds what a smarter chooser could get.

Metrics per split, with the same normalisation as `ocr_eval` (digits → ASCII,
letterforms folded, ZWNJ → space):
  * CER on free text: micro CER over body_text; and over all five fields
  * numeric accuracy: recall of GT numbers (digit-exact) per field, and precision
  * conflict rate: numeric spans where VLM and Tesseract disagreed
  * latency: p50 / p95 of total and per stage

    venv312\Scripts\python.exe -m ocr_service.benchmark --split dev --label qwen25_dev
    venv312\Scripts\python.exe -m ocr_service.benchmark --split dev --primary coreocr --label coreocr_dev
    venv312\Scripts\python.exe -m ocr_service.benchmark --compare qwen25_dev coreocr_dev
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "ocr_eval"))

import normalize as N  # noqa: E402  (ocr_eval/normalize.py)
from harness import cer  # noqa: E402

from .config import PRIMARY_DEFAULT, SECONDARY_DEFAULT, Settings  # noqa: E402
from .numeric_validator import digits_only, to_ascii_digits  # noqa: E402
import collections  # noqa: E402
import re  # noqa: E402


def _atoms(t):
    return collections.Counter(a for a in re.findall(r"\d+", to_ascii_digits(t or "")) if len(a) >= 3)


_GROUPED = re.compile(r"[0-9۰-۹٠-٩]+(?:[/,،.\-][0-9۰-۹٠-٩]+)+")


def _grouped(t):
    """Whole multi-group numbers — amounts, dates, phone lists. Atoms cannot see a
    truncated amount (`۳۲۱/۰۰۰/۰۰` still yields 321 and 000), so this is scored too."""
    out = []
    for m in _GROUPED.finditer(t or ""):
        v = to_ascii_digits(m.group(0))
        if len(re.sub(r"\D", "", v)) >= 4:
            out.append(v)
    return out


def _grouped_hit(gt, cands):
    if any(c == gt for c in cands):
        return True
    parts = [p for p in re.split(r"[^0-9]", gt) if p]
    return any([p for p in re.split(r"[^0-9]", c) if p] == parts[::-1] for c in cands)

EVAL = ROOT / "ocr_eval" / "eval_set.jsonl"
OUT_DIR = ROOT / "ocr_eval" / "benchmarks"
FIELDS = ["sender", "receiver", "subject", "body_text", "contact_info"]


def pct(x, lo=0, hi=1):
    q = sorted(x)
    return q[min(len(q) - 1, int(len(q) * lo))], q[min(len(q) - 1, int(len(q) * hi))]


def micro_cer(pairs: list[tuple[str, str]]) -> float | None:
    num = den = 0
    for g, p in pairs:
        g, p = N.normalize(g, N.RULES_DEFAULT), N.normalize(p or "", N.RULES_DEFAULT)
        if not g:
            continue
        num += cer(g, p) * len(g); den += len(g)
    return round(num / den, 4) if den else None


def _log(*a):
    print(*a, flush=True)       # stdout is usually redirected to a file; never buffer progress


async def run(args, p, cfg, spec) -> dict:
    await p.start()
    docs = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.split != "all":
        docs = [d for d in docs if d["split"] == args.split]
    if args.limit:
        docs = docs[: args.limit]
    _log(f"{len(docs)} documents, split={args.split}, primary={spec.name}, tesseract="
         f"{'available' if p.tesseract.available() else 'MISSING (after == before)'}")

    per_doc = []
    for i, d in enumerate(docs, 1):
        data = Path(d["image"]).read_bytes()
        t0 = time.perf_counter()
        try:
            res = await p.run(data, "image/jpeg", d["doc_id"])
        except Exception as exc:
            _log(f"[{i:3d}] ERROR {type(exc).__name__}: {exc}")
            per_doc.append({"doc_id": d["doc_id"], "error": str(exc)}); continue
        wall = time.perf_counter() - t0
        r = res.model_dump()
        gt_nums = {(n["field"], n["digits"]) for n in d["numeric_gt"]}
        # E15: the VLM's raw numbers are the `value` of every non-added record; the
        # caller-facing answer is `resolved` (+ added records). Atoms (maximal digit
        # runs) are the segmentation-free unit; GT spans are the legacy one.
        before = {(n["field"], digits_only(n["value"])) for n in r["numeric_fields"] if n["confidence"] != "added"}
        atom_gt = {f: _atoms(d["fields"][f]) for f in FIELDS}
        atom_before = {f: collections.Counter() for f in FIELDS}
        atom_after = {f: collections.Counter() for f in FIELDS}
        for n in r["numeric_fields"]:
            if n["confidence"] != "added":
                atom_before[n["field"]].update(_atoms(n["value"]))
            atom_after[n["field"]].update(_atoms(n.get("resolved") or n["value"]))
        a_gt = sum(sum(c.values()) for c in atom_gt.values())
        a_before = sum(sum((atom_gt[f] & atom_before[f]).values()) for f in FIELDS)
        a_after = sum(sum((atom_gt[f] & atom_after[f]).values()) for f in FIELDS)
        a_em_before = sum(sum(c.values()) for c in atom_before.values())
        a_em_after = sum(sum(c.values()) for c in atom_after.values())
        g_gt = g_before = g_after = 0
        for f in FIELDS:
            cb = _grouped(r["fields"][f]) + [to_ascii_digits(n["value"]) for n in r["numeric_fields"] if n["field"] == f]
            ca = _grouped(r["fields"][f]) + [to_ascii_digits(n.get("resolved") or n["value"]) for n in r["numeric_fields"] if n["field"] == f]
            for gnum in _grouped(d["fields"][f]):
                g_gt += 1
                g_before += _grouped_hit(gnum, cb)
                g_after += _grouped_hit(gnum, ca)
        after, oracle = set(), set()
        for n in r["numeric_fields"]:
            after.add((n["field"], digits_only(n.get("resolved") or n["value"])))
            for c in n["candidates"]:
                oracle.add((n["field"], digits_only(c)))
        per_doc.append({
            "doc_id": d["doc_id"], "split": d["split"], "ok": r["primary"]["ok"], "image": d["image"],
            "numeric_fields": r["numeric_fields"],
            "fields_pred": r["fields"], "fields_gt": d["fields"],
            "n_gt_numbers": len(gt_nums),
            "before_hits": len(gt_nums & before), "after_hits": len(gt_nums & after), "oracle_hits": len(gt_nums & oracle),
            "before_emitted": len(before), "after_emitted": len(after),
            "atoms": {"gt": a_gt, "before": a_before, "after": a_after, "emitted_before": a_em_before, "emitted_after": a_em_after},
            "grouped": {"gt": g_gt, "before": g_before, "after": g_after},
            "numeric_summary": r["numeric_summary"], "timing": r["timing"], "wall_s": round(wall, 3),
            "needs_review": r["needs_review"],
        })
        _log(f"[{i:3d}/{len(docs)}] ok={r['primary']['ok']} numbers gt={len(gt_nums)} before={len(gt_nums & before)} "
             f"after={len(gt_nums & after)} conflicts={r['numeric_summary']['low']} {wall:6.1f}s  {d['doc_id'][-10:]}")
    await p.stop()

    # --- sequential cross-check: swap models on the one GPU ------------------
    if args.secondary == "sequential":
        await sequential_cross_check(per_doc, cfg)

    good = [x for x in per_doc if "error" not in x]
    body = micro_cer([(x["fields_gt"]["body_text"] or "", x["fields_pred"]["body_text"]) for x in good])
    allf = micro_cer([(x["fields_gt"][f] or "", x["fields_pred"][f]) for x in good for f in FIELDS])
    per_field = {f: micro_cer([(x["fields_gt"][f] or "", x["fields_pred"][f]) for x in good]) for f in FIELDS}
    n_gt = sum(x["n_gt_numbers"] for x in good) or 1
    at = {k: sum(x["atoms"][k] for x in good) for k in ("gt", "before", "after", "emitted_before", "emitted_after")}
    gp = {k: sum(x["grouped"][k] for x in good) for k in ("gt", "before", "after")}
    n_spans = sum(x["numeric_summary"]["n_numeric"] for x in good) or 1
    totals = [x["timing"]["total_s"] for x in good]
    summary = {
        "label": args.label, "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": args.split, "n_docs": len(good), "n_errors": len(per_doc) - len(good),
        "usable_rate": round(sum(x["ok"] for x in good) / max(len(good), 1), 4),
        "cer_body_text": body, "cer_all_fields": allf, "cer_per_field": per_field,
        "numeric": {
            "gt_numbers": n_gt, "emitted_before": sum(x["before_emitted"] for x in good),
            "recall_before": round(sum(x["before_hits"] for x in good) / n_gt, 4),
            "recall_after": round(sum(x["after_hits"] for x in good) / n_gt, 4),
            "recall_oracle": round(sum(x["oracle_hits"] for x in good) / n_gt, 4),
            "atom_recall_before": round(at["before"] / max(at["gt"], 1), 4),
            "atom_recall_after": round(at["after"] / max(at["gt"], 1), 4),
            "atom_precision_before": round(at["before"] / max(at["emitted_before"], 1), 4),
            "atom_precision_after": round(at["after"] / max(at["emitted_after"], 1), 4),
            "atoms_gt": at["gt"],
            "grouped_gt": gp["gt"],
            "grouped_intact_before": round(gp["before"] / max(gp["gt"], 1), 4),
            "grouped_intact_after": round(gp["after"] / max(gp["gt"], 1), 4),
            "precision_before": round(sum(x["before_hits"] for x in good) / max(sum(x["before_emitted"] for x in good), 1), 4),
            "conflict_rate": round(sum(x["numeric_summary"]["low"] for x in good) / n_spans, 4),
            "unverified_rate": round(sum(x["numeric_summary"]["unverified"] for x in good) / n_spans, 4),
            "high_rate": round(sum(x["numeric_summary"]["high"] for x in good) / n_spans, 4),
        },
        "latency_s": {"p50_total": round(statistics.median(totals), 2) if totals else None,
                      "p95_total": round(pct(totals, 0, 0.95)[1], 2) if totals else None,
                      "mean_primary": round(statistics.fmean(x["timing"]["primary_s"] for x in good), 2) if good else None,
                      "mean_numeric_validation": round(statistics.fmean(x["timing"]["numeric_validation_s"] for x in good), 3) if good else None,
                      "mean_preprocess": round(statistics.fmean(x["timing"]["preprocess_s"] for x in good), 3) if good else None},
        "needs_review_rate": round(sum(x["needs_review"] for x in good) / max(len(good), 1), 4),
        "cross_check": cross_summary(good) if args.secondary else None,
        "tesseract_available": p.tesseract.available(),
        "provenance": cfg.provenance(),
        "per_doc": per_doc,
    }
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"bench_{args.label}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    print_summary(summary)
    return summary


async def sequential_cross_check(per_doc: list[dict], cfg: Settings) -> None:
    """Phase 2 of `--secondary sequential`: the primary engine is down, VRAM is free;
    start the secondary, read every document again, cross-check offline with the
    same function the online path uses."""
    from .backends import make_backend
    from .numeric_validator import NumericField
    from .pipeline import LETTERFORM_ONLY, compute_cross_check, _run
    from .preprocess import prepare
    from .schemas import CrossCheck
    from ocr_pipeline.persian_text import normalize_extraction
    sec = make_backend(cfg.secondary, cfg, cfg.backend_url_secondary)
    _log(f"\n-- sequential cross-check with {cfg.secondary.name} (primary engine stopped, VRAM released)")
    if hasattr(sec, "start_engine"):
        sec.start_engine()
    await sec.start()
    for i, x in enumerate(per_doc, 1):
        if "error" in x:
            continue
        data = Path(x["image"]).read_bytes()
        if cfg.preprocess_incoming:
            data, _ = prepare(data, max_bytes=cfg.preprocess_max_bytes, fmt=cfg.preprocess_format,
                              max_edge=cfg.preprocess_max_edge)
        ext = await sec.extract(data, "image/jpeg", x["doc_id"])
        nums = [NumericField(**{k: v for k, v in n.items() if k != "bbox"},
                             bbox=tuple(n["bbox"]) if n.get("bbox") else None) for n in x["numeric_fields"]]
        if ext.ok:
            cross = compute_cross_check(x["fields_pred"], normalize_extraction(ext.fields, LETTERFORM_ONLY), nums,
                                        min_digits=cfg.numeric_min_digits, policy=cfg.numeric_mismatch_policy,
                                        secondary_run=_run(ext))
        else:
            cross = CrossCheck(enabled=True, secondary=_run(ext))
        x["cross_check"] = cross.model_dump()
        x["secondary_fields"] = ext.fields
        x["numeric_fields"] = [n.as_dict() for n in nums]     # confidences may have been downgraded
        x["needs_review"] = x["needs_review"] or cross.needs_review
        _log(f"[{i:3d}/{len(per_doc)}] secondary ok={ext.ok} mismatches={len(cross.numeric_mismatches)} "
             f"agree(body)={cross.field_agreement.get('body_text')} {ext.latency_s:6.1f}s  {x['doc_id'][-10:]}")
    await sec.stop()


def cross_summary(good: list[dict]) -> dict | None:
    xs = [x["cross_check"] for x in good if x.get("cross_check")]
    if not xs:
        return None
    ok = [x for x in xs if x.get("secondary") and x["secondary"]["ok"]]
    n_num = sum(len(x["numeric_fields"]) for x in good if x.get("cross_check")) or 1
    agree = {f: round(statistics.fmean(x["field_agreement"].get(f, 0.0) for x in ok), 3) for f in FIELDS} if ok else {}
    return {"n_docs": len(xs), "secondary_ok": len(ok),
            "numeric_mismatch_rate": round(sum(len(x["numeric_mismatches"]) for x in ok) / n_num, 4),
            "docs_flagged": sum(1 for x in xs if x.get("needs_review")),
            "mean_field_agreement": agree,
            "secondary_latency_mean_s": round(statistics.fmean(x["secondary"]["latency_s"] for x in ok), 2) if ok else None}


def print_summary(s: dict) -> None:
    n = s["numeric"]; l = s["latency_s"]
    print(f"\n== {s['label']}  n={s['n_docs']} split={s['split']}  tesseract={'yes' if s['tesseract_available'] else 'NO'}")
    print(f"  usable {s['usable_rate']:.1%} | CER body {s['cer_body_text']:.1%} | CER all fields {s['cer_all_fields']:.1%}")
    print(f"  numeric recall  before {n['recall_before']:.1%}  after {n['recall_after']:.1%}  oracle {n['recall_oracle']:.1%}"
          f"  | precision(before) {n['precision_before']:.1%}")
    print(f"  atom recall     before {n['atom_recall_before']:.1%}  after {n['atom_recall_after']:.1%}  | atom precision before {n['atom_precision_before']:.1%}  after {n['atom_precision_after']:.1%}  (GT atoms {n['atoms_gt']})")
    print(f"  grouped numbers intact  before {n.get('grouped_intact_before', float('nan')):.1%}  after {n.get('grouped_intact_after', float('nan')):.1%}  (GT grouped {n.get('grouped_gt')})")
    print(f"  conflict rate {n['conflict_rate']:.1%} | high {n['high_rate']:.1%} | unverified {n['unverified_rate']:.1%}")
    print(f"  latency p50 {l['p50_total']}s p95 {l['p95_total']}s | primary {l['mean_primary']}s | tesseract {l['mean_numeric_validation']}s")
    if s.get("cross_check"):
        c = s["cross_check"]
        print(f"  cross-check: secondary ok {c['secondary_ok']}/{c['n_docs']} | numeric mismatch {c['numeric_mismatch_rate']:.1%} "
              f"| docs flagged {c['docs_flagged']} | agreement {c['mean_field_agreement']} | secondary {c['secondary_latency_mean_s']}s")


def compare(labels: list[str]) -> None:
    rows = [json.loads((OUT_DIR / f"bench_{lab}.json").read_text(encoding="utf-8")) for lab in labels]
    keys = [("usable", lambda s: f"{s['usable_rate']:.1%}"), ("CER body", lambda s: f"{s['cer_body_text']:.1%}"),
            ("CER all", lambda s: f"{s['cer_all_fields']:.1%}"),
            ("num recall before", lambda s: f"{s['numeric']['recall_before']:.1%}"),
            ("num recall after", lambda s: f"{s['numeric']['recall_after']:.1%}"),
            ("num recall oracle", lambda s: f"{s['numeric']['recall_oracle']:.1%}"),
            ("atom recall before", lambda s: f"{s['numeric'].get('atom_recall_before', float('nan')):.1%}"),
            ("atom recall after", lambda s: f"{s['numeric'].get('atom_recall_after', float('nan')):.1%}"),
            ("atom precision after", lambda s: f"{s['numeric'].get('atom_precision_after', float('nan')):.1%}"),
            ("conflict rate", lambda s: f"{s['numeric']['conflict_rate']:.1%}"),
            ("latency p50", lambda s: f"{s['latency_s']['p50_total']}s"), ("latency p95", lambda s: f"{s['latency_s']['p95_total']}s")]
    md = ["| metric | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for k, f in keys:
        md.append(f"| {k} | " + " | ".join(f(s) for s in rows) + " |")
    text = "\n".join(md)
    print(text)
    (OUT_DIR / f"compare_{'_vs_'.join(labels)}.md").write_text(text + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--primary", choices=["qwen25", "coreocr"], default="qwen25")
    ap.add_argument("--secondary", choices=["sequential", "cpu_offload", "concurrent"], default=None,
                    help="cross-check with the secondary model; `sequential` swaps models on one GPU")
    ap.add_argument("--no-preprocess", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", default=None)
    ap.add_argument("--compare", nargs="+")
    a = ap.parse_args()
    if a.compare:
        compare(a.compare); return 0
    a.label = a.label or f"{a.primary}_{a.split}"
    from .pipeline import OcrPipeline
    spec = PRIMARY_DEFAULT if a.primary == "qwen25" else SECONDARY_DEFAULT
    other = SECONDARY_DEFAULT if a.primary == "qwen25" else PRIMARY_DEFAULT
    cfg = Settings(primary=spec, secondary=other,
                   enable_secondary_model=bool(a.secondary), secondary_mode=a.secondary or "sequential",
                   preprocess_incoming=not a.no_preprocess)
    p = OcrPipeline(cfg)
    p.start_engines()                 # synchronous, before the event loop exists (Windows proactor bug)
    asyncio.run(run(a, p, cfg, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
