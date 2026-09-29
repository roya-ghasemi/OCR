# -*- coding: utf-8 -*-
r"""Benchmark the service's transcription on the real-letter split (E19).

Runs `ocr_service.pipeline.OcrPipeline` in-process (no HTTP, no GPU) over every
document of the split that has GT, scores the full transcript with
`fulltext_score` and the rule-cut letter fields with plain per-field CER, and puts
the previous VLM service's stored predictions for the same documents next to it.

    venv312\Scripts\python.exe ocr_eval\tools\bench_fulltext.py --split dev
    venv312\Scripts\python.exe ocr_eval\tools\bench_fulltext.py --split test   # final, once

Tune on dev only. The test split has been looked at twice before (E17, E18).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
ROOT = EVAL.parent
sys.path[:0] = [str(EVAL), str(ROOT)]

from fulltext_score import FIELDS, aggregate, field_cer, score_doc  # noqa: E402

GT = EVAL / "ground_truth_real_fixed_v2.jsonl"
SPLITS = EVAL / "splits_v2.json"
IMAGES = EVAL / "dataset_ex"
# the previous service's stored per-document predictions (E17/E18), for comparison
BASELINE = {"dev": EVAL / "benchmarks" / "bench_qwen25_dev_glyph2.json",
            "test": EVAL / "benchmarks" / "bench_qwen25_test_amounts.json"}


def _field_totals(rows: list[dict]) -> dict:
    out = {}
    for f in FIELDS:
        e = sum(r["per_field"][f][0] for r in rows if f in r["per_field"])
        n = sum(r["per_field"][f][1] for r in rows if f in r["per_field"])
        if n:
            out[f] = round(e / n, 4)
    out["spurious_fields"] = sum(len(r["spurious"]) for r in rows)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    from ocr_service.config import settings
    from ocr_service.pipeline import OcrPipeline

    split = json.loads(SPLITS.read_text(encoding="utf-8"))["assignment"]
    gts = [json.loads(l) for l in GT.read_text(encoding="utf-8").splitlines() if l.strip()]
    docs = [g for g in gts if split.get(g["filename"]) == a.split and any(g["fields"].get(f) for f in FIELDS)]
    if a.limit:
        docs = docs[:a.limit]

    base = {}
    if BASELINE[a.split].is_file():
        b = json.loads(BASELINE[a.split].read_text(encoding="utf-8"))
        base = {d["doc_id"]: d.get("fields_pred") or {} for d in b.get("per_doc", [])}

    pipe = OcrPipeline(settings)
    health = pipe.health()
    if not health["tesseract"]["available"]:
        print("Tesseract with Persian data is not available:", health["tesseract"], file=sys.stderr)
        return 2

    per_doc, new_rows, base_rows, new_f, base_f, secs = [], [], [], [], [], []
    for i, g in enumerate(docs, 1):
        t = time.perf_counter()
        res = pipe.run_sync((IMAGES / g["filename"]).read_bytes(), g["filename"])
        dt = time.perf_counter() - t
        secs.append(dt)
        s = score_doc(res.text, g["fields"])
        fc = field_cer(res.fields.model_dump(), g["fields"])
        new_rows.append(s)
        new_f.append(fc)
        rec = {"doc_id": g["filename"], "seconds": round(dt, 2), "text": res.text, "fields": res.fields.model_dump(),
               "is_letter": res.is_letter, "needs_review": res.needs_review, "score": s, "field_cer": fc}
        if g["filename"] in base:
            bp = base[g["filename"]]
            bs = score_doc("\n".join(bp.get(f) or "" for f in FIELDS), g["fields"])
            base_rows.append(bs)
            base_f.append(field_cer(bp, g["fields"]))
            rec["baseline_score"] = bs
        per_doc.append(rec)
        print(f"[{i}/{len(docs)}] {g['filename'][-10:]} {dt:5.1f}s  cer={s['err'] / max(s['chars'], 1):.3f}", flush=True)

    summary = {
        "label": f"fulltext_{a.split}", "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": a.split, "n_docs": len(docs),
        "transcript": aggregate(new_rows),
        "fields": _field_totals(new_f),
        "latency_s": {"p50": round(statistics.median(secs), 2), "p95": round(sorted(secs)[int(0.95 * (len(secs) - 1))], 2),
                      "mean": round(statistics.mean(secs), 2)} if secs else {},
        "baseline": {"source": str(BASELINE[a.split].name), "n_docs": len(base_rows),
                     "transcript": aggregate(base_rows) if base_rows else None,
                     "fields": _field_totals(base_f) if base_f else None},
        "provenance": settings.provenance(), "health": health,
        "per_doc": per_doc,
    }
    out = a.out or EVAL / "benchmarks" / f"bench_fulltext_{a.split}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    view = {k: summary[k] for k in ("split", "n_docs", "transcript", "fields", "latency_s")}
    view["baseline"] = {k: summary["baseline"][k] for k in ("n_docs", "transcript", "fields")}
    print(json.dumps(view, ensure_ascii=False, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
