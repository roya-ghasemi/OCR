# -*- coding: utf-8 -*-
r"""E16 — does `cache_prompt=false` make `body_text` session-stable? (closes D52)

E11 measured the OLD path (coreOCR on :18234 via `ocr_pipeline.LetterExtractor`)
and found `body_text` differs between processes on 7 of 10 documents at
temperature 0. The hypothesis was llama-server's prompt/KV cache carrying state
across requests. This runs the SHIPPED path (Qwen2.5-VL through `ocr_service`)
with `cache_prompt` as the only variable, once per fresh PROCESS.

Note the reason this test is possible only now: `Settings.cache_prompt` existed
and was stamped into provenance, but nothing put it on the wire (D61). The flag
now reaches the engine through `LetterExtractor(extra_body=...)`.

    venv312\Scripts\python.exe ocr_eval/experiments/repeat_stability/run_service.py --out svc_cache_on_1.jsonl  --cache-prompt 1
    venv312\Scripts\python.exe ocr_eval/experiments/repeat_stability/run_service.py --out svc_cache_on_2.jsonl  --cache-prompt 1
    venv312\Scripts\python.exe ocr_eval/experiments/repeat_stability/run_service.py --out svc_cache_off_1.jsonl --cache-prompt 0
    venv312\Scripts\python.exe ocr_eval/experiments/repeat_stability/run_service.py --out svc_cache_off_2.jsonl --cache-prompt 0
    venv312\Scripts\python.exe ocr_eval/experiments/repeat_stability/run_service.py --compare-pairs svc_cache_on_1.jsonl:svc_cache_on_2.jsonl svc_cache_off_1.jsonl:svc_cache_off_2.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVAL = ROOT / "ocr_eval"
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(ROOT))

F = ["sender", "receiver", "subject", "body_text", "contact_info"]


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


async def _run(names: list[str], out: Path, cache_prompt: bool) -> None:
    from ocr_service.config import PRIMARY_DEFAULT, SECONDARY_DEFAULT, Settings
    from ocr_service.pipeline import OcrPipeline

    cfg = Settings(primary=PRIMARY_DEFAULT, secondary=SECONDARY_DEFAULT, cache_prompt=cache_prompt)
    p = OcrPipeline(cfg)
    p.start_engines()
    await p.start()
    rows = []
    t_all = time.perf_counter()
    for i, fn in enumerate(names, 1):
        data = (EVAL / "dataset_ex" / fn).read_bytes()
        t0 = time.perf_counter()
        res = await p.run(data, "image/jpeg", fn)
        secs = round(time.perf_counter() - t0, 3)
        # `fields` here is AFTER reconciliation; the raw model text is what D52 is
        # about, so both are stored and the comparison uses the raw one.
        rows.append({"filename": fn, "ok": res.primary.ok, "seconds": secs,
                     "fields": res.fields.model_dump(),
                     "numeric": [n.model_dump() for n in res.numeric_fields]})
        body = res.fields.body_text or ""
        print(f"[{i:2d}/{len(names)}] ok={res.primary.ok} body={len(body):4d} {secs:6.2f}s  {fn[-10:]}", flush=True)
    await p.stop()
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8", newline="\n")
    out.with_suffix(".meta.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cache_prompt": cache_prompt, "n": len(rows),
        "wall_seconds": round(time.perf_counter() - t_all, 1),
        "provenance": cfg.provenance(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


def sim(a, b) -> float:
    if a is None and b is None:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def compare_pairs(pairs: list[str]) -> None:
    out = {"experiment": "E16 cache_prompt vs session stability (D52)",
           "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "arms": {}}
    for spec in pairs:
        a_name, b_name = spec.split(":")
        A = {r["filename"]: r for r in jsonl(HERE / a_name)}
        B = {r["filename"]: r for r in jsonl(HERE / b_name)}
        names = sorted(set(A) & set(B))
        meta = json.loads((HERE / a_name).with_suffix(".meta.json").read_text(encoding="utf-8"))
        per_field, unstable = {}, {}
        for f in F:
            sims = [sim(A[n]["fields"][f], B[n]["fields"][f]) for n in names]
            per_field[f] = round(statistics.fmean(sims), 3)
            unstable[f] = [n for n, s in zip(names, sims) if s < 0.9]
        label = f"cache_prompt={meta['cache_prompt']}"
        out["arms"][label] = {"runs": [a_name, b_name], "n": len(names),
                              "mean_similarity": per_field,
                              "documents_below_0.9": {f: v for f, v in unstable.items() if v},
                              "n_below_0.9": {f: len(v) for f, v in unstable.items()}}
        print(f"\n== {label}  ({a_name} vs {b_name}, n={len(names)})")
        for f in F:
            print(f"   {f:13s} {per_field[f]:.3f}   docs<0.9 {len(unstable[f])}/{len(names)}")
    (HERE / "results_E16.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {HERE / 'results_E16.json'}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--cache-prompt", type=int, choices=[0, 1])
    ap.add_argument("--compare-pairs", nargs="+")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    if a.compare_pairs:
        compare_pairs(a.compare_pairs); return 0
    if not a.out or a.cache_prompt is None:
        ap.error("--out and --cache-prompt, or --compare-pairs")
    from ci_gate import dev_sample
    names = dev_sample(a.n)
    print(f"{len(names)} dev documents, cache_prompt={bool(a.cache_prompt)} -> {a.out}")
    asyncio.run(_run(names, HERE / a.out if not a.out.is_absolute() else a.out, bool(a.cache_prompt)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
