# -*- coding: utf-8 -*-
r"""E11 — how many documents have a session-dependent `body_text`? (sizes D52)

Runs `ocr_pipeline.LetterExtractor` defaults (arm B) over the deterministic
10-document dev sample used by ci_gate, once per PROCESS, and compares the runs
field by field. Within a process the engine was seen to be exactly repeatable
(D52), so the unit of repetition is the process, not the request.

    venv312\Scripts\python.exe ocr_eval/experiments/repeat_stability/run.py --out run1.jsonl
    venv312\Scripts\python.exe ocr_eval/experiments/repeat_stability/run.py --out run2.jsonl
    venv312\Scripts\python.exe ocr_eval/experiments/repeat_stability/run.py --compare run1.jsonl run2.jsonl

The stored corpus prediction (`predictions_real_pipeline.jsonl`, 2026-09-03) is
a third, earlier process and is included in the comparison for free.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import difflib
import json
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
ENGINE = "http://127.0.0.1:18234/v1"
CAP = 1400


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def fields_of(raw) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    return {f: (raw.get(f) if isinstance(raw.get(f), str) else None) for f in F}


async def run(names: list[str], out: Path) -> None:
    from openai import AsyncOpenAI
    from ocr_pipeline.extraction import LetterExtractor
    import config

    c = AsyncOpenAI(base_url=ENGINE, api_key="local", timeout=600.0)
    model = (await c.models.list()).data[0].id
    ex = LetterExtractor(c, model)
    rows = []
    t_all = time.perf_counter()
    for i, fn in enumerate(names, 1):
        p = EVAL / "dataset_ex" / fn
        url = "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()
        t0 = time.perf_counter()
        res = await ex.extract(url, doc_id=fn)
        secs = round(time.perf_counter() - t0, 3)
        rows.append({"filename": fn, "ok": res.ok, "seconds": secs, "attempts": res.attempts,
                     "finish_reason": res.finish_reason, "raw": res.data, "error": res.error})
        body = (res.data.get("body_text") or "") if isinstance(res.data, dict) else ""
        print(f"[{i:2d}/{len(names)}] ok={res.ok} body={len(body):4d}{' CAP' if len(body) >= CAP - 10 else '    '} "
              f"{secs:6.2f}s  {fn[-10:]}")
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                   encoding="utf-8", newline="\n")
    out.with_suffix(".meta.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arm": "B — ocr_pipeline.LetterExtractor defaults", "engine": ENGINE, "model": model,
        "sampling": ex.sampling.as_dict(), "n": len(rows),
        "wall_seconds": round(time.perf_counter() - t_all, 1),
        "provenance": config.provenance(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


def sim(a, b):
    if a is None and b is None:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def compare(paths: list[Path]) -> None:
    runs = {p.stem: {r["filename"]: fields_of(r["raw"]) for r in jsonl(p)} for p in paths}
    corpus = {r["filename"]: fields_of(r["raw"]) for r in jsonl(EVAL / "predictions_real_pipeline.jsonl")}
    runs["corpus_2026-09-03"] = corpus
    names = sorted(set.intersection(*[set(v) for v in runs.values()]))
    labels = list(runs)
    pairs = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))]

    per_field = {f: {} for f in F}
    unstable_docs = {f: set() for f in F}
    for f in F:
        for a, b in pairs:
            sims = [sim(runs[a][n][f], runs[b][n][f]) for n in names]
            per_field[f][f"{a} vs {b}"] = round(sum(sims) / len(sims), 3)
            for n, s in zip(names, sims):
                if s < 0.9:
                    unstable_docs[f].add(n)

    cap_hits = {lab: sum(1 for n in names if len(runs[lab][n]["body_text"] or "") >= CAP - 10)
                for lab in labels}

    print(f"\nE11 — {len(names)} documents, {len(labels)} processes: {labels}\n")
    print(f"{'field':13s} " + " ".join(f"{a[:6]}~{b[:6]:<6s}" for a, b in pairs) + "  docs<0.9")
    for f in F:
        vals = " ".join(f"{per_field[f][f'{a} vs {b}']:13.3f}" for a, b in pairs)
        print(f"{f:13s} {vals}  {len(unstable_docs[f]):2d}/{len(names)}")
    print(f"\nbody_text at the {CAP}-char cap, per process: {cap_hits}")
    print("documents with a session-dependent body_text:",
          [n.split('_')[-1] for n in sorted(unstable_docs['body_text'])])

    out = {
        "experiment": "E11 repeat stability across processes (D52)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "documents": names, "processes": labels,
        "mean_similarity_per_field": per_field,
        "unstable_documents_per_field": {f: sorted(v) for f, v in unstable_docs.items()},
        "body_cap_hits_per_process": cap_hits,
        "similarity_metric": "difflib.SequenceMatcher ratio on raw strings; None==None counts as 1.0, one-sided None as 0.0",
    }
    (HERE / "results_E11.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {HERE / 'results_E11.json'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", nargs="+", type=Path)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    if args.compare:
        compare([HERE / p if not p.is_absolute() else p for p in args.compare])
        return 0
    if not args.out:
        ap.error("--out or --compare required")
    from ci_gate import dev_sample
    names = dev_sample(args.n)
    print(f"{len(names)} dev documents (ci_gate.dev_sample) -> {args.out}")
    asyncio.run(run(names, HERE / args.out if not args.out.is_absolute() else args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
