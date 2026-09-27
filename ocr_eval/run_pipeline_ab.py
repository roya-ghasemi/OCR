"""Experiment — shipped inline pipeline vs `ocr_pipeline`, same 90 real documents.

ONE VARIABLE, stated plainly: this changes the *decoding strategy*, not the
model, the images, or the scoring. Arm A is what `/ocr` runs today
(`repeat_penalty=1.2`, free-form JSON prompt, JSON repair, one retry). Arm B is
`ocr_pipeline.LetterExtractor` defaults (GBNF grammar + DRY + `repeat_penalty=1.1`,
bounded field caps, guardrails).

Why it is worth running: arm A returns 89/90 parseable responses but a mean of
2.58 of 5 fields, with `body_text` present in only 20.2%. The reliability
headline is carried almost entirely by `sender`. The open question is whether
that omission is the price of the repetition penalty or an artefact of the
prompt — arm B changes both together, so a win here is a win for the *strategy*,
not evidence about either knob alone.

GT-free. Everything reported is field presence, shape and self-verifying
checksums. No accuracy claim is made or implied.

    venv312\\Scripts\\python.exe ocr_eval/run_pipeline_ab.py
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from openai import AsyncOpenAI  # noqa: E402

from ocr_pipeline.extraction import LetterExtractor  # noqa: E402
from ocr_pipeline.grammar import FIELD_ORDER  # noqa: E402
from ocr_pipeline.validation import Verdict  # noqa: E402

IMAGES = HERE / "dataset_ex"
MANIFEST = HERE / "manifest.json"
OUT = HERE / "predictions_real_pipeline.jsonl"
META = HERE / "predictions_real_pipeline.meta.json"
ENGINE = "http://127.0.0.1:18234/v1"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--base-url", default=ENGINE)
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest = manifest["records"] if isinstance(manifest, dict) else manifest
    names = sorted(r["filename"] for r in manifest if r["source"] == "real")
    if args.limit:
        names = names[:args.limit]

    client = AsyncOpenAI(base_url=args.base_url, api_key="local", timeout=600.0)
    models = await client.models.list()
    model = models.data[0].id
    extractor = LetterExtractor(client, model)

    print(f"{len(names)} images -> {args.base_url}\nmodel: {Path(model).name}\n")
    records = []
    t0 = time.perf_counter()
    for i, fn in enumerate(names, 1):
        p = IMAGES / fn
        data = p.read_bytes()
        url = f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
        t = time.perf_counter()
        try:
            res = await extractor.extract(url, doc_id=fn)
            secs = round(time.perf_counter() - t, 3)
            filled = [k for k in FIELD_ORDER
                      if isinstance(res.data.get(k), str) and res.data[k].strip()]
            v = res.validation
            records.append({
                "filename": fn, "source": "real", "ok": res.ok,
                "seconds": secs, "attempts": res.attempts,
                "finish_reason": res.finish_reason,
                "grammar_used": res.grammar_used,
                "verdict": v.verdict.value if v else None,
                "confidence": v.confidence if v else None,
                "findings": [f.code for f in v.findings] if v else [],
                "n_fields": len(filled),
                "raw": res.data,
                "error": res.error,
            })
            print(f"[{i:3d}/{len(names)}] {'ok ' if res.ok else 'REJ'} "
                  f"fields={len(filled)}/5 att={res.attempts} {secs:6.2f}s  {fn[-12:]}")
        except Exception as exc:
            secs = round(time.perf_counter() - t, 3)
            records.append({"filename": fn, "source": "real", "ok": False,
                            "seconds": secs, "raw": {}, "n_fields": 0,
                            "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{i:3d}/{len(names)}] ERR {type(exc).__name__}  {fn[-12:]}")

    elapsed = round(time.perf_counter() - t0, 1)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = sum(1 for r in records if r["ok"])
    mean_fields = sum(r["n_fields"] for r in records if r["ok"]) / max(ok, 1)
    META.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arm": "B — ocr_pipeline.LetterExtractor defaults",
        "engine": args.base_url,
        "model": model,
        "sampling_profile": extractor.sampling.name,
        "sampling": extractor.sampling.as_dict(),
        "grammar_active": bool(extractor.grammar_spec) and not extractor._grammar_disabled,
        "n_images": len(records),
        "n_ok": ok,
        "mean_fields_when_ok": round(mean_fields, 3),
        "wall_seconds": elapsed,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{ok}/{len(records)} shipped, mean {mean_fields:.2f} fields, "
          f"{elapsed}s -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
