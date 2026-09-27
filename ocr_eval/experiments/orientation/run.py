# -*- coding: utf-8 -*-
r"""E10 — orientation: the stored `_11` (page lying sideways) vs a true 90° rotate.

D51 showed that an EXIF Orientation flag never reaches the model. This runs the
pixel-transposed image through BOTH decoding arms and lays every field next to
the corpus prediction for the same page and next to the human GT.

The rotated image is NOT added to dataset_ex or the manifest: it is an
experiment input, not a corpus document. Outputs stay in this directory.

    venv312\Scripts\python.exe ocr_eval/experiments/orientation/run.py
"""
from __future__ import annotations

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

import httpx  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from run_real import classify  # noqa: E402
from ocr_pipeline.extraction import LetterExtractor  # noqa: E402
from ocr_pipeline.grammar import FIELD_ORDER  # noqa: E402

DOC = "CamScanner ⁨7-1-25 10.26⁩_11.JPG"
ROT = HERE / "CamScanner ⁨7-1-25 10.26⁩_11_rot90.JPG"
GT = EVAL / "ground_truth_real_fixed_v2.jsonl"
OCR_URL = "http://127.0.0.1:8000/ocr"
ENGINE = "http://127.0.0.1:18234/v1"
F = ["sender", "receiver", "subject", "body_text", "contact_info"]


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def fields_of(raw) -> dict:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {}
    raw = raw if isinstance(raw, dict) else {}
    return {f: (raw.get(f) if isinstance(raw.get(f), str) else None) for f in F}


def sim(a: str | None, b: str | None) -> float | None:
    if not a or not b:
        return None
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 3)


def arm_a(path: Path) -> dict:
    t0 = time.perf_counter()
    with httpx.Client(timeout=900.0) as c:
        r = c.post(OCR_URL, files={"file": (path.name, path.read_bytes(), "image/jpeg")})
    secs = round(time.perf_counter() - t0, 3)
    try:
        body = r.json()
    except Exception:
        body = {"_unparseable_response": r.text[:4000]}
    usable, kind = classify(r.status_code, body, None)
    return {"filename": path.name, "http_status": r.status_code, "seconds": secs,
            "usable": usable, "failure_kind": kind,
            "raw": body if usable else None, "failure_body": None if usable else body}


async def arm_b(path: Path) -> dict:
    url = "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()
    c = AsyncOpenAI(base_url=ENGINE, api_key="local", timeout=600.0)
    model = (await c.models.list()).data[0].id
    ex = LetterExtractor(c, model)
    t0 = time.perf_counter()
    res = await ex.extract(url, doc_id=path.name)
    secs = round(time.perf_counter() - t0, 3)
    filled = [k for k in FIELD_ORDER if isinstance(res.data.get(k), str) and res.data[k].strip()]
    return {"filename": path.name, "ok": res.ok, "seconds": secs, "attempts": res.attempts,
            "finish_reason": res.finish_reason, "grammar_used": res.grammar_used,
            "n_fields": len(filled), "raw": res.data, "error": res.error, "model": model}


def main() -> int:
    import config
    gt = next(r for r in jsonl(GT) if r["filename"] == DOC)["fields"]
    corpus_a = fields_of(next(r for r in jsonl(EVAL / "predictions_real.jsonl") if r["filename"] == DOC)["raw"])
    corpus_b = fields_of(next(r for r in jsonl(EVAL / "predictions_real_pipeline.jsonl") if r["filename"] == DOC)["raw"])

    print("arm A on rot90 ...")
    ra = arm_a(ROT)
    print(f"  {ra['seconds']}s usable={ra['usable']}")
    print("arm B on rot90 ...")
    rb = asyncio.run(arm_b(ROT))
    print(f"  {rb['seconds']}s ok={rb['ok']} fields={rb['n_fields']}/5")
    rot_a, rot_b = fields_of(ra["raw"]), fields_of(rb["raw"])

    table = []
    for f in F:
        table.append({
            "field": f,
            "gt_len": len(gt.get(f) or "") if gt.get(f) else 0,
            "armA_sideways_vs_gt": sim(corpus_a[f], gt.get(f)),
            "armA_upright_vs_gt": sim(rot_a[f], gt.get(f)),
            "armB_sideways_vs_gt": sim(corpus_b[f], gt.get(f)),
            "armB_upright_vs_gt": sim(rot_b[f], gt.get(f)),
            "armA_sideways_present": bool(corpus_a[f]), "armA_upright_present": bool(rot_a[f]),
            "armB_sideways_present": bool(corpus_b[f]), "armB_upright_present": bool(rot_b[f]),
        })

    out = {
        "experiment": "E10 orientation — sideways corpus image vs pixel-transposed upright",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "document": DOC,
        "rotated_input": ROT.name,
        "rotation_provenance": json.loads((HERE / "rot90.provenance.json").read_text(encoding="utf-8")),
        "provenance": config.provenance(),
        "similarity_metric": "difflib.SequenceMatcher ratio on raw strings, no normalization",
        "table": table,
        "predictions": {"armA_upright": ra, "armB_upright": rb,
                        "armA_sideways_corpus": corpus_a, "armB_sideways_corpus": corpus_b},
        "gt_fields_present": {f: bool(gt.get(f)) for f in F},
    }
    (HERE / "results_E10.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'field':13s} {'A side':>7s} {'A up':>7s} {'B side':>7s} {'B up':>7s}   (similarity to human GT)")
    for t in table:
        fmt = lambda v: "   -   " if v is None else f"{v:7.3f}"
        print(f"{t['field']:13s} {fmt(t['armA_sideways_vs_gt'])} {fmt(t['armA_upright_vs_gt'])} "
              f"{fmt(t['armB_sideways_vs_gt'])} {fmt(t['armB_upright_vs_gt'])}")
    print(f"\n-> {HERE / 'results_E10.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
