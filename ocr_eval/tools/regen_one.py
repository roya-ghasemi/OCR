# -*- coding: utf-8 -*-
r"""Regenerate the stored prediction for ONE document, in both decoding arms.

Used after `relock_image.py`: when an image is deliberately replaced, every
stored prediction for it was made against the old bytes and is stale. Re-running
the whole corpus would also resample the other 89 documents and break
comparability with every published GT-free number (D41-D46), so this rewrites
exactly the row that must change and leaves the rest byte-identical.

  arm A  POST /ocr on the live service      -> predictions_real.jsonl
  arm B  ocr_pipeline.LetterExtractor       -> predictions_real_pipeline.jsonl

Row construction is imported from / mirrored on the original runners, so a
spliced row is indistinguishable in shape from one the full run would produce.
Each .meta.json gains a `splices` entry naming the document, the time and the
reason — a spliced file is no longer a single-session run and must say so.

Run:  venv312\Scripts\python.exe ocr_eval/tools/regen_one.py "<filename>" --reason "..."
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "ocr_eval"
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(ROOT))

IMAGES = EVAL / "dataset_ex"
PRED_A = EVAL / "predictions_real.jsonl"
PRED_B = EVAL / "predictions_real_pipeline.jsonl"
OCR_URL = "http://127.0.0.1:8000/ocr"
ENGINE = "http://127.0.0.1:18234/v1"
SUFFIX = ".pre_relock.bak"


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                 encoding="utf-8", newline="\n")


def backup(p: Path) -> None:
    b = p.with_name(p.name + SUFFIX)
    if not b.exists() and p.exists():
        shutil.copy2(p, b)


def note_splice(meta_path: Path, target: str, reason: str, extra: dict) -> None:
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.setdefault("splices", []).append({
        "filename": target,
        "spliced_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": reason,
        **extra,
    })
    meta["note_on_provenance"] = (
        "This file is no longer a single uninterrupted run: the rows listed in "
        "`splices` were regenerated later, against re-locked image bytes. "
        "`wall_seconds` and `generated_utc` describe the original run only."
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def summarise(row: dict) -> str:
    data = row.get("raw") or {}
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            data = {}
    if not isinstance(data, dict):
        data = {}
    fields = [k for k in ("sender", "receiver", "subject", "body_text", "contact_info")
              if isinstance(data.get(k), str) and data[k].strip()]
    return f"fields={len(fields)}/5 {fields}"


def run_arm_a(target: str, timeout: float) -> dict:
    import httpx
    from run_real import classify

    p = IMAGES / target
    t0 = time.perf_counter()
    status = body = err = None
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(OCR_URL, files={"file": (p.name, p.read_bytes(), "image/jpeg")})
        status = resp.status_code
        try:
            body = resp.json()
        except Exception:
            body = {"_unparseable_response": resp.text[:4000]}
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    secs = round(time.perf_counter() - t0, 3)
    usable, kind = classify(status, body, err)
    return {
        "filename": p.name, "source": "real", "http_status": status, "error": err,
        "seconds": secs, "usable": usable, "failure_kind": kind,
        "raw": body if usable else None,
        "failure_body": None if usable else body,
    }


async def run_arm_b(target: str) -> dict:
    from openai import AsyncOpenAI
    from ocr_pipeline.extraction import LetterExtractor
    from ocr_pipeline.grammar import FIELD_ORDER

    p = IMAGES / target
    url = f"data:image/jpeg;base64,{base64.b64encode(p.read_bytes()).decode()}"
    client = AsyncOpenAI(base_url=ENGINE, api_key="local", timeout=600.0)
    model = (await client.models.list()).data[0].id
    extractor = LetterExtractor(client, model)

    t = time.perf_counter()
    res = await extractor.extract(url, doc_id=target)
    secs = round(time.perf_counter() - t, 3)
    filled = [k for k in FIELD_ORDER
              if isinstance(res.data.get(k), str) and res.data[k].strip()]
    v = res.validation
    return {
        "filename": target, "source": "real", "ok": res.ok, "seconds": secs,
        "attempts": res.attempts, "finish_reason": res.finish_reason,
        "grammar_used": res.grammar_used,
        "verdict": v.verdict.value if v else None,
        "confidence": v.confidence if v else None,
        "findings": [f.code for f in v.findings] if v else [],
        "n_fields": len(filled), "raw": res.data, "error": res.error,
    }, model


def splice(path: Path, new_row: dict, target: str) -> tuple[dict, int]:
    rows = jsonl(path)
    idx = next((i for i, r in enumerate(rows) if r.get("filename") == target), None)
    if idx is None:
        raise SystemExit(f"{path.name} has no row for {target}")
    old = rows[idx]
    backup(path)
    rows[idx] = new_row
    write_jsonl(path, rows)
    return old, idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("filename")
    ap.add_argument("--reason", default="image re-locked; previous row used stale bytes")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--arm", choices=["a", "b", "both"], default="both")
    args = ap.parse_args()

    target = args.filename
    if not (IMAGES / target).is_file():
        print(f"no such image: {target}")
        return 2

    try:
        import config
        prov = config.provenance()
    except Exception as exc:
        prov = {"error": str(exc)}

    if args.arm in ("a", "both"):
        print(f"arm A: POST {OCR_URL}")
        row = run_arm_a(target, args.timeout)
        old, idx = splice(PRED_A, row, target)
        print(f"  row {idx}: was usable={old.get('usable')} {summarise(old)}")
        print(f"           now usable={row['usable']} {summarise(row)}  {row['seconds']}s")
        rows = jsonl(PRED_A)
        meta = PRED_A.with_suffix(".meta.json")
        note_splice(meta, target, args.reason,
                    {"arm": "A — shipped inline pipeline", "endpoint": OCR_URL,
                     "seconds": row["seconds"], "usable": row["usable"],
                     "previous_usable": old.get("usable"), "provenance": prov})
        m = json.loads(meta.read_text(encoding="utf-8"))
        m["n_usable"] = sum(1 for r in rows if r.get("usable"))
        meta.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  n_usable across the file is now {m['n_usable']}/{len(rows)}")

    if args.arm in ("b", "both"):
        print(f"arm B: {ENGINE} via ocr_pipeline.LetterExtractor")
        row, model = asyncio.run(run_arm_b(target))
        old, idx = splice(PRED_B, row, target)
        print(f"  row {idx}: was ok={old.get('ok')} {summarise(old)}")
        print(f"           now ok={row['ok']} {summarise(row)}  {row['seconds']}s")
        rows = jsonl(PRED_B)
        meta = EVAL / "predictions_real_pipeline.meta.json"
        note_splice(meta, target, args.reason,
                    {"arm": "B — ocr_pipeline.LetterExtractor defaults", "engine": ENGINE,
                     "model": model, "seconds": row["seconds"], "ok": row["ok"],
                     "previous_ok": old.get("ok"), "provenance": prov})
        m = json.loads(meta.read_text(encoding="utf-8"))
        m["n_ok"] = sum(1 for r in rows if r.get("ok"))
        meta.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  n_ok across the file is now {m['n_ok']}/{len(rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
