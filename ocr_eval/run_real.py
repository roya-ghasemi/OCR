"""Phase 2 — run the production endpoint over the 90 real documents.

This runs WITHOUT ground truth. Everything it measures is GT-free and therefore
unblocked by defect D15:

  * usable-output rate and the full failure taxonomy (Phase 2 exit criterion)
  * latency percentiles, failures included and excluded (1.7)
  * letterform conformance -- Arabic yeh/kaf in the API output (D8)
  * digit-system census -- Persian-Indic vs Arabic-Indic vs ASCII (Phase 4)
  * JSON validity and schema conformance

It also stores every prediction, so the moment ground truth arrives the accuracy
metrics can be computed by re-scoring rather than re-running inference.

    venv312\\Scripts\\python.exe ocr_eval/run_real.py
    venv312\\Scripts\\python.exe ocr_eval/run_real.py --limit 5      # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import normalize as N  # noqa: E402

OCR_URL = "http://127.0.0.1:8000/ocr"
IMAGES = HERE / "dataset_ex"
OUT = HERE / "predictions_real.jsonl"
API_FIELDS = ["sender", "receiver", "subject", "body_text", "contact_info"]


def classify(status: int | None, body, err: str | None) -> tuple[bool, str | None]:
    """Usable output = 2xx carrying at least one non-empty field.

    An all-null 200 counts as a FAILURE. The service reports success, nothing
    downstream notices, and that is exactly what makes D4 the more dangerous of
    the two reliability defects.
    """
    if err:
        low = err.lower()
        if "timeout" in low or "timed out" in low:
            return False, "timeout"
        return False, "transport"
    if status == 422:
        return False, "http_422"
    if status is None or status >= 500:
        return False, "http_5xx"
    if status >= 400:
        return False, f"http_{status}"
    if not isinstance(body, dict):
        return False, "non_object_200"
    if all(body.get(f) is None or (isinstance(body.get(f), str) and not body[f].strip())
           for f in API_FIELDS):
        return False, "all_null_200"
    return True, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop after N images")
    ap.add_argument("--url", default=OCR_URL)
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    files = sorted(p for p in IMAGES.iterdir() if p.is_file())
    if args.limit:
        files = files[: args.limit]

    print(f"{len(files)} images -> {args.url}")
    records = []
    t_start = time.time()
    with httpx.Client(timeout=args.timeout) as client:
        for i, p in enumerate(files, 1):
            t0 = time.perf_counter()
            status = body = err = None
            try:
                resp = client.post(
                    args.url,
                    files={"file": (p.name, p.read_bytes(), "image/jpeg")},
                )
                status = resp.status_code
                try:
                    body = resp.json()
                except Exception:
                    body = {"_unparseable_response": resp.text[:4000]}
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
            secs = round(time.perf_counter() - t0, 3)

            usable, kind = classify(status, body, err)
            rec = {
                "filename": p.name,
                "source": "real",
                "http_status": status,
                "error": err,
                "seconds": secs,
                "usable": usable,
                "failure_kind": kind,
                "raw": body if usable else None,
                # The failing body is kept in full: Phase 2a cannot tell token
                # exhaustion from a repetition loop without it.
                "failure_body": None if usable else body,
            }
            records.append(rec)
            flag = "ok " if usable else f"FAIL {kind}"
            print(f"[{i:3d}/{len(files)}] {flag:22s} {secs:7.2f}s  {p.name}")

            # Written incrementally: a crash at image 80 must not lose 79 runs.
            with OUT.open("w", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = round(time.time() - t_start, 1)
    n_ok = sum(1 for r in records if r["usable"])
    print(f"\n{n_ok}/{len(records)} usable in {elapsed}s -> {OUT}")

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoint": args.url,
        "n_images": len(records),
        "n_usable": n_ok,
        "wall_seconds": elapsed,
        "normalizer_version": N.NORMALIZER_VERSION,
    }
    try:
        import config
        meta["provenance"] = config.provenance()
    except Exception as exc:
        meta["provenance"] = {"error": str(exc)}
    (HERE / "predictions_real.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
