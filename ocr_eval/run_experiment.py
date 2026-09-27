"""Phase 2a — one-variable-at-a-time experiments against the reliability defects.

Restarts the service with a single config knob changed, runs a fixed image set
through `/ocr`, and appends the result to `ocr_eval/experiments.md`. One variable
per invocation: the operating rules forbid moving two at once, and the whole
point of D23 is that the obvious hypothesis (raise max_tokens) needs to be
falsified rather than assumed.

    venv312\\Scripts\\python.exe ocr_eval/run_experiment.py --name baseline
    venv312\\Scripts\\python.exe ocr_eval/run_experiment.py --name repeat_penalty_11 \\
        --env REPEAT_PENALTY=1.1
    venv312\\Scripts\\python.exe ocr_eval/run_experiment.py --name max_tokens_4096 \\
        --env MAX_TOKENS=4096

Scored on the DEV split only. The test split is scored once per phase, never to
choose between options.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

IMAGES = HERE / "dataset_ex"
SPLITS = HERE / "splits_v2.json"
EXPERIMENTS = HERE / "experiments.md"
RESULTS_DIR = HERE / "experiments"
HEALTH = "http://127.0.0.1:8000/health"
OCR = "http://127.0.0.1:8000/ocr"
PYEXE = ROOT / "venv312" / "Scripts" / "python.exe"

REPEAT_THRESHOLD = 20
API_FIELDS = ["sender", "receiver", "subject", "body_text", "contact_info"]


def longest_run(s: str) -> int:
    best = n = 0
    prev = ""
    for c in s or "":
        n = n + 1 if c == prev else 1
        prev = c
        best = max(best, n)
    return best


def stop_service() -> None:
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "$c=Get-NetTCPConnection -LocalPort 8000 -State Listen "
         "-ErrorAction SilentlyContinue; if($c){Stop-Process -Id $c.OwningProcess -Force}; "
         "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force"],
        capture_output=True,
    )
    time.sleep(3)


def start_service(env_overrides: dict[str, str]) -> subprocess.Popen:
    env = {**os.environ, **env_overrides}
    proc = subprocess.Popen(
        [str(PYEXE), "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=ROOT, env=env,
        stdout=(ROOT / "server_stdout.log").open("w"),
        stderr=(ROOT / "server_stderr.log").open("w"),
    )
    for _ in range(120):
        try:
            if httpx.get(HEALTH, timeout=5).status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("service did not become healthy within 600s")


def classify(status, body, err) -> tuple[bool, str | None]:
    if err:
        return False, "timeout" if "timeout" in err.lower() else "transport"
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
    ap.add_argument("--name", required=True, help="experiment id, used in experiments.md")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="config override; pass at most ONE for a clean experiment")
    ap.add_argument("--hypothesis", default="", help="what this run is meant to test")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-restart", action="store_true",
                    help="reuse a running service (only valid with no --env)")
    args = ap.parse_args()

    overrides = dict(kv.split("=", 1) for kv in args.env)
    if len(overrides) > 1:
        print(f"WARNING: {len(overrides)} variables changed at once. "
              "The operating rules require one per experiment.", file=sys.stderr)

    splits = json.loads(SPLITS.read_text(encoding="utf-8"))["assignment"]
    files = sorted(p for p in IMAGES.iterdir() if p.is_file())
    if args.split != "all":
        files = [p for p in files if splits.get(p.name) == args.split]
    if args.limit:
        files = files[: args.limit]

    print(f"experiment '{args.name}' | overrides={overrides or 'none'} | "
          f"{len(files)} images ({args.split} split)")

    if not args.no_restart:
        stop_service()
        start_service(overrides)
    elif overrides:
        sys.exit("--no-restart cannot be combined with --env: the running service "
                 "would not pick the override up, and the result would be mislabelled.")

    recs = []
    with httpx.Client(timeout=900.0) as client:
        for i, p in enumerate(files, 1):
            t0 = time.perf_counter()
            status = body = err = None
            try:
                r = client.post(OCR, files={"file": (p.name, p.read_bytes(), "image/jpeg")})
                status = r.status_code
                try:
                    body = r.json()
                except Exception:
                    body = {"_unparseable": r.text[:4000]}
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
            secs = round(time.perf_counter() - t0, 3)
            usable, kind = classify(status, body, err)
            blob = json.dumps(body or "", ensure_ascii=False)
            recs.append({
                "filename": p.name, "http_status": status, "error": err,
                "seconds": secs, "usable": usable, "failure_kind": kind,
                "longest_char_run": longest_run(blob),
                "raw": body if usable else None,
                "failure_body": None if usable else body,
            })
            print(f"[{i:3d}/{len(files)}] {'ok ' if usable else 'FAIL ' + str(kind):22s} "
                  f"{secs:7.2f}s  run={recs[-1]['longest_char_run']:4d}  {p.name}")

    n = len(recs)
    n_ok = sum(1 for r in recs if r["usable"])
    n_degen = sum(1 for r in recs if r["longest_char_run"] >= REPEAT_THRESHOLD)
    lat = sorted(r["seconds"] for r in recs)
    summary = {
        "name": args.name,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hypothesis": args.hypothesis,
        "overrides": overrides,
        "split": args.split,
        "n": n,
        "n_usable": n_ok,
        "usable_rate": round(n_ok / n, 4) if n else None,
        "failure_taxonomy": dict(Counter(r["failure_kind"] for r in recs if not r["usable"])),
        "n_degenerate_runs": n_degen,
        "pct_degenerate": round(100.0 * n_degen / n, 2) if n else None,
        "latency_p50": lat[len(lat) // 2] if lat else None,
        "latency_p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None,
    }
    try:
        import config
        summary["provenance"] = config.provenance()
    except Exception as exc:
        summary["provenance"] = {"error": str(exc)}

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{args.name}.json").write_text(
        json.dumps({"summary": summary, "records": recs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # experiments.md is append-only.
    with EXPERIMENTS.open("a", encoding="utf-8") as fh:
        fh.write(f"\n### {args.name} — {summary['generated_utc']}\n\n")
        if args.hypothesis:
            fh.write(f"**Hypothesis:** {args.hypothesis}\n\n")
        fh.write(f"**Changed:** `{overrides or 'nothing (baseline)'}` · "
                 f"split `{args.split}` · n={n}\n\n")
        fh.write("| metric | value |\n|---|---|\n")
        fh.write(f"| usable-output rate | **{summary['usable_rate']*100:.2f}%** "
                 f"({n_ok}/{n}) |\n")
        fh.write(f"| failures | {summary['failure_taxonomy'] or 'none'} |\n")
        fh.write(f"| degenerate character runs (>={REPEAT_THRESHOLD}) | "
                 f"{n_degen} ({summary['pct_degenerate']}%) |\n")
        fh.write(f"| latency p50 / p95 | {summary['latency_p50']}s / "
                 f"{summary['latency_p95']}s |\n")
        fh.write(f"\nRaw: `ocr_eval/experiments/{args.name}.json`\n")

    print(f"\nusable {n_ok}/{n} = {summary['usable_rate']*100:.2f}% | "
          f"degenerate {n_degen} ({summary['pct_degenerate']}%) | "
          f"appended to experiments.md")


if __name__ == "__main__":
    main()
