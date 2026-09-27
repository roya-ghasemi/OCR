"""Phase 7 — CI regression gate. Ground-truth-free.

Every check here answers "did this change break something we already fixed?"
None of them measure accuracy: accuracy needs ground truth, and the real corpus
has none (see GT_SPEC.md). What they do measure is the set of failures that
actually took this service down — decoding loops, schema collapse, Arabic
letterforms leaking into the API response, all-null 200s — plus the unit suites.

That distinction is the point. A gate that cannot run without GT is a gate that
never runs. This one runs on every commit and would have caught D3, D23, D31 and
D8 before they shipped.

    venv312\\Scripts\\python.exe ocr_eval/ci_gate.py                 # enforce
    venv312\\Scripts\\python.exe ocr_eval/ci_gate.py --update-baseline
    venv312\\Scripts\\python.exe ocr_eval/ci_gate.py --offline       # tests only

Exit codes: 0 pass, 1 regression, 2 could not run.

The live sample is drawn from the DEV split only. The test split is scored once
per phase by hand and must never be consumed by CI.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import normalize as N  # noqa: E402

BASELINE = HERE / "ci_baseline.json"
SPLITS = HERE / "splits_v2.json"
MANIFEST = HERE / "manifest.json"
IMAGES = HERE / "dataset_ex"
OCR_URL = "http://127.0.0.1:8000/ocr"
API_FIELDS = ["sender", "receiver", "subject", "body_text", "contact_info"]

# Unit suites that must stay green. Paths are relative to the repo root.
TEST_SUITES = [
    "tests/test_ocr_pipeline.py",
    "ocr_eval/test_normalize.py",
    "ocr_eval/test_harness.py",
    "ocr_eval/test_phase0_guards.py",
    "test_json_repair.py",
]

# How many dev documents the live sample uses. Small enough to run per-commit.
SAMPLE_N = 12


class Check:
    """One gate condition. `worse` decides the direction of a regression."""

    def __init__(self, name, value, floor=None, ceiling=None, unit="", note=""):
        self.name, self.value = name, value
        self.floor, self.ceiling = floor, ceiling
        self.unit, self.note = unit, note

    @property
    def passed(self) -> bool:
        if self.value is None:
            return False
        if self.floor is not None and self.value < self.floor:
            return False
        if self.ceiling is not None and self.value > self.ceiling:
            return False
        return True

    def render(self) -> str:
        v = "n/a" if self.value is None else f"{self.value:.4g}{self.unit}"
        bound = ""
        if self.floor is not None:
            bound = f" (floor {self.floor:.4g}{self.unit})"
        if self.ceiling is not None:
            bound = f" (ceiling {self.ceiling:.4g}{self.unit})"
        flag = "PASS" if self.passed else "FAIL"
        return f"  [{flag}] {self.name:<34} {v:>10}{bound}  {self.note}"


# ---------------------------------------------------------------------------
# offline checks
# ---------------------------------------------------------------------------

def run_tests() -> list[Check]:
    """Unit suites. A suite that cannot even be collected counts as a failure."""
    checks = []
    py = sys.executable
    for suite in TEST_SUITES:
        p = ROOT / suite
        if not p.exists():
            checks.append(Check(f"tests:{suite}", None, floor=1,
                                note="suite missing"))
            continue
        r = subprocess.run([py, "-m", "pytest", str(p), "-q", "--no-header"],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0:
            checks.append(Check(f"tests:{Path(suite).stem}", 1.0, floor=1.0,
                                note="green"))
        else:
            tail = (r.stdout or r.stderr).strip().splitlines()
            checks.append(Check(f"tests:{Path(suite).stem}", 0.0, floor=1.0,
                                note=tail[-1][:70] if tail else "failed"))
    return checks


def check_corpus_locks() -> list[Check]:
    """The manifest and split locks must still match. A silent corpus change
    invalidates every historical comparison in this repo."""
    import hashlib

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()

    out = []
    for name, f, lock in (("manifest", MANIFEST, HERE / "manifest.sha256"),
                          ("splits_v2", SPLITS, HERE / "splits_v2.sha256")):
        if not (f.exists() and lock.exists()):
            out.append(Check(f"lock:{name}", None, floor=1.0, note="file missing"))
            continue
        match = sha(f) == lock.read_text().strip().split()[0]
        out.append(Check(f"lock:{name}", 1.0 if match else 0.0, floor=1.0,
                         note="hash matches" if match else "HASH CHANGED"))
    return out


# ---------------------------------------------------------------------------
# live checks
# ---------------------------------------------------------------------------

def dev_sample(n: int) -> list[str]:
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))["assignment"]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest = manifest["records"] if isinstance(manifest, dict) else manifest
    real_dev = sorted(r["filename"] for r in manifest
                      if r["source"] == "real" and splits.get(r["filename"]) == "dev")
    if not real_dev:
        return []
    # Deterministic spread across the split rather than the first n, so the
    # sample is not dominated by one scanning session.
    step = max(1, len(real_dev) // n)
    return real_dev[::step][:n]


def run_live(url: str, names: list[str], timeout: float) -> list[Check]:
    try:
        import httpx
    except ImportError:
        return [Check("live:httpx", None, floor=1.0, note="httpx not installed")]

    from ocr_pipeline.sampling import detect_degeneracy
    from ocr_pipeline.persian_text import letterform_conformance

    ok = 0
    latencies: list[float] = []
    degenerate = 0
    all_null = 0
    collapsed = 0
    arabic_forms = 0
    field_counts: list[int] = []
    errors: list[str] = []

    with httpx.Client(timeout=timeout) as client:
        for fn in names:
            p = IMAGES / fn
            if not p.is_file():
                errors.append(f"missing image {fn}")
                continue
            t0 = time.perf_counter()
            try:
                with p.open("rb") as fh:
                    resp = client.post(url, files={"file": (fn, fh, "image/jpeg")})
            except Exception as exc:
                errors.append(f"{fn}: {type(exc).__name__}")
                continue
            latencies.append(time.perf_counter() - t0)
            if resp.status_code != 200:
                continue
            try:
                body = resp.json()
            except Exception:
                continue
            data = body.get("extraction", body)
            if not isinstance(data, dict):
                continue
            vals = {k: data.get(k) for k in API_FIELDS}
            filled = [k for k, v in vals.items() if isinstance(v, str) and v.strip()]
            if not filled:
                all_null += 1
                continue
            ok += 1
            field_counts.append(len(filled))

            blob = "\n".join(v for v in vals.values() if isinstance(v, str))
            if detect_degeneracy(blob).is_degenerate:
                degenerate += 1
            # D31: the whole letter dumped into one header field.
            for hdr in ("sender", "receiver", "subject"):
                v = vals.get(hdr)
                if isinstance(v, str) and len(v) > 200:
                    collapsed += 1
                    break
            # letterform_conformance takes an ITERABLE of values and reports
            # `n_non_conforming`. Passing a single string iterates it character
            # by character and reading a key it does not return yields 0 — a
            # false green, which is worse than no check at all.
            if letterform_conformance(
                [v for v in vals.values() if isinstance(v, str)]
            )["n_non_conforming"]:
                arabic_forms += 1

    n = len(names) or 1
    p95 = (statistics.quantiles(latencies, n=20)[-1]
           if len(latencies) >= 20 else (max(latencies) if latencies else None))
    return [
        Check("live:usable_output_rate", ok / n, floor=None, unit="",
              note=f"{ok}/{n} returned at least one field"),
        Check("live:degeneracy_rate", degenerate / max(ok, 1), ceiling=None,
              note=f"{degenerate} repetition loop(s) in shipped output"),
        Check("live:all_null_200_rate", all_null / n, ceiling=None,
              note=f"{all_null} response(s) were 200 with every field null"),
        Check("live:field_collapse_rate", collapsed / max(ok, 1), ceiling=None,
              note=f"{collapsed} header field(s) over 200 chars"),
        Check("live:arabic_letterform_rate", arabic_forms / max(ok, 1), ceiling=None,
              note=f"{arabic_forms} response(s) still contain Arabic yeh/kaf"),
        Check("live:mean_fields_filled",
              statistics.fmean(field_counts) if field_counts else None, floor=None,
              note=f"of {len(API_FIELDS)} fields"),
        Check("live:latency_p95_s", p95, ceiling=None, unit="s",
              note=f"n={len(latencies)}"),
        Check("live:transport_errors", float(len(errors)), ceiling=0.0,
              note="; ".join(errors[:2]) or "none"),
    ]


# ---------------------------------------------------------------------------
# baseline plumbing
# ---------------------------------------------------------------------------

# Direction and slack for each metric. Slack absorbs sampling noise on a
# 12-document sample; without it the gate cries wolf on every run.
DIRECTION = {
    "live:usable_output_rate":      ("floor", 0.10),
    "live:mean_fields_filled":      ("floor", 0.50),
    "live:degeneracy_rate":         ("ceiling", 0.10),
    "live:all_null_200_rate":       ("ceiling", 0.05),
    "live:field_collapse_rate":     ("ceiling", 0.10),
    "live:arabic_letterform_rate":  ("ceiling", 0.10),
    "live:latency_p95_s":           ("ceiling", 0.50),
    "live:transport_errors":        ("ceiling", 0.0),
}


def apply_baseline(checks: list[Check], base: dict) -> None:
    for c in checks:
        if c.name.startswith(("tests:", "lock:")):
            c.floor = 1.0
            continue
        recorded = base.get("metrics", {}).get(c.name)
        if recorded is None:
            c.note += "  [no baseline — not enforced]"
            continue
        kind, slack = DIRECTION.get(c.name, ("floor", 0.0))
        if kind == "floor":
            c.floor = max(0.0, recorded - slack)
        else:
            c.ceiling = recorded + slack


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--offline", action="store_true",
                    help="skip live checks (no engine required)")
    ap.add_argument("--url", default=OCR_URL)
    ap.add_argument("--sample", type=int, default=SAMPLE_N)
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    print(f"CI gate — {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    checks = check_corpus_locks() + run_tests()

    if not args.offline:
        names = dev_sample(args.sample)
        if not names:
            checks.append(Check("live:sample", None, floor=1.0,
                                note="no real dev documents found"))
        else:
            print(f"  live sample: {len(names)} dev documents -> {args.url}")
            checks += run_live(args.url, names, args.timeout)

    base = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}

    if args.update_baseline:
        metrics = {c.name: c.value for c in checks
                   if not c.name.startswith(("tests:", "lock:")) and c.value is not None}
        BASELINE.write_text(json.dumps({
            "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sample_n": args.sample,
            "normalizer_version": N.VERSION if hasattr(N, "VERSION") else None,
            "note": ("GT-free metrics only. These are reliability and shape "
                     "checks, not accuracy. Re-record deliberately, never to "
                     "make a red gate green."),
            "metrics": metrics,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nBaseline written to {BASELINE}")
        for c in checks:
            print(c.render())
        return 0

    apply_baseline(checks, base)
    print()
    for c in checks:
        print(c.render())
    failed = [c for c in checks if not c.passed]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed.")
    if failed:
        print("REGRESSION:")
        for c in failed:
            print(f"  - {c.name}: {c.value} violates "
                  f"floor={c.floor} ceiling={c.ceiling}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
