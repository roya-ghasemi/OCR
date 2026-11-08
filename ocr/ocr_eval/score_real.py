"""Validate and score ground truth for the 90 real administrative letters.

This is the single command that clears the GT blocker. It does two jobs:

  --validate-only   run the §2.2 GT-validation protocol and write
                    ocr_eval/gt_real_audit.md. Exits non-zero if the file
                    cannot be scored against.

  (default)         validate, then score stored predictions against the GT and
                    write ocr_eval/results_real.json plus
                    ocr_eval/reports/phase_real_baseline.md.

Nothing here re-runs inference. Scoring is a pure function of
(predictions file, GT file, normalizer rules), so a baseline can be recomputed
for any stored prediction set without touching the engine.

PII: this script reads real letters. It writes counts, rates and redacted
excerpts only -- never a full field value into a report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import normalize as N  # noqa: E402
from harness import (  # noqa: E402
    API_FIELDS, DocOutcome, score_document, score_slice, config_hash,
)

GT_REAL = HERE / "ground_truth_real.jsonl"
MANIFEST = HERE / "manifest.json"
SPLITS = HERE / "splits_v2.json"
DEFAULT_PRED = HERE / "predictions_real.jsonl"
AUDIT = HERE / "gt_real_audit.md"
RESULTS = HERE / "results_real.json"
REPORT = HERE / "reports" / "phase_real_baseline.md"

# A field name may be listed in `illegible`; anything else is a typo we should
# catch rather than silently ignore.
ROW_KEYS = {"filename", "sha256", "source", "language_mode", "split",
            "fields", "illegible", "notes"}

_BIDI = re.compile(r"[‎‏‪-‮⁦-⁩]")
_HARAKAT = re.compile(r"[ً-ْ]")
_ASCII_D = re.compile(r"[0-9]")
_ARABIC_INDIC = re.compile(r"[٠-٩]")
_PERSIAN_INDIC = re.compile(r"[۰-۹]")
_ARABIC_LETTERFORMS = "يكة"       # ARABIC YEH / KAF / TEH MARBUTA
_PLACEHOLDER = re.compile(r"\[\s*(ناخوان[اa]?|illegible|todo|xxx)\s*\]", re.I)


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 2.2 -- GT validation
# ---------------------------------------------------------------------------

class Audit:
    """Accumulates findings. `fatal` means the file cannot be scored against."""

    def __init__(self) -> None:
        self.fatal: list[str] = []
        self.warn: list[str] = []
        self.info: dict = {}

    def ok(self) -> bool:
        return not self.fatal


def validate_gt(gt_path: Path) -> tuple[Audit, list[dict]]:
    a = Audit()
    if not gt_path.exists():
        a.fatal.append(
            f"{gt_path.name} does not exist. Fill in gt_real_template.jsonl and "
            f"rename it. See GT_SPEC.md."
        )
        return a, []

    try:
        rows = jsonl(gt_path)
    except json.JSONDecodeError as exc:
        a.fatal.append(f"{gt_path.name} is not valid JSONL: {exc}")
        return a, []

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest = manifest["records"] if isinstance(manifest, dict) else manifest
    real = {r["filename"]: r for r in manifest if r["source"] == "real"}
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))["assignment"]

    a.info["n_rows"] = len(rows)
    a.info["n_expected"] = len(real)

    # --- reconciliation against the hash-locked manifest ---------------------
    seen: Counter = Counter()
    unknown, hash_mismatch, split_mismatch = [], [], []
    for r in rows:
        fn = r.get("filename")
        seen[fn] += 1
        m = real.get(fn)
        if m is None:
            unknown.append(fn)
            continue
        if r.get("sha256") and r["sha256"] != m["sha256"]:
            hash_mismatch.append(fn)
        if r.get("split") and r["split"] != splits.get(fn):
            split_mismatch.append(fn)

    dupes = [f for f, c in seen.items() if c > 1]
    missing = sorted(set(real) - set(seen))

    if unknown:
        a.fatal.append(
            f"{len(unknown)} row(s) name a file not in the real corpus, e.g. "
            f"{unknown[0]!r}. GT rows must correspond 1:1 to manifest records."
        )
    if hash_mismatch:
        a.fatal.append(
            f"{len(hash_mismatch)} row(s) carry a sha256 that does not match the "
            f"manifest. Either the image changed or the GT was written against a "
            f"different file. Scoring is refused."
        )
    if dupes:
        a.fatal.append(f"{len(dupes)} filename(s) appear more than once: {dupes[:3]}")
    if split_mismatch:
        a.warn.append(
            f"{len(split_mismatch)} row(s) disagree with splits_v2.json on `split`. "
            f"The locked split wins; the GT column is ignored."
        )
    if missing:
        a.warn.append(
            f"{len(missing)} of {len(real)} documents have no GT row. They are "
            f"excluded from every number and counted as not-yet-transcribed."
        )

    for r in rows:
        extra = set(r) - ROW_KEYS
        if extra:
            a.warn.append(f"{r.get('filename')}: unrecognised key(s) {sorted(extra)}")
            break

    # --- transcription state -------------------------------------------------
    def is_blank(v):
        return v is None or (isinstance(v, str) and not v.strip())

    transcribed, untranscribed, empty_string = [], [], []
    for r in rows:
        f = r.get("fields") or {}
        if not isinstance(f, dict):
            a.fatal.append(f"{r.get('filename')}: `fields` is not an object.")
            continue
        bad = set(f) - set(API_FIELDS)
        if bad:
            a.fatal.append(f"{r.get('filename')}: unknown field name(s) {sorted(bad)}.")
        if any(isinstance(v, str) and v == "" for v in f.values()):
            empty_string.append(r["filename"])
        (untranscribed if all(is_blank(v) for v in f.values()) else transcribed
         ).append(r["filename"])

    if empty_string:
        a.warn.append(
            f"{len(empty_string)} row(s) use \"\" where GT_SPEC requires null. "
            f"Empty string means 'present but empty'; null means 'absent'. They "
            f"score differently."
        )

    a.info["transcribed"] = len(transcribed)
    a.info["untranscribed"] = len(untranscribed)

    if not transcribed:
        a.fatal.append(
            "No row has any field transcribed. The file is a template, not "
            "ground truth."
        )
    elif len(transcribed) < 20:
        a.warn.append(
            f"Only {len(transcribed)} document(s) transcribed. Below ~20 the "
            f"confidence interval on any headline is too wide to act on."
        )

    # --- coverage, per field -------------------------------------------------
    cov = {}
    for fld in API_FIELDS:
        n = sum(1 for r in rows if not is_blank((r.get("fields") or {}).get(fld)))
        cov[fld] = n
    a.info["coverage"] = cov
    never = [f for f, n in cov.items() if n == 0 and transcribed]
    if never:
        a.warn.append(
            f"Field(s) {never} are null in every transcribed row. They cannot be "
            f"scored and will be reported as unscoreable, not as 100% correct."
        )

    # --- encoding audit ------------------------------------------------------
    enc = Counter()
    for r in rows:
        blob = "\n".join(v for v in (r.get("fields") or {}).values() if isinstance(v, str))
        if not blob:
            continue
        if any(c in blob for c in _ARABIC_LETTERFORMS):
            enc["arabic_letterform"] += 1
        if unicodedata.normalize("NFC", blob) != blob:
            enc["not_nfc"] += 1
        if "ـ" in blob:
            enc["tatweel"] += 1
        if _BIDI.search(blob):
            enc["bidi_control"] += 1
        if _HARAKAT.search(blob):
            enc["harakat"] += 1
        if _ASCII_D.search(blob):
            enc["ascii_digit"] += 1
        if _ARABIC_INDIC.search(blob):
            enc["arabic_indic_digit"] += 1
        if _PERSIAN_INDIC.search(blob):
            enc["persian_indic_digit"] += 1
        if _PLACEHOLDER.search(blob):
            enc["placeholder_text"] += 1
    a.info["encoding"] = dict(enc)

    if enc.get("placeholder_text"):
        a.warn.append(
            f"{enc['placeholder_text']} row(s) contain a placeholder like "
            f"[ناخوانا] inside `fields`. GT_SPEC asks for a best-effort reading "
            f"plus an `illegible` entry instead; a placeholder scores as real text."
        )
    if enc.get("not_nfc"):
        a.warn.append(f"{enc['not_nfc']} row(s) are not NFC-normalised.")

    # --- duplicate GT bodies -------------------------------------------------
    bodies: dict[str, list[str]] = {}
    for r in rows:
        blob = "\n".join(v for v in (r.get("fields") or {}).values() if isinstance(v, str))
        if blob.strip():
            bodies.setdefault(hashlib.sha256(blob.encode()).hexdigest()[:12], []).append(
                r["filename"])
    dgroups = {k: v for k, v in bodies.items() if len(v) > 1}
    a.info["distinct_gt_texts"] = len(bodies)
    a.info["duplicate_groups"] = len(dgroups)
    if dgroups:
        a.warn.append(
            f"{len(dgroups)} group(s) of documents share identical GT text. "
            f"Effective sample size is {len(bodies)}, not {len(transcribed)}; "
            f"confidence intervals are computed over groups as well as images."
        )

    # --- illegible -----------------------------------------------------------
    ill = Counter()
    for r in rows:
        for f in (r.get("illegible") or []):
            if f not in API_FIELDS:
                a.warn.append(f"{r['filename']}: `illegible` names unknown field {f!r}.")
            else:
                ill[f] += 1
    a.info["illegible"] = dict(ill)

    return a, rows


def write_audit(a: Audit, gt_path: Path) -> None:
    L: list[str] = []
    A = L.append
    A("# GT validation — 90 real administrative letters (§2.2)\n")
    A(f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}  ")
    A(f"File: `{gt_path.name}`  ")
    A(f"Verdict: **{'USABLE' if a.ok() else 'NOT USABLE'}**\n")

    if a.fatal:
        A("## Blocking\n")
        for m in a.fatal:
            A(f"- {m}")
        A("")
    if a.warn:
        A("## Warnings — scoring proceeds, but these bound the interpretation\n")
        for m in a.warn:
            A(f"- {m}")
        A("")

    i = a.info
    if i:
        A("## Counts\n")
        A(f"- rows: {i.get('n_rows')} of {i.get('n_expected')} expected")
        A(f"- transcribed: {i.get('transcribed')}   not yet transcribed: "
          f"{i.get('untranscribed')}")
        A(f"- distinct GT texts: {i.get('distinct_gt_texts')} "
          f"(duplicate groups: {i.get('duplicate_groups')})\n")

        if i.get("coverage"):
            A("### Per-field coverage\n")
            A("| field | rows with a value |")
            A("|---|---:|")
            for f in API_FIELDS:
                n = i["coverage"].get(f, 0)
                A(f"| `{f}` | {n} |")
            A("")
        if i.get("encoding"):
            A("### Encoding audit (documents affected)\n")
            A("| property | n |")
            A("|---|---:|")
            for k, v in sorted(i["encoding"].items()):
                A(f"| {k} | {v} |")
            A("")
        if i.get("illegible"):
            A("### Fields marked illegible\n")
            for k, v in sorted(i["illegible"].items()):
                A(f"- `{k}`: {v}")
            A("")
    AUDIT.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def classify_failure(rec: dict) -> tuple[bool, str | None]:
    """Mirrors score_v2.classify_failure so the two runners agree.

    Two row shapes are accepted. Arm A (`run_real.py`) carries `http_status`;
    arm B (`run_pipeline_ab.py`, the `ocr_pipeline` path) carries `ok` and no
    HTTP layer at all. A row with neither is not a prediction.
    """
    if rec.get("error"):
        return False, "transport"
    if "http_status" in rec:
        if rec.get("http_status") != 200:
            return False, f"http_{rec.get('http_status')}"
    elif "ok" in rec:
        if not rec.get("ok"):
            return False, "pipeline_rejected"
    else:
        return False, "unknown_row_shape"
    raw = rec.get("raw")
    if not isinstance(raw, dict):
        return False, "no_object"
    if all(v is None or (isinstance(v, str) and not v.strip()) for v in raw.values()):
        return False, "all_null_200"
    return True, None


def build_docs(rows: list[dict], pred_path: Path, rules: N.Rules) -> list[DocOutcome]:
    preds = {r["filename"]: r for r in jsonl(pred_path)}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest = manifest["records"] if isinstance(manifest, dict) else manifest
    manifest = {r["filename"]: r for r in manifest}
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))["assignment"]

    def is_blank(v):
        return v is None or (isinstance(v, str) and not v.strip())

    docs = []
    for row in rows:
        fn = row["filename"]
        gt_fields = row.get("fields") or {}
        if all(is_blank(v) for v in gt_fields.values()):
            continue                      # not yet transcribed
        rec = preds.get(fn)
        if rec is None:
            continue                      # no stored prediction for this image
        m = manifest.get(fn)
        if m is None:
            continue
        usable, kind = classify_failure(rec)
        SUBJECT_SOURCE[fn] = row.get("subject_source")
        docs.append(DocOutcome(
            filename=fn,
            source="real",
            language_mode=m["language_mode"],
            template_id=m.get("template_id") or "none",
            split=splits.get(fn, "unassigned"),
            usable=usable,
            failure_kind=kind,
            seconds=rec.get("seconds"),
            fields=score_document(fn, rec.get("raw") if usable else None,
                                  gt_fields, rules),
            raw_pred=rec.get("raw") if isinstance(rec.get("raw"), dict) else None,
        ))
    return docs


# `subject_source` is a non-scored GT key: "printed" when the subject annotation
# is also on the page, "annotator" when the annotator supplied it. Sliced on so
# `subject` can be read as an OCR number and as an extraction number separately.
SUBJECT_SOURCE: dict[str, str | None] = {}

SLICES = [
    ("real", lambda d: True),
    ("real|dev", lambda d: d.split == "dev"),
    ("real|test", lambda d: d.split == "test"),
    ("real|bilingual", lambda d: d.language_mode == "bilingual"),
    ("real|persian_only", lambda d: d.language_mode == "persian_only"),
    ("real|subject_printed", lambda d: SUBJECT_SOURCE.get(d.filename) == "printed"),
    ("real|subject_annotator", lambda d: SUBJECT_SOURCE.get(d.filename) == "annotator"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", type=Path, default=GT_REAL)
    ap.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--label", default="baseline",
                    help="name for this scoring run, recorded in the results file")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all",
                    help="score only this split. Use `dev` while tuning; `test` "
                         "is scored once, at the end, and never tuned against.")
    ap.add_argument("--out", type=Path, default=None,
                    help="results file (default results_real.json; with --split, "
                         "results_real.<label>.json so a dev run never "
                         "overwrites a full one)")
    args = ap.parse_args()

    audit, rows = validate_gt(args.gt)
    write_audit(audit, args.gt)
    print(f"GT validation -> {AUDIT}")
    for m in audit.fatal:
        print("  BLOCKING:", m)
    for m in audit.warn:
        print("  warning :", m)
    if not audit.ok():
        print("\nGT is not usable. Nothing was scored.")
        return 2
    if args.validate_only:
        print(f"\nGT is usable: {audit.info['transcribed']} document(s) transcribed.")
        return 0

    if not args.predictions.exists():
        print(f"\nNo prediction file at {args.predictions}. Run ocr_eval/run_real.py first.")
        return 3

    rules = N.RULES_DEFAULT
    docs = build_docs(rows, args.predictions, rules)
    if args.split != "all":
        docs = [d for d in docs if d.split == args.split]
    if not docs:
        print("\nNo document has both a transcribed GT row and a stored prediction.")
        return 4

    pred_meta = args.predictions.with_suffix(".meta.json")
    payload = {
        "label": args.label,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest_sha256": sha256_file(MANIFEST),
        "splits_v2_sha256": sha256_file(SPLITS),
        "gt_sha256": sha256_file(args.gt),
        "predictions_source": args.predictions.name,
        "predictions_meta": (json.loads(pred_meta.read_text(encoding="utf-8"))
                             if pred_meta.exists() else None),
        "n_scored": len(docs),
        "split_filter": args.split,
        "n_gt_rows": len(rows),
        "n_untranscribed": audit.info.get("untranscribed"),
        "gt_audit": audit.info,
        "scope_note": (
            f"Scored {len(docs)} REAL documents. Synthetic results live in "
            f"results_v2.json and are never pooled with these."
        ),
        "slices": {},
    }
    for label, pred in SLICES:
        sel = [d for d in docs if pred(d)]
        if sel:
            payload["slices"][label] = score_slice(sel, label, rules)
    payload["config_hash"] = config_hash(
        {"gt": payload["gt_sha256"], "pred": payload["predictions_source"],
         "rules": N.rules_dict(rules)})

    out = args.out or (RESULTS if args.split == "all"
                       else RESULTS.with_name(f"results_real.{args.label}.json"))
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nScored {len(docs)} real documents (split={args.split}) -> {out}")
    for label in payload["slices"]:
        h = payload["slices"][label]["headline"]
        uor = h["usable_output_rate"]["by_image"]
        ec = h["effective_cer"]["by_image"]
        print(f"  {label:<20} n={h['n_documents']:<3} "
              f"usable={uor['point']:.2%} [{uor['lo']:.2%},{uor['hi']:.2%}]  "
              f"effective_CER={ec['point']:.2%} [{ec['lo']:.2%},{ec['hi']:.2%}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
