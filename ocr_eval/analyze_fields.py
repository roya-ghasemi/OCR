"""Phase 3 — field-assignment structure on the real corpus, without ground truth.

Phase 3 in the brief assumes field misassignment is a large share of error. On
the synthetic corpus that was measured and largely disproved: misassignment
accounts for 14.46% of bilingual CER and 1.43% of Persian-only CER. That
measurement cannot be repeated on the real corpus, because CER needs ground
truth and there is none (D35).

What *can* be measured without a reference is the structural signature of
misassignment — and those signatures are the ones that actually broke this
service:

  * D31 schema collapse   - a header field carrying the whole letter
  * cross-field duplication - the same text emitted into two slots
  * D29 omission          - `contact_info` null on every usable output
  * D36 digit systems     - which digit alphabet each field comes back in
  * identifier integrity  - IBAN / national-ID / phone checksums, which are
                            self-verifying and therefore need no GT at all

The last one is the important one. A checksum is ground truth you already have.

    venv312\\Scripts\\python.exe ocr_eval/analyze_fields.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from harness import API_FIELDS, bootstrap_ci, fmt_ci, similarity  # noqa: E402
from ocr_pipeline.validation import (  # noqa: E402
    extract_identifiers, validate_iban, validate_national_id,
    validate_iranian_mobile,
)

PRED = HERE / "predictions_real.jsonl"
MANIFEST = HERE / "manifest.json"
SPLITS = HERE / "splits_v2.json"
OUT_JSON = HERE / "results_fields_gtfree.json"
OUT_MD = HERE / "reports" / "phase_3.md"

# A header field longer than this is carrying body text, not a header. Chosen
# from the grammar caps in ocr_pipeline/grammar.py (sender/receiver 180,
# subject 240) — the same bound that makes collapse unrepresentable there.
HEADER_MAX = {"sender": 180, "receiver": 180, "subject": 240}

_PERSIAN_INDIC = re.compile(r"[۰-۹]")
_ARABIC_INDIC = re.compile(r"[٠-٩]")
_ASCII_DIGIT = re.compile(r"[0-9]")
_ARABIC_LETTERFORM = re.compile(r"[يكة]")


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    preds = jsonl(PRED)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest = manifest["records"] if isinstance(manifest, dict) else manifest
    manifest = {r["filename"]: r for r in manifest}
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))["assignment"]

    usable = [r for r in preds
              if r.get("http_status") == 200 and isinstance(r.get("raw"), dict)
              and any(isinstance(v, str) and v.strip() for v in r["raw"].values())]

    n_all, n_usable = len(preds), len(usable)

    # --- fill / omission ----------------------------------------------------
    fill = {f: 0 for f in API_FIELDS}
    lengths = {f: [] for f in API_FIELDS}
    for r in usable:
        for f in API_FIELDS:
            v = r["raw"].get(f)
            if isinstance(v, str) and v.strip():
                fill[f] += 1
                lengths[f].append(len(v))

    # --- D31 collapse -------------------------------------------------------
    collapsed_docs, collapse_by_field = [], Counter()
    for r in usable:
        hit = False
        for f, cap in HEADER_MAX.items():
            v = r["raw"].get(f)
            if isinstance(v, str) and len(v) > cap:
                collapse_by_field[f] += 1
                hit = True
        if hit:
            collapsed_docs.append(r["filename"])

    # --- cross-field duplication -------------------------------------------
    dup_docs, dup_pairs = [], Counter()
    for r in usable:
        vals = {f: v.strip() for f, v in r["raw"].items()
                if f in API_FIELDS and isinstance(v, str) and v.strip()}
        hit = False
        names = sorted(vals)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if len(vals[a]) < 12 or len(vals[b]) < 12:
                    continue
                if similarity(vals[a], vals[b]) >= 0.90:
                    dup_pairs[f"{a}~{b}"] += 1
                    hit = True
        if hit:
            dup_docs.append(r["filename"])

    # --- digit systems, per field ------------------------------------------
    digits = {f: Counter() for f in API_FIELDS}
    letterforms = Counter()
    for r in usable:
        for f in API_FIELDS:
            v = r["raw"].get(f)
            if not isinstance(v, str) or not v.strip():
                continue
            if _PERSIAN_INDIC.search(v):
                digits[f]["persian_indic"] += 1
            if _ARABIC_INDIC.search(v):
                digits[f]["arabic_indic"] += 1
            if _ASCII_DIGIT.search(v):
                digits[f]["ascii"] += 1
            if _ARABIC_LETTERFORM.search(v):
                letterforms[f] += 1

    # --- identifier integrity (GT-free: checksums verify themselves) --------
    # Keys returned by extract_identifiers are singular: national_id, iban,
    # mobile, landline. Reading them under plural names silently yields zero
    # identifiers everywhere, which reads as a finding rather than a bug.
    CHECKERS = {"iban": validate_iban, "national_id": validate_national_id,
                "mobile": validate_iranian_mobile}
    ident = {k: [0, 0] for k in CHECKERS}
    ident_docs = set()
    for r in usable:
        blob = "\n".join(v for v in r["raw"].values() if isinstance(v, str))
        found = extract_identifiers(blob)
        for key, checker in CHECKERS.items():
            for v in found.get(key, []):
                ident[key][1] += 1
                ident_docs.add(r["filename"])
                if checker(v):
                    ident[key][0] += 1

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "90 real administrative letters; GT-free structural analysis only",
        "n_documents": n_all,
        "n_usable": n_usable,
        "field_fill": {f: {"n": fill[f], "pct_of_usable": round(fill[f] / max(n_usable, 1), 4)}
                       for f in API_FIELDS},
        "field_length": {f: {"n": len(lengths[f]),
                             "min": min(lengths[f]) if lengths[f] else None,
                             "median": sorted(lengths[f])[len(lengths[f]) // 2] if lengths[f] else None,
                             "max": max(lengths[f]) if lengths[f] else None}
                         for f in API_FIELDS},
        "schema_collapse": {
            "n_documents": len(collapsed_docs),
            "pct_of_usable": round(len(collapsed_docs) / max(n_usable, 1), 4),
            "by_field": dict(collapse_by_field),
            "caps": HEADER_MAX,
        },
        "cross_field_duplication": {
            "n_documents": len(dup_docs),
            "pct_of_usable": round(len(dup_docs) / max(n_usable, 1), 4),
            "pairs": dict(dup_pairs),
        },
        "digit_systems_by_field": {f: dict(digits[f]) for f in API_FIELDS},
        "arabic_letterforms_by_field": dict(letterforms),
        "identifier_integrity": {
            k: {"valid": v[0], "found": v[1],
                "pass_rate": round(v[0] / v[1], 4) if v[1] else None}
            for k, v in ident.items()
        },
        "n_documents_with_identifier": len(ident_docs),
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(f"wrote {OUT_JSON}\nwrote {OUT_MD}")
    print(f"usable={n_usable}/{n_all}  collapse={len(collapsed_docs)}  "
          f"duplication={len(dup_docs)}  identifiers={sum(v[1] for v in ident.values())}")


def write_report(r: dict) -> None:
    L: list[str] = []
    A = L.append
    nu = max(r["n_usable"], 1)
    A("# Phase 3 — field assignment on the real corpus (GT-free)\n")
    A(f"Generated {r['generated_utc']}  ")
    A(f"Scope: {r['scope']}  ")
    A(f"Usable outputs: **{r['n_usable']} / {r['n_documents']}**\n")

    A("## Why this report is GT-free\n")
    A("Phase 3 asks how much of the error is misassignment. That question needs "
      "CER, CER needs ground truth, and the real corpus has none (**D35**). On "
      "synthetic GT it was already measured and largely disproved: misassignment "
      "is 14.46% of bilingual CER and 1.43% of Persian-only CER — real, but not "
      "the dominant term the brief assumed.\n")
    A("What follows measures the *structural signatures* of misassignment, which "
      "need no reference, plus identifier checksums — which are ground truth the "
      "documents carry themselves.\n")

    A("## Field fill\n")
    A("| field | filled | % of usable | min | median | max |")
    A("|---|---:|---:|---:|---:|---:|")
    for f in API_FIELDS:
        ff, fl = r["field_fill"][f], r["field_length"][f]
        A(f"| `{f}` | {ff['n']} | {ff['pct_of_usable']:.1%} | "
          f"{fl['min'] if fl['min'] is not None else '—'} | "
          f"{fl['median'] if fl['median'] is not None else '—'} | "
          f"{fl['max'] if fl['max'] is not None else '—'} |")
    A("")

    sc = r["schema_collapse"]
    A("## D31 — schema collapse\n")
    A(f"A header field longer than its cap is carrying body text. Caps: "
      f"{', '.join(f'`{k}` {v}' for k, v in sc['caps'].items())}.\n")
    A(f"**{sc['n_documents']} of {r['n_usable']} usable outputs "
      f"({sc['pct_of_usable']:.1%})** show collapse.\n")
    if sc["by_field"]:
        A("| field | over cap |")
        A("|---|---:|")
        for k, v in sorted(sc["by_field"].items(), key=lambda kv: -kv[1]):
            A(f"| `{k}` | {v} |")
        A("")

    cd = r["cross_field_duplication"]
    A("## Cross-field duplication\n")
    A(f"Two fields returning near-identical text (similarity ≥ 0.90, both ≥ 12 "
      f"characters). **{cd['n_documents']} of {r['n_usable']} "
      f"({cd['pct_of_usable']:.1%})**.\n")
    if cd["pairs"]:
        A("| field pair | documents |")
        A("|---|---:|")
        for k, v in sorted(cd["pairs"].items(), key=lambda kv: -kv[1]):
            A(f"| {k.replace('~', ' ↔ ')} | {v} |")
        A("")

    A("## Digit systems, per field (D36)\n")
    A("The synthetic corpus contains **zero** Persian-Indic digits, so this table "
      "is the only evidence of what the pipeline actually emits.\n")
    A("| field | Persian-Indic | Arabic-Indic | ASCII |")
    A("|---|---:|---:|---:|")
    for f in API_FIELDS:
        d = r["digit_systems_by_field"][f]
        A(f"| `{f}` | {d.get('persian_indic', 0)} | {d.get('arabic_indic', 0)} | "
          f"{d.get('ascii', 0)} |")
    A("")

    lf = r["arabic_letterforms_by_field"]
    A("## Arabic letterforms in the API response (D8)\n")
    if lf:
        A("| field | documents containing `ي`/`ك`/`ة` |")
        A("|---|---:|")
        for k, v in sorted(lf.items(), key=lambda kv: -kv[1]):
            A(f"| `{k}` | {v} |")
    else:
        A("None. No usable output contains an Arabic yeh, kaf or teh marbuta.")
    A("")

    A("## Identifier integrity — ground truth the documents carry themselves\n")
    A("IBAN (ISO 13616 mod-97), Iranian national ID (mod-11) and mobile numbers "
      "are self-verifying. A failed checksum is a **confirmed** misread with no "
      "reference transcription required.\n")
    A("| identifier | found | checksum-valid | pass rate |")
    A("|---|---:|---:|---:|")
    for k, v in r["identifier_integrity"].items():
        pr = f"{v['pass_rate']:.1%}" if v["pass_rate"] is not None else "—"
        A(f"| {k} | {v['found']} | {v['valid']} | {pr} |")
    A("")
    A(f"Documents containing at least one identifier: "
      f"**{r['n_documents_with_identifier']}**.\n")
    A("A low pass rate here is the strongest accuracy evidence available before "
      "ground truth arrives: these are digits the model demonstrably got wrong.\n")

    A("## PII\n")
    A("This report contains counts and rates only. No field value, name, national "
      "ID, IBAN or phone number from the corpus appears in it.\n")
    OUT_MD.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
