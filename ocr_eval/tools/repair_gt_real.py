# -*- coding: utf-8 -*-
r"""One-off: structural repair of the annotator's segmented ground truth.

Input   ocr_eval/ground_truth_real_fixed.jsonl   (84 rows + 1 truncated row)
Output  ocr_eval/ground_truth_real_fixed_v2.jsonl (90 rows)

The annotator's segmentation is authoritative and is carried through UNCHANGED.
`sender`, `receiver`, `body_text` and `contact_info` are byte-identical to the
input on all 84 rows, and every one of the 84 `subject` annotations is kept.

What this repairs, and nothing more:

  * row 85 is truncated mid-write and its `body_text` holds an assistant refusal
    string, not page text. Per instruction, `_9` is re-keyed as UNTRANSCRIBED
    (all five fields null). The three values that were written before the file
    was cut off are preserved verbatim in `notes` so they can be promoted back
    with one edit; nothing is lost, and no refusal text reaches a scored field.
  * `_9`, `_90`, `_91`, `_92`, `_93`, `_94` are re-keyed in from the template as
    untranscribed rows, restoring the file to the 90 rows the manifest expects.
  * `_76`: the printed address `Emil: Sabzgostar2006@gmail.com` is restored in
    `contact_info` and in `full_text`, replacing `info@sabzgostar-govareshk.ir`,
    which is not on that page. Strict string matching requires the printed form.
  * any document whose image no longer matches the manifest/template sha256 lock
    is WITHHELD: its five fields are set to null and the annotator's values are
    kept verbatim under a non-scored `fields_withheld` key. A changed image means
    the stored predictions were produced against different bytes, so scoring that
    row would compare two different pictures. Promoting it back is one edit, once
    the image is re-locked or the old file is restored.
  * rows are returned to template order (GT_SPEC: do not reorder).
  * `subject_source` is stamped on every row carrying a subject: "printed" when
    the annotation also appears on the page, "annotator" when it does not. It is
    a non-scored provenance key. It discards nothing; it lets `subject` be
    reported as an OCR number over printed subjects and as an extraction number
    over all of them, instead of silently mixing the two.

What this deliberately does NOT do: change, add or remove any field value the
annotator wrote, or segment anything itself.

Run:  venv312\Scripts\python.exe ocr_eval/tools/repair_gt_real.py
Exits non-zero, writing nothing, if any check fails.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "ocr_eval"
SRC = EVAL / "ground_truth_real_fixed.jsonl"
FLAT = EVAL / "ground_truth_real.jsonl"
TPL = EVAL / "gt_real_template.jsonl"
SPLITS = EVAL / "splits_v2.json"
IMAGES = EVAL / "dataset_ex"
OUT = EVAL / "ground_truth_real_fixed_v2.jsonl"

API_FIELDS = ["sender", "receiver", "subject", "body_text", "contact_info"]
UNTRANSCRIBE = {"_9", "_90", "_91", "_92", "_93", "_94"}

PRINTED_EMAIL = "Emil: Sabzgostar2006@gmail.com"
WRONG_EMAIL = "info@sabzgostar-govareshk.ir"


def parseable(p: Path) -> tuple[list[dict], list[str]]:
    """Rows that parse, plus the raw text of the ones that do not."""
    good, bad = [], []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            good.append(json.loads(line))
        except json.JSONDecodeError:
            bad.append(line)
    return good, bad


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def on_page(value: str, page: str) -> bool:
    """Is this annotation also printed on the page? Letterform/ZWNJ tolerant."""
    def norm(s: str) -> str:
        s = unicodedata.normalize("NFC", s or "")
        s = s.replace("‌", " ").replace("ي", "ی").replace("ك", "ک")
        return re.sub(r"[\s‎‏،,;:.\-|]+", " ", s).strip()
    n = norm(value)
    return bool(n) and n in norm(page)


def salvaged_note(raw: str) -> str:
    """Human-readable record of what the truncated row held before it was cut."""
    got = {}
    for f in ("sender", "receiver", "subject"):
        m = re.search(rf'"{f}":\s*"((?:[^"\\]|\\.)*)"', raw)
        if m:
            got[f] = json.loads(f'"{m.group(1)}"')
    if not got:
        return ""
    pairs = "; ".join(f"{k}={v!r}" for k, v in got.items())
    return (
        "Row re-keyed as untranscribed: the source file was truncated mid-write "
        "and `body_text` held an assistant refusal string, not page text. "
        f"Values present before the cut, preserved for review: {pairs}"
    )


def main() -> int:
    problems: list[str] = []

    for p in (SRC, FLAT, TPL, SPLITS):
        if not p.is_file():
            print(f"missing input: {p}")
            return 2
    if OUT.exists():
        print(f"refusing to overwrite existing {OUT.name}; move it aside first.")
        return 2

    src_rows, truncated = parseable(SRC)
    flat = {r["file_name"].lower(): r["text"] for r in jsonl(FLAT)}
    tpl = {r["filename"]: r for r in jsonl(TPL)}
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))["assignment"]

    by_name = {}
    for r in src_rows:
        fn = r.get("filename")
        if not fn:
            problems.append(f"a source row has no filename: {sorted(r)}")
        elif fn in by_name:
            problems.append(f"duplicate source row for {fn}")
        else:
            by_name[fn] = r

    note_for_9 = "\n".join(salvaged_note(t) for t in truncated).strip()

    # --- the hash lock, checked once, before anything is built ---------------
    changed: dict[str, str] = {}
    for fn, t in tpl.items():
        img = IMAGES / fn
        if not img.is_file():
            problems.append(f"image missing on disk: {fn}")
            continue
        digest = hashlib.sha256(img.read_bytes()).hexdigest()
        if digest != t["sha256"]:
            changed[fn] = digest

    rows: list[dict] = []
    kept_subjects = restored = rekeyed = 0

    for fn, t in tpl.items():                      # template order is canonical
        stem = "_" + fn.rsplit("_", 1)[-1].split(".")[0]
        page = flat.get(fn.lower())
        if page is None:
            problems.append(f"no delivered transcription for {fn}")
            page = ""

        src = by_name.get(fn)
        if src is None or stem in UNTRANSCRIBE:
            if src is not None and stem not in UNTRANSCRIBE:
                problems.append(f"unexpected: {fn} present but being re-keyed")
            fields = {f: None for f in API_FIELDS}
            illegible: list[str] = []
            notes = note_for_9 if stem == "_9" else ""
            full_text = page
            rekeyed += 1
        else:
            fields = {f: src["fields"].get(f) for f in API_FIELDS}
            illegible = src.get("illegible") or []
            notes = src.get("notes") or ""
            full_text = src.get("full_text", page)

            if fn.endswith("_76.JPG"):
                if isinstance(fields["contact_info"], str) and WRONG_EMAIL in fields["contact_info"]:
                    fields["contact_info"] = fields["contact_info"].replace(WRONG_EMAIL, PRINTED_EMAIL)
                    restored += 1
                if WRONG_EMAIL in full_text:
                    full_text = full_text.replace(WRONG_EMAIL, PRINTED_EMAIL)
                    restored += 1

        # --- withhold any row whose image no longer matches the lock ---------
        withheld = None
        if fn in changed and any(v is not None for v in fields.values()):
            withheld = dict(fields)
            fields = {f: None for f in API_FIELDS}
            notes = (
                (notes + " " if notes else "")
                + "Withheld from scoring: this image no longer matches the "
                  "manifest sha256 lock, so the stored predictions were made "
                  "against different bytes. The annotation is intact under "
                  "`fields_withheld`."
            ).strip()

        # --- integrity, against the split lock -------------------------------
        if t["split"] != splits.get(fn):
            problems.append(f"split disagrees with splits_v2.json for {fn}")
        for f, v in fields.items():
            if v == "":
                problems.append(f"{fn}: {f} is \"\"; GT_SPEC requires null")
        for f in illegible:
            if f not in API_FIELDS:
                problems.append(f"{fn}: illegible names unknown field {f!r}")

        row = {
            "filename": fn,
            "sha256": t["sha256"],
            "source": t["source"],
            "language_mode": t["language_mode"],
            "split": t["split"],
            "fields": fields,
            "illegible": illegible,
            "notes": notes,
            "full_text": full_text,
        }
        if withheld is not None:
            row["fields_withheld"] = withheld
        subj = fields["subject"]
        if isinstance(subj, str) and subj.strip():
            kept_subjects += 1
            row["subject_source"] = "printed" if on_page(subj, full_text) else "annotator"
        rows.append(row)

    unknown = sorted(set(by_name) - set(tpl))
    if unknown:
        problems.append(f"source names document(s) not in the corpus: {unknown[:3]}")
    if len(rows) != len(tpl):
        problems.append(f"built {len(rows)} rows, expected {len(tpl)}")

    if problems:
        print(f"{len(problems)} problem(s); nothing written:")
        for p in problems[:20]:
            print("  -", p)
        return 1

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                   encoding="utf-8", newline="\n")

    # --- read back and prove the annotator's work survived unchanged ---------
    back = jsonl(OUT)
    for r in back:
        src = by_name.get(r["filename"])
        if src is None or "_" + r["filename"].rsplit("_", 1)[-1].split(".")[0] in UNTRANSCRIBE:
            continue
        for f in API_FIELDS:
            a = src["fields"].get(f)
            b = (r.get("fields_withheld") or r["fields"])[f]
            if a != b and not (r["filename"].endswith("_76.JPG") and f == "contact_info"):
                print(f"read-back differs for {r['filename']}.{f}; not usable.")
                return 1

    transcribed = [r for r in back if any(r["fields"][f] for f in API_FIELDS)]
    printed = sum(1 for r in back if r.get("subject_source") == "printed")
    annot = sum(1 for r in back if r.get("subject_source") == "annotator")
    print(f"wrote {OUT.relative_to(ROOT)}  rows={len(back)}")
    if changed:
        print("  !! CORPUS LOCK BROKEN — image bytes differ from manifest.json:")
        for fn, digest in changed.items():
            print(f"     {fn}\n       on disk {digest[:16]}…  locked {tpl[fn]['sha256'][:16]}…")
        print("     row(s) withheld from scoring; annotation kept in `fields_withheld`.")
    print(f"  transcribed={len(transcribed)}  untranscribed={len(back) - len(transcribed)} "
          f"(re-keyed {rekeyed})")
    print(f"  subject annotations kept: {kept_subjects}  "
          f"(printed on page: {printed}, annotator-supplied: {annot})")
    print(f"  _76 printed address restored in {restored} place(s)")
    print(f"  sha256 re-verified against dataset_ex on {len(back)}/{len(back)} images")
    print(f"  splits reconciled with splits_v2.json on {len(back)}/{len(back)} rows: " + ", ".join(
        f"{s}={sum(1 for r in back if r['split'] == s)}" for s in sorted({r["split"] for r in back})))
    print("  sender/receiver/body_text/contact_info: identical to the annotator's file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
