# -*- coding: utf-8 -*-
r"""Final edits to ground_truth_real_fixed_v2.jsonl, in place, with a .bak.

Two changes, both authorised by the annotator:

  1. `_11` is promoted back to the scored fields. It was withheld because its
     image no longer matched the lock; the change was deliberate (EXIF
     orientation), `relock_image.py` has moved the lock, and `regen_one.py` has
     regenerated both arms' predictions against the new bytes, so the row is
     comparable again. `fields_withheld` is emptied into `fields` verbatim and
     the withholding note is dropped.

  2. `_44`'s `full_text` is restored from the delivered transcription
     (`ground_truth_real.jsonl`), reverting `شماره:` -> `تاریخ:` on the reference
     line so the text matches the printed page. The script asserts that this is
     the ONLY difference before writing; if the delivered text differs anywhere
     else, it refuses rather than silently reverting other edits.

`subject_source` is re-stamped on any row it touches. No other row, field or
character is modified.

Run:  venv312\Scripts\python.exe ocr_eval/tools/finalize_gt_real.py
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "ocr_eval"
GT = EVAL / "ground_truth_real_fixed_v2.jsonl"
FLAT = EVAL / "ground_truth_real.jsonl"
MANIFEST = EVAL / "manifest.json"

API_FIELDS = ["sender", "receiver", "subject", "body_text", "contact_info"]
PROMOTE = "_11.JPG"
RESTORE_TEXT = "_44.JPG"


def on_page(value: str, page: str) -> bool:
    def norm(s: str) -> str:
        s = unicodedata.normalize("NFC", s or "")
        s = s.replace("‌", " ").replace("ي", "ی").replace("ك", "ک")
        return re.sub(r"[\s‎‏،,;:.\-|]+", " ", s).strip()
    n = norm(value)
    return bool(n) and n in norm(page)


def main() -> int:
    rows = [json.loads(l) for l in GT.read_text(encoding="utf-8").splitlines() if l.strip()]
    flat = {r["file_name"].lower(): r["text"] for r in
            (json.loads(l) for l in FLAT.read_text(encoding="utf-8").splitlines() if l.strip())}
    manifest = {r["filename"]: r for r in json.loads(MANIFEST.read_text(encoding="utf-8"))}

    promoted = restored = 0
    for r in rows:
        fn = r["filename"]

        if fn.endswith(PROMOTE):
            withheld = r.get("fields_withheld")
            if not withheld:
                print(f"{fn}: nothing withheld; leaving it alone.")
            else:
                # the reason for withholding must actually be gone
                if r["sha256"] != manifest[fn]["sha256"]:
                    print(f"{fn}: row sha256 still disagrees with the manifest. "
                          f"Re-lock first; refusing to promote.")
                    return 1
                r["fields"] = {f: withheld.get(f) for f in API_FIELDS}
                r.pop("fields_withheld", None)
                r["notes"] = ""
                subj = r["fields"].get("subject")
                if isinstance(subj, str) and subj.strip():
                    r["subject_source"] = ("printed" if on_page(subj, r.get("full_text", ""))
                                           else "annotator")
                promoted += 1

        if fn.endswith(RESTORE_TEXT):
            delivered = flat.get(fn.lower())
            if delivered is None:
                print(f"{fn}: no delivered transcription; refusing.")
                return 1
            current = r.get("full_text", "")
            if current == delivered:
                print(f"{fn}: full_text already matches the delivered transcription.")
            else:
                a, b = current.split("\n"), delivered.split("\n")
                diff = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
                if len(a) != len(b) or len(diff) != 1:
                    print(f"{fn}: expected exactly one differing line, found "
                          f"{len(diff)} (lines {len(a)} vs {len(b)}). Refusing to "
                          f"revert edits this tool was not asked to make.")
                    for i, x, y in diff[:5]:
                        print(f"    line {i}: {x!r}  ->  {y!r}")
                    return 1
                i, was, now = diff[0]
                r["full_text"] = delivered
                restored += 1
                print(f"{fn}: full_text line {i} restored")
                print(f"    was {was!r}")
                print(f"    now {now!r}")

    shutil.copy2(GT, GT.with_name(GT.name + ".pre_finalize.bak"))
    GT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                  encoding="utf-8", newline="\n")

    transcribed = [r for r in rows if any(r["fields"][f] for f in API_FIELDS)]
    printed = sum(1 for r in rows if r.get("subject_source") == "printed")
    annot = sum(1 for r in rows if r.get("subject_source") == "annotator")
    withheld_left = sum(1 for r in rows if "fields_withheld" in r)
    print(f"\nwrote {GT.name}  rows={len(rows)}")
    print(f"  promoted={promoted}  full_text restored={restored}  still withheld={withheld_left}")
    print(f"  transcribed={len(transcribed)}  untranscribed={len(rows) - len(transcribed)}")
    print(f"  subject: printed={printed}  annotator={annot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
