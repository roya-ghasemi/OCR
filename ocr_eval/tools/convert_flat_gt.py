# -*- coding: utf-8 -*-
r"""One-off: re-key the delivered flat ground truth into the GT_SPEC row schema.

Input   ocr_eval/ground_truth_real.jsonl   {"file_name": "...jpg", "text": "..."}
Output  ocr_eval/ground_truth_real_fixed.jsonl

What this does, and nothing more:

  * `file_name` -> `filename`, with the exact manifest casing restored (`.JPG`);
  * re-attaches `sha256`, `source`, `language_mode`, `split` from the hash-locked
    template, so `score_real.py` can reconcile rows against `manifest.json`;
  * restores the template's row ORDER (GT_SPEC: do not reorder);
  * carries the transcription through VERBATIM in a non-scored `full_text` key,
    so the human segmenting it has the page text on the row it belongs to;
  * leaves all five `fields` null — i.e. every row is "not yet transcribed".

What this deliberately does NOT do: split `full_text` into sender / receiver /
subject / body_text / contact_info. That split is a human judgement about what
the pixels say. Inferring it here would be manufacturing ground truth from a
guess, which is the one thing the evaluation may never do (Gate 0, D35).

`full_text` is not in ROW_KEYS, so the validator emits exactly one non-fatal
"unrecognised key" warning. Drop the key once segmentation is done and the
warning goes with it.

Run:  venv312\Scripts\python.exe ocr_eval/tools/convert_flat_gt.py
Exits non-zero, writing nothing, if any check fails.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "ocr_eval"
FLAT = EVAL / "ground_truth_real.jsonl"
TPL = EVAL / "gt_real_template.jsonl"
IMAGES = EVAL / "dataset_ex"
OUT = EVAL / "ground_truth_real_fixed.jsonl"

API_FIELDS = ["sender", "receiver", "subject", "body_text", "contact_info"]


def jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    problems: list[str] = []

    if not FLAT.is_file():
        print(f"missing input: {FLAT}")
        return 2
    if OUT.exists():
        print(f"refusing to overwrite existing {OUT.name}; move it aside first.")
        return 2

    flat = jsonl(FLAT)
    tpl = jsonl(TPL)

    # --- index the flat delivery by lowercased name --------------------------
    by_lower: dict[str, dict] = {}
    for r in flat:
        name = r.get("file_name") or r.get("filename")
        if not name:
            problems.append(f"a flat row has no file_name key: {sorted(r)}")
            continue
        if not isinstance(r.get("text"), str):
            problems.append(f"{name}: `text` is not a string")
            continue
        key = name.lower()
        if key in by_lower:
            problems.append(f"duplicate flat row for {name}")
            continue
        by_lower[key] = r

    # --- reconcile 1:1 with the template, in template order ------------------
    rows: list[dict] = []
    used: set[str] = set()
    for t in tpl:
        fn = t["filename"]
        src = by_lower.get(fn.lower())
        if src is None:
            problems.append(f"no flat row for corpus document {fn}")
            continue
        used.add(fn.lower())

        # the template's sha256 is the manifest lock; re-verify it against the
        # image on disk so this converter cannot launder a stale hash forward.
        img = IMAGES / fn
        if not img.is_file():
            problems.append(f"image missing on disk: {fn}")
        else:
            digest = hashlib.sha256(img.read_bytes()).hexdigest()
            if digest != t["sha256"]:
                problems.append(f"sha256 mismatch for {fn}: image has changed")

        rows.append({
            "filename": fn,
            "sha256": t["sha256"],
            "source": t["source"],
            "language_mode": t["language_mode"],
            "split": t["split"],
            "fields": {f: None for f in API_FIELDS},
            "illegible": [],
            "notes": "",
            "full_text": src["text"],          # verbatim, byte for byte
        })

    for key, r in by_lower.items():
        if key not in used:
            problems.append(f"flat row names a document not in the corpus: "
                            f"{r.get('file_name')!r}")

    if problems:
        print(f"{len(problems)} problem(s); nothing written:")
        for p in problems[:20]:
            print("  -", p)
        return 1

    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    OUT.write_text(payload, encoding="utf-8", newline="\n")

    # --- read back and prove the text survived unchanged ---------------------
    back = jsonl(OUT)
    if len(back) != len(rows):
        print("read-back row count differs; investigate before using this file.")
        return 1
    for r in back:
        original = by_lower[r["filename"].lower()]["text"]
        if r["full_text"] != original:
            print(f"read-back text differs for {r['filename']}")
            return 1
        if any(v is not None for v in r["fields"].values()):
            print(f"{r['filename']}: a field was populated; this tool must not do that.")
            return 1

    print(f"wrote {OUT.relative_to(ROOT)}  rows={len(back)}")
    print(f"  text verified verbatim on {len(back)}/{len(back)} rows")
    print(f"  sha256 re-verified against dataset_ex on {len(back)}/{len(back)} images")
    print(f"  splits: " + ", ".join(
        f"{s}={sum(1 for r in back if r['split'] == s)}"
        for s in sorted({r["split"] for r in back})))
    print(f"  fields: all null on {sum(1 for r in back if all(v is None for v in r['fields'].values()))}/{len(back)} rows "
          f"(segmentation is the human step)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
