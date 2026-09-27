# -*- coding: utf-8 -*-
r"""Build the evaluation file the benchmark consumes (deliverable 6).

Source of truth is the human ground truth `ocr_eval/ground_truth_real_fixed_v2.jsonl`
(84 transcribed of 90). For each transcribed document this writes one row:

    {"doc_id", "image", "split", "fields": {...5 GT fields...},
     "numeric_gt": [{"field","kind","value","digits"}, ...],   # every number on the page, per field
     "subject_source": "printed" | "annotator" | null}

`numeric_gt` is derived with the same span finder the service uses, so "numeric
accuracy" in the benchmark means: of the numbers a human transcribed in that field,
how many did the pipeline return digit-for-digit. Nothing is inferred from images.

    venv312\Scripts\python.exe -m ocr_service.make_eval_set
    -> ocr_eval/eval_set.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "ocr_eval" / "ground_truth_real_fixed_v2.jsonl"
IMAGES = ROOT / "ocr_eval" / "dataset_ex"
OUT = ROOT / "ocr_eval" / "eval_set.jsonl"

sys.path.insert(0, str(ROOT))
from ocr_service.numeric_validator import find_numeric_spans, digits_only  # noqa: E402

FIELDS = ["sender", "receiver", "subject", "body_text", "contact_info"]


def main() -> int:
    rows = [json.loads(l) for l in GT.read_text(encoding="utf-8").splitlines() if l.strip()]
    out, n_num = [], 0
    for r in rows:
        f = r["fields"]
        if all(f.get(k) is None for k in FIELDS):
            continue                                # untranscribed
        img = IMAGES / r["filename"]
        if not img.is_file():
            print("missing image:", r["filename"]); continue
        nums = [{"field": s.field, "kind": s.kind, "value": s.value, "digits": digits_only(s.value)}
                for s in find_numeric_spans(f, min_digits=3)]
        n_num += len(nums)
        out.append({"doc_id": r["filename"], "image": str(img), "split": r["split"], "sha256": r["sha256"],
                    "fields": {k: f.get(k) for k in FIELDS}, "numeric_gt": nums,
                    "subject_source": r.get("subject_source")})
    OUT.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out) + "\n", encoding="utf-8")
    by_split = {s: sum(1 for o in out if o["split"] == s) for s in ("dev", "test")}
    print(f"wrote {OUT}  docs={len(out)} {by_split}  numeric spans={n_num}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
