# -*- coding: utf-8 -*-
r"""Classical digit reader alone vs the human GT — no VLM in the loop.

Unit test for the page-level number locator + glyph reader (D55/D56). For every
transcribed document the reader's `page_words()` runs on the page and the numbers it
returns are compared with the numbers a human transcribed, as ATOMS: maximal digit
runs (any script, folded to ASCII) of >= 3 digits. Atoms sidestep the hyphen/space
segmentation inconsistency of the GT (D57) and of the VLM.

    recall     GT atoms (all fields pooled) found among the reader's atoms
    precision  reader atoms that are GT atoms (the locator's noise)

    venv312\Scripts\python.exe ocr_eval\tools\bench_digit_reader.py --split dev --label glyph_v1
    venv312\Scripts\python.exe ocr_eval\tools\bench_digit_reader.py --split dev --label glyph_v1 --source prepared
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image, ImageOps  # noqa: E402

from ocr_service.numeric_validator import to_ascii_digits  # noqa: E402

EVAL = ROOT / "ocr_eval" / "eval_set.jsonl"
OUT = ROOT / "ocr_eval" / "experiments" / "digits_0922"
FIELDS = ["sender", "receiver", "subject", "body_text", "contact_info"]


def atoms(text: str | None, min_len: int = 3) -> list[str]:
    return [a for a in re.findall(r"\d+", to_ascii_digits(text or "")) if len(a) >= min_len]


# A grouped number: digit groups joined by separators, e.g. ۳۲۱/۰۰۰/۰۰۰ or a phone
# list. Atoms deliberately ignore the grouping, so they cannot see a truncated or
# split amount — `۳۲۱/۰۰۰/۰۰` still contributes the atoms 321 and 000. Whole-number
# exactness is the metric that matches what a caller reads off the page.
_GROUPED = re.compile(r"[0-9۰-۹٠-٩]+(?:[/,،.\-][0-9۰-۹٠-٩]+)+")


def grouped(text: str | None) -> list[str]:
    out = []
    for m in _GROUPED.finditer(text or ""):
        v = to_ascii_digits(m.group(0))
        if len(re.sub(r"\D", "", v)) >= 4:
            out.append(v)
    return out


def gt_atoms(doc: dict) -> collections.Counter:
    c = collections.Counter()
    for f in FIELDS:
        for a in atoms(doc["fields"].get(f)):
            c[a] += 1
    return c


def load_image(doc: dict, source: str) -> Image.Image:
    raw = Path(doc["image"]).read_bytes()
    if source == "prepared":
        from ocr_service.preprocess import prepare
        raw, _ = prepare(raw)
    return ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")


def make_reader(name: str):
    if name == "glyph":
        from ocr_service.digit_reader import GlyphReader
        return GlyphReader()
    if name == "tesseract":
        from ocr_service.config import settings
        from ocr_service.numeric_validator import TesseractReader
        cmd, data = settings.resolved_tesseract()
        return TesseractReader(cmd, settings.tesseract_langs, data)
    mod, _, cls = name.rpartition(":")          # "module.path:ClassName"
    import importlib
    return getattr(importlib.import_module(mod), cls)()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--reader", default="glyph")
    ap.add_argument("--source", choices=["original", "prepared"], default="original")
    ap.add_argument("--label", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default=None, help="substring of doc_id")
    a = ap.parse_args()

    docs = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.split != "all":
        docs = [d for d in docs if d["split"] == a.split]
    if a.only:
        docs = [d for d in docs if a.only in d["doc_id"]]
    if a.limit:
        docs = docs[: a.limit]
    reader = make_reader(a.reader)

    tot_gt = tot_hit = tot_read = tot_read_ok = 0
    grp_gt = grp_exact = grp_digits_only = 0
    grp_miss = []
    by_len = collections.defaultdict(lambda: [0, 0])
    per_doc = []
    t_all = time.perf_counter()
    for i, d in enumerate(docs, 1):
        im = load_image(d, a.source)
        t0 = time.perf_counter()
        words = reader.page_words(im)
        dt = time.perf_counter() - t0
        got = collections.Counter()
        for w in words:
            for at in atoms(w.text):
                got[at] += 1
        gt = gt_atoms(d)
        hit = gt & got
        n_gt, n_hit = sum(gt.values()), sum(hit.values())
        n_read, n_ok = sum(got.values()), sum((got & gt).values())
        tot_gt += n_gt; tot_hit += n_hit; tot_read += n_read; tot_read_ok += n_ok
        for at, k in gt.items():
            by_len[len(at)][0] += k; by_len[len(at)][1] += (hit[at])
        # whole grouped numbers, exactly as printed
        read_texts = [to_ascii_digits(w.text) for w in words]
        for f in FIELDS:
            for gnum in grouped(d["fields"].get(f)):
                grp_gt += 1
                if any(rt == gnum for rt in read_texts):
                    grp_exact += 1
                elif any(re.sub(r"\D", "", rt) == re.sub(r"\D", "", gnum) for rt in read_texts):
                    grp_digits_only += 1
                elif len(grp_miss) < 40:
                    best = max(read_texts, key=lambda rt: len(set(re.sub(r"\D", "", rt)) & set(re.sub(r"\D", "", gnum))), default="")
                    grp_miss.append({"doc": d["doc_id"][-8:], "gt": gnum, "closest_read": best})
        missed = sorted((gt - hit).elements())
        per_doc.append({"doc_id": d["doc_id"], "n_gt": n_gt, "n_hit": n_hit, "n_read": n_read, "n_read_ok": n_ok,
                        "missed": missed, "reads": [w.text for w in words], "seconds": round(dt, 2)})
        print(f"[{i:3d}/{len(docs)}] gt={n_gt:3d} hit={n_hit:3d} read={n_read:3d} ok={n_ok:3d} {dt:5.1f}s "
              f"missed={missed[:6]} {d['doc_id'][-8:]}", flush=True)
    summary = {
        "label": a.label, "reader": a.reader, "source": a.source, "split": a.split, "n_docs": len(docs),
        "gt_atoms": tot_gt, "recall": round(tot_hit / max(tot_gt, 1), 4),
        "reads": tot_read, "precision": round(tot_read_ok / max(tot_read, 1), 4),
        "recall_by_len": {k: [v[0], v[1], round(v[1] / v[0], 3)] for k, v in sorted(by_len.items())},
        "grouped_numbers": {"gt": grp_gt, "exact": grp_exact, "digits_right_separators_wrong": grp_digits_only,
                            "exact_rate": round(grp_exact / max(grp_gt, 1), 4),
                            "missed": grp_miss},
        "seconds_total": round(time.perf_counter() - t_all, 1),
        "reader_version": getattr(reader, "version", None),
        "per_doc": per_doc,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"reader_{a.label}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n== {a.label}: reader={a.reader} source={a.source} n={len(docs)}")
    print(f"   atom recall {summary['recall']:.1%} ({tot_hit}/{tot_gt}) | precision {summary['precision']:.1%} ({tot_read_ok}/{tot_read})")
    print("   by length:", {k: f"{v[1]}/{v[0]}" for k, v in summary["recall_by_len"].items()})
    g = summary["grouped_numbers"]
    print(f"   grouped numbers read EXACTLY (separators and all): {g['exact']}/{g['gt']} = {g['exact_rate']:.1%}"
          f"  (+{g['digits_right_separators_wrong']} with right digits, different separators)")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
