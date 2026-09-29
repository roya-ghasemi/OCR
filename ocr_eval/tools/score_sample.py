# -*- coding: utf-8 -*-
r"""Score one page against a hand-typed transcript.

The corpus benchmark (`bench_fulltext.py`) only covers administrative letters, so
nothing measured the general-document case the service also has to handle. This scores
any image in `ocr_eval/samples/` that has a `<name>.gt.txt` beside it: whole-page CER,
word accuracy, and whether every ground-truth line survives.

    venv312\Scripts\python.exe ocr_eval\tools\score_sample.py
    venv312\Scripts\python.exe ocr_eval\tools\score_sample.py --image ocr_eval/samples/prose_page.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
ROOT = EVAL.parent
sys.path[:0] = [str(EVAL), str(ROOT)]

import Levenshtein  # noqa: E402

from normalize import normalize  # noqa: E402

SAMPLES = EVAL / "samples"


def score(pred: str, gt: str) -> dict:
    p, g = normalize(pred), normalize(gt)
    pw, gw = p.split(), g.split()
    hit = 0
    pool = list(pw)
    for w in gw:
        if w in pool:
            pool.remove(w)
            hit += 1
    return {"cer": round(Levenshtein.distance(p, g) / max(len(g), 1), 4),
            "word_recall": round(hit / max(len(gw), 1), 4),
            "gt_chars": len(g), "pred_chars": len(p), "gt_words": len(gw),
            "len_ratio": round(len(p) / max(len(g), 1), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, default=None)
    a = ap.parse_args()

    from ocr_service.config import Settings
    from ocr_service.pipeline import OcrPipeline

    pairs = ([(a.image, a.image.with_suffix("").with_suffix(".gt.txt"))] if a.image else
             [(p, p.with_suffix("").with_suffix(".gt.txt")) for p in sorted(SAMPLES.glob("*"))
              if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}])
    pairs = [(im, gt) for im, gt in pairs if gt.is_file()]
    if not pairs:
        print(f"no <name>.gt.txt beside an image in {SAMPLES}")
        return 1

    pipe = OcrPipeline(Settings())
    if not pipe.health()["tesseract"]["available"]:
        print("Tesseract with Persian data is not available")
        return 2
    rc = 0
    for im, gtp in pairs:
        r = pipe.run_sync(im.read_bytes(), im.name)
        s = score(r.text, gtp.read_text(encoding="utf-8"))
        print(f"{im.name}: CER {s['cer']:.1%}  word recall {s['word_recall']:.1%}  "
              f"length {s['len_ratio']:.2f}  ({s['gt_words']} GT words, {r.glyph_px:.0f} px text, "
              f"{len(r.lines)} lines, is_letter={r.is_letter})")
        if r.review_reasons:
            print(f"    review: {'; '.join(r.review_reasons)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
