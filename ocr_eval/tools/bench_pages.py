# -*- coding: utf-8 -*-
r"""Whole-page CER / WER against a hand-typed transcript (`ocr_eval/pages`).

`bench_fulltext.py` scores letters against five-field GT. It cannot see three things
this brief is about: word error, a general (non-letter) page, and text the service
printed that is **not on the page at all** — the junk lines are short, so a page-wide
CER barely moves when one is added. This tool measures all three.

    CER            character edit distance / GT characters, after `normalize()`
    WER            word edit distance / GT words, after `normalize()`
    unsupported    predicted lines matching nothing in the GT: a line whose best
                   semi-global distance to the whole GT exceeds `UNSUPPORTED` of its
                   own length. Reported as a count and as a share of predicted
                   characters — this is the "no nonsense text" acceptance criterion
    missing        GT lines matching nothing in the prediction, same test reversed
    structure      lines / paragraphs / tables against what the page actually has

Both sides are normalised identically (`ocr_eval/normalize.py`, RULES_DEFAULT) before
any distance is taken, exactly as the corpus harness does.

    venv312\Scripts\python.exe ocr_eval\tools\bench_pages.py --label before
    venv312\Scripts\python.exe ocr_eval\tools\bench_pages.py --label after --compare before
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
ROOT = EVAL.parent
sys.path[:0] = [str(EVAL), str(ROOT)]

import Levenshtein  # noqa: E402

from fulltext_score import semiglobal  # noqa: E402
from normalize import normalize  # noqa: E402

PAGES = EVAL / "pages"
RUNS = PAGES / "runs"

# A predicted line is "unsupported" when more than this share of it has to be edited to
# find it anywhere in the ground truth. 0.5 is deliberately loose: a line that is half
# wrong is still a reading of something on the page, and only a line that matches almost
# nothing counts as invented.
UNSUPPORTED = 0.5


def wer(pred: str, gt: str) -> tuple[int, int]:
    """(word edits, GT words). Levenshtein over word sequences, not characters."""
    p, g = pred.split(), gt.split()
    if not g:
        return (len(p), 0)
    # map each distinct word to one codepoint so the C implementation can do the DP
    vocab = {w: chr(i + 256) for i, w in enumerate(dict.fromkeys(p + g))}
    return Levenshtein.distance("".join(vocab[w] for w in p),
                                "".join(vocab[w] for w in g)), len(g)


def unsupported_lines(pred_lines: list[str], gt: str) -> list[str]:
    """Predicted lines that match nothing in the ground truth."""
    out = []
    for line in pred_lines:
        t = normalize(line).strip()
        if not t:
            continue
        if semiglobal(t, gt) > UNSUPPORTED * len(t):
            out.append(line)
    return out


def structure(res) -> dict:
    """Lines, paragraphs and tables as the response reports them. A table is a run of
    adjacent lines that were split into two or more cells."""
    tables, run = 0, False
    for L in res.lines:
        has_cells = len(getattr(L, "cells", []) or []) >= 2
        if has_cells and not run:
            tables += 1
        run = has_cells
    paras = len({L.paragraph for L in res.lines}) if res.lines else 0
    return {"lines": len(res.lines), "paragraphs": paras, "tables": tables}


def score_page(res, gt_text: str) -> dict:
    p, g = normalize(res.text), normalize(gt_text)
    we, gw = wer(p, g)
    pred_lines = [L.text for L in res.lines]
    unsup = unsupported_lines(pred_lines, g)
    missing = unsupported_lines([l for l in gt_text.splitlines() if l.strip()], p)
    unsup_chars = sum(len(normalize(l).strip()) for l in unsup)
    return {
        "cer": round(Levenshtein.distance(p, g) / max(len(g), 1), 4),
        "wer": round(we / max(gw, 1), 4),
        "gt_chars": len(g), "pred_chars": len(p), "gt_words": gw,
        "len_ratio": round(len(p) / max(len(g), 1), 3),
        "unsupported_lines": len(unsup),
        "unsupported_char_share": round(unsup_chars / max(len(p), 1), 4),
        "unsupported_text": unsup,
        "missing_gt_lines": len(missing),
        "structure": structure(res),
        "needs_review": res.needs_review,
        "review_reasons": res.review_reasons,
        "glyph_px": res.glyph_px,
    }


def _expect_report(got: dict, expect: dict, is_letter: bool) -> list[str]:
    bad = []
    for k in ("lines", "paragraphs", "tables"):
        if k in expect and got[k] != expect[k]:
            bad.append(f"{k} {got[k]} (expected {expect[k]})")
    if "is_letter" in expect and is_letter != expect["is_letter"]:
        bad.append(f"is_letter {is_letter} (expected {expect['is_letter']})")
    return bad


def _row(pid: str, s: dict) -> str:
    return (f"  {pid:<20} CER {s['cer']:>7.2%}  WER {s['wer']:>7.2%}  "
            f"junk {s['unsupported_lines']:>2} line(s) / {s['unsupported_char_share']:>6.2%}  "
            f"len {s['len_ratio']:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run", help="name this run is stored under")
    ap.add_argument("--compare", default=None, help="label of an earlier run to diff against")
    ap.add_argument("--only", default=None, help="one page id")
    a = ap.parse_args()

    from ocr_service.config import Settings
    from ocr_service.pipeline import OcrPipeline

    manifest = json.loads((PAGES / "manifest.json").read_text(encoding="utf-8"))
    pages = [p for p in manifest["pages"] if not a.only or p["id"] == a.only]
    if not pages:
        print(f"no page matches {a.only!r}", file=sys.stderr)
        return 1

    pipe = OcrPipeline(Settings())
    health = pipe.health()
    if not health["tesseract"]["available"]:
        print("Tesseract with Persian data is not available", file=sys.stderr)
        return 2

    out = {"label": a.label, "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "provenance": Settings().provenance(), "pages": {}}
    for p in pages:
        img = ROOT / p["image"]
        if not img.is_file():
            print(f"missing image: {img}", file=sys.stderr)
            return 1
        t = time.perf_counter()
        res = pipe.run_sync(img.read_bytes(), p["id"])
        dt = time.perf_counter() - t
        s = score_page(res, (PAGES / p["gt"]).read_text(encoding="utf-8"))
        s["seconds"] = round(dt, 2)
        s["expect_failures"] = _expect_report(s["structure"], p.get("expect", {}), res.is_letter)
        out["pages"][p["id"]] = s
        out.setdefault("text", {})[p["id"]] = res.text

    RUNS.mkdir(parents=True, exist_ok=True)
    dest = RUNS / f"{a.label}.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{a.label}  ({health['tesseract']['version']})")
    for pid, s in out["pages"].items():
        print(_row(pid, s))
        if s["expect_failures"]:
            print(f"      structure: {'; '.join(s['expect_failures'])}")
        for line in s["unsupported_text"][:6]:
            print(f"      junk: {line[:90]}")

    if a.compare:
        prev = json.loads((RUNS / f"{a.compare}.json").read_text(encoding="utf-8"))
        print(f"\n  {a.compare} -> {a.label}")
        for pid, s in out["pages"].items():
            b = prev["pages"].get(pid)
            if not b:
                continue
            print(f"  {pid:<20} CER {b['cer']:>7.2%} -> {s['cer']:>7.2%} ({s['cer'] - b['cer']:+.2%})   "
                  f"WER {b['wer']:>7.2%} -> {s['wer']:>7.2%} ({s['wer'] - b['wer']:+.2%})   "
                  f"junk {b['unsupported_lines']} -> {s['unsupported_lines']}")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
