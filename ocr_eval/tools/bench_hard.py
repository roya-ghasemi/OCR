# -*- coding: utf-8 -*-
"""Score the hard set: pages the service could not read.

Most of these have no ground truth and are not supposed to. For handwriting and nastaliq the
correct behaviour is **refusal** — empty `text`, the readings in `unread` — so what is scored
is whether the page was withheld, not how close a transcript came. A transcript appearing for
`hand_poem` is a regression even if it looks plausible; especially then.

Two of the eight are ordinary printed lines with real ground truth, and those are scored on
CER and WER like any other page. They are here because they are the floor: if the service
stops reading those, the legibility gate has been turned up too far.

    python ocr_eval/tools/bench_hard.py [--label NAME] [--compare NAME]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
HARD = ROOT / "ocr_eval" / "hard"
RUNS = HARD / "runs"


def cer(pred: str, gt: str) -> float:
    import Levenshtein
    gt = " ".join(gt.split())
    pred = " ".join((pred or "").split())
    return Levenshtein.distance(pred, gt) / max(1, len(gt))


def wer(pred: str, gt: str) -> float:
    import Levenshtein
    g, p = gt.split(), (pred or "").split()
    vocab = {w: chr(i + 0x3000) for i, w in enumerate(dict.fromkeys(g + p))}
    return Levenshtein.distance("".join(vocab[w] for w in p),
                                "".join(vocab[w] for w in g)) / max(1, len(g))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="hard")
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()

    from ocr_service.config import Settings
    from ocr_service.pipeline import OcrPipeline

    manifest = json.loads((HARD / "manifest.json").read_text(encoding="utf-8"))
    pipe = OcrPipeline(Settings())
    out = {"label": a.label,
           "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "provenance": Settings().provenance(), "pages": {}}

    print(f"\n{a.label}")
    passed = failed = 0
    for p in manifest["pages"]:
        img = ROOT / p["image"]
        if not img.is_file():
            print(f"  missing image: {img}", file=sys.stderr)
            return 1
        res = pipe.run_sync(img.read_bytes(), p["id"])
        lg = res.legibility or {}
        legible = bool(lg.get("legible", True))
        text = (res.text or "").strip()
        exp = p.get("expect", {})

        row = {"legible": legible, "fa_fraction": lg.get("fa_fraction"),
               "chars": len(text), "unread": len(res.unread), "text": text}
        want_legible = exp.get("legible")
        ok = True
        detail = ""
        if want_legible is False:
            ok = (not legible) and not text
            detail = "withheld" if ok else f"LEAKED A TRANSCRIPT: {text[:40]!r}"
        elif exp.get("text_empty"):
            ok = not text
            detail = "nothing returned" if ok else f"unexpected text: {text[:40]!r}"
        elif p.get("gt"):
            c, w = cer(text, p["gt"]), wer(text, p["gt"])
            row["cer"], row["wer"] = round(c, 4), round(w, 4)
            ok = legible and bool(text)
            detail = f"CER {c:6.2%}  WER {w:6.2%}" + ("" if ok else "   WITHHELD — gate too tight")
        else:
            detail = f"{len(text)} chars, fa {lg.get('fa_fraction')}"
        passed, failed = (passed + ok, failed + (not ok))
        row["ok"] = ok
        out["pages"][p["id"]] = row
        print(f"  {'OK ' if ok else '!! '}{p['id']:18} {detail}")

    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{a.label}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(f"\n  {passed} as intended, {failed} not")

    if a.compare:
        prev_path = RUNS / f"{a.compare}.json"
        if prev_path.is_file():
            prev = json.loads(prev_path.read_text(encoding="utf-8"))
            print(f"\n  {a.compare} -> {a.label}")
            for pid, now in out["pages"].items():
                was = prev["pages"].get(pid)
                if not was:
                    continue
                if "cer" in now and "cer" in was:
                    print(f"  {pid:18} CER {was['cer']:6.2%} -> {now['cer']:6.2%}"
                          f"   WER {was['wer']:6.2%} -> {now['wer']:6.2%}")
                elif was["ok"] != now["ok"]:
                    print(f"  {pid:18} {'fixed' if now['ok'] else 'REGRESSED'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
