# -*- coding: utf-8 -*-
r"""Turn `image + .gt.txt` pairs into the `.lstmf` files lstmtraining reads.

Tesseract does this itself: `tesseract <image> <base> --psm 13 lstm.train` reads
`<base>.gt.txt` beside the image and writes `<base>.lstmf`, which holds the line's
normalised image strip and its encoded transcription.

Two details that silently produce a corpus of nothing if you get them wrong:

* **`--psm 13`**, raw line. Any layout mode re-segments the image, and on a tight line crop
  that finds one "line" that is not the one the ground truth describes.
* **The `lstm.train` config lives in the Tesseract install's `tessdata/configs`**, not in
  the project's tessdata. `--tessdata-dir` must therefore point at the install, and the
  starter traineddata must be reachable from there — so it is copied in rather than
  referenced, and removed again afterwards only if we put it there.

A pair whose `.lstmf` comes out empty is dropped and counted. The commonest cause is a label
holding a character the alphabet still cannot encode, which should be zero after
`unicharset.build()` and is worth knowing about if it is not.
"""
from __future__ import annotations

import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config
from .unicharset import STARTER

# The install's tessdata, which is where `configs/lstm.train` ships.
INSTALL_TESSDATA = config.TESS_DIR / "tessdata"
# Ours. Writing into Program Files needs administrator rights, and a training run is no
# reason to ask for them — so a private tessdata is assembled here with the two things
# `tesseract ... lstm.train` needs to find: the starter model and the config.
OUR_TESSDATA = config.TRAIN / "tessdata"


def _stage_starter() -> Path:
    """A tessdata directory holding the starter model and the lstm.train config."""
    (OUR_TESSDATA / "configs").mkdir(parents=True, exist_ok=True)
    dest = OUR_TESSDATA / f"{config.LANG}.traineddata"
    if not dest.is_file() or dest.stat().st_mtime < STARTER.stat().st_mtime:
        shutil.copy2(STARTER, dest)
    cfg = OUR_TESSDATA / "configs" / "lstm.train"
    if not cfg.is_file():
        src = INSTALL_TESSDATA / "configs" / "lstm.train"
        if not src.is_file():
            raise FileNotFoundError(f"{src} is missing — it ships with Tesseract")
        shutil.copy2(src, cfg)
    return OUR_TESSDATA


def _write_list(path: Path, items: list[Path]) -> None:
    """A list file with LF endings, whatever platform wrote it."""
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(str(p) for p in items) + "\n")


ZWNJ = "‌"
# A strong right-to-left letter that is not in the alphabet. The bidi algorithm classes ZWNJ
# as boundary-neutral and *deletes* it, which would quietly strip every «می‌شود» and
# «نیازمندی‌های» out of the targets; standing in a real letter for the reorder and swapping
# it back afterwards keeps it.
_BIDI_GUARD = "ٹ"


def visual_order(text: str) -> str:
    """The characters in the order they appear left-to-right on the page.

    **This is the order the targets must be in, and getting it wrong is silent.** The box
    file this module writes gives every character the *same* rectangle — the whole line —
    because the real per-character geometry is not known. That is the documented line-box
    form, but it means Tesseract has no x-coordinates to sort by, so the order of the file
    *is* the order of the target sequence. The network scans the image left to right, so for
    Persian that sequence is the visual one, not the logical one.

    Measured, because the failure does not announce itself: feeding logical order to the
    stock `fas` model scores BCER 10.9% / **BWER 100%** on a line it reads correctly — the
    characters all match, every word is "wrong", because the model emits them in the
    opposite order. The same line in visual order scores BCER 10.9% / **BWER 33%**. In
    training the effect is worse than wrong: the model spends itself unlearning how to read
    Persian and starts learning it backwards, which is what 95% BCER and a slow "recovery"
    toward 94% actually was.

    Naive reversal is not enough. A date, an amount and an e-mail address inside Persian
    text are left-to-right runs within a right-to-left line, so «۱۴۰۲/۱۲/۲۷» and
    «info@sabzgostar.ir» must come out forwards while everything around them comes out
    backwards. That is the bidi algorithm, and this defers to it.
    """
    from bidi.algorithm import get_display
    return get_display(text.replace(ZWNJ, _BIDI_GUARD), base_dir="R").replace(_BIDI_GUARD, ZWNJ)


def write_line_box(img: Path, text: str) -> Path:
    """The box file `lstm.train` reads, in Tesseract's line-box form.

    This build will not take a bare `.gt.txt` — it asks for `<base>.box` and stops. The
    line-box form is one entry per character, every entry carrying the whole line's
    rectangle, terminated by a tab entry. The characters go in **visual** order; see
    `visual_order` for why, and for what it costs to get that wrong.

    Coordinates are Tesseract's: origin bottom-left, `left bottom right top page`.
    """
    from PIL import Image
    with Image.open(img) as im:
        w, h = im.size
    box = img.with_suffix(".box")
    rows = [f"{c} 0 0 {w} {h} 0" for c in visual_order(text) if c != "\n"]
    rows.append(f"\t 0 0 {w} {h} 0")
    box.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return box


def _one(img: Path) -> tuple[Path, bool]:
    base = img.with_suffix("")
    out = base.with_suffix(".lstmf")
    if out.is_file() and out.stat().st_size > 0:
        return out, True
    gt = base.with_suffix(".gt.txt")
    if not gt.is_file():
        return out, False
    write_line_box(img, gt.read_text(encoding="utf-8").strip())
    cmd = [config.tool("tesseract"), str(img), str(base),
           "--psm", "13", "-l", config.LANG,
           "--tessdata-dir", str(OUR_TESSDATA),
           "lstm.train"]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        return out, False
    return out, out.is_file() and out.stat().st_size > 0


def build(workers: int = 12) -> dict:
    """Make a `.lstmf` for every ground-truth pair, then write the train/eval lists.

    The eval list is the benchmark's held-out fifth and **nothing else** — no synthetic
    lines, because a score on text this pipeline generated itself measures the generator,
    not the model.
    """
    _stage_starter()
    if True:
        images = sorted(config.REAL_DIR.glob("*.png")) + sorted(config.SYNTH_DIR.glob("*.png"))
        images = [p for p in images if p.with_suffix(".gt.txt").is_file()]
        if not images:
            raise RuntimeError("no image/ground-truth pairs found")

        made, failed = [], 0
        with ThreadPoolExecutor(workers) as ex:
            for i, (out, ok) in enumerate(ex.map(_one, images), 1):
                if ok:
                    made.append(out)
                else:
                    failed += 1
                if i % 1000 == 0:
                    print(f"  {i}/{len(images)} ({failed} empty)", flush=True)

        train = [p for p in made if "bench_eval" not in p.name]
        held = [p for p in made if "bench_eval" in p.name]
        config.WORK_DIR.mkdir(parents=True, exist_ok=True)
        # newline="\n" is load-bearing on Windows. Python's default translates "\n" to
        # "\r\n", every path in the list then ends with a carriage return, and lstmtraining
        # fails to open any of them — reporting "Deserialize header failed" with a path that
        # *looks* correct in a terminal, because the \r just moves the cursor back.
        _write_list(config.WORK_DIR / "train.txt", train)
        _write_list(config.WORK_DIR / "eval.txt", held)

        stats = {"pairs": len(images), "lstmf": len(made), "empty": failed,
                 "train": len(train), "eval": len(held)}
        print(f"lstmf: {len(made)} of {len(images)} pairs "
              f"({len(train)} train / {len(held)} held out), {failed} empty")
        return stats
