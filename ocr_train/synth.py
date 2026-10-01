# -*- coding: utf-8 -*-
r"""Render the synthetic corpus to line images with text2image.

text2image, not Pillow. Pillow on this machine reports `features.check('raqm') == False`,
which means no HarfBuzz: it would draw Persian as a row of isolated letterforms, and a model
trained on that is worse than no training at all. The brief flags this exact trap. text2image
drives Pango, which shapes correctly — verified by eye on a rendered sample before any of
this was built.

Two things about text2image that cost time if you do not know them:

* It needs a **fontconfig file** on Windows or it writes an empty box file and exits 0
  (`config.write_fontconfig`).
* It renders a **page**, 3600x4800 at the default resolution, with the text somewhere in it.
  For line training we want a tight line, so the ink bounding box is found and cropped with
  a small margin. That also throws away text2image's `.box` file, which we do not need —
  LSTM training takes `image + .gt.txt`, not character boxes.

Output is the tesstrain ground-truth convention: `<base>.png` beside `<base>.gt.txt`.
"""
from __future__ import annotations

import random
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from . import config
from .corpus import generate

_HAS_LATIN = __import__('re').compile(r'[A-Za-z@]')

MARGIN = 6


def _render_one(args) -> tuple[str, str] | None:
    idx, render, label, font, pt, out_dir, degrade_seed = args
    base = out_dir / f"{config.LANG}_{idx:06d}"
    png, gt = base.with_suffix(".png"), base.with_suffix(".gt.txt")
    if png.is_file() and gt.is_file():
        return (str(png), label)

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "line.txt"
        src.write_text(render + "\n", encoding="utf-8")
        stem = Path(td) / "page"
        cmd = [config.tool("text2image"),
               f"--text={src}", f"--outputbase={stem}",
               f"--font={font}", f"--fonts_dir={config.FONTS_DIR}",
               f"--ptsize={pt}", "--max_pages=1",
               "--leading=32", "--margin=40", "--xsize=4000", "--ysize=400"]
        p = subprocess.run(cmd, capture_output=True, env=config.env(), timeout=120)
        tif = stem.with_suffix(".tif")
        if not tif.is_file():
            return None
        im = Image.open(tif).convert("L")
        a = np.asarray(im) < 128
        if not a.any():
            return None
        ys, xs = np.where(a)
        im = im.crop((max(0, int(xs.min()) - MARGIN), max(0, int(ys.min()) - MARGIN),
                      min(im.width, int(xs.max()) + MARGIN + 1),
                      min(im.height, int(ys.max()) + MARGIN + 1)))
        # A line far taller than it is wide is not a line: the renderer wrapped, and the
        # label no longer describes the image.
        if im.height < 8 or im.width < 16 or im.height > im.width:
            return None
        # Most lines are then photographed badly. A model trained only on clean renders
        # learns features a scan does not have — measured: the first fine-tune improved the
        # benchmark's domain and regressed the scanned letter.
        if degrade_seed is not None:
            from .degrade import apply as degrade
            im = degrade(im, random.Random(degrade_seed))
            if im.height < 8 or im.width < 16:
                return None
        im.save(png)
    gt.write_text(label + "\n", encoding="utf-8")
    return (str(png), label)


def build(n: int = 6000, workers: int = 8, seed: int | None = None) -> int:
    """Render `n` synthetic lines, cycling the fonts and point sizes."""
    config.write_fontconfig()
    config.SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    pairs = generate(config.ROOT, n, seed if seed is not None else config.SEED)

    jobs = []
    fa_i = dual_i = 0
    for i, (render, label) in enumerate(pairs):
        # A line carrying Latin must go to a face that has Latin, or the word renders as a
        # hole and the label lies about the image.
        if _HAS_LATIN.search(label):
            font = config.FONTS_DUAL[dual_i % len(config.FONTS_DUAL)]
            dual_i += 1
        else:
            font = config.FONTS_FA[fa_i % len(config.FONTS_FA)]
            fa_i += 1
        pt = config.PT_SIZES[(i // 7) % len(config.PT_SIZES)]
        seed = None if (i % 10) < config.CLEAN_IN_TEN else config.SEED * 1000003 + i
        jobs.append((i, render, label, font, pt, config.SYNTH_DIR, seed))

    made = 0
    with ThreadPoolExecutor(workers) as ex:
        for k, res in enumerate(ex.map(_render_one, jobs), 1):
            if res:
                made += 1
            if k % 500 == 0:
                print(f"  rendered {k}/{len(jobs)} ({made} usable)", flush=True)
    print(f"synthetic lines: {made} usable of {len(jobs)} in {config.SYNTH_DIR}")
    return made
