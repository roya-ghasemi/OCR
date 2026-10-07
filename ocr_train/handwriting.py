# -*- coding: utf-8 -*-
r"""Carve line pairs out of the benchmark's handwriting split.

The printed split is already (one image, one line). The handwriting split is not: it is 6,669
**page** images, 700x433 to 1502x1302, whose labels run to 94 lines. Nothing in LSTM line
training can use a page, so the lines have to be cut out and matched to the label's lines.

Matching is the whole problem, and the honest way to do it is to refuse most of the data.
A page's label lines are not always its visual lines — an exam form's label lists cell
contents («ردیف», «فروردین», «۱», «قرآن»), forty-three of them across four visible rows, and
any attempt to pair those up produces image/text pairs that are confidently wrong. A wrong
pair is worse than no pair: the model does not ignore it, it learns it.

So a page is used only when the number of ink rows found equals the number of label lines
exactly, and there are at least three. On a 400-page sample that keeps 10% of pages and
yields about 700 line pairs — extrapolating to roughly 11,600 across the split, which is
comparable to the whole synthetic corpus. The other 90% is discarded without being examined
further, and the count of what was dropped and why is printed, because a silent 90% loss and
a silent 90% corruption look identical from the outside.

Ordering is assumed top-to-bottom, which is right for the prose pages that survive the count
test and is the reason the count test is strict: a page whose rows and label lines agree in
number is almost always a page where they agree in order.
"""
from __future__ import annotations

import hashlib
import io
import re

import numpy as np
from PIL import Image

from . import config
from .benchmark import _clean, allowed_chars, split_of

MIN_ROW_H = 10          # an ink run shorter than this is a mark, not a line
MIN_LINES = 3           # a one or two line page gives the count test nothing to check
MAX_LINES = 60          # beyond this it is a form or a table, whatever the rows say
ROW_INK = 0.008         # fraction of a row's width that must be ink for it to be a row
PAD = 4                 # a little air above and below, so descenders survive the cut
MIN_STROKE_PX = 3       # below this the band is a ruled or dotted line, not writing


def ink_rows(im: Image.Image) -> list[tuple[int, int]]:
    """Horizontal bands of ink, top to bottom.

    The threshold is taken from the page's own 60th percentile rather than a constant: these
    are photographs of paper, and a fixed cut either finds nothing on a grey scan or finds
    the paper itself on a bright one.
    """
    a = np.asarray(im.convert("L"), dtype=np.float32)
    thr = float(np.percentile(a, 60)) * 0.80
    ink = a < thr
    prof = ink.mean(axis=1)
    on = prof > ROW_INK
    runs, start = [], None
    for y, v in enumerate(on):
        if v and start is None:
            start = y
        elif not v and start is not None:
            if y - start >= MIN_ROW_H:
                runs.append((start, y))
            start = None
    if start is not None and len(on) - start >= MIN_ROW_H:
        runs.append((start, len(on)))
    return [r for r in runs if _is_text_row(ink[r[0]:r[1]])]


def _is_text_row(ink: np.ndarray) -> bool:
    """Is this band writing, or a ruled line?

    A page of lined paper or a dotted leader counts as an ink row and then shifts every
    label after it by one, which is the failure that matters here: the count test still
    passes, so the page is kept and every pair on it is wrong. One such page was found by
    eye in a sample of six.

    Writing and rules separate cleanly on how tall the ink is in a typical inked column —
    measured across 301 crops, rules sit at a median of 1 px and writing between 7 and 17,
    median 9.
    """
    cols = ink.sum(axis=0)
    inked = cols[cols > 0]
    if inked.size < 8:
        return False
    return float(np.median(inked)) >= MIN_STROKE_PX


def build(limit: int = 0) -> dict:
    """Write `<base>.png` + `<base>.gt.txt` for every line that could be matched."""
    import pyarrow.parquet as pq

    out_dir = config.HAND_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    allowed = allowed_chars()

    files = sorted(config.BENCHMARK.glob("handwriting-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no handwriting-*.parquet under {config.BENCHMARK}")

    stats = {"pages": 0, "used": 0, "dropped_count_mismatch": 0, "dropped_too_few": 0,
             "dropped_too_many": 0, "written": 0, "dropped_charset": 0, "held_out": 0}
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                stats["pages"] += 1
                label = [l.strip() for l in (row["label"] or "").split("\n") if l.strip()]
                if len(label) < MIN_LINES:
                    stats["dropped_too_few"] += 1
                    continue
                if len(label) > MAX_LINES:
                    stats["dropped_too_many"] += 1
                    continue
                raw = row["image"]["bytes"]
                im = Image.open(io.BytesIO(raw)).convert("L")
                rows = ink_rows(im)
                if len(rows) != len(label):
                    stats["dropped_count_mismatch"] += 1
                    continue
                stats["used"] += 1

                page_key = hashlib.sha256(raw).hexdigest()
                held = split_of(raw) == "eval"      # the same hash split as the printed rows
                for i, ((y0, y1), text) in enumerate(zip(rows, label)):
                    text = _clean(text)
                    if not text or set(text) - allowed:
                        stats["dropped_charset"] += 1
                        continue
                    crop = im.crop((0, max(0, y0 - PAD), im.width, min(im.height, y1 + PAD)))
                    if crop.height < 8 or crop.width < 32:
                        continue
                    tag = "hand_eval" if held else "hand_train"
                    base = out_dir / f"{tag}_{page_key[:12]}_{i:03d}"
                    crop.save(base.with_suffix(".png"))
                    base.with_suffix(".gt.txt").write_text(text + "\n", encoding="utf-8")
                    stats["written"] += 1
                    stats["held_out"] += int(held)
                if limit and stats["written"] >= limit:
                    break
            if limit and stats["written"] >= limit:
                break
        if limit and stats["written"] >= limit:
            break

    kept = stats["used"] / max(1, stats["pages"])
    print(f"handwriting: {stats['written']} line pairs from {stats['used']} of "
          f"{stats['pages']} pages ({kept:.0%} kept)")
    print(f"  dropped: {stats['dropped_count_mismatch']} rows/label mismatch, "
          f"{stats['dropped_too_few']} too few lines, {stats['dropped_too_many']} too many, "
          f"{stats['dropped_charset']} lines outside the alphabet")
    print(f"  held out for evaluation: {stats['held_out']}")
    return stats
