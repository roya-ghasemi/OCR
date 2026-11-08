# -*- coding: utf-8 -*-
r"""Carve real Persian line images out of Persian-OCR-Double-Benchmark.

The printed split is already in the shape LSTM training wants: one image, one line of text,
6,669 of them, in a spread of faces, weights, background tints and noise levels that no
synthetic generator here would produce honestly.

**This is a published benchmark and training on it costs something.** Its README describes a
frozen, leakage-controlled set with a leaderboard, and rows that were removed from their
source training repositories precisely so they could serve as a clean test. Training on it
means our number on it is no longer comparable with anyone else's. It is used because the
user asked for it and it is the only real Persian line corpus on this machine — but the split
below is deterministic and a fifth is held out and never trained on, so there is still one
honest measurement available. Every figure taken from that fifth must carry the sentence
"the other four fifths were trained on".

The handwriting split is not touched: it is page-level, its labels contain newlines, and a
printed-text recogniser trained on handwriting gets worse at print.
"""
from __future__ import annotations

import hashlib
import io
import re
import unicodedata

from PIL import Image

from . import config

ZWNJ = "‌"
_WS = re.compile(r"\s+")


def _clean(label: str) -> str:
    """The label as the model should learn to emit it.

    Normalising here and not at scoring time is deliberate: whatever is in the ground truth
    is what the model is taught to produce, so the Arabic letterforms scattered through this
    corpus would otherwise be trained *in* — the exact defect (D8/D43) the service
    post-processes away. Folding them is safe because ي→ی and ك→ک are unambiguous in Persian.
    """
    label = unicodedata.normalize("NFC", label)
    label = label.replace("ي", "ی").replace("ى", "ی").replace("ك", "ک").replace("ۀ", "ه")
    label = label.replace("‏", "").replace("‎", "")
    return _WS.sub(" ", label).strip()


def split_of(image_bytes: bytes) -> str:
    """`train` or `eval`, decided by the image's own content hash.

    Hash-based, not random: the same row lands in the same split on every machine and after
    every re-run, so an eval number stays comparable with the one before it.
    """
    h = int(hashlib.sha256(image_bytes).hexdigest()[:8], 16)
    return "eval" if (h % 1000) < int(config.EVAL_FRACTION * 1000) else "train"


def allowed_chars() -> set[str]:
    """Everything the fine-tuned model will be able to emit."""
    chars = set()
    for line in config.BASE_UNICHARSET.read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip():
            c = line.split(" ")[0]
            if len(c) == 1:
                chars.add(c)
    return chars | set(config.EXTRA_CHARS) | {" ", ZWNJ}


def build(limit: int = 0) -> dict:
    """Write `<base>.png` + `<base>.gt.txt` for every usable printed line."""
    import pyarrow.parquet as pq

    config.REAL_DIR.mkdir(parents=True, exist_ok=True)
    ok = allowed_chars()
    files = sorted(config.BENCHMARK.glob("printed-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no printed-*.parquet under {config.BENCHMARK}")

    stats = {"rows": 0, "written": 0, "train": 0, "eval": 0,
             "dropped_chars": 0, "dropped_shape": 0, "dropped_empty": 0}
    idx = 0
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256):
            imgs = batch.column("image").to_pylist()
            labs = batch.column("label").to_pylist()
            for rec, raw in zip(imgs, labs):
                stats["rows"] += 1
                if limit and stats["written"] >= limit:
                    return _finish(stats)
                label = _clean(raw or "")
                if not label:
                    stats["dropped_empty"] += 1
                    continue
                # A line the alphabet cannot represent would teach the model to emit
                # nothing for those characters. Dropped, and counted, rather than silently
                # stripped — stripping would misalign the label from the image.
                if any(c not in ok for c in label):
                    stats["dropped_chars"] += 1
                    continue
                data = rec["bytes"] if isinstance(rec, dict) else rec
                try:
                    im = Image.open(io.BytesIO(data)).convert("L")
                except Exception:
                    stats["dropped_empty"] += 1
                    continue
                if im.height < 8 or im.width < 16 or im.height > im.width:
                    stats["dropped_shape"] += 1
                    continue
                which = split_of(data)
                base = config.REAL_DIR / f"bench_{which}_{idx:06d}"
                im.save(base.with_suffix(".png"))
                base.with_suffix(".gt.txt").write_text(label + "\n", encoding="utf-8")
                stats[which] += 1
                stats["written"] += 1
                idx += 1
            if stats["rows"] % 2000 < 256:
                print(f"  {stats['rows']} rows -> {stats['written']} lines", flush=True)
    return _finish(stats)


def _finish(stats: dict) -> dict:
    print(f"benchmark lines: {stats['written']} written "
          f"({stats['train']} train / {stats['eval']} held out) of {stats['rows']} rows; "
          f"dropped {stats['dropped_chars']} for characters outside the alphabet, "
          f"{stats['dropped_shape']} for shape, {stats['dropped_empty']} empty")
    return stats
