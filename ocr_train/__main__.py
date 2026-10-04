# -*- coding: utf-8 -*-
r"""One entry point for the fine-tune.

    python -m ocr_train fonts                 probe which faces render, and which have Latin
    python -m ocr_train synth    [--n 6000]   render synthetic lines from the project's fonts
    python -m ocr_train benchmark             carve real lines out of the benchmark
    python -m ocr_train unicharset            extend the alphabet, build the starter model
    python -m ocr_train lstmf                 convert pairs to Tesseract's training format
    python -m ocr_train train    [--iters N]  fine-tune
    python -m ocr_train pack                  checkpoint -> fas_custom.traineddata
    python -m ocr_train install               copy it into tessdata/ beside fas
    python -m ocr_train eval                  character error on the held-out fifth
    python -m ocr_train all                   every step, in order

The steps are separate because they fail for different reasons and take wildly different
times: rendering is minutes, converting is tens of minutes, training is hours. Each is
re-runnable and skips work it has already done.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def _fonts() -> None:
    """Which faces actually render, and which carry Latin. Run this after changing fonts."""
    import numpy as np
    from PIL import Image

    from . import config
    config.write_fontconfig()

    def ink(font: str, text: str) -> int | None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "t.txt"
            src.write_text(text + "\n", encoding="utf-8")
            stem = Path(td) / "p"
            subprocess.run(
                [config.tool("text2image"), f"--text={src}", f"--outputbase={stem}",
                 f"--font={font}", f"--fonts_dir={config.FONTS_DIR}", "--ptsize=20",
                 "--max_pages=1", "--margin=40", "--xsize=2400", "--ysize=300"],
                capture_output=True, env=config.env(), timeout=120)
            tif = stem.with_suffix(".tif")
            if not tif.is_file():
                return None
            return int((np.asarray(Image.open(tif).convert("L")) < 128).sum())

    print(f"{'font':<24}{'persian':>9}{'latin':>8}   verdict")
    for f in config.FONTS:
        fa = ink(f, "شرکت خدماتی سبزگستر، تلفن ۳۸۳۸۸۵۷۵")
        la = ink(f, "Email Contract Manager")
        if not fa:
            print(f"{f:<24}{'DEAD':>9}{'-':>8}   renders nothing — do not use")
            continue
        print(f"{f:<24}{fa:>9}{la or 0:>8}   "
              f"{'persian + latin' if (la or 0) > 0.1 * fa else 'persian only'}")


def main() -> int:
    ap = argparse.ArgumentParser(prog="ocr_train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fonts", "synth", "benchmark", "unicharset",
                                     "segments",
                                     "lstmf", "train", "pack", "install", "eval", "all"])
    ap.add_argument("--n", type=int, default=6000, help="synthetic lines to render")
    ap.add_argument("--iters", type=int, default=None, help="max training iterations")
    ap.add_argument("--lr", type=float, default=None, help="learning rate")
    ap.add_argument("--latest", action="store_true",
                    help="pack the most recent checkpoint, not the lowest-error one "
                         "(use after changing the training data)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--segment", type=int, default=2000,
                    help="iterations per segment; each one is packed and measured on the "
                         "held-out lines, so a run's best model survives whatever follows")
    ap.add_argument("--patience", type=int, default=3,
                    help="stop after this many segments without a new best held-out error")
    ap.add_argument("--from-scratch", action="store_true",
                    help="start from the base weights, setting the previous run's "
                         "checkpoints aside instead of resuming them")
    a = ap.parse_args()

    if a.step == "fonts":
        _fonts()
        return 0

    from . import benchmark, lstmf, segments, synth, train, unicharset

    steps = {
        "synth": lambda: synth.build(n=a.n, workers=a.workers),
        "benchmark": lambda: benchmark.build(),
        "unicharset": lambda: unicharset.build(),
        "lstmf": lambda: lstmf.build(workers=a.workers),
        "train": lambda: train.run(max_iterations=a.iters, learning_rate=a.lr),
        "segments": lambda: segments.run(total=a.iters or 30000, segment=a.segment,
                                         lr=a.lr, patience=a.patience,
                                         from_scratch=a.from_scratch),
        "pack": lambda: train.pack(latest=a.latest),
        "install": lambda: train.install(),
        "eval": lambda: train.evaluate(),
    }
    order = ["synth", "benchmark", "unicharset", "lstmf", "train", "pack", "install", "eval"]
    for name in (order if a.step == "all" else [a.step]):
        print(f"\n=== {name} ===", flush=True)
        steps[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
