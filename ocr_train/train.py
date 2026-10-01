# -*- coding: utf-8 -*-
r"""Drive lstmtraining, then pack the result into `fas_custom.traineddata`.

This is a **fine-tune with an extended alphabet**, which is a specific mode and needs three
arguments that a plain fine-tune does not:

    --continue_from      base/fas.lstm           the float weights we start from
    --traineddata        uc/.../fas_custom.traineddata   the NEW alphabet and recoder
    --old_traineddata    base/fas_best.traineddata       the OLD alphabet

`--old_traineddata` is what makes the extension work. Without it lstmtraining has no way to
map the base model's 95 output classes onto the new 153, so it would either refuse or
silently discard everything the model knows about Persian and start from noise. With it, the
old classes keep their weights and only the new ones — «،», Latin, «@» — begin untrained.

Why the base is `fas.lstm` extracted from the float model and not a checkpoint: there is no
checkpoint; this is the first fine-tune. Tesseract prints `Warning: LSTMTrainer deserialized
an LSTMRecognizer!` when it loads a recognition model rather than a training checkpoint.
That warning is expected here and is not an error.

The learning rate is left at the default. A fine-tune that is allowed to move fast forgets
the language it already knows — the failure mode is a model that reads the training fonts
beautifully and everything else worse than before, which is exactly what the held-out fifth
and `ocr_eval` exist to catch.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import config
from .unicharset import STARTER

CHECKPOINT = config.WORK_DIR / config.LANG
OUT_TRAINEDDATA = config.OUT_DIR / f"{config.LANG}.traineddata"

_ERR = re.compile(r"At iteration (\d+)/(\d+)/(\d+), Mean rms=([\d.]+)%.*?char train=([\d.]+)%.*?"
                  r"skip ratio=([\d.]+)%", re.S)
_EVAL = re.compile(r"(?:BCER|char) eval=([\d.]+)%")


def run(max_iterations: int | None = None, target_error: float | None = None,
        debug_interval: int = 0, learning_rate: float | None = None) -> int:
    """Fine-tune. Returns lstmtraining's exit code."""
    train_list = config.WORK_DIR / "train.txt"
    eval_list = config.WORK_DIR / "eval.txt"
    for p in (train_list, eval_list):
        if not p.is_file() or not p.read_text(encoding="utf-8").strip():
            raise RuntimeError(f"{p} is missing or empty — run the `lstmf` step first")
    if not STARTER.is_file():
        raise RuntimeError("starter traineddata missing — run the `unicharset` step first")

    config.WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Resume from a checkpoint when one exists. A checkpoint already carries the new
    # alphabet and the optimizer state, so `--old_traineddata` must NOT be passed — that
    # flag exists to remap a model built against a *different* unicharset, and giving it a
    # checkpoint that is already remapped would redo the mapping.
    #
    # Resuming matters more than it looks. This fine-tune spends roughly 5,000 iterations on
    # a plateau near 35% BCER while the alignment is learned, and then breaks through: 35% →
    # 21% in under 400 iterations. A run killed during the plateau looks like a failed
    # experiment and is actually an unfinished one.
    resume = best_checkpoint()
    start = [
        "--continue_from", str(resume if resume else config.BASE_LSTM),
        "--traineddata", str(STARTER),
    ]
    if resume is None:
        start += ["--old_traineddata", str(config.BASE_TRAINEDDATA)]
    else:
        print(f"resuming from {resume.name}")
    cmd = [
        config.tool("lstmtraining"),
        *start,
        "--model_output", str(CHECKPOINT),
        "--train_listfile", str(train_list),
        "--eval_listfile", str(eval_list),
        "--max_iterations", str(max_iterations or config.MAX_ITERATIONS),
        "--target_error_rate", str(target_error or config.TARGET_ERROR_RATE),
        "--debug_interval", str(debug_interval),
        "--learning_rate", str(learning_rate or config.LEARNING_RATE),
    ]
    if config.RESET_LEARNING_RATE:
        # Without this the rate in the loaded model wins and --learning_rate is ignored.
        cmd += ["--reset_learning_rate", "true"]
    n_train = len(train_list.read_text(encoding="utf-8").split())
    n_eval = len(eval_list.read_text(encoding="utf-8").split())
    print(f"fine-tuning on {n_train} lines, evaluating on {n_eval} held-out lines")
    print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)

    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
    log = config.WORK_DIR / "train.log"
    with log.open("w", encoding="utf-8") as fh:
        for line in p.stdout:
            fh.write(line)
            fh.flush()          # or the log stays empty for the whole run and nothing can watch it
            # lstmtraining is chatty; surface only progress and anything that looks wrong
            low = line.lower()
            if "at iteration" in low or "eval=" in low or "error" in low[:14] \
                    or "failed" in low or "wrote" in low:
                print("  " + line.rstrip(), flush=True)
    rc = p.wait()
    print(f"lstmtraining exit {rc} after {time.time() - t0:.0f}s; log at {log}")
    return rc


_CKPT = re.compile(rf"_(\d+\.?\d*)_(\d+)_(\d+)\.checkpoint$")


def best_checkpoint() -> Path | None:
    """The lowest-error checkpoint Tesseract kept.

    Tesseract writes two kinds: `<lang>_checkpoint`, which is simply the most recent, and
    `<lang>_<BCER>_<iter>_<iter>.checkpoint`, one per new best. Sorting those by name sorts
    the error rate as a *string*, which puts 9.9 after 10.1 — so the rate is parsed out and
    compared as a number, and the recent-checkpoint file is only a fallback.
    """
    scored = []
    for p in config.WORK_DIR.glob(f"{config.LANG}_*.checkpoint"):
        m = _CKPT.search(p.name)
        if m:
            scored.append((float(m.group(1)), int(m.group(2)), p))
    if scored:
        scored.sort(key=lambda t: (t[0], -t[1]))
        return scored[0][2]
    latest = config.WORK_DIR / f"{config.LANG}_checkpoint"
    return latest if latest.is_file() else None


def pack() -> Path:
    """Turn the best checkpoint into a traineddata the service can load."""
    best = best_checkpoint()
    if best is None:
        raise RuntimeError(f"no checkpoint in {config.WORK_DIR} — did training run?")
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [config.tool("lstmtraining"), "--stop_training",
           "--continue_from", str(best),
           "--traineddata", str(STARTER),
           "--model_output", str(OUT_TRAINEDDATA)]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not OUT_TRAINEDDATA.is_file():
        raise RuntimeError(f"packing failed:\n{p.stdout}\n{p.stderr}")
    print(f"wrote {OUT_TRAINEDDATA} ({OUT_TRAINEDDATA.stat().st_size} bytes) "
          f"from {best.name}")
    return OUT_TRAINEDDATA


def install() -> Path:
    """Put the fine-tune where the service's `_find_up` will see it.

    Beside `fas.traineddata`, not replacing it. Keeping both means the switch is one
    setting (`OCRS_LANG_FA=fas_custom`), an A/B needs no file juggling, and rolling back is
    changing a string rather than restoring a binary.
    """
    tessdata = config.ROOT / "tessdata"
    if not tessdata.is_dir():
        raise FileNotFoundError(f"{tessdata} does not exist")
    if not OUT_TRAINEDDATA.is_file():
        raise RuntimeError(f"{OUT_TRAINEDDATA} is missing — run the `pack` step first")
    dest = tessdata / f"{config.LANG}.traineddata"
    shutil.copy2(OUT_TRAINEDDATA, dest)
    import hashlib
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()[:16]
    print(f"installed {dest} ({dest.stat().st_size} bytes, sha256 {sha})")
    print(f"  use it with:  OCRS_LANG_FA={config.LANG}")
    return dest


def evaluate(model: Path | None = None) -> float | None:
    """Character error on the held-out fifth of the benchmark.

    Remember what this number is: the other four fifths of that benchmark were trained on,
    so it says the model generalises across *rows of the same corpus*. It is not a score on
    the published benchmark and must never be quoted as one. The service's own numbers come
    from `ocr_eval`.
    """
    eval_list = config.WORK_DIR / "eval.txt"
    cmd = [config.tool("lstmeval"),
           "--model", str(model or OUT_TRAINEDDATA),
           "--traineddata", str(STARTER),
           "--eval_listfile", str(eval_list)]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    sys.stdout.write(out[-1500:])
    m = re.findall(r"BCER eval=([\d.]+)%", out) or re.findall(r"char error rate=([\d.]+)", out)
    return float(m[-1]) if m else None
