# -*- coding: utf-8 -*-
r"""Drive training in bounded segments, keeping every segment's model and its measured error.

This exists because of a specific, expensive failure in E29. The degraded-corpus run
descended to BCER 5.85% train at iteration 22,400, then collapsed to 36.9% over the next
3,400 iterations — and **the minimum was unrecoverable**, because of how Tesseract keeps
checkpoints rather than anything it did wrong:

* `lstmtraining` writes a *named* `<lang>_<BCER>_<iter>_<iter>.checkpoint` only when the run
  beats its own recorded best error.
* That recorded best travels **inside the checkpoint**. The run was resumed from a checkpoint
  whose best (4.821) was earned on the earlier, *clean* synthetic corpus.
* On the harder degraded corpus nothing ever beat 4.821, so the run wrote **no named
  checkpoint at all** — and the rolling `<lang>_checkpoint`, which is overwritten every
  hundred iterations, had long since moved past the minimum.

Resuming across a change of training data therefore switches off best-keeping, silently. No
warning, no missing file, just a run whose best result does not exist when it finishes.

The fix here does not try to out-guess Tesseract's internal bookkeeping. It takes the
decision away from it: train a bounded number of iterations, stop, **pack that state into a
real traineddata and measure it on held-out data**, append the number to a ledger, and
continue. The best model of a run is then an artifact on disk with a measured error beside
it, and it stays there whatever the loss does afterwards.

Two further things fall out of segmenting, both of which E29 needed and did not have:

* **A divergence guard.** Held-out error that rises for several segments running is not a
  plateau — a plateau is flat. The run stops instead of spending hours making the model
  worse.
* **Held-out error as the selection criterion.** Training BCER chose the E29 model and chose
  wrong: the model with the lowest training error was overfitting benchmark rows. The ledger
  records `lstmeval` on the held-out fifth, which is a different and harder question.

What the ledger is *not*: a score on the published benchmark, and not evidence the service
got better. Four fifths of those rows were trained on. `ocr_eval/tools/bench_pages.py` on
real pages remains the only thing that decides whether a model ships.
"""
from __future__ import annotations

import json
import re
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import config, train
from .unicharset import STARTER

SEG_DIR = config.OUT_DIR / "segments"
LEDGER = config.WORK_DIR / "segments.jsonl"

_ITER = re.compile(r"At iteration \d+/(\d+)/\d+")
_EVAL = re.compile(r"BCER eval=([\d.]+)%?,\s*BWER eval=([\d.]+)")


def last_iteration() -> int:
    """How many learning iterations the current checkpoint has behind it.

    `--max_iterations` is absolute: it is compared against the iteration count carried in
    the checkpoint, not counted from the start of this invocation. A segment therefore has
    to ask for `here + length`, and asking for `length` on a resumed run makes lstmtraining
    exit immediately having done nothing.
    """
    log = config.WORK_DIR / "train.log"
    if not log.is_file():
        return 0
    hits = _ITER.findall(log.read_text(encoding="utf-8", errors="replace"))
    return int(hits[-1]) if hits else 0


def _segment(target: int, lr: float | None, from_scratch: bool) -> tuple[int, str]:
    """One bounded run of lstmtraining. Returns (exit code, log text)."""
    train_list = config.WORK_DIR / "train.txt"
    eval_list = config.WORK_DIR / "eval.txt"

    resume = None if from_scratch else train.latest_checkpoint()
    start = ["--continue_from", str(resume if resume else config.BASE_LSTM),
             "--traineddata", str(STARTER)]
    if resume is None:
        # Only a model built against the *old* unicharset needs remapping. A checkpoint is
        # already in the new alphabet, and remapping it again would be wrong.
        start += ["--old_traineddata", str(config.BASE_TRAINEDDATA)]

    cmd = [config.tool("lstmtraining"), *start,
           "--model_output", str(train.CHECKPOINT),
           "--train_listfile", str(train_list),
           "--eval_listfile", str(eval_list),
           "--max_iterations", str(target),
           "--target_error_rate", str(config.TARGET_ERROR_RATE),
           "--debug_interval", "0",
           "--learning_rate", str(lr or config.LEARNING_RATE)]
    if config.RESET_LEARNING_RATE and resume is None:
        # Pinning the rate on every resume throws away the schedule Tesseract has adapted,
        # which is a suspect in the E29 late divergence. Reset it when starting from the
        # base weights, then let the run keep its own.
        cmd += ["--reset_learning_rate", "true"]

    # Streamed, not captured. A segment is two thousand iterations; buffering it means the
    # log file is empty and nothing — not the terminal, not a `tail -f` — can see the run
    # until it is already over. The flush per line is what makes the log watchable, and the
    # log is also where `last_iteration` reads the counter back.
    lines: list[str] = []
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
    with (config.WORK_DIR / "train.log").open("a", encoding="utf-8") as fh:
        for line in p.stdout:
            fh.write(line)
            fh.flush()
            lines.append(line)
            low = line.lower()
            m = _ITER.search(line)
            if m and int(m.group(1)) % 500 == 0:
                print("    " + line.rstrip(), flush=True)
            elif not m and ("error" in low[:14] or "failed" in low):
                print("    " + line.rstrip(), flush=True)
    return p.wait(), "".join(lines)


def page_score(model: Path) -> dict | None:
    """Score a packed model on the three real pages, through the whole service.

    **This is the criterion, and the held-out benchmark rows are not.** Measured between
    iteration 4,000 and 10,000 of the E30 run, the two disagreed flatly:

        held-out BCER   35.78%  ->  38.36%    (said the model got worse)
        tech_prose WER  23.44%  ->   7.81%    (it got much better)
        spring     WER  32.00%  ->  16.00%
        letter     WER  20.11%  ->  16.67%

    Four fifths of those benchmark rows were trained on, so `lstmeval` on the remaining
    fifth asks whether the model generalises across rows of one corpus. That is not the
    question. The question is whether it reads an administrative letter and a photographed
    page, and `bench_pages` is the only instrument that sees word error, invented lines, or
    a page that is not a letter at all.

    The cost of getting this wrong is not academic: selecting on the held-out number picked
    the iteration-4,000 model, which is worse on every page, and the patience guard stopped
    a run that was still improving.
    """
    tessdata = config.ROOT / "tessdata"
    if not tessdata.is_dir():
        return None
    shutil.copy2(model, tessdata / f"{config.LANG}.traineddata")

    env = dict(os.environ, OCRS_TESSDATA_DIR="", OCRS_LANG_FA=config.LANG)
    label = f"_seg_{model.stem}"
    p = subprocess.run([sys.executable, str(config.ROOT / "ocr_eval" / "tools" / "bench_pages.py"),
                        "--label", label],
                       cwd=str(config.ROOT), env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = config.ROOT / "ocr_eval" / "pages" / "runs" / f"{label}.json"
    if p.returncode != 0 or not out.is_file():
        return None
    data = json.loads(out.read_text(encoding="utf-8"))
    out.unlink(missing_ok=True)          # a per-segment scratch run, not a published result
    pages = {k: {"cer": v["cer"], "wer": v["wer"], "junk": len(v.get("unsupported_text") or [])}
             for k, v in data["pages"].items()}
    if not pages:
        return None
    n = len(pages)
    return {"pages": pages,
            "mean_wer": round(sum(v["wer"] for v in pages.values()) / n, 4),
            "mean_cer": round(sum(v["cer"] for v in pages.values()) / n, 4),
            "junk": sum(v["junk"] for v in pages.values())}


def _pack_and_measure(tag: str) -> tuple[Path, float | None, float | None]:
    """Pack the current state to its own file and measure it on the held-out lines."""
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = train.latest_checkpoint()
    if ckpt is None:
        raise RuntimeError("no checkpoint to pack")
    out = SEG_DIR / f"{config.LANG}_{tag}.traineddata"
    p = subprocess.run([config.tool("lstmtraining"), "--stop_training",
                        "--continue_from", str(ckpt),
                        "--traineddata", str(STARTER),
                        "--model_output", str(out)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not out.is_file():
        raise RuntimeError(f"packing {tag} failed:\n{p.stdout}\n{p.stderr}")

    e = subprocess.run([config.tool("lstmeval"), "--model", str(out),
                        "--traineddata", str(STARTER),
                        "--eval_listfile", str(config.WORK_DIR / "eval.txt")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = _EVAL.search((e.stdout or "") + (e.stderr or ""))
    return out, (float(m.group(1)) if m else None), (float(m.group(2)) if m else None)


def run(total: int = 30000, segment: int = 2000, lr: float | None = None,
        patience: int = 3, from_scratch: bool = False) -> dict:
    """Train to `total` iterations in `segment`-sized pieces, keeping each piece.

    `patience` segments of rising held-out error stops the run — that is divergence, and
    E29 spent 3,400 iterations proving it does not come back on its own.
    """
    for p in (config.WORK_DIR / "train.txt", config.WORK_DIR / "eval.txt"):
        if not p.is_file() or not p.read_text(encoding="utf-8").strip():
            raise RuntimeError(f"{p} is missing or empty — run the `lstmf` step first")
    if not STARTER.is_file():
        raise RuntimeError("starter traineddata missing — run the `unicharset` step first")
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)

    if from_scratch:
        # A fresh fine-tune must not inherit the diverged rolling checkpoint, and the log is
        # the only record of where the iteration counter is.
        stale = list(config.WORK_DIR.glob(f"{config.LANG}*checkpoint"))
        if stale:
            # Its own folder per run. Checkpoint names repeat across runs — the early ones
            # especially, since they are named by an error rate the model passes through
            # every time — and moving onto an existing name fails on Windows, which would
            # abort a run before it started for no reason worth having.
            attic = config.WORK_DIR / "previous" / time.strftime("%Y%m%d-%H%M%S")
            attic.mkdir(parents=True, exist_ok=True)
            for s in stale:
                shutil.move(str(s), str(attic / s.name))
            print(f"moved {len(stale)} checkpoint(s) of the previous run to {attic}")
        (config.WORK_DIR / "train.log").write_text("", encoding="utf-8")

    n_train = len((config.WORK_DIR / "train.txt").read_text(encoding="utf-8").split())
    n_eval = len((config.WORK_DIR / "eval.txt").read_text(encoding="utf-8").split())
    print(f"training on {n_train} lines, held-out {n_eval}; "
          f"{total} iterations in segments of {segment}", flush=True)
    print(f"{'iter':>8} {'letter WER':>11} {'tech WER':>9} {'spring WER':>11} "
          f"{'mean WER':>9} {'BCER held':>10} {'best':>5}", flush=True)

    rows: list[dict] = []
    best = (None, None)          # (bcer, path)
    worse_runs = 0
    t0 = time.time()
    here = 0 if from_scratch else last_iteration()

    while here < total:
        target = min(here + segment, total)
        rc, log = _segment(target, lr, from_scratch and not rows)
        reached = last_iteration()
        if reached <= here:
            print(f"  lstmtraining made no progress at {target} (exit {rc}) — stopping")
            break
        here = reached

        path, bcer, bwer = _pack_and_measure(f"iter{here:06d}")
        pg = page_score(path)

        # Selection is on the pages. See `page_score` for the measurement that settled it.
        score = pg["mean_wer"] if pg else bcer
        is_best = score is not None and (best[0] is None or score < best[0])
        if is_best:
            best = (score, path)
            worse_runs = 0
        else:
            worse_runs += 1
        rows.append({"iteration": here, "bcer_held": bcer, "bwer_held": bwer,
                     "pages": pg, "selected_on": "mean_wer" if pg else "bcer_held",
                     "model": str(path), "best": is_best,
                     "elapsed_s": round(time.time() - t0)})
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")

        def _w(pid: str) -> str:
            v = (pg or {}).get("pages", {}).get(pid)
            return f"{v['wer'] * 100:8.2f}%" if v else f"{'-':>9}"
        print(f"{here:>8} {_w('letter_sabzgostar'):>11} {_w('tech_prose'):>9} "
              f"{_w('spring_prose'):>11} "
              f"{(f'{score * 100:8.2f}%' if pg and score is not None else '        -'):>9} "
              f"{(f'{bcer:9.2f}%' if bcer is not None else '        -'):>10} "
              f"{'<==' if is_best else '':>5}", flush=True)

        if "target error rate" in log.lower():
            print("  lstmtraining reached the target error rate — stopping")
            break
        if worse_runs >= patience:
            print(f"  held-out error has not improved for {patience} segments — "
                  f"stopping before it diverges further")
            break

    if best[1] is not None:
        shutil.copy2(best[1], train.OUT_TRAINEDDATA)
        print(f"\nbest held-out BCER {best[0]:.2f}% -> {train.OUT_TRAINEDDATA.name} "
              f"(from {best[1].name})")
    print(f"{len(rows)} segment(s) in {time.time() - t0:.0f}s; ledger at {LEDGER}")
    return {"segments": rows, "best_bcer": best[0],
            "best_model": str(best[1]) if best[1] else None}


def best_from_ledger() -> tuple[float, Path] | None:
    """The lowest held-out error any segment of any run reached, from the ledger alone."""
    if not LEDGER.is_file():
        return None
    best = None
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("bcer_held") is None:
            continue
        if best is None or r["bcer_held"] < best[0]:
            best = (r["bcer_held"], Path(r["model"]))
    return best
