# -*- coding: utf-8 -*-
r"""Extend the model's alphabet and build the starter traineddata.

Fine-tuning that only adjusts weights cannot teach a model a character it has no output
class for. The base `fas` model has 95 of them, and three absences shape the whole service:

* **«،» U+060C** — 717 occurrences in 2,000 benchmark labels. `Transcriber._fix_marks` is a
  hundred lines of heuristics restoring a glyph the model is structurally unable to emit.
* **Latin and «@»** — every line is read twice, once as `fas` and once as `eng`, and
  arbitrated, because `fas` has no Latin at all.
* **«٪» and «٬»** — the Persian percent and thousands separators.

So the alphabet is extended and the weights are carried over: `lstmtraining --old_traineddata`
maps the old unicharset onto the new one so the existing Persian stays learned and only the
new classes start from nothing.

The seed file matters. The alphabet must not depend on which characters happened to appear
in a sampled corpus, or two runs produce two different models with two different output
layers. Every character in `config.EXTRA_CHARS` is written to a seed file that is fed to the
extractor alongside the real ground truth, so the alphabet is a declared constant.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from . import config

UC_DIR = config.TRAIN / "uc"
LANGDATA = config.TRAIN / "langdata"
STARTER = UC_DIR / "out" / config.LANG / f"{config.LANG}.traineddata"


def _run(cmd: list[str], what: str) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"{what} failed ({p.returncode}):\n{p.stdout}\n{p.stderr}")


def gt_files() -> list[Path]:
    return sorted(config.REAL_DIR.glob("*.gt.txt")) + sorted(config.SYNTH_DIR.glob("*.gt.txt"))


def build() -> Path:
    """Merged unicharset -> starter traineddata. Returns the traineddata path."""
    if not (LANGDATA / "radical-stroke.txt").is_file():
        raise FileNotFoundError(
            f"{LANGDATA / 'radical-stroke.txt'} is missing. combine_lang_model loads it "
            f"unconditionally in this build, even with --pass_through_recoder. It comes from "
            f"tesseract-ocr/langdata_lstm.")

    UC_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = UC_DIR / "out"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # 1. the declared alphabet, so it does not depend on corpus sampling
    seed = UC_DIR / "charset_seed.gt.txt"
    seed.write_text(" ".join(config.EXTRA_CHARS) + "\n", encoding="utf-8")

    files = gt_files()
    if not files:
        raise RuntimeError("no ground truth found — run the `synth` and `benchmark` steps first")

    # One concatenated file, not 12,000 arguments: Windows caps a command line at 32 KB and
    # passing the ground truth per file overflows it. The extractor only reads text, so the
    # alphabet of the concatenation is the alphabet of the parts.
    all_gt = UC_DIR / "all_gt.txt"
    with all_gt.open("w", encoding="utf-8") as fh:
        fh.write(seed.read_text(encoding="utf-8"))
        for p in files:
            fh.write(p.read_text(encoding="utf-8").rstrip("\n") + "\n")

    # 2. the alphabet of the training data
    new_uc = UC_DIR / "new.unicharset"
    _run([config.tool("unicharset_extractor"),
          "--output_unicharset", str(new_uc),
          "--norm_mode", "2",                      # 2 = normalise like the LSTM engine does
          str(all_gt)],
         "unicharset_extractor")

    # 3. union with the base model's alphabet, so nothing the model already knows is lost
    merged = UC_DIR / "merged.unicharset"
    _run([config.tool("merge_unicharsets"),
          str(config.BASE_UNICHARSET), str(new_uc), str(merged)],
         "merge_unicharsets")

    # 4. the starter traineddata the trainer writes its checkpoints against.
    #    --pass_through_recoder: no compression of the code space. Persian is small enough
    #    that compression buys nothing and it keeps the mapping from the old unicharset
    #    straightforward for --old_traineddata.
    #    --lang_is_rtl: without it the model learns the wrong reading order.
    _run([config.tool("combine_lang_model"),
          "--input_unicharset", str(merged),
          "--script_dir", str(LANGDATA),
          "--output_dir", str(out_dir),
          "--lang", config.LANG,
          "--pass_through_recoder",
          "--lang_is_rtl"],
         "combine_lang_model")

    if not STARTER.is_file():
        raise RuntimeError(f"combine_lang_model did not write {STARTER}")

    base_n = len([l for l in config.BASE_UNICHARSET.read_text(encoding="utf-8").splitlines()[1:] if l.strip()])
    new_n = len([l for l in merged.read_text(encoding="utf-8").splitlines()[1:] if l.strip()])
    print(f"alphabet: {base_n} -> {new_n} characters; starter at {STARTER}")
    return STARTER


def added() -> list[str]:
    """Characters the extended alphabet has that the base did not."""
    def chars(p: Path) -> set[str]:
        return {l.split(" ")[0] for l in p.read_text(encoding="utf-8").splitlines()[1:]
                if l.strip() and len(l.split(" ")[0]) == 1}
    merged = UC_DIR / "merged.unicharset"
    if not merged.is_file():
        return []
    return sorted(chars(merged) - chars(config.BASE_UNICHARSET))
