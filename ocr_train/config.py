# -*- coding: utf-8 -*-
r"""Paths, fonts and the alphabet extension. Everything tunable about the fine-tune.

Artifacts live outside the package, under `train/`, because they are large (the line
corpus runs to hundreds of megabytes) and are not source. `train/` is git-ignored; the
scripts that build it are not.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "train"

BASE_DIR = TRAIN / "base"          # the float model we fine-tune from, unpacked
SYNTH_DIR = TRAIN / "synth"        # line images rendered from the project's fonts
REAL_DIR = TRAIN / "real"          # line images carved out of the benchmark
LSTMF_DIR = TRAIN / "lstmf"        # Tesseract's own training format
WORK_DIR = TRAIN / "work"          # checkpoints
OUT_DIR = TRAIN / "out"            # fas_custom.traineddata
FONTCONFIG = TRAIN / "fontconfig"

FONTS_DIR = ROOT / "ocr_service" / "fonts"
BENCHMARK = ROOT / "ocr_service" / "Persian-OCR-Double-Benchmark" / "data"

BASE_TRAINEDDATA = BASE_DIR / "fas_best.traineddata"
BASE_LSTM = BASE_DIR / "fas.lstm"
BASE_UNICHARSET = BASE_DIR / "fas.lstm-unicharset"

LANG = "fas_custom"

# --- Tesseract binaries ---------------------------------------------------------------
# The training tools ship with the UB-Mannheim build; a distro Tesseract often omits them.
TESS_DIR = Path(os.environ.get("TESSERACT_DIR", r"C:\Program Files\Tesseract-OCR"))


def tool(name: str) -> str:
    """Full path to a Tesseract training binary, or the bare name if it is on PATH."""
    exe = TESS_DIR / f"{name}.exe"
    if exe.is_file():
        return str(exe)
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(
            f"{name} not found in {TESS_DIR} or on PATH. The Tesseract *training* tools "
            f"(lstmtraining, text2image, combine_lang_model, …) are a separate component "
            f"of the installer and must be present to fine-tune."
        )
    return found


def env() -> dict:
    """Environment for text2image.

    Pango on Windows refuses to render without a fontconfig file — it prints
    `Cannot load default config file` and writes an empty box file, which looks like a
    font problem and is not. `write_fontconfig()` lays one down; this points at it.
    """
    e = dict(os.environ)
    e["FONTCONFIG_PATH"] = str(FONTCONFIG)
    e["FONTCONFIG_FILE"] = str(FONTCONFIG / "fonts.conf")
    return e


def write_fontconfig() -> None:
    (FONTCONFIG / "cache").mkdir(parents=True, exist_ok=True)
    (FONTCONFIG / "fonts.conf").write_text(
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n'
        "<fontconfig>\n"
        f"  <dir>{FONTS_DIR.as_posix()}</dir>\n"
        "  <dir>C:/Windows/Fonts</dir>\n"
        f"  <cachedir>{(FONTCONFIG / 'cache').as_posix()}</cachedir>\n"
        "  <config></config>\n"
        "</fontconfig>\n",
        encoding="utf-8")


# --- fonts -----------------------------------------------------------------------------
# The faces the target documents are actually set in. Names are as Pango resolves them,
# which is not always the file name: `--list_available_fonts` is the authority, and
# text2image refuses a name it cannot resolve exactly rather than substituting silently.
# The project's own faces — the ones administrative letters are set in. Every one of them
# was probed (`python -m ocr_train fonts`) and renders Persian; every one of them has
# **zero Latin glyphs**, which matters more than it sounds: a line containing "Tel" renders
# with a blank space where the word should be, and training on that teaches the model that
# Latin text is whitespace. So Latin never goes to these.
#
# The IPT faces are listed by Pango and render **nothing at all** — 0 ink for Persian too.
# They are not in this list for that reason, not by oversight.
FONTS_FA = [
    "B Nazanin Bold",
    "B Nazanin Outline",
    "B Lotus",
    "B Lotus Bold",
    "B Mitra",
    "B Mitra Bold",
    "B Kamran Bold",
    "B Kamran Outline",
    "B Yekan",
]

# Faces carrying both scripts, for the bilingual lines and e-mail addresses. These are
# system fonts rather than project ones, and that is realistic: an address block or an
# English product name in a Persian letter is routinely set in Tahoma or Arial.
FONTS_DUAL = [
    "Tahoma",
    "Tahoma Bold",
    "Arial",
    "Arial Bold",
    "Times New Roman",
    "Segoe UI",
    "Calibri",
    "Microsoft Sans Serif",
]

FONTS = FONTS_FA + FONTS_DUAL

# Rendered sizes. The service upsamples anything under 20 px to ~30 px (E21), so the model
# should see the range it will actually be given, not one clean size.
PT_SIZES = [11, 13, 16, 20, 26]

# --- the alphabet extension --------------------------------------------------------------
# Characters the base `fas` model cannot emit, added here because the pipeline has been
# working around their absence:
#
#   «،»  U+060C  the Persian comma. 717 occurrences in 2,000 benchmark labels, and the
#                single commonest thing wrong with an otherwise correct line (E25). The
#                `_fix_marks` stage exists only to put back a character the model has no
#                output class for.
#   Latin + «@»  every bilingual line currently has to be read twice, once with `fas` and
#                once with `eng`, and arbitrated — because `fas` has no Latin at all. The
#                letter's own e-mail address comes back as «info@sabzgostar-govareshk ۱۲” ;».
#   «ۀ» «٬»      heh with hamza and the Persian thousands separator; both occur in the
#                corpus and both are currently unrepresentable.
EXTRA_CHARS = (
    ["،", "ۀ", "٬", "@", "%", "٪"]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
)

# --- training ----------------------------------------------------------------------------
MAX_ITERATIONS = 6000      # a fine-tune, not a retrain: the base is already good Persian
# lstmtraining's default is 1e-3, which is a rate for training from scratch. Continuing from
# a *recognition* model there is no optimizer state to inherit, so the first steps are large
# and they land on a network that already knows Persian — at 1e-3 this one collapsed, in the
# textbook CTC way: BCER 62% -> 75% -> 97% over 2,000 iterations while rms stayed near 5%,
# and the decoder emitted «ار    ی» for every line. The targets were verified correct at the
# same time (lstmeval printed the truth back perfectly), so the data was never the problem.
LEARNING_RATE = 1e-4
RESET_LEARNING_RATE = True
TARGET_ERROR_RATE = 0.05
EVAL_FRACTION = 0.2        # of the benchmark, held out by content hash and never trained on
SEED = 6669                # the benchmark's own selection seed, reused for the split
