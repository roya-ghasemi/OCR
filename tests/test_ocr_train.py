# -*- coding: utf-8 -*-
"""Guards for the fine-tuning pipeline (E29).

Only the parts whose failure is **silent** are pinned here. Training is not run in the test
suite; what is tested is the data that goes into it, because every bug found while building
this produced a plausible-looking corpus and a model that trained happily on the wrong thing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ZWNJ = "‌"
KASHIDA = "ـ"


def test_targets_are_written_in_visual_order():
    """The costliest bug in this pipeline, and it announces itself as nothing at all.

    The box file gives every character the same rectangle, so Tesseract has no coordinates
    to sort by and the file order *is* the target order. The network scans left to right, so
    Persian targets must be visual. In logical order the stock model scores BWER 100% on a
    line it reads perfectly, and training teaches the model to read backwards.
    """
    from ocr_train.lstmf import visual_order
    assert visual_order("ابج") == "جبا"


def test_visual_order_keeps_latin_and_digits_forwards():
    """A date, an amount or an e-mail inside Persian is a left-to-right run inside a
    right-to-left line. Naive reversal would spell them backwards."""
    from ocr_train.lstmf import visual_order
    for run in ("info@sabzgostar.ir", "۱۴۰۲/۱۲/۲۷", "Tel"):
        assert run in visual_order(f"ایمیل: {run} پایان")


def test_visual_order_keeps_the_zwnj():
    """python-bidi classes ZWNJ as boundary-neutral and deletes it. Unguarded, every
    «می‌شود» and «نیازمندی‌های» would lose its joiner in the targets."""
    from ocr_train.lstmf import visual_order
    src = f"نیازمندی{ZWNJ}های می{ZWNJ}شود"
    out = visual_order(src)
    assert out.count(ZWNJ) == src.count(ZWNJ)
    assert sorted(out) == sorted(src)          # nothing added, nothing dropped


def test_a_box_file_has_one_entry_per_character_and_a_terminator(tmp_path):
    from PIL import Image

    from ocr_train.lstmf import visual_order, write_line_box
    img = tmp_path / "l.png"
    Image.new("L", (200, 40), 255).save(img)
    text = "سلام ۱۲۳"
    box = write_line_box(img, text)
    rows = box.read_text(encoding="utf-8").splitlines()
    assert len(rows) == len(visual_order(text)) + 1
    assert rows[-1].startswith("\t")
    assert all(r.endswith(" 0 0 200 40 0") for r in rows)


def test_kashida_goes_only_where_persian_allows_it():
    """A kashida after «ا د ذ ر ز ژ و» is a floating dash, not an elongation — those letters
    do not join to what follows."""
    from ocr_train.corpus import kashida_positions
    assert kashida_positions("دارد") == []          # every letter is non-connecting
    assert kashida_positions("سلام")                # س and ل join forwards
    for w in ("بلوار", "مشهد", "استخراج"):
        for i in kashida_positions(w):
            assert w[i] not in "اآأإدذرزژوؤةء", f"{w!r} position {i}"


def test_the_label_never_carries_the_stretch():
    """The model is taught that a kashida means nothing, which only works if the ground
    truth is the unstretched text."""
    from ocr_train.corpus import generate
    pairs = generate(ROOT, 150, seed=3)
    stretched = [(r, l) for r, l in pairs if KASHIDA in r]
    assert stretched, "no justified lines were generated at all"
    for render, label in stretched:
        assert KASHIDA not in label
        assert render.replace(KASHIDA, "") == label


def test_the_synthetic_vocabulary_has_no_arabic_letterforms():
    """Tesseract's Persian word list is full of «مدرك» and «ويژگي». Rendering those would
    train in the exact non-canonical forms the service normalises away (D8/D43)."""
    from ocr_train.corpus import load_vocab
    admin, general = load_vocab(ROOT)
    if not general:
        pytest.skip("word lists not present")
    for pool in (admin, general):
        assert not [w for w in pool if set(w) & set("يكىة")]


def test_latin_lines_never_go_to_a_persian_only_face():
    """Every B-series font has zero Latin glyphs: a line containing "Tel" would render with
    a hole where the word is, and the label would describe an image that does not exist."""
    from ocr_train import config
    assert set(config.FONTS_FA).isdisjoint(config.FONTS_DUAL)
    assert config.FONTS_DUAL, "there must be a face that can render Latin"


def test_the_alphabet_extension_covers_what_the_base_model_cannot_emit():
    from ocr_train import config
    for c in ("،", "@", "a", "A", "٪"):
        assert c in config.EXTRA_CHARS


@pytest.mark.skipif(not (ROOT / "train" / "base" / "fas.lstm-unicharset").is_file(),
                    reason="base model not unpacked")
def test_the_base_alphabet_really_is_missing_the_persian_comma():
    """The premise of the whole phase. If this ever fails, the extension is unnecessary."""
    uc = (ROOT / "train" / "base" / "fas.lstm-unicharset").read_text(encoding="utf-8")
    chars = {l.split(" ")[0] for l in uc.splitlines()[1:] if l.strip()}
    assert "،" not in chars
    assert "a" not in chars


def test_degradation_keeps_the_line_a_line(tmp_path):
    """Augmentation may make a line hard to read. It may not make it something else: the
    label still has to describe the image, so the strip stays a strip and stays legible
    enough to have ink in it."""
    import random

    import numpy as np
    from PIL import Image

    from ocr_train.degrade import MAX_TEXT_PX, MIN_TEXT_PX, apply
    rng = random.Random(0)
    src = Image.new("L", (400, 48), 255)
    src.paste(0, (20, 14, 380, 30))                 # a bar of "ink"
    import math
    for _ in range(40):
        out = apply(src, rng)
        assert out.width > out.height, "a text line must stay wider than it is tall"
        # `tilt` rotates with expand=True, so a wide strip gains width*sin(angle) of height
        # on top of whatever the resample asked for. That growth is real and bounded.
        allow = math.ceil(out.width * math.sin(math.radians(0.9))) + 4
        assert MIN_TEXT_PX - 2 <= out.height <= max(MAX_TEXT_PX, src.height) + allow
        a = np.asarray(out)
        assert a.min() < 200, "the ink was degraded out of existence"
        assert a.max() > 120, "the background was degraded into ink"


def test_most_synthetic_lines_are_degraded():
    """Clean renders alone do not transfer to a scan — that was measured, not assumed."""
    from ocr_train import config
    assert 0 <= config.CLEAN_IN_TEN <= 5


def test_the_corpus_covers_the_scale_the_letter_is_actually_read_at():
    """E29's corpus topped out at 58 px and the brief's letter reads at 73 px per line.

    A model that has never seen type at the size it will be handed is being asked to
    generalise across scale, which is exactly the generalisation CTC is worst at.
    """
    from ocr_train.degrade import MAX_TEXT_PX, REGIMES
    scan = next(r for r in REGIMES if r[0] == "scan")
    assert scan[2][0] <= 73 <= scan[2][1], "the scan regime must span the letter's line height"
    assert MAX_TEXT_PX >= 73


def test_the_capture_regimes_are_a_distribution():
    """Each regime is a device, fixing tone *and* resolution together. Shares must sum to
    one or some lines silently fall through to whichever branch is last."""
    from ocr_train.degrade import REGIMES
    assert abs(sum(share for _, share, _ in REGIMES) - 1.0) < 1e-9
    for name, share, (lo, hi) in REGIMES:
        assert 0 < share < 1 and 0 < lo < hi


def test_a_degraded_line_still_has_ink_in_it():
    """The failure this guards is a picture of blank paper carrying a label that describes a
    sentence. Nothing downstream can detect that; it just teaches the model noise."""
    import random

    import numpy as np
    from PIL import Image

    from ocr_train.degrade import READABLE_INK, apply, bright_scan, faint
    rng = random.Random(3)
    src = Image.new("L", (500, 70), 255)
    src.paste(90, (20, 22, 480, 46))            # already-grey ink, the hard case
    for _ in range(60):
        assert np.asarray(bright_scan(faint(src, rng), rng)).min() <= READABLE_INK
    # and the invariant holds for every regime, not only the one that adapts
    for _ in range(120):
        assert np.asarray(apply(src, rng)).min() <= READABLE_INK
