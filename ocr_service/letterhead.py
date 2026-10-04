# -*- coding: utf-8 -*-
r"""Read the printed letterhead — the sender and the registration number.

The letterhead is the one part of an administrative letter the main pipeline cannot see. On
the brief's letter nothing above y=277 reaches the transcript at all, because the text there
is **white on a coloured band** and every stage downstream of binarisation expects dark ink
on light paper. 17 of 18 sampled pages in `ocr_eval/dataset_ex` are built that way.

Inverting the band in the page was measured in E26 and **rejected**: it costs the rest of the
page 2.1 pp of CER (11.3% → 13.4%), because the band's bounding box also contains white paper
carrying the form labels, and inverting that turns «شماره:», «تاریخ:» and «بسمه تعالی» into
white-on-black. This module therefore never touches the page the recogniser reads. It takes
its own crop, reads it separately, and contributes **only** to `fields`.

What separates text from background here is not brightness — the band is a gradient across
several greens with a logo in it, and both adaptive thresholding and global inversion pick up
the wedge edges as ink. It is **colour**: the text is unsaturated and bright, sitting in a
neighbourhood that is saturated. That generalises to any coloured letterhead without knowing
the hue, and it degrades to "no band found" on a plain dark-on-white page, which is the other
case and already works.

The brief's rule governs the output: a value is returned only when it matches the shape it
claims to be. A registration number is digits following a «ثبت» label; a sender is a line
carrying an organisation word. Anything else is left null rather than guessed — in particular
the stylised logotype of a company name is a drawing, and this does not pretend to read it.
"""
from __future__ import annotations

import re

import cv2
import numpy as np
from PIL import Image

# A pixel belongs to the band's background when its neighbourhood is this saturated.
BAND_SAT = 60
# ...and to the text when the pixel itself is this unsaturated and this bright.
TEXT_SAT, TEXT_VAL = 70, 140
# The letterhead lives in the top of the page. Searching further down starts finding
# coloured stamps, which are not letterheads.
TOP_FRAC = 0.25
# Below this there is no band worth cropping, only speckle.
MIN_TEXT_FRAC = 0.004
# A field is only reported when every word carrying it was read at least this confidently.
# Set from measurement: on the sampled letterheads the garbled readings sit well below it,
# which is the point — the band is reported as a region to look at instead.
MIN_FIELD_CONF = 0.80

_ORG = re.compile(r"(شرکت|سازمان|اداره|وزارت|دانشگاه|بانک|موسسه|مؤسسه|گروه|شهرداری|"
                  r"دفتر|هلدینگ|انجمن|کانون|مجتمع|صنایع|تعاونی)")
# «شماره ثبت: ۲۷۸۷۴», «ش.ث ۲۷۸۷۴», «ثبت ۲۷۸۷۴» — the label, then the digits.
_REG = re.compile(r"(?:شماره\s*ثبت|ش\s*\.?\s*ث|ثبت)\s*[:：.]?\s*([۰-۹0-9]{3,12})")
_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def text_mask(bgr: np.ndarray) -> np.ndarray:
    """Light text sitting on a coloured background, as a binary mask.

    The neighbourhood test is what makes this work. White paper is also unsaturated and
    bright; what distinguishes the letterhead's white text is that the pixels *around* it
    are coloured. A blurred saturation map answers that question per pixel, so the white
    form area below the band is excluded without having to know where the band ends.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1].astype(np.float32), hsv[:, :, 2]
    around = cv2.GaussianBlur(sat, (0, 0), 18)
    m = ((around > BAND_SAT) & (sat < TEXT_SAT) & (val > TEXT_VAL)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def text_rows(mask: np.ndarray, min_h: int = 12) -> list[tuple[int, int]]:
    """Row bands that carry ink, by horizontal projection.

    Lines are cut out and read one at a time rather than handing Tesseract the whole band:
    layout analysis on a graphic with wedges and a logo finds structure that is not text,
    and a single-line crop has no layout to analyse.
    """
    ink = mask.mean(axis=1)
    runs, start = [], None
    for i, v in enumerate(ink):
        if v > 0.012 and start is None:
            start = i
        elif v <= 0.012 and start is not None:
            if i - start >= min_h:
                runs.append((start, i))
            start = None
    if start is not None and len(ink) - start >= min_h:
        runs.append((start, len(ink)))
    return runs


def _render(mask: np.ndarray) -> np.ndarray:
    """The mask as black text on white paper, upscaled for a recogniser."""
    img = 255 - mask * 255
    img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _words(img: np.ndarray, tesseract: str, tessdata: str | None,
           lang: str) -> list[tuple[str, float]]:
    """(word, confidence) pairs, confidence as 0..1.

    Through pytesseract's `image_to_data`, the way the rest of the service reads Tesseract.
    The command line is not an option here: per-word confidence needs the `tsv` config, and
    config files are looked up inside `--tessdata-dir` — which for this project holds the
    models and no `configs/` at all, so Tesseract exits with `Can't open tsv`. The data
    directory goes in `TESSDATA_PREFIX` instead, exactly as `TesseractReader._cfg` does.
    """
    try:
        import pytesseract
    except ImportError:
        return []
    if tesseract:
        pytesseract.pytesseract.tesseract_cmd = tesseract
    if tessdata:
        import os
        os.environ["TESSDATA_PREFIX"] = tessdata
    try:
        d = pytesseract.image_to_data(Image.fromarray(img), lang=lang, config="--psm 6",
                                      output_type=pytesseract.Output.DICT)
    except Exception:
        return []
    out = []
    for i, t in enumerate(d.get("text", [])):
        t = (t or "").strip()
        if not t:
            continue
        try:
            c = float(d["conf"][i])
        except (TypeError, ValueError, KeyError):
            continue
        if c >= 0:
            key = (d.get("block_num", [0] * len(d["text"]))[i],
                   d.get("par_num", [0] * len(d["text"]))[i],
                   d.get("line_num", [0] * len(d["text"]))[i])
            out.append((t, c / 100.0, key))
    return out


def _lines(words: list[tuple[str, float, tuple]]) -> list[tuple[str, float]]:
    """Group words back into the lines Tesseract found, with each line's worst confidence.

    A field has to come from one reading of one line. Filtering individual words by
    confidence and joining what survives stitches together fragments of different lines —
    that produced «شرکت خدماتی سرکت», two correct words from one pass and a misread of the
    first word from another, which reads like a company name and is not one.
    """
    groups: dict[tuple, list[tuple[str, float]]] = {}
    for t, c, key in words:
        groups.setdefault(key, []).append((t, c))
    out = []
    for key in sorted(groups):
        ws = groups[key]
        out.append((" ".join(w for w, _ in ws), min(c for _, c in ws)))
    return out


def read(bgr: np.ndarray, tesseract: str, tessdata: str | None, lang: str) -> dict:
    """What the letterhead says, and how sure the recogniser was.

    `sender` and `registration_number` are filled only when the reading is both
    well-formed and confident. On the sampled pages most readings are neither: the company
    name is a stylised logotype and comes back as «۱ا، ۱۰ شرکت خدماتی نعم», the registration
    number as «۵ ۸۷۴ ۲۶ ۲». Returning those would be exactly the invented text the brief
    forbids, so they are dropped and `band` is reported instead — a region for the viewer to
    cut out and a human to read, which is what the brief asks for when something cannot be
    read confidently.
    """
    out: dict = {"sender": None, "registration_number": None,
                 "lines": [], "band": None, "confidence": 0.0}
    if bgr is None or bgr.ndim != 3:
        return out
    h = bgr.shape[0]
    top = bgr[: max(1, int(TOP_FRAC * h))]
    mask = text_mask(top)
    if mask.mean() < MIN_TEXT_FRAC:
        return out                      # a plain letterhead, or none: nothing to add here

    ys, xs = np.where(mask)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    out["band"] = [x0, y0, x1, y1]

    words = _words(_render(mask[y0:y1, x0:x1]), tesseract, tessdata, lang)
    if not words:
        return out
    lines = _lines(words)
    out["lines"] = [t for t, _ in lines]
    out["confidence"] = round(sum(c for _, c, _ in words) / len(words), 3)

    # Both fields come from a single line that was read confidently end to end. A line with
    # one shaky word in it is not a sender; it is a sender-shaped guess.
    for text, conf in lines:
        if conf < MIN_FIELD_CONF:
            continue
        m = _REG.search(text.translate(_DIGITS))
        if m and out["registration_number"] is None:
            out["registration_number"] = m.group(1)
        if out["sender"] is None and _ORG.search(text) and not m:
            out["sender"] = text.strip()
    return out
