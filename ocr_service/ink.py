# -*- coding: utf-8 -*-
r"""Separate the page's ink layers before it is binarised.

Two kinds of ink on an administrative letter are not black text on white paper, and
`flatten()` — which divides by a blurred background and keeps what is darker — gets both
of them wrong:

  * **A stamp or a signature in blue ink.** It is darker than the paper, so it survives
    binarisation and lands on top of the text it covers. On the brief's image 1 the
    company stamp sits over the signature block and «با تشکر / هادی جهان گشته / مدیر
    عامل» comes back as «هادی جهت کته». Blue ink is separable from black by colour and
    nothing else: both are dark.

  * **A coloured letterhead with white text.** The text is *lighter* than its background,
    so dividing by that background makes it 255 — the same value as paper. The sender and
    the registration number are not misread on image 1; they are not in the binarised
    image at all.

Both are fixed here, by colour, before any thresholding happens:

    drop_blue_ink   blue-excess (B above both R and G) erases stamp and signature
    invert_bands    a large region whose own background is dark, holding light strokes,
                    is inverted in place so the existing pipeline sees ordinary dark text

Black ink is defined by the *absence* of colour, so neither step can touch it: the test
for blue is a channel difference, not a brightness, and a near-black pixel has none.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

# How much the blue channel must exceed both others for a pixel to be coloured ink rather
# than black ink that a scanner has tinted. Measured on the corpus stamps: blue ink runs
# 40-90, black body text 0-8.
BLUE_EXCESS = 28
# A pixel this dark is a printed letter, whatever colour a stamp has tinted it, and is
# never erased as stamp ink.
DARK_IS_TEXT = 70
# A band must cover at least this share of the page to be treated as a letterhead. Smaller
# dark patches are shadows, binder rings and stamp bodies, and inverting one of those turns
# a shadow into a page of noise.
MIN_BAND_FRAC = 0.004
# Inside a candidate band, the share of pixels that are much brighter than the band itself.
# Text occupies a few percent of its line box; 0 means a plain dark block (a shadow or a
# photograph) and a high value means the "band" is really paper.
BAND_LIGHT_LO, BAND_LIGHT_HI = 0.015, 0.45


@dataclass
class InkInfo:
    blue_pixels: int = 0
    furniture: int = 0                             # margin objects removed (rings, holes, specks)
    bands: list = field(default_factory=list)      # [x0, y0, x1, y1] per inverted band

    def as_dict(self) -> dict:
        return {"blue_pixels": self.blue_pixels, "furniture_removed": self.furniture,
                "inverted_bands": self.bands}


def blue_mask(bgr: np.ndarray) -> np.ndarray:
    """Pixels whose blue channel stands clearly above both others — stamp and pen ink."""
    b, g, r = (bgr[:, :, i].astype(np.int16) for i in range(3))
    return (b - np.maximum(g, r)) > BLUE_EXCESS


def drop_blue_ink(bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """Erase coloured ink by setting it to the paper around it.

    Two rules keep this from costing more than it buys, and both were measured:

    * **Fill, do not inpaint.** `cv2.inpaint` diffuses colour in from the mask boundary,
      which perturbs pixels near every stamp. Through a pipeline whose line selection is a
      global search, that was enough to change readings far away from any stamp: on the
      18 dev pages that were not cropped at all — where erasing blue was the only change —
      whole numbers fell 5.0 pp, and the brief's image 1 lost the hyphens in its phone
      line. A flat fill with the local paper level changes the masked pixels and nothing
      else.
    * **Never overwrite a dark pixel.** Where the stamp crosses a printed letter, the
      letter's own ink dominates and the pixel is near black. Those pixels are blue enough
      to be masked and must survive anyway, or removing the stamp removes the text it was
      covering — the opposite of the point.
    """
    m = blue_mask(bgr)
    if not m.any():
        return bgr, 0
    # the anti-aliased rim of a stamp stroke is not blue enough to pass the test on its own
    m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    m &= bgr.max(2) >= DARK_IS_TEXT            # keep whatever is really a letter
    if not m.any():
        return bgr, 0
    out = bgr.copy()
    out[m] = np.maximum(_paper_level(bgr)[m], bgr[m])   # at least as light as the paper
    return out, int(m.sum())


def _paper_level(bgr: np.ndarray) -> np.ndarray:
    """The paper colour behind every pixel.

    A blur is the wrong estimate: in the middle of a stamp the blur is still stamp, so
    erasing the stamp left a grey ghost of it that binarised as ink. Paper is the brightest
    thing on the page, so a morphological close (a max filter, then a min) recovers it even
    under a large mark, and a light blur takes the blockiness out.
    """
    k = max(15, int(min(bgr.shape[:2]) / 25)) | 1
    closed = cv2.morphologyEx(bgr, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    closed = cv2.GaussianBlur(closed, (0, 0), max(3, k / 6))
    # A close can only reach `k` pixels in, so the middle of a large solid stamp is still
    # stamp-coloured and the fill would leave a ghost of it that binarises as ink. The
    # page's own paper level is the floor: it is only ever used where the mask says the
    # pixel is coloured ink, so brightening it cannot touch real text.
    level = np.percentile(bgr.reshape(-1, 3), 90, axis=0).astype(np.uint8)
    return np.maximum(closed, level)


def _band_candidates(gray: np.ndarray) -> list[np.ndarray]:
    """Large regions whose own local background is well below the paper level."""
    h, w = gray.shape
    bg = cv2.GaussianBlur(gray, (0, 0), max(8, min(h, w) / 60))
    paper = float(np.percentile(bg, 92))
    if paper <= 1:
        return []
    dark = (bg < 0.80 * paper).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    out = []
    for i in range(1, n):
        if st[i, 4] >= MIN_BAND_FRAC * h * w:
            out.append((lab == i, st[i]))
    return out


def invert_bands(bgr: np.ndarray) -> tuple[np.ndarray, list]:
    """Invert any large dark band that carries light text, so the rest of the pipeline
    sees black-on-white there like everywhere else."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out, boxes = bgr, []
    for region, st in _band_candidates(gray):
        # Measure inside the band, not across its edge. The candidate mask is grown by a
        # morphological close, so its rim straddles the step from band to paper; those rim
        # pixels are "much brighter than the band" and made a solid dark block — a shadow,
        # a photograph — look as though it carried light text.
        # eroded by more than `_band_candidates` grew the region, or the paper just outside
        # it stays in the sample and reads as 3% "light text" on a blank dark block
        core = cv2.erode(region.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
        vals = gray[core if core.any() else region]
        if vals.size < 64:
            continue
        level = float(np.median(vals))
        light = float((vals > level + 45).mean())
        if not (BAND_LIGHT_LO <= light <= BAND_LIGHT_HI):
            continue                        # a plain dark block, or not really a band
        if out is bgr:
            out = bgr.copy()
        out[region] = 255 - out[region]
        x, y, bw, bh, _ = st
        boxes.append([int(x), int(y), int(x + bw), int(y + bh)])
    return out, boxes


def drop_furniture(bgr: np.ndarray, band: float = 0.10) -> tuple[np.ndarray, int]:
    """Erase the binder ring, punch holes and scanner specks in the page margins.

    These are what puts a lone «۰», «سس» or «سح» on the end of an otherwise correct line:
    in right-to-left text the left margin is where lines *end*, so a mark sitting there
    is read as the line's last word. The brief's image 1 has a spiral binder lying across
    its left edge, and its dark tip is the «۰» in «... آن اداره کل ۰».

    Only two kinds of component are removed, and a letter can be neither:

    * **Too big to be a glyph** — taller than 2.5 line heights or wider than six. A binder
      ring, a punch hole, a staple, the shadow of a page edge.
    * **Alone** — a small mark in the margin with no glyph-sized neighbour within two line
      heights. A real word at the end of a line has the rest of the line beside it.
    """
    h, w = bgr.shape[:2]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bgl = cv2.GaussianBlur(g, (0, 0), max(8, min(h, w) / 60))
    ink = (g.astype(np.float32) / np.maximum(bgl, 1) * 255 < 200).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n < 3:
        return bgr, 0
    # The robust estimator, not a plain median of every component. On a 12 px capture the
    # dots and the scan noise outnumber the words, so the plain median lands near 3 px and
    # "taller than 2.5 glyphs" then describes ordinary text: this stage deleted 5% of the
    # low-resolution page (CER 9.0% → 12.3%) until it asked the right question.
    from .transcribe import glyph_height
    gh = glyph_height(st, h)
    if gh <= 0:
        return bgr, 0

    # The margin is defined by where the TEXT is, not by where the paper ends. Measuring it
    # from the page edge cost the letter its first word on six lines — Persian runs
    # right-to-left, so the right margin is where every line begins, and a fixed 10% band
    # reached into «جناب», «سرپرست», «انجام» and «نماینده».
    glyph = [i for i in range(1, n) if 0.5 * gh <= st[i, 3] <= 2.5 * gh and st[i, 4] >= 20]
    if len(glyph) < 20:
        return bgr, 0
    xs = np.array([st[i, 0] for i in glyph] + [st[i, 0] + st[i, 2] for i in glyph])
    lo, hi = float(np.percentile(xs, 1)) - gh, float(np.percentile(xs, 99)) + gh
    lo, hi = min(lo, band * w), max(hi, (1 - band) * w)

    drop = []
    for i in range(1, n):
        x, y, cw, ch, area = st[i]
        if x >= lo and x + cw <= hi:
            continue                                  # inside the text block
        # Only objects no letter could be, and only by HEIGHT. Width says nothing in
        # Persian: a word is a single connected component and a kashida-stretched one is
        # very wide, so a width rule deleted whole words from the low-resolution page
        # (CER 9.0% → 12.3%). A speck is left to `clean_ink`, which already drops one with
        # no glyph beside it.
        if ch > 2.5 * gh:
            drop.append(i)
    if not drop:
        return bgr, 0
    m = np.isin(lab, drop)
    m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    paper = cv2.GaussianBlur(bgr, (0, 0), max(10, min(h, w) / 40))
    out = bgr.copy()
    out[m] = np.maximum(paper[m], 235)
    return out, len(drop)


def separate(im: Image.Image, bands: bool = False,
             furniture: bool = False) -> tuple[Image.Image, InkInfo]:
    """The page with coloured ink erased, and optionally light-on-dark bands flipped.

    `furniture` is off by default, and the measurement is why. Erasing margin objects
    sounds obviously right and buys nothing: across the three pages it changed the letter's
    transcript not at all — the binder ring's «۰» survives it, because the ring lies inside
    the text block's own x-range — while on the low-resolution photo it cost CER 9.0% →
    10.8% and WER 25.2% → 29.9%. The stray marks it was written for are better removed at
    the line level, where a token's neighbours are known, than by guessing at pixels.

    `bands` is off by default, and the measurement is why. Inverting the letterhead does
    put its pixels the right way up, but a letterhead is a multi-tone graphic, and what
    Tesseract then reads off it is «ام ٍ شرکت خدماتی 8۵ وی ی ده» — the company name is in
    there, wrapped in noise that costs more than the name is worth: on the brief's image 1
    whole-page CER 11.3% → 13.4% and WER 19.0% → 25.3%, with one extra junk line. Under
    "correct and incomplete beats complete and wrong" that is a losing trade, so the band
    stays unread until it can be read cleanly (its own crop, its own binarisation, its own
    confidence floor) rather than thrown into the page pass.
    """
    bgr = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    info = InkInfo()
    bgr, info.blue_pixels = drop_blue_ink(bgr)
    if furniture:
        bgr, info.furniture = drop_furniture(bgr)
    if bands:
        bgr, info.bands = invert_bands(bgr)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), info
