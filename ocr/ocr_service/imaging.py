# -*- coding: utf-8 -*-
"""Image operations of the page path: binarisation, rule removal, glyph size, skew,
line finding and kashida squeezing. Pure functions over PIL images and numpy masks."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

# ------------------------------------------------------------------ image preparation
def flatten(im: Image.Image, scale: float = 1.0, thr: int = 200, radius: int = 40) -> Image.Image:
    """Grey -> divide by a blurred background (removes shading, shadows, coloured
    paper) -> binary (text black on white). `radius` 40 is what was measured (E19);
    working images are 2,000-4,200 px after `working_scale`."""
    g = ImageOps.grayscale(im)
    if scale != 1.0:
        g = g.resize((max(1, int(g.width * scale)), max(1, int(g.height * scale))), Image.LANCZOS)
    a = np.asarray(g, dtype=np.float32)
    bg = np.asarray(g.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    return Image.fromarray(np.where(a / np.maximum(bg, 1) * 255 < thr, 0, 255).astype(np.uint8))


# A crop that is one line of text rather than a page: far wider than it is tall, and short
# in absolute terms. Both bounds are needed — a wide banner scan is not a line crop, and a
# tall narrow column is not one either.
LINE_CROP_RATIO = 6.0
LINE_CROP_MAX_H = 200


def is_line_crop(im: Image.Image) -> bool:
    """Is this image one cropped text line rather than a page?"""
    w, h = im.size
    return h > 0 and h <= LINE_CROP_MAX_H and (w / h) >= LINE_CROP_RATIO


def working_scale(im: Image.Image, glyph_h: float = 0.0) -> float:
    """Tesseract reads best with glyphs ~30-40 px tall: upsample phone photos and small
    screenshots (Lanczos, in `flatten`), cap huge scans. `glyph_h` is the text height
    measured at 1x.

    A Persian dot is 1-2 px at 12 px text height, so on a low-resolution photo the marks
    that separate ب/ی/پ/ن and ح/خ/ج fall below the sampling grid and the page comes back
    full of «صبخگاهی» for «صبحگاهی». Resampling cannot invent the missing ink, but it
    gives Tesseract's line model room to resolve what ink there is: on the user's 12 px
    page, resampling to 30 px leaves whole-page CER where it was (8.8% → 9.2%, one page)
    and takes the marker words it reads correctly from 4 of 15 to 9 — «توسعه», «کوچه»,
    «زندگی», «ارزشمند», «اصلاحی». So a page whose text is under 20 px **as supplied** is
    taken to ~30, at most 4x (E21).

    Only such a page. Applying the same target to everything under 30 px *working*
    height upsampled the 24-29 px corpus scans too and cost dev coverage CER
    14.7% → 15.1% and whole numbers 75.8% → 75.2%; a scan that already resolves its own
    dots keeps the rule E19 was measured with. Resampling cannot invent the missing ink —
    the page is still flagged `needs_review` for its resolution."""
    scale = 2.0 if im.height < 2000 else 1.0
    if max(im.size) > 4200:
        scale = 3500 / max(im.size)
    if glyph_h and glyph_h < 20:                     # a low-resolution capture
        scale = min(4.0, 30.0 / glyph_h)
    elif glyph_h and glyph_h * scale < 24:           # small image of small text (E19)
        scale = min(4.0, 32.0 / glyph_h)
    return scale


def measure_glyph_height(im: Image.Image) -> float:
    ink = np.asarray(flatten(im)) == 0
    _, _, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    return glyph_height(st, ink.shape[0])


def glyph_height(st: np.ndarray, H: int) -> float:
    """Median height of glyph-like components (connectedComponentsWithStats rows).
    Whole words are single components in Persian, so this tracks the text size.
    Bounds are loose on purpose: an image may hold a few lines of large text."""
    h, area = st[1:, 3], st[1:, 4]
    base = (area >= 20) & (h > 0.004 * H)
    wide = base & (h <= 0.3 * H)
    if wide.sum() < 3:
        return 0.0
    # Robust estimate: drop everything under half the 75th percentile (dots, marks),
    # then the median. Needed when dots outnumber words (a few large clean lines).
    q = float(np.percentile(h[wide], 75))
    robust = float(np.median(h[wide & (h >= 0.5 * q)]))
    # The scan-scale estimate E19 was measured with; kept unless dots dominate it.
    page = base & (h < 0.05 * H)
    if page.sum() >= 5:
        m = float(np.median(h[page]))
        if m >= 0.4 * robust:
            return m
    return robust


def clean_ink(ink: np.ndarray) -> np.ndarray:
    """Remove straight rules (pixel-wise) and isolated specks; keep glyphs and their dots.

    Rule kernels are sized from the TEXT as well as the page: an alef is a straight
    vertical stroke, and a kashida-stretched word a long horizontal one, so on an
    image of a few large lines a page-relative kernel erased every «ا»."""
    H, W = ink.shape
    u8 = ink.astype(np.uint8)
    _, _, st0, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    gh = glyph_height(st0, H)
    hk = max(25, W // 4, int(15 * gh))
    vk = max(25, H // 25, int(4 * gh))
    hl = cv2.morphologyEx(u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    vl = cv2.morphologyEx(u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
    ink = ink & ~(hl > 0) & ~(vl > 0)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    area = st[:, 4]
    keep = np.ones(n, bool)
    keep[0] = False
    # A small component is a Persian dot next to a letter body, or a speck of scan /
    # shadow noise when nothing big is near it.
    small = area < 40
    big_mask = (keep & ~small)[lab]
    r = max(3, int(0.006 * H), int(0.5 * gh))
    near = cv2.dilate(big_mask.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
    hit = np.zeros(n, bool)
    np.logical_or.at(hit, lab[near & (lab > 0)], True)
    keep &= (~small | hit) & (area >= 3)
    return keep[lab]


def text_skew(im: Image.Image, max_deg: float = 10.0) -> float:
    """Rotation (degrees, counter-clockwise positive) that levels the text rows:
    maximises the sharpness of the row profile of glyph-sized components."""
    small = im.copy()
    small.thumbnail((1400, 1400))
    ink = np.asarray(flatten(small)) == 0
    _, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    h, w, area = st[:, 3], st[:, 2], st[:, 4]
    H, W = ink.shape
    gh = glyph_height(st, H)
    ok = (area >= 8) & (h < max(0.05 * H, 2.5 * gh)) & (w < 0.3 * W)
    ok[0] = False
    m = ok[lab].astype(np.uint8) * 255

    def sharpness(a: float) -> float:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), a, 1.0)
        p = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_NEAREST).sum(1).astype(np.float64)
        return float(np.sum(np.diff(p) ** 2))

    best = max(np.arange(-max_deg, max_deg + 1e-6, 0.5), key=sharpness)
    return float(max(np.arange(best - 0.5, best + 0.5 + 1e-6, 0.1), key=sharpness))


# ------------------------------------------------------------------ line finding
def _rows_to_lines(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Row-profile runs, over-tall runs split at profile valleys."""
    H, W = mask.shape
    prof = np.convolve(mask.sum(1).astype(np.float32), np.ones(3) / 3, "same")
    on = prof > max(2, 0.002 * W)
    runs, y = [], 0
    while y < H:
        if on[y]:
            s = y
            while y < H and on[y]:
                y += 1
            runs.append([s, y])
        else:
            y += 1
    runs = [r for r in runs if r[1] - r[0] >= max(6, int(H * 0.004))]
    if not runs:
        return []
    med = sorted(r[1] - r[0] for r in runs)[len(runs) // 2]
    spans = []
    for s, e in runs:
        if e - s > 1.7 * med:
            k = max(2, round((e - s) / med))
            cuts = []
            for i in range(1, k):
                c = s + int(i * (e - s) / k)
                lo, hi = max(s + 3, c - med // 2), min(e - 3, c + med // 2)
                if hi > lo:
                    cuts.append(lo + int(np.argmin(prof[lo:hi])))
            b = [s] + cuts + [e]
            spans += [(b[i], b[i + 1]) for i in range(len(b) - 1)]
        else:
            spans.append((s, e))
    out = []
    for s, e in spans:
        cols = np.where(mask[s:e].sum(0) > 0)[0]
        if len(cols):
            out.append((int(cols[0]), s, int(cols[-1]) + 1, e))
    return out


def text_lines(ink: np.ndarray) -> tuple[list[tuple[int, int, int, int]], float]:
    """Lines defined by glyph-sized components only: signatures, logos, stamps and
    binder rings do not define (or merge) lines."""
    H, W = ink.shape
    _, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    h, w = st[:, 3], st[:, 2]
    med_h = glyph_height(st, H)
    if not med_h:
        return [], 0.0
    textlike = (h <= 2.2 * med_h) & (w <= 0.6 * W)
    textlike[0] = False
    return _rows_to_lines(textlike[lab]), med_h


def squeeze_kashida(im: Image.Image) -> tuple[Image.Image, np.ndarray | None]:
    """Shorten the elongations justified Persian is stretched with.

    A kashida is a flat horizontal connector — a run of columns with the same top and
    bottom edge, joined to a glyph at both ends. A dash is just as flat but stands
    alone, so it is left alone. Stretched to 45 px a connector grows teeth under
    Tesseract: «بلــوار» came back as «بلسوار» on 21 of the 51 dev pages, «مشهـــد» as
    «مشهصد». Cut back to an ordinary connector's width, the word reads (D74).

    Returns the narrowed image and the map from its columns back to the original x —
    word boxes must still be reported in page coordinates.
    """
    ink = np.asarray(im.convert("L")) < 128
    H, W = ink.shape
    cols = ink.any(0)
    if cols.sum() < 8:
        return im, None
    top = np.argmax(ink, 0)
    bot = H - 1 - np.argmax(ink[::-1], 0)
    h = np.where(cols, bot - top + 1, 0)
    runs = (np.diff(ink.astype(np.int8), axis=0) == 1).sum(0) + ink[0].astype(int)
    xh = float(np.percentile(h[cols], 90))
    body = cols & (h >= 0.5 * xh)
    if not body.any():
        return im, None
    base = float(np.median(bot[body]))                   # where the letters stand
    # a full stroke, thinner than a letter, level along its whole length
    flat = cols & (runs == 1) & (h >= 0.22 * xh) & (h <= 0.5 * xh)
    flat[:-1] &= (np.abs(np.diff(top)) <= 1) & (np.abs(np.diff(bot)) <= 1)
    idx = np.flatnonzero(flat)
    if not len(idx):
        return im, None
    keep, least = max(2, int(0.30 * xh)), max(5, int(0.38 * xh))
    drop = np.zeros(W, bool)
    for r in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
        a, b = int(r[0]), int(r[-1]) + 1
        if b - a < least or b - a <= keep or a == 0 or b >= W - 1:
            continue
        # a kashida runs along the baseline and joins a letter at each end; a dash floats
        # above it, and an underline or a rule remnant sits below — leave those alone
        if cols[a - 1] and cols[b + 1] and abs(bot[a] - base) <= 0.15 * xh:
            drop[a + keep:b] = True
    if not drop.any():
        return im, None
    return Image.fromarray(np.where(ink[:, ~drop], 0, 255).astype(np.uint8)), np.flatnonzero(~drop)


def _crop(im: Image.Image, box, pad: int = 2) -> Image.Image:
    x0, y0, x1, y1 = box
    return im.crop((max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad)))


def _yov(a, b) -> float:
    return max(0, min(a[3], b[3]) - max(a[1], b[1])) / max(1, min(a[3] - a[1], b[3] - b[1]))


def _xov(a, b) -> float:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) / max(1, min(a[2] - a[0], b[2] - b[0]))


# A screenshot or a rendered page is mostly runs of identical pixels; a scan or a photo never
# is, because sensor noise and JPEG make neighbours differ. Over the corpus every scan and
# photo is at most 0.80 and every digital render at least 0.92 (E39).
SCREEN_FLAT_FRAC = 0.90
# How far a pixel's colour must be from its neighbourhood's to count as ink on a screen image.
SCREEN_INK_CONTRAST = 64


def is_screen_image(im: Image.Image) -> bool:
    """Is this a digital image (screenshot, rendered page) rather than a scan or photo?"""
    small = im.convert("RGB")
    small.thumbnail((1600, 1600), Image.NEAREST)
    a = np.asarray(small, dtype=np.int16)
    if a.shape[1] < 2:
        return False
    return float((np.abs(np.diff(a, axis=1)).sum(2) == 0).mean()) >= SCREEN_FLAT_FRAC


def screen_ink(im: Image.Image) -> Image.Image:
    """Black text on white from a screen image, whatever its colours.

    A pixel is ink when its colour stands out from the median colour around it, in either
    direction: dark labels on white, white text on a blue or green button and purple text on
    white all come out the same. The window is wide enough that text strokes never make up
    half of it, so the median is the background.
    """
    a = np.asarray(im.convert("RGB"))
    k = max(21, min(a.shape[:2]) // 20) | 1
    bg = cv2.medianBlur(a, k)
    d = np.abs(a.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    return Image.fromarray(np.where(d >= SCREEN_INK_CONTRAST, 0, 255).astype(np.uint8)).convert("RGB")
