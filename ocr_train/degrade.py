# -*- coding: utf-8 -*-
r"""Make a cleanly rendered line look like a page someone actually photographed.

text2image produces a perfect raster: pure black on pure white, every stroke crisp, no
compression, no lighting, no paper. Nothing the service is given looks like that. Its inputs
are CamScanner exports and phone photos — JPEG artefacts, a shadow across the sheet, the
grain of the paper, a degree of tilt, and text that may be 12 px tall.

That gap showed up as a measurement. The first fine-tune, trained only on clean renders plus
the benchmark, improved the benchmark's own domain by 5.3 pp CER and **regressed** the
scanned administrative letter. A model that has only seen clean type learns features that a
scan does not have.

Every transform here is something that happens to a real document, and each is applied
independently so a line gets a plausible combination rather than all of them at once. The
label never changes — these are the same words, photographed badly.

Deliberately NOT included: anything that moves ink around. No elastic warps, no dilation or
erosion of strokes. In Persian a thickened stroke can close the gap that distinguishes two
letters, and a model taught on that learns a confusion rather than a robustness.
"""
from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageFilter

# The service upsamples anything under 20 px to about 30 px (E21), so the model should see
# the whole range it will be handed — including the bad end, which is where the dots of
# ب/ی/پ/ن stop being resolvable at all.
MIN_TEXT_PX, MAX_TEXT_PX = 14, 58


def _to_gray(im: Image.Image) -> Image.Image:
    return im if im.mode == "L" else im.convert("L")


def rescale(im: Image.Image, rng: random.Random) -> Image.Image:
    """Resample to a random text height, through the same bottleneck a real capture has."""
    target = rng.randint(MIN_TEXT_PX, MAX_TEXT_PX)
    if im.height < 4:
        return im
    f = target / im.height
    if 0.95 < f < 1.05:
        return im
    w, h = max(8, int(im.width * f)), max(4, int(im.height * f))
    # Down then back is not the same as never having gone down: detail that falls below the
    # sampling grid does not come back, which is exactly what a low-DPI scan does.
    return im.resize((w, h), Image.LANCZOS)


def blur(im: Image.Image, rng: random.Random) -> Image.Image:
    return im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1)))


def jpeg(im: Image.Image, rng: random.Random) -> Image.Image:
    buf = io.BytesIO()
    _to_gray(im).save(buf, "JPEG", quality=rng.randint(28, 75))
    buf.seek(0)
    return Image.open(buf).convert("L")


def noise(im: Image.Image, rng: random.Random) -> Image.Image:
    a = np.asarray(_to_gray(im), dtype=np.float32)
    a += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(3, 14), a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def uneven_light(im: Image.Image, rng: random.Random) -> Image.Image:
    """A shadow or a fold across the line: a smooth gradient over the whole strip."""
    a = np.asarray(_to_gray(im), dtype=np.float32)
    h, w = a.shape
    gx = np.linspace(rng.uniform(-1, 1), rng.uniform(-1, 1), w)[None, :]
    gy = np.linspace(rng.uniform(-1, 1), rng.uniform(-1, 1), h)[:, None]
    return Image.fromarray(
        np.clip(a + (gx + gy) * rng.uniform(8, 38), 0, 255).astype(np.uint8))


def paper(im: Image.Image, rng: random.Random) -> Image.Image:
    """Tint the white towards paper and give it grain, instead of a pure 255 background."""
    a = np.asarray(_to_gray(im), dtype=np.float32)
    g = np.random.default_rng(rng.randrange(1 << 30))
    tint = rng.uniform(216, 250)
    grain = g.normal(0, rng.uniform(1.5, 6.0), a.shape)
    bright = a > 160                                  # only the background, not the ink
    a = np.where(bright, tint + grain, a * rng.uniform(0.75, 1.0))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def tilt(im: Image.Image, rng: random.Random) -> Image.Image:
    """A fraction of a degree. The transcriber deskews at 1 degree and above, so what
    survives into a line crop is the small residual — which is what this teaches."""
    return im.rotate(rng.uniform(-0.9, 0.9), resample=Image.BICUBIC,
                     expand=True, fillcolor=245)


def faint(im: Image.Image, rng: random.Random) -> Image.Image:
    """Worn toner or a light photocopy: ink that is grey rather than black."""
    a = np.asarray(_to_gray(im), dtype=np.float32)
    k = rng.uniform(0.35, 0.75)
    return Image.fromarray(np.clip(255 - (255 - a) * k, 0, 255).astype(np.uint8))


# (transform, probability). Order matters: geometry before sensor, sensor before codec, so
# the JPEG artefacts land on the finished picture the way they do in a real file.
PIPELINE = [
    (tilt, 0.30),
    (rescale, 0.80),
    (faint, 0.20),
    (paper, 0.55),
    (uneven_light, 0.35),
    (blur, 0.45),
    (noise, 0.40),
    (jpeg, 0.55),
]


def apply(im: Image.Image, rng: random.Random) -> Image.Image:
    """One plausible capture of this line."""
    out = _to_gray(im)
    for fn, p in PIPELINE:
        if rng.random() < p:
            out = fn(out, rng)
    return out
