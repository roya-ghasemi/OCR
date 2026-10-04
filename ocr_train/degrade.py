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

The second version of this file did not fix it either, and the reason is worth keeping. Its
transforms were a single soft-scan pipeline, which put the corpus here:

    corpus                     line height   >240    <40   60-200
    E29 degraded synthetic          53 px   0.539  0.115   0.086
    the brief's letter              73 px   0.817  0.068   0.056
    tech_prose (phone photo)        26 px   0.000  0.000   0.639
    spring_prose (phone photo)      31 px   0.000  0.001   0.842

Those are per text line, in the page the recogniser is actually handed. Three domains, not
one — and the corpus sat between them, matching none. The phone photos have **no** saturated
paper and **no** solid ink at all; they are entirely midtone, because a camera with automatic
exposure never reaches either end of the range. The letter is the opposite: paper blown to
white, thin ink, almost nothing in between. And the corpus could not even reach the letter's
line height, because the resample capped at 58 px.

So a regime here is a *device*: it fixes the tone curve and the resolution together, because
in reality those are not independent. `bright_scan` reproduces the first row, `photo_tone`
the last two.

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
# ب/ی/پ/ن stop being resolvable at all, and the top end, which is where a 300 DPI scan sits.
# The brief's letter reads at 73 px per line; a cap of 58 could not even reach it.
MIN_TEXT_PX, MAX_TEXT_PX = 14, 86

# The lightest the darkest pixel of a line may be. Above this the text is not in the picture
# any more, whatever the label says.
READABLE_INK = 160

# Where a scanner's auto-levels is allowed to put the ink, as a fraction of the output
# range. Above this the stretch has turned text into paper.
INK_CEILING = 0.35


def _to_gray(im: Image.Image) -> Image.Image:
    return im if im.mode == "L" else im.convert("L")


def rescale(im: Image.Image, rng: random.Random,
            bounds: tuple[int, int] | None = None) -> Image.Image:
    """Resample to a random text height, through the same bottleneck a real capture has."""
    lo, hi = bounds or (MIN_TEXT_PX, MAX_TEXT_PX)
    target = rng.randint(lo, hi)
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


def bright_scan(im: Image.Image, rng: random.Random) -> Image.Image:
    """The look of a phone scanner app: paper blown to white, ink thinned to grey.

    Measured on the brief's letter, per text line, in the page the recogniser is actually
    handed: **82% of pixels above 240, only 6.8% below 40, 5.6% in the midtones.** A
    CamScanner export is not a soft noisy scan. It has had its white point pulled down hard
    until the paper saturates, and what that does to the text is make it *thinner and
    lighter* — the ink is grey with a bright halo, not the solid black of a clean render.

    The white point is pulled hard and the ink is left where it falls. An earlier version
    lifted the ink with a floor as well, on the theory that a scan's text is grey; measured
    against the letter that made it too grey — 3.2% of pixels below 40 against the letter's
    6.8% — and the thin ink came out of the *stretch*, not out of a floor.
    """
    src = _to_gray(im)

    # First, flatten the illumination. This is the thing a scanner app does that a plain
    # contrast curve cannot: it estimates the background across the page and divides it out,
    # so a shadow or a fold stops existing rather than being stretched along with the text.
    # Without it the uneven lighting survives into the result and holds a fifth of the
    # background below saturation — 0.63 of pixels above 240 where the letter has 0.82.
    a = np.asarray(src, dtype=np.float32)
    field = np.asarray(src.filter(ImageFilter.GaussianBlur(max(6.0, src.height * 0.45))),
                       dtype=np.float32)
    a = np.clip(a * (float(np.percentile(field, 90)) / np.maximum(field, 1.0)), 0, 255)

    black_in = rng.uniform(0, 45)
    white_in = rng.uniform(150, 205)                 # everything above this saturates

    # The one case the fixed white point gets wrong: a line that `faint` has already
    # lightened past it. The stretch then clips text and paper to the same 255 and the
    # difference is simply gone — no later guard can recover it, and the result is a picture
    # of blank paper carrying a label that describes a sentence. Only then does the white
    # point move, and only as far as it must, so an ordinary render is untouched and keeps
    # the histogram this was measured to produce.
    ink = float(np.percentile(a, 2))
    if ink > white_in * INK_CEILING:
        white_in = max(white_in, black_in + (ink - black_in) / INK_CEILING)

    a = np.clip((a - black_in) * (255.0 / max(1.0, white_in - black_in)), 0, 255)
    out = Image.fromarray(a.astype(np.uint8))

    # And it sharpens. Without this the regime stops at 20% midtones against the letter's
    # 5.6%: a resampled glyph has a wide anti-aliased perimeter, and a scanner app's
    # sharpening is what collapses that perimeter into either ink or paper. It is the step
    # that makes the result look bilevel without ever thresholding — which matters, because
    # a real threshold would break strokes and teach the model letters that do not exist.
    return out.filter(ImageFilter.UnsharpMask(
        radius=rng.uniform(1.0, 2.6), percent=rng.randint(90, 230), threshold=rng.randint(0, 3)))


def ensure_ink(im: Image.Image, rng: random.Random) -> Image.Image:
    """Put the contrast back if a stack of transforms has lightened the text away.

    The failure this prevents is the worst kind of training pair: a picture of blank paper
    carrying a label that describes a sentence. Nothing downstream can detect it — the
    `.lstmf` builds, the line is the right shape, the label is valid — so it simply teaches
    the model that those glyphs look like nothing.

    It is a backstop, not the mechanism. `bright_scan` adapts its white point to the line's
    own ink so it does not create this in the first place; `faint` on an already-light
    render still can, in any regime.
    """
    a = np.asarray(_to_gray(im), dtype=np.float32)
    darkest = float(a.min())
    if darkest <= READABLE_INK:
        return im
    a = 255 - (255 - a) * ((255 - rng.uniform(70, READABLE_INK)) / max(1.0, 255 - darkest))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def photo_tone(im: Image.Image, rng: random.Random) -> Image.Image:
    """A page photographed in room light: no white, no black, everything in the middle.

    Also measured, on the two photo pages: **zero pixels above 240, zero below 40, and 64%
    to 84% of them between 60 and 200.** A phone camera with automatic exposure never
    reaches either end of the range, and a model trained only on material that does has
    never seen the contrast it will be given.
    """
    a = np.asarray(_to_gray(im), dtype=np.float32)
    lo = rng.uniform(55, 115)
    hi = rng.uniform(150, 215)
    if hi - lo < 40:
        hi = lo + 40
    return Image.fromarray(np.clip(lo + a * (hi - lo) / 255, 0, 255).astype(np.uint8))


# Capture transforms, applied between the resample and the tone regime. Order matters:
# geometry before sensor, sensor before codec, so the JPEG artefacts land on the finished
# picture the way they do in a real file.
PIPELINE = [
    (tilt, 0.30),
    (faint, 0.20),
    (paper, 0.35),
    (uneven_light, 0.35),
    (blur, 0.45),
    (noise, 0.40),
]

# A capture regime is a *device*, so it fixes both the tone and the resolution — those are
# not independent, and treating them as independent is what produced a corpus that sat
# between the two real domains and matched neither.
#
#   name      share  line height   because
#   scan       0.50    55-86 px    the brief's letter reads at 73 px per line
#   photo      0.35    18-42 px    the two photo pages read at 26 and 31 px
#   plain      0.15    24-80 px    clean print exists too, at any size
#
# The scan leads because administrative letters are what the product is for, and because it
# was the one domain the E29 corpus had nothing resembling — which is what made every
# fine-tune in that experiment read the letter worse the longer it trained.
REGIMES = [("scan", 0.50, (55, MAX_TEXT_PX)),
           ("photo", 0.35, (18, 42)),
           ("plain", 0.15, (24, 80))]


def _regime(rng: random.Random) -> tuple[str, tuple[int, int]]:
    r = rng.random()
    acc = 0.0
    for name, share, bounds in REGIMES:
        acc += share
        if r < acc:
            return name, bounds
    return REGIMES[-1][0], REGIMES[-1][2]


def apply(im: Image.Image, rng: random.Random) -> Image.Image:
    """One plausible capture of this line, by one plausible device.

    The regime is chosen first because it decides the resolution as well as the tone. Then
    the capture transforms say what happened to the paper and the lens — a scanner app and a
    phone camera can both be handed a tilted, blurred, unevenly lit page — and the tone curve
    last says what kind of file came out. The histogram is what the network sees first, and
    E29 proved it is what transfers or fails to.
    """
    name, bounds = _regime(rng)
    out = rescale(_to_gray(im), rng, bounds)
    for fn, p in PIPELINE:
        if rng.random() < p:
            out = fn(out, rng)

    if name == "scan":
        out = bright_scan(out, rng)
    elif name == "photo":
        out = photo_tone(out, rng)

    if rng.random() < 0.55:
        out = jpeg(out, rng)        # last, so the artefacts land on the finished picture
    return ensure_ink(out, rng)
