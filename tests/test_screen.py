# -*- coding: utf-8 -*-
"""Screen images: telling them from scans, and reading coloured text on them (E39)."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ocr_service.imaging import is_screen_image, screen_ink  # noqa: E402


def _ui() -> Image.Image:
    im = Image.new("RGB", (400, 200), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle((20, 20, 200, 70), fill=(15, 139, 230))           # blue button
    d.text((60, 35), "SAVE", fill=(255, 255, 255))                 # white text on it
    d.text((40, 120), "label", fill=(51, 51, 51))                  # dark text on white
    d.text((240, 120), "bp7S", fill=(123, 111, 224))               # purple text on white
    return im


def test_a_rendered_interface_is_a_screen_image():
    assert is_screen_image(_ui())


def test_a_noisy_capture_is_not():
    rng = np.random.default_rng(0)
    a = np.clip(np.asarray(_ui(), np.int16) + rng.normal(0, 4, (200, 400, 3)), 0, 255).astype(np.uint8)
    assert not is_screen_image(Image.fromarray(a))


def test_text_of_every_colour_comes_out_dark_on_white():
    out = np.asarray(screen_ink(_ui()).convert("L"))
    assert (out[33:50, 58:90] == 0).any()          # white text on the blue button
    assert (out[118:135, 38:75] == 0).any()        # dark label
    assert (out[118:135, 238:270] == 0).any()      # purple text
    assert (out[25:32, 100:190] == 255).all()      # the button's flat fill is background
