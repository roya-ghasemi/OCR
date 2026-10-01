# -*- coding: utf-8 -*-
"""Page cutting and ink separation (E26).

Each test pins a property the measurement paid for, not an implementation detail: the
stage must take the page out of a photographed frame, and must never be the thing that
loses a line of the letter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ocr_service.ink import blue_mask, drop_blue_ink, invert_bands  # noqa: E402
from ocr_service.page_prep import FULL_FRAME_FRAC, crop_page, find_quad  # noqa: E402

PAGES = ROOT / "ocr_eval" / "pages"


def _page_photo(angle: float = 0.0, desk: int = 150) -> Image.Image:
    """A white sheet with text-like rows, lying on a textured desk."""
    rng = np.random.default_rng(0)
    frame = np.clip(rng.normal(desk, 12, (900, 700, 3)), 0, 255).astype(np.uint8)
    sheet = np.full((700, 520, 3), 248, np.uint8)
    for y in range(60, 420, 40):
        cv2.rectangle(sheet, (60, y), (460, y + 16), (20, 20, 20), -1)
    if angle:
        M = cv2.getRotationMatrix2D((260, 350), angle, 1.0)
        sheet = cv2.warpAffine(sheet, M, (520, 700), borderValue=(int(desk),) * 3)
    frame[100:800, 90:610] = sheet
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def test_page_on_a_desk_is_cut_out_of_the_frame():
    im = _page_photo()
    out, prep = crop_page(im)
    assert prep.cropped
    assert out.size[0] * out.size[1] < 0.85 * im.size[0] * im.size[1]


def test_a_page_filling_image_is_returned_untouched():
    """A scan is already the page. Cropping it was measured to make three corpus
    documents worse and help none, so it must not happen."""
    a = np.full((900, 700, 3), 250, np.uint8)
    for y in range(60, 840, 40):
        cv2.rectangle(a, (60, y), (640, y + 16), (20, 20, 20), -1)
    im = Image.fromarray(a)
    out, prep = crop_page(im)
    assert not prep.cropped and out.size == im.size


def test_crop_is_refused_when_it_would_discard_text():
    """Text outside the detected page means the detection is wrong, not that the text is
    rubbish — a letterhead or a footer. The frame is kept whole instead."""
    rng = np.random.default_rng(1)
    frame = np.clip(rng.normal(150, 12, (900, 700, 3)), 0, 255).astype(np.uint8)
    frame[100:700, 90:610] = 248
    for y in range(140, 660, 40):                       # text on the "page"
        cv2.rectangle(frame, (120, y), (560, y + 16), (20, 20, 20), -1)
    for y in range(750, 850, 30):                       # and more below it, outside
        cv2.rectangle(frame, (120, y), (560, y + 14), (20, 20, 20), -1)
    im = Image.fromarray(frame)
    out, prep = crop_page(im)
    assert not prep.cropped and out.size == im.size


def test_to_original_maps_a_cropped_point_back():
    im = _page_photo()
    _, prep = crop_page(im)
    x, y = prep.to_original(0, 0)
    assert 0 <= x <= im.size[0] and 0 <= y <= im.size[1]
    assert (x, y) != (0, 0)                             # the crop really moved the origin


# --- ink ---------------------------------------------------------------------------

def test_blue_mask_finds_stamp_ink_and_not_black_text():
    a = np.full((60, 200, 3), 250, np.uint8)
    a[10:20, 10:60] = (20, 20, 20)                      # black text (BGR)
    a[30:40, 10:60] = (190, 90, 60)                     # blue stamp
    m = blue_mask(a)
    assert m[30:40, 10:60].all()
    assert not m[10:20, 10:60].any()


def test_stamp_removal_keeps_the_letters_underneath():
    """Where a stamp crosses a printed letter the pixel is blue AND dark. Erasing it
    would remove the text the stamp was covering, which is the opposite of the point."""
    a = np.full((800, 800, 3), 250, np.uint8)
    cv2.circle(a, (400, 400), 120, (200, 120, 90), -1)  # stamp over paper
    a[390:410, 340:460] = (60, 30, 20)                  # a letter under the stamp
    out, n = drop_blue_ink(a)
    assert n > 0
    assert out[390:410, 340:460].max() < 120            # the letter is still dark
    assert out[300:330, 380:420].min() > 200            # the stamp alone is paper again


def test_blue_removal_changes_nothing_on_a_page_with_no_colour():
    a = np.full((60, 200, 3), 250, np.uint8)
    a[10:20, 10:60] = (20, 20, 20)
    out, n = drop_blue_ink(a)
    assert n == 0 and np.array_equal(out, a)


def test_band_inversion_leaves_a_plain_dark_block_alone():
    """A shadow or a photograph is dark without being a letterhead; inverting one turns a
    page of nothing into a page of noise."""
    a = np.full((400, 400, 3), 250, np.uint8)
    a[20:160, 20:380] = 40                              # solid dark block, no light strokes
    out, boxes = invert_bands(a)
    assert boxes == []
    assert np.array_equal(out, a)


@pytest.mark.skipif(not (PAGES / "tech_prose.jpg").is_file(), reason="page corpus absent")
def test_the_brief_photo_loses_its_desk():
    """The page this stage exists for: whole-page CER 20.1% → 4.4% and four fabricated
    lines gone, purely because the desk stopped reaching the recogniser."""
    im = Image.open(PAGES / "tech_prose.jpg")
    out, prep = crop_page(im)
    assert prep.cropped
    assert out.size[0] * out.size[1] < 0.85 * im.size[0] * im.size[1]


@pytest.mark.skipif(not (PAGES / "spring_prose.jpg").is_file(), reason="page corpus absent")
def test_a_crop_keeps_substantially_all_of_the_page():
    """A photo whose sheet nearly fills the frame may still be trimmed, but only at the
    edge: a crop that removed a tenth of this page would be removing text."""
    im = Image.open(PAGES / "spring_prose.jpg")
    out, _ = crop_page(im)
    assert out.size[0] * out.size[1] > 0.9 * im.size[0] * im.size[1]


def test_full_frame_gate_is_the_measured_one():
    """Pinned because it is a tuned constant: at 0.93 and 0.97 the crop reached corpus
    scans and made every document it touched worse."""
    assert FULL_FRAME_FRAC == 0.90
