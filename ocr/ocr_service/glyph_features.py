# -*- coding: utf-8 -*-
"""Glyph features of the CNN digit reader, shared by inference (digit_reader_v2.py) and
training (ocr_research/digit_cnn_data.py) so the two cannot drift apart."""
from __future__ import annotations

import cv2
import numpy as np

G = 32                                     # glyph canvas
PERSIAN = "۰۱۲۳۴۵۶۷۸۹"
LATIN = "0123456789"
# 0-9 Persian, 10-19 Latin, 20-23 separators, 24 reject
CLASSES = list(PERSIAN) + list(LATIN) + ["/", "-", ",", ".", "<rej>"]
REJECT = 24


def binarize(gray: np.ndarray) -> np.ndarray:
    """Local threshold — the pages carry shading and a coloured letterhead; a global
    Otsu loses light strokes. Same function at training and inference."""
    blur = cv2.GaussianBlur(gray, (0, 0), 0.6)
    win = 51 if min(gray.shape) > 60 else 15
    b = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, win, 15)
    return b


def normalize_glyph(bin_img: np.ndarray) -> np.ndarray:
    ys, xs = np.where(bin_img > 0)
    if len(xs) == 0:
        return np.zeros((G, G), np.uint8)
    g = bin_img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = g.shape
    side = max(h, w)
    canvas = np.zeros((side, side), np.uint8)
    canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = g
    canvas = cv2.resize(canvas, (G - 4, G - 4), interpolation=cv2.INTER_AREA)
    out = np.zeros((G, G), np.uint8)
    out[2:-2, 2:-2] = (canvas > 96).astype(np.uint8) * 255
    return out


def line_context(boxes: list[tuple[int, int, int, int]]) -> tuple[float, float]:
    """(ref_h, line_cy) from the tall components of a line; shared with inference."""
    hs = np.array([b[3] - b[1] for b in boxes], np.float32)
    if len(hs) == 0:
        return 1.0, 0.0
    # percentile-based so one oversized blob (a shadow, a logo fragment) on the line
    # cannot drag the reference height away from the glyphs
    p80 = float(np.percentile(hs, 80))
    tall = (hs >= 0.5 * p80) & (hs <= 2.0 * p80)
    if not tall.any():
        tall = hs >= 0.5 * hs.max()
    ref = float(np.median(hs[tall]))
    cys = np.array([(b[1] + b[3]) / 2 for b in boxes], np.float32)[tall]
    return max(ref, 1.0), float(np.median(cys))


def scalars(box, ref_h: float, line_cy: float) -> np.ndarray:
    x0, y0, x1, y1 = box
    return np.array([(y1 - y0) / ref_h, ((y0 + y1) / 2 - line_cy) / ref_h, (x1 - x0) / ref_h], np.float32)
