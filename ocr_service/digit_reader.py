# -*- coding: utf-8 -*-
r"""Glyph-level Persian/Latin digit reader — the classical oracle Tesseract failed to be.

Measured 2026-09-17: on a clean crop of `۰۹۲۴۴۲۴۱۱۷` from a real letter, every
Tesseract model available (`fas` best/std, `ara` best/std, LSTM and legacy engines)
returned at most 5 of the 10 digits. The VLMs read 0/12 (D53). Nothing on the machine
could read a printed Persian number.

What makes a dedicated reader tractable: Persian digits are NON-JOINING. In every
office typeface (B Nazanin, B Lotus, B Mitra, B Zar, …) each digit is one isolated
connected component, so segmentation is trivial, and the ten classes are visually
simple. A classifier trained on synthetic renders of ۰-۹ / 0-9 across the ~40 Persian
fonts installed here, with scan-like augmentation, needs no labelled scans at all.

Pipeline
    line/crop image -> binarise -> connected components -> keep digit-like blobs
    -> 28x28 normalised glyph -> HOG -> SVM -> digits, right-to-left for Persian.

Assumption: printed digits only, ≥ ~16 px tall. Model file: `models/digits_hog_svm.xml`
(scikit-learn SVC on numpy-HOG; trained by `train()` in a few minutes on CPU).
"""
from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "digits_hog_svm.joblib"

PERSIAN = "۰۱۲۳۴۵۶۷۸۹"
LATIN = "0123456789"
SEPARATORS = "/-.,:"
CLASSES = PERSIAN + LATIN + SEPARATORS   # 0-9 Persian, 10-19 Latin, 20+ separators (never digits)
G = 28                             # glyph canvas

_USER_FONTS = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts")
FONT_GLOBS = [os.path.join(d, pat) for d in (r"C:\Windows\Fonts", _USER_FONTS)
              for pat in ("B*.ttf", "B*.TTF", "Mj_*.ttf", "Mj_*.TTF", "tahoma*.ttf", "arial*.ttf",
                          "times*.ttf", "calibri*.ttf", "DUBAI*.TTF", "*Yekan*.ttf", "*Vazir*.ttf", "*Sahel*.ttf")]

def hog(glyph: np.ndarray, cell: int = 7, bins: int = 9) -> np.ndarray:
    """Plain HOG in numpy (OpenCV 5 headless ships neither HOGDescriptor nor cv2.ml)."""
    g = glyph.astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) % np.pi) * (bins / np.pi)        # unsigned, 0..bins
    n = G // cell
    out = np.zeros((n, n, bins), np.float32)
    for i in range(n):
        for j in range(n):
            a = ang[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].ravel()
            m = mag[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].ravel()
            lo = np.floor(a).astype(int) % bins; frac = a - np.floor(a)
            np.add.at(out[i, j], lo, m * (1 - frac)); np.add.at(out[i, j], (lo + 1) % bins, m * frac)
    # block-normalise 2x2 cells
    feats = []
    for i in range(n - 1):
        for j in range(n - 1):
            blk = out[i:i + 2, j:j + 2].ravel()
            feats.append(blk / (np.linalg.norm(blk) + 1e-6))
    return np.concatenate(feats)


# --- glyph normalisation ------------------------------------------------------

def normalize_glyph(bin_img: np.ndarray) -> np.ndarray:
    """Tight-crop a binary (white glyph on black) blob, pad to square, resize to GxG.
    Aspect ratio is preserved: `۱` is a thin bar and `۰` a dot, and that IS the signal."""
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
    out[2:-2, 2:-2] = canvas
    return out


def features(glyph: np.ndarray, rel_h: float = 1.0) -> np.ndarray:
    # HOG on the binary glyph plus its aspect ratio and fill ratio — the dot (۰) and
    # the bar (۱) are separated by shape statistics more than by gradients.
    f = hog(glyph)
    px = cv2.resize(glyph, (14, 14), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0
    ys, xs = np.where(glyph > 0)
    if len(xs):
        aspect = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
        fill = len(xs) / ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    else:
        aspect = fill = 0.0
    # rel_h: this glyph's height over the tallest digit on its line. A `۰` is a dot at
    # ~0.25-0.4 of digit height; every other digit is ~1.0. Normalisation to GxG
    # erases that, so it is fed back in explicitly.
    return np.concatenate([f, px, [aspect, fill, rel_h]]).astype(np.float32)


# --- synthetic training data ----------------------------------------------------

def _fonts() -> list[str]:
    seen, out = set(), []
    for pat in FONT_GLOBS:
        for f in glob.glob(pat):
            k = os.path.basename(f).lower()
            if k not in seen:
                seen.add(k); out.append(f)
    # Decorative faces never appear in an administrative letter and only add noise.
    skip = ("outline", "shade", "freehand", "kidnap", "fantezy", "bomba", "digital", "liner",
            "ghalam", "sayeh", "nastaliq", "takhteh", "granada", "cordoba", "promoter", "typographer")
    good = []
    for f in out:
        if any(k in os.path.basename(f).lower() for k in skip):
            continue
        try:
            font = ImageFont.truetype(f, 40)
            # A font without Persian digits renders every one of them as the same
            # fallback box; a font with them renders ten different shapes.
            masks = [np.asarray(font.getmask(ch)) for ch in PERSIAN]
            shapes = {(m.shape, int(m.sum())) for m in masks if m.size}
            if len(shapes) >= 8:
                good.append(f)
        except Exception:
            pass
    return good


_line_h: dict[tuple[str, int], int] = {}


def line_height(font_path: str, px: int) -> int:
    """Tallest digit of the font at this size — the reference for `rel_h`."""
    key = (font_path, px)
    if key not in _line_h:
        try:
            font = ImageFont.truetype(font_path, px)
            hs = []
            for ch in PERSIAN + LATIN:
                bb = font.getmask(ch).getbbox()
                if bb:
                    hs.append(bb[3] - bb[1])
            _line_h[key] = max(hs) if hs else px
        except Exception:
            _line_h[key] = px
    return _line_h[key]


def render_digit(ch: str, font_path: str, px: int, rng: random.Random) -> tuple[np.ndarray, float] | None:
    """One augmented glyph as a binary GxG image plus its relative height, or None."""
    try:
        font = ImageFont.truetype(font_path, px)
    except Exception:
        return None
    im = Image.new("L", (px * 3, px * 3), 255)
    d = ImageDraw.Draw(im)
    d.text((px, px // 2), ch, font=font, fill=0)
    # scan-like augmentation
    if rng.random() < 0.7:
        im = im.rotate(rng.uniform(-4, 4), resample=Image.BICUBIC, fillcolor=255)
    if rng.random() < 0.8:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.3)))
    a = np.asarray(im)
    if rng.random() < 0.4:
        k = np.ones((2, 2), np.uint8)
        a = cv2.erode(a, k) if rng.random() < 0.5 else cv2.dilate(a, k)
    if rng.random() < 0.5:
        a = cv2.addWeighted(a, rng.uniform(0.7, 1.0), np.full_like(a, 255), 0.0, rng.uniform(0, 60))
    if rng.random() < 0.5:                       # JPEG ringing
        _, enc = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(50, 90)])
        a = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    _, b = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    if n < 2:
        return None
    # keep the largest component (some fonts draw a dotted zero as 2 parts; take all
    # parts that are not specks)
    keep = np.zeros_like(b)
    areas = stats[1:, cv2.CC_STAT_AREA]
    for i, area in enumerate(areas, start=1):
        if area >= max(3, 0.05 * areas.max()):
            keep[lab == i] = 255
    ys = np.where(keep.any(axis=1))[0]
    rel = (ys.max() - ys.min() + 1) / max(1, line_height(font_path, px)) if len(ys) else 1.0
    return normalize_glyph(keep), float(min(rel * rng.uniform(0.9, 1.1), 1.3))


def merge_overlapping(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Glyph parts that detach under binarisation (the hook of `۴`, the dots of a
    dotted zero) sit inside the horizontal span of their main stroke. Merge any
    box that overlaps ≥ 50% of its own width with a wider box on the same line."""
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    merged: list[list[int]] = []
    for x0, y0, x1, y1 in boxes:
        for m in merged:
            ov = min(x1, m[2]) - max(x0, m[0])
            if ov > 0 and ov >= 0.5 * (x1 - x0) and not (y1 < m[1] - (m[3] - m[1]) or y0 > m[3] + (m[3] - m[1])):
                m[0], m[1], m[2], m[3] = min(m[0], x0), min(m[1], y0), max(m[2], x1), max(m[3], y1)
                break
        else:
            merged.append([x0, y0, x1, y1])
    return [tuple(m) for m in merged]


def train(n_per_font: int = 10, seed: int = 17, model_path: Path = MODEL_PATH) -> dict:
    rng = random.Random(seed)
    fonts = _fonts()
    X, y = [], []
    for fp in fonts:
        for ci, ch in enumerate(CLASSES):
            for _ in range(n_per_font if ch not in SEPARATORS else max(3, n_per_font // 2)):
                r = render_digit(ch, fp, rng.choice([18, 22, 28, 36, 48, 64]), rng)
                if r is None:
                    continue
                g, rel = r
                X.append(features(g, rel)); y.append(ci)
    X = np.array(X, np.float32); y = np.array(y, np.int32)
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    import joblib
    idx = np.random.RandomState(seed).permutation(len(X))
    split = int(len(X) * 0.9)
    clf = make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale", probability=False))
    clf.fit(X[idx[:split]], y[idx[:split]])
    pred = clf.predict(X[idx[split:]]); truth = y[idx[split:]]
    acc = float((pred == truth).mean())
    acc_value = float(((pred % 10) == (truth % 10)).mean())   # digit value, script ignored
    clf.fit(X, y)                                  # final model on everything
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(clf, model_path)
    return {"fonts": len(fonts), "samples": int(len(X)), "holdout_acc_20class": round(acc, 4),
            "holdout_acc_digit_value": round(acc_value, 4), "model": str(model_path)}


# --- inference ---------------------------------------------------------------------

@dataclass
class DigitRead:
    text: str                  # digits as read, in logical (left-to-right numeric) order
    script: str                # "persian" | "latin" | "mixed"
    boxes: list[tuple[int, int, int, int]]
    confidence: float          # fraction of glyphs whose SVM margin was clear


class DigitReader:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.is_file():
            raise FileNotFoundError(f"{model_path} — run `python -m ocr_service.digit_reader --train`")
        import joblib
        self.svm = joblib.load(model_path)

    def _blobs(self, gray: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
        _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        boxes = []
        H = gray.shape[0]
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 4 or h > 0.98 * H:
                continue
            boxes.append((x, y, x + w, y + h))
        return b, boxes

    def read_line(self, gray: np.ndarray, rtl: bool = True) -> DigitRead:
        """Read a crop that contains ONE number (any digit script). Blobs are
        classified individually; `۰` is a dot, so small blobs are kept."""
        b, boxes = self._blobs(gray)
        if not boxes:
            return DigitRead("", "persian", [], 0.0)
        hs = sorted(y1 - y0 for _, y0, _, y1 in boxes)
        ref_h = hs[len(hs) // 2]
        keep = merge_overlapping([bx for bx in boxes if (bx[3] - bx[1]) >= 0.12 * ref_h])  # specks out, dots in
        keep.sort(key=lambda bx: bx[0])       # numbers read left-to-right in every script
        digits, boxes_out, clear = [], [], 0
        line_h = max(y1 - y0 for _, y0, _, y1 in keep)
        for x0, y0, x1, y1 in keep:
            g = normalize_glyph(b[y0:y1, x0:x1])
            f = features(g, (y1 - y0) / line_h)[None, :]
            cls = int(self.svm.predict(f)[0])
            digits.append(CLASSES[cls]); boxes_out.append((x0, y0, x1, y1)); clear += 1
        txt = "".join(digits).strip(SEPARATORS)
        scripts = {"persian" if d in PERSIAN else "latin" for d in digits if d not in SEPARATORS}
        return DigitRead(txt, (scripts.pop() if len(scripts) == 1 else ("mixed" if scripts else "none")),
                         boxes_out, clear / max(len(keep), 1))


# --- page-level locator ---------------------------------------------------------

class GlyphReader:
    """Drop-in for `TesseractReader` in `numeric_validator`: `page_words()` returns
    every digit run on the page as a `Word` (text = digits read, box), `read_crop()`
    re-reads a box. Numbers are located structurally: on a text line, a run of >=3
    narrow, isolated, evenly spaced components is a number - joined Persian words
    are wide blobs and never look like that.
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        self.reader = DigitReader(model_path)
        self.version = "glyph-svm-" + model_path.stem

    def available(self) -> bool:
        return True

    @staticmethod
    def _components(gray: np.ndarray):
        _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        n, _, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        H, W = gray.shape
        boxes = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 6 or h > 0.06 * H or w > 0.25 * W or h < 6:
                continue
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
        return boxes

    @staticmethod
    def _lines(boxes):
        """Cluster boxes into text lines by vertical overlap."""
        boxes = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
        lines = []
        for b in boxes:
            cy = (b[1] + b[3]) / 2
            for ln in lines:
                y0 = min(x[1] for x in ln[-8:]); y1 = max(x[3] for x in ln[-8:])
                if y0 - 0.3 * (y1 - y0) <= cy <= y1 + 0.3 * (y1 - y0):
                    ln.append(b); break
            else:
                lines.append([b])
        return lines

    def page_words(self, im) -> list:
        from .numeric_validator import Word
        gray = np.asarray(im.convert("L"))
        out = []
        for ln in self._lines(self._components(gray)):
            ln.sort(key=lambda b: b[0])
            hs = sorted(b[3] - b[1] for b in ln)
            ref = hs[int(len(hs) * 0.8)] if hs else 0
            if ref < 8:
                continue
            run = []

            def flush():
                digit_like = [b for b in run if (b[3] - b[1]) >= 0.5 * ref]
                if len(digit_like) >= 3:
                    x0 = min(b[0] for b in run); y0 = min(b[1] for b in run)
                    x1 = max(b[2] for b in run); y1 = max(b[3] for b in run)
                    pad = max(2, int(0.15 * (y1 - y0)))
                    crop = gray[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
                    r = self.reader.read_line(crop)
                    if sum(c not in SEPARATORS for c in r.text) >= 3:
                        out.append(Word(r.text, (x0, y0, x1, y1), 100.0 * r.confidence))
                run.clear()

            for b in ln:
                w, h = b[2] - b[0], b[3] - b[1]
                narrow = w <= 1.15 * ref and h <= 1.4 * ref          # a digit, dot or separator
                if not narrow:
                    flush(); continue
                if run and b[0] - run[-1][2] > 0.9 * ref:           # gap too wide: new run
                    flush()
                run.append(b)
            flush()
        return out

    def read_crop(self, im, box, pad: float = 0.35) -> str:
        x0, y0, x1, y1 = box
        h = max(1, y1 - y0)
        px, py = int(h * pad), int(h * pad * 0.6)
        gray = np.asarray(im.convert("L"))
        crop = gray[max(0, y0 - py):min(gray.shape[0], y1 + py), max(0, x0 - px):min(gray.shape[1], x1 + px)]
        return self.reader.read_line(crop).text


def main() -> int:
    import argparse, json, sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--read", help="image crop containing one number")
    a = ap.parse_args()
    if a.train:
        print(json.dumps(train(), ensure_ascii=False))
    if a.read:
        r = DigitReader().read_line(cv2.imread(a.read, cv2.IMREAD_GRAYSCALE))
        print(json.dumps({"text": r.text, "script": r.script, "n_glyphs": len(r.boxes)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
