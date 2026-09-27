# -*- coding: utf-8 -*-
r"""Synthetic glyph dataset for the v2 digit reader (CNN with a reject class).

Why a new dataset (D55/D56, E14): the v1 HOG-SVM reader has no "not a digit" class,
so every narrow Persian letter fragment the geometric locator admits is forced into a
digit label — measured precision 10% on the dev pages. Here every training sample
comes from a rendered TEXT LINE that mixes Persian words, numbers, Latin words and
punctuation, so the classifier sees the letter pieces it must reject next to the
digits it must read, with the same per-line context features the page reader uses.

Words come from the Dehkhoda headword table (312k entries, no overlap with the
evaluation corpus by construction); Arabic joining is done with a small shaper that
emits Presentation Forms-B code points, because Pillow on this machine has no raqm.

Output: `models/digit_glyphs.npz` with
    X   (N, 32, 32) uint8 binary glyphs, aspect preserved
    S   (N, 3)      float32 [rel_h, rel_cy, rel_w] relative to the line
    y   (N,)        int16 class index into CLASSES
    font(N,)        int16 font index (for a by-font holdout)

    venv312\Scripts\python.exe -m ocr_service.digit_cnn_data --lines-per-font 80
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .digit_reader import _fonts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "digit_glyphs.npz"
DEHKHODA = ROOT / "dehkhoda" / "dehkhoda.db"

G = 32                                     # glyph canvas
PERSIAN = "۰۱۲۳۴۵۶۷۸۹"
ARABIC = "٠١٢٣٤٥٦٧٨٩"
LATIN = "0123456789"
# class layout: 0-9 Persian/Arabic-Indic digit value, 10-19 Latin digit value,
# 20 slash, 21 dash, 22 comma, 23 dot (also each dot of ':'), 24 reject
CLASSES = list(PERSIAN) + list(LATIN) + ["/", "-", ",", ".", "<rej>"]
REJECT = 24
SEP_CLASS = {"/": 20, "-": 21, ",": 22, "،": 22, "٫": 22, ".": 23, ":": 23}

# --- Arabic shaping via Presentation Forms-B ------------------------------------
# letter -> (isolated, final, initial, medial); None = the form does not exist
_F = {
    "ا": (0xFE8D, 0xFE8E, None, None), "آ": (0xFE81, 0xFE82, None, None),
    "أ": (0xFE83, 0xFE84, None, None), "إ": (0xFE87, 0xFE88, None, None),
    "ب": (0xFE8F, 0xFE90, 0xFE91, 0xFE92), "پ": (0xFB56, 0xFB57, 0xFB58, 0xFB59),
    "ت": (0xFE95, 0xFE96, 0xFE97, 0xFE98), "ث": (0xFE99, 0xFE9A, 0xFE9B, 0xFE9C),
    "ج": (0xFE9D, 0xFE9E, 0xFE9F, 0xFEA0), "چ": (0xFB7A, 0xFB7B, 0xFB7C, 0xFB7D),
    "ح": (0xFEA1, 0xFEA2, 0xFEA3, 0xFEA4), "خ": (0xFEA5, 0xFEA6, 0xFEA7, 0xFEA8),
    "د": (0xFEA9, 0xFEAA, None, None), "ذ": (0xFEAB, 0xFEAC, None, None),
    "ر": (0xFEAD, 0xFEAE, None, None), "ز": (0xFEAF, 0xFEB0, None, None), "ژ": (0xFB8A, 0xFB8B, None, None),
    "س": (0xFEB1, 0xFEB2, 0xFEB3, 0xFEB4), "ش": (0xFEB5, 0xFEB6, 0xFEB7, 0xFEB8),
    "ص": (0xFEB9, 0xFEBA, 0xFEBB, 0xFEBC), "ض": (0xFEBD, 0xFEBE, 0xFEBF, 0xFEC0),
    "ط": (0xFEC1, 0xFEC2, 0xFEC3, 0xFEC4), "ظ": (0xFEC5, 0xFEC6, 0xFEC7, 0xFEC8),
    "ع": (0xFEC9, 0xFECA, 0xFECB, 0xFECC), "غ": (0xFECD, 0xFECE, 0xFECF, 0xFED0),
    "ف": (0xFED1, 0xFED2, 0xFED3, 0xFED4), "ق": (0xFED5, 0xFED6, 0xFED7, 0xFED8),
    "ک": (0xFB8E, 0xFB8F, 0xFB90, 0xFB91), "ك": (0xFED9, 0xFEDA, 0xFEDB, 0xFEDC),
    "گ": (0xFB92, 0xFB93, 0xFB94, 0xFB95), "ل": (0xFEDD, 0xFEDE, 0xFEDF, 0xFEE0),
    "م": (0xFEE1, 0xFEE2, 0xFEE3, 0xFEE4), "ن": (0xFEE5, 0xFEE6, 0xFEE7, 0xFEE8),
    "ه": (0xFEE9, 0xFEEA, 0xFEEB, 0xFEEC), "و": (0xFEED, 0xFEEE, None, None), "ؤ": (0xFE85, 0xFE86, None, None),
    "ی": (0xFBFC, 0xFBFD, 0xFBFE, 0xFBFF), "ي": (0xFEF1, 0xFEF2, 0xFEF3, 0xFEF4),
    "ى": (0xFEEF, 0xFEF0, None, None), "ئ": (0xFE89, 0xFE8A, 0xFE8B, 0xFE8C),
    "ة": (0xFE93, 0xFE94, None, None), "ء": (0xFE80, None, None, None),
}


def shape(word: str) -> str:
    """Presentation-form string in VISUAL left-to-right order (what PIL should draw)."""
    letters = [c for c in word if c in _F]
    out = []
    for i, c in enumerate(letters):
        iso, fin, ini, med = _F[c]
        prev_joins = i > 0 and _F[letters[i - 1]][2] is not None       # previous letter joins forward
        dual = ini is not None
        has_next = i + 1 < len(letters)
        if prev_joins and dual and has_next:
            cp = med
        elif prev_joins:
            cp = fin
        elif dual and has_next:
            cp = ini
        else:
            cp = iso
        out.append(chr(cp if cp else iso))
    return "".join(reversed(out))


# --- token generators ---------------------------------------------------------------

def _words(n: int, rng: random.Random) -> list[str]:
    con = sqlite3.connect(str(DEHKHODA))
    total = con.execute("select count(*) from words").fetchone()[0]
    ids = rng.sample(range(1, total + 1), min(n, total))
    q = "select word_norm from words where id in (%s)" % ",".join(map(str, ids))
    rows = [r[0] for r in con.execute(q) if r[0]]
    con.close()
    out = []
    for w in rows:
        for part in w.replace("‌", " ").split():
            if 2 <= len(part) <= 10 and all(c in _F for c in part):
                out.append(part)
    return out


def _digits(s: str, script: str) -> str:
    tbl = {"persian": PERSIAN, "arabic": ARABIC, "latin": LATIN}[script]
    return "".join(tbl[int(c)] for c in s)


def number_token(rng: random.Random) -> str:
    script = rng.choices(["persian", "arabic", "latin"], [0.7, 0.1, 0.2])[0]
    kind = rng.choices(["plain", "date", "amount", "phone_list", "iban", "ref"], [0.35, 0.15, 0.15, 0.15, 0.05, 0.15])[0]
    d = lambda k: "".join(rng.choice("0123456789") for _ in range(k))
    if kind == "plain":
        s = d(rng.randint(3, 12))
    elif kind == "date":
        s = f"{rng.choice(['13', '14'])}{d(2)}{rng.choice('/-.')}{rng.randint(1, 12):02d}{rng.choice('/-.')}{rng.randint(1, 30):02d}"
    elif kind == "amount":
        groups = [d(rng.randint(1, 3))] + [d(3) for _ in range(rng.randint(1, 4))]
        s = rng.choice(["/", ",", "،"]).join(groups)
    elif kind == "phone_list":
        s = "-".join(d(rng.choice([4, 7, 8, 11])) for _ in range(rng.randint(1, 3)))
    elif kind == "iban":
        s = "IR" + d(24)
    else:
        parts = [d(rng.randint(2, 5)) for _ in range(rng.randint(2, 5))]
        s = "/".join(parts)
    out = []
    for c in s:
        out.append(_digits(c, script) if c.isdigit() else c)
    return "".join(out)


_LATIN_WORDS = ["Email", "info", "yahoo", "gmail", "com", "ir", "Sabz", "gostar", "Tel", "Fax", "No", "www", "@", "IR", "Ref",
                "Address", "Web", "Mobile", "Post", "Code", "Iran", "Mashhad", "Tehran"]
_ALPHA = "abcdefghijklmnopqrstuvwxyz"


def latin_token(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.4:
        return rng.choice(_LATIN_WORDS) + rng.choice(["", ".", ":", "@", "_", "-"])
    if r < 0.7:                                          # email-like: letters, digits, @ . _
        user = "".join(rng.choice(_ALPHA) for _ in range(rng.randint(3, 9)))
        return f"{user}{rng.choice(['', '_', '.'])}{rng.choice(['', '8', '2006', '110'])}@{rng.choice(['gmail', 'yahoo', 'mail'])}.com"
    word = "".join(rng.choice(_ALPHA + _ALPHA.upper()) for _ in range(rng.randint(2, 8)))
    return word


# --- rendering -------------------------------------------------------------------

def render_line(tokens: list[tuple[str, str]], font: ImageFont.FreeTypeFont, px: int, rng: random.Random):
    """tokens: (kind, text) where kind in {word, number, latin, punct}. Laid out right to
    left. Returns (gray image, [(x0, x1, kind, text)]) with per-token x-ranges, and for
    numbers the per-character x-ranges."""
    space = max(3, int(px * rng.uniform(0.25, 0.6)))
    widths = [max(1, int(font.getlength(t))) for _, t in tokens]
    W = sum(widths) + space * (len(tokens) + 1) + 2 * px
    H = int(px * 2.6)
    im = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(im)
    x = W - px
    y = int(px * 0.6)
    spans = []
    ring_zero = rng.random() < 0.3       # some office fonts draw ۰ as a small hollow ring
    for (kind, text), w in zip(tokens, widths):
        x -= w
        chars = None
        if kind == "number":
            chars = []
            for i, c in enumerate(text):
                x0 = x + font.getlength(text[:i]); x1 = x + font.getlength(text[:i + 1])
                chars.append((x0, x1, c))
            if ring_zero and any(c in "۰٠" for c in text):
                # draw char by char so the zeros can be replaced by rings
                ref = font.getbbox("۵")
                top, bot = y + ref[1], y + ref[3]
                dh = max(4, bot - top)
                for x0, x1, c in chars:
                    if c in "۰٠":
                        r = dh * rng.uniform(0.22, 0.32); cx = (x0 + x1) / 2; cy = (top + bot) / 2 + dh * rng.uniform(-0.05, 0.08)
                        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=0, width=max(1, int(px / rng.uniform(8, 14))))
                    else:
                        d.text((x0, y), c, font=font, fill=0)
            else:
                d.text((x, y), text, font=font, fill=0)
        else:
            d.text((x, y), text, font=font, fill=0)
        spans.append((x, x + w, kind, text, chars))
        x -= space
    return im, spans


def augment(im: Image.Image, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        im = im.rotate(rng.uniform(-1.2, 1.2), resample=Image.BICUBIC, fillcolor=255)
    if rng.random() < 0.85:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.4)))
    a = np.asarray(im)
    if rng.random() < 0.4:
        k = np.ones((2, 2), np.uint8)
        a = cv2.erode(a, k) if rng.random() < 0.5 else cv2.dilate(a, k)
    if rng.random() < 0.6:                       # contrast / shadow
        gain = rng.uniform(0.6, 1.0); bias = rng.uniform(0, 70)
        a = np.clip(a.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        a = np.clip(a.astype(np.float32) + np.random.normal(0, rng.uniform(2, 12), a.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.6:                       # JPEG ringing, as the scans have
        _, enc = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(40, 90)])
        a = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return a


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


def components(b: np.ndarray, min_area: int = 3):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        out.append((int(x), int(y), int(x + w), int(y + h), i))
    return lab, out


def latin_is_persian_shaped(font: ImageFont.FreeTypeFont) -> bool:
    """The classic B-series fonts draw ASCII 0-9 with Persian glyph shapes. Labels
    describe pixels, so for such a font a Latin code point is a Persian digit."""
    same = 0
    for a, b in zip(LATIN, PERSIAN):
        ma, mb = np.asarray(font.getmask(a)), np.asarray(font.getmask(b))
        if ma.shape == mb.shape and np.array_equal(ma, mb):
            same += 1
    return same >= 7


def label_component(box, spans, latin_as_persian: bool = False) -> int:
    x0, y0, x1, y1, _ = box
    cx = (x0 + x1) / 2
    for sx0, sx1, kind, text, chars in spans:
        if sx0 - 2 <= cx <= sx1 + 2:
            if kind != "number":
                return REJECT
            hits = [c for (cx0, cx1, c) in chars if cx0 - 1 <= cx <= cx1 + 1]
            if not hits:
                return REJECT
            # a component wider than ~1.4 characters is two touching glyphs: reject
            covering = [c for (cx0, cx1, c) in chars if x0 < cx1 and x1 > cx0 and (min(x1, cx1) - max(x0, cx0)) > 0.35 * (cx1 - cx0)]
            if len(covering) > 1:
                return REJECT
            c = hits[0]
            if c in PERSIAN:
                return PERSIAN.index(c)
            if c in ARABIC:
                return ARABIC.index(c)
            if c in LATIN:
                return LATIN.index(c) if latin_as_persian else 10 + LATIN.index(c)
            return SEP_CLASS.get(c, REJECT)
    return REJECT


def make_tokens(words: list[str], rng: random.Random) -> list[tuple[str, str]]:
    n = rng.randint(3, 9)
    toks = []
    for _ in range(n):
        r = rng.random()
        if r < 0.5:
            toks.append(("word", shape(rng.choice(words))))
        elif r < 0.82:
            toks.append(("number", number_token(rng)))
        elif r < 0.95:
            toks.append(("latin", latin_token(rng)))
        else:
            toks.append(("punct", rng.choice([":", "،", ".", "-", "(", ")", "«", "»"])))
    return toks


def build(lines_per_font: int, seed: int, out: Path) -> dict:
    rng = random.Random(seed)
    np.random.seed(seed)
    fonts = _fonts()
    words = _words(20000, rng)
    X, S, y, F = [], [], [], []
    t0 = time.perf_counter()
    counts = np.zeros(len(CLASSES), np.int64)
    latin_shaped: dict[str, bool] = {}
    for fi, fp in enumerate(fonts):
        for _ in range(lines_per_font):
            px = rng.choice([16, 18, 20, 24, 28, 34, 40, 48, 60])
            try:
                font = ImageFont.truetype(fp, px)
            except Exception:
                break
            toks = make_tokens(words, rng)
            lap = latin_shaped.setdefault(fp, latin_is_persian_shaped(font))
            im, spans = render_line(toks, font, px, rng)
            gray = augment(im, rng)
            b = binarize(gray)
            lab, comps = components(b)
            if not comps:
                continue
            ref_h, cy = line_context([c[:4] for c in comps])
            for c in comps:
                x0, y0, x1, y1, idx = c
                if (y1 - y0) < 3 and (x1 - x0) < 3:
                    continue
                cls = label_component(c, spans, lap)
                # keep the reject class from swamping the digits: subsample it
                if cls == REJECT and rng.random() < 0.6:
                    continue
                glyph = normalize_glyph((lab[y0:y1, x0:x1] == idx).astype(np.uint8) * 255)
                X.append(glyph); S.append(scalars(c[:4], ref_h, cy)); y.append(cls); F.append(fi)
                counts[cls] += 1
        if (fi + 1) % 20 == 0:
            print(f"  fonts {fi + 1}/{len(fonts)}  samples {len(X)}  {time.perf_counter() - t0:.0f}s", flush=True)
    X = np.stack(X).astype(np.uint8); S = np.stack(S).astype(np.float32)
    y = np.array(y, np.int16); F = np.array(F, np.int16)
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, X=X, S=S, y=y, font=F, classes=np.array(CLASSES), fonts=np.array(fonts))
    info = {"samples": int(len(y)), "fonts": len(fonts), "fonts_latin_persian_shaped": sum(latin_shaped.values()), "per_class": {CLASSES[i]: int(counts[i]) for i in range(len(CLASSES))},
            "seconds": round(time.perf_counter() - t0, 1), "out": str(out)}
    return info


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines-per-font", type=int, default=80)
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    print(json.dumps(build(a.lines_per_font, a.seed, Path(a.out)), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
