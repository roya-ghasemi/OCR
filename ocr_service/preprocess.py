# -*- coding: utf-8 -*-
r"""Edge / server pre-processing: auto-crop the page, deskew, compress under a byte cap.

Used in two places with one code path:
  * server side — `prepare(image_bytes)` runs on every incoming image when
    `OCRS_PREPROCESS_INCOMING=1` (raw phone photos arrive un-cropped and skewed);
  * client spec — `python -m ocr_service.preprocess in.jpg out.jpg` is the reference
    behaviour for the mobile team, and prints the same stats the server logs.

Order matters: crop first (so the deskew angle is estimated from the page, not the
desk), deskew second, resize + compress last.

Assumptions, each marked in code:
  * page = largest bright quadrilateral-ish contour covering ≥ 20% of the frame;
    if none, the frame is kept whole (a CamScanner export is already cropped);
  * skew = dominant angle of long Hough lines within ±15°; text-line projection is
    used as a tie-break. Angles below 0.3° are not corrected;
  * output JPEG (not WebP) by default because the llama.cpp image loader has no
    WebP decoder; the mobile client may send WebP to the gateway, which transcodes.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class PrepStats:
    in_bytes: int
    out_bytes: int
    in_size: tuple[int, int]
    out_size: tuple[int, int]
    cropped: bool
    crop_box: tuple[int, int, int, int] | None
    skew_deg: float
    deskewed: bool
    quality: int
    format: str
    seconds: float


def _to_cv(im: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)


def _to_pil(a: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))


# --- 1. auto-crop -----------------------------------------------------------

def find_page(bgr: np.ndarray, min_area_frac: float = 0.20) -> tuple[int, int, int, int] | None:
    """Bounding box of the page, or None if the frame already is the page."""
    h, w = bgr.shape[:2]
    scale = 800 / max(h, w)
    small = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Paper is the bright region; Otsu separates it from a darker desk.
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area_frac * small.shape[0] * small.shape[1]:
        return None
    x, y, bw, bh = cv2.boundingRect(c)
    # Assumption: if the page fills ≥ 97% of the frame there is nothing to crop.
    if bw * bh >= 0.97 * small.shape[0] * small.shape[1]:
        return None
    s = 1 / scale if scale < 1 else 1.0
    return (int(x * s), int(y * s), int((x + bw) * s), int((y + bh) * s))


# --- 2. deskew ----------------------------------------------------------------

def estimate_skew(bgr: np.ndarray, max_deg: float = 15.0) -> float:
    """Angle to pass to `rotate()` (OpenCV convention, CCW positive) to level the
    text lines. Text rising to the right has a negative image-space angle and needs
    a clockwise (negative) rotation, so the median line angle is returned as-is."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    scale = 1200 / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=120,
                            minLineLength=gray.shape[1] // 4, maxLineGap=20)
    angles: list[float] = []
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            a = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(a) <= max_deg:
                angles.append(a)
    if angles:
        return float(np.median(angles))
    # Tie-break / fallback: text-row projection sharpness over candidate angles.
    inv = 255 - gray
    best, best_score = 0.0, -1.0
    for a in np.arange(-max_deg, max_deg + 0.01, 0.5):
        M = cv2.getRotationMatrix2D((inv.shape[1] / 2, inv.shape[0] / 2), a, 1.0)
        r = cv2.warpAffine(inv, M, (inv.shape[1], inv.shape[0]), flags=cv2.INTER_NEAREST)
        proj = r.sum(axis=1).astype(np.float64)
        score = float(np.var(proj))
        if score > best_score:
            best, best_score = float(a), score
    return best


def rotate(bgr: np.ndarray, deg: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(bgr, M, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# --- 3. compress -------------------------------------------------------------

def compress(im: Image.Image, max_bytes: int, fmt: str = "JPEG", max_edge: int = 3200,
             q_hi: int = 92, q_lo: int = 40) -> tuple[bytes, int, tuple[int, int]]:
    """Largest quality (then largest size) whose encoded output fits `max_bytes`."""
    im = ImageOps.exif_transpose(im).convert("RGB")
    if max(im.size) > max_edge:
        im.thumbnail((max_edge, max_edge), Image.LANCZOS)
    while True:
        lo, hi, best = q_lo, q_hi, None
        while lo <= hi:                          # binary search on quality
            q = (lo + hi) // 2
            buf = io.BytesIO()
            im.save(buf, fmt, quality=q, optimize=True, **({"method": 4} if fmt == "WEBP" else {}))
            if buf.tell() <= max_bytes:
                best, lo = (buf.getvalue(), q), q + 1
            else:
                hi = q - 1
        if best:
            return best[0], best[1], im.size
        # Even q_lo is too large: shrink 15% and try again.
        im = im.resize((int(im.width * 0.85), int(im.height * 0.85)), Image.LANCZOS)


# --- pipeline ---------------------------------------------------------------

def prepare(image_bytes: bytes, *, max_bytes: int = 300_000, fmt: str = "JPEG",
            max_edge: int = 3200, do_crop: bool = True, do_deskew: bool = True,
            min_skew_deg: float = 0.3) -> tuple[bytes, PrepStats]:
    t0 = time.perf_counter()
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
    in_size = im.size
    bgr = _to_cv(im)

    box = find_page(bgr) if do_crop else None
    if box:
        x0, y0, x1, y1 = box
        bgr = bgr[y0:y1, x0:x1]

    skew = estimate_skew(bgr) if do_deskew else 0.0
    deskewed = do_deskew and abs(skew) >= min_skew_deg
    if deskewed:
        bgr = rotate(bgr, skew)

    out, q, out_size = compress(_to_pil(bgr), max_bytes, fmt, max_edge)
    stats = PrepStats(len(image_bytes), len(out), in_size, out_size, bool(box), box,
                      round(skew, 2), deskewed, q, fmt, round(time.perf_counter() - t0, 3))
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Reference client-side pre-processing.")
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--max-kb", type=int, default=300)
    ap.add_argument("--format", choices=["JPEG", "WEBP"], default="JPEG")
    ap.add_argument("--no-crop", action="store_true"); ap.add_argument("--no-deskew", action="store_true")
    a = ap.parse_args()
    data, st = prepare(open(a.src, "rb").read(), max_bytes=a.max_kb * 1000, fmt=a.format,
                       do_crop=not a.no_crop, do_deskew=not a.no_deskew)
    open(a.dst, "wb").write(data)
    print(json.dumps(asdict(st), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
