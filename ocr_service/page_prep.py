# -*- coding: utf-8 -*-
r"""Find the sheet of paper in the frame and hand the transcriber that, and only that.

A phone photo is not a page: it is a page lying on a desk, under a shadow, at an angle.
Everything in the frame that is not paper is texture, and `flatten()`'s background
division turns texture into ink — on the brief's image 2 the empty grey desk below the
sheet produced four "lines" of output, one of them a whole fabricated table row
(«18 8 تا دا که ۱8111353112»). No amount of downstream filtering can fix that
reliably, because by then the noise looks exactly like faint text. It has to not be in
the image.

    find_quad   the paper's four corners, or None when the frame already is the page
    warp        perspective-correct to a rectangle (also removes the tilt, so a warped
                page needs no further deskew)
    crop_page   the whole decision: quad -> warp, else bounding box, else unchanged

What is deliberately NOT done here:

  * No binarisation, no contrast work, no resampling. This stage only answers "which
    pixels are the page"; `transcribe.flatten()` owns everything after that, and the
    two must not both decide the same thing.
  * A page that already fills the frame is returned untouched. The 90-document corpus
    is CamScanner output, already cropped, and warping an already-rectangular page
    costs a resample for nothing.

`Prep.to_original` maps a point in the prepared image back to the frame the caller
uploaded, so boxes can still be drawn on the original photo.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

# A contour must cover at least this much of the frame to be the page. Below it we are
# looking at a label, a shadow or a desk edge.
MIN_PAGE_FRAC = 0.25
# Above this, the page already is the frame: a scan, or a photo cropped by the camera app.
#
# This number is a deliberately blunt instrument, and the reason is a negative result worth
# recording. Five candidate ways to tell "scan that needs no crop" from "photo of a page on
# a desk" were measured on the 54 dev scans against the two phone photos — brightness of the
# surround, its texture, the page's bounding box, the quad's perspective offset, and its
# rectangularity — and **every one of them overlaps**: the least rectangular corpus scan is
# less rectangular than the photo, the darkest surround belongs to a scan, and so on. There
# is no cheap geometric signal here that separates the two populations.
#
# So the gate asks only the question it can answer honestly: is a large part of this frame
# not page? At 0.93 and 0.97 the crop fired on corpus scans and made every one of the three
# documents it touched worse (CER +1.4 to +7.1 pp, numbers and atoms both down). At 0.90 it
# touches no corpus scan at all and still takes the brief's image 2, whose page covers 0.867
# of its frame. A page surrounded by a thin border is left alone — that border costs little,
# and cropping it has been measured to cost more.
FULL_FRAME_FRAC = 0.90
# A quadrilateral this close to its own bounding box is a rectangle photographed straight
# on; cropping it is free, warping it is a resample that buys nothing.
MIN_WARP_OFFSET = 0.015
# How far a quadrilateral may depart from its own bounding box before we distrust it as a
# perspective estimate and fall back to a straight crop.
MAX_QUAD_SKEW = 0.25


@dataclass
class Prep:
    """What was done to the image, and how to get back to the original."""
    cropped: bool = False
    warped: bool = False
    quad: list | None = None                       # 4 corners in original coordinates
    box: tuple | None = None                       # (x0, y0, x1, y1) in original coordinates
    original_size: tuple = (0, 0)
    size: tuple = (0, 0)
    inverse: np.ndarray | None = field(default=None, repr=False)   # prepared -> original

    def to_original(self, x: float, y: float) -> tuple[float, float]:
        if self.inverse is None:
            return (x, y)
        v = self.inverse @ np.array([x, y, 1.0])
        return (float(v[0] / v[2]), float(v[1] / v[2]))

    def as_dict(self) -> dict:
        return {"cropped": self.cropped, "warped": self.warped, "quad": self.quad,
                "box": list(self.box) if self.box else None,
                "original_size": list(self.original_size), "size": list(self.size)}


def _to_cv(im: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)


def _to_pil(a: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))


def _page_mask(bgr: np.ndarray) -> np.ndarray:
    """Paper is the smooth region; a desk is a textured one.

    Brightness alone does not work and the measurement says so: on the brief's image 2 the
    concrete desk sits at V 150-175 and the paper at 160-210, overlapping so heavily that
    Otsu puts the threshold at 105 and calls 94% of the frame page. Local standard
    deviation separates the same two surfaces with no overlap at all — paper 1.0-1.4,
    desk 11-15 — because paper is smooth and a desk is speckled.

    Text is textured too, so the text block would be cut out of its own page. That is why
    the largest contour is taken filled: the text lines are holes inside the sheet, and
    filling the outer contour puts them back.
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = 11
    mu = cv2.blur(g, (k, k))
    sd = np.sqrt(np.maximum(cv2.blur(g * g, (k, k)) - mu * mu, 0))
    sd = cv2.blur(sd, (41, 41))                     # one stroke is not a texture
    t, _ = cv2.threshold(np.clip(sd * 4, 0, 255).astype(np.uint8), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    smooth = (sd * 4 <= t)
    # Whatever is as bright as the paper is paper, however textured: this is what keeps a
    # dense block of text, and a coloured letterhead, inside the page.
    lum = cv2.blur(g, (41, 41))
    paper = float(np.percentile(mu, 97))
    bright = lum >= 0.88 * paper
    # Smoothness alone is not enough to call something paper. Otsu puts the threshold
    # between the two textures actually present, so on a page whose text is high contrast
    # and whose desk is only mildly speckled it lands *above* the desk and declares the
    # desk smooth — and then the whole frame is "page". A surface well darker than the
    # paper is not paper however flat it is.
    mask = ((bright | (smooth & (lum >= 0.70 * paper))).astype(np.uint8)) * 255
    kk = max(9, min(bgr.shape[:2]) // 50) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kk, kk), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return filled


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def find_quad(bgr: np.ndarray) -> tuple[np.ndarray | None, tuple | None]:
    """(quad, bounding box) of the page in original coordinates, each possibly None.

    The quad is only returned when the contour really is a quadrilateral: a page held
    flat against a desk gives one, a curled or partly occluded page does not, and
    warping to a wrong quad stretches the text out of shape.
    """
    h, w = bgr.shape[:2]
    scale = 900 / max(h, w)
    small = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr
    sh, sw = small.shape[:2]
    cnts, _ = cv2.findContours(_page_mask(small), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_PAGE_FRAC * sh * sw:
        return None, None
    # "Is there anything here but page?" is asked of the CONVEX HULL, and the two wrong
    # answers either side of it were both measured. The bounding box says no for a sheet
    # lying at an angle, whose box fills the frame while its corners are still desk — that
    # left the brief's image 2 uncropped. The raw contour says yes for a clean scan, whose
    # mask is ragged at the edges, so 34 of 54 corpus scans were cropped and 13 of them
    # perspective-warped for nothing: whole numbers -7.7 pp and CER +3.2 pp on exactly
    # those 13. A page is convex, so its hull is the honest measure of how much of the
    # frame it occupies.
    if cv2.contourArea(cv2.convexHull(c)) >= FULL_FRAME_FRAC * sh * sw:
        return None, None                       # already a page-filling image

    x, y, bw, bh = cv2.boundingRect(c)
    s = 1 / scale if scale < 1 else 1.0
    box = (int(x * s), int(y * s), int((x + bw) * s), int((y + bh) * s))

    quad = None
    peri = cv2.arcLength(c, True)
    for eps in (0.02, 0.03, 0.04, 0.05):
        ap = cv2.approxPolyDP(c, eps * peri, True)
        if len(ap) == 4 and cv2.isContourConvex(ap):
            q = _order_quad(ap)
            # trust it only if it is a mild perspective, not a wild shape
            if cv2.contourArea(ap) >= 0.75 * bw * bh:
                side = max(np.ptp(q[:, 0][[0, 3]]), np.ptp(q[:, 0][[1, 2]]),
                           np.ptp(q[:, 1][[0, 1]]), np.ptp(q[:, 1][[2, 3]]))
                if side <= MAX_QUAD_SKEW * max(bw, bh):
                    quad = q * s
            break
    return quad, box


def warp(bgr: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perspective-correct to a rectangle. Returns the image and the inverse transform."""
    tl, tr, br, bl = quad
    wid = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    hei = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    wid, hei = max(wid, 8), max(hei, 8)
    dst = np.array([[0, 0], [wid - 1, 0], [wid - 1, hei - 1], [0, hei - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    out = cv2.warpPerspective(bgr, M, (wid, hei), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return out, np.linalg.inv(M)


def _text_outside(bgr: np.ndarray, keep: np.ndarray) -> int:
    """Glyph-sized ink components lying outside the region we are about to keep.

    The crop must never be the thing that loses a line of the letter. A letterhead band,
    a footer, a marginal note: if the detector got the page edge wrong, this sees real
    letters in the offcut and the crop is abandoned. Losing text is unrecoverable;
    keeping a strip of desk is not.
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(g, (0, 0), max(8, min(g.shape) / 60))
    ink = (g.astype(np.float32) / np.maximum(bg, 1) * 255 < 200).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n < 3:
        return 0
    h = st[1:, 3]
    med = float(np.median(h[h > 1])) if (h > 1).any() else 0.0
    if med <= 0:
        return 0
    # Counting loose components is not enough: a speckled desk produces thousands of
    # glyph-sized specks, and on the brief's image 2 that refused the very crop that
    # removes them. Text is distinguished from texture by *alignment* — several
    # components of similar height sharing a baseline — so rows are counted, not blobs.
    rows: dict[int, int] = {}
    for i in range(1, n):
        x, y, w, hh, area = st[i]
        if not (0.5 * med <= hh <= 2.5 * med and area >= 20):
            continue
        if keep[min(y + hh // 2, keep.shape[0] - 1), min(x + w // 2, keep.shape[1] - 1)]:
            continue
        rows[int((y + hh / 2) // max(med, 1))] = rows.get(int((y + hh / 2) // max(med, 1)), 0) + 1
    return sum(1 for c in rows.values() if c >= 4)


def crop_page(im: Image.Image) -> tuple[Image.Image, Prep]:
    """The page, cut out of the frame it was photographed in."""
    prep = Prep(original_size=im.size, size=im.size)
    bgr = _to_cv(im)
    quad, box = find_quad(bgr)

    # A quad that is already a rectangle gets a plain crop: same pixels kept, no resample.
    if quad is not None:
        bx = (quad[:, 0].min(), quad[:, 1].min(), quad[:, 0].max(), quad[:, 1].max())
        off = max(np.abs(quad[:, 0] - np.where(quad[:, 0] < (bx[0] + bx[2]) / 2, bx[0], bx[2])).max(),
                  np.abs(quad[:, 1] - np.where(quad[:, 1] < (bx[1] + bx[3]) / 2, bx[1], bx[3])).max())
        if off < MIN_WARP_OFFSET * max(bx[2] - bx[0], bx[3] - bx[1]):
            box = (int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3]))
            quad = None

    # Refuse any crop that would throw away text.
    if quad is not None or box is not None:
        keep = np.zeros(bgr.shape[:2], bool)
        if quad is not None:
            cv2.fillConvexPoly(keep.view(np.uint8), quad.astype(np.int32), 1)
            keep = keep.view(np.uint8).astype(bool)
        else:
            x0, y0, x1, y1 = box
            keep[y0:y1, x0:x1] = True
        if _text_outside(bgr, keep) >= 2:
            return im, prep

    if quad is not None:
        out, inv = warp(bgr, quad)
        prep.cropped = prep.warped = True
        prep.quad = [[float(a), float(b)] for a, b in quad]
        prep.inverse = inv
    elif box is not None:
        x0, y0, x1, y1 = box
        out = bgr[y0:y1, x0:x1]
        prep.cropped = True
        prep.box = box
        prep.inverse = np.array([[1.0, 0.0, x0], [0.0, 1.0, y0], [0.0, 0.0, 1.0]])
    else:
        return im, prep

    if out.shape[0] < 32 or out.shape[1] < 32:    # a bad detection; keep the frame
        return im, Prep(original_size=im.size, size=im.size)
    pil = _to_pil(out)
    prep.size = pil.size
    return pil, prep
