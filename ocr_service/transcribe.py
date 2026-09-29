# -*- coding: utf-8 -*-
r"""Full-page transcription of ANY image: every line of printed text, in reading order.

    image -> [deskew] -> page binarisation + clean-up -> line candidates from three
    Tesseract layout passes + a connected-component line finder read line by line
    (Persian, English) -> best reading per physical line -> CNN digit reader
    replaces number tokens -> junk-line filter -> text + lines + numbers

CPU only; no language model anywhere in the path, so nothing can be invented: every
character in the output was recognised from pixels by Tesseract or the digit CNN.

Why this design (E19, ocr_eval/experiments.md), all on the 51-letter dev split vs the
previous Qwen2.5-VL letter-JSON path:

  * The VLM path capped `body_text` at 1,400 characters (GBNF grammar) and dropped
    whole paragraphs of longer pages; its letter-only prompt made it invent a sender
    and receiver for pages that are not letters; and on dense Persian text it emits
    words out of right-to-left order. Full-page CER 28.5% -> 15.7% here.
  * One Tesseract layout mode is not enough: each drops different lines (psm 4 loses
    lines next to a binder-ring shadow, psm 6 merges tight lines). Candidates from
    all passes compete per physical line on confident-character mass.
  * `--psm 7` sometimes returns nothing for a clean line that `--psm 13` reads
    perfectly; both are tried.
  * `-l fas+eng` corrupts Persian words with Latin fragments; each line is read with
    `fas` and with `eng` separately and the English read wins only on a Latin line.
  * Tesseract misreads Persian thousands separators as digits; the CNN glyph reader
    (digit_reader_v2) replaces a number token only where it saw digits at the same
    place. Digit-group recall 70.8% -> 81.3%, whole numbers 54.8% -> 65.3%.
  * Straight rules are removed with long line kernels, NOT by dropping components:
    a signature or table grid touching text would take the text with it. The kernel
    is a quarter of the page width because Persian baselines (kashida) are long.
  * Deskew only at >= 1 degree: resampling a nearly straight page costs the digit
    reader more than the tiny rotation gains.

Coordinates: `bbox` values are in the pixel space of the (deskewed) input image;
`skew_deg` is the rotation that was applied (PIL convention, counter-clockwise).
"""
from __future__ import annotations

import csv
import io
import logging
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

log = logging.getLogger(__name__)

_NUMTOK = re.compile(r"[0-9۰-۹٠-٩]")
_PERSIAN_DIGITS = str.maketrans("0123456789٠١٢٣٤٥٦٧٨٩", "۰۱۲۳۴۵۶۷۸۹۰۱۲۳۴۵۶۷۸۹")
_ASCII_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_KEY = re.compile(r"[ـ‌ً-ْ\s]")
_BIDI = re.compile("[‎‏‪-‮⁦-⁩]")
_HARAKAT = re.compile("[ً-ْ]")
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")


def amount_grouping(tok: str) -> str:
    """How a number token uses thousands separators.

    `ok` — a correctly grouped amount: one separator kind, 1-3 digits then groups of
    exactly three («۱۲۵,۰۰۰,۰۰۰», «۷/۷۷۵/۰۰۰/۰۰۰»).
    `broken` — grouped, but a group is not three digits («۳۲۱/۰۰۰/۰۰»,
    «۱۲.۹۷۸۰۲۱۹.۵۷۱», «۱۶۵.۰۰۰۰۰۰» where two groups ran together): a digit was lost or
    invented, so the amount must not be trusted. It is flagged, never repaired —
    filling the gap in would mean printing a digit nobody read.
    `none` — not a grouped amount: dates, reference, account and phone numbers.

    Checked against the corpus ground truth: it calls 27 amounts `ok` and raises no
    false alarm on a correct one.
    """
    t = re.sub(r"^\D+|\D+$", "", tok.translate(_ASCII_DIGITS))
    if not re.fullmatch(r"\d+(?:[,./]\d+)+", t) or len({*re.findall(r"[,./]", t)}) != 1:
        return "none"
    head, *rest = re.split(r"[,./]", t)
    if not 1 <= len(head) <= 3:
        return "none"                       # «۱۴۰۳/۰۹/۲۰» is a date, not an amount
    if len(rest) >= 2 and all(len(g) == 3 for g in rest):
        return "ok"
    if len(rest) >= 2 and any(len(g) == 3 for g in rest):
        return "broken"                     # grouped, but one group is not whole
    if any(len(g) > 4 for g in rest) and len(re.sub(r"\D", "", t)) >= 6:
        return "broken"                     # a separator was missed: «۱۶۵.۰۰۰۰۰۰»
    return "none"


def _key(w: str) -> str:
    return _KEY.sub("", w).replace("ي", "ی").replace("ك", "ک")


@dataclass
class Word:
    text: str
    conf: float                                   # 0..100 (Tesseract); 101 = set by the digit reader
    box: tuple[int, int, int, int]                # working-image coordinates


@dataclass
class Line:
    words: list[Word]
    bbox: tuple[int, int, int, int]               # working-image coordinates
    source: str                                   # psm3 | psm4 | psm6 | line-fas | line-eng
    score: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def mean_conf(self) -> float:
        return sum(min(w.conf, 100.0) for w in self.words) / max(1, len(self.words))


@dataclass
class NumberOut:
    value: str
    value_ascii: str
    bbox: list[int]
    source: str                                   # glyph (CNN digit reader) | tesseract
    confidence: float                             # 0..1
    grouping: str = "none"                        # ok | broken | none (see amount_grouping)


@dataclass
class LineOut:
    text: str
    bbox: list[int]
    confidence: float                             # mean word confidence, 0..1
    source: str
    paragraph: int
    cells: list = field(default_factory=list)     # [{text, bbox}] when the line is split into
    #                                               table cells; empty for ordinary prose


@dataclass
class Transcript:
    text: str
    lines: list[LineOut]
    numbers: list[NumberOut]
    skew_deg: float
    image_size: tuple[int, int]
    timing: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


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


def working_scale(im: Image.Image, glyph_h: float = 0.0) -> float:
    """Tesseract reads best with glyphs ~25-40 px tall: upsample phone photos and small
    screenshots, cap huge scans. `glyph_h` is the text height measured at 1x; when the
    page-size rule would still leave glyphs under 24 px (a small image of small text),
    scale to ~32 px, at most 4x. Pages whose glyphs already reach 24 px keep the rule
    E19 was measured with."""
    scale = 2.0 if im.height < 2000 else 1.0
    if max(im.size) > 4200:
        scale = 3500 / max(im.size)
    if glyph_h and glyph_h * scale < 24:
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
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
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
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    h, w, area = st[:, 3], st[:, 2], st[:, 4]
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


# ------------------------------------------------------------------ transcriber
class Transcriber:
    def __init__(self, tesseract_cmd: str | None, tessdata_dir: str | None, digit_model: Path | None,
                 layout_psms: tuple[str, ...] = ("3", "4", "6"), deskew_min_deg: float = 1.0,
                 min_line_conf: float = 30.0, workers: int = 8, number_min_prob: float = 0.6):
        self.cmd, self.tessdata = tesseract_cmd, tessdata_dir
        self.layout_psms = tuple(layout_psms)
        self.deskew_min_deg = deskew_min_deg
        self.min_line_conf = min_line_conf
        self.workers = workers
        self.number_min_prob = number_min_prob
        self.digit_model = digit_model
        self._glyph = None
        self._glyph_error: str | None = None
        self.version: str | None = None

    # -- availability ------------------------------------------------------------
    def glyph_reader(self):
        if self._glyph is None and self._glyph_error is None:
            try:
                from .digit_reader_v2 import GlyphReaderV2
                kw = {"model_path": self.digit_model} if self.digit_model else {}
                self._glyph = GlyphReaderV2(min_digits=1, **kw)
            except Exception as exc:                      # model file missing: numbers stay Tesseract's
                self._glyph_error = f"{type(exc).__name__}: {exc}"
                log.warning("digit reader unavailable, numbers come from Tesseract only: %s", exc)
        return self._glyph

    def health(self) -> dict:
        ok, version, langs = False, None, []
        if self.cmd:
            try:
                p = subprocess.run([self.cmd, "--version"], capture_output=True, text=True, timeout=20)
                version = (p.stdout or p.stderr).splitlines()[0].strip()
                args = [self.cmd, "--list-langs"] + (["--tessdata-dir", self.tessdata] if self.tessdata else [])
                q = subprocess.run(args, capture_output=True, text=True, timeout=20)
                langs = [l.strip() for l in q.stdout.splitlines()[1:] if l.strip()]
                ok = "fas" in langs
            except Exception as exc:
                version = f"error: {exc}"
        self.version = version
        g = self.glyph_reader()
        return {"tesseract": {"available": ok, "version": version, "langs": langs, "cmd": self.cmd,
                              "tessdata_dir": self.tessdata},
                "digit_reader": {"available": g is not None, "version": getattr(g, "version", None),
                                 "error": self._glyph_error}}

    # -- tesseract ---------------------------------------------------------------
    def _run(self, img: Image.Image, psm: str, lang: str) -> list[dict]:
        buf = io.BytesIO()
        img.save(buf, "PNG")
        args = [self.cmd, "stdin", "stdout", "-l", lang, "--psm", psm, "-c", "tessedit_create_tsv=1"]
        if self.tessdata:
            args[5:5] = ["--tessdata-dir", self.tessdata]
        p = subprocess.run(args, input=buf.getvalue(), capture_output=True, timeout=120)
        rows = csv.DictReader(io.StringIO(p.stdout.decode("utf-8", "replace")), delimiter="\t",
                              quoting=csv.QUOTE_NONE)
        out = []
        for r in rows:
            t = (r.get("text") or "").strip()
            try:
                c = float(r.get("conf") or -1)
            except ValueError:
                c = -1.0
            if r.get("level") == "5" and t and c >= 0:
                x, y, w, h = (int(r[k]) for k in ("left", "top", "width", "height"))
                out.append({"text": t, "conf": c, "box": (x, y, x + w, y + h),
                            "key": (r["block_num"], r["par_num"], r["line_num"])})
        return out

    def _layout(self, page: Image.Image, psm: str) -> list[Line]:
        groups: dict[tuple, list[Word]] = {}
        for r in self._run(page, psm, "fas"):
            groups.setdefault(r["key"], []).append(Word(r["text"], r["conf"], r["box"]))
        lines = []
        for ws in groups.values():
            bbox = (min(w.box[0] for w in ws), min(w.box[1] for w in ws),
                    max(w.box[2] for w in ws), max(w.box[3] for w in ws))
            lines.append(Line(ws, bbox, "psm" + psm))
        return lines

    def _read_line(self, img: Image.Image, lang: str) -> list[Word]:
        """psm 7 (single line), with psm 13 (raw line) when 7 is empty or unsure:
        psm 7 returns nothing for some clean lines that 13 reads perfectly."""
        img, xmap = squeeze_kashida(img)
        framed = ImageOps.expand(img, border=30, fill=255)

        def read(psm):
            ws = [Word(r["text"], r["conf"], tuple(v - 30 for v in r["box"])) for r in self._run(framed, psm, lang)]
            return ws, (sum(w.conf for w in ws) / len(ws) if ws else -1.0)

        a, ma = read("7")
        if ma < 70:
            b, mb = read("13")
            a = b if mb > ma else a
        if xmap is None:
            return a
        last = len(xmap) - 1
        return [Word(w.text, w.conf, (int(xmap[min(max(w.box[0], 0), last)]), w.box[1],
                                      int(xmap[min(max(w.box[2] - 1, 0), last)]) + 1, w.box[3])) for w in a]

    # -- selection -----------------------------------------------------------------
    @staticmethod
    def _score(line: Line) -> float:
        return sum(len(_key(w.text)) * (min(w.conf, 100.0) / 100.0) ** 2 for w in line.words if w.conf >= 40)

    def _line_ok(self, line: Line) -> bool:
        good = [w for w in line.words if w.conf >= 60 and len(_key(w.text)) >= 2]
        return bool(good) and line.mean_conf >= self.min_line_conf

    @staticmethod
    def _order_segments(line: Line) -> None:
        """Split a line at gaps wider than a line-height (table cells, a label far from
        its value) and order the pieces right-to-left (left-to-right on an English
        line). Tesseract keeps a run of number cells in visual left-to-right order
        («۱۳ ۲۳» for the row «۲۳ | ۱۳»); within a piece its order is kept."""
        ws = line.words
        if len(ws) < 2:
            return
        h = max(1, line.bbox[3] - line.bbox[1])
        segs, cur = [], [ws[0]]
        for a, b in zip(ws, ws[1:]):
            gap = max(0, max(a.box[0], b.box[0]) - min(a.box[2], b.box[2]))
            if gap > 1.2 * h:
                segs.append(cur)
                cur = []
            cur.append(b)
        segs.append(cur)
        if len(segs) < 2:
            return
        rtl = line.source != "line-eng"
        segs.sort(key=lambda s: max(w.box[2] for w in s), reverse=rtl)
        line.words = [w for s in segs for w in s]

    @staticmethod
    def _trim_edges(line: Line) -> None:
        """Drop a lone 1-character token at either end of a line when it stands at
        least a line-height away from its neighbour: binder holes, margin ticks and
        page-edge specks read as «۰», «5», «ی». A real single-character word sits at
        normal word spacing and is kept."""
        h = max(1, line.bbox[3] - line.bbox[1])

        def gap(a: Word, b: Word) -> int:
            return max(0, max(a.box[0], b.box[0]) - min(a.box[2], b.box[2]))

        for end in (0, -1):
            ws = line.words
            if len(ws) >= 2:
                w, nb = (ws[0], ws[1]) if end == 0 else (ws[-1], ws[-2])
                if len(_key(w.text)) <= 1 and gap(w, nb) >= h:
                    ws.pop(end)

    def _candidates(self, page: Image.Image, ink: np.ndarray) -> list[Line]:
        with ThreadPoolExecutor(max(1, len(self.layout_psms))) as ex:
            layouts = list(ex.map(lambda p: self._layout(page, p), self.layout_psms))
        cands = [L for ls in layouts for L in ls]
        boxes, _ = text_lines(ink)
        crops = [_crop(page, b) for b in boxes]
        offs = [(max(0, b[0] - 2), max(0, b[1] - 2)) for b in boxes]
        with ThreadPoolExecutor(self.workers) as ex:
            fas = list(ex.map(lambda c: self._read_line(c, "fas"), crops))
            eng = list(ex.map(lambda c: self._read_line(c, "eng"), crops))
        for b, (ox, oy), wf, we in zip(boxes, offs, fas, eng):
            for ws, lang in ((wf, "fas"), (we, "eng")):
                if not ws:
                    continue
                if lang == "eng" and not any(_LATIN_WORD.search(w.text) and w.conf >= 70 for w in ws):
                    continue                            # an English read only competes on a Latin line
                moved = [Word(w.text, w.conf, (w.box[0] + ox, w.box[1] + oy, w.box[2] + ox, w.box[3] + oy)) for w in ws]
                cands.append(Line(moved, b, "line-" + lang))
        return cands

    @staticmethod
    def _conflict(a: Line, b: Line) -> bool:
        return _yov(a.bbox, b.bbox) > 0.4 and _xov(a.bbox, b.bbox) > 0.3

    @classmethod
    def _select(cls, cands: list[Line]) -> list[Line]:
        """The set of non-overlapping readings with the most confident characters,
        in reading order (rows top-down; within a row, right to left).

        Greedy-by-score alone is wrong: a slightly tilted line is read by psm 3/4 as two
        clean halves and by psm 6 as one piece merged with the next line. The single
        piece outscores each half, so greedy kept it and lost the words at the line's
        left end («نسیم خنک صبحگاهی» → «سیم بحگاهی»). After the greedy pass, each chosen
        reading is therefore compared with every compatible combination of the readings
        it displaced (exact search over a small neighbourhood)."""
        for L in cands:
            L.score = cls._score(L)
        pool = sorted((L for L in cands if L.score > 0), key=lambda L: -L.score)
        chosen: list[Line] = []
        for L in pool:
            if not any(cls._conflict(L, C) for C in chosen):
                chosen.append(L)
        for _ in range(3):
            improved = False
            for L in list(chosen):
                if L not in chosen:
                    continue
                others = [C for C in chosen if C is not L]
                alt = [c for c in pool if c not in chosen and cls._conflict(c, L)
                       and not any(cls._conflict(c, O) for O in others)][:8]
                if len(alt) < 2:
                    continue                                   # one displaced reading scored lower already
                clash = [sum(1 << j for j in range(len(alt)) if j != i and cls._conflict(alt[i], alt[j]))
                         for i in range(len(alt))]
                best, best_mask = L.score * 1.05, 0            # a swap must win clearly
                for mask in range(1, 1 << len(alt)):
                    bits = [i for i in range(len(alt)) if mask >> i & 1]
                    if any(clash[i] & mask for i in bits):
                        continue
                    s = sum(alt[i].score for i in bits)
                    if s > best:
                        best, best_mask = s, mask
                if best_mask:
                    chosen.remove(L)
                    chosen.extend(alt[i] for i in range(len(alt)) if best_mask >> i & 1)
                    improved = True
            if not improved:
                break
        chosen.sort(key=lambda L: ((L.bbox[1] + L.bbox[3]) / 2, -L.bbox[2]))
        rows: list[list[Line]] = []
        for L in chosen:
            if rows and _yov(L.bbox, rows[-1][0].bbox) > 0.5:
                rows[-1].append(L)
            else:
                rows.append([L])
        # pieces stay separate here: each is cleaned and junk-filtered on its own
        # before `_join_row`, or a speck read as «اس ی» rides along with a real line
        return [sorted(r, key=lambda L: -L.bbox[2]) for r in rows]

    def _finish_rows(self, rows: list[list[Line]]) -> list[Line]:
        """Per piece: order table cells, trim lone edge marks, drop junk; then join
        the surviving pieces of each row into one line."""
        for r in rows:
            for p in r:
                self._order_segments(p)
                self._trim_edges(p)
        rows = [[p for p in r if p.words and self._line_ok(p)] for r in rows]
        return [self._join_row(r) for r in rows if r]

    @staticmethod
    def _join_row(pieces: list[Line]) -> Line:
        """Pieces of one physical row (halves of a tilted line, table cells read as
        separate lines) become one line, right to left — or left to right when every
        piece is an English read."""
        if len(pieces) == 1:
            return pieces[0]
        if all(p.source == "line-eng" for p in pieces):
            pieces = sorted(pieces, key=lambda p: p.bbox[0])
        words = [w for p in pieces for w in p.words]
        bbox = (min(p.bbox[0] for p in pieces), min(p.bbox[1] for p in pieces),
                max(p.bbox[2] for p in pieces), max(p.bbox[3] for p in pieces))
        src = "+".join(dict.fromkeys(p.source for p in pieces))
        return Line(words, bbox, src, sum(p.score for p in pieces))

    # -- numbers --------------------------------------------------------------------
    def _fix_numbers(self, lines: list[Line], reads, scale: float) -> None:
        """Replace a Tesseract number token with the CNN digit reader's reading where
        the reader saw enough confident digits at the same place."""
        def hov(a, b):
            return max(0, min(a[2], b[2]) - max(a[0], b[0]))

        def vov(a, b):
            return max(0, min(a[3], b[3]) - max(a[1], b[1])) / max(1, min(a[3] - a[1], b[3] - b[1]))

        for L in lines:
            for i, w in enumerate(L.words):
                if not _NUMTOK.search(w.text):
                    continue
                ob = tuple(v / scale for v in w.box)
                rs = [r for r in reads if vov(r.box, ob) > 0.5 and hov(r.box, ob) > 0.5 * (r.box[2] - r.box[0])]
                if not rs:
                    continue
                if sum(r.n_digits for r in rs) < max(1, 0.6 * len(_NUMTOK.findall(w.text))):
                    continue
                if min(r.mean_prob for r in rs) < self.number_min_prob:
                    continue
                rs.sort(key=lambda r: r.box[0])
                core, joined = rs[0].text, True
                for prev, nxt in zip(rs, rs[1:]):
                    # the reader split one number in two, so put it back together — but
                    # never invent a separator that is not printed on the page: guessing
                    # one turned «۱۴۰۳/۰۹/۰۶» into «۱۴۰۳/۰۹/۰/۶» and «۱۳» into «۱/۳» (D73)
                    if nxt.box[0] - prev.box[2] > 0.35 * max(prev.line_h, nxt.line_h):
                        joined = False
                        break
                    core += nxt.text
                if not joined:
                    continue
                # between two readings of one amount, keep the one whose thousands
                # groups are whole: «۳۲۱/۰۰۰/۰۰» is a lost digit, «۳۲۱/۰۰۰/۰۰۰» is not
                if amount_grouping(core) == "broken" and amount_grouping(w.text) == "ok":
                    continue
                m = re.match(r"^(\D*)(.*?)(\D*)$", w.text)      # keep the token's own punctuation
                pre, suf = (m.group(1), m.group(3)) if m else ("", "")
                L.words[i] = Word(pre + core.replace(",", "،") + suf, 101.0 + min(r.mean_prob for r in rs), w.box)

    # -- main ---------------------------------------------------------------------------
    def transcribe(self, im: Image.Image) -> Transcript:
        t0 = time.perf_counter()
        timing: dict[str, float] = {}
        im = ImageOps.exif_transpose(im).convert("RGB")
        skew = text_skew(im)
        if abs(skew) >= self.deskew_min_deg:
            im = im.rotate(skew, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
        else:
            skew = 0.0
        timing["deskew_s"] = round(time.perf_counter() - t0, 3)

        t = time.perf_counter()
        scale = working_scale(im, measure_glyph_height(im))
        ink = clean_ink(np.asarray(flatten(im, scale)) == 0)
        page = Image.fromarray(np.where(ink, 0, 255).astype(np.uint8))
        timing["binarise_s"] = round(time.perf_counter() - t, 3)

        t = time.perf_counter()
        rows = self._select(self._candidates(page, ink))
        pieces = [p for r in rows for p in r]
        timing["tesseract_s"] = round(time.perf_counter() - t, 3)

        t = time.perf_counter()
        reader = self.glyph_reader()
        if reader is not None:
            self._fix_numbers(pieces, reader.read_page(im), scale)
        timing["digits_s"] = round(time.perf_counter() - t, 3)

        lines = self._finish_rows(rows)
        out_lines, numbers = self._assemble(lines, scale)
        text = self._join(out_lines)
        timing["total_s"] = round(time.perf_counter() - t0, 3)
        return Transcript(text=text, lines=out_lines, numbers=numbers, skew_deg=round(skew, 2),
                          image_size=im.size, timing=timing)

    @staticmethod
    def _clean_word(w: str, latin: bool) -> str:
        w = _BIDI.sub("", w)
        if not latin:
            w = w.replace("ي", "ی").replace("ك", "ک")
            # Tesseract invents harakat on bold letterhead-style text («بِمُدیرِیَت»);
            # printed Persian carries at most one per word («احتراماً», «بهارِ»).
            if len(_HARAKAT.findall(w)) >= 2:
                w = _HARAKAT.sub("", w)
        return w

    @staticmethod
    def _cells(words: list[Word], line_h: float, scale: float) -> list[dict]:
        """The line's table cells: the pieces `_order_segments` separates, split at the
        same wide gap. A line of ordinary prose has none, and says so with an empty list."""
        cuts = [i for i in range(1, len(words))
                if max(words[i - 1].box[0] - words[i].box[2],
                       words[i].box[0] - words[i - 1].box[2]) > 1.2 * line_h]
        if not cuts:
            return []
        out = []
        for a, b in zip([0] + cuts, cuts + [len(words)]):
            seg = words[a:b]
            out.append({"text": " ".join(w.text for w in seg),
                        "bbox": [int(round(min(w.box[0] for w in seg) / scale)),
                                 int(round(min(w.box[1] for w in seg) / scale)),
                                 int(round(max(w.box[2] for w in seg) / scale)),
                                 int(round(max(w.box[3] for w in seg) / scale))]})
        return out

    def _assemble(self, lines: list[Line], scale: float) -> tuple[list[LineOut], list[NumberOut]]:
        # paragraph breaks: a vertical gap clearly larger than the usual line pitch
        centers = [(L.bbox[1] + L.bbox[3]) / 2 for L in lines]
        gaps = [b - a for a, b in zip(centers, centers[1:]) if b - a > 0]
        pitch = float(np.median(gaps)) if gaps else 0.0
        para, out, numbers = 0, [], []
        for i, L in enumerate(lines):
            if i and pitch and centers[i] - centers[i - 1] > 1.6 * pitch:
                para += 1
            latin = L.source == "line-eng"
            words = [Word(self._clean_word(w.text, latin), w.conf, w.box) for w in L.words]
            words = [w for w in words if w.text]
            if not words:
                continue
            bbox = [int(round(v / scale)) for v in L.bbox]
            conf = sum(min(w.conf, 100.0) for w in words) / len(words) / 100.0
            out.append(LineOut(" ".join(w.text for w in words), bbox, round(conf, 3), L.source, para,
                               self._cells(words, L.bbox[3] - L.bbox[1], scale)))
            for w in words:
                if _NUMTOK.search(w.text) and len(_NUMTOK.findall(w.text)) >= 2:
                    glyph = w.conf > 100.0
                    numbers.append(NumberOut(
                        value=w.text, value_ascii=w.text.translate(_ASCII_DIGITS),
                        bbox=[int(round(v / scale)) for v in w.box],
                        source="glyph" if glyph else "tesseract",
                        confidence=round((w.conf - 101.0) if glyph else w.conf / 100.0, 3),
                        grouping=amount_grouping(w.text)))
        return out, numbers

    @staticmethod
    def _join(lines: list[LineOut]) -> str:
        parts, prev = [], None
        for L in lines:
            if prev is not None and L.paragraph != prev:
                parts.append("")
            parts.append(L.text)
            prev = L.paragraph
        return "\n".join(parts)
