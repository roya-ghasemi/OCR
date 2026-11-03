# -*- coding: utf-8 -*-
r"""Full-page transcription of ANY image: every line of printed text, in reading order.

    image -> [deskew] -> page binarisation + clean-up -> line candidates from three
    Tesseract layout passes + a connected-component line finder read line by line
    (Persian, English) -> best reading per physical line -> CNN digit reader
    replaces number tokens -> junk-line filter -> text + lines + numbers

CPU only; no language model anywhere in the path, so nothing can be invented: every
character in the output was recognised from pixels by Tesseract or the digit CNN.

Why this design (E19, ocr_eval/experiments.md), all on the 51-letter dev split vs the
previous vision-language-model letter-JSON path:

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

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# Image operations live in imaging.py; the names are re-exported here for callers that
# have always imported them from this module.
from .imaging import (_crop, _xov, _yov, clean_ink, flatten, glyph_height, is_line_crop,  # noqa: F401
                      is_screen_image, measure_glyph_height, screen_ink, squeeze_kashida, text_lines,
                      text_skew, working_scale)
from .config import repo_relative
from .tesseract import Tesseract

log = logging.getLogger(__name__)

# psm 7 (single line) is retried with psm 13 (raw line) below this mean confidence.
RETRY_RAW_LINE_CONF = 70
# Padding around a line crop before it is read; Tesseract misreads glyphs that touch the edge.
LINE_FRAME_PX = 30

_NUMTOK = re.compile(r"[0-9۰-۹٠-٩]")
_PERSIAN_DIGITS = str.maketrans("0123456789٠١٢٣٤٥٦٧٨٩", "۰۱۲۳۴۵۶۷۸۹۰۱۲۳۴۵۶۷۸۹")
_ASCII_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_KEY = re.compile(r"[ـ‌ً-ْ\s]")
_BIDI = re.compile("[‎‏‪-‮⁦-⁩]")
_HARAKAT = re.compile("[ً-ْ]")
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
_PERSIAN_WORD = re.compile(r"^[ء-ی‌]{2,}$")
# A Persian comma is a small mark sitting on the baseline; Tesseract reads it as one of
# these about as often as it reads it right.
_COMMA_LOOKALIKE = {"»", "«", "۰", "٠", "0", "+", ",", "ء", "٫"}
# The same mark glued to a word: «خدماتی»تولیدی»فنی» for «خدماتی،تولیدی،فنی». Digits and
# the full stop are NOT here — inside a token they belong to a number or a sentence.
_COMMA_IN_WORD = "»«,٫ء"
_PERSIAN_CHAR = re.compile(r"[ء-ی]")
# Printed labels of a letter's form fields. What follows one of these on the page is
# routinely handwritten, and a handwritten value must never be transcribed into the body.
_FORM_LABEL = re.compile(r"^(شماره|تاریخ|پیوست|رونوشت)\s*[:：]$")
# Confidence a token must reach to stand as printed text where a handwritten value is
# expected — after «شماره:» on the letter's own form. The body of these letters reads at
# 0.86-0.93, so a printed «پیوست: ندارد» clears this and a pen stroke does not.
PRINTED_AFTER_LABEL = 85.0
# A token mixing Persian-Indic and ASCII digits — «۱7», «۱/2». No real page does this; it
# is the recogniser reading a handwritten mark as whichever digit each stroke resembled.
_MIXED_DIGITS = re.compile(r"(?=.*[۰-۹])(?=.*[0-9])")
# Confidence under which a token that is not a known word is taken as unread rather than
# read. Body text on the corpus runs 0.86-0.93; the handwritten header runs 0.49-0.68.
UNREAD_CONF = 70.0


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
    conf: float                                   # 0..100, Tesseract's word confidence
    box: tuple[int, int, int, int]                # working-image coordinates
    digit_prob: float | None = None               # set when the digit reader replaced the text


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
    glyph_px: float = 0.0                         # text height in the image as supplied
    timing: dict = field(default_factory=dict)
    prep: dict = field(default_factory=dict)      # what page_prep did (crop / warp)
    unread: list = field(default_factory=list)    # regions nobody read: handwriting, marks
    prepared: Image.Image | None = None           # the page as it was read

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ transcriber
class Transcriber:
    def __init__(self, tesseract_cmd: str | None, tessdata_dir: str | None, digit_model: Path | None,
                 layout_psms: tuple[str, ...] = ("3", "4", "6"), deskew_min_deg: float = 1.0,
                 min_line_conf: float = 30.0, workers: int = 8, number_min_prob: float = 0.6,
                 spellfix: bool = True, crop_page: bool = True, ink_separate: bool = True,
                 lang_fa: str = "fas_custom", text_psms: tuple[str, ...] = ("11",)):
        self.cmd, self.tessdata = tesseract_cmd, tessdata_dir
        self.tess = Tesseract(tesseract_cmd, tessdata_dir)
        self.layout_psms = tuple(layout_psms)
        self.text_psms = tuple(text_psms)
        self.crop_page = crop_page
        self.ink_separate = ink_separate
        # Which Persian model reads the page. `fas` is the stock one; a fine-tune
        # (E29) ships as `fas_custom` and is selected here, so switching models is a
        # setting and not an edit.
        self.lang_fa = lang_fa
        # Both normalisations below are right for Persian and destructive for Arabic: ي and
        # ك are the correct letters there, not misspellings of ی and ک, and a vocalised page
        # carries harakat on nearly every word rather than inventing them. Reading Arabic
        # with the Persian folding on produces «الولمه» for «الْعِلْمِ».
        self.arabic_script = lang_fa.startswith("ara")
        self.deskew_min_deg = deskew_min_deg
        self.min_line_conf = min_line_conf
        self.workers = workers
        self.number_min_prob = number_min_prob
        self.spellfix = spellfix
        self.digit_model = digit_model
        self._glyph = None
        self._spellfix = None
        self._glyph_error: str | None = None
        self.version: str | None = None

    # -- availability ------------------------------------------------------------
    def spellfix_stage(self):
        """The dot-confusion repair (spellfix.py), loaded once."""
        if self._spellfix is None:
            from .spellfix import SpellFix
            self._spellfix = SpellFix()
        return self._spellfix

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
        version, langs = self.tess.probe()
        ok = self.lang_fa in langs
        self.version = version
        g = self.glyph_reader()
        sp = self.spellfix_stage() if self.spellfix else None
        return {"tesseract": {"available": ok, "version": version, "langs": langs, "cmd": self.cmd,
                              "lang_fa": self.lang_fa,
                              "tessdata_dir": repo_relative(self.tessdata)},
                "digit_reader": {"available": g is not None, "version": getattr(g, "version", None),
                                 "error": self._glyph_error},
                "spellfix": {"enabled": bool(self.spellfix),
                             "available": bool(sp and sp.available()),
                             "words": len(sp.words) if sp else 0}}

    # -- tesseract ---------------------------------------------------------------
    def _layout(self, page: Image.Image, psm: str) -> list[Line]:
        groups: dict[tuple, list[Word]] = {}
        for r in self.tess.read(page, psm, self.lang_fa):
            groups.setdefault(r["key"], []).append(Word(r["text"], r["conf"], r["box"]))
        lines = []
        for ws in groups.values():
            bbox = (min(w.box[0] for w in ws), min(w.box[1] for w in ws),
                    max(w.box[2] for w in ws), max(w.box[3] for w in ws))
            lines.append(Line(ws, bbox, "psm" + psm))
        return lines

    def _read_batches(self, ex: ThreadPoolExecutor, jobs: list[tuple[list[Image.Image], str, str]]):
        """Read several (images, psm, lang) jobs, each split into one batch per worker.
        Returns futures that resolve to the per-image rows of each job, in order."""
        def chunks(n: int) -> list[range]:
            k = max(1, min(self.workers, n))
            return [range(i * n // k, (i + 1) * n // k) for i in range(k)]

        pending = []
        for imgs, psm, lang in jobs:
            parts = [(r, ex.submit(self.tess.read_many, [imgs[i] for i in r], psm, lang))
                     for r in chunks(len(imgs)) if len(r)]
            pending.append((len(imgs), parts))

        def collect(n, parts):
            out: list[list[dict]] = [[] for _ in range(n)]
            for r, f in parts:
                for i, rows in zip(r, f.result()):
                    out[i] = rows
            return out
        return [collect(n, parts) for n, parts in pending]

    def _read_lines(self, ex: ThreadPoolExecutor, crops: list[Image.Image],
                    langs: tuple[str, ...]) -> dict[str, list[list[Word]]]:
        """Each crop read as one line in each language: psm 7 (single line), and psm 13
        (raw line) where 7 came back empty or unsure. psm 7 returns nothing for some clean
        lines that 13 reads perfectly. Word boxes are in crop coordinates."""
        squeezed = [squeeze_kashida(c) for c in crops]
        framed = [ImageOps.expand(img, border=LINE_FRAME_PX, fill=255) for img, _ in squeezed]

        def words(rows):
            ws = [Word(r["text"], r["conf"], tuple(v - LINE_FRAME_PX for v in r["box"])) for r in rows]
            return ws, (sum(w.conf for w in ws) / len(ws) if ws else -1.0)

        first = dict(zip(langs, self._read_batches(ex, [(framed, "7", lang) for lang in langs])))
        best = {lang: [words(rows) for rows in first[lang]] for lang in langs}
        retry = {lang: [i for i, (_, m) in enumerate(best[lang]) if m < RETRY_RAW_LINE_CONF] for lang in langs}
        second = self._read_batches(ex, [([framed[i] for i in retry[lang]], "13", lang) for lang in langs])
        for lang, rows13 in zip(langs, second):
            for i, rows in zip(retry[lang], rows13):
                b = words(rows)
                if b[1] > best[lang][i][1]:
                    best[lang][i] = b

        out: dict[str, list[list[Word]]] = {}
        for lang in langs:
            out[lang] = []
            for (ws, _), (_, xmap) in zip(best[lang], squeezed):
                if xmap is not None:                    # back from squeezed to crop columns
                    last = len(xmap) - 1
                    ws = [Word(w.text, w.conf, (int(xmap[min(max(w.box[0], 0), last)]), w.box[1],
                                                int(xmap[min(max(w.box[2] - 1, 0), last)]) + 1, w.box[3]))
                          for w in ws]
                out[lang].append(ws)
        return out

    # -- selection -----------------------------------------------------------------
    @staticmethod
    def _score(line: Line) -> float:
        return sum(len(_key(w.text)) * (min(w.conf, 100.0) / 100.0) ** 2 for w in line.words if w.conf >= 40)

    def _line_ok(self, line: Line) -> bool:
        """Confidence only. A junk filter that also judged the line's mean token length
        deleted real text with the noise — the salutation «با سلام (», whose stray
        bracket pulls the mean under three, and a table cell holding «۱۰» at full
        confidence. Measured on dev: it cost coverage CER 14.65% → 14.72% and 1-3% of
        the output. Handwriting and stamps are surfaced by `confidence`, not deleted:
        losing a line is unrecoverable, marking one is not (D79)."""
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
        normal word spacing and is kept.

        Widening this to two characters was tried and reverted: the marks it was meant to
        catch — «سا» after «پیوست:», «سر» before «آدرس:» — are both words in Tesseract's
        own Persian list, so a lexicon test keeps them and a length test would also delete
        «از», «به», «در» and «را». They are removed instead by `_unread`, which knows they
        follow a printed form label."""
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
        boxes, _ = text_lines(ink)
        crops = [_crop(page, b) for b in boxes]
        offs = [(max(0, b[0] - 2), max(0, b[1] - 2)) for b in boxes]
        # The whole-page layout passes are the slowest single calls, so they start first and
        # run alongside the line batches in the same pool of `workers` processes.
        with ThreadPoolExecutor(max(self.workers, len(self.layout_psms) + len(self.text_psms))) as ex:
            layouts = [ex.submit(self._layout, page, p) for p in self.layout_psms]
            text_only = [ex.submit(self._layout, page, p) for p in self.text_psms]
            read = self._read_lines(ex, crops, (self.lang_fa, "eng"))
            cands = [L for f in layouts for L in f.result()]
            sparse = [L for f in text_only for L in f.result()]
        for b, (ox, oy), wf, we in zip(boxes, offs, read[self.lang_fa], read["eng"]):
            for ws, lang in ((wf, "fas"), (we, "eng")):   # tags, not model names
                if not ws:
                    continue
                if lang == "eng" and not any(_LATIN_WORD.search(w.text) and w.conf >= 70 for w in ws):
                    continue                            # an English read only competes on a Latin line
                moved = [Word(w.text, w.conf, (w.box[0] + ox, w.box[1] + oy, w.box[2] + ox, w.box[3] + oy)) for w in ws]
                cands.append(Line(moved, b, "line-" + lang))
        # psm 11 (sparse text) recovers words the other passes merge or drop, but splits number
        # groups apart. Its lines compete only in regions where no reading holds a digit, so
        # the numbers are decided exactly as without it (E38).
        def has_digit(L: Line) -> bool:
            return any(_NUMTOK.search(w.text) for w in L.words)
        numeric = [c for c in cands if has_digit(c)]
        return cands + [L for L in sparse
                        if not has_digit(L) and not any(self._conflict(L, c) for c in numeric)]

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
    def _drop_duplicates(pieces: list[Line]) -> list[Line]:
        """Two readings of the same words, kept because they barely overlap in x.

        `_conflict` divides the overlap by the *narrower* box, so two readings of one row
        that start at opposite ends — psm 6 taking «که پسمه تعالی» from x 1028 and psm 4
        «بسمه تعالی پیوست: سا» from x 83 — share only 17% of the wider one and are both
        selected. Joined, the row says «بسمه تعالی» twice. Geometry cannot separate these;
        the text can: pieces of one row that *read the same words* are one reading, and the
        better-scoring one is kept.
        """
        def lcs(a: str, b: str) -> int:
            if not a or not b:
                return 0
            prev = [0] * (len(b) + 1)
            best = 0
            for ca in a:
                cur = [0] * (len(b) + 1)
                for j, cb in enumerate(b, 1):
                    if ca == cb:
                        cur[j] = prev[j - 1] + 1
                        best = max(best, cur[j])
                prev = cur
            return best

        keys = [_key(p.text) for p in pieces]
        order = sorted(range(len(pieces)), key=lambda i: -pieces[i].score)
        kept: list[int] = []
        for i in order:
            dup = False
            for j in kept:
                short = min(len(keys[i]), len(keys[j]))
                if short >= 5 and lcs(keys[i], keys[j]) >= max(5, 0.5 * short):
                    dup = True
                    break
            if not dup:
                kept.append(i)
        return [pieces[i] for i in sorted(kept)]

    @staticmethod
    def _join_row(pieces: list[Line]) -> Line:
        """Pieces of one physical row (halves of a tilted line, table cells read as
        separate lines) become one line, right to left — or left to right when every
        piece is an English read."""
        if len(pieces) == 1:
            return pieces[0]
        pieces = Transcriber._drop_duplicates(pieces)
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
                L.words[i] = Word(pre + core.replace(",", "،") + suf, 100.0, w.box,
                                  digit_prob=min(r.mean_prob for r in rs))

    # -- main ---------------------------------------------------------------------------
    def transcribe_line_crop(self, im: Image.Image) -> Transcript:
        """One cropped line, read as one line at the size it arrived.

        The page path (crop, deskew, line finding, layout passes) is skipped: a 640x48
        strip has no frame, no measurable skew and one line. Sent through it, `working_scale`
        upsampled such strips 2.5-3.3x and psm 7 then returned nothing at all (D82). The
        fine-tune was trained on lines at native scale.
        """
        t0 = time.perf_counter()
        im = ImageOps.exif_transpose(im).convert("RGB")
        # Read without the white frame the page path adds around a line: this strip already
        # has its own margin, and on 9 px text inside 48 px the extra border made psm 7
        # find nothing.
        words = [Word(r["text"], r["conf"], r["box"])
                 for r in self.tess.read(im, "7", self.lang_fa) if r["text"].strip()]
        if not words:
            words = [Word(r["text"], r["conf"], r["box"])
                     for r in self.tess.read(im, "13", self.lang_fa) if r["text"].strip()]
        # `_assemble` does the cleaning and the spell-fix, keyed off the line's source, so
        # the words go in raw exactly as they do on the page path.
        line = Line(words, (0, 0, im.width, im.height), "psm7")
        self._trim_edges(line)
        out_lines, numbers, unread = self._assemble([line], 1.0)
        return Transcript(text=self._join(out_lines), lines=out_lines, numbers=numbers,
                          skew_deg=0.0, image_size=im.size,
                          glyph_px=round(measure_glyph_height(im), 1),
                          timing={"total_s": round(time.perf_counter() - t0, 3),
                                  "route": "line_crop"},
                          unread=unread, prepared=im, prep={"line_crop": True})

    def transcribe(self, im: Image.Image, screen: bool = False) -> Transcript:
        """Read a page. `screen=True` reads a digital image (screenshot, rendered page) the
        way a screen draws text: any colour on any colour, no sheet to cut out, no stamp."""
        t0 = time.perf_counter()
        timing: dict[str, float] = {}
        im = ImageOps.exif_transpose(im).convert("RGB")
        prep, ink_info, shown = None, None, None
        if screen:
            # The paper steps below erase blue text and drop white text on buttons, so the
            # page is read from a black-on-white rendering of its contrast instead (E39).
            # Boxes are the same in both, so the original is what the viewer shows.
            shown, im = im, screen_ink(im)
        else:
            # Cut the page out of the frame first. Everything that is not paper — desk,
            # shadow, the edge of the next sheet — becomes ink under `flatten`, and by the
            # time it reaches the line finder it is indistinguishable from faint text.
            if self.crop_page:
                from .page_prep import crop_page
                im, prep = crop_page(im)
            # Then split the ink layers: erase the blue stamp that sits on the signature
            # block, and turn a white-on-coloured letterhead the right way up. Both are
            # invisible to a threshold on brightness alone (ink.py).
            if self.ink_separate:
                from .ink import separate
                im, ink_info = separate(im)
        timing["prep_s"] = round(time.perf_counter() - t0, 3)

        skew = text_skew(im)
        if abs(skew) >= self.deskew_min_deg:
            im = im.rotate(skew, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
        else:
            skew = 0.0
        timing["deskew_s"] = round(time.perf_counter() - t0, 3)

        t = time.perf_counter()
        glyph_px = measure_glyph_height(im)
        scale = working_scale(im, glyph_px)
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
        out_lines, numbers, unread = self._assemble(lines, scale)
        text = self._join(out_lines)
        timing["total_s"] = round(time.perf_counter() - t0, 3)
        return Transcript(text=text, lines=out_lines, numbers=numbers, skew_deg=round(skew, 2),
                          image_size=im.size, glyph_px=round(glyph_px, 1), timing=timing, unread=unread,
                          prepared=shown or im,
                          prep={"screen": screen, **(prep.as_dict() if prep else {}),
                                **(ink_info.as_dict() if ink_info else {})})

    def _clean_word(self, w: str, latin: bool) -> str:
        w = _BIDI.sub("", w)
        if not latin and not self.arabic_script:
            w = w.replace("ي", "ی").replace("ك", "ک")
            # Tesseract invents harakat on bold letterhead-style text («بِمُدیرِیَت»);
            # printed Persian carries at most one per word («احتراماً», «بهارِ»).
            if len(_HARAKAT.findall(w)) >= 2:
                w = _HARAKAT.sub("", w)
        return w

    @staticmethod
    def _fix_marks(words: list[Word], cells: int) -> list[Word]:
        """Put back the Persian comma Tesseract keeps misreading.

        «احتراما ، پیرو» comes back «احتراما ۰ پیرو» and «می باشد ،» comes back
        «می باشد »» — 65 times across dev+test, the single commonest thing wrong with an
        otherwise correct line. A mark standing **alone between two Persian words** is
        that comma: a lone zero is not a word, and a closing guillemet with nothing to
        close is not a quotation. Inside a table row a lone «۰» can be a cell value, so a
        row with cells is left alone, and a mark at either end of the line is left alone
        because it has no two neighbours to stand between (D81).
        """
        if cells >= 2 or not words:
            return words
        # «...» is a Persian quotation. If this line opens one, its closing mark is real.
        quoted = any("«" in w.text for w in words)
        out = list(words)
        for i, w in enumerate(out):
            t = w.text
            # the mark standing alone, between two Persian words
            if (0 < i < len(out) - 1 and t in _COMMA_LOOKALIKE and not (t in "«»" and quoted)
                    and _PERSIAN_WORD.match(out[i - 1].text) and _PERSIAN_WORD.match(out[i + 1].text)):
                out[i] = replace(w, text="،")
                continue
            # the mark glued to a word, with a Persian letter on its left
            if not any(c in _COMMA_IN_WORD for c in t):
                continue
            ch = list(t)
            for j, c in enumerate(ch):
                if c not in _COMMA_IN_WORD or (c in "«»" and quoted):
                    continue
                if j == 0:
                    # «,طبقه»: a comma that drifted to the front of the next word. Only
                    # the unambiguous ones — a leading «»» could open nothing, a leading
                    # «ء» belongs to the word.
                    if c in ",٫" and len(ch) > 1 and _PERSIAN_CHAR.match(ch[1]):
                        ch[j] = "،"
                    continue
                if not _PERSIAN_CHAR.match(ch[j - 1]):
                    continue
                after = ch[j + 1] if j + 1 < len(ch) else ""
                # between two letters, or trailing the word — but «شیء» keeps its hamza
                if _PERSIAN_CHAR.match(after) or (after == "" and c in "»,"):
                    ch[j] = "،"
            if "".join(ch) != t:
                out[i] = replace(w, text="".join(ch))
        return out

    @staticmethod
    def _cut_positions(words: list[Word], line_h: float) -> list[int]:
        """Indices where the line has a gap wide enough to be a column boundary."""
        return [i for i in range(1, len(words))
                if max(words[i - 1].box[0] - words[i].box[2],
                       words[i].box[0] - words[i - 1].box[2]) > 1.2 * line_h]

    @staticmethod
    def _cells(words: list[Word], line_h: float, scale: float, cuts: list[int]) -> list[dict]:
        """The line's table cells, split at the column boundaries `cuts` names. A line of
        ordinary prose has none, and says so with an empty list."""
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

    def _unread(self, words: list[Word], floor: float) -> tuple[list[Word], list[dict]]:
        """Drop the tokens nobody actually read, and say where they were.

        Two kinds, and neither belongs in a transcript:

        * **A handwritten form value.** «شماره:» and «تاریخ:» are printed; what follows
          them is written by hand, and the recogniser answers with whichever letters each
          stroke resembled — «شماره: ۱7 یک», «تاریخ: امه ۱/2». The requirement is that a
          handwritten region must never reach the body text, so the printed label is kept
          and the value is reported as a region to look at.
        * **A mark that is not text at all.** A speck or a binder ring that survived as
          «زب». A run of tokens where nothing is a word and the reader was unsure of every
          one of them is not a reading; it is the absence of one.

        A token counts as read when it is a word in the list, a clean number, or Latin —
        so this can only ever remove something the recogniser itself was unsure of AND
        that matches nothing known. Where it removes something, the caller surfaces the
        region for review rather than silently deleting it.
        """
        lex = self.spellfix_stage().words if self.spellfix else set()

        def read(w: Word) -> bool:
            t = w.text.strip("()[]{}«».,،؛:;\"'")
            if not t:
                return False
            if _MIXED_DIGITS.search(t) and re.fullmatch(r"[0-9۰-۹٠-٩/.,:-]+", t):
                return False                      # «۱7»: two digit systems in one token
            if _NUMTOK.search(t):
                return len(_NUMTOK.findall(t)) >= 2 or w.conf >= floor
            if _LATIN_WORD.search(t):
                return True
            if t in lex or _key(t) in lex:
                return True
            return w.conf >= floor

        # A form label counts only with its colon and only at the head of the line. Without
        # both, the «شماره» in «به قرار داد منعقده به شماره ۲۹۰۶۳» is a form label and the
        # contract number after it is a handwritten value.
        label = next((i for i, w in enumerate(words) if _FORM_LABEL.match(w.text)), None)

        flags = [not read(w) for w in words]
        if label is not None:
            # What follows the label on a form line is written by hand unless it is plainly
            # printed: «شماره: یک» is a pen stroke the recogniser offered a word for.
            for i in range(label + 1, len(words)):
                if words[i].conf < PRINTED_AFTER_LABEL:
                    flags[i] = True
        if not any(flags):
            return words, []
        keep, regions = [], []
        i = 0
        while i < len(words):
            if not flags[i]:
                keep.append(words[i])
                i += 1
                continue
            j = i
            while j < len(words) and flags[j]:
                j += 1
            run = words[i:j]
            # after a printed form label the value is handwritten, however short the run
            handwritten = label is not None and i > label
            # Away from a form label, low confidence alone is not enough to delete text.
            # On a 12 px capture every word reads badly, and «برناسه‌ریزی انجام‌شده» is
            # damaged text, not noise — removing it cost that page 3.3 pp of CER. What a
            # speck or a binder ring produces is always SHORT: «زب», «سس», «۰». So a run is
            # only dropped as unreadable when no token in it is long enough to be a word.
            noise = all(len(_key(w.text)) <= 3 for w in run)
            if handwritten or (noise and (len(run) >= 2 or len(words) == len(run))):
                regions.append({"kind": "handwritten" if handwritten else "illegible",
                                "bbox": [min(w.box[0] for w in run), min(w.box[1] for w in run),
                                         max(w.box[2] for w in run), max(w.box[3] for w in run)],
                                "read_as": " ".join(w.text for w in run),
                                "confidence": round(min(w.conf for w in run) / 100.0, 3)})
            else:
                keep.extend(run)                  # a single stray token stays; it may be real
            i = j
        return keep, regions

    @staticmethod
    def _table_cuts(lines: list[Line], words_per_line: list[list[Word]]) -> list[list[int]]:
        """Which of each line's wide gaps are really column boundaries.

        A wide gap is evidence of a table only when the line above or below breaks in the
        *same place*. Justified Persian is stretched by widening the spaces, so a single
        line of prose has gaps as wide as any table's — which is why a per-line rule
        reported «۲ جدول» on a page that is one paragraph, and 3 more on the letter. A
        column, unlike a stretched space, is in the same position on the next row.
        """
        gaps: list[list[tuple[int, float]]] = []
        for L, ws in zip(lines, words_per_line):
            h = max(1.0, L.bbox[3] - L.bbox[1])
            gaps.append([(i, (ws[i - 1].box[0] + ws[i].box[2]) / 2)
                         for i in Transcriber._cut_positions(ws, h)])
        out: list[list[int]] = []
        for j, g in enumerate(gaps):
            if not g:
                out.append([])
                continue
            tol = 0.6 * max(1.0, lines[j].bbox[3] - lines[j].bbox[1])
            near = [x for k in (j - 1, j + 1) if 0 <= k < len(gaps) for _, x in gaps[k]]
            out.append([i for i, x in g if any(abs(x - y) <= tol for y in near)])
        return out

    def _assemble(self, lines: list[Line], scale: float) -> tuple[list[LineOut], list[NumberOut], list[dict]]:
        # paragraph breaks: a vertical gap clearly larger than the usual line pitch
        centers = [(L.bbox[1] + L.bbox[3]) / 2 for L in lines]
        gaps = [b - a for a, b in zip(centers, centers[1:]) if b - a > 0]
        pitch = float(np.median(gaps)) if gaps else 0.0
        # Clean every line first: whether a gap is a column can only be judged once the
        # neighbouring lines' words are known.
        cleaned: list[list[Word]] = []
        for L in lines:
            latin = L.source == "line-eng"
            words = [replace(w, text=self._clean_word(w.text, latin)) for w in L.words]
            if self.spellfix and not latin:
                sp = self.spellfix_stage()
                words = [replace(w, text=sp.fix(w.text, w.conf)) for w in words]
            cleaned.append([w for w in words if w.text])

        # Remove what was never read — a handwritten form value, a mark that is not text —
        # before the columns are judged, so a dropped mark cannot define one.
        # "Unread" is relative to how well THIS page reads, not an absolute score. A 12 px
        # capture reads every word at low confidence, so an absolute floor called correct
        # text unread and deleted 3% of that page (CER 7.4% -> 10.7%). The floor is the
        # page's own typical confidence, well below it, and never above UNREAD_CONF.
        confs = sorted(w.conf for ws in cleaned for w in ws)
        typical = confs[len(confs) // 2] if confs else 0.0
        floor = min(UNREAD_CONF, max(35.0, typical - 20.0))
        unread: list[dict] = []
        for i, ws in enumerate(cleaned):
            cleaned[i], regions = self._unread(ws, floor)
            unread += regions
        table_cuts = self._table_cuts(lines, cleaned)

        para, out, numbers = 0, [], []
        for i, L in enumerate(lines):
            if i and pitch and centers[i] - centers[i - 1] > 1.6 * pitch:
                para += 1
            words = cleaned[i]
            if not words:
                continue
            bbox = [int(round(v / scale)) for v in L.bbox]
            conf = sum(min(w.conf, 100.0) for w in words) / len(words) / 100.0
            cells = self._cells(words, L.bbox[3] - L.bbox[1], scale, table_cuts[i])
            words = self._fix_marks(words, len(cells))
            out.append(LineOut(" ".join(w.text for w in words), bbox, round(conf, 3), L.source, para,
                               cells))
            for w in words:
                if _NUMTOK.search(w.text) and len(_NUMTOK.findall(w.text)) >= 2:
                    glyph = w.digit_prob is not None
                    numbers.append(NumberOut(
                        value=w.text, value_ascii=w.text.translate(_ASCII_DIGITS),
                        bbox=[int(round(v / scale)) for v in w.box],
                        source="glyph" if glyph else "tesseract",
                        confidence=round(w.digit_prob if glyph else w.conf / 100.0, 3),
                        grouping=amount_grouping(w.text)))
        for r in unread:
            r["bbox"] = [int(round(v / scale)) for v in r["bbox"]]
        return out, numbers, unread

    @staticmethod
    def _join(lines: list[LineOut]) -> str:
        parts, prev = [], None
        for L in lines:
            if prev is not None and L.paragraph != prev:
                parts.append("")
            parts.append(L.text)
            prev = L.paragraph
        return "\n".join(parts)
