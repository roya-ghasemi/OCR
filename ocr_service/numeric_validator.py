# -*- coding: utf-8 -*-
"""Numeric validation layer — Tesseract cross-checks every number the VLM produced.

Why this exists (measured, D53): the VLMs in use read Latin digits 12/12 and
Persian-Indic digits 0/12 on clean cards; the numbers they emit are generated, not
read. Tesseract's `fas` model is a conventional glyph classifier and has no such
blind spot on printed digits. So every numeric span in the VLM output is located on
the page, re-read by Tesseract from a crop, and compared.

Decoupled from the VLM: input is `(image, fields)`, output is a list of
`NumericField`. Nothing here imports a backend.

Locating the region. The VLMs return no bounding boxes on this stack, so the crop is
found the other way round: one sparse-text Tesseract pass over the page yields every
word box; digit-bearing boxes are matched to each VLM number by digit-string
similarity; the best box (padded) is re-read with a digit whitelist. If the page
pass finds no digit boxes at all, a coarse position heuristic per field (header /
body / footer band) is tried. Assumption: printed administrative letters; no
handwriting.

Confidence contract (Section 2.3 of the brief):
    high        Tesseract's digits == VLM's digits
    low         both read something, they differ -> `candidates` carries both
    unverified  Tesseract unavailable, or found no number to compare against
"""
from __future__ import annotations

import difflib
import logging
import re
import shutil
from dataclasses import dataclass, field
from typing import Iterable

from PIL import Image

log = logging.getLogger(__name__)

FIELDS = ["sender", "receiver", "subject", "body_text", "contact_info"]
PERSIAN = "۰۱۲۳۴۵۶۷۸۹"
ARABIC = "٠١٢٣٤٥٦٧٨٩"
TO_ASCII = str.maketrans(PERSIAN + ARABIC, "0123456789" * 2)
SEPARATORS = ",./-،٫٬ "
# A numeric span: digits of any script with internal separators; ≥ N digits total.
_SPAN_RE = re.compile(r"(?:IR)?[0-9۰-۹٠-٩](?:[0-9۰-۹٠-٩,./\-،٫٬]*[0-9۰-۹٠-٩])?")
WHITELIST = "0123456789" + PERSIAN + ARABIC + ",./-"


def to_ascii_digits(s: str) -> str:
    return (s or "").translate(TO_ASCII)


def digits_only(s: str) -> str:
    return re.sub(r"\D", "", to_ascii_digits(s))


def classify(value: str) -> str:
    a = to_ascii_digits(value).upper()
    d = digits_only(a)
    if a.startswith("IR") and len(d) == 24:
        return "iban"
    if re.fullmatch(r"1[34]\d{2}[/\-.]\d{1,2}[/\-.]\d{1,2}", a):
        return "date"
    if len(d) == 11 and d.startswith("09"):
        return "mobile"
    if len(d) == 10 and d.startswith("9") and not any(c in a for c in ",/"):
        return "mobile"                                   # 9xxxxxxxxx without the leading 0
    if len(d) == 10 and not any(c in a for c in ",/"):
        return "national_id"                              # 09xx… is a valid national-ID prefix
    if any(c in a for c in ",/") and len(d) >= 6:
        return "amount" if a.endswith(("000", "00")) or "," in a else "reference"
    if len(d) >= 6:
        return "reference"
    return "other"


@dataclass
class NumericSpan:
    field: str
    value: str            # as the VLM printed it
    value_ascii: str      # digits + separators, ASCII digits
    kind: str
    start: int
    end: int


@dataclass
class NumericField:
    field: str
    kind: str
    value: str
    value_ascii: str
    confidence: str                       # high | low | unverified
    tesseract_value: str | None = None
    candidates: list[str] = field(default_factory=list)
    bbox: tuple[int, int, int, int] | None = None
    similarity: float | None = None
    note: str | None = None
    resolved: str | None = None            # the value a downstream consumer should use

    def as_dict(self) -> dict:
        return {"field": self.field, "kind": self.kind, "value": self.value,
                "value_ascii": self.value_ascii, "confidence": self.confidence,
                "tesseract_value": self.tesseract_value, "candidates": self.candidates,
                "bbox": list(self.bbox) if self.bbox else None,
                "similarity": self.similarity, "note": self.note, "resolved": self.resolved}


def resolve(vlm: str, classical: str | None, kind: str) -> str:
    """Which reading wins when they differ.

    Measured on the dev split (2026-09-17): a "plausible-looking" override rule
    (same digit count, >=60% agreement) fixed 4 numbers and broke 20. A classical
    reading therefore overrides the VLM ONLY when it is self-verifying: the mod-11
    national-ID checksum or the IBAN mod-97 passes on it and fails on the VLM's.
    Otherwise the VLM value stands and the conflict stays flagged (`low`)."""
    if not classical:
        return vlm
    a, b = digits_only(vlm), digits_only(classical)
    if kind == "national_id" and len(b) == 10 and _nid_ok(b) and not _nid_ok(a):
        return classical
    if kind == "iban" and _iban_ok(to_ascii_digits(classical)) and not _iban_ok(to_ascii_digits(vlm)):
        return classical
    return vlm


def _nid_ok(d: str) -> bool:
    if len(d) != 10 or not d.isdigit() or len(set(d)) == 1:
        return False
    s = sum(int(x) * (10 - i) for i, x in enumerate(d[:9])) % 11
    return (s < 2 and int(d[9]) == s) or (s >= 2 and int(d[9]) == 11 - s)


def _iban_ok(v: str) -> bool:
    s = re.sub(r"[\s-]", "", v or "").upper()
    if not re.fullmatch(r"IR\d{24}", s):
        return False
    num = "".join(str(ord(c) - 55) if c.isalpha() else c for c in s[4:] + s[:4])
    return int(num) % 97 == 1


def find_numeric_spans(fields: dict[str, str | None], min_digits: int = 3) -> list[NumericSpan]:
    spans: list[NumericSpan] = []
    for f in FIELDS:
        text = fields.get(f)
        if not isinstance(text, str):
            continue
        for m in _SPAN_RE.finditer(text):
            v = m.group(0)
            if len(digits_only(v)) < min_digits:
                continue
            spans.append(NumericSpan(f, v, to_ascii_digits(v), classify(v), m.start(), m.end()))
    return spans


# ---------------------------------------------------------------------------
# Tesseract
# ---------------------------------------------------------------------------

@dataclass
class Word:
    text: str
    box: tuple[int, int, int, int]
    conf: float

    @property
    def digits(self) -> str:
        return digits_only(self.text)


class TesseractReader:
    """Thin wrapper; `available()` is the only thing the pipeline needs before use."""

    def __init__(self, cmd: str | None = None, langs: str = "fas+eng", tessdata_dir: str | None = None):
        self.langs = langs
        self.tessdata_dir = tessdata_dir
        self._pt = None
        self._version: str | None = None
        try:
            import pytesseract
            if cmd:
                pytesseract.pytesseract.tesseract_cmd = cmd
            self._pt = pytesseract
        except ImportError:
            self._pt = None
        self._cmd = cmd

    def available(self) -> bool:
        if self._pt is None:
            return False
        if self._version is not None:
            return self._version != ""
        try:
            self._version = str(self._pt.get_tesseract_version())
        except Exception:
            self._version = ""
            exe = self._cmd or shutil.which("tesseract")
            log.warning("Tesseract not available (cmd=%s). Numeric fields will be 'unverified'. "
                        "Install: https://github.com/UB-Mannheim/tesseract/wiki + fas.traineddata, "
                        "then OCRS_TESSERACT_CMD=<path to tesseract.exe>.", exe)
        return self._version != ""

    @property
    def version(self) -> str | None:
        return self._version or None

    def _cfg(self, extra: str) -> str:
        # pytesseract keeps quotes literally on Windows, so the data dir goes in the
        # environment (TESSDATA_PREFIX, honoured by tesseract) instead of the config string.
        if self.tessdata_dir:
            import os
            os.environ["TESSDATA_PREFIX"] = self.tessdata_dir
        return extra

    def page_words(self, im: Image.Image) -> list[Word]:
        """One sparse-text pass over the page: every word box with its text."""
        # Assumption: --psm 11 (sparse text, no layout assumptions) finds numbers in
        # letterhead tables and footers that block-based modes merge or drop.
        d = self._pt.image_to_data(im, lang=self.langs, config=self._cfg("--psm 11"),
                                   output_type=self._pt.Output.DICT)
        out = []
        for i, t in enumerate(d["text"]):
            t = (t or "").strip()
            if not t:
                continue
            try:
                conf = float(d["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            out.append(Word(t, (d["left"][i], d["top"][i], d["left"][i] + d["width"][i], d["top"][i] + d["height"][i]), conf))
        return out

    def read_crop(self, im: Image.Image, box: tuple[int, int, int, int], pad: float = 0.35) -> str:
        x0, y0, x1, y1 = box
        h = max(1, y1 - y0)
        px, py = int(h * pad * 2), int(h * pad)
        crop = im.crop((max(0, x0 - px), max(0, y0 - py), min(im.width, x1 + px), min(im.height, y1 + py)))
        if crop.height < 48:                       # upscale small crops; LSTM wants ~30px x-height
            s = 48 / crop.height
            crop = crop.resize((int(crop.width * s), 48), Image.LANCZOS)
        cfg = self._cfg(f"--psm 7 -c tessedit_char_whitelist={WHITELIST}")
        return self._pt.image_to_string(crop, lang=self.langs, config=cfg).strip()


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------

def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _is_sep(w: Word) -> bool:
    return bool(w.text) and all(c in SEPARATORS for c in w.text)


def _merge_adjacent(words: list[Word], max_gap: int = 40, max_run: int = 8) -> list[Word]:
    """Numbers split by separators come back as several boxes on one line; also
    offer merged neighbours so `1403 / 4 / 524` can match `1403/4/524`. Input is
    every digit-bearing or separator-only box; output is the input plus merges."""
    merged: list[Word] = []
    by_line = sorted(words, key=lambda w: (w.box[1] // 20, w.box[0]))
    for i, w in enumerate(by_line):
        if _is_sep(w):
            continue
        cur = w
        for nxt in by_line[i + 1:i + max_run]:
            same_line = abs(nxt.box[1] - cur.box[1]) < max(cur.box[3] - cur.box[1], 12)
            close = 0 <= nxt.box[0] - cur.box[2] <= max_gap or 0 <= cur.box[0] - nxt.box[2] <= max_gap
            if not (same_line and close) or not (nxt.digits or _is_sep(nxt)):
                break
            cur = Word(cur.text + nxt.text,
                       (min(cur.box[0], nxt.box[0]), min(cur.box[1], nxt.box[1]),
                        max(cur.box[2], nxt.box[2]), max(cur.box[3], nxt.box[3])),
                       min(cur.conf, nxt.conf))
            if nxt.digits:
                merged.append(cur)
    return words + merged


_BANDS = {  # Assumption: fallback bands by relative page position (top,bottom fractions)
    "sender": (0.0, 0.30), "receiver": (0.10, 0.45), "subject": (0.15, 0.50),
    "body_text": (0.25, 0.85), "contact_info": (0.75, 1.0),
}


class NumericValidator:
    def __init__(self, reader=None, min_digits: int = 3, match_threshold: float = 1.0,
                 candidate_floor: float = 0.4, extra_reader=None):
        self.reader = reader or TesseractReader()
        self.extra = extra_reader            # optional second opinion, adds candidates only
        self.min_digits = min_digits
        self.match_threshold = match_threshold
        self.candidate_floor = candidate_floor

    def validate(self, image: Image.Image, fields: dict[str, str | None]) -> list[NumericField]:
        spans = find_numeric_spans(fields, self.min_digits)
        if not spans:
            return []
        if not self.reader.available():
            return [NumericField(s.field, s.kind, s.value, s.value_ascii, "unverified",
                                 candidates=[s.value], note="classical reader unavailable", resolved=s.value)
                    for s in spans]

        im = image.convert("RGB")
        try:
            words = [w for w in self.reader.page_words(im) if w.digits or _is_sep(w)]
        except Exception as exc:
            log.exception("tesseract page pass failed")
            return [NumericField(s.field, s.kind, s.value, s.value_ascii, "unverified",
                                 candidates=[s.value], note=f"reader error: {exc}", resolved=s.value)
                    for s in spans]
        digit_words = [w for w in _merge_adjacent(words) if len(w.digits) >= self.min_digits]

        out: list[NumericField] = []
        for s in spans:
            target = digits_only(s.value)
            best, best_sim = None, 0.0
            for w in digit_words:
                sim = _similarity(target, w.digits)
                if sim > best_sim:
                    best, best_sim = w, sim
            if best is None or best_sim < self.candidate_floor:
                cand = self._band_fallback(im, s)
                if cand is None:
                    out.append(NumericField(s.field, s.kind, s.value, s.value_ascii, "unverified",
                                            candidates=[s.value], note="no digit region found on page",
                                            resolved=s.value))
                    continue
                best, best_sim = cand, _similarity(target, cand.digits)
            try:
                tess = self.reader.read_crop(im, best.box)
            except Exception as exc:
                tess = ""
                log.warning("crop re-read failed: %s", exc)
            tess_digits = digits_only(tess) or best.digits
            tess_display = to_ascii_digits(tess).strip() or to_ascii_digits(best.text)
            sim = _similarity(target, tess_digits)
            extra_display = None
            if self.extra is not None and getattr(self.extra, "available", lambda: False)():
                try:
                    extra_display = to_ascii_digits(self.extra.read_crop(im, best.box)).strip() or None
                except Exception:
                    extra_display = None
            if tess_digits and sim >= self.match_threshold:
                out.append(NumericField(s.field, s.kind, s.value, s.value_ascii, "high",
                                        tesseract_value=tess_display, candidates=[s.value],
                                        bbox=best.box, similarity=round(sim, 3), resolved=s.value))
            else:
                cands = [c for c in (s.value, tess_display, extra_display) if c]
                out.append(NumericField(s.field, s.kind, s.value, s.value_ascii, "low",
                                        tesseract_value=tess_display,
                                        candidates=list(dict.fromkeys(cands)),
                                        bbox=best.box, similarity=round(sim, 3),
                                        note="vlm/classical reader disagree",
                                        resolved=resolve(s.value, tess_display, s.kind)))
        return out

    def _band_fallback(self, im: Image.Image, s: NumericSpan) -> Word | None:
        top, bot = _BANDS.get(s.field, (0.0, 1.0))
        band = im.crop((0, int(im.height * top), im.width, int(im.height * bot)))
        try:
            words = [w for w in self.reader.page_words(band) if len(w.digits) >= self.min_digits]
        except Exception:
            return None
        if not words:
            return None
        target = digits_only(s.value)
        best = max(words, key=lambda w: _similarity(target, w.digits))
        if _similarity(target, best.digits) < self.candidate_floor:
            return None
        x0, y0, x1, y1 = best.box
        off = int(im.height * top)
        return Word(best.text, (x0, y0 + off, x1, y1 + off), best.conf)


def summarize(results: Iterable[NumericField]) -> dict:
    rs = list(results)
    n = len(rs)
    return {"n_numeric": n,
            "high": sum(r.confidence == "high" for r in rs),
            "low": sum(r.confidence == "low" for r in rs),
            "unverified": sum(r.confidence == "unverified" for r in rs),
            "conflict_rate": (sum(r.confidence == "low" for r in rs) / n) if n else 0.0}
