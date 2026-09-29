# -*- coding: utf-8 -*-
r"""v2 page-level digit reader: CNN glyph classifier WITH a reject class, numpy inference.

Replaces the v1 HOG-SVM (`digit_reader.py`, kept for provenance) whose page locator
had 10% precision on the dev pages (E15): it had no way to say "this is a letter".
Here every connected component on the page is classified into
{۰-۹, 0-9, / - , ., <reject>} with a small CNN trained on synthetic text lines
(`digit_cnn_data.py`); a number is a run of confidently-classified digit glyphs on one
text line, and Persian letter fragments simply fall into the reject class.

No torch at inference: the exported weights (`models/digit_cnn.npz`) are run with a
numpy forward pass so the service venv needs nothing new.

Same pre-processing as training (`binarize`, `normalize_glyph`, `line_context`,
`scalars` are imported from the data module so the two can never drift).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .digit_cnn_data import CLASSES, G, PERSIAN, LATIN, REJECT, binarize, line_context, normalize_glyph, scalars

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "digit_cnn.npz"
SEP_CHARS = {20: "/", 21: "-", 22: ",", 23: "."}


# --- numpy CNN --------------------------------------------------------------------

def _conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x (N,C,H,W), w (O,C,3,3) -> (N,O,H,W), padding 1."""
    N, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(2, 3))     # N,C,H,W,3,3
    out = np.tensordot(win, w, axes=([1, 4, 5], [1, 2, 3]))                     # N,H,W,O
    return np.transpose(out, (0, 3, 1, 2)) + b[None, :, None, None]


def _pool2(x: np.ndarray) -> np.ndarray:
    N, C, H, W = x.shape
    return x.reshape(N, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))


class NumpyCNN:
    def __init__(self, path: Path = MODEL_PATH):
        z = np.load(path, allow_pickle=False)
        self.p = {k: z[k] for k in z.files if k != "classes"}
        self.classes = [str(c) for c in z["classes"]]
        self.version = "glyph-cnn-v2"

    def __call__(self, X: np.ndarray, S: np.ndarray, batch: int = 512) -> np.ndarray:
        """X (N,32,32) uint8, S (N,3) -> softmax probabilities (N, n_classes)."""
        outs = []
        p = self.p
        for i in range(0, len(X), batch):
            x = X[i:i + batch].astype(np.float32)[:, None] / 255.0
            x = _pool2(np.maximum(_conv3x3(x, p["c1.weight"], p["c1.bias"]), 0))
            x = _pool2(np.maximum(_conv3x3(x, p["c2.weight"], p["c2.bias"]), 0))
            x = _pool2(np.maximum(_conv3x3(x, p["c3.weight"], p["c3.bias"]), 0))
            h = np.concatenate([x.reshape(len(x), -1), S[i:i + batch].astype(np.float32)], 1)
            h = np.maximum(h @ p["f1.weight"].T + p["f1.bias"], 0)
            logits = h @ p["f2.weight"].T + p["f2.bias"]
            logits -= logits.max(1, keepdims=True)
            e = np.exp(logits)
            outs.append(e / e.sum(1, keepdims=True))
        return np.concatenate(outs) if outs else np.zeros((0, len(self.classes)), np.float32)


# --- page reader ------------------------------------------------------------------

@dataclass
class Glyph:
    box: tuple[int, int, int, int]
    idx: int                 # label index in the CC map
    cls: int = REJECT
    prob: float = 0.0
    probs: np.ndarray | None = None
    stacked: bool = False    # part of a vertical dot pair (a colon, a two-dot letter mark)
    value: int = -1          # digit value 0-9 pooled over scripts
    vprob: float = 0.0
    script: str = "persian"


@dataclass
class NumberRead:
    text: str                                   # visual left-to-right, separators kept
    box: tuple[int, int, int, int]
    confidence: float                           # min digit probability in the run
    mean_prob: float
    n_digits: int
    line_h: float
    script: str                                 # persian | latin | mixed
    parts: list = field(default_factory=list)   # [(Glyph, kind, value)] in visual order


class GlyphReaderV2:
    """`page_words()` returns every number found on the page as a `Word`
    (same contract as the Tesseract/v1 readers used by `NumericValidator`);
    `read_page()` returns the richer `NumberRead` list."""

    def __init__(self, model_path: Path = MODEL_PATH, p_digit: float = 0.5, p_run: float = 0.60,
                 min_digits: int = 3, p_weak: float = 0.35, min_glyph_h: int = 8,
                 seed_height_ratio: float = 0.62, extend_gap: float = 0.75,
                 extend_zero_p: float = 0.40, extend_digit_p: float = 0.50,
                 pitch_lo: float = 0.55, pitch_hi: float = 1.6):
        if not model_path.is_file():
            raise FileNotFoundError(f"{model_path} — build with digit_cnn_data + digit_cnn_train")
        self.net = NumpyCNN(model_path)
        self.version = self.net.version
        self.p_digit = p_digit          # a glyph counts as a digit above this probability
        self.p_run = p_run              # a run is kept if its mean digit probability is above this
        self.min_digits = min_digits
        self.p_weak = p_weak            # a digit glyph below this splits a run
        self.min_glyph_h = min_glyph_h  # lines whose glyphs are smaller than this are noise
        self.seed_height_ratio = seed_height_ratio   # a glyph this tall (of the page text height) may seed a line
        self.extend_gap = extend_gap                 # run extension: max gap, in digit heights
        self.extend_zero_p = extend_zero_p           # min P(zero) to absorb a small blob as `0`
        self.extend_digit_p = extend_digit_p         # min digit-value probability to absorb a full-size glyph
        self.pitch_lo = pitch_lo                     # absorbed glyph must continue the run's own pitch
        self.pitch_hi = pitch_hi

    def available(self) -> bool:
        return True

    # -- components and lines ----------------------------------------------------
    @staticmethod
    def _components(b: np.ndarray) -> tuple[np.ndarray, list[Glyph]]:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        H, W = b.shape
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 3 or h > 0.05 * H or w > 0.12 * W or (h < 3 and w < 3 * h):
                continue                              # (thin, wide 2-px blobs are dashes: kept)
            if h >= 0.02 * H and w < 3:            # rules / page edges
                continue
            out.append(Glyph((int(x), int(y), int(x + w), int(y + h)), i))
        return lab, out

    def _lines(self, glyphs: list[Glyph]) -> list[list[Glyph]]:
        """Two-pass line clustering. Tall glyphs seed the lines (a line is a band of
        vertically overlapping tall members; an outsized blob may join but never
        widens the band); small blobs — dots, dashes, colon halves — are then attached
        to the line whose band contains them, so a colon's two dots land together."""
        if not glyphs:
            return []
        hs = np.array([g.box[3] - g.box[1] for g in glyphs], np.float32)
        # Typical TEXT height, robust to the many small fragments a scan produces
        # (letter dots, speckle): the median of the upper half of the distribution.
        # A plain 75th percentile over every component came out at 19 px on a page
        # whose digits are 40 px, which made a 12 px dot-zero tall enough to seed a
        # line of its own — and the trailing zero of an amount was lost with it.
        upper = hs[hs >= float(np.median(hs))]
        h_ref = float(np.median(upper)) if len(upper) else float(np.median(hs))
        # Only a genuinely full-height glyph may SEED a line. A dot-zero is ~0.3-0.45
        # of digit height and sat just above the old 0.45 threshold, so it seeded a
        # line of its own and the trailing zero of `۳۲۱/۰۰۰/۰۰۰` was lost with it.
        # Everything shorter is placed by horizontal adjacency below.
        seed_h = max(6.0, self.seed_height_ratio * h_ref)
        tall = [g for g in glyphs if (g.box[3] - g.box[1]) >= seed_h]
        small = [g for g in glyphs if (g.box[3] - g.box[1]) < seed_h]
        if not tall:                                  # a line of nothing but small marks
            tall, small = glyphs, []
        tall.sort(key=lambda g: (g.box[1] + g.box[3]) / 2)
        lines: list[dict] = []
        for g in tall:
            y0, y1 = g.box[1], g.box[3]
            h = y1 - y0
            best, best_ov = None, 0.0
            # lines are created in cy order; scan back until bands are clearly above
            for ln in reversed(lines):
                if ln["y1"] < y0 - 4 * h and ln["y0"] < y0 - 6 * h:
                    break
                ov = min(y1, ln["y1"]) - max(y0, ln["y0"])
                if ov > 0 and ov >= 0.5 * min(h, ln["y1"] - ln["y0"]):
                    if ov > best_ov:
                        best, best_ov = ln, ov
            if best is None:
                lines.append({"y0": y0, "y1": y1, "items": [g], "hs": [h]})
            else:
                best["items"].append(g); best["hs"].append(h)
                # outsized blobs must not widen an established line; while the line is
                # young its median height is not yet meaningful (a colon dot may be first)
                if len(best["hs"]) < 6 or h <= 2.0 * float(np.median(best["hs"])):
                    best["y0"] = min(best["y0"], y0); best["y1"] = max(best["y1"], y1)
        # Left to right, so that a zero already attached extends the line for the next
        # one: the zeros of `/۰۰۰` chain onto their own number instead of each being
        # measured against the gap left by the slash.
        for g in sorted(small, key=lambda g: g.box[0]):
            cy = (g.box[1] + g.box[3]) / 2
            cand = []
            for ln in lines:
                band = ln["y1"] - ln["y0"]
                if ln["y0"] - 0.35 * band <= cy <= ln["y1"] + 0.35 * band:
                    cand.append(ln)
            if not cand:
                continue
            # A dot-zero belongs to the number it sits next to, so choose by HORIZONTAL
            # adjacency first and vertical distance only as a tie-break. Choosing by cy
            # alone put the last zero of `۳۲۱/۰۰۰/۰۰۰` on a neighbouring line and
            # deleted it from the amount.
            def key(ln):
                dx = min((max(m.box[0] - g.box[2], g.box[0] - m.box[2], 0) for m in ln["items"]), default=1e9)
                dcy = abs(cy - (ln["y0"] + ln["y1"]) / 2)
                return (dx, dcy)
            min(cand, key=key)["items"].append(g)
        out = []
        for ln in lines:
            med = float(np.median(ln["hs"]))
            if ln["y1"] - ln["y0"] > 1.7 * med and len(ln["items"]) >= 6:
                out.extend(GlyphReaderV2._split_by_adjacency(ln["items"]))
            else:
                out.append(ln["items"])
        return out

    @staticmethod
    def _split_by_adjacency(items: list[Glyph]) -> list[list[Glyph]]:
        """A cluster whose band is much taller than its glyphs has swallowed a
        neighbouring text line (page skew or curvature drifts a line across the
        band). Re-split it with LOCAL links only: two glyphs belong together when they
        are horizontal neighbours that overlap vertically."""
        items = sorted(items, key=lambda g: g.box[0])
        parent = list(range(len(items)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a

        for i, a in enumerate(items):
            ha = a.box[3] - a.box[1]
            for j in range(i + 1, min(len(items), i + 25)):
                b = items[j]
                hb = b.box[3] - b.box[1]
                if b.box[0] - a.box[2] > 1.5 * max(ha, hb):
                    break
                ov = min(a.box[3], b.box[3]) - max(a.box[1], b.box[1])
                if ov >= 0.4 * min(ha, hb):
                    parent[find(i)] = find(j)
        groups: dict[int, list[Glyph]] = {}
        for i, g in enumerate(items):
            groups.setdefault(find(i), []).append(g)
        return list(groups.values())

    # -- classification ----------------------------------------------------------
    def _classify(self, lab: np.ndarray, lines: list[list[Glyph]]) -> None:
        X, S, refs = [], [], []
        for ln in lines:
            ref_h, cy = line_context([g.box for g in ln])
            for g in ln:
                x0, y0, x1, y1 = g.box
                X.append(normalize_glyph((lab[y0:y1, x0:x1] == g.idx).astype(np.uint8) * 255))
                S.append(scalars(g.box, ref_h, cy)); refs.append(g)
        if not X:
            return
        P = self.net(np.stack(X), np.stack(S))
        for g, p in zip(refs, P):
            g.cls = int(p.argmax()); g.prob = float(p[g.cls]); g.probs = p
            # digit VALUE probability pools both scripts: the Latin `0` and Persian `۰`
            # classes describe the same value, so an email's `2006` should not lose
            # confidence to the script split
            pv = p[:10] + p[10:20]
            g.value = int(pv.argmax()); g.vprob = float(pv[g.value])
            g.script = "persian" if p[g.value] >= p[10 + g.value] else "latin"

    # -- runs ---------------------------------------------------------------------
    @staticmethod
    def _mark_stacked(ln: list[Glyph], ref_h: float) -> None:
        """Two small blobs sharing an x-range and separated vertically are a colon (or
        a letter's dot pair), never a dot-zero."""
        small = [g for g in ln if (g.box[3] - g.box[1]) < 0.6 * ref_h and (g.box[2] - g.box[0]) < 0.7 * ref_h]
        for i, a in enumerate(small):
            for b in small[i + 1:]:
                ov = min(a.box[2], b.box[2]) - max(a.box[0], b.box[0])
                vgap = max(b.box[1] - a.box[3], a.box[1] - b.box[3])
                if ov > 0.5 * min(a.box[2] - a.box[0], b.box[2] - b.box[0]) and -2 <= vgap <= 0.8 * ref_h:
                    a.stacked = b.stacked = True

    def _runs(self, ln: list[Glyph]) -> list[NumberRead]:
        ln = sorted(ln, key=lambda g: g.box[0])
        ref_h, line_cy = line_context([g.box for g in ln])
        out: list[NumberRead] = []
        if ref_h < self.min_glyph_h:                 # specks along a page edge, not text
            return out
        self._mark_stacked(ln, ref_h)

        def digit_mass(g: Glyph) -> float:
            return float(g.probs[:20].sum()) if g.probs is not None else 0.0

        def is_small(g: Glyph, dh: float) -> bool:
            h, w = g.box[3] - g.box[1], g.box[2] - g.box[0]
            return h < 0.5 * dh and w < 0.6 * dh

        def mid_height(g: Glyph, dh: float) -> bool:
            return abs((g.box[1] + g.box[3]) / 2 - line_cy) < 0.35 * dh

        # 1. label every glyph: digit value, separator, dot-zero candidate, or reject
        kind: list[tuple[str, int]] = []           # ("d", value) | ("s", cls) | ("z", 0) | ("r", -1)
        for g in ln:
            if g.stacked or g.probs is None:
                kind.append(("r", -1)); continue
            h = g.box[3] - g.box[1]
            if digit_mass(g) >= self.p_digit:
                # a blob well under digit height cannot be any digit but the dot-zero
                if h < 0.5 * ref_h and mid_height(g, ref_h):
                    kind.append(("z", 0))
                else:
                    kind.append(("d", g.value))
            elif 20 <= g.cls < 24 and g.prob >= 0.4:
                kind.append(("s", g.cls))
            elif is_small(g, ref_h) and mid_height(g, ref_h) and g.probs[0] > 0.10:
                kind.append(("z", 0))
            else:
                kind.append(("r", -1))

        # 2. runs of digits: a run may absorb separators and dot-zeros between/beside
        #    digits; anything else, or a wide gap, ends it
        i, n = 0, len(ln)
        while i < n:
            if kind[i][0] not in ("d", "z"):
                i += 1; continue
            run = [i]
            j = i + 1
            bridged_idx: set[int] = set()
            while j < n:
                gap = ln[j].box[0] - max(ln[k].box[2] for k in run)
                dh = float(np.median([ln[k].box[3] - ln[k].box[1] for k in run if kind[k][0] == "d"] or [ref_h]))
                if gap > 0.75 * dh:
                    break
                if kind[j][0] == "r":
                    # one smudged digit inside a long number: bridge it if it is digit-sized,
                    # has some digit probability, and a digit follows immediately
                    gj = ln[j]; hj, wj = gj.box[3] - gj.box[1], gj.box[2] - gj.box[0]
                    nxt_ok = j + 1 < n and kind[j + 1][0] == "d" and ln[j + 1].box[0] - gj.box[2] <= 0.75 * dh
                    strong_so_far = [ln[k].vprob for k in run if kind[k][0] == "d"]
                    if (not bridged_idx and len(strong_so_far) >= 3 and float(np.mean(strong_so_far)) >= 0.75
                            and nxt_ok and ln[j + 1].vprob >= 0.6 and not gj.stacked
                            and 0.6 * dh <= hj <= 1.3 * dh and wj <= 1.2 * dh and digit_mass(gj) >= 0.25):
                        kind[j] = ("d", gj.value); bridged_idx.add(j)
                    else:
                        break
                run.append(j); j += 1
            # split the candidate at weak glyphs (letters that carry some digit
            # probability drag a run's mean down; the clean digits beside them must
            # survive), then keep every strong segment on its own
            segments, cur = [], []
            for k in run:
                weak = kind[k][0] == "d" and ln[k].vprob < self.p_weak and k not in bridged_idx
                if weak:
                    if cur:
                        segments.append(cur)
                    cur = []
                else:
                    cur.append(k)
            if cur:
                segments.append(cur)
            for seg in segments:
                rd = self._finish(seg, kind, ln, ref_h, line_cy)
                if rd is not None:
                    out.append(rd)
            i = max(j, i + 1)
        return out

    def _finish(self, run: list[int], kind: list, ln: list[Glyph], ref_h: float, line_cy: float) -> NumberRead | None:
        def mid_height(g: Glyph, dh: float) -> bool:
            return abs((g.box[1] + g.box[3]) / 2 - line_cy) < 0.35 * dh

        real = [k for k in run if kind[k][0] == "d"]
        if not real:
            return None

        def zero_p(k: int) -> float:
            return float(ln[k].probs[0] + ln[k].probs[10])

        def digit_h(r: list[int]) -> float:
            hs = [ln[k].box[3] - ln[k].box[1] for k in r if kind[k][0] == "d"]
            return float(np.median(hs)) if hs else ref_h

        def spacing(r: list[int]) -> float:
            """The run's own inter-glyph gap, ignoring the widest one (a separator, or
            the suspect end itself): a number is set at a regular pitch."""
            gs = sorted(float(ln[b].box[0] - ln[a].box[2]) for a, b in zip(r, r[1:]))
            return float(np.median(gs[:-1] or gs)) if gs else 0.0

        # re-judge zeros against the run's own digit height: the line reference is
        # inflated by letter ascenders, so a bold dot-zero can pass as a full digit.
        # Only where the classifier gives zero real weight, or a comma made «۸۲۲۸» into
        # «۸۲۲۸۰» and a letter's dot made «۶۰۱» into «۶۰۱۰» (D72).
        dh = digit_h(run)
        for k in run:
            if (kind[k][0] == "d" and (ln[k].box[3] - ln[k].box[1]) < 0.68 * dh
                    and mid_height(ln[k], dh) and zero_p(k) >= 0.25):
                kind[k] = ("z", 0)

        # ends: a digit, a separator, or a dot-zero the classifier believes in, standing
        # at the run's own spacing. Stray specks, letter dots and commas are none of those.
        while len(run) > 1:
            lim = max(0.35 * digit_h(run), 2.2 * spacing(run))
            for end, nb in ((0, 1), (-1, -2)):
                k, o = run[end], run[nb]
                if (kind[k][0] == "s" or (kind[k][0] == "z" and zero_p(k) < 0.4)
                        or ln[max(k, o)].box[0] - ln[min(k, o)].box[2] > lim):
                    run.pop(end)
                    break
            else:
                break

        digits = [k for k in run if kind[k][0] in ("d", "z")]
        real = [k for k in run if kind[k][0] == "d"]
        if len(digits) < self.min_digits or not real:
            return None
        probs = [ln[k].vprob if kind[k][0] == "d" else max(float(ln[k].probs[0] + ln[k].probs[10]), 0.5) for k in digits]
        if float(np.mean(probs)) < self.p_run:
            return None
        chars = [CLASSES[kind[k][1]] if kind[k][0] in ("d", "z") else SEP_CHARS[kind[k][1]] for k in run]
        scripts = {ln[k].script for k in real}
        dh = float(np.median([ln[k].box[3] - ln[k].box[1] for k in real]))
        x0 = min(ln[k].box[0] for k in run); y0 = min(ln[k].box[1] for k in run)
        x1 = max(ln[k].box[2] for k in run); y1 = max(ln[k].box[3] for k in run)
        parts = [(ln[k], kind[k][0], kind[k][1]) for k in run]
        return NumberRead("".join(chars), (x0, y0, x1, y1), float(min(probs)), float(np.mean(probs)),
                          len(digits), dh, scripts.pop() if len(scripts) == 1 else "mixed", parts)

    # -- public --------------------------------------------------------------------
    def _extend(self, reads: list[NumberRead], glyphs: list[Glyph]) -> list[NumberRead]:
        """Absorb digit glyphs that sit against a number but were clustered onto a
        different text line.

        Line clustering uses page-wide height statistics, and on a noisy scan those
        are unreliable for a dot-zero: the final `۰` of `۳۲۱/۰۰۰/۰۰۰` was landing on a
        neighbouring line and the amount came back as `۳۲۱/۰۰۰/۰۰` — a financial
        amount silently divided by ten. Adjacency to the number is local evidence and
        does not depend on any page statistic, so it decides here.
        """
        used = {id(g) for r in reads for g, _, _ in r.parts}
        free = [g for g in glyphs if id(g) not in used and not g.stacked and g.probs is not None]
        if not free:
            return reads
        free.sort(key=lambda g: g.box[0])
        for r in reads:
            if not r.parts:
                continue
            dh = r.line_h or float(np.median([g.box[3] - g.box[1] for g, k, _ in r.parts if k == "d"] or [10]))
            grew = True
            while grew:
                grew = False
                digits = [g for g, k, _ in r.parts if k in ("d", "z")]
                cy = float(np.median([(g.box[1] + g.box[3]) / 2 for g in digits]))
                left, right = r.parts[0][0].box[0], r.parts[-1][0].box[2]
                # A number is set at a regular pitch. The next glyph of THIS number
                # continues it; a dot belonging to a nearby Persian word does not.
                lefts = [g.box[0] for g, _, _ in r.parts]
                steps = [b - a for a, b in zip(lefts, lefts[1:]) if b > a]
                pitch = float(np.median(steps)) if steps else dh
                for g in free:
                    if id(g) in used:
                        continue
                    gh, gw = g.box[3] - g.box[1], g.box[2] - g.box[0]
                    gcy = (g.box[1] + g.box[3]) / 2
                    if gh > 1.3 * dh or gw > 1.3 * dh or abs(gcy - cy) > 0.45 * dh:
                        continue
                    gap_r, gap_l = g.box[0] - right, left - g.box[2]
                    side = "r" if 0 <= gap_r <= self.extend_gap * dh else ("l" if 0 <= gap_l <= self.extend_gap * dh else None)
                    if side is None:
                        continue
                    step = (g.box[0] - lefts[-1]) if side == "r" else (lefts[0] - g.box[0])
                    if not (self.pitch_lo * pitch <= step <= self.pitch_hi * pitch):
                        continue
                    pv = g.probs[:10] + g.probs[10:20]
                    val, vprob = int(pv.argmax()), float(pv[pv.argmax()])
                    is_small = gh < 0.68 * dh
                    if is_small:                       # only a zero is drawn this small
                        if float(g.probs[0] + g.probs[10]) < self.extend_zero_p:
                            continue
                        # ...but so is half a colon. A dot with a same-sized dot above
                        # or below it, at this run's scale, is punctuation, not a zero.
                        if self._has_vertical_twin(g, glyphs, dh):
                            continue
                        val, vprob, kind = 0, float(g.probs[0] + g.probs[10]), "z"
                    elif float(g.probs[:20].sum()) >= self.p_digit and vprob >= self.extend_digit_p:
                        kind = "d"
                    else:
                        continue
                    part = (g, kind, val)
                    r.parts = ([part] + r.parts) if side == "l" else (r.parts + [part])
                    used.add(id(g)); grew = True
                    break
            self._rebuild(r)
        return reads

    @staticmethod
    def _has_vertical_twin(g: Glyph, others: list[Glyph], dh: float) -> bool:
        gh, gw = g.box[3] - g.box[1], g.box[2] - g.box[0]
        for o in others:
            if o is g:
                continue
            oh, ow = o.box[3] - o.box[1], o.box[2] - o.box[0]
            if oh > 1.6 * dh or max(gh, oh) > 2.0 * min(gh, oh):
                continue
            ov = min(g.box[2], o.box[2]) - max(g.box[0], o.box[0])
            if ov <= 0.5 * min(gw, ow):
                continue
            vgap = max(o.box[1] - g.box[3], g.box[1] - o.box[3])
            if -2 <= vgap <= 0.9 * dh:
                return True
        return False

    @staticmethod
    def _rebuild(r: NumberRead) -> None:
        chars = [CLASSES[v] if k in ("d", "z") else SEP_CHARS[v] for _, k, v in r.parts]
        boxes = [g.box for g, _, _ in r.parts]
        digits = [(g, k, v) for g, k, v in r.parts if k in ("d", "z")]
        probs = [float(g.probs[v]) if k == "d" else max(float(g.probs[0] + g.probs[10]), 0.5) for g, k, v in digits]
        r.text = "".join(chars)
        r.box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
        r.n_digits = len(digits)
        if probs:
            r.confidence, r.mean_prob = float(min(probs)), float(np.mean(probs))

    def read_page(self, im: Image.Image) -> list[NumberRead]:
        gray = np.asarray(im.convert("L"))
        b = binarize(gray)
        lab, glyphs = self._components(b)
        lines = self._lines(glyphs)
        self._classify(lab, lines)
        for ln in lines:
            self._mark_stacked(ln, line_context([g.box for g in ln])[0])
        reads: list[NumberRead] = []
        for ln in lines:
            reads.extend(self._runs(ln))
        return self._extend(reads, glyphs)

    def page_words(self, im: Image.Image) -> list:
        from .numeric_validator import Word
        return [Word(r.text, r.box, 100.0 * r.confidence) for r in self.read_page(im)]

    def read_crop(self, im: Image.Image, box: tuple[int, int, int, int], pad: float = 0.35) -> str:
        x0, y0, x1, y1 = box
        h = max(1, y1 - y0)
        px, py = int(h * pad), int(h * pad * 0.6)
        crop = im.crop((max(0, x0 - px), max(0, y0 - py), min(im.width, x1 + px), min(im.height, y1 + py)))
        reads = self.read_page(crop)
        if not reads:
            return ""
        return max(reads, key=lambda r: r.n_digits).text
