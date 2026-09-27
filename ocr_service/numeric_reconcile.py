# -*- coding: utf-8 -*-
r"""Reconcile the VLM's numbers with the v2 glyph reader's page reads (E15).

Why: the VLMs generate Persian-Indic digits instead of reading them (D53); on the
dev split 74% of their numbers are wrong, but they are wrong in a *useful* way — for
48% of the ground-truth numbers the model emits a near-miss in the right field, and
omits only 14%. The glyph reader (`digit_reader_v2`) reads 79% of the page's numbers
exactly. So the job here is alignment, not reading: every digit run the VLM wrote is
matched to a classical read, and a confident classical read replaces it.

Units are ATOMS — maximal digit runs — because both the VLM and the human GT
segment hyphenated lists inconsistently (D57), and because a phone list is really
three numbers.

Decision per VLM atom (`decide`):
    confirmed   a classical atom has exactly these digits
    corrected   the best-aligned classical atom differs, is confident, and the
                alignment is unambiguous -> its digits replace the VLM's
    conflict    aligned but the classical read is not confident enough to override
    unverified  nothing on the page aligns with it

Then `inject`: confident classical atoms that no VLM atom consumed, long enough to
be distinctive, located in the footer band, are added to `contact_info` — the field
whose content (a printed letterhead footer) the VLM omits most.

Pure: takes fields + reads, returns patched fields + per-number records. Nothing
here touches a model or an image.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from .numeric_validator import FIELDS, PERSIAN, classify, digits_only, to_ascii_digits

_ATOM_RE = re.compile(r"[0-9۰-۹٠-٩]+")
_SPAN_RE = re.compile(r"(?:IR)?[0-9۰-۹٠-٩](?:[0-9۰-۹٠-٩,./\-،٫٬]*[0-9۰-۹٠-٩])?")
_ASCII_TO_PERSIAN = str.maketrans("0123456789", PERSIAN)


@dataclass
class ClassicalAtom:
    digits: str                 # ASCII
    conf: float                 # min glyph probability of the containing read
    mean: float
    box: tuple[int, int, int, int]
    y_rel: float                # vertical centre / page height
    script: str
    used: int = 0


@dataclass
class AtomDecision:
    field: str
    vlm: str                    # ASCII digits the VLM wrote
    resolved: str               # ASCII digits to use
    status: str                 # confirmed | corrected | conflict | unverified
    classical: str | None = None
    similarity: float = 0.0
    conf: float = 0.0
    box: tuple[int, int, int, int] | None = None


@dataclass
class NumberRecord:
    field: str
    kind: str
    value: str                  # as the VLM printed it (or the classical read, if added)
    value_ascii: str
    resolved: str               # value with corrected atoms substituted
    confidence: str             # high | corrected | low | unverified | added
    classical_value: str | None = None
    candidates: list[str] = field(default_factory=list)
    bbox: tuple[int, int, int, int] | None = None
    similarity: float | None = None
    note: str | None = None
    atoms: list[AtomDecision] = field(default_factory=list)


@dataclass
class Policy:
    align_long: float = 0.5         # min similarity to align an atom of >= long_digits
    align_short: float = 0.75       # min similarity for shorter atoms
    long_digits: int = 7
    margin_short: float = 0.1       # short atoms: best must beat the runner-up by this
    override_conf: float = 0.35     # classical min-glyph probability needed to override (dev sweep, E15)
    override_mean: float = 0.60
    inject: bool = True
    inject_min_digits: int = 7      # single, ungrouped number: digits needed to inject
    inject_conf: float = 0.55
    inject_mean: float = 0.78
    footer_band: float = 0.80       # y_rel above this is the footer
    inject_text: bool = True        # also append injected numbers to contact_info text
    align_span: float = 0.6         # whole-number similarity to align a multi-part number
    inject_body: bool = True        # unmatched confident page numbers in the body band -> records only
    body_band: tuple[float, float] = (0.18, 0.80)
    # Dev sweep 2026-09-22: 3 beats 4 on both recall (76.7% vs 75.3%) and precision
    # (75.0% vs 74.7%) — the excluded chunks were the groups of large amounts.
    inject_body_min_digits: int = 3     # single, ungrouped number in the body band
    # A GROUPED number (`۳۲۱/۰۰۰/۰۰۰`, a phone list) is injected as ONE unit with its
    # separators exactly as read, and qualifies on its TOTAL digit count. Judging its
    # chunks separately discarded whole amounts whose groups are 3 digits long —
    # every trailing `/۰۰۰` of a large sum.
    inject_group_min_digits: int = 6
    inject_group_min_parts: int = 2


def classical_atoms(reads, page_h: int) -> list[ClassicalAtom]:
    out = []
    for r in reads:
        y_rel = ((r.box[1] + r.box[3]) / 2) / max(page_h, 1)
        for m in _ATOM_RE.finditer(r.text):
            d = to_ascii_digits(m.group(0))
            if len(d) >= 3:
                out.append(ClassicalAtom(d, r.confidence, r.mean_prob, r.box, y_rel, r.script))
    return out


def _consumed(a: "ClassicalAtom", flat: list["ClassicalAtom"]) -> bool:
    return any(f.used and f.digits == a.digits and f.box == a.box for f in flat)


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def _render(digits: str, like: str) -> str:
    """Write ASCII digits in the script the VLM used for `like`."""
    return digits.translate(_ASCII_TO_PERSIAN) if any(c in PERSIAN or "٠" <= c <= "٩" for c in like) else digits


def decide(field_name: str, vlm_atom: str, atoms: list[ClassicalAtom], pol: Policy) -> AtomDecision:
    target = to_ascii_digits(vlm_atom)
    scored = sorted(((_sim(target, a.digits), a) for a in atoms), key=lambda t: -t[0])
    if not scored:
        return AtomDecision(field_name, target, target, "unverified")
    best_sim, best = scored[0]
    if best_sim == 1.0:
        best.used += 1
        return AtomDecision(field_name, target, target, "confirmed", best.digits, 1.0, best.conf, best.box)
    thr = pol.align_long if len(target) >= pol.long_digits else pol.align_short
    if best_sim < thr:
        return AtomDecision(field_name, target, target, "unverified", best.digits, best_sim)
    if len(target) < pol.long_digits:
        runner = scored[1][0] if len(scored) > 1 and scored[1][1].digits != best.digits else 0.0
        if best_sim - runner < pol.margin_short:
            return AtomDecision(field_name, target, target, "conflict", best.digits, best_sim, best.conf, best.box)
    if best.conf >= pol.override_conf and best.mean >= pol.override_mean:
        best.used += 1
        return AtomDecision(field_name, target, best.digits, "corrected", best.digits, best_sim, best.conf, best.box)
    return AtomDecision(field_name, target, target, "conflict", best.digits, best_sim, best.conf, best.box)


@dataclass
class ClassicalSpan:
    digits: str                 # ASCII digits with separators, e.g. 2/588/808/588
    atoms: list[ClassicalAtom]


_KEEP = re.compile(r"[^0-9/,.\-]")


def classical_spans(reads, page_h: int) -> list[ClassicalSpan]:
    """Whole reads with their atoms, for aligning a multi-part VLM number as a unit."""
    out = []
    for r in reads:
        y_rel = ((r.box[1] + r.box[3]) / 2) / max(page_h, 1)
        ats = [ClassicalAtom(to_ascii_digits(m.group(0)), r.confidence, r.mean_prob, r.box, y_rel, r.script)
               for m in _ATOM_RE.finditer(r.text)]
        if len(ats) >= 2:
            out.append(ClassicalSpan(_KEEP.sub("", to_ascii_digits(r.text)), ats))
    return out


def align_span(field_name: str, span: str, vlm_atoms: list[str], spans: list[ClassicalSpan],
               atoms: list[ClassicalAtom], pol: Policy) -> list[AtomDecision] | None:
    """A multi-part VLM number (date, amount, reference) is aligned to a whole
    classical read first: when the two agree on structure (same number of parts,
    similar digits overall) each part is corrected by position, so a short part
    such as 587 -> 588 no longer needs to pass the per-atom threshold on its own."""
    if len(vlm_atoms) < 2:
        return None
    target = _KEEP.sub("", to_ascii_digits(span))
    best, best_sim, best_atoms = None, 0.0, None
    for cs in spans:
        if len(cs.atoms) != len(vlm_atoms):
            continue
        # The page is right-to-left: a multi-group number is often laid out with its
        # groups in the opposite order to the logical value (`۱۴۰۳/۰۸/۰۳` is printed
        # `۰۳/۰۸/۱۴۰۳`). The reader reports what it sees, so try both orders and keep
        # the model's ordering with the reader's digits.
        for atoms_try in (cs.atoms, list(reversed(cs.atoms))):
            sim = _sim(target, "/".join(a.digits for a in atoms_try))
            if sim > best_sim:
                best, best_sim, best_atoms = cs, sim, atoms_try
    if best is None or best_sim < pol.align_span:
        return None
    out = []
    for v, ca in zip(vlm_atoms, best_atoms):
        v_ascii = to_ascii_digits(v)
        if ca.digits == v_ascii:
            out.append(AtomDecision(field_name, v_ascii, v_ascii, "confirmed", ca.digits, 1.0, ca.conf, ca.box))
        elif ca.conf >= pol.override_conf and ca.mean >= pol.override_mean:
            out.append(AtomDecision(field_name, v_ascii, ca.digits, "corrected", ca.digits, round(best_sim, 3), ca.conf, ca.box))
        else:
            out.append(AtomDecision(field_name, v_ascii, v_ascii, "conflict", ca.digits, round(best_sim, 3), ca.conf, ca.box))
    # mark the same atoms in the flat index as consumed so they are not injected
    for ca in best.atoms:
        for a in atoms:
            if a.digits == ca.digits and a.box == ca.box:
                a.used += 1
    return out


def reconcile(fields: dict[str, str | None], reads, page_h: int, pol: Policy | None = None,
              min_digits: int = 3) -> tuple[dict[str, str | None], list[NumberRecord]]:
    pol = pol or Policy()
    atoms = classical_atoms(reads, page_h)
    spans_cl = classical_spans(reads, page_h)
    out_fields = dict(fields)
    records: list[NumberRecord] = []
    for f in FIELDS:
        text = fields.get(f)
        if not isinstance(text, str) or not text:
            continue
        pieces, pos = [], 0
        for m in _SPAN_RE.finditer(text):
            span = m.group(0)
            if len(digits_only(span)) < min_digits:
                continue
            decisions, new_span, p = [], [], 0
            parts = list(_ATOM_RE.finditer(span))
            pre = align_span(f, span, [am.group(0) for am in parts], spans_cl, atoms, pol)
            for i, am in enumerate(parts):
                if pre is not None:
                    d = pre[i]
                elif len(am.group(0)) < min_digits:
                    continue
                else:
                    d = decide(f, am.group(0), atoms, pol)
                decisions.append(d)
                new_span.append(span[p:am.start()])
                new_span.append(_render(d.resolved, am.group(0)) if d.status == "corrected" else am.group(0))
                p = am.end()
            new_span.append(span[p:])
            resolved = "".join(new_span)
            statuses = {d.status for d in decisions}
            if statuses <= {"confirmed"}:
                conf = "high"
            elif "conflict" in statuses:
                conf = "low"
            elif "corrected" in statuses:
                conf = "corrected"
            else:
                conf = "unverified"
            classical = "-".join(d.classical for d in decisions if d.classical) or None
            cands = [span]
            if resolved != span:
                cands.append(resolved)
            for d in decisions:
                if d.status == "conflict" and d.classical:
                    cands.append(_render(d.classical, span))
            boxes = [d.box for d in decisions if d.box]
            bbox = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)) if boxes else None
            records.append(NumberRecord(f, classify(span), span, to_ascii_digits(span), resolved, conf, classical,
                                        list(dict.fromkeys(cands)), bbox,
                                        round(min((d.similarity for d in decisions), default=0.0), 3),
                                        None if conf in ("high", "corrected") else
                                        ("vlm/classical reader disagree" if conf == "low" else "no digit region found on page"),
                                        decisions))
            pieces.append(text[pos:m.start()]); pieces.append(resolved); pos = m.end()
        pieces.append(text[pos:])
        out_fields[f] = "".join(pieces)

    if pol.inject:
        added = []
        for a in atoms:
            if a.used or len(a.digits) < pol.inject_min_digits or a.conf < pol.inject_conf or a.mean < pol.inject_mean:
                continue
            if any(a.digits == r.value_ascii or a.digits in r.value_ascii for r in records):
                continue
            if any(a.digits == x.digits for x in added):
                continue
            if a.y_rel >= pol.footer_band:
                added.append(a)
        # --- grouped numbers first, as whole units -------------------------------
        group_added: list[tuple[str, ClassicalSpan, float]] = []
        for cs in spans_cl:
            if len(cs.atoms) < pol.inject_group_min_parts:
                continue
            # `decide()` marks consumption on the FLAT atom index, not on these
            # span-local copies, so ask the flat index — otherwise a phone list whose
            # parts the model already carries gets appended a second time and every
            # duplicate counts against precision.
            if any(_consumed(a, atoms) for a in cs.atoms):
                continue
            total = sum(len(a.digits) for a in cs.atoms)
            if total < pol.inject_group_min_digits:
                continue
            a0 = cs.atoms[0]
            if a0.conf < pol.inject_conf or a0.mean < pol.inject_mean:
                continue
            if any(cs.digits == r.value_ascii or _KEEP.sub("", r.value_ascii) == cs.digits for r in records):
                continue
            if any(cs.digits == g[0] for g in group_added):
                continue
            field = "contact_info" if a0.y_rel >= pol.footer_band else "body_text"
            if field == "body_text" and not pol.inject_body:
                continue
            group_added.append((cs.digits, cs, a0.y_rel))
            for a in cs.atoms:
                a.used += 1
                for b in atoms:
                    if b.digits == a.digits and b.box == a.box:
                        b.used += 1

        body_added = []
        if pol.inject_body:
            for a in atoms:
                if a.used or len(a.digits) < pol.inject_body_min_digits or a.conf < pol.inject_conf or a.mean < pol.inject_mean:
                    continue
                if not (pol.body_band[0] <= a.y_rel < pol.body_band[1]):
                    continue
                if any(a.digits == r.value_ascii or a.digits in r.value_ascii for r in records):
                    continue
                if any(a.digits == x.digits for x in body_added):
                    continue
                body_added.append(a)
        def _as_read(a: ClassicalAtom) -> str:
            # injected numbers carry the script the page printed them in
            return a.digits if a.script == "latin" else a.digits.translate(_ASCII_TO_PERSIAN)

        for a in added:
            v = _as_read(a)
            records.append(NumberRecord("contact_info", classify(a.digits), v, a.digits, v, "added", a.digits, [v],
                                        a.box, None, "read from the page footer; not in the model output"))
        for a in body_added:
            v = _as_read(a)
            records.append(NumberRecord("body_text", classify(a.digits), v, a.digits, v, "added", a.digits, [v],
                                        a.box, None, "read from the page body; not in the model output"))
        for digits, cs, y_rel in group_added:
            a0 = cs.atoms[0]
            field = "contact_info" if y_rel >= pol.footer_band else "body_text"
            v = digits if a0.script == "latin" else digits.translate(_ASCII_TO_PERSIAN)
            records.append(NumberRecord(field, classify(digits), v, digits, v, "added", digits, [v],
                                        a0.box, None, "grouped number read from the page; not in the model output"))
            if pol.inject_text and field == "contact_info":
                base = out_fields.get("contact_info") or ""
                out_fields["contact_info"] = (base + "\n" + v).strip() if base else v
        if added and pol.inject_text:
            extra = " ".join(_as_read(a) for a in added)
            base = out_fields.get("contact_info") or ""
            out_fields["contact_info"] = (base + "\n" + extra).strip() if base else extra
    return out_fields, records


def summarize(records: list[NumberRecord]) -> dict:
    n = len(records)
    c = {k: sum(r.confidence == k for r in records) for k in ("high", "corrected", "low", "unverified", "added")}
    return {"n_numeric": n, **c, "conflict_rate": (c["low"] / n) if n else 0.0}
