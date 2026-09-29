# -*- coding: utf-8 -*-
"""Scoring a full-page transcript against the five-field real-letter GT (E19).

coverage CER   For each non-null GT field, the minimum edit distance between the field
               and ANY substring of the transcript (semi-global alignment), summed and
               divided by the summed field lengths. Text the GT does not carry
               (signature blocks, form labels, handwriting) is not charged; text that is
               missing, misread or out of order is. Both sides are normalised with
               `normalize.normalize` (RULES_DEFAULT) first.
atom recall    Digit groups (thousands-grouped numbers kept whole, other separators
               split) of the GT fields found in the transcript, as a multiset.
number exact   Whole numbers (>= 2 digits) found exactly, separator type and group order
               ignored (bidi layout legitimately reverses date groups).
len ratio      transcript length / GT length - below 1 means content is missing.

A GT field whose text is contained in another field (e.g. a subject lifted out of the
body) is dropped from the number counts so its numbers are not counted twice.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from normalize import normalize

FIELDS = ["sender", "receiver", "subject", "body_text", "contact_info"]
_NUM = re.compile(r"\d+(?:[.,/:\-]\d+)*")


def semiglobal(needle: str, hay: str) -> int:
    """Minimum edit distance between `needle` and any substring of `hay`."""
    n = len(needle)
    if n == 0:
        return 0
    if not hay:
        return n
    t = np.frombuffer(hay.encode("utf-32-le"), dtype=np.uint32)
    m = len(t)
    idx = np.arange(m + 1, dtype=np.int64)
    prev = np.zeros(m + 1, dtype=np.int64)              # free start anywhere in hay
    for i, ch in enumerate(needle, 1):
        cost = (t != ord(ch)).astype(np.int64)
        cand = np.empty(m + 1, dtype=np.int64)
        cand[0] = i
        cand[1:] = np.minimum(prev[:-1] + cost, prev[1:] + 1)
        prev = np.minimum.accumulate(cand - idx) + idx  # insertions from hay, in one pass
    return int(prev.min())                              # free end


def atoms(text: str) -> list[str]:
    out = []
    for m in _NUM.finditer(text):
        tok = m.group(0)
        parts = tok.split(",")
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]) and all(p.isdigit() for p in parts):
            out.append("".join(parts))
            continue
        out += [p for p in re.split(r"[.,/:\-]", tok) if p]
    return out


def numbers(text: str) -> list[str]:
    out = []
    for m in _NUM.finditer(text):
        tok = m.group(0)
        if len(re.sub(r"\D", "", tok)) < 2:
            continue
        groups = re.split(r"[.,/:\-]", tok)
        out.append("|".join(sorted(groups)) if len(groups) > 1 else tok)
    return out


def _dedup(fields: dict) -> dict:
    norm = {f: normalize(fields.get(f)) for f in FIELDS if fields.get(f)}
    return {f: s for f, s in norm.items()
            if not any(g != f and len(s) < len(o) and s.replace(" ", "") in o.replace(" ", "")
                       for g, o in norm.items())}


def score_doc(transcript: str, gt: dict) -> dict:
    p = normalize(transcript)
    per = {}
    for f in FIELDS:
        g = normalize(gt.get(f))
        if g:
            per[f] = (semiglobal(g, p), len(g))
    keep = _dedup(gt)
    ga = Counter(a for s in keep.values() for a in atoms(s))
    pa = Counter(atoms(p))
    gn = Counter(n for s in keep.values() for n in numbers(s))
    pn = Counter(numbers(p))
    return {"err": sum(e for e, _ in per.values()), "chars": sum(n for _, n in per.values()), "per_field": per,
            "atoms_gt": sum(ga.values()), "atoms_hit": sum((ga & pa).values()),
            "nums_gt": sum(gn.values()), "nums_hit": sum((gn & pn).values()),
            "pred_len": len(p), "gt_len": sum(len(s) for s in keep.values())}


def aggregate(rows: list[dict]) -> dict:
    E = sum(r["err"] for r in rows)
    C = sum(r["chars"] for r in rows)
    pf = {}
    for f in FIELDS:
        e = sum(r["per_field"][f][0] for r in rows if f in r["per_field"])
        n = sum(r["per_field"][f][1] for r in rows if f in r["per_field"])
        if n:
            pf[f] = round(e / n, 4)
    ag, ah = sum(r["atoms_gt"] for r in rows), sum(r["atoms_hit"] for r in rows)
    ng, nh = sum(r["nums_gt"] for r in rows), sum(r["nums_hit"] for r in rows)
    return {"n": len(rows), "coverage_cer": round(E / max(C, 1), 4), "coverage_cer_per_field": pf,
            "atom_recall": round(ah / max(ag, 1), 4), "number_exact_recall": round(nh / max(ng, 1), 4),
            "len_ratio": round(sum(r["pred_len"] for r in rows) / max(sum(r["gt_len"] for r in rows), 1), 3)}


def field_cer(pred: dict, gt: dict) -> dict:
    """Plain per-field CER for extracted fields: (edits, gt_chars) per non-null GT field
    (a null prediction costs every GT character), plus fields predicted where GT is null."""
    import Levenshtein
    out, spurious = {}, []
    for f in FIELDS:
        g, p = normalize(gt.get(f)), normalize(pred.get(f))
        if g:
            out[f] = (Levenshtein.distance(g, p), len(g))
        elif p:
            spurious.append(f)
    return {"per_field": out, "spurious": spurious}
