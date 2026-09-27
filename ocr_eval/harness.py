"""Phase 1 evaluation harness -- per-field scoring with separated error classes.

Replaces the flat-blob CER in `make_report.py`. The design rules it enforces:

  * Ground truth is a 5-field OBJECT, never a concatenated blob (1.1).
  * The five error classes are measured separately and NEVER merged (1.2):
    field-assignment, omission, hallucination, character error, duplication.
  * The headline is always a PAIR -- usable-output rate together with
    CER-given-output -- plus effective CER. CER-given-output is never
    publishable on its own (1.3, defect D9).
  * Every headline number carries a bootstrap 95% CI, over images AND over
    templates, because effective n on the synthetic set is 6 templates, not 36
    images (1.8, defect D14).
  * Real and synthetic are scored and reported separately, never pooled (2.1).
  * A GT field that is ABSENT from the schema is `unscoreable`; a GT field that
    is present-and-null is scoreable via null-agreement. Conflating the two
    makes omission unmeasurable (2.2).

Usage:
    from harness import score_corpus
    result = score_corpus(predictions, ground_truth, manifest, splits)
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import normalize as N  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

HARNESS_VERSION = "1.0.0"

# The five fields the API contract emits. The harness always iterates this list,
# so a field the ground truth never covers shows up as `unscoreable` rather than
# silently disappearing (defect D19).
API_FIELDS = ["sender", "receiver", "subject", "body_text", "contact_info"]

BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 20260901

# Two strings are "the same content in a different field" when they are at least
# this similar. Deliberately strict: a loose threshold turns ordinary misreads
# into phantom assignment errors.
ASSIGNMENT_SIM = 0.80
# Duplication uses the same bar: the same GT content echoed into 2+ pred fields.
DUPLICATION_SIM = 0.80


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def cer(ref: str, hyp: str) -> float:
    """Character error rate. Undefined against an empty reference: callers must
    route empty-ref cases to null-agreement instead of calling this."""
    if not ref:
        return float("nan")
    return Levenshtein.distance(ref, hyp) / len(ref)


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return float("nan")
    return Levenshtein.distance(r, h) / len(r)


def similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 1.0 - Levenshtein.distance(a, b) / max(len(a), len(b))


def _script_of(ch: str) -> str:
    """Coarse script bucket. Cross-script substitutions are alignment noise and
    are reported separately, never mixed into the confusion table (1.5)."""
    if ch.isspace():
        return "space"
    o = ord(ch)
    if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0xFB50 <= o <= 0xFDFF:
        return "arabic"
    if ch.isascii() and ch.isalpha():
        return "latin"
    if ch.isdigit():
        return "digit"
    if unicodedata.category(ch).startswith("P"):
        return "punct"
    return "other"


# ---------------------------------------------------------------------------
# per-field outcome
# ---------------------------------------------------------------------------

@dataclass
class FieldResult:
    filename: str
    field: str
    # outcome is exactly one of:
    #   unscoreable   -- GT schema does not cover this field for this document
    #   null_agree    -- GT null and prediction null
    #   omission      -- GT has content, prediction is null/empty
    #   hallucination -- GT null, prediction has content
    #   scored        -- both present; cer/wer/exact are populated
    outcome: str
    gt: str | None = None
    pred: str | None = None
    cer: float | None = None
    wer: float | None = None
    exact: bool | None = None
    # set when this prediction's content actually belongs to a different GT field
    misassigned_to: str | None = None
    # set when this GT content also appears in another predicted field
    duplicated_in: list[str] = field(default_factory=list)


def _is_blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def score_document(
    filename: str,
    pred_fields: dict | None,
    gt_fields: dict,
    rules: N.Rules,
) -> list[FieldResult]:
    """Score one document, one row per API field.

    `pred_fields=None` means the request produced no usable output at all; the
    caller handles that at the document level (usable-output rate), and every
    field is recorded as an omission so effective CER can charge 100%.
    """
    results: list[FieldResult] = []
    pred_fields = pred_fields or {}

    norm_gt = {
        f: N.normalize(gt_fields[f], rules)
        for f in API_FIELDS
        if f in gt_fields and not _is_blank(gt_fields[f])
    }
    norm_pred = {
        f: N.normalize(pred_fields.get(f), rules)
        for f in API_FIELDS
        if not _is_blank(pred_fields.get(f))
    }

    for f in API_FIELDS:
        gt_present = f in gt_fields
        gt_val = norm_gt.get(f, "")
        pred_val = norm_pred.get(f, "")

        if not gt_present:
            # The evaluation schema never covered this field for this document.
            # Recording it as `unscoreable` rather than as a null keeps defect
            # D19 visible instead of scoring a hallucination against nothing.
            # The normalized prediction is still stored, because the assignment
            # pass below must be able to see content that landed here -- the
            # canonical D5 symptom is subject text dumped into `receiver`, and
            # `receiver` is exactly the field the ground truth never covers.
            results.append(FieldResult(filename, f, "unscoreable", None, pred_val))
            continue

        if not gt_val and not pred_val:
            results.append(FieldResult(filename, f, "null_agree", gt_val, pred_val))
            continue
        if gt_val and not pred_val:
            results.append(FieldResult(filename, f, "omission", gt_val, pred_val))
            continue
        if not gt_val and pred_val:
            results.append(FieldResult(filename, f, "hallucination", gt_val, pred_val))
            continue

        results.append(FieldResult(
            filename, f, "scored", gt_val, pred_val,
            cer=cer(gt_val, pred_val),
            wer=wer(gt_val, pred_val),
            exact=(gt_val == pred_val),
        ))

    # Field-assignment pass. Runs over EVERY field that received predicted
    # content, including `unscoreable` ones. Restricting it to scored fields
    # reported 0 assignment errors on a corpus whose documented failure mode is
    # assignment (D5), because the field the content lands in -- `receiver` --
    # is the one the ground truth never covers.
    for r in results:
        if not r.pred:
            continue
        own = norm_gt.get(r.field, "")
        best_other, best_sim = None, 0.0
        for g, gv in norm_gt.items():
            if g == r.field:
                continue
            sim = similarity(gv, r.pred)
            if sim > best_sim:
                best_other, best_sim = g, sim
        if best_other and best_sim >= ASSIGNMENT_SIM and best_sim > similarity(own, r.pred):
            r.misassigned_to = best_other

    # Duplication: one GT field's content echoed into two or more pred fields.
    # Also spans unscoreable fields, for the same reason as the assignment pass.
    for f, gv in norm_gt.items():
        hits = [p for p, pv in norm_pred.items() if similarity(gv, pv) >= DUPLICATION_SIM]
        if len(hits) > 1:
            for r in results:
                if r.field == f:
                    r.duplicated_in = sorted(hits)
    return results


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(
    units: list[Any],
    statistic,
    iters: int = BOOTSTRAP_ITERS,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap over whatever `units` are -- images or templates.

    Resampling templates rather than images is the only honest CI when the corpus
    has 36 images drawn from 6 templates (D14) or 90 documents from one
    letterhead (D16).
    """
    point = statistic(units)
    if point is None or (isinstance(point, float) and math.isnan(point)) or len(units) < 2:
        return {"point": point, "lo": None, "hi": None, "n_units": len(units),
                "note": "too few units to bootstrap"}
    rng = random.Random(seed)
    n = len(units)
    draws = []
    for _ in range(iters):
        sample = [units[rng.randrange(n)] for _ in range(n)]
        v = statistic(sample)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            draws.append(v)
    if not draws:
        return {"point": point, "lo": None, "hi": None, "n_units": n}
    draws.sort()
    lo = draws[int((alpha / 2) * len(draws))]
    hi = draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))]
    return {"point": round(point, 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "n_units": n, "iters": iters}


def fmt_ci(ci: dict, pct: bool = True, digits: int = 2) -> str:
    """Render as 'point [lo, hi]'. CIs are printed beside the number, never in a
    footnote (1.8)."""
    if ci.get("point") is None:
        return "n/a"
    m = 100.0 if pct else 1.0
    u = "%" if pct else ""
    if ci.get("lo") is None:
        return f"{ci['point']*m:.{digits}f}{u} [CI n/a, n={ci['n_units']}]"
    return (f"{ci['point']*m:.{digits}f}{u} "
            f"[{ci['lo']*m:.{digits}f}, {ci['hi']*m:.{digits}f}]")


# ---------------------------------------------------------------------------
# corpus-level aggregation
# ---------------------------------------------------------------------------

@dataclass
class DocOutcome:
    filename: str
    source: str
    language_mode: str
    template_id: str
    split: str
    usable: bool
    failure_kind: str | None      # http_422 | all_null | transport | timeout | None
    seconds: float | None
    fields: list[FieldResult]
    raw_pred: dict | None
    # populated lazily by _oracle_doc; not part of the reported record
    _oracle_cache: tuple[int, int] | None = None
    _assigned_cache: tuple[int, int] | None = None


def _micro_cer(docs: list[DocOutcome], only_assigned: bool = True) -> float | None:
    """Character error over correctly-assigned pairs only (1.2). Micro-averaged:
    total edits over total reference characters, so long fields carry their real
    weight."""
    num = den = 0
    for d in docs:
        for f in d.fields:
            if f.outcome != "scored":
                continue
            if only_assigned and f.misassigned_to:
                continue
            num += f.cer * len(f.gt)
            den += len(f.gt)
    return num / den if den else None


def _oracle_cer(docs: list[DocOutcome]) -> float | None:
    """Micro-CER under the BEST POSSIBLE assignment of predicted fields to ground
    -truth fields, found by brute force over all permutations (at most 5! = 120).

    This is the threshold-free counterpart to `misassigned_to`. The similarity
    threshold can only catch a prediction that is a near-clean copy of another
    field; when the model both misreads AND misassigns, the copy is too corrupted
    to trip the threshold and the assignment error hides inside CER. The oracle
    has no such blind spot: it simply asks what CER would be if every predicted
    string were routed to its best-fitting field.

    `assignment_cost = as_assigned_cer - oracle_cer` is therefore the share of
    character error that is pure routing, not reading -- the number Phase 1 must
    be able to state and the number the Phase 3 decision turns on.
    """
    num = den = 0
    for d in docs:
        c = _oracle_doc(d)
        num += c[0]
        den += c[1]
    return num / den if den else None


def _oracle_doc(d: DocOutcome) -> tuple[int, int]:
    """Per-document oracle cost: the minimum total edit distance achievable by
    routing the predicted strings to ground-truth fields, each prediction used at
    most once.

    Exact, via a bitmask DP over which predicted fields have been consumed. With
    at most 5 fields that is 2**5 states -- instant, and it always finds the true
    optimum. The earlier permutation version padded `None` incorrectly and so
    could not reach every assignment, which made the "oracle" score WORSE than
    the actual routing on the Persian-only slice. Since routing pred field f to
    gt field f is itself a feasible assignment, `oracle <= as_assigned` always
    holds; the assertion below fails loudly if that invariant ever breaks again.

    The bootstrap resamples documents thousands of times but a document's oracle
    cost never changes, so it is computed once and cached on the DocOutcome.
    """
    cached = getattr(d, "_oracle_cache", None)
    if cached is not None:
        return cached

    gt = {f.field: f.gt for f in d.fields if f.gt}
    if not gt:
        d._oracle_cache = (0, 0)
        return d._oracle_cache

    gt_names = list(gt)
    pred_names = [f for f in API_FIELDS if (d_pred := _pred_of(d, f))]
    preds = [_pred_of(d, f) for f in pred_names]

    # cost[i][j] = edits to turn predicted string j into ground-truth field i.
    # cost_none[i] = edits when field i receives nothing at all (a pure omission).
    cost = [[Levenshtein.distance(gt[g], pv) for pv in preds] for g in gt_names]
    cost_none = [len(gt[g]) for g in gt_names]

    INF = float("inf")
    n_pred = len(preds)
    # dp[mask] = best cost for the first `i` gt fields, having consumed `mask`.
    dp = {0: 0}
    for i in range(len(gt_names)):
        nxt: dict[int, float] = {}
        for mask, c in dp.items():
            # option 1: this field gets nothing
            if nxt.get(mask, INF) > c + cost_none[i]:
                nxt[mask] = c + cost_none[i]
            # option 2: this field takes an unused prediction
            for j in range(n_pred):
                bit = 1 << j
                if mask & bit:
                    continue
                v = c + cost[i][j]
                if nxt.get(mask | bit, INF) > v:
                    nxt[mask | bit] = v
        dp = nxt

    best = int(min(dp.values()))
    assigned = sum(Levenshtein.distance(f.gt, f.pred or "") for f in d.fields if f.gt)
    assert best <= assigned, (
        f"oracle {best} exceeded as-assigned {assigned} on {d.filename}; "
        "the oracle must be a lower bound by construction"
    )
    d._oracle_cache = (best, sum(len(v) for v in gt.values()))
    return d._oracle_cache


def _pred_of(d: DocOutcome, fieldname: str) -> str:
    for f in d.fields:
        if f.field == fieldname:
            return f.pred or ""
    return ""


def _as_assigned_cer(docs: list[DocOutcome]) -> float | None:
    """Micro-CER exactly as the API assigned the fields -- routing errors and all.
    The companion to `_oracle_cer`; their difference is the assignment cost."""
    num = den = 0
    for d in docs:
        cached = getattr(d, "_assigned_cache", None)
        if cached is None:
            n = sum(Levenshtein.distance(f.gt, f.pred or "") for f in d.fields if f.gt)
            m = sum(len(f.gt) for f in d.fields if f.gt)
            cached = d._assigned_cache = (n, m)
        num += cached[0]
        den += cached[1]
    return num / den if den else None


def _effective_cer(docs: list[DocOutcome]) -> float | None:
    """CER with no-output documents charged at 100% error (1.3). This is the
    number a stakeholder actually cares about."""
    num = den = 0
    for d in docs:
        for f in d.fields:
            if f.outcome == "unscoreable":
                continue
            if f.outcome in ("omission",) or not d.usable:
                num += len(f.gt or "")
                den += len(f.gt or "")
            elif f.outcome == "scored":
                num += f.cer * len(f.gt)
                den += len(f.gt)
            elif f.outcome == "hallucination":
                # No reference characters to divide by; charged to the
                # hallucination-rate metric instead of silently inflating CER.
                continue
    return num / den if den else None


def _pp_gap(docs: list[DocOutcome]) -> float | None:
    a, o = _as_assigned_cer(docs), _oracle_cer(docs)
    return round((a - o) * 100, 2) if (a is not None and o is not None) else None


def _assignment_share(docs: list[DocOutcome]) -> float | None:
    """What fraction of the as-assigned CER is routing rather than reading."""
    a, o = _as_assigned_cer(docs), _oracle_cer(docs)
    if a is None or o is None or a <= 0:
        return None
    return round(100.0 * (a - o) / a, 2)


def _rate(docs: list[DocOutcome], pred) -> float | None:
    return (sum(1 for d in docs if pred(d)) / len(docs)) if docs else None


def _field_rate(docs: list[DocOutcome], pred) -> float | None:
    rows = [f for d in docs for f in d.fields if f.outcome != "unscoreable"]
    return (sum(1 for f in rows if pred(f)) / len(rows)) if rows else None


def _all_field_rate(docs: list[DocOutcome], pred) -> float | None:
    """Rate over every field slot, unscoreable included -- for classes that are
    detectable without ground truth for the field the content landed in."""
    rows = [f for d in docs for f in d.fields]
    return (sum(1 for f in rows if pred(f)) / len(rows)) if rows else None


def _by_template(docs: list[DocOutcome]) -> list[list[DocOutcome]]:
    g = defaultdict(list)
    for d in docs:
        g[d.template_id].append(d)
    return list(g.values())


def _flat(groups: list[list[DocOutcome]]) -> list[DocOutcome]:
    return [d for g in groups for d in g]


def headline(docs: list[DocOutcome]) -> dict:
    """The headline pair plus effective CER, each with an image-level and a
    template-level CI. Never publish `cer_given_output` without its companions."""
    tmpl = _by_template(docs)
    def over_templates(stat):
        return lambda groups: stat(_flat(groups))
    return {
        "n_documents": len(docs),
        "n_templates": len(tmpl),
        "usable_output_rate": {
            "by_image": bootstrap_ci(docs, lambda ds: _rate(ds, lambda d: d.usable)),
            "by_template": bootstrap_ci(tmpl, over_templates(lambda ds: _rate(ds, lambda d: d.usable))),
        },
        "cer_given_output": {
            "by_image": bootstrap_ci([d for d in docs if d.usable],
                                     lambda ds: _micro_cer(ds)),
            "by_template": bootstrap_ci([[d for d in g if d.usable] for g in tmpl],
                                        over_templates(lambda ds: _micro_cer(ds))),
        },
        "effective_cer": {
            "by_image": bootstrap_ci(docs, _effective_cer),
            "by_template": bootstrap_ci(tmpl, over_templates(_effective_cer)),
        },
    }


def error_classes(docs: list[DocOutcome]) -> dict:
    """The five classes of 1.2, each as its own rate. Never summed together."""
    rows = [f for d in docs for f in d.fields]
    scoreable = [f for f in rows if f.outcome != "unscoreable"]
    n = len(scoreable) or 1
    return {
        "n_field_slots": len(rows),
        "n_scoreable_slots": len(scoreable),
        "n_unscoreable_slots": len(rows) - len(scoreable),
        # Counted over ALL rows (see the assignment pass in score_document):
        # excluding unscoreable rows here would re-hide exactly the errors that
        # the pass exists to find.
        "field_assignment_error": {
            "count": sum(1 for f in rows if f.misassigned_to),
            "rate": bootstrap_ci(docs, lambda ds: _all_field_rate(ds, lambda f: bool(f.misassigned_to))),
        },
        "content_in_unscoreable_field": {
            "count": sum(1 for f in rows if f.outcome == "unscoreable" and f.pred),
            "note": ("Predicted content in a field the ground truth never covers. "
                     "Not scoreable as right or wrong -- it is the measurement gap D19, "
                     "and it is where the D5 assignment errors hide."),
        },
        "omission": {
            "count": sum(1 for f in scoreable if f.outcome == "omission"),
            "rate": bootstrap_ci(docs, lambda ds: _field_rate(ds, lambda f: f.outcome == "omission")),
        },
        "hallucination": {
            "count": sum(1 for f in scoreable if f.outcome == "hallucination"),
            "rate": bootstrap_ci(docs, lambda ds: _field_rate(ds, lambda f: f.outcome == "hallucination")),
        },
        "duplication": {
            "count": sum(1 for f in rows if f.duplicated_in),
            "rate": bootstrap_ci(docs, lambda ds: _all_field_rate(ds, lambda f: bool(f.duplicated_in))),
        },
        "character_error_correctly_assigned_only": {
            "micro_cer": bootstrap_ci(docs, lambda ds: _micro_cer(ds, only_assigned=True)),
        },
        # Threshold-free decomposition of CER into routing vs reading.
        # Computed over USABLE documents only: a no-output failure is neither a
        # reading error nor a routing error, it is a reliability failure, and it
        # is already counted in the headline pair. Including it here would let a
        # 422 masquerade as a misread.
        "assignment_vs_reading": {
            "scope": "usable documents only",
            "n_usable": sum(1 for d in docs if d.usable),
            "as_assigned_cer": bootstrap_ci([d for d in docs if d.usable], _as_assigned_cer),
            "oracle_assigned_cer": bootstrap_ci([d for d in docs if d.usable], _oracle_cer),
            "assignment_cost_pp": _pp_gap([d for d in docs if d.usable]),
            "share_of_cer_from_misassignment_pct": _assignment_share([d for d in docs if d.usable]),
            "note": ("oracle = best possible routing of the SAME predicted strings. "
                     "The gap is character error caused by putting correct text in "
                     "the wrong field; the remainder is genuine misreading."),
        },
        "null_agreement": {
            "count": sum(1 for f in scoreable if f.outcome == "null_agree"),
            "rate": bootstrap_ci(docs, lambda ds: _field_rate(ds, lambda f: f.outcome == "null_agree")),
        },
    }


def per_field_table(docs: list[DocOutcome]) -> dict:
    """1.1 -- CER, WER, exact-match and null-agreement for each of the 5 fields."""
    out = {}
    for fname in API_FIELDS:
        rows = [f for d in docs for f in d.fields if f.field == fname]
        scoreable = [f for f in rows if f.outcome != "unscoreable"]
        scored = [f for f in scoreable if f.outcome == "scored"]
        num = sum(f.cer * len(f.gt) for f in scored)
        den = sum(len(f.gt) for f in scored)
        wnum = sum(f.wer * len(f.gt.split()) for f in scored if f.gt.split())
        wden = sum(len(f.gt.split()) for f in scored)
        out[fname] = {
            "n_slots": len(rows),
            "n_unscoreable": len(rows) - len(scoreable),
            "n_scored": len(scored),
            "cer": round(num / den, 4) if den else None,
            "wer": round(wnum / wden, 4) if wden else None,
            "exact_match": round(sum(1 for f in scored if f.exact) / len(scored), 4) if scored else None,
            "null_agreement": round(
                sum(1 for f in scoreable if f.outcome == "null_agree") / len(scoreable), 4
            ) if scoreable else None,
            "omissions": sum(1 for f in scoreable if f.outcome == "omission"),
            "hallucinations": sum(1 for f in scoreable if f.outcome == "hallucination"),
            "misassigned": sum(1 for f in scoreable if f.misassigned_to),
        }
    return out


def confusion(docs: list[DocOutcome], top: int = 30) -> dict:
    """1.5 -- substitution table over correctly-assigned pairs only, split into
    same-script (signal) and cross-script (alignment noise)."""
    same, cross = Counter(), Counter()
    for d in docs:
        for f in d.fields:
            if f.outcome != "scored" or f.misassigned_to:
                continue
            for op in Levenshtein.opcodes(f.gt, f.pred):
                if op.tag != "replace":
                    continue
                a = f.gt[op.src_start:op.src_end]
                b = f.pred[op.dest_start:op.dest_end]
                for ca, cb in zip(a, b):
                    (same if _script_of(ca) == _script_of(cb) else cross)[(ca, cb)] += 1
    fmt = lambda c: [
        {"gt": a, "pred": b, "gt_cp": f"U+{ord(a):04X}", "pred_cp": f"U+{ord(b):04X}", "count": n}
        for (a, b), n in c.most_common(top)
    ]
    return {
        "same_script": fmt(same),
        "cross_script_ALIGNMENT_NOISE": fmt(cross),
        "note": ("Cross-script substitutions are an artefact of aligning RTL and LTR "
                 "runs and are listed separately; they are not evidence of a "
                 "character confusion."),
    }


# 1.6 -- reversal detection, four buckets.
_ID_PREFIX = ("INV-", "EMP-", "REF-", "DOC-", "NO-")


def reversal_buckets(docs: list[DocOutcome]) -> dict:
    """Classify every token-level mismatch containing Latin/digits into exactly
    one of four buckets, so a bidi bug is never reported as a misread."""
    buckets = Counter()
    examples = defaultdict(list)
    for d in docs:
        for f in d.fields:
            if f.outcome != "scored":
                continue
            gt_toks = [t for t in f.gt.split() if any(c.isascii() and c.isalnum() for c in t)]
            pred_toks = f.pred.split()
            for g in gt_toks:
                if g in pred_toks:
                    continue
                cand = max(pred_toks, key=lambda p: similarity(g, p), default="")
                if not cand or similarity(g, cand) < 0.4:
                    buckets["absent"] += 1
                    examples["absent"].append({"file": d.filename, "field": f.field, "gt": g})
                    continue
                if cand == g[::-1]:
                    b = "exact_reversal"
                elif sorted(cand) == sorted(g):
                    b = "any_permutation"
                elif any(g.startswith(p) and cand == g[len(p):] for p in _ID_PREFIX):
                    b = "prefix_stripped"
                else:
                    b = "genuine_misread"
                buckets[b] += 1
                if len(examples[b]) < 12:
                    examples[b].append({"file": d.filename, "field": f.field,
                                        "gt": g, "pred": cand})
    total = sum(buckets.values())
    return {
        "total_latin_digit_token_mismatches": total,
        "buckets": {k: {"count": v, "pct": round(100.0 * v / total, 2) if total else 0.0}
                    for k, v in buckets.most_common()},
        "examples": {k: v for k, v in examples.items()},
    }


def latency(docs: list[DocOutcome]) -> dict:
    """1.7 -- percentiles computed twice: failures included and excluded."""
    def pct(vals, p):
        if not vals:
            return None
        vals = sorted(vals)
        return round(vals[min(len(vals) - 1, int(p * len(vals)))], 3)

    def block(vals):
        return {
            "n": len(vals),
            "mean": round(sum(vals) / len(vals), 3) if vals else None,
            "p50": pct(vals, 0.50), "p90": pct(vals, 0.90),
            "p95": pct(vals, 0.95), "p99": pct(vals, 0.99),
        }

    allv = [d.seconds for d in docs if d.seconds is not None]
    okv = [d.seconds for d in docs if d.usable and d.seconds is not None]
    return {
        "failures_included": block(allv),
        "failures_excluded": block(okv),
        "note": "Retry count is reported separately once Phase 2 introduces retries.",
    }


def failure_taxonomy(docs: list[DocOutcome]) -> dict:
    c = Counter(d.failure_kind for d in docs if not d.usable)
    return {"n_failures": sum(c.values()), "by_kind": dict(c.most_common())}


def score_slice(docs: list[DocOutcome], label: str, rules: N.Rules) -> dict:
    return {
        "label": label,
        "normalization": N.rules_dict(rules),
        "headline": headline(docs),
        "error_classes": error_classes(docs),
        "per_field": per_field_table(docs),
        "confusion": confusion(docs),
        "reversal": reversal_buckets(docs),
        "latency": latency(docs),
        "failures": failure_taxonomy(docs),
        "letterform_conformance_RAW": N.letterform_report(
            [v for d in docs if d.raw_pred for v in d.raw_pred.values() if isinstance(v, str)]
        ),
    }


def config_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
