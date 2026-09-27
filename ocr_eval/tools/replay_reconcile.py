# -*- coding: utf-8 -*-
r"""Replay the numeric reconciliation offline on stored VLM predictions.

Takes the per-document `fields_pred` from an existing benchmark file (the VLM's raw
output), runs the v2 glyph reader on the prepared page, applies
`numeric_reconcile.reconcile`, and scores before/after against the human GT — no
model inference, so a policy change is measured in a minute.

Metrics (dev split unless told otherwise):
    span recall   the benchmark's own unit: (field, digits of a GT numeric span)
    atom recall   (field, maximal digit run >= 3) — the segmentation-free unit
    override precision   corrected atoms whose new digits are in the GT field
    injection precision  added numbers that are in the GT field

    venv312\Scripts\python.exe ocr_eval\tools\replay_reconcile.py --from qwen25_dev_glyph --label recon_v1
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image  # noqa: E402

from ocr_service.numeric_reconcile import Policy, reconcile  # noqa: E402
from ocr_service.numeric_validator import digits_only, to_ascii_digits  # noqa: E402
from ocr_service.preprocess import prepare  # noqa: E402

EVAL = ROOT / "ocr_eval" / "eval_set.jsonl"
BENCH = ROOT / "ocr_eval" / "benchmarks"
OUT = ROOT / "ocr_eval" / "experiments" / "digits_0922"
FIELDS = ["sender", "receiver", "subject", "body_text", "contact_info"]
READS_CACHE = OUT / "reads_cache.json"


def atoms(t: str | None) -> collections.Counter:
    return collections.Counter(a for a in re.findall(r"\d+", to_ascii_digits(t or "")) if len(a) >= 3)


_GROUPED = re.compile(r"[0-9۰-۹٠-٩]+(?:[/,،.\-][0-9۰-۹٠-٩]+)+")


def grouped(text: str | None) -> list[str]:
    """Whole multi-group numbers — amounts, dates, phone lists — as printed."""
    out = []
    for m in _GROUPED.finditer(text or ""):
        v = to_ascii_digits(m.group(0))
        if len(re.sub(r"\D", "", v)) >= 4:
            out.append(v)
    return out


def _parts(v: str) -> list[str]:
    return [p for p in re.split(r"[^0-9]", v) if p]


def grouped_match(gt: str, cands: list[str]) -> str | None:
    """Exact, or exact once the group order is flipped — the page lays a multi-group
    number out right-to-left, so both orders describe the same printed number."""
    for c in cands:
        if c == gt:
            return "exact"
    g = _parts(gt)
    for c in cands:
        if _parts(c) == g[::-1]:
            return "reversed"
    return None


def span_set(fields: dict) -> set[tuple[str, str]]:
    from ocr_service.numeric_validator import find_numeric_spans
    return {(s.field, digits_only(s.value)) for s in find_numeric_spans(fields, 3)}


class _Read:
    def __init__(self, d):
        self.text, self.box, self.confidence, self.mean_prob, self.script = d["text"], tuple(d["box"]), d["conf"], d["mean"], d["script"]


def get_reads(doc: dict, reader, cache: dict) -> tuple[list, int]:
    key = doc["sha256"]
    if key in cache:
        c = cache[key]
        return [_Read(x) for x in c["reads"]], c["page_h"]
    raw, _ = prepare(Path(doc["image"]).read_bytes())
    im = Image.open(io.BytesIO(raw))
    reads = reader.read_page(im)
    cache[key] = {"page_h": im.height, "reads": [{"text": r.text, "box": list(r.box), "conf": r.confidence,
                                                   "mean": r.mean_prob, "script": r.script} for r in reads]}
    return reads, im.height


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default="qwen25_dev_glyph", help="benchmark label whose fields_pred to replay")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--label", required=True)
    ap.add_argument("--no-inject", action="store_true")
    ap.add_argument("--policy", default=None, help="JSON overrides for Policy fields")
    a = ap.parse_args()

    bench = json.loads((BENCH / f"bench_{a.src}.json").read_text(encoding="utf-8"))
    ev = {json.loads(l)["doc_id"]: json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()}
    pol = Policy(**(json.loads(a.policy) if a.policy else {}))
    if a.no_inject:
        pol.inject = False
    from ocr_service.digit_reader_v2 import GlyphReaderV2
    reader = GlyphReaderV2()
    cache = json.loads(READS_CACHE.read_text(encoding="utf-8")) if READS_CACHE.is_file() else {}

    tot = collections.Counter()
    per_field = collections.defaultdict(collections.Counter)
    per_doc = []
    t0 = time.perf_counter()
    for d in bench["per_doc"]:
        if "error" in d or d["split"] != a.split:
            continue
        doc = ev[d["doc_id"]]
        reads, page_h = get_reads(doc, reader, cache)
        before = d["fields_pred"]
        after, records = reconcile(before, reads, page_h, pol)

        gt_spans = {(n["field"], n["digits"]) for n in doc["numeric_gt"]}
        before_spans, after_spans = span_set(before), span_set(after)
        tot["span_gt"] += len(gt_spans)
        tot["span_before"] += len(gt_spans & before_spans)
        tot["span_after"] += len(gt_spans & after_spans)

        # what the caller sees in numeric_fields: resolved values + added records
        rec_atoms = collections.defaultdict(collections.Counter)
        for r in records:
            for at in atoms(r.resolved):
                rec_atoms[r.field][at] += 1
        # whole grouped numbers, delivered intact
        for f in FIELDS:
            cands_before = grouped(before.get(f)) + [to_ascii_digits(r.value) for r in records if r.field == f]
            cands_after = grouped(after.get(f)) + [to_ascii_digits(r.resolved) for r in records if r.field == f]
            for gnum in grouped(doc["fields"][f]):
                tot["grp_gt"] += 1
                if grouped_match(gnum, cands_before):
                    tot["grp_before"] += 1
                m = grouped_match(gnum, cands_after)
                if m:
                    tot["grp_after"] += 1
                    tot["grp_" + m] += 1
        for f in FIELDS:
            g, b, af = atoms(doc["fields"][f]), atoms(before.get(f)), atoms(after.get(f))
            per_field[f]["records"] += sum((g & rec_atoms[f]).values())
            per_field[f]["records_emitted"] += sum(rec_atoms[f].values())
            per_field[f]["gt"] += sum(g.values()); per_field[f]["before"] += sum((g & b).values()); per_field[f]["after"] += sum((g & af).values())
            per_field[f]["emitted_before"] += sum(b.values()); per_field[f]["emitted_after"] += sum(af.values())
            per_field[f]["wrong_before"] += sum((b - g).values()); per_field[f]["wrong_after"] += sum((af - g).values())

        for r in records:
            g = atoms(doc["fields"].get(r.field))
            for dcn in r.atoms:
                if dcn.status == "corrected":
                    tot["corrected"] += 1
                    tot["corrected_right"] += int(g[dcn.resolved] > 0)
                    tot["corrected_was_right"] += int(g[dcn.vlm] > 0)
                elif dcn.status == "confirmed":
                    tot["confirmed"] += 1; tot["confirmed_right"] += int(g[dcn.vlm] > 0)
                elif dcn.status == "conflict":
                    tot["conflict"] += 1; tot["conflict_vlm_right"] += int(g[dcn.vlm] > 0)
                    tot["conflict_classical_right"] += int(dcn.classical is not None and g[dcn.classical] > 0)
                else:
                    tot["unverified"] += 1; tot["unverified_vlm_right"] += int(g[dcn.vlm] > 0)
            if r.confidence == "added":
                tot["added"] += 1; tot["added_right"] += int(g[r.value_ascii] > 0)
        per_doc.append({"doc_id": d["doc_id"], "fields_after": after,
                        "records": [{"field": r.field, "kind": r.kind, "value": r.value, "resolved": r.resolved,
                                     "confidence": r.confidence, "classical": r.classical_value} for r in records]})
    READS_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    def pct(n, d):
        return f"{n / d:.1%}" if d else "-"
    summary = {
        "label": a.label, "from": a.src, "split": a.split, "n_docs": len(per_doc), "policy": pol.__dict__,
        "span_recall_before": round(tot["span_before"] / max(tot["span_gt"], 1), 4),
        "span_recall_after": round(tot["span_after"] / max(tot["span_gt"], 1), 4),
        "atom_recall_before": round(sum(v["before"] for v in per_field.values()) / max(sum(v["gt"] for v in per_field.values()), 1), 4),
        "atom_recall_after": round(sum(v["after"] for v in per_field.values()) / max(sum(v["gt"] for v in per_field.values()), 1), 4),
        "atom_precision_before": round(sum(v["before"] for v in per_field.values()) / max(sum(v["emitted_before"] for v in per_field.values()), 1), 4),
        "atom_precision_after": round(sum(v["after"] for v in per_field.values()) / max(sum(v["emitted_after"] for v in per_field.values()), 1), 4),
        "grouped_numbers": {"gt": tot["grp_gt"], "before": tot["grp_before"], "after": tot["grp_after"],
                            "exact_order": tot["grp_exact"], "reversed_order": tot["grp_reversed"],
                            "rate_before": round(tot["grp_before"] / max(tot["grp_gt"], 1), 4),
                            "rate_after": round(tot["grp_after"] / max(tot["grp_gt"], 1), 4)},
        "record_atom_recall": round(sum(v["records"] for v in per_field.values()) / max(sum(v["gt"] for v in per_field.values()), 1), 4),
        "record_atom_precision": round(sum(v["records"] for v in per_field.values()) / max(sum(v["records_emitted"] for v in per_field.values()), 1), 4),
        "per_field": {f: dict(c) for f, c in per_field.items() if c["gt"] or c["emitted_before"]},
        "decisions": dict(tot),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"recon_{a.label}.json").write_text(json.dumps({**summary, "per_doc": per_doc}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"== {a.label} (from {a.src}, {a.split}, n={len(per_doc)})")
    print(f"  span recall   before {summary['span_recall_before']:.1%}  after {summary['span_recall_after']:.1%}   (GT spans {tot['span_gt']})")
    print(f"  atom recall   before {summary['atom_recall_before']:.1%}  after {summary['atom_recall_after']:.1%}")
    print(f"  atom precision before {summary['atom_precision_before']:.1%}  after {summary['atom_precision_after']:.1%}")
    gg = summary["grouped_numbers"]
    print(f"  GROUPED numbers intact (amounts, dates, phone lists): before {gg['before']}/{gg['gt']} = {gg['rate_before']:.1%}"
          f"   after {gg['after']}/{gg['gt']} = {gg['rate_after']:.1%}  ({gg['exact_order']} as printed, {gg['reversed_order']} group order flipped)")
    print(f"  numeric_fields view: atom recall {summary['record_atom_recall']:.1%}  precision {summary['record_atom_precision']:.1%}")
    for f, c in summary["per_field"].items():
        print(f"    {f:13s} gt {c['gt']:3d}  before {c['before']:3d}  after {c['after']:3d}  records {c['records']:3d}  wrong-emitted before {c['wrong_before']:3d} after {c['wrong_after']:3d}")
    print(f"  confirmed {tot['confirmed']} (right {pct(tot['confirmed_right'], tot['confirmed'])}) | "
          f"corrected {tot['corrected']} (now right {pct(tot['corrected_right'], tot['corrected'])}, was right {pct(tot['corrected_was_right'], tot['corrected'])}) | "
          f"conflict {tot['conflict']} (vlm right {pct(tot['conflict_vlm_right'], tot['conflict'])}, classical right {pct(tot['conflict_classical_right'], tot['conflict'])}) | "
          f"unverified {tot['unverified']} (vlm right {pct(tot['unverified_vlm_right'], tot['unverified'])}) | "
          f"added {tot['added']} (right {pct(tot['added_right'], tot['added'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
