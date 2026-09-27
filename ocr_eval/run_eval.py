"""
Accuracy-testing harness for the Persian + English OCR service.

Calls the REAL /ocr endpoint on every image in the synthetic test set, flattens
the returned JSON fields into text, and scores it against the ground truth with
CER / WER (via jiwer). Persian-aware normalization is applied, and every headline
number is reported BOTH with and without digit-style normalization so the effect
of normalization is visible rather than hidden.

Outputs (in ocr_eval/):
  predictions.jsonl   raw model output per image (re-usable, so --report-only can
                      recompute metrics without re-running inference)
  results.json        all computed metrics + per-image rows + confusion pairs
  REPORT.md           human-readable report (written by make_report.py)

Usage:
  python ocr_eval/run_eval.py                 # run inference then score
  python ocr_eval/run_eval.py --report-only   # rescore from predictions.jsonl
  OCR_URL=http://127.0.0.1:8000/ocr           # override endpoint
"""
import argparse
import difflib
import io
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import jiwer

HERE = Path(__file__).resolve().parent
IMG_DIR = HERE / "images"
GT_PATH = HERE / "ground_truth.jsonl"
PRED_PATH = HERE / "predictions.jsonl"
RESULTS_PATH = HERE / "results.json"

OCR_URL = os.getenv("OCR_URL", "http://127.0.0.1:8000/ocr")
REQUEST_TIMEOUT = float(os.getenv("OCR_TIMEOUT", "300"))

# Order in which the extraction fields are flattened into one text blob.
FIELD_ORDER = ["sender", "receiver", "subject", "body_text", "contact_info"]

# ── Regexes for language-segment extraction ────────────────────────────────────
_FA_RUN = re.compile(r"[؀-ۿﭐ-﷿ﹰ-﻿]+")
_EN_RUN = re.compile(r"[A-Za-z0-9@._\-/:]+")
_LATIN_DIGIT = re.compile(r"[A-Za-z0-9]")
_EN_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-/@]*")


# ── Persian-aware normalization ────────────────────────────────────────────────
_CHAR_MAP = {
    "ي": "ی",  # Arabic yeh    ي -> Persian yeh ی
    "ى": "ی",  # alef maksura  ى -> ی
    "ك": "ک",  # Arabic kaf    ك -> Persian kaf ک
    "ة": "ه",  # teh marbuta   ة -> ه
    "ـ": "",         # tatweel/kashida ـ -> removed
}
_ZERO_WIDTH = dict.fromkeys(
    [0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0xFEFF], None
)
_PUNCT_MAP = {
    "،": ",",   # Arabic comma ،
    "؛": ";",   # Arabic semicolon ؛
    "؟": "?",   # Arabic question mark ؟
    "«": '"', "»": '"',
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-",
}
_FA_DIGITS = {ord("۰") + i: str(i) for i in range(10)}          # U+06F0..
_AR_DIGITS = {ord("٠") + i: str(i) for i in range(10)}          # U+0660..


def normalize(text: str, level: str = "digit") -> str:
    """level: 'raw' (whitespace only) | 'nodigit' (Persian norm, keep digit styles)
    | 'digit' (Persian norm + Persian/Arabic digits -> Latin)."""
    if not text:
        return ""
    if level == "raw":
        return re.sub(r"\s+", " ", text).strip()
    text = text.translate(_ZERO_WIDTH)
    for a, b in _CHAR_MAP.items():
        text = text.replace(a, b)
    for a, b in _PUNCT_MAP.items():
        text = text.replace(a, b)
    if level == "digit":
        text = text.translate(_FA_DIGITS).translate(_AR_DIGITS)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Backwards-friendly wrapper used in a couple of call sites.
def norm_digit(text: str) -> str:
    return normalize(text, "digit")


def extract_fa(text: str) -> str:
    return " ".join(_FA_RUN.findall(text)).strip()


def extract_en(text: str) -> str:
    toks = [t for t in _EN_RUN.findall(text) if _LATIN_DIGIT.search(t)]
    return " ".join(toks).strip()


# ── Metrics ────────────────────────────────────────────────────────────────────
def safe_cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.cer(ref, hyp))


def safe_wer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.wer(ref, hyp))


def corpus_cer(refs, hyps) -> float:
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs:
        return 0.0
    r, h = zip(*pairs)
    return float(jiwer.cer(list(r), list(h)))


def corpus_wer(refs, hyps) -> float:
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs:
        return 0.0
    r, h = zip(*pairs)
    return float(jiwer.wer(list(r), list(h)))


def substitution_pairs(ref: str, hyp: str):
    """Yield (ref_char, hyp_char) substitutions via char-level alignment."""
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            a, b = ref[i1:i2], hyp[j1:j2]
            for k in range(min(len(a), len(b))):
                if a[k] != b[k] and not a[k].isspace() and not b[k].isspace():
                    yield (a[k], b[k])


# ── RTL/LTR ordering check ─────────────────────────────────────────────────────
_DIGIT_RUN = re.compile(r"\d{3,}")
_ALPHA_RUN = re.compile(r"[A-Za-z]{3,}")


def ordering_stats(ref_en: str, pred_norm: str):
    """Classify Latin-letter runs and digit runs (>=3 chars) from the reference as
    exact / reversed / other, checked against the (digit-normalized) prediction.
    Digit-group reversal is the main bilingual RTL/LTR failure mode, e.g. a code
    like 0093 appearing as 3900."""
    ref_norm = norm_digit(ref_en)
    units = _DIGIT_RUN.findall(ref_norm) + _ALPHA_RUN.findall(ref_norm)
    exact = reversed_ = other = 0
    reversed_examples = []
    for u in units:
        if u in pred_norm:
            exact += 1
        elif u != u[::-1] and u[::-1] in pred_norm:
            reversed_ += 1
            reversed_examples.append(f"{u}->{u[::-1]}")
        else:
            other += 1
    return {
        "units": len(units),
        "exact": exact,
        "reversed": reversed_,
        "other": other,
        "reversed_examples": reversed_examples,
    }


# ── Inference ──────────────────────────────────────────────────────────────────
def flatten_prediction(payload: dict) -> str:
    parts = []
    for field in FIELD_ORDER:
        val = payload.get(field)
        if val:
            parts.append(str(val))
    return "\n".join(parts)


def run_inference(rows):
    preds = []
    n = len(rows)
    for i, row in enumerate(rows, 1):
        img_path = IMG_DIR / row["filename"]
        t0 = time.time()
        pred_text, raw, err = "", None, None
        try:
            with open(img_path, "rb") as fh:
                resp = httpx.post(
                    OCR_URL,
                    files={"file": (row["filename"], fh, "image/png")},
                    timeout=REQUEST_TIMEOUT,
                )
            if resp.status_code == 200:
                raw = resp.json()
                pred_text = flatten_prediction(raw)
            else:
                err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        dt = time.time() - t0
        preds.append({
            "filename": row["filename"],
            "source": row["source"],
            "pred_text": pred_text,
            "raw": raw,
            "error": err,
            "seconds": round(dt, 2),
        })
        status = "OK " if err is None else "ERR"
        print(f"[{i:>3}/{n}] {status} {row['filename']:<28} {dt:5.1f}s"
              + (f"  {err}" if err else ""))
    with open(PRED_PATH, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return preds


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Scoring ────────────────────────────────────────────────────────────────────
def score(rows, preds):
    gt = {r["filename"]: r for r in rows}
    per_image = []
    confusion = Counter()

    for p in preds:
        r = gt[p["filename"]]
        ref_text = r["text"]
        pred_text = p["pred_text"]

        ref_d, hyp_d = norm_digit(ref_text), norm_digit(pred_text)

        # Per-language segment CER (digits normalized).
        ref_fa = norm_digit(r.get("ref_fa", ""))
        ref_en = norm_digit(r.get("ref_en", ""))
        pred_fa = norm_digit(extract_fa(pred_text))
        pred_en = norm_digit(extract_en(pred_text))

        if p["error"] is not None:
            status = "error"          # HTTP 422 / exception: no usable output
        elif not hyp_d.strip():
            status = "empty"          # 200 OK but all fields null: no output
        else:
            status = "ok"

        row = {
            "filename": p["filename"],
            "source": p["source"],
            "font": r.get("font"),
            "size": r.get("size"),
            "rotation": r.get("rotation"),
            "error": p["error"],
            "status": status,
            "ref_text": ref_text,
            "pred_text": pred_text,
            "cer": safe_cer(ref_d, hyp_d),
            "wer": safe_wer(ref_d, hyp_d),
            "cer_fa": safe_cer(ref_fa, pred_fa) if ref_fa else None,
            "cer_en": safe_cer(ref_en, pred_en) if ref_en else None,
            "ordering": ordering_stats(r.get("ref_en", ""), hyp_d) if ref_en else None,
        }
        per_image.append(row)

        if status == "ok":
            for a, b in substitution_pairs(ref_d, hyp_d):
                confusion[f"{a} -> {b}"] += 1

    # ── Corpus aggregates over images that PRODUCED output; no-output tracked apart ─
    ok = [x for x in per_image if x["status"] == "ok"]

    def agg(subset, level):
        refs = [normalize(x["ref_text"], level) for x in subset]
        hyps = [normalize(x["pred_text"], level) for x in subset]
        return corpus_cer(refs, hyps), corpus_wer(refs, hyps)

    sources = sorted({row["source"] for row in per_image})
    by_source = {}
    for src in ["ALL"] + sources:
        subset = ok if src == "ALL" else [x for x in ok if x["source"] == src]
        total = per_image if src == "ALL" else [x for x in per_image if x["source"] == src]
        cer_d, wer_d = agg(subset, "digit")
        by_source[src] = {
            "n": len(total),
            "n_scored": len(subset),
            "n_empty": sum(1 for x in total if x["status"] == "empty"),
            "n_errors": sum(1 for x in total if x["status"] == "error"),
            "cer": cer_d, "wer": wer_d,
        }

    # Normalization sensitivity at ALL level: raw vs Persian-norm vs +digit.
    cer_raw, wer_raw = agg(ok, "raw")
    cer_nd, wer_nd = agg(ok, "nodigit")
    cer_dg, wer_dg = agg(ok, "digit")
    normalization_sensitivity = {
        "raw":     {"cer": cer_raw, "wer": wer_raw},
        "persian": {"cer": cer_nd,  "wer": wer_nd},
        "persian_plus_digits": {"cer": cer_dg, "wer": wer_dg},
    }

    # Per-language aggregate on the bilingual set (successful images only).
    bi = [x for x in ok if x["source"] == "synthetic_bilingual"]
    fa_refs = [norm_digit(gt[x["filename"]]["ref_fa"]) for x in bi]
    fa_hyps = [norm_digit(extract_fa(x["pred_text"])) for x in bi]
    en_refs = [norm_digit(gt[x["filename"]]["ref_en"]) for x in bi]
    en_hyps = [norm_digit(extract_en(x["pred_text"])) for x in bi]
    language_breakdown = {
        "persian_segment_cer": corpus_cer(fa_refs, fa_hyps),
        "latin_digit_segment_cer": corpus_cer(en_refs, en_hyps),
    }

    # Ordering aggregate on bilingual set (successful images only).
    tot = ex = rev = oth = 0
    rev_ex = []
    for x in bi:
        o = x["ordering"]
        if not o:
            continue
        tot += o["units"]; ex += o["exact"]; rev += o["reversed"]; oth += o["other"]
        rev_ex += o["reversed_examples"]
    ordering_agg = {
        "units": tot, "exact": ex, "reversed": rev, "other": oth,
        "reversed_rate": (rev / tot) if tot else 0.0,
        "other_rate": (oth / tot) if tot else 0.0,
        "reversed_examples": sorted(set(rev_ex))[:20],
    }

    n_err = sum(1 for x in per_image if x["status"] == "error")
    n_empty = sum(1 for x in per_image if x["status"] == "empty")
    results = {
        "endpoint": OCR_URL,
        "n_images": len(per_image),
        "n_ok": len(ok),
        "n_errors": n_err,
        "n_empty": n_empty,
        "no_output_rate": (n_err + n_empty) / max(len(per_image), 1),
        "error_rate": n_err / max(len(per_image), 1),
        "avg_seconds": round(sum(p["seconds"] for p in preds) / max(len(preds), 1), 2),
        "by_source": by_source,
        "normalization_sensitivity": normalization_sensitivity,
        "language_breakdown": language_breakdown,
        "ordering": ordering_agg,
        "confusion_top": confusion.most_common(30),
        "per_image": per_image,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def print_summary(res):
    print("\n" + "=" * 66)
    print(f"Endpoint: {res['endpoint']}")
    print(f"Images: {res['n_images']}   no-output: {res['n_errors']} err(422) + "
          f"{res['n_empty']} empty(200-null) = {res['no_output_rate']*100:.1f}%   "
          f"Avg: {res['avg_seconds']}s/img")
    print("CER/WER below are over images that PRODUCED output; no-output tracked apart.")
    print("-" * 66)
    print(f"{'source':<22}{'scored':>7}{'empty':>6}{'err':>5}  {'CER':>8} {'WER':>8}")
    for src, m in res["by_source"].items():
        print(f"{src:<22}{m['n_scored']:>7}{m['n_empty']:>6}{m['n_errors']:>5}  "
              f"{m['cer']*100:7.2f}% {m['wer']*100:7.2f}%")
    print("-" * 66)
    ns = res["normalization_sensitivity"]
    print("Normalization sensitivity (ALL, CER):")
    print(f"  raw (no normalization) : {ns['raw']['cer']*100:.2f}%")
    print(f"  + Persian normalization: {ns['persian']['cer']*100:.2f}%")
    print(f"  + digit normalization  : {ns['persian_plus_digits']['cer']*100:.2f}%")
    lb = res["language_breakdown"]
    print("Bilingual language segments (CER):")
    print(f"  Persian-script : {lb['persian_segment_cer']*100:.2f}%")
    print(f"  Latin/digit    : {lb['latin_digit_segment_cer']*100:.2f}%")
    o = res["ordering"]
    print(f"RTL/LTR ordering on Latin/digit runs: reversed {o['reversed']}/{o['units']} "
          f"({o['reversed_rate']*100:.1f}%), other-misread {o['other']}/{o['units']} "
          f"({o['other_rate']*100:.1f}%)")
    if o["reversed_examples"]:
        print("  reversed examples:", ", ".join(o["reversed_examples"][:8]))
    print("=" * 66)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true",
                    help="skip inference, rescore from predictions.jsonl")
    args = ap.parse_args()

    rows = load_jsonl(GT_PATH)
    if args.report_only:
        if not PRED_PATH.exists():
            sys.exit("predictions.jsonl not found; run without --report-only first.")
        preds = load_jsonl(PRED_PATH)
    else:
        print(f"Running inference on {len(rows)} images via {OCR_URL}\n")
        preds = run_inference(rows)

    res = score(rows, preds)
    print_summary(res)
    print(f"\nWrote {RESULTS_PATH.name} and {PRED_PATH.name}.")
    print("Now run:  python ocr_eval/make_report.py")


if __name__ == "__main__":
    main()
