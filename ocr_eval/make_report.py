"""
Render ocr_eval/results.json into a human-readable ocr_eval/REPORT.md.
Run run_eval.py first to produce results.json.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.json"
REPORT = HERE / "REPORT.md"

# Commonly-confused Persian letter pairs to look for in the confusion summary.
WATCH_PAIRS = ["ب/پ", "ت/ث", "ج/چ", "ح/خ", "س/ش", "ص/ض",
               "ط/ظ", "ع/غ", "ف/ق", "ک/گ", "ز/ژ", "د/ذ", "ر/ز"]


def pct(x):
    return f"{x*100:.2f}%"


def esc(s: str) -> str:
    """Make text safe + readable inside a markdown table cell."""
    return (s or "").replace("\n", " ⏎ ").replace("|", "\\|").replace("\u200c", "")


def main():
    res = json.loads(RESULTS.read_text(encoding="utf-8"))
    per_image = res["per_image"]
    ok = [x for x in per_image if x.get("status") == "ok"]
    errored = [x for x in per_image if x.get("status") == "error"]
    empties = [x for x in per_image if x.get("status") == "empty"]

    by = res["by_source"]
    ns = res["normalization_sensitivity"]
    lb = res["language_breakdown"]
    o = res["ordering"]

    worst = sorted(ok, key=lambda x: x["cer"], reverse=True)[:15]

    L = []
    L.append("# OCR Accuracy Report\n")
    L.append(f"Endpoint tested: `{res['endpoint']}` (the real `/ocr` product endpoint).  ")
    L.append(f"Model output (5 JSON fields) is flattened into one text blob and scored "
             f"against ground truth.\n")

    # ── Headline ──
    a = by["ALL"]
    L.append("## 1. Headline numbers\n")
    L.append(f"- **Images:** {res['n_images']}  ")
    L.append(f"- **No output:** {res['n_errors']} hard failures (HTTP 422, unparseable JSON) "
             f"+ {res['n_empty']} empty (200 OK but all fields null) = "
             f"**{pct(res['no_output_rate'])}**  ")
    L.append(f"- **Images that produced text:** {res['n_ok']}  ")
    L.append(f"- **CER (over images with output):** **{pct(a['cer'])}**  ")
    L.append(f"- **WER (over images with output):** **{pct(a['wer'])}**  ")
    L.append(f"- **Avg latency:** {res['avg_seconds']} s/image\n")
    L.append("> There are two distinct failure modes, kept separate from reading accuracy so "
             "neither hides the other: (1) **no output** — the model either loops into invalid "
             "JSON (422) or returns all-null fields; (2) **reading errors** — measured by "
             "CER/WER only on the images where the model actually produced text.\n")

    # ── By source ──
    L.append("## 2. Accuracy by dataset source\n")
    L.append("| Source | Scored | Empty | 422 | CER | WER |")
    L.append("|---|--:|--:|--:|--:|--:|")
    for src, m in by.items():
        L.append(f"| {src} | {m['n_scored']} | {m.get('n_empty',0)} | {m['n_errors']} | "
                 f"{pct(m['cer'])} | {pct(m['wer'])} |")
    L.append("")

    # ── Normalization sensitivity ──
    L.append("## 3. Normalization sensitivity (ALL, CER)\n")
    L.append("Persian CER is famously sensitive to normalization choices, so the same "
             "predictions are scored three ways:\n")
    L.append("| Normalization | CER | WER |")
    L.append("|---|--:|--:|")
    L.append(f"| raw (whitespace only) | {pct(ns['raw']['cer'])} | {pct(ns['raw']['wer'])} |")
    L.append(f"| + Persian (ی/ك unify, ZWNJ, punctuation) | {pct(ns['persian']['cer'])} "
             f"| {pct(ns['persian']['wer'])} |")
    L.append(f"| + digit style (۰-۹ → 0-9) | {pct(ns['persian_plus_digits']['cer'])} "
             f"| {pct(ns['persian_plus_digits']['wer'])} |")
    L.append("")
    d_persian = ns['raw']['cer'] - ns['persian']['cer']
    d_digit = ns['persian']['cer'] - ns['persian_plus_digits']['cer']
    L.append(f"- Persian normalization changes CER by {pct(abs(d_persian))} "
             f"({'helps' if d_persian>0 else 'no gain'}) — mostly ZWNJ and ی/ك unification.  ")
    L.append(f"- Digit-style normalization changes CER by {pct(abs(d_digit))}. In this set the "
             f"ground-truth digits are Latin and the model also emits Latin digits, so it has "
             f"**near-zero effect here** — but it would matter for documents whose numbers are "
             f"written in Persian digits (۰-۹).\n")

    # ── Language segments ──
    L.append("## 4. Bilingual: per-language segment accuracy\n")
    L.append("On the bilingual letters, Persian-script runs and Latin/digit runs are scored "
             "separately (a single blended score would hide which side is weaker):\n")
    L.append("| Segment | CER |")
    L.append("|---|--:|")
    L.append(f"| Persian-script | {pct(lb['persian_segment_cer'])} |")
    L.append(f"| Latin / digits | {pct(lb['latin_digit_segment_cer'])} |")
    L.append("")
    if lb['latin_digit_segment_cer'] > lb['persian_segment_cer']:
        L.append(f"**The Latin/digit side is the weaker one** "
                 f"({pct(lb['latin_digit_segment_cer'])} vs {pct(lb['persian_segment_cer'])} CER). "
                 f"Codes, dates, and account numbers embedded in Persian text are read less "
                 f"reliably than the surrounding Persian prose.\n")

    # ── Ordering ──
    L.append("## 5. RTL/LTR ordering errors (Latin/digit runs)\n")
    L.append(f"- Latin-letter / digit runs checked: **{o['units']}**  ")
    L.append(f"- Reversed (e.g. a number's digits flipped): **{o['reversed']}** "
             f"({pct(o['reversed_rate'])})  ")
    L.append(f"- Other misreads (wrong characters, not a clean reversal): **{o['other']}** "
             f"({pct(o['other_rate'])})  ")
    if o["reversed_examples"]:
        L.append(f"- Reversal examples: " + ", ".join(f"`{e}`" for e in o["reversed_examples"]))
    L.append("")
    L.append("> Clean full-reversals are rare here, but ~1 in 3 Latin/digit runs is misread in "
             "some way — the dominant bilingual failure is character-level misreading of "
             "embedded codes/numbers, not whole-token reversal.\n")

    # ── Worst cases ──
    L.append("## 6. Worst 15 images (highest CER, successful only)\n")
    L.append("| # | File | CER | Ground truth → Prediction |")
    L.append("|--:|---|--:|---|")
    for i, x in enumerate(worst, 1):
        L.append(f"| {i} | {x['filename']} | {pct(x['cer'])} | "
                 f"**GT:** {esc(x['ref_text'])[:300]} <br> **PR:** {esc(x['pred_text'])[:300]} |")
    L.append("")

    # ── No-output images ──
    if errored or empties:
        L.append("## 7. No-output failures\n")
        if errored:
            L.append("**HTTP 422 (model looped into unterminated JSON):** the same repetition "
                     "failure seen on the original sample letter before `IMAGE_MIN_TOKENS` was "
                     "raised; it still triggers occasionally on denser bilingual content.\n")
            L.append("| File | Error (truncated) |")
            L.append("|---|---|")
            for x in errored:
                L.append(f"| {x['filename']} | {esc(x['error'])[:160]} |")
            L.append("")
        if empties:
            L.append("**Empty (200 OK, all fields null):** the model returned a valid but empty "
                     "extraction — it declined to read the letter at all.\n")
            L.append("| File | Source |")
            L.append("|---|---|")
            for x in empties:
                L.append(f"| {x['filename']} | {x['source']} |")
            L.append("")

    # ── Confusion ──
    L.append("## 8. Character confusion summary\n")
    L.append("Most frequent character substitutions (ground-truth → predicted), after "
             "normalization:\n")
    L.append("| ref → pred | count |")
    L.append("|---|--:|")
    for pair, cnt in res["confusion_top"][:25]:
        L.append(f"| `{esc(pair)}` | {cnt} |")
    L.append("")
    seen_pairs = {p for p, _ in res["confusion_top"]}
    hits = []
    for wp in WATCH_PAIRS:
        a1, b1 = wp.split("/")
        if f"{a1} -> {b1}" in seen_pairs or f"{b1} -> {a1}" in seen_pairs:
            hits.append(wp)
    if hits:
        L.append(f"Commonly-confused Persian pairs actually observed: "
                 + ", ".join(f"`{h}`" for h in hits) + ".\n")
    else:
        L.append("None of the classic dot-only Persian confusions (ب/پ, ج/چ, …) dominated; "
                 "errors are spread across many characters.\n")

    # ── Conclusion ──
    L.append("## 9. Plain-language conclusion\n")
    fa_cer = by["synthetic_fa"]["cer"]
    bi_cer = by["synthetic_bilingual"]["cer"]
    L.append(f"- **Printed Persian prose is the model's strong suit** — CER {pct(fa_cer)} on "
             f"Persian-only letters, and only {pct(lb['persian_segment_cer'])} on the Persian "
             f"runs inside bilingual letters.  ")
    L.append(f"- **Embedded Latin/digit content is the weak spot** — {pct(lb['latin_digit_segment_cer'])} "
             f"CER on codes, dates, emails and account numbers, with ~{pct(o['other_rate'])} of "
             f"such runs misread and occasional digit reversal.  ")
    L.append(f"- **Robustness:** {pct(res['no_output_rate'])} of letters return no usable text "
             f"({res['n_errors']} as HTTP 422, {res['n_empty']} as all-null 200). For a "
             f"production path, catching the 422 / empty response and retrying with a higher "
             f"token budget would remove most of these.  ")
    L.append(f"- **Normalization:** report digit-normalized CER as the headline, but the raw vs. "
             f"normalized gap is small here ({pct(ns['raw']['cer'])} → {pct(ns['persian_plus_digits']['cer'])}), "
             f"so these numbers are not an artifact of aggressive normalization.\n")
    L.append("### How to re-run\n")
    L.append("```bash")
    L.append("# regenerate images (optional)")
    L.append("venv312\\Scripts\\python.exe ocr_eval\\generate_dataset.py")
    L.append("# run inference + score (server must be running)")
    L.append("venv312\\Scripts\\python.exe ocr_eval\\run_eval.py")
    L.append("# rebuild this report from the last run")
    L.append("venv312\\Scripts\\python.exe ocr_eval\\make_report.py")
    L.append("```")

    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
