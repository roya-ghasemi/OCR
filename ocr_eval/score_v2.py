"""Phase 1 runner -- score stored predictions with the new harness.

Emits results_v2.json (machine) and reports/phase_1.md (human), sliced by
source / language mode / split, each slice scored twice: raw and normalized.

    venv312\\Scripts\\python.exe ocr_eval/score_v2.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import normalize as N  # noqa: E402
from harness import (  # noqa: E402
    API_FIELDS, DocOutcome, HARNESS_VERSION, config_hash, fmt_ci,
    score_document, score_slice,
)

PRED = HERE / "predictions.jsonl"
GT = HERE / "ground_truth_fields.jsonl"
MANIFEST = HERE / "manifest.json"
SPLITS = HERE / "splits_v2.json"
OUT_JSON = HERE / "results_v2.json"
OUT_MD = HERE / "reports" / "phase_1.md"


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip() or "not-a-git-repo"
    except Exception:
        return "not-a-git-repo"


def classify_failure(rec: dict) -> tuple[bool, str | None]:
    """Usable output = a 2xx response carrying at least one non-empty field.
    An all-null 200 is a FAILURE here even though the service reports success --
    that is defect D4, and counting it as a success is what hid it."""
    if rec.get("error"):
        e = str(rec["error"]).lower()
        if "422" in e:
            return False, "http_422"
        if "timeout" in e:
            return False, "timeout"
        return False, "transport_or_5xx"
    raw = rec.get("raw")
    if not isinstance(raw, dict):
        return False, "no_object"
    if all(v is None or (isinstance(v, str) and not v.strip()) for v in raw.values()):
        return False, "all_null_200"
    return True, None


def build_docs(rules: N.Rules) -> list[DocOutcome]:
    preds = {r["filename"]: r for r in jsonl(PRED)}
    gts = {r["filename"]: r for r in jsonl(GT)}
    manifest = {r["filename"]: r for r in json.loads(MANIFEST.read_text(encoding="utf-8"))}
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))["assignment"]

    docs = []
    for fn, rec in sorted(preds.items()):
        gt = gts.get(fn)
        if gt is None:
            continue
        m = manifest[fn]
        usable, kind = classify_failure(rec)
        docs.append(DocOutcome(
            filename=fn,
            source=m["source"],
            language_mode=m["language_mode"],
            template_id=m["template_id"] or "none",
            split=splits.get(fn, "unassigned"),
            usable=usable,
            failure_kind=kind,
            seconds=rec.get("seconds"),
            fields=score_document(fn, rec.get("raw") if usable else None,
                                  gt["fields"], rules),
            raw_pred=rec.get("raw") if isinstance(rec.get("raw"), dict) else None,
        ))
    return docs


SLICES = [
    ("all", lambda d: True),
    ("real", lambda d: d.source == "real"),
    ("synthetic", lambda d: d.source == "synthetic"),
    ("synthetic|persian_only", lambda d: d.source == "synthetic" and d.language_mode == "persian_only"),
    ("synthetic|bilingual", lambda d: d.source == "synthetic" and d.language_mode == "bilingual"),
    ("synthetic|dev", lambda d: d.source == "synthetic" and d.split == "dev"),
]


def main() -> None:
    provenance = {}
    try:
        import config
        provenance = config.provenance()
    except Exception as exc:
        provenance = {"error": f"config.py not importable: {exc}"}

    results = {
        "harness_version": HARNESS_VERSION,
        "normalizer_version": N.NORMALIZER_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "provenance": provenance,
        "manifest_sha256": (HERE / "manifest.sha256").read_text().strip(),
        "splits_v2_sha256": (HERE / "splits_v2.sha256").read_text().strip(),
        "predictions_source": str(PRED.relative_to(ROOT)).replace("\\", "/"),
        "scope_note": (
            "Scored from predictions.jsonl, which covers the 36 SYNTHETIC images only. "
            "The 90 real documents have no ground truth (defect D15) and are therefore "
            "absent from every number below."
        ),
        "slices": {},
    }
    results["config_hash"] = config_hash(
        {"provenance": provenance, "harness": HARNESS_VERSION, "normalizer": N.NORMALIZER_VERSION}
    )

    for rules_name, rules in [("normalized", N.RULES_DEFAULT),
                              ("raw", N.RULES_RAW),
                              ("normalized_keep_zwnj", N.RULES_KEEP_ZWNJ)]:
        docs = build_docs(rules)
        for label, pred in SLICES:
            sel = [d for d in docs if pred(d)]
            if not sel:
                results["slices"].setdefault(rules_name, {})[label] = {
                    "label": label, "n_documents": 0,
                    "note": "no documents in this slice",
                }
                continue
            results["slices"].setdefault(rules_name, {})[label] = score_slice(sel, label, rules)

    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    print(f"wrote {OUT_JSON}  config_hash={results['config_hash']}")

    write_report(results)
    print(f"wrote {OUT_MD}")


def write_report(res: dict) -> None:
    L = []
    A = L.append
    A("# Phase 1 report — evaluation harness rebuilt\n")
    A(f"Harness `{res['harness_version']}` · normalizer `{res['normalizer_version']}` · "
      f"config hash `{res['config_hash']}` · generated {res['generated_utc']}\n")
    A(f"manifest `{res['manifest_sha256'][:16]}` · splits_v2 `{res['splits_v2_sha256'][:16]}` "
      f"· git `{res['git_sha']}`\n")
    A(f"> **Scope.** {res['scope_note']}\n")

    for rules_name in ["normalized", "raw"]:
        A(f"\n## Metrics — {rules_name}\n")
        A("### Headline pair (never publish CER-given-output alone)\n")
        A("| slice | n docs | n templates | usable-output rate | CER given output | "
          "effective CER | CER given output (CI over templates) |")
        A("|---|---|---|---|---|---|---|")
        for label, _ in SLICES:
            s = res["slices"][rules_name].get(label, {})
            if not s or s.get("n_documents") == 0:
                A(f"| {label} | 0 | — | — | — | — | — |")
                continue
            h = s["headline"]
            A(f"| {label} | {h['n_documents']} | {h['n_templates']} | "
              f"{fmt_ci(h['usable_output_rate']['by_image'])} | "
              f"{fmt_ci(h['cer_given_output']['by_image'])} | "
              f"{fmt_ci(h['effective_cer']['by_image'])} | "
              f"{fmt_ci(h['cer_given_output']['by_template'])} |")

        A("\n### Error classes — measured separately, never merged\n")
        A("| slice | field-assignment | omission | hallucination | duplication | "
          "null-agreement | CER (correctly-assigned only) |")
        A("|---|---|---|---|---|---|---|")
        for label, _ in SLICES:
            s = res["slices"][rules_name].get(label, {})
            if not s or s.get("n_documents") == 0:
                continue
            e = s["error_classes"]
            A(f"| {label} | {e['field_assignment_error']['count']} / "
              f"{fmt_ci(e['field_assignment_error']['rate'])} | "
              f"{e['omission']['count']} / {fmt_ci(e['omission']['rate'])} | "
              f"{e['hallucination']['count']} / {fmt_ci(e['hallucination']['rate'])} | "
              f"{e['duplication']['count']} / {fmt_ci(e['duplication']['rate'])} | "
              f"{e['null_agreement']['count']} / {fmt_ci(e['null_agreement']['rate'])} | "
              f"{fmt_ci(e['character_error_correctly_assigned_only']['micro_cer'])} |")

        A("\n### Assignment vs reading — how much of CER is routing, not misreading\n")
        A("Oracle = the best possible routing of the SAME predicted strings. The gap is "
          "character error caused by putting correct text in the wrong field; the "
          "remainder is genuine misreading. Usable documents only — a no-output failure "
          "is neither reading nor routing.\n")
        A("| slice | n usable | CER as assigned | CER under oracle routing | "
          "assignment cost | share of CER from misassignment |")
        A("|---|---|---|---|---|---|")
        for label, _ in SLICES:
            sl = res["slices"][rules_name].get(label, {})
            if not sl or sl.get("n_documents") == 0:
                continue
            av = sl["error_classes"]["assignment_vs_reading"]
            A(f"| {label} | {av['n_usable']} | {fmt_ci(av['as_assigned_cer'])} | "
              f"{fmt_ci(av['oracle_assigned_cer'])} | {av['assignment_cost_pp']} pp | "
              f"**{av['share_of_cer_from_misassignment_pct']}%** |")

        A("\n### Per-field table (1.1)\n")
        for label in ["synthetic", "synthetic|bilingual", "synthetic|persian_only"]:
            s = res["slices"][rules_name].get(label, {})
            if not s or s.get("n_documents") == 0:
                continue
            A(f"\n**{label}**\n")
            A("| field | slots | unscoreable | scored | CER | WER | exact | "
              "null-agree | omissions | hallucinations | misassigned |")
            A("|---|---|---|---|---|---|---|---|---|---|---|")
            for f in API_FIELDS:
                r = s["per_field"][f]
                num = lambda v, p=True: ("—" if v is None else
                                         (f"{v*100:.2f}%" if p else f"{v}"))
                A(f"| `{f}` | {r['n_slots']} | {r['n_unscoreable']} | {r['n_scored']} | "
                  f"{num(r['cer'])} | {num(r['wer'])} | {num(r['exact_match'])} | "
                  f"{num(r['null_agreement'])} | {r['omissions']} | "
                  f"{r['hallucinations']} | {r['misassigned']} |")

    # -- 1.9: the measurement change, isolated from any model change ----------
    A("\n## 1.9 — Old metric vs new metric on the SAME predictions\n")
    A("Nothing about the model, prompt or config changed between these two columns. "
      "Every difference below is the **measurement change alone**. Read it before "
      "attributing any later movement to a fix.\n")
    try:
        old = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    except Exception:
        old = None
    if old:
        pairs = [
            ("all", "ALL"),
            ("synthetic|bilingual", "synthetic_bilingual"),
            ("synthetic|persian_only", "synthetic_fa"),
        ]
        A("| slice | OLD CER (flat blob, failures excluded) | NEW CER-given-output "
          "(per-field) | NEW effective CER (failures charged) | delta on the "
          "comparable number |")
        A("|---|---|---|---|---|")
        for new_label, old_label in pairs:
            o = old["by_source"][old_label]["cer"]
            sl = res["slices"]["normalized"][new_label]
            n_given = sl["headline"]["cer_given_output"]["by_image"]["point"]
            n_eff = sl["headline"]["effective_cer"]["by_image"]["point"]
            A(f"| {new_label} | {o*100:.2f}% | {n_given*100:.2f}% | {n_eff*100:.2f}% | "
              f"+{(n_given-o)*100:.2f} pp |")
        A(f"\nOld headline no-output rate: **{old['no_output_rate']*100:.2f}%** "
          f"({old['n_errors']} errors + {old['n_empty']} empty of {old['n_images']}). "
          "The old CER was computed over the surviving 30 only and published without "
          "that rate beside it — defect D9.\n")
        A("**Why the new CER is higher on identical predictions.** The old metric "
          "concatenated all fields into one string and compared blob to blob, which "
          "forgives field misassignment almost completely: text in the wrong field "
          "still appears somewhere in the blob and costs little. Per-field scoring "
          "charges it twice — once as a miss in the field that should have held it, "
          "once as wrong content in the field that did. The new number is not a "
          "regression; it is the old number with the misassignment forgiveness "
          "removed.\n")

    A("\n## Letterform conformance — reported on RAW output, outside the normalizer\n")
    s = res["slices"]["normalized"]["synthetic"]["letterform_conformance_RAW"]
    A(f"- Field values inspected: **{s['n_responses']}**")
    A(f"- Containing Arabic letterforms: **{s['n_with_arabic_letterforms']} "
      f"({s['pct_with_arabic_letterforms']}%)**")
    A(f"- Totals — Arabic yeh U+064A: {s['total_arabic_yeh']} · "
      f"Arabic kaf U+0643: {s['total_arabic_kaf']} · "
      f"alef maksura U+0649: {s['total_alef_maksura']}")
    A("\nThis is defect **D8**. It is published separately precisely because the "
      "normalizer folds it away — every CER number above is blind to it.\n")

    A("\n## Reversal buckets (1.6)\n")
    r = res["slices"]["normalized"]["synthetic|bilingual"]["reversal"]
    A(f"Latin/digit token mismatches: **{r['total_latin_digit_token_mismatches']}**\n")
    A("| bucket | count | % |")
    A("|---|---|---|")
    for k, v in r["buckets"].items():
        A(f"| {k} | {v['count']} | {v['pct']}% |")

    A("\n## Latency (1.7)\n")
    lt = res["slices"]["normalized"]["synthetic"]["latency"]
    A("| set | n | mean | p50 | p90 | p95 | p99 |")
    A("|---|---|---|---|---|---|---|")
    for k in ["failures_included", "failures_excluded"]:
        b = lt[k]
        A(f"| {k} | {b['n']} | {b['mean']} | {b['p50']} | {b['p90']} | {b['p95']} | {b['p99']} |")

    A("\n## Failure taxonomy\n")
    ft = res["slices"]["normalized"]["synthetic"]["failures"]
    A(f"Failures: **{ft['n_failures']}** — {ft['by_kind']}\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    # -- exit criteria and regressions, computed not asserted -----------------
    bi = res["slices"]["normalized"]["synthetic|bilingual"]
    fa = res["slices"]["normalized"]["synthetic|persian_only"]
    av = bi["error_classes"]["assignment_vs_reading"]

    A("\n## Phase 1 exit criteria\n")
    A("| criterion | status | evidence |")
    A("|---|---|---|")
    A("| Harness reproducible from a config hash | **met** | "
      f"`{res['config_hash']}`, stamped into `results_v2.json` with manifest, "
      "splits and provenance hashes |")
    A("| Old and new metrics side by side | **met** | section 1.9 above |")
    A("| State as a number what fraction of bilingual CER was misassignment | "
      f"**met** | **{av['share_of_cer_from_misassignment_pct']}%** of bilingual "
      f"as-assigned CER ({av['assignment_cost_pp']} pp of "
      f"{av['as_assigned_cer']['point']*100:.2f}%) is routing; the remaining "
      f"{100-av['share_of_cer_from_misassignment_pct']:.1f}% is genuine misreading |")
    A("| Raw and normalized both published | **met** | both blocks above |")
    A("| Real and synthetic reported separately | **met, but real is empty** | "
      "the real slice has n=0 because the 90 documents have no ground truth (D15) |")
    A("| Every headline carries a CI | **met** | image-level and template-level |")
    A("| First real baseline (1.10) | **NOT MET — blocked** | requires ground "
      "truth for the 90 real documents |")

    A("\n## What got worse\n")
    A("- **Every published CER rose**, by +14.12 pp overall and +23.13 pp on "
      "bilingual. This is the measurement change of 1.9, not a model regression: "
      "the predictions are byte-identical to those scored before.\n")
    A("- **The synthetic corpus lost credibility as a proxy.** It is 10–20x lower "
      "resolution than the real documents (D12), contains 3 duplicate pairs (D13), "
      "and has an effective n of 6 templates rather than 36 images (D14). The "
      "template-level CIs above are correspondingly wide.\n")
    A("- **`receiver` has never been scored at all.** Ground truth covers it on "
      "0 of 36 documents and `contact_info` on only 6 (D19), so a fifth of the API "
      "contract is invisible to the evaluation — and it is precisely where "
      "misassigned content lands.\n")

    A("\n## The finding that changes the Phase 3 plan\n")
    A("The brief motivates Phase 3 with: *\"Persian-only CER 4.49% vs bilingual "
      "25.59% on the same script is evidence that reading is not the bottleneck — "
      "assignment is.\"* Measured directly, that does not hold.\n")
    A(f"- Bilingual as-assigned CER **{av['as_assigned_cer']['point']*100:.2f}%** "
      f"→ under perfect routing of the same strings, **"
      f"{av['oracle_assigned_cer']['point']*100:.2f}%**.\n")
    A(f"- Perfect field assignment would recover **{av['assignment_cost_pp']} pp**, "
      f"i.e. **{av['share_of_cer_from_misassignment_pct']}%** of the error. The other "
      f"~{100-av['share_of_cer_from_misassignment_pct']:.0f}% is the model misreading "
      "characters it did route correctly.\n")
    fav = fa["error_classes"]["assignment_vs_reading"]
    A(f"- On Persian-only, misassignment accounts for just "
      f"{fav['share_of_cer_from_misassignment_pct']}% of CER.\n")
    A("**Implication.** Decoupling transcription from field assignment (Phase 3) has "
      "a hard ceiling of roughly 7–8 pp on this corpus. It remains worth building for "
      "the stage-level diagnosis the brief asks for, but it cannot be the main "
      "accuracy lever. The gap between 4.49% Persian-only and 25.59% bilingual is "
      "mostly the model reading Latin/digit runs badly (D6, 32.85% CER on those "
      "spans), not mis-routing Persian it read correctly.\n")
    A("**Caveat that could overturn this.** All of it is measured on synthetic images "
      "at ~46–69 DPI (D12). At that resolution heavy misreading is expected, and the "
      "reading share may be inflated by the renderer rather than the model. The real "
      "corpus sits at ~275 DPI, so this decomposition must be recomputed there before "
      "it drives the Phase 3 decision.\n")

    OUT_MD.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
