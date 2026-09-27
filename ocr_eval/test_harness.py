"""Guard tests for the Phase 1 harness.

These exist because two harness bugs (D21, D22) each produced a plausible-looking
number that pointed the whole project in the wrong direction. Each test below
pins an invariant one of those bugs violated.

Run: venv312\\Scripts\\python.exe -m pytest ocr_eval/test_harness.py -q
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import normalize as N  # noqa: E402
import harness as H  # noqa: E402


def doc(fields, gt, usable=True, seconds=1.0, template="t1", source="synthetic"):
    return H.DocOutcome(
        filename="x.png", source=source, language_mode="bilingual",
        template_id=template, split="dev", usable=usable, failure_kind=None,
        seconds=seconds,
        fields=H.score_document("x.png", fields if usable else None, gt, N.RULES_DEFAULT),
        raw_pred=fields,
    )


# -- D22: assignment errors must be visible even in unscoreable fields --------

def test_assignment_error_detected_when_gt_lacks_that_field():
    """The canonical D5 symptom: subject text dumped into `receiver`, which the
    ground truth never covers. Scoring only `scored` rows reported zero."""
    gt = {"sender": "شرکت الف", "subject": "درخواست تمدید قرارداد"}
    pred = {"sender": "شرکت الف", "receiver": "درخواست تمدید قرارداد"}
    d = doc(pred, gt)
    recv = next(f for f in d.fields if f.field == "receiver")
    assert recv.outcome == "unscoreable"
    assert recv.misassigned_to == "subject", "assignment error must be caught here"
    assert H.error_classes([d])["field_assignment_error"]["count"] == 1


def test_unscoreable_is_not_scored_as_hallucination():
    gt = {"sender": "شرکت الف"}
    d = doc({"sender": "شرکت الف", "receiver": "چیزی"}, gt)
    assert H.error_classes([d])["hallucination"]["count"] == 0
    assert H.error_classes([d])["content_in_unscoreable_field"]["count"] == 1


# -- D21: the oracle must be a lower bound -----------------------------------

def test_oracle_never_exceeds_as_assigned_on_swapped_fields():
    gt = {"sender": "الف الف الف", "subject": "ب ب ب ب"}
    swapped = {"sender": "ب ب ب ب", "subject": "الف الف الف"}
    d = doc(swapped, gt)
    a = H._as_assigned_cer([d])
    o = H._oracle_cer([d])
    assert o <= a
    assert o == pytest.approx(0.0, abs=1e-9), "a pure swap is free under oracle routing"
    assert H._assignment_share([d]) == pytest.approx(100.0, abs=0.01)


def test_oracle_equals_as_assigned_when_routing_is_already_optimal():
    gt = {"sender": "الف", "subject": "ب"}
    d = doc({"sender": "الف", "subject": "ب"}, gt)
    assert H._oracle_cer([d]) == H._as_assigned_cer([d])
    assert H._assignment_share([d]) is None or H._assignment_share([d]) == 0.0


def test_pure_misreading_is_not_attributed_to_assignment():
    gt = {"sender": "الفالفالف", "subject": "بببب"}
    d = doc({"sender": "xxxxxxxxx", "subject": "بببب"}, gt)
    assert H._pp_gap([d]) == 0.0, "no routing gain is available; this is all reading"


# -- null vs missing vs omission ---------------------------------------------

def test_gt_present_but_null_is_null_agreement_not_omission():
    d = doc({"sender": "الف"}, {"sender": "الف", "subject": None})
    subj = next(f for f in d.fields if f.field == "subject")
    assert subj.outcome == "null_agree"


def test_gt_present_with_content_and_no_prediction_is_omission():
    d = doc({"sender": "الف"}, {"sender": "الف", "subject": "ب"})
    subj = next(f for f in d.fields if f.field == "subject")
    assert subj.outcome == "omission"


def test_gt_null_but_prediction_present_is_hallucination():
    d = doc({"sender": "الف", "subject": "ب"}, {"sender": "الف", "subject": None})
    subj = next(f for f in d.fields if f.field == "subject")
    assert subj.outcome == "hallucination"


def test_missing_gt_key_and_null_gt_value_are_different_outcomes():
    """2.2: conflating these makes omission unmeasurable."""
    missing = doc({"sender": "الف"}, {"sender": "الف"})
    null = doc({"sender": "الف"}, {"sender": "الف", "subject": None})
    assert next(f for f in missing.fields if f.field == "subject").outcome == "unscoreable"
    assert next(f for f in null.fields if f.field == "subject").outcome == "null_agree"


# -- D9: the headline pair -----------------------------------------------------

def test_failed_document_costs_full_effective_cer_but_is_absent_from_cer_given_output():
    gt = {"sender": "الفالفالف"}
    good = doc({"sender": "الفالفالف"}, gt)
    bad = doc(None, gt, usable=False)
    h = H.headline([good, bad])
    assert h["usable_output_rate"]["by_image"]["point"] == pytest.approx(0.5)
    assert h["cer_given_output"]["by_image"]["point"] == pytest.approx(0.0)
    assert h["effective_cer"]["by_image"]["point"] == pytest.approx(0.5)


def test_duplication_is_detected_across_two_fields():
    gt = {"subject": "درخواست تمدید قرارداد پشتیبانی"}
    d = doc({"subject": "درخواست تمدید قرارداد پشتیبانی",
             "body_text": "درخواست تمدید قرارداد پشتیبانی"}, gt)
    assert H.error_classes([d])["duplication"]["count"] >= 1


# -- 1.5: cross-script substitutions are not confusion signal ------------------

def test_confusion_separates_cross_script_substitutions():
    gt = {"sender": "ابجد"}
    d = doc({"sender": "abcd"}, gt)
    c = H.confusion([d])
    assert c["same_script"] == []
    assert len(c["cross_script_ALIGNMENT_NOISE"]) > 0


# -- 1.8: CIs ------------------------------------------------------------------

def test_bootstrap_ci_brackets_the_point_estimate():
    docs = [doc({"sender": "الف"}, {"sender": "الف"}) for _ in range(10)]
    docs += [doc(None, {"sender": "الف"}, usable=False) for _ in range(10)]
    ci = H.bootstrap_ci(docs, lambda ds: H._rate(ds, lambda d: d.usable))
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["n_units"] == 20


def test_template_level_ci_is_wider_than_image_level_when_templates_are_few():
    """D14: 36 images from 6 templates must not be reported as n=36."""
    docs = []
    for t in range(3):
        for _ in range(12):
            docs.append(doc({"sender": "الف"}, {"sender": "الف"},
                            usable=(t != 0), template=f"t{t}"))
    h = H.headline(docs)
    assert h["n_templates"] == 3
    img = h["usable_output_rate"]["by_image"]
    tmpl = h["usable_output_rate"]["by_template"]
    assert (tmpl["hi"] - tmpl["lo"]) >= (img["hi"] - img["lo"])


# -- dataset integrity ---------------------------------------------------------

def test_splits_v2_hash_is_locked():
    """splits_v2.json must never be regenerated: every comparison is against it."""
    import hashlib
    actual = hashlib.sha256((HERE / "splits_v2.json").read_bytes()).hexdigest()
    expected = (HERE / "splits_v2.sha256").read_text().strip()
    assert actual == expected, "splits_v2.json changed; comparisons are invalidated"


def test_synthetic_and_real_are_never_globbed_from_one_directory():
    m = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    dirs = {r["source"]: r["corpus_dir"] for r in m}
    assert dirs["synthetic"] == "images" and dirs["real"] == "dataset_ex"
    assert sum(1 for r in m if r["source"] == "synthetic") == 36
    assert sum(1 for r in m if r["source"] == "real") == 90


def test_both_language_modes_appear_in_the_test_split():
    """A stratification bug once emptied the Persian-only test split entirely."""
    m = {r["filename"]: r for r in json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))}
    sp = json.loads((HERE / "splits_v2.json").read_text(encoding="utf-8"))["assignment"]
    test_modes = {m[f]["language_mode"] for f, s in sp.items()
                  if s == "test" and m[f]["source"] == "synthetic"}
    assert test_modes == {"bilingual", "persian_only"}
