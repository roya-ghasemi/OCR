"""
Phase 0 regression guards. Run with: venv312\\Scripts\\python.exe -m pytest ocr_eval -q

These lock in the Phase 0 decisions so a later change cannot silently undo them:

  * the dev/test split is frozen and hash-locked          (0.7)
  * no lexicon/dictionary stage is wired into the runtime (0.3 policy)
  * no prompt example value can leak into a prediction    (0.4 policy)
  * the model identity stays pinned                       (0.6)
"""
import hashlib
import json
import re
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPLITS = HERE / "splits.json"
MAIN = ROOT / "main.py"

# Locked 2026-09-01. The split is frozen by policy: tuning on dev must never
# transfer to test. If this hash changes, the split was regenerated and every
# number measured against it is void.
SPLITS_SHA256 = "6d558226b3ea2b9058042a3b72301d0dc61e0ae45aa7770942e32f1266a32489"


# ── 0.7 — the split is frozen ─────────────────────────────────────────────────
def test_splits_file_hash_is_unchanged():
    actual = hashlib.sha256(SPLITS.read_bytes()).hexdigest()
    assert actual == SPLITS_SHA256, (
        f"ocr_eval/splits.json changed (sha256 {actual}).\n"
        "The split is frozen. Every metric measured before this change is void.\n"
        "If the change is deliberate, record why in ocr_eval/experiments.md, "
        "re-baseline, and update SPLITS_SHA256."
    )


def test_no_ground_truth_text_spans_both_splits():
    """The 36 images are only 18 distinct texts. All renders of one text must stay
    in the same split or dev tuning leaks into test."""
    s = json.loads(SPLITS.read_text(encoding="utf-8"))
    gt = {}
    for line in (HERE / "ground_truth.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            gt[r["filename"]] = hashlib.sha1(r["text"].encode()).hexdigest()[:10]

    dev = {gt[f] for f in s["dev"]}
    test = {gt[f] for f in s["test"]}
    overlap = dev & test
    assert not overlap, f"text group(s) present in BOTH splits: {sorted(overlap)}"
    assert set(s["dev"]) & set(s["test"]) == set(), "an image is in both splits"
    assert len(s["dev"]) + len(s["test"]) == len(gt), "split does not cover every image"


# ── 0.3 — no lexicon stage may return to the runtime ──────────────────────────
LEXICON_TOKENS = re.compile(
    r"dehkhoda|corrector|lexicon|levenshtein|edit_distance|rapidfuzz|difflib|"
    r"spell|دهخدا",
    re.IGNORECASE,
)

RUNTIME_FILES = ["main.py", "engine.py", "config.py"]


def _executable_source(path: Path) -> str:
    """Source with comments and docstrings stripped out.

    The guard is about what the runtime DOES, not what its prose mentions. A
    comment explaining why the Dehkhoda stage was removed is documentation, and
    must not fail the very test that keeps it removed.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))

    # Blank every string constant. Wiring a corrector back in requires an import,
    # an attribute access or a call -- never a bare string. Text-substituting the
    # literals out does not work: Python folds adjacent literals into one
    # Constant at parse time, so the folded value never matches the source. Going
    # through the AST and unparsing sidesteps that, and drops comments and
    # docstrings for free.
    class _Blank(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value=""), node)
            return node

    stripped = ast.fix_missing_locations(_Blank().visit(tree))
    return ast.unparse(stripped)


@pytest.mark.parametrize("name", RUNTIME_FILES)
def test_no_lexicon_stage_in_runtime(name):
    """Phase 0.3 removed the Dehkhoda stage. Reintroducing any lexicon/spell
    correction requires the four preconditions in dehkhoda/REMOVED.md, including a
    pre-registered A/B whose CI excludes zero. Failing this test is the reminder."""
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    hits = LEXICON_TOKENS.findall(_executable_source(path))
    assert not hits, (
        f"{name} executes a lexicon/correction stage: {sorted(set(h.lower() for h in hits))}.\n"
        "See dehkhoda/REMOVED.md for the conditions under which one may return."
    )


def test_corrected_endpoint_runs_no_correction_stage():
    """`/ocr/corrected` exists again, but only as a deprecation passthrough for
    out-of-repo callers that would otherwise get a 404 (defect D11).

    The invariant worth guarding was never the absence of the route -- it is the
    absence of a correction stage behind it. Asserting the route is gone made the
    test fail for a change that is safe, while a corrector smuggled into `/ocr`
    itself would have passed. This checks the thing that matters.
    """
    import ast

    src = _executable_source(MAIN)
    assert "correct_text" not in src, "a corrector call reappeared in the runtime"
    assert "_get_corrector" not in src, "the lazy corrector loader reappeared"

    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    handler = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == "extract_text_corrected"),
        None,
    )
    if handler is None:
        return  # route fully removed; nothing left to guard
    body = ast.unparse(handler)
    assert "deprecated" in body, "the passthrough must declare itself deprecated"
    assert "_run_ocr" in body, "the passthrough must go through the same OCR path"


# ── 0.4 — the prompt must not seed values that can appear in output ───────────
def test_prompt_contains_no_concrete_example_values():
    """The `Roya Ghasemi` investigation concluded the prompt is example-free (the
    name comes from the rendered page, not the prompt). Keep it that way: a few-shot
    example in the system prompt is a fabricated-identity risk, because the model can
    emit the example when it cannot read the document.

    The prompt legitimately contains a JSON skeleton with <angle-bracket> Persian
    field descriptions. What it must not contain is a filled-in value."""
    src = MAIN.read_text(encoding="utf-8")
    m = re.search(r'_SYSTEM_PROMPT\s*=\s*"""(.*?)"""', src, re.S)
    assert m, "_SYSTEM_PROMPT not found in main.py"
    prompt = m.group(1)

    # No Latin-script personal/company-looking name, and no digit run long enough to
    # be an ID, phone, invoice or date. Both are the shapes a leak would take.
    latin_names = re.findall(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b", prompt)
    assert not latin_names, f"prompt contains a concrete name-like value: {latin_names}"

    digit_runs = [d for d in re.findall(r"\d{3,}", prompt)]
    assert not digit_runs, f"prompt contains a concrete numeric value: {digit_runs}"


def test_no_prediction_contains_a_value_absent_from_its_ground_truth():
    """Regression guard for fabricated identities. For every stored prediction, any
    name token it emits must be present in that image's ground truth.

    This is the test that settles D2 empirically and keeps it settled."""
    pred_path = HERE / "predictions.jsonl"
    if not pred_path.exists():
        pytest.skip("no predictions.jsonl to check")

    gt = {}
    for line in (HERE / "ground_truth.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            gt[r["filename"]] = r["text"]

    # Every proper noun the corpus can legitimately contain.
    NAME_TOKENS = ["Roya", "Ghasemi", "Ali", "Rezaei", "Maryam",
                   "رویا", "قاسمی", "علی", "رضایی", "مریم"]

    offenders = []
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        text = p.get("pred_text") or ""
        ref = gt.get(p["filename"], "")
        for tok in NAME_TOKENS:
            if tok in text and tok not in ref:
                offenders.append((p["filename"], tok))

    assert not offenders, (
        "prediction contains a name absent from its ground truth "
        f"(fabricated identity): {offenders}"
    )


# ── 0.6 — the model stays pinned ──────────────────────────────────────────────
def test_model_identity_is_pinned():
    import sys
    sys.path.insert(0, str(ROOT))
    import config

    prov = config.provenance()
    assert prov["model_id"] == "coreOCR-7B-050325-preview.Q4_K_S.gguf"
    assert prov["mmproj_sha256"] == (
        "34933952a9ae2f1ac7af7a908189b3fc22106d3999fd3fae43b520ac4d308a78")
    # Decoding must stay deterministic, or no A/B in this project means anything.
    assert prov["temperature"] == 0.0, "temperature must stay 0.0 for reproducibility"
