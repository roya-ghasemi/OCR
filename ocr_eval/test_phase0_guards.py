"""
Phase 0 regression guards. Run with: venv312\\Scripts\\python.exe -m pytest ocr_eval -q

These lock in the Phase 0 decisions so a later change cannot silently undo them:

  * the dev/test split is frozen and hash-locked          (0.7)
  * no lexicon/dictionary stage is wired into the runtime (0.3 policy)
  * no language model is wired into the runtime           (E19: nothing may be generated)
  * no stored prediction invents a name                   (0.4 policy)
  * the recognition engine identity stays pinned          (0.6, E19)
"""
import hashlib
import json
import re
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPLITS = HERE / "splits.json"

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
# `spellfix` is the ONE exception, admitted in E24 under the four conditions in
# dehkhoda/REMOVED.md: it repairs dot confusions only (a word is replaced solely by a
# word that is identical once dots are folded away), never changes a letter's shape or a
# word's length, never touches digits or Latin, leaves any word the reader was sure of,
# and acts only where exactly ONE lexicon word matches. Everything else stays blocked —
# a general edit-distance corrector is what damaged 47 of 49 tokens in the 2026-09-01
# audit, and what reproduced as 11 corruptions in D77.
LEXICON_TOKENS = re.compile(
    r"dehkhoda|corrector|lexicon|levenshtein|edit_distance|rapidfuzz|difflib|"
    r"spell(?!fix|_?fix)|دهخدا",
    re.IGNORECASE,
)

# The serving path since E19 (the LLM path - main.py, engine.py, ocr_pipeline/ - is gone).
RUNTIME_FILES = ["ocr_service/transcribe.py", "ocr_service/letter_fields.py", "ocr_service/pipeline.py",
                 "ocr_service/api.py", "ocr_service/config.py", "ocr_service/tasks.py",
                 "ocr_service/spellfix.py"]


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


# -- E19 - nothing in the runtime can generate text ------------------------------
LLM_TOKENS = re.compile(
    r"openai|llama|vllm|triton|chat/completions|completions|LetterExtractor|gguf|mmproj|"
    r"ollama|transformers|anthropic|grammar|prompt",
    re.IGNORECASE,
)


@pytest.mark.parametrize("name", RUNTIME_FILES)
def test_no_language_model_in_runtime(name):
    """Every character of the output must be recognised from pixels (Tesseract or the
    digit CNN). A language model in this path is what invented senders, receivers and
    subjects for pages that had none, and dropped paragraphs past a grammar cap (E19).
    Bringing one back is a design decision to be measured, not a quiet import."""
    path = ROOT / name
    assert path.exists(), f"{name} missing"
    hits = LLM_TOKENS.findall(_executable_source(path))
    assert not hits, f"{name} wires a language model into the runtime: {sorted(set(h.lower() for h in hits))}"


def test_the_llm_serving_path_is_gone():
    for gone in ("main.py", "engine.py", "config.py", "ocr_pipeline", "ocr_service/backends"):
        assert not (ROOT / gone).exists(), f"{gone} is back"


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


# ── 0.6 — the recognition engine stays pinned (E19) ─────────────────────────────
def test_engine_identity_is_pinned():
    """Accuracy was measured with THESE files. A different fas.traineddata (e.g. a distro
    package) or digit model changes every number in bench_fulltext_*.json."""
    import sys
    sys.path.insert(0, str(ROOT))
    from ocr_service.config import Settings

    prov = Settings().provenance()
    assert prov["engine"] == "tesseract(fas,eng) + digit_cnn"
    assert prov["layout_psms"] == "3,4,6" and prov["deskew_min_deg"] == 1.0
    fas = prov["tesseract"]["fas_sha256"]
    if fas is not None:
        assert fas == "99e420969b5ddd2c", f"fas.traineddata differs from the measured one: {fas}"
    dm = prov["digit_model"]["sha256"]
    if dm is not None:
        assert dm == "e0e286aaf5595898", f"models/digit_cnn.npz differs from the measured one: {dm}"


def test_the_spell_fix_can_only_move_dots():
    """The one correction stage allowed back in (E24). It may swap a letter for another
    of the SAME shape and nothing else, so it cannot invent a word: the failures that
    kept every previous corrector out — «شادی»→«هادی», «گذشته»→«گشته», «۱۴۰۴»→«۱۴۰۳» —
    are all unreachable, and so is any change to a word the reader was confident of."""
    import sys
    sys.path[:0] = [str(ROOT)]
    from ocr_service.spellfix import SpellFix
    sf = SpellFix()
    if not sf.available():
        pytest.skip("Persian word list not present")

    assert sf.fix("صبخگاهی", 50.0) == "صبحگاهی"          # a dot moves: repaired
    assert sf.fix("هوسمند", 50.0) == "هوشمند"
    assert sf.fix("گزارس", 50.0) == "گزارش"

    for w in ("شادی", "گذشته", "انسان", "باید", "میان", "بدون", "باقری", "هادی", "گشته"):
        assert sf.fix(w, 50.0) == w, f"{w} is a word and must never be touched"
    assert sf.fix("۱۴۰۴", 50.0) == "۱۴۰۴"                # digits are never touched
    assert sf.fix("125,000,000", 50.0) == "125,000,000"
    assert sf.fix("Email", 50.0) == "Email"              # nor Latin
    assert sf.fix("صبخگاهی", 95.0) == "صبخگاهی"          # nor a confident reading
    assert sf.fix("بمار", 50.0) == "بمار"                # two moved dots: too far to be sure

    for bad, good in (("صبخگاهی", "صبحگاهی"), ("گزارس", "گزارش")):
        assert len(sf.fix(bad, 50.0)) == len(bad)        # length never changes
        assert good == sf.fix(bad, 50.0)
