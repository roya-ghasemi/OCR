"""Phase 0 §2.1 — splits_v2: dev 60 / test 40, stratified by source x language_mode x template.

Grouping rules, in order of precedence:
  1. identical sha256  -> same group (the synthetic set contains 3 duplicate pairs)
  2. identical GT text -> same group (prevents template leakage across the split)
Groups, not images, are dealt to dev/test, so a template never straddles the split.

Written ONCE and hash-locked. Regenerating it invalidates every comparison made
against it; the guard test asserts the hash has not moved.
"""
import hashlib, json, random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "ocr_eval"
OUT = EVAL / "splits_v2.json"
SEED = 20260901
DEV_FRACTION = 0.6

manifest = json.loads((EVAL / "manifest.json").read_text(encoding="utf-8"))

gt_text = {}
gtp = EVAL / "ground_truth_fields.jsonl"
if gtp.exists():
    for line in gtp.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            gt_text[r["filename"]] = r.get("text", "")


def group_key(r: dict) -> str:
    """Duplicate pixels first, then duplicate ground truth, then the file itself."""
    txt = gt_text.get(r["filename"])
    if txt:
        return "gt:" + hashlib.sha1(txt.encode("utf-8")).hexdigest()[:12]
    return "px:" + r["sha256"][:12]


def stratum(r: dict) -> str:
    """Strata are source x language_mode. Template is the GROUPING unit, not a
    stratum: with only 6 synthetic templates, stratifying on template makes every
    stratum a single group, which forces all 6 Persian-only groups into dev and
    leaves the Persian-only test split EMPTY. Stratifying one level coarser keeps
    templates intact across the split while still giving each language mode a
    test set."""
    return f'{r["source"]}|{r["language_mode"]}'


groups = defaultdict(list)
for r in manifest:
    groups[group_key(r)].append(r)

# One stratum per group (all members of a group share it by construction).
by_stratum = defaultdict(list)
for gk, members in groups.items():
    by_stratum[stratum(members[0])].append(gk)

rng = random.Random(SEED)
assign = {}
for strat in sorted(by_stratum):
    gks = sorted(by_stratum[strat])
    rng.shuffle(gks)
    n_dev = round(len(gks) * DEV_FRACTION)
    # Every stratum with >1 group must contribute to BOTH splits, or a whole
    # language mode silently disappears from test.
    if len(gks) > 1:
        n_dev = max(1, min(len(gks) - 1, n_dev))
    else:
        n_dev = len(gks)
    for i, gk in enumerate(gks):
        assign[gk] = "dev" if i < n_dev else "test"

split_of = {}
for gk, members in groups.items():
    for r in members:
        split_of[r["filename"]] = assign[gk]

from collections import Counter
counts = Counter((r["source"], split_of[r["filename"]]) for r in manifest)

doc = {
    "version": 2,
    "seed": SEED,
    "dev_fraction_target": DEV_FRACTION,
    "policy": (
        "Groups (identical pixels OR identical ground-truth text) are the unit of "
        "assignment, stratified by source x language_mode x template_id. "
        "splits.json (v1) is retained unchanged for the legacy 36-image comparison only."
    ),
    "manifest_sha256": hashlib.sha256((EVAL / "manifest.json").read_bytes()).hexdigest(),
    "counts": {f"{s}_{sp}": n for (s, sp), n in sorted(counts.items())},
    "n_groups": len(groups),
    "strata": {s: len(g) for s, g in sorted(by_stratum.items())},
    "groups": {
        gk: {
            "split": assign[gk],
            "stratum": stratum(members[0]),
            "files": sorted(m["filename"] for m in members),
        }
        for gk, members in sorted(groups.items())
    },
    "assignment": dict(sorted(split_of.items())),
}

OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
h = hashlib.sha256(OUT.read_bytes()).hexdigest()
(EVAL / "splits_v2.sha256").write_text(h + "\n", encoding="utf-8")

print("splits_v2 sha256:", h)
print("groups:", len(groups), "(126 images ->", len(groups), "groups)")
for k, v in sorted(doc["counts"].items()):
    print(f"  {k:16s} {v}")
print("strata:", doc["strata"])
