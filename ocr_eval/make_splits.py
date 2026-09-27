"""
Create the frozen dev/test split — run ONCE, then never again.

Constraints, in priority order:
  1. No leakage: the 36 images are only 18 distinct ground-truth texts, each
     rendered 1-3 times with different font/size/rotation/noise. All renders of a
     given text therefore stay in the SAME split, otherwise tuning on dev partly
     transfers to test.
  2. Stratified by source: both layouts (synthetic_fa, synthetic_bilingual) appear
     in both splits, so neither split is layout-homogeneous.
  3. Target 60% dev / 40% test by image count, per source.

Writes ocr_eval/splits.json. Refuses to overwrite an existing file.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
GT = HERE / "ground_truth.jsonl"
OUT = HERE / "splits.json"
SEED = 20260829
DEV_FRACTION = 0.60


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists — the split is frozen. Delete it "
                         f"deliberately if you really mean to regenerate.")

    rows = [json.loads(l) for l in GT.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Group images by (source, distinct text) -> the leakage unit.
    groups = defaultdict(list)
    for r in rows:
        tid = hashlib.sha1(r["text"].encode("utf-8")).hexdigest()[:10]
        groups[(r["source"], tid)].append(r["filename"])

    by_source = defaultdict(list)
    for (source, tid), files in groups.items():
        by_source[source].append((tid, sorted(files)))

    split = {"dev": [], "test": []}
    meta = {}

    for source in sorted(by_source):
        # Deterministic order, then greedily fill dev until the image quota is met.
        units = sorted(by_source[source], key=lambda u: (-len(u[1]), u[0]))
        total = sum(len(f) for _, f in units)
        quota = round(total * DEV_FRACTION)
        dev_n = 0
        for tid, files in units:
            target = "dev" if dev_n < quota else "test"
            if target == "dev":
                dev_n += len(files)
            split[target].extend(files)
            meta[tid] = {"source": source, "split": target, "n_renders": len(files)}

    for k in split:
        split[k] = sorted(split[k])

    payload = {
        "seed": SEED,
        "dev_fraction_target": DEV_FRACTION,
        "policy": ("grouped by distinct ground-truth text to prevent leakage; "
                   "stratified by source"),
        "counts": {
            "dev": len(split["dev"]),
            "test": len(split["test"]),
            "dev_by_source": {},
            "test_by_source": {},
        },
        "text_groups": meta,
        "dev": split["dev"],
        "test": split["test"],
    }
    src_of = {r["filename"]: r["source"] for r in rows}
    for s in ("dev", "test"):
        c = defaultdict(int)
        for f in split[s]:
            c[src_of[f]] += 1
        payload["counts"][f"{s}_by_source"] = dict(c)

    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"dev : {len(split['dev'])} images  {payload['counts']['dev_by_source']}")
    print(f"test: {len(split['test'])} images  {payload['counts']['test_by_source']}")
    print(f"distinct text groups: {len(meta)}  (none spans both splits)")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
