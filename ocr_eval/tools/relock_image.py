# -*- coding: utf-8 -*-
r"""Re-lock the corpus after a DELIBERATE change to one image.

A corpus change is normally a defect (D51). This tool exists so that an
intentional one is recorded rather than papered over: it moves the lock to the
new bytes, in every place a lock is held, and leaves a .bak of each file it
touches.

The lock is a chain, and a partial re-lock is worse than none:

    dataset_ex/<image>            the bytes themselves
    manifest.json                 sha256 + size + PIL metadata for that record
    manifest.sha256               hash OF manifest.json
    splits_v2.json                carries `manifest_sha256`
    splits_v2.sha256              hash OF splits_v2.json
    gt_real_template.jsonl        per-row sha256
    ground_truth_real_fixed_v2.jsonl  per-row sha256 (if present)

`ci_gate.check_corpus_locks()` verifies the two .sha256 files, and
`score_real.validate_gt()` refuses to score a GT row whose sha256 disagrees with
the manifest, so all of them must move together.

This does NOT regenerate predictions. Every stored prediction for the re-locked
image was produced against the OLD bytes and is stale until re-run; the tool
prints which files contain such a row.

Run:  venv312\Scripts\python.exe ocr_eval/tools/relock_image.py "<filename>"
Exits non-zero, writing nothing, if any file cannot be round-tripped exactly.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "ocr_eval"
IMAGES = EVAL / "dataset_ex"
MANIFEST = EVAL / "manifest.json"
MANIFEST_LOCK = EVAL / "manifest.sha256"
SPLITS = EVAL / "splits_v2.json"
SPLITS_LOCK = EVAL / "splits_v2.sha256"
JSONL_LOCKS = [EVAL / "gt_real_template.jsonl", EVAL / "ground_truth_real_fixed_v2.jsonl"]
PREDICTIONS = [EVAL / "predictions_real.jsonl", EVAL / "predictions_real_pipeline.jsonl"]

SUFFIX = ".pre_relock.bak"


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dumps(obj, trailing: str = "") -> str:
    """The formatting every JSON file in ocr_eval/ is written in.

    `trailing` carries the original file's line ending through, so re-locking
    changes exactly one record and not every byte after it.
    """
    return json.dumps(obj, indent=2, ensure_ascii=False) + trailing


def backup(p: Path) -> None:
    b = p.with_name(p.name + SUFFIX)
    if not b.exists():
        shutil.copy2(p, b)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[-3])
        return 2
    target = sys.argv[1]

    img = IMAGES / target
    if not img.is_file():
        print(f"no such image: {img}")
        return 2

    # --- prove we can round-trip the JSON files before changing anything -----
    tail: dict[Path, str] = {}
    for p in (MANIFEST, SPLITS):
        raw = p.read_text(encoding="utf-8")
        end = "\n" if raw.endswith("\n") else ""
        if dumps(json.loads(raw), end) != raw:
            print(f"{p.name} does not round-trip under the standard formatting; "
                  f"refusing to rewrite it.")
            return 1
        tail[p] = end

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rec = next((r for r in manifest if r["filename"] == target), None)
    if rec is None:
        print(f"{target} is not in manifest.json")
        return 2

    new_sha = sha_file(img)
    if new_sha == rec["sha256"]:
        print(f"{target} already matches the lock; nothing to do.")
        return 0

    with Image.open(img) as im:
        new_meta = {
            "bytes": img.stat().st_size,
            "format": im.format,
            "mode": im.mode,
            "width": im.width,
            "height": im.height,
            "exif_orientation": im.getexif().get(274, 1),
        }

    old = {"sha256": rec["sha256"], **{k: rec.get(k) for k in new_meta}}

    print(f"re-locking {target}")
    print(f"  sha256           {old['sha256'][:16]}…  ->  {new_sha[:16]}…")
    for k, v in new_meta.items():
        mark = "   (unchanged)" if old.get(k) == v else "   <-- CHANGED"
        print(f"  {k:16s} {str(old.get(k)):>12s}  ->  {str(v):<12s}{mark}")

    # --- 1. manifest record ---------------------------------------------------
    backup(MANIFEST)
    rec["sha256"] = new_sha
    rec.update(new_meta)
    MANIFEST.write_text(dumps(manifest, tail[MANIFEST]), encoding="utf-8")

    # --- 2. hash of the manifest ---------------------------------------------
    backup(MANIFEST_LOCK)
    man_sha = sha_file(MANIFEST)
    MANIFEST_LOCK.write_text(man_sha + "\n", encoding="utf-8")

    # --- 3. splits_v2 carries the manifest hash, then re-hash it -------------
    backup(SPLITS)
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    prev = splits.get("manifest_sha256")
    splits["manifest_sha256"] = man_sha
    SPLITS.write_text(dumps(splits, tail[SPLITS]), encoding="utf-8")
    backup(SPLITS_LOCK)
    SPLITS_LOCK.write_text(sha_file(SPLITS) + "\n", encoding="utf-8")

    print(f"  manifest.json    {prev[:16]}…  ->  {man_sha[:16]}…  (splits_v2 + both .sha256 updated)")

    # --- 4. per-row sha256 in every jsonl that carries one -------------------
    for p in JSONL_LOCKS:
        if not p.is_file():
            continue
        rows, hit = [], 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("filename") == target and r.get("sha256") != new_sha:
                r["sha256"] = new_sha
                hit += 1
            rows.append(r)
        if hit:
            backup(p)
            p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                         encoding="utf-8", newline="\n")
            print(f"  {p.name}: row sha256 updated")

    # --- 5. say plainly what is now stale ------------------------------------
    stale = []
    for p in PREDICTIONS:
        if p.is_file() and any(json.loads(l).get("filename") == target
                               for l in p.read_text(encoding="utf-8").splitlines() if l.strip()):
            stale.append(p.name)
    if stale:
        print("\n  STALE — these hold a prediction made against the OLD bytes:")
        for s in stale:
            print(f"     {s}")
        print("     Re-run that document before scoring it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
