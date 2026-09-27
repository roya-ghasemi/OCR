"""Phase 0 §2.1 — one manifest record per image across BOTH corpora.

Synthetic (ocr_eval/images) and real (ocr_eval/dataset_ex) are enumerated from
separate roots and tagged with `source`. Nothing globs a single mixed directory,
so the 36-image legacy baseline stays independently re-scoreable.
"""
import hashlib, json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "ocr_eval"
SYNTH_DIR = EVAL / "images"
REAL_DIR = EVAL / "dataset_ex"
GT_FIELDS = EVAL / "ground_truth_fields.jsonl"
OUT = EVAL / "manifest.json"

# A4 short edge in mm; used to turn pixel width into an effective scan density,
# because the JPEG DPI tag on the real corpus is a 72 placeholder.
A4_SHORT_EDGE_MM = 210.0
MM_PER_INCH = 25.4
DPI_FLOOR = 200.0   # below this, a scan is flagged as under-resolved


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_synth_gt() -> dict:
    gt = {}
    if GT_FIELDS.exists():
        for line in GT_FIELDS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                gt[r["filename"]] = r
    return gt


def template_id(rec: dict | None) -> str | None:
    """Synthetic letters are drawn from a small pool of body/subject templates.
    The template is identified by the ground-truth *subject*, which is the field
    the generator varies per template. Effective n is the number of templates,
    not the number of images -- §1.8 depends on this being recorded."""
    if not rec:
        return None
    subj = (rec.get("fields") or {}).get("subject")
    if not subj:
        return None
    return "t_" + hashlib.sha1(subj.encode("utf-8")).hexdigest()[:8]


def probe(p: Path) -> dict:
    im = Image.open(p)
    w, h = im.size
    dpi = im.info.get("dpi")
    exif = {}
    try:
        exif = im.getexif() or {}
    except Exception:
        pass
    est_dpi = round(min(w, h) / (A4_SHORT_EDGE_MM / MM_PER_INCH), 1)
    return {
        "format": im.format,
        "mode": im.mode,
        "width": w,
        "height": h,
        "n_frames": getattr(im, "n_frames", 1),
        "dpi_tag": [float(x) for x in dpi] if dpi else None,
        "dpi_tag_is_placeholder": bool(dpi and set(float(x) for x in dpi) == {72.0}),
        "estimated_dpi_a4": est_dpi,
        "below_dpi_floor": est_dpi < DPI_FLOOR,
        "exif_orientation": int(exif.get(274)) if 274 in exif else None,
    }


records = []
synth_gt = load_synth_gt()

for p in sorted(SYNTH_DIR.iterdir()):
    if p.is_dir():
        continue
    rec = synth_gt.get(p.name)
    lang = "bilingual" if p.name.startswith("synthetic_bilingual") else "persian_only"
    r = {
        "filename": p.name,
        "relpath": str(p.relative_to(EVAL)).replace("\\", "/"),
        "sha256": sha256(p),
        "bytes": p.stat().st_size,
        "source": "synthetic",
        "corpus_dir": "images",
        "language_mode": lang,
        "template_id": template_id(rec),
        "gt_exists": rec is not None,
        "gt_path": str(GT_FIELDS.relative_to(EVAL)).replace("\\", "/") if rec else None,
        "gt_kind": "generator_intent" if rec else None,
    }
    r.update(probe(p))
    records.append(r)

for p in sorted(REAL_DIR.iterdir()):
    if p.is_dir():
        continue
    r = {
        "filename": p.name,
        "relpath": str(p.relative_to(EVAL)).replace("\\", "/"),
        "sha256": sha256(p),
        "bytes": p.stat().st_size,
        "source": "real",
        "corpus_dir": "dataset_ex",
        # Determined by inspection: Persian body text with Latin/ASCII runs
        # (email addresses) in the footer. Confirm per-document once GT lands.
        "language_mode": "bilingual",
        # All sampled letters share one sender + letterhead. Until GT arrives
        # this is a single provisional template group, which keeps the §1.8
        # template-level bootstrap honest instead of pretending n=90.
        "template_id": "real_unassigned",
        "gt_exists": False,
        "gt_path": None,
        "gt_kind": None,
    }
    r.update(probe(p))
    records.append(r)

OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

manifest_hash = hashlib.sha256(OUT.read_bytes()).hexdigest()
(EVAL / "manifest.sha256").write_text(manifest_hash + "\n", encoding="utf-8")

from collections import Counter
print(f"manifest: {len(records)} records -> {OUT}")
print("sha256:", manifest_hash)
print("by source:", Counter(r["source"] for r in records))
print("by lang:  ", Counter((r["source"], r["language_mode"]) for r in records))
print("gt_exists:", Counter((r["source"], r["gt_exists"]) for r in records))
print("templates(synthetic):", len({r["template_id"] for r in records if r["source"]=="synthetic"}))
print("below dpi floor:", sum(1 for r in records if r["below_dpi_floor"]),
      "| dup sha256:", len(records) - len({r["sha256"] for r in records}))
