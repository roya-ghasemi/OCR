"""Characterize the real-scan corpus: dimensions, DPI, colour, EXIF, quality proxies."""
import hashlib, json, sys
from pathlib import Path
from PIL import Image, ExifTags
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "ocr_eval" / "dataset_ex"
OUT = ROOT / "ocr_eval" / "reports" / "real_corpus_characterization.json"

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

recs = []
for p in sorted(SRC.iterdir()):
    if p.is_dir():
        continue
    r = {"filename": p.name, "sha256": sha256(p), "bytes": p.stat().st_size}
    try:
        im = Image.open(p)
        r["format"] = im.format
        r["mode"] = im.mode
        r["width"], r["height"] = im.size
        r["n_frames"] = getattr(im, "n_frames", 1)
        dpi = im.info.get("dpi")
        r["dpi_tag"] = [float(x) for x in dpi] if dpi else None
        exif = None
        try:
            exif = im.getexif()
        except Exception:
            pass
        r["exif_orientation"] = int(exif.get(274)) if exif and 274 in exif else None
        r["exif_xres"] = float(exif.get(282)) if exif and 282 in exif else None

        g = im.convert("L")
        a = np.asarray(g, dtype=np.float32)
        r["mean_lum"] = round(float(a.mean()), 2)
        r["std_lum"] = round(float(a.std()), 2)
        # contrast: 5th-95th percentile spread
        lo, hi = np.percentile(a, [5, 95])
        r["p5_p95_spread"] = round(float(hi - lo), 2)
        # sharpness proxy: variance of Laplacian (3x3)
        k = a[1:-1,1:-1]*4 - a[:-2,1:-1] - a[2:,1:-1] - a[1:-1,:-2] - a[1:-1,2:]
        r["lap_var"] = round(float(k.var()), 2)
        # colourfulness: is it really colour?
        if im.mode in ("RGB","RGBA"):
            rgb = np.asarray(im.convert("RGB"), dtype=np.int16)
            r["max_channel_delta_mean"] = round(float(np.abs(rgb.max(2)-rgb.min(2)).mean()), 2)
        else:
            r["max_channel_delta_mean"] = 0.0
        # ink fraction (Otsu-ish, simple)
        thr = a.mean() - a.std()
        r["dark_pixel_frac"] = round(float((a < thr).mean()), 4)
        r["megapixels"] = round(im.size[0]*im.size[1]/1e6, 2)
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
    recs.append(r)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")

ok = [r for r in recs if "error" not in r]
print(f"images={len(recs)} readable={len(ok)} errors={len(recs)-len(ok)}")
def dist(key):
    v = sorted(r[key] for r in ok if r.get(key) is not None)
    if not v: return "n/a"
    import statistics as st
    q = lambda p: v[min(len(v)-1, int(p*len(v)))]
    return f"min={v[0]} p10={q(.1)} p50={q(.5)} p90={q(.9)} max={v[-1]}"
for k in ["width","height","megapixels","mean_lum","std_lum","p5_p95_spread","lap_var","max_channel_delta_mean","dark_pixel_frac","bytes"]:
    print(f"{k:24s} {dist(k)}")
from collections import Counter
for k in ["format","mode","dpi_tag","exif_orientation","n_frames"]:
    print(k, Counter(str(r.get(k)) for r in ok).most_common())
