# -*- coding: utf-8 -*-
r"""E13b — memorisation vs reading: the real letter layout with names it has never seen.

Renders letters in the corpus layout (letterhead, شماره/تاریخ block, receiver
line, salutation, body with an amount and a national ID, closing, signature,
footer) with company names, person names and numbers drawn at random — none of
them in the 90-document corpus — plus one CONTROL letter carrying the corpus
letterhead `شرکت خدماتی سبز گستر`. Arm B (`ocr_pipeline.LetterExtractor`) reads
them; each field is scored against the known text with the project's CER.

If the corpus sender name reads far better than novel names in the same layout
and font, that is memorisation of the template. If novel names read as well,
the header fields are being read, and the body/digit failure is something else.

    venv312\Scripts\python.exe ocr_eval/experiments/diagnostics_0913/novel_headers.py
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVAL = ROOT / "ocr_eval"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(EVAL))
sys.stdout.reconfigure(encoding="utf-8")

import normalize as N  # noqa: E402
from harness import cer  # noqa: E402

FONT = r"C:\Windows\Fonts\tahoma.ttf"
W, H = 2300, 3200                       # ~ the real scans (2304 x 3016)
FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
ENGINE = "http://127.0.0.1:18234/v1"

COMPANIES = ["شرکت بازرگانی نیلوفر کویر", "موسسه فنی مهندسی آذرخش پارس", "شرکت تعاونی دامداران البرز",
             "شرکت مهندسی سپهر آب زاگرس", "کارگزاری بیمه ارغوان", "شرکت حمل و نقل رعد شرق"]
RECEIVERS = ["مدیر محترم شعبه بانک ملت بلوار وکیل آباد", "ریاست محترم اداره گاز شهرستان نیشابور",
             "سرپرست محترم بیمارستان فارابی", "مدیر عامل محترم شرکت پخش دارویی هجرت",
             "ریاست محترم دانشکده کشاورزی دانشگاه فردوسی", "مدیر محترم امور مالی شهرداری منطقه ۹"]
PEOPLE = ["فرزانه توکلی راد", "بهرام کیانی مقدم", "نسرین عبدالهی", "کاوه رستم پور", "لیلا شریفی نسب", "مجتبی نوروزی"]
STREETS = ["بلوار سجاد، خیابان بهار، پلاک", "خیابان احمدآباد، نبش عارف، پلاک", "بلوار پیروزی، میدان حر، پلاک"]

CONTROL_COMPANY = "شرکت خدماتی سبز گستر"


def shape(s: str) -> str:
    return get_display(arabic_reshaper.reshape(s))


def rtl(d: ImageDraw.ImageDraw, xy, s: str, font, fill="black"):
    """Right-anchored, shaped Persian. Numbers are direction-neutral and survive."""
    d.text(xy, shape(s), font=font, fill=fill, anchor="ra")


def make_letter(i: int, company: str, rng: random.Random) -> tuple[Image.Image, dict]:
    receiver = RECEIVERS[i % len(RECEIVERS)]
    person = PEOPLE[i % len(PEOPLE)]
    signer = PEOPLE[(i + 3) % len(PEOPLE)]
    reg = str(rng.randint(10000, 99999))
    ref = f"1403/{rng.randint(1, 12)}/{rng.randint(100, 999)}"
    date = f"1403/{rng.randint(1, 12):02d}/{rng.randint(1, 30):02d}"
    amount = f"{rng.randint(11, 989)}/{rng.randint(100, 999)}/000"
    nid = "0" + "".join(str(rng.randint(0, 9)) for _ in range(9))
    phone = "-".join(str(rng.randint(30000000, 39999999)) for _ in range(2))
    plate = str(rng.randint(2, 240))

    body = (f"با سلام؛\nاحتراما ، بدین وسیله {person} به شماره ملی {nid.translate(FA)} به عنوان نماینده این شرکت "
            f"جهت پیگیری مطالبات به مبلغ {amount.translate(FA)} ریال به حضور معرفی می گردد. "
            f"خواهشمند است دستور فرمایید اقدام مقتضی مبذول گردد.\nپیشاپیش از همکاری شما سپاسگزاریم.")
    contact = (f"آدرس : مشهد - {STREETS[i % len(STREETS)]} {plate.translate(FA)}\n"
               f"تلفن: {phone.translate(FA)}")
    gt = {"sender": company, "receiver": receiver, "subject": None, "body_text": body,
          "contact_info": contact, "_amount": amount, "_nid": nid, "_ref": ref}

    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    f_head = ImageFont.truetype(FONT, 78)
    f_meta = ImageFont.truetype(FONT, 36)
    f_body = ImageFont.truetype(FONT, 44)
    f_foot = ImageFont.truetype(FONT, 34)

    rtl(d, (W - 150, 140), company, f_head)
    rtl(d, (W - 150, 260), f"شماره ثبت: {reg.translate(FA)}", f_meta)
    rtl(d, (620, 160), f"شماره: {ref.translate(FA)}", f_meta)
    rtl(d, (620, 220), f"تاریخ: {date.translate(FA)}", f_meta)
    rtl(d, (620, 280), "پیوست: ...............", f_meta)
    rtl(d, (W - 150, 560), receiver, f_body)
    y = 680
    for line in body.split("\n"):
        # wrap at ~55 chars, right to left
        words, cur = line.split(" "), ""
        for w_ in words:
            if len(cur) + len(w_) > 60:
                rtl(d, (W - 150, y), cur, f_body); y += 70; cur = w_
            else:
                cur = (cur + " " + w_).strip()
        rtl(d, (W - 150, y), cur, f_body); y += 90
    rtl(d, (700, y + 60), "با تشکر", f_body)
    rtl(d, (700, y + 130), signer, f_body)
    rtl(d, (700, y + 200), "مدیر عامل", f_body)
    for k, line in enumerate(contact.split("\n")):
        rtl(d, (W - 150, H - 260 + k * 55), line, f_foot)
    d.line([(150, H - 300), (W - 150, H - 300)], fill="black", width=3)
    gt["_signer"] = signer
    return im, gt


async def read_all(paths: list[Path]) -> list[dict]:
    from openai import AsyncOpenAI
    from ocr_pipeline.extraction import LetterExtractor
    c = AsyncOpenAI(base_url=ENGINE, api_key="local", timeout=600.0)
    model = (await c.models.list()).data[0].id
    ex = LetterExtractor(c, model)
    out = []
    for p in paths:
        url = "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()
        r = await ex.extract(url, doc_id=p.name)
        out.append(r.data if isinstance(r.data, dict) else {})
    return out


def digits(s: str | None) -> str:
    return re.sub(r"\D", "", N.normalize(s or "", N.RULES_DEFAULT))


def main() -> int:
    rng = random.Random(913)
    out_dir = HERE / "novel_letters"; out_dir.mkdir(exist_ok=True)
    docs = []
    for i in range(6):
        im, gt = make_letter(i, COMPANIES[i], rng)
        p = out_dir / f"novel_{i:02d}.jpg"; im.save(p, "JPEG", quality=90)
        docs.append((p, gt, "novel"))
    for i in range(2):
        im, gt = make_letter(i, CONTROL_COMPANY, rng)
        p = out_dir / f"control_{i:02d}.jpg"; im.save(p, "JPEG", quality=90)
        docs.append((p, gt, "control"))

    preds = asyncio.run(read_all([p for p, _, _ in docs]))
    rows = []
    print(f"{'doc':12s} {'kind':7s} {'sender':>7s} {'recv':>7s} {'body':>7s} {'contact':>7s}  amount   nat-id")
    for (p, gt, kind), pr in zip(docs, preds):
        row = {"doc": p.name, "kind": kind}
        for f in ("sender", "receiver", "body_text", "contact_info"):
            g = N.normalize(gt[f], N.RULES_DEFAULT); h = N.normalize(pr.get(f), N.RULES_DEFAULT)
            row[f"cer_{f}"] = round(cer(g, h), 3) if g else None
            row[f"pred_{f}"] = pr.get(f)
        blob = " ".join(v for v in pr.values() if isinstance(v, str))
        row["amount_found"] = digits(gt["_amount"]) in digits(blob)
        row["nid_found"] = gt["_nid"] in digits(blob)
        rows.append(row)
        print(f"{p.name:12s} {kind:7s} {row['cer_sender']:7.2f} {row['cer_receiver']:7.2f} "
              f"{row['cer_body_text']:7.2f} {row['cer_contact_info']:7.2f}  "
              f"{'OK ' if row['amount_found'] else 'BAD'}      {'OK' if row['nid_found'] else 'BAD'}")

    def mean(kind, key):
        v = [r[key] for r in rows if r["kind"] == kind and r[key] is not None]
        return round(sum(v) / len(v), 3) if v else None
    summary = {kind: {k: mean(kind, f"cer_{k}") for k in ("sender", "receiver", "body_text", "contact_info")}
               for kind in ("novel", "control")}
    summary["digits"] = {"amount_found": sum(r["amount_found"] for r in rows), "nid_found": sum(r["nid_found"] for r in rows), "n": len(rows)}
    (HERE / "results_E13b_novel_headers.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arm": "B — ocr_pipeline.LetterExtractor defaults", "font": "Tahoma", "size": [W, H],
        "summary": summary, "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nmean CER  novel  :", summary["novel"]); print("mean CER  control:", summary["control"]); print("digits:", summary["digits"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
