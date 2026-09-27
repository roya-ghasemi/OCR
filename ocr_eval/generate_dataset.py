
"""
Generate a synthetic, labeled OCR test set for the Persian + English OCR service.

IMPORTANT design note
---------------------
The service's /ocr endpoint is a *letter field extractor*, not a general text
transcriber: given a bare sentence it correctly returns all-null. So the test
images are shaped as real Persian business letters (sender / subject / body /
signature, plus bilingual field lines), which is what the model is built to read.
The ground truth for each image is the full rendered letter text in logical
(reading) order; the harness flattens the model's five JSON fields back into one
text blob and scores that against this ground truth, so field assignment does not
matter — only whether the characters were read correctly.

Two sources:
  synthetic_fa        - Persian-only letters
  synthetic_bilingual - letters whose fields mix Persian + English + digits
                        (names, codes, dates, amounts, emails, IBANs, ...)

Persian is shaped with arabic_reshaper and ordered with python-bidi before being
drawn. The stored ground truth is always the ORIGINAL logical string.

Output:
  ocr_eval/images/*.png
  ocr_eval/ground_truth.jsonl
"""
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

SEED = 20260827
random.seed(SEED)
np.random.seed(SEED)

HERE = Path(__file__).resolve().parent
IMG_DIR = HERE / "images"
GT_PATH = HERE / "ground_truth.jsonl"

FONTS = {
    "tahoma": r"C:\Windows\Fonts\tahoma.ttf",
    "arial":  r"C:\Windows\Fonts\arial.ttf",
}

_FA_RUN = re.compile(r"[؀-ۿﭐ-﷿ﹰ-﻿]+")
_EN_RUN = re.compile(r"[A-Za-z0-9@._\-/:]+")
_LATIN_DIGIT = re.compile(r"[A-Za-z0-9]")


# ── Content pieces ─────────────────────────────────────────────────────────────
SENDERS_FA = [
    "شرکت سپهرداده دیجیتال",
    "اداره کل فناوری اطلاعات",
    "معاونت پشتیبانی و خدمات",
    "دفتر مدیریت پروژه",
    "سازمان نظام صنفی رایانه‌ای",
    "شرکت داده‌پردازان نوین",
]
SENDERS_BI = [
    "شرکت سپهرداده دیجیتال - Sepehr Digital Data Co.",
    "اداره فناوری اطلاعات - IT Department",
    "شرکت داده‌پردازان نوین - NovinData Ltd.",
    "دفتر پروژه - Project Management Office",
]
SUBJECTS_FA = [
    "درخواست راه‌اندازی سرویس اینترنت پرسرعت",
    "پیگیری قرارداد پشتیبانی سالانه",
    "اعلام نتایج ارزیابی عملکرد سه‌ماهه",
    "درخواست تمدید مجوز دسترسی",
    "گزارش وضعیت سامانه و اقدامات اصلاحی",
    "دعوت به جلسه هماهنگی فنی",
]
BODIES_FA = [
    "احتراماً به استحضار می‌رساند که با توجه به نیاز مجموعه، درخواست بررسی و اقدام لازم را داریم.",
    "پیرو مکاتبات قبلی، خواهشمند است دستور فرمایید موضوع در اسرع وقت پیگیری شود.",
    "بدین‌وسیله گزارش عملکرد واحد فنی جهت استحضار و بهره‌برداری تقدیم می‌گردد.",
    "با عنایت به بررسی‌های انجام‌شده، اصلاحات پیشنهادی جهت تصویب به پیوست ارسال می‌شود.",
    "نظر به اهمیت موضوع، حضور نماینده تام‌الاختیار در جلسه مورد انتظار است.",
    "خواهشمند است مساعدت لازم جهت تسریع در انجام امور مربوطه صورت پذیرد.",
]
CLOSINGS_FA = [
    "پیشاپیش از همکاری شما سپاسگزاریم.",
    "از بذل توجه جنابعالی قدردانی می‌شود.",
    "خواهشمند است اقدام مقتضی به عمل آید.",
]
SIGNATURES_FA = [
    "مدیرعامل\nرویا قاسمی فر",
    "معاون فنی\nعلی رضایی",
    "مدیر پروژه\nمریم احمدی",
]
SIGNATURES_BI = [
    "مدیرعامل CEO\nRoya Ghasemi - رویا قاسمی فر",
    "معاون فنی CTO\nAli Rezaei - علی رضایی",
]

# Bilingual field lines embedded in bilingual letters (Persian label + Latin/digit value).
BI_FIELDS = [
    "کد اقتصادی: 51096365",
    "شماره قرارداد: INV-2026-0093",
    "تاریخ: 2026-08-27",
    "مبلغ کل: 1,250,000 ریال",
    "ایمیل: info@sepehr-digital.co",
    "کد ملی: 0071234567",
    "شماره تماس: +98 21 8899 1234",
    "شماره پرسنلی: EMP-4821",
    "شماره حساب IBAN: IR820170000000123456789",
    "کد رهگیری: TRK-99A45B",
    "درصد تخفیف: 25%",
    "نسخه سامانه: version 3.4.1",
]
DATES = ["1405-06-05", "2026-08-27", "2026/07/13"]


# ── Rendering ──────────────────────────────────────────────────────────────────
def _reshape(line: str) -> str:
    return get_display(arabic_reshaper.reshape(line)) if line.strip() else ""


def render(text: str, font_path: str, size: int, rotation: float) -> Image.Image:
    font = ImageFont.truetype(font_path, size)
    lines = text.split("\n")
    shaped = [_reshape(ln) for ln in lines]

    scratch = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    line_h = size + max(10, size // 3)
    widths = []
    for s in shaped:
        if s:
            b = scratch.textbbox((0, 0), s, font=font)
            widths.append(b[2] - b[0])
        else:
            widths.append(0)
    pad = max(30, size)
    W = max(max(widths), 300) + 2 * pad
    H = line_h * len(shaped) + 2 * pad

    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    for i, s in enumerate(shaped):
        y = pad + i * line_h
        if s:
            x = W - pad - widths[i]        # right-align (RTL document)
            draw.text((x, y), s, font=font, fill="black")

    if rotation:
        img = img.rotate(rotation, expand=True, fillcolor="white", resample=Image.BICUBIC)
    return img


def add_noise(img: Image.Image, sigma: float) -> Image.Image:
    if sigma <= 0:
        return img
    arr = np.asarray(img).astype(np.float32) + np.random.normal(0, sigma, (img.height, img.width, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def derive_refs(text: str):
    flat = text.replace("\n", " ")
    fa = " ".join(_FA_RUN.findall(flat)).strip()
    en = " ".join(t for t in _EN_RUN.findall(flat) if _LATIN_DIGIT.search(t)).strip()
    return fa, en


# ── Letter assembly ────────────────────────────────────────────────────────────
def make_fa_letter(i: int) -> str:
    return "\n".join([
        SENDERS_FA[i % len(SENDERS_FA)],
        f"موضوع: {SUBJECTS_FA[i % len(SUBJECTS_FA)]}",
        "",
        "با سلام و احترام،",
        BODIES_FA[i % len(BODIES_FA)],
        CLOSINGS_FA[i % len(CLOSINGS_FA)],
        "",
        SIGNATURES_FA[i % len(SIGNATURES_FA)],
    ])


def make_bi_letter(i: int) -> str:
    f1 = BI_FIELDS[(2 * i) % len(BI_FIELDS)]
    f2 = BI_FIELDS[(2 * i + 1) % len(BI_FIELDS)]
    return "\n".join([
        SENDERS_BI[i % len(SENDERS_BI)],
        f"موضوع: {SUBJECTS_FA[(i + 2) % len(SUBJECTS_FA)]}",
        f"تاریخ: {DATES[i % len(DATES)]}",
        "",
        "با سلام و احترام،",
        BODIES_FA[(i + 1) % len(BODIES_FA)],
        f1,
        f2,
        CLOSINGS_FA[i % len(CLOSINGS_FA)],
        "",
        SIGNATURES_BI[i % len(SIGNATURES_BI)],
    ])


# ── Build ──────────────────────────────────────────────────────────────────────
def build(n_fa: int = 18, n_bi: int = 18) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for old in IMG_DIR.glob("*.png"):
        old.unlink()

    fonts = list(FONTS.keys())
    records = []
    idx = 0

    def emit(text, source, i):
        nonlocal idx
        idx += 1
        font_name = fonts[i % len(fonts)]
        size = [26, 28, 30][i % 3]
        rot = random.choice([0, 0, 0, -1.5, 1.5])
        sigma = random.choice([0, 0, 0, 4])
        img = render(text, FONTS[font_name], size, rot)
        img = add_noise(img, sigma)
        fname = f"{source}_{idx:03d}.png"
        img.save(IMG_DIR / fname)
        fa, en = derive_refs(text)
        records.append({
            "filename": fname, "source": source, "text": text,
            "ref_fa": fa, "ref_en": en,
            "font": font_name, "size": size, "rotation": round(rot, 1),
        })

    for i in range(n_fa):
        emit(make_fa_letter(i), "synthetic_fa", i)
    for i in range(n_bi):
        emit(make_bi_letter(i), "synthetic_bilingual", i)

    with open(GT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Generated {len(records)} letter images -> {IMG_DIR}")
    print(f"  synthetic_fa        : {n_fa}")
    print(f"  synthetic_bilingual : {n_bi}")
    print(f"Ground truth -> {GT_PATH}")


if __name__ == "__main__":
    build()
