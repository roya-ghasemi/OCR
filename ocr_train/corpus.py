# -*- coding: utf-8 -*-
r"""The text the synthetic lines are rendered from.

The benchmark supplies real Persian prose, which teaches the model nothing about the two
things this service actually fails on:

* **Kashida.** Administrative Persian is justified by *stretching letters*, not by widening
  spaces. The recogniser reads a long connector as a letter («بلــوار» → «بلسوار») or as a
  gap («استخراج» → «اس‌تخراج»), and E20/E27 both spend code on it. A model that has seen
  stretched text does not need either. The label is always the **unstretched** text, so the
  model is taught that a kashida carries no meaning.
* **The administrative register.** Letter numbers, Persian dates, 8-digit landlines, amounts
  with thousands separators, e-mail addresses, and the fixed formulas («بسمه تعالی»,
  «احتراماً», «رونوشت»). These carry the information a letter is read *for*, and they are
  exactly where a general corpus is thinnest.

Every generator here returns `(text_to_render, ground_truth)`. They differ only when the
rendering is deliberately distorted — kashida — and that difference is the point.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

ZWNJ = "‌"
KASHIDA = "ـ"

# Letters that do not join to what follows them. A kashida after one of these would be a
# free-floating dash, not an elongation, so it is never inserted there. This is the rule the
# brief states and it is also simply how the script works.
NON_CONNECTING = set("اآأإدذرزژوؤةء")
PERSIAN_LETTER = re.compile(r"[ء-غف-يٰ-ۓ]")

# Fixed formulas: every administrative letter carries some of these, and the recogniser is
# currently repairing them against a word list after the fact (E27).
FORMULAS = [
    "بسمه تعالی", "باسمه تعالی", "به نام خدا", "با سلام", "با سلام و احترام",
    "سلام علیکم", "احتراماً", "احتراما", "با تشکر", "با تشکر و احترام",
    "رونوشت:", "پیوست:", "شماره:", "تاریخ:", "موضوع:", "مدیر عامل", "مدیرعامل",
    "جناب آقای", "سرکار خانم", "ریاست محترم", "معاون محترم", "سرپرست محترم",
    "خواهشمند است", "مبذول فرمایید", "به استحضار می رساند", "پیرو نامه شماره",
    "عطف به نامه شماره", "در خصوص", "به قید فوریت", "اقدام لازم",
]

LATIN_BITS = [
    "Email", "Tel", "Fax", "No", "Ref", "www", "http", "Page", "Rev", "ID",
    "OCR", "PDF", "Excel", "Word", "Invoice", "Contract", "Department", "Manager",
]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


# --- kashida -----------------------------------------------------------------------------
def kashida_positions(word: str) -> list[int]:
    """Indices in `word` after which a kashida may legally be inserted."""
    out = []
    for i, ch in enumerate(word[:-1]):
        if not PERSIAN_LETTER.match(ch) or ch in NON_CONNECTING:
            continue
        nxt = word[i + 1]
        if not PERSIAN_LETTER.match(nxt) or nxt == ZWNJ:
            continue
        out.append(i)
    return out


def stretch_word(word: str, total: int, rng: random.Random) -> str:
    """`word` with `total` kashida characters distributed over its legal join points."""
    spots = kashida_positions(word)
    if not spots or total <= 0:
        return word
    share = [0] * len(spots)
    for _ in range(total):
        share[rng.randrange(len(spots))] += 1
    out = []
    for i, ch in enumerate(word):
        out.append(ch)
        if i in spots:
            out.append(KASHIDA * share[spots.index(i)])
    return "".join(out)


def justify(text: str, target: int, rng: random.Random) -> str:
    """Stretch a line towards `target` characters the way a word processor does — by
    elongating words, not by padding with spaces. Returns the rendered form; the caller
    keeps `text` as the label."""
    words = text.split()
    if len(words) < 2:
        return text
    need = target - len(text)
    if need <= 0:
        return text
    stretchable = [i for i, w in enumerate(words) if kashida_positions(w)]
    if not stretchable:
        return text
    rng.shuffle(stretchable)
    per = max(1, need // max(1, len(stretchable)))
    left = need
    for i in stretchable:
        if left <= 0:
            break
        n = min(left, rng.randint(1, max(1, per * 2)))
        words[i] = stretch_word(words[i], n, rng)
        left -= n
    return " ".join(words)


# --- the administrative register -----------------------------------------------------------
def _digits(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("۰۱۲۳۴۵۶۷۸۹") for _ in range(n))


def jalali_date(rng: random.Random) -> str:
    y = rng.randint(1390, 1410)
    m = rng.randint(1, 12)
    d = rng.randint(1, 30)
    sep = rng.choice(["/", "/", "/", "-"])
    fa = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
    return f"{y}{sep}{m:02d}{sep}{d:02d}".translate(fa)


def amount(rng: random.Random) -> str:
    groups = rng.randint(1, 4)
    head = _digits(rng.randint(1, 3), rng)
    sep = rng.choice(["،", ",", ".", "/"])
    return head + "".join(sep + _digits(3, rng) for _ in range(groups))


def phone(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.4:
        return _digits(8, rng)
    if kind < 0.7:
        return "۰" + _digits(2, rng) + "-" + _digits(8, rng)
    return "۰۹" + _digits(9, rng)


def email(rng: random.Random) -> str:
    user = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(4, 10)))
    host = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(4, 12)))
    tld = rng.choice(["ir", "com", "org", "net", "co.ir"])
    if rng.random() < 0.3:
        host += "-" + "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 8)))
    return f"{user}@{host}.{tld}"


# Arabic letterforms. Tesseract's Persian word list is full of them — «مدرك», «ويژگي»,
# «كودكان» — because it was built from mixed sources. Rendering those would train the model
# to emit exactly the non-canonical forms the service spends code normalising away (D8/D43),
# so a word carrying one is dropped rather than folded: folding would invent a spelling the
# list does not actually vouch for.
_ARABIC_FORMS = set("يكىة")


def _usable(w: str) -> bool:
    return bool(w) and not (_ARABIC_FORMS & set(w)) and not w.startswith("#")


def load_vocab(root: Path) -> tuple[list[str], list[str]]:
    """(administrative words, general words) from the lists the service already ships."""
    admin, general = [], []
    p = root / "ocr_service" / "data" / "fas_admin.txt"
    if p.is_file():
        admin = [w.strip(ZWNJ) for w in map(str.strip, p.read_text(encoding="utf-8").splitlines())
                 if _usable(w)]
    p = root / "ocr_service" / "data" / "fas_words.txt"
    if p.is_file():
        general = [w.strip(ZWNJ) for w in map(str.strip, p.read_text(encoding="utf-8").splitlines())
                   if _usable(w)]
    return [w for w in admin if w], [w for w in general if w]


def load_real_sentences(root: Path) -> list[str]:
    """Sentences off the pages this service is actually judged on. These are few — the
    point is register and letterhead vocabulary, not volume."""
    out = []
    for p in sorted((root / "ocr_eval" / "pages").glob("*.gt.txt")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if len(line) >= 8:
                out.append(line)
                for part in re.split(r"[.؛]", line):
                    part = part.strip()
                    if 12 <= len(part) <= 90:
                        out.append(part)
    return out


# --- line generation -------------------------------------------------------------------------
def generate(root: Path, n: int, seed: int) -> list[tuple[str, str]]:
    """`n` (render, label) pairs. `render` may carry kashida; `label` never does."""
    rng = _rng(seed)
    admin, general = load_vocab(root)
    real = load_real_sentences(root)
    pool = admin + general
    if not pool:
        raise RuntimeError("no word lists found — expected ocr_service/data/fas_words.txt")

    out: list[tuple[str, str]] = []
    while len(out) < n:
        r = rng.random()

        if r < 0.22 and real:                                   # real administrative prose
            label = rng.choice(real)

        elif r < 0.40:                                          # a formula plus a clause
            label = rng.choice(FORMULAS) + " " + " ".join(rng.choice(pool) for _ in range(rng.randint(3, 9)))

        elif r < 0.52:                                          # the numeric register
            kind = rng.random()
            if kind < 0.3:
                label = f"شماره: {_digits(rng.randint(3, 6), rng)} تاریخ: {jalali_date(rng)}"
            elif kind < 0.55:
                label = f"مبلغ {amount(rng)} ریال"
            elif kind < 0.8:
                label = f"تلفن: {phone(rng)} فکس: {phone(rng)}"
            else:
                label = f"پیرو قرارداد شماره {_digits(5, rng)} مورخ {jalali_date(rng)}"

        elif r < 0.60:                                          # bilingual / e-mail
            kind = rng.random()
            if kind < 0.5:
                label = f"{rng.choice(['ایمیل', 'پست الکترونیک', 'Email'])}: {email(rng)}"
            else:
                label = (" ".join(rng.choice(pool) for _ in range(rng.randint(2, 5)))
                         + " " + rng.choice(LATIN_BITS) + " "
                         + " ".join(rng.choice(pool) for _ in range(rng.randint(1, 4))))

        else:                                                   # plain prose
            label = " ".join(rng.choice(pool) for _ in range(rng.randint(4, 12)))

        # Commas: the character the base alphabet is missing. Put them where Persian puts
        # them — between clauses — so the model learns the context, not just the glyph.
        if rng.random() < 0.45:
            words = label.split()
            if len(words) > 3:
                at = rng.randrange(1, len(words) - 1)
                words[at] = words[at] + "،"
                label = " ".join(words)

        label = re.sub(r"\s+", " ", label).strip()
        if not (6 <= len(label) <= 110):
            continue

        # Three lines in ten are justified, which is roughly how often an administrative
        # page is. The label stays unstretched.
        render = label
        if rng.random() < 0.30:
            render = justify(label, int(len(label) * rng.uniform(1.08, 1.45)), rng)

        out.append((render, label))
    return out
