"""
OCR-aware Dehkhoda corrector (v2).

Why v2 exists
-------------
v1 treated OCR output like a typo-ridden text and allowed any single edit over the
whole 35-letter Persian alphabet. Measured on the benchmark it made accuracy WORSE
(+1.04 CER), and the diagnosis was unambiguous:

    HARM substitution 60   (س→ا ×14, ژ→ت ×11, د→م ×7, ح→ن ×7 ...)
    HARM transposition 10  (and ZERO helpful transpositions)
    HARM deletion 13 / HELP deletion 5

Those substitutions are not OCR errors — no OCR confuses س with ا. They are *typing*
errors. Real OCR errors are VISUAL: same letter skeleton with different dots
(ب پ ت ث ن ی), or same shape family (ج چ ح خ). And transposition is a keyboard
artifact that OCR essentially never produces.

v2 therefore restricts the edit model to what an OCR can actually get wrong:
  * substitution ONLY between visually confusable letters (same skeleton family)
  * deletion ONLY of a doubled letter (استحضاار → استحضار), the one deletion that helped
  * NO insertions, NO transpositions
  * optional: only accept a correction when it is UNAMBIGUOUS (exactly one
    dictionary candidate), so the corrector never guesses between rivals

Everything else is unchanged from v1: only Persian-script tokens are considered,
tokens already in the dictionary are never touched, Latin/digits/codes never touched.
"""
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr_eval"))
from run_eval import normalize  # noqa: E402

DB_PATH = Path(__file__).resolve().parent / "dehkhoda.db"

_PERSIAN_CORE = re.compile(r"[ء-يٰ-ۓۺ-ۼ‌]+")
_HAS_PERSIAN = re.compile(r"[ء-يٰ-ۓۺ-ۼ]")

# Visually confusable Persian letter families: identical (or near-identical)
# skeletons that differ only in dots/diacritics or a small stroke. These are the
# substitutions an OCR genuinely makes.
CONFUSION_GROUPS = [
    "بپتثنی",     # the tooth skeleton — the single biggest OCR confusion family
    "جچحخ",       # same bowl shape, dots differ
    "دذ",
    "رزژ",
    "سش",
    "صض",
    "طظ",
    "عغ",
    "فق",
    "کگ",
    "اآأإ",
    "هة",
    "وؤ",
    "ئي",
]


def _build_confusion_map():
    m: dict[str, set] = {}
    for group in CONFUSION_GROUPS:
        for ch in group:
            m.setdefault(ch, set()).update(c for c in group if c != ch)
    return m


CONFUSABLE = _build_confusion_map()


class CorrectorV2:
    # Defaults are the ONLY configuration measured to be non-harmful on the
    # benchmark (see ablation.py):
    #
    #   v1  any-edit                     17.04% -> 18.08%  (+1.04)  83 harmful
    #   v2a confusable-sub + doubled-del 17.04% -> 17.28%  (+0.24)  23 harmful
    #   v2b  ... + unambiguous only      17.04% -> 17.13%  (+0.09)  11 harmful
    #   v2c confusable-sub only          17.04% -> 17.15%  (+0.11)  11 harmful
    #   v2d doubled-del only             17.04% -> 17.01%  (-0.03)   0 harmful  <-- default
    #
    # Confusable substitution stays available behind a flag, but it is off by
    # default: Dehkhoda lacks modern loanwords (پروژه، سامانه، دیجیتال), so a
    # visually-plausible edit can still snap a correct modern word onto a
    # classical one (پروژه → پروره).
    def __init__(self, db_path: Path = DB_PATH, *,
                 allow_substitution: bool = False,
                 allow_doubled_deletion: bool = True,
                 unambiguous_only: bool = True,
                 min_len: int = 3):
        if not Path(db_path).exists():
            raise FileNotFoundError(f"{db_path} not found; run build_db.py first.")
        con = sqlite3.connect(str(db_path))
        self.vocab = set()
        for (wn,) in con.execute("SELECT word_norm FROM words WHERE word_norm<>''"):
            for tok in wn.split():
                if tok:
                    self.vocab.add(tok)
        con.close()
        self.allow_substitution = allow_substitution
        self.allow_doubled_deletion = allow_doubled_deletion
        self.unambiguous_only = unambiguous_only
        self.min_len = min_len
        self._cache: dict[str, str | None] = {}

    # ── OCR-plausible edit candidates ─────────────────────────────────────────
    def _candidates(self, w: str) -> set:
        out = set()
        if self.allow_substitution:
            for i, ch in enumerate(w):
                for alt in CONFUSABLE.get(ch, ()):
                    out.add(w[:i] + alt + w[i + 1:])
        if self.allow_doubled_deletion:
            for i in range(1, len(w)):
                if w[i] == w[i - 1]:
                    out.add(w[:i] + w[i + 1:])
        out.discard(w)
        return out

    def _best_correction(self, norm: str):
        if norm in self.vocab:
            return None
        if len(norm) < self.min_len:
            return None
        cands = sorted(c for c in self._candidates(norm) if c in self.vocab)
        if not cands:
            return None
        if self.unambiguous_only and len(cands) > 1:
            return None                      # rival candidates: refuse to guess
        return cands[0]

    def correct_token(self, token: str) -> str:
        if not _HAS_PERSIAN.search(token):
            return token
        m = _PERSIAN_CORE.search(token)
        if not m:
            return token
        pre, core, post = token[:m.start()], m.group(0), token[m.end():]
        norm = normalize(core, "digit")
        if norm not in self._cache:
            self._cache[norm] = self._best_correction(norm)
        repl = self._cache[norm]
        if repl is None or repl == norm:
            return token
        return pre + repl + post

    def correct_text(self, text: str):
        tokens = text.split(" ")
        out, n_persian, n_oov, n_changed = [], 0, 0, 0
        for t in tokens:
            if _HAS_PERSIAN.search(t):
                n_persian += 1
                m = _PERSIAN_CORE.search(t)
                norm = normalize(m.group(0), "digit") if m else ""
                if norm and norm not in self.vocab:
                    n_oov += 1
                ct = self.correct_token(t)
                if ct != t:
                    n_changed += 1
                out.append(ct)
            else:
                out.append(t)
        return " ".join(out), {"tokens": len(tokens), "persian": n_persian,
                               "oov": n_oov, "changed": n_changed}


if __name__ == "__main__":
    c = CorrectorV2()
    print(f"vocab={len(c.vocab):,}  confusable letters={len(CONFUSABLE)}")
    tests = ["سامانه", "سرویس", "دیجیتال", "داریم", "آید", "پروژه",   # correct-but-OOV: must NOT change
             "اسححضار", "استحضاار", "پیشایپیش", "فرمانیه"]            # real OCR errors
    for w in tests:
        fixed, _ = c.correct_text(w)
        flag = "unchanged" if fixed == w else f"-> {fixed}"
        print(f"  {w:12} {flag}")
