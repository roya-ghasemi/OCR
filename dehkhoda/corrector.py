"""
Dictionary-based post-correction layer for Persian OCR output.

Backed by the Dehkhoda SQLite database (dehkhoda/dehkhoda.db). At load time it
builds an in-memory set of normalized single-token headwords (multi-word entries
are split into tokens). Correction is deliberately conservative, because ~22% of
correctly-read modern words (فرمایید، سامانه، اینترنت، مدیرعامل …) are simply not
in Dehkhoda — blindly "fixing" every out-of-vocabulary token would damage them:

  * Only Persian-script tokens are touched. Latin letters, digits, emails, codes
    and punctuation are left exactly as the OCR produced them (the dictionary
    cannot help there, and that is where the OCR is weakest).
  * A token already in the dictionary is never changed.
  * An out-of-vocabulary token is replaced only if a single-edit (Levenshtein
    distance 1) neighbour exists in the dictionary; the best such neighbour is
    chosen deterministically. Tokens shorter than 2 letters are left alone.

Public API:
    c = Corrector()
    fixed, stats = c.correct_text(text)     # stats: tokens / persian / oov / changed
    fixed = c.correct_token("سامانه")
"""
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr_eval"))
from run_eval import normalize  # noqa: E402

DB_PATH = Path(__file__).resolve().parent / "dehkhoda.db"

# Persian/Arabic LETTERS only (plus ZWNJ) — deliberately excludes the Persian
# comma ، semicolon ؛ question ؟ and the Persian/Arabic digit ranges, which all
# live in the 06xx block but must not be pulled into a "word".
_PERSIAN_CORE = re.compile(r"[ء-يٰ-ۓۺ-ۼ‌]+")
_HAS_PERSIAN = re.compile(r"[ء-يٰ-ۓۺ-ۼ]")


class Corrector:
    def __init__(self, db_path: Path = DB_PATH, max_len_for_edit: int = 20):
        if not Path(db_path).exists():
            raise FileNotFoundError(f"{db_path} not found; run build_db.py first.")
        con = sqlite3.connect(str(db_path))
        self.vocab = set()
        alphabet = set()
        for (wn,) in con.execute("SELECT word_norm FROM words WHERE word_norm<>''"):
            for tok in wn.split():
                if tok:
                    self.vocab.add(tok)
                    alphabet.update(tok)
        con.close()
        # Persian letters only in the edit alphabet (no ZWNJ, no digits).
        self.alphabet = sorted(ch for ch in alphabet if "؀" <= ch <= "ۿ")
        self.max_len_for_edit = max_len_for_edit
        self._cache: dict[str, str | None] = {}

    # ── edit-distance-1 generate-and-test ─────────────────────────────────────
    def _edits1(self, w: str):
        splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits if b for c in self.alphabet]
        inserts = [a + c + b for a, b in splits for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def _best_correction(self, norm: str):
        if norm in self.vocab:
            return None                         # already valid, no change
        if len(norm) < 2 or len(norm) > self.max_len_for_edit:
            return None
        cands = [c for c in self._edits1(norm) if c in self.vocab]
        if not cands:
            return None
        # Deterministic pick: prefer same length as the input, then shortest,
        # then lexicographic. (No frequency signal in Dehkhoda to rank on.)
        cands.sort(key=lambda c: (abs(len(c) - len(norm)), len(c), c))
        return cands[0]

    def correct_token(self, token: str) -> str:
        """Correct one whitespace-delimited token, preserving non-Persian affixes."""
        if not _HAS_PERSIAN.search(token):
            return token                        # Latin / digits / punctuation: untouched
        m = _PERSIAN_CORE.search(token)
        if not m:
            return token
        pre, core, post = token[:m.start()], m.group(0), token[m.end():]
        norm = normalize(core, "digit")
        if norm in self._cache:
            repl = self._cache[norm]
        else:
            repl = self._best_correction(norm)
            self._cache[norm] = repl
        if repl is None or repl == norm:
            return token
        return pre + repl + post

    def correct_text(self, text: str):
        tokens = text.split(" ")
        out = []
        n_persian = n_oov = n_changed = 0
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
        stats = {"tokens": len(tokens), "persian": n_persian,
                 "oov": n_oov, "changed": n_changed}
        return " ".join(out), stats


if __name__ == "__main__":
    c = Corrector()
    print(f"vocab={len(c.vocab):,}  alphabet={len(c.alphabet)} letters")
    for w in ["سامانه", "اسححضار", "فرمانیه", "اعلان", "مجموعه", "INV-2026-0093", "پیشایپیش"]:
        fixed, st = c.correct_text(w)
        print(f"  {w!r:24} -> {fixed!r:24} {st}")
