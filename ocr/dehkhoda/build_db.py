"""
Load the 33 gzip-compressed Dehkhoda MySQL dumps into a local SQLite database.

Every dump ships the same schema:
    CREATE TABLE `words` (`id` int, `word` text, `meaning` text)

MySQL-only syntax (ENGINE=, COLLATE=, int(255), ALTER TABLE ... MODIFY, the
/*!... */ conditional comments) is ignored: we recreate a clean SQLite table and
parse the INSERT ... VALUES tuples ourselves, respecting backslash-escaped string
literals (phpMyAdmin escapes quotes as \\' , not by doubling).

Result: dehkhoda/dehkhoda.db
    words(id INTEGER PRIMARY KEY, word TEXT, word_norm TEXT, meaning TEXT)
    index on word_norm  -> fast normalized lookup for the corrector

Usage:
    python dehkhoda/build_db.py            # load everything
    python dehkhoda/build_db.py --no-meaning   # skip meanings (smaller/faster)
"""
import argparse
import glob
import gzip
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

# Reuse the exact same normalization the eval harness uses, so dictionary keys
# and OCR tokens are compared on equal footing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr_eval"))
from run_eval import normalize  # noqa: E402

HERE = Path(__file__).resolve().parent
DUMP_DIR = HERE.parent / "Dehkhoda-SQL-master"
DB_PATH = HERE / "dehkhoda.db"

_INSERT_RE = re.compile(r"INSERT\s+INTO\s+`?words`?\s*\([^)]*\)\s*VALUES", re.IGNORECASE)


def parse_values(text: str, start: int):
    """Parse the `(...),(...);` tuple list beginning at `start` (just after VALUES).
    Yields (id, word, meaning) tuples. Returns the index just past the closing ';'."""
    i, n = start, len(text)
    rows = []
    while i < n:
        # advance to next '(' or end of statement
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == ";":
            i += 1
            break
        if text[i] != "(":
            i += 1
            continue
        i += 1  # past '('
        fields = []
        buf = []
        in_str = False
        while i < n:
            c = text[i]
            if in_str:
                if c == "\\" and i + 1 < n:
                    nxt = text[i + 1]
                    buf.append({"n": "\n", "r": "\r", "t": "\t", "0": "\0"}.get(nxt, nxt))
                    i += 2
                    continue
                if c == "'":
                    in_str = False
                    i += 1
                    continue
                buf.append(c)
                i += 1
            else:
                if c == "'":
                    in_str = True
                    i += 1
                elif c == ",":
                    fields.append("".join(buf).strip())
                    buf = []
                    i += 1
                elif c == ")":
                    fields.append("".join(buf).strip())
                    i += 1
                    break
                else:
                    buf.append(c)
                    i += 1
        if len(fields) >= 3:
            try:
                wid = int(re.sub(r"[^0-9]", "", fields[0]) or 0)
            except ValueError:
                wid = 0
            rows.append((wid, fields[1], fields[2]))
    return rows, i


def iter_rows(sql_text: str):
    for m in _INSERT_RE.finditer(sql_text):
        rows, _ = parse_values(sql_text, m.end())
        yield from rows


def build(keep_meaning: bool = True) -> None:
    dumps = sorted(glob.glob(str(DUMP_DIR / "*.sql.gz")))
    if not dumps:
        sys.exit(f"No .sql.gz dumps under {DUMP_DIR}")

    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute(
        "CREATE TABLE words (id INTEGER PRIMARY KEY, word TEXT, word_norm TEXT, meaning TEXT)"
    )

    total = 0
    t0 = time.time()
    for path in dumps:
        name = os.path.basename(path)
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            sql_text = fh.read()
        batch = []
        for wid, word, meaning in iter_rows(sql_text):
            wn = normalize(word, "digit")
            batch.append((wid, word, wn, meaning if keep_meaning else None))
            if len(batch) >= 5000:
                con.executemany("INSERT OR IGNORE INTO words VALUES (?,?,?,?)", batch)
                total += len(batch)
                batch = []
        if batch:
            con.executemany("INSERT OR IGNORE INTO words VALUES (?,?,?,?)", batch)
            total += len(batch)
        con.commit()
        print(f"  {name:<14} loaded (running total {total:,})")

    con.execute("CREATE INDEX idx_word_norm ON words(word_norm)")
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM words").fetchone()[0]
    n_norm = con.execute("SELECT COUNT(DISTINCT word_norm) FROM words WHERE word_norm<>''").fetchone()[0]
    con.close()
    print(f"\nRows loaded: {n:,}   distinct normalized words: {n_norm:,}")
    print(f"DB: {DB_PATH}  ({DB_PATH.stat().st_size/1e6:.1f} MB)  in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-meaning", action="store_true", help="do not store meaning HTML")
    args = ap.parse_args()
    build(keep_meaning=not args.no_meaning)
