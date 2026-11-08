# -*- coding: utf-8 -*-
"""Tesseract command-line runner.

Every call starts a tesseract process, and a page is read as several dozen line images.
Started one process per line, start-up cost more CPU than recognition did. `read_many`
gives a batch of images to a single process through a list file instead; the TSV's
`page_num` column maps the rows back to their images. The LSTM recogniser keeps no state
between pages, so the words and confidences are the same as reading each image alone.
"""
from __future__ import annotations

import csv
import io
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

# The pipeline already runs one tesseract process per core; OpenMP threads inside each
# process would only compete with the others.
_ENV = {**os.environ, "OMP_THREAD_LIMIT": "1"}

Row = dict          # {"text", "conf", "box": (x0, y0, x1, y1), "key": (block, par, line)}


class Tesseract:
    def __init__(self, cmd: str | None, tessdata: str | None, timeout_s: float = 120.0):
        self.cmd = cmd
        self.tessdata = tessdata
        self.timeout_s = timeout_s

    def probe(self) -> tuple[str | None, list[str]]:
        """Engine version and the languages its tessdata directory provides."""
        if not self.cmd:
            return None, []
        try:
            p = subprocess.run([self.cmd, "--version"], capture_output=True, text=True, timeout=20)
            version = (p.stdout or p.stderr).splitlines()[0].strip()
            args = [self.cmd, "--list-langs"] + (["--tessdata-dir", self.tessdata] if self.tessdata else [])
            q = subprocess.run(args, capture_output=True, text=True, timeout=20)
            return version, [x.strip() for x in q.stdout.splitlines()[1:] if x.strip()]
        except (OSError, subprocess.SubprocessError, IndexError) as exc:
            return f"error: {exc}", []

    def _args(self, src: str, lang: str, psm: str) -> list[str]:
        args = [self.cmd, src, "stdout", "-l", lang]
        if self.tessdata:
            args += ["--tessdata-dir", self.tessdata]
        return args + ["--psm", psm, "-c", "tessedit_create_tsv=1"]

    def _exec(self, args: list[str], stdin: bytes | None, timeout: float) -> str:
        p = subprocess.run(args, input=stdin, capture_output=True, timeout=timeout, env=_ENV)
        if p.returncode != 0:
            err = p.stderr.decode("utf-8", "replace").strip().splitlines()
            log.warning("tesseract exited %s (%s, psm %s): %s", p.returncode,
                        args[args.index("-l") + 1], args[args.index("--psm") + 1],
                        err[-1] if err else "")
            return ""
        return p.stdout.decode("utf-8", "replace")

    def read(self, img: Image.Image, psm: str, lang: str) -> list[Row]:
        """Word rows of one image."""
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return parse_tsv(self._exec(self._args("stdin", lang, psm), buf.getvalue(), self.timeout_s)).get(1, [])

    def read_many(self, imgs: list[Image.Image], psm: str, lang: str) -> list[list[Row]]:
        """Word rows of each image, read by one tesseract process."""
        if not imgs:
            return []
        if len(imgs) == 1:
            return [self.read(imgs[0], psm, lang)]
        with tempfile.TemporaryDirectory(prefix="ocr_") as d:
            paths = []
            for i, im in enumerate(imgs):
                p = Path(d) / f"{i:05d}.png"
                im.save(p, "PNG")
                paths.append(str(p))
            listing = Path(d) / "images.txt"
            listing.write_text("\n".join(paths) + "\n", encoding="utf-8")
            pages = parse_tsv(self._exec(self._args(str(listing), lang, psm), None,
                                         self.timeout_s * len(imgs)))
        return [pages.get(i + 1, []) for i in range(len(imgs))]


def parse_tsv(tsv: str) -> dict[int, list[Row]]:
    """Recognised words (TSV level 5) per page number."""
    out: dict[int, list[Row]] = {}
    for r in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        t = (r.get("text") or "").strip()
        try:
            c = float(r.get("conf") or -1)
        except ValueError:
            c = -1.0
        if r.get("level") == "5" and t and c >= 0:
            x, y, w, h = (int(r[k]) for k in ("left", "top", "width", "height"))
            out.setdefault(int(r["page_num"]), []).append(
                {"text": t, "conf": c, "box": (x, y, x + w, y + h),
                 "key": (r["block_num"], r["par_num"], r["line_num"])})
    return out
