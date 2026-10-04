# -*- coding: utf-8 -*-
r"""Remember what a page said, so the same page is not read twice.

Reading a page costs about three and a half seconds of CPU. The same file arrives again all
the time — a retry after a dropped connection, a user pressing the button twice, a batch
re-run after a crash, the same attachment forwarded by two people — and every one of those is
the identical computation.

**The key is not the image.** That is the mistake this is built to avoid. A transcript is a
function of the image *and* of everything that decided how it was read: which Persian model,
which digit model, which layout passes, whether the spell-fixer and the page crop and the ink
separation were on. Key on the image alone and the first thing that happens after an engine
change is that the service confidently serves a transcript the current engine would never
produce — with the new model's hash stamped on it by `provenance()`, because the stamp is
written at response time. A results file that describes a run that never happened is the one
failure mode this project has spent its whole history avoiding.

So the key folds in the provenance: the model hashes, the service version, and every setting
that changes the output. Change any of them and the old entries are simply never looked up
again. They are not deleted — they age out — which means rolling a setting back finds its
cached work still there.

What is cached is the response body as JSON, minus the prepared image: that is megabytes of
base64 per page, it is only wanted by the viewer, and recomputing it is cheap next to the
reading. A hit therefore serves a response without `prepared_image`, and a request that asks
for one is read properly rather than answered from here.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

# The settings that change what the transcript says. A setting absent from this list is one
# whose value cannot alter the output; adding a setting that *can* and forgetting it here is
# how a cache starts lying, so the list is explicit rather than "everything on Settings".
KEYED_SETTINGS = (
    "lang_fa", "layout_psms", "crop_page", "ink_separate", "deskew_min_deg",
    "min_line_conf", "number_min_prob", "letter_fields", "spellfix",
)


def key(image_bytes: bytes, settings) -> str:
    """One hex digest standing for "this image, read this way, by these models"."""
    h = hashlib.sha256()
    h.update(hashlib.sha256(image_bytes).digest())
    prov = settings.provenance()
    tess = prov.get("tesseract", {}) if isinstance(prov, dict) else {}
    parts = [
        str(prov.get("service_version") or prov.get("version") or ""),
        str(tess.get("lang_fa_sha256") or ""),
        str(tess.get("fas_sha256") or ""),
        str(tess.get("eng_sha256") or ""),
        str((prov.get("digit_model") or {}).get("sha256") or ""),
    ]
    parts += [f"{name}={getattr(settings, name, None)!r}" for name in KEYED_SETTINGS]
    h.update("\x1f".join(parts).encode("utf-8"))
    return h.hexdigest()


class ResultCache:
    """A bounded store of finished transcripts on disk.

    On disk rather than in memory because the win is largest across restarts — a crashed
    batch resumes without re-reading what it already read — and because a few thousand
    transcripts is tens of megabytes, which does not belong in a worker's heap.

    Eviction is least-recently-used by modification time, which a filesystem gives for free.
    """

    def __init__(self, directory: str | os.PathLike, max_entries: int = 2000,
                 enabled: bool = True):
        self.dir = Path(directory)
        self.max_entries = max_entries
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        if self.enabled:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.enabled = False           # an unwritable cache directory is not an error

    def _path(self, k: str) -> Path:
        # Two levels of fan-out: a single directory with thousands of entries is slow to
        # list on Windows, and eviction has to list it.
        return self.dir / k[:2] / f"{k}.json"

    def get(self, k: str) -> dict | None:
        if not self.enabled:
            return None
        p = self._path(k)
        try:
            raw = p.read_text(encoding="utf-8")
        except (OSError, ValueError):
            self.misses += 1
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # A half-written entry from a kill mid-write. Drop it and read the page.
            p.unlink(missing_ok=True)
            self.misses += 1
            return None
        self.hits += 1
        try:
            os.utime(p, None)                  # mark it recently used, for eviction
        except OSError:
            pass
        return data

    def put(self, k: str, payload: dict) -> None:
        if not self.enabled:
            return
        p = self._path(k)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            # Write beside it and replace, so a reader never sees a half-written file and a
            # kill mid-write leaves the old entry rather than a corrupt one.
            tmp = p.with_suffix(f".{os.getpid()}.part")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(p)
        except OSError:
            return
        self._evict()

    def _evict(self) -> None:
        try:
            entries = list(self.dir.glob("*/*.json"))
        except OSError:
            return
        if len(entries) <= self.max_entries:
            return
        entries.sort(key=lambda f: f.stat().st_mtime if f.exists() else 0)
        for f in entries[: len(entries) - self.max_entries]:
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass

    def stats(self) -> dict:
        n = -1
        if self.enabled:
            try:
                n = sum(1 for _ in self.dir.glob("*/*.json"))
            except OSError:
                n = -1
        return {"enabled": self.enabled, "entries": n, "max_entries": self.max_entries,
                "hits": self.hits, "misses": self.misses,
                "dir": str(self.dir) if self.enabled else None}
