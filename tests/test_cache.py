# -*- coding: utf-8 -*-
"""Guards for the transcript cache (E31).

The failure this exists to prevent is silent and serious: serving a transcript produced by an
engine that is no longer installed, stamped with the current engine's hash. Every test here
is about the key.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMG = b"pretend-this-is-a-jpeg"


def test_the_same_page_read_the_same_way_is_the_same_key():
    from ocr_service.cache import key
    from ocr_service.config import Settings
    assert key(IMG, Settings()) == key(IMG, Settings())
    assert key(IMG, Settings()) != key(b"a different page", Settings())


def test_changing_the_model_changes_the_key():
    """The whole point. A transcript is a function of the image *and* of what read it. Keyed
    on the image alone, the first thing that happens after an engine change is that the
    service serves a transcript the current engine would never produce — with the new
    model's hash stamped on it, because `provenance()` is written at response time."""
    from ocr_service.cache import key
    from ocr_service.config import Settings
    assert key(IMG, Settings(lang_fa="fas")) != key(IMG, Settings(lang_fa="fas_custom"))


def test_every_setting_that_changes_the_output_changes_the_key():
    from ocr_service.cache import key
    from ocr_service.config import Settings
    base = key(IMG, Settings())
    for name, alt in (("spellfix", False), ("layout_psms", "3,6"), ("crop_page", False),
                      ("ink_separate", False), ("min_line_conf", 50.0),
                      ("letter_fields", False), ("number_min_prob", 0.9),
                      ("deskew_min_deg", 2.0)):
        assert key(IMG, Settings(**{name: alt})) != base, name


def test_a_setting_that_cannot_change_the_output_is_not_keyed():
    """`workers` only decides how many processes do the reading, and `max_upload_mb` whether
    the request is accepted at all. Keying on them would throw away usable entries for
    nothing."""
    from ocr_service.cache import KEYED_SETTINGS
    assert "workers" not in KEYED_SETTINGS
    assert "max_upload_mb" not in KEYED_SETTINGS
    assert "redis_url" not in KEYED_SETTINGS


def test_a_half_written_entry_is_dropped_not_served(tmp_path):
    """A kill mid-write used to be able to leave a truncated file. Entries are written beside
    and replaced, but a corrupt one that does appear must read as a miss."""
    from ocr_service.cache import ResultCache
    c = ResultCache(tmp_path, enabled=True)
    c.put("abc123", {"text": "fine"})
    assert c.get("abc123") == {"text": "fine"}
    c._path("abc123").write_text("{not json", encoding="utf-8")
    assert c.get("abc123") is None
    assert not c._path("abc123").exists(), "the corrupt entry should be removed"


def test_eviction_keeps_the_most_recently_used(tmp_path):
    import os
    import time

    from ocr_service.cache import ResultCache
    c = ResultCache(tmp_path, max_entries=3, enabled=True)
    for i in range(3):
        c.put(f"{i:02d}" + "f" * 10, {"n": i})
        os.utime(c._path(f"{i:02d}" + "f" * 10), (time.time() - 100 + i, time.time() - 100 + i))
    c.put("zz" + "f" * 10, {"n": 99})
    assert c.get("zz" + "f" * 10) == {"n": 99}
    assert c.get("00" + "f" * 10) is None, "the oldest entry should have been evicted"


def test_a_disabled_cache_never_stores_or_serves(tmp_path):
    from ocr_service.cache import ResultCache
    c = ResultCache(tmp_path, enabled=False)
    c.put("k" * 12, {"text": "x"})
    assert c.get("k" * 12) is None
    assert c.stats()["enabled"] is False
