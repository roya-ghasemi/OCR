# -*- coding: utf-8 -*-
"""HTTP surface that does not need a page read: liveness and upload validation."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from ocr_service.api import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with fastapi_testclient.TestClient(app) as c:
        yield c


def test_liveness_answers_without_touching_tesseract(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "alive"}


def test_an_upload_that_is_not_an_image_is_refused(client):
    r = client.post("/ocr", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_an_empty_upload_is_refused(client):
    r = client.post("/ocr", files={"file": ("x.png", b"", "image/png")})
    assert r.status_code == 400


def test_health_never_shows_the_broker_password(client, monkeypatch):
    from ocr_service import api
    monkeypatch.setattr(api.settings, "redis_url", "redis://:s3cret@redis:6379/0")
    body = client.get("/health").json()
    assert "s3cret" not in body["queue"]["broker"]
