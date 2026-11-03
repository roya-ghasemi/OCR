# -*- coding: utf-8 -*-
"""HTTP surface that does not need a page read: liveness and upload validation."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from ocr_service.api import app


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


def test_with_a_key_set_reading_needs_it(client, monkeypatch):
    from ocr_service import api
    monkeypatch.setattr(api.settings, "api_key", "k1")
    assert client.post("/ocr", files={"file": ("x.png", b"", "image/png")}).status_code == 401
    r = client.post("/ocr", files={"file": ("x.png", b"", "image/png")}, headers={"X-API-Key": "k1"})
    assert r.status_code == 400                     # past the key, refused for the empty upload
    assert client.get("/healthz").status_code == 200
