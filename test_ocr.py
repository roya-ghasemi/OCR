"""
test_ocr.py – FastAPI TestClient tests for the Persian OCR service.

Run with:  python test_ocr.py
Or:        pytest test_ocr.py -v
"""

import sys
import numpy as np
import cv2
from fastapi.testclient import TestClient

from main import app


# ── Helpers ────────────────────────────────────────────────────────────────────
def make_blank_image(width: int = 600, height: int = 800) -> bytes:
    """Return a minimal white JPEG – valid image, no real text."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (20, 20), (580, 80), (0, 0, 0), 2)
    for y in range(120, 700, 50):
        cv2.line(img, (40, y), (560, y), (80, 80, 80), 1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def run(label: str, passed: bool, detail: str = ""):
    mark = "PASS" if passed else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{mark}]  {label}{suffix}")
    if not passed:
        sys.exit(1)


# ── Tests (all receive an active client with lifespan already started) ─────────
def test_health(client: TestClient):
    r = client.get("/health")
    run("GET /health → 200", r.status_code == 200)
    run("status = healthy", r.json()["status"] == "healthy")


def test_root(client: TestClient):
    r = client.get("/")
    run("GET / → 200", r.status_code == 200)


def test_invalid_mime(client: TestClient):
    r = client.post("/ocr", files={"file": ("doc.txt", b"hello", "text/plain")})
    run("Unsupported MIME → 400", r.status_code == 400)


def test_empty_file(client: TestClient):
    r = client.post("/ocr", files={"file": ("empty.jpg", b"", "image/jpeg")})
    run("Empty file → 400", r.status_code == 400)


def test_synthetic_image(client: TestClient):
    img_bytes = make_blank_image()
    r = client.post("/ocr", files={"file": ("synth.jpg", img_bytes, "image/jpeg")})
    run("Synthetic image → 200", r.status_code == 200)
    data = r.json()
    run("Response has full_text key", "full_text" in data)
    run("Response has segments key", "segments" in data)
    run("success = True", data["success"] is True)
    print(f"         segments detected: {data['total_segments']}")
    print(f"         average confidence: {data['average_confidence']:.2%}")


def test_real_letter(client: TestClient, path: str = r"d:\ocr\exampel_paper.png"):
    """Test with the actual sample letter – skipped if file not found."""
    try:
        with open(path, "rb") as fh:
            img_bytes = fh.read()
    except FileNotFoundError:
        print(f"  [SKIP]  Real letter test – file not found: {path}")
        return

    r = client.post("/ocr", files={"file": ("letter.png", img_bytes, "image/png")})
    run("Real letter → 200", r.status_code == 200, f"status={r.status_code}")
    data = r.json()
    run("full_text non-empty", bool(data.get("full_text", "").strip()))

    print("\n" + "─" * 60)
    print("EXTRACTED TEXT (real letter):")
    print("─" * 60)
    print(data["full_text"])
    print("─" * 60)
    print(f"Segments : {data['total_segments']}")
    print(f"Avg conf : {data['average_confidence']:.2%}")
    print("─" * 60 + "\n")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Persian OCR Service – Test Suite")
    print("=" * 60)
    print()

    # Use context-manager form so lifespan (= EasyOCR model loading) fires
    with TestClient(app) as client:
        print("[1] Basic endpoint tests")
        test_health(client)
        test_root(client)
        test_invalid_mime(client)
        test_empty_file(client)
        print()
        print("[2] Synthetic image OCR")
        test_synthetic_image(client)
        print()
        print("[3] Real letter OCR  (exampel_paper.png)")
        test_real_letter(client)

    print()
    print("All tests passed.")
