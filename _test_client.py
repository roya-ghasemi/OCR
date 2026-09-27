"""Temporary OCR test client: POST the sample letter and print the JSON result."""
import json, sys
import httpx

URL = "http://localhost:8001/ocr"
IMG = r"D:\ocr\exampel_paper.png"

with open(IMG, "rb") as fh:
    files = {"file": ("exampel_paper.png", fh.read(), "image/png")}

print(f"[client] POST {URL}  (image: {IMG})", flush=True)
try:
    r = httpx.post(URL, files=files, timeout=600.0)
except Exception as exc:
    print(f"[client] REQUEST FAILED: {exc}")
    sys.exit(2)

print(f"[client] HTTP {r.status_code}")
try:
    data = r.json()
    print(json.dumps(data, ensure_ascii=False, indent=2))
except Exception:
    print("[client] non-JSON response:")
    print(r.text)
sys.exit(0 if r.status_code == 200 else 1)
