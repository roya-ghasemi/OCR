"""Hybrid Persian OCR service — VLM reader + Tesseract numeric validation.

Layout
    config.py            settings (env-driven), model paths, backend selection
    backends/            InferenceBackend interface + llama.cpp / vLLM / Triton
    numeric_validator.py Tesseract cross-check of every numeric span, backend-agnostic
    preprocess.py        deskew / auto-crop / compress (server fallback + client spec)
    pipeline.py          preprocess -> primary VLM -> [secondary VLM] -> numeric validation
    schemas.py           response contract (fields, per-number confidence, candidates)
    api.py               FastAPI: POST /ocr, GET /health, GET /jobs/{id}
    tasks.py             Celery app + task (Redis broker/backend)
    benchmark.py         before/after report on a labelled folder
    make_eval_set.py     builds the evaluation file the benchmark consumes

The evaluated legacy path (`main.py`, `/ocr` on :8000) is untouched; this package is
mounted separately so every historical number in `ocr_eval/` stays comparable.
"""
__version__ = "0.1.0"
