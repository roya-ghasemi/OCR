"""Persian OCR service — full-text transcription of any image, CPU only.

Entry points: `api.py` (FastAPI app), `tasks.py` (Celery worker), `pipeline.OcrPipeline`
(in-process use). Module map in ARCHITECTURE.md.
"""
__version__ = "1.3.0"
